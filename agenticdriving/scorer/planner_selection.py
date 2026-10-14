"""Pipeline-side trajectory selection for learned planners.

The planner subprocess only returns ``(proposals, scores)``. Everything that
used to live in each planner's ``client.py`` after inference — building the
candidate pool (carry-previous, top-k, recovery candidates, default fallbacks),
running the VLM / AgenticDriving selection, and assembling the CARLA plan payload —
now happens here, on the pipeline (pixi) side, once for all learned planners.

``proposals`` are expected in CARLA ego-local coordinates ([x_right, y_forward]),
shape ``[N, T, 2]``, so this module is planner-agnostic; per-planner labels are
passed in via the constructor.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from agenticdriving.adapters.carla.defaults import get_default_trajectories
from agenticdriving.adapters.carla.geometry import (
    info_to_pose,
    local_plan_to_world,
    path_length,
    truncate_plan,
    world_points_to_current_local,
)
from agenticdriving.scorer.payloads import build_carla_plan_payload
from agenticdriving.scorer.selection_strategies import (
    ArgmaxStrategy,
    RecoveryStrategy,
)
from agenticdriving.planners.fallback_rule_based.provider import (
    build_primitive_candidate_rows,
    get_primitive_proposals_and_scores,
)
from agenticdriving.orchestration.directional_options import (
    junction_route_heading_violation,
    proposal_moves_left,
)

PLAN_DT_SEC = 0.5
TOPK_PROPOSALS_TO_SEND = 20


def _identity_converter(trajectory: np.ndarray) -> np.ndarray:
    # Proposals already arrive in CARLA ego-local coordinates, so no conversion
    # is needed when handing them to build_carla_plan_payload.
    return np.asarray(trajectory, dtype=np.float32)


def build_carry_plan_candidate(
    previous_plan: Optional[np.ndarray],
    previous_pose: Optional[np.ndarray],
    previous_selected_score: Optional[float],
    previous_timestamp: Optional[float],
    current_info: Dict[str, object],
    vlm_cfg,
) -> Optional[Dict[str, object]]:
    if not vlm_cfg.carry_previous_enabled or previous_plan is None or previous_pose is None or previous_timestamp is None:
        return None

    current_timestamp = float(current_info.get("timestamp", previous_timestamp))
    elapsed_sec = max(0.0, current_timestamp - float(previous_timestamp))
    elapsed_pose_steps = int(round(elapsed_sec / PLAN_DT_SEC))
    if elapsed_pose_steps >= len(previous_plan):
        return None

    trimmed_plan = np.asarray(previous_plan[elapsed_pose_steps:], dtype=np.float32)
    if len(trimmed_plan) < vlm_cfg.carry_previous_min_points:
        return None

    points_world = local_plan_to_world(trimmed_plan, np.asarray(previous_pose, dtype=np.float32))
    current_local = world_points_to_current_local(points_world, info_to_pose(current_info))

    valid_mask = current_local[:, 1] > 0.0
    if not np.any(valid_mask):
        return None
    first_valid_idx = int(np.argmax(valid_mask))
    current_local = current_local[first_valid_idx:]

    if len(current_local) < vlm_cfg.carry_previous_min_points:
        return None
    if path_length(current_local) < vlm_cfg.carry_previous_min_path_m:
        return None

    return {
        "source": "carry_prev",
        "proposal_index": None,
        "proposal_score": 0.0,
        "proposal_score_norm": 0.0,
        "origin_selected_score_raw": None if previous_selected_score is None else float(previous_selected_score),
        "local_plan": current_local.astype(np.float32),
        "execution_plan": current_local.astype(np.float32),
        "carry_elapsed_sec": elapsed_sec,
        "carry_elapsed_pose_steps": elapsed_pose_steps,
    }


class LearnedPlannerSelector:
    """Stateful per-run selector. Call :meth:`select` once per frame.

    Holds the configured recovery selector, the AgenticDriving runtime, the
    resolved configs, and the carry-previous state. Produces the CARLA plan
    payload the simulator consumes.
    """

    def __init__(
        self,
        *,
        vlm_selector,
        runtime_name: str = "agenticdriving",
        agenticdriving_cfg,
        vlm_cfg,
        recovery_cfg,
        current_source_name: str = "current_rap",
        learned_default_source: str = "fallback_rap_argmax",
        plain_source: str = "rap_argmax",
        score_fallback_key: str = "rap_score",
        planner_log_name: str = "RAP",
        strict_learned_argmax_lookup: bool = True,
        q_key_prefix: bool = True,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.vlm_selector = vlm_selector
        self.runtime_name = str(runtime_name)
        self.agenticdriving_cfg = agenticdriving_cfg
        self.vlm_cfg = vlm_cfg
        self.recovery_cfg = recovery_cfg
        self.current_source_name = current_source_name
        self.learned_default_source = learned_default_source
        self.plain_source = plain_source
        self.score_fallback_key = score_fallback_key
        self.planner_log_name = planner_log_name
        self.strict_learned_argmax_lookup = strict_learned_argmax_lookup
        self.q_key_prefix = q_key_prefix
        self.logger = logger or logging.getLogger("planner_selection")

        self.frame_index = 0
        self.previous_selected_plan: Optional[np.ndarray] = None
        self.previous_selected_pose: Optional[np.ndarray] = None
        self.previous_selected_score: Optional[float] = None
        self.previous_selected_timestamp: Optional[float] = None
        self.previous_selected_source: Optional[str] = None
        self.argmax_strategy = ArgmaxStrategy(self)
        self.recovery_strategy = RecoveryStrategy(self)

    # ------------------------------------------------------------------ payload
    def _build_plan_payload(self, proposals, scores, output_num_poses, *, topk=TOPK_PROPOSALS_TO_SEND, **kwargs) -> Dict[str, object]:
        return build_carla_plan_payload(
            proposals=proposals,
            scores=scores,
            output_num_poses=output_num_poses,
            plan_converter=_identity_converter,
            default_source_name=self.current_source_name,
            default_trajectory_provider=get_default_trajectories,
            topk=topk,
            **kwargs,
        )

    # --------------------------------------------------------------- candidates
    def _build_learned_candidate_rows(
        self,
        proposals: np.ndarray,
        scores: np.ndarray,
        output_num_poses: int,
        current_info: Dict[str, object],
        reserved_candidate_slots: int,
    ) -> Tuple[List[Dict[str, object]], bool]:
        vlm_cfg = self.vlm_cfg
        allow_carry_previous = not (
            self.previous_selected_source is not None
            and str(self.previous_selected_source).startswith("default_fallback_")
        )
        carry_candidate = build_carry_plan_candidate(
            previous_plan=self.previous_selected_plan if allow_carry_previous else None,
            previous_pose=self.previous_selected_pose if allow_carry_previous else None,
            previous_selected_score=self.previous_selected_score if allow_carry_previous else None,
            previous_timestamp=self.previous_selected_timestamp if allow_carry_previous else None,
            current_info=current_info,
            vlm_cfg=vlm_cfg,
        )

        sorted_indices = np.argsort(scores)[::-1]
        carry_slot_count = 1 if carry_candidate is not None else 0
        current_candidate_limit = max(
            1,
            int(vlm_cfg.candidate_limit) - carry_slot_count - max(0, int(reserved_candidate_slots)),
        )
        current_candidate_limit = min(current_candidate_limit, int(len(sorted_indices)))
        candidate_indices = sorted_indices[:current_candidate_limit]

        candidate_rows: List[Dict[str, object]] = []
        if carry_candidate is not None:
            candidate_rows.append(carry_candidate)

        for idx in candidate_indices:
            # Proposals are already CARLA-local and truncated to output_num_poses.
            full_plan = np.asarray(proposals[idx], dtype=np.float32)
            candidate_rows.append(
                {
                    "source": self.current_source_name,
                    "proposal_index": int(idx),
                    "proposal_score": float(scores[idx]),
                    "local_plan": full_plan,
                    "execution_plan": full_plan.copy(),
                }
            )

        if vlm_cfg.include_default_candidates:
            for default_idx, default_plan in enumerate(get_default_trajectories(output_num_poses)):
                candidate_rows.append(
                    {
                        "source": f"default_fallback_{default_idx}",
                        "proposal_index": None,
                        "proposal_score": 0.0,
                        "local_plan": default_plan,
                        "execution_plan": default_plan.copy(),
                    }
                )

        carry_row = next((row for row in candidate_rows if row.get("source") == "carry_prev"), None)
        shared_horizon = len(carry_row["local_plan"]) if carry_row is not None else output_num_poses
        shared_horizon = max(1, int(shared_horizon))
        for row in candidate_rows:
            execution_plan = np.asarray(row.get("execution_plan", row["local_plan"]), dtype=np.float32)
            row["execution_plan"] = execution_plan
            row["local_plan"] = truncate_plan(execution_plan, shared_horizon)

        # Apply the same map policy at the VLM boundary as in Loop 1 scoring.
        # This also covers carry/default rows, so a stale left-going plan is
        # never rendered or described to the model as an available option.
        candidate_rows = [
            row
            for row in candidate_rows
            if not proposal_moves_left(row["local_plan"], current_info)
            and not junction_route_heading_violation(
                row["local_plan"], current_info
            )
        ]

        # A junction may invalidate every moving learned/default/carry row.
        # Keep a real stationary option so the recovery critic never receives
        # an empty family and no forbidden trajectory is resurrected.
        if not candidate_rows:
            stop_plan = np.zeros((max(2, output_num_poses), 2), dtype=np.float32)
            candidate_rows.append(
                {
                    "source": "hard_filter:junction_route_stop",
                    "proposal_index": None,
                    "proposal_score": 0.0,
                    "local_plan": stop_plan,
                    "execution_plan": stop_plan.copy(),
                }
            )

        return candidate_rows, allow_carry_previous

    def _build_rule_based_candidate_rows(
        self,
        obs,
        info,
        info_history,
        privileged_info,
        output_num_poses: int,
    ) -> List[Dict[str, object]]:
        if not self.recovery_cfg.enabled:
            return []
        try:
            # In-repo primitive ladder (already CARLA-local); rows carry
            # execution_states / primitive_name for open-loop execution.
            prim_proposals, prim_scores, prim_names, prim_states = (
                get_primitive_proposals_and_scores(
                    info=info,
                    privileged_info=privileged_info,
                    output_num_poses=output_num_poses,
                    allowed_primitives=self.recovery_cfg.allowed_primitives,
                    lane_bypass_min_gap_m=getattr(
                        self.recovery_cfg, "lane_bypass_min_gap_m", 0.0
                    ),
                    lane_bypass_clearance_buffer_m=getattr(
                        self.recovery_cfg,
                        "lane_bypass_clearance_buffer_m",
                        1.0,
                    ),
                    lane_bypass_required_time_sec=getattr(
                        self.recovery_cfg,
                        "lane_bypass_required_time_sec",
                        3.5,
                    ),
                    lane_bypass_min_time_gap_sec=getattr(
                        self.recovery_cfg,
                        "lane_bypass_min_time_gap_sec",
                        4.5,
                    ),
                    obstacle_clearance_m=getattr(
                        self.recovery_cfg, "obstacle_clearance_m", 8.0
                    ),
                )
            )
            rows = build_primitive_candidate_rows(
                prim_proposals,
                prim_scores,
                prim_names,
                prim_states,
                output_num_poses=output_num_poses,
                source_name=self.recovery_cfg.source_name,
                topk=self.recovery_cfg.topk,
            )
            # Orchestrator recovery memory: maneuvers already attempted this
            # episode are hard-excluded on the dummy path (the key is only
            # set then); the VLM path re-ranks the full ladder itself.
            excluded = set(info.get("recovery_excluded_primitives") or [])
            if excluded:
                rows = [
                    row for row in rows
                    if str(row.get("primitive_name")) not in excluded
                ]
            rows = [
                row
                for row in rows
                if not junction_route_heading_violation(
                    row["local_plan"], info
                )
            ]
            return rows
        except Exception:
            self.logger.exception("Failed to append rule-based %s recovery candidates", self.planner_log_name)
            return []

    # -------------------------------------------------------------------- select
    def select(
        self,
        *,
        proposals: np.ndarray,
        scores: np.ndarray,
        obs: Dict[str, Any],
        info: Dict[str, Any],
        info_history: Sequence[Dict[str, Any]],
        privileged_info: Optional[Any] = None,
        strategy: Optional[str] = None,
    ) -> Dict[str, object]:
        """Pick one trajectory from ``proposals``.

        ``strategy``: ``"argmax"`` forces the lightweight argmax path (used by
        the orchestrator's first-round scorer stage, where scores have already
        been shaped by the scoring/ modules); ``None`` keeps the config-driven
        dispatch (argmax, or the agentic recovery loop when AgenticDriving is
        enabled) — the recovery stage relies on that dispatch.
        """
        proposals = np.asarray(proposals, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        output_num_poses = int(proposals.shape[1]) if proposals.ndim == 3 else 0
        # VLM one-shot selection is currently disabled: the pipeline only runs
        # the argmax path, or the agentic recovery loop when it is explicitly
        # enabled. The recovery loop can use either the VLM selector or the
        # no-VLM dummy selector configured by AgenticDrivingConfig.selector.
        if strategy == "argmax":
            chosen_strategy = self.argmax_strategy
        elif self.agenticdriving_cfg.enabled and self.vlm_selector is not None:
            chosen_strategy = self.recovery_strategy
        else:
            if self.agenticdriving_cfg.enabled:
                self.logger.warning(
                    "AgenticDriving enabled but no recovery selector is available; using argmax"
                )
            chosen_strategy = self.argmax_strategy

        outcome = chosen_strategy.select(
            proposals=proposals,
            scores=scores,
            output_num_poses=output_num_poses,
            obs=obs,
            info=info,
            info_history=info_history,
            privileged_info=privileged_info,
        )
        if outcome.advance_frame:
            self.frame_index += 1

        self.previous_selected_plan = np.asarray(outcome.selected_plan, dtype=np.float32).copy()
        self.previous_selected_pose = info_to_pose(info)
        self.previous_selected_score = outcome.selected_score_raw
        self.previous_selected_timestamp = float(info.get("timestamp", 0.0))
        self.previous_selected_source = outcome.selected_source
        return outcome.plan_payload
