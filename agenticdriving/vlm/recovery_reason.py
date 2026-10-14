"""One-shot VLM explanation of why the verifier just triggered recovery.

Decoupled from :class:`agenticdriving.scorer.vlm_selector.VLMPlanSelector`: it
runs regardless of ``agenticdriving.selector`` (dummy or vlm), since it only
*explains* the verifier's decision rather than participating in candidate
selection. Wired up via ``setup_recovery_explainer`` in
``adapters/carla_agent/agenticdriving_carla_helper.py``, configured from the
yaml ``recovery_reason:`` block (``enabled: true`` to turn it on).
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from agenticdriving.adapters.carla.context import (
    describe_task_target_hint,
    extract_current_ego_speed_mps,
    resolve_route_instruction,
    resolve_route_waypoint,
    resolve_vlm_camera_order,
)
from agenticdriving.adapters.carla.overlays import render_candidate_overlays
from agenticdriving.prompts.recovery import (
    RECOVERY_PLAN_SYSTEM_PROMPT,
    build_recovery_plan_prompt,
    build_recovery_primitive_selection_prompt,
    build_recovery_reason_prompt,
    build_recovery_sequence_prompt,
    recovery_plan_bypass_first_system_prompt,
)
from agenticdriving.prompts.semantic_verifier import build_semantic_verifier_prompt
from agenticdriving.prompts.stop_sign import build_stop_sign_prompt
from agenticdriving.planners.fallback_rule_based.primitives import (
    recovery_lateral_distance_range,
)
from agenticdriving.scorer.candidates import build_candidate_rows
from agenticdriving.vlm.debug import append_episode_vlm_summary, append_jsonl
from agenticdriving.vlm.parsing import (
    coerce_recovery_reason_result,
    coerce_recovery_plan_result,
    coerce_recovery_selection_result,
    coerce_recovery_sequence_result,
    coerce_semantic_verifier_result,
    coerce_stop_sign_result,
    empty_token_usage,
    normalize_token_usage,
    try_parse_json,
)

LOG = logging.getLogger(__name__)


class RecoveryQueryBudgetExhausted(RuntimeError):
    """No inference was issued; this is not a VLM query/error summary entry."""


@dataclass
class RecoveryReasonConfig:
    enabled: bool = False
    # When True, the same VLM also PICKS the recovery primitive from the
    # recovery candidate set (see RecoveryReasonExplainer.select_primitive),
    # consuming the failure reason produced by explain() as context. Independent
    # of ``enabled``: it can run without the reason, but the intended setup is
    # both on so the model reasons once about the failure and reuses that here.
    select_primitive: bool = False
    # When enabled, recovery selection returns a bounded list of whitelisted
    # actions rather than exactly one ladder primitive. The whole materialized
    # sequence still passes through the existing final PDMS and commit/hold path.
    select_sequence: bool = False
    # Concise replacement for sequence prompting: explain the visible blockage,
    # choose an escape direction, then rank 1--5 preverified primitives. When
    # enabled it supersedes ``select_sequence``.
    select_recovery_plan: bool = False
    # Optional mutually exclusive recovery policy: first decide whether a
    # complete adjacent-lane pass-and-return maneuver is needed. The borrowed
    # lane may be same-direction or opposing. A true decision may select only
    # that primitive; false falls back to the regular safe menu.
    recovery_plan_bypass_first: bool = False
    recovery_plan_bypass_primitive: str = "lane_bypass_left"
    # Give concise recovery-plan selection a temporal view of the scene. The
    # orchestrator continuously records the selected camera, and a recovery
    # call sends the newest frame at least this old before the current frame.
    # Zero disables the historical image and preserves single-image behavior.
    recovery_plan_temporal_image_gap_sec: float = 1.0
    # Bound memory by sampling the camera history instead of copying every
    # simulator tick. Zero records every orchestrator observation.
    recovery_plan_temporal_image_sample_sec: float = 0.2
    recovery_location_repeat_limit: int = 0
    recovery_location_reset_distance_m: float = 3.0
    # Fixed aggregate trajectory budget shared across VLM candidates according
    # to their probabilities. Parameterless primitives still have only one
    # unique plan even when their allocated budget is larger than one.
    probability_sample_multiplier: float = 30.0
    # Scorer-side test-time scaling.  The environment remains frozen while a
    # deterministic temporal tree expands speed/duration/phase schedules for
    # every primitive.  The VLM receives the reduced scorecards and chooses one
    # already verified winning leaf; it never regenerates numerical parameters.
    temporal_tree_search: bool = False
    temporal_tree_budget_per_primitive: int = 18
    temporal_tree_duration_scales: Tuple[float, ...] = (0.75, 1.0, 1.25)
    # Short frozen-world delays searched before a non-stop primitive begins.
    # Each delay is materialized as a hard-brake/stationary prefix. Dynamic
    # lane-bypass waiting remains owned by its measured traffic-gap latch.
    temporal_tree_start_delays_sec: Tuple[float, ...] = (0.0, 0.5, 1.0, 1.5)
    temporal_tree_bypass_target_speeds_mps: Tuple[float, ...] = (
        6.0, 8.0, 10.0, 12.0,
    )
    temporal_tree_bypass_accels_mps2: Tuple[float, ...] = (3.5, 4.5)
    temporal_tree_bypass_total_steps: Tuple[int, ...] = (8, 9, 10, 12)
    temporal_tree_bypass_entry_steps: Tuple[int, ...] = (2, 3, 4)
    temporal_tree_bypass_return_steps: Tuple[int, ...] = (2, 3)
    temporal_tree_bypass_settle_steps: Tuple[int, ...] = (1, 2)
    # Posture mode: cap back_left/back_right lateral requests (0.3 => ~7 deg
    # of nose rotation with the swept footprint provably in-lane). None (the
    # default) keeps the full spec range.
    sequence_back_steer_max_lateral_m: Optional[float] = None
    # Exposure caution: reserved gate for marking the ego's position as
    # exposed (road_context["exposed_position"] renders a stopping-here-is-
    # dangerous caution in the sequence prompt). No in-orchestrator trigger
    # currently sets the key; wiring a trigger is a separate decision.
    sequence_exposure_guidance: bool = False
    # Optional subset of allowed_primitives exposed to the VLM's sequence
    # menu; the fallback ladder keeps the full allowed set. Empty = menu
    # equals the allowed set (shipped behavior).
    sequence_menu_primitives: Tuple[str, ...] = ()
    # Optional second-stage menu opened after the ego remains within a small
    # position radius for the configured simulator-time interval. Empty/zero
    # disables escalation.
    sequence_menu_escalation_primitives: Tuple[str, ...] = ()
    sequence_menu_escalate_after_stationary_sec: float = 0.0
    sequence_menu_stationary_position_tolerance_m: float = 0.5
    # Immediately expose the escalation primitives when a non-vehicle,
    # non-pedestrian obstacle overlaps the ego corridor within this bumper gap.
    # Zero disables the scene trigger while retaining stationary escalation.
    sequence_menu_front_obstacle_distance_m: float = 0.0
    # Also expose the escalation primitives when the same vehicle remains in
    # that front corridor with negligible speed and world-position drift for
    # this many simulator seconds. Zero disables stopped-vehicle tracking.
    sequence_menu_front_stopped_vehicle_stationary_sec: float = 0.0
    sequence_menu_front_stopped_vehicle_speed_threshold_mps: float = 0.05
    sequence_menu_front_stopped_vehicle_position_tolerance_m: float = 0.05
    # Hide the full left bypass for a nearby obstruction offset toward the
    # right edge of the route lane. Other recovery choices retain their gates.
    sequence_menu_mask_roadside_lane_bypass_left: bool = True
    sequence_menu_roadside_offset_ratio: float = 0.25
    sequence_menu_roadside_distance_m: float = 10.0
    # Deterministic tool rule for persistent, verified front blockages. Once
    # the obstacle/menu dwell reaches the threshold, bypass the visual chooser
    # and commit the first configured lateral primitive whose nominal rollout
    # has already passed PDMS, the HD-map footprint gate, and the no-progress
    # tabu filter. Empty/false preserves the VLM-only behavior.
    tool_rule_blocked_escape_enabled: bool = False
    tool_rule_blocked_escape_after_sec: float = 10.0
    tool_rule_blocked_escape_primitives: Tuple[str, ...] = (
        "lane_bypass_left",
        "lane_bypass_right",
        "lane_left",
        "lane_right",
        "swerve_left",
        "swerve_right",
    )
    sequence_max_steps: int = 4
    # Road-context prompts (default off => prompts byte-identical to this
    # stack without the feature). When True the orchestrator computes ego
    # on/off-corridor state, bearing+distance to the nearest driving lane, and
    # a per-candidate ends_on_driving_lane flag, and both recovery prompts
    # include them. Motivated by ep 1023: without road context the VLM reports
    # "blocked by foliage" while the ego is 7m off-road and recovery walks it
    # further out (52 events, deviation 4->21m); with it, 5 events and a
    # monotonic 7.2->3.1m return.
    road_context: bool = False
    # v2 graded signals (requires road_context): off-lane candidates also carry
    # distance_to_lane_m, rows are ordered best-return-first, and the argmin is
    # tagged closest_to_lane when NO candidate reaches the lane (the binary
    # flag saturates beyond the <=3m primitive envelopes).
    road_context_distance: bool = False
    # Brake was historically a hidden last resort. Recovery Table 2 exposes it
    # as a real B1/B4/B5 choice, so those configs opt in explicitly.
    include_brake: bool = False
    # B1-B4 choose only a discrete primitive. B5 additionally lets the VLM
    # choose the validated lateral-distance parameter for supported families.
    select_parameters: bool = True
    # Progressive recovery ablations opt in; preserve legacy behavior otherwise.
    ignore_parameters_when_disabled: bool = False
    force_brake_on_selection_exhausted: bool = False
    exclude_rejected_primitives_on_retry: bool = False
    # B5 can pre-sweep each parameterized primitive with PDMS and expose only
    # the widest sampled PDMS-threshold-safe interval. This makes selection one-shot.
    preverify_parameter_ranges: bool = False
    parameter_range_samples: int = 21
    camera_mode: str = "front_only"
    backend: str = "anthropic"
    model_id: str = "claude-opus-5"
    api_key_env: str = "ANTHROPIC_API_KEY"
    claude_effort: str = "low"
    # Responses limits include reasoning as well as visible output tokens.
    # Per-episode recovery inference attempts (including errors). -1 = unlimited,
    # 0 = random/default-parameter recovery immediately. STOP/semantic are separate.
    max_queries_per_episode: int = -1
    device: str = "auto"
    python_bin: str = ""
    max_new_tokens: int = 220
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 20
    enable_thinking: bool = False
    timeout_sec: float = 20.0
    # Loading an 8B VLM (disk -> GPU) is much slower than a single inference,
    # especially with several shards warming up at once, so preload gets its own
    # generous budget instead of reusing the per-inference timeout_sec (a 20s
    # inference budget was killing cold model loads mid-flight).
    preload_timeout_sec: float = 180.0
    select_max_new_tokens: int = 400
    select_timeout_sec: float = 20.0
    select_max_attempts: int = 3
    save_debug_artifacts: bool = True
    # explain() and select_primitive() now write to two sibling folders under
    # the run output dir instead of one shared "recovery_reason_debug" dir, so
    # the two tasks' artifacts (json + overlay jpg + jsonl log) never mix.
    explain_dir_name: str = "recovery_explain"
    select_dir_name: str = "recovery_select"
    log_file_name: str = "recovery_reason.jsonl"
    preload_on_init: bool = True


class RecoveryReasonExplainer:
    """Explains a single verifier-triggered recovery event via one VLM call."""

    def __init__(
        self,
        cfg: RecoveryReasonConfig,
        output_dir: Path,
        *,
        extra_active: bool = False,
    ) -> None:
        self.cfg = cfg
        from agenticdriving.vlm.location_actions import LocationActionMemory
        self.location_actions = LocationActionMemory(
            cfg.recovery_location_repeat_limit, cfg.recovery_location_reset_distance_m)
        self._extra_active = bool(extra_active)
        self.output_dir = Path(output_dir)
        self.episode_output_dir = (
            self.output_dir.parent.parent
            if self.output_dir.parent.name == "scenarios"
            else self.output_dir
        )
        self.explain_dir = self.output_dir / cfg.explain_dir_name
        self.select_dir = self.output_dir / cfg.select_dir_name
        self.explain_log_path = self.explain_dir / cfg.log_file_name
        self.select_log_path = self.select_dir / cfg.log_file_name
        self._sequence_call_counts: Dict[int, int] = {}
        self._recovery_camera_history = deque()
        self._recovery_camera_history_last_ts: Optional[float] = None
        self._selector: Optional[object] = None
        self._disabled_reason: Optional[str] = None
        self._recovery_query_count = 0
        self._recovery_query_lock = threading.Lock()
        self._recovery_budget_path = self.episode_output_dir / "recovery_query_budget.json"
        if cfg.max_queries_per_episode >= 0:
            # An infrastructure retry of the same episode keeps its API budget.
            # Also migrate runs that have summaries but predate this journal.
            for path in (self._recovery_budget_path, self.episode_output_dir / "vlm_summary.json"):
                if not path.exists():
                    continue
                try:
                    saved = json.loads(path.read_text())
                    count = (int(saved["attempts"]) if path == self._recovery_budget_path else
                             sum(str(c.get("task", "")).startswith("recovery_")
                                 for c in saved.get("calls", [])))
                    self._recovery_query_count = max(self._recovery_query_count, count)
                except (OSError, ValueError, KeyError, TypeError):
                    # A damaged journal must not reset an already spent budget.
                    self._recovery_query_count = cfg.max_queries_per_episode
                    LOG.warning("Unreadable recovery budget history; preserving the query cap")
        # Semantic and stop-sign checks are daemonized and share one backend.
        # Serialize those optional calls so a subprocess socket never receives
        # overlapping requests. The orchestrator also reads the busy property
        # to skip lower-priority recovery VLM work while either check is active.
        self._optional_inference_lock = threading.Lock()
        self._optional_busy_lock = threading.Lock()
        self._optional_busy_count = 0
        if self.active and cfg.save_debug_artifacts:
            self.explain_dir.mkdir(parents=True, exist_ok=True)
            self.select_dir.mkdir(parents=True, exist_ok=True)

    @property
    def active(self) -> bool:
        """True when either the explanation or the primitive selection is on;
        both share one VLM backend, so either turns the model on."""
        return bool(
            self.cfg.enabled
            or self.cfg.select_primitive
            or self.cfg.select_sequence
            or self.cfg.select_recovery_plan
            or self._extra_active
        )

    @property
    def optional_async_busy(self) -> bool:
        with self._optional_busy_lock:
            return self._optional_busy_count > 0

    @property
    def recovery_query_count(self) -> int:
        with self._recovery_query_lock:
            return self._recovery_query_count

    @property
    def recovery_query_budget_exhausted(self) -> bool:
        limit = self.cfg.max_queries_per_episode
        return limit >= 0 and self.recovery_query_count >= limit

    def _infer_recovery_prompt(self, selector: object, **kwargs) -> Dict[str, Any]:
        with self._recovery_query_lock:
            limit = self.cfg.max_queries_per_episode
            if limit >= 0 and self._recovery_query_count >= limit:
                raise RecoveryQueryBudgetExhausted("recovery_query_budget_exhausted")
            # Count attempts before invocation, so timeouts/bad JSON cannot
            # reopen the budget. Persist before network I/O so a killed request
            # still counts when the same episode is restarted.
            self._recovery_query_count += 1
            if limit >= 0:
                self.episode_output_dir.mkdir(parents=True, exist_ok=True)
                temporary = self._recovery_budget_path.with_suffix(".tmp")
                temporary.write_text(json.dumps({"attempts": self._recovery_query_count}))
                temporary.replace(self._recovery_budget_path)
        return selector.infer_prompt(**kwargs)

    def _infer_optional_prompt(self, selector: object, **kwargs) -> Dict[str, Any]:
        with self._optional_busy_lock:
            self._optional_busy_count += 1
        try:
            with self._optional_inference_lock:
                return selector.infer_prompt(**kwargs)
        finally:
            with self._optional_busy_lock:
                self._optional_busy_count = max(0, self._optional_busy_count - 1)

    def _normalized_backend(self) -> str:
        return str(self.cfg.backend or "").strip()

    def _ensure_selector(self) -> Optional[object]:
        if self._selector is not None:
            return self._selector
        if self._disabled_reason is not None:
            return None
        backend = self._normalized_backend()
        if backend == "anthropic":
            from agenticdriving.vlm.claude import ClaudeTrajectorySelector
            selector_factory = lambda: ClaudeTrajectorySelector(
                model_id=self.cfg.model_id,
                max_new_tokens=self.cfg.max_new_tokens,
                api_key_env=getattr(self.cfg, "api_key_env", "ANTHROPIC_API_KEY"),
                effort=getattr(self.cfg, "claude_effort", "low"),
                enable_thinking=getattr(self.cfg, "enable_thinking", False),
            )
        else:
            self._disabled_reason = f"unsupported_backend:{self.cfg.backend}"
            return None
        try:
            self._selector = selector_factory()
            LOG.info(
                "Initialized recovery-reason VLM backend=%s model=%s device=%s",
                backend, self.cfg.model_id, getattr(self._selector, "device", self.cfg.device),
            )
            return self._selector
        except Exception as exc:
            self._disabled_reason = str(exc)
            LOG.exception("Failed to initialize recovery-reason VLM backend")
            return None

    def preload(self, timeout_sec: Optional[float] = None) -> None:
        if not self.active or not self.cfg.preload_on_init:
            return
        selector = self._ensure_selector()
        if selector is None:
            return
        preload_fn = getattr(selector, "preload", None)
        if preload_fn is not None:
            preload_fn(timeout_sec=timeout_sec if timeout_sec is not None else self.cfg.preload_timeout_sec)

    def _log(self, record: Dict[str, object], log_path: Path) -> None:
        if self.cfg.max_queries_per_episode >= 0:
            record["recovery_query_count"] = self.recovery_query_count
            record["recovery_query_limit"] = self.cfg.max_queries_per_episode
        if not self.active or not self.cfg.save_debug_artifacts:
            return
        try:
            append_jsonl(log_path, record)
        except Exception:
            LOG.exception("Failed to append recovery-reason debug log")

    def observe_camera_frame(
        self,
        camera_images: Dict[str, np.ndarray],
        timestamp: float,
    ) -> None:
        """Retain sampled front-camera frames for temporal recovery reasoning."""
        if not self.cfg.select_recovery_plan:
            return
        gap_sec = max(
            0.0, float(self.cfg.recovery_plan_temporal_image_gap_sec or 0.0)
        )
        if gap_sec <= 0.0:
            return
        camera_order = resolve_vlm_camera_order(self.cfg.camera_mode)
        # Temporal recovery is deliberately a two-image input, not two full
        # multiview bundles. Keep existing multiview behavior unchanged.
        if tuple(camera_order) != ("CAM_FRONT",):
            return
        frame = camera_images.get("CAM_FRONT")
        if frame is None:
            return
        try:
            now = float(timestamp)
        except (TypeError, ValueError):
            return
        last_ts = self._recovery_camera_history_last_ts
        if last_ts is not None and now + 1e-6 < last_ts:
            self._recovery_camera_history.clear()
            last_ts = None
        sample_sec = max(
            0.0,
            float(self.cfg.recovery_plan_temporal_image_sample_sec or 0.0),
        )
        if last_ts is not None and now - last_ts + 1e-6 < sample_sec:
            return
        self._recovery_camera_history.append(
            (now, {"CAM_FRONT": np.asarray(frame).copy()})
        )
        self._recovery_camera_history_last_ts = now
        retention_sec = gap_sec + max(1.0, 2.0 * sample_sec)
        cutoff = now - retention_sec
        while (
            len(self._recovery_camera_history) > 1
            and self._recovery_camera_history[1][0] < cutoff
        ):
            self._recovery_camera_history.popleft()

    def _recovery_plan_historical_frame(
        self,
        timestamp: float,
        camera_order: Sequence[str],
    ) -> Optional[Tuple[float, Dict[str, np.ndarray]]]:
        if tuple(camera_order) != ("CAM_FRONT",):
            return None
        gap_sec = max(
            0.0, float(self.cfg.recovery_plan_temporal_image_gap_sec or 0.0)
        )
        if gap_sec <= 0.0:
            return None
        target_ts = float(timestamp) - gap_sec
        for frame_ts, images in reversed(self._recovery_camera_history):
            if frame_ts <= target_ts + 1e-6:
                return float(frame_ts), images
        return None

    def _record_vlm_summary(
        self,
        *,
        task: str,
        frame_index: int,
        timestamp: float,
        input_fields: Dict[str, Any],
        inference_result: Optional[Dict[str, Any]],
        accepted_result: Dict[str, Any],
        error: Optional[str],
        image_paths: Sequence[Path] = (),
    ) -> None:
        """Write one compact, readable call without duplicating the prompt."""
        inference = dict(inference_result or {})
        raw_text = inference.get("raw_output")
        parsed_json = inference.get("parsed_output")
        try:
            append_episode_vlm_summary(
                self.episode_output_dir,
                {
                    "task": str(task),
                    "frame_index": int(frame_index),
                    "simulation_timestamp_sec": float(timestamp),
                    "status": "error" if error else "ok",
                    "input": dict(input_fields),
                    "answer": {
                        "raw_text": raw_text,
                        "parsed_json": parsed_json,
                        "accepted_result": dict(accepted_result),
                    },
                    "runtime": {
                        "elapsed_sec": float(inference.get("elapsed_sec", 0.0) or 0.0),
                        "token_usage": normalize_token_usage(
                            inference.get("token_usage")
                        ),
                        "provider_usage": inference.get("provider_usage") or {},
                        "error": error or inference.get("error"),
                    },
                    "artifacts": {
                        "images": [str(Path(path)) for path in image_paths],
                        "scenario_output_dir": str(self.output_dir),
                    },
                },
            )
        except Exception:
            LOG.exception("Failed to update episode VLM summary")

    def check_semantic_takeover(
        self,
        *,
        frame_index: int,
        camera_images: Dict[str, np.ndarray],
        info: Dict[str, object],
        selected_plan: np.ndarray,
        trigger: str,
        max_new_tokens: int,
        timeout_sec: float,
        debug_dir: Path,
        save_debug_artifacts: bool,
    ) -> Dict[str, object]:
        """Run one conservative semantic takeover check.

        Called only from the daemonized sparse-event manager.  It shares this
        object's shared Claude backend with recovery explanation and
        primitive selection, so semantic supervision does not load a second 8B
        model.  The orchestrator skips optional recovery-VLM work if this call
        is still in flight when PDMS hard-triggers.
        """
        timestamp = float(info.get("timestamp", 0.0) or 0.0)
        route_instruction = resolve_route_instruction(info)

        def _result(
            *,
            on_track: bool = True,
            confidence: Optional[float] = None,
            severity: str = "none",
            reason_code: str = "UNCERTAIN",
            takeover_recommended: bool = False,
            reasoning: Optional[str] = None,
            error: Optional[str] = None,
            elapsed_sec: float = 0.0,
            token_usage: Optional[Dict[str, int]] = None,
            raw: Optional[Dict[str, object]] = None,
        ) -> Dict[str, object]:
            return {
                "semantic_verifier_accepted": bool(on_track),
                "semantic_verifier_on_track": bool(on_track),
                "semantic_verifier_confidence": confidence,
                "semantic_verifier_severity": str(severity),
                "semantic_verifier_reason_code": str(reason_code),
                "semantic_verifier_takeover_recommended": bool(takeover_recommended),
                "semantic_verifier_reasoning": reasoning,
                "semantic_verifier_error": error,
                "semantic_verifier_elapsed_sec": float(elapsed_sec),
                "semantic_verifier_token_usage": token_usage or empty_token_usage(),
                "semantic_verifier_raw": raw,
                "route_instruction": route_instruction,
                "trigger": str(trigger),
                "frame_index": int(frame_index),
                "timestamp": timestamp,
            }

        selector = self._ensure_selector()
        if selector is None:
            return _result(error=self._disabled_reason or "selector_unavailable")

        camera_order = resolve_vlm_camera_order(self.cfg.camera_mode)
        candidate_rows = build_candidate_rows(
            [{"source": "basic_loop_selected", "local_plan": np.asarray(selected_plan, dtype=np.float32)}],
            current_ego_speed_mps=extract_current_ego_speed_mps(info),
        )
        try:
            overlays = render_candidate_overlays(
                camera_images,
                info,
                candidate_rows,
                camera_order=camera_order,
            )
        except Exception as exc:
            LOG.exception("Failed to render semantic-takeover overlay")
            return _result(error=f"overlay_failed:{exc}")

        debug_dir = Path(debug_dir)
        if save_debug_artifacts:
            debug_dir.mkdir(parents=True, exist_ok=True)
        trigger_stem = "".join(
            ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(trigger)
        )
        frame_stem = f"frame_{int(frame_index):06d}_{trigger_stem}"
        temp_paths: List[Path] = []
        image_paths: List[Path] = []
        for cam_name in camera_order:
            overlay = overlays.get(cam_name)
            if overlay is None:
                continue
            if save_debug_artifacts:
                image_path = debug_dir / f"{frame_stem}_{cam_name}.jpg"
            else:
                image_path = self.output_dir / f"{frame_stem}_{cam_name}_tmp.jpg"
                temp_paths.append(image_path)
            cv2.imwrite(str(image_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            image_paths.append(image_path)

        prompt = build_semantic_verifier_prompt(
            candidate_rows[0],
            route_instruction,
            task_target_hint=describe_task_target_hint(info),
            trigger_reason=str(trigger),
            camera_order=camera_order,
        )
        selected_plan_array = np.asarray(selected_plan, dtype=np.float32)
        summary_input = {
            "trigger": str(trigger),
            "route_instruction": route_instruction,
            "camera_order": list(camera_order),
            "selected_plan": {
                "pose_count": int(len(selected_plan_array)),
                "start_xy": (
                    selected_plan_array[0, :2].tolist()
                    if len(selected_plan_array) else None
                ),
                "end_xy": (
                    selected_plan_array[-1, :2].tolist()
                    if len(selected_plan_array) else None
                ),
            },
        }
        started = time.time()
        try:
            inference_result = self._infer_optional_prompt(
                selector,
                image_paths=image_paths,
                prompt=prompt,
                max_new_tokens=max(1, int(max_new_tokens)),
                timeout_sec=max(0.1, float(timeout_sec)),
            )
        except Exception as exc:
            LOG.exception("Semantic-takeover VLM call failed")
            for temp_path in temp_paths:
                temp_path.unlink(missing_ok=True)
            self._record_vlm_summary(
                task="semantic_route_check",
                frame_index=frame_index,
                timestamp=timestamp,
                input_fields=summary_input,
                inference_result={"elapsed_sec": time.time() - started},
                accepted_result={"on_track": True},
                error=str(exc),
                image_paths=image_paths,
            )
            return _result(error=str(exc), elapsed_sec=time.time() - started)

        elapsed_sec = float(inference_result.get("elapsed_sec", time.time() - started))
        token_usage = normalize_token_usage(inference_result.get("token_usage"))
        parsed = coerce_semantic_verifier_result(
            inference_result.get("parsed_output")
            or try_parse_json(str(inference_result.get("raw_output", "")))
        )
        error = parsed.error
        if elapsed_sec > float(timeout_sec):
            error = f"semantic_takeover_timeout:{elapsed_sec:.3f}"
        elif inference_result.get("error"):
            error = str(inference_result.get("error"))

        for temp_path in temp_paths:
            temp_path.unlink(missing_ok=True)
        if save_debug_artifacts:
            result_path = debug_dir / f"{frame_stem}.json"
            result_path.write_text(
                json.dumps(
                    {"inference_result": inference_result},
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
        accepted_result = {
            "on_track": True if error is not None else parsed.on_track,
            "confidence": parsed.confidence,
            "severity": parsed.severity,
            "reason_code": parsed.reason_code,
            "takeover_recommended": parsed.takeover_recommended,
            "reasoning": parsed.reasoning,
        }
        self._record_vlm_summary(
            task="semantic_route_check",
            frame_index=frame_index,
            timestamp=timestamp,
            input_fields=summary_input,
            inference_result=inference_result,
            accepted_result=accepted_result,
            error=error,
            image_paths=image_paths,
        )
        return _result(
            on_track=accepted_result["on_track"],
            confidence=parsed.confidence,
            severity=parsed.severity,
            reason_code=parsed.reason_code,
            takeover_recommended=parsed.takeover_recommended,
            reasoning=parsed.reasoning,
            error=error,
            elapsed_sec=elapsed_sec,
            token_usage=token_usage,
            raw=parsed.raw,
        )

    def check_perception_box(self, **kwargs):
        """Optional object arbitration sharing this episode's serialized VLM."""
        from agenticdriving.vlm.perception_box import check_perception_box
        # Mark the whole query busy, including image preparation/backend
        # initialization, so lower-priority recovery does not race the query.
        with self._optional_busy_lock:
            self._optional_busy_count += 1
        try:
            return check_perception_box(self, **kwargs)
        finally:
            with self._optional_busy_lock:
                self._optional_busy_count -= 1

    def check_stop_sign(
        self,
        *,
        frame_index: int,
        camera_images: Dict[str, np.ndarray],
        info: Dict[str, object],
        junction_distance_m: float,
        junction_key: str,
        camera_mode: str,
        max_new_tokens: int,
        timeout_sec: float,
        debug_dir: Path,
        save_debug_artifacts: bool,
    ) -> Dict[str, object]:
        """Run one raw-image stop-sign classification for a mapped junction."""
        timestamp = float(info.get("timestamp", 0.0) or 0.0)
        route_instruction = resolve_route_instruction(info)

        def _result(
            *,
            visible: bool = False,
            applies: bool = False,
            must_stop: bool = False,
            confidence: Optional[float] = None,
            reasoning: Optional[str] = None,
            error: Optional[str] = None,
            elapsed_sec: float = 0.0,
            token_usage: Optional[Dict[str, int]] = None,
            raw: Optional[Dict[str, object]] = None,
        ) -> Dict[str, object]:
            return {
                "stop_sign_visible": bool(visible),
                "applies_to_ego_lane": bool(applies),
                "must_stop": bool(must_stop),
                "confidence": confidence,
                "reasoning": reasoning,
                "stop_sign_error": error,
                "elapsed_sec": float(elapsed_sec),
                "token_usage": token_usage or empty_token_usage(),
                "raw": raw,
                "route_instruction": route_instruction,
                "junction_key": str(junction_key),
                "junction_distance_m": float(junction_distance_m),
                "frame_index": int(frame_index),
                "timestamp": timestamp,
            }

        selector = self._ensure_selector()
        if selector is None:
            return _result(error=self._disabled_reason or "selector_unavailable")

        camera_order = resolve_vlm_camera_order(camera_mode)
        debug_dir = Path(debug_dir)
        if save_debug_artifacts:
            debug_dir.mkdir(parents=True, exist_ok=True)
        frame_stem = f"frame_{int(frame_index):06d}_{str(junction_key).replace(':', '_')}"
        image_paths: List[Path] = []
        temp_paths: List[Path] = []
        for camera_name in camera_order:
            image = camera_images.get(camera_name)
            if image is None:
                continue
            if save_debug_artifacts:
                image_path = debug_dir / f"{frame_stem}_{camera_name}.jpg"
            else:
                image_path = self.output_dir / f"{frame_stem}_{camera_name}_tmp.jpg"
                temp_paths.append(image_path)
            cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            image_paths.append(image_path)
        if not image_paths:
            return _result(error="no_stop_sign_camera_images")

        prompt = build_stop_sign_prompt(
            route_instruction,
            junction_distance_m=float(junction_distance_m),
            camera_order=camera_order,
        )
        summary_input = {
            "route_instruction": route_instruction,
            "junction_key": str(junction_key),
            "junction_distance_m": float(junction_distance_m),
            "camera_order": list(camera_order),
        }
        started = time.time()
        try:
            inference_result = self._infer_optional_prompt(
                selector,
                image_paths=image_paths,
                prompt=prompt,
                max_new_tokens=max(1, int(max_new_tokens)),
                timeout_sec=max(0.1, float(timeout_sec)),
            )
        except Exception as exc:
            LOG.exception("Stop-sign VLM call failed")
            for temp_path in temp_paths:
                temp_path.unlink(missing_ok=True)
            self._record_vlm_summary(
                task="stop_sign_check",
                frame_index=frame_index,
                timestamp=timestamp,
                input_fields=summary_input,
                inference_result={"elapsed_sec": time.time() - started},
                accepted_result={"must_stop": False},
                error=str(exc),
                image_paths=image_paths,
            )
            return _result(error=str(exc), elapsed_sec=time.time() - started)

        elapsed_sec = float(inference_result.get("elapsed_sec", time.time() - started))
        token_usage = normalize_token_usage(inference_result.get("token_usage"))
        parsed = coerce_stop_sign_result(
            inference_result.get("parsed_output")
            or try_parse_json(str(inference_result.get("raw_output", "")))
        )
        error = parsed.error
        if elapsed_sec > float(timeout_sec):
            error = f"stop_sign_timeout:{elapsed_sec:.3f}"
        elif inference_result.get("error"):
            error = str(inference_result.get("error"))

        for temp_path in temp_paths:
            temp_path.unlink(missing_ok=True)
        if save_debug_artifacts:
            result_path = debug_dir / f"{frame_stem}.json"
            result_path.write_text(
                json.dumps({"inference_result": inference_result}, indent=2, default=str),
                encoding="utf-8",
            )
        accepted_result = {
            "stop_sign_visible": parsed.stop_sign_visible,
            "applies_to_ego_lane": parsed.applies_to_ego_lane,
            "must_stop": False if error is not None else parsed.must_stop,
            "confidence": parsed.confidence,
            "reasoning": parsed.reasoning,
        }
        self._record_vlm_summary(
            task="stop_sign_check",
            frame_index=frame_index,
            timestamp=timestamp,
            input_fields=summary_input,
            inference_result=inference_result,
            accepted_result=accepted_result,
            error=error,
            image_paths=image_paths,
        )
        return _result(
            visible=parsed.stop_sign_visible,
            applies=parsed.applies_to_ego_lane,
            must_stop=accepted_result["must_stop"],
            confidence=parsed.confidence,
            reasoning=parsed.reasoning,
            error=error,
            elapsed_sec=elapsed_sec,
            token_usage=token_usage,
            raw=parsed.raw,
        )

    def explain(
        self,
        *,
        frame_index: int,
        camera_images: Dict[str, np.ndarray],
        info: Dict[str, object],
        selected_plan: np.ndarray,
        verifier_score: float,
        verifier_threshold: float,
        verifier_gates: Dict[str, float],
        verifier_soft: Dict[str, float],
        road_context: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        """Ask the VLM to explain, in one shot, why recovery just triggered.

        Best-effort: any failure (VLM disabled/unavailable, timeout, bad JSON)
        degrades to a result with ``reason=None`` and an ``error`` string; the
        caller should never block on this.
        """
        timestamp = float(info.get("timestamp", 0.0) or 0.0)
        route_instruction = resolve_route_instruction(info)

        def _result(
            *,
            reason: Optional[str],
            primary_cause: Optional[str] = None,
            confidence: Optional[float] = None,
            error: Optional[str] = None,
            elapsed_sec: float = 0.0,
            token_usage: Optional[Dict[str, int]] = None,
        ) -> Dict[str, object]:
            record = {
                "frame_index": frame_index,
                "timestamp": timestamp,
                "route_instruction": route_instruction,
                "verifier_score": verifier_score,
                "verifier_threshold": verifier_threshold,
                "verifier_gates": dict(verifier_gates),
                "verifier_soft": dict(verifier_soft),
                "reason": reason,
                "primary_cause": primary_cause,
                "confidence": confidence,
                "error": error,
                "elapsed_sec": elapsed_sec,
                "token_usage": token_usage or empty_token_usage(),
            }
            self._log(record, self.explain_log_path)
            return record

        if not self.cfg.enabled:
            return _result(reason=None, error="recovery_reason_disabled")
        if self.recovery_query_budget_exhausted:
            return _result(reason=None, error="recovery_query_budget_exhausted")
        selector = self._ensure_selector()
        if selector is None:
            return _result(reason=None, error=self._disabled_reason or "selector_unavailable")

        camera_order = resolve_vlm_camera_order(self.cfg.camera_mode)
        candidate_rows = build_candidate_rows(
            [{"source": "rejected_plan", "local_plan": np.asarray(selected_plan, dtype=np.float32)}],
            current_ego_speed_mps=extract_current_ego_speed_mps(info),
        )

        try:
            # No plan polyline and no task-target marker: the projected plan
            # misleads the VLM (a straight-ahead plan renders as a short hooked
            # stub), so the explain path feeds a fully clean scene and conveys
            # the blocked direction in text instead.
            overlays = render_candidate_overlays(
                camera_images, info, candidate_rows, camera_order=camera_order,
                draw_plan=False, draw_task_target=False,
            )
        except Exception as exc:
            LOG.exception("Failed to render recovery-reason overlay")
            return _result(reason=None, error=f"overlay_failed:{exc}")

        frame_stem = f"frame_{int(frame_index):04d}"
        temp_paths: List[Path] = []
        image_paths: List[Path] = []
        for cam_name in camera_order:
            overlay = overlays.get(cam_name)
            if overlay is None:
                continue
            if self.cfg.save_debug_artifacts:
                image_path = self.explain_dir / f"{frame_stem}_recovery_reason_{cam_name}.jpg"
            else:
                image_path = self.output_dir / f"{frame_stem}_recovery_reason_{cam_name}_tmp.jpg"
                temp_paths.append(image_path)
            cv2.imwrite(str(image_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            image_paths.append(image_path)

        prompt = build_recovery_reason_prompt(
            candidate_rows[0],
            verifier_score=verifier_score,
            verifier_threshold=verifier_threshold,
            verifier_gates=verifier_gates,
            verifier_soft=verifier_soft,
            camera_order=camera_order,
            road_context=road_context,
        )
        summary_input = {
            "route_instruction": route_instruction,
            "camera_order": list(camera_order),
            "verifier": {
                "score": float(verifier_score),
                "threshold": float(verifier_threshold),
                "gates": dict(verifier_gates),
                "soft": dict(verifier_soft),
            },
            "road_context": dict(road_context or {}),
        }

        started = time.time()
        try:
            inference_result = self._infer_recovery_prompt(
                selector,
                image_paths=image_paths,
                prompt=prompt,
                max_new_tokens=self.cfg.max_new_tokens,
                timeout_sec=self.cfg.timeout_sec,
            )
        except RecoveryQueryBudgetExhausted as exc:
            for temp_path in temp_paths:
                temp_path.unlink(missing_ok=True)
            return _result(reason=None, error=str(exc))
        except Exception as exc:
            LOG.exception("Recovery-reason VLM call failed")
            for temp_path in temp_paths:
                temp_path.unlink(missing_ok=True)
            self._record_vlm_summary(
                task="recovery_failure_explanation",
                frame_index=frame_index,
                timestamp=timestamp,
                input_fields=summary_input,
                inference_result={"elapsed_sec": time.time() - started},
                accepted_result={"reason": None},
                error=str(exc),
                image_paths=image_paths,
            )
            return _result(reason=None, error=str(exc), elapsed_sec=time.time() - started)

        elapsed_sec = float(inference_result.get("elapsed_sec", time.time() - started))
        token_usage = normalize_token_usage(inference_result.get("token_usage"))
        parsed = coerce_recovery_reason_result(
            inference_result.get("parsed_output") or try_parse_json(str(inference_result.get("raw_output", "")))
        )
        error = parsed.error
        if elapsed_sec > self.cfg.timeout_sec:
            error = f"recovery_reason_timeout:{elapsed_sec:.3f}"
        elif inference_result.get("error"):
            error = str(inference_result.get("error"))

        for temp_path in temp_paths:
            temp_path.unlink(missing_ok=True)
        if self.cfg.save_debug_artifacts:
            result_path = self.explain_dir / f"{frame_stem}_recovery_reason.json"
            result_path.write_text(
                json.dumps(
                    {"inference_result": inference_result},
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )

        accepted_result = {
            "reason": parsed.reason,
            "primary_cause": parsed.primary_cause,
            "confidence": parsed.confidence,
        }
        self._record_vlm_summary(
            task="recovery_failure_explanation",
            frame_index=frame_index,
            timestamp=timestamp,
            input_fields=summary_input,
            inference_result=inference_result,
            accepted_result=accepted_result,
            error=error,
            image_paths=image_paths,
        )

        return _result(
            reason=parsed.reason,
            primary_cause=parsed.primary_cause,
            confidence=parsed.confidence,
            error=error,
            elapsed_sec=elapsed_sec,
            token_usage=token_usage,
        )

    def select_primitive(
        self,
        *,
        frame_index: int,
        camera_images: Dict[str, np.ndarray],
        info: Dict[str, object],
        candidate_plans: List[np.ndarray],
        candidate_names: List[str],
        failure_reason: Optional[str],
        previous_attempts: Optional[List[str]] = None,
        failure_memory: Optional[str] = None,
        verifier_score: Optional[float] = None,
        verifier_threshold: Optional[float] = None,
        verifier_gates: Optional[Dict[str, float]] = None,
        verifier_soft: Optional[Dict[str, float]] = None,
        candidate_lateral_distance_ranges: Optional[
            List[Optional[Tuple[float, float]]]
        ] = None,
        road_context: Optional[Dict[str, object]] = None,
        candidate_road_flags: Optional[List[Optional[Dict[str, object]]]] = None,
    ) -> Dict[str, object]:
        """Pick one recovery primitive and an optional lateral-distance value.

        Reuses the SAME VLM backend as :meth:`explain` (one model load), taking
        the earlier ``failure_reason`` as context. ``candidate_*`` are parallel
        lists over the available primitive families. The caller materializes the
        chosen parameter and runs the verifier before execution.

        Returns a dict with ``selected_index`` (position into the candidate
        lists, or ``None`` on any failure), ``reasoning`` and ``error``. The
        caller must treat ``selected_index is None`` as "fall back to your own
        default ordering" — this method never raises.
        """
        timestamp = float(info.get("timestamp", 0.0) or 0.0)
        route_instruction = resolve_route_instruction(info)
        route_waypoint = resolve_route_waypoint(info)
        num_candidates = len(candidate_names)

        def _result(
            *,
            selected_index: Optional[int],
            reasoning: Optional[str] = None,
            confidence: Optional[float] = None,
            lateral_distance_m: Optional[float] = None,
            error: Optional[str] = None,
            elapsed_sec: float = 0.0,
            token_usage: Optional[Dict[str, int]] = None,
        ) -> Dict[str, object]:
            record = {
                "kind": "select_primitive",
                "frame_index": frame_index,
                "timestamp": timestamp,
                "route_instruction": route_instruction,
                "route_waypoint": dict(route_waypoint) if route_waypoint else None,
                "candidate_names": list(candidate_names),
                "road_context": dict(road_context) if road_context else None,
                "candidate_road_flags": (
                    list(candidate_road_flags) if candidate_road_flags is not None else None
                ),
                "candidate_lateral_distance_ranges": (
                    None
                    if candidate_lateral_distance_ranges is None
                    else list(candidate_lateral_distance_ranges)
                ),
                "failure_reason": failure_reason,
                "verifier_score": verifier_score,
                "verifier_threshold": verifier_threshold,
                "verifier_gates": dict(verifier_gates or {}),
                "verifier_soft": dict(verifier_soft or {}),
                "selected_index": selected_index,
                "selected_name": (
                    candidate_names[selected_index]
                    if selected_index is not None and 0 <= selected_index < num_candidates
                    else None
                ),
                "reasoning": reasoning,
                "confidence": confidence,
                "lateral_distance_m": lateral_distance_m,
                "previous_attempts": list(previous_attempts or []),
                "failure_memory": failure_memory,
                "error": error,
                "elapsed_sec": elapsed_sec,
                "token_usage": token_usage or empty_token_usage(),
            }
            self._log(record, self.select_log_path)
            return record

        if candidate_lateral_distance_ranges is not None and len(
            candidate_lateral_distance_ranges
        ) != num_candidates:
            return _result(
                selected_index=None,
                error="candidate_lateral_distance_ranges_length_mismatch",
            )
        if num_candidates == 0:
            return _result(selected_index=None, error="no_candidates")
        if self.recovery_query_budget_exhausted:
            return _result(selected_index=None, error="recovery_query_budget_exhausted")
        selector = self._ensure_selector()
        if selector is None:
            return _result(selected_index=None, error=self._disabled_reason or "selector_unavailable")

        camera_order = resolve_vlm_camera_order(self.cfg.camera_mode)
        ego_speed = extract_current_ego_speed_mps(info)
        candidate_rows = build_candidate_rows(
            [
                {"source": f"recovery_{name}", "local_plan": np.asarray(plan, dtype=np.float32)}
                for name, plan in zip(candidate_names, candidate_plans)
            ],
            current_ego_speed_mps=ego_speed,
        )
        for row, name in zip(candidate_rows, candidate_names):
            row["primitive_name"] = name
        if candidate_lateral_distance_ranges is None:
            effective_ranges = (
                [recovery_lateral_distance_range(name) for name in candidate_names]
                if self.cfg.select_parameters
                else [None for _name in candidate_names]
            )
        else:
            effective_ranges = list(candidate_lateral_distance_ranges)
        ignore_parameters = bool(
            not self.cfg.select_parameters and self.cfg.ignore_parameters_when_disabled
        )
        if ignore_parameters:
            effective_ranges = [None] * num_candidates
        for row, lateral_range in zip(candidate_rows, effective_ranges):
            row["lateral_distance_range_m"] = lateral_range
            row["lateral_distance_range_preverified"] = bool(
                lateral_range is not None
                and self.cfg.preverify_parameter_ranges
            )
        # Road-context arm: stamp each row's endpoint road relationship
        # (ends_on_driving_lane, and distance_to_lane_m in the graded v2 arm).
        if candidate_road_flags is not None:
            for row, flag in zip(candidate_rows, candidate_road_flags):
                if isinstance(flag, dict):
                    row.update(flag)
                elif flag is not None:  # legacy bare-bool form
                    row["ends_on_driving_lane"] = bool(flag)

        try:
            # Raw scene only: candidates are conveyed as the numeric list in the
            # prompt, so we draw neither the plan polylines nor the task marker.
            overlays = render_candidate_overlays(
                camera_images, info, candidate_rows, camera_order=camera_order,
                draw_plan=False, draw_task_target=False,
            )
        except Exception as exc:
            LOG.exception("Failed to render recovery-selection overlay")
            return _result(selected_index=None, error=f"overlay_failed:{exc}")

        frame_stem = f"frame_{int(frame_index):04d}"
        attempt_stem = f"{frame_stem}_attempt_{len(previous_attempts or []) + 1:02d}"
        temp_paths: List[Path] = []
        image_paths: List[Path] = []
        for cam_name in camera_order:
            overlay = overlays.get(cam_name)
            if overlay is None:
                continue
            if self.cfg.save_debug_artifacts:
                image_path = self.select_dir / f"{attempt_stem}_recovery_select_{cam_name}.jpg"
            else:
                image_path = self.output_dir / f"{attempt_stem}_recovery_select_{cam_name}_tmp.jpg"
                temp_paths.append(image_path)
            cv2.imwrite(str(image_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            image_paths.append(image_path)

        prompt = build_recovery_primitive_selection_prompt(
            candidate_rows,
            route_instruction,
            failure_reason=failure_reason,
            previous_attempts=previous_attempts,
            failure_memory=failure_memory,
            camera_order=camera_order,
            verifier_score=verifier_score,
            verifier_threshold=verifier_threshold,
            verifier_gates=verifier_gates,
            verifier_soft=verifier_soft,
            road_context=road_context,
            route_waypoint=route_waypoint,
        )
        summary_input = {
            "route_instruction": route_instruction,
            "route_waypoint": dict(route_waypoint) if route_waypoint else None,
            "failure_reason": failure_reason,
            "failure_memory": failure_memory,
            "previous_attempts": list(previous_attempts or []),
            "camera_order": list(camera_order),
            "candidates": [
                {
                    "index": index,
                    "action": name,
                    "lateral_distance_range_m": effective_ranges[index],
                    "road_context": (
                        candidate_road_flags[index]
                        if candidate_road_flags is not None
                        and index < len(candidate_road_flags)
                        else None
                    ),
                }
                for index, name in enumerate(candidate_names)
            ],
        }

        started = time.time()
        try:
            inference_result = self._infer_recovery_prompt(
                selector,
                image_paths=image_paths,
                prompt=prompt,
                max_new_tokens=self.cfg.select_max_new_tokens,
                timeout_sec=self.cfg.select_timeout_sec,
            )
        except RecoveryQueryBudgetExhausted as exc:
            for temp_path in temp_paths:
                temp_path.unlink(missing_ok=True)
            return _result(selected_index=None, error=str(exc))
        except Exception as exc:
            LOG.exception("Recovery-selection VLM call failed")
            for temp_path in temp_paths:
                temp_path.unlink(missing_ok=True)
            self._record_vlm_summary(
                task="recovery_primitive_selection",
                frame_index=frame_index,
                timestamp=timestamp,
                input_fields=summary_input,
                inference_result={"elapsed_sec": time.time() - started},
                accepted_result={"selected_index": None, "selected_action": None},
                error=str(exc),
                image_paths=image_paths,
            )
            return _result(selected_index=None, error=str(exc), elapsed_sec=time.time() - started)

        elapsed_sec = float(inference_result.get("elapsed_sec", time.time() - started))
        token_usage = normalize_token_usage(inference_result.get("token_usage"))
        selection_output = inference_result.get("parsed_output") or try_parse_json(
            str(inference_result.get("raw_output", ""))
        )
        if ignore_parameters and isinstance(selection_output, dict):
            selection_output = dict(selection_output, lateral_distance_m=None)
        parsed = coerce_recovery_selection_result(
            selection_output,
            num_candidates=num_candidates,
            candidate_names=candidate_names,
            lateral_distance_ranges=effective_ranges,
        )
        error = parsed.error
        if elapsed_sec > self.cfg.select_timeout_sec:
            error = f"recovery_selection_timeout:{elapsed_sec:.3f}"
        elif inference_result.get("error"):
            error = str(inference_result.get("error"))
        selected_index = None if error is not None else parsed.selected_index

        for temp_path in temp_paths:
            temp_path.unlink(missing_ok=True)
        if self.cfg.save_debug_artifacts:
            result_path = self.select_dir / f"{attempt_stem}_recovery_select.json"
            debug_inference_result = {
                **inference_result,
                "camera_order": list(camera_order),
                "route_instruction": route_instruction,
                "route_waypoint": (
                    dict(route_waypoint) if route_waypoint else None
                ),
                "road_context": dict(road_context or {}),
                "failure_reason": failure_reason,
                "failure_memory": failure_memory,
                "previous_attempts": list(previous_attempts or []),
                "candidate_names": list(candidate_names),
                "candidate_lateral_distance_ranges": list(effective_ranges),
            }
            result_path.write_text(
                json.dumps(
                    {"inference_result": debug_inference_result},
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )

        accepted_result = {
            "selected_index": selected_index,
            "selected_action": (
                candidate_names[selected_index]
                if selected_index is not None
                else None
            ),
            "lateral_distance_m": parsed.lateral_distance_m,
            "reasoning": parsed.reasoning,
            "confidence": parsed.confidence,
        }
        self._record_vlm_summary(
            task="recovery_primitive_selection",
            frame_index=frame_index,
            timestamp=timestamp,
            input_fields=summary_input,
            inference_result=inference_result,
            accepted_result=accepted_result,
            error=error,
            image_paths=image_paths,
        )

        return _result(
            selected_index=selected_index,
            reasoning=parsed.reasoning,
            confidence=parsed.confidence,
            lateral_distance_m=parsed.lateral_distance_m,
            error=error,
            elapsed_sec=elapsed_sec,
            token_usage=token_usage,
        )

    def select_sequence(
        self,
        *,
        frame_index: int,
        camera_images: Dict[str, np.ndarray],
        info: Dict[str, object],
        action_specs: Dict[str, Dict[str, object]],
        failure_reason: Optional[str],
        first_action_names: Optional[Sequence[str]] = None,
        failure_memory: Optional[str] = None,
        sequence_context: Optional[Dict[str, object]] = None,
        verifier_score: Optional[float] = None,
        verifier_threshold: Optional[float] = None,
        verifier_gates: Optional[Dict[str, float]] = None,
        verifier_soft: Optional[Dict[str, float]] = None,
        road_context: Optional[Dict[str, object]] = None,
        first_action_evaluations: Optional[Dict[str, Dict[str, object]]] = None,
        previous_attempts: Optional[Sequence[str]] = None,
        memory_guidance: str = "avoid",
    ) -> Dict[str, object]:
        """Plan either the legacy sequence or one concise recovery plan."""
        timestamp = float(info.get("timestamp", 0.0) or 0.0)
        excluded = self.location_actions.exclusions(info)
        location_memory_reset = False
        action_specs = {k: v for k, v in action_specs.items() if k not in excluded}
        if first_action_names is not None:
            first_action_names = [k for k in first_action_names if k not in excluded]
        if excluded:
            failure_memory = (failure_memory or "") + "\nRepeated without progress here; unavailable: " + ", ".join(sorted(excluded)) + "."
        route_instruction = resolve_route_instruction(info)
        plan_mode = bool(self.cfg.select_recovery_plan)
        route_waypoint = None if plan_mode else resolve_route_waypoint(info)

        def _result(
            *,
            steps: Optional[List[Dict[str, Any]]] = None,
            candidates: Optional[List[Dict[str, Any]]] = None,
            front_blockage: Optional[str] = None,
            escape_direction: Optional[str] = None,
            needs_lane_bypass_left: Optional[bool] = None,
            needs_lane_bypass_right: Optional[bool] = None,
            reasoning: Optional[str] = None,
            confidence: Optional[float] = None,
            error: Optional[str] = None,
            elapsed_sec: float = 0.0,
            token_usage: Optional[Dict[str, int]] = None,
        ) -> Dict[str, object]:
            record: Dict[str, object] = {
                "kind": (
                    "select_recovery_plan" if plan_mode else "select_sequence"
                ),
                "frame_index": frame_index,
                "timestamp": timestamp,
                "route_instruction": route_instruction,
                "route_waypoint": dict(route_waypoint) if route_waypoint else None,
                "action_names": list(action_specs),
                "location_excluded_actions": sorted(excluded),
                "location_repeat_counts": dict(self.location_actions.counts),
                "location_memory_reset": location_memory_reset,
                "first_action_names": (
                    list(action_specs)
                    if first_action_names is None
                    else list(first_action_names)
                ),
                "failure_reason": failure_reason,
                "failure_memory": failure_memory,
                "sequence_context": dict(sequence_context or {}),
                "verifier_score": verifier_score,
                "verifier_threshold": verifier_threshold,
                "verifier_gates": dict(verifier_gates or {}),
                "verifier_soft": dict(verifier_soft or {}),
                "first_action_evaluations": {
                    name: dict(evaluation)
                    for name, evaluation in (first_action_evaluations or {}).items()
                },
                "previous_attempts": list(previous_attempts or []),
                "steps": list(steps or []),
                "candidates": list(candidates or []),
                "front_blockage": front_blockage,
                "escape_direction": escape_direction,
                "needs_lane_bypass_left": False if "lane_bypass_left" in excluded else needs_lane_bypass_left,
                "needs_lane_bypass_right": False if "lane_bypass_right" in excluded else needs_lane_bypass_right,
                "reasoning": reasoning,
                "confidence": confidence,
                "error": error,
                "elapsed_sec": elapsed_sec,
                "token_usage": token_usage or empty_token_usage(),
            }
            self._log(record, self.select_log_path)
            return record

        bypass_available = bool(plan_mode and self.cfg.recovery_plan_bypass_first
                                and self.cfg.recovery_plan_bypass_primitive in action_specs)
        if excluded and (not action_specs or first_action_names == [] and not bypass_available):
            self.location_actions.reset()
            location_memory_reset = True
            return _result(error="recovery_location_menu_exhausted")
        if not action_specs:
            return _result(
                error=(
                    "no_recovery_plan_actions"
                    if plan_mode
                    else "no_sequence_actions"
                )
            )
        if self.recovery_query_budget_exhausted:
            return _result(error="recovery_query_budget_exhausted")
        selector = self._ensure_selector()
        if selector is None:
            return _result(error=self._disabled_reason or "selector_unavailable")

        camera_order = resolve_vlm_camera_order(self.cfg.camera_mode)
        frame_key = int(frame_index)
        sequence_call_index = self._sequence_call_counts.get(frame_key, 0) + 1
        self._sequence_call_counts[frame_key] = sequence_call_index
        frame_stem = f"frame_{frame_key:04d}_attempt_{sequence_call_index:02d}"
        temp_paths: List[Path] = []
        image_paths: List[Path] = []
        temporal_image_age_sec: Optional[float] = None
        historical_frame = (
            self._recovery_plan_historical_frame(timestamp, camera_order)
            if plan_mode
            else None
        )
        if historical_frame is not None:
            historical_ts, historical_images = historical_frame
            historical_front = historical_images.get("CAM_FRONT")
            if (
                historical_front is not None
                and camera_images.get("CAM_FRONT") is not None
            ):
                temporal_image_age_sec = max(0.0, timestamp - historical_ts)
                if self.cfg.save_debug_artifacts:
                    history_path = self.select_dir / (
                        f"{frame_stem}_recovery_sequence_CAM_FRONT_"
                        f"tminus_{temporal_image_age_sec:.2f}s.jpg"
                    )
                else:
                    history_path = self.output_dir / (
                        f"{frame_stem}_recovery_sequence_CAM_FRONT_"
                        f"tminus_{temporal_image_age_sec:.2f}s_tmp.jpg"
                    )
                    temp_paths.append(history_path)
                cv2.imwrite(
                    str(history_path),
                    cv2.cvtColor(
                        np.asarray(historical_front), cv2.COLOR_RGB2BGR
                    ),
                )
                image_paths.append(history_path)
        for cam_name in camera_order:
            frame = camera_images.get(cam_name)
            if frame is None:
                continue
            if self.cfg.save_debug_artifacts:
                image_path = self.select_dir / f"{frame_stem}_recovery_sequence_{cam_name}.jpg"
            else:
                image_path = self.output_dir / f"{frame_stem}_recovery_sequence_{cam_name}_tmp.jpg"
                temp_paths.append(image_path)
            cv2.imwrite(
                str(image_path),
                cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR),
            )
            image_paths.append(image_path)

        bypass_action = str(
            getattr(
                self.cfg,
                "recovery_plan_bypass_primitive",
                "lane_bypass_left",
            )
            or "lane_bypass_left"
        ).strip()
        # The orchestrator may remove a directionally invalid bypass before
        # this call.  Do not let the global config re-inject that action into
        # either the system or user prompt.
        bypass_first = bool(
            plan_mode
            and getattr(self.cfg, "recovery_plan_bypass_first", False)
            and bypass_action in action_specs
        )
        system_prompt = (
            recovery_plan_bypass_first_system_prompt(bypass_action)
            if bypass_first
            else (RECOVERY_PLAN_SYSTEM_PROMPT if plan_mode else None)
        )
        if plan_mode:
            prompt = build_recovery_plan_prompt(
                route_instruction,
                action_specs=action_specs,
                first_action_names=first_action_names,
                failure_memory=failure_memory,
                sequence_context=sequence_context,
                camera_order=camera_order,
                road_context=road_context,
                previous_attempts=previous_attempts,
                first_action_evaluations=first_action_evaluations,
                bypass_first=bypass_first,
                bypass_action=bypass_action,
                temporal_image_age_sec=temporal_image_age_sec,
            )
        else:
            prompt = build_recovery_sequence_prompt(
                route_instruction,
                action_specs=action_specs,
                first_action_names=first_action_names,
                max_steps=self.cfg.sequence_max_steps,
                failure_reason=failure_reason,
                failure_memory=failure_memory,
                sequence_context=sequence_context,
                camera_order=camera_order,
                verifier_score=verifier_score,
                verifier_threshold=verifier_threshold,
                verifier_gates=verifier_gates,
                verifier_soft=verifier_soft,
                road_context=road_context,
                route_waypoint=route_waypoint,
                first_action_evaluations=first_action_evaluations,
                previous_attempts=previous_attempts,
                memory_guidance=memory_guidance,
            )
        summary_input = {
            "route_instruction": route_instruction,
            "route_waypoint": dict(route_waypoint) if route_waypoint else None,
            "failure_reason": failure_reason,
            "failure_memory": failure_memory,
            "previous_sequence": dict(sequence_context or {}),
            "camera_order": list(camera_order),
            "temporal_image_age_sec": temporal_image_age_sec,
            "first_action_names": (
                list(action_specs)
                if first_action_names is None
                else list(first_action_names)
            ),
            "available_actions": {
                name: dict(spec) for name, spec in action_specs.items()
            },
            "first_action_evaluations": {
                name: dict(evaluation)
                for name, evaluation in (first_action_evaluations or {}).items()
            },
            "previous_attempts": list(previous_attempts or []),
            "recovery_plan_bypass_first": bypass_first,
            "recovery_plan_bypass_primitive": bypass_action,
        }
        started = time.time()
        try:
            inference_result = self._infer_recovery_prompt(
                selector,
                image_paths=image_paths,
                prompt=prompt,
                system_prompt=system_prompt,
                max_new_tokens=self.cfg.select_max_new_tokens,
                timeout_sec=self.cfg.select_timeout_sec,
            )
        except RecoveryQueryBudgetExhausted as exc:
            for temp_path in temp_paths:
                temp_path.unlink(missing_ok=True)
            return _result(error=str(exc))
        except Exception as exc:
            LOG.exception(
                "%s VLM call failed",
                "Recovery-plan" if plan_mode else "Recovery-sequence",
            )
            for temp_path in temp_paths:
                temp_path.unlink(missing_ok=True)
            self._record_vlm_summary(
                task=(
                    "recovery_plan_selection"
                    if plan_mode
                    else "recovery_sequence_selection"
                ),
                frame_index=frame_index,
                timestamp=timestamp,
                input_fields=summary_input,
                inference_result={"elapsed_sec": time.time() - started},
                accepted_result={"steps": []},
                error=str(exc),
                image_paths=image_paths,
            )
            return _result(error=str(exc), elapsed_sec=time.time() - started)

        elapsed_sec = float(inference_result.get("elapsed_sec", time.time() - started))
        token_usage = normalize_token_usage(inference_result.get("token_usage"))
        raw_parsed = inference_result.get("parsed_output") or try_parse_json(
            str(inference_result.get("raw_output", ""))
        )
        if plan_mode:
            parsed_plan = coerce_recovery_plan_result(
                raw_parsed,
                action_specs=action_specs,
                first_action_names=first_action_names,
                bypass_first=bypass_first,
                bypass_action=bypass_action,
            )
            parsed = parsed_plan
        else:
            parsed = coerce_recovery_sequence_result(
                raw_parsed,
                action_specs=action_specs,
                max_steps=self.cfg.sequence_max_steps,
                first_action_names=first_action_names,
            )
        error = parsed.error
        if elapsed_sec > self.cfg.select_timeout_sec:
            error = (
                "recovery_plan_timeout"
                if plan_mode
                else "recovery_sequence_timeout"
            ) + f":{elapsed_sec:.3f}"
        elif inference_result.get("error"):
            error = str(inference_result.get("error"))

        for temp_path in temp_paths:
            temp_path.unlink(missing_ok=True)
        if self.cfg.save_debug_artifacts:
            result_kind = "recovery_plan" if plan_mode else "recovery_sequence"
            result_path = self.select_dir / f"{frame_stem}_{result_kind}.json"
            debug_inference_result = {
                **inference_result,
                "system_prompt": system_prompt,
                "camera_order": list(camera_order),
                "temporal_image_age_sec": temporal_image_age_sec,
                "route_instruction": route_instruction,
                "route_waypoint": (
                    dict(route_waypoint) if route_waypoint else None
                ),
                "road_context": dict(road_context or {}),
                "failure_reason": failure_reason,
                "failure_memory": failure_memory,
                "sequence_context": dict(sequence_context or {}),
                "action_specs": action_specs,
                "first_action_names": (
                    list(action_specs)
                    if first_action_names is None
                    else list(first_action_names)
                ),
                "first_action_evaluations": {
                    name: dict(evaluation)
                    for name, evaluation in (first_action_evaluations or {}).items()
                },
                "previous_attempts": list(previous_attempts or []),
                "max_steps": int(self.cfg.sequence_max_steps),
            }
            result_path.write_text(
                json.dumps(
                    {"inference_result": debug_inference_result},
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
        if plan_mode:
            accepted_steps = []
            accepted_candidates = (
                []
                if error is not None
                else [candidate.to_dict() for candidate in parsed.candidates]
            )
            front_blockage = parsed.front_blockage
            escape_direction = parsed.escape_direction
            needs_lane_bypass_left = parsed.needs_lane_bypass_left
            needs_lane_bypass_right = parsed.needs_lane_bypass_right
            reasoning = parsed.front_blockage
            confidence = None
        else:
            accepted_steps = (
                [] if error is not None else [step.to_dict() for step in parsed.steps]
            )
            accepted_candidates = []
            front_blockage = None
            escape_direction = None
            needs_lane_bypass_left = None
            needs_lane_bypass_right = None
            reasoning = parsed.reasoning
            confidence = parsed.confidence
        # Invalid, empty, timed-out or failed model replies are not choices.
        accepted_choices = accepted_candidates if plan_mode else accepted_steps
        if accepted_choices:
            self.location_actions.record(accepted_choices[0]["action"])
        elif (parsed.error is not None and elapsed_sec <= self.cfg.select_timeout_sec
              and not inference_result.get("error")):
            # Reject this reply, but restore the menu for the next decision.
            self.location_actions.reset()
            location_memory_reset = True
        accepted_result = {
            "steps": accepted_steps,
            "candidates": accepted_candidates,
            "front_blockage": front_blockage,
            "escape_direction": escape_direction,
            "needs_lane_bypass_left": needs_lane_bypass_left,
            "needs_lane_bypass_right": needs_lane_bypass_right,
            "first_action_executed_now": (
                accepted_steps[0] if accepted_steps else None
            ),
            "remaining_actions_saved_as_intent": accepted_steps[1:],
            "reasoning": reasoning,
            "confidence": confidence,
        }
        self._record_vlm_summary(
            task=(
                "recovery_plan_selection"
                if plan_mode
                else "recovery_sequence_selection"
            ),
            frame_index=frame_index,
            timestamp=timestamp,
            input_fields=summary_input,
            inference_result=inference_result,
            accepted_result=accepted_result,
            error=error,
            image_paths=image_paths,
        )
        return _result(
            steps=accepted_steps,
            candidates=accepted_candidates,
            front_blockage=front_blockage,
            escape_direction=escape_direction,
            needs_lane_bypass_left=needs_lane_bypass_left,
            needs_lane_bypass_right=needs_lane_bypass_right,
            reasoning=reasoning,
            confidence=confidence,
            error=error,
            elapsed_sec=elapsed_sec,
            token_usage=token_usage,
        )
