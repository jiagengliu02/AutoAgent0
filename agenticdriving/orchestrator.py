"""Per-frame orchestrator: the AgenticDriving main entry point.

This is the CARLA-era re-materialization of AutoAgent0's closed-loop body
(``autoagent0/orchestration/loop.py`` + ``main.py``). Under the leaderboard
the simulator owns the ``while`` loop and calls the agent back once per tick,
so one :meth:`Orchestrator.step` call == one iteration of the old loop, with
the stages kept explicit instead of collapsed into the selector:

    # trajectory generation   planner.process()            (PlannerService)
    # scorer                  independent proposals -> PDMS argmax per branch
    #   + select              selector.select(strategy="argmax")
    # verifier                verifier.score()             (agenticdriving.verifiers)
    # recovery                rule-based proposals -> selector.select() again
    #                         (config-driven dispatch: may run the agentic loop)
    #                         proposals come from the in-repo primitive ladder
    #                         (fallback_rule_based) by default; set
    #                         recovery provider="external" for the old
    #                         Rule-Planner repo path.

The first-round scorer stage uses no VLM. When End2End and rule-based scoring
are both enabled, End2End's native Top-K and the complete leaderboard
target-point region are evaluated independently by PDMS. Each branch proposes
its PDMS argmax and the higher-PDMS proposal advances to verification. The
agentic recovery loop only ever runs inside the recovery stage.

Planner-agnostic by construction: anything implementing
:class:`agenticdriving.planners.base.PlannerService` (RAP, DrivoR, in-process
CARLA wrappers, FIFO-backed services, ...) plugs in unchanged. Simulator
specifics — sensor parsing, control laws, recording — stay in the adapters;
this module only consumes native CARLA ``obs``/``info`` and returns
the selected plan + payload.

Note on state: like AutoAgent0, a recovery re-runs ``selector.select`` on the
rule-based proposals, so the selector's carry-previous state ends up tracking
the plan that is actually executed. When the agentic recovery loop is enabled
this advances the selector's ``frame_index`` twice in one tick (debug artifact
numbering only).
"""
from __future__ import annotations

import logging
import math
import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from agenticdriving.adapters.carla.geometry import (
    info_to_pose,
    local_plan_to_world,
    world_points_to_current_local,
)
from agenticdriving.memory import RecoveryAttemptTracker, RecoveryFailureMemory
from agenticdriving.prompts.recovery import RECOVERY_PLAN_EXCLUDED_PRIMITIVES
from agenticdriving.planners.fallback_rule_based.provider import (
    LANE_BYPASS_ENTRY_RETREAT_BASE_DISTANCE_M,
    LANE_BYPASS_SETTLE_STEPS,
    RECOVERY_PLAN_DT_SEC,
    get_recovery_sequence_action_specs,
    get_primitive_proposals_and_scores,
    materialize_hd_map_lane_bypass,
    materialize_hd_map_lane_bypass_retreat,
    materialize_recovery_action_profile,
    materialize_recovery_primitive,
    materialize_recovery_sequence,
    lane_bypass_execution_assessment,
    lane_bypass_gap_assessment,
    lane_bypass_entry_retreat_assessment,
    front_corridor_object_observations,
    roadside_lane_bypass_left_assessment,
)
from agenticdriving.planners.fallback_rule_based.recovery_waypoint import (
    materialize_recovery_waypoint,
)
from agenticdriving.planners.fallback_rule_based.primitives import (
    recovery_lateral_distance_range,
)
from agenticdriving.planners.base import PlannerService
from agenticdriving.orchestration.candidate_scoring import (
    CandidateScoringConfig,
    CandidateScoringPipeline,
)
from agenticdriving.orchestration.directional_options import (
    adjacent_driving_lane_absent,
    is_leftward_primitive,
    junction_route_heading_violation_mask,
    suppress_leftward_options,
)
from agenticdriving.orchestration.recovery.hold import RecoveryHoldMixin
from agenticdriving.orchestration.recovery.safety import RecoverySafetyMixin
from agenticdriving.orchestration.recovery.state import (
    RecoverySequenceActionManager,
    observe_position_dwell,
)
from agenticdriving.orchestration.recovery.temporal_search import (
    TemporalSearchLeaf,
    TemporalSearchResult,
    build_temporal_profile_tree,
    reduce_temporal_search,
)
from agenticdriving.scorer.scoring import BaseScorer, PlannerNativeScorer
from agenticdriving.verifiers.base import BaseVerifier

LOG = logging.getLogger(__name__)

LANE_BYPASS_CONE_CHAIN_CLEARANCE_M = 35.0
LANE_BYPASS_STATIC_TRAILING_CLEARANCE_M = 8.0


@dataclass
class StepResult:
    """Everything one orchestrator step produced, for the adapter to act on."""

    selected_plan: np.ndarray
    selected_score_raw: Optional[float]
    selected_source: str
    plan_payload: Dict[str, Any]
    proposals: np.ndarray
    scores: np.ndarray
    verifier_score: Optional[float] = None
    verifier_passed: bool = True
    recovered: bool = False
    debug: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _ForceBrakeFallback:
    """Sequence recovery failed; bypass every other fallback and brake."""

    reason: str



def _unpack_payload(plan_payload: Dict[str, Any]) -> tuple[np.ndarray, Optional[float], str]:
    selected_plan = np.asarray(plan_payload["selected_plan"], dtype=np.float32)
    selected_score = plan_payload.get("selected_score")
    selected_score_raw = None if selected_score is None else float(selected_score)
    selected_source = str(plan_payload.get("selected_source", ""))
    return selected_plan, selected_score_raw, selected_source


class Orchestrator(RecoveryHoldMixin, RecoverySafetyMixin):
    """Drives generate -> select -> verify -> recover for one frame.

    Args:
        planner: trajectory generation backend (``PlannerService``).
        selector: the pipeline-side ``LearnedPlannerSelector``. Round-1 uses it
            with ``strategy="argmax"`` over the final selection-score vector; the
            recovery stage uses its config-driven dispatch (which may run the
            agentic recovery loop).
        end2end_scorer: learned per-candidate scorer ("End2end Scorer" box).
            ``None`` disables the end2end term (rule-based-only scoring). When
            BOTH ``end2end_scorer`` and ``rule_based_scorer`` are ``None`` it
            defaults to :class:`PlannerNativeScorer`, preserving the pre-scorer
            behaviour; at least one scorer must always be active.
        instruction_scorer: instruction-compliance per-candidate gate
            ("Rule-based scorer" box), e.g. ``InstructionComplianceScorer``.
            ``None`` disables gating.
        rule_based_scorer: optional per-candidate PDMS scorer
            (:class:`RuleBasedPDMSScorer`). With the end2end scorer active, two
            independent proposal branches run: End2End sends its native Top-K
            to PDMS, while the rule-based branch randomly samples trajectories
            from every layered region and sends the union to PDMS.
            Each branch proposes its PDMS argmax and the higher of those two
            PDMS scores is sent to the verifier.
        scorer_end2end_pdms_branch: independently enable the End2End Top-K ->
            PDMS proposal branch. ``None`` preserves legacy auto-dispatch.
        scorer_target_region_pdms_branch: independently enable the all-region
            random-sampling -> PDMS branch. ``None`` preserves legacy dispatch.
        verifier: optional trajectory verifier; ``None`` disables the
            verify/recover stage (behaviour identical to the pre-verifier path).
        verifier_threshold: scores below this reject the selected trajectory.
        pedestrian_ttc_brake_hold_sec: simulator-time duration of the hard
            brake latch after an in-range pedestrian trips ``ttc_vru``.
        pedestrian_ttc_brake_distance_m: forward extent of that trigger range.
        pedestrian_ttc_brake_lateral_m: lateral half-width of that range.
        recovery_explainer: optional ``RecoveryReasonExplainer``; when set, one
            VLM call explains why the verifier rejected the trajectory the
            instant recovery triggers (independent of ``selector``/``dummy`` vs
            ``vlm`` mode). ``None`` (default) disables it, no-op on failure.
        semantic_takeover_manager: optional sparse asynchronous VLM supervisor.
            It observes only instruction-change and post-recovery events, never
            blocks loop 1, and may request the existing recovery executor after
            a conservative confirmed semantic mismatch.
        recovery_cfg: config for the rule-based recovery generator;
            recovery is skipped (with a warning) when disabled/absent.
    """

    def __init__(
        self,
        *,
        planner: PlannerService,
        selector: Any,
        end2end_scorer: Optional[BaseScorer] = None,
        instruction_scorer: Optional[BaseScorer] = None,
        rule_based_scorer: Optional[BaseScorer] = None,
        scorer_region_sector_counts: Sequence[int] = (1, 3, 5),
        scorer_end2end_topk: int = 30,
        scorer_region_samples_per_region: int = 3,
        scorer_region_random_seed: int = 0,
        scorer_target_region_max_candidates: Optional[int] = None,
        scorer_target_region_random_seed: Optional[int] = None,
        scorer_region_min_bearing_deg: float = -45.0,
        scorer_region_max_bearing_deg: float = 45.0,
        scorer_end2end_pdms_branch: Optional[bool] = None,
        scorer_target_region_pdms_branch: Optional[bool] = None,
        scorer_pdms_gate_native_argmax: bool = False,
        scorer_pdms_gate_threshold: Optional[float] = None,
        verifier: Optional[BaseVerifier] = None,
        lane_tier_probe: Optional[Any] = None,
        verifier_threshold: float = 0.5,
        pedestrian_ttc_brake_hold_sec: float = 1.0,
        pedestrian_ttc_brake_distance_m: float = 30.0,
        pedestrian_ttc_brake_lateral_m: float = 6.0,
        recovery_explainer: Optional[Any] = None,
        semantic_takeover_manager: Optional[Any] = None,
        recovery_cfg: Optional[Any] = None,
        recovery_commit_and_hold: bool = False,
        recovery_random_order: bool = True,
        recovery_seed: Optional[int] = None,
        recovery_tier_veto: bool = False,
        recovery_hd_map_endpoint_only: bool = False,
        recovery_full_plan_dac: bool = False,
        recovery_hold_dac_guard: bool = False,
        recovery_hold_dac_guard_ticks: int = 3,
        recovery_bypass_return_nc_guard: bool = False,
        lane_bypass_wait_timeout_sec: float = 30.0,
        recovery_allow_failed_primitive_escalation: bool = True,
        recovery_tie_break_order: Sequence[str] = (),
        recovery_tabu_exempt: Sequence[str] = (),
        recovery_rank_without_ep: bool = False,
        normal_ep_gate: bool = False,
        normal_ep_min: float = 0.05,
        normal_ep_gate_steps: int = 3,
        stuck_recovery_enabled: bool = True,
        stuck_recovery_timeout_sec: float = 10.0,
        stuck_recovery_position_tolerance_m: float = 0.5,
        recovery_vlm_reuse_sec: float = 0.0,
        failure_memory_enabled: bool = False,
        failure_memory_clear_after_loop1_sec: float = 1.0,
        failure_memory_max_events: int = 64,
        failure_memory_guidance_mode: str = "avoid",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.planner = planner
        self.selector = selector
        # At least one of the two ranking scorers must be active. When neither
        # is supplied, default end2end to the planner-native pass-through so a
        # bare Orchestrator() keeps its pre-scorer behaviour.
        if end2end_scorer is None and rule_based_scorer is None:
            end2end_scorer = PlannerNativeScorer()
        self.end2end_scorer = end2end_scorer  # may be None (rule-based-only)
        self.instruction_scorer = instruction_scorer
        self.rule_based_scorer = rule_based_scorer
        # Backward compatibility: historically supplying both scorers always
        # enabled both PDMS proposal branches. New ablation configs can control
        # the branches independently while still sharing one PDMS scorer.
        legacy_parallel = (
            self.end2end_scorer is not None and self.rule_based_scorer is not None
        )
        if (
            scorer_end2end_pdms_branch is None
            and scorer_target_region_pdms_branch is None
        ):
            self.scorer_end2end_pdms_branch = legacy_parallel
            self.scorer_target_region_pdms_branch = legacy_parallel
        else:
            self.scorer_end2end_pdms_branch = bool(
                scorer_end2end_pdms_branch
            )
            self.scorer_target_region_pdms_branch = bool(
                scorer_target_region_pdms_branch
            )
        if self.scorer_end2end_pdms_branch and self.end2end_scorer is None:
            raise ValueError("end2end PDMS branch requires an end2end scorer")
        if (
            self.scorer_end2end_pdms_branch
            or self.scorer_target_region_pdms_branch
        ) and self.rule_based_scorer is None:
            raise ValueError("PDMS proposal branches require a rule-based scorer")
        self.scorer_region_sector_counts = tuple(
            int(v) for v in scorer_region_sector_counts
        )
        if not self.scorer_region_sector_counts:
            raise ValueError("scorer region sector counts must be non-empty")
        self.scorer_end2end_topk = max(1, int(scorer_end2end_topk))
        # Deprecated target-region arguments remain accepted as aliases so
        # older experiment launchers continue to work under the new all-region
        # sampling semantics.
        if scorer_target_region_max_candidates is not None:
            scorer_region_samples_per_region = int(
                scorer_target_region_max_candidates
            )
        if scorer_target_region_random_seed is not None:
            scorer_region_random_seed = int(scorer_target_region_random_seed)
        self.scorer_region_samples_per_region = max(
            1, int(scorer_region_samples_per_region)
        )
        self.scorer_region_random_seed = int(
            scorer_region_random_seed
        )
        self.scorer_region_min_bearing_deg = float(scorer_region_min_bearing_deg)
        self.scorer_region_max_bearing_deg = float(scorer_region_max_bearing_deg)
        if self.scorer_region_max_bearing_deg <= self.scorer_region_min_bearing_deg:
            raise ValueError("scorer region max bearing must exceed min bearing")
        self.verifier = verifier
        self._lane_tier_probe = lane_tier_probe
        self.verifier_threshold = float(verifier_threshold)
        self.pedestrian_ttc_brake_hold_sec = max(
            0.0, float(pedestrian_ttc_brake_hold_sec)
        )
        self.pedestrian_ttc_brake_distance_m = max(
            0.0, float(pedestrian_ttc_brake_distance_m)
        )
        self.pedestrian_ttc_brake_lateral_m = max(
            0.0, float(pedestrian_ttc_brake_lateral_m)
        )
        self.recovery_explainer = recovery_explainer
        self.semantic_takeover_manager = semantic_takeover_manager
        self.recovery_cfg = recovery_cfg
        self.log = logger or LOG
        self._candidate_scoring = CandidateScoringPipeline(
            end2end_scorer=self.end2end_scorer,
            instruction_scorer=self.instruction_scorer,
            rule_based_scorer=self.rule_based_scorer,
            config=CandidateScoringConfig(
                region_sector_counts=self.scorer_region_sector_counts,
                end2end_topk=self.scorer_end2end_topk,
                region_samples_per_region=self.scorer_region_samples_per_region,
                region_random_seed=self.scorer_region_random_seed,
                region_min_bearing_deg=self.scorer_region_min_bearing_deg,
                region_max_bearing_deg=self.scorer_region_max_bearing_deg,
                end2end_pdms_branch=self.scorer_end2end_pdms_branch,
                target_region_pdms_branch=(
                    self.scorer_target_region_pdms_branch
                ),
                pdms_gate_native_argmax=bool(scorer_pdms_gate_native_argmax),
                # Gate defaults to the verifier threshold so "acceptable to
                # execute" and "acceptable to select" stay one number unless
                # explicitly split in config.
                pdms_gate_threshold=float(
                    scorer_pdms_gate_threshold
                    if scorer_pdms_gate_threshold is not None
                    else self.verifier_threshold
                ),
            ),
            logger=self.log,
        )
        # Cross-tick recovery memory: which ladder primitives were already
        # attempted this episode, settled by later verifier verdicts.
        self.recovery_tracker = RecoveryAttemptTracker()
        # Episode-scoped reflection memory. Unlike RecoveryAttemptTracker this
        # captures the complete recovery process (problem, VLM choice, PDMS
        # feedback, and execution outcome) and is retained across repeated
        # recovery entries until Loop 1 has stayed healthy for the configured
        # continuous interval.
        self.failure_memory = RecoveryFailureMemory(
            enabled=failure_memory_enabled,
            clear_after_loop1_sec=failure_memory_clear_after_loop1_sec,
            max_events=failure_memory_max_events,
            guidance_mode=failure_memory_guidance_mode,
        )
        # A VLM sequence is an intent queue, not one long open-loop maneuver.
        # Only its first primitive is committed. After that primitive ends,
        # Loop 1 runs again; if recovery is still needed, the next VLM call sees
        # the completed and remaining steps and may keep or revise the intent.
        self._recovery_sequence_manager = RecoverySequenceActionManager()
        self._recovery_sequence_clear_after_loop1_sec = max(
            0.0, float(failure_memory_clear_after_loop1_sec)
        )

        # --- "commit-and-hold" recovery mode -------------------------------
        # When on: on a verifier failure, try each primitive in random order,
        # verify each one, and commit the FIRST that passes for that action's
        # fixed SIM-time duration (open-loop hold, no re-verify), then resume
        # the normal loop. Off keeps the legacy per-frame recovery behavior.
        #
        # The hold is bounded by the sim clock (``info["timestamp"]``), NOT by a
        # count of orchestrator.step() calls. Under the async replan pipeline
        # step() only runs once per replan (e.g. 2 Hz), not once per control
        # tick (e.g. 20 Hz), so a frame counter would stretch the intended hold
        # by control_frequency/replan_frequency (10x: 6 s -> 60 s of blind
        # open-loop driving). Timing off the sim clock keeps the window correct
        # at any replan cadence.
        self.recovery_commit_and_hold = bool(recovery_commit_and_hold)
        self.recovery_random_order = bool(recovery_random_order)
        self._recovery_rng = random.Random(recovery_seed)
        self.recovery_tier_veto = bool(recovery_tier_veto)
        self.recovery_hd_map_endpoint_only = bool(
            recovery_hd_map_endpoint_only
        )
        self.recovery_full_plan_dac = bool(recovery_full_plan_dac)
        self.recovery_hold_dac_guard = bool(recovery_hold_dac_guard)
        self.recovery_hold_dac_guard_ticks = max(1, int(recovery_hold_dac_guard_ticks))
        # Re-arms the mid-hold NC gate for pedestrians over every exclusive
        # lane-bypass phase. Route-lane vehicles join over the return blend --
        # the merge slot no admission gate inspects.
        self.recovery_bypass_return_nc_guard = bool(
            recovery_bypass_return_nc_guard
        )
        # Upper bound on how long a latched-but-never-committed bypass may
        # suppress SparseDrive and the rest of the recovery menu. The latch is
        # deliberately exclusive so its gates are the only thing reconsidered,
        # but "gate closed" and "gate unevaluable" are not the same claim: an
        # unknown adjacent-lane topology keeps the wait alive forever, and
        # because the wait also blocks the planner the ego never moves, so the
        # lookup that produced "unknown" never gets a different answer. Cap the
        # wait so the deadlock resolves into normal planning instead of a
        # route timeout. <= 0 restores the old unbounded wait.
        self.lane_bypass_wait_timeout_sec = float(lane_bypass_wait_timeout_sec)
        # Fallback-selection policy (ports from the x-series ablations; all
        # defaults preserve shipped behavior byte-identically):
        # tie_break_order -- names preferred among EXACT fallback-argmax score
        # ties (today's tie-break is iteration-order accident); tabu_exempt --
        # primitives the no-progress tabu never bans (creep_back's backward
        # gain is negative by design); rank_without_ep -- rank trial-passing
        # fallback candidates by gates*(5*ttc+2*lk)/7 so route-progress greed
        # stops deciding WHICH safe maneuver runs (eligibility keeps full
        # canonical PDMS; measured: EP prices candidates by forward-lunge
        # distance, ~+0.014 score per meter of endpoint).
        self._recovery_tie_break_rank: Dict[str, int] = {
            str(name): rank for rank, name in enumerate(recovery_tie_break_order)
        }
        self._recovery_tabu_exempt = frozenset(
            str(name) for name in recovery_tabu_exempt
        )
        self.recovery_rank_without_ep = bool(recovery_rank_without_ep)
        self._trial_rank_scores: Dict[int, float] = {}
        self.recovery_allow_failed_primitive_escalation = bool(
            recovery_allow_failed_primitive_escalation
        )
        # Loop-1-only route-progress guard. Recovery primitives intentionally
        # bypass this counter because reversing/bypass maneuvers can have EP=0
        # while still being the correct way back to the route corridor.
        self.normal_ep_gate = bool(normal_ep_gate)
        self.normal_ep_min = max(0.0, float(normal_ep_min))
        self.normal_ep_gate_steps = max(1, int(normal_ep_gate_steps))
        self._normal_ep_low_streak = 0
        # Independent hard recovery trigger: if the ego remains within a small
        # XY radius for the configured SIM-time interval, enter the same
        # recovery path even when the verifier accepts the learned plan. The
        # radius absorbs localization/physics jitter around a genuinely stuck
        # vehicle; meaningful displacement restarts the timer.
        self.stuck_recovery_enabled = bool(stuck_recovery_enabled)
        self.stuck_recovery_timeout_sec = max(
            0.0, float(stuck_recovery_timeout_sec)
        )
        self.stuck_recovery_position_tolerance_m = max(
            0.0, float(stuck_recovery_position_tolerance_m)
        )
        self._stuck_anchor_xy: Optional[np.ndarray] = None
        self._stuck_anchor_ts: Optional[float] = None
        # Optional short-lived reuse of the last recovery VLM answer. A normal
        # Loop-1 pass clears it immediately, so only a continuous run of
        # recovery-triggering frames can reuse the answer. Default zero keeps
        # every existing BridgeDrive/SparseDrive configuration unchanged.
        self.recovery_vlm_reuse_sec = max(0.0, float(recovery_vlm_reuse_sec))
        self._recovery_vlm_answer: Optional[Dict[str, Any]] = None
        self._recovery_vlm_answer_ts: Optional[float] = None
        # Consecutive commit-and-hold cycles that ended in a forced brake while
        # the ego was parked; drives the anti-deadlock escalation.
        self._forced_brake_streak = 0
        # No-progress tabu state: {primitive_name: cycles_remaining} plus the
        # (name, route_station) of the most recent commit, judged next cycle.
        self._tabu: Dict[str, int] = {}
        self._last_commit_progress: Optional[tuple[str, float]] = None
        # Position-based sequence-menu escalation. Unlike the normal stuck
        # recovery timer, this observation persists across recovery attempts
        # and commit/hold windows so repeated one-tick brakes cannot prevent a
        # genuinely stationary ego from opening the larger VLM menu.
        self._sequence_menu_stationary_anchor_xy: Optional[np.ndarray] = None
        self._sequence_menu_stationary_anchor_ts: Optional[float] = None
        self._sequence_menu_stationary_sec = 0.0
        self._sequence_menu_front_obstacle: Dict[str, Any] = {
            "enabled": False,
            "triggered": False,
        }
        self._sequence_menu_front_vehicle_dwell: Dict[
            str, Dict[str, Any]
        ] = {}
        # Times each primitive has been committed via escalation this episode;
        # the escalation ranks least-committed first (diversity).
        self._escalation_commits: Dict[str, int] = {}
        # In-flight committed maneuver:
        # {"name", "start_ts", "execute_seconds", "prog",
        #  "output_num_poses", "world_plan", "states", "plan_dt_sec"}.
        self._recovery_hold: Optional[Dict[str, Any]] = None
        # Positive VLM bypass-necessity decision, held independently from the
        # live traffic/topology execution gate.  While set, normal planning and
        # all other recovery primitives remain suspended.
        self._lane_bypass_intent: Optional[Dict[str, Any]] = None
        # action -> timestamp until which that bypass may not be re-latched,
        # armed when a latch is abandoned on timeout. Without it the very next
        # recovery cycle re-latches the same never-executable maneuver and the
        # timeout buys nothing.
        self._lane_bypass_latch_cooldown: Dict[str, float] = {}
        self._target_lane_static_wait: Optional[Dict[str, Any]] = None
        # Optional legacy explanation of the current recovery trigger. Table-2
        # configs keep this None and select directly from scene + PDMS context.
        self._pending_recovery_reason: Optional[str] = None
        self._pending_recovery_verifier_context: Optional[Dict[str, Any]] = None
        # Covers both commit-and-hold and legacy one-tick recovery.  It lets
        # the sparse semantic manager suppress A checks during recovery and
        # emit exactly one E event when control returns to the basic loop.
        self._recovery_episode_active = False
        # A pedestrian-caused TTC failure inside the configured caution range
        # latches an unconditional stop. The hold bypasses both Loop-1
        # replanning and the maneuver-recovery ladder for its complete window.
        self._pedestrian_ttc_brake_hold: Optional[Dict[str, Any]] = None

    def _clear_recovery_vlm_answer(self) -> None:
        self._recovery_vlm_answer = None
        self._recovery_vlm_answer_ts = None

    def _cached_recovery_vlm_answer(
        self,
        info: Dict[str, Any],
    ) -> Optional[tuple[Dict[str, Any], float]]:
        if (
            self.recovery_vlm_reuse_sec <= 0.0
            or self._recovery_vlm_answer is None
            or self._recovery_vlm_answer_ts is None
        ):
            return None
        now = float(info.get("timestamp", 0.0) or 0.0)
        age_sec = now - self._recovery_vlm_answer_ts
        if age_sec < 0.0 or age_sec > self.recovery_vlm_reuse_sec:
            self._clear_recovery_vlm_answer()
            return None
        return deepcopy(self._recovery_vlm_answer), float(age_sec)

    def _remember_recovery_vlm_answer(
        self,
        info: Dict[str, Any],
        answer: Dict[str, Any],
    ) -> None:
        if self.recovery_vlm_reuse_sec <= 0.0:
            return
        if answer.get("error") or not answer.get("reason"):
            return
        self._recovery_vlm_answer = deepcopy(answer)
        self._recovery_vlm_answer_ts = float(
            info.get("timestamp", 0.0) or 0.0
        )

    @staticmethod
    def _make_step_result(**kwargs: Any) -> StepResult:
        return StepResult(**kwargs)

    @staticmethod
    def _vru_ttc_vetoed(verifier_result: Optional[Any]) -> bool:
        if verifier_result is None:
            return False
        try:
            value = float((verifier_result.gates or {}).get("ttc_vru", 1.0))
        except (TypeError, ValueError):
            return False
        return bool(np.isfinite(value) and value <= 0.0)

    def _nearest_forward_vru(
        self,
        info: Dict[str, Any],
        privileged_info: Optional[List[Dict[str, Any]]],
    ) -> Optional[Dict[str, Any]]:
        """Nearest pedestrian inside the controller's forward caution range."""
        ego_box = np.asarray(info.get("ego_box", []), dtype=np.float64).reshape(-1)
        ego_half_width = (
            max(0.1, 0.5 * float(ego_box[4])) if ego_box.size > 4 else 1.0
        )
        observations = front_corridor_object_observations(
            info=info,
            privileged_info=privileged_info,
            max_distance_m=self.pedestrian_ttc_brake_distance_m,
            lateral_buffer_m=max(
                0.0, self.pedestrian_ttc_brake_lateral_m - ego_half_width
            ),
        )
        return next(
            (
                observation
                for observation in observations
                if str(observation.get("object_type", "")).startswith(
                    ("pedestrian", "walker")
                )
            ),
            None,
        )

    @staticmethod
    def _pedestrian_ttc_brake_debug_verifier(
        verifier_result: Optional[Any],
    ) -> Dict[str, Any]:
        if verifier_result is None:
            return {"gates": {"ttc_vru": 0.0}, "soft": {"ttc": 0.0}}
        try:
            return dict(verifier_result.to_debug_dict())
        except (AttributeError, TypeError, ValueError):
            return {
                "gates": dict(getattr(verifier_result, "gates", {}) or {}),
                "soft": dict(getattr(verifier_result, "soft", {}) or {}),
            }

    def _pedestrian_ttc_brake_result(
        self,
        *,
        info: Dict[str, Any],
        base_result: Optional[StepResult] = None,
        triggered: bool = False,
    ) -> StepResult:
        """Return a controller-ready all-zero plan for the active brake hold."""
        hold = dict(self._pedestrian_ttc_brake_hold or {})
        now = float(info.get("timestamp", 0.0) or 0.0)
        until_ts = float(hold.get("until_ts", now))
        plan_points = max(2, int(hold.get("plan_points", 2)))
        plan = np.zeros((plan_points, 2), dtype=np.float32)
        states = np.zeros((plan_points, 2), dtype=np.float32)
        source = "safety:pedestrian_ttc_brake"
        payload = {
            "selected_idx": None,
            "selected_plan": plan,
            "selected_score": 0.0,
            "selected_source": source,
            "fallback_primitive": "brake",
            "fallback_mode": "pedestrian_ttc_hold",
            "fallback_states": states,
            "fallback_execute_seconds": self.pedestrian_ttc_brake_hold_sec,
        }
        debug = {} if base_result is None else dict(base_result.debug)
        debug["verifier"] = dict(hold.get("verifier", {}))
        debug["pedestrian_ttc_brake"] = {
            "triggered": bool(triggered),
            "active": True,
            "start_ts": float(hold.get("start_ts", now)),
            "until_ts": until_ts,
            "remaining_sec": max(0.0, until_ts - now),
            "duration_sec": self.pedestrian_ttc_brake_hold_sec,
            "nearest_pedestrian": hold.get("nearest_pedestrian"),
        }
        return self._make_step_result(
            selected_plan=plan,
            selected_score_raw=0.0,
            selected_source=source,
            plan_payload=payload,
            proposals=(
                plan[np.newaxis, ...]
                if base_result is None
                else base_result.proposals
            ),
            scores=(
                np.asarray([0.0], dtype=np.float32)
                if base_result is None
                else base_result.scores
            ),
            verifier_score=(
                float(hold.get("verifier_score", 0.0))
                if base_result is None
                else base_result.verifier_score
            ),
            verifier_passed=False,
            recovered=False,
            debug=debug,
        )

    def _start_pedestrian_ttc_brake_hold(
        self,
        *,
        info: Dict[str, Any],
        result: StepResult,
        verifier_result: Optional[Any],
        nearest_pedestrian: Dict[str, Any],
    ) -> StepResult:
        now = float(info.get("timestamp", 0.0) or 0.0)
        self._pedestrian_ttc_brake_hold = {
            "start_ts": now,
            "until_ts": now + self.pedestrian_ttc_brake_hold_sec,
            "plan_points": max(2, len(result.selected_plan)),
            "verifier_score": float(result.verifier_score or 0.0),
            "verifier": self._pedestrian_ttc_brake_debug_verifier(verifier_result),
            "nearest_pedestrian": dict(nearest_pedestrian),
        }
        self._normal_ep_low_streak = 0
        self.log.warning(
            "pedestrian TTC=0 inside %.1fm/±%.1fm range; forcing brake for %.2fs "
            "(actor=%s distance=%.2fm)",
            self.pedestrian_ttc_brake_distance_m,
            self.pedestrian_ttc_brake_lateral_m,
            self.pedestrian_ttc_brake_hold_sec,
            nearest_pedestrian.get("agent_id", "unknown"),
            float(nearest_pedestrian.get("distance_m", 0.0)),
        )
        return self._pedestrian_ttc_brake_result(
            info=info,
            base_result=result,
            triggered=True,
        )

    def _step_pedestrian_ttc_brake_hold(
        self, info: Dict[str, Any]
    ) -> Optional[StepResult]:
        hold = self._pedestrian_ttc_brake_hold
        if hold is None:
            return None
        now = float(info.get("timestamp", 0.0) or 0.0)
        if now >= float(hold.get("until_ts", now)):
            self._pedestrian_ttc_brake_hold = None
            return None
        return self._pedestrian_ttc_brake_result(info=info)

    @staticmethod
    def _info_to_pose(info: Dict[str, Any]):
        return info_to_pose(info)

    @staticmethod
    def _local_plan_to_world(plan: np.ndarray, pose: Any):
        return local_plan_to_world(plan, pose)

    @staticmethod
    def _world_points_to_current_local(points: np.ndarray, pose: Any):
        return world_points_to_current_local(points, pose)

    @property
    def verifier_enabled(self) -> bool:
        return self.verifier is not None

    @property
    def candidate_pdms_enabled(self) -> bool:
        """True when round-one candidate ranking needs obstacle perception."""
        return self.rule_based_scorer is not None

    @property
    def recovery_hold_active(self) -> bool:
        """True while a committed recovery maneuver is still in its hold window.

        The async control path reads this to leave the hold alone: its
        every-tick verifier must not force an early replan mid-maneuver, since
        commit-and-hold is defined as open-loop for its whole window.
        """
        return self._recovery_hold is not None

    def _verifier_accepts(self, score: float) -> bool:
        """Keep Loop 1 only while the PDMS score meets its threshold."""
        return float(score) >= float(self.verifier_threshold)

    def _refine_rejected_loop1_plan(self, plan, verify_info, source):
        """Optional planner-specific refinement; existing planners keep their plan."""
        return None

    def _note_recovery_started(self, info: Dict[str, Any], *, source: str) -> None:
        self._recovery_episode_active = True
        # Recovery breaks a sequence of Loop-1 observations. A low-EP plan
        # after the maneuver must earn a fresh consecutive streak instead of
        # inheriting the trigger count from before recovery.
        self._normal_ep_low_streak = 0
        # A recovery attempt starts a new motion episode. Do not carry the
        # pre-recovery stationary interval into the next Loop-1 attempt.
        self._stuck_anchor_xy = None
        self._stuck_anchor_ts = None
        failure_memory = getattr(self, "failure_memory", None)
        if failure_memory is not None:
            failure_memory.start_or_resume(
                float(info.get("timestamp", 0.0) or 0.0),
                source=source,
            )
        self._sequence_action_manager().cancel_loop1_clear()
        manager = self.semantic_takeover_manager
        if manager is not None:
            manager.note_recovery_started(
                float(info.get("timestamp", 0.0) or 0.0),
                source=source,
            )

    def _observe_stuck_recovery(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Track whether the ego stayed at one XY location long enough."""
        state, self._stuck_anchor_xy, self._stuck_anchor_ts = (
            observe_position_dwell(
                info,
                enabled=self.stuck_recovery_enabled,
                timeout_sec=self.stuck_recovery_timeout_sec,
                position_tolerance_m=(
                    self.stuck_recovery_position_tolerance_m
                ),
                anchor_xy=self._stuck_anchor_xy,
                anchor_ts=self._stuck_anchor_ts,
            )
        )
        return state
    def _note_recovery_ended(self, info: Dict[str, Any]) -> None:
        if not self._recovery_episode_active:
            return
        self._recovery_episode_active = False
        self._pending_recovery_verifier_context = None
        manager = self.semantic_takeover_manager
        if manager is not None:
            manager.note_recovery_ended(info)

    def _record_failure_memory(
        self,
        info: Dict[str, Any],
        *,
        event_type: str,
        problem: Optional[str] = None,
        method: Optional[str] = None,
        outcome: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Best-effort write to the optional recovery failure memory."""
        failure_memory = getattr(self, "failure_memory", None)
        if failure_memory is None:
            return
        failure_memory.record(
            timestamp=float(info.get("timestamp", 0.0) or 0.0),
            event_type=event_type,
            problem=problem,
            method=method,
            outcome=outcome,
            details=details,
        )

    def _failure_memory_context(
        self, available_actions: Optional[Sequence[str]] = None
    ) -> Optional[str]:
        failure_memory = getattr(self, "failure_memory", None)
        if failure_memory is None:
            return None
        return failure_memory.describe(available_actions=available_actions)

    def _sequence_action_manager(self) -> RecoverySequenceActionManager:
        manager = getattr(self, "_recovery_sequence_manager", None)
        if manager is None:
            manager = RecoverySequenceActionManager()
            self._recovery_sequence_manager = manager
        return manager

    def _recovery_sequence_context(self) -> Optional[Dict[str, Any]]:
        return self._sequence_action_manager().context()

    def _latch_lane_bypass_intent(
        self,
        *,
        action: str,
        info: Dict[str, Any],
        output_num_poses: int,
        front_blockage: Optional[object],
    ) -> bool:
        """Persist a positive bypass necessity decision until completion.

        Returns whether the latch is now held for ``action``. A False result
        means current blockage evidence is absent, or a previous latch is
        still in its timeout cooldown; the caller must not execute that bypass.
        """
        existing = getattr(self, "_lane_bypass_intent", None)
        if existing is not None and str(existing.get("action")) == str(action):
            return True
        front = getattr(self, "_sequence_menu_front_obstacle", {})
        if front.get("enabled") and not front.get("triggered"):
            self.log.info("refusing lane bypass without a current route blocker")
            return False
        now = float(info.get("timestamp", 0.0) or 0.0)
        blocked_until = float(
            getattr(self, "_lane_bypass_latch_cooldown", {}).get(str(action), 0.0)
        )
        if now < blocked_until:
            self.log.warning(
                "refusing to re-latch lane bypass '%s': its previous latch was "
                "abandoned on timeout, cooldown has %.1fs left",
                action,
                blocked_until - now,
            )
            return False
        self._lane_bypass_intent = {
            "action": str(action),
            "latched_ts": now,
            "output_num_poses": max(2, int(output_num_poses)),
            "front_blockage": (
                None if front_blockage is None else str(front_blockage)
            ),
            "committed": False,
            # Only executed motion resets the wait budget; predicted gaps do not.
            "last_progress_ts": now,
            "wait_ticks": 0,
        }
        self.log.warning(
            "latched lane bypass intent '%s'; waiting exclusively for its "
            "traffic/topology and PDMS gates",
            action,
        )
        return True

    def _retain_lane_bypass_obstacle_clearance(
        self,
        assessment: Optional[Dict[str, Any]],
    ) -> Optional[float]:
        """Keep the largest observed route-blocker bbox clearance on a latch.

        Static-object snapshots can be intermittent.  Once any tick has seen
        the complete construction bbox cluster, a later sparse snapshot must
        not shorten the committed bypass and return the ego between props.
        """
        intent = getattr(self, "_lane_bypass_intent", None)
        if intent is None:
            return None
        cfg = getattr(self, "recovery_cfg", None)
        retained = max(
            0.0,
            float(getattr(cfg, "obstacle_clearance_m", 8.0)),
            float(intent.get("required_obstacle_clearance_m", 0.0) or 0.0),
        )
        observed = 0.0
        if isinstance(assessment, dict):
            try:
                observed = float(assessment.get("obstacle_clearance_m", 0.0))
            except (TypeError, ValueError):
                observed = 0.0
            if math.isfinite(observed):
                retained = max(retained, observed)
            if assessment.get("route_blocker_type") == "static":
                retained = max(
                    retained,
                    min(
                        LANE_BYPASS_CONE_CHAIN_CLEARANCE_M,
                        observed + LANE_BYPASS_STATIC_TRAILING_CLEARANCE_M,
                    ),
                )
            assessment["latched_obstacle_clearance_m"] = retained
        if "cone" in str(intent.get("front_blockage", "")).lower():
            retained = max(retained, LANE_BYPASS_CONE_CHAIN_CLEARANCE_M)
        intent["required_obstacle_clearance_m"] = retained
        return retained

    @staticmethod
    def _mark_lane_bypass_payload(
        payload: Dict[str, Any], action: str
    ) -> None:
        """Mark an exclusive bypass payload without assuming traffic side."""
        side = "right" if str(action).endswith("_right") else "left"
        payload["recovery_needs_lane_bypass"] = True
        payload[f"recovery_needs_lane_bypass_{side}"] = True

    def _pending_lane_bypass_brake_result(self) -> StepResult:
        """Controller-ready brake used while the latched bypass is gated."""
        intent = getattr(self, "_lane_bypass_intent", None) or {}
        action = str(intent.get("action", "lane_bypass_left"))
        plan = np.zeros((2, 2), dtype=np.float32)
        states = np.zeros((2, 2), dtype=np.float32)
        payload = {
            "selected_plan": plan,
            "selected_score": 0.0,
            "selected_source": f"recovery_wait:{action}",
            "fallback_primitive": self._RECOVERY_BRAKE_PRIMITIVE,
            "fallback_mode": "single_tick",
            "fallback_states": states,
            "recovery_pending_primitive": str(
                action
            ),
        }
        self._mark_lane_bypass_payload(payload, action)
        return self._make_step_result(
            selected_plan=plan,
            selected_score_raw=0.0,
            selected_source=str(payload["selected_source"]),
            plan_payload=payload,
            proposals=plan[np.newaxis, ...],
            scores=np.asarray([0.0], dtype=np.float32),
            verifier_score=0.0,
            verifier_passed=False,
            recovered=True,
            debug={"lane_bypass_intent": dict(intent)},
        )

    def _abandon_lane_bypass_intent_if_stale(self, info: Dict[str, Any]) -> bool:
        """Expire an uncommitted latch unless a static blocker is still observed.

        The latch only makes sense while its gates are actually being
        evaluated. When the target lane's topology is unknown the primitive is
        never even materialized, so the "gate" reported every tick is the
        initial ``traffic_or_topology_gate_closed`` placeholder rather than a
        verdict -- and since the latch also suppresses the planner, the ego
        cannot move to a pose where the lookup would succeed. Time the wait out
        so that deadlock can end in normal planning once the static obstruction
        is no longer observed.
        A committed latch is mid-maneuver and is never interrupted here.
        """
        intent = getattr(self, "_lane_bypass_intent", None)
        timeout_sec = float(getattr(self, "lane_bypass_wait_timeout_sec", 0.0))
        if intent is None or timeout_sec <= 0.0:
            return False
        if bool(intent.get("committed", False)):
            return False
        now = float(info.get("timestamp", 0.0) or 0.0)
        since_ts = intent.get("last_progress_ts", intent.get("latched_ts", now))
        waited_sec = now - float(since_ts if since_ts is not None else now)
        total_wait_sec = now - float(intent.get("latched_ts", since_ts))
        # Repeated commits that immediately abort cannot renew a latch forever.
        if waited_sec < timeout_sec and total_wait_sec < 2.0 * timeout_sec:
            return False
        # A scheduling timeout is not evidence that the obstruction cleared.
        # Keep evaluating the bypass while a measured static blocker remains;
        # _step_pending_lane_bypass releases the latch after observed clearance.
        front = getattr(self, "_sequence_menu_front_obstacle", {})
        if front.get("triggered") and front.get("object_type") == "static":
            return False
        action = str(intent.get("action") or "lane_bypass_left")
        self._lane_bypass_intent = None
        cooldown = getattr(self, "_lane_bypass_latch_cooldown", None)
        if cooldown is None:
            cooldown = {}
            self._lane_bypass_latch_cooldown = cooldown
        cooldown[action] = now + timeout_sec
        self.log.warning(
            "latched lane bypass '%s' did not become executable in %.1fs "
            "(last_gate=%s, wait_ticks=%d); abandoning the latch and resuming "
            "normal planning, re-latch blocked for %.1fs",
            action,
            total_wait_sec,
            intent.get("last_gate", "traffic_or_topology_gate_closed"),
            int(intent.get("wait_ticks", 0)),
            timeout_sec,
        )
        return True

    def _step_pending_lane_bypass(
        self,
        *,
        obs: Dict[str, Any],
        info: Dict[str, Any],
        privileged_info: Optional[List[Dict[str, Any]]],
    ) -> Optional[StepResult]:
        """Brake or commit the sole latched bypass without normal replanning.

        Returns None when the blocker clears or an uncommitted latch expires
        without a still-observed static obstruction, handing the tick back to
        normal planning.
        """
        intent = self._lane_bypass_intent
        assert intent is not None
        # A positive visual decision is not a permanent obstacle. Once fresh
        # observations no longer support a stopped/static route blocker,
        # return to normal planning (which still has to pass the verifier).
        front = getattr(self, "_sequence_menu_front_obstacle", {})
        if (
            privileged_info is not None
            and front.get("enabled")
            and not intent.get("committed")
        ):
            now = float(info.get("timestamp", 0.0) or 0.0)
            if front.get("triggered"):
                intent.pop("clear_since_ts", None)
            elif now - intent.setdefault("clear_since_ts", now) >= 1.0:
                self.log.info(
                    "lane bypass blocker cleared; releasing '%s' for normal verification",
                    intent["action"],
                )
                self._lane_bypass_intent = None
                return None
        self._refresh_lane_bypass_wait_progress(
            info=info,
            privileged_info=privileged_info,
        )
        if self._abandon_lane_bypass_intent_if_stale(info):
            return None
        intent["wait_ticks"] = int(intent.get("wait_ticks", 0)) + 1
        self._note_recovery_started(info, source="lane_bypass_wait")
        brake_result = self._pending_lane_bypass_brake_result()
        return self._begin_commit_and_hold(
            brake_result,
            obs=obs,
            info=info,
            privileged_info=privileged_info,
            output_num_poses=max(2, int(intent.get("output_num_poses", 2))),
            verifier_score=0.0,
            allow_vlm_selection=False,
            trigger_reason=(
                "latched lane bypass intent waiting for a safe execution window"
            ),
            # ``brake_result`` is already a stop, so "keep result" IS the
            # fail-safe here. Latching the fixed-duration MRM brake instead
            # would stop this path from polling for its execution window every
            # tick, which is the whole point of the wait.
            fail_closed=False,
        )

    def _refresh_lane_bypass_wait_progress(
        self,
        *,
        info: Dict[str, Any],
        privileged_info: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Track a scheduled safe gap while an exclusive bypass is latched."""
        intent = self._lane_bypass_intent
        assert intent is not None
        action = str(intent.get("action") or "lane_bypass_left")
        direction = "right" if action.endswith("_right") else "left"
        cfg = getattr(self, "recovery_cfg", None)
        assessment = lane_bypass_execution_assessment(
            lane_bypass_gap_assessment(
                info=info,
                privileged_info=privileged_info,
                direction=direction,
                min_gap_m=getattr(cfg, "lane_bypass_min_gap_m", 0.0),
                clearance_buffer_m=getattr(
                    cfg, "lane_bypass_clearance_buffer_m", 1.0
                ),
                required_time_sec=getattr(
                    cfg, "lane_bypass_required_time_sec", 3.5
                ),
                min_time_gap_sec=getattr(
                    cfg, "lane_bypass_min_time_gap_sec", 4.5
                ),
                obstacle_clearance_m=getattr(
                    cfg, "obstacle_clearance_m", 8.0
                ),
            )
        )
        self._retain_lane_bypass_obstacle_clearance(assessment)
        intent["lane_bypass_assessment"] = assessment
        if bool(assessment.get("ready_now", False)):
            intent["last_gate"] = "ready_now"
            intent["last_entry_delay_sec"] = 0.0
            return assessment

        if bool(assessment.get("scheduled_available", False)):
            delay = max(
                0.0,
                float(assessment.get("entry_delay_sec", 0.0) or 0.0),
            )
            intent["last_entry_delay_sec"] = delay
            intent["last_gate"] = "waiting_entry_delay"
            return assessment

        intent["last_gate"] = str(
            assessment.get("reason") or "traffic_or_topology_gate_closed"
        )
        intent.pop("last_entry_delay_sec", None)
        return assessment

    @staticmethod
    def _fresh_recovery_agents(agents):
        """Allow 0.5s sensor lag; older memory cannot initiate a maneuver."""
        return [a for a in agents or []
                if float(a.get("perception_age_sec", 0.0) or 0.0) <= 0.5]

    def _static_wait_bypass_assessments(self, info, privileged_info):
        """Recheck adjacent paths using only fresh obstacle observations."""
        privileged_info = self._fresh_recovery_agents(privileged_info)
        cfg = getattr(self, "recovery_cfg", None)
        return {
            side: lane_bypass_gap_assessment(
                info=info, privileged_info=privileged_info, direction=side,
                min_gap_m=getattr(cfg, "lane_bypass_min_gap_m", 0.0),
                clearance_buffer_m=getattr(cfg, "lane_bypass_clearance_buffer_m", 1.0),
                required_time_sec=getattr(cfg, "lane_bypass_required_time_sec", 3.5),
                min_time_gap_sec=getattr(cfg, "lane_bypass_min_time_gap_sec", 4.5),
                obstacle_clearance_m=getattr(cfg, "obstacle_clearance_m", 8.0),
            ) for side in ("left", "right")
        }

    def _road_closure_wait_if_needed(
        self, info: Dict[str, Any], privileged_info: Optional[List[Dict[str, Any]]]
    ) -> Optional[StepResult]:
        """Wait for a sensed whole-road closure, independent of wall texture."""
        privileged_info = self._fresh_recovery_agents(privileged_info)
        front = getattr(self, "_sequence_menu_front_obstacle", {})
        # A nearer vehicle (including a car printed on a wall) must not hide
        # a separately measured static closure farther along the same corridor.
        if front.get("enabled") and front.get("object_type") != "static":
            static_fronts = [a for a in front_corridor_object_observations(
                info=info, privileged_info=privileged_info,
                max_distance_m=float(front.get("distance_limit_m", 0.0)),
            ) if a.get("object_type") == "static"]
            if static_fronts:
                front = {**static_fronts[0], "triggered": True}
        if not front.get("triggered") or front.get("object_type") != "static":
            return None
        assessments = self._static_wait_bypass_assessments(info, privileged_info)
        reasons = {a.get("reason") for a in assessments.values()}
        blocker = next((a for a in privileged_info or []
                        if a.get("agent_id") == front.get("agent_id")), {})
        ego_box = np.asarray(info.get("ego_box", []), dtype=float).reshape(-1)
        ego_width = float(ego_box[4]) if ego_box.size > 4 else 2.0
        extent = blocker.get("agent_extent") or [0.0, 0.0]
        # With no adjacent lane, require a broad measured face, rather than
        # stopping indefinitely for a small object that an in-lane swerve can pass.
        wide_face = (
            blocker.get("perception_source") == "bridgedrive_lidar_corridor"
            and float(extent[1]) >= 2.0 * ego_width
        )
        if not (
            ("target_lane_static_blocked" in reasons or wide_face)
            and reasons <= {"target_lane_static_blocked", "no_adjacent_driving_lane"}
        ):
            return None
        direction = next((s for s, a in assessments.items()
                          if a.get("reason") == "target_lane_static_blocked"), "left")
        previous = getattr(self, "_target_lane_static_wait", None) or {}
        bypass_intent = previous.get("bypass_intent") or self._lane_bypass_intent
        self._lane_bypass_intent = None
        self._target_lane_static_wait = {
            "direction": direction,
            "started_ts": previous.get("started_ts", float(info.get("timestamp", 0.0))),
            "wait_ticks": int(previous.get("wait_ticks", 0)) + 1,
            "full_road_closure": True,
            "bypass_intent": bypass_intent,
            "conflicting_agent_id": assessments[direction].get("conflicting_agent_id"),
        }
        if not previous:
            self._note_recovery_started(info, source="road_closure_wait")
            self.log.info("sensed static road closure with no adjacent driving path; waiting for clearance")
        return self._pending_target_lane_static_wait_result()

    def _pending_target_lane_static_wait_result(self) -> StepResult:
        """Controller-ready stop while a fixed adjacent lane is obstructed."""
        plan = np.zeros((2, 2), dtype=np.float32)
        states = np.zeros((2, 2), dtype=np.float32)
        payload = {
            "selected_plan": plan,
            "selected_score": 0.0,
            "selected_source": "recovery_wait:target_lane_static_blocked",
            "fallback_primitive": "wait",
            "fallback_mode": "single_tick",
            "fallback_states": states,
        }
        return self._make_step_result(
            selected_plan=plan,
            selected_score_raw=0.0,
            selected_source=str(payload["selected_source"]),
            plan_payload=payload,
            proposals=plan[np.newaxis, ...],
            scores=np.asarray([0.0], dtype=np.float32),
            verifier_score=0.0,
            verifier_passed=False,
            recovered=True,
            debug={
                "target_lane_static_wait": dict(
                    getattr(self, "_target_lane_static_wait", None) or {}
                )
            },
        )

    def _step_pending_target_lane_static_wait(
        self,
        *,
        info: Dict[str, Any],
        privileged_info: Optional[List[Dict[str, Any]]],
    ) -> Optional[StepResult]:
        """Suppress all planners until the full-road static closure clears."""
        wait_state = getattr(self, "_target_lane_static_wait", None)
        if wait_state is None:
            return None
        if wait_state.get("full_road_closure"):
            # Recheck both sides and the actual forward blockage, not just the
            # lane that originally caused the wait. Clearing returns to NC verification.
            result = self._road_closure_wait_if_needed(info, privileged_info)
            if result is not None:
                return result
            front = getattr(self, "_sequence_menu_front_obstacle", {})
            static_ahead = (
                front.get("triggered") and front.get("object_type") == "static"
            ) or any(a.get("object_type") == "static" for a in
                front_corridor_object_observations(
                    info=info, privileged_info=self._fresh_recovery_agents(privileged_info),
                    max_distance_m=float(front.get("distance_limit_m", 20.0)),
                ))
            if static_ahead and wait_state.get("bypass_intent"):
                # A lane opening resumes the original bypass safety gates,
                # not ordinary planning toward the still-present obstruction.
                self._lane_bypass_intent = wait_state["bypass_intent"]
            elif static_ahead:
                # A changed traffic verdict alone is insufficient. Reopen a
                # bypass only when an adjacent path is executable now; the
                # pending intent still passes the normal trajectory/PDMS gates.
                wait_state.pop("clear_since_ts", None)
                for side, assessment in self._static_wait_bypass_assessments(
                    info, privileged_info
                ).items():
                    if lane_bypass_execution_assessment(assessment).get("ready_now"):
                        if self._latch_lane_bypass_intent(
                            action=f"lane_bypass_{side}", info=info,
                            output_num_poses=getattr(self, "_last_output_num_poses", 2),
                            front_blockage=front.get("agent_id"),
                        ):
                            break
                if self._lane_bypass_intent is None:
                    return self._pending_target_lane_static_wait_result()
            else:
                now = float(info.get("timestamp", 0.0) or 0.0)
                if now - wait_state.setdefault("clear_since_ts", now) < 1.0:
                    return self._pending_target_lane_static_wait_result()
            self._target_lane_static_wait = None
            self._note_recovery_ended(info)
            return None
        cfg = getattr(self, "recovery_cfg", None)
        assessment = lane_bypass_gap_assessment(
            info=info,
            privileged_info=privileged_info,
            direction=str(wait_state.get("direction", "left")),
            min_gap_m=getattr(cfg, "lane_bypass_min_gap_m", 0.0),
            clearance_buffer_m=getattr(
                cfg, "lane_bypass_clearance_buffer_m", 1.0
            ),
            required_time_sec=getattr(
                cfg, "lane_bypass_required_time_sec", 3.5
            ),
            min_time_gap_sec=getattr(
                cfg, "lane_bypass_min_time_gap_sec", 4.5
            ),
            obstacle_clearance_m=getattr(cfg, "obstacle_clearance_m", 8.0),
        )
        timeout_sec = float(getattr(self, "lane_bypass_wait_timeout_sec", 0.0))
        waited_sec = float(info.get("timestamp", 0.0) or 0.0) - float(
            wait_state.get("started_ts", 0.0)
        )
        if (
            assessment.get("reason") == "target_lane_static_blocked"
            and (timeout_sec <= 0.0 or waited_sec < timeout_sec)
        ):
            wait_state["wait_ticks"] = int(wait_state.get("wait_ticks", 0)) + 1
            wait_state["conflicting_agent_id"] = assessment.get(
                "conflicting_agent_id"
            )
            self._note_recovery_started(info, source="target_lane_static_wait")
            return self._pending_target_lane_static_wait_result()

        self.log.info(
            "rechecking adjacent-lane static closure after %d wait ticks; "
            "normal planning and verification may resume",
            int(wait_state.get("wait_ticks", 0)),
        )
        self._target_lane_static_wait = None
        self._note_recovery_ended(info)
        return None

    def _finish_sequence_hold(
        self,
        hold: Dict[str, Any],
        *,
        outcome: str,
        completed: bool,
    ) -> None:
        """Settle one queued sequence primitive without auto-running the next."""
        if not hold.get("sequence_step"):
            return
        self._sequence_action_manager().finish_current(
            outcome,
            completed=completed,
        )

    # ------------------------------------------------------------------ step
    def step(
        self,
        *,
        obs: Dict[str, Any],
        info: Dict[str, Any],
        info_history: Sequence[Dict[str, Any]],
        privileged_info: Optional[List[Dict[str, Any]]] = None,
        frame_index: Optional[int] = None,
    ) -> StepResult:
        """One loop iteration: generate -> select -> verify -> recover."""
        if frame_index is not None:
            self.selector.frame_index = int(frame_index)

        if self.recovery_explainer is not None:
            location_actions = getattr(self.recovery_explainer, "location_actions", None)
            if location_actions is not None:
                location_actions.observe(info)
            observe_camera_frame = getattr(
                self.recovery_explainer, "observe_camera_frame", None
            )
            if callable(observe_camera_frame):
                observe_camera_frame(
                    (obs or {}).get("rgb", {}) or {},
                    float(info.get("timestamp", 0.0) or 0.0),
                )

        semantic_manager = self.semantic_takeover_manager
        recovery_active = bool(
            self._recovery_episode_active or self._recovery_hold is not None
        )
        if semantic_manager is not None:
            semantic_manager.observe(info, recovery_active=recovery_active)

        pedestrian_brake = self._step_pedestrian_ttc_brake_hold(info)
        if pedestrian_brake is not None:
            return pedestrian_brake

        sequence_menu_stationary = self._observe_sequence_menu_stationary(info)
        sequence_menu_front_obstacle = (
            self._observe_sequence_menu_front_obstacle(info, privileged_info)
        )

        # Committed recovery maneuver in flight: keep executing it (open-loop
        # hold) for the committed window without re-planning or re-verifying,
        # then fall through to the normal loop once the window elapses.
        if self._recovery_hold is not None:
            held = self._step_recovery_hold(info, privileged_info)
            if self._recovery_hold is None:
                self._note_recovery_ended(info)
            if held is not None:
                return held

        static_wait = self._step_pending_target_lane_static_wait(
            info=info,
            privileged_info=privileged_info,
        )
        if static_wait is not None:
            return static_wait

        # Once the VLM says this blockage requires a complete adjacent-lane
        # pass-and-return bypass, only its live safety gates are reconsidered.
        # Waiting here deliberately
        # skips SparseDrive, the normal recovery menu, and any further VLM call.
        if getattr(self, "_lane_bypass_intent", None) is not None:
            pending_bypass = self._step_pending_lane_bypass(
                obs=obs,
                info=info,
                privileged_info=privileged_info,
            )
            # None means the latch timed out and was dropped; fall through to
            # the normal generate -> verify -> recover loop on this same tick.
            if pending_bypass is not None:
                return pending_bypass

        stuck_recovery = self._observe_stuck_recovery(info)

        # trajectory generation
        planner_result = self.planner.process(
            obs, info, list(info_history), {"privileged_info": privileged_info}
        )
        proposals = np.asarray(planner_result.proposals, dtype=np.float32)
        if proposals.ndim == 3:
            self._last_output_num_poses = int(proposals.shape[1])
        scores = np.asarray(planner_result.scores, dtype=np.float32)

        scoring = self._candidate_scoring.score(
            proposals=proposals,
            native_scores=scores,
            obs=obs,
            info=info,
            privileged_info=privileged_info,
        )
        combined_scores = scoring.scores
        scoring_debug = scoring.debug

        # Fast HD-map target gate. PDMS has already reduced the planner's 1024
        # proposals to a small evaluated shortlist, so only probe the final
        # target point of those candidates (plus any independently-finite
        # fallback score). All endpoints are sent through one lane-probe call;
        # unchecked proposals remain unselectable. Recovery paths retain the
        # stricter complete-trajectory footprint gate.
        shortlist = scoring_debug.get("shortlist_indices")
        if shortlist is None:
            checked_indices = np.arange(len(proposals), dtype=np.int64)
            check_scope = "all_proposals_no_pdms_shortlist"
        else:
            checked_indices = np.asarray(shortlist, dtype=np.int64).reshape(-1)
            checked_indices = checked_indices[
                (checked_indices >= 0) & (checked_indices < len(proposals))
            ]
            finite_indices = np.flatnonzero(np.isfinite(combined_scores))
            checked_indices = np.unique(
                np.concatenate([checked_indices, finite_indices])
            ).astype(np.int64)
            check_scope = "pdms_shortlist"

        hd_map_rejected = np.zeros(len(proposals), dtype=bool)
        precomputed_indices = np.asarray(
            scoring_debug.get("shortlist_indices", []), dtype=np.int64
        ).reshape(-1)
        precomputed_tiers = np.asarray(
            scoring_debug.get("shortlist_endpoint_tiers", []), dtype=np.float64
        ).reshape(-1)
        precomputed_by_index = (
            {
                int(index): float(tier)
                for index, tier in zip(precomputed_indices, precomputed_tiers)
            }
            if len(precomputed_indices) == len(precomputed_tiers)
            else {}
        )
        missing_indices = []
        for index in checked_indices:
            tier = precomputed_by_index.get(int(index))
            if tier is None:
                missing_indices.append(int(index))
            else:
                hd_map_rejected[int(index)] = tier < 1.0
        if missing_indices:
            missing_array = np.asarray(missing_indices, dtype=np.int64)
            hd_map_rejected[missing_array] = self._endpoint_offroad_batch(
                proposals[missing_array], info
            )
        if np.any(hd_map_rejected):
            combined_scores = np.asarray(combined_scores, dtype=np.float32).copy()
            combined_scores[hd_map_rejected] = -np.inf
        if shortlist is not None:
            unchecked = np.ones(len(proposals), dtype=bool)
            unchecked[checked_indices] = False
            combined_scores = np.asarray(combined_scores, dtype=np.float32).copy()
            combined_scores[unchecked] = -np.inf
        safe_indices = checked_indices[~hd_map_rejected[checked_indices]]
        scoring_debug["hd_map_hard_filter"] = {
            "active": True,
            "scope": check_scope,
            "checked_indices": checked_indices.tolist(),
            "checked_count": int(len(checked_indices)),
            "skipped_count": int(len(proposals) - len(checked_indices)),
            "reused_pdms_endpoint_count": int(
                len(checked_indices) - len(missing_indices)
            ),
            "new_endpoint_probe_count": int(len(missing_indices)),
            "rejected_indices": np.flatnonzero(hd_map_rejected).tolist(),
            "safe_indices": safe_indices.tolist(),
            "reason": "final_target_outside_driving_corridor",
        }

        # If every learned prediction violates the map, hand the controller a
        # stationary plan while the normal recovery path searches for a legal
        # alternative. This also keeps the hard guarantee when no verifier is
        # configured: an unsafe learned plan is never used as a fallback.
        all_loop1_candidates_rejected = (
            len(safe_indices) == 0
            or not np.any(np.isfinite(combined_scores))
        )
        selector_proposals = proposals
        if all_loop1_candidates_rejected:
            plan_len = int(proposals.shape[1]) if proposals.ndim == 3 else 2
            stop_plan = np.zeros((max(2, plan_len), 2), dtype=np.float32)
            selector_proposals = stop_plan[np.newaxis, ...]
            combined_scores = np.asarray([0.0], dtype=np.float32)
            self.log.warning(
                "HD-map hard filter rejected all %d Loop-1 predictions; "
                "substituting a stationary plan and forcing recovery",
                len(proposals),
            )

        # Round one is always a lightweight argmax. In the two-scorer path the
        # vector contains the PDMS score of each branch's single proposal.
        plan_payload = self.selector.select(
            proposals=selector_proposals,
            scores=combined_scores,
            obs=obs,
            info=info,
            info_history=list(info_history),
            privileged_info=privileged_info,
            strategy="argmax",
        )
        selected_plan, selected_score_raw, selected_source = _unpack_payload(plan_payload)
        pdms_branch_active = self.rule_based_scorer is not None and (
            self.scorer_end2end_pdms_branch
            or self.scorer_target_region_pdms_branch
        )
        if pdms_branch_active and plan_payload.get("selected_idx") is not None:
            selected_idx = int(plan_payload["selected_idx"])
            proposed_by = []
            if selected_idx == scoring_debug.get("end2end_proposal_index"):
                proposed_by.append("end2end")
            if selected_idx == scoring_debug.get("rule_based_proposal_index"):
                proposed_by.append("rule_based")
            selected_scorer = "+".join(proposed_by) if proposed_by else "fallback"
            plan_payload["selected_scorer"] = selected_scorer
            scoring_debug["selected_scorer"] = selected_scorer
            scoring_debug["selected_proposal_index"] = selected_idx
            scoring_debug["selected_proposal_pdms"] = selected_score_raw

        result = StepResult(
            selected_plan=selected_plan,
            selected_score_raw=selected_score_raw,
            selected_source=selected_source,
            plan_payload=plan_payload,
            proposals=proposals,
            scores=scores,
            debug={"scoring": scoring_debug},
        )
        if self.stuck_recovery_enabled:
            result.debug["stuck_recovery"] = stuck_recovery
        if sequence_menu_stationary.get("enabled"):
            result.debug["sequence_menu_stationary"] = sequence_menu_stationary
        if sequence_menu_front_obstacle.get("enabled"):
            result.debug["sequence_menu_front_obstacle"] = (
                sequence_menu_front_obstacle
            )

        if all_loop1_candidates_rejected:
            result.plan_payload["selected_source"] = "hard_filter:hd_map_stop"
            result.selected_source = "hard_filter:hd_map_stop"
            result.debug["scoring"]["hd_map_hard_filter"]["all_rejected"] = True

        # verifier
        if self.verifier is None:
            return result

        # A/E requests are launched only after loop 1 has produced its selected
        # trajectory. This copies a snapshot and returns immediately; VLM work
        # happens on a daemon thread and never stalls the basic loop.
        if semantic_manager is not None:
            semantic_manager.maybe_submit(
                frame_index=getattr(self.selector, "frame_index", 0),
                obs=obs,
                info=info,
                selected_plan=selected_plan,
                recovery_active=self._recovery_episode_active,
            )
        verify_info = dict(info)
        verify_info["privileged_agents"] = privileged_info
        verifier_score = float(self.verifier.score(selected_plan, verify_info))
        if not all_loop1_candidates_rejected and not self._verifier_accepts(verifier_score):
            refinement = self._refine_rejected_loop1_plan(
                selected_plan, verify_info, selected_source
            )
            if refinement is not None:
                selected_plan, verifier_score, refinement_payload = refinement
                plan_payload.update(refinement_payload)
                plan_payload["selected_plan"] = selected_plan
                plan_payload["selected_score"] = verifier_score
                selected_source = str(plan_payload["selected_source"])
                selected_score_raw = verifier_score
                result.selected_plan = selected_plan
                result.selected_source = selected_source
                result.selected_score_raw = selected_score_raw
        verifier_result = getattr(self.verifier, "last_result", None)
        result.verifier_score = verifier_score
        result.verifier_passed = self._verifier_accepts(verifier_score)
        if all_loop1_candidates_rejected:
            result.verifier_passed = False
        normal_ep_gate_triggered = False
        normal_ep = None
        if verifier_result is not None:
            try:
                normal_ep = float((verifier_result.soft or {}).get("ep"))
            except (TypeError, ValueError):
                normal_ep = None
        if self.normal_ep_gate and normal_ep is not None:
            if normal_ep < self.normal_ep_min:
                self._normal_ep_low_streak += 1
            else:
                self._normal_ep_low_streak = 0
            normal_ep_gate_triggered = (
                result.verifier_passed
                and self._normal_ep_low_streak >= self.normal_ep_gate_steps
            )
            if normal_ep_gate_triggered:
                result.verifier_passed = False
        else:
            self._normal_ep_low_streak = 0
        if verifier_result is not None:
            result.debug["verifier"] = verifier_result.to_debug_dict()
        if self.normal_ep_gate:
            result.debug["normal_ep_gate"] = {
                "enabled": True,
                "ep": normal_ep,
                "minimum": self.normal_ep_min,
                "low_streak": self._normal_ep_low_streak,
                "required_steps": self.normal_ep_gate_steps,
                "triggered": normal_ep_gate_triggered,
            }

        # A pedestrian-caused TTC failure inside the caution range is an
        # unconditional one-second brake. Do not search progress-scaled
        # variants and do not enter the maneuver-recovery ladder: both allowed
        # the planner to release the yield and accelerate again too early.
        vru_observation = self._nearest_forward_vru(info, privileged_info)
        if (
            vru_observation is not None
            and self._vru_ttc_vetoed(verifier_result)
        ):
            return self._start_pedestrian_ttc_brake_hold(
                info=info,
                result=result,
                verifier_result=verifier_result,
                nearest_pedestrian=vru_observation,
            )

        # This is a hard trigger, independent of the verifier score. Preserve
        # PDMS/EP as the primary cause when either already rejected the plan;
        # otherwise override the passing verdict and enter recovery.
        stuck_recovery_triggered = bool(stuck_recovery.get("triggered"))
        stuck_forced_recovery = bool(
            stuck_recovery_triggered
            and result.verifier_passed
        )
        if stuck_forced_recovery:
            result.verifier_passed = False
            stuck_recovery["forced_recovery"] = True
            self.log.warning(
                "ego stayed within %.2fm for %.2fs; forcing recovery",
                self.stuck_recovery_position_tolerance_m,
                float(stuck_recovery.get("stationary_sec", 0.0)),
            )
        elif stuck_recovery_triggered:
            stuck_recovery["forced_recovery"] = False

        # Log the score on EVERY step (pass and fail). The recovery branches
        # below only log on failure, so without this the terminal would only
        # ever show sub-threshold scores and look like the verifier is always 0.
        terms = ""
        if verifier_result is not None:
            submetrics = {**verifier_result.gates, **verifier_result.soft}
            terms = " " + " ".join(
                "%s=%.2f" % (name, float(value)) for name, value in submetrics.items()
            )
        score_relation = ">=" if verifier_score >= self.verifier_threshold else "<"
        verdict_label = (
            "recover:stuck"
            if stuck_forced_recovery
            else "recover:normal_ep_gate" if normal_ep_gate_triggered
            else "pass" if result.verifier_passed else "recover"
        )
        self.log.info(
            "verifier score %.3f %s %.3f (%s)%s%s",
            verifier_score,
            score_relation,
            self.verifier_threshold,
            verdict_label,
            terms,
            (
                " normal_ep_gate=%d/%d"
                % (self._normal_ep_low_streak, self.normal_ep_gate_steps)
                if normal_ep_gate_triggered else ""
            ),
        )

        # Settle the recovery primitive selected on an earlier tick: the plan
        # was already handed to CARLA back then, so its outcome only becomes
        # observable here — verifier passing means the maneuver resolved the
        # situation (episode memory resets); failing marks it as a failed
        # attempt so the re-selection below can exclude it.
        settled = self.recovery_tracker.observe_verifier(
            result.verifier_passed,
            failure_reason=(
                "stuck:stationary_sec=%.3f,tolerance_m=%.3f"
                % (
                    float(stuck_recovery.get("stationary_sec", 0.0)),
                    self.stuck_recovery_position_tolerance_m,
                )
                if stuck_forced_recovery
                else "normal_ep_gate:ep=%.3f,streak=%d/%d"
                % (normal_ep, self._normal_ep_low_streak, self.normal_ep_gate_steps)
                if normal_ep_gate_triggered and normal_ep is not None
                else f"verifier_score={verifier_score:.3f}"
            ),
            frame_index=frame_index,
            selector=self._recovery_selector_kind(),
        )
        if settled is not None:
            self.log.info(
                "recovery attempt '%s' settled: %s",
                settled.primitive, "success" if settled.success else "failed",
            )
            self._record_failure_memory(
                info,
                event_type="legacy_recovery_verdict",
                problem=getattr(self, "_pending_recovery_reason", None),
                method=settled.primitive,
                outcome="resolved" if settled.success else "rejected",
                details={
                    "verifier_score": verifier_score,
                    "verifier_threshold": self.verifier_threshold,
                    "gates": (
                        dict(verifier_result.gates)
                        if verifier_result is not None else {}
                    ),
                    "soft": (
                        dict(verifier_result.soft)
                        if verifier_result is not None else {}
                    ),
                },
            )
        semantic_takeover = None
        if result.verifier_passed:
            self._clear_recovery_vlm_answer()
            # Failure memory is intentionally NOT cleared merely because one
            # Loop 1 trajectory passed. It must remain in Loop 1 continuously
            # for the configured grace interval (1 s by default); a recovery
            # re-entry before then cancels this timer and preserves history.
            failure_memory = getattr(self, "failure_memory", None)
            if failure_memory is not None:
                cleared = failure_memory.observe_loop1(
                    float(info.get("timestamp", 0.0) or 0.0)
                )
                if cleared:
                    self.log.info(
                        "recovery failure memory cleared after %.2fs stable in Loop 1",
                        failure_memory.clear_after_loop1_sec,
                    )
            sequence_cleared = self._sequence_action_manager().observe_loop1(
                float(info.get("timestamp", 0.0) or 0.0),
                clear_after_sec=float(
                    getattr(self, "_recovery_sequence_clear_after_loop1_sec", 1.0)
                ),
            )
            if sequence_cleared:
                self.log.info(
                    "recovery sequence intent cleared after stable Loop 1"
                )
            # Resolved with nothing in flight: the episode is over, drop its
            # exclusions.
            if self.recovery_tracker.pending is None and len(self.recovery_tracker.memory) > 0:
                self.recovery_tracker.reset()
            self._note_recovery_ended(info)
            if semantic_manager is not None:
                semantic_takeover = semantic_manager.poll(
                    info,
                    recovery_active=False,
                )
            if semantic_takeover is None:
                return result

            # PDMS passed, but two sparse high-confidence semantic checks agree
            # that the basic loop is severely off-route. Keep the PDMS verdict
            # intact and enter the existing recovery executor as a soft gate.
            result.debug["semantic_takeover"] = semantic_takeover
            self._pending_recovery_reason = str(
                semantic_takeover.get("reasoning")
                or semantic_takeover.get("reason_code")
                or "semantic route mismatch"
            )
            self._note_recovery_started(info, source="semantic_vlm")
            self.log.warning(
                "semantic VLM requested recovery: code=%s confidence=%.3f reason=%s",
                semantic_takeover.get("reason_code"),
                float(semantic_takeover.get("confidence", 0.0) or 0.0),
                self._pending_recovery_reason,
            )
        else:
            # A sub-threshold PDMS score always takes priority. Invalidate any
            # semantic result immediately; if its VLM call is still in flight,
            # the deterministic recovery path below must not wait for it.
            self._note_recovery_started(
                info,
                source=(
                    "stuck"
                    if stuck_forced_recovery
                    else "normal_ep_gate" if normal_ep_gate_triggered else "pdms"
                ),
            )

        # Recovery is now definitely triggering. Explanation and primitive
        # selection are independent: the recovery-table ablations disable the
        # former and directly ask the VLM to reason over the scene + PDMS result.
        # A completed semantic-takeover decision is a recovery *trigger*, not a
        # replacement for the common B5 recovery selector. Once the async
        # semantic call has completed, every C-table configuration still lets
        # the shared VLM choose the primitive/parameters. Only an actually
        # in-flight semantic call suppresses a second concurrent VLM request.
        allow_recovery_vlm = not (
            semantic_manager is not None and semantic_manager.busy
        ) and not (
            self.recovery_explainer is not None
            and bool(getattr(self.recovery_explainer, "optional_async_busy", False))
        )
        if semantic_takeover is None:
            self._pending_recovery_reason = (
                "ego remained within %.2fm for %.2fs"
                % (
                    self.stuck_recovery_position_tolerance_m,
                    float(stuck_recovery.get("stationary_sec", 0.0)),
                )
                if stuck_forced_recovery
                else None
            )
        explain_enabled = bool(
            self.recovery_explainer is not None
            and getattr(getattr(self.recovery_explainer, "cfg", None), "enabled", False)
        )
        cached_vlm_answer = (
            self._cached_recovery_vlm_answer(info) if explain_enabled else None
        )
        if explain_enabled and cached_vlm_answer is not None:
            explanation, cache_age_sec = cached_vlm_answer
            explanation["cache_reused"] = True
            explanation["cache_age_sec"] = cache_age_sec
            result.debug["recovery_reason"] = explanation
            self._pending_recovery_reason = explanation.get("reason")
            self.log.warning(
                "reusing recovery VLM answer from %.2f simulator seconds ago: %s",
                cache_age_sec,
                explanation.get("reason"),
            )
        elif explain_enabled and allow_recovery_vlm:
            try:
                explanation = self.recovery_explainer.explain(
                    frame_index=getattr(self.selector, "frame_index", 0),
                    camera_images=obs.get("rgb", {}),
                    info=info,
                    selected_plan=selected_plan,
                    verifier_score=verifier_score,
                    verifier_threshold=self.verifier_threshold,
                    verifier_gates=dict(verifier_result.gates) if verifier_result is not None else {},
                    verifier_soft=dict(verifier_result.soft) if verifier_result is not None else {},
                    road_context=self._recovery_road_context(info),
                )
                result.debug["recovery_reason"] = explanation
                self._pending_recovery_reason = explanation.get("reason")
                self._remember_recovery_vlm_answer(info, explanation)
                if explanation.get("reason"):
                    self.log.warning("recovery reason: %s", explanation["reason"])
                elif explanation.get("error"):
                    self.log.debug("recovery reason unavailable: %s", explanation["error"])
            except Exception:
                self.log.exception("recovery_explainer.explain failed")
        elif explain_enabled and not allow_recovery_vlm:
            self.log.info(
                "Skipping optional recovery VLM work; semantic VLM owns the shared backend"
            )

        verifier_gates = (
            dict(verifier_result.gates) if verifier_result is not None else {}
        )
        verifier_soft = (
            dict(verifier_result.soft) if verifier_result is not None else {}
        )
        self._pending_recovery_verifier_context = {
            "score": verifier_score,
            "threshold": self.verifier_threshold,
            "gates": verifier_gates,
            "soft": verifier_soft,
        }
        self._record_failure_memory(
            info,
            event_type="recovery_problem",
            problem=getattr(self, "_pending_recovery_reason", None),
            outcome="requires_recovery",
            details={
                "source": (
                    "semantic_vlm"
                    if semantic_takeover is not None
                    else (
                        "stuck"
                        if stuck_forced_recovery
                        else "normal_ep_gate" if normal_ep_gate_triggered else "pdms"
                    )
                ),
                "verifier_score": verifier_score,
                "verifier_threshold": self.verifier_threshold,
                "gates": verifier_gates,
                "soft": verifier_soft,
                "route_instruction": str(
                    info.get("route_instruction")
                    or info.get("instruction")
                    or info.get("command")
                    or ""
                ),
            },
        )

        output_num_poses = int(proposals.shape[1]) if proposals.ndim == 3 else len(selected_plan)

        # Commit-and-hold recovery: verify each primitive, commit the first that
        # passes for a fixed window, and bypass the legacy per-frame path.
        if self.recovery_commit_and_hold:
            return self._begin_commit_and_hold(
                result,
                obs=obs,
                info=info,
                privileged_info=privileged_info,
                output_num_poses=output_num_poses,
                verifier_score=verifier_score,
                allow_vlm_selection=allow_recovery_vlm,
                trigger_reason=(
                    f"semantic VLM: {self._pending_recovery_reason}"
                    if semantic_takeover is not None
                    else (
                        "stuck trigger: ego remained within %.2fm for %.2fs"
                        % (
                            self.stuck_recovery_position_tolerance_m,
                            float(stuck_recovery.get("stationary_sec", 0.0)),
                        )
                        if stuck_forced_recovery
                        else (
                            "normal EP gate: ep=%.3f for %d/%d consecutive Loop-1 steps"
                            % (
                                normal_ep,
                                self._normal_ep_low_streak,
                                self.normal_ep_gate_steps,
                            )
                            if normal_ep_gate_triggered and normal_ep is not None
                            else None
                        )
                    )
                ),
            )

        # recovery: rule-based proposals through the SAME selection pipeline
        recovered_payload = self._recover(
            obs=obs,
            info=info,
            info_history=info_history,
            privileged_info=privileged_info,
            output_num_poses=output_num_poses,
            allow_vlm_selection=allow_recovery_vlm,
        )
        if recovered_payload is None:
            self.log.warning(
                "%s; recovery produced nothing, keeping learned plan",
                (
                    "stuck trigger"
                    if stuck_forced_recovery
                    else "verifier score %.3f < %.3f"
                    % (verifier_score, self.verifier_threshold)
                ),
            )
            return result

        self.log.warning(
            "%s; recovered via rule-based planner",
            (
                "stuck trigger"
                if stuck_forced_recovery
                else "verifier score %.3f < %.3f"
                % (verifier_score, self.verifier_threshold)
            ),
        )
        selected_plan, selected_score_raw, selected_source = _unpack_payload(recovered_payload)
        result.selected_plan = selected_plan
        result.selected_score_raw = selected_score_raw
        result.selected_source = selected_source
        result.plan_payload = recovered_payload
        result.recovered = True
        return result



    def _observe_sequence_menu_stationary(
        self, info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Track position dwell time for opening the larger sequence menu."""
        cfg = getattr(getattr(self, "recovery_explainer", None), "cfg", None)
        timeout_sec = float(
            getattr(cfg, "sequence_menu_escalate_after_stationary_sec", 0.0)
            or 0.0
        )
        tolerance_m = float(
            getattr(
                cfg,
                "sequence_menu_stationary_position_tolerance_m",
                0.5,
            )
            or 0.0
        )
        state, anchor_xy, anchor_ts = observe_position_dwell(
            info,
            enabled=timeout_sec > 0.0,
            timeout_sec=timeout_sec,
            position_tolerance_m=tolerance_m,
            anchor_xy=getattr(
                self, "_sequence_menu_stationary_anchor_xy", None
            ),
            anchor_ts=getattr(
                self, "_sequence_menu_stationary_anchor_ts", None
            ),
        )
        self._sequence_menu_stationary_anchor_xy = anchor_xy
        self._sequence_menu_stationary_anchor_ts = anchor_ts
        self._sequence_menu_stationary_sec = float(state["stationary_sec"])
        return state

    def _observe_sequence_menu_front_obstacle(
        self,
        info: Dict[str, Any],
        privileged_info: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Observe nearby obstacles that should open the recovery menu."""
        cfg = getattr(getattr(self, "recovery_explainer", None), "cfg", None)
        distance_limit_m = max(
            0.0,
            float(
                getattr(
                    cfg,
                    "sequence_menu_front_obstacle_distance_m",
                    0.0,
                )
                or 0.0
            ),
        )
        stopped_vehicle_sec = max(
            0.0,
            float(
                getattr(
                    cfg,
                    "sequence_menu_front_stopped_vehicle_stationary_sec",
                    0.0,
                )
                or 0.0
            ),
        )
        speed_threshold_mps = max(
            0.0,
            float(
                getattr(
                    cfg,
                    "sequence_menu_front_stopped_vehicle_speed_threshold_mps",
                    0.05,
                )
                or 0.0
            ),
        )
        position_tolerance_m = max(
            0.0,
            float(
                getattr(
                    cfg,
                    "sequence_menu_front_stopped_vehicle_position_tolerance_m",
                    0.05,
                )
                or 0.0
            ),
        )
        state: Dict[str, Any] = {
            "enabled": distance_limit_m > 0.0,
            "triggered": False,
            "distance_limit_m": distance_limit_m,
            "stopped_vehicle_timeout_sec": stopped_vehicle_sec,
            "stopped_vehicle_speed_threshold_mps": speed_threshold_mps,
            "stopped_vehicle_position_tolerance_m": position_tolerance_m,
        }
        observations = (
            front_corridor_object_observations(
                info=info,
                privileged_info=self._fresh_recovery_agents(privileged_info),
                max_distance_m=distance_limit_m,
            )
            if distance_limit_m > 0.0
            else []
        )
        triggered_obstacles: List[Dict[str, Any]] = []
        for observation in observations:
            object_type = str(observation.get("object_type", "unknown"))
            if not (
                object_type.startswith("vehicle")
                or object_type.startswith("pedestrian")
                or object_type.startswith("walker")
            ):
                triggered_obstacles.append({
                    "triggered": True,
                    "trigger_kind": "non_traffic_obstacle",
                    **observation,
                })

        tracker = getattr(self, "_sequence_menu_front_vehicle_dwell", {})
        try:
            now = float(info["timestamp"])
            if not np.isfinite(now):
                raise ValueError("non-finite timestamp")
        except (KeyError, TypeError, ValueError):
            now = None

        vehicle_tracking_enabled = bool(
            distance_limit_m > 0.0 and stopped_vehicle_sec > 0.0
        )
        seen_vehicle_ids: set[str] = set()
        if vehicle_tracking_enabled and now is not None:
            for observation in observations:
                object_type = str(observation.get("object_type", "unknown"))
                if not object_type.startswith("vehicle"):
                    continue
                agent_id = str(observation.get("agent_id", "unknown"))
                world_position = observation.get("world_position")
                speed_mps = observation.get("speed_mps")
                if agent_id == "unknown":
                    continue
                seen_vehicle_ids.add(agent_id)
                try:
                    xy = np.asarray(world_position, dtype=np.float64).reshape(-1)[:2]
                    speed = float(speed_mps)
                    if (
                        xy.size != 2
                        or not np.isfinite(xy).all()
                        or not np.isfinite(speed)
                    ):
                        raise ValueError("invalid vehicle motion observation")
                except (TypeError, ValueError):
                    tracker.pop(agent_id, None)
                    continue

                previous = tracker.get(agent_id)
                if speed > speed_threshold_mps:
                    tracker.pop(agent_id, None)
                    continue
                if previous is None or now < float(previous["anchor_ts"]):
                    previous = {
                        "anchor_world_xy": xy.copy(),
                        "anchor_ts": now,
                    }
                distance_from_anchor_m = float(
                    np.linalg.norm(xy - previous["anchor_world_xy"])
                )
                if distance_from_anchor_m > position_tolerance_m:
                    previous = {
                        "anchor_world_xy": xy.copy(),
                        "anchor_ts": now,
                    }
                    distance_from_anchor_m = 0.0
                stationary_sec = max(0.0, now - float(previous["anchor_ts"]))
                previous.update({
                    "last_seen_ts": now,
                    "stationary_sec": stationary_sec,
                    "distance_from_anchor_m": distance_from_anchor_m,
                })
                tracker[agent_id] = previous
                if stationary_sec >= stopped_vehicle_sec:
                    triggered_obstacles.append({
                        "triggered": True,
                        "trigger_kind": "stationary_vehicle",
                        **observation,
                        "stationary_sec": stationary_sec,
                        "distance_from_anchor_m": distance_from_anchor_m,
                    })

            for agent_id in tuple(tracker):
                if agent_id not in seen_vehicle_ids:
                    tracker.pop(agent_id, None)
        else:
            tracker.clear()

        self._sequence_menu_front_vehicle_dwell = tracker
        if triggered_obstacles:
            state.update(
                min(
                    triggered_obstacles,
                    key=lambda item: float(item.get("distance_m", float("inf"))),
                )
            )
        self._sequence_menu_front_obstacle = state
        return state

    def _sequence_vlm_menu_subset(
        self,
        info: Optional[Dict[str, Any]] = None,
    ) -> tuple[frozenset[str], tuple[str, ...]]:
        """Return the current sequence menu and newly exposed escalation ids.

        The expanded menu contains ordinary-road avoidance maneuvers such as
        swerves, lane changes, and lane bypasses.  Keep those actions hidden
        throughout an HD-map junction encounter (including its configured
        lookahead), where they can confuse a legal route branch with an
        adjacent lane and latch an unexecutable bypass intent.
        """
        cfg = self.recovery_explainer.cfg
        base = tuple(
            str(name)
            for name in getattr(cfg, "sequence_menu_primitives", ()) or ()
        )
        extras = tuple(
            str(name)
            for name in getattr(
                cfg, "sequence_menu_escalation_primitives", ()
            )
            or ()
        )
        junction_context = (
            info.get("junction_route_heading_filter")
            if isinstance(info, dict)
            else None
        )
        if isinstance(junction_context, dict) and bool(
            junction_context.get("active", False)
        ):
            return frozenset(base), ()
        trigger_sec = max(
            0.0,
            float(
                getattr(
                    cfg,
                    "sequence_menu_escalate_after_stationary_sec",
                    0.0,
                )
                or 0.0
            ),
        )
        stationary_sec = float(
            getattr(self, "_sequence_menu_stationary_sec", 0.0)
        )
        # Once front sensing is enabled, stopped time alone is not evidence
        # that the road still requires a lateral maneuver.
        stationary_triggered = bool(
            trigger_sec > 0.0 and stationary_sec >= trigger_sec
            and not getattr(self, "_sequence_menu_front_obstacle", {}).get("enabled")
        )
        front_obstacle_triggered = bool(
            getattr(self, "_sequence_menu_front_obstacle", {}).get(
                "triggered", False
            )
        )
        # Empty base already means "all allowed primitives", so there is
        # nothing for a second-stage menu to add.
        if (
            not base
            or not extras
            or not (stationary_triggered or front_obstacle_triggered)
        ):
            return frozenset(base), ()

        merged = list(base)
        added = []
        for name in extras:
            if name not in merged:
                merged.append(name)
                added.append(name)
        return frozenset(merged), tuple(added)

    def _tool_rule_blocked_escape_choice(
        self,
        *,
        info: Dict[str, Any],
        nominal_eligible: Sequence[tuple[int, float]],
        names: Sequence[str],
    ) -> Optional[tuple[int, float, Dict[str, Any]]]:
        """Choose a verified lateral tool after a persistent front blockage.

        This runs only after nominal PDMS, complete-footprint HD-map,
        junction-heading, and no-progress tabu filtering. It changes who
        chooses among safe tools, not the safety gates themselves.
        """
        cfg = getattr(getattr(self, "recovery_explainer", None), "cfg", None)
        if not bool(getattr(cfg, "tool_rule_blocked_escape_enabled", False)):
            return None
        junction_context = info.get("junction_route_heading_filter")
        if isinstance(junction_context, dict) and bool(
            junction_context.get("active", False)
        ):
            return None

        front = dict(
            getattr(self, "_sequence_menu_front_obstacle", {}) or {}
        )
        if not bool(front.get("triggered", False)):
            return None
        required_sec = max(
            0.0,
            float(getattr(cfg, "tool_rule_blocked_escape_after_sec", 10.0)),
        )
        stationary_sec = max(
            float(getattr(self, "_sequence_menu_stationary_sec", 0.0) or 0.0),
            float(front.get("stationary_sec", 0.0) or 0.0),
        )
        if stationary_sec + 1e-6 < required_sec:
            return None

        priority = tuple(
            str(name)
            for name in (
                getattr(cfg, "tool_rule_blocked_escape_primitives", ()) or ()
            )
        )
        eligible_by_name = {
            str(names[int(idx)]): (int(idx), float(score))
            for idx, score in nominal_eligible
            if 0 <= int(idx) < len(names)
        }
        for action in priority:
            if is_leftward_primitive(action) and suppress_leftward_options(info):
                continue
            if action.startswith("lane_"):
                direction = "right" if action.endswith("_right") else "left"
                if adjacent_driving_lane_absent(info, direction):
                    continue
            if (
                action == "lane_bypass_right"
                and "lane_right" not in eligible_by_name
            ):
                continue
            selected = eligible_by_name.get(action)
            if selected is None:
                continue
            idx, score = selected
            return idx, score, {
                "tool_rule": "persistent_front_blockage_escape",
                "trigger_kind": front.get("trigger_kind"),
                "obstacle_type": front.get("object_type"),
                "obstacle_id": front.get("agent_id"),
                "obstacle_distance_m": front.get("distance_m"),
                "stationary_sec": stationary_sec,
                "required_stationary_sec": required_sec,
                "priority": list(priority),
            }
        return None

    def _build_hold_payload(
        self,
        idx: int,
        names: List[str],
        proposals: np.ndarray,
        states: List[np.ndarray],
        score: float,
        *,
        selected_plan: Optional[np.ndarray] = None,
        selected_states: Optional[np.ndarray] = None,
        lateral_distance_m: Optional[float] = None,
        recovery_name: Optional[str] = None,
        sequence_steps: Optional[List[Dict[str, Any]]] = None,
        waypoint: Optional[Dict[str, float]] = None,
        execute_seconds: Optional[float] = None,
        front_blockage: Optional[str] = None,
        escape_direction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Assemble the commit-and-hold payload for primitive ``idx``."""
        name = str(recovery_name or names[idx])
        payload = {
            "selected_plan": np.asarray(
                proposals[idx] if selected_plan is None else selected_plan,
                dtype=np.float32,
            ),
            "selected_score": score,
            "selected_source": f"recovery:{name}",
            "fallback_primitive": name,
            "fallback_mode": "hold",
            "fallback_states": (
                np.asarray(
                    states[idx] if selected_states is None else selected_states,
                    dtype=np.float32,
                )
                if (selected_states is not None or idx < len(states))
                else None
            ),
        }
        if lateral_distance_m is not None:
            payload["fallback_parameters"] = {
                "lateral_distance_m": float(lateral_distance_m),
            }
        if sequence_steps:
            payload["fallback_sequence"] = [dict(step) for step in sequence_steps]
            payload["fallback_parameters"] = {
                "sequence": [dict(step) for step in sequence_steps],
            }
        if waypoint is not None:
            payload["fallback_waypoint"] = {
                "forward_m": float(waypoint["forward_m"]),
                "right_m": float(waypoint["right_m"]),
            }
            payload["fallback_parameters"] = {
                "waypoint": dict(payload["fallback_waypoint"]),
            }
        if execute_seconds is not None:
            payload["fallback_execute_seconds"] = float(execute_seconds)
        if front_blockage:
            payload["recovery_front_blockage"] = str(front_blockage)
        if escape_direction:
            payload["recovery_escape_direction"] = str(escape_direction)
        return payload

    def _pdms_feasible_parameter_range(
        self,
        *,
        primitive_name: str,
        info: Dict[str, Any],
        privileged_info: Optional[List[Dict[str, Any]]],
        output_num_poses: int,
        sample_count: int,
    ) -> Optional[tuple[float, float]]:
        """Return the widest sampled PDMS-threshold-safe lateral interval.

        B5 uses this before the VLM call. Every grid point in the returned run
        meets the PDMS score threshold and the existing footprint/
        full-DAC guards. Disjoint safe runs are reduced to the widest one so the
        VLM receives one simple inclusive range and chooses only once.
        """
        static_range = recovery_lateral_distance_range(primitive_name)
        if static_range is None:
            return None
        lower, upper = (float(static_range[0]), float(static_range[1]))
        count = max(2, int(sample_count))
        values = np.linspace(lower, upper, num=count, dtype=np.float64)
        verify_info = dict(info)
        verify_info["privileged_agents"] = privileged_info
        generated_plans: List[Optional[np.ndarray]] = []
        for value in values:
            generated = materialize_recovery_primitive(
                primitive_name=primitive_name,
                info=info,
                privileged_info=privileged_info,
                output_num_poses=output_num_poses,
                lateral_distance_m=float(value),
            )
            if generated is None:
                generated_plans.append(None)
                continue
            traj, _states = generated
            generated_plans.append(np.asarray(traj, dtype=np.float32))

        feasible: List[bool] = [False] * len(generated_plans)
        valid_indices = [
            idx for idx, traj in enumerate(generated_plans) if traj is not None
        ]
        valid_plans = [generated_plans[idx] for idx in valid_indices]
        batch_score = getattr(self.verifier, "score_batch", None)
        can_batch = bool(
            callable(batch_score)
            and valid_plans
            and len({plan.shape for plan in valid_plans}) == 1
        )
        if can_batch:
            stacked = np.stack(valid_plans, axis=0)
            scores = np.asarray(batch_score(stacked, verify_info), dtype=np.float64)
            passed = np.asarray(
                [self._verifier_accepts(float(score)) for score in scores],
                dtype=bool,
            )
            passed_positions = np.flatnonzero(passed)
            if len(passed_positions):
                endpoint_tiers = getattr(
                    self.verifier, "last_batch_endpoint_tiers", None
                )
                if (
                    bool(getattr(self, "recovery_hd_map_endpoint_only", False))
                    and endpoint_tiers is not None
                    and len(endpoint_tiers) == len(stacked)
                ):
                    passed[passed_positions] &= (
                        np.asarray(endpoint_tiers, dtype=np.float64)[passed_positions]
                        >= 1.0
                    )
                else:
                    passed[passed_positions] &= ~self._footprint_offroad_batch(
                        stacked[passed_positions], info
                    )
            for position, idx in enumerate(valid_indices):
                feasible[idx] = bool(passed[position])
        else:
            for idx in valid_indices:
                traj = generated_plans[idx]
                score = float(self.verifier.score(traj, verify_info))
                passed = self._verifier_accepts(score)
                if passed and self._verified_trajectory_offroad(traj, info):
                    passed = False
                if (
                    passed
                    and self.recovery_full_plan_dac
                    and self._primitive_tier_gate_fails(traj, info)
                ):
                    passed = False
                feasible[idx] = bool(passed)

        runs: List[tuple[int, int]] = []
        start: Optional[int] = None
        for idx, passed in enumerate(feasible + [False]):
            if passed and start is None:
                start = idx
            elif not passed and start is not None:
                runs.append((start, idx - 1))
                start = None
        if not runs:
            self.log.info(
                "B5 parameter preverification: '%s' has no PDMS-safe samples",
                primitive_name,
            )
            return None
        best_start, best_end = max(
            runs,
            key=lambda run: (
                float(values[run[1]] - values[run[0]]),
                run[1] - run[0] + 1,
            ),
        )
        result = (float(values[best_start]), float(values[best_end]))
        self.log.info(
            "B5 parameter preverification: '%s' safe lateral range "
            "[%.3f, %.3f] from %d samples",
            primitive_name,
            result[0],
            result[1],
            count,
        )
        return result

    @staticmethod
    def _recovery_probability_sample_counts(
        probabilities: Sequence[float],
        total_count: float,
    ) -> List[int]:
        """Allocate one exact trajectory budget across VLM probabilities.

        Largest-remainder apportionment keeps the aggregate equal to the
        configured budget.  Every supplied primitive receives at least one
        sample (the VLM returns at most five candidates, while the shipped
        budget is 30).
        """
        weights = [max(0.0, float(value)) for value in probabilities]
        if not weights:
            return []
        budget = max(
            len(weights),
            int(math.floor(max(0.0, float(total_count)) + 0.5)),
        )
        weight_sum = float(sum(weights))
        if weight_sum <= 0.0:
            weights = [1.0] * len(weights)
            weight_sum = float(len(weights))
        quotas = [budget * weight / weight_sum for weight in weights]
        counts = [int(math.floor(quota)) for quota in quotas]
        remaining = budget - sum(counts)
        remainder_order = sorted(
            range(len(weights)),
            key=lambda idx: (
                quotas[idx] - counts[idx],
                weights[idx],
                -idx,
            ),
            reverse=True,
        )
        for idx in remainder_order[:remaining]:
            counts[idx] += 1

        # Hamilton allocation can assign zero to a very-low-probability item.
        # Preserve the existing guarantee that each VLM candidate is tried.
        for recipient in [idx for idx, count in enumerate(counts) if count == 0]:
            donors = [idx for idx, count in enumerate(counts) if count > 1]
            if not donors:
                break
            donor = max(
                donors,
                key=lambda idx: (
                    counts[idx] - quotas[idx],
                    counts[idx],
                    weights[idx],
                    -idx,
                ),
            )
            counts[donor] -= 1
            counts[recipient] = 1
        return counts

    def _materialize_recovery_action_profile(self, **kwargs):
        return materialize_recovery_action_profile(**kwargs)

    def _search_recovery_temporal_trees(
        self,
        *,
        action_specs: Dict[str, Dict[str, object]],
        candidate_indices: Sequence[int],
        names: Sequence[str],
        info: Dict[str, Any],
        privileged_info: Optional[List[Dict[str, Any]]],
        output_num_poses: int,
        lateral_distance_ranges: Optional[Dict[int, tuple[float, float]]],
        obstacle_clearance_m: Optional[float] = None,
    ) -> Dict[str, TemporalSearchResult]:
        """Expand and score frozen-world temporal trees for every action."""
        cfg = self.recovery_explainer.cfg
        runtime_recovery_cfg = getattr(self, "recovery_cfg", None)
        configured_obstacle_clearance = float(
            getattr(runtime_recovery_cfg, "obstacle_clearance_m", 8.0)
        )
        effective_obstacle_clearance = configured_obstacle_clearance
        if obstacle_clearance_m is not None:
            effective_obstacle_clearance = max(
                configured_obstacle_clearance,
                float(obstacle_clearance_m),
            )
        budget = max(
            1,
            int(getattr(cfg, "temporal_tree_budget_per_primitive", 18)),
        )
        verify_info = dict(info)
        verify_info["privileged_agents"] = privileged_info
        results: Dict[str, TemporalSearchResult] = {}
        previous_debug = getattr(self.verifier, "_debug", None)
        if previous_debug is not None:
            self.verifier._debug = False
        try:
            for raw_idx in candidate_indices:
                idx = int(raw_idx)
                action = str(names[idx])
                spec = action_specs.get(action)
                if spec is None:
                    continue
                profiles = build_temporal_profile_tree(
                    action=action,
                    spec=spec,
                    budget=budget,
                    duration_scales=getattr(
                        cfg,
                        "temporal_tree_duration_scales",
                        (0.75, 1.0, 1.25),
                    ),
                    start_delays_sec=getattr(
                        cfg,
                        "temporal_tree_start_delays_sec",
                        (0.0, 0.5, 1.0, 1.5),
                    ),
                    lateral_range=(
                        None
                        if lateral_distance_ranges is None
                        else lateral_distance_ranges.get(idx)
                    ),
                    bypass_target_speeds_mps=getattr(
                        cfg,
                        "temporal_tree_bypass_target_speeds_mps",
                        (6.0, 8.0, 10.0, 12.0),
                    ),
                    bypass_accels_mps2=getattr(
                        cfg,
                        "temporal_tree_bypass_accels_mps2",
                        (3.5, 4.5),
                    ),
                    bypass_total_steps=getattr(
                        cfg,
                        "temporal_tree_bypass_total_steps",
                        (8, 9, 10),
                    ),
                    bypass_entry_steps=getattr(
                        cfg,
                        "temporal_tree_bypass_entry_steps",
                        (2, 3, 4),
                    ),
                    bypass_return_steps=getattr(
                        cfg,
                        "temporal_tree_bypass_return_steps",
                        (2, 3),
                    ),
                    bypass_settle_steps=getattr(
                        cfg,
                        "temporal_tree_bypass_settle_steps",
                        (1, 2),
                    ),
                )
                leaves: List[TemporalSearchLeaf] = []
                for profile_index, profile in enumerate(profiles):
                    profile_record = {
                        "profile_id": f"{action}/temporal_{profile_index:03d}",
                        **dict(profile),
                    }
                    if action.startswith("lane_bypass_"):
                        profile_record["obstacle_clearance_m"] = float(
                            effective_obstacle_clearance
                        )
                    generated = self._materialize_recovery_action_profile(
                        action=action,
                        profile=profile_record,
                        info=info,
                        privileged_info=privileged_info,
                        output_num_poses=output_num_poses,
                        lane_bypass_min_gap_m=getattr(
                            runtime_recovery_cfg, "lane_bypass_min_gap_m", 0.0
                        ),
                        lane_bypass_clearance_buffer_m=getattr(
                            runtime_recovery_cfg,
                            "lane_bypass_clearance_buffer_m",
                            1.0,
                        ),
                        lane_bypass_required_time_sec=getattr(
                            runtime_recovery_cfg,
                            "lane_bypass_required_time_sec",
                            3.5,
                        ),
                        lane_bypass_min_time_gap_sec=getattr(
                            runtime_recovery_cfg,
                            "lane_bypass_min_time_gap_sec",
                            4.5,
                        ),
                        obstacle_clearance_m=effective_obstacle_clearance,
                    )
                    if generated is None:
                        leaves.append(
                            TemporalSearchLeaf(
                                profile=profile_record,
                                score=0.0,
                                accepted=False,
                                failure_flags=["generation_failed"],
                            )
                        )
                        continue
                    trajectory, selected_states = generated
                    score = float(self.verifier.score(trajectory, verify_info))
                    last = getattr(self.verifier, "last_result", None)
                    gates = dict(getattr(last, "gates", {}) or {})
                    soft = dict(getattr(last, "soft", {}) or {})
                    verifier_passed = self._verifier_accepts(score)
                    footprint_failed = bool(
                        verifier_passed
                        and self._verified_trajectory_offroad(trajectory, info)
                    )
                    full_dac_failed = bool(
                        verifier_passed
                        and not footprint_failed
                        and self.recovery_full_plan_dac
                        and self._primitive_tier_gate_fails(trajectory, info)
                    )
                    failure_flags: List[str] = []
                    if not verifier_passed:
                        failure_flags.append("below_verifier_threshold")
                    if footprint_failed:
                        failure_flags.append("footprint_offroad")
                    if full_dac_failed:
                        failure_flags.append("full_plan_dac_failed")
                    state_array = np.asarray(selected_states, dtype=np.float32)
                    active_rows = np.flatnonzero(
                        np.any(np.abs(state_array[:, :2]) > 1e-4, axis=1)
                    )
                    if active_rows.size:
                        profile_record["execute_seconds"] = float(
                            (int(active_rows[-1]) + 1)
                            * RECOVERY_PLAN_DT_SEC
                        )
                        if action.startswith("lane_bypass_"):
                            # The active horizon ends after the route-lane
                            # return plus its straight wheel-recentering rows.
                            # Record the exact return completion separately so
                            # temporal reduction can prefer minimum opposing-
                            # lane exposure instead of rewarding extra settle
                            # progress. Legacy-auto uses the provider default.
                            settle_steps = int(
                                profile_record.get(
                                    "settle_steps", LANE_BYPASS_SETTLE_STEPS
                                )
                            )
                            profile_record["return_complete_seconds"] = max(
                                0.0,
                                profile_record["execute_seconds"]
                                - max(0, settle_steps) * RECOVERY_PLAN_DT_SEC,
                            )
                    leaves.append(
                        TemporalSearchLeaf(
                            profile=profile_record,
                            score=score,
                            accepted=bool(
                                verifier_passed
                                and not footprint_failed
                                and not full_dac_failed
                            ),
                            trajectory=np.asarray(trajectory, dtype=np.float32),
                            states=state_array,
                            gates=gates,
                            soft=soft,
                            failure_flags=failure_flags,
                        )
                    )
                result = reduce_temporal_search(action, leaves)
                results[action] = result
                self.log.info(
                    "temporal tree '%s': selected=%.3f peak=%.3f robust=%.3f "
                    "feasible=%d/%d profile=%s",
                    action,
                    result.best_score,
                    max(
                        (
                            float(leaf.score)
                            for leaf in result.leaves
                            if leaf.accepted
                        ),
                        default=0.0,
                    ),
                    result.robust_score,
                    result.feasible_count,
                    result.expanded_count,
                    (
                        result.best_leaf.profile
                        if result.best_leaf is not None
                        else None
                    ),
                )
        finally:
            if previous_debug is not None:
                self.verifier._debug = previous_debug
        return results

    def _select_vlm_temporal_tree_result(
        self,
        *,
        plan_candidates: Sequence[Dict[str, Any]],
        temporal_results: Dict[str, TemporalSearchResult],
        names: Sequence[str],
        info: Dict[str, Any],
        front_blockage: Optional[object],
        escape_direction: Optional[object],
        needs_lane_bypass: Optional[bool] = None,
        lane_bypass_side: str = "left",
    ) -> tuple | _ForceBrakeFallback:
        """Bind the VLM's action choice to an already verified tree leaf."""
        ordered = sorted(
            enumerate(plan_candidates),
            key=lambda item: (
                float(item[1].get("probability", 0.0)),
                -item[0],
            ),
            reverse=True,
        )
        selected_candidate: Optional[Dict[str, Any]] = None
        selected_result: Optional[TemporalSearchResult] = None
        for _position, candidate in ordered:
            action = str(candidate.get("action") or "").strip()
            result = temporal_results.get(action)
            if result is not None and result.best_leaf is not None:
                selected_candidate = dict(candidate)
                selected_result = result
                break
        if selected_candidate is None or selected_result is None:
            return _ForceBrakeFallback("vlm_selected_no_temporal_safe_leaf")
        leaf = selected_result.best_leaf
        assert (
            leaf is not None
            and leaf.trajectory is not None
            and leaf.states is not None
        )
        action = selected_result.action
        try:
            idx = list(names).index(action)
        except ValueError:
            return _ForceBrakeFallback("temporal_selected_action_missing")
        metadata = {
            "front_blockage": front_blockage,
            "escape_direction": escape_direction,
            "probability": float(selected_candidate.get("probability", 0.0)),
            "temporal_tree_search": True,
            "selected_profile": dict(leaf.profile),
            "temporal_scorecard": selected_result.scorecard(),
            "execute_seconds": leaf.profile.get("execute_seconds"),
        }
        if needs_lane_bypass is not None:
            side = (
                "right"
                if str(lane_bypass_side).lower() == "right"
                else "left"
            )
            metadata[f"needs_lane_bypass_{side}"] = bool(needs_lane_bypass)
        self._record_failure_memory(
            info,
            event_type="primitive_selected",
            problem=getattr(self, "_pending_recovery_reason", None),
            method=action,
            outcome="selected_by_vlm_from_temporal_tree",
            details=metadata,
        )
        return (
            idx,
            float(leaf.score),
            np.asarray(leaf.trajectory, dtype=np.float32),
            np.asarray(leaf.states, dtype=np.float32),
            leaf.profile.get("lateral_distance_m"),
            [],
            metadata,
        )

    def _sample_and_score_recovery_plan_candidates(
        self,
        *,
        candidates: Sequence[Dict[str, Any]],
        action_specs: Dict[str, Dict[str, object]],
        candidate_indices: Sequence[int],
        names: Sequence[str],
        info: Dict[str, Any],
        privileged_info: Optional[List[Dict[str, Any]]],
        output_num_poses: int,
        lateral_distance_ranges: Optional[Dict[int, tuple[float, float]]],
        front_blockage: Optional[object],
        escape_direction: Optional[object],
        needs_lane_bypass_left: Optional[bool] = None,
        lane_bypass_side: str = "left",
    ) -> tuple | _ForceBrakeFallback:
        """Sample trajectory variants and return the highest-scoring safe one.

        VLM probabilities split one fixed aggregate search budget across the
        supplied primitives. The winning trajectory is selected by its raw
        verifier score; probability is used only as a deterministic tie-break
        after score.
        """
        runtime_recovery_cfg = getattr(self, "recovery_cfg", None)
        total_sample_count = float(
            getattr(
                self.recovery_explainer.cfg,
                "probability_sample_multiplier",
                30.0,
            )
        )
        rng = getattr(self, "_recovery_rng", None)
        if rng is None:
            rng = random.Random(0)
            self._recovery_rng = rng
        allowed_indices = {int(idx) for idx in candidate_indices}
        name_to_index = {
            str(name): idx
            for idx, name in enumerate(names)
            if idx in allowed_indices
        }
        verify_info = dict(info)
        verify_info["privileged_agents"] = privileged_info
        sampled_records: List[Dict[str, Any]] = []
        best: Optional[tuple] = None
        requested_counts = self._recovery_probability_sample_counts(
            [float(candidate["probability"]) for candidate in candidates],
            total_sample_count,
        )

        for candidate_order, candidate in enumerate(candidates):
            action = str(candidate.get("action") or "").strip()
            idx = name_to_index.get(action)
            if idx is None or action not in action_specs:
                sampled_records.append(
                    {
                        "action": action,
                        "probability": candidate.get("probability"),
                        "error": "action_unavailable",
                    }
                )
                continue
            probability = float(candidate["probability"])
            requested_count = requested_counts[candidate_order]
            spec = action_specs[action]
            static_range = spec.get("lateral_distance_m")
            preverified_range = (
                None
                if lateral_distance_ranges is None
                else lateral_distance_ranges.get(idx)
            )
            static_bounds = (
                None
                if static_range is None
                else (float(static_range[0]), float(static_range[1]))
            )
            if preverified_range is not None and static_bounds is not None:
                sample_range = (
                    max(float(preverified_range[0]), static_bounds[0]),
                    min(float(preverified_range[1]), static_bounds[1]),
                )
            elif preverified_range is not None:
                sample_range = tuple(map(float, preverified_range))
            else:
                sample_range = static_bounds
            if sample_range is None:
                evaluated_count = 1
                sampled_laterals: List[Optional[float]] = [None]
            else:
                low, high = map(float, sample_range)
                if high < low:
                    sampled_records.append(
                        {
                            "action": action,
                            "probability": probability,
                            "requested_sample_count": requested_count,
                            "evaluated_sample_count": 0,
                            "error": "no_reasonable_parameter_overlap",
                        }
                    )
                    continue
                evaluated_count = requested_count if high > low else 1
                sampled_laterals = [
                    low if high <= low else float(rng.uniform(low, high))
                    for _ in range(evaluated_count)
                ]

            candidate_record: Dict[str, Any] = {
                "action": action,
                "probability": probability,
                "requested_sample_count": requested_count,
                "evaluated_sample_count": evaluated_count,
                "samples": [],
            }
            sampled_records.append(candidate_record)
            for sample_order, sampled_lateral_m in enumerate(sampled_laterals):
                step = {
                    "action": action,
                    "target_speed_mps": candidate.get("target_speed_mps"),
                    "lateral_distance_m": sampled_lateral_m,
                }
                generated = materialize_recovery_sequence(
                    steps=[step],
                    info=info,
                    privileged_info=privileged_info,
                    output_num_poses=output_num_poses,
                    lane_bypass_min_gap_m=getattr(
                        runtime_recovery_cfg, "lane_bypass_min_gap_m", 0.0
                    ),
                    lane_bypass_clearance_buffer_m=getattr(
                        runtime_recovery_cfg,
                        "lane_bypass_clearance_buffer_m",
                        1.0,
                    ),
                    lane_bypass_required_time_sec=getattr(
                        runtime_recovery_cfg,
                        "lane_bypass_required_time_sec",
                        3.5,
                    ),
                    lane_bypass_min_time_gap_sec=getattr(
                        runtime_recovery_cfg,
                        "lane_bypass_min_time_gap_sec",
                        4.5,
                    ),
                    obstacle_clearance_m=getattr(
                        runtime_recovery_cfg, "obstacle_clearance_m", 8.0
                    ),
                )
                sample_record: Dict[str, Any] = {
                    "sample_index": sample_order,
                    "sampled_lateral_m": sampled_lateral_m,
                }
                candidate_record["samples"].append(sample_record)
                if generated is None:
                    sample_record.update(
                        {"accepted": False, "failure_flags": ["generation_failed"]}
                    )
                    continue
                trajectory, selected_states = generated
                score = float(self.verifier.score(trajectory, verify_info))
                last = getattr(self.verifier, "last_result", None)
                verifier_passed = self._verifier_accepts(score)
                footprint_failed = (
                    verifier_passed
                    and self._verified_trajectory_offroad(trajectory, info)
                )
                full_dac_failed = (
                    verifier_passed
                    and not footprint_failed
                    and self.recovery_full_plan_dac
                    and self._primitive_tier_gate_fails(trajectory, info)
                )
                accepted = bool(
                    verifier_passed and not footprint_failed and not full_dac_failed
                )
                failure_flags: List[str] = []
                if not verifier_passed:
                    failure_flags.append("below_verifier_threshold")
                if footprint_failed:
                    failure_flags.append("footprint_offroad")
                if full_dac_failed:
                    failure_flags.append("full_plan_dac_failed")
                sample_record.update(
                    {
                        "verifier_score": score,
                        "accepted": accepted,
                        "failure_flags": failure_flags,
                        "gates": dict(getattr(last, "gates", {}) or {}),
                        "soft": dict(getattr(last, "soft", {}) or {}),
                    }
                )
                if not accepted:
                    continue
                rank = (score, probability, -candidate_order, -sample_order)
                if best is None or rank > best[0]:
                    best = (
                        rank,
                        idx,
                        score,
                        np.asarray(trajectory, dtype=np.float32),
                        np.asarray(selected_states, dtype=np.float32),
                        sampled_lateral_m,
                        candidate,
                        requested_count,
                    )

        if best is None:
            self._record_failure_memory(
                info,
                event_type="pdms_verification",
                problem=getattr(self, "_pending_recovery_reason", None),
                outcome="all_probability_samples_rejected",
                details={
                    "probability_sample_multiplier": total_sample_count,
                    "total_requested_sample_count": sum(requested_counts),
                    "sampled_candidates": sampled_records,
                },
            )
            return _ForceBrakeFallback("recovery_plan_samples_exhausted")

        (
            _rank,
            idx,
            score,
            trajectory,
            selected_states,
            lateral_m,
            candidate,
            requested_count,
        ) = best
        decision_side = (
            "right" if str(lane_bypass_side).lower() == "right" else "left"
        )
        metadata = {
            "front_blockage": front_blockage,
            "escape_direction": escape_direction,
            "needs_lane_bypass_left": (
                needs_lane_bypass_left if decision_side == "left" else None
            ),
            "needs_lane_bypass_right": (
                needs_lane_bypass_left if decision_side == "right" else None
            ),
            "probability": float(candidate["probability"]),
            "probability_sample_multiplier": total_sample_count,
            "total_requested_sample_count": sum(requested_counts),
            "requested_sample_count": requested_count,
            "sampled_candidates": sampled_records,
        }
        self._record_failure_memory(
            info,
            event_type="primitive_selected",
            problem=getattr(self, "_pending_recovery_reason", None),
            method=str(candidate["action"]),
            outcome="highest_scoring_probability_sample",
            details={
                **metadata,
                "sampled_lateral_m": lateral_m,
                "verifier_score": score,
                "reasoning": candidate.get("reasoning"),
            },
        )
        self.log.warning(
            "commit-hold probability sampling selected '%s' (VLM p=%.3f, "
            "verifier %.3f) from %d evaluated trajectory variants",
            candidate["action"],
            float(candidate["probability"]),
            score,
            sum(
                int(record.get("evaluated_sample_count", 0))
                for record in sampled_records
            ),
        )
        return (
            idx,
            score,
            trajectory,
            selected_states,
            lateral_m,
            [],
            metadata,
        )

    @staticmethod
    def _filter_right_lane_bypass_candidates(
        *,
        candidate_indices: Sequence[int],
        names: Sequence[str],
        first_action_evaluations: Optional[Dict[str, Dict[str, Any]]],
        road_context: Optional[Dict[str, Any]],
    ) -> List[int]:
        """Filter lane bypasses using HD-map policy and the legacy right gate.

        The opposing-left/same-flow-right route-lane topology suppresses every
        leftward primitive before the VLM call.  Elsewhere,
        ``lane_bypass_right`` is an aggressive specialization of moving to
        the right.  It must not become a loophole when the ordinary
        ``lane_right`` proposal failed its PDMS/road checks.  It is also invalid
        while the ego is already off its driving corridor: in that state a
        rightward lane change may be a return-to-road action, whereas another
        rightward pass-and-return bypass would move farther away from the route.
        """
        candidates = [int(idx) for idx in candidate_indices]
        if road_context is not None:
            candidates = [
                idx
                for idx in candidates
                if not (
                    names[idx].startswith("lane_bypass_")
                    and adjacent_driving_lane_absent(
                        road_context,
                        names[idx][len("lane_bypass_"):],
                    )
                )
            ]
        if road_context is not None and suppress_leftward_options(road_context):
            return [
                idx
                for idx in candidates
                if not is_leftward_primitive(names[idx])
            ]
        evaluations = first_action_evaluations or {}
        lane_right_valid = bool(
            (evaluations.get("lane_right") or {}).get("eligible", False)
        )
        ego_on_corridor = not (
            road_context is not None
            and road_context.get("ego_on_corridor") is False
        )
        allow_right_bypass = lane_right_valid and ego_on_corridor
        return [
            idx
            for idx in candidates
            if not (
                names[idx] == "lane_bypass_right"
                and not allow_right_bypass
            )
        ]

    def _nominal_recovery_fallback(
        self,
        *,
        nominal_eligible: Sequence[tuple[int, float]],
        names: Sequence[str],
        proposals: np.ndarray,
        states: Sequence[np.ndarray],
        excluded_actions: Sequence[str] = (),
    ) -> Optional[tuple]:
        """Best verifier-passed regular recovery choice.

        The same ranking is used after ordinary VLM selection failure and when
        an exclusive lane-bypass latch is on cooldown.  Keeping it in one place
        prevents the cooldown path from silently degrading a set of safe
        regular actions into repeated fixed-duration brakes.
        """
        excluded = {str(action) for action in excluded_actions}
        candidates = [
            (int(idx), float(score))
            for idx, score in nominal_eligible
            if 0 <= int(idx) < len(names)
            and str(names[int(idx)]) not in excluded
        ]
        if not candidates:
            return None
        tie_break_rank = getattr(self, "_recovery_tie_break_rank", None)
        rank_scores = (
            getattr(self, "_trial_rank_scores", None)
            if getattr(self, "recovery_rank_without_ep", False)
            else None
        )
        unranked = len(tie_break_rank) if tie_break_rank else 0

        def fallback_key(item: tuple[int, float]) -> tuple:
            idx, verifier_score = item
            base = (
                verifier_score
                if not rank_scores
                else rank_scores.get(idx, verifier_score)
            )
            if tie_break_rank:
                return (base, -tie_break_rank.get(str(names[idx]), unranked))
            return (base,)

        idx, score = max(candidates, key=fallback_key)
        return (
            idx,
            score,
            np.asarray(proposals[idx], dtype=np.float32),
            np.asarray(states[idx], dtype=np.float32),
            None,
        )

    def _temporal_motion_recovery_fallback(
        self,
        *,
        temporal_results: Dict[str, TemporalSearchResult],
        names: Sequence[str],
        excluded_actions: Sequence[str] = (),
        front_blockage: Optional[object] = None,
    ) -> Optional[tuple]:
        """Best verified time-parameterized motion outside lane bypass.

        A primitive's fixed/default rollout can fail while a delayed or longer
        temporal leaf is safe.  Cooldown after an abandoned exclusive bypass
        must preserve that verified result instead of collapsing it to brake.
        """
        excluded = {str(action) for action in excluded_actions}
        tie_break_rank = getattr(self, "_recovery_tie_break_rank", None)
        unranked = len(tie_break_rank) if tie_break_rank else 0
        candidates: List[tuple[int, TemporalSearchResult]] = []
        for idx, raw_name in enumerate(names):
            action = str(raw_name)
            if (
                action in excluded
                or action in {"brake", "wait"}
                or action.startswith("lane_bypass_")
            ):
                continue
            result = temporal_results.get(action)
            leaf = result.best_leaf if result is not None else None
            if (
                result is None
                or leaf is None
                or leaf.trajectory is None
                or leaf.states is None
            ):
                continue
            candidates.append((idx, result))
        if not candidates:
            return None

        def fallback_key(
            item: tuple[int, TemporalSearchResult],
        ) -> tuple:
            idx, result = item
            action = str(names[idx])
            return (
                float(result.best_score),
                float(result.robust_score),
                float(result.feasible_ratio),
                -(
                    tie_break_rank.get(action, unranked)
                    if tie_break_rank
                    else idx
                ),
            )

        idx, result = max(candidates, key=fallback_key)
        leaf = result.best_leaf
        assert leaf is not None and leaf.trajectory is not None and leaf.states is not None
        metadata = {
            "front_blockage": front_blockage,
            "temporal_tree_search": True,
            "temporal_cooldown_fallback": True,
            "selected_profile": dict(leaf.profile),
            "temporal_scorecard": result.scorecard(),
            "execute_seconds": leaf.profile.get("execute_seconds"),
        }
        return (
            idx,
            float(leaf.score),
            np.asarray(leaf.trajectory, dtype=np.float32),
            np.asarray(leaf.states, dtype=np.float32),
            leaf.profile.get("lateral_distance_m"),
            [],
            metadata,
        )

    def _vlm_pick_recovery_primitive(
        self,
        *,
        obs: Dict[str, Any],
        info: Dict[str, Any],
        candidate_indices: List[int],
        nominal_eligible: List[tuple[int, float]],
        names: List[str],
        proposals: np.ndarray,
        states: List[np.ndarray],
        privileged_info: Optional[List[Dict[str, Any]]],
        output_num_poses: int,
        lateral_distance_ranges: Optional[Dict[int, tuple[float, float]]] = None,
        sequence_first_action_names: Optional[Sequence[str]] = None,
        sequence_first_action_evaluations: Optional[
            Dict[str, Dict[str, Any]]
        ] = None,
    ) -> Optional[tuple] | _ForceBrakeFallback:
        """Run VLM selection, materialize the choice, and verify it.

        B5 preverification remains one-shot when ``select_max_attempts`` is 1.
        C3/C4 deliberately configure more attempts: a rejected parameterized
        primitive or sequence is then recorded in failure memory and sent back
        to the VLM for bounded self-refinement before falling back.
        """
        plan_mode = bool(
            getattr(self.recovery_explainer.cfg, "select_recovery_plan", False)
        )
        bypass_first = bool(
            plan_mode
            and getattr(
                self.recovery_explainer.cfg,
                "recovery_plan_bypass_first",
                False,
            )
        )
        bypass_action = str(
            getattr(
                self.recovery_explainer.cfg,
                "recovery_plan_bypass_primitive",
                "lane_bypass_left",
            )
            or "lane_bypass_left"
        ).strip()
        bypass_direction = (
            "right" if bypass_action.endswith("_right") else "left"
        )
        bypass_decision_field = (
            f"needs_lane_bypass_{bypass_direction}"
        )
        runtime_recovery_cfg = getattr(self, "recovery_cfg", None)
        sequence_mode = bool(
            getattr(self.recovery_explainer.cfg, "select_sequence", False)
            and not plan_mode
        )
        structured_mode = bool(plan_mode or sequence_mode)
        force_brake_on_exhaustion = bool(getattr(
            self.recovery_explainer.cfg, "force_brake_on_selection_exhausted", False
        ))
        exclude_rejected = bool(getattr(
            self.recovery_explainer.cfg, "exclude_rejected_primitives_on_retry", False
        )) and not structured_mode
        if not candidate_indices:
            return (
                _ForceBrakeFallback("no_recovery_candidates")
                if structured_mode or force_brake_on_exhaustion
                else None
            )
        configured_attempts = max(
            1,
            int(getattr(self.recovery_explainer.cfg, "select_max_attempts", 3)),
        )
        preverified_selection = bool(
            structured_mode
            or (
                getattr(self.recovery_explainer.cfg, "select_parameters", False)
                and getattr(
                    self.recovery_explainer.cfg,
                    "preverify_parameter_ranges",
                    False,
                )
            )
        )
        one_shot_parameter_selection = bool(
            preverified_selection and configured_attempts == 1
        )
        max_attempts = configured_attempts
        previous_attempts: List[str] = []
        attempted_choices = set()
        verify_info = dict(info)
        verify_info["privileged_agents"] = privileged_info

        # Road-context arm (default off): compute once per selection and flag
        # each candidate's endpoint tier for all attempts.
        road_context = self._recovery_road_context(info)
        directional_context = dict(road_context or {})
        directional_context["adjacent_lane_flows"] = info.get(
            "adjacent_lane_flows"
        )
        candidate_indices = self._filter_right_lane_bypass_candidates(
            candidate_indices=candidate_indices,
            names=names,
            first_action_evaluations=sequence_first_action_evaluations,
            road_context=directional_context,
        )
        roadside_mask = {"masked": False, "reason": "disabled"}
        if bool(getattr(
            self.recovery_explainer.cfg,
            "sequence_menu_mask_roadside_lane_bypass_left",
            True,
        )):
            roadside_mask = roadside_lane_bypass_left_assessment(
                info=info,
                privileged_info=self._fresh_recovery_agents(privileged_info),
                max_distance_m=getattr(
                    self.recovery_explainer.cfg,
                    "sequence_menu_roadside_distance_m", 10.0,
                ),
                offset_ratio=getattr(
                    self.recovery_explainer.cfg,
                    "sequence_menu_roadside_offset_ratio", 0.25,
                ),
            )
        self._sequence_menu_roadside_bypass = roadside_mask
        roadside_bypass_masked = bool(roadside_mask["masked"])
        if roadside_bypass_masked:
            candidate_indices = [
                idx for idx in candidate_indices if names[idx] != "lane_bypass_left"
            ]
            if sequence_first_action_evaluations is not None:
                sequence_first_action_evaluations = {
                    name: value for name, value in sequence_first_action_evaluations.items()
                    if name != "lane_bypass_left"
                }
            self.log.info(
                "recovery menu masks lane_bypass_left for right-edge "
                "route obstacle(s): %s",
                roadside_mask,
            )
        if not candidate_indices:
            return _ForceBrakeFallback("no_directionally_valid_recovery_candidates")
        candidate_road_flags: Optional[List[Optional[Dict[str, Any]]]] = None
        if road_context is not None:
            candidate_road_flags = self._candidate_ends_on_driving_lane_batch(
                proposals[candidate_indices], info
            )
        # Optional VLM-menu restriction: the fallback ladder keeps the full
        # allowed set; only the model's action space narrows. An MRM-style
        # menu can dynamically add verified swerves/bypasses/lane changes
        # for a nearby static obstruction or sustained stationary dwell.
        menu_subset, menu_escalation_added = (
            self._sequence_vlm_menu_subset(info)
        )
        if roadside_bypass_masked:
            menu_escalation_added = tuple(
                name for name in menu_escalation_added if name != "lane_bypass_left"
            )
        if menu_escalation_added:
            front_obstacle = getattr(
                self, "_sequence_menu_front_obstacle", {}
            )
            if front_obstacle.get("triggered"):
                if front_obstacle.get("trigger_kind") == "stationary_vehicle":
                    self.log.warning(
                        "commit-hold VLM menu escalation for vehicle %s "
                        "stationary %.1fs at %.1fm ahead: exposing %s",
                        front_obstacle.get("agent_id", "unknown"),
                        float(front_obstacle.get("stationary_sec", 0.0)),
                        float(front_obstacle.get("distance_m", 0.0)),
                        ", ".join(menu_escalation_added),
                    )
                else:
                    self.log.warning(
                        "commit-hold VLM menu escalation for %s obstacle %s "
                        "%.1fm ahead: exposing %s",
                        front_obstacle.get("object_type", "unknown"),
                        front_obstacle.get("agent_id", "unknown"),
                        float(front_obstacle.get("distance_m", 0.0)),
                        ", ".join(menu_escalation_added),
                    )
            else:
                self.log.warning(
                    "commit-hold VLM menu escalation after %.1fs within the "
                    "stationary position radius: exposing %s",
                    float(
                        getattr(self, "_sequence_menu_stationary_sec", 0.0)
                    ),
                    ", ".join(menu_escalation_added),
                )
        sequence_action_specs = (
            get_recovery_sequence_action_specs(
                [
                    name
                    for name in (names[idx] for idx in candidate_indices)
                    if not menu_subset or name in menu_subset
                    if not (
                        plan_mode
                        and name in RECOVERY_PLAN_EXCLUDED_PRIMITIVES
                    )
                ],
                back_steer_max_lateral_m=getattr(
                    self.recovery_explainer.cfg,
                    "sequence_back_steer_max_lateral_m",
                    None,
                ),
            )
            if structured_mode
            else {}
        )
        # Keep the bypass in the *necessity* question while a live traffic gate
        # or an unknown map lookup hides its trajectory, but only after the
        # current sequence menu exposes it.  In particular, an MRM escalation
        # menu must not be bypassed by ``recovery_plan_bypass_first``.  A
        # definitive HD-map ``none`` is also terminal: no such maneuver exists,
        # so do not expose its action spec to the VLM at all.
        bypass_menu_allowed = bool(
            (not menu_subset or bypass_action in menu_subset)
            and not (roadside_bypass_masked and bypass_action == "lane_bypass_left")
        )
        bypass_directionally_allowed = not (
            (
                is_leftward_primitive(bypass_action)
                and suppress_leftward_options(info)
            )
            or adjacent_driving_lane_absent(info, bypass_direction)
        )
        bypass_first = bool(
            bypass_first
            and bypass_menu_allowed
            and bypass_directionally_allowed
        )
        if (
            bypass_first
            and bypass_directionally_allowed
            and bypass_action not in sequence_action_specs
        ):
            sequence_action_specs.update(
                get_recovery_sequence_action_specs([bypass_action])
            )
        temporal_tree_enabled = bool(
            plan_mode
            and getattr(
                self.recovery_explainer.cfg,
                "temporal_tree_search",
                False,
            )
        )
        temporal_results: Dict[str, TemporalSearchResult] = {}
        if temporal_tree_enabled:
            temporal_results = self._search_recovery_temporal_trees(
                action_specs=sequence_action_specs,
                candidate_indices=candidate_indices,
                names=names,
                info=info,
                privileged_info=privileged_info,
                output_num_poses=output_num_poses,
                lateral_distance_ranges=lateral_distance_ranges,
            )
            merged_evaluations = {
                str(name): dict(value)
                for name, value in (
                    sequence_first_action_evaluations or {}
                ).items()
            }
            for action, search_result in temporal_results.items():
                evaluation = merged_evaluations.setdefault(action, {})
                evaluation["temporal_search"] = search_result.scorecard()
                evaluation["eligible"] = bool(search_result.eligible)
                evaluation["pdms_score"] = float(search_result.best_score)
                evaluation["pdms_threshold"] = float(self.verifier_threshold)
                if search_result.best_leaf is not None:
                    evaluation["gates"] = dict(
                        search_result.best_leaf.gates or {}
                    )
                    evaluation["soft"] = dict(
                        search_result.best_leaf.soft or {}
                    )
                    evaluation["rejection_reasons"] = []
                else:
                    evaluation["rejection_reasons"] = [
                        "temporal_tree_no_safe_leaf"
                    ]
            sequence_first_action_evaluations = merged_evaluations
            first_action_names = [
                action
                for action in sequence_action_specs
                if (
                    action in temporal_results
                    and temporal_results[action].eligible
                )
            ]
            # Bypass-first asks a semantic necessity question even while its
            # traffic gate is closed.  Keep that one action visible so it can
            # latch; execution still requires a later safe temporal leaf.
            if bypass_first and bypass_action not in first_action_names:
                first_action_names.append(bypass_action)
            if not first_action_names:
                return _ForceBrakeFallback("no_temporal_tree_safe_leaf")

            # A nominal rollout may fail while temporal search finds a safe
            # delayed/longer lateral leaf. Preserve that verified profile and
            # apply the persistent-blockage rule before consulting the VLM.
            temporal_eligible = [
                (idx, float(search_result.best_score))
                for idx, action in enumerate(names)
                for search_result in [temporal_results.get(str(action))]
                if (
                    search_result is not None
                    and search_result.eligible
                    and search_result.best_leaf is not None
                )
            ]
            temporal_rule_choice = self._tool_rule_blocked_escape_choice(
                info=info,
                nominal_eligible=temporal_eligible,
                names=names,
            )
            if temporal_rule_choice is not None:
                idx, score, rule_debug = temporal_rule_choice
                action = str(names[idx])
                search_result = temporal_results[action]
                leaf = search_result.best_leaf
                assert (
                    leaf is not None
                    and leaf.trajectory is not None
                    and leaf.states is not None
                )
                self._record_failure_memory(
                    info,
                    event_type="primitive_selected",
                    problem=getattr(self, "_pending_recovery_reason", None),
                    method=action,
                    outcome="persistent_blockage_tool_rule",
                    details={
                        **rule_debug,
                        "verifier_score": score,
                        "selected_profile": dict(leaf.profile),
                    },
                )
                return (
                    idx,
                    score,
                    np.asarray(leaf.trajectory, dtype=np.float32),
                    np.asarray(leaf.states, dtype=np.float32),
                    leaf.profile.get("lateral_distance_m"),
                    [],
                    {
                        "front_blockage": (
                            f"tool rule: persistent "
                            f"{rule_debug.get('obstacle_type') or 'front'} obstacle"
                        ),
                        "temporal_tree_search": True,
                        "selected_profile": dict(leaf.profile),
                        "temporal_scorecard": search_result.scorecard(),
                        "execute_seconds": leaf.profile.get("execute_seconds"),
                        "tool_rule": rule_debug,
                    },
                )

        elif structured_mode and sequence_first_action_names is not None:
            first_action_names = [
                str(name)
                for name in sequence_first_action_names
                if str(name) in sequence_action_specs
            ]
            if (
                bypass_first
                and bypass_directionally_allowed
                and bypass_action not in first_action_names
            ):
                first_action_names.append(bypass_action)
            if not first_action_names:
                self.log.warning(
                    "commit-hold sequence selection skipped: no primitive passed "
                    "the first-action PDMS/road-safety gate"
                )
                return _ForceBrakeFallback("no_pdms_passed_first_action")
        else:
            first_action_names = None

        rejected_primitives = set()
        for attempt_index in range(1, max_attempts + 1):
            if exclude_rejected and rejected_primitives:
                keep = [pos for pos, idx in enumerate(candidate_indices)
                        if names[idx] not in rejected_primitives]
                candidate_indices = [candidate_indices[pos] for pos in keep]
                if candidate_road_flags is not None:
                    candidate_road_flags = [candidate_road_flags[pos] for pos in keep]
                if not candidate_indices:
                    break
            try:
                common_context = {
                    "frame_index": getattr(self.selector, "frame_index", 0),
                    "camera_images": obs.get("rgb", {}),
                    "info": info,
                    "failure_reason": self._pending_recovery_reason,
                    "failure_memory": self._failure_memory_context(
                        available_actions=(
                            list(sequence_action_specs) if structured_mode else None
                        )
                    ),
                    "verifier_gates": dict(
                        (getattr(self, "_pending_recovery_verifier_context", None) or {}).get(
                            "gates", {}
                        )
                    ),
                    "verifier_soft": dict(
                        (getattr(self, "_pending_recovery_verifier_context", None) or {}).get(
                            "soft", {}
                        )
                    ),
                    "verifier_score": (
                        getattr(self, "_pending_recovery_verifier_context", None) or {}
                    ).get("score"),
                    "verifier_threshold": (
                        getattr(self, "_pending_recovery_verifier_context", None) or {}
                    ).get("threshold", self.verifier_threshold),
                    "road_context": road_context,
                }
                if structured_mode:
                    result = self.recovery_explainer.select_sequence(
                        **common_context,
                        action_specs=sequence_action_specs,
                        first_action_names=first_action_names,
                        first_action_evaluations=sequence_first_action_evaluations,
                        previous_attempts=previous_attempts,
                        sequence_context=self._recovery_sequence_context(),
                        memory_guidance=getattr(
                            getattr(self, "failure_memory", None),
                            "guidance_mode",
                            "avoid",
                        ),
                    )
                else:
                    result = self.recovery_explainer.select_primitive(
                        **common_context,
                        candidate_plans=[
                            np.asarray(proposals[idx], dtype=np.float32)
                            for idx in candidate_indices
                        ],
                        candidate_names=[names[idx] for idx in candidate_indices],
                        previous_attempts=previous_attempts,
                        candidate_lateral_distance_ranges=[
                            None
                            if lateral_distance_ranges is None
                            else lateral_distance_ranges.get(idx)
                            for idx in candidate_indices
                        ],
                        candidate_road_flags=candidate_road_flags,
                    )
            except Exception as exc:
                self.log.exception("recovery VLM selection failed")
                self._record_failure_memory(
                    info,
                    event_type="primitive_selection_error",
                    problem=getattr(self, "_pending_recovery_reason", None),
                    outcome="vlm_call_failed",
                    details={"attempt": attempt_index, "error": str(exc)},
                )
                if structured_mode:
                    return _ForceBrakeFallback("sequence_vlm_call_failed")
                break

            if plan_mode:
                plan_candidates = list(result.get("candidates") or [])
                if result.get("error") is not None:
                    selection_error = str(
                        result.get("error")
                    )
                    self._record_failure_memory(
                        info,
                        event_type="primitive_selection_error",
                        problem=getattr(self, "_pending_recovery_reason", None),
                        outcome="invalid_vlm_output",
                        details={"attempt": attempt_index, "error": selection_error},
                    )
                    return _ForceBrakeFallback(selection_error)
                needs_lane_bypass_left = result.get(
                    bypass_decision_field
                )
                if bypass_first:
                    candidate_actions = [
                        str(candidate.get("action") or "").strip()
                        for candidate in plan_candidates
                    ]
                    exclusive_bypass = bool(
                        needs_lane_bypass_left is True
                        and candidate_actions == [bypass_action]
                        and abs(
                            float(plan_candidates[0].get("probability", 0.0))
                            - 1.0
                        )
                        <= 1e-6
                        and str(result.get("escape_direction") or "").lower()
                        == bypass_direction
                    )
                    regular_fallback = bool(
                        needs_lane_bypass_left is False
                        and bypass_action not in candidate_actions
                    )
                    if exclusive_bypass:
                        direction = (
                            "left"
                            if bypass_action.endswith("_left")
                            else "right"
                        )
                        assessment = lane_bypass_gap_assessment(
                            info=info,
                            privileged_info=privileged_info,
                            direction=direction,
                            min_gap_m=getattr(
                                runtime_recovery_cfg,
                                "lane_bypass_min_gap_m",
                                0.0,
                            ),
                            clearance_buffer_m=getattr(
                                runtime_recovery_cfg,
                                "lane_bypass_clearance_buffer_m",
                                1.0,
                            ),
                            required_time_sec=getattr(
                                runtime_recovery_cfg,
                                "lane_bypass_required_time_sec",
                                3.5,
                            ),
                            min_time_gap_sec=getattr(
                                runtime_recovery_cfg,
                                "lane_bypass_min_time_gap_sec",
                                4.5,
                            ),
                            obstacle_clearance_m=getattr(
                                runtime_recovery_cfg,
                                "obstacle_clearance_m",
                                8.0,
                            ),
                        )
                        if assessment.get("reason") == "target_lane_static_blocked":
                            wait_idx = next(
                                (
                                    idx
                                    for idx in candidate_indices
                                    if names[int(idx)] == "wait"
                                ),
                                None,
                            )
                            if (
                                wait_idx is None
                                or "wait" not in sequence_action_specs
                            ):
                                return _ForceBrakeFallback(
                                    "lane_bypass_target_lane_static_blocked"
                                )
                            self.log.warning(
                                "recovery VLM requested '%s', but exact static "
                                "bbox %s occupies its fixed adjacent lane; "
                                "substituting regular 'wait' without latching",
                                bypass_action,
                                assessment.get("conflicting_agent_id"),
                            )
                            self._target_lane_static_wait = {
                                "direction": direction,
                                "started_ts": float(
                                    info.get("timestamp", 0.0) or 0.0
                                ),
                                "wait_ticks": 0,
                                "conflicting_agent_id": assessment.get(
                                    "conflicting_agent_id"
                                ),
                            }
                            plan_candidates = [
                                {"action": "wait", "probability": 1.0}
                            ]
                            candidate_actions = ["wait"]
                            needs_lane_bypass_left = False
                            result["escape_direction"] = "stop"
                            exclusive_bypass = False
                            regular_fallback = True
                    if not (exclusive_bypass or regular_fallback):
                        self.log.warning(
                            "recovery VLM violated mutually exclusive bypass "
                            "policy: decision=%r candidates=%s",
                            needs_lane_bypass_left,
                            candidate_actions,
                        )
                        return _ForceBrakeFallback(
                            "invalid_bypass_first_recovery_plan"
                        )
                    if exclusive_bypass:
                        if not self._latch_lane_bypass_intent(
                            action=bypass_action,
                            info=info,
                            output_num_poses=output_num_poses,
                            front_blockage=result.get("front_blockage"),
                        ):
                            front = getattr(self, "_sequence_menu_front_obstacle", {})
                            if front.get("enabled") and not front.get("triggered"):
                                return _ForceBrakeFallback("no_current_route_blocker")
                            # The same exclusive maneuver already spent its
                            # full wait window without becoming executable.
                            # Preserve a safe time-parameterized motion even
                            # when the primitive's default rollout failed.
                            temporal_fallback = (
                                self._temporal_motion_recovery_fallback(
                                    temporal_results=temporal_results,
                                    names=names,
                                    excluded_actions={bypass_action},
                                    front_blockage=result.get("front_blockage"),
                                )
                            )
                            if temporal_fallback is not None:
                                fallback_name = names[temporal_fallback[0]]
                                self._record_failure_memory(
                                    info,
                                    event_type="primitive_selection_fallback",
                                    problem=getattr(
                                        self, "_pending_recovery_reason", None
                                    ),
                                    method=fallback_name,
                                    outcome=(
                                        "lane_bypass_cooldown_using_temporal_pass"
                                    ),
                                    details={
                                        "blocked_bypass": bypass_action,
                                        "verifier_score": temporal_fallback[1],
                                        "selected_profile": dict(
                                            temporal_fallback[6].get(
                                                "selected_profile", {}
                                            )
                                        ),
                                    },
                                )
                                self.log.warning(
                                    "lane bypass '%s' is on cooldown; using "
                                    "verifier-passed temporal primitive '%s' "
                                    "instead of another forced brake",
                                    bypass_action,
                                    fallback_name,
                                )
                                return temporal_fallback

                            # If no temporal motion exists, keep the original
                            # already-verified regular fallback path.
                            fallback = self._nominal_recovery_fallback(
                                nominal_eligible=nominal_eligible,
                                names=names,
                                proposals=proposals,
                                states=states,
                                excluded_actions={bypass_action},
                            )
                            if fallback is not None:
                                self._record_failure_memory(
                                    info,
                                    event_type="primitive_selection_fallback",
                                    problem=getattr(
                                        self, "_pending_recovery_reason", None
                                    ),
                                    method=names[fallback[0]],
                                    outcome="lane_bypass_cooldown_using_nominal_pass",
                                    details={
                                        "blocked_bypass": bypass_action,
                                        "verifier_score": fallback[1],
                                    },
                                )
                                self.log.warning(
                                    "lane bypass '%s' is on cooldown; using "
                                    "verifier-passed regular primitive '%s' "
                                    "instead of another forced brake",
                                    bypass_action,
                                    names[fallback[0]],
                                )
                                return fallback
                            return _ForceBrakeFallback(
                                "lane_bypass_latch_on_cooldown_no_nominal_pass"
                            )
                        self._retain_lane_bypass_obstacle_clearance(assessment)
                        self.log.warning(
                            "recovery VLM requested direct complete primitive "
                            "'%s'; bypassing all regular recovery candidates",
                            bypass_action,
                        )
                if not plan_candidates:
                    self.log.warning(
                        "recovery VLM declined every supplied primitive; selecting brake"
                    )
                    return _ForceBrakeFallback("vlm_declined_all_candidates")
                if temporal_tree_enabled:
                    return self._select_vlm_temporal_tree_result(
                        plan_candidates=plan_candidates,
                        temporal_results=temporal_results,
                        names=names,
                        info=info,
                        front_blockage=result.get("front_blockage"),
                        escape_direction=result.get("escape_direction"),
                        needs_lane_bypass=(
                            bool(needs_lane_bypass_left)
                            if bypass_first
                            else None
                        ),
                        lane_bypass_side=bypass_direction,
                    )
                return self._sample_and_score_recovery_plan_candidates(
                    candidates=plan_candidates,
                    action_specs=sequence_action_specs,
                    candidate_indices=candidate_indices,
                    names=names,
                    info=info,
                    privileged_info=privileged_info,
                    output_num_poses=output_num_poses,
                    lateral_distance_ranges=lateral_distance_ranges,
                    front_blockage=result.get("front_blockage"),
                    escape_direction=result.get("escape_direction"),
                    needs_lane_bypass_left=(
                        bool(needs_lane_bypass_left)
                        if bypass_first
                        else None
                    ),
                    lane_bypass_side=bypass_direction,
                )

            sequence_steps = list(result.get("steps") or []) if structured_mode else []
            selected_waypoint = (
                dict(result.get("waypoint"))
                if plan_mode and isinstance(result.get("waypoint"), dict)
                else None
            )
            selected = result.get("selected_index")
            selection_invalid = (
                not sequence_steps and selected_waypoint is None
                if structured_mode
                else selected is None or not (0 <= int(selected) < len(candidate_indices))
            )
            if selection_invalid:
                selection_error = str(
                    result.get("error")
                    or ("invalid_recovery_plan" if structured_mode else "invalid_selected_index")
                )
                self.log.warning(
                    "commit-hold VLM select unavailable on attempt %d/%d (%s)",
                    attempt_index,
                    max_attempts,
                    selection_error,
                )
                self._record_failure_memory(
                    info,
                    event_type="primitive_selection_error",
                    problem=getattr(self, "_pending_recovery_reason", None),
                    outcome="invalid_vlm_output",
                    details={
                        "attempt": attempt_index,
                        "error": selection_error,
                    },
                )
                if structured_mode:
                    return _ForceBrakeFallback(selection_error)
                if (
                    not one_shot_parameter_selection
                    and selection_error.startswith("recovery_selection_")
                ):
                    previous_attempts.append(
                        f"attempt {attempt_index}: output rejected before trajectory "
                        f"generation ({selection_error}); return a valid index, maneuver, "
                        "and in-range lateral_distance_m"
                    )
                    continue
                break
            if selected_waypoint is not None:
                idx = candidate_indices[0]
                lateral_distance_m = None
                method_name = "waypoint"
                choice_key = (
                    "waypoint",
                    round(float(selected_waypoint["forward_m"]), 3),
                    round(float(selected_waypoint["right_m"]), 3),
                )
            elif structured_mode:
                idx = candidate_indices[0]
                lateral_distance_m = None
                method_name = (
                    "sequence:" + ">".join(
                        str(step.get("action")) for step in sequence_steps
                    )
                    if sequence_mode
                    else str(sequence_steps[0].get("action"))
                )
                choice_key = tuple(
                    (
                        str(step.get("action")),
                        None
                        if step.get("target_speed_mps") is None
                        else round(float(step["target_speed_mps"]), 3),
                        None
                        if step.get("lateral_distance_m") is None
                        else round(float(step["lateral_distance_m"]), 3),
                    )
                    for step in sequence_steps
                )
            else:
                idx = candidate_indices[int(selected)]
                lateral_distance_m = result.get("lateral_distance_m")
                if (not getattr(self.recovery_explainer.cfg, "select_parameters", True)
                        and getattr(self.recovery_explainer.cfg, "ignore_parameters_when_disabled", False)):
                    lateral_distance_m = None
                method_name = names[idx]
                choice_key = (
                    names[idx],
                    None if lateral_distance_m is None else round(float(lateral_distance_m), 3),
                )
            if structured_mode and selected_waypoint is None:
                productive_first_actions = [
                    name
                    for name in (first_action_names or [])
                    if name not in {"brake", "wait"}
                ]
                stop_only = bool(sequence_steps) and all(
                    str(step.get("action")) in {"brake", "wait"}
                    for step in sequence_steps
                )
                pending_gates = dict(
                    (
                        getattr(self, "_pending_recovery_verifier_context", None)
                        or {}
                    ).get("gates", {})
                )
                immediate_collision = float(pending_gates.get("nc", 1.0)) <= 0.0
                if stop_only and productive_first_actions and not immediate_collision:
                    feedback = (
                        f"attempt {attempt_index}: {method_name} rejected as "
                        "over-conservative because PDMS-passed movement actions "
                        f"are available: {', '.join(productive_first_actions)}"
                    )
                    previous_attempts.append(feedback)
                    self._record_failure_memory(
                        info,
                        event_type="recovery_selection",
                        problem=getattr(self, "_pending_recovery_reason", None),
                        method=method_name,
                        outcome="rejected",
                        details={
                            "attempt": attempt_index,
                            "sequence": sequence_steps,
                            "failure_flags": ["overconservative_stop_only"],
                            "productive_first_actions": productive_first_actions,
                            "normal_plan_gates": pending_gates,
                        },
                    )
                    self.log.info("commit-hold VLM retry feedback: %s", feedback)
                    continue
            self._record_failure_memory(
                info,
                event_type="primitive_selected",
                problem=getattr(self, "_pending_recovery_reason", None),
                method=method_name,
                outcome="sequence_selected_by_vlm" if sequence_mode else "selected_by_vlm",
                details={
                    "attempt": attempt_index,
                    "lateral_distance_m": lateral_distance_m,
                    "sequence": sequence_steps,
                    "waypoint": selected_waypoint,
                    "front_blockage": result.get("front_blockage"),
                    "escape_direction": result.get("escape_direction"),
                    "executed_step": sequence_steps[0] if sequence_steps else None,
                    "reasoning": result.get("reasoning"),
                    "confidence": result.get("confidence"),
                },
            )
            if choice_key in attempted_choices:
                if not one_shot_parameter_selection:
                    previous_attempts.append(
                        f"attempt {attempt_index}: {names[idx]} lateral_distance_m="
                        f"{lateral_distance_m} repeated an already rejected choice"
                    )
                self._record_failure_memory(
                    info,
                    event_type="primitive_selection_repeated",
                    problem=getattr(self, "_pending_recovery_reason", None),
                    method=method_name,
                    outcome="rejected_without_generation",
                    details={
                        "attempt": attempt_index,
                        "lateral_distance_m": lateral_distance_m,
                        "sequence": sequence_steps,
                        "waypoint": selected_waypoint,
                    },
                )
                continue
            attempted_choices.add(choice_key)
            if exclude_rejected:
                # Successful verification returns immediately. Only failed
                # generation/verification can reach another query with this set.
                rejected_primitives.add(names[idx])

            waypoint_execute_seconds = None
            if selected_waypoint is not None:
                generated_waypoint = materialize_recovery_waypoint(
                    forward_m=float(selected_waypoint["forward_m"]),
                    right_m=float(selected_waypoint["right_m"]),
                    output_num_poses=output_num_poses,
                    forward_range_m=(
                        float(
                            getattr(
                                self.recovery_explainer.cfg,
                                "recovery_waypoint_forward_min_m",
                                1.0,
                            )
                        ),
                        float(
                            getattr(
                                self.recovery_explainer.cfg,
                                "recovery_waypoint_forward_max_m",
                                6.0,
                            )
                        ),
                    ),
                    right_range_m=(
                        float(
                            getattr(
                                self.recovery_explainer.cfg,
                                "recovery_waypoint_right_min_m",
                                -3.0,
                            )
                        ),
                        float(
                            getattr(
                                self.recovery_explainer.cfg,
                                "recovery_waypoint_right_max_m",
                                3.0,
                            )
                        ),
                    ),
                )
                if generated_waypoint is None:
                    generated = None
                else:
                    generated = generated_waypoint[:2]
                    waypoint_execute_seconds = float(generated_waypoint[2])
            elif structured_mode:
                # The VLM may return future intent, but only the first primitive
                # is materialized and safety-checked from the current scene.
                # The rest is reconsidered after Loop 1 runs again.
                generated = materialize_recovery_sequence(
                    steps=sequence_steps[:1],
                    info=info,
                    privileged_info=privileged_info,
                    output_num_poses=output_num_poses,
                    lane_bypass_min_gap_m=getattr(
                        runtime_recovery_cfg, "lane_bypass_min_gap_m", 0.0
                    ),
                    lane_bypass_clearance_buffer_m=getattr(
                        runtime_recovery_cfg,
                        "lane_bypass_clearance_buffer_m",
                        1.0,
                    ),
                    lane_bypass_required_time_sec=getattr(
                        runtime_recovery_cfg,
                        "lane_bypass_required_time_sec",
                        3.5,
                    ),
                    lane_bypass_min_time_gap_sec=getattr(
                        runtime_recovery_cfg,
                        "lane_bypass_min_time_gap_sec",
                        4.5,
                    ),
                    obstacle_clearance_m=getattr(
                        runtime_recovery_cfg, "obstacle_clearance_m", 8.0
                    ),
                )
            else:
                generated = materialize_recovery_primitive(
                    primitive_name=names[idx],
                    info=info,
                    privileged_info=privileged_info,
                    output_num_poses=output_num_poses,
                    lateral_distance_m=(
                        None if lateral_distance_m is None else float(lateral_distance_m)
                    ),
                    lane_bypass_min_gap_m=getattr(
                        runtime_recovery_cfg, "lane_bypass_min_gap_m", 0.0
                    ),
                    lane_bypass_clearance_buffer_m=getattr(
                        runtime_recovery_cfg,
                        "lane_bypass_clearance_buffer_m",
                        1.0,
                    ),
                    lane_bypass_required_time_sec=getattr(
                        runtime_recovery_cfg,
                        "lane_bypass_required_time_sec",
                        3.5,
                    ),
                    lane_bypass_min_time_gap_sec=getattr(
                        runtime_recovery_cfg,
                        "lane_bypass_min_time_gap_sec",
                        4.5,
                    ),
                    obstacle_clearance_m=getattr(
                        runtime_recovery_cfg, "obstacle_clearance_m", 8.0
                    ),
                )
            if generated is None:
                if not one_shot_parameter_selection:
                    previous_attempts.append(
                        f"attempt {attempt_index}: {method_name} could not generate "
                        "a valid trajectory"
                    )
                self._record_failure_memory(
                    info,
                    event_type="primitive_generation",
                    problem=getattr(self, "_pending_recovery_reason", None),
                    method=method_name,
                    outcome="generation_failed",
                    details={
                        "attempt": attempt_index,
                        "lateral_distance_m": lateral_distance_m,
                        "sequence": sequence_steps,
                    },
                )
                continue
            traj, selected_states = generated
            score = float(self.verifier.score(traj, verify_info))
            last = getattr(self.verifier, "last_result", None)
            passed = self._verifier_accepts(score)
            footprint_failed = passed and self._verified_trajectory_offroad(
                traj, info
            )
            full_dac_failed = (
                passed
                and not footprint_failed
                and self.recovery_full_plan_dac
                and self._primitive_tier_gate_fails(traj, info)
            )
            passed = passed and not footprint_failed and not full_dac_failed
            gates = dict(getattr(last, "gates", {}) or {}) if last is not None else {}
            soft = dict(getattr(last, "soft", {}) or {}) if last is not None else {}
            if passed:
                if structured_mode and sequence_steps:
                    self._sequence_action_manager().select(
                        sequence_steps,
                        timestamp=float(info.get("timestamp", 0.0) or 0.0),
                        reasoning=(
                            None
                            if result.get("reasoning") is None
                            else str(result.get("reasoning"))
                        ),
                        confidence=(
                            None
                            if result.get("confidence") is None
                            else float(result.get("confidence"))
                        ),
                    )
                self._record_failure_memory(
                    info,
                    event_type="pdms_verification",
                    problem=getattr(self, "_pending_recovery_reason", None),
                    method=method_name,
                    outcome="accepted",
                    details={
                        "attempt": attempt_index,
                        "lateral_distance_m": lateral_distance_m,
                        "sequence": sequence_steps,
                        "waypoint": selected_waypoint,
                        "verified_step": sequence_steps[0] if sequence_steps else None,
                        "verifier_score": score,
                        "verifier_threshold": self.verifier_threshold,
                        "gates": gates,
                        "soft": soft,
                    },
                )
                self.log.warning(
                    "commit-hold VLM selected '%s' on attempt %d/%d "
                    "(verifier %.3f): %s",
                    method_name,
                    attempt_index,
                    max_attempts,
                    score,
                    result.get("reasoning"),
                )
                chosen = (
                    idx,
                    score,
                    np.asarray(traj, dtype=np.float32),
                    np.asarray(selected_states, dtype=np.float32),
                    None if lateral_distance_m is None else float(lateral_distance_m),
                )
                if plan_mode:
                    return (
                        *chosen,
                        sequence_steps,
                        {
                            "plan_type": (
                                "waypoint" if selected_waypoint is not None else "primitive"
                            ),
                            "waypoint": selected_waypoint,
                            "front_blockage": result.get("front_blockage"),
                            "escape_direction": result.get("escape_direction"),
                            "execute_seconds": waypoint_execute_seconds,
                        },
                    )
                if sequence_mode:
                    return (*chosen, sequence_steps)
                return chosen

            failure_flags = []
            if not self._verifier_accepts(score):
                failure_flags.append(
                    f"verifier_score={score:.3f}<threshold={self.verifier_threshold:.3f}"
                )
            if footprint_failed:
                failure_flags.append("footprint_offroad")
            if full_dac_failed:
                failure_flags.append("full_plan_dac_failed")
            feedback = (
                f"attempt {attempt_index}: {method_name} rejected "
                f"({', '.join(failure_flags)}; gates={gates}; soft={soft})"
            )
            if not one_shot_parameter_selection:
                previous_attempts.append(feedback)
            self._record_failure_memory(
                info,
                event_type="pdms_verification",
                problem=getattr(self, "_pending_recovery_reason", None),
                method=method_name,
                outcome="rejected",
                details={
                    "attempt": attempt_index,
                    "lateral_distance_m": lateral_distance_m,
                    "sequence": sequence_steps,
                    "waypoint": selected_waypoint,
                    "failure_flags": failure_flags,
                    "verifier_score": score,
                    "verifier_threshold": self.verifier_threshold,
                    "gates": gates,
                    "soft": soft,
                },
            )
            if one_shot_parameter_selection:
                self.log.info(
                    "B5 one-shot VLM choice failed final safety assertion; "
                    "no VLM reselection: %s",
                    feedback,
                )
            else:
                self.log.info("commit-hold VLM retry feedback: %s", feedback)

        if structured_mode or force_brake_on_exhaustion:
            # A valid sequence was never accepted. Do not select a nominal
            # primitive behind the VLM's back and do not enter escalation; the
            # caller routes this marker directly to the brake safety fallback.
            return _ForceBrakeFallback(
                "sequence_attempts_exhausted" if structured_mode else "primitive_attempts_exhausted"
            )

        fallback = self._nominal_recovery_fallback(
            nominal_eligible=nominal_eligible,
            names=names,
            proposals=proposals,
            states=states,
        )
        if fallback is not None:
            idx, score = fallback[:2]
            self._record_failure_memory(
                info,
                event_type="primitive_selection_fallback",
                problem=getattr(self, "_pending_recovery_reason", None),
                method=names[idx],
                outcome="vlm_retries_exhausted_using_nominal_pass",
                details={"verifier_score": score},
            )
            self.log.warning(
                "commit-hold VLM parameter selection exhausted; using nominal "
                "verifier-passed primitive '%s'",
                names[idx],
            )
            return fallback
        return None

    def _latched_lane_bypass_choice(
        self,
        *,
        info: Dict[str, Any],
        privileged_info: Optional[List[Dict[str, Any]]],
        names: List[str],
        proposals: np.ndarray,
        states: List[np.ndarray],
        brake_idx: Optional[int],
    ) -> tuple[str, Dict[str, Any], float]:
        """Try only the latched bypass; otherwise return a one-tick brake."""
        intent = getattr(self, "_lane_bypass_intent", None)
        assert intent is not None
        action = str(intent.get("action") or "lane_bypass_left")
        direction = "right" if action.endswith("_right") else "left"
        action_idx = next((i for i, name in enumerate(names) if name == action), None)
        verify_info = dict(info)
        verify_info["privileged_agents"] = privileged_info

        rejection = "traffic_or_topology_gate_closed"
        action_score = 0.0
        # The traffic gate can open only after the ego has waited immediately
        # behind a wide construction sign.  Before committing the forward
        # maneuver, use that sign's exact bbox to check that the stopped car has
        # enough longitudinal room to physically reach the fixed adjacent lane.
        # If it does not, remain inside the same exclusive primitive and create
        # space once on the route lane.  Never attempt the infeasible forward
        # trajectory and never expose another primitive/SparseDrive action.
        if (
            action_idx is not None
            and int(intent.get("retreat_count", 0)) < 1
        ):
            entry_assessment = lane_bypass_entry_retreat_assessment(
                info=info,
                privileged_info=privileged_info,
                direction=direction,
            )
            intent["entry_retreat_assessment"] = entry_assessment
            if bool(entry_assessment.get("requires_retreat", False)):
                forward_trajectory = np.asarray(
                    proposals[action_idx], dtype=np.float32
                )
                retreat = materialize_hd_map_lane_bypass_retreat(
                    info=info,
                    privileged_info=privileged_info,
                    direction=direction,
                    output_num_poses=max(
                        2,
                        int(
                            intent.get(
                                "output_num_poses",
                                forward_trajectory.shape[0],
                            )
                        ),
                    ),
                    allow_route_centered=True,
                    retreat_distance_m=(
                        LANE_BYPASS_ENTRY_RETREAT_BASE_DISTANCE_M
                        + max(
                            0.0,
                            float(
                                entry_assessment[
                                    "minimum_entry_distance_m"
                                ]
                            )
                            - float(entry_assessment["entry_distance_m"]),
                        )
                    ),
                    blocker_id_hint=entry_assessment.get("blocker_id"),
                )
                if retreat is not None:
                    retreat_plan, retreat_states = retreat
                    retreat_score = float(
                        self.verifier.score(retreat_plan, verify_info)
                    )
                    retreat_last = getattr(self.verifier, "last_result", None)
                    retreat_gates = (
                        dict(getattr(retreat_last, "gates", {}) or {})
                        if retreat_last is not None
                        else {}
                    )
                    retreat_soft = (
                        dict(getattr(retreat_last, "soft", {}) or {})
                        if retreat_last is not None
                        else {}
                    )
                    retreat_passed = bool(
                        retreat_last is not None
                        and float(retreat_gates.get("nc", 0.0)) > 0.0
                        and float(retreat_gates.get("dac", 0.0)) > 0.0
                        and float(retreat_soft.get("ttc", 0.0)) > 0.0
                    )
                    if retreat_passed and not self._verified_trajectory_offroad(
                        retreat_plan, info, allow_tier_half_crossing=True
                    ):
                        intent["committed"] = True
                        intent["last_progress_ts"] = float(
                            info.get("timestamp", 0.0) or 0.0
                        )
                        intent["retreat_count"] = int(
                            intent.get("retreat_count", 0)
                        ) + 1
                        intent["last_gate"] = "bbox_entry_internal_retreat"
                        payload = self._build_hold_payload(
                            action_idx,
                            names,
                            proposals,
                            states,
                            retreat_score,
                            selected_plan=retreat_plan,
                            selected_states=retreat_states,
                            recovery_name=action,
                            execute_seconds=(
                                max(
                                    1,
                                    int(
                                        np.count_nonzero(
                                            np.asarray(retreat_states)[:, 0]
                                            < -1e-4
                                        )
                                    ),
                                )
                                * 0.5
                            ),
                            front_blockage=intent.get("front_blockage"),
                            escape_direction=direction,
                        )
                        self._mark_lane_bypass_payload(payload, action)
                        payload["recovery_direct_primitive"] = True
                        payload["lane_bypass_internal_phase"] = "retreat"
                        payload["preserve_lane_bypass_intent"] = True
                        payload["lane_bypass_entry_assessment"] = (
                            entry_assessment
                        )
                        self.log.warning(
                            "latched lane bypass '%s' has only %.2fm before "
                            "blocker bbox %s (minimum %.2fm); committing its "
                            "single HD-map route-lane retreat before the fixed "
                            "forward bypass (verifier %.3f)",
                            action,
                            float(entry_assessment["entry_distance_m"]),
                            entry_assessment.get("blocker_id"),
                            float(entry_assessment["minimum_entry_distance_m"]),
                            retreat_score,
                        )
                        return action, payload, retreat_score
                # Fail closed.  The forward candidate is geometrically valid
                # for PDMS but physically cannot move sideways before impact.
                rejection = "bbox_entry_internal_retreat_gate_failed"
                action_idx = None
        # A closed traffic gate intentionally hides the forward bypass from the
        # provider.  If Loop 1 has already put the ego in that fixed borrowed
        # lane, hiding the forward candidate must not also hide the same
        # primitive's reverse-to-route safety phase.  Once stopped, attempt that
        # map-bound retreat directly; the positive VLM decision remains latched
        # and no regular primitive/SparseDrive action is exposed in between.
        if (
            action_idx is None
            and int(intent.get("retreat_count", 0)) < 1
            and names
        ):
            retreat = materialize_hd_map_lane_bypass_retreat(
                info=info,
                privileged_info=privileged_info,
                direction=direction,
                output_num_poses=max(
                    2,
                    int(intent.get("output_num_poses", 2)),
                ),
                allow_route_centered=False,
            )
            if retreat is not None:
                retreat_plan, retreat_states = retreat
                retreat_score = float(
                    self.verifier.score(retreat_plan, verify_info)
                )
                retreat_last = getattr(self.verifier, "last_result", None)
                retreat_gates = (
                    dict(getattr(retreat_last, "gates", {}) or {})
                    if retreat_last is not None
                    else {}
                )
                retreat_soft = (
                    dict(getattr(retreat_last, "soft", {}) or {})
                    if retreat_last is not None
                    else {}
                )
                retreat_passed = bool(
                    retreat_last is not None
                    and float(retreat_gates.get("nc", 0.0)) > 0.0
                    and float(retreat_gates.get("dac", 0.0)) > 0.0
                    and float(retreat_soft.get("ttc", 0.0)) > 0.0
                )
                if retreat_passed and not self._verified_trajectory_offroad(
                    retreat_plan, info, allow_tier_half_crossing=True
                ):
                    payload_idx = (
                        int(brake_idx)
                        if brake_idx is not None
                        else 0
                    )
                    active_retreat_steps = max(
                        1,
                        int(
                            np.count_nonzero(
                                np.asarray(retreat_states)[:, 0] < -1e-4
                            )
                        ),
                    )
                    intent["committed"] = True
                    intent["last_progress_ts"] = float(
                        info.get("timestamp", 0.0) or 0.0
                    )
                    intent["retreat_count"] = int(
                        intent.get("retreat_count", 0)
                    ) + 1
                    intent["last_gate"] = (
                        "closed_traffic_gate_internal_retreat"
                    )
                    payload = self._build_hold_payload(
                        payload_idx,
                        names,
                        proposals,
                        states,
                        retreat_score,
                        selected_plan=retreat_plan,
                        selected_states=retreat_states,
                        recovery_name=action,
                        execute_seconds=active_retreat_steps * 0.5,
                        front_blockage=intent.get("front_blockage"),
                        escape_direction=direction,
                    )
                    self._mark_lane_bypass_payload(payload, action)
                    payload["recovery_direct_primitive"] = True
                    payload["lane_bypass_internal_phase"] = "retreat"
                    payload["preserve_lane_bypass_intent"] = True
                    self.log.warning(
                        "latched lane bypass '%s' traffic gate is closed "
                        "while ego occupies its fixed target lane; committing "
                        "the same primitive's %.1fs HD-map retreat "
                        "(verifier %.3f)",
                        action,
                        active_retreat_steps * 0.5,
                        retreat_score,
                    )
                    return action, payload, retreat_score
                rejection = "closed_gate_internal_retreat_gate_failed"
        if action_idx is not None:
            temporal_tree_enabled = bool(
                getattr(self, "recovery_explainer", None) is not None
                and getattr(
                    self.recovery_explainer.cfg,
                    "temporal_tree_search",
                    False,
                )
            )
            if temporal_tree_enabled:
                temporal_results = self._search_recovery_temporal_trees(
                    action_specs=get_recovery_sequence_action_specs([action]),
                    candidate_indices=[action_idx],
                    names=names,
                    info=info,
                    privileged_info=privileged_info,
                    output_num_poses=max(
                        2,
                        int(
                            intent.get(
                                "output_num_poses",
                                proposals.shape[1],
                            )
                        ),
                    ),
                    lateral_distance_ranges=None,
                    obstacle_clearance_m=intent.get(
                        "required_obstacle_clearance_m"
                    ),
                )
                temporal_result = temporal_results.get(action)
                if (
                    temporal_result is not None
                    and temporal_result.best_leaf is not None
                    and temporal_result.best_leaf.trajectory is not None
                    and temporal_result.best_leaf.states is not None
                ):
                    leaf = temporal_result.best_leaf
                    intent["committed"] = True
                    intent["last_progress_ts"] = float(
                        info.get("timestamp", 0.0) or 0.0
                    )
                    intent["last_gate"] = "temporal_tree_passed"
                    payload = self._build_hold_payload(
                        action_idx,
                        names,
                        proposals,
                        states,
                        float(leaf.score),
                        selected_plan=leaf.trajectory,
                        selected_states=leaf.states,
                        recovery_name=action,
                        execute_seconds=leaf.profile.get("execute_seconds"),
                        front_blockage=intent.get("front_blockage"),
                        escape_direction=direction,
                    )
                    self._mark_lane_bypass_payload(payload, action)
                    payload["recovery_direct_primitive"] = True
                    payload["recovery_temporal_tree"] = {
                        "selected_profile": dict(leaf.profile),
                        "scorecard": temporal_result.scorecard(),
                    }
                    self.log.warning(
                        "latched lane bypass '%s' safety window opened; "
                        "temporal tree selected %s (verifier %.3f)",
                        action,
                        leaf.profile.get("profile_id"),
                        float(leaf.score),
                    )
                    return action, payload, float(leaf.score)
            trajectory = np.asarray(proposals[action_idx], dtype=np.float32)
            trajectory_states = np.asarray(states[action_idx], dtype=np.float32)
            retained_clearance_met = True
            if temporal_tree_enabled:
                regenerated = materialize_hd_map_lane_bypass(
                    info=info,
                    privileged_info=privileged_info,
                    direction=direction,
                    output_num_poses=max(
                        2,
                        int(intent.get("output_num_poses", trajectory.shape[0])),
                    ),
                    entry_delay_sec=0.0,
                    obstacle_clearance_m=float(
                        intent.get("required_obstacle_clearance_m", 0.0) or 0.0
                    ),
                    temporal_profile={"profile_kind": "legacy_auto"},
                )
                if regenerated is None:
                    retained_clearance_met = False
                else:
                    trajectory, trajectory_states = regenerated
            action_score = float(self.verifier.score(trajectory, verify_info))
            last = getattr(self.verifier, "last_result", None)
            gates = dict(getattr(last, "gates", {}) or {}) if last else {}
            soft = dict(getattr(last, "soft", {}) or {}) if last else {}
            ttc_passed = bool(float(soft.get("ttc", 0.0)) > 0.0)
            gating_nc = []
            gating_ttc = []
            if last is not None and not ttc_passed:
                # The intentional route-lane blocker can trip the TTC probe
                # before the trajectory has completed its lateral departure.
                # Permit that specific static conflict, but never a vehicle,
                # pedestrian, or unknown dynamic conflict.  TTC's collision
                # records retain object_type even when PDMS debug output is off.
                ttc_probe = getattr(self.verifier, "_ttc", None)
                collision_map = getattr(ttc_probe, "last_collisions", {}) or {}
                gating_nc = [
                    collision
                    for collision in list(collision_map.get("nc", []) or [])
                    if not bool(collision.get("beyond_gate_horizon", False))
                ]
                gating_ttc = [
                    collision
                    for collision in list(collision_map.get("ttc", []) or [])
                    if not bool(collision.get("beyond_gate_horizon", False))
                ]
                ttc_passed = bool(
                    gating_ttc
                    and all(
                        str(collision.get("object_type", "unknown")) == "static"
                        for collision in gating_ttc
                    )
                )
                dynamic_ttc = any(
                    str(collision.get("object_type", "unknown")) != "static"
                    for collision in gating_ttc
                )
            else:
                dynamic_ttc = False
            static_nc_entrapment = any(
                str(collision.get("object_type", "unknown")) == "static"
                for collision in gating_nc
            )
            static_ttc_entrapment = any(
                str(collision.get("object_type", "unknown")) == "static"
                for collision in gating_ttc
            )
            # Once necessity is latched, the provider's live target-lane traffic
            # gate determines the candidate entry time.  At commit, require NC,
            # DAC *and TTC*: Route 69 showed that NC/DAC can pass at the first
            # pose while TTC already predicts a head-on conflict 4.5 s later.
            # EP/LK remain soft because an intentional lane borrow necessarily
            # departs the nominal route/lane for part of the maneuver.
            passed = bool(
                last is not None
                and retained_clearance_met
                and float(gates.get("nc", 0.0)) > 0.0
                and float(gates.get("dac", 0.0)) > 0.0
                and ttc_passed
            )
            if not passed:
                rejection = (
                    "retained_clearance_generation_failed"
                    if not retained_clearance_met
                    else
                    "ttc_gate_failed"
                    if last is not None
                    and float(gates.get("nc", 0.0)) > 0.0
                    and float(gates.get("dac", 0.0)) > 0.0
                    and not ttc_passed
                    else "hard_pdms_gate_failed"
                )
            elif self._verified_trajectory_offroad(
                trajectory, info, allow_tier_half_crossing=True
            ):
                passed = False
                rejection = "footprint_offroad"
            elif (
                self.recovery_full_plan_dac
                and self._primitive_tier_gate_fails(trajectory, info)
            ):
                passed = False
                rejection = "full_plan_dac_failed"
            if passed:
                intent["committed"] = True
                intent["last_progress_ts"] = float(
                    info.get("timestamp", 0.0) or 0.0
                )
                intent["last_gate"] = "passed"
                payload = self._build_hold_payload(
                    action_idx,
                    names,
                    proposals,
                    states,
                    action_score,
                    selected_plan=trajectory,
                    selected_states=trajectory_states,
                    recovery_name=action,
                    front_blockage=intent.get("front_blockage"),
                    escape_direction=direction,
                )
                self._mark_lane_bypass_payload(payload, action)
                payload["recovery_direct_primitive"] = True
                self.log.warning(
                    "latched lane bypass '%s' safety window opened; "
                    "NC/DAC/TTC gates passed; committing the complete "
                    "primitive (aggregate verifier %.3f)",
                    action,
                    action_score,
                )
                return action, payload, action_score

            # Loop 1 may already have entered the route's fixed left-adjacent
            # lane by the time the VLM latches the bypass. If live opposing
            # actor then closes the TTC gate, braking there blocks that actor
            # forever.  Stay inside the same exclusive primitive: reverse along
            # the borrowed HD-map centreline until the route-lane blocker bbox
            # is behind the ego, blend once onto the route centreline, and keep
            # the latch alive so the complete forward bypass is retried after
            # the actor passes.
            if (
                (
                    dynamic_ttc
                    or static_nc_entrapment
                    or static_ttc_entrapment
                )
                and int(intent.get("retreat_count", 0)) < 1
            ):
                retreat = materialize_hd_map_lane_bypass_retreat(
                    info=info,
                    privileged_info=privileged_info,
                    direction=direction,
                    output_num_poses=max(
                        2, int(intent.get("output_num_poses", trajectory.shape[0]))
                    ),
                    allow_route_centered=(
                        static_nc_entrapment or static_ttc_entrapment
                    ),
                )
                if retreat is not None:
                    retreat_plan, retreat_states = retreat
                    retreat_score = float(
                        self.verifier.score(retreat_plan, verify_info)
                    )
                    retreat_last = getattr(self.verifier, "last_result", None)
                    retreat_gates = (
                        dict(getattr(retreat_last, "gates", {}) or {})
                        if retreat_last is not None
                        else {}
                    )
                    retreat_soft = (
                        dict(getattr(retreat_last, "soft", {}) or {})
                        if retreat_last is not None
                        else {}
                    )
                    retreat_probe = getattr(self.verifier, "_ttc", None)
                    retreat_collision_map = (
                        getattr(retreat_probe, "last_collisions", {}) or {}
                    )
                    retreat_gating_collisions = [
                        collision
                        for term in ("nc", "ttc")
                        for collision in list(
                            retreat_collision_map.get(term, []) or []
                        )
                        if not bool(collision.get("beyond_gate_horizon", False))
                    ]
                    retreat_static_only = bool(
                        retreat_gating_collisions
                        and all(
                            str(collision.get("object_type", "unknown"))
                            == "static"
                            for collision in retreat_gating_collisions
                        )
                    )
                    retreat_passed = bool(
                        retreat_last is not None
                        and float(retreat_gates.get("dac", 0.0)) > 0.0
                        and (
                            (
                                float(retreat_gates.get("nc", 0.0)) > 0.0
                                and float(retreat_soft.get("ttc", 0.0)) > 0.0
                            )
                            or retreat_static_only
                        )
                    )
                    if retreat_passed and not self._verified_trajectory_offroad(
                        retreat_plan, info, allow_tier_half_crossing=True
                    ):
                        intent["committed"] = True
                        intent["last_progress_ts"] = float(
                            info.get("timestamp", 0.0) or 0.0
                        )
                        intent["retreat_count"] = int(
                            intent.get("retreat_count", 0)
                        ) + 1
                        intent["last_gate"] = "dynamic_ttc_internal_retreat"
                        payload = self._build_hold_payload(
                            action_idx,
                            names,
                            proposals,
                            states,
                            retreat_score,
                            selected_plan=retreat_plan,
                            selected_states=retreat_states,
                            recovery_name=action,
                            execute_seconds=(
                                max(
                                    1,
                                    int(
                                        np.count_nonzero(
                                            np.asarray(retreat_states)[:, 0]
                                            < -1e-4
                                        )
                                    ),
                                )
                                * 0.5
                            ),
                            front_blockage=intent.get("front_blockage"),
                            escape_direction=direction,
                        )
                        self._mark_lane_bypass_payload(payload, action)
                        payload["recovery_direct_primitive"] = True
                        payload["lane_bypass_internal_phase"] = "retreat"
                        payload["preserve_lane_bypass_intent"] = True
                        self.log.warning(
                            "latched lane bypass '%s' has a live TTC "
                            "conflict while trapped by the blocker bbox; "
                            "committing its HD-map/bbox reverse-to-route phase "
                            "(verifier %.3f)",
                            action,
                            retreat_score,
                        )
                        return action, payload, retreat_score
                    rejection = "internal_retreat_gate_failed"

        intent["committed"] = False
        intent["last_gate"] = rejection
        intent["last_score"] = action_score
        self.log.info(
            "latched lane bypass '%s' still waiting (%s, verifier %.3f); "
            "holding brake and suppressing every other planner/action",
            action,
            rejection,
            action_score,
        )
        if brake_idx is not None:
            brake_plan = np.asarray(proposals[brake_idx], dtype=np.float32)
            brake_states = (
                np.asarray(states[brake_idx], dtype=np.float32)
                if brake_idx < len(states)
                else np.zeros_like(brake_plan)
            )
            brake_score = float(self.verifier.score(brake_plan, verify_info))
        else:
            brake_plan = np.zeros((2, 2), dtype=np.float32)
            brake_states = np.zeros((2, 2), dtype=np.float32)
            brake_score = 0.0
        payload = {
            "selected_plan": brake_plan,
            "selected_score": brake_score,
            "selected_source": f"recovery_wait:{action}",
            "fallback_primitive": self._RECOVERY_BRAKE_PRIMITIVE,
            "fallback_mode": "single_tick",
            "fallback_states": brake_states,
            "recovery_pending_primitive": action,
            "recovery_wait_reason": rejection,
        }
        self._mark_lane_bypass_payload(payload, action)
        return self._RECOVERY_BRAKE_PRIMITIVE, payload, brake_score

    def _recover_verify_first_pass(
        self,
        *,
        obs: Dict[str, Any],
        info: Dict[str, Any],
        privileged_info: Optional[List[Dict[str, Any]]],
        output_num_poses: int,
        allow_vlm_selection: bool = True,
    ) -> Optional[tuple[str, Dict[str, Any], float]]:
        """Verify each primitive and commit one, as ``(name, payload, score)``.

        Default (``recovery_explainer.select_primitive`` off): returns the FIRST
        primitive (in random order) whose verifier score clears the threshold.

        VLM-select on: evaluates nominal trajectories for a safe fallback, then
        asks the VLM for a primitive plus any tunable lateral distance. The exact
        parameterized rollout is verified before commit; rejection feedback is
        returned to the VLM for another choice. Exhausted retries fall back to a
        nominal verifier-passed candidate or the escalation/brake path."""
        cfg = self.recovery_cfg
        if cfg is None or not getattr(cfg, "enabled", False):
            self.log.warning("commit-hold recovery requested but recovery is disabled")
            return None
        if self.verifier is None:
            return None
        try:
            proposals, _scores, names, states = get_primitive_proposals_and_scores(
                info=info,
                privileged_info=privileged_info,
                output_num_poses=output_num_poses,
                allowed_primitives=getattr(cfg, "allowed_primitives", ()),
                lane_bypass_min_gap_m=getattr(
                    cfg, "lane_bypass_min_gap_m", 0.0
                ),
                lane_bypass_clearance_buffer_m=getattr(
                    cfg, "lane_bypass_clearance_buffer_m", 1.0
                ),
                lane_bypass_required_time_sec=getattr(
                    cfg, "lane_bypass_required_time_sec", 3.5
                ),
                lane_bypass_min_time_gap_sec=getattr(
                    cfg, "lane_bypass_min_time_gap_sec", 4.5
                ),
                obstacle_clearance_m=getattr(cfg, "obstacle_clearance_m", 8.0),
            )
        except Exception:
            self.log.exception("commit-hold: primitive generation failed")
            return None
        if names is None or len(names) == 0:
            return None

        # Remove map-invalid primitives before nominal PDMS trials, random
        # ordering, escalation, or any VLM prompt. This is the commit-and-hold
        # counterpart of the legacy recovery filter in ``_recover``.
        hd_map_rejected = self._footprint_offroad_batch(proposals, info)
        if np.any(hd_map_rejected):
            rejected_names = [names[idx] for idx in np.flatnonzero(hd_map_rejected)]
            keep = np.flatnonzero(~hd_map_rejected).tolist()
            self.log.warning(
                "HD-map hard filter removed commit-hold primitives: %s",
                rejected_names,
            )
            if not keep:
                return None
            proposals = proposals[keep]
            names = [names[idx] for idx in keep]
            states = [states[idx] for idx in keep]

        junction_heading_rejected = junction_route_heading_violation_mask(
            proposals, info
        )
        if np.any(junction_heading_rejected):
            rejected_names = [
                names[idx] for idx in np.flatnonzero(junction_heading_rejected)
            ]
            keep = np.flatnonzero(~junction_heading_rejected).tolist()
            self.log.warning(
                "junction route-heading filter removed commit-hold primitives: %s",
                rejected_names,
            )
            if not keep:
                return None
            proposals = proposals[keep]
            names = [names[idx] for idx in keep]
            states = [states[idx] for idx in keep]

        # In VLM-selection mode nominal verification provides a safe fallback
        # and escalation diagnostics. Sequence selection receives the complete
        # per-primitive PDMS precheck table; legacy single-primitive selection
        # keeps its no-scalar-score prompt. Either path is verified again after
        # the VLM materializes an exact choice.
        budget_random_defaults = bool(
            self.recovery_explainer is not None
            and getattr(self.recovery_explainer, "recovery_query_budget_exhausted", False)
        )
        vlm_select = (
            not budget_random_defaults
            and allow_vlm_selection
            and self.recovery_explainer is not None
            and (
                getattr(self.recovery_explainer.cfg, "select_primitive", False)
                or getattr(self.recovery_explainer.cfg, "select_sequence", False)
                or getattr(
                    self.recovery_explainer.cfg, "select_recovery_plan", False
                )
            )
        )
        order = list(range(len(names)))
        if self.recovery_random_order or budget_random_defaults:
            self._recovery_rng.shuffle(order)
        if budget_random_defaults:
            self.log.info(
                "Recovery VLM query budget exhausted (%s); random primitives with default parameters",
                self.recovery_explainer.recovery_query_count,
            )
        # In deterministic/non-VLM mode brake remains the last resort. Direct
        # VLM recovery must see it, however: B1 explicitly compares brake with
        # creep, and B4/B5 expose the complete primitive library.
        brake_idx = next(
            (i for i, n in enumerate(names) if n == self._RECOVERY_BRAKE_PRIMITIVE),
            None,
        )
        if getattr(self, "_lane_bypass_intent", None) is not None:
            return self._latched_lane_bypass_choice(
                info=info,
                privileged_info=privileged_info,
                names=names,
                proposals=proposals,
                states=states,
                brake_idx=brake_idx,
            )
        vlm_include_brake = bool(
            vlm_select
            and getattr(self.recovery_explainer.cfg, "include_brake", False)
        )
        if not vlm_include_brake:
            order = [i for i in order if i != brake_idx]

        # ---- no-progress tabu -------------------------------------------
        # Judge the PREVIOUS commit by route progress since it was issued: a
        # primitive that gained < _RECOVERY_TABU_MIN_GAIN_M of route station is
        # banned for _RECOVERY_TABU_CYCLES commit cycles. Without this, two
        # mirror maneuvers that both pass the verifier (lane_left/lane_right at
        # 0.5) ping-pong the ego side to side forever on a fully-blocked road
        # -- observed 26+26 alternating commits to route_timeout. Banning the
        # unproductive one forces rotation through the whole library
        # (swerves, back_bypass, three_point) instead.
        station = self._route_station(info)
        prev = self._last_commit_progress
        self._tabu = {n: ttl - 1 for n, ttl in self._tabu.items() if ttl > 1}
        if prev is not None and station is not None:
            prev_name, prev_station = prev
            gain = station - prev_station
            # Productivity ledger: record what the previous commit actually
            # bought in route terms (brake included) so memory can distinguish
            # accepted-and-useful from accepted-and-stationary.
            self._record_failure_memory(
                info,
                event_type="commit_progress",
                method=prev_name,
                details={"gain_m": float(gain)},
            )
            if (
                prev_name != self._RECOVERY_BRAKE_PRIMITIVE
                and prev_name not in getattr(self, "_recovery_tabu_exempt", ())
                and gain < self._RECOVERY_TABU_MIN_GAIN_M
            ):
                self._tabu[prev_name] = self._RECOVERY_TABU_CYCLES
                self.log.info(
                    "commit-hold tabu: '%s' gained %.2fm of route -> banned "
                    "for %d cycles", prev_name, gain, self._RECOVERY_TABU_CYCLES,
                )
        self._last_commit_progress = None
        if self._tabu:
            allowed = [i for i in order if names[i] not in self._tabu]
            # Never tabu ourselves into an empty candidate set.
            if allowed:
                order = allowed

        verify_info = dict(info)
        verify_info["privileged_agents"] = privileged_info
        # Per-trial (idx, score, nc_gate, ep) records for the anti-deadlock
        # escalation below: when NOTHING passes for several consecutive commit
        # cycles and the ego is parked, we re-consider the collision-free
        # (nc == 1) trials instead of braking forever.
        trials: list[tuple[int, float, float, float]] = []
        self._trial_rank_scores = {}
        # Nominal verifier-passed candidates are retained as a safe fallback if
        # all parameterized VLM attempts fail.
        eligible: list[tuple[int, float]] = []  # (idx, score), verifier-passed + footprint-safe
        nominal_passed: set[int] = set()
        sequence_first_action_evaluations: Dict[str, Dict[str, Any]] = {}
        # Trial-scoring must not pollute the verifier's per-step debug log.
        prev_debug = getattr(self.verifier, "_debug", None)
        if prev_debug is not None:
            self.verifier._debug = False
        try:
            for idx in order:
                traj = np.asarray(proposals[idx], dtype=np.float32)
                score = float(self.verifier.score(traj, verify_info))
                last = getattr(self.verifier, "last_result", None)
                passed = self._verifier_accepts(score)
                nc = float((last.gates or {}).get("nc", 1.0)) if last else 1.0
                ep = float((last.soft or {}).get("ep", 0.0)) if last else 0.0
                gates = dict(getattr(last, "gates", {}) or {}) if last else {}
                soft = dict(getattr(last, "soft", {}) or {}) if last else {}
                rejection_reasons: List[str] = []
                if not passed:
                    rejection_reasons.append("below_pdms_threshold")
                if self.recovery_rank_without_ep:
                    self._trial_rank_scores[idx] = (
                        float(gates.get("nc", 1.0))
                        * float(gates.get("dac", 1.0))
                        * (
                            5.0 * float(soft.get("ttc", 1.0))
                            + 2.0 * float(soft.get("lk", 1.0))
                        )
                        / 7.0
                    )
                trials.append((idx, score, nc, ep))
                self.log.info(
                    "commit-hold trial primitive '%s': verifier %.3f (%s)",
                    names[idx], score, "pass" if passed else "fail",
                )
                if passed and self._verified_trajectory_offroad(traj, info):
                    self.log.info(
                        "commit-hold trial primitive '%s': footprint leaves "
                        "the mapped road -> vetoed despite verifier pass",
                        names[idx],
                    )
                    passed = False
                    rejection_reasons.append("footprint_offroad")
                # Full-maneuver DAC-tier trial gate (recovery_full_plan_dac,
                # default off): the verifier gated only the plan head; probe
                # the whole materialized centerline for tier-0.5 samples.
                if passed and self.recovery_full_plan_dac and self._primitive_tier_gate_fails(traj, info):
                    self.log.info(
                        "commit-hold trial primitive '%s': full-plan DAC "
                        "tier<=0.5 -> vetoed despite verifier pass",
                        names[idx],
                    )
                    passed = False
                    rejection_reasons.append("full_plan_dac_failed")
                sequence_first_action_evaluations[names[idx]] = {
                    "pdms_score": score,
                    "pdms_threshold": float(self.verifier_threshold),
                    "eligible": bool(passed),
                    "gates": gates,
                    "soft": soft,
                    "rejection_reasons": rejection_reasons,
                }
                if passed:
                    nominal_passed.add(idx)
                    if vlm_select:
                        # The VLM sees brake, but an unavailable/invalid model
                        # must not silently turn a productive recovery into a
                        # long brake hold. Reserve brake for the explicit VLM
                        # choice or the final safety fallback below.
                        if idx != brake_idx:
                            eligible.append((idx, score))
                        continue
                    self._forced_brake_streak = 0
                    if station is not None:
                        self._last_commit_progress = (names[idx], station)
                    payload = self._build_hold_payload(idx, names, proposals, states, score)
                    if budget_random_defaults:
                        payload["recovery_selection_mode"] = "budget_random_defaults"
                        payload["recovery_query_count"] = self.recovery_explainer.recovery_query_count
                    return names[idx], payload, score

            tool_rule_choice = self._tool_rule_blocked_escape_choice(
                info=info,
                nominal_eligible=eligible,
                names=names,
            )
            if vlm_select and tool_rule_choice is not None:
                idx, score, rule_debug = tool_rule_choice
                action = str(names[idx])
                self._forced_brake_streak = 0
                if station is not None:
                    self._last_commit_progress = (action, station)
                payload = self._build_hold_payload(
                    idx,
                    names,
                    proposals,
                    states,
                    score,
                    front_blockage=(
                        f"tool rule: persistent {rule_debug.get('obstacle_type') or 'front'} "
                        "obstacle"
                    ),
                )
                payload["selected_source"] = f"tool_rule:{action}"
                payload["recovery_tool_rule"] = rule_debug
                self.log.warning(
                    "persistent-blockage tool rule committed verified '%s' "
                    "(PDMS %.3f, stationary %.1fs, obstacle=%s %s at %sm)",
                    action,
                    score,
                    float(rule_debug["stationary_sec"]),
                    rule_debug.get("obstacle_type", "unknown"),
                    rule_debug.get("obstacle_id", "unknown"),
                    rule_debug.get("obstacle_distance_m", "unknown"),
                )
                return action, payload, score

            # VLM chooses a primitive plus any tunable lateral distance. The
            # resulting concrete trajectory is generated and verified; a
            # B5 first restricts each parameterized family to a sampled
            # PDMS-safe interval, then makes exactly one VLM call. Other modes
            # retain the legacy retry behavior. Nominal verifier-passed
            # candidates remain the safe fallback if selection is unavailable.
            if vlm_select:
                candidate_indices = list(order)
                feasible_ranges: Optional[Dict[int, tuple[float, float]]] = None
                preverify_ranges = bool(
                    not getattr(
                        self.recovery_explainer.cfg,
                        "temporal_tree_search",
                        False,
                    )
                    and not (
                        getattr(self.recovery_explainer.cfg, "select_sequence", False)
                        and not getattr(
                            self.recovery_explainer.cfg,
                            "select_recovery_plan",
                            False,
                        )
                    )
                    and getattr(self.recovery_explainer.cfg, "select_parameters", False)
                    and getattr(
                        self.recovery_explainer.cfg,
                        "preverify_parameter_ranges",
                        False,
                    )
                )
                if preverify_ranges:
                    feasible_ranges = {}
                    filtered_indices: List[int] = []
                    sample_count = max(
                        2,
                        int(
                            getattr(
                                self.recovery_explainer.cfg,
                                "parameter_range_samples",
                                21,
                            )
                        ),
                    )
                    for idx in candidate_indices:
                        static_range = recovery_lateral_distance_range(names[idx])
                        if static_range is None:
                            if idx in nominal_passed:
                                filtered_indices.append(idx)
                            continue
                        safe_range = self._pdms_feasible_parameter_range(
                            primitive_name=names[idx],
                            info=info,
                            privileged_info=privileged_info,
                            output_num_poses=output_num_poses,
                            sample_count=sample_count,
                        )
                        if safe_range is not None:
                            filtered_indices.append(idx)
                            feasible_ranges[idx] = safe_range
                    candidate_indices = filtered_indices
                    self.log.info(
                        "B5 one-shot VLM candidates after PDMS parameter sweep: %s",
                        [names[idx] for idx in candidate_indices],
                    )
                chosen = self._vlm_pick_recovery_primitive(
                    obs=obs,
                    info=info,
                    candidate_indices=candidate_indices,
                    nominal_eligible=eligible,
                    names=names,
                    proposals=proposals,
                    states=states,
                    privileged_info=privileged_info,
                    output_num_poses=output_num_poses,
                    lateral_distance_ranges=feasible_ranges,
                    sequence_first_action_names=(
                        [
                            names[idx]
                            for idx in candidate_indices
                            if idx in nominal_passed
                            or (
                                getattr(
                                    self.recovery_explainer.cfg,
                                    "select_recovery_plan",
                                    False,
                                )
                                and feasible_ranges is not None
                            )
                        ]
                        if (
                            getattr(self.recovery_explainer.cfg, "select_sequence", False)
                            or getattr(
                                self.recovery_explainer.cfg,
                                "select_recovery_plan",
                                False,
                            )
                        )
                        else None
                    ),
                    sequence_first_action_evaluations=(
                        sequence_first_action_evaluations
                        if (
                            getattr(
                                self.recovery_explainer.cfg,
                                "select_sequence",
                                False,
                            )
                            or getattr(
                                self.recovery_explainer.cfg,
                                "select_recovery_plan",
                                False,
                            )
                        )
                        else None
                    ),
                )
                force_brake_only = isinstance(chosen, _ForceBrakeFallback)
                if force_brake_only:
                    self._record_failure_memory(
                        info,
                        event_type="primitive_selection_fallback",
                        problem=getattr(self, "_pending_recovery_reason", None),
                        method=self._RECOVERY_BRAKE_PRIMITIVE,
                        # Preserve the existing memory schema for legacy
                        # sequence consumers; plan mode shares this fallback.
                        outcome="sequence_unavailable_force_brake",
                        details={"reason": chosen.reason},
                    )
                    self.log.warning(
                        "commit-hold structured recovery unavailable (%s); bypassing nominal "
                        "and escalation fallbacks and selecting brake",
                        chosen.reason,
                    )
                elif chosen is not None:
                    chosen_idx, chosen_score, chosen_plan, chosen_states, lateral_distance_m = chosen[:5]
                    sequence_steps = list(chosen[5]) if len(chosen) > 5 else []
                    plan_metadata = (
                        dict(chosen[6])
                        if len(chosen) > 6 and isinstance(chosen[6], dict)
                        else {}
                    )
                    selected_waypoint = (
                        dict(plan_metadata["waypoint"])
                        if isinstance(plan_metadata.get("waypoint"), dict)
                        else None
                    )
                    sequence_name = (
                        "sequence:" + ">".join(
                            str(step.get("action")) for step in sequence_steps
                        )
                        if sequence_steps
                        else ("waypoint" if selected_waypoint is not None else names[chosen_idx])
                    )
                    recovery_name = (
                        str(sequence_steps[0].get("action"))
                        if sequence_steps
                        else ("waypoint" if selected_waypoint is not None else names[chosen_idx])
                    )
                    self._forced_brake_streak = 0
                    if station is not None:
                        self._last_commit_progress = (recovery_name, station)
                    payload = self._build_hold_payload(
                        chosen_idx,
                        names,
                        proposals,
                        states,
                        chosen_score,
                        selected_plan=chosen_plan,
                        selected_states=chosen_states,
                        lateral_distance_m=lateral_distance_m,
                        recovery_name=recovery_name,
                        sequence_steps=sequence_steps,
                        waypoint=selected_waypoint,
                        execute_seconds=plan_metadata.get("execute_seconds"),
                        front_blockage=plan_metadata.get("front_blockage"),
                        escape_direction=plan_metadata.get("escape_direction"),
                    )
                    if "sampled_candidates" in plan_metadata:
                        payload["recovery_probability_sampling"] = {
                            "selected_probability": plan_metadata.get("probability"),
                            "sample_multiplier": plan_metadata.get(
                                "probability_sample_multiplier"
                            ),
                            "total_requested_sample_count": plan_metadata.get(
                                "total_requested_sample_count"
                            ),
                            "requested_sample_count": plan_metadata.get(
                                "requested_sample_count"
                            ),
                            "candidates": list(
                                plan_metadata.get("sampled_candidates") or []
                            ),
                        }
                    if plan_metadata.get("temporal_tree_search"):
                        payload["recovery_temporal_tree"] = {
                            "selected_probability": plan_metadata.get(
                                "probability"
                            ),
                            "selected_profile": dict(
                                plan_metadata.get("selected_profile") or {}
                            ),
                            "scorecard": dict(
                                plan_metadata.get("temporal_scorecard") or {}
                            ),
                        }
                    if isinstance(plan_metadata.get("tool_rule"), dict):
                        payload["selected_source"] = f"tool_rule:{recovery_name}"
                        payload["recovery_tool_rule"] = dict(
                            plan_metadata["tool_rule"]
                        )
                    for bypass_side in ("left", "right"):
                        decision = plan_metadata.get(
                            f"needs_lane_bypass_{bypass_side}"
                        )
                        if decision is None:
                            continue
                        payload[
                            f"recovery_needs_lane_bypass_{bypass_side}"
                        ] = bool(decision)
                        payload["recovery_needs_lane_bypass"] = bool(decision)
                        payload["recovery_direct_primitive"] = bool(decision)
                    if sequence_steps:
                        payload["selected_source"] = f"recovery:{sequence_name}"
                        payload["fallback_sequence_name"] = sequence_name
                    elif selected_waypoint is not None:
                        payload["selected_source"] = "recovery:waypoint"
                    return (
                        recovery_name,
                        payload,
                        chosen_score,
                    )
            else:
                force_brake_only = False

            # ---- anti-deadlock escalation --------------------------------
            # A fully-blocked route can zero EVERY primitive forever (any
            # bypass leaves the corridor -> DAC=0; backing up has EP=0), so the
            # "force brake" fallback below deadlocks: brake 6s -> all fail ->
            # brake ... until the leaderboard calls the agent blocked (~2-6
            # composed points). After ESCALATE_AFTER consecutive forced-brake
            # commits with the ego parked, commit the best collision-free
            # (nc == 1) primitive instead -- ranked by (verifier score, ep) so
            # forward bypasses beat reversing holds. NC stays a hard veto:
            # a predicted collision is never escalated into.
            ego_speed = abs(float(info.get("ego_velo", 0.0)))
            if not force_brake_only and ego_speed > 1.0:
                self._forced_brake_streak = 0
            if (
                not force_brake_only
                and self.recovery_allow_failed_primitive_escalation
                and self._forced_brake_streak >= self._RECOVERY_ESCALATE_AFTER
                and ego_speed <= 1.0
            ):
                safe = [
                    t for t in trials
                    if t[2] >= 1.0
                    and not self._footprint_offroad(
                        np.asarray(proposals[t[0]], dtype=np.float32), info
                    )
                ]
                if safe:
                    # Diversity term: without it the two mirror lane changes
                    # (both scoring ~0.25) alternate forever while the wedged
                    # ego never gets to try back_*/swerve/three_point -- rank
                    # least-escalated first so repeated escalations sweep the
                    # whole library instead of replaying the same failed pair.
                    idx, score, _nc, ep = max(
                        safe,
                        key=lambda t: (
                            -self._escalation_commits.get(names[t[0]], 0),
                            t[1],
                            t[3],
                        ),
                    )
                    self._escalation_commits[names[idx]] = (
                        self._escalation_commits.get(names[idx], 0) + 1
                    )
                    self._forced_brake_streak = 0
                    self.log.warning(
                        "commit-hold ESCALATION: %d consecutive forced brakes "
                        "with ego parked; committing best collision-free "
                        "primitive '%s' (verifier %.3f, ep %.3f) despite "
                        "threshold %.3f",
                        self._RECOVERY_ESCALATE_AFTER, names[idx], score, ep,
                        self.verifier_threshold,
                    )
                    traj = np.asarray(proposals[idx], dtype=np.float32)
                    if station is not None:
                        self._last_commit_progress = (names[idx], station)
                    payload = {
                        "selected_plan": traj,
                        "selected_score": score,
                        "selected_source": f"recovery:{names[idx]}",
                        "fallback_primitive": names[idx],
                        "fallback_mode": "hold",
                        "fallback_states": (
                            np.asarray(states[idx], dtype=np.float32)
                            if idx < len(states) else None
                        ),
                    }
                    return names[idx], payload, score
            # Every non-brake primitive failed the verifier: force brake instead
            # of returning to loop 1. Brake is committed through the exact same
            # commit-and-hold path as any passing primitive. Scored here (still
            # inside the debug-suppressed region) only to report its verdict.
            if brake_idx is not None:
                if not force_brake_only:
                    self._forced_brake_streak += 1
                traj = np.asarray(proposals[brake_idx], dtype=np.float32)
                brake_score = float(self.verifier.score(traj, verify_info))
                brake_passed = self._verifier_accepts(brake_score)
                if force_brake_only:
                    self.log.info(
                        "commit-hold: sequence fallback selecting brake '%s' for "
                        "the fixed brake hold regardless of verifier %.3f",
                        names[brake_idx],
                        brake_score,
                    )
                else:
                    self.log.info(
                        "commit-hold: all non-brake primitives failed; selecting brake "
                        "'%s': verifier %.3f (%s)", names[brake_idx], brake_score,
                        "pass" if brake_passed else "fixed-duration safety fallback",
                    )
                payload = {
                    "selected_plan": traj,
                    "selected_score": brake_score,
                    "selected_source": f"recovery:{names[brake_idx]}",
                    "fallback_primitive": names[brake_idx],
                    # A recovery-selection safety fallback is a real brake
                    # primitive, not a one-control-tick placeholder.  Commit it
                    # through the normal hold path so the system-owned brake
                    # duration (currently 1.0 s) is applied consistently even
                    # when the VLM returns no usable action or brake itself
                    # fails the PDMS threshold.
                    "fallback_mode": "hold",
                    "fallback_states": (
                        np.asarray(states[brake_idx], dtype=np.float32)
                        if brake_idx < len(states) else None
                    ),
                }
                return names[brake_idx], payload, brake_score
        finally:
            if prev_debug is not None:
                self.verifier._debug = prev_debug
        return None

    # --------------------------------------------------------------- recover
    def _recovery_selector_kind(self) -> str:
        """``'vlm'`` only when the agentic loop will actually consult a VLM;
        every other configuration (dummy selector, agentic loop disabled,
        selector unavailable -> argmax) is ``'dummy'`` and hard-excludes tried
        maneuvers instead of relying on the model to avoid them."""
        cfg = getattr(self.selector, "agenticdriving_cfg", None)
        if cfg is None or not getattr(cfg, "enabled", False):
            return "dummy"
        if getattr(self.selector, "vlm_selector", None) is None:
            return "dummy"
        return "vlm" if str(getattr(cfg, "selector", "vlm")).strip().lower() == "vlm" else "dummy"

    def _recover(
        self,
        *,
        obs: Dict[str, Any],
        info: Dict[str, Any],
        info_history: Sequence[Dict[str, Any]],
        privileged_info: Optional[List[Dict[str, Any]]],
        output_num_poses: int,
        allow_vlm_selection: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Regenerate with the rule-based planner and re-select (AutoAgent0's
        ``_build_recover_plan``). Defensive: failures degrade to ``None``."""
        cfg = self.recovery_cfg
        if cfg is None or not getattr(cfg, "enabled", False):
            self.log.warning("recovery requested but recovery is disabled; skipping")
            return None
        selector_kind = (
            self._recovery_selector_kind() if allow_vlm_selection else "dummy"
        )
        try:
            primitive_names: Optional[List[str]] = None
            primitive_states: Optional[List[np.ndarray]] = None
            # In-repo primitive ladder. VehicleSim integrates directly in the
            # CARLA ego-local frame.
            proposals_carla_local, scores, primitive_names, primitive_states = (
                get_primitive_proposals_and_scores(
                    info=info,
                    privileged_info=privileged_info,
                    output_num_poses=output_num_poses,
                    allowed_primitives=getattr(cfg, "allowed_primitives", ()),
                    lane_bypass_min_gap_m=getattr(
                        cfg, "lane_bypass_min_gap_m", 0.0
                    ),
                    lane_bypass_clearance_buffer_m=getattr(
                        cfg, "lane_bypass_clearance_buffer_m", 1.0
                    ),
                    lane_bypass_required_time_sec=getattr(
                        cfg, "lane_bypass_required_time_sec", 3.5
                    ),
                    lane_bypass_min_time_gap_sec=getattr(
                        cfg, "lane_bypass_min_time_gap_sec", 4.5
                    ),
                    obstacle_clearance_m=getattr(cfg, "obstacle_clearance_m", 8.0),
                )
            )
            # Apply the same hard HD-map gate before either recovery selector.
            # Slice every aligned array so rejected primitives cannot appear in
            # a VLM prompt or be resurrected by argmax fallback behavior.
            hd_map_rejected = self._footprint_offroad_batch(
                proposals_carla_local, info
            )
            if np.any(hd_map_rejected):
                rejected_names = [
                    primitive_names[idx]
                    for idx in np.flatnonzero(hd_map_rejected)
                ]
                keep = np.flatnonzero(~hd_map_rejected).tolist()
                self.log.warning(
                    "HD-map hard filter removed recovery primitives: %s",
                    rejected_names,
                )
                if not keep:
                    self.log.warning(
                        "HD-map hard filter rejected every recovery primitive"
                    )
                    return None
                proposals_carla_local = proposals_carla_local[keep]
                scores = np.asarray(scores)[keep]
                primitive_names = [primitive_names[idx] for idx in keep]
                primitive_states = [primitive_states[idx] for idx in keep]
            junction_heading_rejected = junction_route_heading_violation_mask(
                proposals_carla_local, info
            )
            if np.any(junction_heading_rejected):
                rejected_names = [
                    primitive_names[idx]
                    for idx in np.flatnonzero(junction_heading_rejected)
                ]
                keep = np.flatnonzero(~junction_heading_rejected).tolist()
                self.log.warning(
                    "junction route-heading filter removed recovery primitives: %s",
                    rejected_names,
                )
                if not keep:
                    return None
                proposals_carla_local = proposals_carla_local[keep]
                scores = np.asarray(scores)[keep]
                primitive_names = [primitive_names[idx] for idx in keep]
                primitive_states = [primitive_states[idx] for idx in keep]
            # Cross-tick memory: dummy/argmax paths hard-exclude maneuvers
            # already attempted this episode; the VLM path keeps the full
            # set and sees the attempt history as critique context instead.
            info = dict(info)
            history = self.recovery_tracker.describe()
            if history:
                info["recovery_attempt_history"] = history
            if selector_kind == "dummy":
                excluded = self.recovery_tracker.excluded()
                if excluded:
                    keep = [i for i, n in enumerate(primitive_names) if n not in excluded]
                    if not keep:
                        self.log.warning(
                            "recovery ladder exhausted (%s all failed); resetting memory",
                            sorted(excluded),
                        )
                        self.recovery_tracker.reset()
                        return None
                    proposals_carla_local = proposals_carla_local[keep]
                    scores = np.asarray(scores)[keep]
                    primitive_names = [primitive_names[i] for i in keep]
                    primitive_states = [primitive_states[i] for i in keep]
                    info["recovery_excluded_primitives"] = sorted(excluded)
            if scores is None or len(scores) == 0:
                return None
            # recovery selector
            payload = self.selector.select(
                proposals=proposals_carla_local,
                scores=np.asarray(scores, dtype=np.float32),
                obs=obs,
                info=info,
                info_history=list(info_history),
                privileged_info=privileged_info,
                strategy=None if allow_vlm_selection else "argmax",
            )
            selected_recovery_plan = np.asarray(
                payload.get("selected_plan"), dtype=np.float32
            )
            if self._footprint_offroad(selected_recovery_plan, info):
                # Defense in depth for selector-injected candidates such as a
                # carry-previous plan that was not in the primitive arrays.
                self.log.warning(
                    "HD-map hard filter rejected selector-produced recovery "
                    "trajectory"
                )
                return None
            if primitive_names:
                self._attach_primitive_execution(
                    payload, proposals_carla_local, primitive_names, primitive_states
                )
            # The selected primitive becomes the in-flight attempt; a later
            # tick's verifier verdict settles it (see step()). Its signed state
            # sequence is retained only so the controller can execute reverse
            # phases correctly; recovery failures are never physically undone.
            selected_primitive = payload.get("fallback_primitive")
            if selected_primitive:
                self.recovery_tracker.note_attempt(
                    str(selected_primitive),
                    frame_index=self.selector.frame_index,
                )
                self._record_failure_memory(
                    info,
                    event_type="recovery_committed",
                    problem=getattr(self, "_pending_recovery_reason", None),
                    method=str(selected_primitive),
                    outcome="legacy_one_tick_execution",
                    details={
                        "selector": selector_kind,
                        "parameters": dict(payload.get("fallback_parameters") or {}),
                    },
                )
            else:
                self._record_failure_memory(
                    info,
                    event_type="recovery_selection",
                    problem=getattr(self, "_pending_recovery_reason", None),
                    outcome="selector_returned_no_primitive",
                    details={"selector": selector_kind},
                )
            return payload
        except Exception as exc:
            self.log.exception("rule-based recovery failed")
            self._record_failure_memory(
                info,
                event_type="recovery_selection_error",
                problem=getattr(self, "_pending_recovery_reason", None),
                outcome="rule_based_recovery_failed",
                details={"error": str(exc)},
            )
            return None

    @staticmethod
    def _attach_primitive_execution(
        payload: Dict[str, Any],
        proposals: np.ndarray,
        names: List[str],
        states: List[np.ndarray],
    ) -> None:
        """Tag the payload with the selected maneuver's signed (speed, steer_rad)
        sequence so the adapter can drive reverse segments open-loop instead of
        through a forward-only tracker. Only applies when the selection traces
        back to one of the primitive proposals (the agentic loop may pick a
        carry-previous candidate instead, whose plan won't match)."""
        if payload.get("fallback_states") is not None:
            return
        idx = payload.get("selected_idx")
        if idx is None or not (0 <= int(idx) < len(names)):
            return
        idx = int(idx)
        plan = np.asarray(payload.get("selected_plan"), dtype=np.float32)
        candidate = np.asarray(proposals[idx][: len(plan)], dtype=np.float32)
        if plan.shape != candidate.shape or not np.allclose(plan, candidate, atol=1e-3):
            return
        payload["fallback_states"] = np.asarray(states[idx], dtype=np.float32)
        payload["fallback_primitive"] = str(names[idx])
