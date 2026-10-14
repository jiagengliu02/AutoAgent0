"""Agentic recovery-loop trajectory selection strategy."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from agenticdriving.scorer.agent_schemas import DesignChangeRequest, FrameUncertainty
from agenticdriving.scorer.agent_trace import OrchestratorToolLog, attach_recovery_trace
from agenticdriving.scorer.selection_strategies.base import (
    BaseSelectionStrategy,
    LearnedPlannerSelection,
    SelectionOutcome,
)
from agenticdriving.scorer.uncertainty import (
    ROUTING_ZONE_LEAN_RULE_BASED,
    ROUTING_ZONE_NORMAL,
    ROUTING_ZONE_RULE_BASED_FALLBACK,
    classify_routing_zone,
    compute_cross_family_disagreement,
    compute_intra_learned_disagreement,
    compute_mode_count,
)


PHASE_DEFAULT_ACCEPTED = "default_accepted"
PHASE_DEFAULT_REJECTED = "default_rejected"
PHASE_REDESIGN_REJECTED_RETRY = "redesign_rejected_retry"
PHASE_REDESIGN_ACCEPTED = "redesign_accepted"
PHASE_REDESIGN_SELECTED_AT_LIMIT = "redesign_selected_at_critic_limit"


@dataclass(frozen=True)
class ExpandedDesign:
    """Expanded candidate pool requested after an AgenticDriving critique rejection."""

    rows: List[Dict[str, object]]
    learned_budget: int


@dataclass(frozen=True)
class FinalRecoveryDecision:
    """Orchestrator decision after the revised candidate critique."""

    final_rejected: bool
    continue_redesign: bool
    use_fallback: bool
    phase: str
    fallback_reason: Optional[str]


class RecoveryStrategy(BaseSelectionStrategy):
    """Agentic critique/redesign loop path."""

    def select(
        self,
        *,
        proposals: np.ndarray,
        scores: np.ndarray,
        output_num_poses: int,
        obs: Dict[str, Any],
        info: Dict[str, Any],
        info_history: Sequence[Dict[str, Any]],
        privileged_info: Optional[Any] = None,
    ) -> SelectionOutcome:
        sel = self.selector
        selector_name = str(
            getattr(sel.agenticdriving_cfg, "selector", "vlm")
        ).strip().lower()
        critic_name = (
            "agenticdriving_vlm_critic"
            if selector_name == "vlm"
            else "agenticdriving_dummy_critic"
        )
        scorer_name = (
            "agenticdriving_vlm_planner"
            if selector_name == "vlm"
            else "agenticdriving_dummy_selector"
        )
        reserved_candidate_slots = (
            max(0, int(sel.recovery_cfg.topk))
            if sel.recovery_cfg.enabled and not sel.vlm_cfg.planner_gate_enabled
            else 0
        )
        learned_candidate_rows, allow_carry_previous = sel._build_learned_candidate_rows(
            proposals, scores, output_num_poses, info, reserved_candidate_slots,
        )
        rule_based_candidate_rows = sel._build_rule_based_candidate_rows(
            obs, info, info_history, privileged_info, output_num_poses,
        )

        tool_log = OrchestratorToolLog(runtime_name=sel.runtime_name)
        tool_log.reset()
        tool_log.record(
            "request_initial_design",
            designer="learned",
            candidate_count=len(learned_candidate_rows),
        )
        tool_log.record(
            "request_designer",
            learned_candidate_count=len(learned_candidate_rows),
            rule_based_candidate_count=0,
            combined_candidate_count=0,
        )
        default_selected_index, default_selected_source = self._default_selection_for_family(
            candidate_rows=learned_candidate_rows,
            selected_planner="learned",
            scores=scores,
            learned_source_name=sel.current_source_name,
            learned_default_source=sel.learned_default_source,
            strict_learned_argmax_lookup=sel.strict_learned_argmax_lookup,
        )
        default_row = dict(learned_candidate_rows[default_selected_index])
        default_plan = np.asarray(default_row.get("execution_plan", default_row["local_plan"]), dtype=np.float32)
        default_idx = default_row.get("proposal_index")
        default_score, default_score_raw = self._score_from_row(default_row, sel.score_fallback_key)

        frame_uncertainty = self._maybe_compute_frame_uncertainty(
            learned_candidate_rows=learned_candidate_rows,
            rule_based_candidate_rows=rule_based_candidate_rows,
            default_row=default_row,
        )
        if frame_uncertainty is not None:
            tool_log.record(
                "compute_uncertainty",
                intra_learned_m=frame_uncertainty.intra_learned_m,
                cross_family_m=frame_uncertainty.cross_family_m,
                mode_count=frame_uncertainty.mode_count,
                routing_zone=frame_uncertainty.routing_zone,
            )

        tool_log.record(
            "request_critique",
            phase="default",
            critic=critic_name,
            candidate_count=1,
        )
        # Cross-tick recovery memory (orchestrator): past attempted maneuvers
        # and their outcomes, surfaced to the VLM as critique context.
        attempt_history = info.get("recovery_attempt_history")
        critique_result = sel.vlm_selector.critique_agenticdriving_candidate(
            frame_index=sel.frame_index,
            camera_images=obs["rgb"],
            info=info,
            candidate_row=default_row,
            stage="default",
            previous_feedback=attempt_history,
        )
        should_redesign = self._critique_requests_redesign(critique_result)

        if not should_redesign:
            tool_log.record(
                "select_final_actions",
                phase=PHASE_DEFAULT_ACCEPTED,
                selected_source=default_selected_source,
                selected_candidate_index=default_selected_index,
            )
            selection_debug = dict(critique_result)
            selection_debug.update(
                {
                    "agenticdriving_mode": "recovery_loop",
                    "agenticdriving_selector": selector_name,
                    "agenticdriving_critic": critic_name,
                    "agenticdriving_scorer": scorer_name,
                    "agenticdriving_phase": PHASE_DEFAULT_ACCEPTED,
                    "agenticdriving_redesign_triggered": False,
                    "agenticdriving_default_critique": critique_result,
                    "agenticdriving_fallback_reason": None,
                    "fallback_selected_idx": int(default_selected_index),
                    "fallback_selected_source": default_selected_source,
                    "planner_gate_selected_planner": "learned",
                    "agenticdriving_frame_uncertainty": self._frame_uncertainty_debug(frame_uncertainty),
                }
            )
            selection = LearnedPlannerSelection(
                selected_row=default_row,
                selected_plan=default_plan,
                selected_idx=None if default_idx is None else int(default_idx),
                selected_score=default_score,
                selected_score_raw=default_score_raw,
                selected_source=default_selected_source,
                selected_planner="learned",
                candidate_rows=[default_row],
                planner_gate_result=self._disabled_planner_gate_result(),
                default_selected_index=0,
                default_selected_source=default_selected_source,
                selection_debug=selection_debug,
            )
            selection = replace(selection, selection_debug=attach_recovery_trace(
                runtime_name=sel.runtime_name,
                frame_index=sel.frame_index,
                info=info,
                learned_candidate_rows=learned_candidate_rows,
                rule_based_candidate_rows=rule_based_candidate_rows,
                selection=selection,
                phase=PHASE_DEFAULT_ACCEPTED,
                tool_log=tool_log,
                logger=sel.logger,
            ))
            return self._to_outcome(
                proposals=proposals,
                scores=scores,
                output_num_poses=output_num_poses,
                selection=selection,
            )

        design_change_request = self._build_design_change_request(
            critique_result,
            redesign_candidate_budget=sel.agenticdriving_cfg.redesign_candidate_budget,
            available_rule_based_count=len(rule_based_candidate_rows),
            has_rule_based_candidates=bool(rule_based_candidate_rows),
            frame_uncertainty=frame_uncertainty,
        )
        tool_log.record(
            "request_design_change",
            phase=PHASE_DEFAULT_REJECTED,
            reason=design_change_request.reason,
            corrective_action=design_change_request.corrective_action,
            candidate_budget=design_change_request.candidate_budget,
            learned_budget=design_change_request.learned_budget,
            rule_based_budget=design_change_request.rule_based_budget,
            allocation_strategy=design_change_request.allocation_strategy,
            include_rule_based=design_change_request.include_rule_based,
        )
        expanded_design = self._build_expanded_design(
            learned_candidate_rows=learned_candidate_rows,
            rule_based_candidate_rows=rule_based_candidate_rows,
            default_row=default_row,
            design_change_request=design_change_request,
        )
        expanded_rows = expanded_design.rows
        tool_log.record(
            "request_revised_design",
            designer="learned+rule_based",
            candidate_count=len(expanded_rows),
            candidate_budget=design_change_request.candidate_budget,
            learned_budget=design_change_request.learned_budget,
            rule_based_budget=design_change_request.rule_based_budget,
            allocation_strategy=design_change_request.allocation_strategy,
        )
        tool_log.record(
            "request_designer",
            learned_candidate_count=len(learned_candidate_rows[:expanded_design.learned_budget]),
            rule_based_candidate_count=len(rule_based_candidate_rows),
            combined_candidate_count=len(expanded_rows),
        )
        max_attempts = max(1, int(sel.agenticdriving_cfg.max_redesign_attempts))
        previous_feedback = design_change_request.reason
        if attempt_history:
            previous_feedback = f"{attempt_history} {previous_feedback}"
        attempt_records = []
        redesign_result = None
        final_critique = None
        final_decision = None
        revised_row = default_row
        revised_plan = default_plan
        revised_idx = default_idx
        revised_score = default_score
        revised_score_raw = default_score_raw

        for attempt_index in range(1, max_attempts + 1):
            attempt_stage = f"redesign_attempt_{attempt_index}"
            tool_log.record(
                "select_final_actions",
                phase=attempt_stage,
                scorer=scorer_name,
                candidate_count=len(expanded_rows),
                attempt_index=attempt_index,
                max_attempts=max_attempts,
            )
            redesign_result = sel.vlm_selector.score_agenticdriving_candidates(
                frame_index=sel.frame_index,
                camera_images=obs["rgb"],
                info=info,
                candidate_rows=expanded_rows,
                default_selected_index=0,
                default_selected_source="agenticdriving_redesign_default",
                critique_reason=previous_feedback,
                corrective_action=design_change_request.corrective_action,
                stage=attempt_stage,
            )
            revised_row = dict(redesign_result["selected_candidate_row"])
            revised_plan = np.asarray(revised_row.get("execution_plan", revised_row["local_plan"]), dtype=np.float32)
            revised_idx = revised_row.get("proposal_index")
            revised_score, revised_score_raw = self._score_from_row(revised_row, sel.score_fallback_key)

            tool_log.record(
                "request_critique",
                phase=attempt_stage,
                critic=critic_name,
                candidate_count=1,
                attempt_index=attempt_index,
                max_attempts=max_attempts,
            )
            final_critique = sel.vlm_selector.critique_agenticdriving_candidate(
                frame_index=sel.frame_index,
                camera_images=obs["rgb"],
                info=info,
                candidate_row=revised_row,
                stage=attempt_stage,
                previous_feedback=previous_feedback,
            )
            final_decision = self._decide_final_recovery_action(
                final_critique,
                attempt_index=attempt_index,
                max_redesign_attempts=max_attempts,
            )
            attempt_records.append(
                {
                    "attempt_index": attempt_index,
                    "phase": final_decision.phase,
                    "selected_source": redesign_result.get("selected_source"),
                    "selected_candidate_index": redesign_result.get("vlm_candidate_index"),
                    "selected_candidate_source": revised_row.get("source"),
                    "selected_proposal_index": revised_row.get("proposal_index"),
                    "critique_rejected": final_decision.final_rejected,
                    "critique_reasoning": final_critique.get("agenticdriving_critique_reasoning"),
                    "fallback_reason": final_decision.fallback_reason,
                }
            )
            if not final_decision.continue_redesign:
                break
            previous_feedback = (
                final_critique.get("agenticdriving_critique_reasoning")
                or final_decision.fallback_reason
                or previous_feedback
            )

        if redesign_result is None or final_critique is None or final_decision is None:
            raise RuntimeError("AgenticDriving redesign loop did not produce a revised candidate")

        selected_row = revised_row
        selected_plan = revised_plan
        selected_idx = None if revised_idx is None else int(revised_idx)
        selected_score = revised_score
        selected_score_raw = revised_score_raw
        selected_source = str(redesign_result.get("selected_source", "agenticdriving_redesign_selected"))

        tool_log.record(
            "emit_final_actions",
            phase=final_decision.phase,
            selected_source=selected_source,
            selected_candidate_index=selected_idx,
            redesign_attempt_count=len(attempt_records),
        )
        actual_learned_count = sum(
            1 for row in expanded_rows if str(row.get("source", "")).startswith(sel.current_source_name)
        )
        actual_rule_based_count = sum(
            1 for row in expanded_rows if str(row.get("source", "")) == "rule_based"
        )
        missing_rule_based_count = max(0, int(design_change_request.rule_based_budget) - actual_rule_based_count)
        selection_debug = dict(redesign_result)
        selection_debug.update(
            {
                "agenticdriving_mode": "recovery_loop",
                "agenticdriving_selector": selector_name,
                "agenticdriving_critic": critic_name,
                "agenticdriving_scorer": scorer_name,
                "agenticdriving_phase": final_decision.phase,
                "agenticdriving_redesign_triggered": True,
                "agenticdriving_default_critique": critique_result,
                "agenticdriving_design_change_request": {
                    "reason": design_change_request.reason,
                    "corrective_action": design_change_request.corrective_action,
                    "candidate_budget": design_change_request.candidate_budget,
                    "learned_budget": design_change_request.learned_budget,
                    "rule_based_budget": design_change_request.rule_based_budget,
                    "allocation_strategy": design_change_request.allocation_strategy,
                    "include_learned": design_change_request.include_learned,
                    "include_rule_based": design_change_request.include_rule_based,
                    "routing_mode": design_change_request.routing_mode,
                },
                "agenticdriving_redesign_request": {
                    "reason": design_change_request.reason,
                    "corrective_action": design_change_request.corrective_action,
                    "candidate_budget": design_change_request.candidate_budget,
                    "learned_budget": design_change_request.learned_budget,
                    "rule_based_budget": design_change_request.rule_based_budget,
                    "allocation_strategy": design_change_request.allocation_strategy,
                    "include_learned": design_change_request.include_learned,
                    "include_rule_based": design_change_request.include_rule_based,
                    "routing_mode": design_change_request.routing_mode,
                },
                "agenticdriving_frame_uncertainty": self._frame_uncertainty_debug(frame_uncertainty),
                "agenticdriving_revised_candidate_count": len(expanded_rows),
                "agenticdriving_requested_learned_candidate_count": design_change_request.learned_budget,
                "agenticdriving_requested_rule_based_candidate_count": design_change_request.rule_based_budget,
                "agenticdriving_revised_learned_candidate_count": actual_learned_count,
                "agenticdriving_revised_rule_based_candidate_count": actual_rule_based_count,
                "agenticdriving_missing_rule_based_candidate_count": missing_rule_based_count,
                "agenticdriving_rule_based_budget_satisfied": missing_rule_based_count == 0,
                "agenticdriving_final_critique": final_critique,
                "agenticdriving_fallback_reason": final_decision.fallback_reason,
                "agenticdriving_redesign_attempt_count": len(attempt_records),
                "agenticdriving_redesign_attempts": attempt_records,
                "agenticdriving_max_redesign_attempts": int(sel.agenticdriving_cfg.max_redesign_attempts),
                "fallback_selected_idx": 0,
                "fallback_selected_source": "agenticdriving_redesign_default",
                "planner_gate_selected_planner": "agenticdriving",
            }
        )
        selection = LearnedPlannerSelection(
            selected_row=selected_row,
            selected_plan=selected_plan,
            selected_idx=selected_idx,
            selected_score=float(selected_score),
            selected_score_raw=float(selected_score_raw),
            selected_source=selected_source,
            selected_planner="agenticdriving",
            candidate_rows=list(expanded_rows),
            planner_gate_result=self._disabled_planner_gate_result(),
            default_selected_index=0,
            default_selected_source="agenticdriving_redesign_default",
            selection_debug=selection_debug,
        )
        selection = replace(selection, selection_debug=attach_recovery_trace(
            runtime_name=sel.runtime_name,
            frame_index=sel.frame_index,
            info=info,
            learned_candidate_rows=learned_candidate_rows,
            rule_based_candidate_rows=rule_based_candidate_rows,
            selection=selection,
            phase=final_decision.phase,
            tool_log=tool_log,
            logger=sel.logger,
        ))
        return self._to_outcome(
            proposals=proposals,
            scores=scores,
            output_num_poses=output_num_poses,
            selection=selection,
        )

    def _to_outcome(
        self,
        *,
        proposals: np.ndarray,
        scores: np.ndarray,
        output_num_poses: int,
        selection: LearnedPlannerSelection,
    ) -> SelectionOutcome:
        sel = self.selector
        plan_payload = sel._build_plan_payload(
            proposals,
            scores,
            output_num_poses,
            selected_idx=None if selection.selected_idx is None else int(selection.selected_idx),
            selected_source=selection.selected_source,
            selection_debug=selection.selection_debug,
            selected_plan_override=selection.selected_plan,
            selected_score_override=selection.selected_score,
            candidate_pool_rows=selection.candidate_rows,
        )
        # Primitive-backed rows carry the signed (speed, steer_rad) sequence;
        # forward it so reverse maneuvers can be executed open-loop.
        execution_states = selection.selected_row.get("execution_states")
        if execution_states is not None:
            plan_payload["fallback_states"] = np.asarray(
                execution_states,
                dtype=np.float32,
            )
            plan_payload["fallback_primitive"] = selection.selected_row.get(
                "primitive_name"
            )
        return SelectionOutcome(
            plan_payload=plan_payload,
            selected_plan=selection.selected_plan,
            selected_score_raw=selection.selected_score_raw,
            selected_source=str(selection.selected_row.get("source", "vlm_selected")),
            advance_frame=True,
        )

    @staticmethod
    def _score_from_row(selected_row: Dict[str, object], fallback_key: str) -> tuple[float, float]:
        selected_score_value = selected_row.get("proposal_score")
        if selected_score_value is None:
            selected_score_value = selected_row.get(fallback_key, 0.0)
        selected_score = float(selected_score_value)
        selected_score_raw_value = selected_row.get("origin_selected_score_raw")
        if selected_score_raw_value is None:
            selected_score_raw_value = selected_score
        return selected_score, float(selected_score_raw_value)

    @staticmethod
    def _critique_requests_redesign(critique_result: Dict[str, object]) -> bool:
        should_redesign = critique_result.get("agenticdriving_critique_rejected") is True
        critique_error = critique_result.get("agenticdriving_critique_error") or critique_result.get("error")
        if critique_error is not None:
            return False
        return bool(should_redesign)

    def _build_design_change_request(
        self,
        critique_result: Dict[str, object],
        *,
        redesign_candidate_budget: int,
        available_rule_based_count: int,
        has_rule_based_candidates: bool,
        frame_uncertainty: Optional[FrameUncertainty] = None,
    ) -> DesignChangeRequest:
        corrective_action = critique_result.get("agenticdriving_critique_corrective_action")
        rejection_reason = critique_result.get("agenticdriving_critique_reasoning") or "vlm_critic_requested_redesign"
        candidate_budget = max(1, int(redesign_candidate_budget))
        learned_budget = 8
        rule_based_budget = 5 if has_rule_based_candidates else 0
        allocation_strategy = "learned8_rule5_static_v1"
        routing_mode = ROUTING_ZONE_NORMAL

        if frame_uncertainty is not None and has_rule_based_candidates:
            zone = frame_uncertainty.routing_zone
            if zone == ROUTING_ZONE_RULE_BASED_FALLBACK:
                learned_budget = 0
                rule_based_budget = candidate_budget
                allocation_strategy = (
                    f"rule_based_only_uncertainty_intra{frame_uncertainty.intra_learned_m:.2f}"
                    f"_cross{frame_uncertainty.cross_family_m:.2f}"
                    f"_modes{frame_uncertainty.mode_count}"
                )
                routing_mode = ROUTING_ZONE_RULE_BASED_FALLBACK
            elif zone == ROUTING_ZONE_LEAN_RULE_BASED:
                shift = max(1, learned_budget // 2)
                learned_budget = max(0, learned_budget - shift)
                rule_based_budget = max(rule_based_budget, candidate_budget - learned_budget)
                allocation_strategy = f"{allocation_strategy}_lean_rule_based"
                routing_mode = ROUTING_ZONE_LEAN_RULE_BASED

        if not has_rule_based_candidates:
            rule_based_budget = 0
            allocation_strategy = f"{allocation_strategy}_no_rule_based_available"

        return DesignChangeRequest(
            reason=str(rejection_reason),
            corrective_action=None if corrective_action is None else str(corrective_action),
            candidate_budget=candidate_budget,
            learned_budget=learned_budget,
            rule_based_budget=rule_based_budget,
            allocation_strategy=allocation_strategy,
            include_learned=True,
            include_rule_based=bool(has_rule_based_candidates and int(available_rule_based_count) > 0),
            routing_mode=routing_mode,
        )

    def _maybe_compute_frame_uncertainty(
        self,
        *,
        learned_candidate_rows: Sequence[Dict[str, object]],
        rule_based_candidate_rows: Sequence[Dict[str, object]],
        default_row: Dict[str, object],
    ) -> Optional[FrameUncertainty]:
        cfg = self.selector.agenticdriving_cfg
        if not getattr(cfg, "uncertainty_enabled", False):
            return None

        horizon = int(getattr(cfg, "uncertainty_horizon_steps", 0)) or None
        intra = compute_intra_learned_disagreement(
            learned_candidate_rows,
            horizon_steps=horizon,
        )
        cross = compute_cross_family_disagreement(
            default_row,
            rule_based_candidate_rows,
            horizon_steps=horizon,
        )
        modes = compute_mode_count(
            learned_candidate_rows,
            k_max=int(cfg.uncertainty_mode_k_max),
            horizon_steps=horizon,
            backend=str(cfg.uncertainty_kmeans_backend),
        )
        zone = classify_routing_zone(
            intra["disagreement_m"],
            cross["disagreement_m"],
            modes["mode_count"],
            t_intra=float(cfg.uncertainty_t_intra),
            t_cross=float(cfg.uncertainty_t_cross),
            mode_count_high=int(cfg.uncertainty_mode_count_high),
        )
        return FrameUncertainty(
            intra_learned_m=float(intra["disagreement_m"]),
            cross_family_m=float(cross["disagreement_m"]),
            mode_count=int(modes["mode_count"]),
            routing_zone=zone,
            metadata={
                "intra": intra,
                "cross": cross,
                "modes": modes,
                "thresholds": {
                    "t_intra": float(cfg.uncertainty_t_intra),
                    "t_cross": float(cfg.uncertainty_t_cross),
                    "mode_count_high": int(cfg.uncertainty_mode_count_high),
                },
            },
        )

    @staticmethod
    def _frame_uncertainty_debug(frame_uncertainty: Optional[FrameUncertainty]) -> Optional[Dict[str, Any]]:
        if frame_uncertainty is None:
            return None
        return {
            "intra_learned_m": frame_uncertainty.intra_learned_m,
            "cross_family_m": frame_uncertainty.cross_family_m,
            "mode_count": frame_uncertainty.mode_count,
            "routing_zone": frame_uncertainty.routing_zone,
            "metadata": frame_uncertainty.metadata,
        }

    @staticmethod
    def _build_expanded_design(
        *,
        learned_candidate_rows: Sequence[Dict[str, object]],
        rule_based_candidate_rows: Sequence[Dict[str, object]],
        default_row: Dict[str, object],
        design_change_request: DesignChangeRequest,
    ) -> ExpandedDesign:
        learned_budget = max(1, int(design_change_request.learned_budget))
        rule_based_budget = max(0, int(design_change_request.rule_based_budget))
        expanded_rows = list(learned_candidate_rows[:learned_budget]) + list(rule_based_candidate_rows[:rule_based_budget])
        if not expanded_rows:
            expanded_rows = [default_row]
        return ExpandedDesign(rows=expanded_rows, learned_budget=learned_budget)

    @staticmethod
    def _decide_final_recovery_action(
        final_critique: Dict[str, object],
        *,
        attempt_index: int,
        max_redesign_attempts: int,
    ) -> FinalRecoveryDecision:
        final_rejected = final_critique.get("agenticdriving_critique_rejected") is True
        final_critique_error = final_critique.get("agenticdriving_critique_error") or final_critique.get("error")
        if final_critique_error is not None:
            final_rejected = False

        reached_redesign_limit = int(attempt_index) >= max(1, int(max_redesign_attempts))
        if final_rejected and not reached_redesign_limit:
            fallback_reason = (
                final_critique.get("agenticdriving_critique_reasoning")
                or "final_critic_rejected_revised_candidate"
            )
            return FinalRecoveryDecision(
                final_rejected=True,
                continue_redesign=True,
                use_fallback=False,
                phase=PHASE_REDESIGN_REJECTED_RETRY,
                fallback_reason=str(fallback_reason),
            )

        if final_rejected:
            return FinalRecoveryDecision(
                final_rejected=True,
                continue_redesign=False,
                use_fallback=False,
                phase=PHASE_REDESIGN_SELECTED_AT_LIMIT,
                fallback_reason="final_critic_rejected_but_max_redesign_attempts_reached_use_vlm_scorer_selection",
            )

        return FinalRecoveryDecision(
            final_rejected=False,
            continue_redesign=False,
            use_fallback=False,
            phase=PHASE_REDESIGN_ACCEPTED,
            fallback_reason=None,
        )
