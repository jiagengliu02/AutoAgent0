"""Debug/trace assembly for the AgenticDriving selection flows.

Consolidates everything that builds the ``agent_trace`` debug payload: the
per-frame tool-call log, the passive-verifier result, and the trace builders
shared by the scorer flow and the recovery flow.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence

from agenticdriving.scorer.agent_schemas import (
    AgentStepTrace,
    OrchestratorDecision,
    SceneContext,
    VerifierResult,
)


TRACE_SCHEMA_VERSION = "agenticdriving.trace.v1"

# The verifier is passive in phase 1: it always accepts and never alters control.
# Kept as a constant so the trace's ``verifier`` field is produced without a class.
PASSIVE_VERIFIER_RESULT = VerifierResult(
    accepted=True,
    mode="passive",
    rejection_reason=None,
    checks={},
)


class OrchestratorToolLog:
    """Per-frame SceneSmith-style tool-call trace recorder."""

    def __init__(self, *, runtime_name: str):
        self.runtime_name = str(runtime_name)
        self._tool_calls: List[Dict[str, object]] = []

    def reset(self) -> None:
        self._tool_calls = []

    def record(self, name: str, **metadata: object) -> None:
        self._tool_calls.append(
            {
                "name": name,
                "runtime": self.runtime_name,
                **metadata,
            }
        )

    def to_debug_list(self) -> List[Dict[str, object]]:
        return list(self._tool_calls)


def summarize_sources(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    return dict(Counter(str(row.get("source") or "unknown") for row in rows))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def route_instruction_from_info(info: Dict[str, object]) -> str:
    for key in ("task_instruction", "route_instruction", "command"):
        value = info.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def first_present(*values: object) -> object:
    for value in values:
        if value is not None:
            return value
    return None


def optional_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None




def build_agent_trace(
    *,
    frame_index: int,
    route_instruction: str,
    info: Optional[Dict[str, Any]] = None,
    candidate_rows: Sequence[Dict[str, Any]] = (),
    learned_candidate_rows: Sequence[Dict[str, Any]] = (),
    rule_based_candidate_rows: Sequence[Dict[str, Any]] = (),
    decision_type: str,
    selected_source: Optional[str] = None,
    selected_planner: Optional[str] = None,
    selected_candidate_index: Optional[int] = None,
    confidence: Optional[float] = None,
    reasoning: Optional[str] = None,
    previous_verifier_feedback: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    info = info or {}
    all_rows = list(candidate_rows) or list(learned_candidate_rows) + list(rule_based_candidate_rows)
    scene = SceneContext(
        frame_index=int(frame_index),
        route_instruction=str(route_instruction),
        timestamp=_safe_float(info.get("timestamp", 0.0)),
        task_instruction=info.get("task_instruction") if isinstance(info.get("task_instruction"), str) else None,
        metadata={
            "command": info.get("command"),
            "task_type": info.get("task_type"),
        },
    )
    designer = {
        "candidate_count": len(all_rows),
        "learned_candidate_count": len(learned_candidate_rows),
        "rule_based_candidate_count": len(rule_based_candidate_rows),
        "candidate_sources": summarize_sources(all_rows),
    }
    decision = OrchestratorDecision(
        decision_type=decision_type,
        selected_source=selected_source,
        selected_planner=selected_planner,
        selected_candidate_index=selected_candidate_index,
        confidence=confidence,
        reasoning=reasoning,
    )
    trace = AgentStepTrace(
        schema_version=TRACE_SCHEMA_VERSION,
        scene=scene,
        designer=designer,
        orchestrator=decision,
        verifier=PASSIVE_VERIFIER_RESULT,
        previous_verifier_feedback=previous_verifier_feedback,
    )
    return trace.to_debug_dict()




def attach_recovery_trace(
    *,
    runtime_name: str,
    frame_index: int,
    info: Dict[str, object],
    learned_candidate_rows: Sequence[Dict[str, object]],
    rule_based_candidate_rows: Sequence[Dict[str, object]],
    selection: Any,
    phase: str,
    tool_log: OrchestratorToolLog,
    logger: Any = None,
    previous_verifier_feedback: Optional[Dict[str, Any]] = None,
) -> Dict[str, object]:
    selection_debug = dict(selection.selection_debug)
    try:
        confidence = first_present(
            selection_debug.get("vlm_confidence"),
            selection_debug.get("intervention_confidence"),
        )
        reasoning = first_present(
            selection_debug.get("selected_path_reasoning"),
            selection_debug.get("vlm_reasoning"),
            selection_debug.get("intervention_reasoning"),
        )
        trace = build_agent_trace(
            frame_index=frame_index,
            route_instruction=route_instruction_from_info(info),
            info=info,
            candidate_rows=selection.candidate_rows,
            learned_candidate_rows=learned_candidate_rows,
            rule_based_candidate_rows=rule_based_candidate_rows,
            decision_type="agenticdriving_recovery_loop",
            selected_source=selection.selected_source,
            selected_planner=selection.selected_planner,
            selected_candidate_index=selection.selected_idx,
            confidence=optional_float(confidence),
            reasoning=None if reasoning is None else str(reasoning),
            previous_verifier_feedback=previous_verifier_feedback or {},
        )
        trace["runtime"] = {
            "name": runtime_name,
            "behavior_mode": "agentic_recovery_loop",
            "scene_smith_style": True,
            "phase": phase,
        }
        trace["tool_calls"] = tool_log.to_debug_list()
        trace["critique"] = {
            "critic": selection_debug.get(
                "agenticdriving_critic",
                "agenticdriving_vlm_critic",
            ),
            "default": selection_debug.get("agenticdriving_default_critique"),
            "final": selection_debug.get("agenticdriving_final_critique"),
        }
        trace["design_change_request"] = selection_debug.get("agenticdriving_design_change_request")
        trace["redesign_request"] = selection_debug.get("agenticdriving_redesign_request")
        trace["redesign_attempt_count"] = selection_debug.get("agenticdriving_redesign_attempt_count")
        trace["redesign_attempts"] = selection_debug.get("agenticdriving_redesign_attempts")
        trace["fallback_reason"] = selection_debug.get("agenticdriving_fallback_reason")
        selection_debug["agent_trace"] = trace
    except Exception as exc:
        if logger is not None:
            logger.exception("Failed to attach AgenticDriving recovery trace: %s", exc)
        selection_debug["agent_trace_error"] = str(exc)
    return selection_debug
