from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class CritiqueResult:
    """Structured result parsed from the active AgenticDriving critic."""

    accepted: bool
    severity_score: float
    corrective_action: str
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    error: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


def try_parse_json(text: str) -> Optional[Dict[str, object]]:
    raw = str(text).strip()
    try:
        return json.loads(raw)
    except Exception:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, flags=re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except Exception:
            pass

    blob = re.search(r"(\{.*\})", raw, flags=re.DOTALL)
    if blob:
        try:
            return json.loads(blob.group(1))
        except Exception:
            pass
    return None


def empty_token_usage() -> Dict[str, int]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def normalize_token_usage(token_usage: Optional[object]) -> Dict[str, int]:
    if not isinstance(token_usage, dict):
        return empty_token_usage()
    prompt_tokens = int(token_usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(token_usage.get("completion_tokens", 0) or 0)
    total_tokens = int(token_usage.get("total_tokens", prompt_tokens + completion_tokens) or 0)
    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "prompt_tokens": max(0, prompt_tokens),
        "completion_tokens": max(0, completion_tokens),
        "total_tokens": max(0, total_tokens),
    }


@dataclass(frozen=True)
class StopSignParseResult:
    stop_sign_visible: bool = False
    applies_to_ego_lane: bool = False
    must_stop: bool = False
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    error: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


def coerce_stop_sign_result(parsed: object) -> StopSignParseResult:
    """Validate the dedicated stop-sign classifier's small JSON contract."""
    if not isinstance(parsed, dict):
        return StopSignParseResult(error="stop_sign_output_invalid")

    visible = parsed.get("stop_sign_visible")
    applies = parsed.get("applies_to_ego_lane")
    must_stop = parsed.get("must_stop")
    if not isinstance(visible, bool):
        return StopSignParseResult(error="stop_sign_visible_missing", raw=dict(parsed))
    if not isinstance(applies, bool):
        return StopSignParseResult(
            stop_sign_visible=visible,
            error="stop_sign_applies_missing",
            raw=dict(parsed),
        )
    if not isinstance(must_stop, bool):
        return StopSignParseResult(
            stop_sign_visible=visible,
            applies_to_ego_lane=applies,
            error="stop_sign_must_stop_missing",
            raw=dict(parsed),
        )

    try:
        confidence = float(parsed.get("confidence"))
    except (TypeError, ValueError):
        return StopSignParseResult(
            stop_sign_visible=visible,
            applies_to_ego_lane=applies,
            must_stop=False,
            error="stop_sign_confidence_invalid",
            raw=dict(parsed),
        )
    if not 0.0 <= confidence <= 1.0:
        return StopSignParseResult(
            stop_sign_visible=visible,
            applies_to_ego_lane=applies,
            must_stop=False,
            error=f"stop_sign_confidence_out_of_range:{confidence}",
            raw=dict(parsed),
        )

    reasoning = parsed.get("reasoning")
    if reasoning is not None and not isinstance(reasoning, str):
        reasoning = str(reasoning)
    # Enforce the prompt's invariant locally instead of trusting inconsistent
    # JSON such as visible=false, must_stop=true.
    validated_must_stop = bool(must_stop and visible and applies)
    return StopSignParseResult(
        stop_sign_visible=visible,
        applies_to_ego_lane=applies,
        must_stop=validated_must_stop,
        confidence=confidence,
        reasoning=reasoning,
        raw=dict(parsed),
    )


def normalize_corrective_action(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    direct = {
        "left": "left",
        "right": "right",
        "straight": "straight",
        "forward": "straight",
        "go straight": "straight",
        "keep straight": "straight",
        "continue straight": "straight",
        "turn left": "left",
        "go left": "left",
        "veer left": "left",
        "turn right": "right",
        "go right": "right",
        "veer right": "right",
    }
    if text in direct:
        return direct[text]
    if "left" in text:
        return "left"
    if "right" in text:
        return "right"
    if "straight" in text or "forward" in text or "continue" in text:
        return "straight"
    return None


def coerce_candidate_scores(raw_scores: object, num_candidates: int) -> Tuple[Optional[List[float]], Optional[str]]:
    if not isinstance(raw_scores, dict):
        return None, "candidate_scores_missing"

    scores: List[Optional[float]] = [None] * num_candidates
    for raw_key, raw_value in raw_scores.items():
        try:
            idx = int(raw_key)
        except Exception:
            return None, f"candidate_score_bad_key:{raw_key}"
        if idx < 0 or idx >= num_candidates:
            return None, f"candidate_score_key_out_of_range:{idx}"
        try:
            scores[idx] = float(raw_value)
        except Exception:
            return None, f"candidate_score_bad_value:{raw_key}"

    if any(score is None for score in scores):
        missing = [idx for idx, score in enumerate(scores) if score is None]
        return None, f"candidate_scores_incomplete:{missing}"
    return [float(score) for score in scores], None


def coerce_vlm_candidate_scores(
    raw_scores: object,
    num_candidates: int,
) -> Tuple[Optional[List[float]], Optional[str]]:
    return coerce_candidate_scores(raw_scores, num_candidates)


def intervention_severity_band(
    severity_score: Optional[float],
    *,
    action_threshold: float,
    high_threshold: float,
) -> Optional[str]:
    if severity_score is None:
        return None
    if severity_score >= high_threshold:
        return "high"
    if severity_score >= action_threshold:
        return "medium"
    return "low"


def coerce_intervention_decision(
    parsed: object,
) -> Tuple[Optional[bool], Optional[float], Optional[str], Optional[float], Optional[str], Optional[str]]:
    if not isinstance(parsed, dict):
        return None, None, None, None, None, "intervention_output_invalid"

    raw_flag = parsed.get("should_intervene")
    if not isinstance(raw_flag, bool):
        return None, None, None, None, None, "intervention_flag_missing"

    raw_severity_score = parsed.get("severity_score")
    if raw_severity_score is None:
        return None, None, None, None, None, "intervention_severity_score_missing"
    try:
        severity_score = float(raw_severity_score)
    except Exception:
        return None, None, None, None, None, "intervention_severity_score_invalid"
    if not (0.0 <= severity_score <= 1.0):
        return None, None, None, None, None, f"intervention_severity_score_out_of_range:{severity_score}"

    corrective_action = normalize_corrective_action(parsed.get("corrective_action"))
    if raw_flag and corrective_action is None:
        return None, None, None, None, None, "intervention_corrective_action_missing"

    confidence = None
    if "confidence" in parsed and parsed.get("confidence") is not None:
        raw_confidence = parsed.get("confidence")
        try:
            confidence = float(raw_confidence)
        except Exception:
            return None, None, None, None, None, "intervention_confidence_invalid"

    reasoning = parsed.get("reasoning")
    if reasoning is not None and not isinstance(reasoning, str):
        reasoning = str(reasoning)
    return bool(raw_flag), severity_score, corrective_action, confidence, reasoning, None


def coerce_critique_result(parsed: object) -> CritiqueResult:
    if not isinstance(parsed, dict):
        return CritiqueResult(
            accepted=False,
            severity_score=1.0,
            corrective_action="straight",
            error="critique_output_invalid",
        )

    raw_accepted = parsed.get("accepted")
    if not isinstance(raw_accepted, bool):
        return CritiqueResult(
            accepted=False,
            severity_score=1.0,
            corrective_action="straight",
            error="critique_accepted_missing",
            raw=dict(parsed),
        )

    try:
        severity_score = float(parsed.get("severity_score"))
    except Exception:
        return CritiqueResult(
            accepted=False,
            severity_score=1.0,
            corrective_action="straight",
            error="critique_severity_score_invalid",
            raw=dict(parsed),
        )
    if not (0.0 <= severity_score <= 1.0):
        return CritiqueResult(
            accepted=False,
            severity_score=1.0,
            corrective_action="straight",
            error=f"critique_severity_score_out_of_range:{severity_score}",
            raw=dict(parsed),
        )

    corrective_action = normalize_corrective_action(parsed.get("corrective_action")) or "straight"
    confidence = None
    if parsed.get("confidence") is not None:
        try:
            confidence = float(parsed.get("confidence"))
        except Exception:
            return CritiqueResult(
                accepted=False,
                severity_score=severity_score,
                corrective_action=corrective_action,
                error="critique_confidence_invalid",
                raw=dict(parsed),
            )
    reasoning = parsed.get("reasoning")
    if reasoning is not None and not isinstance(reasoning, str):
        reasoning = str(reasoning)

    return CritiqueResult(
        accepted=bool(raw_accepted),
        severity_score=severity_score,
        corrective_action=corrective_action,
        confidence=confidence,
        reasoning=reasoning,
        raw=dict(parsed),
    )


@dataclass(frozen=True)
class SemanticVerifierParseResult:
    """Structured result parsed from the semantic verifier VLM output."""

    on_track: bool
    confidence: Optional[float] = None
    severity: str = "none"
    reason_code: str = "UNCERTAIN"
    takeover_recommended: bool = False
    reasoning: Optional[str] = None
    error: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


def coerce_semantic_verifier_result(parsed: object) -> SemanticVerifierParseResult:
    if not isinstance(parsed, dict):
        return SemanticVerifierParseResult(
            on_track=True,
            error="semantic_verifier_output_invalid",
        )

    raw_on_track = parsed.get("on_track")
    if raw_on_track is None:
        raw_on_track = parsed.get("accepted")
    if not isinstance(raw_on_track, bool):
        return SemanticVerifierParseResult(
            on_track=True,
            error="semantic_verifier_on_track_missing",
            raw=dict(parsed),
        )

    confidence = None
    if parsed.get("confidence") is not None:
        try:
            confidence = float(parsed.get("confidence"))
        except Exception:
            return SemanticVerifierParseResult(
                on_track=bool(raw_on_track),
                error="semantic_verifier_confidence_invalid",
                raw=dict(parsed),
            )
        if not (0.0 <= confidence <= 1.0):
            return SemanticVerifierParseResult(
                on_track=bool(raw_on_track),
                error=f"semantic_verifier_confidence_out_of_range:{confidence}",
                raw=dict(parsed),
            )

    reasoning = parsed.get("reasoning")
    if reasoning is not None and not isinstance(reasoning, str):
        reasoning = str(reasoning)

    severity = str(parsed.get("severity") or "none").strip().lower()
    if severity not in {"none", "minor", "severe"}:
        severity = "none"
    reason_code = str(parsed.get("reason_code") or "UNCERTAIN").strip().upper()
    if reason_code not in {
        "ALIGNED", "WRONG_TURN", "OFF_ROUTE", "ROAD_DEPARTURE",
        "WRONG_WAY", "UNEXPECTED_ROAD_OBSTACLE", "UNCERTAIN",
    }:
        reason_code = "UNCERTAIN"
    raw_takeover = parsed.get("takeover_recommended", False)
    takeover_recommended = raw_takeover if isinstance(raw_takeover, bool) else False

    return SemanticVerifierParseResult(
        on_track=bool(raw_on_track),
        confidence=confidence,
        severity=severity,
        reason_code=reason_code,
        takeover_recommended=takeover_recommended,
        reasoning=reasoning,
        raw=dict(parsed),
    )


@dataclass(frozen=True)
class RecoveryReasonParseResult:
    """Structured result parsed from the recovery-trigger explanation VLM call."""

    reason: Optional[str] = None
    primary_cause: Optional[str] = None
    confidence: Optional[float] = None
    error: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


def coerce_recovery_reason_result(parsed: object) -> RecoveryReasonParseResult:
    if not isinstance(parsed, dict):
        return RecoveryReasonParseResult(error="recovery_reason_output_invalid")

    raw_reason = parsed.get("reason")
    if not isinstance(raw_reason, str) or not raw_reason.strip():
        return RecoveryReasonParseResult(error="recovery_reason_missing", raw=dict(parsed))

    primary_cause = parsed.get("primary_cause")
    if primary_cause is not None and not isinstance(primary_cause, str):
        primary_cause = str(primary_cause)

    confidence = None
    if parsed.get("confidence") is not None:
        try:
            confidence = float(parsed.get("confidence"))
        except Exception:
            return RecoveryReasonParseResult(
                reason=raw_reason.strip(),
                primary_cause=primary_cause,
                error="recovery_reason_confidence_invalid",
                raw=dict(parsed),
            )

    return RecoveryReasonParseResult(
        reason=raw_reason.strip(),
        primary_cause=primary_cause,
        confidence=confidence,
        raw=dict(parsed),
    )


@dataclass(frozen=True)
class RecoverySelectionParseResult:
    """Structured result parsed from the recovery primitive-selection VLM call."""

    selected_index: Optional[int] = None
    reasoning: Optional[str] = None
    confidence: Optional[float] = None
    lateral_distance_m: Optional[float] = None
    error: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RecoverySequenceStep:
    """One validated action in a structured recovery sequence."""

    action: str
    duration_sec: float
    target_speed_mps: Optional[float] = None
    lateral_distance_m: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "target_speed_mps": self.target_speed_mps,
            "lateral_distance_m": self.lateral_distance_m,
        }


@dataclass(frozen=True)
class RecoverySequenceParseResult:
    """Structured result parsed from the recovery sequence-planning VLM call."""

    steps: Tuple[RecoverySequenceStep, ...] = ()
    reasoning: Optional[str] = None
    confidence: Optional[float] = None
    error: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RecoveryPlanCandidate:
    """One VLM-ranked primitive; geometry is sampled by the orchestrator."""

    action: str
    probability: float
    target_speed_mps: Optional[float] = None
    reasoning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "target_speed_mps": self.target_speed_mps,
            "probability": self.probability,
            "reasoning": self.reasoning,
        }


@dataclass(frozen=True)
class RecoveryPlanParseResult:
    """Validated concise assessment plus probabilistic primitive candidates."""

    front_blockage: Optional[str] = None
    escape_direction: Optional[str] = None
    needs_lane_bypass_left: Optional[bool] = None
    needs_lane_bypass_right: Optional[bool] = None
    candidates: Tuple[RecoveryPlanCandidate, ...] = ()
    error: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


def coerce_recovery_sequence_result(
    parsed: object,
    *,
    action_specs: Dict[str, Dict[str, object]],
    max_steps: int,
    first_action_names: Optional[Sequence[str]] = None,
) -> RecoverySequenceParseResult:
    """Validate a bounded sequence and its optional first-step PDMS whitelist."""
    if not isinstance(parsed, dict):
        return RecoverySequenceParseResult(error="recovery_sequence_output_invalid")
    raw_steps = parsed.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        return RecoverySequenceParseResult(
            error="recovery_sequence_steps_missing", raw=dict(parsed)
        )
    if len(raw_steps) > max(1, int(max_steps)):
        return RecoverySequenceParseResult(
            error=f"recovery_sequence_too_many_steps:{len(raw_steps)}", raw=dict(parsed)
        )

    steps: List[RecoverySequenceStep] = []
    first_action_set = (
        None
        if first_action_names is None
        else {str(name).strip() for name in first_action_names}
    )
    for index, raw_step in enumerate(raw_steps):
        if not isinstance(raw_step, dict):
            return RecoverySequenceParseResult(
                error=f"recovery_sequence_step_invalid:{index}", raw=dict(parsed)
            )
        action = str(raw_step.get("action") or "").strip()
        spec = action_specs.get(action)
        if spec is None:
            return RecoverySequenceParseResult(
                error=f"recovery_sequence_action_unavailable:{action}", raw=dict(parsed)
            )
        if index == 0 and first_action_set is not None and action not in first_action_set:
            return RecoverySequenceParseResult(
                error=f"recovery_sequence_first_action_not_pdms_passed:{action}",
                raw=dict(parsed),
            )
        if raw_step.get("duration_sec") is not None:
            return RecoverySequenceParseResult(
                error=f"recovery_sequence_unexpected_duration:{action}", raw=dict(parsed)
            )
        try:
            duration_sec = float(spec.get("fixed_duration_sec"))
        except Exception:
            return RecoverySequenceParseResult(
                error=f"recovery_sequence_spec_invalid:{action}", raw=dict(parsed)
            )
        if duration_sec <= 0.0:
            return RecoverySequenceParseResult(
                error=f"recovery_sequence_spec_invalid:{action}", raw=dict(parsed)
            )

        target_speed_mps = None
        speed_range = spec.get("target_speed_mps")
        raw_speed = raw_step.get("target_speed_mps")
        if speed_range is not None:
            try:
                target_speed_mps = float(raw_speed)
            except Exception:
                return RecoverySequenceParseResult(
                    error=f"recovery_sequence_speed_invalid:{index}", raw=dict(parsed)
                )
            speed_lower, speed_upper = map(float, speed_range)
            if not speed_lower <= target_speed_mps <= speed_upper:
                return RecoverySequenceParseResult(
                    error=(
                        f"recovery_sequence_speed_out_of_range:{action}:"
                        f"{target_speed_mps:.3f} not in [{speed_lower:.3f},{speed_upper:.3f}]"
                    ),
                    raw=dict(parsed),
                )
        elif raw_speed is not None:
            return RecoverySequenceParseResult(
                error=f"recovery_sequence_unexpected_speed:{action}", raw=dict(parsed)
            )

        lateral_distance_m = None
        lateral_range = spec.get("lateral_distance_m")
        raw_lateral = raw_step.get("lateral_distance_m")
        if lateral_range is not None:
            try:
                lateral_distance_m = float(raw_lateral)
            except Exception:
                return RecoverySequenceParseResult(
                    error=f"recovery_sequence_lateral_invalid:{index}", raw=dict(parsed)
                )
            lateral_lower, lateral_upper = map(float, lateral_range)
            if not lateral_lower <= lateral_distance_m <= lateral_upper:
                return RecoverySequenceParseResult(
                    error=(
                        f"recovery_sequence_lateral_out_of_range:{action}:"
                        f"{lateral_distance_m:.3f} not in "
                        f"[{lateral_lower:.3f},{lateral_upper:.3f}]"
                    ),
                    raw=dict(parsed),
                )
        elif raw_lateral is not None:
            return RecoverySequenceParseResult(
                error=f"recovery_sequence_unexpected_lateral:{action}", raw=dict(parsed)
            )

        steps.append(
            RecoverySequenceStep(
                action=action,
                duration_sec=duration_sec,
                target_speed_mps=target_speed_mps,
                lateral_distance_m=lateral_distance_m,
            )
        )

    confidence = None
    if parsed.get("confidence") is not None:
        try:
            confidence = float(parsed.get("confidence"))
        except Exception:
            return RecoverySequenceParseResult(
                error="recovery_sequence_confidence_invalid", raw=dict(parsed)
            )
        if not 0.0 <= confidence <= 1.0:
            return RecoverySequenceParseResult(
                error="recovery_sequence_confidence_out_of_range", raw=dict(parsed)
            )

    reasoning = parsed.get("reasoning")
    if reasoning is not None and not isinstance(reasoning, str):
        reasoning = str(reasoning)
    return RecoverySequenceParseResult(
        steps=tuple(steps),
        reasoning=reasoning,
        confidence=confidence,
        raw=dict(parsed),
    )


def coerce_recovery_plan_result(
    parsed: object,
    *,
    action_specs: Dict[str, Dict[str, object]],
    first_action_names: Optional[Sequence[str]] = None,
    bypass_first: bool = False,
    bypass_action: str = "lane_bypass_left",
) -> RecoveryPlanParseResult:
    """Validate a 0--5 item primitive distribution without geometry output."""
    if not isinstance(parsed, dict):
        return RecoveryPlanParseResult(error="recovery_plan_output_invalid")
    raw = dict(parsed)
    front_blockage = parsed.get("front_blockage")
    if not isinstance(front_blockage, str) or not front_blockage.strip():
        return RecoveryPlanParseResult(
            error="recovery_plan_front_blockage_missing", raw=raw
        )
    front_blockage = front_blockage.strip()

    escape_direction = str(parsed.get("escape_direction") or "").strip().lower()
    if escape_direction not in {"forward", "left", "right", "back", "stop"}:
        return RecoveryPlanParseResult(
            front_blockage=front_blockage,
            error=f"recovery_plan_escape_direction_invalid:{escape_direction}",
            raw=raw,
        )

    bypass_action = str(bypass_action).strip()
    bypass_side = "right" if bypass_action.endswith("_right") else "left"
    bypass_field = f"needs_lane_bypass_{bypass_side}"
    needs_lane_bypass: Optional[bool] = None
    if bypass_first:
        raw_bypass = parsed.get(bypass_field)
        if not isinstance(raw_bypass, bool):
            return RecoveryPlanParseResult(
                front_blockage=front_blockage,
                escape_direction=escape_direction,
                error="recovery_plan_bypass_decision_missing",
                raw=raw,
            )
        needs_lane_bypass = raw_bypass

    def bypass_fields(value: Optional[bool]) -> Dict[str, Optional[bool]]:
        return {
            "needs_lane_bypass_left": (
                value if bypass_side == "left" else None
            ),
            "needs_lane_bypass_right": (
                value if bypass_side == "right" else None
            ),
        }

    raw_candidates = parsed.get("candidates")
    if not isinstance(raw_candidates, list):
        return RecoveryPlanParseResult(
            front_blockage=front_blockage,
            escape_direction=escape_direction,
            error="recovery_plan_candidates_missing",
            raw=raw,
        )
    if len(raw_candidates) > 5:
        return RecoveryPlanParseResult(
            front_blockage=front_blockage,
            escape_direction=escape_direction,
            error=f"recovery_plan_too_many_candidates:{len(raw_candidates)}",
            raw=raw,
        )
    if not raw_candidates:
        if needs_lane_bypass:
            return RecoveryPlanParseResult(
                front_blockage=front_blockage,
                escape_direction=escape_direction,
                **bypass_fields(True),
                error="recovery_plan_bypass_requires_single_candidate",
                raw=raw,
            )
        if escape_direction != "stop":
            return RecoveryPlanParseResult(
                front_blockage=front_blockage,
                escape_direction=escape_direction,
                error="recovery_plan_empty_candidates_require_stop",
                raw=raw,
            )
        return RecoveryPlanParseResult(
            front_blockage=front_blockage,
            escape_direction=escape_direction,
            **bypass_fields(needs_lane_bypass),
            candidates=(),
            raw=raw,
        )

    allowed = (
        set(action_specs)
        if first_action_names is None
        else {str(name) for name in first_action_names}
    )
    # In bypass-first mode this list represents actions executable *now*.
    # A positive necessity decision is allowed to name the bypass even while
    # its independent traffic/topology gate is closed; the orchestrator latches
    # that intent and brakes until execution becomes safe.
    if bypass_first and needs_lane_bypass:
        allowed.add(bypass_action)
    seen = set()
    candidates: List[RecoveryPlanCandidate] = []
    for index, raw_candidate in enumerate(raw_candidates):
        if not isinstance(raw_candidate, dict):
            return RecoveryPlanParseResult(
                front_blockage=front_blockage,
                escape_direction=escape_direction,
                error=f"recovery_plan_candidate_invalid:{index}",
                raw=raw,
            )
        raw_primitive = raw_candidate.get("primitive")
        if not isinstance(raw_primitive, dict):
            return RecoveryPlanParseResult(
                front_blockage=front_blockage,
                escape_direction=escape_direction,
                error=f"recovery_plan_primitive_missing:{index}",
                raw=raw,
            )
        action = str(raw_primitive.get("action") or "").strip()
        spec = action_specs.get(action)
        if spec is None or action not in allowed:
            return RecoveryPlanParseResult(
                front_blockage=front_blockage,
                escape_direction=escape_direction,
                error=f"recovery_plan_action_unavailable:{action}",
                raw=raw,
            )
        if action in seen:
            return RecoveryPlanParseResult(
                front_blockage=front_blockage,
                escape_direction=escape_direction,
                error=f"recovery_plan_action_repeated:{action}",
                raw=raw,
            )
        seen.add(action)
        if "lateral_distance_m" in raw_primitive:
            return RecoveryPlanParseResult(
                front_blockage=front_blockage,
                escape_direction=escape_direction,
                error=f"recovery_plan_unexpected_lateral:{action}",
                raw=raw,
            )

        raw_speed = raw_primitive.get("target_speed_mps")
        speed_range = spec.get("target_speed_mps")
        target_speed_mps = None
        if speed_range is not None:
            try:
                target_speed_mps = float(raw_speed)
            except (TypeError, ValueError):
                return RecoveryPlanParseResult(
                    front_blockage=front_blockage,
                    escape_direction=escape_direction,
                    error=f"recovery_plan_speed_invalid:{action}",
                    raw=raw,
                )
            speed_lo, speed_hi = map(float, speed_range)
            if not speed_lo <= target_speed_mps <= speed_hi:
                return RecoveryPlanParseResult(
                    front_blockage=front_blockage,
                    escape_direction=escape_direction,
                    error=(
                        f"recovery_plan_speed_out_of_range:{action}:"
                        f"{target_speed_mps:.3f} not in [{speed_lo:.3f},{speed_hi:.3f}]"
                    ),
                    raw=raw,
                )
        elif raw_speed is not None:
            return RecoveryPlanParseResult(
                front_blockage=front_blockage,
                escape_direction=escape_direction,
                error=f"recovery_plan_unexpected_speed:{action}",
                raw=raw,
            )

        try:
            probability = float(raw_candidate.get("probability"))
        except (TypeError, ValueError):
            return RecoveryPlanParseResult(
                front_blockage=front_blockage,
                escape_direction=escape_direction,
                error=f"recovery_plan_probability_invalid:{action}",
                raw=raw,
            )
        if not math.isfinite(probability) or not 0.0 < probability <= 1.0:
            return RecoveryPlanParseResult(
                front_blockage=front_blockage,
                escape_direction=escape_direction,
                error=f"recovery_plan_probability_out_of_range:{action}:{probability}",
                raw=raw,
            )
        reasoning = raw_candidate.get("reasoning")
        if reasoning is not None and not isinstance(reasoning, str):
            reasoning = str(reasoning)
        candidates.append(
            RecoveryPlanCandidate(
                action=action,
                probability=probability,
                target_speed_mps=target_speed_mps,
                reasoning=reasoning,
            )
        )

    if bypass_first:
        selected_actions = [candidate.action for candidate in candidates]
        if needs_lane_bypass:
            if escape_direction != bypass_side:
                return RecoveryPlanParseResult(
                    front_blockage=front_blockage,
                    escape_direction=escape_direction,
                    **bypass_fields(True),
                    error=f"recovery_plan_bypass_requires_{bypass_side}_escape",
                    raw=raw,
                )
            if selected_actions != [bypass_action]:
                return RecoveryPlanParseResult(
                    front_blockage=front_blockage,
                    escape_direction=escape_direction,
                    **bypass_fields(True),
                    error="recovery_plan_bypass_requires_exclusive_primitive",
                    raw=raw,
                )
            if not math.isclose(
                candidates[0].probability, 1.0, rel_tol=0.0, abs_tol=1e-6
            ):
                return RecoveryPlanParseResult(
                    front_blockage=front_blockage,
                    escape_direction=escape_direction,
                    **bypass_fields(True),
                    error="recovery_plan_bypass_probability_must_be_one",
                    raw=raw,
                )
        elif bypass_action in selected_actions:
            return RecoveryPlanParseResult(
                front_blockage=front_blockage,
                escape_direction=escape_direction,
                **bypass_fields(False),
                error="recovery_plan_regular_candidates_include_bypass",
                raw=raw,
            )

    return RecoveryPlanParseResult(
        front_blockage=front_blockage,
        escape_direction=escape_direction,
        **bypass_fields(needs_lane_bypass),
        candidates=tuple(candidates),
        raw=raw,
    )


def coerce_recovery_selection_result(
    parsed: object,
    *,
    num_candidates: int,
    candidate_names: Optional[Sequence[str]] = None,
    lateral_distance_ranges: Optional[Sequence[Optional[Tuple[float, float]]]] = None,
) -> RecoverySelectionParseResult:
    if not isinstance(parsed, dict):
        return RecoverySelectionParseResult(error="recovery_selection_output_invalid")

    raw_index = parsed.get("selected_index")
    if raw_index is None:
        raw_index = parsed.get("best_candidate_index")
    try:
        selected_index = int(raw_index)
    except Exception:
        return RecoverySelectionParseResult(
            error="recovery_selection_index_invalid", raw=dict(parsed)
        )
    if not (0 <= selected_index < int(num_candidates)):
        return RecoverySelectionParseResult(
            error=f"recovery_selection_index_out_of_range:{selected_index}",
            raw=dict(parsed),
        )

    if candidate_names is not None and selected_index < len(candidate_names):
        parsed_maneuver = parsed.get("maneuver")
        expected_maneuver = str(candidate_names[selected_index])
        if parsed_maneuver is None:
            return RecoverySelectionParseResult(
                selected_index=selected_index,
                error="recovery_selection_maneuver_missing",
                raw=dict(parsed),
            )
        if str(parsed_maneuver) != expected_maneuver:
            return RecoverySelectionParseResult(
                selected_index=selected_index,
                error=(
                    "recovery_selection_maneuver_mismatch:"
                    f"{parsed_maneuver}!={expected_maneuver}"
                ),
                raw=dict(parsed),
            )

    lateral_distance_m = None
    selected_range = None
    if lateral_distance_ranges is not None and selected_index < len(lateral_distance_ranges):
        selected_range = lateral_distance_ranges[selected_index]
    if selected_range is not None:
        raw_lateral = parsed.get("lateral_distance_m")
        try:
            lateral_distance_m = float(raw_lateral)
        except Exception:
            return RecoverySelectionParseResult(
                selected_index=selected_index,
                error="recovery_selection_lateral_distance_invalid",
                raw=dict(parsed),
            )
        lower, upper = float(selected_range[0]), float(selected_range[1])
        if not (lower <= lateral_distance_m <= upper):
            return RecoverySelectionParseResult(
                selected_index=selected_index,
                lateral_distance_m=lateral_distance_m,
                error=(
                    "recovery_selection_lateral_distance_out_of_range:"
                    f"{lateral_distance_m:.3f} not in [{lower:.3f},{upper:.3f}]"
                ),
                raw=dict(parsed),
            )
    elif parsed.get("lateral_distance_m") is not None:
        return RecoverySelectionParseResult(
            selected_index=selected_index,
            error="recovery_selection_unexpected_lateral_distance",
            raw=dict(parsed),
        )

    confidence = None
    if parsed.get("confidence") is not None:
        try:
            confidence = float(parsed.get("confidence"))
        except Exception:
            return RecoverySelectionParseResult(
                selected_index=selected_index,
                error="recovery_selection_confidence_invalid",
                raw=dict(parsed),
            )
        if not (0.0 <= confidence <= 1.0):
            return RecoverySelectionParseResult(
                selected_index=selected_index,
                lateral_distance_m=lateral_distance_m,
                error="recovery_selection_confidence_out_of_range",
                raw=dict(parsed),
            )

    reasoning = parsed.get("reasoning")
    if reasoning is not None and not isinstance(reasoning, str):
        reasoning = str(reasoning)

    return RecoverySelectionParseResult(
        selected_index=selected_index,
        reasoning=reasoning,
        confidence=confidence,
        lateral_distance_m=lateral_distance_m,
        raw=dict(parsed),
    )


def select_from_vlm_scores(
    candidate_rows: Sequence[Dict[str, object]],
    vlm_scores: Sequence[float],
) -> Dict[str, object]:
    vlm_scores = [float(score) for score in vlm_scores]
    if len(vlm_scores) != len(candidate_rows):
        raise ValueError("VLM score count does not match candidate rows")

    selected_idx = int(max(range(len(vlm_scores)), key=lambda idx: vlm_scores[idx]))
    selected_row = dict(candidate_rows[selected_idx])
    row_source = str(selected_row.get("source", "current_rap"))
    if row_source == "carry_prev":
        selected_source = "vlm_selected_carry_prev"
        decision = "vlm_selected_reuse_prev"
    elif row_source.startswith("default_fallback_"):
        selected_source = "vlm_selected_default_fallback"
        decision = "vlm_selected_default_fallback"
    else:
        selected_source = "vlm_selected_current"
        decision = "vlm_selected_current"

    sorted_scores = sorted(((float(score), idx) for idx, score in enumerate(vlm_scores)), reverse=True)
    score_gap_top2 = None
    if len(sorted_scores) >= 2:
        score_gap_top2 = float(sorted_scores[0][0] - sorted_scores[1][0])

    return {
        "selected_candidate_index": selected_idx,
        "selected_candidate_row": selected_row,
        "selected_source": selected_source,
        "adaptive_replan_decision": decision,
        "vlm_q_score_gap_top2": score_gap_top2,
    }


def selected_path_reasoning(
    selected_row: Dict[str, object],
    selected_candidate_index: int,
    selected_source: str,
    vlm_scores: Optional[Sequence[float]],
    parsed_reasoning: Optional[object],
) -> str:
    if isinstance(parsed_reasoning, str) and parsed_reasoning.strip():
        return parsed_reasoning.strip()

    summary = selected_row.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        summary = "selected path"

    score_text = ""
    if vlm_scores is not None and 0 <= selected_candidate_index < len(vlm_scores):
        score_text = f" with highest q-score {float(vlm_scores[selected_candidate_index]):.3f}"
    return (
        f"Selected {selected_source} candidate {int(selected_candidate_index)}{score_text}: "
        f"{summary}."
    )


__all__ = [
    "CritiqueResult",
    "coerce_candidate_scores",
    "coerce_critique_result",
    "coerce_intervention_decision",
    "coerce_vlm_candidate_scores",
    "empty_token_usage",
    "intervention_severity_band",
    "normalize_corrective_action",
    "normalize_token_usage",
    "select_from_vlm_scores",
    "selected_path_reasoning",
    "try_parse_json",
]
