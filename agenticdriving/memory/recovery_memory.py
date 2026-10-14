"""Recovery-attempt memory: which maneuvers were executed and how they ended.

Standalone module (no simulator / selector imports) behind the orchestrator's
per-tick recovery: each ``_recover`` call selects ONE primitive and returns its
trajectory to CARLA; whether that attempt worked is only observable on a later
tick (verifier passes -> success; recovery re-triggers -> failure).
:class:`RecoveryAttemptTracker` does that cross-tick settlement on top of
:class:`RecoveryMemory`; the dummy selector then hard-excludes tried maneuvers,
while the VLM selector receives the :meth:`RecoveryMemory.describe` summary as
critique context and re-ranks the full set itself.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set


@dataclass
class AttemptRecord:
    """One executed recovery maneuver and its outcome."""

    primitive: str
    success: bool
    failure_reason: Optional[str] = None
    selector: str = "dummy"
    frame_index: Optional[int] = None
    timestamp: float = field(default_factory=time.time)
    extra: Dict[str, Any] = field(default_factory=dict)


class RecoveryMemory:
    """Append-only log of executed recovery attempts for one trigger.

    Call :meth:`reset` when a new recovery episode starts, so exclusions from
    a previous episode do not leak into the next one. This is bookkeeping only;
    failed maneuvers are never physically rolled back.
    """

    def __init__(self) -> None:
        self._records: List[AttemptRecord] = []

    # --- write -------------------------------------------------------------
    def record(
        self,
        primitive: str,
        success: bool,
        *,
        failure_reason: Optional[str] = None,
        selector: str = "dummy",
        frame_index: Optional[int] = None,
        **extra: Any,
    ) -> AttemptRecord:
        rec = AttemptRecord(
            primitive=str(primitive),
            success=bool(success),
            failure_reason=failure_reason,
            selector=selector,
            frame_index=frame_index,
            extra=dict(extra),
        )
        self._records.append(rec)
        return rec

    def reset(self) -> None:
        self._records.clear()

    # --- read --------------------------------------------------------------
    @property
    def records(self) -> List[AttemptRecord]:
        return list(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def tried(self) -> Set[str]:
        """Names of every maneuver executed so far (success or failure)."""
        return {r.primitive for r in self._records}

    def failed(self) -> Set[str]:
        return {r.primitive for r in self._records if not r.success}

    def last(self) -> Optional[AttemptRecord]:
        return self._records[-1] if self._records else None

    def filter_untried(
        self,
        rows: Sequence[Dict[str, object]],
        *,
        key: str = "primitive_name",
    ) -> List[Dict[str, object]]:
        """Candidate rows whose ``key`` has not been executed yet (dummy path)."""
        tried = self.tried()
        return [row for row in rows if str(row.get(key)) not in tried]

    def describe(self) -> Optional[str]:
        """Human/VLM-readable summary of past attempts (``None`` when empty)."""
        if not self._records:
            return None
        lines = []
        for i, r in enumerate(self._records, 1):
            outcome = "succeeded" if r.success else f"FAILED ({r.failure_reason or 'unknown reason'})"
            lines.append(f"attempt {i}: '{r.primitive}' {outcome}")
        return "Previously executed recovery maneuvers: " + "; ".join(lines) + "."

    # --- persistence -------------------------------------------------------
    def to_dicts(self) -> List[Dict[str, Any]]:
        return [asdict(r) for r in self._records]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dicts(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "RecoveryMemory":
        mem = cls()
        for item in json.loads(Path(path).read_text()):
            mem._records.append(AttemptRecord(**item))
        return mem


@dataclass
class FailureMemoryEvent:
    """One problem, attempted method, or outcome in a recovery episode."""

    timestamp: float
    event_type: str
    problem: Optional[str] = None
    method: Optional[str] = None
    outcome: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_prompt_line(self, index: int) -> str:
        fields = [f"event {index}", f"t={self.timestamp:.2f}", f"type={self.event_type}"]
        if self.problem:
            fields.append(f"problem={self.problem}")
        if self.method:
            fields.append(f"method={self.method}")
        if self.outcome:
            fields.append(f"outcome={self.outcome}")
        if self.details:
            fields.append(
                "details=" + json.dumps(self.details, sort_keys=True, default=str)
            )
        return " | ".join(fields)


@dataclass
class RecoveryDecisionAttempt:
    """Compact state for one high-level recovery decision attempt.

    Raw verifier/debug details remain in :class:`FailureMemoryEvent`.  This
    structure intentionally keeps only the fields needed to understand the
    problem-solving process and build a short selector context.
    """

    method: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    precheck_result: Optional[str] = None
    execution_result: Optional[str] = None
    # x4 additions: the sequence steps (with their parameters) this attempt
    # proposed, a short verifier note ("score 0.40 vs 0.43, zeroed: ttc") and
    # the route progress its execution bought (written back on the NEXT
    # recovery entry, when the station delta is known).
    steps: List[Dict[str, Any]] = field(default_factory=list)
    note: Optional[str] = None
    progress_m: Optional[float] = None


class RecoveryFailureMemory:
    """Episode-scoped high-level tracker for one unresolved recovery problem.

    The append-only event trace is retained for auditing, while a small reducer
    tracks decision attempts and exposes a bounded four-line context to the
    recovery selector.  The context deliberately omits timestamps, repeated
    verifier metrics, VLM reasoning, and the raw event history; those are
    already available through the current scene/verifier inputs and debug log.

    An action ending is not treated as problem resolution.  The active episode
    is cleared only after the normal Loop 1 verifier remains healthy for
    ``clear_after_loop1_sec`` continuously.  Re-entering recovery before that
    deadline marks the last attempt ineffective and resumes the same episode.
    """

    _FAILED_EXECUTION_OUTCOMES = frozenset(
        {
            "projection_failed",
            "phase_stalled_abort",
            "nc_guard_abort",
            "dac_guard_abort",
            "world_anchor_failed_one_tick_execution",
            "world_plan_too_short_one_tick_execution",
        }
    )
    _FINISHED_EXECUTION_OUTCOMES = frozenset(
        {"action_complete", "time_budget_elapsed"}
    )
    _FAILED_SELECTION_OUTCOMES = frozenset(
        {
            "rejected",
            "rejected_without_generation",
            "generation_failed",
            "invalid_vlm_output",
            "vlm_call_failed",
            "no_primitive_available",
            "selector_returned_no_primitive",
            "rule_based_recovery_failed",
        }
    )
    _CONTEXT_RESULT_NAMES = {
        "nc_guard_abort": "collision_guard_abort",
        "dac_guard_abort": "road_guard_abort",
        "phase_stalled_abort": "stalled_abort",
        "projection_failed": "projection_failed",
        "rejected": "verifier_rejected",
        "generation_failed": "generation_failed",
        "problem_recurred": "problem_recurred",
    }

    def __init__(
        self,
        *,
        enabled: bool = False,
        clear_after_loop1_sec: float = 1.0,
        max_events: int = 64,
        guidance_mode: str = "avoid",
    ) -> None:
        self.enabled = bool(enabled)
        self.clear_after_loop1_sec = max(0.0, float(clear_after_loop1_sec))
        self.max_events = max(1, int(max_events))
        # "avoid": legacy four-line context (byte-identical to the shipped
        # design). "productivity": parameterized recent-attempts table with
        # verifier margins and route progress instead of the bare avoid list.
        if guidance_mode not in ("avoid", "productivity"):
            raise ValueError(
                f"guidance_mode must be 'avoid' or 'productivity', got {guidance_mode!r}"
            )
        self.guidance_mode = guidance_mode
        self._events: List[FailureMemoryEvent] = []
        self._active = False
        self._loop1_since: Optional[float] = None
        self._state = "idle"
        self._attempts: List[RecoveryDecisionAttempt] = []
        self._avoid_actions: List[str] = []

    @property
    def active(self) -> bool:
        return bool(self.enabled and self._active)

    @property
    def events(self) -> List[FailureMemoryEvent]:
        return list(self._events)

    @property
    def loop1_since(self) -> Optional[float]:
        return self._loop1_since

    @property
    def state(self) -> str:
        return self._state

    @property
    def attempts(self) -> List[RecoveryDecisionAttempt]:
        return list(self._attempts)

    @property
    def avoid_actions(self) -> List[str]:
        return list(self._avoid_actions)

    @staticmethod
    def _parameters(details: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        raw = dict(details or {})
        parameters = dict(raw.get("parameters") or {})
        lateral = raw.get("lateral_distance_m")
        if lateral is not None:
            parameters["lateral_distance_m"] = lateral
        return parameters

    @staticmethod
    def _attempt_label(attempt: RecoveryDecisionAttempt) -> str:
        lateral = attempt.parameters.get("lateral_distance_m")
        if lateral is None:
            return attempt.method
        try:
            return f"{attempt.method}({float(lateral):g}m)"
        except (TypeError, ValueError):
            return attempt.method

    @staticmethod
    def _round_half(value: Any) -> Optional[float]:
        try:
            return round(float(value) * 2.0) / 2.0
        except (TypeError, ValueError):
            return None

    @classmethod
    def _parameterized_label(cls, attempt: RecoveryDecisionAttempt) -> str:
        """Chain label with per-step parameters, rounded to 0.5 so retries at
        nearby values aggregate while genuinely different ones stay distinct."""
        if not attempt.steps:
            return cls._attempt_label(attempt)
        parts = []
        for step in attempt.steps:
            bits = []
            lat = cls._round_half(step.get("lateral_distance_m"))
            spd = cls._round_half(step.get("target_speed_mps"))
            dur = cls._round_half(step.get("duration_sec"))
            if lat is not None:
                bits.append(f"{lat:g}m")
            if spd is not None:
                bits.append(f"{spd:g}m/s")
            if dur is not None:
                bits.append(f"{dur:g}s")
            name = str(step.get("action") or "?")
            parts.append(name + (f"({','.join(bits)})" if bits else ""))
        return ">".join(parts)

    def _current_attempt(self) -> Optional[RecoveryDecisionAttempt]:
        return self._attempts[-1] if self._attempts else None

    def _ensure_attempt(
        self,
        method: Optional[str],
        details: Optional[Dict[str, Any]],
    ) -> Optional[RecoveryDecisionAttempt]:
        if method is None:
            return self._current_attempt()
        method = str(method)
        steps = list((details or {}).get("sequence") or [])
        current = self._current_attempt()
        if (
            current is not None
            and current.method == method
            and current.execution_result is None
        ):
            current.parameters.update(self._parameters(details))
            if steps and not current.steps:
                current.steps = steps
            return current
        attempt = RecoveryDecisionAttempt(
            method=method,
            parameters=self._parameters(details),
            steps=steps,
        )
        self._attempts.append(attempt)
        return attempt

    def _avoid(self, attempt: Optional[RecoveryDecisionAttempt]) -> None:
        if attempt is None:
            return
        label = self._attempt_label(attempt)
        if label in self._avoid_actions:
            self._avoid_actions.remove(label)
        self._avoid_actions.append(label)

    def _reduce(
        self,
        *,
        event_type: str,
        method: Optional[str],
        outcome: Optional[str],
        details: Optional[Dict[str, Any]],
    ) -> None:
        """Update the compact problem-solving state from one raw event."""
        event_type = str(event_type)
        outcome = None if outcome is None else str(outcome)

        if event_type == "recovery_problem":
            if not self._attempts:
                self._state = "deciding"
            return

        if event_type == "primitive_selected":
            self._ensure_attempt(method, details)
            self._state = "deciding"
            return

        if event_type == "primitive_selection_fallback":
            attempt = self._ensure_attempt(method, details)
            if attempt is not None:
                attempt.precheck_result = "passed_fallback"
            self._state = "deciding"
            return

        if event_type == "pdms_verification":
            attempt = self._ensure_attempt(method, details)
            if attempt is not None:
                attempt.precheck_result = "passed" if outcome == "accepted" else "failed"
                # x4: keep the margin and the zeroed terms for the
                # productivity-mode table.
                raw = details or {}
                score = raw.get("verifier_score")
                threshold = raw.get("verifier_threshold")
                zeroed = sorted(
                    str(name)
                    for src in (raw.get("gates") or {}, raw.get("soft") or {})
                    for name, value in src.items()
                    if isinstance(value, (int, float)) and float(value) == 0.0
                )
                if score is not None and threshold is not None:
                    note = f"score {float(score):.2f} vs {float(threshold):.2f}"
                    if zeroed:
                        note += f", zeroed: {', '.join(zeroed)}"
                    attempt.note = note
            if outcome != "accepted":
                if attempt is not None:
                    attempt.execution_result = outcome or "verifier_rejected"
                self._avoid(attempt)
                self._state = "retry_required"
            return

        if event_type == "commit_progress":
            attempt = self._current_attempt()
            if attempt is not None and (method is None or attempt.method == method):
                gain = (details or {}).get("gain_m")
                try:
                    attempt.progress_m = float(gain)
                except (TypeError, ValueError):
                    pass
            return

        if event_type == "recovery_committed":
            self._ensure_attempt(method, details)
            self._state = "executing"
            return

        if event_type == "recovery_execution":
            attempt = self._ensure_attempt(method, details)
            if attempt is not None:
                attempt.execution_result = outcome
            if outcome in self._FAILED_EXECUTION_OUTCOMES:
                self._avoid(attempt)
                self._state = "retry_required"
            elif outcome in self._FINISHED_EXECUTION_OUTCOMES:
                self._state = "resolution_pending"
            return

        if event_type in {
            "primitive_selection_error",
            "primitive_selection_repeated",
            "primitive_generation",
            "recovery_selection",
            "recovery_selection_error",
        }:
            attempt = self._ensure_attempt(method, details)
            if outcome in self._FAILED_SELECTION_OUTCOMES:
                if attempt is not None:
                    attempt.execution_result = outcome
                    self._avoid(attempt)
                self._state = "retry_required"
            return

        if event_type == "legacy_recovery_verdict":
            attempt = self._ensure_attempt(method, details)
            if attempt is not None:
                attempt.execution_result = outcome
            if outcome == "rejected":
                self._avoid(attempt)
                self._state = "retry_required"
            elif outcome == "resolved":
                self._state = "resolution_pending"
            return

        if event_type == "loop1_returned":
            self._state = "resolution_pending"

    def _append(self, event: FailureMemoryEvent) -> None:
        if not self.enabled:
            return
        self._events.append(event)
        if len(self._events) > self.max_events:
            del self._events[: len(self._events) - self.max_events]

    def start_or_resume(self, timestamp: float, *, source: str) -> None:
        if not self.enabled:
            return
        timestamp = float(timestamp)
        was_waiting_in_loop1 = self._loop1_since is not None
        self._loop1_since = None
        if not self._active:
            self._active = True
            self._state = "deciding"
            self._append(
                FailureMemoryEvent(
                    timestamp=timestamp,
                    event_type="recovery_started",
                    outcome="active",
                    details={"source": str(source)},
                )
            )
        elif was_waiting_in_loop1:
            attempt = self._current_attempt()
            if attempt is not None:
                attempt.execution_result = "problem_recurred"
                self._avoid(attempt)
            self._state = "retry_required"
            self._append(
                FailureMemoryEvent(
                    timestamp=timestamp,
                    event_type="recovery_resumed",
                    outcome="loop1_stability_interrupted",
                    details={"source": str(source)},
                )
            )

    def record(
        self,
        *,
        timestamp: float,
        event_type: str,
        problem: Optional[str] = None,
        method: Optional[str] = None,
        outcome: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled:
            return
        self._active = True
        normalized_details = dict(details or {})
        self._reduce(
            event_type=str(event_type),
            method=None if method is None else str(method),
            outcome=None if outcome is None else str(outcome),
            details=normalized_details,
        )
        self._append(
            FailureMemoryEvent(
                timestamp=float(timestamp),
                event_type=str(event_type),
                problem=None if problem is None else str(problem),
                method=None if method is None else str(method),
                outcome=None if outcome is None else str(outcome),
                details=normalized_details,
            )
        )

    def observe_loop1(self, timestamp: float) -> bool:
        """Return True when one second of continuous Loop 1 clears memory."""
        if not self.active:
            return False
        timestamp = float(timestamp)
        if self._loop1_since is None:
            self._loop1_since = timestamp
            self._state = "resolution_pending"
            self._append(
                FailureMemoryEvent(
                    timestamp=timestamp,
                    event_type="loop1_returned",
                    outcome="stability_timer_started",
                    details={"clear_after_sec": self.clear_after_loop1_sec},
                )
            )
            if self.clear_after_loop1_sec > 0.0:
                return False
        if timestamp - float(self._loop1_since) < self.clear_after_loop1_sec:
            return False
        self.clear()
        return True

    def describe(
        self, available_actions: Optional[Sequence[str]] = None
    ) -> Optional[str]:
        """Return the bounded context consumed only by the recovery selector.

        ``available_actions`` (the current sequence-action menu) adds an
        ``Untried:`` line naming menu actions that appear in no attempt so far
        — the avoid list says what failed, this says what was never explored.
        Omitted (None) keeps the four-line output byte-identical to before.
        """
        if not self.active:
            return None
        untried_line: Optional[str] = None
        if available_actions is not None:
            tried_tokens: Set[str] = set()
            for item in self._attempts:
                for token in re.split(r"[:>()]+", item.method):
                    if token:
                        tried_tokens.add(token)
            untried = [
                str(name) for name in available_actions
                if str(name) not in tried_tokens
            ]
            untried_line = f"Untried: {', '.join(untried) or 'none'}"

        if self.guidance_mode == "productivity":
            lines = [
                f"Recovery state: {self._state}",
                f"Attempt: {len(self._attempts) + 1}",
            ]
            # Aggregate attempts by parameterized label, most recent groups
            # first, capped at 4 groups so the block stays bounded.
            groups: Dict[str, Dict[str, Any]] = {}
            order: List[str] = []
            for item in reversed(self._attempts):
                label = self._parameterized_label(item)
                if label not in groups:
                    groups[label] = {
                        "count": 0, "progress": 0.0, "has_progress": False,
                        "rejected": 0, "note": None,
                    }
                    order.append(label)
                g = groups[label]
                g["count"] += 1
                if item.progress_m is not None:
                    g["progress"] += float(item.progress_m)
                    g["has_progress"] = True
                result = item.execution_result or item.precheck_result or "pending"
                if result in ("failed",) or item.execution_result in (
                    self._FAILED_SELECTION_OUTCOMES
                    | self._FAILED_EXECUTION_OUTCOMES
                    | {"verifier_rejected", "problem_recurred"}
                ):
                    g["rejected"] += 1
                if g["note"] is None and item.note:
                    g["note"] = item.note
            if order:
                lines.append("Recent attempts:")
                for label in order[:4]:
                    g = groups[label]
                    if g["rejected"] >= g["count"] and g["note"]:
                        verdict = f"rejected x{g['count']}: {g['note']}"
                    elif g["rejected"] >= g["count"]:
                        verdict = f"rejected x{g['count']}"
                    elif g["has_progress"]:
                        verdict = (
                            f"accepted x{g['count']}, "
                            f"route {g['progress']:+.1f}m total"
                        )
                    else:
                        verdict = f"accepted x{g['count']}"
                    lines.append(f"  {label} -> {verdict}")
            if untried_line is not None:
                lines.append(untried_line)
            return "\n".join(lines)

        attempt = self._current_attempt()
        if attempt is None:
            last_action = "none"
        else:
            result = attempt.execution_result or attempt.precheck_result or "pending"
            result = self._CONTEXT_RESULT_NAMES.get(result, result)
            last_action = f"{self._attempt_label(attempt)} -> {result}"
        avoid = ", ".join(self._avoid_actions[-3:]) or "none"
        lines = [
            f"Recovery state: {self._state}",
            f"Attempt: {len(self._attempts) + 1}",
            f"Last action: {last_action}",
            f"Avoid: {avoid}",
        ]
        if untried_line is not None:
            lines.append(untried_line)
        return "\n".join(lines)

    def clear(self) -> None:
        self._events.clear()
        self._active = False
        self._loop1_since = None
        self._state = "idle"
        self._attempts.clear()
        self._avoid_actions.clear()


class RecoveryAttemptTracker:
    """Settles the in-flight recovery attempt from later verifier verdicts.

    Per-tick protocol (driven by ``Orchestrator.step``):

    1. after the verifier runs, call :meth:`observe_verifier` — it resolves the
       primitive selected on an earlier tick: verifier passed -> recorded as
       success and the episode memory resets; failed -> recorded as failed, so
       the upcoming re-selection can exclude it;
    2. when ``_recover`` returns a primitive-backed plan, call
       :meth:`note_attempt` with its name — it becomes the new in-flight
       attempt.

    A failed settlement is recorded as feedback for the next recovery
    selection. The vehicle is never physically driven backward to undo the
    failed attempt.
    """

    def __init__(self, memory: Optional[RecoveryMemory] = None) -> None:
        self.memory = memory if memory is not None else RecoveryMemory()
        self._pending: Optional[str] = None

    @property
    def pending(self) -> Optional[str]:
        """Primitive selected on a previous tick and not yet settled."""
        return self._pending

    def note_attempt(
        self,
        primitive: str,
        *,
        frame_index: Optional[int] = None,
    ) -> None:
        """Register the newly selected maneuver as the in-flight attempt."""
        self._pending = str(primitive)
        self._pending_frame = frame_index

    def observe_verifier(
        self,
        passed: bool,
        *,
        failure_reason: Optional[str] = None,
        frame_index: Optional[int] = None,
        selector: str = "dummy",
    ) -> Optional[AttemptRecord]:
        """Settle the in-flight attempt; no-op when nothing is pending."""
        if self._pending is None:
            return None
        rec = self.memory.record(
            self._pending,
            bool(passed),
            failure_reason=None if passed else (failure_reason or "verifier_rejected"),
            selector=selector,
            frame_index=frame_index,
        )
        self._pending = None
        if passed:
            # Episode resolved: a later trigger is a new situation, so past
            # exclusions must not leak into it.
            self.memory.reset()
        return rec

    def excluded(self) -> Set[str]:
        """Maneuver names the dummy path must not re-select this episode."""
        names = self.memory.tried()
        if self._pending is not None:
            names = names | {self._pending}
        return names

    def describe(self) -> Optional[str]:
        return self.memory.describe()

    def reset(self) -> None:
        self._pending = None
        self.memory.reset()
