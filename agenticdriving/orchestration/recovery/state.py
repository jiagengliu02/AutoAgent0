"""Cross-tick state objects used by recovery orchestration."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def observe_position_dwell(
    info: Dict[str, Any],
    *,
    enabled: bool,
    timeout_sec: float,
    position_tolerance_m: float,
    anchor_xy: Optional[np.ndarray],
    anchor_ts: Optional[float],
) -> tuple[Dict[str, Any], Optional[np.ndarray], Optional[float]]:
    """Observe how long the ego stays inside a position-radius window."""
    timeout_sec = max(0.0, float(timeout_sec))
    position_tolerance_m = max(0.0, float(position_tolerance_m))
    state: Dict[str, Any] = {
        "enabled": bool(enabled) and timeout_sec > 0.0,
        "timeout_sec": timeout_sec,
        "position_tolerance_m": position_tolerance_m,
        "stationary_sec": 0.0,
        "distance_from_anchor_m": 0.0,
        "triggered": False,
    }
    if not state["enabled"]:
        return state, None, None

    try:
        now = float(info["timestamp"])
        xy = np.asarray(info["ego_pos"], dtype=np.float64).reshape(-1)[:2]
        if xy.size != 2 or not np.isfinite(now) or not np.all(np.isfinite(xy)):
            raise ValueError("non-finite ego pose")
    except (KeyError, TypeError, ValueError):
        state["available"] = False
        return state, None, None

    state["available"] = True
    if anchor_xy is None or anchor_ts is None or now < anchor_ts:
        return state, xy.copy(), now

    distance = float(np.linalg.norm(xy - anchor_xy))
    state["distance_from_anchor_m"] = distance
    if distance > position_tolerance_m:
        state["timer_reset"] = True
        return state, xy.copy(), now

    stationary_sec = max(0.0, now - float(anchor_ts))
    state["stationary_sec"] = stationary_sec
    state["triggered"] = stationary_sec >= timeout_sec
    return state, anchor_xy, anchor_ts


@dataclass
class RecoverySequenceActionManager:
    """Keep multi-step VLM intent while executing one primitive at a time."""

    sequence_id: int = 0
    original_steps: List[Dict[str, Any]] = field(default_factory=list)
    completed_steps: List[Dict[str, Any]] = field(default_factory=list)
    remaining_steps: List[Dict[str, Any]] = field(default_factory=list)
    in_flight_step: Optional[Dict[str, Any]] = None
    reasoning: Optional[str] = None
    confidence: Optional[float] = None
    last_outcome: Optional[str] = None
    loop1_since: Optional[float] = None

    @property
    def active(self) -> bool:
        return bool(
            self.original_steps
            or self.completed_steps
            or self.remaining_steps
            or self.in_flight_step is not None
        )

    def select(
        self,
        steps: Sequence[Dict[str, Any]],
        *,
        timestamp: Optional[float] = None,
        reasoning: Optional[str],
        confidence: Optional[float],
    ) -> None:
        # Accepted for compatibility with older callers; sequence aging is
        # governed by ``loop1_since`` rather than selection wall time.
        del timestamp
        normalized = [dict(step) for step in steps]
        self.sequence_id += 1
        self.original_steps = [dict(step) for step in normalized]
        self.completed_steps = []
        self.remaining_steps = normalized
        self.in_flight_step = None
        self.reasoning = reasoning
        self.confidence = confidence
        self.last_outcome = "selected"
        self.loop1_since = None

    def start_first(self) -> Optional[Dict[str, Any]]:
        if self.in_flight_step is not None:
            return dict(self.in_flight_step)
        if not self.remaining_steps:
            return None
        self.in_flight_step = dict(self.remaining_steps[0])
        self.last_outcome = "executing"
        self.loop1_since = None
        return dict(self.in_flight_step)

    def finish_current(self, outcome: str, *, completed: bool) -> None:
        current = self.in_flight_step
        if completed and current is not None:
            if self.remaining_steps and self.remaining_steps[0] == current:
                self.remaining_steps.pop(0)
            self.completed_steps.append(dict(current))
        self.in_flight_step = None
        self.last_outcome = str(outcome)
        self.loop1_since = None

    def cancel_loop1_clear(self) -> None:
        self.loop1_since = None

    def observe_loop1(self, timestamp: float, *, clear_after_sec: float) -> bool:
        if not self.active:
            return False
        timestamp = float(timestamp)
        if self.loop1_since is None:
            self.loop1_since = timestamp
        if timestamp - self.loop1_since < max(0.0, float(clear_after_sec)):
            return False
        self.clear()
        return True

    def clear(self) -> None:
        self.original_steps = []
        self.completed_steps = []
        self.remaining_steps = []
        self.in_flight_step = None
        self.reasoning = None
        self.confidence = None
        self.last_outcome = None
        self.loop1_since = None

    def context(self) -> Optional[Dict[str, Any]]:
        if not self.active:
            return None
        status = (
            "executing"
            if self.in_flight_step is not None
            else ("pending" if self.remaining_steps else "completed")
        )
        return {
            "sequence_id": self.sequence_id,
            "status": status,
            "original_steps": [dict(step) for step in self.original_steps],
            "completed_steps": [dict(step) for step in self.completed_steps],
            "remaining_steps": [dict(step) for step in self.remaining_steps],
            "in_flight_step": (
                None
                if self.in_flight_step is None
                else dict(self.in_flight_step)
            ),
            "last_outcome": self.last_outcome,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
        }
