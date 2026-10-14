"""Low-frequency asynchronous semantic gate for recovery takeover.

The high-frequency planner/scorer/verifier loop never waits for this module.
Only two sparse events are eligible to launch a VLM request:

* ``periodic`` -- once per configured interval of simulation time while the
  basic loop is active;
* ``post_recovery`` -- once after a committed recovery hands control back to
  the basic loop.

The VLM is deliberately a *soft* gate.  A rejection must be severe,
high-confidence, fresh, recovery-relevant, and (by default) confirmed by a
second sparse request before it can ask the orchestrator to enter recovery.
PDMS remains the immediate deterministic hard gate.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from agenticdriving.adapters.carla.context import (
    resolve_route_instruction,
    resolve_vlm_camera_order,
)
from agenticdriving.vlm.debug import append_jsonl


LOG = logging.getLogger(__name__)

RECOVERY_REASON_CODES = frozenset(
    {
        "WRONG_TURN",
        "OFF_ROUTE",
        "ROAD_DEPARTURE",
        "WRONG_WAY",
        "UNEXPECTED_ROAD_OBSTACLE",
    }
)


@dataclass(frozen=True)
class SemanticTakeoverConfig:
    enabled: bool = False
    trigger_periodically: bool = True
    trigger_after_recovery: bool = True
    periodic_check_interval_sec: float = 5.0
    post_recovery_delay_sec: float = 3.0
    min_check_interval_sec: float = 5.0
    confirmation_delay_sec: float = 1.5
    require_confirmation: bool = True
    confidence_threshold: float = 0.92
    result_max_age_sec: float = 6.0
    semantic_takeover_cooldown_sec: float = 30.0
    post_recovery_retakeover_min_sec: float = 12.0
    max_semantic_takeovers_per_episode: int = 2
    max_new_tokens: int = 180
    timeout_sec: float = 12.0
    save_debug_artifacts: bool = True
    debug_dir_name: str = "semantic_takeover"
    log_file_name: str = "semantic_takeover.jsonl"


def _instruction_key(info: Dict[str, Any]) -> str:
    command = info.get("command")
    instruction = resolve_route_instruction(info)
    return f"{command!r}:{instruction}"


def _snapshot_value(value: Any) -> Any:
    """Copy mutable observation data without importing simulator types."""
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _snapshot_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_snapshot_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_snapshot_value(item) for item in value)
    return value


class AsyncSemanticTakeoverManager:
    """Latest-only, non-blocking VLM semantic takeover manager."""

    def __init__(
        self,
        cfg: SemanticTakeoverConfig,
        *,
        checker: Any,
        output_dir: Path,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.cfg = cfg
        self.checker = checker
        self.log = logger or LOG
        self.output_dir = Path(output_dir)
        self.debug_dir = self.output_dir / cfg.debug_dir_name
        self.log_path = self.debug_dir / cfg.log_file_name
        if cfg.enabled and cfg.save_debug_artifacts:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._result: Optional[Dict[str, Any]] = None
        self._request: Optional[Dict[str, Any]] = None
        self._generation = 0
        self._last_instruction_key: Optional[str] = None
        self._next_periodic_check_ts: Optional[float] = None
        self._pending_event: Optional[Dict[str, Any]] = None
        self._last_check_ts = float("-inf")
        self._last_semantic_takeover_ts = float("-inf")
        self._last_recovery_end_ts = float("-inf")
        self._semantic_takeovers = 0
        self._confirmation: Optional[Dict[str, Any]] = None
        self._closed = False

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.enabled and self.checker is not None and not self._closed)

    @property
    def busy(self) -> bool:
        thread = self._thread
        return bool(thread is not None and thread.is_alive())

    def _record(self, record: Dict[str, Any]) -> None:
        if not self.cfg.save_debug_artifacts:
            return
        try:
            append_jsonl(self.log_path, record)
        except Exception:
            self.log.exception("Failed to append semantic-takeover log")

    def _schedule(self, trigger: str, due_ts: float, *, confirmation: bool = False) -> None:
        self._generation += 1
        self._pending_event = {
            "trigger": str(trigger),
            "due_ts": float(due_ts),
            "generation": self._generation,
            "confirmation": bool(confirmation),
        }
        self._record(
            {
                "kind": "scheduled",
                "trigger": trigger,
                "due_ts": float(due_ts),
                "generation": self._generation,
                "confirmation": bool(confirmation),
            }
        )

    def observe(self, info: Dict[str, Any], *, recovery_active: bool) -> None:
        """Schedule a sparse check every configured simulation-time interval."""
        if not self.enabled:
            return
        key = _instruction_key(info)
        now = float(info.get("timestamp", 0.0) or 0.0)
        self._last_instruction_key = key
        if not self.cfg.trigger_periodically:
            return

        interval = max(0.0, float(self.cfg.periodic_check_interval_sec))
        if interval <= 0.0:
            return
        if self._next_periodic_check_ts is None:
            self._next_periodic_check_ts = now + interval
            return
        if now < self._next_periodic_check_ts:
            return

        # Keep the cadence tied to simulator time rather than wall-clock VLM
        # latency. Missed intervals are skipped instead of causing a burst.
        missed_intervals = int((now - self._next_periodic_check_ts) // interval) + 1
        self._next_periodic_check_ts += missed_intervals * interval
        if recovery_active or self.busy:
            return

        pending = self._pending_event
        if pending is not None:
            # Never replace a confirmation request. A due sparse request can
            # also run as-is; only a later post-recovery request is superseded
            # so the five-second periodic cadence is not accidentally delayed.
            if pending.get("confirmation", False) or float(pending["due_ts"]) <= now:
                return
        self._confirmation = None
        self._schedule("periodic", now)

    def note_recovery_started(self, timestamp: float, *, source: str) -> None:
        """Invalidate soft decisions as soon as either gate enters recovery."""
        if not self.enabled:
            return
        self._generation += 1
        self._pending_event = None
        with self._lock:
            self._result = None
            self._request = None
        self._confirmation = None
        self._record(
            {
                "kind": "recovery_started",
                "timestamp": float(timestamp),
                "source": str(source),
                "generation": self._generation,
            }
        )

    def note_recovery_ended(self, info: Dict[str, Any]) -> None:
        """Schedule the sparse E check after control returns to loop 1."""
        if not self.enabled:
            return
        now = float(info.get("timestamp", 0.0) or 0.0)
        self._last_recovery_end_ts = now
        self._confirmation = None
        if self.cfg.trigger_after_recovery:
            # E is deliberately later than the nominal handoff delay when the
            # retakeover guard is larger. This avoids spending two VLM calls on
            # a decision that is categorically forbidden until the guard ends.
            delay = max(
                0.0,
                float(self.cfg.post_recovery_delay_sec),
                float(self.cfg.post_recovery_retakeover_min_sec),
            )
            self._schedule(
                "post_recovery",
                now + delay,
            )

    def maybe_submit(
        self,
        *,
        frame_index: int,
        obs: Dict[str, Any],
        info: Dict[str, Any],
        selected_plan: np.ndarray,
        recovery_active: bool,
    ) -> bool:
        """Launch one daemon VLM request if a sparse event is due."""
        if not self.enabled or recovery_active or self.busy:
            return False
        event = self._pending_event
        if event is None:
            return False
        now = float(info.get("timestamp", 0.0) or 0.0)
        if now < float(event["due_ts"]):
            return False
        if not event.get("confirmation", False):
            earliest = self._last_check_ts + max(0.0, float(self.cfg.min_check_interval_sec))
            if now < earliest:
                event["due_ts"] = earliest
                return False
        camera_images = (obs or {}).get("rgb") or {}
        if not camera_images:
            self._record(
                {
                    "kind": "skipped",
                    "trigger": event["trigger"],
                    "timestamp": now,
                    "reason": "no_camera_images",
                }
            )
            self._pending_event = None
            return False

        # Copy only the views the shared checker will actually consume. The
        # snapshot happens on the loop-1 thread for data safety, so keeping a
        # front-only check to one image materially reduces handoff overhead.
        camera_mode = getattr(
            getattr(self.checker, "cfg", None),
            "camera_mode",
            "front_only",
        )
        camera_order = resolve_vlm_camera_order(str(camera_mode))
        selected_images = {
            name: camera_images[name]
            for name in camera_order
            if name in camera_images
        }
        if not selected_images:
            selected_images = dict(camera_images)

        request = {
            **dict(event),
            "frame_index": int(frame_index),
            "snapshot_ts": now,
            "instruction_key": _instruction_key(info),
            "camera_images": _snapshot_value(selected_images),
            "info": _snapshot_value(dict(info)),
            "selected_plan": np.asarray(selected_plan, dtype=np.float32).copy(),
        }
        self._pending_event = None
        self._last_check_ts = now
        with self._lock:
            self._request = request
            self._result = None
        self._thread = threading.Thread(
            target=self._run_request,
            args=(request,),
            name="semantic-takeover-vlm",
            daemon=True,
        )
        self._thread.start()
        self._record(
            {
                "kind": "submitted",
                "trigger": request["trigger"],
                "frame_index": request["frame_index"],
                "snapshot_ts": now,
                "generation": request["generation"],
                "confirmation": request["confirmation"],
            }
        )
        return True

    def _run_request(self, request: Dict[str, Any]) -> None:
        try:
            result = self.checker.check_semantic_takeover(
                frame_index=request["frame_index"],
                camera_images=request["camera_images"],
                info=request["info"],
                selected_plan=request["selected_plan"],
                trigger=str(request["trigger"]),
                max_new_tokens=int(self.cfg.max_new_tokens),
                timeout_sec=float(self.cfg.timeout_sec),
                debug_dir=self.debug_dir,
                save_debug_artifacts=bool(self.cfg.save_debug_artifacts),
            )
        except Exception as exc:
            self.log.exception("Asynchronous semantic takeover check failed")
            result = {
                "semantic_verifier_error": str(exc),
                "semantic_verifier_on_track": True,
            }
        with self._lock:
            # A newer instruction/recovery event may have invalidated this
            # generation.  Keep the result for poll(), which records why it was
            # discarded instead of silently dropping observability.
            self._result = {"request": request, "result": result}

    def poll(
        self,
        info: Dict[str, Any],
        *,
        recovery_active: bool,
    ) -> Optional[Dict[str, Any]]:
        """Consume a completed result; return a rare confirmed takeover."""
        if not self.enabled:
            return None
        with self._lock:
            completed = self._result
            self._result = None
        if completed is None:
            return None
        request = completed["request"]
        result = completed["result"]
        now = float(info.get("timestamp", 0.0) or 0.0)
        disposition = "continue"
        reason = None

        if recovery_active:
            disposition, reason = "discard", "recovery_active"
        elif int(request["generation"]) != int(self._generation):
            disposition, reason = "discard", "stale_generation"
        elif request["instruction_key"] != _instruction_key(info):
            disposition, reason = "discard", "instruction_changed"
        elif now - float(request["snapshot_ts"]) > float(self.cfg.result_max_age_sec):
            disposition, reason = "discard", "result_too_old"
        elif result.get("semantic_verifier_error"):
            disposition, reason = "continue", "vlm_error"
        else:
            on_track = bool(result.get("semantic_verifier_on_track", True))
            confidence = result.get("semantic_verifier_confidence")
            confidence = 0.0 if confidence is None else float(confidence)
            severity = str(result.get("semantic_verifier_severity") or "none").lower()
            reason_code = str(result.get("semantic_verifier_reason_code") or "UNCERTAIN").upper()
            recommended = bool(result.get("semantic_verifier_takeover_recommended", False))
            severe_reject = (
                not on_track
                and recommended
                and severity == "severe"
                and reason_code in RECOVERY_REASON_CODES
                and confidence >= float(self.cfg.confidence_threshold)
            )
            if severe_reject:
                if self.cfg.require_confirmation and not request.get("confirmation", False):
                    self._confirmation = {
                        "reason_code": reason_code,
                        "instruction_key": request["instruction_key"],
                    }
                    self._schedule(
                        f"confirm:{request['trigger']}",
                        now + max(0.0, float(self.cfg.confirmation_delay_sec)),
                        confirmation=True,
                    )
                    disposition, reason = "confirm", "first_severe_reject"
                else:
                    confirmation = self._confirmation
                    same_reason = (
                        confirmation is None
                        or confirmation.get("reason_code") == reason_code
                    )
                    cooldown_ok = (
                        now - self._last_semantic_takeover_ts
                        >= float(self.cfg.semantic_takeover_cooldown_sec)
                    )
                    post_recovery_ok = (
                        now - self._last_recovery_end_ts
                        >= float(self.cfg.post_recovery_retakeover_min_sec)
                    )
                    count_ok = (
                        self._semantic_takeovers
                        < max(0, int(self.cfg.max_semantic_takeovers_per_episode))
                    )
                    if same_reason and cooldown_ok and post_recovery_ok and count_ok:
                        self._semantic_takeovers += 1
                        self._last_semantic_takeover_ts = now
                        self._confirmation = None
                        disposition = "takeover"
                        decision = {
                            "source": "semantic_vlm",
                            "trigger": request["trigger"],
                            "reason_code": reason_code,
                            "reasoning": result.get("semantic_verifier_reasoning"),
                            "confidence": confidence,
                            "snapshot_ts": request["snapshot_ts"],
                            "decision_ts": now,
                            "raw": result,
                        }
                        self._record(
                            {
                                "kind": "result",
                                "disposition": disposition,
                                **decision,
                            }
                        )
                        return decision
                    disposition = "continue"
                    blocked = []
                    if not same_reason:
                        blocked.append("confirmation_reason_changed")
                    if not cooldown_ok:
                        blocked.append("semantic_takeover_cooldown")
                    if not post_recovery_ok:
                        blocked.append("post_recovery_cooldown")
                    if not count_ok:
                        blocked.append("episode_takeover_limit")
                    reason = ",".join(blocked) or "conservative_gate"
                    self._confirmation = None
            else:
                self._confirmation = None
                reason = "not_severe_high_confidence_recovery_mismatch"

        self._record(
            {
                "kind": "result",
                "trigger": request["trigger"],
                "snapshot_ts": request["snapshot_ts"],
                "decision_ts": now,
                "disposition": disposition,
                "reason": reason,
                "result": result,
            }
        )
        return None

    def finalize(self) -> None:
        # The worker is daemonized and may be inside an uninterruptible model
        # call.  Invalidate it and return immediately; never hold evaluator
        # teardown on a soft semantic check.
        self._closed = True
        self._generation += 1
        self._pending_event = None
        self._confirmation = None
