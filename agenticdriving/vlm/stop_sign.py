"""Asynchronous HD-map-triggered stop-sign VLM and braking state machine."""
from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

from agenticdriving.adapters.carla.context import resolve_vlm_camera_order
from agenticdriving.vlm.debug import append_jsonl


LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class StopSignConfig:
    enabled: bool = False
    lookahead_m: float = 30.0
    # Query once at each descending HD-map distance. A negative verdict moves
    # to the next distance instead of ending the junction encounter.
    check_distances_m: Tuple[float, ...] = (30.0, 22.0, 12.0)
    waypoint_step_m: float = 2.0
    confidence_threshold: float = 0.65
    result_max_age_sec: float = 20.0
    stop_speed_threshold_mps: float = 0.1
    hold_sec: float = 0.5
    stop_line_margin_m: float = 0.5
    creep_speed_mps: float = 1.0
    # Begin full braking at target_offset + v*t_reaction + v^2/(2*a). When the
    # HD map exposes an OpenDRIVE type-206 STOP landmark, its lane distance is
    # the target; otherwise the mapped junction boundary is the fallback.
    stop_offset_before_junction_m: float = 0.5
    assumed_brake_decel_mps2: float = 10.0
    brake_reaction_sec: float = 0.1
    min_brake_trigger_distance_m: float = 4.0
    # Once the first check is negative (or still pending this close), limit
    # approach speed so the final close-range query still leaves room to stop.
    negative_speed_cap_start_m: float = 22.0
    negative_speed_cap_mps: float = 6.0
    rearm_clear_sec: float = 3.0
    max_new_tokens: int = 100
    timeout_sec: float = 12.0
    camera_mode: str = "front_only"
    save_debug_artifacts: bool = True
    debug_dir_name: str = "stop_sign"
    log_file_name: str = "stop_sign.jsonl"


@dataclass(frozen=True)
class JunctionObservation:
    key: str
    distance_m: float
    is_current: bool = False
    stop_landmark_distance_m: Optional[float] = None
    # Lane-projected position and forward vector of the physical STOP sign.
    stop_target: Optional[Tuple[float, float, float, float]] = None


def _waypoint_key(waypoint: Any) -> Tuple[object, ...]:
    return (
        getattr(waypoint, "road_id", None),
        getattr(waypoint, "section_id", None),
        getattr(waypoint, "lane_id", None),
        round(float(getattr(waypoint, "s", 0.0) or 0.0), 1),
    )


def _junction_key(waypoint: Any, successors: Iterable[Any] = ()) -> str:
    try:
        junction = waypoint.get_junction()
        junction_id = getattr(junction, "id", None)
        if junction_id is not None:
            return f"junction:{int(junction_id)}"
    except Exception:
        pass
    successor_keys = sorted(str(_waypoint_key(item)) for item in successors)
    if successor_keys:
        return "branch:" + "|".join(successor_keys)
    return "junction_wp:" + str(_waypoint_key(waypoint))


def _upcoming_stop_landmark_distance(
    ego_waypoint: Any,
    *,
    lookahead_m: float,
) -> Tuple[Optional[float], Optional[Tuple[float, float, float, float]]]:
    """Keep the braking distance and locate the physical sign on its lane."""
    try:
        landmarks = ego_waypoint.get_landmarks(
            max(0.0, float(lookahead_m)) + 10.0,
            False,
        )
    except Exception:
        return None, None

    distances = []
    targets = []
    for landmark in landmarks or ():
        if str(getattr(landmark, "type", "")) != "206":
            continue
        try:
            distance = float(getattr(landmark, "distance"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(distance) and distance >= 0.0:
            distances.append(distance)
            # Town13 also labels the upstream painted STOP word as type 206.
            # Its position must not satisfy completion of the physical stop.
            if str(getattr(landmark, "name", "")) == "Sign_Stop":
                transform = landmark.waypoint.transform
                location = transform.location
                forward = transform.get_forward_vector()
                targets.append((distance, (location.x, location.y, forward.x, forward.y)))
    target = min(targets, key=lambda item: item[0])[1] if targets else None
    return (min(distances) if distances else None), target


def find_upcoming_junction(
    ego_waypoint: Any,
    *,
    lookahead_m: float,
    waypoint_step_m: float,
) -> Optional[JunctionObservation]:
    """Find the nearest mapped junction/branch in front of an ego waypoint."""
    if ego_waypoint is None:
        return None
    stop_landmark_distance, stop_target = _upcoming_stop_landmark_distance(
        ego_waypoint,
        lookahead_m=lookahead_m,
    )
    if bool(getattr(ego_waypoint, "is_junction", False)):
        return JunctionObservation(
            _junction_key(ego_waypoint),
            0.0,
            True,
            stop_landmark_distance,
            stop_target,
        )

    step = max(0.5, float(waypoint_step_m))
    max_steps = max(1, int(math.ceil(max(0.0, float(lookahead_m)) / step)))
    frontier = [ego_waypoint]
    seen = {_waypoint_key(ego_waypoint)}
    for step_index in range(1, max_steps + 1):
        next_frontier = []
        for waypoint in frontier:
            try:
                successors = [item for item in waypoint.next(step) if item is not None]
            except Exception:
                successors = []
            if len(successors) > 1:
                junction_successor = next(
                    (item for item in successors if bool(getattr(item, "is_junction", False))),
                    successors[0],
                )
                return JunctionObservation(
                    _junction_key(junction_successor, successors),
                    min(float(lookahead_m), step_index * step),
                    bool(getattr(junction_successor, "is_junction", False)),
                    stop_landmark_distance,
                    stop_target,
                )
            for successor in successors:
                if bool(getattr(successor, "is_junction", False)):
                    return JunctionObservation(
                        _junction_key(successor),
                        min(float(lookahead_m), step_index * step),
                        False,
                        stop_landmark_distance,
                        stop_target,
                    )
                key = _waypoint_key(successor)
                if key not in seen:
                    seen.add(key)
                    next_frontier.append(successor)
        if not next_frontier:
            return None
        # A normal lane has one successor. Bound malformed maps defensively.
        frontier = next_frontier[:16]
    return None


def _snapshot(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    return value


class AsyncStopSignManager:
    """Multi-range VLM checks per mapped junction plus a hard-stop latch."""

    def __init__(
        self,
        cfg: StopSignConfig,
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
        self._completed: Optional[Dict[str, Any]] = None
        self._closed = False
        self._encounter_key: Optional[str] = None
        configured_distances = tuple(
            float(item) for item in self.cfg.check_distances_m
            if float(item) > 0.0
        )
        if not configured_distances:
            configured_distances = (float(self.cfg.lookahead_m),)
        self._check_distances_m = tuple(sorted(set(configured_distances), reverse=True))
        self._next_check_index = 0
        self._active_request_index: Optional[int] = None
        self._negative_results = 0
        self._speed_cap_active = False
        self._clear_since: Optional[float] = None
        self._state = "IDLE"
        self._target_key: Optional[str] = None
        self._stop_target: Optional[Tuple[float, float, float, float]] = None
        self._hold_started_ts: Optional[float] = None
        self._last_result: Optional[Dict[str, Any]] = None

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.enabled and self.checker is not None and not self._closed)

    @property
    def busy(self) -> bool:
        return bool(self._thread is not None and self._thread.is_alive())

    @property
    def state(self) -> str:
        return self._state

    @property
    def braking(self) -> bool:
        return self._state in {"BRAKING", "HOLD"}

    @property
    def speed_cap_mps(self) -> Optional[float]:
        if self._state == "CREEP":
            return float(self.cfg.creep_speed_mps)
        if not self._speed_cap_active:
            return None
        return max(0.0, float(self.cfg.negative_speed_cap_mps))

    @property
    def last_result(self) -> Optional[Dict[str, Any]]:
        return None if self._last_result is None else dict(self._last_result)

    def _record(self, record: Dict[str, Any]) -> None:
        if not self.cfg.save_debug_artifacts:
            return
        try:
            append_jsonl(self.log_path, record)
        except Exception:
            self.log.exception("Failed to append stop-sign log")

    def _reset_encounter(self, junction_key: Optional[str]) -> None:
        self._encounter_key = junction_key
        self._next_check_index = 0
        self._active_request_index = None
        self._negative_results = 0
        self._speed_cap_active = False
        self._state = "IDLE"
        self._target_key = None
        self._stop_target = None
        self._hold_started_ts = None

    def _set_speed_cap(
        self,
        active: bool,
        *,
        now: float,
        junction: Optional[JunctionObservation],
    ) -> None:
        active = bool(active and float(self.cfg.negative_speed_cap_mps) > 0.0)
        if active == self._speed_cap_active:
            return
        self._speed_cap_active = active
        self._record({
            "kind": "speed_cap_started" if active else "speed_cap_released",
            "timestamp": now,
            "junction_key": self._encounter_key,
            "junction_distance_m": None if junction is None else float(junction.distance_m),
            "speed_cap_mps": float(self.cfg.negative_speed_cap_mps),
        })

    def _run_request(self, request: Dict[str, Any]) -> None:
        try:
            result = self.checker.check_stop_sign(
                frame_index=request["frame_index"],
                camera_images=request["camera_images"],
                info=request["info"],
                junction_distance_m=request["junction_distance_m"],
                junction_key=request["junction_key"],
                camera_mode=str(self.cfg.camera_mode),
                max_new_tokens=int(self.cfg.max_new_tokens),
                timeout_sec=float(self.cfg.timeout_sec),
                debug_dir=self.debug_dir,
                save_debug_artifacts=bool(self.cfg.save_debug_artifacts),
            )
        except Exception as exc:
            self.log.exception("Asynchronous stop-sign VLM check failed")
            result = {"stop_sign_error": str(exc), "must_stop": False}
        with self._lock:
            self._completed = {"request": request, "result": result}

    def _submit(
        self,
        *,
        frame_index: int,
        obs: Dict[str, Any],
        info: Dict[str, Any],
        junction: JunctionObservation,
        check_index: int,
    ) -> bool:
        camera_images = (obs or {}).get("rgb") or {}
        camera_order = resolve_vlm_camera_order(self.cfg.camera_mode)
        selected = {
            name: camera_images[name]
            for name in camera_order
            if name in camera_images
        }
        if not selected:
            self._record({
                "kind": "skipped",
                "timestamp": float(info.get("timestamp", 0.0) or 0.0),
                "junction_key": junction.key,
                "reason": "no_camera_images",
            })
            return False
        request = {
            "frame_index": int(frame_index),
            "snapshot_ts": float(info.get("timestamp", 0.0) or 0.0),
            "junction_key": junction.key,
            "junction_distance_m": float(junction.distance_m),
            "check_index": int(check_index),
            "check_distance_m": float(self._check_distances_m[check_index]),
            "camera_images": _snapshot(selected),
            "info": _snapshot(dict(info)),
        }
        with self._lock:
            self._completed = None
        self._thread = threading.Thread(
            target=self._run_request,
            args=(request,),
            name="stop-sign-vlm",
            daemon=True,
        )
        self._thread.start()
        self._active_request_index = int(check_index)
        self._record({"kind": "submitted", **{k: request[k] for k in (
            "frame_index", "snapshot_ts", "junction_key", "junction_distance_m",
            "check_index", "check_distance_m",
        )}})
        return True

    def _poll(self, now: float, junction: Optional[JunctionObservation]) -> None:
        with self._lock:
            completed = self._completed
            self._completed = None
        if completed is None:
            return
        request = completed["request"]
        result = completed["result"]
        check_index = int(request.get("check_index", 0))
        self._active_request_index = None
        error = result.get("stop_sign_error")
        same_encounter = (
            self._encounter_key == request["junction_key"]
            and junction is not None
            and junction.key == request["junction_key"]
        )
        fresh = now - float(request["snapshot_ts"]) <= float(self.cfg.result_max_age_sec)
        confidence = float(result.get("confidence", 0.0) or 0.0)
        must_stop = bool(result.get("must_stop", False))
        accepted = bool(
            same_encounter
            and fresh
            and not error
            and must_stop
            and confidence >= float(self.cfg.confidence_threshold)
        )
        self._last_result = {
            **dict(result),
            "accepted": accepted,
            "junction_key": request["junction_key"],
            "decision_ts": now,
            "check_index": check_index,
            "check_distance_m": request.get("check_distance_m"),
        }
        if accepted:
            self._state = "APPROACH"
            self._target_key = str(request["junction_key"])
            self._stop_target = junction.stop_target
            self.log.warning(
                "Stop-sign VLM accepted at %s (confidence=%.3f); arming hard stop",
                self._target_key,
                confidence,
            )
        else:
            can_retry = bool(
                same_encounter
                and fresh
                and check_index + 1 < len(self._check_distances_m)
            )
            if can_retry:
                self._negative_results += 1
                self._next_check_index = check_index + 1
                self._state = "WAIT_RECHECK"
            else:
                self._state = "DONE"
                self._speed_cap_active = False
        self._record({
            "kind": "result",
            "snapshot_ts": request["snapshot_ts"],
            "decision_ts": now,
            "junction_key": request["junction_key"],
            "accepted": accepted,
            "same_encounter": same_encounter,
            "fresh": fresh,
            "check_index": check_index,
            "check_distance_m": request.get("check_distance_m"),
            "will_retry": self._state == "WAIT_RECHECK",
            "next_check_distance_m": (
                self._check_distances_m[self._next_check_index]
                if self._state == "WAIT_RECHECK"
                else None
            ),
            "result": result,
        })

    def _brake_trigger_distance(self, speed_mps: float) -> float:
        speed = max(0.0, float(speed_mps))
        decel = max(0.1, float(self.cfg.assumed_brake_decel_mps2))
        distance = (
            float(self.cfg.stop_offset_before_junction_m)
            + speed * max(0.0, float(self.cfg.brake_reaction_sec))
            + speed * speed / (2.0 * decel)
        )
        return max(float(self.cfg.min_brake_trigger_distance_m), distance)

    def step(
        self,
        *,
        frame_index: int,
        obs: Dict[str, Any],
        info: Dict[str, Any],
        speed_mps: float,
        junction: Optional[JunctionObservation],
    ) -> bool:
        """Advance detection/control state and return whether full brake is required."""
        if not self.enabled:
            return False
        now = float(info.get("timestamp", 0.0) or 0.0)

        if junction is None:
            self._set_speed_cap(False, now=now, junction=None)
            if self._clear_since is None:
                self._clear_since = now
            elif (
                now - self._clear_since >= max(0.0, float(self.cfg.rearm_clear_sec))
                and self._state in {"IDLE", "WAIT_RECHECK", "DONE"}
                and not self.busy
            ):
                self._reset_encounter(None)
        else:
            self._clear_since = None
            if self._encounter_key != junction.key and self._state not in {"APPROACH", "CREEP", "BRAKING", "HOLD"}:
                self._reset_encounter(junction.key)

        self._poll(now, junction)

        if junction is not None and junction.key == self._encounter_key:
            cap_due_to_pending_first_check = bool(
                self._active_request_index == 0
                and float(junction.distance_m) <= float(self.cfg.negative_speed_cap_start_m)
            )
            cap_due_to_negative = bool(
                self._negative_results > 0
                and self._state in {"WAIT_RECHECK", "APPROACH"}
                and float(junction.distance_m) <= float(self.cfg.negative_speed_cap_start_m)
            )
            self._set_speed_cap(
                cap_due_to_pending_first_check or cap_due_to_negative,
                now=now,
                junction=junction,
            )

            if (
                self._state in {"IDLE", "WAIT_RECHECK"}
                and not self.busy
                and self._next_check_index < len(self._check_distances_m)
                and float(junction.distance_m)
                <= float(self._check_distances_m[self._next_check_index])
            ):
                self._submit(
                    frame_index=frame_index,
                    obs=obs,
                    info=info,
                    junction=junction,
                    check_index=self._next_check_index,
                )

        if self._state == "APPROACH":
            target_visible = junction is not None and junction.key == self._target_key
            trigger_distance = self._brake_trigger_distance(speed_mps)
            landmark_distance = (
                None if junction is None else junction.stop_landmark_distance_m
            )
            target_distance = (
                None
                if junction is None
                else (
                    float(landmark_distance)
                    if landmark_distance is not None
                    else float(junction.distance_m)
                )
            )
            if not target_visible or target_distance <= trigger_distance:
                self._state = "BRAKING"
                self._set_speed_cap(False, now=now, junction=junction)
                self._record({
                    "kind": "braking_started",
                    "timestamp": now,
                    "junction_key": self._target_key,
                    "junction_distance_m": None if junction is None else junction.distance_m,
                    "stop_landmark_distance_m": landmark_distance,
                    "brake_target_distance_m": target_distance,
                    "brake_target_source": (
                        "stop_landmark" if landmark_distance is not None else "junction"
                    ),
                    "speed_mps": float(speed_mps),
                    "trigger_distance_m": trigger_distance,
                })

        front_distance = None
        if self._stop_target is not None:
            x, y, forward_x, forward_y = self._stop_target
            position = info["ego_pos"]
            half_length = float(info["ego_box"][3]) / 2.0
            front_distance = (
                (x - float(position[0])) * forward_x
                + (y - float(position[1])) * forward_y - half_length
            )

        if self._state == "CREEP":
            if front_distance > float(self.cfg.stop_line_margin_m):
                return False
            self._state = "BRAKING"

        if self._state == "BRAKING":
            if float(speed_mps) < float(self.cfg.stop_speed_threshold_mps):
                if front_distance is not None and front_distance > float(self.cfg.stop_line_margin_m):
                    self._state = "CREEP"
                    self._record({
                        "kind": "creep_started", "timestamp": now,
                        "junction_key": self._target_key,
                        "front_distance_m": front_distance,
                        "stop_target": self._stop_target,
                    })
                    return False
                self._state = "HOLD"
                self._hold_started_ts = now
                self._record({
                    "kind": "hold_started",
                    "timestamp": now,
                    "junction_key": self._target_key,
                    "speed_mps": float(speed_mps),
                    "front_distance_m": front_distance,
                })
            return True

        if self._state == "HOLD":
            if float(speed_mps) >= float(self.cfg.stop_speed_threshold_mps):
                self._hold_started_ts = now
                return True
            hold_start = now if self._hold_started_ts is None else self._hold_started_ts
            if now - hold_start < max(0.0, float(self.cfg.hold_sec)):
                return True
            self._state = "DONE"
            self._set_speed_cap(False, now=now, junction=junction)
            self._record({
                "kind": "released",
                "timestamp": now,
                "junction_key": self._target_key,
                "hold_sec": now - hold_start,
            })
        return False

    def finalize(self) -> None:
        self._closed = True
