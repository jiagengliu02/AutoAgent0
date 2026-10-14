"""SparseDriveV2 PID tracking and recovery trajectory control."""
from __future__ import annotations


import numpy as np


def plan_travel_direction(plan_xy: np.ndarray, horizon_idx: Optional[int] = None) -> float:
    """Return +1.0 for forward travel, -1.0 for reverse.

    Sign of the net FORWARD (ego +y) progress. The origin (ego at t=0) is
    prepended so the first segment counts. When ``horizon_idx`` is given, only
    the segments up to and including that index are considered (the portion the
    controller acts on this step); otherwise the whole plan is used.
    """
    xy = np.asarray(plan_xy, dtype=np.float64)[:, :2]
    if len(xy) == 0:
        return 1.0
    pts = np.concatenate([np.zeros((1, 2), dtype=np.float64), xy], axis=0)
    forward = np.diff(pts, axis=0)[:, 1]
    if horizon_idx is not None:
        forward = forward[: int(horizon_idx) + 1]
    return -1.0 if float(np.sum(forward)) < 0.0 else 1.0




def reverse_steer(steer: float, reverse: bool) -> float:
    """Invert the steer command when reversing (pairs with reverse_track_angle)."""
    return -float(steer) if reverse else float(steer)
import os
from collections import deque
from typing import Any, Dict, Optional, Tuple

import carla

import agenticdriving_carla_helper as _helper

LOG = _helper.LOG

# The Route 21 opposing-lane maneuver reverses from its outward steer to a
# single 0.5 s counter-steer phase. CARLA's physical steering response reached
# only about 40% of the bicycle model's intended yaw correction during that
# transition, even though steady-state steering normalization was correct.
# Boost the first same-side, decreasing-magnitude counter-steer. The final
# return counter-steer now keeps its planned angle; an extra straight settle and
# bounded endpoint feedback finish recentering without an opposite-lock swing.
LANE_BYPASS_FIRST_COUNTER_STEER_GAIN = 1.6
# The final return counter-steer is followed by an in-lane straight settle. A
# second 1.6x boost made RouteScenario_1007 overshoot the planned -3 m lateral
# excursion to about -4.8 m, then demand opposite lock at the endpoint. Keep the
# primitive's planned physical angle here and let the bounded settle feedback
# remove the remaining error.
LANE_BYPASS_FINAL_COUNTER_STEER_GAIN = 1.0
# The primitive simulator accelerates toward its 8 m/s target continuously, but
# its exported state rows contain the reached speeds at 0.5 s intervals. Tracking
# those rows literally made the real PID lift at every 1.75 m/s stair step. The
# lane bypass instead tracks the peak speed of its current forward phase.
# Throttle remains physically capped at 1.0; this only removes artificial lifts.
LANE_BYPASS_SETTLE_MAX_STEER = 0.35
# At 20 Hz, do not let endpoint feedback cross the rack centre in one frame.
# RouteScenario_1025 jumped from -0.350 to +0.315 in 50 ms after passing a
# repeated terminal waypoint.  The forward sentinel fixes that target, while
# this slew guard makes the settle phase robust to residual localization/PID
# noise without weakening steering during the actual lane changes.
LANE_BYPASS_SETTLE_MAX_STEER_DELTA = 0.05
# The latched bypass's only reverse phase is its HD-map safety retreat.  CARLA
# testing on Route 69 showed that unrestricted reverse feedback can rotate the
# ego nearly sideways while crossing one lane, blocking the actor it is trying
# to yield to.  Limit normalized rack command while reversing this primitive;
# the map path now spreads the lane blend over two full seconds.
LANE_BYPASS_REVERSE_MAX_STEER = 0.22


def _forward_pedestrian_caution(
    info: Dict[str, Any],
    *,
    stationary_speed_mps: float,
    max_distance_m: float,
    max_lateral_m: float,
    planned_path_local_xy: Optional[np.ndarray] = None,
    proximity_distance_m: float = 20.0,
) -> Optional[Dict[str, Any]]:
    """Nearest pedestrian that should cap forward speed.

    Moving/on-road pedestrians receive the conflict cap. A stationary
    off-road pedestrian is only a lower-severity proximity caution: its
    measured bbox is not velocity-advanced or treated as a future crossing.
    During a lane bypass, the current bbox is also checked against the
    committed path's lateral sweep so turning the ego cannot make a pedestrian
    disappear merely by rotating it outside a fixed ego-heading window.
    """
    if max_distance_m <= 0.0 or max_lateral_m <= 0.0:
        return None
    ego_pos = np.asarray(info.get("ego_pos", ()), dtype=np.float64).reshape(-1)
    ego_rot = np.asarray(info.get("ego_rot", ()), dtype=np.float64).reshape(-1)
    if ego_pos.size < 2 or ego_rot.size < 3:
        return None
    yaw = float(ego_rot[2])
    forward_world = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float64)
    right_world = np.array([-np.sin(yaw), np.cos(yaw)], dtype=np.float64)
    ego_box = np.asarray(info.get("ego_box", ()), dtype=np.float64).reshape(-1)
    ego_half_width = (
        max(0.1, 0.5 * float(ego_box[4])) if ego_box.size > 4 else 1.0
    )
    ego_half_length = (
        max(0.1, 0.5 * float(ego_box[3])) if ego_box.size > 3 else 2.25
    )
    path_bounds = None
    if planned_path_local_xy is not None:
        try:
            path = np.asarray(planned_path_local_xy, dtype=np.float64)
            if path.ndim == 2 and path.shape[1] >= 2:
                path = path[:, :2]
                path = path[np.isfinite(path).all(axis=1)]
                if len(path):
                    path = np.vstack([np.zeros((1, 2), dtype=np.float64), path])
                    path_bounds = (
                        float(path[:, 0].min()),
                        float(path[:, 0].max()),
                        float(path[:, 1].min()),
                        float(path[:, 1].max()),
                    )
        except (TypeError, ValueError):
            path_bounds = None
    nearest = None
    for actor in info.get("privileged_agents") or ():
        if str(actor.get("object_type", "")) != "pedestrian":
            continue
        velocity = np.asarray(
            actor.get("agent_vel_vec") or (0.0, 0.0), dtype=np.float64
        ).reshape(-1)
        if velocity.size < 2 or not np.isfinite(velocity[:2]).all():
            continue
        actor_speed = float(np.linalg.norm(velocity[:2]))
        on_driving_lane = actor.get("agent_on_driving_lane")
        pos = np.asarray(
            actor.get("agent_pos_world", ()), dtype=np.float64
        ).reshape(-1)
        if pos.size < 2 or not np.isfinite(pos[:2]).all():
            continue
        delta = pos[:2] - ego_pos[:2]
        lateral = float(np.dot(delta, right_world))
        forward = float(np.dot(delta, forward_world))
        extent = np.asarray(
            actor.get("agent_extent") or (0.6, 0.6), dtype=np.float64
        ).reshape(-1)
        actor_radius = (
            max(0.1, 0.5 * float(np.linalg.norm(extent[:2])))
            if extent.size > 1 and np.isfinite(extent[:2]).all()
            else 0.45
        )
        lateral_min = lateral - actor_radius
        lateral_max = lateral + actor_radius
        forward_min = forward - actor_radius
        forward_max = forward + actor_radius
        corners = np.asarray(
            actor.get("bbox_world_corners") or (), dtype=np.float64
        )
        if corners.ndim == 2 and corners.shape[1] >= 2 and len(corners):
            corners = corners[np.isfinite(corners[:, :2]).all(axis=1), :2]
            if len(corners):
                corner_delta = corners - ego_pos[:2]
                corner_lateral = corner_delta @ right_world
                corner_forward = corner_delta @ forward_world
                lateral_min = float(corner_lateral.min())
                lateral_max = float(corner_lateral.max())
                forward_min = float(corner_forward.min())
                forward_max = float(corner_forward.max())
        if forward_max <= 0.0 or forward_min > max_distance_m:
            continue
        nominal_lateral_overlap = bool(
            lateral_max >= -max_lateral_m
            and lateral_min <= max_lateral_m
        )
        bbox_in_ego_corridor = bool(
            lateral_max >= -(ego_half_width + 0.3)
            and lateral_min <= ego_half_width + 0.3
        )
        bbox_in_planned_sweep = False
        if path_bounds is not None:
            path_left, path_right, path_back, path_front = path_bounds
            bbox_in_planned_sweep = bool(
                lateral_max >= path_left - ego_half_width - 0.3
                and lateral_min <= path_right + ego_half_width + 0.3
                and forward_max >= path_back - ego_half_length - 0.3
                and forward_min <= path_front + ego_half_length + 0.3
            )
        if not (nominal_lateral_overlap or bbox_in_planned_sweep):
            continue
        moving = bool(actor_speed >= stationary_speed_mps)
        conflict = bool(
            moving
            or on_driving_lane is True
            or bbox_in_ego_corridor
            or bbox_in_planned_sweep
        )
        bbox_dx = (
            0.0
            if lateral_min <= 0.0 <= lateral_max
            else min(abs(lateral_min), abs(lateral_max))
        )
        bbox_dy = max(0.0, forward_min)
        bbox_distance = float(np.hypot(bbox_dx, bbox_dy))
        if not conflict and (
            on_driving_lane is not False
            or bbox_distance > max(0.0, proximity_distance_m)
        ):
            continue
        candidate = {
            "agent_id": str(actor.get("agent_id", "?")),
            "distance_m": bbox_distance,
            "forward_m": forward,
            "lateral_m": lateral,
            "speed_mps": actor_speed,
            "on_driving_lane": on_driving_lane,
            "bbox_in_ego_corridor": bbox_in_ego_corridor,
            "bbox_in_planned_sweep": bbox_in_planned_sweep,
            "risk_level": "conflict" if conflict else "proximity",
        }
        if nearest is None or (
            (candidate["risk_level"] == "conflict")
            > (nearest["risk_level"] == "conflict")
        ) or (
            candidate["risk_level"] == nearest["risk_level"]
            and bbox_distance < float(nearest["distance_m"])
        ):
            nearest = candidate
    return nearest


def _apply_pedestrian_caution_speed_cap(
    desired_speed: float,
    caution: Optional[Dict[str, Any]],
    speed_cap_mps: float,
    proximity_speed_cap_mps: Optional[float] = None,
) -> Tuple[float, bool]:
    """Apply the soft pedestrian cap and report whether it changed the target."""
    target = max(0.0, float(desired_speed))
    cap = max(0.0, float(speed_cap_mps))
    if (
        caution is not None
        and caution.get("risk_level") == "proximity"
        and proximity_speed_cap_mps is not None
    ):
        cap = max(cap, float(proximity_speed_cap_mps))
    if caution is None or cap <= 0.0 or target <= cap:
        return target, False
    return cap, True


def _apply_vru_speed_policy(
    desired_speed: float,
    *,
    reverse: bool,
    caution: Optional[Dict[str, Any]],
    speed_cap_mps: float,
    proximity_speed_cap_mps: Optional[float] = None,
    bbox_stop_distance_m: float = 0.0,
    zeroed_verifier_metrics: Tuple[str, ...],
) -> Tuple[float, bool, bool]:
    """Apply pedestrian caution or an emergency current-bbox/TTC stop.

    Returns ``(target_speed, caution_cap_applied, emergency_stop_applied)``.
    A hard pedestrian TTC veto always wins. For forward motion, a close
    pedestrian whose *measured current bbox* overlaps the ego corridor also
    commands a stop. This has no velocity extrapolation and releases as soon as
    the observed bbox leaves the corridor. Reverse recovery remains available
    to create clearance from a pedestrian in front of the vehicle.
    """
    target = max(0.0, float(desired_speed))
    if "ttc_vru" in zeroed_verifier_metrics:
        return 0.0, False, True
    if reverse:
        return target, False, False
    if (
        caution is not None
        and caution.get("risk_level") == "conflict"
        and caution.get("bbox_in_ego_corridor") is True
        and float(caution.get("distance_m", float("inf")))
        <= max(0.0, float(bbox_stop_distance_m))
    ):
        return 0.0, False, True
    target, capped = _apply_pedestrian_caution_speed_cap(
        target,
        caution,
        speed_cap_mps,
        proximity_speed_cap_mps,
    )
    return target, capped, False


def _obb_overlap_local(
    first_center: np.ndarray,
    first_heading: np.ndarray,
    first_half_length: float,
    first_half_width: float,
    second_center: np.ndarray,
    second_heading: np.ndarray,
    second_half_length: float,
    second_half_width: float,
) -> bool:
    """Separating-axis overlap for two ego-local oriented rectangles."""
    first_lateral = np.array(
        [-first_heading[1], first_heading[0]], dtype=np.float64
    )
    second_lateral = np.array(
        [-second_heading[1], second_heading[0]], dtype=np.float64
    )
    delta = second_center - first_center
    for axis in (
        first_heading,
        first_lateral,
        second_heading,
        second_lateral,
    ):
        distance = abs(float(np.dot(delta, axis)))
        first_radius = (
            first_half_length * abs(float(np.dot(first_heading, axis)))
            + first_half_width * abs(float(np.dot(first_lateral, axis)))
        )
        second_radius = (
            second_half_length * abs(float(np.dot(second_heading, axis)))
            + second_half_width * abs(float(np.dot(second_lateral, axis)))
        )
        if distance > first_radius + second_radius + 1e-6:
            return False
    return True


def _planned_path_conflicts_vehicle(
    planned_path_local_xy: np.ndarray,
    *,
    actor_center_local: np.ndarray,
    actor_velocity_local: np.ndarray,
    actor_heading_local: np.ndarray,
    actor_half_length: float,
    actor_half_width: float,
    ego_half_length: float,
    ego_half_width: float,
    plan_dt_sec: float,
    horizon_sec: float,
) -> bool:
    """Whether the near-term planned ego footprint reaches one vehicle bbox."""
    path = np.asarray(planned_path_local_xy, dtype=np.float64)
    if (
        path.ndim != 2
        or len(path) == 0
        or path.shape[1] < 2
        or not np.isfinite(path[:, :2]).all()
    ):
        return True
    dt = max(1e-3, float(plan_dt_sec))
    horizon = max(dt, float(horizon_sec))
    row_count = min(len(path), max(1, int(np.ceil(horizon / dt))))
    points = np.vstack(
        [np.zeros((1, 2), dtype=np.float64), path[:row_count, :2]]
    )
    previous_heading = np.array([0.0, 1.0], dtype=np.float64)
    for row_index, (start, end) in enumerate(zip(points[:-1], points[1:])):
        delta = end - start
        distance = float(np.linalg.norm(delta))
        if distance > 1e-6:
            previous_heading = delta / distance
        steps = max(1, int(np.ceil(distance / 0.25)))
        start_time = row_index * dt
        for sample_index in range(1, steps + 1):
            alpha = sample_index / steps
            sample_time = min(horizon, start_time + alpha * dt)
            ego_center = start + alpha * delta
            actor_center = (
                actor_center_local + actor_velocity_local * sample_time
            )
            if _obb_overlap_local(
                ego_center,
                previous_heading,
                ego_half_length,
                ego_half_width,
                actor_center,
                actor_heading_local,
                actor_half_length,
                actor_half_width,
            ):
                return True
    return False


def _forward_vehicle_bbox_guard(
    info: Dict[str, Any],
    *,
    max_clearance_m: float,
    planned_path_local_xy: Optional[np.ndarray] = None,
    plan_dt_sec: float = 0.5,
    path_horizon_sec: float = 1.0,
) -> Optional[Dict[str, Any]]:
    """Nearest current front-vehicle bbox inside the ego body corridor.

    This is deliberately instantaneous geometry. It does not project the
    vehicle forward with TTC or assume an acceleration profile. Oncoming
    actors are excluded using their measured longitudinal velocity; stationary
    and same-direction vehicles remain eligible.
    """
    if max_clearance_m <= 0.0:
        return None
    ego_pos = np.asarray(info.get("ego_pos", ()), dtype=np.float64).reshape(-1)
    ego_rot = np.asarray(info.get("ego_rot", ()), dtype=np.float64).reshape(-1)
    if ego_pos.size < 2 or ego_rot.size < 3:
        return None
    yaw = float(ego_rot[2])
    forward_world = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float64)
    right_world = np.array([-np.sin(yaw), np.cos(yaw)], dtype=np.float64)
    ego_box = np.asarray(info.get("ego_box", ()), dtype=np.float64).reshape(-1)
    ego_half_width = (
        max(0.1, 0.5 * float(ego_box[4])) if ego_box.size > 4 else 1.0
    )
    ego_half_length = (
        max(0.1, 0.5 * float(ego_box[3])) if ego_box.size > 3 else 2.25
    )
    nearest = None
    now = float(info.get("timestamp", 0.0) or 0.0)
    for actor in info.get("privileged_agents") or ():
        if str(actor.get("object_type", "")) != "vehicle":
            continue
        if actor.get("vehicle_bbox_stop_radar_required"):
            confirmation_times = []
            if actor.get("radar_bbox_confirmed"):
                confirmation_times.append(actor.get("radar_bbox_confirmation_ts"))
            if actor.get("lidar_bbox_confirmed"):
                confirmation_times.append(actor.get("lidar_bbox_confirmation_ts"))
            confirmation_times = [
                float(value) for value in confirmation_times if value is not None
            ]
            if not confirmation_times:
                continue
            confirmation_ts = max(confirmation_times)
            max_confirmation_age = float(
                actor.get("radar_bbox_confirmation_max_age_sec", 0.8)
            )
            if now - confirmation_ts > max_confirmation_age:
                continue
        velocity = np.asarray(
            actor.get("agent_vel_vec") or (0.0, 0.0), dtype=np.float64
        ).reshape(-1)
        if velocity.size < 2 or not np.isfinite(velocity[:2]).all():
            continue
        longitudinal_speed = float(np.dot(velocity[:2], forward_world))
        lateral_speed = float(np.dot(velocity[:2], right_world))
        if longitudinal_speed < -1.0:
            continue
        pos = np.asarray(
            actor.get("agent_pos_world", ()), dtype=np.float64
        ).reshape(-1)
        if pos.size < 2 or not np.isfinite(pos[:2]).all():
            continue
        delta = pos[:2] - ego_pos[:2]
        lateral = float(np.dot(delta, right_world))
        forward = float(np.dot(delta, forward_world))
        extent = np.asarray(
            actor.get("agent_extent") or (4.5, 2.0), dtype=np.float64
        ).reshape(-1)
        actor_half_length = (
            max(0.1, 0.5 * float(extent[0]))
            if extent.size > 0 and np.isfinite(extent[0])
            else 2.25
        )
        actor_half_width = (
            max(0.1, 0.5 * float(extent[1]))
            if extent.size > 1 and np.isfinite(extent[1])
            else 1.0
        )
        lateral_min = lateral - actor_half_width
        lateral_max = lateral + actor_half_width
        forward_min = forward - actor_half_length
        forward_max = forward + actor_half_length
        corners = np.asarray(
            actor.get("bbox_world_corners") or (), dtype=np.float64
        )
        if corners.ndim == 2 and corners.shape[1] >= 2 and len(corners):
            corners = corners[np.isfinite(corners[:, :2]).all(axis=1), :2]
            if len(corners):
                corner_delta = corners - ego_pos[:2]
                lateral_values = corner_delta @ right_world
                forward_values = corner_delta @ forward_world
                lateral_min = float(lateral_values.min())
                lateral_max = float(lateral_values.max())
                forward_min = float(forward_values.min())
                forward_max = float(forward_values.max())
        if forward_max <= ego_half_length:
            continue
        bbox_in_ego_corridor = bool(
            lateral_max >= -(ego_half_width + 0.2)
            and lateral_min <= ego_half_width + 0.2
        )
        if not bbox_in_ego_corridor:
            continue
        clearance = max(0.0, forward_min - ego_half_length)
        if clearance > max_clearance_m:
            continue
        actor_heading = np.array([0.0, 1.0], dtype=np.float64)
        try:
            heading_world = float(actor.get("agent_heading"))
            heading_vector_world = np.array(
                [np.cos(heading_world), np.sin(heading_world)],
                dtype=np.float64,
            )
            actor_heading = np.array(
                [
                    float(np.dot(heading_vector_world, right_world)),
                    float(np.dot(heading_vector_world, forward_world)),
                ],
                dtype=np.float64,
            )
            heading_norm = float(np.linalg.norm(actor_heading))
            if heading_norm > 1e-6:
                actor_heading /= heading_norm
            else:
                actor_heading = np.array([0.0, 1.0], dtype=np.float64)
        except (TypeError, ValueError):
            pass
        if planned_path_local_xy is not None and not _planned_path_conflicts_vehicle(
            planned_path_local_xy,
            actor_center_local=np.array([lateral, forward], dtype=np.float64),
            actor_velocity_local=np.array(
                [lateral_speed, longitudinal_speed], dtype=np.float64
            ),
            actor_heading_local=actor_heading,
            actor_half_length=actor_half_length,
            actor_half_width=actor_half_width,
            ego_half_length=ego_half_length,
            ego_half_width=ego_half_width,
            plan_dt_sec=plan_dt_sec,
            horizon_sec=path_horizon_sec,
        ):
            continue
        candidate = {
            "agent_id": str(actor.get("agent_id", "?")),
            "clearance_m": float(clearance),
            "forward_m": forward,
            "lateral_m": lateral,
            "longitudinal_speed_mps": longitudinal_speed,
            "bbox_in_ego_corridor": True,
            "planned_path_conflict": planned_path_local_xy is not None,
        }
        if nearest is None or clearance < float(nearest["clearance_m"]):
            nearest = candidate
    return nearest


def _apply_vehicle_bbox_stop_policy(
    desired_speed: float,
    *,
    reverse: bool,
    guard: Optional[Dict[str, Any]],
) -> Tuple[float, bool]:
    """Stop forward creep only while a measured front-vehicle bbox is close."""
    target = max(0.0, float(desired_speed))
    if reverse or guard is None:
        return target, False
    return 0.0, True


def _normalized_heading_error(aim: np.ndarray) -> float:
    """Return signed ego-heading error for ``[x_right, y_forward]``.

    ``atan2(x, y)`` uses the forward axis as zero and, unlike the historical
    ``pi/2 - atan2(y, x)`` expression, keeps left-rear targets on the negative
    side of the steering controller instead of wrapping them toward +3.
    SparseDrive's PID is calibrated for a normalized quarter-turn error, so
    rearward targets saturate at the appropriate steering lock.
    """
    point = np.asarray(aim, dtype=np.float64).reshape(-1)
    if point.size < 2 or not np.isfinite(point[:2]).all():
        return 0.0
    angle = np.arctan2(point[0], point[1]) / (np.pi / 2.0)
    return float(np.clip(angle, -1.0, 1.0))


def _blend_recovery_steer(
    feedback: float,
    feedforward: float,
    heading_error: float,
    *,
    conflict_error_threshold: float = 0.08,
    allow_feedback_override: bool = True,
) -> float:
    """Blend primitive curvature with PID while preserving corrective authority.

    Feed-forward remains dominant while the primitive is tracking normally.
    Once a material heading error asks for the opposite steering direction,
    however, continuing to apply 90% feed-forward makes the error unrecoverable.
    In that case use feedback alone until the trajectory is recaptured.
    """
    feedback = float(np.clip(feedback, -1.0, 1.0))
    feedforward = float(np.clip(feedforward, -1.0, 1.0))
    conflict = feedback * feedforward < 0.0
    if (
        allow_feedback_override
        and conflict
        and abs(float(heading_error)) >= conflict_error_threshold
    ):
        return feedback
    return float(np.clip(0.1 * feedback + 0.9 * feedforward, -1.0, 1.0))


class PID(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D

        self._window = deque([0 for _ in range(n)], maxlen=n)
        self._max = 0.0
        self._min = 0.0

    def step(self, error):
        self._window.append(error)
        self._max = max(self._max, abs(error))
        self._min = -abs(self._max)

        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = self._window[-1] - self._window[-2]
        else:
            integral = 0.0
            derivative = 0.0

        return self._K_P * error + self._K_I * integral + self._K_D * derivative


def _zero_collision_verifier_metrics(info) -> Tuple[str, ...]:
    """Return collision-safety metrics that explicitly veto this frame."""
    metrics = info.get("verifier_metrics") or {}
    zeroed = []
    for name in ("nc", "ttc", "ttc_vru"):
        value = metrics.get(name)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric) and numeric <= 0.0:
            zeroed.append(name)
    return tuple(zeroed)


class SparseDriveV2PIDController(object):
    """Verbatim SparseDriveV2 PID gains/logic; numpy I/O instead of torch."""

    def __init__(
        self,
        turn_KP=1.0,
        turn_KI=0.75,
        turn_KD=0.4,
        turn_n=10,
        speed_KP=5.0,
        speed_KI=0.5,
        speed_KD=1.0,
        speed_n=10,
        alpha=0.5,
        beta=2.5,
        min_aim_dis=4.0,
        max_aim_dis=8.0,
        max_throttle=1.0,
        brake_speed=0.2,
        brake_ratio=1.1,
        clip_delta=1.0,
        speed_lookahead_sec=1.5,
        lane_change_speed_lookahead_sec=1.5,
        lane_change_accel_hold_sec=0.0,
        lane_change_brake_ratio=1.1,
        max_accel_mps2=2.4,
        max_brake_mps2=4.0,
        target_speed_max_drop_mps2=0.0,
        model_speed_gain=1.0,
        model_speed_gain_max_mps=20.0,
        model_speed_gain_min_mps=5.5,
        pedestrian_caution_speed_mps=0.0,
        pedestrian_proximity_speed_mps=7.0,
        pedestrian_proximity_distance_m=20.0,
        pedestrian_bbox_stop_distance_m=8.0,
        vehicle_bbox_stop_clearance_m=3.0,
        pedestrian_caution_stationary_speed_mps=0.3,
        pedestrian_caution_distance_m=30.0,
        pedestrian_caution_lateral_m=6.0,
    ):
        self.turn_controller = PID(K_P=turn_KP, K_I=turn_KI, K_D=turn_KD, n=turn_n)
        self.speed_controller = PID(K_P=speed_KP, K_I=speed_KI, K_D=speed_KD, n=speed_n)
        self.alpha = alpha
        self.beta = beta
        self.min_aim_dis = min_aim_dis
        self.max_aim_dis = max_aim_dis
        self.max_throttle = max_throttle
        self.brake_speed = brake_speed
        self.brake_ratio = brake_ratio
        self.clip_delta = clip_delta
        # Track a short future model speed, then clamp it to what the current
        # vehicle state can physically reach over that horizon. This preserves a
        # positive launch target at standstill without jumping directly to a
        # distant future speed when the immediate trajectory asks to slow down.
        self.speed_lookahead_sec = float(speed_lookahead_sec)
        # Loop-1 lane-change commands often expose a conservative immediate
        # longitudinal head followed by a faster speed later in the same model
        # sequence. Track that later, still model-predicted speed while the HD-map
        # re-entry state or the explicit RoadOption says a lane change is active.
        self.lane_change_speed_lookahead_sec = float(
            lane_change_speed_lookahead_sec
        )
        self.lane_change_accel_hold_sec = max(
            0.0, float(lane_change_accel_hold_sec)
        )
        self.lane_change_brake_ratio = max(
            float(brake_ratio), float(lane_change_brake_ratio)
        )
        self.max_accel_mps2 = float(max_accel_mps2)
        self.max_brake_mps2 = float(max_brake_mps2)
        # Optional cross-frame slew limit for learned/model candidate targets.
        # A single noisy longitudinal head must not turn a 20 Hz replan into a
        # full-brake pulse. Explicit stop targets and recovery phases bypass the
        # filter so safety stops retain their original response time.
        self.target_speed_max_drop_mps2 = max(
            0.0, float(target_speed_max_drop_mps2)
        )
        # Multiplicative gain applied to the model's cruising target speed only
        # (model_candidate:* sources). Purely a domain-adaptation knob for the
        # Bench2Drive planner's conservative speed prior: it scales UP a positive
        # cruise target, but 0 (an explicit stop) stays 0, so safety stops for
        # pedestrians/obstacles are preserved. Default 1.0 == no change.
        self.model_speed_gain = max(0.0, float(model_speed_gain))
        self.model_speed_gain_max_mps = max(0.0, float(model_speed_gain_max_mps))
        # Gate: only amplify when the model's OWN target already asks for at least
        # this speed. A low model target means the planner is being cautious
        # (construction/clutter/narrow), so we leave it alone to avoid over-driving
        # into static layout; a high target means open road, safe to speed up.
        self.model_speed_gain_min_mps = max(0.0, float(model_speed_gain_min_mps))
        # Moving/on-road/current-path pedestrians use the conflict cap. A
        # stationary off-road bbox uses the milder proximity cap; no velocity
        # is invented for that actor.
        self.pedestrian_caution_speed_mps = max(
            0.0, float(pedestrian_caution_speed_mps)
        )
        self.pedestrian_proximity_speed_mps = max(
            self.pedestrian_caution_speed_mps,
            float(pedestrian_proximity_speed_mps),
        )
        self.pedestrian_proximity_distance_m = max(
            0.0, float(pedestrian_proximity_distance_m)
        )
        self.pedestrian_bbox_stop_distance_m = max(
            0.0, float(pedestrian_bbox_stop_distance_m)
        )
        self.vehicle_bbox_stop_clearance_m = max(
            0.0, float(vehicle_bbox_stop_clearance_m)
        )
        self.pedestrian_caution_stationary_speed_mps = max(
            0.0, float(pedestrian_caution_stationary_speed_mps)
        )
        self.pedestrian_caution_distance_m = max(
            0.0, float(pedestrian_caution_distance_m)
        )
        self.pedestrian_caution_lateral_m = max(
            0.0, float(pedestrian_caution_lateral_m)
        )
        self._last_model_target_speed = None
        self._last_model_target_timestamp = None

    def reset_model_target_speed_filter(self) -> None:
        """Forget model-target history when control leaves the learned planner."""
        self._last_model_target_speed = None
        self._last_model_target_timestamp = None

    def filter_model_target_speed(
        self,
        desired_speed: float,
        timestamp: float,
        *,
        bypass: bool = False,
    ) -> float:
        """Limit only cross-frame target-speed drops.

        Acceleration remains unfiltered because ``_bounded_model_target_speed``
        already caps it by physical reachability. A target at or below the
        controller's explicit stop threshold bypasses the limiter. Safety
        callers may also bypass it explicitly; doing so clears the prior high
        target so it cannot leak back in on the next frame.
        """
        target = max(0.0, float(desired_speed))
        now = float(timestamp)
        if bypass:
            self.reset_model_target_speed_filter()
            return target
        previous = self._last_model_target_speed
        previous_ts = self._last_model_target_timestamp

        if (
            self.target_speed_max_drop_mps2 <= 0.0
            or target <= self.brake_speed
            or previous is None
            or previous_ts is None
            or not np.isfinite(now)
            or now <= float(previous_ts)
        ):
            filtered = target
        else:
            max_drop = self.target_speed_max_drop_mps2 * (
                now - float(previous_ts)
            )
            filtered = max(target, float(previous) - max_drop)

        self._last_model_target_speed = float(filtered)
        self._last_model_target_timestamp = now
        return float(filtered)

    def control_pid(
        self,
        waypoints,
        desired_speed,
        speed,
        *,
        allow_steer_while_braking=False,
        brake_ratio=None,
    ):
        """Predicts vehicle control with the exact SparseDriveV2 PID logic.

        Args:
            waypoints: ``lat_reg_final`` [N, 2] in ego ``[x_right, y_forward]``.
            desired_speed: ``lon_reg_final`` target speed (m/s).
            speed: current speed (m/s).
        """
        desired_speed = np.float64(desired_speed)
        speed = np.float64(speed)

        active_brake_ratio = (
            self.brake_ratio if brake_ratio is None else float(brake_ratio)
        )
        # brake
        brake = desired_speed < self.brake_speed or (speed / desired_speed) > active_brake_ratio
        # throttle
        delta = np.clip(desired_speed - speed, 0.0, self.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self.max_throttle)
        throttle = throttle if not brake else 0.0

        # angle: adaptive look-ahead aim point over the predicted waypoints
        aim_dist = np.clip(self.alpha * speed + self.beta, self.min_aim_dis, self.max_aim_dis)
        norms = np.linalg.norm(waypoints[:-1], axis=1)
        closest_index = np.abs(norms - aim_dist).argmin()
        aim = waypoints[closest_index]  # aim location
        # steer
        angle = _normalized_heading_error(aim)
        if speed < 0.01:
            # When we don't move we don't want the angle error to accumulate in
            # the integral.
            angle = 0.0
        if brake and not (
            allow_steer_while_braking and desired_speed >= self.brake_speed
        ):
            angle = 0.0

        steer = self.turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)

        metadata = {
            "speed": float(speed),
            "steer": float(steer),
            "throttle": float(throttle),
            "brake": float(brake),
            "aim": tuple(np.asarray(aim, dtype=np.float64)),
            "aim_dist": float(aim_dist),
            "desired_speed": float(desired_speed),
            "brake_ratio": float(active_brake_ratio),
            "angle": float(angle),
        }
        return steer, throttle, brake, metadata


def _get_controller(agent) -> SparseDriveV2PIDController:
    controller = getattr(agent, "_sparsedrivev2_pid_controller", None)
    if controller is None:
        cfg = _helper._control_cfg(agent)
        controller = SparseDriveV2PIDController(
            brake_ratio=_helper._cfg_float(cfg, "brake_ratio", 1.1),
            speed_lookahead_sec=_helper._cfg_float(cfg, "speed_lookahead_sec", 1.5),
            lane_change_speed_lookahead_sec=_helper._cfg_float(
                cfg, "lane_change_speed_lookahead_sec", 1.5
            ),
            lane_change_accel_hold_sec=_helper._cfg_float(
                cfg, "lane_change_accel_hold_sec", 0.0
            ),
            lane_change_brake_ratio=_helper._cfg_float(
                cfg, "lane_change_brake_ratio", 1.1
            ),
            max_accel_mps2=_helper._cfg_float(cfg, "max_accel_mps2", 2.4),
            max_brake_mps2=_helper._cfg_float(cfg, "max_brake_mps2", 4.0),
            target_speed_max_drop_mps2=_helper._cfg_float(
                cfg, "target_speed_max_drop_mps2", 0.0
            ),
            model_speed_gain=_helper._cfg_float(
                cfg,
                "model_speed_gain",
                float(os.environ.get("AGENTICDRIVING_SD_SPEED_GAIN", "1.0") or 1.0),
            ),
            model_speed_gain_max_mps=_helper._cfg_float(
                cfg, "model_speed_gain_max_mps", 20.0
            ),
            model_speed_gain_min_mps=_helper._cfg_float(
                cfg,
                "model_speed_gain_min_mps",
                float(os.environ.get("AGENTICDRIVING_SD_SPEED_GAIN_MIN", "5.5") or 5.5),
            ),
            pedestrian_caution_speed_mps=_helper._cfg_float(
                cfg, "pedestrian_caution_speed_mps", 0.0
            ),
            pedestrian_proximity_speed_mps=_helper._cfg_float(
                cfg, "pedestrian_proximity_speed_mps", 7.0
            ),
            pedestrian_proximity_distance_m=_helper._cfg_float(
                cfg, "pedestrian_proximity_distance_m", 20.0
            ),
            pedestrian_bbox_stop_distance_m=_helper._cfg_float(
                cfg, "pedestrian_bbox_stop_distance_m", 8.0
            ),
            vehicle_bbox_stop_clearance_m=_helper._cfg_float(
                cfg, "vehicle_bbox_stop_clearance_m", 3.0
            ),
            pedestrian_caution_stationary_speed_mps=_helper._cfg_float(
                cfg, "pedestrian_caution_stationary_speed_mps", 0.3
            ),
            pedestrian_caution_distance_m=_helper._cfg_float(
                cfg, "pedestrian_caution_distance_m", 30.0
            ),
            pedestrian_caution_lateral_m=_helper._cfg_float(
                cfg, "pedestrian_caution_lateral_m", 6.0
            ),
        )
        agent._sparsedrivev2_pid_controller = controller
        if controller.model_speed_gain != 1.0:
            LOG.info(
                "sparsedrivev2_pid: model_speed_gain=%.2f (gated: only when model "
                "target >= %.2f m/s, cap %.1f m/s)",
                controller.model_speed_gain,
                controller.model_speed_gain_min_mps,
                controller.model_speed_gain_max_mps,
            )
        LOG.info(
            "sparsedrivev2_pid: enabled (SparseDriveV2 defaults: turn=1.0/0.75/0.40, "
            "speed=5.0/0.5/1.0, aim=clip(0.5*v+2.5, 4, 8) m, max_throttle=1.0, "
            "speed_lookahead=%.2fs lane_change_lookahead=%.2fs hold=%.2fs "
            "accel=%.2f brake=%.2f m/s^2 target_drop=%.2f m/s^2 "
            "brake_ratio=%.2f/%.2f)",
            controller.speed_lookahead_sec,
            controller.lane_change_speed_lookahead_sec,
            controller.lane_change_accel_hold_sec,
            controller.max_accel_mps2,
            controller.max_brake_mps2,
            controller.target_speed_max_drop_mps2,
            controller.brake_ratio,
            controller.lane_change_brake_ratio,
        )
    return controller


def _planner_debug(agent) -> Dict[str, Any]:
    debug = getattr(getattr(agent, "_sparsedrive", None), "last_debug", None)
    return debug if isinstance(debug, dict) else {}


def _cached_elapsed_sec(agent) -> float:
    """Simulation time elapsed since the cached plan's first control tick."""
    controls = max(
        1, int(getattr(agent, "_sparsedrivev2_controls_since_replan", 1))
    )
    frequency = max(
        1e-6, float(getattr(agent, "_sparsedrivev2_control_frequency", 20.0))
    )
    return float(controls - 1) / frequency


def _reproject_cached_model_waypoints(
    agent, waypoints: np.ndarray, info: Dict[str, Any]
) -> Optional[np.ndarray]:
    """Express cached model waypoints in the current ego frame.

    ``lat_reg_final`` is emitted in the ego frame of the inference tick.  The
    async reuse path advances the vehicle before calling the PID again, so using
    that array unchanged would make the controller steer toward an old frame.
    Reuse the same world-frame reprojection and trimming policy as the selected
    cached plan.
    """
    cached_pose = getattr(agent, "_sparsedrivev2_cached_selected_pose", None)
    if cached_pose is None:
        return None
    projected = _helper._transform_cached_local_plan(
        agent,
        np.asarray(waypoints, dtype=np.float32),
        cached_pose,
        info,
        planner_name="SparseDriveV2 PID targets",
    )
    projected = _helper._trim_cached_local_plan(
        agent, projected, planner_key="sparsedrivev2"
    )
    projected = np.asarray(projected, dtype=np.float64)
    if projected.ndim != 2 or projected.shape[0] < 2 or projected.shape[1] < 2:
        return None
    return projected[:, :2]


def _valid_waypoints(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        return None
    arr = arr[:, :2]
    return arr if np.isfinite(arr).all() else None


def _valid_speed_sequence(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64).ravel()
    if arr.size == 0 or not np.isfinite(arr).all():
        return None
    return arr


def _matching_model_candidate(
    debug: Dict[str, Any],
    selected_plan: np.ndarray,
    preferred_idx: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[int]]:
    """Return model heads aligned with ``selected_plan``, if it is top-K.

    SparseDriveV2's decoder emits a separate lateral anchor and longitudinal
    speed sequence for every flattened (lat, lon) candidate.  The inference
    adapter stores those heads in the same order as ``candidate_plans_xy``.
    Match by trajectory values because the generic orchestrator deliberately
    passes only the selected plan through its planner-agnostic control seam.
    """
    selected = _valid_waypoints(selected_plan)
    if selected is None:
        return None, None, None

    plans = debug.get("candidate_plans_xy")
    lateral = debug.get("candidate_lat_reg")
    speeds = debug.get("candidate_lon_reg_speed_mps")
    if plans is not None and lateral is not None and speeds is not None:
        try:
            plan_arr = np.asarray(plans, dtype=np.float64)
            if plan_arr.ndim == 3 and len(plan_arr) == len(lateral) == len(speeds):
                indices = list(range(len(plan_arr)))
                if preferred_idx is not None and 0 <= int(preferred_idx) < len(plan_arr):
                    preferred_idx = int(preferred_idx)
                    indices = [preferred_idx] + [i for i in indices if i != preferred_idx]
                for idx in indices:
                    candidate = plan_arr[idx]
                    candidate_xy = _valid_waypoints(candidate)
                    if (
                        candidate_xy is not None
                        and candidate_xy.shape == selected.shape
                        and np.allclose(candidate_xy, selected, rtol=0.0, atol=1e-4)
                    ):
                        lat = _valid_waypoints(lateral[idx])
                        lon = _valid_speed_sequence(speeds[idx])
                        if lat is not None and lon is not None:
                            return lat, lon, int(idx)
        except (TypeError, ValueError):
            pass

    # Compatibility with debug payloads produced before candidate-aligned heads
    # were added. This preserves exact upstream behaviour for the native argmax.
    native_plan = debug.get("carla_local_plan_xy") or debug.get("traj_final_xy")
    native = _valid_waypoints(native_plan)
    if (
        native is not None
        and native.shape == selected.shape
        and np.allclose(native, selected, rtol=0.0, atol=1e-4)
    ):
        lat = _valid_waypoints(debug.get("lat_reg_final"))
        lon = _valid_speed_sequence(debug.get("lon_reg_final_speed_mps"))
        if lat is not None and lon is not None:
            return lat, lon, 0
    return None, None, None


def _selected_plan_target_speed(selected_plan: np.ndarray, plan_dt: float) -> float:
    """Immediate target speed encoded by a selected trajectory's first pose."""
    plan = _valid_waypoints(selected_plan)
    if plan is None:
        return 0.0
    return float(np.linalg.norm(plan[0]) / max(float(plan_dt), 1e-3))




def _bounded_model_target_speed(
    desired_speed: float,
    current_speed: float,
    horizon_sec: float,
    max_accel_mps2: float,
    max_brake_mps2: float,
) -> float:
    """Clamp a future model speed to what the ego can reach by that time."""
    horizon = max(float(horizon_sec), 0.0)
    current = max(float(current_speed), 0.0)
    return float(
        np.clip(
            float(desired_speed),
            max(0.0, current - float(max_brake_mps2) * horizon),
            current + float(max_accel_mps2) * horizon,
        )
    )


def _loop1_lane_change_active(info: Dict[str, Any], debug: Dict[str, Any]) -> bool:
    """Whether the learned planner is executing/undoing a lane change."""
    if bool(info.get("route_reentry_active", False)):
        return True
    road_option = info.get("sparsedrive_command")
    if road_option is None:
        input_info = debug.get("input_info")
        if isinstance(input_info, dict):
            road_option = input_info.get("road_option")
    try:
        # Bench2Drive RoadOption: 5=CHANGELANELEFT, 6=CHANGELANERIGHT.
        return int(road_option) in {5, 6}
    except (TypeError, ValueError):
        return False


def _is_explicit_brake_source(selected_source: str) -> bool:
    source = str(selected_source)
    return source.startswith("recovery:brake") or source.startswith(
        "recovery_hold:brake"
    )


def _recovery_max_steer_rad(agent) -> float:
    """Return the CARLA wheel-lock angle used by normalized ``VehicleControl``.

    Primitive states store a physical front-wheel angle in radians (the
    bicycle model is capped at 35 degrees), while ``VehicleControl.steer`` is a
    normalized fraction of the current vehicle's actual steering lock.  The
    Lincoln MKZ used by Leaderboard exposes a substantially larger lock than
    35 degrees, so dividing by the bicycle-model cap over-commanded every
    recovery curve and made a planned 3 m lane shift reach the far-side fence.
    """
    cached = getattr(agent, "_max_steer_rad", None)
    try:
        cached_value = float(cached)
    except (TypeError, ValueError):
        cached_value = 0.0
    if np.isfinite(cached_value) and cached_value > 1e-4:
        return cached_value

    try:
        hero = _helper.find_hero(agent)
        if hero is not None:
            value = float(_helper._get_max_steer_rad(agent, hero))
            if np.isfinite(value) and value > 1e-4:
                return value
    except Exception:
        pass

    # Unit tests and non-CARLA callers have no live vehicle. Preserve the
    # historical bicycle-model normalization as a safe compatibility fallback.
    return float(np.deg2rad(35.0))


def _recovery_steer_gain(selected_source: str, states: np.ndarray) -> float:
    """Return phase-local calibration for a committed recovery steer.

    For a left bypass the first counter-steer and the following return steer
    are both positive, with the counter-steer having the larger magnitude.
    The right maneuver is exactly mirrored. This pattern identifies the one
    transition that needs real-vehicle response compensation. The final return
    counter-steer is identified separately and deliberately keeps unit gain.
    """
    source = str(selected_source)
    if "lane_bypass_left" in source:
        direction_sign = 1.0
    elif "lane_bypass_right" in source:
        direction_sign = -1.0
    else:
        return 1.0
    if states.ndim != 2 or states.shape[0] < 2 or states.shape[1] < 2:
        return 1.0
    current = float(states[0, 1]) * direction_sign
    following = float(states[1, 1]) * direction_sign
    first_counter_steer = (
        current > 1e-4
        and following > 1e-4
        and current > following + 1e-4
    )
    final_return_counter_steer = current < -1e-4 and following >= -1e-4
    if first_counter_steer:
        return LANE_BYPASS_FIRST_COUNTER_STEER_GAIN
    if final_return_counter_steer:
        return LANE_BYPASS_FINAL_COUNTER_STEER_GAIN
    return 1.0


def _is_lane_bypass(selected_source: str) -> bool:
    source = str(selected_source)
    return "lane_bypass_left" in source or "lane_bypass_right" in source


def _lane_bypass_straight_settle(
    selected_source: str,
    states: np.ndarray,
) -> bool:
    """Whether the remaining bypass state is the final straight settle."""
    if not _is_lane_bypass(selected_source):
        return False
    arr = np.asarray(states, dtype=np.float64)
    return bool(
        arr.ndim == 2
        and arr.shape[0] > 0
        and arr.shape[1] >= 2
        and float(arr[0, 0]) > 1e-4
        and np.max(np.abs(arr[:, 1])) <= 1e-4
    )


def _bound_lane_bypass_settle_steer(steer: float) -> float:
    """Keep endpoint feedback useful without allowing opposite full lock."""
    return float(
        np.clip(
            float(steer),
            -LANE_BYPASS_SETTLE_MAX_STEER,
            LANE_BYPASS_SETTLE_MAX_STEER,
        )
    )


def _slew_lane_bypass_settle_steer(
    steer: float,
    previous_steer: Optional[float],
) -> float:
    """Bound settle feedback and prevent a one-frame opposite-lock jump."""
    bounded = _bound_lane_bypass_settle_steer(steer)
    if previous_steer is None:
        return bounded
    try:
        previous_value = float(previous_steer)
    except (TypeError, ValueError):
        return bounded
    if not np.isfinite(previous_value):
        return bounded
    previous = _bound_lane_bypass_settle_steer(previous_value)
    return float(
        np.clip(
            bounded,
            previous - LANE_BYPASS_SETTLE_MAX_STEER_DELTA,
            previous + LANE_BYPASS_SETTLE_MAX_STEER_DELTA,
        )
    )


def _bound_lane_bypass_reverse_steer(
    selected_source: str,
    reverse: bool,
    steer: float,
) -> float:
    if not reverse or not _is_lane_bypass(selected_source):
        return float(steer)
    return float(
        np.clip(
            float(steer),
            -LANE_BYPASS_REVERSE_MAX_STEER,
            LANE_BYPASS_REVERSE_MAX_STEER,
        )
    )


def _recovery_phase_target(
    agent,
    selected_source: str,
    waypoints: np.ndarray,
) -> Optional[Tuple[np.ndarray, float, bool, bool, float]]:
    """Return phase-local targets from a primitive's signed state sequence.

    Recovery plans can stop, reverse, and move forward again inside one
    trajectory. Geometry-only target speed/direction loses that temporal
    information, so the adapter exposes the state rows aligned with the current
    plan slice. Restrict lateral tracking to the contiguous same-direction phase
    to stop SparseDrive's 4--8 m lookahead from aiming through a later gear
    change (for example, through the forward half of ``back_bypass`` while the
    ego should still be reversing straight).
    """
    states = getattr(agent, "_sparsedrivev2_recovery_states", None)
    if states is None:
        return None
    arr = np.asarray(states, dtype=np.float64)
    points = _valid_waypoints(waypoints)
    if (
        points is None
        or arr.ndim != 2
        or arr.shape[0] == 0
        or arr.shape[1] < 2
        or not np.isfinite(arr[:, :2]).all()
    ):
        return None

    signed_speed = float(arr[0, 0])
    if _is_explicit_brake_source(selected_source):
        signed_speed = 0.0
    reverse = signed_speed < -1e-4
    desired_speed = abs(signed_speed)

    phase_sign = -1 if reverse else (1 if signed_speed > 1e-4 else 0)
    phase_count = 1
    if phase_sign != 0:
        limit = min(len(arr), len(points))
        while phase_count < limit:
            next_speed = float(arr[phase_count, 0])
            next_sign = -1 if next_speed < -1e-4 else (1 if next_speed > 1e-4 else 0)
            if next_sign != phase_sign:
                break
            phase_count += 1

    # Follow the current verified state, including its acceleration ramp.
    # Using a later phase's peak speed makes entry outrun its lateral motion.

    # SparseDrive's aim selection requires at least two waypoints. Including one
    # extra stop pose for a one-row phase is harmless because brake forces angle
    # zero, while it avoids an empty ``waypoints[:-1]`` reduction.
    # ``control_pid`` searches ``waypoints[:-1]``. Keep the stop/boundary pose
    # beyond the active phase eligible, plus one sentinel pose that gets excluded
    # by that slice. Otherwise the last active pose becomes the excluded sentinel;
    # once the ego passes the preceding pose the PID aims backward and saturates.
    point_count = min(len(points), max(2, phase_count + 2))
    allow_steer_while_braking = bool(
        phase_sign != 0
        and np.max(np.abs(arr[: max(phase_count, 1), 1])) > 1e-4
    )
    steer_feedforward = float(
        np.clip(
            arr[0, 1]
            * _recovery_steer_gain(selected_source, arr)
            / _recovery_max_steer_rad(agent),
            -1.0,
            1.0,
        )
    )
    return (
        points[:point_count],
        desired_speed,
        reverse,
        allow_steer_while_braking,
        steer_feedforward,
    )


def _model_targets(
    agent,
    selected_plan: np.ndarray,
    plan_dt: float,
    lookahead_sec: float,
    *,
    selected_source: str = "",
    info: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, float, str]:
    """Recover (lateral waypoints, target speed, source) for the PID.

    Uses model-native lateral/speed heads only when they are aligned with the
    SparseDriveV2 top-K candidate that AgenticDriving actually selected. For a
    recovery/carry trajectory, or when no aligned heads are available,
    it tracks ``selected_plan`` itself and derives its immediate target speed.

    On an async cached tick, reprojects ``lat_reg_final`` from the inference
    ego frame into the current ego frame and advances the longitudinal sequence
    by the elapsed cache time.  A fresh inference tick still uses upstream's
    exact ``lat_reg_final`` / ``lon_reg_final[0]`` behaviour when lookahead is 0.
    """
    debug = _planner_debug(agent)
    cached_reuse = str(selected_source).endswith("_cached")
    elapsed_sec = _cached_elapsed_sec(agent) if cached_reuse else 0.0

    if cached_reuse:
        waypoints = _valid_waypoints(
            getattr(agent, "_sparsedrivev2_pid_cached_waypoints", None)
        )
        speed_sequence = _valid_speed_sequence(
            getattr(agent, "_sparsedrivev2_pid_cached_speeds", None)
        )
        candidate_idx = getattr(agent, "_sparsedrivev2_pid_cached_candidate_idx", None)
    else:
        allow_model_candidate = str(selected_source) in {
            "sparsedrivev2",
            "sparsedrivev2_argmax",
        }
        if allow_model_candidate:
            waypoints, speed_sequence, candidate_idx = _matching_model_candidate(
                debug,
                selected_plan,
                preferred_idx=getattr(
                    agent, "_sparsedrivev2_selected_candidate_idx", None
                ),
            )
        else:
            waypoints, speed_sequence, candidate_idx = None, None, None
        # The async ticks reuse exactly the heads selected on this fresh tick.
        agent._sparsedrivev2_pid_cached_waypoints = (
            None if waypoints is None else np.asarray(waypoints).copy()
        )
        agent._sparsedrivev2_pid_cached_speeds = (
            None if speed_sequence is None else np.asarray(speed_sequence).copy()
        )
        agent._sparsedrivev2_pid_cached_candidate_idx = candidate_idx

    if waypoints is not None and speed_sequence is not None:
        if cached_reuse:
            waypoints = _reproject_cached_model_waypoints(
                agent, waypoints, info or {}
            )
        if waypoints is not None:
            dt = max(plan_dt, 1e-3)
            lookahead_idx = int(round(max(lookahead_sec, 0.0) / dt))
            elapsed_idx = int(np.floor(max(elapsed_sec, 0.0) / dt + 1e-9))
            idx = int(
                np.clip(lookahead_idx + elapsed_idx, 0, speed_sequence.size - 1)
            )
            source = f"model_candidate:{candidate_idx}"
            if cached_reuse:
                source += "_cached_reprojected"
            return waypoints, float(speed_sequence[idx]), source

    # Non-model choices must never inherit the native argmax's targets. The
    # first selected pose is at plan_dt, so its distance from the ego is the
    # immediate speed encoded by this trajectory. A brake plan therefore maps
    # to exactly 0 m/s even if the model argmax wanted to accelerate.
    selected_waypoints = np.asarray(selected_plan, dtype=np.float64)[:, :2]
    desired_speed = (
        0.0
        if _is_explicit_brake_source(selected_source)
        else _selected_plan_target_speed(selected_waypoints, plan_dt)
    )
    source = "selected_plan_cached" if cached_reuse else "selected_plan"
    return selected_waypoints, desired_speed, source


def apply(agent, selected_plan, info, selected_score_raw, selected_source):
    _helper._clear_ackermann_control(agent)

    controller = _get_controller(agent)
    plan_dt = float(getattr(agent, "_plan_dt_sec", _helper.PLAN_DT_SEC))
    plan_xy = np.asarray(selected_plan, dtype=np.float64)[:, :2]
    speed = np.asarray(float(info.get("ego_velo", 0.0)), dtype=np.float64)
    source_text = str(selected_source)
    model_source = source_text in {
        "sparsedrivev2",
        "sparsedrivev2_argmax",
        "sparsedrivev2_cached",
        "sparsedrivev2_argmax_cached",
    }
    lane_change_detected = model_source and _loop1_lane_change_active(
        info, _planner_debug(agent)
    )
    now = float(info.get("timestamp", 0.0) or 0.0)
    if lane_change_detected:
        agent._sparsedrivev2_last_loop1_lane_change_ts = now
    last_lane_change_ts = getattr(
        agent, "_sparsedrivev2_last_loop1_lane_change_ts", None
    )
    lane_change_hold = bool(
        model_source
        and last_lane_change_ts is not None
        and now >= float(last_lane_change_ts)
        and now - float(last_lane_change_ts)
        <= controller.lane_change_accel_hold_sec
    )
    loop1_lane_change = bool(lane_change_detected or lane_change_hold)
    speed_lookahead_sec = (
        max(
            controller.speed_lookahead_sec,
            controller.lane_change_speed_lookahead_sec,
        )
        if loop1_lane_change
        else controller.speed_lookahead_sec
    )
    waypoints, desired_speed, target_source = _model_targets(
        agent,
        selected_plan,
        plan_dt,
        speed_lookahead_sec,
        selected_source=selected_source,
        info=info,
    )
    model_target = target_source.startswith("model_candidate:")
    if (
        model_target
        and controller.model_speed_gain != 1.0
        and desired_speed >= controller.model_speed_gain_min_mps
    ):
        # Domain-adaptation gain: scale UP the conservative Bench2Drive cruise
        # target. Gated on model_speed_gain_min_mps so cautious low-speed targets
        # (construction/clutter) are left untouched; only confident open-road
        # cruise is amplified. A zero/low target is thus never boosted, so
        # pedestrian/obstacle stops keep their full braking response.
        desired_speed = min(
            desired_speed * controller.model_speed_gain,
            controller.model_speed_gain_max_mps,
        )
    if model_target:
        desired_speed = _bounded_model_target_speed(
            desired_speed,
            float(speed),
            max(plan_dt, speed_lookahead_sec),
            controller.max_accel_mps2,
            controller.max_brake_mps2,
        )
    unfiltered_desired_speed = float(desired_speed)
    target_speed_filter_bypass_metrics = _zero_collision_verifier_metrics(info)
    bypass_caution_path = (
        waypoints if "lane_bypass_" in str(selected_source) else None
    )
    pedestrian_caution = _forward_pedestrian_caution(
        info,
        stationary_speed_mps=controller.pedestrian_caution_stationary_speed_mps,
        max_distance_m=controller.pedestrian_caution_distance_m,
        max_lateral_m=controller.pedestrian_caution_lateral_m,
        planned_path_local_xy=bypass_caution_path,
        proximity_distance_m=controller.pedestrian_proximity_distance_m,
    )
    vehicle_bbox_guard = None
    pedestrian_caution_cap_applied = False
    vehicle_bbox_stop_applied = False
    pedestrian_emergency_stop_applied = False

    phase_target = _recovery_phase_target(agent, selected_source, waypoints)
    if phase_target is not None:
        (
            waypoints,
            desired_speed,
            reverse,
            phase_allow_steer,
            phase_steer_feedforward,
        ) = phase_target
        target_source = "recovery_state_phase"
        unfiltered_desired_speed = float(desired_speed)
        controller.reset_model_target_speed_filter()
    else:
        # Reverse-only plans can still use geometry. Multi-phase recovery plans
        # take the signed-state path above so each gear change happens on time.
        reverse = plan_travel_direction(plan_xy) < 0.0
        phase_allow_steer = False
        phase_steer_feedforward = 0.0
        if model_target:
            desired_speed = controller.filter_model_target_speed(
                desired_speed,
                now,
                bypass=bool(target_speed_filter_bypass_metrics),
            )
        else:
            controller.reset_model_target_speed_filter()
    source_lower = str(selected_source or "").lower()
    lateral_recovery_path = (
        waypoints
        if any(
            token in source_lower
            for token in (
                "lane_left",
                "lane_right",
                "lane_bypass_left",
                "lane_bypass_right",
                "swerve_left",
                "swerve_right",
            )
        )
        else None
    )
    vehicle_bbox_guard = _forward_vehicle_bbox_guard(
        info,
        max_clearance_m=controller.vehicle_bbox_stop_clearance_m,
        planned_path_local_xy=lateral_recovery_path,
        plan_dt_sec=plan_dt,
        path_horizon_sec=1.0,
    )
    # Apply after model filtering and recovery phase decoding. This makes the
    # policy uniform: nearby pedestrians cap both learned and explicitly
    # materialized trajectories, while a hard TTC_VRU veto commands a stop
    # without entering the maneuver-recovery stack.
    (
        desired_speed,
        pedestrian_caution_cap_applied,
        pedestrian_emergency_stop_applied,
    ) = _apply_vru_speed_policy(
        desired_speed,
        reverse=reverse,
        caution=pedestrian_caution,
        speed_cap_mps=controller.pedestrian_caution_speed_mps,
        proximity_speed_cap_mps=controller.pedestrian_proximity_speed_mps,
        bbox_stop_distance_m=controller.pedestrian_bbox_stop_distance_m,
        zeroed_verifier_metrics=target_speed_filter_bypass_metrics,
    )
    desired_speed, vehicle_bbox_stop_applied = _apply_vehicle_bbox_stop_policy(
        desired_speed,
        reverse=reverse,
        guard=vehicle_bbox_guard,
    )
    pid_waypoints = -waypoints if reverse else waypoints

    try:
        steer_traj, throttle_traj, brake_traj, metadata_traj = controller.control_pid(
            pid_waypoints,
            desired_speed,
            speed,
            allow_steer_while_braking=phase_allow_steer,
            brake_ratio=(
                controller.lane_change_brake_ratio
                if loop1_lane_change
                else controller.brake_ratio
            ),
        )
    except Exception:
        LOG.exception("sparsedrivev2_pid controller failed")
        fallback = _helper.agent_brake_control(agent)
        _helper._record_applied_control(
            agent, info, control_mode="vehicle_pid_sparsedrivev2", vehicle_ctrl=fallback
        )
        return fallback

    if phase_target is not None and desired_speed > controller.brake_speed and brake_traj:
        # Upstream's brake law is boolean: ego/target > 1.1 means full brake.
        # That is appropriate for a learned trajectory head, but primitive states
        # encode an exact next-step speed. A mild 5->4 m/s phase (three_point,
        # creep) must not receive the same brake=1 command as an emergency stop.
        # Scale positive-target braking by overspeed; a >=2 m/s error still maps
        # to full brake, preserving the hard-stop phases of back_* maneuvers.
        overspeed = max(float(speed) - float(desired_speed), 0.0)
        brake_traj = float(np.clip(overspeed / 2.0, 0.0, 1.0))

    # Exact post-processing from sparsedrive_b2d_agent.py.
    if brake_traj < 0.05:
        brake_traj = 0.0
    if throttle_traj > brake_traj:
        brake_traj = 0.0

    if reverse:
        steer_traj = reverse_steer(steer_traj, True)
    if phase_target is not None and abs(phase_steer_feedforward) > 1e-4:
        # Primitive steering is a bicycle-model feed-forward command aligned to
        # this exact phase. Blend it with SparseDrive's feedback correction: the
        # feed-forward supplies the high curvature of tight recovery turns while
        # the PID still corrects tracking error and model mismatch.
        steer_traj = _blend_recovery_steer(
            feedback=float(steer_traj),
            feedforward=phase_steer_feedforward,
            heading_error=float(metadata_traj.get("angle", 0.0)),
            # The back-bypass arc deliberately changes steering direction near
            # its end and needs early counter-steer. Three-point maneuvers have
            # repeated gear/steer transitions where an unconditional override
            # makes the existing coarse 0.5 s tracker substantially worse.
            allow_feedback_override="back_bypass" in str(selected_source),
        )
    recovery_states = getattr(agent, "_sparsedrivev2_recovery_states", None)
    if (
        phase_target is not None
        and recovery_states is not None
        and _lane_bypass_straight_settle(selected_source, recovery_states)
    ):
        # Once the return S-curve has ended, do not let a repeated/near endpoint
        # flip the command from counter-steer to full opposite lock.  Preserve
        # feedback authority, but bound and slew-limit it while the straight
        # settle row gives the physical rack time to recenter.
        steer_traj = _slew_lane_bypass_settle_steer(
            steer_traj,
            getattr(agent, "_last_steer", None),
        )
    steer_traj = _bound_lane_bypass_reverse_steer(
        selected_source,
        reverse,
        steer_traj,
    )

    ctrl = carla.VehicleControl()
    ctrl.steer = np.clip(float(steer_traj), -1, 1)
    ctrl.throttle = np.clip(float(throttle_traj), 0, 1)
    ctrl.brake = np.clip(float(brake_traj), 0, 1)
    ctrl.reverse = bool(reverse)
    ctrl.hand_brake = False
    ctrl.manual_gear_shift = False
    agent._last_steer = ctrl.steer

    metadata_traj["agent"] = "only_traj"
    metadata_traj["steer"] = ctrl.steer
    metadata_traj["throttle"] = ctrl.throttle
    metadata_traj["brake"] = ctrl.brake
    metadata_traj["steer_traj"] = float(steer_traj)
    metadata_traj["throttle_traj"] = float(throttle_traj)
    metadata_traj["brake_traj"] = float(brake_traj)
    metadata_traj["target_source"] = target_source
    metadata_traj["unfiltered_desired_speed"] = float(unfiltered_desired_speed)
    metadata_traj["target_speed_filter_applied"] = bool(
        float(desired_speed) > float(unfiltered_desired_speed) + 1e-6
    )
    metadata_traj["target_speed_filter_bypass_metrics"] = list(
        target_speed_filter_bypass_metrics
    )
    metadata_traj["target_speed_max_drop_mps2"] = float(
        controller.target_speed_max_drop_mps2
    )
    metadata_traj["pedestrian_caution_speed_cap_applied"] = bool(
        pedestrian_caution_cap_applied
    )
    metadata_traj["pedestrian_emergency_stop_applied"] = bool(
        pedestrian_emergency_stop_applied
    )
    metadata_traj["pedestrian_caution"] = pedestrian_caution
    metadata_traj["vehicle_bbox_stop_applied"] = bool(vehicle_bbox_stop_applied)
    metadata_traj["vehicle_bbox_guard"] = vehicle_bbox_guard
    metadata_traj["loop1_lane_change"] = bool(loop1_lane_change)
    metadata_traj["speed_lookahead_sec"] = float(speed_lookahead_sec)
    metadata_traj["plan"] = plan_xy.tolist()

    if _helper.coerce_bool(_helper._control_cfg(agent).get("debug", True), default=True):
        LOG.info(
            "sparsedrivev2_pid src=%s tgt=%s ego=%.2f m/s desired=%.2f raw=%.2f aim_dist=%.2f "
            "thr=%.2f brk=%.0f steer=%.3f rev=%d lanechg=%d lookahead=%.2f",
            selected_source,
            target_source,
            float(speed),
            float(metadata_traj["desired_speed"]),
            float(unfiltered_desired_speed),
            float(metadata_traj["aim_dist"]),
            ctrl.throttle,
            ctrl.brake,
            ctrl.steer,
            int(bool(reverse)),
            int(bool(loop1_lane_change)),
            float(speed_lookahead_sec),
        )

    plan_step1_speed = _helper._plan_step1_speed(selected_plan, plan_dt)
    _helper._record_applied_control(
        agent,
        info,
        control_mode="vehicle_pid_sparsedrivev2",
        acc_cmd=None,
        plan_step1_speed=plan_step1_speed,
        vehicle_ctrl=ctrl,
    )
    if isinstance(getattr(agent, "_last_applied_control", None), dict):
        agent._last_applied_control.update(
            {
                "target_speed": float(metadata_traj["desired_speed"]),
                "unfiltered_target_speed": float(unfiltered_desired_speed),
                "target_speed_filter_applied": bool(
                    metadata_traj["target_speed_filter_applied"]
                ),
                "target_speed_filter_bypass_metrics": list(
                    target_speed_filter_bypass_metrics
                ),
                "pedestrian_caution_speed_cap_applied": bool(
                    metadata_traj["pedestrian_caution_speed_cap_applied"]
                ),
                "pedestrian_emergency_stop_applied": bool(
                    metadata_traj["pedestrian_emergency_stop_applied"]
                ),
                "pedestrian_caution": pedestrian_caution,
                "vehicle_bbox_stop_applied": bool(vehicle_bbox_stop_applied),
                "vehicle_bbox_guard": vehicle_bbox_guard,
                "aim_distance": float(metadata_traj["aim_dist"]),
                "angle": float(metadata_traj["angle"]),
                "target_source": target_source,
                "loop1_lane_change": bool(loop1_lane_change),
                "speed_lookahead_sec": float(speed_lookahead_sec),
            }
        )
    return ctrl

def resolve_mode(mode):
    if mode not in {"sparsedrivev2_pid", "sparsedrivev2", "sparsedrive_pid"}:
        raise ValueError(f"Unsupported release control mode: {mode}")
    return "sparsedrivev2_pid"

def dispatch(agent, selected_plan, info, selected_score_raw, selected_source):
    resolve_mode(_helper._control_mode(agent))
    return apply(agent, selected_plan, info, selected_score_raw, selected_source)
