"""Primitive-backed proposal provider for the orchestrator recovery stage.

In-repo replacement for the external Rule-Planner (``privileged_planner``)
dependency: recovery candidates are materialized from the fallback primitive
ladder (:data:`primitives.CANDIDATE_LADDER`) through the same restricted
executor the VLM codegen path uses, so the recovery pipeline needs no
``repo_root``/``config_path`` pointing at another repository.

Frame note: :class:`kinematics.VehicleSim` integrates directly in the CARLA ego-local
local frame (x=right, y=forward), so — unlike the external Rule-Planner output
— these proposals do not need another coordinate conversion.

Score note: there is no per-candidate scorer here; scores encode the ladder's
safest-first ordering. The argmax recovery path therefore picks the ladder
head, while the agentic recovery loop re-ranks the set with the VLM anyway.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from agenticdriving.adapters.carla.context import (
    command_to_route_instruction,
    describe_task_target_hint,
    extract_current_ego_speed_mps,
)
from agenticdriving.adapters.carla.geometry import (
    info_to_pose,
    world_points_to_current_local,
)
from agenticdriving.planners.fallback_rule_based.executor import run_codegen
from agenticdriving.planners.fallback_rule_based.primitives import (
    CANDIDATE_LADDER,
    DrivingContext,
    LANE_BYPASS_ACCEL_MPS2,
    LANE_BYPASS_BORROW_TIME_SEC,
    LANE_BYPASS_COUNTER_STEER_DURATION_SEC,
    LANE_BYPASS_LANE_CHANGE_DURATION_SEC,
    LANE_BYPASS_OBSTACLE_CLEARANCE_M,
    LANE_BYPASS_TARGET_SPEED_MPS,
    PrimitiveRuntime,
    front_vehicle_distance,
    ladder_for_speed,
    recovery_lateral_distance_range,
    validate_recovery_lateral_distance,
)
from agenticdriving.orchestration.directional_options import (
    is_leftward_primitive,
    suppress_leftward_options,
)

LOG = logging.getLogger(__name__)

# Ladder-order prior mapped onto [_SCORE_BOTTOM, _SCORE_TOP], head first.
_SCORE_TOP = 1.0
_SCORE_BOTTOM = 0.5

# Minimum poses to materialize per recovery maneuver, independent of the model's
# native horizon (SparseDrive hands us only 6). Lane changes now fit exactly in
# those six 0.5 s rows; longer recovery combinations still use the 16-row
# materialization budget and shorter primitives pad their final pose.
# The orchestrator still slices its own output_num_poses lookahead window while
# playing the hold.
RECOVERY_MIN_GEN_POSES = 16
# A maximum-delay lane bypass needs up to 20 rows: ten stationary wait rows,
# five rows for the outward S-curve, one obstacle-clearance row, two return
# rows, and two zero-steer settle rows. Sixteen truncated the settle in Route
# 1021 and handed control back while the steering was still reversed.
LANE_BYPASS_RECOVERY_MIN_GEN_POSES = 20
# Physical time between adjacent recovery waypoints/states. Keep this aligned
# with ``DrivingContext.dt``; the orchestrator uses it to advance a committed
# multi-phase primitive by simulation time instead of nearest-point geometry.
RECOVERY_PLAN_DT_SEC = 0.5

# Dynamic prompt gate for the aggressive adjacent-lane primitive.  The old
# implementation imposed a fixed 90 m inter-vehicle floor.  That conflated a
# spawn-distance setting with the actual time available to the ego and hid
# perfectly usable low-speed gaps.  Availability is now decided from the next
# oncoming bbox's projected time to conflict at the scheduled lane entry.
LANE_BYPASS_MIN_GAP_M = 0.0
# Same-direction lane insertion keeps a small bbox-to-bbox longitudinal
# clearance.  Opposing traffic uses its independent 1 m bbox margin below.
LANE_BYPASS_CLEARANCE_BUFFER_M = 1.0
# Reserve an extra half second beyond the nominal 3.0 s borrowed-lane profile
# for steering-response and tracking uncertainty in the real CARLA vehicle.
LANE_BYPASS_REQUIRED_TIME_SEC = LANE_BYPASS_BORROW_TIME_SEC + 0.5
LANE_BYPASS_MIN_VEHICLE_SPEED_MPS = 1.0
LANE_BYPASS_MAX_ENTRY_DELAY_SEC = 5.0
# PDMS additionally probes every trajectory pose 0.5 s forward.  Its TTC gate
# remains unchanged and independent from this prompt-availability calculation.
LANE_BYPASS_TTC_PROBE_MARGIN_SEC = 0.5
# The full primitive lasts about 4.5 s.  Only expose it to the VLM when the
# next target-lane bbox remains outside the ego conflict point for strictly longer
# than this interval, using the measured bbox longitudinal velocity.
LANE_BYPASS_MIN_TIME_GAP_SEC = 4.5

# The committed moving maneuver has a six-second upper budget. Traffic waiting
# happens outside it in the latched recovery state; once the return and its
# settle row finish, a close-blocker profile still ends early instead of
# spending the unused budget cruising quickly in the re-entered route lane.
# The extra three half-second rows are only consumed by long work-zone tapers
# such as Route 1069's separated warning frame plus cone chain.
LANE_BYPASS_FIXED_DURATION_SEC = 6.0
LANE_BYPASS_FIXED_STEPS = int(
    round(LANE_BYPASS_FIXED_DURATION_SEC / RECOVERY_PLAN_DT_SEC)
)
LANE_BYPASS_ENTRY_STEPS = 3
# Give the physical CARLA steering rack a full second to return from the fixed
# adjacent lane, then keep one final route-centreline row for wheel recentering.
# The previous single 0.5 s return row ended Route 69 while the HD map still
# classified the ego in the opposing lane; SparseDrive then issued a second
# lane change.  Three fixed rows remain inside the same 4.5 s primitive.
LANE_BYPASS_RETURN_STEPS = 2
LANE_BYPASS_SETTLE_STEPS = 1
LANE_BYPASS_TRACKING_CLEARANCE_STEPS = 0
# Preserve the original fast-pass target for ordinary blockers even though a
# long construction taper may now consume the extended six-second ceiling.
# The auto solver first tries to clear by three seconds; only when the measured
# bbox chain cannot physically fit does it use later rows.
LANE_BYPASS_NOMINAL_CLEAR_DEADLINE_STEPS = 6
LANE_BYPASS_BBOX_MARGIN_M = 1.0
# Treat a row of construction props as one obstruction when each next bbox is
# still close enough that returning to the route lane between the two objects
# is not useful.  Route 1069 places its warning frame roughly ten metres ahead
# of the first cone, followed by a tight cone chain.  A four-metre link split
# that one work zone after the frame and returned directly into cones 4711--
# 4714.  The transitive rule still stops at a genuinely separated obstacle,
# while a twelve-metre link keeps common sign-to-cone tapers in one bypass.
LANE_BYPASS_CLUSTER_MAX_GAP_M = 13.0
LANE_BYPASS_DEFAULT_ROUTE_LANE_WIDTH_M = 3.5
LANE_BYPASS_ROUTE_LANE_EDGE_MARGIN_M = 0.3
# Finish the single move into the immediate adjacent lane before the ego front
# can reach a close construction bbox.  The real CARLA rack lags the ideal
# bicycle rollout, so never spend more than three forward metres entering.
LANE_BYPASS_ENTRY_RESPONSE_MARGIN_M = 2.0
LANE_BYPASS_MAX_ENTRY_DISTANCE_M = 3.0
# A stopped CARLA vehicle cannot translate one lane sideways before reaching a
# blocker when less than roughly two metres of forward travel remain.  In that
# case the latched lane-bypass primitive must first use its one permitted internal
# HD-map retreat; merely saturating the steering rack still clips a wide sign's
# exact bbox (Route 69).
LANE_BYPASS_MIN_EXECUTABLE_ENTRY_DISTANCE_M = 2.0
# Track the exact adjacent-lane centre while passing a route-lane blocker.
# A fixed outward reserve reduces the remaining margin to the far lane edge
# and, under a fast temporal profile, can turn tracking error into a second
# lane crossing.  Obstacle admission remains bbox-checked before execution;
# lateral placement itself is owned by the authoritative HD-map centreline.
LANE_BYPASS_TARGET_OUTWARD_BUFFER_M = 0.0
# The map-bound rollout needs to cover a stopped blocker whose front edge is
# roughly 20 m away before beginning its one-second return.  Route 69's exact
# stop pose varies by about half a metre between runs; 4.5 m/s^2 puts row seven
# at 20.75 m.  If one safety retreat has increased the bbox distance, the
# materializer may solve a faster profile, still within the same fixed 4.5 s.
LANE_BYPASS_HD_MAP_ACCEL_MPS2 = 4.5
LANE_BYPASS_HD_MAP_MAX_ACCEL_MPS2 = 8.0
LANE_BYPASS_HD_MAP_MAX_SPEED_MPS = 12.0
# When Loop 1 has already entered the borrowed lane before recovery latches,
# braking in place can deadlock with the first target-lane actor.  The latched
# primitive may therefore execute one map-bound reverse retreat of the same
# sub-five-second duration, returning behind the route-lane blocker before it
# retries the forward bypass.  This remains an internal phase of
# ``lane_bypass_left``; it is never exposed as another primitive.
# Reserve the final two seconds for the one HD-map lane blend back to the route.
# A close/wide bbox can require about 11 m of longitudinal reverse clearance;
# reaching that by row five needs a short 6 m/s profile.  The previous 4 m/s
# profile consumed seven rows and left only one second for the lane return, so
# CARLA remained in the opposing lane despite the nominal endpoint.
LANE_BYPASS_RETREAT_TARGET_SPEED_MPS = 6.0
LANE_BYPASS_RETREAT_ACCEL_MPS2 = 5.0
LANE_BYPASS_RETREAT_RETURN_STEPS = 4
LANE_BYPASS_RETREAT_MIN_RETURN_STEPS = LANE_BYPASS_RETREAT_RETURN_STEPS
LANE_BYPASS_RETREAT_MIN_DISTANCE_M = 6.0
# A route-centred ego that is only short of the physical lane-entry distance
# needs a small bbox-driven setup move, not the full borrowed-lane retreat.
LANE_BYPASS_ENTRY_RETREAT_BASE_DISTANCE_M = 2.0


def _distance_under_acceleration(
    initial_speed_mps: float,
    target_speed_mps: float,
    accel_mps2: float,
    duration_sec: float,
) -> float:
    """Conservative forward distance covered during an accelerate-and-hold leg."""
    initial = max(0.0, float(initial_speed_mps))
    target = max(0.0, float(target_speed_mps))
    duration = max(0.0, float(duration_sec))
    if initial >= target:
        return initial * duration
    accel = max(float(accel_mps2), 1e-3)
    accel_time = min((target - initial) / accel, duration)
    return (
        initial * accel_time
        + 0.5 * accel * accel_time * accel_time
        + target * (duration - accel_time)
    )


def _polyline_point_at_distance(points: np.ndarray, distance_m: float) -> np.ndarray:
    """Interpolate a 2-D centerline by arc length, clamping at its endpoint."""
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        raise ValueError("centerline needs at least two finite 2-D points")
    arr = arr[:, :2]
    if not np.isfinite(arr).all():
        raise ValueError("centerline contains non-finite points")
    segment_lengths = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    distance = float(np.clip(float(distance_m), 0.0, cumulative[-1]))
    idx = int(np.searchsorted(cumulative, distance, side="right") - 1)
    idx = int(np.clip(idx, 0, len(arr) - 2))
    length = float(segment_lengths[idx])
    if length <= 1e-6:
        return arr[idx + 1].copy()
    alpha = (distance - float(cumulative[idx])) / length
    return (1.0 - alpha) * arr[idx] + alpha * arr[idx + 1]


def _polyline_point_at_signed_distance(
    points: np.ndarray,
    distance_m: float,
) -> np.ndarray:
    """Interpolate forward, or extend the initial HD-map tangent backward."""
    distance = float(distance_m)
    if distance >= 0.0:
        return _polyline_point_at_distance(points, distance)
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        raise ValueError("centerline needs at least two finite 2-D points")
    arr = arr[:, :2]
    if not np.isfinite(arr).all():
        raise ValueError("centerline contains non-finite points")
    tangent = None
    for delta in np.diff(arr, axis=0):
        norm = float(np.linalg.norm(delta))
        if norm > 1e-6:
            tangent = delta / norm
            break
    if tangent is None:
        raise ValueError("centerline has no non-zero initial tangent")
    return arr[0] + distance * tangent


def _smoothstep(value: float) -> float:
    tau = float(np.clip(float(value), 0.0, 1.0))
    return tau * tau * (3.0 - 2.0 * tau)


def _centerline_right_at_forward(
    points: np.ndarray,
    forward_m: float,
) -> float:
    """Interpolate one HD-map centreline's local right coordinate.

    CARLA publishes the sampled centreline in the current vehicle frame.  Its
    forward coordinate is monotonic for the short non-junction segment used by
    the bypass, so matching actors by this curve is more stable than using one
    fixed lateral band.  Clamp outside the sampled interval; distant curved
    lanes retain the existing conservative corridor fallback below.
    """
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        raise ValueError("centerline needs at least two finite 2-D points")
    arr = arr[:, :2]
    if not np.isfinite(arr).all():
        raise ValueError("centerline contains non-finite points")
    order = np.argsort(arr[:, 1], kind="stable")
    forward = arr[order, 1]
    right = arr[order, 0]
    unique_forward, unique_indices = np.unique(forward, return_index=True)
    if unique_forward.size < 2:
        raise ValueError("centerline has no forward span")
    unique_right = right[unique_indices]
    return float(
        np.interp(
            float(forward_m),
            unique_forward,
            unique_right,
            left=float(unique_right[0]),
            right=float(unique_right[-1]),
        )
    )


def _agent_local_bbox_corners(
    agent: Dict[str, Any],
    pose: np.ndarray,
) -> Optional[np.ndarray]:
    """Return one privileged object's exact/conservative bbox in ego local XY."""
    raw_corners = agent.get("bbox_world_corners")
    if raw_corners is not None:
        try:
            corners = np.asarray(raw_corners, dtype=np.float64)
            if corners.ndim == 2 and corners.shape[0] >= 4 and corners.shape[1] >= 2:
                if corners.shape[1] == 2:
                    corners = np.column_stack([corners, np.zeros(len(corners))])
                local = world_points_to_current_local(corners[:, :3], pose)
                if np.isfinite(local[:, :2]).all():
                    return np.asarray(local[:, :2], dtype=np.float64)
        except (TypeError, ValueError):
            pass
    try:
        center_world = np.asarray(agent.get("agent_pos_world"), dtype=np.float64).reshape(-1)
        if center_world.size < 2:
            return None
        if center_world.size < 3:
            center_world = np.pad(center_world, (0, 3 - center_world.size))
        extent = np.asarray(agent.get("agent_extent") or (4.5, 2.0), dtype=np.float64)
        half_length = max(0.1, 0.5 * float(extent[0]))
        half_width = max(0.1, 0.5 * float(extent[1]))
        heading = float(agent.get("agent_heading", 0.0) or 0.0)
        forward = np.array([math.cos(heading), math.sin(heading)], dtype=np.float64)
        right = np.array([-forward[1], forward[0]], dtype=np.float64)
        world_xy = np.asarray(
            [
                center_world[:2] + sx * half_length * forward + sy * half_width * right
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
            ],
            dtype=np.float64,
        )
        world_xyz = np.column_stack([world_xy, np.full(4, center_world[2])])
        return np.asarray(
            world_points_to_current_local(world_xyz, pose)[:, :2], dtype=np.float64
        )
    except (TypeError, ValueError):
        return None


def front_corridor_object_observations(
    *,
    info: Dict[str, Any],
    privileged_info: Any,
    max_distance_m: float,
    lateral_buffer_m: float = 0.3,
) -> List[Dict[str, Any]]:
    """Return bbox observations overlapping the ego's forward corridor.

    Distance is the longitudinal free space from the ego front bumper to the
    object's nearest bbox edge. Results are ordered nearest-first.
    """
    limit_m = max(0.0, float(max_distance_m))
    if limit_m <= 0.0:
        return []
    agents = privileged_info
    if isinstance(privileged_info, dict):
        agents = privileged_info.get("agents") or privileged_info.get("objects")
    if agents is None:
        return []
    try:
        pose = info_to_pose(info)
    except Exception:
        return []
    ego_box = np.asarray(info.get("ego_box", []), dtype=np.float64).reshape(-1)
    ego_half_length = (
        max(0.1, 0.5 * float(ego_box[3])) if ego_box.size > 3 else 2.25
    )
    ego_half_width = (
        max(0.1, 0.5 * float(ego_box[4])) if ego_box.size > 4 else 1.0
    )
    corridor_half_width = ego_half_width + max(0.0, float(lateral_buffer_m))
    observations: List[Dict[str, Any]] = []
    for agent in agents:
        if not isinstance(agent, dict):
            continue
        object_type = str(agent.get("object_type", "vehicle")).strip().lower()
        corners = _agent_local_bbox_corners(agent, pose)
        if corners is None or len(corners) < 4:
            continue
        min_right = float(np.min(corners[:, 0]))
        max_right = float(np.max(corners[:, 0]))
        rear = float(np.min(corners[:, 1]))
        front = float(np.max(corners[:, 1]))
        if not np.isfinite([min_right, max_right, rear, front]).all():
            continue
        if front <= 0.0:
            continue
        if min_right > corridor_half_width or max_right < -corridor_half_width:
            continue
        distance_m = max(0.0, rear - ego_half_length)
        if distance_m > limit_m:
            continue
        world_position = None
        try:
            raw_position = np.asarray(
                agent.get("agent_pos_world"), dtype=np.float64
            ).reshape(-1)
            if raw_position.size >= 2 and np.isfinite(raw_position[:2]).all():
                world_position = raw_position[:2].tolist()
        except (TypeError, ValueError):
            pass
        speed_mps = None
        try:
            raw_speed = agent.get("agent_velo")
            if raw_speed is not None:
                speed_mps = abs(float(raw_speed))
            else:
                velocity = np.asarray(
                    agent.get("agent_vel_vec"), dtype=np.float64
                ).reshape(-1)
                if velocity.size >= 2 and np.isfinite(velocity[:2]).all():
                    speed_mps = float(np.linalg.norm(velocity[:2]))
        except (TypeError, ValueError):
            pass
        if speed_mps is not None and not np.isfinite(speed_mps):
            speed_mps = None
        observations.append({
            "distance_m": float(distance_m),
            "bbox_min_right_m": min_right,
            "bbox_max_right_m": max_right,
            "bbox_rear_m": rear,
            "bbox_front_m": front,
            "agent_id": str(agent.get("agent_id", "unknown")),
            "object_type": object_type or "unknown",
            "world_position": world_position,
            "speed_mps": speed_mps,
        })
    observations.sort(key=lambda item: float(item["distance_m"]))
    return observations


def roadside_lane_bypass_left_assessment(
    *,
    info: Dict[str, Any],
    privileged_info: Any,
    max_distance_m: float = 10.0,
    offset_ratio: float = 0.25,
) -> Dict[str, Any]:
    """Mask the left-bypass menu item for a right-edge front obstruction.

    This is a menu policy, not a collision-free-path assertion. Measure bbox
    offset against the route/source lane, never the ego heading or the whole
    road. Missing geometry leaves the menu unchanged. A simultaneous central
    blocker (including a wide bbox spanning the lane) keeps bypass available.
    """
    result: Dict[str, Any] = {"masked": False, "reason": "unknown_route_geometry"}
    descriptors = info.get("adjacent_lane_flows")
    descriptor = descriptors.get("left") if isinstance(descriptors, dict) else None
    if not isinstance(descriptor, dict):
        return result
    try:
        source = np.asarray(descriptor.get("source_centerline_local_xy"), dtype=np.float64)
        # Live adapters publish the route width separately from the adjacent
        # lane's width. Retain the old descriptor fallback for saved inputs.
        width = float(descriptor.get("source_lane_width_m", descriptor.get("lane_width_m")))
        ratio = float(offset_ratio)
        if not math.isfinite(width) or width <= 0.0:
            return result
        if not math.isfinite(ratio) or not 0.0 < ratio < 0.5:
            return result
        _centerline_right_at_forward(source, 0.0)
    except (TypeError, ValueError):
        return result

    threshold = width * ratio
    observations = front_corridor_object_observations(
        info=info,
        privileged_info=privileged_info,
        max_distance_m=max_distance_m,
        # Ego may already be offset within its lane; inspect the route lane
        # rather than limiting the policy to the ego's narrow straight sweep.
        lateral_buffer_m=width,
    )
    blockers: List[Dict[str, Any]] = []
    for observation in observations:
        object_type = observation["object_type"]
        if object_type not in {"static", "vehicle"}:
            continue
        if object_type == "vehicle":
            speed = observation["speed_mps"]
            if speed is None or speed > 0.5:
                continue
        rear = observation["bbox_rear_m"]
        front = observation["bbox_front_m"]
        # Do not extrapolate the lane shape beyond the available map samples.
        if rear < float(np.min(source[:, 1])) or front > float(np.max(source[:, 1])):
            continue
        center_forward = 0.5 * (rear + front)
        route_right = _centerline_right_at_forward(source, center_forward)
        bbox_left = observation["bbox_min_right_m"] - route_right
        bbox_right = observation["bbox_max_right_m"] - route_right
        if bbox_left >= width * 0.5 or bbox_right <= -width * 0.5:
            continue
        offset = 0.5 * (bbox_left + bbox_right)
        blockers.append({
            **observation,
            "route_center_right_m": route_right,
            "offset_from_route_center_m": offset,
            "roadside": bool(offset >= threshold and bbox_left > -threshold),
        })
    if not blockers:
        return {**result, "reason": "no_nearby_stationary_route_blocker"}
    nearest = blockers[0]
    # Include all bboxes overlapping the nearest obstruction longitudinally,
    # so a roadside proxy/cone cannot hide a central obstacle beside it.
    front_group = [
        item for item in blockers
        if item["bbox_rear_m"] <= nearest["bbox_front_m"]
    ]
    masked = all(item["roadside"] for item in front_group)
    return {
        "masked": masked,
        "reason": "right_edge_front_obstacle" if masked else "central_front_obstacle",
        "lane_width_m": width,
        "offset_threshold_m": threshold,
        "blockers": front_group,
    }




def _lane_bypass_bbox_clearance_m(
    *,
    info: Dict[str, Any],
    privileged_info: Any,
    source_centerline: np.ndarray,
    fallback_clearance_m: float,
    route_lane_width_m: Optional[float] = None,
) -> Tuple[float, Optional[float], Optional[str], Optional[str]]:
    """Find entry/exit distances for the bbox cluster blocking the route lane."""
    agents = privileged_info
    if isinstance(privileged_info, dict):
        agents = privileged_info.get("agents") or privileged_info.get("objects")
    if agents is None:
        return max(0.0, float(fallback_clearance_m)), None, None, None
    try:
        pose = info_to_pose(info)
    except Exception:
        return max(0.0, float(fallback_clearance_m)), None, None, None
    ego_box = np.asarray(info.get("ego_box", []), dtype=np.float64).reshape(-1)
    ego_half_length = max(0.1, 0.5 * float(ego_box[3])) if ego_box.size > 3 else 2.25
    ego_half_width = max(0.1, 0.5 * float(ego_box[4])) if ego_box.size > 4 else 1.0
    try:
        route_lane_width = max(1.0, float(route_lane_width_m))
    except (TypeError, ValueError):
        route_lane_width = LANE_BYPASS_DEFAULT_ROUTE_LANE_WIDTH_M

    blockers: List[Tuple[float, float, str, str]] = []
    for agent in agents:
        if not isinstance(agent, dict):
            continue
        object_type = str(agent.get("object_type", "vehicle"))
        if object_type not in {"vehicle", "static"}:
            continue
        if object_type == "vehicle":
            try:
                raw_speed = agent.get("agent_velo")
                if raw_speed is None:
                    velocity = np.asarray(
                        agent.get("agent_vel_vec") or (0.0, 0.0),
                        dtype=np.float64,
                    ).reshape(-1)
                    speed = float(np.linalg.norm(velocity[:2]))
                else:
                    speed = float(raw_speed or 0.0)
                if speed > 2.0:
                    continue
            except (TypeError, ValueError):
                continue
        corners = _agent_local_bbox_corners(agent, pose)
        if corners is None or len(corners) < 4:
            continue
        min_right = float(np.min(corners[:, 0]))
        max_right = float(np.max(corners[:, 0]))
        rear = float(np.min(corners[:, 1]))
        front = float(np.max(corners[:, 1]))
        if front <= 0.0 or rear > 30.0:
            continue
        center_forward = max(0.0, 0.5 * (rear + front))
        try:
            source_point = _polyline_point_at_distance(
                source_centerline, center_forward
            )
        except ValueError:
            continue
        source_right = float(source_point[0])
        # Clear every prop occupying the route lane, not only objects that
        # intersect the ego's current centreline footprint.  Work-zone cones
        # are commonly staggered toward a lane edge; ignoring them makes the
        # return blend sweep through their measured bboxes.
        lateral_margin = max(
            ego_half_width + 0.3,
            0.5 * route_lane_width + LANE_BYPASS_ROUTE_LANE_EDGE_MARGIN_M,
        )
        overlaps_route_lane = bool(
            min_right <= source_right + lateral_margin
            and max_right >= source_right - lateral_margin
        )
        if overlaps_route_lane:
            blockers.append(
                (rear, front, str(agent.get("agent_id", "unknown")), object_type)
            )

    if not blockers:
        return max(0.0, float(fallback_clearance_m)), None, None, None
    blockers.sort(key=lambda item: item[0])
    nearest_rear = blockers[0][0]
    cluster = [blockers[0]]
    cluster_front = float(blockers[0][1])
    for item in blockers[1:]:
        if float(item[0]) > cluster_front + LANE_BYPASS_CLUSTER_MAX_GAP_M:
            break
        cluster.append(item)
        cluster_front = max(cluster_front, float(item[1]))
    clearance = max(
        float(fallback_clearance_m),
        cluster_front + ego_half_length + LANE_BYPASS_BBOX_MARGIN_M,
    )
    entry_distance = max(
        0.75,
        min(
            LANE_BYPASS_MAX_ENTRY_DISTANCE_M,
            nearest_rear
            - ego_half_length
            - LANE_BYPASS_ENTRY_RESPONSE_MARGIN_M,
        ),
    )
    return (
        float(clearance),
        float(entry_distance),
        ",".join(item[2] for item in cluster),
        "static" if any(item[3] == "static" for item in cluster) else "vehicle",
    )


def _lane_bypass_active_points(
    *,
    source_centerline: np.ndarray,
    target_centerline: np.ndarray,
    direction: str,
    profile: Dict[str, Any],
    already_in_target_lane: bool,
    route_blocker_id: Optional[str],
) -> np.ndarray:
    """Build the exact HD-map centre path shared by gating and execution."""
    source = np.asarray(source_centerline, dtype=np.float64)[:, :2]
    target = np.asarray(target_centerline, dtype=np.float64)[:, :2]
    progresses = [float(value) for value in profile["progresses_m"]]
    entry_steps = max(1, int(profile["entry_steps"]))
    return_steps = max(1, int(profile["return_steps"]))
    return_start_step = int(profile["return_start_step"])
    outward_sign = -1.0 if str(direction).lower().startswith("l") else 1.0
    points: List[np.ndarray] = []
    for step, distance in enumerate(progresses, start=1):
        source_point = _polyline_point_at_distance(source, distance)
        target_point = _polyline_point_at_distance(target, distance).copy()
        if route_blocker_id is not None:
            target_point[0] += outward_sign * LANE_BYPASS_TARGET_OUTWARD_BUFFER_M
        current_heading_point = np.array([0.0, distance], dtype=np.float64)
        map_alignment = _smoothstep(step / entry_steps)
        aligned_source = (
            (1.0 - map_alignment) * current_heading_point
            + map_alignment * source_point
        )
        if already_in_target_lane:
            lane_weight = (
                1.0
                if step <= return_start_step
                else (
                    1.0
                    - _smoothstep(
                        (step - return_start_step) / return_steps
                    )
                    if step <= return_start_step + return_steps
                    else 0.0
                )
            )
        elif step <= entry_steps:
            lane_weight = _smoothstep(step / entry_steps)
        elif step <= return_start_step:
            lane_weight = 1.0
        elif step <= return_start_step + return_steps:
            lane_weight = 1.0 - _smoothstep(
                (step - return_start_step) / return_steps
            )
        else:
            lane_weight = 0.0
        points.append(
            (1.0 - lane_weight) * aligned_source + lane_weight * target_point
        )
    return np.asarray(points, dtype=np.float64)


def _lane_bypass_target_lane_static_conflicts(
    *,
    info: Dict[str, Any],
    privileged_info: Any,
    source_centerline: np.ndarray,
    target_centerline: np.ndarray,
    planned_path_local_xy: np.ndarray,
) -> List[str]:
    """Return target-lane static bboxes intersecting the actual bypass sweep.

    A route-lane blocker is the object this primitive passes.  A different
    static object centred in the adjacent lane matters only when the ego's
    complete outward/pass/return footprint actually reaches it.  The previous
    fixed ``rear <= 35m`` corridor veto rejected objects well after the return
    point (ConstructionPermutations 0016), even though the borrowed lane was
    empty over the maneuver.  Keep the exact centreline assignment, then use a
    dense separating-axis sweep against the exact bbox polygon.
    """
    agents = privileged_info
    if isinstance(privileged_info, dict):
        agents = privileged_info.get("agents") or privileged_info.get("objects")
    if agents is None:
        return []
    try:
        pose = info_to_pose(info)
    except Exception:
        return []
    ego_box = np.asarray(info.get("ego_box", []), dtype=np.float64).reshape(-1)
    ego_half_length = (
        max(0.1, 0.5 * float(ego_box[3])) if ego_box.size > 3 else 2.25
    )
    ego_half_width = (
        max(0.1, 0.5 * float(ego_box[4])) if ego_box.size > 4 else 1.0
    )
    path = np.asarray(planned_path_local_xy, dtype=np.float64)
    if (
        path.ndim != 2
        or len(path) == 0
        or path.shape[1] < 2
        or not np.isfinite(path[:, :2]).all()
    ):
        return []

    def convex_hull(points: np.ndarray) -> np.ndarray:
        unique = sorted(
            {
                (float(point[0]), float(point[1]))
                for point in np.asarray(points, dtype=np.float64)[:, :2]
                if np.isfinite(point[:2]).all()
            }
        )
        if len(unique) <= 2:
            return np.asarray(unique, dtype=np.float64)

        def cross(origin, first, second) -> float:
            return (
                (first[0] - origin[0]) * (second[1] - origin[1])
                - (first[1] - origin[1]) * (second[0] - origin[0])
            )

        lower = []
        for point in unique:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
                lower.pop()
            lower.append(point)
        upper = []
        for point in reversed(unique):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
                upper.pop()
            upper.append(point)
        return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)

    def polygons_overlap(first: np.ndarray, second: np.ndarray) -> bool:
        if len(first) < 3 or len(second) < 3:
            return False
        for polygon in (first, second):
            edges = np.roll(polygon, -1, axis=0) - polygon
            for edge in edges:
                norm = float(np.linalg.norm(edge))
                if norm <= 1e-9:
                    continue
                axis = np.array([-edge[1], edge[0]], dtype=np.float64) / norm
                first_projection = first @ axis
                second_projection = second @ axis
                if (
                    float(np.max(first_projection))
                    < float(np.min(second_projection)) - 1e-6
                    or float(np.max(second_projection))
                    < float(np.min(first_projection)) - 1e-6
                ):
                    return False
        return True

    # Densify the centre path so a sparse 0.5 s plan cannot jump over a thin
    # barrier between two native rows.  Heading follows the active segment.
    path_with_origin = np.vstack(
        [np.zeros((1, 2), dtype=np.float64), path[:, :2]]
    )
    sweep_samples: List[Tuple[np.ndarray, np.ndarray]] = []
    previous_heading = np.array([0.0, 1.0], dtype=np.float64)
    for start, end in zip(path_with_origin[:-1], path_with_origin[1:]):
        delta = end - start
        distance = float(np.linalg.norm(delta))
        if distance > 1e-6:
            previous_heading = delta / distance
        steps = max(1, int(math.ceil(distance / 0.25)))
        for index in range(1, steps + 1):
            center = start + delta * (index / steps)
            sweep_samples.append((center, previous_heading.copy()))

    def ego_footprint(center: np.ndarray, heading: np.ndarray) -> np.ndarray:
        lateral = np.array([-heading[1], heading[0]], dtype=np.float64)
        return np.asarray(
            [
                center + ego_half_length * heading + ego_half_width * lateral,
                center + ego_half_length * heading - ego_half_width * lateral,
                center - ego_half_length * heading - ego_half_width * lateral,
                center - ego_half_length * heading + ego_half_width * lateral,
            ],
            dtype=np.float64,
        )

    conflicts: List[Tuple[float, str]] = []
    for agent in agents:
        if not isinstance(agent, dict):
            continue
        object_type = str(agent.get("object_type", "vehicle"))
        if object_type != "static":
            continue
        corners = _agent_local_bbox_corners(agent, pose)
        if corners is None or len(corners) < 4:
            continue
        min_right = float(np.min(corners[:, 0]))
        max_right = float(np.max(corners[:, 0]))
        rear = float(np.min(corners[:, 1]))
        front = float(np.max(corners[:, 1]))
        if front <= -ego_half_length:
            continue
        center_right = 0.5 * (min_right + max_right)
        center_forward = 0.5 * (rear + front)
        try:
            source_right = _centerline_right_at_forward(
                source_centerline, center_forward
            )
            target_right = _centerline_right_at_forward(
                target_centerline, center_forward
            )
        except ValueError:
            continue
        source_distance = abs(center_right - source_right)
        target_distance = abs(center_right - target_right)
        centred_in_target = bool(target_distance + 0.25 < source_distance)
        # A measured wall in the route lane can hit the borrowed-lane vehicle
        # even when its edge stops short of that lane's centreline. Let the
        # footprint sweep below decide clearance instead of centre assignment.
        measured_cross_lane_surface = bool(
            agent.get("perception_source") == "bridgedrive_lidar_corridor"
            and min_right <= source_right <= max_right
        )
        target_corridor_overlap = bool(
            min_right <= target_right + ego_half_width + 0.3
            and max_right >= target_right - ego_half_width - 0.3
        )
        actor_polygon = convex_hull(corners[:, :2])
        swept_overlap = bool(
            (centred_in_target or measured_cross_lane_surface)
            and target_corridor_overlap
            and len(actor_polygon) >= 3
            and any(
                polygons_overlap(ego_footprint(center, heading), actor_polygon)
                for center, heading in sweep_samples
            )
        )
        # Diagnostic-only trace for the static-target-lane veto.  Keep this
        # before the final conjunction so a visually empty lane can be
        # explained by the exact bbox, centreline assignment, or corridor
        # margin that caused (or nearly caused) the veto.  This does not alter
        # the gate decision.
        LOG.warning(
            "lane bypass static-target probe id=%s type=%s lane_key=%s "
            "bbox_local=[r %.2f..%.2f,f %.2f..%.2f] center=(%.2f,%.2f) "
            "source_center=%.2f target_center=%.2f "
            "dist_source=%.2f dist_target=%.2f centered=%s corridor_overlap=%s "
            "swept_overlap=%s "
            "ego_half_width=%.2f corridor_margin=0.30 front/rear=(%.2f,%.2f)",
            str(agent.get("agent_id", "unknown")),
            object_type,
            agent.get("agent_lane_key"),
            min_right,
            max_right,
            rear,
            front,
            center_right,
            center_forward,
            source_right,
            target_right,
            source_distance,
            target_distance,
            centred_in_target,
            target_corridor_overlap,
            swept_overlap,
            ego_half_width,
            rear,
            front,
        )
        if swept_overlap:
            conflicts.append((rear, str(agent.get("agent_id", "unknown"))))
    conflicts.sort(key=lambda item: item[0])
    return [agent_id for _rear, agent_id in conflicts]


def _lane_bypass_forward_profile(
    *,
    initial_speed_mps: float,
    bbox_clearance_m: float,
    target_speed_mps: Optional[float] = None,
    accel_mps2: Optional[float] = None,
    total_steps: int = LANE_BYPASS_FIXED_STEPS,
    entry_steps: int = LANE_BYPASS_ENTRY_STEPS,
    return_steps: int = LANE_BYPASS_RETURN_STEPS,
    settle_steps: int = LANE_BYPASS_SETTLE_STEPS,
) -> Optional[Dict[str, Any]]:
    """Solve the fixed bypass's longitudinal profile and target-lane window.

    The prompt gate and the materializer must use the same acceleration rows.
    In particular, an ego starting from rest does not instantly travel at the
    target-speed cap, and it stops conflicting with opposing traffic as soon as
    the single HD-map return reaches the route lane.
    """
    dt = RECOVERY_PLAN_DT_SEC
    initial_speed = max(0.0, float(initial_speed_mps))
    clearance = max(0.0, float(bbox_clearance_m))
    total_steps = max(2, int(total_steps))
    requested_total_steps = total_steps
    entry_steps = max(1, int(entry_steps))
    return_steps = max(1, int(return_steps))
    settle_steps = max(0, int(settle_steps))
    latest_return_start = (
        total_steps
        - return_steps
        - settle_steps
    )
    if latest_return_start < entry_steps:
        return None
    latest_clear_step = max(
        entry_steps,
        latest_return_start - LANE_BYPASS_TRACKING_CLEARANCE_STEPS,
    )

    def speed_profile(
        target_speed_mps: float,
        accel_mps2: float,
    ) -> Tuple[List[float], List[float]]:
        speed = initial_speed
        progress = 0.0
        profile_progresses: List[float] = []
        profile_speeds: List[float] = []
        for _ in range(total_steps):
            next_speed = min(
                float(target_speed_mps),
                speed + float(accel_mps2) * dt,
            )
            progress += 0.5 * (speed + next_speed) * dt
            profile_progresses.append(float(progress))
            profile_speeds.append(float(next_speed))
            speed = next_speed
        return profile_progresses, profile_speeds

    selected_target_speed = (
        LANE_BYPASS_TARGET_SPEED_MPS
        if target_speed_mps is None
        else max(0.1, float(target_speed_mps))
    )
    selected_accel = (
        LANE_BYPASS_HD_MAP_ACCEL_MPS2
        if accel_mps2 is None
        else max(0.1, float(accel_mps2))
    )
    progresses, speeds = speed_profile(selected_target_speed, selected_accel)
    if target_speed_mps is None and accel_mps2 is None:
        preferred_clear_step = min(
            latest_clear_step,
            LANE_BYPASS_NOMINAL_CLEAR_DEADLINE_STEPS,
        )
    else:
        preferred_clear_step = latest_clear_step
    if (
        target_speed_mps is None
        and accel_mps2 is None
        and progresses[preferred_clear_step - 1] + 1e-6 < clearance
    ):
        selected_accel = LANE_BYPASS_HD_MAP_MAX_ACCEL_MPS2
        deadlines = [preferred_clear_step]
        if latest_clear_step != preferred_clear_step:
            deadlines.append(latest_clear_step)
        found_profile = False
        for clear_deadline_step in deadlines:
            for candidate_target in np.arange(
                LANE_BYPASS_TARGET_SPEED_MPS,
                LANE_BYPASS_HD_MAP_MAX_SPEED_MPS + 0.25,
                0.5,
                dtype=np.float64,
            ):
                candidate_progresses, candidate_speeds = speed_profile(
                    float(candidate_target), selected_accel
                )
                if (
                    candidate_progresses[clear_deadline_step - 1] + 1e-6
                    >= clearance
                ):
                    selected_target_speed = float(candidate_target)
                    progresses = candidate_progresses
                    speeds = candidate_speeds
                    found_profile = True
                    break
            if found_profile:
                break

    clear_step = next(
        (idx + 1 for idx, value in enumerate(progresses) if value >= clearance),
        None,
    )
    if clear_step is None:
        return None
    return_start_step = max(
        entry_steps,
        int(clear_step) + LANE_BYPASS_TRACKING_CLEARANCE_STEPS,
    )
    if return_start_step > latest_return_start:
        return None

    # Row ``return_start_step`` is still on the borrowed centreline.  The next
    # row completes the only return blend, so target-lane bbox overlap ends
    # there rather than at the fixed plan endpoint by definition.
    conflict_end_step = min(
        total_steps,
        return_start_step + return_steps,
    )
    execution_steps = min(
        total_steps,
        conflict_end_step + settle_steps,
    )
    return {
        "progresses_m": progresses[:execution_steps],
        "speeds_mps": speeds[:execution_steps],
        "target_speed_mps": float(selected_target_speed),
        "accel_mps2": float(selected_accel),
        "clear_step": int(clear_step),
        "return_start_step": int(return_start_step),
        "conflict_start_step": 1,
        "conflict_end_step": int(conflict_end_step),
        "conflict_duration_sec": float(conflict_end_step * dt),
        "total_steps": int(execution_steps),
        "requested_total_steps": int(requested_total_steps),
        "entry_steps": int(entry_steps),
        "return_steps": int(return_steps),
        "settle_steps": int(settle_steps),
    }


def lane_bypass_entry_retreat_assessment(
    *,
    info: Dict[str, Any],
    privileged_info: Any,
    direction: str,
    obstacle_clearance_m: float = 4.0,
) -> Dict[str, Any]:
    """Report whether the exact blocker bbox leaves enough room to enter.

    This is deliberately separate from the opposing-traffic gate.  It answers
    a vehicle-dynamics question: can a stopped ego reach the fixed adjacent
    HD-map lane before its front reaches the nearest blocking bbox?  If not,
    the already-latched primitive creates space once by reversing on the route
    lane; no different recovery action is exposed or selected.
    """
    normalized_direction = (
        "left" if str(direction).lower().startswith("l") else "right"
    )
    descriptors = info.get("adjacent_lane_flows")
    descriptor = (
        descriptors.get(normalized_direction)
        if isinstance(descriptors, dict)
        else None
    )
    if not isinstance(descriptor, dict):
        return {
            "requires_retreat": False,
            "reason": "no_adjacent_lane_descriptor",
            "entry_distance_m": None,
            "blocker_id": None,
        }
    try:
        source = np.asarray(
            descriptor.get("source_centerline_local_xy"), dtype=np.float64
        )[:, :2]
        target = np.asarray(
            descriptor.get("target_centerline_local_xy"), dtype=np.float64
        )[:, :2]
    except (TypeError, ValueError, IndexError):
        return {
            "requires_retreat": False,
            "reason": "invalid_lane_centerline",
            "entry_distance_m": None,
            "blocker_id": None,
        }
    if (
        source.ndim != 2
        or target.ndim != 2
        or len(source) < 2
        or len(target) < 2
        or not np.isfinite(source).all()
        or not np.isfinite(target).all()
    ):
        return {
            "requires_retreat": False,
            "reason": "invalid_lane_centerline",
            "entry_distance_m": None,
            "blocker_id": None,
        }
    already_in_target_lane = bool(
        abs(float(target[0, 0])) + 0.1 < abs(float(source[0, 0]))
    )
    if already_in_target_lane:
        return {
            "requires_retreat": False,
            "reason": "already_in_fixed_target_lane",
            "entry_distance_m": None,
            "minimum_entry_distance_m": (
                LANE_BYPASS_MIN_EXECUTABLE_ENTRY_DISTANCE_M
            ),
            "blocker_id": None,
        }
    _clearance, entry_distance, blocker_id, _blocker_type = (
        _lane_bypass_bbox_clearance_m(
        info=info,
        privileged_info=privileged_info,
        source_centerline=source,
        fallback_clearance_m=obstacle_clearance_m,
        route_lane_width_m=descriptor.get("lane_width_m"),
        )
    )
    requires_retreat = bool(
        blocker_id is not None
        and entry_distance is not None
        and float(entry_distance)
        < LANE_BYPASS_MIN_EXECUTABLE_ENTRY_DISTANCE_M
    )
    return {
        "requires_retreat": requires_retreat,
        "reason": (
            "blocker_bbox_too_close_for_forward_lane_entry"
            if requires_retreat
            else "forward_lane_entry_space_sufficient"
        ),
        "entry_distance_m": (
            None if entry_distance is None else float(entry_distance)
        ),
        "minimum_entry_distance_m": (
            LANE_BYPASS_MIN_EXECUTABLE_ENTRY_DISTANCE_M
        ),
        "blocker_id": blocker_id,
    }


def _lane_bypass_retreat_clearance_m(
    *,
    info: Dict[str, Any],
    privileged_info: Any,
    source_centerline: np.ndarray,
) -> Tuple[float, Optional[str]]:
    """Return reverse distance needed to re-enter behind route-lane bboxes.

    The endpoint condition is the mirror of the forward bypass clearance: the
    ego front, plus a one-metre margin, must lie behind the rearmost bbox edge
    in the nearby blocker cluster before the reverse path blends into the route
    lane.  Exact CARLA bbox corners are preferred when available.
    """
    agents = privileged_info
    if isinstance(privileged_info, dict):
        agents = privileged_info.get("agents") or privileged_info.get("objects")
    if agents is None:
        return LANE_BYPASS_RETREAT_MIN_DISTANCE_M, None
    try:
        pose = info_to_pose(info)
    except Exception:
        return LANE_BYPASS_RETREAT_MIN_DISTANCE_M, None
    ego_box = np.asarray(info.get("ego_box", []), dtype=np.float64).reshape(-1)
    ego_half_length = (
        max(0.1, 0.5 * float(ego_box[3])) if ego_box.size > 3 else 2.25
    )
    ego_half_width = (
        max(0.1, 0.5 * float(ego_box[4])) if ego_box.size > 4 else 1.0
    )

    blockers: List[Tuple[float, float, str]] = []
    for agent in agents:
        if not isinstance(agent, dict):
            continue
        object_type = str(agent.get("object_type", "vehicle"))
        if object_type not in {"vehicle", "static"}:
            continue
        if object_type == "vehicle":
            try:
                if float(agent.get("agent_velo", 0.0) or 0.0) > 2.0:
                    continue
            except (TypeError, ValueError):
                continue
        corners = _agent_local_bbox_corners(agent, pose)
        if corners is None or len(corners) < 4:
            continue
        min_right = float(np.min(corners[:, 0]))
        max_right = float(np.max(corners[:, 0]))
        rear = float(np.min(corners[:, 1]))
        front = float(np.max(corners[:, 1]))
        if front < -25.0 or rear > 30.0:
            continue
        center_forward = 0.5 * (rear + front)
        try:
            source_point = _polyline_point_at_signed_distance(
                source_centerline, center_forward
            )
        except ValueError:
            continue
        source_right = float(source_point[0])
        lateral_margin = ego_half_width + 0.3
        if (
            min_right <= source_right + lateral_margin
            and max_right >= source_right - lateral_margin
        ):
            blockers.append((rear, front, str(agent.get("agent_id", "unknown"))))

    if not blockers:
        return LANE_BYPASS_RETREAT_MIN_DISTANCE_M, None
    # Keep the installation immediately beside/ahead of the ego.  Distant
    # route objects should not make the vehicle reverse through an unrelated
    # part of the scene.
    nearby = [item for item in blockers if item[0] <= 20.0 and item[1] >= -20.0]
    if not nearby:
        return LANE_BYPASS_RETREAT_MIN_DISTANCE_M, None
    rearmost_edge = min(item[0] for item in nearby)
    distance = max(
        LANE_BYPASS_RETREAT_MIN_DISTANCE_M,
        ego_half_length + LANE_BYPASS_BBOX_MARGIN_M - rearmost_edge,
    )
    return float(distance), ",".join(item[2] for item in nearby)


def materialize_hd_map_lane_bypass_retreat(
    *,
    info: Dict[str, Any],
    privileged_info: Any,
    direction: str,
    output_num_poses: int,
    allow_route_centered: bool = False,
    retreat_distance_m: Optional[float] = None,
    blocker_id_hint: Optional[str] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Build the latched bypass's map-bound reverse-to-route safety phase."""
    if (extract_current_ego_speed_mps(info) or 0.0) > 0.75:
        return None
    normalized_direction = (
        "left" if str(direction).lower().startswith("l") else "right"
    )
    descriptors = info.get("adjacent_lane_flows")
    descriptor = (
        descriptors.get(normalized_direction)
        if isinstance(descriptors, dict)
        else None
    )
    if not isinstance(descriptor, dict):
        return None
    try:
        source = np.asarray(
            descriptor.get("source_centerline_local_xy"), dtype=np.float64
        )[:, :2]
        target = np.asarray(
            descriptor.get("target_centerline_local_xy"), dtype=np.float64
        )[:, :2]
    except (TypeError, ValueError, IndexError):
        return None
    if (
        source.ndim != 2
        or target.ndim != 2
        or len(source) < 2
        or len(target) < 2
        or not np.isfinite(source).all()
        or not np.isfinite(target).all()
    ):
        return None
    # Do not depend on a discrete lane-membership guess here.  Route 69 can
    # report the route lane while the ego visually occupies the borrowed lane,
    # and the two sampled centerlines can be nearly equidistant while the car
    # straddles their boundary.  Start exactly at the current vehicle origin,
    # follow the tangent of whichever HD-map centerline is nearer, then blend
    # only toward the fixed route/source centerline.
    source_distance = abs(float(source[0, 0]))
    target_distance = abs(float(target[0, 0]))
    # A clearly route-centred ego should simply wait for opposing traffic; the
    # retreat is only for a borrowed/straddling ego that would block it.
    if source_distance + 0.5 < target_distance and not allow_route_centered:
        return None
    current_centerline = (
        target
        if target_distance <= source_distance
        else source
    )
    current_origin = np.asarray(current_centerline[0], dtype=np.float64)

    if retreat_distance_m is None:
        retreat_distance, blocker_id = _lane_bypass_retreat_clearance_m(
            info=info,
            privileged_info=privileged_info,
            source_centerline=source,
        )
    else:
        retreat_distance = max(0.75, float(retreat_distance_m))
        blocker_id = blocker_id_hint
    dt = RECOVERY_PLAN_DT_SEC
    # End at the exact bbox clearance instead of continuing to accelerate for
    # all 4.5 seconds.  Also choose the shortest waypoint-aligned smoothstep
    # that respects the reverse-speed cap: stretching a six-metre retreat over
    # the full primitive left the ego in the borrowed lane for two unnecessary
    # seconds and let Route 69's Mustang reach it.  The remaining output rows
    # hold the route-lane endpoint at zero speed.
    active_steps = None
    progresses: List[float] = []
    # Keep at least 2.5 seconds/five controller rows so the one route-lane
    # blend is trackable, while still completing materially sooner than the
    # old 4.5-second retreat.
    for candidate_steps in range(5, LANE_BYPASS_FIXED_STEPS + 1):
        candidate_progresses = [
            float(retreat_distance) * _smoothstep(step / candidate_steps)
            for step in range(1, candidate_steps + 1)
        ]
        candidate_segments = np.diff(
            np.asarray([0.0, *candidate_progresses], dtype=np.float64)
        )
        if (
            np.all(candidate_segments > 1e-4)
            and float(np.max(candidate_segments / dt))
            <= LANE_BYPASS_RETREAT_TARGET_SPEED_MPS + 1e-6
        ):
            active_steps = candidate_steps
            progresses = candidate_progresses
            break
    if active_steps is None:
        LOG.info(
            "HD-map lane-bypass retreat hidden: bbox rear clearance %.2fm for %s "
            "exceeds the %.1fs reverse speed budget",
            retreat_distance,
            blocker_id,
            LANE_BYPASS_FIXED_DURATION_SEC,
        )
        return None
    # Keep one extra blend row for the common shorter clearance, matching the
    # gentler existing route-return geometry; longer bboxes use the reserved
    # four-row blend.
    requested_return_steps = (
        LANE_BYPASS_RETREAT_MIN_RETURN_STEPS + 1
        if retreat_distance < 8.0
        else LANE_BYPASS_RETREAT_MIN_RETURN_STEPS
    )
    actual_return_steps = min(
        int(active_steps) - 1,
        int(requested_return_steps),
    )
    return_start_step = int(active_steps) - actual_return_steps

    points: List[np.ndarray] = []
    for step, distance in enumerate(progresses, start=1):
        signed_distance = -distance
        source_point = _polyline_point_at_signed_distance(source, signed_distance)
        current_point = (
            _polyline_point_at_signed_distance(
                current_centerline, signed_distance
            )
            - current_origin
        )
        if step <= return_start_step:
            source_weight = 0.0
        else:
            source_weight = _smoothstep(
                (step - return_start_step) / actual_return_steps
            )
        points.append(
            (1.0 - source_weight) * current_point
            + source_weight * source_point
        )

    active = np.asarray(points, dtype=np.float64)
    segments = np.diff(
        np.vstack([np.zeros((1, 2), dtype=np.float64), active]), axis=0
    )
    segment_lengths = np.linalg.norm(segments, axis=1)
    if np.any(segment_lengths <= 1e-4):
        return None
    num_poses = max(int(output_num_poses), LANE_BYPASS_FIXED_STEPS)
    trajectory = np.repeat(active[-1][None, :], num_poses, axis=0)
    trajectory[:active_steps] = active
    states = np.zeros((num_poses, 2), dtype=np.float64)
    states[:active_steps, 0] = -(segment_lengths / dt)
    # The reverse PID tracks the fixed HD-map geometry directly.  Keeping the
    # feed-forward column zero avoids applying forward-bicycle curvature with
    # the wrong sign during the final reverse lane blend.
    states[:active_steps, 1] = 0.0
    LOG.info(
        "HD-map lane-bypass retreat %s: duration=%.1fs bbox_rear_clearance=%.2fm "
        "blocker=%s return_start=%.1fs source_center_right=%.2fm",
        normalized_direction,
        active_steps * dt,
        retreat_distance,
        blocker_id,
        return_start_step * dt,
        float(source[0, 0]),
    )
    return (
        np.asarray(trajectory, dtype=np.float32),
        np.asarray(states, dtype=np.float32),
    )


def materialize_hd_map_lane_bypass(
    *,
    info: Dict[str, Any],
    privileged_info: Any,
    direction: str,
    output_num_poses: int,
    entry_delay_sec: float,
    obstacle_clearance_m: float,
    temporal_profile: Optional[Dict[str, Any]] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Build a bounded bypass between sampled HD-map lane centerlines.

    The path is expressed in the current vehicle frame, but every active point
    is interpolated from the route-lane and adjacent-lane centerlines sampled
    from CARLA's HD map.  This removes the old dependence on accumulated vehicle
    yaw and prevents a nominal one-lane shift from turning into a shoulder-sized
    lateral excursion.
    """
    if float(entry_delay_sec or 0.0) > 1e-3:
        # The latched state owns traffic waiting.  Do not bury a multi-second
        # wait inside the committed maneuver and violate its <5 s contract.
        return None
    normalized_direction = "left" if str(direction).lower().startswith("l") else "right"
    descriptors = info.get("adjacent_lane_flows")
    descriptor = descriptors.get(normalized_direction) if isinstance(descriptors, dict) else None
    if not isinstance(descriptor, dict):
        LOG.info(
            "HD-map lane bypass hidden: no %s adjacent-lane descriptor",
            normalized_direction,
        )
        return None
    try:
        source = np.asarray(
            descriptor.get("source_centerline_local_xy"), dtype=np.float64
        )
        target = np.asarray(
            descriptor.get("target_centerline_local_xy"), dtype=np.float64
        )
    except (TypeError, ValueError) as exc:
        LOG.info(
            "HD-map lane bypass hidden: unusable %s centerline samples (%s)",
            normalized_direction,
            exc,
        )
        return None
    if (
        source.ndim != 2 or target.ndim != 2
        or source.shape[0] < 2 or target.shape[0] < 2
        or source.shape[1] < 2 or target.shape[1] < 2
        or not np.isfinite(source[:, :2]).all()
        or not np.isfinite(target[:, :2]).all()
    ):
        LOG.info(
            "HD-map lane bypass hidden: invalid %s centerlines "
            "(source_shape=%s target_shape=%s)",
            normalized_direction,
            getattr(source, "shape", None),
            getattr(target, "shape", None),
        )
        return None
    source = source[:, :2]
    target = target[:, :2]
    # The topology is route-anchored.  If Loop 1 has already placed the ego in
    # the borrowed lane, that target centerline is near the vehicle origin while
    # the route/source centerline is one lane away.  Do not issue another lane
    # change; hold the existing target centerline until the route-lane blocker
    # bbox is clear, then perform the single return.
    already_in_target_lane = bool(
        abs(float(target[0, 0])) + 0.1 < abs(float(source[0, 0]))
    )

    clearance, entry_distance, blocker_id, _blocker_type = (
        _lane_bypass_bbox_clearance_m(
        info=info,
        privileged_info=privileged_info,
        source_centerline=source,
        fallback_clearance_m=obstacle_clearance_m,
        route_lane_width_m=descriptor.get("lane_width_m"),
        )
    )
    dt = RECOVERY_PLAN_DT_SEC
    initial_speed = max(0.0, extract_current_ego_speed_mps(info) or 0.0)
    temporal = dict(temporal_profile or {})
    legacy_auto = str(temporal.get("profile_kind", "")) == "legacy_auto"
    profile = _lane_bypass_forward_profile(
        initial_speed_mps=initial_speed,
        bbox_clearance_m=clearance,
        target_speed_mps=(
            None if legacy_auto else temporal.get("target_speed_mps")
        ),
        accel_mps2=None if legacy_auto else temporal.get("accel_mps2"),
        total_steps=int(
            temporal.get("total_steps", LANE_BYPASS_FIXED_STEPS)
        ),
        entry_steps=int(
            temporal.get("entry_steps", LANE_BYPASS_ENTRY_STEPS)
        ),
        return_steps=int(
            temporal.get("return_steps", LANE_BYPASS_RETURN_STEPS)
        ),
        settle_steps=int(
            temporal.get("settle_steps", LANE_BYPASS_SETTLE_STEPS)
        ),
    )
    if profile is None:
        LOG.warning(
            "HD-map lane bypass hidden: bbox clearance %.2fm for %s is not "
            "reachable before the %.1fs return deadline",
            clearance,
            blocker_id,
            int(temporal.get("total_steps", LANE_BYPASS_FIXED_STEPS)) * dt,
        )
        return None
    progresses = list(profile["progresses_m"])
    speeds = list(profile["speeds_mps"])
    selected_target_speed = float(profile["target_speed_mps"])
    selected_accel = float(profile["accel_mps2"])
    return_start_step = int(profile["return_start_step"])
    total_steps = int(profile["total_steps"])

    # Keep longitudinal progress consistent with the speed states.  The old
    # bbox entry clamp squeezed the first three rows into about 2.5 m but left
    # row four at its original 12 m station.  That discontinuity made the PID
    # cut back toward the blocker exactly while entering the adjacent lane.
    # Bbox proximity is handled before commit by the same-primitive retreat;
    # the forward trajectory itself must remain one smooth HD-map path.
    path_distances = list(progresses)

    active = _lane_bypass_active_points(
        source_centerline=source,
        target_centerline=target,
        direction=normalized_direction,
        profile=profile,
        already_in_target_lane=already_in_target_lane,
        route_blocker_id=blocker_id,
    )
    static_conflicts = _lane_bypass_target_lane_static_conflicts(
        info=info,
        privileged_info=privileged_info,
        source_centerline=source,
        target_centerline=target,
        planned_path_local_xy=active,
    )
    if static_conflicts:
        LOG.warning(
            "HD-map lane bypass hidden: concrete %s profile intersects "
            "target-lane static bbox(es) %s",
            normalized_direction,
            static_conflicts,
        )
        return None
    segments = np.diff(
        np.vstack([np.zeros((1, 2), dtype=np.float64), active]), axis=0
    )
    segment_lengths = np.linalg.norm(segments, axis=1)
    if np.any(segment_lengths <= 1e-4):
        # CARLA can transiently return source/target samples whose blended
        # positions coincide at the single-row return boundary (especially
        # immediately after a reverse retreat).  The lane identities are still
        # valid; dropping the whole candidate leaves the latched bypass braking
        # forever.  Repair only the degenerate interval along the route's
        # current HD-map tangent and translate the remaining suffix by the same
        # amount.  Lateral coordinates, lane choice, and the one-return-only
        # topology are therefore unchanged.
        repaired_rows: List[int] = []
        for idx in range(len(active)):
            previous = (
                np.zeros(2, dtype=np.float64)
                if idx == 0
                else active[idx - 1]
            )
            segment = active[idx] - previous
            if float(np.linalg.norm(segment)) > 1e-4:
                continue
            distance = path_distances[idx]
            tangent_probe = max(0.25, min(1.0, speeds[idx] * dt))
            tangent = (
                _polyline_point_at_distance(source, distance + tangent_probe)
                - _polyline_point_at_distance(source, distance)
            )
            tangent_norm = float(np.linalg.norm(tangent))
            if tangent_norm <= 1e-6:
                tangent = np.array([0.0, 1.0], dtype=np.float64)
            else:
                tangent = tangent / tangent_norm
                if float(tangent[1]) < 0.0:
                    tangent = -tangent
            previous_progress = path_distances[idx - 1] if idx else 0.0
            expected_step = max(0.25, distance - previous_progress)
            replacement = previous + expected_step * tangent
            correction = replacement - active[idx]
            active[idx:] += correction
            repaired_rows.append(idx + 1)
        segments = np.diff(
            np.vstack([np.zeros((1, 2), dtype=np.float64), active]), axis=0
        )
        segment_lengths = np.linalg.norm(segments, axis=1)
        if np.any(segment_lengths <= 1e-4):
            LOG.info(
                "HD-map lane bypass hidden: degenerate %s path rows=%s "
                "source0=%s target0=%s",
                normalized_direction,
                np.flatnonzero(segment_lengths <= 1e-4).tolist(),
                source[0].tolist(),
                target[0].tolist(),
            )
            return None
        LOG.info(
            "HD-map lane bypass %s repaired coincident path rows=%s "
            "along the route tangent",
            normalized_direction,
            repaired_rows,
        )
    # Do not attach an open-loop steering schedule to the map path.  The
    # committed trajectory is continuously transformed into the current ego
    # frame by the hold controller, so SparseDrive's lateral PID already sees
    # the live heading and cross-track error to the fixed HD-map centreline.
    #
    # A response-compensation experiment used four alternating full/partial
    # steering phases here.  In the physical CARLA vehicle those phases did
    # not line up with the planned 0.5 s rows: Route 69 crossed the adjacent
    # lane, returned, crossed again, and ended on the roadside.  Zero
    # feed-forward makes this primitive one geometric move into the unique
    # route-left lane followed by one geometric return; steering changes now
    # happen only when the live ego-to-map tracking error requires them.
    steer = np.zeros(total_steps, dtype=np.float64)

    num_poses = max(int(output_num_poses), total_steps)
    trajectory = np.repeat(active[-1][None, :], num_poses, axis=0)
    trajectory[:total_steps] = active
    # Padding rows are stationary commands, but their geometry is still used
    # as look-ahead by the lateral PID.  Repeating the final active waypoint
    # leaves that target behind the moving ego during the last 0.5 s row.  The
    # controller then tries to turn back toward it and can flip directly to the
    # opposite steering limit.  Keep zero-speed padding semantics while placing
    # its poses ahead along the final route-aligned tangent.  These are control
    # sentinels only: the state rows below remain [0, 0], so hold duration and
    # maneuver completion timing are unchanged.
    if num_poses > total_steps:
        final_segment = active[-1] - active[-2]
        final_step = float(np.linalg.norm(final_segment))
        if final_step <= 1e-6:
            final_tangent = np.array([0.0, 1.0], dtype=np.float64)
            final_step = max(0.25, float(speeds[-1]) * dt)
        else:
            final_tangent = final_segment / final_step
        # One forward point is sufficient: ``control_pid`` excludes the last
        # item from aim selection, so the following repeated row is its
        # sentinel.  Extending every padding row would invent a long zero-speed
        # route suffix that other trajectory consumers could misinterpret.
        trajectory[total_steps:] = active[-1] + final_step * final_tangent
    states = np.zeros((num_poses, 2), dtype=np.float64)
    states[:total_steps, 0] = np.minimum(
        segment_lengths / dt,
        selected_target_speed,
    )
    states[:total_steps, 1] = steer
    LOG.info(
        "HD-map lane bypass %s: duration=%.1fs bbox_clearance=%.2fm "
        "bbox_entry=%.2fm blocker=%s return_start=%.1fs "
        "target_center_right=%.2fm "
        "target_speed=%.1fm/s accel=%.1fm/s2",
        normalized_direction,
        total_steps * dt,
        clearance,
        float(entry_distance) if entry_distance is not None else float("nan"),
        blocker_id,
        return_start_step * dt,
        float(target[0, 0]),
        selected_target_speed,
        selected_accel,
    )
    return (
        np.asarray(trajectory, dtype=np.float32),
        np.asarray(states, dtype=np.float32),
    )


def materialize_hd_map_lane_change(
    *,
    info: Dict[str, Any],
    direction: str,
    output_num_poses: int,
    duration_sec: float = 3.0,
    start_delay_sec: float = 0.0,
    target_speed_mps: float = 8.0,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Materialize one ordinary lane change on HD-map centreline geometry.

    Unlike the legacy fixed 3.5 m bicycle S-curve, this follows the sampled
    source and target lane centre lines, so curved roads, non-standard lane
    widths, and an initially offset ego all finish on the actual adjacent lane.
    """
    normalized_direction = (
        "left" if str(direction).lower().startswith("l") else "right"
    )
    descriptors = info.get("adjacent_lane_flows")
    descriptor = (
        descriptors.get(normalized_direction)
        if isinstance(descriptors, dict)
        else None
    )
    if not isinstance(descriptor, dict):
        LOG.info(
            "HD-map lane change hidden: no %s adjacent-lane descriptor",
            normalized_direction,
        )
        return None
    if str(descriptor.get("flow") or "unknown").lower() in {"none", "unknown"}:
        LOG.info(
            "HD-map lane change hidden: %s adjacent-lane flow=%s",
            normalized_direction,
            descriptor.get("flow"),
        )
        return None
    try:
        source = np.asarray(
            descriptor.get("source_centerline_local_xy"), dtype=np.float64
        )[:, :2]
        target = np.asarray(
            descriptor.get("target_centerline_local_xy"), dtype=np.float64
        )[:, :2]
    except (TypeError, ValueError, IndexError):
        return None
    if (
        source.ndim != 2
        or target.ndim != 2
        or len(source) < 2
        or len(target) < 2
        or not np.isfinite(source).all()
        or not np.isfinite(target).all()
    ):
        return None

    dt = RECOVERY_PLAN_DT_SEC
    duration = max(2.0 * dt, float(duration_sec))
    delay = max(0.0, float(start_delay_sec))
    active_steps = max(2, int(math.ceil(duration / dt)))
    delay_steps = max(0, int(math.ceil(delay / dt - 1e-9)))
    initial_speed = max(0.0, extract_current_ego_speed_mps(info) or 0.0)
    target_speed = max(0.5, float(target_speed_mps))
    accel_mps2 = 2.0
    progresses: List[float] = []
    speeds: List[float] = []
    progress = 0.0
    previous_speed = initial_speed
    for step in range(1, active_steps + 1):
        speed = min(target_speed, initial_speed + accel_mps2 * step * dt)
        progress += max(0.25, 0.5 * (previous_speed + speed) * dt)
        progresses.append(progress)
        speeds.append(speed)
        previous_speed = speed

    active_points: List[np.ndarray] = []
    for step, distance in enumerate(progresses, start=1):
        source_point = _polyline_point_at_distance(source, distance)
        target_point = _polyline_point_at_distance(target, distance)
        current_heading_point = np.array([0.0, distance], dtype=np.float64)
        map_alignment = _smoothstep(step / active_steps)
        aligned_source = (
            (1.0 - map_alignment) * current_heading_point
            + map_alignment * source_point
        )
        lane_weight = _smoothstep(step / active_steps)
        active_points.append(
            (1.0 - lane_weight) * aligned_source + lane_weight * target_point
        )
    active = np.asarray(active_points, dtype=np.float64)
    if len(active) < 2 or not np.isfinite(active).all():
        return None

    moving_segments = np.diff(
        np.vstack([np.zeros((1, 2), dtype=np.float64), active]), axis=0
    )
    moving_lengths = np.linalg.norm(moving_segments, axis=1)
    if np.any(moving_lengths <= 1e-4):
        return None

    total_steps = delay_steps + active_steps
    num_poses = max(int(output_num_poses), total_steps)
    trajectory = np.zeros((num_poses, 2), dtype=np.float64)
    if delay_steps:
        trajectory[:delay_steps] = 0.0
    trajectory[delay_steps:total_steps] = active
    states = np.zeros((num_poses, 2), dtype=np.float64)
    states[delay_steps:total_steps, 0] = np.minimum(
        moving_lengths / dt,
        target_speed,
    )

    if num_poses > total_steps:
        final_segment = active[-1] - active[-2]
        final_step = float(np.linalg.norm(final_segment))
        if final_step <= 1e-6:
            final_tangent = np.array([0.0, 1.0], dtype=np.float64)
            final_step = max(0.25, speeds[-1] * dt)
        else:
            final_tangent = final_segment / final_step
        trajectory[total_steps:] = active[-1] + final_step * final_tangent

    LOG.info(
        "HD-map lane change %s: duration=%.1fs delay=%.1fs "
        "source0=%s target0=%s endpoint=%s",
        normalized_direction,
        active_steps * dt,
        delay_steps * dt,
        source[0].tolist(),
        target[0].tolist(),
        active[-1].tolist(),
    )
    return (
        np.asarray(trajectory, dtype=np.float32),
        np.asarray(states, dtype=np.float32),
    )


def _quantized_duration_to_cover_distance(
    *,
    initial_speed_mps: float,
    target_speed_mps: float,
    accel_mps2: float,
    distance_m: float,
    dt: float = RECOVERY_PLAN_DT_SEC,
) -> float:
    """Return a waypoint-aligned duration that covers at least ``distance_m``."""
    distance = max(0.0, float(distance_m))
    if distance <= 0.0:
        return 0.0
    duration = 0.0
    step = max(float(dt), 1e-3)
    while _distance_under_acceleration(
        initial_speed_mps,
        target_speed_mps,
        accel_mps2,
        duration,
    ) + 1e-6 < distance:
        duration += step
    return duration


def lane_bypass_gap_assessment(
    *,
    info: Dict[str, Any],
    privileged_info: Any,
    direction: str,
    min_gap_m: float = LANE_BYPASS_MIN_GAP_M,
    clearance_buffer_m: float = LANE_BYPASS_CLEARANCE_BUFFER_M,
    required_time_sec: float = LANE_BYPASS_REQUIRED_TIME_SEC,
    min_time_gap_sec: float = LANE_BYPASS_MIN_TIME_GAP_SEC,
    obstacle_clearance_m: float = LANE_BYPASS_OBSTACLE_CLEARANCE_M,
) -> Dict[str, Any]:
    """Return whether traffic in the target adjacent lane admits the bypass.

    HD-map topology first classifies the immediate target lane as opposing or
    same-direction. Opposing traffic uses the projected arrival-time gate. A
    same-direction lane instead checks both its front and rear vehicles over
    the complete lane-borrow interval. Unknown topology fails closed: visible
    actors cannot establish that an adjacent lane has the intended direction.

    Missing privileged data or ego pose also fails closed.
    """
    normalized_direction = (
        "left" if str(direction).lower().startswith("l") else "right"
    )
    lane_descriptors = info.get("adjacent_lane_flows")
    lane_descriptor: Dict[str, Any] = {}
    if isinstance(lane_descriptors, dict):
        candidate = lane_descriptors.get(normalized_direction)
        if isinstance(candidate, dict):
            lane_descriptor = candidate
        elif isinstance(candidate, str):
            lane_descriptor = {"flow": candidate}
    target_lane_flow = str(lane_descriptor.get("flow") or "unknown").lower()
    if target_lane_flow in {"same", "same_flow", "same-direction"}:
        target_lane_flow = "same_direction"
    elif target_lane_flow in {"opposite", "oncoming", "reverse"}:
        target_lane_flow = "opposing"
    elif target_lane_flow not in {
        "same_direction", "opposing", "none", "unknown"
    }:
        target_lane_flow = "unknown"
    result: Dict[str, Any] = {
        "available": False,
        "direction": normalized_direction,
        "target_lane_flow": target_lane_flow,
        "target_lane_key": lane_descriptor.get("lane_key"),
        "reason": "unavailable",
    }
    if target_lane_flow == "none":
        result["reason"] = "no_adjacent_driving_lane"
        return result
    if target_lane_flow == "unknown":
        result["reason"] = "adjacent_lane_flow_unknown"
        return result
    if privileged_info is None:
        result["reason"] = "no_vehicle_data"
        return result
    agents = privileged_info
    if isinstance(privileged_info, dict):
        if "agents" in privileged_info:
            agents = privileged_info.get("agents")
        elif "objects" in privileged_info:
            agents = privileged_info.get("objects")
        else:
            agents = None
    if agents is None:
        result["reason"] = "no_vehicle_data"
        return result
    try:
        pose = info_to_pose(info)
        rot = np.asarray(pose[:3, :3], dtype=np.float64)
    except Exception:
        result["reason"] = "missing_ego_pose"
        return result

    source_centerline: Optional[np.ndarray] = None
    try:
        source_centerline_candidate = np.asarray(
            lane_descriptor.get("source_centerline_local_xy"),
            dtype=np.float64,
        )[:, :2]
        if (
            source_centerline_candidate.ndim == 2
            and len(source_centerline_candidate) >= 2
            and np.isfinite(source_centerline_candidate).all()
        ):
            source_centerline = source_centerline_candidate
    except (TypeError, ValueError, IndexError):
        source_centerline = None

    target_centerline: Optional[np.ndarray] = None
    try:
        target_centerline_candidate = np.asarray(
            lane_descriptor.get("target_centerline_local_xy"),
            dtype=np.float64,
        )[:, :2]
        if (
            target_centerline_candidate.ndim == 2
            and len(target_centerline_candidate) >= 2
            and np.isfinite(target_centerline_candidate).all()
        ):
            target_centerline = target_centerline_candidate
    except (TypeError, ValueError, IndexError):
        target_centerline = None

    def normalized_lane_key(value: Any) -> Optional[Tuple[int, int, int]]:
        try:
            items = list(value)
            if len(items) != 3:
                return None
            return tuple(int(item) for item in items)
        except (TypeError, ValueError):
            return None

    target_lane_keys = {
        key
        for key in (
            normalized_lane_key(value)
            for value in (lane_descriptor.get("target_lane_keys") or ())
        )
        if key is not None
    }
    immediate_target_lane_key = normalized_lane_key(
        lane_descriptor.get("lane_key")
    )
    if immediate_target_lane_key is not None:
        target_lane_keys.add(immediate_target_lane_key)

    vehicles: List[Dict[str, Any]] = []
    for agent in agents:
        if not isinstance(agent, dict) or str(agent.get("object_type", "vehicle")) != "vehicle":
            continue
        try:
            local_pos_value = agent.get("agent_pos_local")
            if local_pos_value is not None:
                local_pos = np.asarray(local_pos_value, dtype=np.float64).reshape(-1)
                if local_pos.size < 2:
                    continue
                center = local_pos[:2]
            else:
                pos_world = np.asarray(
                    agent.get("agent_pos_world"), dtype=np.float64
                ).reshape(-1)
                if pos_world.size < 2:
                    continue
                if pos_world.size < 3:
                    pos_world = np.pad(pos_world, (0, 3 - pos_world.size))
                center = world_points_to_current_local(
                    pos_world[None, :3], pose
                )[0, :2].astype(np.float64)

            local_vel_value = agent.get("agent_vel_local")
            if local_vel_value is not None:
                local_vel = np.asarray(local_vel_value, dtype=np.float64).reshape(-1)
                if local_vel.size < 2:
                    continue
                velocity = local_vel[:2]
            else:
                vel_vec = np.asarray(
                    agent.get("agent_vel_vec"), dtype=np.float64
                ).reshape(-1)
                if vel_vec.size < 2:
                    continue
                vel_world = np.array(
                    [float(vel_vec[0]), float(vel_vec[1]), 0.0],
                    dtype=np.float64,
                )
                vel_local3 = rot.T @ vel_world
                velocity = np.array(
                    [vel_local3[1], vel_local3[0]], dtype=np.float64
                )
        except (TypeError, ValueError):
            continue

        # Use CARLA's exact bbox for both lane membership and longitudinal
        # clearance.  The compact actor centre/extent remains a fallback for
        # wrappers that do not expose get_world_vertices().
        bbox_corners = _agent_local_bbox_corners(agent, pose)
        if bbox_corners is not None and len(bbox_corners) >= 4:
            bbox_min = np.min(bbox_corners[:, :2], axis=0)
            bbox_max = np.max(bbox_corners[:, :2], axis=0)
            center = 0.5 * (bbox_min + bbox_max)
            bbox_half_width = max(
                0.1, 0.5 * float(bbox_max[0] - bbox_min[0])
            )
            bbox_half_length = max(
                0.1, 0.5 * float(bbox_max[1] - bbox_min[1])
            )
        else:
            bbox_half_width = None
            bbox_half_length = None

        lateral_m = float(center[0])
        forward_m = float(center[1])
        forward_speed_mps = float(velocity[1])
        # Privileged actor snapshots can retain a vehicle for one or more
        # frames after its transform/velocity has become invalid.  Counting a
        # NaN centre as traffic leaves it in neither the ahead nor behind set,
        # so a latched bypass can wait forever with no pair to assess.
        if not all(math.isfinite(value) for value in (
            lateral_m,
            forward_m,
            forward_speed_mps,
        )):
            continue
        extent = list(agent.get("agent_extent") or (4.5, 2.0))
        half_length = (
            float(bbox_half_length)
            if bbox_half_length is not None
            else max(0.1, 0.5 * float(extent[0]))
        )
        half_width = (
            float(bbox_half_width)
            if bbox_half_width is not None
            else max(0.1, 0.5 * float(extent[1]))
        )
        if not math.isfinite(half_length) or not math.isfinite(half_width):
            continue

        target_center_value = lane_descriptor.get("center_right_m")
        lane_width_value = lane_descriptor.get("lane_width_m")
        try:
            target_center_m = float(target_center_value)
        except (TypeError, ValueError):
            target_center_m = math.nan
        try:
            target_lane_width_m = max(1.0, float(lane_width_value))
        except (TypeError, ValueError):
            target_lane_width_m = 3.5

        actor_lane_key = normalized_lane_key(agent.get("agent_lane_key"))
        if actor_lane_key is not None and target_lane_keys:
            # Runtime CARLA actors carry their authoritative map waypoint lane
            # key.  Use the same topology membership test at every range.
            side_matches = actor_lane_key in target_lane_keys
        elif target_centerline is not None:
            # Metadata-less callers (unit tests and non-CARLA perception
            # adapters) use the same HD-map centreline match at every range.
            # The descriptor now spans the full privileged-agent horizon, so
            # there is no distant ego-axis corridor fallback.
            try:
                target_center_at_actor = _centerline_right_at_forward(
                    target_centerline,
                    forward_m,
                )
                # Match centres inside the actual target lane. Adjacent parked
                # vehicles remain obstacles in the full trajectory NC check.
                lateral_tolerance = 0.5 * target_lane_width_m
                side_matches = bool(
                    abs(lateral_m - target_center_at_actor)
                    <= lateral_tolerance
                )
            except ValueError:
                side_matches = False
        elif target_lane_flow == "same_direction" and math.isfinite(target_center_m):
            # Same-direction roads commonly have several lanes on the same
            # side. Match only the immediate lane that this primitive enters.
            lateral_tolerance = 0.5 * target_lane_width_m
            side_matches = abs(lateral_m - target_center_m) <= lateral_tolerance
        else:
            # No authoritative lane identity or usable map centreline means
            # lane membership is unknown. Fail closed instead of guessing from
            # the actor's distance to the ego's current straight axis.
            side_matches = False
        if not side_matches:
            continue

        if target_lane_flow == "opposing":
            oncoming_speed = -forward_speed_mps
            # A car currently stopped in the target lane is still part of the
            # safety window.  It may start and pass later, at which point the
            # latch recomputes the bbox window and proceeds.  Treat a stopped
            # or temporarily wrong-way snapshot as zero-speed occupancy rather
            # than incorrectly declaring the lane empty.
            speed_mps = max(0.0, oncoming_speed)
        else:
            # Include stopped and wrong-way actors too: either one can occupy a
            # same-direction insertion gap even though its signed speed differs.
            speed_mps = forward_speed_mps
        vehicles.append(
            {
                "agent_id": str(agent.get("agent_id", "unknown")),
                "center_forward_m": float(center[1]),
                "half_length_m": half_length,
                "half_width_m": half_width,
                "speed_mps": speed_mps,
            }
        )
    nominal_required_time = max(0.0, float(required_time_sec))
    minimum_time_gap = max(0.0, float(min_time_gap_sec))
    obstacle_clearance = max(0.0, float(obstacle_clearance_m))
    route_blocker_entry_distance: Optional[float] = None
    route_blocker_id: Optional[str] = None
    route_blocker_type: Optional[str] = None
    if source_centerline is not None:
        (
            obstacle_clearance,
            route_blocker_entry_distance,
            route_blocker_id,
            route_blocker_type,
        ) = _lane_bypass_bbox_clearance_m(
            info=info,
            privileged_info=privileged_info,
            source_centerline=source_centerline,
            fallback_clearance_m=obstacle_clearance,
            route_lane_width_m=lane_descriptor.get("lane_width_m"),
        )
    ego_speed = extract_current_ego_speed_mps(info) or 0.0
    ego_box = np.asarray(info.get("ego_box", []), dtype=np.float64).reshape(-1)
    ego_half_length = (
        max(0.1, 0.5 * float(ego_box[3])) if ego_box.size > 3 else 2.25
    )
    ego_half_width = (
        max(0.1, 0.5 * float(ego_box[4])) if ego_box.size > 4 else 1.0
    )
    dt = RECOVERY_PLAN_DT_SEC
    accel = max(LANE_BYPASS_ACCEL_MPS2, 1e-3)

    # Steering begins on the first moving tick.  Entry delay is reserved only
    # for a stationary wait for traffic; it must never become straight travel
    # toward the blockage.
    accel_lead_sec = 0.0
    bypass_entry_speed = max(
        ego_speed,
        min(
            LANE_BYPASS_TARGET_SPEED_MPS,
            ego_speed + accel * accel_lead_sec,
        ),
    )
    outward_duration = (
        LANE_BYPASS_LANE_CHANGE_DURATION_SEC
        + LANE_BYPASS_COUNTER_STEER_DURATION_SEC
    )
    borrowed_lane_speed = min(
        LANE_BYPASS_TARGET_SPEED_MPS,
        bypass_entry_speed
        + accel * outward_duration,
    )
    outward_distance = _distance_under_acceleration(
        bypass_entry_speed,
        LANE_BYPASS_TARGET_SPEED_MPS,
        accel,
        outward_duration,
    )
    remaining_obstacle_clearance = max(0.0, obstacle_clearance - outward_distance)
    obstacle_clearance_time = _quantized_duration_to_cover_distance(
        initial_speed_mps=borrowed_lane_speed,
        target_speed_mps=LANE_BYPASS_TARGET_SPEED_MPS,
        accel_mps2=accel,
        distance_m=remaining_obstacle_clearance,
        dt=dt,
    )
    required_time = nominal_required_time + obstacle_clearance_time
    bypass_distance = _distance_under_acceleration(
        bypass_entry_speed,
        LANE_BYPASS_TARGET_SPEED_MPS,
        accel,
        required_time,
    )

    def _common_assessment(**extra: Any) -> Dict[str, Any]:
        assessment: Dict[str, Any] = {
            "direction": result["direction"],
            "target_lane_flow": target_lane_flow,
            "target_lane_key": lane_descriptor.get("lane_key"),
            "target_lane_vehicle_count": len(vehicles),
            "oncoming_vehicle_count": (
                len(vehicles) if target_lane_flow == "opposing" else 0
            ),
            "accel_lead_sec": float(accel_lead_sec),
            "ego_distance_m": float(bypass_distance),
            "required_time_sec": float(required_time),
            "nominal_required_time_sec": float(nominal_required_time),
            "obstacle_clearance_m": float(obstacle_clearance),
            "route_blocker_id": route_blocker_id,
            "route_blocker_type": route_blocker_type,
            "route_blocker_entry_distance_m": (
                None
                if route_blocker_entry_distance is None
                else float(route_blocker_entry_distance)
            ),
            "outward_distance_m": float(outward_distance),
            "remaining_obstacle_clearance_m": float(
                remaining_obstacle_clearance
            ),
            "obstacle_clearance_time_sec": float(obstacle_clearance_time),
            "ttc_probe_margin_sec": float(LANE_BYPASS_TTC_PROBE_MARGIN_SEC),
            "min_time_gap_sec": float(minimum_time_gap),
            # Retained only so old configs/log parsers do not break.  It is no
            # longer an availability condition.
            "legacy_min_gap_m": max(0.0, float(min_gap_m)),
            "legacy_clearance_buffer_m": max(
                0.0, float(clearance_buffer_m)
            ),
        }
        assessment.update(extra)
        return assessment

    # Static geometry is independent of traffic flow.  Build the same default
    # HD-map path that execution will track and veto only exact bbox overlap
    # with that complete outward/pass/return footprint.  Temporal leaves repeat
    # this check against their concrete profile in the materializer.
    if source_centerline is not None and target_centerline is not None:
        static_profile = _lane_bypass_forward_profile(
            initial_speed_mps=ego_speed,
            bbox_clearance_m=obstacle_clearance,
        )
        if static_profile is not None:
            already_in_target_for_static = bool(
                abs(float(target_centerline[0, 0])) + 0.1
                < abs(float(source_centerline[0, 0]))
            )
            static_path = _lane_bypass_active_points(
                source_centerline=source_centerline,
                target_centerline=target_centerline,
                direction=normalized_direction,
                profile=static_profile,
                already_in_target_lane=already_in_target_for_static,
                route_blocker_id=route_blocker_id,
            )
            target_static_conflicts = (
                _lane_bypass_target_lane_static_conflicts(
                    info=info,
                    privileged_info=privileged_info,
                    source_centerline=source_centerline,
                    target_centerline=target_centerline,
                    planned_path_local_xy=static_path,
                )
            )
            if target_static_conflicts:
                return _common_assessment(
                    available=False,
                    reason="target_lane_static_blocked",
                    pair_mode=None,
                    entry_delay_sec=None,
                    target_lane_static_conflicting_agent_ids=(
                        target_static_conflicts
                    ),
                    conflicting_agent_id=target_static_conflicts[0],
                    bbox_clearance_m=float(obstacle_clearance),
                )

    if target_lane_flow == "same_direction":
        if not vehicles:
            return _common_assessment(
                available=True,
                reason="same_direction_lane_clear",
                pair_mode="same_direction_empty",
                entry_delay_sec=float(accel_lead_sec),
                pre_entry_ego_distance_m=float(
                    _distance_under_acceleration(
                        ego_speed,
                        LANE_BYPASS_TARGET_SPEED_MPS,
                        accel,
                        accel_lead_sec,
                    )
                ),
                gap_m=None,
                required_gap_m=2.0 * max(0.0, float(clearance_buffer_m)),
                minimum_predicted_clearance_m=None,
                required_clearance_buffer_m=max(
                    0.0, float(clearance_buffer_m)
                ),
                conflicting_agent_id=None,
            )

        required_clearance = max(0.0, float(clearance_buffer_m))
        conflict_duration = (
            required_time + LANE_BYPASS_TTC_PROBE_MARGIN_SEC
        )
        best_failed: Optional[Dict[str, Any]] = None
        entry_times = np.arange(
            accel_lead_sec,
            LANE_BYPASS_MAX_ENTRY_DELAY_SEC + 0.5 * dt,
            dt,
            dtype=np.float64,
        )
        for entry_delay_value in entry_times:
            entry_delay = float(entry_delay_value)
            wait_duration = max(0.0, entry_delay - accel_lead_sec)
            accel_initial_speed = 0.0 if wait_duration > 0.0 else ego_speed

            def ego_distance_at(time_sec: float) -> float:
                moving_time = max(0.0, float(time_sec) - wait_duration)
                return _distance_under_acceleration(
                    accel_initial_speed,
                    LANE_BYPASS_TARGET_SPEED_MPS,
                    accel,
                    moving_time,
                )

            sample_times = np.arange(
                entry_delay,
                entry_delay + conflict_duration + 0.5 * dt,
                dt,
                dtype=np.float64,
            )
            minimum_clearance = math.inf
            conflicting_agent_id: Optional[str] = None
            entry_relative_centers: List[float] = []
            for car in vehicles:
                for sample_time_value in sample_times:
                    sample_time = float(sample_time_value)
                    relative_center = (
                        float(car["center_forward_m"])
                        + float(car["speed_mps"]) * sample_time
                        - ego_distance_at(sample_time)
                    )
                    clearance = (
                        abs(relative_center)
                        - ego_half_length
                        - float(car["half_length_m"])
                    )
                    if clearance < minimum_clearance:
                        minimum_clearance = float(clearance)
                        conflicting_agent_id = str(car["agent_id"])
                entry_relative_centers.append(
                    float(car["center_forward_m"])
                    + float(car["speed_mps"]) * entry_delay
                    - ego_distance_at(entry_delay)
                )

            rear_centers = [value for value in entry_relative_centers if value < 0.0]
            front_centers = [value for value in entry_relative_centers if value >= 0.0]
            rear_center = max(rear_centers) if rear_centers else None
            front_center = min(front_centers) if front_centers else None
            gap_at_entry = (
                float(front_center - rear_center)
                if rear_center is not None and front_center is not None
                else None
            )
            available = minimum_clearance > required_clearance + 1e-6
            assessment = _common_assessment(
                available=bool(available),
                reason=(
                    "safe_same_direction_gap"
                    if available
                    else "same_direction_gap_conflict"
                ),
                pair_mode="same_direction_gap",
                entry_delay_sec=entry_delay,
                pre_entry_ego_distance_m=float(ego_distance_at(entry_delay)),
                gap_m=gap_at_entry,
                gap_at_entry_m=gap_at_entry,
                required_gap_m=2.0 * required_clearance,
                rear_center_at_entry_m=rear_center,
                front_center_at_entry_m=front_center,
                minimum_predicted_clearance_m=float(minimum_clearance),
                required_clearance_buffer_m=float(required_clearance),
                conflict_duration_sec=float(conflict_duration),
                conflicting_agent_id=conflicting_agent_id,
            )
            if available:
                return assessment
            if best_failed is None or minimum_clearance > float(
                best_failed.get("minimum_predicted_clearance_m", -math.inf)
            ):
                best_failed = assessment
        if best_failed is not None:
            return best_failed
        return _common_assessment(
            available=False,
            reason="same_direction_gap_unavailable",
            pair_mode="same_direction_gap",
            entry_delay_sec=None,
        )

    bbox_clearance = obstacle_clearance
    already_in_target_lane = False
    lane_center_separation = max(
        1.0,
        float(lane_descriptor.get("lane_width_m") or 3.5),
    )
    try:
        source = np.asarray(
            lane_descriptor.get("source_centerline_local_xy"),
            dtype=np.float64,
        )[:, :2]
        target = np.asarray(
            lane_descriptor.get("target_centerline_local_xy"),
            dtype=np.float64,
        )[:, :2]
        if (
            source.ndim == 2
            and target.ndim == 2
            and len(source) >= 2
            and len(target) >= 2
            and np.isfinite(source).all()
            and np.isfinite(target).all()
        ):
            bbox_clearance, _entry_distance, _blocker_id, _blocker_type = (
                _lane_bypass_bbox_clearance_m(
                    info=info,
                    privileged_info=privileged_info,
                    source_centerline=source,
                    fallback_clearance_m=obstacle_clearance,
                    route_lane_width_m=lane_descriptor.get("lane_width_m"),
                )
            )
            already_in_target_lane = bool(
                abs(float(target[0, 0])) + 0.1 < abs(float(source[0, 0]))
            )
            lane_center_separation = max(
                1.0,
                float(np.linalg.norm(target[0] - source[0])),
            )
    except (TypeError, ValueError, IndexError):
        pass

    required_bbox_margin = LANE_BYPASS_BBOX_MARGIN_M

    def inverse_smoothstep(value: float) -> float:
        target_value = float(np.clip(value, 0.0, 1.0))
        lo = 0.0
        hi = 1.0
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            if _smoothstep(mid) < target_value:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    max_target_half_width = max(
        [float(car["half_width_m"]) for car in vehicles] or [1.0]
    )
    lateral_overlap_width = ego_half_width + max_target_half_width
    overlap_ratio = float(np.clip(
        lateral_overlap_width / lane_center_separation,
        0.0,
        1.0,
    ))
    entry_conflict_fraction = inverse_smoothstep(1.0 - overlap_ratio)
    return_conflict_fraction = inverse_smoothstep(overlap_ratio)

    def profile_for_entry(entry_delay_sec: float) -> Optional[Dict[str, Any]]:
        # Waiting is owned by the latch and happens at a full stop.  An
        # immediate entry retains measured ego speed; every delayed candidate
        # starts the same atomic primitive from rest.
        start_speed = ego_speed if float(entry_delay_sec) <= 1e-6 else 0.0
        return _lane_bypass_forward_profile(
            initial_speed_mps=start_speed,
            bbox_clearance_m=bbox_clearance,
        )

    def sampled_conflict_clearances(
        *,
        car: Dict[str, Any],
        entry_delay_sec: float,
        profile: Dict[str, Any],
        side: str,
    ) -> Tuple[float, float, float]:
        progresses = list(profile["progresses_m"])
        conflict_start = (
            0.0
            if already_in_target_lane
            else entry_conflict_fraction * LANE_BYPASS_ENTRY_STEPS * dt
        )
        conflict_end = (
            float(profile["return_start_step"]) * dt
            + return_conflict_fraction * LANE_BYPASS_RETURN_STEPS * dt
        )
        sample_times = [conflict_start]
        sample_times.extend(
            float(value)
            for value in np.arange(
                math.ceil(conflict_start / dt) * dt,
                conflict_end,
                dt,
                dtype=np.float64,
            )
            if float(value) > conflict_start + 1e-6
        )
        if conflict_end > conflict_start + 1e-6:
            sample_times.append(conflict_end)

        def ego_progress_at(relative_time: float) -> float:
            time_value = max(0.0, float(relative_time))
            if time_value <= 0.0:
                return 0.0
            scaled = time_value / dt
            lower_step = int(math.floor(scaled))
            if lower_step <= 0:
                return float(progresses[0]) * scaled
            if lower_step >= len(progresses):
                return float(progresses[-1])
            alpha = scaled - lower_step
            return (
                (1.0 - alpha) * float(progresses[lower_step - 1])
                + alpha * float(progresses[lower_step])
            )

        clearances: List[float] = []
        for relative_time in sample_times:
            ego_progress = ego_progress_at(relative_time)
            center = (
                float(car["center_forward_m"])
                - float(car["speed_mps"])
                * (float(entry_delay_sec) + relative_time)
            )
            if side == "behind":
                clearance = (
                    ego_progress
                    - ego_half_length
                    - center
                    - float(car["half_length_m"])
                )
            else:
                clearance = (
                    center
                    - float(car["half_length_m"])
                    - ego_progress
                    - ego_half_length
                )
            clearances.append(float(clearance))
        return (
            float(min(clearances)),
            float(clearances[0]),
            float(clearances[-1]),
        )

    def opposing_assessment(
        *,
        pair_mode: str,
        entry_delay_sec: float,
        previous_car: Optional[Dict[str, Any]],
        next_car: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        profile = profile_for_entry(entry_delay_sec)
        if profile is None:
            return None
        previous_min = math.inf
        previous_entry = None
        if previous_car is not None:
            previous_min, previous_entry, _previous_end = (
                sampled_conflict_clearances(
                    car=previous_car,
                    entry_delay_sec=entry_delay_sec,
                    profile=profile,
                    side="behind",
                )
            )
        next_min = math.inf
        next_entry = None
        next_end = None
        if next_car is not None:
            next_min, next_entry, next_end = sampled_conflict_clearances(
                car=next_car,
                entry_delay_sec=entry_delay_sec,
                profile=profile,
                side="ahead",
            )
        minimum_clearance = min(previous_min, next_min)
        conflicting_agent_id = None
        if previous_min <= next_min and previous_car is not None:
            conflicting_agent_id = str(previous_car["agent_id"])
        elif next_car is not None:
            conflicting_agent_id = str(next_car["agent_id"])
        available = bool(minimum_clearance > required_bbox_margin + 1e-6)
        gap_at_entry = None
        if previous_car is not None and next_car is not None:
            previous_front = (
                float(previous_car["center_forward_m"])
                + float(previous_car["half_length_m"])
                - float(previous_car["speed_mps"]) * float(entry_delay_sec)
            )
            next_rear = (
                float(next_car["center_forward_m"])
                - float(next_car["half_length_m"])
                - float(next_car["speed_mps"]) * float(entry_delay_sec)
            )
            gap_at_entry = float(next_rear - previous_front)
        conflict_duration = float(profile["conflict_duration_sec"])
        conflict_start = (
            0.0
            if already_in_target_lane
            else entry_conflict_fraction * LANE_BYPASS_ENTRY_STEPS * dt
        )
        conflict_end = (
            float(profile["return_start_step"]) * dt
            + return_conflict_fraction * LANE_BYPASS_RETURN_STEPS * dt
        )
        progresses = list(profile["progresses_m"])
        end_scaled = conflict_end / dt
        end_lower = int(math.floor(end_scaled))
        if end_lower <= 0:
            ego_conflict_distance = float(progresses[0]) * end_scaled
        elif end_lower >= len(progresses):
            ego_conflict_distance = float(progresses[-1])
        else:
            end_alpha = end_scaled - end_lower
            ego_conflict_distance = (
                (1.0 - end_alpha) * float(progresses[end_lower - 1])
                + end_alpha * float(progresses[end_lower])
            )
        conflict_duration = float(conflict_end)
        initial_required_ahead_clearance = (
            None
            if next_car is None
            else ego_conflict_distance
            + float(next_car["speed_mps"]) * conflict_duration
            + required_bbox_margin
        )
        required_ahead_clearance = (
            None if next_car is None else required_bbox_margin
        )
        ahead_clearance = next_entry
        relative_closing_speed = (
            None
            if next_car is None
            else float(profile["target_speed_mps"])
            + float(next_car["speed_mps"])
        )
        time_gap = (
            None
            if next_car is None or ahead_clearance is None
            else float(ahead_clearance)
            / max(float(relative_closing_speed), 1e-3)
        )
        return _common_assessment(
            available=available,
            reason=(
                "safe_profiled_bbox_gap"
                if available
                else "profiled_bbox_conflict"
            ),
            pair_mode=pair_mode,
            entry_delay_sec=float(entry_delay_sec),
            pre_entry_ego_distance_m=0.0,
            gap_m=gap_at_entry,
            gap_at_entry_m=gap_at_entry,
            required_gap_m=initial_required_ahead_clearance,
            previous_car_speed_mps=(
                None
                if previous_car is None
                else float(previous_car["speed_mps"])
            ),
            next_car_speed_mps=(
                None if next_car is None else float(next_car["speed_mps"])
            ),
            next_car_bbox_speed_mps=(
                None if next_car is None else float(next_car["speed_mps"])
            ),
            ahead_clearance_m=ahead_clearance,
            projected_ahead_clearance_m=next_min if next_car is not None else None,
            required_ahead_clearance_m=required_ahead_clearance,
            initial_required_ahead_clearance_m=initial_required_ahead_clearance,
            minimum_predicted_clearance_m=float(minimum_clearance),
            conflicting_agent_id=conflicting_agent_id,
            previous_minimum_clearance_m=(
                None if previous_car is None else float(previous_min)
            ),
            next_minimum_clearance_m=(
                None if next_car is None else float(next_min)
            ),
            required_bbox_margin_m=float(required_bbox_margin),
            bbox_clearance_m=float(bbox_clearance),
            conflict_start_sec=(
                float(conflict_start)
            ),
            conflict_end_sec=float(conflict_end),
            conflict_duration_sec=float(conflict_end - conflict_start),
            ego_conflict_distance_m=ego_conflict_distance,
            profile_target_speed_mps=float(profile["target_speed_mps"]),
            profile_accel_mps2=float(profile["accel_mps2"]),
            return_start_sec=float(profile["return_start_step"]) * dt,
            time_gap_sec=time_gap,
            relative_closing_speed_mps=relative_closing_speed,
        )

    if not vehicles:
        profile = profile_for_entry(0.0)
        if profile is None:
            return _common_assessment(
                available=False,
                reason="bypass_profile_cannot_clear_blocker_bbox",
                pair_mode="empty",
                entry_delay_sec=0.0,
                bbox_clearance_m=float(bbox_clearance),
            )
        return _common_assessment(
            available=True,
            reason="no_oncoming_vehicles",
            pair_mode="empty",
            entry_delay_sec=0.0,
            pre_entry_ego_distance_m=0.0,
            gap_m=None,
            required_gap_m=0.0,
            projected_ahead_clearance_m=None,
            required_ahead_clearance_m=0.0,
            minimum_predicted_clearance_m=None,
            required_bbox_margin_m=float(required_bbox_margin),
            bbox_clearance_m=float(bbox_clearance),
            conflict_end_sec=float(profile["conflict_duration_sec"]),
            return_start_sec=float(profile["return_start_step"]) * dt,
            profile_target_speed_mps=float(profile["target_speed_mps"]),
            profile_accel_mps2=float(profile["accel_mps2"]),
            time_gap_sec=None,
            relative_closing_speed_mps=None,
        )

    behind = [
        item for item in vehicles
        if item["center_forward_m"] < 0.0
    ]
    ahead = [
        item for item in vehicles
        if item["center_forward_m"] >= 0.0
    ]
    ahead.sort(key=lambda item: item["center_forward_m"])
    entry_times = np.arange(
        0.0,
        LANE_BYPASS_MAX_ENTRY_DELAY_SEC + 0.5 * dt,
        dt,
        dtype=np.float64,
    )

    if len(vehicles) == 1:
        car = vehicles[0]
        previous_car = car if car in behind else None
        next_car = car if car in ahead else None
        best_failed: Optional[Dict[str, Any]] = None
        for entry_delay_value in entry_times:
            assessment = opposing_assessment(
                pair_mode=("single_behind" if previous_car else "single_ahead"),
                entry_delay_sec=float(entry_delay_value),
                previous_car=previous_car,
                next_car=next_car,
            )
            if assessment is None:
                continue
            if assessment["available"]:
                assessment["reason"] = (
                    "single_vehicle_passed"
                    if previous_car is not None
                    else "single_vehicle_clear_ahead"
                )
                return assessment
            if best_failed is None or float(
                assessment["minimum_predicted_clearance_m"]
            ) > float(best_failed["minimum_predicted_clearance_m"]):
                best_failed = assessment
        if best_failed is not None:
            best_failed["reason"] = "single_vehicle_too_close"
            return best_failed
        return _common_assessment(
            available=False,
            reason="single_vehicle_overlaps_entry",
            pair_mode="single_overlap",
            entry_delay_sec=None,
        )

    pair_candidates: List[
        Tuple[str, Dict[str, Any], Optional[Dict[str, Any]]]
    ] = []
    if behind and ahead:
        pair_candidates.append((
            "bracketed",
            max(behind, key=lambda item: item["center_forward_m"]),
            ahead[0],
        ))
    pair_candidates.extend(
        ("two_ahead", ahead[index], ahead[index + 1])
        for index in range(len(ahead) - 1)
    )
    if behind and not ahead:
        nearest_behind = max(behind, key=lambda item: item["center_forward_m"])
        pair_candidates.append(("all_behind", nearest_behind, None))

    best_failed: Optional[Dict[str, Any]] = None
    for pair_mode, previous_car, next_car in pair_candidates:
        for entry_delay_value in entry_times:
            assessment = opposing_assessment(
                pair_mode=pair_mode,
                entry_delay_sec=float(entry_delay_value),
                previous_car=previous_car,
                next_car=next_car,
            )
            if assessment is None:
                continue
            if assessment["available"]:
                return assessment
            if best_failed is None or float(
                assessment["minimum_predicted_clearance_m"]
            ) > float(best_failed["minimum_predicted_clearance_m"]):
                best_failed = assessment

    if best_failed is not None:
        return best_failed
    return _common_assessment(
        available=False,
        reason="no_pair_clears_before_delay_limit",
        pair_mode=None,
        entry_delay_sec=None,
        oncoming_vehicle_count=len(vehicles),
    )


def _apply_lane_bypass_assessment_to_context(
    ctx: DrivingContext,
    direction: str,
    assessment: Dict[str, Any],
) -> None:
    """Keep runtime enforcement and its scheduled entry aligned with the gate."""
    assessment = lane_bypass_execution_assessment(assessment)
    is_left = str(direction).lower().startswith("l")
    available = bool(assessment.get("available", False))
    delay = max(0.0, float(assessment.get("entry_delay_sec", 0.0) or 0.0))
    ctx.lane_bypass_obstacle_clearance_m = max(
        0.0,
        float(
            assessment.get(
                "obstacle_clearance_m",
                ctx.lane_bypass_obstacle_clearance_m,
            )
            or 0.0
        ),
    )
    if is_left:
        ctx.lane_bypass_left_available = available
        ctx.lane_bypass_left_entry_delay_sec = delay
    else:
        ctx.lane_bypass_right_available = available
        ctx.lane_bypass_right_entry_delay_sec = delay


def lane_bypass_execution_assessment(
    assessment: Dict[str, Any],
) -> Dict[str, Any]:
    """Translate a scheduled traffic gap into immediate execution readiness.

    ``lane_bypass_gap_assessment`` may report a safe gap that begins after an
    ``entry_delay_sec``. The latched orchestrator owns that wait; the fixed
    HD-map primitive is only materializable once the delay reaches zero.
    Keeping these states distinct prevents an ``available=True`` assessment
    from being exposed as a candidate that necessarily returns no trajectory.
    """
    execution = dict(assessment or {})
    scheduled_available = bool(
        execution.get(
            "scheduled_available",
            execution.get("available", False),
        )
    )
    delay = max(
        0.0,
        float(execution.get("entry_delay_sec", 0.0) or 0.0),
    )
    ready_now = bool(scheduled_available and delay <= 1e-3)
    execution["scheduled_available"] = scheduled_available
    execution["ready_now"] = ready_now
    execution["available"] = ready_now
    if scheduled_available and not ready_now:
        execution["scheduled_reason"] = execution.get("reason")
        execution["reason"] = "waiting_entry_delay"
    return execution


# Structured sequence actions exposed to the recovery VLM.  The keys are the
# exact stable ids from CANDIDATE_LADDER: the model, the nominal PDMS sweep, the
# failure memory, and the executor must all talk about the same maneuver.  Do
# not introduce aliases here (the old shift_left/shift_right/forward layer was
# lossy: reverse primitives could pass PDMS and execute as a fallback without
# ever appearing in the model's action space).
RECOVERY_SEQUENCE_ACTION_SPECS: Dict[str, Dict[str, object]] = {
    "brake": {
        "description": "brake to a full stop",
        "fixed_duration_sec": 1.0,
    },
    "wait": {
        "description": "remain stopped and wait",
        "fixed_duration_sec": 1.0,
    },
    "creep": {
        "description": "crawl straight forward at low speed",
        "fixed_duration_sec": 1.5,
        "target_speed_mps": (0.5, 2.0),
    },
    "creep_back": {
        "description": (
            "stop, then back straight up at low speed (heading preserved; "
            "creates space without rotating the nose)"
        ),
        "fixed_duration_sec": 3.0,
        "target_speed_mps": (0.5, 2.0),
    },
    "swerve_left": {
        "description": "swerve left around an obstacle, then stop",
        "fixed_duration_sec": 2.5,
        "lateral_distance_m": (0.5, 2.5),
    },
    "swerve_right": {
        "description": "swerve right around an obstacle, then stop",
        "fixed_duration_sec": 2.5,
        "lateral_distance_m": (0.5, 2.5),
    },
    "back_left": {
        "description": "stop, reverse while steering left, then stop",
        "fixed_duration_sec": 5.0,
        "lateral_distance_m": (0.3, 1.5),
    },
    "back_right": {
        "description": "stop, reverse while steering right, then stop",
        "fixed_duration_sec": 5.0,
        "lateral_distance_m": (0.3, 1.5),
    },
    "back_bypass_left": {
        "description": "reverse, then drive forward around the obstacle on the left",
        "fixed_duration_sec": 6.0,
        "lateral_distance_m": (1.0, 3.0),
    },
    "back_bypass_right": {
        "description": "reverse, then drive forward around the obstacle on the right",
        "fixed_duration_sec": 6.0,
        "lateral_distance_m": (1.0, 3.0),
    },
    "three_point": {
        "description": "perform a bounded three-point turn",
        "fixed_duration_sec": 6.0,
    },
    "lane_left": {
        "description": "change one lane to the left and finish aligned",
        "fixed_duration_sec": 3.0,
    },
    "lane_right": {
        "description": "change one lane to the right and finish aligned",
        "fixed_duration_sec": 3.0,
    },
    "lane_bypass_left": {
        "description": (
            "temporarily borrow the adjacent driving lane on the left, pass "
            "a static obstacle, and return to the original lane"
        ),
        "fixed_duration_sec": LANE_BYPASS_FIXED_DURATION_SEC,
    },
    "lane_bypass_right": {
        "description": (
            "temporarily borrow the adjacent driving lane on the right, pass "
            "a static obstacle, and return to the original lane"
        ),
        "fixed_duration_sec": LANE_BYPASS_FIXED_DURATION_SEC,
    },
}


def get_recovery_sequence_action_specs(
    allowed_primitives: Optional[Sequence[str]] = None,
    *,
    back_steer_max_lateral_m: Optional[float] = None,
) -> Dict[str, Dict[str, object]]:
    """Return the exact configured primitive ids supported in a sequence."""
    allowed = list(dict.fromkeys(
        str(name).strip()
        for name in (allowed_primitives or ())
        if str(name).strip()
    ))
    def _cap_back_steer(specs: Dict[str, Dict[str, object]]) -> Dict[str, Dict[str, object]]:
        # Posture mode: cap the steered backs' lateral request. At the range
        # floor (0.3 m => ~7 deg of nose rotation) the swept footprint stays
        # provably inside the lane on both edges.
        if back_steer_max_lateral_m is None:
            return specs
        for name in ("back_left", "back_right"):
            if name in specs and "lateral_distance_m" in specs[name]:
                lo, hi = specs[name]["lateral_distance_m"]
                specs[name] = {
                    **specs[name],
                    "lateral_distance_m": (
                        float(lo),
                        max(float(lo), min(float(hi), float(back_steer_max_lateral_m))),
                    ),
                }
        return specs

    if not allowed:
        return _cap_back_steer(
            {name: dict(spec) for name, spec in RECOVERY_SEQUENCE_ACTION_SPECS.items()}
        )
    unsupported = [
        name for name in allowed if name not in RECOVERY_SEQUENCE_ACTION_SPECS
    ]
    if unsupported:
        raise ValueError(
            "recovery sequence primitive spec missing for: %s"
            % ", ".join(unsupported)
        )
    return _cap_back_steer(
        {name: dict(RECOVERY_SEQUENCE_ACTION_SPECS[name]) for name in allowed}
    )


def recovery_action_fixed_duration_sec(action: str) -> Optional[float]:
    """Return the system-owned execution time for one recovery action."""
    spec = RECOVERY_SEQUENCE_ACTION_SPECS.get(str(action).strip())
    if spec is None:
        return None
    try:
        duration = float(spec["fixed_duration_sec"])
    except (KeyError, TypeError, ValueError):
        return None
    return duration if duration > 0.0 else None




def _build_recovery_sequence_code(steps: Sequence[Dict[str, Any]]) -> str:
    lines = ["def recovery_sequence():"]
    for step_index, step in enumerate(steps):
        action = str(step["action"])
        duration = float(step["duration_sec"])
        speed = step.get("target_speed_mps")
        lateral = step.get("lateral_distance_m")
        direction = "left" if action.endswith("_left") else "right"
        if action == "brake":
            call = f"emergency_stop(duration={duration!r})"
        elif action == "wait":
            call = f"wait(duration={duration!r})"
        elif action == "creep":
            call = f"creep(target_speed={float(speed)!r}, duration={duration!r})"
        elif action == "creep_back":
            call = (
                "creep_back("
                f"target_speed={float(speed)!r}, duration={duration!r})"
            )
        elif action.startswith("swerve_"):
            # Reserve 1.5 s of the fixed step budget for the stop phase.
            swerve_duration = max(0.5, duration - 1.5)
            call = (
                "swerve_and_stop("
                f"direction={direction!r}, lateral_m={float(lateral)!r}, "
                f"swerve_duration={swerve_duration!r}, settle_duration=1.5)"
            )
        elif action.startswith("back_bypass_"):
            call = (
                "back_then_bypass("
                f"direction={direction!r}, lateral_m={float(lateral)!r})"
            )
        elif action.startswith("back_"):
            call = (
                "back_up_and_steer("
                f"reverse_duration=2.0, direction={direction!r}, "
                f"lateral_m={float(lateral)!r})"
            )
        elif action == "three_point":
            call = "three_point_turn()"
        elif action.startswith("lane_bypass_"):
            call = f"lane_bypass(direction={direction!r})"
        elif action.startswith("lane_"):
            call = (
                "change_lane("
                f"direction={direction!r}, lateral_m=3.5, seg_duration=1.5, "
                f"total_duration={duration!r})"
            )
        else:
            raise ValueError(f"unsupported recovery sequence action: {action!r}")
        tick_limit = max(1, int(round(duration / RECOVERY_PLAN_DT_SEC)))
        tick_var = f"_tick_{step_index}"
        index_var = f"_index_{step_index}"
        lines.extend(
            [
                f"    for {index_var}, {tick_var} in enumerate({call}):",
                f"        if {index_var} >= {tick_limit}:",
                "            break",
                f"        yield {tick_var}",
            ]
        )
    return "\n".join(lines) + "\n"


def materialize_recovery_sequence(
    *,
    steps: Sequence[Dict[str, Any]],
    info: Dict[str, Any],
    privileged_info: Any,
    output_num_poses: int,
    lane_bypass_min_gap_m: float = LANE_BYPASS_MIN_GAP_M,
    lane_bypass_clearance_buffer_m: float = LANE_BYPASS_CLEARANCE_BUFFER_M,
    lane_bypass_required_time_sec: float = LANE_BYPASS_REQUIRED_TIME_SEC,
    lane_bypass_min_time_gap_sec: float = LANE_BYPASS_MIN_TIME_GAP_SEC,
    obstacle_clearance_m: float = LANE_BYPASS_OBSTACLE_CLEARANCE_M,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Materialize a validated multi-action recovery plan as one trajectory."""
    if not steps:
        return None
    if suppress_leftward_options(info) and any(
        is_leftward_primitive(step.get("action")) for step in steps
    ):
        return None
    specs = RECOVERY_SEQUENCE_ACTION_SPECS
    total_duration = 0.0
    normalized: List[Dict[str, Any]] = []
    ctx = build_primitive_context(info, privileged_info)
    for step in steps:
        action = str(step.get("action") or "")
        spec = specs.get(action)
        if spec is None:
            return None
        if action.startswith("lane_bypass_"):
            direction = action[len("lane_bypass_"):]
            assessment = lane_bypass_gap_assessment(
                info=info,
                privileged_info=privileged_info,
                direction=direction,
                min_gap_m=lane_bypass_min_gap_m,
                clearance_buffer_m=lane_bypass_clearance_buffer_m,
                required_time_sec=lane_bypass_required_time_sec,
                min_time_gap_sec=lane_bypass_min_time_gap_sec,
                obstacle_clearance_m=obstacle_clearance_m,
            )
            assessment = lane_bypass_execution_assessment(assessment)
            _apply_lane_bypass_assessment_to_context(ctx, direction, assessment)
            if not assessment["available"]:
                return None
        try:
            duration = float(spec["fixed_duration_sec"])
        except (KeyError, TypeError, ValueError):
            return None
        speed_range = spec.get("target_speed_mps")
        speed = step.get("target_speed_mps")
        if speed_range is None:
            if speed is not None:
                return None
        else:
            try:
                speed = float(speed)
            except (TypeError, ValueError):
                return None
            if not float(speed_range[0]) <= speed <= float(speed_range[1]):
                return None
        lateral_range = spec.get("lateral_distance_m")
        lateral = step.get("lateral_distance_m")
        if lateral_range is None:
            if lateral is not None:
                return None
        else:
            try:
                lateral = float(lateral)
            except (TypeError, ValueError):
                return None
            if not float(lateral_range[0]) <= lateral <= float(lateral_range[1]):
                return None
        total_duration += duration
        normalized.append(
            {
                "action": action,
                "duration_sec": duration,
                "target_speed_mps": speed,
                "lateral_distance_m": lateral,
            }
        )
    code = _build_recovery_sequence_code(normalized)
    duration_poses = int(math.ceil(total_duration / RECOVERY_PLAN_DT_SEC))
    contains_lane_bypass = any(
        str(step.get("action", "")).startswith("lane_bypass_")
        for step in normalized
    )
    if contains_lane_bypass:
        # The adjacent-lane bypass is intentionally atomic and exclusive.
        # Do not splice its HD-map-centreline contract into a longer open-loop
        # sequence whose later actions could invalidate the lane/bbox anchors.
        if len(normalized) != 1:
            return None
        action = str(normalized[0]["action"])
        direction = action[len("lane_bypass_"):]
        assessment = lane_bypass_gap_assessment(
            info=info,
            privileged_info=privileged_info,
            direction=direction,
            min_gap_m=lane_bypass_min_gap_m,
            clearance_buffer_m=lane_bypass_clearance_buffer_m,
            required_time_sec=lane_bypass_required_time_sec,
            min_time_gap_sec=lane_bypass_min_time_gap_sec,
            obstacle_clearance_m=obstacle_clearance_m,
        )
        assessment = lane_bypass_execution_assessment(assessment)
        if not assessment.get("available", False):
            return None
        return materialize_hd_map_lane_bypass(
            info=info,
            privileged_info=privileged_info,
            direction=direction,
            output_num_poses=max(int(output_num_poses), LANE_BYPASS_RECOVERY_MIN_GEN_POSES),
            entry_delay_sec=float(assessment.get("entry_delay_sec", 0.0) or 0.0),
            obstacle_clearance_m=obstacle_clearance_m,
        )
    contains_lane_change = any(
        str(step.get("action", "")) in {"lane_left", "lane_right"}
        for step in normalized
    )
    if contains_lane_change:
        # A map-anchored lane change cannot be spliced with later codegen atoms
        # without re-anchoring those atoms in the target-lane frame.
        if len(normalized) != 1:
            return None
        action = str(normalized[0]["action"])
        lane_change_result = materialize_hd_map_lane_change(
            info=info,
            direction=action[len("lane_"):],
            output_num_poses=max(int(output_num_poses), RECOVERY_MIN_GEN_POSES),
            duration_sec=float(normalized[0]["duration_sec"]),
        )
        if lane_change_result is not None:
            return lane_change_result
    gen_poses = max(
        int(output_num_poses),
        (
            LANE_BYPASS_RECOVERY_MIN_GEN_POSES
            if contains_lane_bypass
            else RECOVERY_MIN_GEN_POSES
        ),
        duration_poses,
    )
    result = run_codegen(code, PrimitiveRuntime(ctx), gen_poses, name="sequence")
    if result is None:
        return None
    traj, states = result
    return np.asarray(traj, dtype=np.float32), np.asarray(states, dtype=np.float32)


def materialize_recovery_action_profile(
    *,
    action: str,
    profile: Dict[str, Any],
    info: Dict[str, Any],
    privileged_info: Any,
    output_num_poses: int,
    lane_bypass_min_gap_m: float = LANE_BYPASS_MIN_GAP_M,
    lane_bypass_clearance_buffer_m: float = LANE_BYPASS_CLEARANCE_BUFFER_M,
    lane_bypass_required_time_sec: float = LANE_BYPASS_REQUIRED_TIME_SEC,
    lane_bypass_min_time_gap_sec: float = LANE_BYPASS_MIN_TIME_GAP_SEC,
    obstacle_clearance_m: float = LANE_BYPASS_OBSTACLE_CLEARANCE_M,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Materialize one scorer-owned temporal leaf for an action.

    Unlike the VLM-facing sequence API, the duration here is chosen by the
    deterministic temporal tree.  All exposed speed/lateral values remain
    inside the action spec, and lane-bypass profiles retain the same topology
    and traffic admission gates as the legacy fixed primitive.
    """
    name = str(action).strip()
    spec = RECOVERY_SEQUENCE_ACTION_SPECS.get(name)
    if spec is None:
        return None
    temporal = dict(profile or {})
    if name.startswith("lane_bypass_"):
        direction = name[len("lane_bypass_"):]
        assessment = lane_bypass_gap_assessment(
            info=info,
            privileged_info=privileged_info,
            direction=direction,
            min_gap_m=lane_bypass_min_gap_m,
            clearance_buffer_m=lane_bypass_clearance_buffer_m,
            required_time_sec=lane_bypass_required_time_sec,
            min_time_gap_sec=lane_bypass_min_time_gap_sec,
            obstacle_clearance_m=obstacle_clearance_m,
        )
        assessment = lane_bypass_execution_assessment(assessment)
        if not assessment.get("available", False):
            return None
        return materialize_hd_map_lane_bypass(
            info=info,
            privileged_info=privileged_info,
            direction=direction,
            output_num_poses=max(
                int(output_num_poses),
                LANE_BYPASS_RECOVERY_MIN_GEN_POSES,
            ),
            entry_delay_sec=float(
                assessment.get("entry_delay_sec", 0.0) or 0.0
            ),
            obstacle_clearance_m=obstacle_clearance_m,
            temporal_profile=temporal,
        )

    try:
        fixed_duration = float(spec["fixed_duration_sec"])
        duration = float(temporal.get("duration_sec", fixed_duration))
        start_delay = float(temporal.get("start_delay_sec", 0.0))
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(start_delay) or start_delay < 0.0:
        return None
    min_duration = max(RECOVERY_PLAN_DT_SEC, 0.5 * fixed_duration)
    max_duration = max(min_duration, 1.5 * fixed_duration)
    if not min_duration <= duration <= max_duration:
        return None

    if name in {"lane_left", "lane_right"}:
        lane_change_result = materialize_hd_map_lane_change(
            info=info,
            direction=name[len("lane_"):],
            output_num_poses=max(int(output_num_poses), RECOVERY_MIN_GEN_POSES),
            duration_sec=duration,
            start_delay_sec=start_delay,
        )
        if lane_change_result is not None:
            return lane_change_result

    speed_range = spec.get("target_speed_mps")
    speed = temporal.get("target_speed_mps")
    if speed_range is None:
        if speed is not None:
            return None
    else:
        try:
            speed = float(speed)
        except (TypeError, ValueError):
            return None
        if not float(speed_range[0]) <= speed <= float(speed_range[1]):
            return None

    lateral_range = spec.get("lateral_distance_m")
    lateral = temporal.get("lateral_distance_m")
    if lateral_range is None:
        if lateral is not None:
            return None
    else:
        try:
            lateral = float(lateral)
        except (TypeError, ValueError):
            return None
        if not float(lateral_range[0]) <= lateral <= float(lateral_range[1]):
            return None

    normalized: List[Dict[str, Any]] = []
    if start_delay > 0.0:
        # A delayed action means "be stopped until this offset, then begin".
        # Use the hard-brake atom so a moving ego decelerates safely during the
        # prefix; from rest this produces an exactly stationary trajectory.
        normalized.append(
            {
                "action": "brake",
                "duration_sec": start_delay,
                "target_speed_mps": None,
                "lateral_distance_m": None,
            }
        )
    normalized.append(
        {
            "action": name,
            "duration_sec": duration,
            "target_speed_mps": speed,
            "lateral_distance_m": lateral,
        }
    )
    code = _build_recovery_sequence_code(normalized)
    duration_poses = int(
        math.ceil((start_delay + duration) / RECOVERY_PLAN_DT_SEC)
    )
    gen_poses = max(int(output_num_poses), RECOVERY_MIN_GEN_POSES, duration_poses)
    result = run_codegen(
        code,
        PrimitiveRuntime(build_primitive_context(info, privileged_info)),
        gen_poses,
        name=f"temporal:{name}",
    )
    if result is None:
        return None
    traj, states = result
    return np.asarray(traj, dtype=np.float32), np.asarray(states, dtype=np.float32)


def build_recovery_primitive_code(
    primitive_name: str,
    *,
    lateral_distance_m: Optional[float] = None,
) -> Optional[str]:
    """Build executable source for a ladder primitive, optionally parameterized."""
    name = str(primitive_name).strip()
    lateral_range = recovery_lateral_distance_range(name)
    if lateral_distance_m is None or lateral_range is None:
        return None

    if name.startswith("swerve_"):
        family = "swerve"
        direction = name[len("swerve_"):]
        call = (
            "swerve_and_stop("
            f"direction={direction!r}, lateral_m={validate_recovery_lateral_distance(family, lateral_distance_m)!r}, "
            "swerve_duration=1.2)"
        )
    elif name.startswith("back_bypass_"):
        family = "back_bypass"
        direction = name[len("back_bypass_"):]
        call = (
            "back_then_bypass("
            f"direction={direction!r}, lateral_m={validate_recovery_lateral_distance(family, lateral_distance_m)!r})"
        )
    elif name.startswith("back_"):
        family = "back"
        direction = name[len("back_"):]
        call = (
            "back_up_and_steer("
            f"reverse_duration=2.0, direction={direction!r}, "
            f"lateral_m={validate_recovery_lateral_distance(family, lateral_distance_m)!r})"
        )
    else:
        return None
    return f"def m():\n    yield from {call}\n"


def materialize_recovery_primitive(
    *,
    primitive_name: str,
    info: Dict[str, Any],
    privileged_info: Any,
    output_num_poses: int,
    lateral_distance_m: Optional[float] = None,
    lane_bypass_min_gap_m: float = LANE_BYPASS_MIN_GAP_M,
    lane_bypass_clearance_buffer_m: float = LANE_BYPASS_CLEARANCE_BUFFER_M,
    lane_bypass_required_time_sec: float = LANE_BYPASS_REQUIRED_TIME_SEC,
    lane_bypass_min_time_gap_sec: float = LANE_BYPASS_MIN_TIME_GAP_SEC,
    obstacle_clearance_m: float = LANE_BYPASS_OBSTACLE_CLEARANCE_M,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Materialize one named primitive using an optional VLM-selected parameter."""
    ctx = build_primitive_context(info, privileged_info)
    ladder = dict(ladder_for_speed(ctx.ego_speed_mps))
    name = str(primitive_name)
    if suppress_leftward_options(info) and is_leftward_primitive(name):
        return None
    if name.startswith("lane_bypass_"):
        direction = name[len("lane_bypass_"):]
        assessment = lane_bypass_gap_assessment(
            info=info,
            privileged_info=privileged_info,
            direction=direction,
            min_gap_m=lane_bypass_min_gap_m,
            clearance_buffer_m=lane_bypass_clearance_buffer_m,
            required_time_sec=lane_bypass_required_time_sec,
            min_time_gap_sec=lane_bypass_min_time_gap_sec,
            obstacle_clearance_m=obstacle_clearance_m,
        )
        assessment = lane_bypass_execution_assessment(assessment)
        _apply_lane_bypass_assessment_to_context(ctx, direction, assessment)
        if not assessment["available"]:
            return None
        return materialize_hd_map_lane_bypass(
            info=info,
            privileged_info=privileged_info,
            direction=direction,
            output_num_poses=max(
                int(output_num_poses), LANE_BYPASS_RECOVERY_MIN_GEN_POSES
            ),
            entry_delay_sec=float(assessment.get("entry_delay_sec", 0.0) or 0.0),
            obstacle_clearance_m=obstacle_clearance_m,
        )
    if name in {"lane_left", "lane_right"}:
        lane_change_result = materialize_hd_map_lane_change(
            info=info,
            direction=name[len("lane_"):],
            output_num_poses=max(int(output_num_poses), RECOVERY_MIN_GEN_POSES),
        )
        if lane_change_result is not None:
            return lane_change_result
    default_code = ladder.get(name)
    if default_code is None:
        return None
    code = build_recovery_primitive_code(
        name,
        lateral_distance_m=lateral_distance_m,
    ) or default_code
    gen_poses = max(
        int(output_num_poses),
        (
            LANE_BYPASS_RECOVERY_MIN_GEN_POSES
            if name.startswith("lane_bypass_")
            else RECOVERY_MIN_GEN_POSES
        ),
    )
    result = run_codegen(code, PrimitiveRuntime(ctx), gen_poses, name=name)
    if result is None:
        return None
    traj, states = result
    return np.asarray(traj, dtype=np.float32), np.asarray(states, dtype=np.float32)


def build_primitive_context(
    info: Dict[str, Any],
    privileged_info: Any,
) -> DrivingContext:
    """Build the per-frame :class:`DrivingContext` the primitives read from."""
    ego_speed = extract_current_ego_speed_mps(info) or 0.0
    route_cmd = command_to_route_instruction(
        info.get("command") if isinstance(info, dict) else None
    )
    if route_cmd not in {"left", "right", "straight"}:
        route_cmd = "straight"
    return DrivingContext(
        ego_speed_mps=float(ego_speed),
        route_command=route_cmd,
        task_target_hint=describe_task_target_hint(info),
        front_vehicle_distance_m=front_vehicle_distance(privileged_info),
    )


def get_primitive_proposals_and_scores(
    *,
    info: Dict[str, Any],
    privileged_info: Any,
    output_num_poses: int,
    topk: Optional[int] = None,
    allowed_primitives: Optional[Sequence[str]] = None,
    lane_bypass_min_gap_m: float = LANE_BYPASS_MIN_GAP_M,
    lane_bypass_clearance_buffer_m: float = LANE_BYPASS_CLEARANCE_BUFFER_M,
    lane_bypass_required_time_sec: float = LANE_BYPASS_REQUIRED_TIME_SEC,
    lane_bypass_min_time_gap_sec: float = LANE_BYPASS_MIN_TIME_GAP_SEC,
    obstacle_clearance_m: float = LANE_BYPASS_OBSTACLE_CLEARANCE_M,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[np.ndarray]]:
    """Materialize the primitive ladder into recovery proposals.

    Returns ``(proposals, scores, names, states)``: ``proposals`` is
    ``[K, output_num_poses, 2]`` already in the CARLA ego-local frame, ``scores``
    the ``[K]`` ladder-order prior, ``names`` the ladder keys, and ``states``
    per-candidate signed ``[K_i, 2]`` ``(speed, steer_rad)`` sequences needed
    to execute reverse maneuvers open-loop. Candidates whose rollout fails are
    skipped; ``K`` may be smaller than the ladder (0 when everything failed).
    """
    ctx = build_primitive_context(info, privileged_info)
    limit = None if topk is None else max(1, int(topk))
    allowed = None
    if allowed_primitives:
        allowed = {str(name).strip() for name in allowed_primitives if str(name).strip()}
        known = {name for name, _code in CANDIDATE_LADDER}
        unknown = sorted(allowed - known)
        if unknown:
            # A stale or renamed ladder id must not take the whole recovery
            # path down with it. Raising here propagated into the
            # orchestrator's blanket ``except`` in _recover_verify_first_pass,
            # which reported "no primitive available" -- and the caller then
            # kept driving the plan the verifier had already rejected. Seen on
            # routes 70/1077 of the 2026-08-18 200-route run: the pre-rename
            # ids oncoming_bypass_{left,right} were still configured, so 22
            # consecutive PDMS=0.000 ticks (NC=0 on a pedestrian 4.8 m ahead)
            # were answered with full throttle. Drop the unknown ids instead
            # and let the remaining ladder be evaluated.
            allowed -= set(unknown)
            LOG.error(
                "unknown recovery primitive(s) ignored: %s; expected stable "
                "ladder ids from CANDIDATE_LADDER (%s)",
                ", ".join(unknown),
                ", ".join(name for name, _code in CANDIDATE_LADDER),
            )
            if not allowed:
                # Every configured id was unknown. Do NOT fall back to the
                # full ladder: that would silently re-enable maneuvers the
                # config meant to forbid. An empty ``allowed`` set yields no
                # candidates, and the caller's no-primitive path brakes.
                LOG.error(
                    "no valid recovery primitive remains after dropping %s; "
                    "recovery will report no primitive available",
                    ", ".join(unknown),
                )
    # Materialize at least RECOVERY_MIN_GEN_POSES so maneuvers that need a longer
    # body + realign leg (change_lane) are not truncated. The orchestrator's own
    # output_num_poses stays the per-tick lookahead window when the hold plays.
    includes_lane_bypass = allowed is None or any(
        name.startswith("lane_bypass_") for name in allowed
    )
    gen_poses = max(
        int(output_num_poses),
        (
            LANE_BYPASS_RECOVERY_MIN_GEN_POSES
            if includes_lane_bypass
            else RECOVERY_MIN_GEN_POSES
        ),
    )
    proposals: List[np.ndarray] = []
    names: List[str] = []
    states_list: List[np.ndarray] = []
    lane_bypass_assessments = {
        direction: lane_bypass_execution_assessment(
            lane_bypass_gap_assessment(
                info=info,
                privileged_info=privileged_info,
                direction=direction,
                min_gap_m=lane_bypass_min_gap_m,
                clearance_buffer_m=lane_bypass_clearance_buffer_m,
                required_time_sec=lane_bypass_required_time_sec,
                min_time_gap_sec=lane_bypass_min_time_gap_sec,
                obstacle_clearance_m=obstacle_clearance_m,
            )
        )
        for direction in ("left", "right")
    }
    for direction, assessment in lane_bypass_assessments.items():
        _apply_lane_bypass_assessment_to_context(ctx, direction, assessment)
    for key, code_str in ladder_for_speed(ctx.ego_speed_mps):
        if allowed is not None and key not in allowed:
            continue
        if suppress_leftward_options(info) and is_leftward_primitive(key):
            continue
        if key.startswith("lane_bypass_"):
            direction = key[len("lane_bypass_"):]
            assessment = lane_bypass_assessments.get(direction, {})
            LOG.warning(
                "lane bypass gate '%s': available=%s scheduled_available=%s "
                "reason=%s "
                "target_lane_flow=%s vehicles=%s pair_mode=%s "
                "entry_delay_sec=%s gap_m=%s required_gap_m=%s "
                "previous_speed_mps=%s next_speed_mps=%s "
                "projected_ahead_clearance_m=%s required_time_sec=%s "
                "time_gap_sec=%s min_time_gap_sec=%s closing_speed_mps=%s "
                "minimum_predicted_clearance_m=%s required_clearance_buffer_m=%s "
                "conflicting_agent_id=%s "
                "ego_distance_m=%s obstacle_clearance_m=%s "
                "outward_distance_m=%s remaining_obstacle_clearance_m=%s "
                "obstacle_clearance_time_sec=%s",
                key,
                assessment.get("available", False),
                assessment.get("scheduled_available", False),
                assessment.get("reason", "unknown"),
                assessment.get("target_lane_flow", "unknown"),
                assessment.get("target_lane_vehicle_count", 0),
                assessment.get("pair_mode"),
                assessment.get("entry_delay_sec"),
                assessment.get("gap_m"),
                assessment.get("required_gap_m"),
                assessment.get("previous_car_speed_mps"),
                assessment.get("next_car_speed_mps"),
                assessment.get("projected_ahead_clearance_m"),
                assessment.get("required_time_sec"),
                assessment.get("time_gap_sec"),
                assessment.get("min_time_gap_sec"),
                assessment.get("relative_closing_speed_mps"),
                assessment.get("minimum_predicted_clearance_m"),
                assessment.get("required_clearance_buffer_m"),
                assessment.get("conflicting_agent_id"),
                assessment.get("ego_distance_m"),
                assessment.get("obstacle_clearance_m"),
                assessment.get("outward_distance_m"),
                assessment.get("remaining_obstacle_clearance_m"),
                assessment.get("obstacle_clearance_time_sec"),
            )
            if not assessment.get("available", False):
                LOG.debug(
                    "primitive recovery candidate '%s' hidden: %s",
                    key,
                    assessment,
                )
                continue
            result = materialize_hd_map_lane_bypass(
                info=info,
                privileged_info=privileged_info,
                direction=direction,
                output_num_poses=gen_poses,
                entry_delay_sec=float(
                    assessment.get("entry_delay_sec", 0.0) or 0.0
                ),
                obstacle_clearance_m=obstacle_clearance_m,
            )
        elif key in {"lane_left", "lane_right"}:
            result = materialize_hd_map_lane_change(
                info=info,
                direction=key[len("lane_"):],
                output_num_poses=gen_poses,
            )
            if result is None:
                # Unit/non-CARLA callers may not publish HD-map descriptors.
                # Preserve the historical primitive as a compatibility
                # fallback; runtime CARLA recovery takes the map path above.
                result = run_codegen(
                    code_str, PrimitiveRuntime(ctx), gen_poses, name=key
                )
        else:
            # Fresh runtime per candidate: the sim integrates state across steps.
            result = run_codegen(code_str, PrimitiveRuntime(ctx), gen_poses, name=key)
        if result is None:
            LOG.warning("primitive recovery candidate '%s' produced no trajectory", key)
            continue
        traj, states = result
        proposals.append(np.asarray(traj, dtype=np.float32))
        states_list.append(np.asarray(states, dtype=np.float32))
        names.append(key)
        if limit is not None and len(proposals) >= limit:
            break
    if not proposals:
        return (
            np.zeros((0, gen_poses, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            [],
            [],
        )
    count = len(proposals)
    if count == 1:
        scores = np.array([_SCORE_TOP], dtype=np.float32)
    else:
        scores = np.linspace(_SCORE_TOP, _SCORE_BOTTOM, num=count).astype(np.float32)
    return np.stack(proposals, axis=0), scores, names, states_list


def build_primitive_candidate_rows(
    proposals: np.ndarray,
    scores: np.ndarray,
    names: List[str],
    states: List[np.ndarray],
    *,
    output_num_poses: int,
    source_name: str,
    topk: int,
) -> List[Dict[str, object]]:
    """Convert primitive proposals into selector candidate rows.

    Rows carry ``execution_states`` (signed speed/steer sequence) and
    ``primitive_name`` on top of the standard row keys so the selected
    maneuver can be executed open-loop and described to the VLM.
    """
    if len(scores) == 0:
        return []
    candidate_limit = max(1, min(int(topk), int(len(scores))))
    rows: List[Dict[str, object]] = []
    # Scores already descend in ladder order; take the head directly.
    for idx in range(candidate_limit):
        full_plan = np.asarray(proposals[idx, :output_num_poses], dtype=np.float32)
        rows.append(
            {
                "source": source_name,
                "proposal_index": int(idx),
                "proposal_score": float(scores[idx]),
                "local_plan": full_plan,
                "execution_plan": full_plan.copy(),
                "execution_states": np.asarray(states[idx], dtype=np.float32),
                "primitive_name": str(names[idx]),
            }
        )
    return rows
