"""CARLA/Bench2Drive input conventions for SparseDriveV2.

This module is intentionally dependency-light so both the CARLA leaderboard
agent and the dedicated SparseDriveV2 worker can share one authoritative sensor
and pose definition.
"""
from __future__ import annotations

import logging
import math
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


LOG = logging.getLogger(__name__)

CAMERAS: List[str] = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

IMAGE_WIDTH = 704
IMAGE_HEIGHT = 396
MODEL_FRAME_RATE = 10.0
JPEG_QUALITY = 20

# The privileged traffic gate can receive actors out to 200 m.  Publish the
# adjacent HD-map lane over the same horizon so every actor is matched against
# one map-derived lane representation instead of switching to an ego-axis
# corridor for distant traffic.
ADJACENT_LANE_CENTERLINE_SAMPLE_DISTANCES_M: Tuple[float, ...] = tuple(
    float(distance_m)
    for distance_m in (
        *range(0, 41, 4),
        *range(50, 201, 10),
    )
)

# Exact sensor definitions from SparseDriveV2's Bench2Drive leaderboard agent.
BENCH2DRIVE_CAMERA_SPECS: Dict[str, Dict[str, float]] = {
    "CAM_FRONT": {
        "x": 0.80, "y": 0.0, "z": 1.60,
        "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
        "width": IMAGE_WIDTH, "height": IMAGE_HEIGHT, "fov": 70.0,
    },
    "CAM_FRONT_RIGHT": {
        "x": 0.27, "y": 0.55, "z": 1.60,
        "roll": 0.0, "pitch": 0.0, "yaw": 55.0,
        "width": IMAGE_WIDTH, "height": IMAGE_HEIGHT, "fov": 70.0,
    },
    "CAM_FRONT_LEFT": {
        "x": 0.27, "y": -0.55, "z": 1.60,
        "roll": 0.0, "pitch": 0.0, "yaw": -55.0,
        "width": IMAGE_WIDTH, "height": IMAGE_HEIGHT, "fov": 70.0,
    },
    "CAM_BACK": {
        "x": -2.0, "y": 0.0, "z": 1.60,
        "roll": 0.0, "pitch": 0.0, "yaw": 180.0,
        "width": IMAGE_WIDTH, "height": IMAGE_HEIGHT, "fov": 110.0,
    },
    "CAM_BACK_LEFT": {
        "x": -0.32, "y": -0.55, "z": 1.60,
        "roll": 0.0, "pitch": 0.0, "yaw": -110.0,
        "width": IMAGE_WIDTH, "height": IMAGE_HEIGHT, "fov": 70.0,
    },
    "CAM_BACK_RIGHT": {
        "x": -0.32, "y": 0.55, "z": 1.60,
        "roll": 0.0, "pitch": 0.0, "yaw": 110.0,
        "width": IMAGE_WIDTH, "height": IMAGE_HEIGHT, "fov": 70.0,
    },
}

LIDAR_MOUNT = np.array([-0.39, 0.0, 1.84], dtype=np.float64)

# CARLA world is left-handed (x forward, y right, z up). SparseDriveV2's
# converted Bench2Drive metadata uses a right-handed global frame and a lidar
# local frame ordered as (x right, y forward, z up).
_LEFT_TO_RIGHT_GLOBAL = np.diag([1.0, -1.0, 1.0, 1.0])
_MODEL_LIDAR_TO_CARLA_SENSOR = np.array(
    [
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def carla_transform_matrix(position, rotation) -> np.ndarray:
    """Build CARLA's actor-to-world matrix from radians.

    ``position`` is ``[x, y, z]`` and ``rotation`` is
    ``[roll, pitch, yaw]``. The signs match CARLA's left-handed transform
    convention rather than scipy's default right-handed Euler convention.
    """
    position = np.asarray(position, dtype=np.float64).reshape(3)
    roll, pitch, yaw = np.asarray(rotation, dtype=np.float64).reshape(3)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.array(
        [
            [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
            [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
            [sp, -cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )
    transform[:3, 3] = position
    return transform


def carla_bbox_world_pose(
    actor_position,
    actor_rotation,
    bbox_location,
    bbox_rotation,
) -> Tuple[np.ndarray, float]:
    """Return a CARLA bounding box's world center and planar heading.

    CARLA stores ``actor.bounding_box.location`` and ``rotation`` in the actor's
    local frame.  Using the actor transform directly is therefore only correct
    for blueprints whose bbox-local transform is identity.  Compose both
    transforms so static props and regular vehicles share the same world OBB
    convention before they enter PDMS.

    All rotations are ``[roll, pitch, yaw]`` in radians.  The returned heading is
    the world angle of the bbox-local +X (length) axis.
    """
    actor_to_world = carla_transform_matrix(actor_position, actor_rotation)
    bbox_to_actor = carla_transform_matrix(bbox_location, bbox_rotation)
    bbox_to_world = actor_to_world @ bbox_to_actor
    center = bbox_to_world[:3, 3].astype(np.float64)
    length_axis_xy = bbox_to_world[:2, 0]
    if float(np.linalg.norm(length_axis_xy)) <= 1e-9:
        heading = float(np.asarray(actor_rotation, dtype=np.float64).reshape(3)[2])
    else:
        heading = math.atan2(float(length_axis_xy[1]), float(length_axis_xy[0]))
    return center, heading


def carla_pose_to_lidar2global(position, rotation) -> np.ndarray:
    """Reproduce the Bench2Drive converter's CARLA lidar-to-global pose."""
    ego2world = carla_transform_matrix(position, rotation)
    lidar2ego_carla = np.eye(4, dtype=np.float64)
    lidar2ego_carla[:3, 3] = LIDAR_MOUNT
    lidar2world_carla = ego2world @ lidar2ego_carla
    return (
        _LEFT_TO_RIGHT_GLOBAL
        @ lidar2world_carla
        @ _MODEL_LIDAR_TO_CARLA_SENSOR
    )


def command_onehot(command: int) -> np.ndarray:
    """Convert CARLA/Bench2Drive RoadOption value 1..6 to a six-way one-hot."""
    try:
        value = int(command)
    except (TypeError, ValueError):
        value = 4
    if value < 1 or value > 6:
        value = 4
    onehot = np.zeros(6, dtype=np.float32)
    onehot[value - 1] = 1.0
    return onehot


def coarse_command_to_road_option(command: int) -> int:
    """Fallback mapping for the legacy coarse command when no route is present."""
    try:
        value = int(command)
    except (TypeError, ValueError):
        value = -1
    return {1: 1, 0: 2, 2: 4}.get(value, 4)


def road_option_value(option: Any) -> int:
    """Normalize a CARLA RoadOption enum/name to Bench2Drive value 1..6."""
    value = getattr(option, "value", None)
    if value is not None:
        try:
            value = int(value)
            if 1 <= value <= 6:
                return value
        except (TypeError, ValueError):
            pass
    name = str(getattr(option, "name", option)).upper()
    if "CHANGELANELEFT" in name or ("CHANGE" in name and "LEFT" in name):
        return 5
    if "CHANGELANERIGHT" in name or ("CHANGE" in name and "RIGHT" in name):
        return 6
    if "LEFT" in name:
        return 1
    if "RIGHT" in name:
        return 2
    if "STRAIGHT" in name:
        return 3
    return 4


def _closest_polyline_projection(
    points: np.ndarray,
    query_xy: np.ndarray,
    *,
    segment_hint: Optional[int] = None,
    search_back: int = 20,
    search_ahead: int = 200,
) -> Tuple[np.ndarray, int, float]:
    """Return the closest point on a directed polyline.

    The segment index follows the route direction: index ``i`` denotes the
    segment from ``points[i]`` to ``points[i + 1]``.  Degenerate polylines are
    handled without exposing special cases to callers.
    """
    if len(points) == 1:
        projection = points[0].copy()
        return projection, 0, float(np.linalg.norm(projection - query_xy))

    segments = points[1:] - points[:-1]
    segment_start = 0
    segment_end = len(segments)
    if segment_hint is not None and len(segments) > 0:
        try:
            hint = int(segment_hint)
        except (TypeError, ValueError):
            hint = 0
        segment_start = max(0, hint - max(0, int(search_back)))
        segment_end = min(
            len(segments), hint + max(0, int(search_ahead)) + 1
        )
        if segment_start >= segment_end:
            segment_start, segment_end = 0, len(segments)

    candidate_segments = segments[segment_start:segment_end]
    candidate_points = points[segment_start:segment_end]
    lengths_sq = np.einsum("ij,ij->i", candidate_segments, candidate_segments)
    relative = query_xy[None, :] - candidate_points
    fractions = np.zeros(len(candidate_segments), dtype=np.float64)
    valid = lengths_sq > 1e-12
    fractions[valid] = np.einsum(
        "ij,ij->i", relative[valid], candidate_segments[valid]
    ) / lengths_sq[valid]
    fractions = np.clip(fractions, 0.0, 1.0)
    projections = candidate_points + fractions[:, None] * candidate_segments
    distances = np.linalg.norm(projections - query_xy[None, :], axis=1)
    local_segment_index = int(distances.argmin())
    segment_index = segment_start + local_segment_index
    return (
        projections[local_segment_index],
        segment_index,
        float(distances[local_segment_index]),
    )


def route_projection_from_polyline(
    points_xy: Sequence[Sequence[float]],
    ego_position: Sequence[float],
    *,
    segment_hint: Optional[int] = None,
    search_back: int = 20,
    search_ahead: int = 200,
) -> Optional[Tuple[np.ndarray, float, int]]:
    """Return the closest route projection, distance, and segment index.

    The projection is used to keep route progress continuous and to construct a
    forward re-entry target. It is deliberately not used to decide whether the
    ego changed lanes; that decision belongs to CARLA's HD-map lane topology.
    """
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 2:
        return None
    ego = np.asarray(ego_position, dtype=np.float64).reshape(-1)
    if ego.size < 2 or not np.isfinite(ego[:2]).all():
        return None
    projection, segment_index, distance = _closest_polyline_projection(
        points,
        ego[:2],
        segment_hint=segment_hint,
        search_back=search_back,
        search_ahead=search_ahead,
    )
    return projection, float(distance), int(segment_index)


def carla_waypoint_lane_key(waypoint: Any) -> Optional[Tuple[int, int, int]]:
    """Return CARLA's stable lane identity tuple for a waypoint-like object."""
    if waypoint is None:
        return None
    try:
        return (
            int(waypoint.road_id),
            int(waypoint.section_id),
            int(waypoint.lane_id),
        )
    except (AttributeError, TypeError, ValueError):
        return None


# ``get_waypoint(project_to_road=False, lane_type=Driving)`` returns None for
# any query point that is not inside a Driving lane. An ego straddling the
# boundary with an adjacent Parking/Shoulder lane therefore resolves to no
# waypoint at all, which silently drops the entire HD-map topology:
# ``adjacent_lane_flows`` stays "unknown" and the recovery bypass gate treats
# unknown as fail-closed, so a latched lane bypass can never open. Projecting
# instead keeps the topology usable, but an unbounded projection would happily
# snap across a median or onto a different road. Accept the projected waypoint
# only within this planar distance -- under one lane width, so a correctly
# positioned ego can never be re-anchored a whole lane away, while the
# half-car-width straddle that motivates the fallback still resolves.
DRIVING_WAYPOINT_PROJECTION_FALLBACK_M = 3.0


def driving_waypoint_at(
    carla_map: Any,
    location: Any,
    *,
    driving_lane_type: Any = None,
    max_projection_m: float = DRIVING_WAYPOINT_PROJECTION_FALLBACK_M,
) -> Tuple[Any, bool]:
    """Resolve a Driving-lane waypoint, projecting only when containment fails.

    Returns ``(waypoint, projected)``. ``projected`` marks the bounded fallback
    so callers that need exact lane membership can tell an on-lane answer from
    a snapped one. Both are None/False when no Driving lane is close enough --
    see DRIVING_WAYPOINT_PROJECTION_FALLBACK_M for why the fallback is bounded.
    """
    if carla_map is None or location is None:
        return None, False

    def lookup(project: bool) -> Any:
        kwargs: Dict[str, Any] = {"project_to_road": project}
        if driving_lane_type is not None:
            kwargs["lane_type"] = driving_lane_type
        try:
            return carla_map.get_waypoint(location, **kwargs)
        except (RuntimeError, TypeError):
            LOG.debug(
                "CARLA Driving-lane lookup failed (project_to_road=%s)",
                project,
                exc_info=True,
            )
            return None

    waypoint = lookup(False)
    if waypoint is not None:
        return waypoint, False
    projected = lookup(True)
    projected_location = getattr(
        getattr(projected, "transform", None), "location", None
    )
    if projected_location is None:
        return None, False
    offset_m = math.hypot(
        float(projected_location.x) - float(location.x),
        float(projected_location.y) - float(location.y),
    )
    if offset_m > float(max_projection_m):
        LOG.debug(
            "Driving-lane projection rejected: nearest lane centre is %.2fm "
            "away (limit %.2fm)",
            offset_m,
            float(max_projection_m),
        )
        return None, False
    # No log line here: this fires every tick the ego stays off-lane, and this
    # module's logger has no handler in the leaderboard client. The caller owns
    # the (rate-limited) reporting via the returned ``projected`` flag.
    return projected, True


def carla_adjacent_lane_flows(
    ego_waypoint: Any,
    *,
    driving_lane_type: Any = None,
    ego_transform: Any = None,
    centerline_sample_distances_m: Sequence[
        float
    ] = ADJACENT_LANE_CENTERLINE_SAMPLE_DISTANCES_M,
) -> Dict[str, Dict[str, Any]]:
    """Describe the immediate left/right driving lanes and their traffic flow.

    The recovery provider must not infer lane direction from whichever actors
    happen to be visible.  CARLA waypoints already encode the authoritative
    travel direction, so publish a small serializable descriptor for each
    physical side of the ego lane.  ``flow`` is one of ``same_direction``,
    ``opposing``, ``none`` (no adjacent driving lane), or ``unknown``.
    """

    def descriptor(
        flow: str,
        waypoint: Any = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        lane_key = carla_waypoint_lane_key(waypoint)
        result: Dict[str, Any] = {
            "flow": flow,
            "lane_key": list(lane_key) if lane_key is not None else None,
        }
        result.update(extra)
        return result

    unknown = {
        "left": descriptor("unknown"),
        "right": descriptor("unknown"),
    }
    ego_key = carla_waypoint_lane_key(ego_waypoint)
    if ego_key is None or bool(getattr(ego_waypoint, "is_junction", False)):
        return unknown

    def is_driving(waypoint: Any) -> bool:
        lane_type = getattr(waypoint, "lane_type", None)
        if driving_lane_type is None:
            return "DRIVING" in str(lane_type).upper()
        try:
            return bool(lane_type & driving_lane_type)
        except TypeError:
            return lane_type == driving_lane_type

    def planar_pose(waypoint: Any) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        transform = getattr(waypoint, "transform", None)
        location = getattr(transform, "location", None)
        if transform is None or location is None:
            return None
        try:
            forward_value = transform.get_forward_vector()
            forward = np.array(
                [float(forward_value.x), float(forward_value.y)],
                dtype=np.float64,
            )
        except Exception:
            rotation = getattr(transform, "rotation", None)
            try:
                yaw = math.radians(float(rotation.yaw))
            except (AttributeError, TypeError, ValueError):
                return None
            forward = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
        norm = float(np.linalg.norm(forward))
        if norm <= 1e-6:
            return None
        return (
            np.array([float(location.x), float(location.y)], dtype=np.float64),
            forward / norm,
        )

    ego_pose = planar_pose(ego_waypoint)
    vehicle_pose = planar_pose(SimpleNamespace(transform=ego_transform))
    reference_pose = vehicle_pose if vehicle_pose is not None else ego_pose

    def local_xy(position: np.ndarray) -> Optional[List[float]]:
        if vehicle_pose is None:
            return None
        vehicle_position, vehicle_forward = vehicle_pose
        vehicle_right = np.array(
            [-vehicle_forward[1], vehicle_forward[0]], dtype=np.float64
        )
        delta = np.asarray(position, dtype=np.float64) - vehicle_position
        return [
            float(np.dot(delta, vehicle_right)),
            float(np.dot(delta, vehicle_forward)),
        ]

    def lane_centerline_local(
        waypoint: Any,
    ) -> Tuple[Optional[List[List[float]]], List[List[int]]]:
        """Sample one lane centerline in the ego vehicle's travel direction.

        CARLA's ``next`` follows a lane's legal traffic direction.  Select
        ``next`` or ``previous`` independently for every lane by comparing its
        HD-map tangent with the *current vehicle* tangent.  This also handles an
        ego that is already travelling against the legal direction of its
        current waypoint while completing an earlier lane borrow.
        """
        if vehicle_pose is None or waypoint is None or reference_pose is None:
            return None, []
        start_pose = planar_pose(waypoint)
        if start_pose is None:
            return None, []
        start_position, lane_forward = start_pose
        _, reference_forward = reference_pose
        accessor = (
            "next"
            if float(np.dot(lane_forward, reference_forward)) >= 0.0
            else "previous"
        )
        reverse_accessor = "previous" if accessor == "next" else "next"
        getter = getattr(waypoint, accessor, None)
        if getter is None:
            return None, []
        samples: List[List[float]] = []
        lane_keys = set()
        start_key = carla_waypoint_lane_key(waypoint)
        if start_key is not None:
            lane_keys.add(start_key)
        for raw_distance in centerline_sample_distances_m:
            try:
                distance = max(0.0, float(raw_distance))
            except (TypeError, ValueError):
                continue
            candidates = [waypoint]
            if distance > 1e-6:
                try:
                    candidates = list(getter(distance))
                except Exception:
                    candidates = []
            best_position = None
            best_progress = -math.inf
            for candidate in candidates:
                candidate_pose = planar_pose(candidate)
                if candidate_pose is None or bool(
                    getattr(candidate, "is_junction", False)
                ):
                    continue
                candidate_key = carla_waypoint_lane_key(candidate)
                if candidate_key is not None:
                    # Every reachable branch can feed traffic into the borrowed
                    # lane, so retain all of their lane identities even though
                    # one branch is selected for the visualization polyline.
                    lane_keys.add(candidate_key)
                candidate_position, _ = candidate_pose
                progress = float(
                    np.dot(candidate_position - start_position, reference_forward)
                )
                if progress > best_progress:
                    best_progress = progress
                    best_position = candidate_position
            if best_position is None:
                continue
            point = local_xy(best_position)
            if point is None:
                continue
            if samples and point[1] <= samples[-1][1] + 1e-3:
                continue
            samples.append(point)

        # Vehicles that have just passed the ego lie behind the adjacent-lane
        # anchor. Include the reverse reachable lane chain as well so they and
        # the next approaching vehicle form one authoritative bracketed gap.
        reverse_getter = getattr(waypoint, reverse_accessor, None)
        if reverse_getter is not None:
            for raw_distance in centerline_sample_distances_m:
                try:
                    distance = max(0.0, float(raw_distance))
                except (TypeError, ValueError):
                    continue
                if distance <= 1e-6:
                    continue
                try:
                    reverse_candidates = list(reverse_getter(distance))
                except Exception:
                    reverse_candidates = []
                for candidate in reverse_candidates:
                    if bool(getattr(candidate, "is_junction", False)):
                        continue
                    candidate_key = carla_waypoint_lane_key(candidate)
                    if candidate_key is not None:
                        lane_keys.add(candidate_key)
        serialized_keys = [list(key) for key in sorted(lane_keys)]
        return (samples if len(samples) >= 2 else None), serialized_keys

    source_centerline_local, source_lane_keys = lane_centerline_local(ego_waypoint)
    try:
        source_lane_width_m = float(ego_waypoint.lane_width)
    except (AttributeError, TypeError, ValueError):
        source_lane_width_m = None
    vehicle_reversed_against_source = False
    if ego_pose is not None and reference_pose is not None:
        _, ego_forward = ego_pose
        _, reference_forward = reference_pose
        vehicle_reversed_against_source = bool(
            float(np.dot(ego_forward, reference_forward)) < 0.0
        )
    result: Dict[str, Dict[str, Any]] = {}
    side_accessors = (
        (("left", "get_right_lane"), ("right", "get_left_lane"))
        if vehicle_reversed_against_source
        else (("left", "get_left_lane"), ("right", "get_right_lane"))
    )
    for side, accessor in side_accessors:
        getter = getattr(ego_waypoint, accessor, None)
        if getter is None:
            result[side] = descriptor("none")
            continue
        try:
            adjacent = getter()
        except Exception:
            result[side] = descriptor("unknown")
            continue
        if adjacent is None or not is_driving(adjacent):
            result[side] = descriptor("none", adjacent)
            continue

        adjacent_key = carla_waypoint_lane_key(adjacent)
        adjacent_pose = planar_pose(adjacent)
        heading_dot: Optional[float] = None
        center_right_m: Optional[float] = None
        if ego_pose is not None and adjacent_pose is not None and reference_pose is not None:
            ego_position, _ = ego_pose
            _, reference_forward = reference_pose
            adjacent_position, adjacent_forward = adjacent_pose
            heading_dot = float(np.dot(reference_forward, adjacent_forward))
            reference_right = np.array(
                [-reference_forward[1], reference_forward[0]], dtype=np.float64
            )
            center_right_m = float(
                np.dot(adjacent_position - ego_position, reference_right)
            )

        if heading_dot is not None and heading_dot >= 0.5:
            flow = "same_direction"
        elif heading_dot is not None and heading_dot <= -0.5:
            flow = "opposing"
        elif adjacent_key is not None and adjacent_key[:2] == ego_key[:2]:
            flow = (
                "opposing"
                if adjacent_key[2] * ego_key[2] < 0
                else "same_direction"
            )
        else:
            flow = "unknown"

        try:
            lane_width_m = float(adjacent.lane_width)
        except (AttributeError, TypeError, ValueError):
            lane_width_m = None
        target_centerline_local, target_lane_keys = lane_centerline_local(adjacent)
        result[side] = descriptor(
            flow,
            adjacent,
            heading_dot=heading_dot,
            center_right_m=center_right_m,
            lane_width_m=lane_width_m,
            source_lane_width_m=source_lane_width_m,
            source_centerline_local_xy=source_centerline_local,
            source_lane_keys=source_lane_keys,
            target_centerline_local_xy=target_centerline_local,
            target_lane_keys=target_lane_keys,
        )
    return result


def carla_route_lane_relation(
    route_waypoint: Any,
    ego_waypoint: Any,
    *,
    max_adjacent_lanes: int = 2,
    driving_lane_type: Any = None,
) -> Optional[str]:
    """Classify the ego lane relative to the HD-map route lane.

    Returns ``"route"``, ``"left"``, or ``"right"`` when the ego occupies the
    route lane or one of its adjacent lanes. ``None`` means the map relation is
    unavailable/ambiguous. Left/right are spatially relative to the route lane,
    so the result remains meaningful even when the adjacent lane is oncoming.
    """
    route_key = carla_waypoint_lane_key(route_waypoint)
    ego_key = carla_waypoint_lane_key(ego_waypoint)
    if route_key is None or ego_key is None:
        return None
    if route_key == ego_key:
        return "route"

    def is_driving(waypoint: Any) -> bool:
        lane_type = getattr(waypoint, "lane_type", None)
        if driving_lane_type is None:
            return "DRIVING" in str(lane_type).upper()
        try:
            return bool(lane_type & driving_lane_type)
        except TypeError:
            return lane_type == driving_lane_type

    for side, accessor in (("left", "get_left_lane"), ("right", "get_right_lane")):
        waypoint = route_waypoint
        visited = {route_key}
        for _ in range(max(0, int(max_adjacent_lanes))):
            getter = getattr(waypoint, accessor, None)
            if getter is None:
                break
            try:
                waypoint = getter()
            except Exception:
                break
            key = carla_waypoint_lane_key(waypoint)
            if key is None or key in visited or not is_driving(waypoint):
                break
            if key == ego_key:
                return side
            visited.add(key)
    return None


def carla_route_waypoint_at_ego_progress(
    route_waypoint: Any,
    ego_waypoint: Any,
    lane_relation: Optional[str],
    *,
    max_adjacent_lanes: int = 2,
    driving_lane_type: Any = None,
) -> Any:
    """Return the route-lane waypoint laterally aligned with the ego.

    A Leaderboard route sample can be tens of metres ahead of the ego. Calling
    ``route_waypoint.next(lookahead)`` from that sparse sample therefore does
    not mean "lookahead metres from the ego".  CARLA's adjacent-lane accessors
    preserve longitudinal progress, so walk from the *ego waypoint* back to the
    known route lane and use that aligned waypoint as the HD-map lookahead
    anchor.

    ``lane_relation`` follows :func:`carla_route_lane_relation`.  The inverse
    walk searches both sides because CARLA defines left/right relative to each
    waypoint's travel direction; across an oncoming lane the accessor that
    leads back to the route can therefore have the same name rather than the
    intuitive opposite name.
    """
    route_key = carla_waypoint_lane_key(route_waypoint)
    ego_key = carla_waypoint_lane_key(ego_waypoint)
    if route_key is None or ego_key is None:
        return None
    if lane_relation == "route":
        return ego_waypoint if ego_key == route_key else None

    if lane_relation not in {"left", "right"}:
        return None

    def is_driving(waypoint: Any) -> bool:
        lane_type = getattr(waypoint, "lane_type", None)
        if driving_lane_type is None:
            return "DRIVING" in str(lane_type).upper()
        try:
            return bool(lane_type & driving_lane_type)
        except TypeError:
            return lane_type == driving_lane_type

    for accessor in ("get_left_lane", "get_right_lane"):
        waypoint = ego_waypoint
        visited = {ego_key}
        for _ in range(max(0, int(max_adjacent_lanes))):
            getter = getattr(waypoint, accessor, None)
            if getter is None:
                break
            try:
                waypoint = getter()
            except Exception:
                break
            key = carla_waypoint_lane_key(waypoint)
            if key is None or key in visited or not is_driving(waypoint):
                break
            if key == route_key:
                return waypoint
            visited.add(key)
    return None


def projected_route_condition_from_polyline(
    points_xy: Sequence[Sequence[float]],
    options: Sequence[Any],
    ego_position,
    ego_rotation,
    lookahead_m: float = 5.0,
    *,
    segment_hint: Optional[int] = None,
    search_back: int = 20,
    search_ahead: int = 200,
) -> Tuple[Optional[List[float]], int]:
    """Build a forward route target from the ego's polyline projection.

    This construction is intentionally reserved for active route re-entry.  A
    borrowed-lane ego must spend the lookahead budget advancing along the
    original route rather than first aiming at a sparse vertex behind it.
    """
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] != 2:
        return None, 4
    if len(options) != len(points):
        raise ValueError("route points and options must have the same length")

    ego_position = np.asarray(ego_position, dtype=np.float64).reshape(3)
    ego_xy = ego_position[:2]
    route_projection, segment_index, _ = _closest_polyline_projection(
        points,
        ego_xy,
        segment_hint=segment_hint,
        search_back=search_back,
        search_ahead=search_ahead,
    )

    remaining = max(0.0, float(lookahead_m))
    cursor = route_projection
    target_world = points[-1]
    target_index = len(points) - 1
    first_route_index = min(segment_index + 1, len(points) - 1)
    for idx in range(first_route_index, len(points)):
        segment = points[idx] - cursor
        segment_length = float(np.linalg.norm(segment))
        if segment_length < 1e-6:
            continue
        if segment_length >= remaining:
            target_world = cursor + segment * (remaining / segment_length)
            target_index = idx
            break
        remaining -= segment_length
        cursor = points[idx]

    waypoint = np.array(
        [target_world[0], target_world[1], ego_position[2], 1.0],
        dtype=np.float64,
    )
    ego_point = np.linalg.inv(
        carla_transform_matrix(ego_position, ego_rotation)
    ) @ waypoint
    right, forward = float(ego_point[1]), float(ego_point[0])
    target = None if forward <= 0.0 else [right, forward]
    return target, road_option_value(options[target_index])


def route_condition_from_polyline(
    points_xy: Sequence[Sequence[float]],
    options: Sequence[Any],
    ego_position,
    ego_rotation,
    lookahead_m: float = 5.0,
) -> Tuple[Optional[List[float]], int]:
    """Interpolate the original CARLA route near point and RoadOption.

    Keep the historical nearest-vertex construction byte-for-byte for NORMAL
    SparseDriveV2 operation.  Active route re-entry uses the explicitly named
    projected helper above, so episodes without a lane departure are untouched.
    """
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] != 2:
        return None, 4
    if len(options) != len(points):
        raise ValueError("route points and options must have the same length")

    ego_position = np.asarray(ego_position, dtype=np.float64).reshape(3)
    ego_xy = ego_position[:2]
    nearest = int(np.linalg.norm(points - ego_xy[None, :], axis=1).argmin())

    remaining = max(0.0, float(lookahead_m))
    cursor = ego_xy
    target_world = points[-1]
    target_index = len(points) - 1
    for idx in range(nearest, len(points)):
        segment = points[idx] - cursor
        segment_length = float(np.linalg.norm(segment))
        if segment_length < 1e-6:
            continue
        if segment_length >= remaining:
            target_world = cursor + segment * (remaining / segment_length)
            target_index = idx
            break
        remaining -= segment_length
        cursor = points[idx]

    waypoint = np.array(
        [target_world[0], target_world[1], ego_position[2], 1.0],
        dtype=np.float64,
    )
    ego_point = np.linalg.inv(
        carla_transform_matrix(ego_position, ego_rotation)
    ) @ waypoint
    right, forward = float(ego_point[1]), float(ego_point[0])
    target = None if forward <= 0.0 else [right, forward]
    return target, road_option_value(options[target_index])
