"""Close-range LiDAR geometry validation for BridgeDrive obstacle boxes.

BridgeDrive supplies the semantic detection and orientation.  LiDAR is used
inside a short range to re-measure geometry; a narrow forward-corridor
backstop can also promote a dense, non-ground return when the semantic head
misses an unboxed wall or barrier.  Vehicles retain their model-predicted
hidden face because a solid body normally occludes it; static obstacles use
both longitudinal LiDAR bounds because sparse prop clusters are visible
through their gaps and a delayed model far face would otherwise make
lane-bypass clearance too long.

Inputs use BridgeDrive/CARLA ego axes: x forward, y right, metres.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from agenticdriving.planners.bridgedrive.carla_inputs import (
    BRIDGEDRIVE_SENSOR_SPECS,
)
from agenticdriving.planners.sparsedrivev2.carla_inputs import carla_transform_matrix


_LIDAR_SPECS = {
    str(spec["id"]): dict(spec)
    for spec in BRIDGEDRIVE_SENSOR_SPECS
    if spec.get("type") == "sensor.lidar.ray_cast"
}


def _rotation_matrix_deg(roll: float, pitch: float, yaw: float) -> np.ndarray:
    roll, pitch, yaw = np.radians([roll, pitch, yaw])
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.asarray([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.asarray([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def lidar_points_to_ego(
    lidar_input: Mapping[str, Any] | None,
    *,
    sensor_ids: Tuple[str, ...] = ("lidar1", "lidar2"),
) -> np.ndarray:
    """Convert leaderboard LiDAR payloads to ``[x, y, z, sensor]``.

    The transform intentionally matches LEAD's ``lidar_to_ego_coordinate``
    including its historical half-mount-height z adjustment, so these points
    align with the BridgeDrive boxes produced from the same sensor frames.
    """
    rows = []
    lidar_input = lidar_input or {}
    for sensor_index, sensor_id in enumerate(sensor_ids):
        payload = lidar_input.get(sensor_id)
        if payload is None:
            continue
        if isinstance(payload, tuple) and len(payload) == 2:
            payload = payload[1]
        raw = np.asarray(payload, dtype=np.float64)
        if raw.size == 0:
            continue
        raw = raw.reshape(-1, raw.shape[-1])
        if raw.shape[1] < 3:
            continue
        xyz = raw[:, :3]
        xyz = xyz[np.all(np.isfinite(xyz), axis=1)]
        if len(xyz) == 0:
            continue

        spec = _LIDAR_SPECS[sensor_id]
        rotation = _rotation_matrix_deg(
            float(spec.get("roll", 0.0)),
            float(spec.get("pitch", 0.0)),
            float(spec.get("yaw", 0.0)),
        )
        translation = np.asarray(
            [spec.get("x", 0.0), spec.get("y", 0.0), spec.get("z", 0.0)],
            dtype=np.float64,
        )
        ego_xyz = xyz @ rotation.T + translation.reshape(1, 3)
        ego_xyz[:, 2] -= 0.5 * translation[2]
        rows.append(
            np.column_stack(
                [ego_xyz, np.full(len(ego_xyz), sensor_index, dtype=np.float64)]
            )
        )
    if not rows:
        return np.empty((0, 4), dtype=np.float64)
    return np.concatenate(rows, axis=0)


def front_corridor_static_agent(
    points: np.ndarray,
    ego_pos,
    ego_rot,
    timestamp: float,
    *,
    known_boxes: np.ndarray | None = None,
    score_threshold: float = 0.35,
    max_distance_m: float = 20.0,
    lateral_limit_m: float = 1.1,
    min_points: int = 12,
) -> Dict[str, Any] | None:
    """Add a current collision surface for an unboxed forward obstacle.

    Require dense low returns inside the driving corridor, after removing
    detected traffic. Retain the connected lateral face of the nearest slice
    so a whole-road wall is not reduced to an ego-lane-sized bypass blocker.
    The surface has no motion history and must not enter stationary memory.
    """
    local, _stats = _non_ground_points(
        points,
        max_distance_m=max_distance_m,
        lateral_limit_m=8.0,
        ground_clearance_m=0.2,
        max_object_height_m=1.6,
    )
    # This is a miss backstop, not a second stationary copy of a detected car.
    # Include a small association margin for box/point capture noise.
    boxes = np.asarray(known_boxes if known_boxes is not None else []).reshape(-1, 9)
    for box in boxes:
        if (
            not np.isfinite(box).all()
            or box[8] < score_threshold
            or min(box[2:4]) < 0.05
            or int(box[7]) not in {0, 1, 4, 6, 7}
        ):
            continue
        relative = _box_local_points(local[:, :2], box)
        covered = (np.abs(relative[:, 0]) <= box[2] + 1.0) & (
            np.abs(relative[:, 1]) <= box[3] + 0.3
        )
        local = local[~covered]
    if len(local) < max(1, int(min_points)):
        return None
    front = local[
        (local[:, 0] > 1.0)
        & (local[:, 0] <= float(max_distance_m))
        & (np.abs(local[:, 1]) <= float(lateral_limit_m))
    ]
    if len(front) < max(1, int(min_points)):
        return None
    bins = np.floor(front[:, 0] / 0.5).astype(np.int64)
    unique, counts = np.unique(bins, return_counts=True)
    viable = unique[counts >= max(1, int(min_points))]
    viable = [b for b in viable if np.ptp(front[bins == b, 2]) >= 0.25]
    if len(viable) == 0:
        return None
    nearest_bin = int(viable[np.argmin(viable)])
    # A wall crossing the driving corridor must retain its lateral face;
    # clipping its box to the ego lane falsely creates a bypass through it.
    selected = local[np.floor(local[:, 0] / 0.5).astype(np.int64) == nearest_bin]
    selected = selected[np.abs(selected[:, 1]) <= 8.0]
    # Keep only the contiguous lateral component that touches the corridor.
    y_bins = np.floor(selected[:, 1] / 0.5).astype(np.int64)
    occupied = set(y_bins.tolist())
    component = set(np.floor(front[bins == nearest_bin, 1] / 0.5).astype(np.int64).tolist())
    pending = list(component)
    while pending:
        b = pending.pop()
        for neighbor in (b - 1, b + 1):
            if neighbor in occupied and neighbor not in component:
                component.add(neighbor)
                pending.append(neighbor)
    selected = selected[np.isin(y_bins, list(component))]
    if len(selected) < max(1, int(min_points)):
        return None
    x_lo, x_hi = float(selected[:, 0].min()), float(selected[:, 0].max())
    y_lo, y_hi = float(selected[:, 1].min()), float(selected[:, 1].max())
    z = float(np.median(selected[:, 2]))
    center_local = np.asarray([(x_lo + x_hi) * 0.5, (y_lo + y_hi) * 0.5, 0.0, 1.0])
    world = carla_transform_matrix(ego_pos, ego_rot) @ center_local
    return {
        "agent_id": "bd_lidar_front",
        "timestamp": float(timestamp),
        "agent_pos_world": [float(world[0]), float(world[1]), float(world[2])],
        "agent_heading": float(np.asarray(ego_rot).reshape(-1)[2]),
        "agent_velo": 0.0,
        "agent_vel_vec": [0.0, 0.0],
        "agent_extent": [max(0.5, x_hi - x_lo + 0.2), max(0.5, y_hi - y_lo + 0.2)],
        "object_type": "static",
        "perception_source": "bridgedrive_lidar_corridor",
        "lidar_bbox_confirmed": True,
        "stationary_memory_confirmed": False,
        "lidar_point_count": int(len(selected)),
        "lidar_center_local": [float(center_local[0]), float(center_local[1]), z],
    }


def _box_local_points(points_xy: np.ndarray, box: np.ndarray) -> np.ndarray:
    center = np.asarray(box[:2], dtype=np.float64)
    yaw = float(box[4])
    c, s = math.cos(yaw), math.sin(yaw)
    return (points_xy - center.reshape(1, 2)) @ np.asarray(
        [[c, -s], [s, c]], dtype=np.float64
    )


def _local_delta_to_ego(delta: np.ndarray, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.asarray(delta, dtype=np.float64) @ np.asarray(
        [[c, s], [-s, c]], dtype=np.float64
    )


def box_is_eligible_for_lidar_validation(
    box: np.ndarray,
    *,
    score_threshold: float,
    max_distance_m: float,
    lateral_limit_m: float,
) -> bool:
    """Whether a model box lies in the close-range LiDAR validation gate."""
    row = np.asarray(box, dtype=np.float64).reshape(-1)
    if row.size < 9:
        return False
    x, y = float(row[0]), float(row[1])
    return bool(
        float(row[8]) >= float(score_threshold)
        and int(row[7]) in {0, 4, 5, 6}
        and math.hypot(x, y) <= float(max_distance_m)
        and abs(y) <= float(lateral_limit_m)
    )


def _non_ground_points(
    points: np.ndarray,
    *,
    max_distance_m: float,
    lateral_limit_m: float,
    ground_clearance_m: float,
    max_object_height_m: float,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Remove ego returns and fit a local road plane from the lowest points."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 4)
    stats: Dict[str, Any] = {
        "lidar_point_count": int(len(points)),
        "lidar_non_ground_point_count": 0,
        "lidar_ground_plane": None,
    }
    if len(points) == 0:
        return points, stats

    finite = np.all(np.isfinite(points[:, :3]), axis=1)
    radius = np.linalg.norm(points[:, :2], axis=1)
    # LEAD removes the complete ego footprint before consuming LiDAR.
    outside_ego = (np.abs(points[:, 0]) > 2.4508416652679443) | (
        np.abs(points[:, 1]) > 1.0641621351242065
    )
    region = (
        finite
        & outside_ego
        & (radius <= max_distance_m + 3.0)
        & (np.abs(points[:, 1]) <= lateral_limit_m + 4.0)
    )
    local = points[region]
    if len(local) < 32:
        return np.empty((0, 4), dtype=np.float64), stats

    # In the close/mid-range validation region road points dominate.
    # Iteratively fitting the lower part of the cloud is stable on grades
    # while rejecting obstacle faces.
    z_low, z_high = np.quantile(local[:, 2], [0.02, 0.55])
    ground = local[(local[:, 2] >= z_low - 0.1) & (local[:, 2] <= z_high)]
    plane = None
    for _ in range(3):
        if len(ground) < 16:
            break
        design = np.column_stack([ground[:, 0], ground[:, 1], np.ones(len(ground))])
        plane, *_ = np.linalg.lstsq(design, ground[:, 2], rcond=None)
        # A pathological fit must not turn walls into a road plane.
        plane[:2] = np.clip(plane[:2], -0.35, 0.35)
        height = local[:, 2] - (
            plane[0] * local[:, 0] + plane[1] * local[:, 1] + plane[2]
        )
        ground = local[(height >= -0.2) & (height <= 0.2)]
    if plane is None:
        return np.empty((0, 4), dtype=np.float64), stats

    height = local[:, 2] - (
        plane[0] * local[:, 0] + plane[1] * local[:, 1] + plane[2]
    )
    keep = (height >= ground_clearance_m) & (height <= max_object_height_m)
    result = local[keep]
    stats["lidar_non_ground_point_count"] = int(len(result))
    stats["lidar_ground_plane"] = [float(value) for value in plane]
    return result, stats


def refine_close_boxes_with_lidar(
    boxes_vehicle_system: np.ndarray,
    lidar_points_ego: np.ndarray,
    *,
    score_threshold: float,
    max_distance_m: float = 20.0,
    lateral_limit_m: float = 8.0,
    longitudinal_association_margin_m: float = 1.0,
    lateral_association_margin_m: float = 0.75,
    min_points: int = 12,
    min_vertical_span_m: float = 0.25,
    ground_clearance_m: float = 0.18,
    static_min_points: int | None = None,
    static_min_vertical_span_m: float | None = None,
    static_ground_clearance_m: float | None = None,
    max_object_height_m: float = 4.0,
    quantile: float = 0.05,
    longitudinal_padding_m: float = 0.10,
    lateral_padding_m: float = 0.20,
    blend: float = 0.85,
    min_extent_scale: float = 0.4,
    max_extent_scale: float = 1.25,
    max_center_shift_m: float = 1.5,
    max_near_edge_shift_m: float = 3.0,
    rebuild_static_longitudinal_bounds: bool = True,
) -> tuple[np.ndarray, Dict[int, Dict[str, Any]], Dict[str, Any]]:
    """Boundedly refine model boxes using close non-ground LiDAR returns."""
    boxes = np.asarray(boxes_vehicle_system, dtype=np.float64).reshape(-1, 9).copy()
    diagnostics: Dict[int, Dict[str, Any]] = {}
    eligible = [
        index
        for index, box in enumerate(boxes)
        if box_is_eligible_for_lidar_validation(
            box,
            score_threshold=score_threshold,
            max_distance_m=max_distance_m,
            lateral_limit_m=lateral_limit_m,
        )
    ]
    static_min_points = max(1, int(
        min_points if static_min_points is None else static_min_points
    ))
    static_min_vertical_span_m = (
        min_vertical_span_m if static_min_vertical_span_m is None
        else max(0.0, float(static_min_vertical_span_m))
    )
    static_ground_clearance_m = (
        ground_clearance_m if static_ground_clearance_m is None
        else max(0.0, float(static_ground_clearance_m))
    )
    points, stats = _non_ground_points(
        lidar_points_ego,
        max_distance_m=max_distance_m,
        lateral_limit_m=lateral_limit_m,
        ground_clearance_m=max(0.0, min(ground_clearance_m, static_ground_clearance_m)),
        max_object_height_m=max(ground_clearance_m, float(max_object_height_m)),
    )
    stats["eligible_box_count"] = len(eligible)
    stats["confirmed_box_count"] = 0
    if not eligible or len(points) == 0:
        return boxes, diagnostics, stats

    plane = stats["lidar_ground_plane"]
    heights = points[:, 2] - (plane[0] * points[:, 0] + plane[1] * points[:, 1] + plane[2])
    quantile = min(0.35, max(0.0, float(quantile)))
    blend = min(1.0, max(0.0, float(blend)))
    min_points = max(1, int(min_points))
    min_extent_scale = max(0.1, float(min_extent_scale))
    max_extent_scale = max(min_extent_scale, float(max_extent_scale))
    assigned: Dict[int, list[int]] = {index: [] for index in eligible}
    for point_index, point in enumerate(points):
        best = None
        for box_index in eligible:
            box = boxes[box_index]
            local = _box_local_points(point[None, :2], box)[0]
            gate_len = max(0.25, float(box[2])) + longitudinal_association_margin_m
            gate_wid = max(0.25, float(box[3])) + lateral_association_margin_m
            if abs(local[0]) > gate_len or abs(local[1]) > gate_wid:
                continue
            distance = (local[0] / gate_len) ** 2 + (local[1] / gate_wid) ** 2
            if best is None or distance < best[0]:
                best = (distance, box_index)
        if best is not None:
            assigned[best[1]].append(point_index)

    for box_index in eligible:
        is_static = int(boxes[box_index, 7]) == 5
        clearance = static_ground_clearance_m if is_static else ground_clearance_m
        # Assign first, then filter: low prop returns must neither count toward
        # vehicle confirmation nor change ownership of higher vehicle returns.
        indices = [i for i in assigned[box_index] if heights[i] >= max(0.0, clearance)]
        required_points = static_min_points if is_static else min_points
        required_span = static_min_vertical_span_m if is_static else min_vertical_span_m
        # Cone-sized props may span only two LiDAR beams. Requiring the
        # vertical support of a larger object erases valid small obstacles.
        small_static = (
            int(boxes[box_index, 7]) == 5
            and max(boxes[box_index, 2:4]) < 0.25
        )
        required_points = min(required_points, 2) if small_static else required_points
        required_span = (
            min(float(required_span), 0.08)
            if small_static else float(required_span)
        )
        if len(indices) < required_points:
            continue
        samples = points[indices]
        low_z, high_z = np.quantile(samples[:, 2], [quantile, 1.0 - quantile])
        vertical_span = float(high_z - low_z)
        if vertical_span < required_span:
            continue

        box = boxes[box_index].copy()
        local = _box_local_points(samples[:, :2], box)
        original_half_len = max(0.05, float(box[2]))
        original_half_wid = max(0.05, float(box[3]))
        original_length = 2.0 * original_half_len
        original_width = 2.0 * original_half_wid
        # A few returns on a small prop reveal its surface, not a smaller
        # physical footprint. Preserve the detector's size lower bound.
        extent_scale_floor = (
            max(1.0, min_extent_scale)
            if int(box[7]) == 5 and min(original_length, original_width) < 0.5
            else min_extent_scale
        )
        original_min_x, original_max_x = -original_half_len, original_half_len
        original_min_y, original_max_y = -original_half_wid, original_half_wid
        low_x, high_x = np.quantile(local[:, 0], [quantile, 1.0 - quantile])
        low_y, high_y = np.quantile(local[:, 1], [quantile, 1.0 - quantile])

        ego_local = _box_local_points(np.zeros((1, 2), dtype=np.float64), box)[0]
        # Dense, vertically supported returns on a stopped object's near face
        # need not retain 15% of an inward-biased detector edge. Only shorten
        # the box; keep the padded measurement and all existing shift bounds.
        measured_near_face = bool(
            int(box[7]) in {0, 4, 6} and abs(float(box[5])) <= 0.3
            and len(samples) >= max(100, required_points)
            and vertical_span >= max(0.5, required_span)
            and abs(float(ego_local[0])) > original_half_len
        )
        lidar_min_x = float(low_x) - longitudinal_padding_m
        lidar_max_x = float(high_x) + longitudinal_padding_m
        measured_longitudinal_span = max(0.0, lidar_max_x - lidar_min_x)
        rebuild_full_length = bool(
            rebuild_static_longitudinal_bounds
            and int(box[7]) == 5
            # A single opaque face does not reveal the hidden edge. Require
            # meaningful depth support before treating a static detection as
            # a penetrable prop cluster with both boundaries observed.
            and measured_longitudinal_span
            >= max(0.5, 0.2 * original_length)
        )
        if rebuild_full_length:
            # Custom/static props (cones, gnomes, barriers) are not a single
            # opaque body. LiDAR commonly sees the complete cluster, so keep
            # neither model edge as an anchor. This specifically removes the
            # delayed far edge that lengthens a committed lane bypass.
            lidar_min_x = float(
                np.clip(
                    lidar_min_x,
                    original_min_x - max_near_edge_shift_m,
                    original_min_x + max_near_edge_shift_m,
                )
            )
            lidar_max_x = float(
                np.clip(
                    lidar_max_x,
                    original_max_x - max_near_edge_shift_m,
                    original_max_x + max_near_edge_shift_m,
                )
            )
            fused_min_x = (1.0 - blend) * original_min_x + blend * lidar_min_x
            fused_max_x = (1.0 - blend) * original_max_x + blend * lidar_max_x
            near_error = (
                lidar_min_x - original_min_x
                if ego_local[0] <= 0.0
                else lidar_max_x - original_max_x
            )
            far_error = (
                lidar_max_x - original_max_x
                if ego_local[0] <= 0.0
                else lidar_min_x - original_min_x
            )
            anchor_positive = None
        elif ego_local[0] <= 0.0:
            lidar_near = lidar_min_x
            lidar_near = float(
                np.clip(
                    lidar_near,
                    original_min_x - max_near_edge_shift_m,
                    original_min_x + max_near_edge_shift_m,
                )
            )
            fused_min_x = (1.0 - blend) * original_min_x + blend * lidar_near
            fused_max_x = original_max_x
            near_error = lidar_near - original_min_x
            far_error = 0.0
            anchor_positive = True
        else:
            lidar_near = lidar_max_x
            lidar_near = float(
                np.clip(
                    lidar_near,
                    original_max_x - max_near_edge_shift_m,
                    original_max_x + max_near_edge_shift_m,
                )
            )
            fused_min_x = original_min_x
            fused_max_x = (1.0 - blend) * original_max_x + blend * lidar_near
            near_error = lidar_near - original_max_x
            far_error = 0.0
            anchor_positive = False

        bounded_length = float(
            np.clip(
                fused_max_x - fused_min_x,
                original_length * extent_scale_floor,
                original_length * max_extent_scale,
            )
        )
        if anchor_positive is True:
            fused_min_x = fused_max_x - bounded_length
        elif anchor_positive is False:
            fused_max_x = fused_min_x + bounded_length
        else:
            longitudinal_center = 0.5 * (fused_min_x + fused_max_x)
            fused_min_x = longitudinal_center - 0.5 * bounded_length
            fused_max_x = longitudinal_center + 0.5 * bounded_length

        lidar_min_y = float(low_y) - lateral_padding_m
        lidar_max_y = float(high_y) + lateral_padding_m
        fused_min_y = (1.0 - blend) * original_min_y + blend * lidar_min_y
        fused_max_y = (1.0 - blend) * original_max_y + blend * lidar_max_y
        bounded_width = float(
            np.clip(
                fused_max_y - fused_min_y,
                original_width * extent_scale_floor,
                original_width * max_extent_scale,
            )
        )
        center_y = 0.5 * (fused_min_y + fused_max_y)
        fused_min_y = center_y - 0.5 * bounded_width
        fused_max_y = center_y + 0.5 * bounded_width

        local_center_delta = np.asarray(
            [0.5 * (fused_min_x + fused_max_x), 0.5 * (fused_min_y + fused_max_y)],
            dtype=np.float64,
        )
        shift_norm = float(np.linalg.norm(local_center_delta))
        center_shift_limit = float(max_center_shift_m)
        if rebuild_full_length:
            center_shift_limit = max(center_shift_limit, float(max_near_edge_shift_m))
        if center_shift_limit <= 0.0:
            local_center_delta[:] = 0.0
        elif shift_norm > center_shift_limit:
            local_center_delta *= center_shift_limit / shift_norm
        near_face_sign = 0
        if measured_near_face:
            sign = -1 if ego_local[0] <= 0.0 else 1
            current_near = local_center_delta[0] + sign * bounded_length / 2.0
            trim = min(max(0.0, sign * (current_near - lidar_near)),
                       max(0.0, bounded_length - original_length * extent_scale_floor))
            corrected_delta = local_center_delta + np.array([-sign * trim / 2.0, 0.0])
            # Apply after the coupled centre-shift limit: the correction must
            # be a subset of the original fit, never move its lateral/far edge.
            if trim > 1e-6 and np.linalg.norm(corrected_delta) <= center_shift_limit:
                local_center_delta = corrected_delta
                bounded_length -= trim
                near_face_sign = sign
        ego_delta = _local_delta_to_ego(local_center_delta, float(box[4]))
        boxes[box_index, :2] = box[:2] + ego_delta
        boxes[box_index, 2] = 0.5 * bounded_length
        boxes[box_index, 3] = 0.5 * bounded_width

        diagnostics[box_index] = {
            "lidar_bbox_confirmed": True,
            "lidar_point_count": int(len(samples)),
            "lidar_sensor_count": int(len(np.unique(samples[:, 3].astype(np.int64)))),
            "lidar_vertical_span_m": vertical_span,
            "lidar_original_extent": [original_length, original_width],
            "lidar_refined_extent": [bounded_length, bounded_width],
            "lidar_center_shift_m": float(np.linalg.norm(ego_delta)),
            "lidar_near_edge_error_m": float(near_error),
            "lidar_far_edge_error_m": float(far_error),
            "lidar_measured_longitudinal_bounds": [lidar_min_x, lidar_max_x],
            "lidar_measured_longitudinal_span_m": measured_longitudinal_span,
            "lidar_bbox_full_extent_confirmed": rebuild_full_length,
            "lidar_near_face_sign": near_face_sign,
        }

    stats["confirmed_box_count"] = len(diagnostics)
    return boxes, diagnostics, stats
