"""Sensor-only radar validation for BridgeDrive's forward obstacle boxes.

CARLA radar returns are sparse surface samples, not object point clouds.  In
particular, the hidden far face of an obstacle is usually unobservable.  This
module therefore uses radar to refine only geometry that the sensor actually
supports: the visible longitudinal face and the lateral span.  The opposite
longitudinal face remains anchored to BridgeDrive's prediction, and all
changes are bounded before they can reach the judge.

Both inputs use BridgeDrive/CARLA ego axes: x forward, y right, metres.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from agenticdriving.planners.bridgedrive.carla_inputs import (
    BRIDGEDRIVE_SENSOR_SPECS,
)


_RADAR_SPECS = {
    str(spec["id"]): dict(spec)
    for spec in BRIDGEDRIVE_SENSOR_SPECS
    if spec.get("type") == "sensor.other.radar"
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


def radar_points_to_ego(
    radar_input: Mapping[str, Any] | None,
    *,
    sensor_ids: Tuple[str, ...] = ("radar1", "radar2"),
) -> np.ndarray:
    """Convert leaderboard radar payloads to ``[x, y, z, velocity, sensor]``.

    BridgeDrive/LEAD consumes each callback row as
    ``[depth, altitude, azimuth, velocity]`` (the contract used by
    ``common_utils.radar_points_to_ego``). Values may be supplied either
    directly or in leaderboard ``(frame, array)`` form.
    """
    rows = []
    radar_input = radar_input or {}
    for sensor_index, sensor_id in enumerate(sensor_ids):
        payload = radar_input.get(sensor_id)
        if payload is None:
            continue
        if isinstance(payload, tuple) and len(payload) == 2:
            payload = payload[1]
        raw = np.asarray(payload, dtype=np.float64)
        if raw.size == 0:
            continue
        raw = raw.reshape(-1, 4)
        valid = np.all(np.isfinite(raw), axis=1) & (raw[:, 0] > 0.0)
        raw = raw[valid]
        if len(raw) == 0:
            continue

        depth, altitude, azimuth, velocity = raw.T
        cos_altitude = np.cos(altitude)
        sensor_xyz = np.stack(
            [
                depth * np.cos(azimuth) * cos_altitude,
                depth * np.sin(azimuth) * cos_altitude,
                depth * np.sin(altitude),
            ],
            axis=1,
        )
        spec = _RADAR_SPECS[sensor_id]
        rotation = _rotation_matrix_deg(
            float(spec.get("roll", 0.0)),
            float(spec.get("pitch", 0.0)),
            float(spec.get("yaw", 0.0)),
        )
        translation = np.asarray(
            [spec.get("x", 0.0), spec.get("y", 0.0), spec.get("z", 0.0)],
            dtype=np.float64,
        )
        ego_xyz = sensor_xyz @ rotation.T + translation.reshape(1, 3)
        rows.append(
            np.column_stack(
                [
                    ego_xyz,
                    velocity,
                    np.full(len(raw), sensor_index, dtype=np.float64),
                ]
            )
        )
    if not rows:
        return np.empty((0, 5), dtype=np.float64)
    return np.concatenate(rows, axis=0)


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


def box_is_eligible_for_radar_validation(
    box: np.ndarray,
    *,
    score_threshold: float,
    max_distance_m: float,
    lateral_limit_m: float,
    max_heading_error_deg: float,
) -> bool:
    """Whether a BridgeDrive box lies in the reliable forward radar gate."""
    row = np.asarray(box, dtype=np.float64).reshape(-1)
    if row.size < 9:
        return False
    x, y = float(row[0]), float(row[1])
    heading_error = abs(
        math.atan2(math.sin(float(row[4])), math.cos(float(row[4])))
    )
    heading_error = min(heading_error, abs(math.pi - heading_error))
    return bool(
        float(row[8]) >= float(score_threshold)
        and int(row[7]) in {0, 4, 5, 6}
        and x > 0.0
        and math.hypot(x, y) <= float(max_distance_m)
        and abs(y) <= float(lateral_limit_m)
        and heading_error
        <= math.radians(max(0.0, float(max_heading_error_deg)))
    )


def refine_forward_boxes_with_radar(
    boxes_vehicle_system: np.ndarray,
    radar_points_ego: np.ndarray,
    *,
    score_threshold: float,
    max_distance_m: float = 35.0,
    lateral_limit_m: float = 8.0,
    max_heading_error_deg: float = 45.0,
    association_margin_m: float = 1.5,
    min_points: int = 5,
    min_unique_sensors: int = 1,
    quantile: float = 0.1,
    longitudinal_padding_m: float = 0.15,
    lateral_padding_m: float = 0.25,
    blend: float = 0.6,
    min_extent_scale: float = 0.7,
    max_extent_scale: float = 1.3,
    max_center_shift_m: float = 0.75,
) -> tuple[np.ndarray, Dict[int, Dict[str, Any]], Dict[str, Any]]:
    """Boundedly refine directly-forward vehicle/static BridgeDrive boxes.

    Points are associated one-to-one with the closest expanded predicted box.
    Only vehicle/parking/special/static classes (0, 4, 5, 6) participate;
    sparse radar is intentionally not used to resize pedestrians.
    """
    boxes = np.asarray(boxes_vehicle_system, dtype=np.float64).reshape(-1, 9).copy()
    points = np.asarray(radar_points_ego, dtype=np.float64).reshape(-1, 5)
    diagnostics: Dict[int, Dict[str, Any]] = {}
    stats: Dict[str, Any] = {
        "radar_point_count": int(len(points)),
        "eligible_box_count": 0,
        "confirmed_box_count": 0,
    }
    if len(boxes) == 0:
        return boxes, diagnostics, stats

    quantile = min(0.45, max(0.0, float(quantile)))
    blend = min(1.0, max(0.0, float(blend)))
    min_points = max(1, int(min_points))
    min_unique_sensors = max(1, int(min_unique_sensors))
    min_extent_scale = max(0.1, float(min_extent_scale))
    max_extent_scale = max(min_extent_scale, float(max_extent_scale))
    eligible = []
    for index, box in enumerate(boxes):
        if box_is_eligible_for_radar_validation(
            box,
            score_threshold=score_threshold,
            max_distance_m=max_distance_m,
            lateral_limit_m=lateral_limit_m,
            max_heading_error_deg=max_heading_error_deg,
        ):
            eligible.append(index)
    stats["eligible_box_count"] = len(eligible)
    if not eligible or len(points) == 0:
        return boxes, diagnostics, stats

    # Only forward returns can confirm a forward obstacle.  This also rejects
    # occasional side/rear multipath samples before box association.
    points = points[points[:, 0] > 0.0]
    assigned: Dict[int, list[int]] = {index: [] for index in eligible}
    for point_index, point in enumerate(points):
        best = None
        for box_index in eligible:
            box = boxes[box_index]
            local = _box_local_points(point[None, :2], box)[0]
            half_len = max(0.05, float(box[2]))
            half_wid = max(0.05, float(box[3]))
            gate_len = half_len + association_margin_m
            gate_wid = half_wid + association_margin_m
            if abs(local[0]) > gate_len or abs(local[1]) > gate_wid:
                continue
            normalized_distance = (
                (local[0] / gate_len) ** 2 + (local[1] / gate_wid) ** 2
            )
            if best is None or normalized_distance < best[0]:
                best = (normalized_distance, box_index)
        if best is not None:
            assigned[best[1]].append(point_index)

    for box_index in eligible:
        indices = assigned[box_index]
        if len(indices) < min_points:
            continue
        samples = points[indices]
        sensor_count = int(len(np.unique(samples[:, 4].astype(np.int64))))
        if sensor_count < min_unique_sensors:
            continue

        box = boxes[box_index].copy()
        local = _box_local_points(samples[:, :2], box)
        original_half_len = max(0.05, float(box[2]))
        original_half_wid = max(0.05, float(box[3]))
        original_length = 2.0 * original_half_len
        original_width = 2.0 * original_half_wid
        original_min_x, original_max_x = -original_half_len, original_half_len
        original_min_y, original_max_y = -original_half_wid, original_half_wid

        low_x, high_x = np.quantile(local[:, 0], [quantile, 1.0 - quantile])
        low_y, high_y = np.quantile(local[:, 1], [quantile, 1.0 - quantile])
        # Locate the ego in box-local axes to identify the radar-visible face.
        ego_local = _box_local_points(np.zeros((1, 2), dtype=np.float64), box)[0]
        if ego_local[0] <= 0.0:
            radar_near = float(low_x) - longitudinal_padding_m
            fused_min_x = (1.0 - blend) * original_min_x + blend * radar_near
            fused_max_x = original_max_x
            near_error = radar_near - original_min_x
            anchor_positive = True
        else:
            radar_near = float(high_x) + longitudinal_padding_m
            fused_min_x = original_min_x
            fused_max_x = (1.0 - blend) * original_max_x + blend * radar_near
            near_error = radar_near - original_max_x
            anchor_positive = False

        target_length = fused_max_x - fused_min_x
        bounded_length = float(
            np.clip(
                target_length,
                original_length * min_extent_scale,
                original_length * max_extent_scale,
            )
        )
        if anchor_positive:
            fused_min_x = fused_max_x - bounded_length
        else:
            fused_max_x = fused_min_x + bounded_length

        radar_min_y = float(low_y) - lateral_padding_m
        radar_max_y = float(high_y) + lateral_padding_m
        fused_min_y = (1.0 - blend) * original_min_y + blend * radar_min_y
        fused_max_y = (1.0 - blend) * original_max_y + blend * radar_max_y
        target_width = fused_max_y - fused_min_y
        bounded_width = float(
            np.clip(
                target_width,
                original_width * min_extent_scale,
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
        if max_center_shift_m <= 0.0:
            local_center_delta[:] = 0.0
        elif shift_norm > max_center_shift_m:
            local_center_delta *= max_center_shift_m / shift_norm
        ego_delta = _local_delta_to_ego(local_center_delta, float(box[4]))
        boxes[box_index, :2] = box[:2] + ego_delta
        boxes[box_index, 2] = 0.5 * bounded_length
        boxes[box_index, 3] = 0.5 * bounded_width

        diagnostics[box_index] = {
            "radar_bbox_confirmed": True,
            "radar_point_count": int(len(samples)),
            "radar_sensor_count": sensor_count,
            "radar_original_extent": [original_length, original_width],
            "radar_refined_extent": [bounded_length, bounded_width],
            "radar_center_shift_m": float(np.linalg.norm(ego_delta)),
            "radar_near_edge_error_m": float(near_error),
        }

    stats["confirmed_box_count"] = len(diagnostics)
    return boxes, diagnostics, stats
