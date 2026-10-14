"""BEVFormer-versus-simulator box overlays for CARLA top-down recordings."""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np


BEVFORMER_BOX_COLOR_BGR = (0, 165, 255)  # orange
GROUND_TRUTH_BOX_COLOR_BGR = (64, 224, 64)  # green
PDMS_GROUND_TRUTH_BOX_COLOR_BGR = (0, 0, 255)  # red
# Explicit comparison palette for sensor-only debug recordings. Keep the
# historical BEVFormer/shadow palette above unchanged: GT is red and sensed
# perception is green in the comparison view.
PERCEPTION_COMPARISON_BOX_COLOR_BGR = (0, 220, 0)  # green
PERCEPTION_COMPARISON_GT_COLOR_BGR = (0, 0, 255)  # red
# BEV-semantic OBSTACLE-mask components (bridgedrive judge statics feed)
MASK_STATIC_BOX_COLOR_BGR = (255, 0, 255)  # magenta
SPARSEDRIVE_TARGET_POINT_COLOR_BGR = (255, 255, 0)  # cyan
SPARSEDRIVE_REENTRY_TARGET_POINT_COLOR_BGR = (0, 165, 255)  # orange
_BOX_LABEL_COLOR_BGR = (245, 245, 245)


def _agent_box_in_ego_local(
    agent: Dict[str, Any],
    info: Dict[str, Any],
    *,
    target_timestamp: Optional[float] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray, float, float]]:
    """Return center, heading direction, half-length and half-width.

    The returned center/direction use the planner/top-down convention
    ``[right, forward]``. Positions are advanced with the recorded world-frame
    velocity when a cached shadow snapshot is reused on control-only ticks.
    """
    try:
        pos = np.asarray(agent["agent_pos_world"], dtype=np.float64).reshape(-1)
        ego_pos = np.asarray(info["ego_pos"], dtype=np.float64).reshape(-1)
        ego_rot = np.asarray(info["ego_rot"], dtype=np.float64).reshape(-1)
        if len(pos) < 2 or len(ego_pos) < 2 or len(ego_rot) < 3:
            return None

        pos_xy = pos[:2].copy()
        if target_timestamp is not None:
            source_timestamp = float(agent.get("timestamp", target_timestamp))
            dt = max(0.0, min(1.0, float(target_timestamp) - source_timestamp))
            velocity = np.asarray(
                agent.get("agent_vel_vec") or (0.0, 0.0), dtype=np.float64
            ).reshape(-1)
            if len(velocity) >= 2:
                pos_xy += velocity[:2] * dt

        ego_yaw = float(ego_rot[2])
        cos_yaw, sin_yaw = math.cos(ego_yaw), math.sin(ego_yaw)
        forward_world = np.array([cos_yaw, sin_yaw], dtype=np.float64)
        right_world = np.array([-sin_yaw, cos_yaw], dtype=np.float64)
        delta = pos_xy - ego_pos[:2]
        center = np.array(
            [np.dot(delta, right_world), np.dot(delta, forward_world)],
            dtype=np.float64,
        )

        heading = float(agent.get("agent_heading", 0.0))
        heading_world = np.array(
            [math.cos(heading), math.sin(heading)], dtype=np.float64
        )
        direction = np.array(
            [np.dot(heading_world, right_world), np.dot(heading_world, forward_world)],
            dtype=np.float64,
        )
        norm = float(np.linalg.norm(direction))
        direction = direction / norm if norm > 1e-6 else np.array([0.0, 1.0])

        extent = np.asarray(agent.get("agent_extent") or (1.0, 1.0), dtype=np.float64)
        if len(extent) < 2:
            return None
        half_length = max(0.1, float(extent[0]) * 0.5)
        half_width = max(0.1, float(extent[1]) * 0.5)
    except (KeyError, TypeError, ValueError):
        return None

    if not np.isfinite(center).all() or not np.isfinite(direction).all():
        return None
    if not np.isfinite([half_length, half_width]).all():
        return None
    return center, direction, half_length, half_width


def _local_box_corners(
    center: np.ndarray,
    direction: np.ndarray,
    half_length: float,
    half_width: float,
) -> np.ndarray:
    right = np.array([direction[1], -direction[0]], dtype=np.float64)
    forward_offset = direction * float(half_length)
    right_offset = right * float(half_width)
    return np.asarray(
        [
            center + forward_offset + right_offset,
            center + forward_offset - right_offset,
            center - forward_offset - right_offset,
            center - forward_offset + right_offset,
        ],
        dtype=np.float64,
    )


def _local_to_pixels(
    points: np.ndarray,
    *,
    center_xy: Tuple[int, int],
    pixels_per_meter: float,
) -> np.ndarray:
    cx, cy = center_xy
    points = np.asarray(points, dtype=np.float64)
    pixels = np.column_stack(
        [
            cx + points[:, 0] * pixels_per_meter,
            cy - points[:, 1] * pixels_per_meter,
        ]
    )
    return np.round(pixels).astype(np.int32)


def _world_points_to_ego_local(
    points_world: np.ndarray,
    info: Dict[str, Any],
) -> Optional[np.ndarray]:
    """Map CARLA world XYZ/XY points to top-down ``[right, forward]``."""
    try:
        points = np.asarray(points_world, dtype=np.float64)
        ego_pos = np.asarray(info["ego_pos"], dtype=np.float64).reshape(-1)
        ego_rot = np.asarray(info["ego_rot"], dtype=np.float64).reshape(-1)
        if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 2:
            return None
        if len(ego_pos) < 2 or len(ego_rot) < 3:
            return None
        ego_yaw = float(ego_rot[2])
        cos_yaw, sin_yaw = math.cos(ego_yaw), math.sin(ego_yaw)
        forward_world = np.array([cos_yaw, sin_yaw], dtype=np.float64)
        right_world = np.array([-sin_yaw, cos_yaw], dtype=np.float64)
        delta = points[:, :2] - ego_pos[None, :2]
        local = np.column_stack(
            [delta @ right_world, delta @ forward_world]
        )
    except (KeyError, TypeError, ValueError):
        return None
    return local if np.isfinite(local).all() else None


def _exact_ground_truth_footprint(
    agent: Dict[str, Any], info: Dict[str, Any]
) -> Optional[np.ndarray]:
    """Return the convex XY footprint of CARLA's actual world-space bbox.

    ``bbox_world_corners`` comes directly from
    ``actor.bounding_box.get_world_vertices(actor.get_transform())``.  It includes
    the bounding box's local location and rotation. The compact PDMS OBB should
    describe the same footprint using center, heading and extents.
    """
    corners = agent.get("bbox_world_corners")
    if corners is None:
        return None
    local = _world_points_to_ego_local(corners, info)
    if local is None or len(local) < 4:
        return None
    return local


def _agent_label(agent: Dict[str, Any]) -> str:
    object_type = str(agent.get("object_type", "object"))
    actor_type_id = str(agent.get("actor_type_id") or object_type)
    agent_id = str(agent.get("agent_id", "?"))
    yaw_offset = agent.get("bbox_yaw_offset_deg")
    if yaw_offset is None:
        return f"{actor_type_id}({agent_id})"
    return f"{actor_type_id}({agent_id}) dyaw={float(yaw_offset):+.0f}"


def _draw_corner_brackets(
    image_bgr: np.ndarray,
    corner_pixels: np.ndarray,
    *,
    color_bgr: Tuple[int, int, int],
    thickness: int = 2,
    fraction: float = 0.28,
) -> None:
    """Draw a box as four corner L-brackets instead of a closed rectangle.

    Used for reference layers in the decluttered ``judge`` overlay: brackets
    stay legible when a full detection rectangle is drawn exactly on top —
    a matched object reads as "orange box with red corner ticks", a miss as
    naked brackets, a phantom as a box without brackets."""
    pts = corner_pixels.reshape(-1, 2)
    n = len(pts)
    for i in range(n):
        prev_vec = pts[(i - 1) % n] - pts[i]
        next_vec = pts[(i + 1) % n] - pts[i]
        for vec in (prev_vec, next_vec):
            tip = (pts[i] + vec * fraction).astype(np.int32)
            cv2.line(
                image_bgr,
                tuple(pts[i].astype(np.int32)),
                tuple(tip),
                color_bgr,
                thickness,
                cv2.LINE_AA,
            )


def render_ground_truth_boxes_on_topdown(
    image_rgb: np.ndarray,
    info: Dict[str, Any],
    *,
    ground_truth_agents: Sequence[Dict[str, Any]],
    pixels_per_meter: float,
    ego_center_xy: Optional[Tuple[int, int]] = None,
    show_pdms_boxes: bool = True,
    style: str = "full",
) -> np.ndarray:
    """Draw every simulator actor box used by PDMS on the BEV recording.

    style="full" (default): green exact CARLA footprint + red compact PDMS
    OBB + per-agent labels — the transform-regression tripwire (green/red
    should coincide). style="judge": the decluttered perception-review
    layer — ONE GT reference only, drawn as red corner brackets so sensed
    detection rectangles drawn on top stay legible; no green, no labels.
    """
    image_bgr = cv2.cvtColor(np.asarray(image_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    height, width = image_bgr.shape[:2]
    center_xy = ego_center_xy or (width // 2, height // 2)
    target_timestamp = float(info.get("timestamp", 0.0) or 0.0)
    max_visible_m = math.hypot(width, height) / max(2.0 * pixels_per_meter, 1e-6)

    judge_style = str(style).lower() == "judge"
    for agent in ground_truth_agents:
        geometry = _agent_box_in_ego_local(
            agent, info, target_timestamp=target_timestamp
        )
        exact_footprint = _exact_ground_truth_footprint(agent, info)
        if geometry is None and exact_footprint is None:
            continue

        if exact_footprint is not None and not judge_style:
            exact_center = np.mean(exact_footprint, axis=0)
            if float(np.linalg.norm(exact_center)) <= max_visible_m + 10.0:
                exact_pixels = _local_to_pixels(
                    exact_footprint,
                    center_xy=center_xy,
                    pixels_per_meter=pixels_per_meter,
                )
                hull = cv2.convexHull(exact_pixels.reshape(-1, 1, 2))
                cv2.polylines(
                    image_bgr,
                    [hull],
                    isClosed=True,
                    color=GROUND_TRUTH_BOX_COLOR_BGR,
                    thickness=3,
                    lineType=cv2.LINE_AA,
                )
                label_anchor = tuple(
                    np.min(hull.reshape(-1, 2), axis=0)
                    + np.array([2, -4], dtype=np.int32)
                )
                label = _agent_label(agent)
                cv2.putText(
                    image_bgr, label, label_anchor, cv2.FONT_HERSHEY_SIMPLEX,
                    0.36, (0, 0, 0), 3, cv2.LINE_AA,
                )
                cv2.putText(
                    image_bgr, label, label_anchor, cv2.FONT_HERSHEY_SIMPLEX,
                    0.36, GROUND_TRUTH_BOX_COLOR_BGR, 1, cv2.LINE_AA,
                )

        if show_pdms_boxes and geometry is not None:
            pdms_center, direction, half_length, half_width = geometry
            if float(np.linalg.norm(pdms_center)) <= max_visible_m + half_length + half_width:
                pdms_pixels = _local_to_pixels(
                    _local_box_corners(
                        pdms_center, direction, half_length, half_width
                    ),
                    center_xy=center_xy,
                    pixels_per_meter=pixels_per_meter,
                )
                if judge_style:
                    _draw_corner_brackets(
                        image_bgr,
                        pdms_pixels,
                        color_bgr=PDMS_GROUND_TRUTH_BOX_COLOR_BGR,
                        thickness=2,
                    )
                else:
                    cv2.polylines(
                        image_bgr,
                        [pdms_pixels],
                        isClosed=True,
                        color=PDMS_GROUND_TRUTH_BOX_COLOR_BGR,
                        thickness=2,
                        lineType=cv2.LINE_AA,
                    )

    legend = (
        "GT=red brackets  detection=orange  mask=magenta"
        if judge_style
        else "GT bbox=green  PDMS bbox=red"
    )
    (legend_w, _), _ = cv2.getTextSize(legend, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.putText(
        image_bgr,
        legend,
        (max(8, width - legend_w - 12), height - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        _BOX_LABEL_COLOR_BGR,
        1,
        cv2.LINE_AA,
    )
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def render_sparsedrive_target_point_on_topdown(
    image_rgb: np.ndarray,
    target_point: Optional[Sequence[float]],
    *,
    pixels_per_meter: float,
    ego_center_xy: Optional[Tuple[int, int]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Draw SparseDriveV2's actual ``tp_near`` and route-conditioning state."""
    image_bgr = cv2.cvtColor(np.asarray(image_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    if target_point is None:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    try:
        point = np.asarray(target_point, dtype=np.float64).reshape(-1)
        if len(point) < 2 or not np.isfinite(point[:2]).all():
            return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    except (TypeError, ValueError):
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    height, width = image_bgr.shape[:2]
    center_xy = ego_center_xy or (width // 2, height // 2)
    metadata = dict(metadata or {})
    reentry_active = bool(metadata.get("route_reentry_active", False))
    target_color = (
        SPARSEDRIVE_REENTRY_TARGET_POINT_COLOR_BGR
        if reentry_active
        else SPARSEDRIVE_TARGET_POINT_COLOR_BGR
    )
    pixel = _local_to_pixels(
        point[:2][None, :],
        center_xy=center_xy,
        pixels_per_meter=pixels_per_meter,
    )[0]
    endpoint = int(pixel[0]), int(pixel[1])
    cv2.arrowedLine(
        image_bgr,
        center_xy,
        endpoint,
        target_color,
        2,
        cv2.LINE_AA,
        tipLength=0.12,
    )
    marker = np.asarray(
        [
            [endpoint[0], endpoint[1] - 8],
            [endpoint[0] + 8, endpoint[1]],
            [endpoint[0], endpoint[1] + 8],
            [endpoint[0] - 8, endpoint[1]],
        ],
        dtype=np.int32,
    )
    cv2.polylines(image_bgr, [marker], True, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.polylines(
        image_bgr,
        [marker],
        True,
        target_color,
        2,
        cv2.LINE_AA,
    )
    state = "REENTRY" if reentry_active else "NORMAL"
    labels = [f"SDV2 TP [{state}] r={point[0]:+.1f}m f={point[1]:.1f}m"]
    command = metadata.get("road_option")
    command_label = {
        1: "LEFT",
        2: "RIGHT",
        3: "STRAIGHT",
        4: "LANEFOLLOW",
        5: "CHANGE_LEFT",
        6: "CHANGE_RIGHT",
    }.get(command, str(command) if command is not None else "?")
    command_details = [f"cmd={command_label}"]
    route_distance = metadata.get("route_distance_m")
    if route_distance is not None:
        try:
            command_details.append(f"route={float(route_distance):.2f}m")
        except (TypeError, ValueError):
            pass
    labels.append("  ".join(command_details))

    route_details = []
    lookahead = metadata.get("route_target_lookahead_m")
    if lookahead is not None:
        try:
            route_details.append(f"lookahead={float(lookahead):.1f}m")
        except (TypeError, ValueError):
            pass
    primitive = metadata.get("route_reentry_primitive")
    if primitive:
        route_details.append(f"primitive={primitive}")
    reentry_side = metadata.get("route_reentry_side")
    if reentry_side:
        route_details.append(f"side={reentry_side}")
    trigger_reason = metadata.get("route_reentry_trigger_reason")
    if trigger_reason:
        route_details.append(f"trigger={trigger_reason}")
    lane_relation = metadata.get("route_lane_relation")
    if lane_relation:
        route_details.append(f"lane={lane_relation}")
    if route_details:
        labels.append("  ".join(route_details))

    lane_keys = []
    route_lane_key = metadata.get("route_lane_key")
    if route_lane_key is not None:
        lane_keys.append(f"route_lane={route_lane_key}")
    ego_lane_key = metadata.get("ego_lane_key")
    if ego_lane_key is not None:
        lane_keys.append(f"ego_lane={ego_lane_key}")
    if lane_keys:
        labels.append("  ".join(lane_keys))

    max_label_width = max(
        cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)[0][0]
        for label in labels
    )
    label_x = max(8, min(width - max_label_width - 8, endpoint[0] + 10))
    label_y = endpoint[1] - (18 * len(labels) + 12)
    if label_y < 18:
        label_y = min(
            height - 18 * len(labels),
            endpoint[1] + 28,
        )
    for line_index, label in enumerate(labels):
        label_xy = (label_x, label_y + 18 * line_index)
        cv2.putText(
            image_bgr, label, label_xy, cv2.FONT_HERSHEY_SIMPLEX,
            0.46, (0, 0, 0), 3, cv2.LINE_AA,
        )
        cv2.putText(
            image_bgr, label, label_xy, cv2.FONT_HERSHEY_SIMPLEX,
            0.46, target_color, 1, cv2.LINE_AA,
        )
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _draw_agent_boxes(
    image_bgr: np.ndarray,
    agents: Sequence[Dict[str, Any]],
    info: Dict[str, Any],
    *,
    color_bgr: Tuple[int, int, int],
    thickness: int,
    pixels_per_meter: float,
    center_xy: Tuple[int, int],
    target_timestamp: float,
    show_scores: bool,
) -> int:
    height, width = image_bgr.shape[:2]
    max_visible_m = math.hypot(width, height) / max(2.0 * pixels_per_meter, 1e-6)
    drawn = 0
    for agent in agents:
        geometry = _agent_box_in_ego_local(
            agent, info, target_timestamp=target_timestamp
        )
        if geometry is None:
            continue
        center, direction, half_length, half_width = geometry
        if float(np.linalg.norm(center)) > max_visible_m + half_length + half_width:
            continue

        corners = _local_box_corners(
            center, direction, half_length, half_width
        )
        corner_pixels = _local_to_pixels(
            corners, center_xy=center_xy, pixels_per_meter=pixels_per_meter
        )
        cv2.polylines(
            image_bgr,
            [corner_pixels],
            isClosed=True,
            color=color_bgr,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

        heading_pixels = _local_to_pixels(
            np.asarray([center, center + direction * half_length]),
            center_xy=center_xy,
            pixels_per_meter=pixels_per_meter,
        )
        cv2.line(
            image_bgr,
            tuple(heading_pixels[0]),
            tuple(heading_pixels[1]),
            color_bgr,
            thickness,
            cv2.LINE_AA,
        )

        if show_scores and agent.get("detection_score") is not None:
            anchor = tuple(corner_pixels[0] + np.array([3, -3], dtype=np.int32))
            label = f"{float(agent['detection_score']):.2f}"
            cv2.putText(
                image_bgr, label, anchor, cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (0, 0, 0), 3, cv2.LINE_AA,
            )
            cv2.putText(
                image_bgr, label, anchor, cv2.FONT_HERSHEY_SIMPLEX,
                0.38, color_bgr, 1, cv2.LINE_AA,
            )
        drawn += 1
    return drawn


def _draw_legend(
    image_bgr: np.ndarray,
    *,
    detected_count: int,
    ground_truth_count: int,
    shadow_age_sec: float,
    detected_color_bgr: Tuple[int, int, int] = BEVFORMER_BOX_COLOR_BGR,
    ground_truth_color_bgr: Tuple[int, int, int] = GROUND_TRUTH_BOX_COLOR_BGR,
    detected_label: str = "detections",
    ground_truth_label: str = "GT boxes",
) -> None:
    height, width = image_bgr.shape[:2]
    panel_width, panel_height = 330, 78
    x0 = max(8, width - panel_width - 12)
    y0 = 12
    x1 = min(width - 8, x0 + panel_width)
    y1 = min(height - 8, y0 + panel_height)
    shade = image_bgr.copy()
    cv2.rectangle(shade, (x0, y0), (x1, y1), (8, 8, 8), thickness=-1)
    cv2.addWeighted(shade, 0.62, image_bgr, 0.38, 0, image_bgr)

    rows = (
        (ground_truth_color_bgr, f"{ground_truth_label}: {ground_truth_count}"),
        (detected_color_bgr, f"{detected_label}: {detected_count}"),
    )
    for index, (color, label) in enumerate(rows):
        y = y0 + 21 + index * 23
        cv2.rectangle(image_bgr, (x0 + 10, y - 9), (x0 + 30, y + 5), color, 2)
        cv2.putText(
            image_bgr, label, (x0 + 40, y + 3), cv2.FONT_HERSHEY_SIMPLEX,
            0.46, _BOX_LABEL_COLOR_BGR, 1, cv2.LINE_AA,
        )
    cv2.putText(
        image_bgr,
        f"shadow age: {max(0.0, shadow_age_sec):.2f}s",
        (x0 + 10, y0 + 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (190, 190, 190),
        1,
        cv2.LINE_AA,
    )


def render_perception_comparison_on_topdown(
    image_rgb: np.ndarray,
    info: Dict[str, Any],
    *,
    detected_agents: Sequence[Dict[str, Any]],
    ground_truth_agents: Sequence[Dict[str, Any]],
    pixels_per_meter: float,
    shadow_timestamp: Optional[float] = None,
    ego_center_xy: Optional[Tuple[int, int]] = None,
    draw_ground_truth: bool = True,
    mask_static_agents: Sequence[Dict[str, Any]] = (),
    detected_color_bgr: Tuple[int, int, int] = BEVFORMER_BOX_COLOR_BGR,
    ground_truth_color_bgr: Tuple[int, int, int] = GROUND_TRUTH_BOX_COLOR_BGR,
    detected_label: str = "detections",
    ground_truth_label: str = "GT boxes",
) -> np.ndarray:
    """Overlay oriented GT and sensed boxes on an ego-aligned BEV image.

    Defaults preserve the historical BEVFormer shadow colors. Recording-only
    comparison mode passes an explicit palette with GT red and perception
    green.
    """
    image_bgr = cv2.cvtColor(np.asarray(image_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    height, width = image_bgr.shape[:2]
    center_xy = ego_center_xy or (width // 2, height // 2)
    target_timestamp = float(info.get("timestamp", shadow_timestamp or 0.0))
    shadow_timestamp = (
        target_timestamp if shadow_timestamp is None else float(shadow_timestamp)
    )

    # Draw the wider GT stroke first and the narrower prediction on top. A
    # close match therefore shows both outlines instead of hiding one layer.
    gt_count = len(ground_truth_agents)
    if draw_ground_truth:
        gt_count = _draw_agent_boxes(
            image_bgr,
            ground_truth_agents,
            info,
            color_bgr=ground_truth_color_bgr,
            thickness=4,
            pixels_per_meter=pixels_per_meter,
            center_xy=center_xy,
            target_timestamp=target_timestamp,
            show_scores=False,
        )
    detected_count = _draw_agent_boxes(
        image_bgr,
        detected_agents,
        info,
        color_bgr=detected_color_bgr,
        thickness=2,
        pixels_per_meter=pixels_per_meter,
        center_xy=center_xy,
        target_timestamp=target_timestamp,
        show_scores=True,
    )
    if mask_static_agents:
        # Semantic-mask component boxes (bdocc_*): the judge's unboxed-statics
        # feed, drawn thin so overlap with a CenterNet box stays readable.
        _draw_agent_boxes(
            image_bgr,
            mask_static_agents,
            info,
            color_bgr=MASK_STATIC_BOX_COLOR_BGR,
            thickness=1,
            pixels_per_meter=pixels_per_meter,
            center_xy=center_xy,
            target_timestamp=target_timestamp,
            show_scores=False,
        )
    _draw_legend(
        image_bgr,
        detected_count=detected_count,
        ground_truth_count=gt_count,
        shadow_age_sec=target_timestamp - shadow_timestamp,
        detected_color_bgr=detected_color_bgr,
        ground_truth_color_bgr=ground_truth_color_bgr,
        detected_label=detected_label,
        ground_truth_label=ground_truth_label,
    )
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
