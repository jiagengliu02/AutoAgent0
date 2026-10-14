"""Task-target overlays for CARLA views."""
from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np

def _task_label(task_type: str) -> str:
    if task_type == "stop_at_target":
        return "STOP"
    if task_type == "park_at_target":
        return "PARK"
    return "GOAL"


def _task_color_bgr(task_type: str) -> tuple[int, int, int]:
    if task_type == "stop_at_target":
        return (0, 64, 255)
    if task_type == "park_at_target":
        return (255, 0, 255)
    return (0, 200, 255)


def _goal_distance_text(info: Dict[str, object]) -> Optional[str]:
    goal_status = info.get("task_goal_status")
    if not isinstance(goal_status, dict):
        return None
    try:
        position_error = float(goal_status.get("position_error_m"))
    except (TypeError, ValueError):
        return None
    text = f"{position_error:.1f}m"
    reached = bool(goal_status.get("reached"))
    if reached:
        text += " reached"
    return text


def _camera_matrix(intrinsic) -> np.ndarray:
    if intrinsic is None:
        return np.eye(4, dtype=np.float32)
    fx = intrinsic.get("fx")
    fy = intrinsic.get("fy")
    if fx is None:
        fx = intrinsic.get("focal_x")
    if fy is None:
        fy = intrinsic.get("focal_y")
    if fx is None:
        fovx = float(intrinsic["fovx"])
        width = float(intrinsic["W"])
        fx = width / (2.0 * np.tan(fovx / 2.0))
    if fy is None:
        fovy = float(intrinsic["fovy"])
        height = float(intrinsic["H"])
        fy = height / (2.0 * np.tan(fovy / 2.0))
    cx = float(intrinsic["cx"])
    cy = float(intrinsic["cy"])
    K = np.eye(4, dtype=np.float32)
    K[0, 0] = float(fx)
    K[1, 1] = float(fy)
    K[0, 2] = cx
    K[1, 2] = cy
    return K


def _project_world_points_to_image(points_world, intrinsic, c2w):
    points_world = np.asarray(points_world, dtype=np.float32)
    if len(points_world) == 0:
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0,), dtype=bool)

    K = _camera_matrix(intrinsic)
    homogeneous = np.concatenate(
        [points_world, np.ones((len(points_world), 1), dtype=np.float32)],
        axis=1,
    )
    camera_points = (np.linalg.inv(np.asarray(c2w, dtype=np.float32)) @ homogeneous.T).T[:, :3]
    depth = camera_points[:, 2]
    valid = depth > 1e-4
    pixels = np.zeros((len(points_world), 2), dtype=np.int32)
    if np.any(valid):
        image_points = (K[:3, :3] @ camera_points[valid].T).T
        pixels_valid = image_points[:, :2] / np.clip(image_points[:, 2:3], 1e-3, None)
        pixels[valid] = np.round(pixels_valid).astype(np.int32)
    return pixels, valid


def draw_task_target_overlay(
    image: np.ndarray,
    info: Dict[str, object],
    intrinsic,
    camera_c2w,
    *,
    draw_status_badge: bool = False,
) -> Dict[str, object]:
    task_type = str(info.get("task_type", "")).strip()
    target_world = info.get("task_target_world")
    label = _task_label(task_type)
    result = {
        "task_active": bool(info.get("task_active")) and bool(task_type),
        "task_type": task_type or None,
        "task_label": label,
        "projection_state": "inactive",
        "visible": False,
        "pixel": None,
        "instruction": info.get("task_instruction"),
    }
    if not result["task_active"]:
        return result

    marker_color = _task_color_bgr(task_type)
    distance_text = _goal_distance_text(info)
    h, w = image.shape[:2]

    if draw_status_badge:
        badge_text = label if not distance_text else f"{label} {distance_text}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.58
        thickness = 2
        text_w, text_h = cv2.getTextSize(badge_text, font, font_scale, thickness)[0]
        pad_x = 12
        pad_y = 10
        box_x1 = w - 18
        box_y0 = 18
        box_x0 = max(0, box_x1 - text_w - pad_x * 2)
        box_y1 = box_y0 + text_h + pad_y * 2
        overlay = image.copy()
        cv2.rectangle(overlay, (box_x0, box_y0), (box_x1, box_y1), (12, 12, 12), thickness=-1)
        cv2.addWeighted(overlay, 0.55, image, 0.45, 0, image)
        cv2.rectangle(image, (box_x0, box_y0), (box_x1, box_y1), marker_color, thickness=2)
        cv2.putText(
            image,
            badge_text,
            (box_x0 + pad_x, box_y1 - pad_y),
            font,
            font_scale,
            marker_color,
            thickness,
            cv2.LINE_AA,
        )

    if not isinstance(target_world, (list, tuple)) or len(target_world) < 3:
        result["projection_state"] = "missing_target"
        return result

    pixels, valid_mask = _project_world_points_to_image(
        np.asarray([target_world[:3]], dtype=np.float32),
        intrinsic,
        camera_c2w,
    )
    if len(pixels) == 0:
        result["projection_state"] = "missing_target"
        return result

    px, py = pixels[0]
    valid = bool(valid_mask[0]) if len(valid_mask) > 0 else False
    try:
        px_f = float(px)
        py_f = float(py)
    except (TypeError, ValueError):
        px_f = 0.0
        py_f = 0.0
    result["pixel"] = [round(px_f, 2), round(py_f, 2)]

    if valid and 0 <= px_f < w and 0 <= py_f < h:
        px_i = int(round(px_f))
        py_i = int(round(py_f))
        cv2.drawMarker(
            image,
            (px_i, py_i),
            marker_color,
            markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=26,
            thickness=2,
            line_type=cv2.LINE_AA,
        )
        cv2.circle(image, (px_i, py_i), 12, marker_color, 2, cv2.LINE_AA)
        cv2.circle(image, (px_i, py_i), 4, (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.62
        thickness = 2
        text_w, text_h = cv2.getTextSize(label, font, font_scale, thickness)[0]
        box_x0 = min(max(8, px_i + 10), max(8, w - text_w - 18))
        box_y0 = min(max(8, py_i - text_h - 18), max(8, h - text_h - 18))
        box_x1 = box_x0 + text_w + 10
        box_y1 = box_y0 + text_h + 10
        overlay = image.copy()
        cv2.rectangle(overlay, (box_x0, box_y0), (box_x1, box_y1), (12, 12, 12), thickness=-1)
        cv2.addWeighted(overlay, 0.55, image, 0.45, 0, image)
        cv2.rectangle(image, (box_x0, box_y0), (box_x1, box_y1), marker_color, thickness=2)
        cv2.putText(
            image,
            label,
            (box_x0 + 5, box_y1 - 6),
            font,
            font_scale,
            marker_color,
            thickness,
            cv2.LINE_AA,
        )
        result["projection_state"] = "visible"
        result["visible"] = True
        result["pixel"] = [px_i, py_i]
        return result

    if valid:
        result["projection_state"] = "offscreen"
        arrow = "LEFT" if px_f < 0 else "RIGHT" if px_f >= w else "UP" if py_f < 0 else "DOWN"
    else:
        result["projection_state"] = "behind_camera"
        arrow = "AHEAD"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.52
    thickness = 2
    offscreen_text = f"{label} {arrow}"
    if distance_text:
        offscreen_text = f"{offscreen_text} {distance_text}"
    text_w, text_h = cv2.getTextSize(offscreen_text, font, font_scale, thickness)[0]
    box_x0 = 18
    box_y0 = max(70, 18)
    box_x1 = min(w - 18, box_x0 + text_w + 18)
    box_y1 = min(h - 18, box_y0 + text_h + 16)
    overlay = image.copy()
    cv2.rectangle(overlay, (box_x0, box_y0), (box_x1, box_y1), (12, 12, 12), thickness=-1)
    cv2.addWeighted(overlay, 0.55, image, 0.45, 0, image)
    cv2.rectangle(image, (box_x0, box_y0), (box_x1, box_y1), marker_color, thickness=2)
    cv2.putText(
        image,
        offscreen_text,
        (box_x0 + 9, box_y1 - 6),
        font,
        font_scale,
        marker_color,
        thickness,
        cv2.LINE_AA,
    )
    return result
