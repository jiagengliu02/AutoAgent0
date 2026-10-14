"""Trajectory overlays for CARLA camera observations."""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

from agenticdriving.adapters.carla.task_overlay import draw_task_target_overlay


PLAN_VIS_FORWARD_OFFSET_M = 4.5
PLAN_RESAMPLE_SPACING_M = 0.08
VLM_CAMERA_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
)
PLANNER_GATE_LEARNED_COLOR_BGR = (64, 224, 64)
PLANNER_GATE_RULE_BASED_COLOR_BGR = (0, 165, 255)


def _fov2focal(fov: float, pixels: float) -> float:
    return pixels / (2.0 * math.tan(fov / 2.0))


def _get_camera_matrix(intrinsic: Dict[str, float]) -> np.ndarray:
    K = np.eye(4, dtype=np.float32)
    K[0, 0] = _fov2focal(intrinsic["fovx"], intrinsic["W"])
    K[1, 1] = _fov2focal(intrinsic["fovy"], intrinsic["H"])
    K[0, 2] = intrinsic["cx"]
    K[1, 2] = intrinsic["cy"]
    return K


def _rt2pose(r: Sequence[float], t: Sequence[float]) -> np.ndarray:
    from scipy.spatial.transform import Rotation as SCR

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = SCR.from_euler("XYZ", r, degrees=False).as_matrix().astype(np.float32)
    pose[:3, 3] = np.asarray(t, dtype=np.float32)
    return pose


def _get_camera_c2w(cam_params: Dict[str, Dict[str, np.ndarray]], ego_pose: np.ndarray, cam_name: str) -> np.ndarray:
    params = cam_params[cam_name]
    if "front2cam" in params:
        return ego_pose @ np.asarray(params["front2cam"], dtype=np.float32)

    v2front = np.asarray(cam_params["CAM_FRONT"]["v2c"], dtype=np.float32)
    v2c = np.asarray(params["v2c"], dtype=np.float32)
    c2front = v2front @ np.linalg.inv(v2c)
    return ego_pose @ c2front


def _local_plan_to_front_world(plan_traj: np.ndarray, front_c2w: np.ndarray, front_v2c: np.ndarray) -> np.ndarray:
    plan_traj = np.asarray(plan_traj, dtype=np.float32)
    if len(plan_traj) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    local_xyz = np.zeros((len(plan_traj) + 1, 3), dtype=np.float32)
    local_xyz[1:, :2] = plan_traj
    local_xyz[:, 1] += float(PLAN_VIS_FORWARD_OFFSET_M)

    camera_in_vehicle = np.linalg.inv(np.asarray(front_v2c, dtype=np.float32))[:3, 3]
    camera_in_local = np.array(
        [-camera_in_vehicle[1], camera_in_vehicle[0], camera_in_vehicle[2]],
        dtype=np.float32,
    )
    local_to_front_cam = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    points_cam = (local_to_front_cam @ (local_xyz - camera_in_local).T).T
    homogeneous = np.concatenate(
        [points_cam, np.ones((len(points_cam), 1), dtype=np.float32)],
        axis=1,
    )
    return (np.asarray(front_c2w, dtype=np.float32) @ homogeneous.T).T[:, :3].astype(np.float32)


def _resample_polyline(points_world: np.ndarray, spacing: float = PLAN_RESAMPLE_SPACING_M) -> np.ndarray:
    points_world = np.asarray(points_world, dtype=np.float32)
    if len(points_world) < 2:
        return points_world

    segment_lengths = np.linalg.norm(np.diff(points_world[:, [0, 2]], axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total_length = cumulative[-1]
    if total_length < 1e-4:
        return points_world[:1]

    sample_distances = np.arange(0.0, total_length + spacing, spacing, dtype=np.float32)
    sample_distances[-1] = min(sample_distances[-1], total_length)

    resampled = []
    seg_idx = 0
    for dist in sample_distances:
        while seg_idx + 1 < len(cumulative) and cumulative[seg_idx + 1] < dist:
            seg_idx += 1
        if seg_idx + 1 >= len(points_world):
            resampled.append(points_world[-1])
            continue
        seg_start = cumulative[seg_idx]
        seg_end = cumulative[seg_idx + 1]
        denom = max(seg_end - seg_start, 1e-6)
        alpha = float((dist - seg_start) / denom)
        point = (1.0 - alpha) * points_world[seg_idx] + alpha * points_world[seg_idx + 1]
        resampled.append(point)
    return np.asarray(resampled, dtype=np.float32)


def _draw_projected_polyline_camera_clipped(
    image: np.ndarray,
    points_world: np.ndarray,
    intrinsic: Dict[str, float],
    c2w: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int = 3,
    near: float = 1e-3,
) -> np.ndarray:
    points_world = np.asarray(points_world, dtype=np.float32)
    if len(points_world) < 2:
        return image

    K = _get_camera_matrix(intrinsic)[:3, :3]
    w2c = np.linalg.inv(np.asarray(c2w, dtype=np.float32))
    homogeneous = np.concatenate(
        [points_world, np.ones((len(points_world), 1), dtype=np.float32)],
        axis=1,
    )
    camera_points = (w2c @ homogeneous.T).T[:, :3]

    h, w = image.shape[:2]
    rect = (0, 0, w, h)

    def project_point(point_cam: np.ndarray) -> Tuple[int, int]:
        projected = K @ point_cam
        uv = projected[:2] / np.clip(projected[2], near, None)
        return tuple(np.round(uv).astype(np.int32))

    for p0, p1 in zip(camera_points[:-1], camera_points[1:]):
        z0, z1 = float(p0[2]), float(p1[2])
        if z0 <= near and z1 <= near:
            continue

        q0 = p0.copy()
        q1 = p1.copy()
        if z0 <= near:
            alpha = (near - z0) / max(z1 - z0, 1e-6)
            q0 = p0 + alpha * (p1 - p0)
            q0[2] = near
        if z1 <= near:
            alpha = (near - z1) / max(z0 - z1, 1e-6)
            q1 = p1 + alpha * (p0 - p1)
            q1[2] = near

        pixel0 = project_point(q0)
        pixel1 = project_point(q1)
        ok, clipped_p0, clipped_p1 = cv2.clipLine(rect, pixel0, pixel1)
        if not ok:
            continue
        cv2.line(image, clipped_p0, clipped_p1, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    return image


def _project_world_points_to_image(
    points_world: np.ndarray,
    intrinsic: Dict[str, float],
    c2w: np.ndarray,
) -> List[Tuple[int, int, bool]]:
    if len(points_world) == 0:
        return []

    K = _get_camera_matrix(intrinsic)
    homogeneous = np.concatenate(
        [np.asarray(points_world, dtype=np.float32), np.ones((len(points_world), 1), dtype=np.float32)],
        axis=1,
    )
    camera_points = (np.linalg.inv(c2w) @ homogeneous.T).T[:, :3]

    depth = camera_points[:, 2]
    valid = depth > 1e-4
    points: List[Tuple[int, int, bool]] = []
    for idx, point_cam in enumerate(camera_points):
        if not valid[idx]:
            points.append((0, 0, False))
            continue
        image_point = K[:3, :3] @ point_cam
        uv = image_point[:2] / np.clip(image_point[2], 1e-3, None)
        points.append((int(round(float(uv[0]))), int(round(float(uv[1]))), True))
    return points


def render_candidate_overlay(
    camera_image: np.ndarray,
    info: Dict[str, object],
    candidate_rows: Sequence[Dict[str, object]],
    cam_name: str = "CAM_FRONT",
    show_candidate_labels: bool = True,
    draw_plan: bool = True,
    draw_task_target: bool = True,
) -> np.ndarray:
    overlay = np.asarray(camera_image, dtype=np.uint8).copy()
    ego_pose = _rt2pose(info["ego_rot"], info["ego_pos"])
    cam_params = info["cam_params"]
    intrinsic = cam_params[cam_name]["intrinsic"]
    camera_c2w = _get_camera_c2w(cam_params, ego_pose, cam_name)
    camera_v2c = cam_params[cam_name]["v2c"]

    # ``draw_plan=False`` keeps only the task-target marker (drawn below) and
    # leaves the raw camera clean: the projected plan polyline is unreliable for
    # VLM geometric reading, so callers that only need the scene skip it.
    for row in candidate_rows if draw_plan else []:
        plan_local = np.asarray(row["local_plan"], dtype=np.float32)
        plan_world = _local_plan_to_front_world(plan_local, camera_c2w, camera_v2c)
        plan_world_dense = _resample_polyline(plan_world, spacing=PLAN_RESAMPLE_SPACING_M)
        color = tuple(int(v) for v in row["color_bgr"])
        _draw_projected_polyline_camera_clipped(
            overlay,
            plan_world_dense,
            intrinsic,
            camera_c2w,
            color=color,
            thickness=4 if row["candidate_rank"] == 0 else 2,
        )

        if show_candidate_labels and len(plan_world_dense) > 0:
            h, w = overlay.shape[:2]
            pixels = _project_world_points_to_image(plan_world_dense, intrinsic, camera_c2w)
            label_anchor = None
            for px, py, valid in reversed(pixels):
                if valid and 0 <= px < w and 0 <= py < h:
                    label_anchor = (int(px), int(py))
                    break
            if label_anchor is not None:
                cv2.putText(
                    overlay,
                    str(row["candidate_index"]),
                    (label_anchor[0] + 4, label_anchor[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )

    if draw_task_target:
        draw_task_target_overlay(
            overlay,
            info,
            intrinsic,
            camera_c2w,
            draw_status_badge=False,
        )
    return overlay






def render_candidate_overlays(
    camera_images: Dict[str, np.ndarray],
    info: Dict[str, object],
    candidate_rows: Sequence[Dict[str, object]],
    camera_order: Sequence[str] = VLM_CAMERA_ORDER,
    draw_plan: bool = True,
    draw_task_target: bool = True,
) -> Dict[str, np.ndarray]:
    overlays: Dict[str, np.ndarray] = {}
    for cam_name in camera_order:
        image = camera_images.get(cam_name)
        if image is None:
            continue
        if cam_name == "CAM_FRONT":
            overlays[cam_name] = render_candidate_overlay(
                image,
                info,
                candidate_rows,
                cam_name=cam_name,
                draw_plan=draw_plan,
                draw_task_target=draw_task_target,
            )
        else:
            overlays[cam_name] = np.asarray(image, dtype=np.uint8).copy()
    return overlays
