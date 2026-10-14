"""Geometry helpers for CARLA world and ego-local trajectories."""
from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np




def forward_left_basis(yaw: float) -> Tuple[np.ndarray, np.ndarray]:
    forward_dir = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float32)
    left_dir = np.array([-math.sin(yaw), math.cos(yaw)], dtype=np.float32)
    return forward_dir, left_dir


def world_delta_to_local_components(delta_world: np.ndarray, yaw: float) -> np.ndarray:
    forward_dir, left_dir = forward_left_basis(yaw)
    return np.array(
        [
            float(np.dot(delta_world, forward_dir)),
            float(np.dot(delta_world, left_dir)),
        ],
        dtype=np.float32,
    )


def timestamp_delta_seconds(
    prev_info: Dict[str, object],
    next_info: Dict[str, object],
    default_dt: float = 0.25,
) -> float:
    dt = float(next_info["timestamp"]) - float(prev_info["timestamp"])
    return dt if dt > 1e-6 else default_dt






def info_to_pose(info: Dict[str, object]) -> np.ndarray:
    from agenticdriving.planners.sparsedrivev2.carla_inputs import carla_transform_matrix

    return carla_transform_matrix(info["ego_pos"], info["ego_rot"]).astype(np.float32)


def local_plan_to_world(plan_traj: np.ndarray, ego_pose: np.ndarray) -> np.ndarray:
    plan_traj = np.asarray(plan_traj, dtype=np.float32)
    if len(plan_traj) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    origin = ego_pose[:3, 3]
    right_dir = ego_pose[:3, 1]
    forward_dir = ego_pose[:3, 0]
    return np.asarray(
        [origin + float(right) * right_dir + float(forward) * forward_dir
         for right, forward in plan_traj],
        dtype=np.float32,
    )


def world_points_to_current_local(points_world: np.ndarray, ego_pose: np.ndarray) -> np.ndarray:
    if len(points_world) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    homogeneous = np.concatenate(
        [np.asarray(points_world, dtype=np.float32),
         np.ones((len(points_world), 1), dtype=np.float32)],
        axis=1,
    )
    ego_points = (np.linalg.inv(ego_pose) @ homogeneous.T).T[:, :3]
    return np.stack([ego_points[:, 1], ego_points[:, 0]], axis=1).astype(np.float32)


def path_length(plan_traj: np.ndarray) -> float:
    plan_traj = np.asarray(plan_traj, dtype=np.float32)
    if len(plan_traj) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(plan_traj, axis=0), axis=1).sum())


def truncate_plan(plan_traj: np.ndarray, num_poses: int) -> np.ndarray:
    plan_traj = np.asarray(plan_traj, dtype=np.float32)
    if num_poses <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(plan_traj[: min(len(plan_traj), int(num_poses))], dtype=np.float32)




def normalize_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    if len(scores) == 0:
        return np.zeros((0,), dtype=np.float32)
    score_min = float(scores.min())
    score_max = float(scores.max())
    if score_max - score_min < 1e-6:
        return np.ones_like(scores, dtype=np.float32)
    return (scores - score_min) / (score_max - score_min)
