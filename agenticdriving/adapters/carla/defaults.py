"""Default trajectories in CARLA ego-local coordinates."""
from __future__ import annotations

import numpy as np


def get_default_trajectories(num_poses: int) -> np.ndarray:
    num_poses = max(2, int(num_poses))
    t = np.linspace(0.0, 1.0, num_poses, dtype=np.float32)
    forward = np.stack([np.zeros_like(t), 40.0 * t], axis=1)
    slight_left = np.stack([-5.0 * (t ** 2), 38.0 * t], axis=1)
    slight_right = np.stack([5.0 * (t ** 2), 38.0 * t], axis=1)
    sharp_left = np.stack([-25.0 * (t ** 3), 30.0 * t], axis=1)
    sharp_right = np.stack([25.0 * (t ** 3), 30.0 * t], axis=1)
    return np.stack([forward, slight_left, slight_right, sharp_left, sharp_right], axis=0).astype(np.float32)
