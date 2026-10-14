"""Materialize one VLM-selected local recovery waypoint.

The recovery planner uses the CARLA ego-local convention ``[x_right, y_forward]``.
This module turns a bounded waypoint into a short smooth path plus the signed
``[speed, steer_rad]`` states expected by the existing commit/hold controller.
The resulting path is still checked by the normal recovery verifier before it
can be executed.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from agenticdriving.planners.fallback_rule_based.kinematics import (
    DEFAULT_WHEELBASE_M,
    MAX_STEER_RAD,
)
from agenticdriving.planners.fallback_rule_based.provider import (
    RECOVERY_PLAN_DT_SEC,
)


RECOVERY_WAYPOINT_FORWARD_RANGE_M = (1.0, 6.0)
RECOVERY_WAYPOINT_RIGHT_RANGE_M = (-3.0, 3.0)
RECOVERY_WAYPOINT_TARGET_SPEED_MPS = 1.25


def materialize_recovery_waypoint(
    *,
    forward_m: float,
    right_m: float,
    output_num_poses: int,
    target_speed_mps: float = RECOVERY_WAYPOINT_TARGET_SPEED_MPS,
    forward_range_m: Tuple[float, float] = RECOVERY_WAYPOINT_FORWARD_RANGE_M,
    right_range_m: Tuple[float, float] = RECOVERY_WAYPOINT_RIGHT_RANGE_M,
) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """Return ``(trajectory, states, execute_seconds)`` for a local waypoint.

    The lateral component follows a cubic smoothstep so the path starts and
    finishes aligned with the ego heading. Only forward waypoints are accepted:
    with a front-only image, reverse free-space should be expressed through one
    of the verified reverse primitives instead of an unobserved waypoint.
    """
    try:
        forward = float(forward_m)
        right = float(right_m)
        target_speed = float(target_speed_mps)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (forward, right, target_speed)):
        return None
    try:
        forward_lo, forward_hi = map(float, forward_range_m)
        right_lo, right_hi = map(float, right_range_m)
    except (TypeError, ValueError):
        return None
    if not all(
        math.isfinite(value)
        for value in (forward_lo, forward_hi, right_lo, right_hi)
    ):
        return None
    if forward_lo > forward_hi or right_lo > right_hi:
        return None
    if not forward_lo <= forward <= forward_hi:
        return None
    if not right_lo <= right <= right_hi or target_speed <= 0.0:
        return None

    dt = float(RECOVERY_PLAN_DT_SEC)
    path_length_estimate = math.hypot(forward, right)
    active_steps = max(
        4,
        int(math.ceil(path_length_estimate / max(target_speed * dt, 1e-3))),
    )
    num_poses = max(2, int(output_num_poses), active_steps + 1)

    tau = np.linspace(
        1.0 / active_steps,
        1.0,
        num=active_steps,
        dtype=np.float64,
    )
    lateral_blend = 3.0 * tau**2 - 2.0 * tau**3
    active_points = np.column_stack(
        [right * lateral_blend, forward * tau]
    )
    trajectory = np.repeat(active_points[-1][None, :], num_poses, axis=0)
    trajectory[:active_steps] = active_points

    segments = np.diff(
        np.vstack([np.zeros((1, 2), dtype=np.float64), active_points]),
        axis=0,
    )
    segment_lengths = np.linalg.norm(segments, axis=1)
    if np.any(segment_lengths <= 1e-6):
        return None
    headings = np.arctan2(segments[:, 0], segments[:, 1])
    previous_headings = np.concatenate([[0.0], headings[:-1]])
    heading_delta = np.arctan2(
        np.sin(headings - previous_headings),
        np.cos(headings - previous_headings),
    )
    curvature = heading_delta / segment_lengths
    raw_steer = np.arctan(DEFAULT_WHEELBASE_M * curvature)
    if np.max(np.abs(raw_steer)) > MAX_STEER_RAD + 1e-6:
        return None

    states = np.zeros((num_poses, 2), dtype=np.float64)
    states[:active_steps, 0] = np.minimum(segment_lengths / dt, 2.0)
    states[:active_steps, 1] = raw_steer
    execute_seconds = active_steps * dt
    return (
        np.asarray(trajectory, dtype=np.float32),
        np.asarray(states, dtype=np.float32),
        float(execute_seconds),
    )


__all__ = [
    "RECOVERY_WAYPOINT_FORWARD_RANGE_M",
    "RECOVERY_WAYPOINT_RIGHT_RANGE_M",
    "RECOVERY_WAYPOINT_TARGET_SPEED_MPS",
    "materialize_recovery_waypoint",
]
