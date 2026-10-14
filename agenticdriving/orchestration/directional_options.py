"""HD-map based filtering for directionally impossible planning options."""
from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np


DEFAULT_JUNCTION_ROUTE_MAX_ANGLE_DEG = 60.0
_MIN_HEADING_DISPLACEMENT_M = 0.5


LEFTWARD_PRIMITIVES = frozenset(
    {
        "swerve_left",
        "back_left",
        "back_bypass_left",
        "lane_left",
        "lane_bypass_left",
    }
)


def adjacent_driving_lane_absent(info: Dict[str, Any], direction: object) -> bool:
    """Return whether the HD map explicitly reports no lane on ``direction``.

    Missing or unknown descriptors are not treated as proof that a lane is
    absent.  This distinction lets transient map lookup failures keep the
    existing fail-closed execution/wait behavior while ensuring a definitive
    ``flow: none`` can never expose an impossible lane-bypass function to
    the VLM.
    """
    normalized_direction = (
        "left"
        if str(direction or "").strip().lower().startswith("l")
        else "right"
    )
    descriptors = info.get("adjacent_lane_flows")
    if not isinstance(descriptors, dict):
        return False
    descriptor = descriptors.get(normalized_direction)
    if isinstance(descriptor, dict):
        flow = descriptor.get("flow")
    else:
        flow = descriptor
    return str(flow or "unknown").strip().lower().replace("-", "_") in {
        "none",
        "no_lane",
        "absent",
    }


def suppress_leftward_options(info: Dict[str, Any]) -> bool:
    """Return whether the route lane has opposing-left/same-flow-right topology.

    The descriptors are anchored to the Leaderboard route lane by the CARLA
    adapter.  In the four-lane layout discussed in RouteScenario 11 this is
    exactly the lane-3 situation: lane 2 is opposing on the left and lane 4 is
    same-direction on the right.  Recovery policy deliberately removes the
    left-side family and keeps the same-direction right-side options.
    """
    descriptors = info.get("adjacent_lane_flows")
    if not isinstance(descriptors, dict):
        return False
    left = descriptors.get("left")
    right = descriptors.get("right")
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False

    def flow(value: object) -> str:
        normalized = str(value or "unknown").strip().lower().replace("-", "_")
        if normalized in {"opposite", "oncoming", "reverse"}:
            return "opposing"
        if normalized in {"same", "same_flow", "same_direction"}:
            return "same_direction"
        return normalized

    return (
        flow(left.get("flow")) == "opposing"
        and flow(right.get("flow")) == "same_direction"
    )


def is_leftward_primitive(name: object) -> bool:
    """Return whether a stable recovery primitive belongs to the left menu."""
    return str(name or "").strip() in LEFTWARD_PRIMITIVES


def _source_centerline(info: Dict[str, Any]) -> np.ndarray | None:
    descriptors = info.get("adjacent_lane_flows")
    if not isinstance(descriptors, dict):
        return None
    for side in ("left", "right"):
        descriptor = descriptors.get(side)
        if not isinstance(descriptor, dict):
            continue
        raw = descriptor.get("source_centerline_local_xy")
        try:
            points = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if (
            points.ndim == 2
            and points.shape[0] >= 2
            and points.shape[1] >= 2
            and np.isfinite(points[:, :2]).all()
        ):
            return points[:, :2]
    return None


def proposal_moves_left(
    proposal: Sequence[Sequence[float]] | np.ndarray,
    info: Dict[str, Any],
    *,
    lateral_threshold_m: float = 0.5,
) -> bool:
    """Classify a learned proposal relative to the route-lane centreline.

    Proposals use ``[right, forward]`` coordinates.  Comparing every meaningful
    point with the curved HD-map centreline catches both left lane changes and
    leftward excursions that later return, without treating a road bend as a
    lane change.  The 0.5 m threshold matches the direction labels used in the
    VLM candidate summaries.
    """
    if not suppress_leftward_options(info):
        return False
    centerline = _source_centerline(info)
    if centerline is None:
        return False
    try:
        plan = np.asarray(proposal, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    if plan.ndim != 2 or plan.shape[0] == 0 or plan.shape[1] < 2:
        return False
    finite = np.isfinite(plan[:, :2]).all(axis=1)
    if not np.any(finite):
        return False
    points = plan[finite, :2]

    order = np.argsort(centerline[:, 1], kind="stable")
    forward = centerline[order, 1]
    right = centerline[order, 0]
    unique_forward, unique_idx = np.unique(forward, return_index=True)
    if unique_forward.size < 2:
        return False
    route_right = np.interp(
        points[:, 1],
        unique_forward,
        right[unique_idx],
        left=float(right[unique_idx][0]),
        right=float(right[unique_idx][-1]),
    )
    offsets = points[:, 0] - route_right
    return bool(
        np.any(offsets < -max(0.0, float(lateral_threshold_m)))
    )


def leftward_proposal_mask(
    proposals: np.ndarray,
    info: Dict[str, Any],
) -> np.ndarray:
    """Return a boolean mask aligned with a ``[K,T,2]`` proposal batch."""
    array = np.asarray(proposals)
    if array.ndim != 3 or not suppress_leftward_options(info):
        return np.zeros(len(array) if array.ndim else 0, dtype=bool)
    return np.asarray(
        [proposal_moves_left(proposal, info) for proposal in array],
        dtype=bool,
    )


def _junction_route_filter_context(
    info: Dict[str, Any],
) -> tuple[bool, float, np.ndarray | None]:
    """Return ``(active, max_angle_deg, route_vector)`` from adapter context.

    The CARLA adapter publishes the junction state from the HD map and the
    Leaderboard-route target in the planner's ``[right, forward]`` frame.  All
    fields deliberately fail open: a transient map/route lookup failure must
    not turn a stationary or otherwise safe trajectory into an invalid one.
    """
    context = info.get("junction_route_heading_filter")
    if not isinstance(context, dict) or not bool(context.get("enabled", True)):
        return False, DEFAULT_JUNCTION_ROUTE_MAX_ANGLE_DEG, None
    if not bool(context.get("active", False)):
        return False, DEFAULT_JUNCTION_ROUTE_MAX_ANGLE_DEG, None
    try:
        max_angle_deg = float(
            context.get(
                "max_angle_deg", DEFAULT_JUNCTION_ROUTE_MAX_ANGLE_DEG
            )
        )
    except (TypeError, ValueError):
        max_angle_deg = DEFAULT_JUNCTION_ROUTE_MAX_ANGLE_DEG
    raw_route = context.get(
        "route_heading_local_xy", info.get("target_point")
    )
    try:
        route = np.asarray(raw_route, dtype=np.float64).reshape(-1)[:2]
    except (TypeError, ValueError):
        return False, max_angle_deg, None
    if route.size < 2 or not np.isfinite(route).all():
        return False, max_angle_deg, None
    if float(np.linalg.norm(route)) < _MIN_HEADING_DISPLACEMENT_M:
        return False, max_angle_deg, None
    return True, max(0.0, min(180.0, max_angle_deg)), route


def junction_route_heading_angle_deg(
    proposal: Sequence[Sequence[float]] | np.ndarray,
    info: Dict[str, Any],
) -> float | None:
    """Return proposal/route heading separation while the junction gate is on.

    Proposal heading uses the first-to-last meaningful displacement rather
    than the first predicted point, which is unstable for low-speed plans.
    Brake/wait trajectories whose net displacement is below 0.5 m return
    ``None`` and remain available as the safe fallback.
    """
    active, _max_angle_deg, route = _junction_route_filter_context(info)
    if not active or route is None:
        return None
    try:
        plan = np.asarray(proposal, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if plan.ndim != 2 or plan.shape[0] < 2 or plan.shape[1] < 2:
        return None
    finite = np.isfinite(plan[:, :2]).all(axis=1)
    points = plan[finite, :2]
    if len(points) < 2:
        return None
    trajectory = points[-1] - points[0]
    trajectory_norm = float(np.linalg.norm(trajectory))
    route_norm = float(np.linalg.norm(route))
    if (
        trajectory_norm < _MIN_HEADING_DISPLACEMENT_M
        or route_norm < _MIN_HEADING_DISPLACEMENT_M
    ):
        return None
    cosine = float(
        np.clip(
            np.dot(trajectory, route) / (trajectory_norm * route_norm),
            -1.0,
            1.0,
        )
    )
    return float(np.degrees(np.arccos(cosine)))


def junction_route_heading_angles_deg(
    proposal: Sequence[Sequence[float]] | np.ndarray,
    info: Dict[str, Any],
) -> Dict[str, float | None]:
    """Return full-horizon and tail-horizon route heading separations.

    A trajectory may go straight for its first half and then take the wrong
    junction branch.  Its first-to-last displacement can remain just below
    90 degrees even though the actual turn in the latter half exceeds it.
    Checking a multi-step tail displacement catches that case without relying
    on noisy individual one-step headings.
    """
    full_angle = junction_route_heading_angle_deg(proposal, info)
    active, _max_angle_deg, route = _junction_route_filter_context(info)
    if not active or route is None:
        return {"full": full_angle, "tail": None}
    try:
        plan = np.asarray(proposal, dtype=np.float64)
    except (TypeError, ValueError):
        return {"full": full_angle, "tail": None}
    if plan.ndim != 2 or plan.shape[0] < 3 or plan.shape[1] < 2:
        return {"full": full_angle, "tail": full_angle}
    finite = np.isfinite(plan[:, :2]).all(axis=1)
    points = plan[finite, :2]
    if len(points) < 3:
        return {"full": full_angle, "tail": full_angle}
    midpoint = max(0, (len(points) - 1) // 2)
    tail = points[-1] - points[midpoint]
    tail_norm = float(np.linalg.norm(tail))
    route_norm = float(np.linalg.norm(route))
    if (
        tail_norm < _MIN_HEADING_DISPLACEMENT_M
        or route_norm < _MIN_HEADING_DISPLACEMENT_M
    ):
        return {"full": full_angle, "tail": None}
    cosine = float(
        np.clip(
            np.dot(tail, route) / (tail_norm * route_norm),
            -1.0,
            1.0,
        )
    )
    return {
        "full": full_angle,
        "tail": float(np.degrees(np.arccos(cosine))),
    }


def junction_route_heading_violation(
    proposal: Sequence[Sequence[float]] | np.ndarray,
    info: Dict[str, Any],
) -> bool:
    """Reject a moving proposal only when its junction angle is strictly high."""
    active, max_angle_deg, _route = _junction_route_filter_context(info)
    if not active:
        return False
    angles = junction_route_heading_angles_deg(proposal, info)
    return any(
        angle_deg is not None and angle_deg > max_angle_deg + 1e-6
        for angle_deg in angles.values()
    )


def junction_route_heading_violation_mask(
    proposals: np.ndarray,
    info: Dict[str, Any],
) -> np.ndarray:
    """Return the junction route-angle hard-filter mask for ``[K,T,2]``."""
    array = np.asarray(proposals)
    if array.ndim != 3:
        return np.zeros(len(array) if array.ndim else 0, dtype=bool)
    active, _max_angle_deg, _route = _junction_route_filter_context(info)
    if not active:
        return np.zeros(len(array), dtype=bool)
    return np.asarray(
        [junction_route_heading_violation(plan, info) for plan in array],
        dtype=bool,
    )
