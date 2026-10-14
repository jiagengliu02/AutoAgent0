"""Context helpers for native CARLA observations."""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple



VLM_CAMERA_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
)




def command_to_route_instruction(command: object) -> str:
    mapping = {
        0: "right",
        1: "left",
        2: "straight",
    }
    try:
        return mapping.get(int(command), "Drive safely and choose the most reasonable trajectory.")
    except Exception:
        return "Drive safely and choose the most reasonable trajectory."


def resolve_route_instruction(info: Dict[str, object]) -> str:
    task_instruction = info.get("task_instruction")
    if isinstance(task_instruction, str) and task_instruction.strip():
        return task_instruction.strip()
    return command_to_route_instruction(info.get("command"))


def resolve_route_waypoint(info: Dict[str, object]) -> Optional[Dict[str, float]]:
    """Return the next route waypoint in a prompt-friendly ego-frame form.

    The CARLA helper stores ``target_point`` as ``[right, forward]`` metres in
    the ego frame. Named fields keep that convention explicit for the VLM.
    """
    target_point = info.get("target_point")
    if target_point is None or isinstance(target_point, (str, bytes)):
        return None
    try:
        right_m = float(target_point[0])  # type: ignore[index]
        forward_m = float(target_point[1])  # type: ignore[index]
    except (IndexError, KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(right_m) or not math.isfinite(forward_m):
        return None
    return {
        "forward_m": forward_m,
        "right_m": right_m,
        "distance_m": math.hypot(forward_m, right_m),
        "bearing_deg": math.degrees(math.atan2(right_m, forward_m)),
    }


def describe_task_target_hint(info: Dict[str, object]) -> Optional[str]:
    task_type = str(info.get("task_type", "")).strip()
    if not task_type:
        return None
    if task_type == "park_at_target":
        return "park target"
    if task_type == "stop_at_target":
        return "stop target"
    return "goal target"


def resolve_vlm_camera_order(camera_mode: str) -> Tuple[str, ...]:
    mode = str(camera_mode or "multiview").strip().lower()
    if mode in {"front", "front_only", "single_front"}:
        return ("CAM_FRONT",)
    return tuple(VLM_CAMERA_ORDER)


def describe_vlm_camera_inputs(camera_order: Sequence[str]) -> Tuple[str, str]:
    camera_order = tuple(camera_order)
    if camera_order == ("CAM_FRONT",):
        return (
            "A front-facing driving image.",
            "The image has the trajectory overlay.",
        )
    return (
        "Four driving images in this exact order: front, left, right, back.",
        "Only the front image has the trajectory overlay; left, right, and back are unannotated context images.",
    )




def extract_current_ego_speed_mps(info: Dict[str, object]) -> Optional[float]:
    try:
        return float(info["ego_velo"])
    except Exception:
        return None
