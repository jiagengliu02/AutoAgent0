# carla_agent/agenticdriving_carla_helper.py

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agenticdriving.scorer.selector_config import VLMSelectorConfig
    from agenticdriving.vlm.recovery_reason import RecoveryReasonConfig

import logging
import math
import os
import sys
import json
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import carla
import cv2
import numpy as np

from streaming_video import StreamingVideoWriter, video_frame_is_due

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from agenticdriving.planners.sparsedrivev2.carla_inputs import (
    BENCH2DRIVE_CAMERA_SPECS,
    CAMERAS as SPARSEDRIVEV2_CAMERAS,
    carla_adjacent_lane_flows,
    carla_bbox_world_pose,
    carla_route_lane_relation,
    carla_route_waypoint_at_ego_progress,
    carla_transform_matrix,
    carla_waypoint_lane_key,
    driving_waypoint_at,
    projected_route_condition_from_polyline,
    route_condition_from_polyline,
    route_projection_from_polyline,
)

LOG = logging.getLogger(__name__)


def _ensure_console_logging(level: int = logging.INFO) -> None:
    """Attach an INFO ``StreamHandler`` to this module's logger exactly once.

    The leaderboard client never calls ``logging.basicConfig``, so the root
    logger stays at WARNING and ``LOG.info(...)`` from this module (including the
    ``speed_diag`` line) gets swallowed. This wires a stdout handler directly on
    our logger so those diagnostics actually print. Idempotent / import-safe.
    """
    if getattr(_ensure_console_logging, "_done", False):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
    )
    LOG.addHandler(handler)
    LOG.setLevel(level)
    LOG.propagate = False  # avoid double-printing if root later gets a handler
    _ensure_console_logging._done = True


_ensure_console_logging()

EGO_HISTORY_FRAMES = 4
PRIVILEGED_AGENT_RADIUS_M = 50.0
# Reject agents whose z is farther than this from the ego. CARLA sometimes
# parks walkers it could not place on the navmesh ~100 m under the map; they
# share the ego's (x, y) so a 2D radius test keeps them, and their vertical
# offset projects into the ego-local plane via ego pitch, tripping the NC
# collision gate. Real road-level agents stay within a few metres of ego z.
PRIVILEGED_AGENT_MAX_VERT_SEP_M = 4.0

# Default time (s) between consecutive plan waypoints. Learned planners
# (DrivoR/RAP) emit poses every 0.5 s; the rule planner emits every 0.25 s.
# The actual per-planner value is stored on ``agent._plan_dt_sec`` at setup.
PLAN_DT_SEC = 0.5
# Must match ``control/ilqr/lqr.py`` ILQRSolverParameters.discretization_time.
LQR_DISCRETIZATION_TIME = 0.5

# Generic CARLA camera mounts used by planners other than SparseDriveV2.
CAMERA_MOUNTS: Dict[str, Dict[str, float]] = {
    "CAM_FRONT": {"x": 0.7, "y": 0.0, "z": 1.6, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
    "CAM_FRONT_LEFT": {"x": 0.7, "y": -0.4, "z": 1.6, "roll": 0.0, "pitch": 0.0, "yaw": -55.0},
    "CAM_FRONT_RIGHT": {"x": 0.7, "y": 0.4, "z": 1.6, "roll": 0.0, "pitch": 0.0, "yaw": 55.0},
    "CAM_BACK": {"x": -1.0, "y": 0.0, "z": 1.6, "roll": 0.0, "pitch": 0.0, "yaw": 180.0},
    "CAM_BACK_LEFT": {"x": -0.7, "y": -0.4, "z": 1.6, "roll": 0.0, "pitch": 0.0, "yaw": -110.0},
    "CAM_BACK_RIGHT": {"x": -0.7, "y": 0.4, "z": 1.6, "roll": 0.0, "pitch": 0.0, "yaw": 110.0},
}

CAMERA_RIGS: Dict[str, List[str]] = {
    "front_only": ["CAM_FRONT"],
    "4cam": [
        "CAM_BACK",
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
    ],
    "full_6cam": list(CAMERA_MOUNTS.keys()),
}

# Bench2DriveZoo UniAD uses the original 1600x900 six-camera calibration plus
# a 512x512 top-down semantic camera and raw IMU/GNSS.  Keep this rig explicit
# so its model input is byte/geometry compatible with the reference agent.

VIDEO_GRID_LAYOUT = [
    ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"],
    ["CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT"],
]

TOPDOWN_CAMERA_ID = "CAM_TOPDOWN"
# Recording-only high-res duplicate of CAM_FRONT (front pane of front_topdown).
# Never enters obs["rgb"], so planner input is byte-identical with or without it.
RECORDING_CAMERA_ID = "CAM_FRONT_REC"
TOPDOWN_VIZ_TOPK = 0  # <= 0 means draw all pre-scorer proposal trajectories.
TOPDOWN_SELECTED_COLOR_BGR = (0, 255, 255)
TOPDOWN_SELECTED_START_COLOR_BGR = (0, 255, 0)
TOPDOWN_SELECTED_END_COLOR_BGR = (0, 0, 255)
TOPDOWN_SELECTED_OUTLINE_BGR = (0, 0, 0)
TOPDOWN_RAW_MODEL_COLOR_BGR = (255, 0, 255)
TOPDOWN_EGO_COLOR_BGR = (255, 255, 255)
# Top-K candidate trajectories from the planning head are drawn as purple,
# slightly-bolder lines; the selected plan mirrors Bench2DriveZoo UniAD's
# black-outline/yellow-line waypoint rendering so it stands out on top.
TOPDOWN_CANDIDATE_COLOR_BGR = (255, 0, 255)  # purple/magenta (BGR)
# Line thickness for the top-down trajectory overlay.
TOPDOWN_LINE_THICKNESS = 1
TOPDOWN_CANDIDATE_LINE_THICKNESS = 2
TOPDOWN_RAW_MODEL_LINE_THICKNESS = 1
TOPDOWN_SELECTED_OUTLINE_THICKNESS = 5
TOPDOWN_SELECTED_LINE_THICKNESS = 3
# Thin variant used on the single replan tick per async cycle -- the one frame
# that also draws all the candidate trajectories -- so the selected line does
# not mask the candidates. The reuse ticks that follow (which show only the
# selected line) use the bold thickness above.
TOPDOWN_SELECTED_OUTLINE_THICKNESS_THIN = 3
TOPDOWN_SELECTED_LINE_THICKNESS_THIN = 1


def navsim_proposal_to_carla_local_plan(trajectory: np.ndarray) -> np.ndarray:
    traj = np.asarray(trajectory, dtype=np.float32)
    if traj.ndim == 1:
        traj = traj.reshape(1, -1)
    right = -traj[:, 1]
    forward = traj[:, 0]
    return np.stack([right, forward], axis=-1).astype(np.float32)


def build_step_viz_payload(
    *,
    selected_plan: np.ndarray,
    selected_source: str,
    selected_score: Optional[float],
    proposals: np.ndarray,
    scores: np.ndarray,
    output_num_poses: int,
    topk: int = TOPDOWN_VIZ_TOPK,
    proposals_already_local: bool = False,
    recovered: bool = False,
    verifier_passed: bool = True,
    verifier_score: Optional[float] = None,
    fallback_primitive: Optional[str] = None,
    fallback_mode: Optional[str] = None,
    verifier_metrics: Optional[Dict[str, float]] = None,
    selected_bold: bool = True,
    candidate_color_scores: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    proposals = np.asarray(proposals, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if proposals.ndim == 2:
        proposals = proposals[np.newaxis, ...]
        scores = scores.reshape(1)

    # Optional final visualization/ranking score (legacy debug key
    # ``combined_scores``), aligned with proposals. In parallel proposal mode
    # this contains PDMS only at the two branch winners.
    # Ignored on a length mismatch, which falls back to native ranking / flat purple.
    color_scores = None
    if candidate_color_scores is not None:
        color_scores = np.asarray(candidate_color_scores, dtype=np.float32).reshape(-1)
        if color_scores.shape[0] != scores.shape[0]:
            color_scores = None
    rank_scores = color_scores if color_scores is not None else scores

    if int(topk) <= 0:
        top_indices = np.argsort(rank_scores)[::-1]
    else:
        topk = max(1, min(int(topk), int(len(rank_scores))))
        top_indices = np.argsort(rank_scores)[-topk:][::-1]
    candidate_plans: List[np.ndarray] = []
    candidate_scores: List[float] = []
    candidate_color: List[float] = []
    for idx in top_indices:
        if proposals_already_local:
            plan = np.asarray(proposals[int(idx)], dtype=np.float32)
        else:
            plan = navsim_proposal_to_carla_local_plan(
                proposals[int(idx), :output_num_poses]
            )
        candidate_plans.append(plan)
        candidate_scores.append(float(scores[int(idx)]))
        if color_scores is not None:
            candidate_color.append(float(color_scores[int(idx)]))

    return {
        "selected_plan": np.asarray(selected_plan, dtype=np.float32),
        "selected_source": str(selected_source),
        "selected_score": (
            None if selected_score is None else float(selected_score)
        ),
        "candidate_plans": candidate_plans,
        "candidate_scores": candidate_scores,
        # Combined score per candidate (same order as candidate_plans) for the
        # purple->white shading; None when no color scores were supplied.
        "candidate_color_scores": (candidate_color if color_scores is not None else None),
        "recovered": bool(recovered),
        "verifier_passed": bool(verifier_passed),
        "verifier_score": (
            None if verifier_score is None else float(verifier_score)
        ),
        "fallback_primitive": (
            None if fallback_primitive is None else str(fallback_primitive)
        ),
        "fallback_mode": (
            None if fallback_mode is None else str(fallback_mode)
        ),
        # PDMS sub-metrics {nc, dac, ttc, ep} for the top-down overlay.
        "verifier_metrics": (
            {str(k): float(v) for k, v in verifier_metrics.items()}
            if verifier_metrics else None
        ),
        # Whether the selected plan is drawn bold. False on the async replan tick
        # (the frame that also shows all candidates) so it does not mask them.
        "selected_bold": bool(selected_bold),
    }


def resolve_predictions_dir(agent) -> Optional[Path]:
    raw = agent._carla_cfg.get("predictions_dir")
    if not raw:
        return None
    path = Path(str(raw)).expanduser()
    auto_run_id = coerce_bool(agent._carla_cfg.get("auto_run_id", True), default=True)
    if auto_run_id:
        rel = path.name if path.is_absolute() else str(path)
        if rel in {".", ""}:
            rel = "predictions"
        return agent._output_dir / rel
    if not path.is_absolute():
        path = agent._output_dir / path
    return path


def build_topdown_camera_spec(carla_cfg: Dict[str, Any]) -> Dict[str, Any]:
    td_cfg = carla_cfg.get("topdown_camera") or {}
    width = int(td_cfg.get("width", 800))
    height = int(td_cfg.get("height", width))
    return {
        "id": TOPDOWN_CAMERA_ID,
        "type": "sensor.camera.rgb",
        "x": float(td_cfg.get("x", 0.0)),
        "y": float(td_cfg.get("y", 0.0)),
        "z": float(td_cfg.get("z", 45.0)),
        "roll": float(td_cfg.get("roll", 0.0)),
        "pitch": float(td_cfg.get("pitch", -90.0)),
        "yaw": float(td_cfg.get("yaw", 0.0)),
        "width": width,
        "height": height,
        "fov": float(td_cfg.get("fov", 90.0)),
    }


def build_recording_camera_spec(
    cam_cfg: Optional[Dict[str, Any]], camera_specs: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Recording-only camera spec from a camera-config dict (either the
    legacy ``carla.recording_camera`` block or the dict synthesized by the
    simple ``recording.camera:`` selector). Defaults duplicate the active
    rig's CAM_FRONT mount/FOV so framing matches the planner's view; only
    the pixel count differs. Returns None when absent/disabled."""
    cfg = cam_cfg or {}
    if not cfg or not coerce_bool(cfg.get("enabled", True), default=True):
        return None
    front = next(
        (spec for spec in camera_specs if spec["id"] == "CAM_FRONT"), None
    )
    mount_keys = ("x", "y", "z", "roll", "pitch", "yaw", "fov")
    if front is not None:
        mount = {k: float(front[k]) for k in mount_keys}
    else:
        mount = {
            "x": 0.80, "y": 0.0, "z": 1.60,
            "roll": 0.0, "pitch": 0.0, "yaw": 0.0, "fov": 70.0,
        }
    # ``name`` selects a view PRESET and labels the artifacts
    # (<name>.mp4 / <name>/ frame dir). Minimal config is just:
    #   recording_camera: {name: chase_back, save_video: true}
    # front_1080 (default): the rig's CAM_FRONT view at 1920x1080.
    # chase_back: fail2drive-hero style third-person chase.
    # Explicit mount/size keys in the block still override the preset.
    name = re.sub(
        r"[^A-Za-z0-9_-]+", "_", str(cfg.get("name", "front_1080")).strip()
    ) or "front_1080"
    presets = {
        "front_1080": {},
        "chase_back": {"x": -6.0, "z": 3.0, "pitch": -15.0, "fov": 90.0},
    }
    mount.update(presets.get(name, {}))
    return {
        "id": RECORDING_CAMERA_ID,
        "type": "sensor.camera.rgb",
        "name": name,
        **{k: float(cfg.get(k, mount[k])) for k in mount_keys},
        "width": int(cfg.get("width", 1920)),
        "height": int(cfg.get("height", 1080)),
    }


def topdown_pixels_per_meter(width: int, fov_deg: float, altitude_m: float) -> float:
    ground_span_m = 2.0 * altitude_m * math.tan(math.radians(fov_deg * 0.5))
    return float(width) / max(ground_span_m, 1e-3)


def _plan_to_pixel_points(
    plan: np.ndarray,
    *,
    center_xy: Tuple[int, int],
    pixels_per_meter: float,
) -> np.ndarray:
    plan = np.asarray(plan, dtype=np.float32)
    if len(plan) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    cx, cy = center_xy
    xs = cx + plan[:, 0] * pixels_per_meter
    ys = cy - plan[:, 1] * pixels_per_meter
    return np.round(np.stack([xs, ys], axis=1)).astype(np.int32)


def _prepend_plan_origin(plan: np.ndarray) -> np.ndarray:
    plan = np.asarray(plan, dtype=np.float32)
    if plan.ndim != 2 or plan.shape[0] == 0:
        return plan.reshape(0, 2)
    xy = plan[:, :2]
    if np.linalg.norm(xy[0]) < 1e-3:
        return xy
    return np.concatenate([np.zeros((1, 2), dtype=np.float32), xy], axis=0)




def _draw_selected_plan_polyline(
    img: np.ndarray, pts: np.ndarray, *, bold: bool = True
) -> None:
    """Draw the post-scorer trajectory like Bench2DriveZoo's UniAD BEV overlay.

    ``bold`` (default) draws the thick UniAD-style line. Pass ``bold=False`` on
    the async replan tick -- the single frame per cycle that also shows every
    candidate trajectory -- so the thin selected line does not mask them; the
    reuse ticks that follow (selected line only) keep the bold form.
    """
    if len(pts) < 2:
        return

    outline_thickness = (
        TOPDOWN_SELECTED_OUTLINE_THICKNESS if bold
        else TOPDOWN_SELECTED_OUTLINE_THICKNESS_THIN
    )
    line_thickness = (
        TOPDOWN_SELECTED_LINE_THICKNESS if bold
        else TOPDOWN_SELECTED_LINE_THICKNESS_THIN
    )
    # Shrink the waypoint dots on the thin variant so they match the slimmer line.
    radius_delta = 0 if bold else -2

    cv2.polylines(
        img,
        [pts],
        isClosed=False,
        color=TOPDOWN_SELECTED_OUTLINE_BGR,
        thickness=outline_thickness,
        lineType=cv2.LINE_AA,
    )
    cv2.polylines(
        img,
        [pts],
        isClosed=False,
        color=TOPDOWN_SELECTED_COLOR_BGR,
        thickness=line_thickness,
        lineType=cv2.LINE_AA,
    )

    height, width = img.shape[:2]
    for idx, point in enumerate(pts):
        x, y = int(point[0]), int(point[1])
        if not (0 <= x < width and 0 <= y < height):
            continue
        if idx == 0:
            color = TOPDOWN_SELECTED_START_COLOR_BGR
            radius = 4
        elif idx == len(pts) - 1:
            color = TOPDOWN_SELECTED_END_COLOR_BGR
            radius = 4
        else:
            color = TOPDOWN_SELECTED_COLOR_BGR
            radius = 3
        radius = max(1, radius + radius_delta)
        cv2.circle(
            img,
            (x, y),
            radius + 2,
            TOPDOWN_SELECTED_OUTLINE_BGR,
            -1,
            cv2.LINE_AA,
        )
        cv2.circle(img, (x, y), radius, color, -1, cv2.LINE_AA)


def _raw_model_plan_from_debug(viz_payload: Dict[str, Any]) -> Optional[np.ndarray]:
    debug = viz_payload.get("planner_debug")
    if not isinstance(debug, dict):
        return None
    raw_plan = debug.get("carla_local_plan_xy")
    if raw_plan is None:
        return None
    plan = np.asarray(raw_plan, dtype=np.float32)
    if plan.ndim != 2 or plan.shape[1] < 2 or len(plan) == 0:
        return None
    return plan[:, :2]


def _draw_raw_model_plan(img: np.ndarray, pts: np.ndarray) -> None:
    if len(pts) < 2:
        return
    cv2.polylines(
        img,
        [pts],
        isClosed=False,
        color=TOPDOWN_RAW_MODEL_COLOR_BGR,
        thickness=TOPDOWN_RAW_MODEL_LINE_THICKNESS,
        lineType=cv2.LINE_AA,
    )


def _draw_wrapped_text_block(
    img: np.ndarray,
    text: str,
    *,
    origin_x: int,
    bottom_y: int,
    max_width: int,
    font_scale: float = 0.6,
    color_bgr: Tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
    line_gap: int = 8,
) -> None:
    """Word-wrap ``text`` to ``max_width`` px and draw it bottom-up from
    ``bottom_y`` so the block grows upward, away from the top-left HUD lines.

    Each line is stroked black first, then filled, so it stays legible over any
    BEV background. In-place edit of the BGR ``img``."""
    words = str(text).split()
    if not words:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines: List[str] = []
    cur = words[0]
    for word in words[1:]:
        trial = f"{cur} {word}"
        (w, _), _ = cv2.getTextSize(trial, font, font_scale, thickness)
        if w <= max_width:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    lines.append(cur)
    (_, line_h), _ = cv2.getTextSize("Ag", font, font_scale, thickness)
    step = line_h + line_gap
    y = bottom_y
    for line in reversed(lines):
        cv2.putText(img, line, (origin_x, y), font, font_scale, (0, 0, 0),
                    thickness + 2, cv2.LINE_AA)
        cv2.putText(img, line, (origin_x, y), font, font_scale, color_bgr,
                    thickness, cv2.LINE_AA)
        y -= step


def _candidate_gradient_color_bgr(t: float) -> Tuple[int, int, int]:
    """White (t=0, lowest score) -> purple (t=1, highest score), in BGR.

    Purple is (255, 0, 255); white is (255, 255, 255), so only the green channel
    ramps with the normalized score ``t`` -- inverted so the highest-scoring
    candidate is the most saturated purple and the lowest fades to white."""
    t = max(0.0, min(1.0, float(t)))
    return (255, int(round(255 * (1.0 - t))), 255)


def overlay_trajectories_on_topdown(
    image_rgb: np.ndarray,
    viz_payload: Dict[str, Any],
    *,
    pixels_per_meter: float,
    ego_center_xy: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    height, width = img.shape[:2]
    center = ego_center_xy or (width // 2, height // 2)

    # Candidates are already ordered best -> worst, so the index is the rank.
    candidate_plans = viz_payload.get("candidate_plans", [])
    # Per-candidate final display score used to shade each line on a
    # purple(lowest)->white(highest) gradient. In the two-scorer path this shows
    # gated End2End rank; it does not add PDMS and End2End scores together.
    color_scores = viz_payload.get("candidate_color_scores")
    color_arr = None
    if (
        isinstance(color_scores, (list, tuple, np.ndarray))
        and len(color_scores) == len(candidate_plans)
        and len(candidate_plans) > 0
    ):
        color_arr = np.asarray(color_scores, dtype=np.float64)
        cmin = float(color_arr.min())
        cspan = float(color_arr.max()) - cmin
    for rank_idx, plan in enumerate(candidate_plans):
        pts = _plan_to_pixel_points(
            _prepend_plan_origin(plan),
            center_xy=center,
            pixels_per_meter=pixels_per_meter,
        )
        if len(pts) < 2:
            continue
        # Shade by combined score when available (purple->white); otherwise fall
        # back to the flat purple used by the other planners. The selected plan
        # (yellow, below) is drawn on top regardless.
        if color_arr is not None:
            t = 0.5 if cspan <= 1e-9 else (float(color_arr[rank_idx]) - cmin) / cspan
            color = _candidate_gradient_color_bgr(t)
        else:
            color = TOPDOWN_CANDIDATE_COLOR_BGR
        cv2.polylines(
            img,
            [pts],
            isClosed=False,
            color=color,
            thickness=TOPDOWN_CANDIDATE_LINE_THICKNESS,
            lineType=cv2.LINE_AA,
        )
        end_pt = (int(pts[-1][0]), int(pts[-1][1]))
        cv2.putText(
            img,
            str(rank_idx + 1),
            end_pt,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )

    selected_plan = viz_payload.get("selected_plan")
    if selected_plan is not None and len(selected_plan) > 0:
        pts = _plan_to_pixel_points(
            _prepend_plan_origin(selected_plan),
            center_xy=center,
            pixels_per_meter=pixels_per_meter,
        )
        if len(pts) >= 2:
            _draw_selected_plan_polyline(
                img, pts, bold=bool(viz_payload.get("selected_bold", True))
            )

    raw_model_plan = _raw_model_plan_from_debug(viz_payload)
    if raw_model_plan is not None:
        pts = _plan_to_pixel_points(
            _prepend_plan_origin(raw_model_plan),
            center_xy=center,
            pixels_per_meter=pixels_per_meter,
        )
        if len(pts) >= 2:
            _draw_raw_model_plan(img, pts)

    cv2.circle(
        img, center, 6, TOPDOWN_EGO_COLOR_BGR, thickness=-1, lineType=cv2.LINE_AA
    )

    # All three HUD lines are drawn on every single frame, with placeholders when
    # a value is unavailable. Nothing here may be conditional: a line that only
    # appears on replan ticks reads as flicker at control_frequency.

    # Line 1 -- the policy currently driving the ego.
    policy = _policy_label(viz_payload.get("selected_source", "")) or "unknown"
    score = viz_payload.get("selected_score")
    score_text = "--" if score is None else f"{float(score):.2f}"
    cv2.putText(
        img,
        f"POLICY: {policy} ({score_text})",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        TOPDOWN_SELECTED_COLOR_BGR,
        2,
        cv2.LINE_AA,
    )

    # Line 2 -- verifier verdict + ego speed. Red while a recovery is driving.
    recovery_label, recovery_color = _recovery_strategy_label(viz_payload)
    cv2.putText(
        img,
        recovery_label,
        (12, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        recovery_color,
        2,
        cv2.LINE_AA,
    )

    # Line 3 -- PDMS sub-metrics: NC/DAC (hard gates) + TTC/EP/LK (soft terms).
    # A term is red when it is 0.0 (i.e. what dragged PDMS down). Every term gets
    # a fixed-width slot so the line never reflows when one is missing (LK shows
    # "--" when the term is disabled or the probe lacks .details()).
    metrics = viz_payload.get("verifier_metrics") or {}
    slot_width, _ = cv2.getTextSize(
        "DAC=0.00  ", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    )[0]
    for slot, name in enumerate(("nc", "dac", "ttc", "ep", "lk")):
        value = metrics.get(name)
        if value is None:
            text, color = f"{name.upper()}=--", (170, 170, 170)
        else:
            value = float(value)
            color = (0, 0, 255) if value == 0.0 else (255, 255, 255)
            text = f"{name.upper()}={value:.2f}"
        cv2.putText(
            img, text, (12 + slot * slot_width, 84),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
        )

    # Bottom block -- the recovery-reason VLM output (explain()): why the
    # verifier rejected the plan, in the model's own words. Present only while a
    # recovery event is active (carried across the commit-and-hold reuse ticks by
    # the sticky-viz group); absent frames draw nothing.
    vlm = viz_payload.get("vlm_reason")
    if isinstance(vlm, dict):
        reason = str(vlm.get("reason") or "").strip()
        if reason:
            cause = str(vlm.get("primary_cause") or "").strip()
            conf = vlm.get("confidence")
            header = "VLM"
            if cause:
                header += f" [{cause}]"
            if conf is not None:
                header += f" ({float(conf):.2f})"
            # Sit above the bottom-left "BEV" caption that compose_front_topdown_
            # frame stamps at height-12, so the two never overlap.
            _draw_wrapped_text_block(
                img, f"{header}: {reason}",
                origin_x=12, bottom_y=height - 44, max_width=width - 24,
            )
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# Green (pass) / red (recovering) in BGR for the recovery-strategy overlay line.
_RECOVERY_OK_COLOR_BGR = (0, 200, 0)
_RECOVERY_ACTIVE_COLOR_BGR = (0, 0, 255)

# Verifier fields only exist on replan ticks; on the reuse ticks in between they
# are absent. Carry the last known values forward so the overlay lines stay put
# instead of blinking at (control_frequency / replan_frequency) Hz.
_STICKY_VIZ_KEYS = (
    "recovered",
    "verifier_passed",
    "verifier_score",
    "verifier_metrics",
    "fallback_primitive",
    "fallback_mode",
    # VLM recovery-reason text: produced on the recovery-trigger replan tick,
    # carried through the commit-and-hold reuse ticks with the verifier group,
    # then cleared on the next verifier run that does not re-trigger recovery.
    "vlm_reason",
)


def _policy_label(selected_source: Any) -> str:
    """Name of the policy actually driving the ego, for the top-down overlay.

    The planner runs at replan_frequency while control runs faster, so between
    replans the source is tagged ``_cached``. That is a scheduling detail, not a
    different policy -- strip it so the label does not flicker frame to frame."""
    source = str(selected_source or "")
    if source.endswith("_cached"):
        source = source[: -len("_cached")]
    return source


def _carry_sticky_viz_fields(agent, viz_payload: Dict[str, Any]) -> None:
    """Backfill replan-only fields on reuse ticks from the last replan tick.

    These six fields are produced as one group by a verifier run, and
    ``build_step_viz_payload`` always emits all of them -- filled in on a replan
    tick, at their defaults (``verifier_score=None``) on a reuse tick. So a
    per-key "is it present?" test cannot tell the two apart; the group is carried
    atomically instead, keyed off whether a verifier actually ran this tick."""
    sticky = getattr(agent, "_last_verifier_viz", None)
    if viz_payload.get("verifier_score") is not None:
        agent._last_verifier_viz = {
            key: viz_payload.get(key) for key in _STICKY_VIZ_KEYS
        }
    elif sticky:
        viz_payload.update(sticky)


def _recovery_strategy_label(
    viz_payload: Dict[str, Any],
) -> Tuple[str, Tuple[int, int, int]]:
    """Build the verifier line drawn under the POLICY line on the top-down view.

    The recovery maneuver itself is already named by the POLICY line, so this
    line only carries the verdict plus ego speed. Always returns a non-empty
    ``(text, bgr_color)`` -- the caller draws it on every frame, so "no verifier
    ran" has to render as a placeholder rather than as a missing line."""
    recovered = bool(viz_payload.get("recovered", False))
    verifier_score = viz_payload.get("verifier_score")
    verifier_passed = bool(viz_payload.get("verifier_passed", True))
    # `v=` suffix now reports the ego vehicle speed (km/h) instead of the
    # verifier aggregate score.
    ego_speed = viz_payload.get("ego_speed_mps")
    speed = "--" if ego_speed is None else f"{float(ego_speed) * 3.6:.1f}"

    if verifier_score is None:
        verdict, color = "off", _RECOVERY_OK_COLOR_BGR
    else:
        verdict = "OK" if verifier_passed else "FAIL"
        color = _RECOVERY_ACTIVE_COLOR_BGR if recovered else _RECOVERY_OK_COLOR_BGR
    return f"verifier: {verdict}  v={speed}km/h", color


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def save_predictions_frame(
    agent,
    frame_idx: int,
    viz_payload: Dict[str, Any],
    info: Dict[str, Any],
) -> None:
    predictions_dir = getattr(agent, "_predictions_dir", None)
    if predictions_dir is None:
        return
    predictions_dir.mkdir(parents=True, exist_ok=True)
    selected_only = bool(viz_payload.get("_prediction_selected_only", False))
    payload = {
        "frame_index": int(frame_idx),
        "timestamp": float(info.get("timestamp", 0.0)),
        "selected_source": str(viz_payload.get("selected_source", "")),
        "selected_score": viz_payload.get("selected_score"),
        "selected_plan": np.asarray(
            viz_payload.get("selected_plan", []), dtype=np.float32
        ).tolist(),
        # Commanded control + previous-tick pedals CARLA applied (applied_*).
        "control": getattr(agent, "_last_applied_control", None),
        "candidates": [] if selected_only else [
            {
                "score": float(score),
                "plan": np.asarray(plan, dtype=np.float32).tolist(),
            }
            for plan, score in zip(
                viz_payload.get("candidate_plans", []),
                viz_payload.get("candidate_scores", []),
            )
        ],
    }
    if "_prediction_planner_debug" in viz_payload:
        planner_debug = viz_payload.get("_prediction_planner_debug")
    else:
        planner_debug = viz_payload.get("planner_debug")
    if planner_debug is not None:
        payload["planner_debug"] = _jsonable(planner_debug)
    out_path = predictions_dir / f"{frame_idx:05d}.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2)


def record_post_step_visualization(agent, ego_state: Dict[str, Any]) -> None:
    viz_payload = getattr(agent, "_step_viz", None)
    agent._step_viz = None
    # Ticks that bailed to a brake fallback never built a payload. Still emit a
    # video frame for them so the recording keeps one frame per control tick --
    # skipping them desyncs the video from the run. They are not written to
    # predictions/, which stays a record of actual planner output.
    planned = viz_payload is not None
    if not planned:
        viz_payload = {"selected_source": "braking", "selected_score": None}

    _carry_sticky_viz_fields(agent, viz_payload)

    # Ego speed (m/s) for the top-down recovery line's `v=` suffix.
    ego_speed = ego_state.get("speed_mps")
    if ego_speed is not None:
        viz_payload["ego_speed_mps"] = float(ego_speed)

    info = ego_state.get("info", {})
    frame_idx = agent._frame_index
    if planned:
        save_predictions_frame(agent, frame_idx, viz_payload, info)

    want_combined = getattr(agent, "_recording_save_combined_video", False)
    if not want_combined:
        return

    topdown_rgb = ego_state.get("topdown_rgb")
    if topdown_rgb is None:
        spec = getattr(agent, "_topdown_camera_spec", None) or {}
        width = int(spec.get("width", 800))
        height = int(spec.get("height", width))
        topdown_rgb = np.full((height, width, 3), 32, dtype=np.uint8)

    pixels_per_meter = float(
        getattr(agent, "_topdown_pixels_per_meter", 10.0)
    )

    # Optional simulator-object debug overlay. Strict sensor-only perception
    # modes normally never query CARLA actors/EnvironmentObjects here. The
    # explicit ``comparison`` profile is a recording-only diagnostic escape
    # hatch: it queries GT solely for visualization and never feeds it back to
    # planning or scoring.
    # Overlay profile via
    # carla.recording.overlay: "full" (default — green exact + red PDMS +
    # labels, the transform-regression tripwire), "judge" (decluttered
    # perception review: GT as red corner brackets only, no labels, so the
    # sensed rectangles drawn on top stay legible), "comparison" (GT red +
    # live sensor perception green), or "minimal" (no boxes).
    recording_cfg = (getattr(agent, "_carla_cfg", {}) or {}).get("recording") or {}
    overlay_mode = str(recording_cfg.get("overlay", "full")).strip().lower()
    comparison_mode = overlay_mode in {"comparison", "compare", "gt_perception"}
    from perception_helper import object_ground_truth_allowed

    ground_truth_agents = []
    # ``comparison`` is intentionally opt-in and recording-only. It does not
    # alter object_ground_truth_allowed(), which remains the guard for every
    # control/scoring perception path.
    if object_ground_truth_allowed(agent) or comparison_mode:
        try:
            ground_truth_agents = get_privileged_info(agent, info)
        except Exception:
            LOG.exception("Failed to collect GT boxes for the top-down overlay")
    if comparison_mode:
        from agenticdriving.adapters.carla.perception_overlay import (
            PERCEPTION_COMPARISON_BOX_COLOR_BGR,
            PERCEPTION_COMPARISON_GT_COLOR_BGR,
            render_perception_comparison_on_topdown,
        )

        perception = getattr(agent, "_perception", None)
        detected_agents = []
        detection_timestamp = float(info.get("timestamp", 0.0) or 0.0)
        if perception is not None and getattr(perception, "source", None) not in {
            "ground_truth",
            "none",
        }:
            detected_agents = list(
                getattr(perception, "last_agents", None)
                or getattr(perception, "cache_agents", None)
                or ()
            )
            if getattr(perception, "last_detection_ts", None) is not None:
                detection_timestamp = float(perception.last_detection_ts)

        topdown_rgb = render_perception_comparison_on_topdown(
            topdown_rgb,
            info,
            detected_agents=detected_agents,
            ground_truth_agents=ground_truth_agents,
            pixels_per_meter=pixels_per_meter,
            shadow_timestamp=detection_timestamp,
            detected_color_bgr=PERCEPTION_COMPARISON_BOX_COLOR_BGR,
            ground_truth_color_bgr=PERCEPTION_COMPARISON_GT_COLOR_BGR,
            detected_label="perception (green)",
            ground_truth_label="GT (red)",
        )
    elif ground_truth_agents and overlay_mode != "minimal":
        from agenticdriving.adapters.carla.perception_overlay import (
            render_ground_truth_boxes_on_topdown,
        )

        topdown_rgb = render_ground_truth_boxes_on_topdown(
            topdown_rgb,
            info,
            ground_truth_agents=ground_truth_agents,
            pixels_per_meter=pixels_per_meter,
            style="judge" if overlay_mode == "judge" else "full",
        )

    # The top-down overlay is only an intermediate now: it is composed with the
    # front camera into front_topdown.mp4. The standalone topdown.mp4 was removed.
    overlay = overlay_trajectories_on_topdown(
        topdown_rgb,
        viz_payload,
        pixels_per_meter=pixels_per_meter,
    )
    if getattr(agent, "_planner_type", None) == "sparsedrivev2":
        from agenticdriving.adapters.carla.perception_overlay import (
            render_sparsedrive_target_point_on_topdown,
        )

        overlay = render_sparsedrive_target_point_on_topdown(
            overlay,
            viz_payload.get("sparsedrivev2_target_point"),
            pixels_per_meter=pixels_per_meter,
            metadata=viz_payload.get("sparsedrivev2_target_point_meta"),
        )
    # Prefer the recording-only high-res front camera when attached; fall
    # back to the planner rig's CAM_FRONT (upscaled by the composite).
    front_src = ego_state.get("rec_front_rgb")
    if front_src is None:
        front_src = ego_state.get("front_rgb")
    combined = compose_front_topdown_frame(front_src, overlay)
    if combined is not None:
        streamed = record_streaming_video_frame(
            agent,
            "front_topdown",
            "front_topdown.mp4",
            combined,
        )
        save_frames = getattr(agent, "_recording_save_combined_frames", False)
        combined_bgr = None
        if not streamed or save_frames:
            combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        if not streamed:
            ok, enc = cv2.imencode(".jpg", combined_bgr)
            if ok:
                agent._combined_video_buffer.append(enc)
        if save_frames:
            frames_dir = agent._recording_dir / "front_topdown"
            frames_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(
                    frames_dir
                    / f"{frame_idx:05d}.{agent._recording_frame_ext}"
                ),
                combined_bgr,
            )



def _sanitize_run_id(run_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(run_id).strip())
    return cleaned or "run"


def resolve_scenario_output_dir(
    carla_cfg: Dict[str, Any],
    *,
    base_output_dir: Path,
    planner_type: str,
    scenario_name: Optional[str] = None,
    repetition_index: int = 0,
    config_path: Optional[str] = None,
) -> Path:
    """Create a per-scenario output directory under base_output_dir/scenarios/.

    When auto_run_id is true (default), each leaderboard route (e.g. RouteScenario_1085)
    gets its own directory. Re-running the same scenario overwrites that directory.
    """
    auto_run_id = coerce_bool(carla_cfg.get("auto_run_id", True), default=True)
    if not auto_run_id:
        base_output_dir.mkdir(parents=True, exist_ok=True)
        return base_output_dir

    scenario_id = carla_cfg.get("scenario_id") or carla_cfg.get("run_id")
    if not scenario_id:
        if scenario_name:
            scenario_id = str(scenario_name)
            if int(repetition_index) > 0:
                scenario_id = f"{scenario_id}_rep{int(repetition_index)}"
        else:
            scenario_id = f"{planner_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    scenario_id = _sanitize_run_id(str(scenario_id))

    scenarios_parent = (
        str(carla_cfg.get("scenarios_dir_name", carla_cfg.get("run_dir_name", "scenarios"))).strip()
        or "scenarios"
    )
    scenario_output_dir = base_output_dir / scenarios_parent / scenario_id

    overwrite = coerce_bool(carla_cfg.get("overwrite_scenario_output", True), default=True)
    if scenario_output_dir.exists() and overwrite:
        shutil.rmtree(scenario_output_dir)
    scenario_output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "scenario_id": scenario_id,
        "scenario_name": scenario_name,
        "repetition_index": int(repetition_index),
        "planner_type": planner_type,
        "helper_path": str(Path(__file__).resolve()),
        "base_output_dir": str(base_output_dir),
        "scenario_output_dir": str(scenario_output_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": config_path,
        "control_cfg": dict(carla_cfg.get("control", {}) or {}),
        "overwrite": overwrite,
    }
    manifest_path = scenario_output_dir / "run_info.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    LOG.info(
        "Scenario output directory: %s (scenario_id=%s, overwrite=%s)",
        scenario_output_dir,
        scenario_id,
        overwrite,
    )
    return scenario_output_dir


def coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _camera_defaults(carla_cfg: Dict[str, Any]) -> Dict[str, float]:
    return {
        "width": float(carla_cfg.get("camera_width", 800)),
        "height": float(carla_cfg.get("camera_height", 600)),
        "fov": float(carla_cfg.get("camera_fov", 100.0)),
    }


def _build_camera_spec(cam_id: str, carla_cfg: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    defaults = _camera_defaults(carla_cfg)
    mount = dict(CAMERA_MOUNTS.get(cam_id, CAMERA_MOUNTS["CAM_FRONT"]))
    if overrides:
        mount.update({k: overrides[k] for k in overrides if overrides[k] is not None})
    return {
        "id": cam_id,
        "type": "sensor.camera.rgb",
        "x": float(mount.get("x", 0.7)),
        "y": float(mount.get("y", 0.0)),
        "z": float(mount.get("z", 1.6)),
        "roll": float(mount.get("roll", 0.0)),
        "pitch": float(mount.get("pitch", 0.0)),
        "yaw": float(mount.get("yaw", 0.0)),
        "width": int(overrides.get("width", defaults["width"]) if overrides else defaults["width"]),
        "height": int(overrides.get("height", defaults["height"]) if overrides else defaults["height"]),
        "fov": float(overrides.get("fov", defaults["fov"]) if overrides else defaults["fov"]),
    }


def resolve_camera_specs(
    carla_cfg: Dict[str, Any],
    *,
    attach_legacy: bool,
    planner_type: str,
    vlm_enabled: bool,
    perception_source: str = "ground_truth",
) -> List[Dict[str, Any]]:
    """Resolve which CARLA RGB cameras to attach from config."""
    explicit = carla_cfg.get("cameras")
    if planner_type == "sparsedrivev2":
        rig_name = str(carla_cfg.get("camera_rig", "full_6cam")).strip()
        if explicit:
            raise ValueError(
                "SparseDriveV2 uses the fixed Bench2Drive CARLA camera rig; "
                "remove carla.cameras overrides"
            )
        if rig_name != "full_6cam":
            raise ValueError(
                "SparseDriveV2 requires carla.camera_rig=full_6cam with the "
                "official Bench2Drive calibration"
            )
        return [
            {
                "id": cam_id,
                "type": "sensor.camera.rgb",
                **dict(BENCH2DRIVE_CAMERA_SPECS[cam_id]),
            }
            for cam_id in SPARSEDRIVEV2_CAMERAS
        ]
    raise ValueError("This release requires the SparseDriveV2 camera rig")


def rgb_from_input_data(input_data: Dict[str, Any], cam_id: str) -> Optional[np.ndarray]:
    if cam_id not in input_data:
        return None
    raw = np.array(input_data[cam_id][1])
    if raw.ndim != 3:
        return None
    if raw.shape[2] == 4:
        return cv2.cvtColor(raw, cv2.COLOR_BGRA2RGB)
    if raw.shape[2] == 3:
        return cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    return raw


def _resize_for_grid(image: np.ndarray, target_height: int) -> np.ndarray:
    if image.shape[0] == target_height:
        return image
    width = max(1, int(round(image.shape[1] * (target_height / image.shape[0]))))
    return cv2.resize(image, (width, target_height), interpolation=cv2.INTER_LINEAR)


def compose_grid_frame(obs_rgb: Dict[str, np.ndarray], layout: List[List[str]]) -> Optional[np.ndarray]:
    present = {name for name in obs_rgb if obs_rgb[name] is not None}
    needed = [name for row in layout for name in row if name in present]
    if not needed:
        return None
    target_height = max(obs_rgb[name].shape[0] for name in needed)
    rows = []
    for row_names in layout:
        tiles = []
        for name in row_names:
            if name not in present:
                continue
            tiles.append(_resize_for_grid(obs_rgb[name], target_height))
        if tiles:
            rows.append(np.concatenate(tiles, axis=1))
    if not rows:
        return None
    target_width = max(row.shape[1] for row in rows)
    padded_rows = []
    for row in rows:
        if row.shape[1] < target_width:
            pad = target_width - row.shape[1]
            row = np.pad(row, ((0, 0), (0, pad), (0, 0)), mode="constant")
        padded_rows.append(row)
    return np.concatenate(padded_rows, axis=0)


def _caption_view(image_rgb: np.ndarray, text: str) -> np.ndarray:
    """Stamp a small view label (e.g. 'FRONT' / 'BEV') in the bottom-left."""
    img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    height = img.shape[0]
    cv2.putText(
        img, text, (10, height - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def compose_front_topdown_frame(
    front_rgb: Optional[np.ndarray], topdown_rgb: np.ndarray
) -> Optional[np.ndarray]:
    """Side-by-side dual view: front camera (left) + top-down overlay (right).

    Both panes are resized to a common height and concatenated horizontally.
    Returns ``None`` when the front image is unavailable so the caller can skip.
    """
    if front_rgb is None or topdown_rgb is None:
        return None
    target_height = int(topdown_rgb.shape[0])
    left = _resize_for_grid(_caption_view(front_rgb, "FRONT"), target_height)
    right = _resize_for_grid(_caption_view(topdown_rgb, "BEV"), target_height)
    return np.concatenate([left, right], axis=1)





_PLAN2CONTROL = None







def _coerce_vlm_value(value: Any, default: Any) -> Any:
    if isinstance(default, bool):
        return coerce_bool(value, default=default)
    if isinstance(default, int) and not isinstance(default, bool):
        return int(value)
    if isinstance(default, float):
        return float(value)
    if isinstance(default, (tuple, list)):
        # yaml lists reach dataclass tuple fields as real sequences; a bare
        # string is treated as comma-separated ids.
        if isinstance(value, (list, tuple, set)):
            items = tuple(value)
        else:
            items = tuple(
                s.strip() for s in str(value).split(",") if s.strip()
            )
        exemplar = default[0] if default else None
        if isinstance(exemplar, bool):
            return tuple(coerce_bool(item, default=exemplar) for item in items)
        if isinstance(exemplar, int) and not isinstance(exemplar, bool):
            return tuple(int(item) for item in items)
        if isinstance(exemplar, float):
            return tuple(float(item) for item in items)
        return tuple(str(item) for item in items)
    if default is None:
        # Optional fields (e.g. Optional[float]) keep their yaml-typed value;
        # str() here would silently stringify numbers.
        return value
    return str(value)


def build_vlm_selector_config_from_dict(
    vlm_dict: Optional[Dict[str, Any]],
    *,
    enabled: Optional[bool] = None,
    python_bin: str = "",
) -> "VLMSelectorConfig":
    from agenticdriving.scorer.selector_config import VLMSelectorConfig

    raw = dict(vlm_dict or {})
    fields_by_name = {
        field.name: field
        for field in VLMSelectorConfig.__dataclass_fields__.values()
    }
    kwargs: Dict[str, Any] = {}
    for name, field in fields_by_name.items():
        if name not in raw:
            continue
        kwargs[name] = _coerce_vlm_value(raw[name], field.default)
    if python_bin and not kwargs.get("python_bin"):
        kwargs["python_bin"] = str(python_bin)
    if enabled is not None:
        kwargs["enabled"] = bool(enabled)
    return VLMSelectorConfig(**kwargs)


def apply_agenticdriving_env(agenticdriving_dict: Optional[Dict[str, Any]]) -> None:
    from agenticdriving.config import AGENTICDRIVING_ENV_FIELD_NAMES

    cfg = dict(agenticdriving_dict or {})
    if "selector" not in cfg and "selector_type" in cfg:
        cfg["selector"] = cfg["selector_type"]
    for suffix, field_name in AGENTICDRIVING_ENV_FIELD_NAMES.items():
        if field_name not in cfg or cfg[field_name] is None:
            continue
        os.environ[f"AGENTICDRIVING_{suffix}"] = str(cfg[field_name])


def get_agenticdriving_recovery_cfg(agent_cfg: Dict[str, Any]) -> Dict[str, Any]:
    agenticdriving_cfg = dict((agent_cfg or {}).get("agenticdriving") or {})
    recovery_cfg = agenticdriving_cfg.get("recovery")
    if isinstance(recovery_cfg, dict):
        return dict(recovery_cfg)
    return {}


def init_selection_state(agent) -> None:
    agent._previous_selected_plan = None
    agent._previous_selected_pose = None
    agent._previous_selected_score = None
    agent._previous_selected_timestamp = None
    agent._previous_selected_source = None
    agent._last_applied_control = None
    agent._last_ackermann_control = None
    agent._previous_pedals = None
    agent._ackermann_gear_initialized = False
    agent._max_steer_rad = None

def _control_cfg(agent) -> Dict[str, Any]:
    """Control settings under ``carla.control`` (fallback: legacy ``carla.longitudinal``)."""
    carla_cfg = getattr(agent, "_carla_cfg", {}) or {}
    control = dict(carla_cfg.get("control", {}) or {})
    if not control:
        legacy = dict(carla_cfg.get("longitudinal", {}) or {})
        if legacy:
            mode = str(legacy.get("mode", "vehicle")).lower()
            if mode in ("pid", "raw", "gain"):
                mode = "vehicle"
            control = {**legacy, "mode": mode}
    return control


def _control_mode(agent) -> str:
    return str(_control_cfg(agent).get("mode", "vehicle")).lower()


def _clear_ackermann_control(agent) -> None:
    agent._last_ackermann_control = None


def _get_max_steer_rad(agent, hero: carla.Vehicle) -> float:
    cached = getattr(agent, "_max_steer_rad", None)
    if cached is not None:
        return float(cached)
    try:
        wheels = hero.get_physics_control().wheels
        max_deg = max(float(w.max_steer_angle) for w in wheels) if wheels else 70.0
    except Exception:
        max_deg = 70.0
    agent._max_steer_rad = math.radians(max_deg)
    return float(agent._max_steer_rad)




def capture_previous_pedals(agent, hero: carla.Vehicle) -> None:
    """Store pedals CARLA applied on the previous tick (for diagnostics)."""
    try:
        prev = hero.get_control()
        agent._previous_pedals = {
            "throttle": float(prev.throttle),
            "brake": float(prev.brake),
            "steer": float(prev.steer),
        }
    except Exception:
        agent._previous_pedals = None


def apply_recovery_env(
    recovery_dict: Optional[Dict[str, Any]],
    python_bin: str,
    prefixes: tuple[str, ...],
) -> None:
    from agenticdriving.planners.fallback_rule_based.recovery_env import build_prefixed_rule_based_env

    env_values = build_prefixed_rule_based_env(
        recovery_dict or {},
        planner_python_bin=str(python_bin),
        prefixes=prefixes,
    )
    for key, value in env_values.items():
        os.environ[str(key)] = str(value)





def setup_vlm_selector(
    agent,
    vlm_dict: Optional[Dict[str, Any]] = None,
    *,
    planner_cfg: Optional[Dict[str, Any]] = None,
) -> None:
    from agenticdriving.config import resolve_agenticdriving_config
    from agenticdriving.scorer.dummy_selector import DummyPlanSelector

    planner_cfg = dict(planner_cfg or {})
    agent_cfg = getattr(agent, "_cfg", None) or {}
    agenticdriving_dict = dict(agent_cfg.get("agenticdriving") or {})
    apply_agenticdriving_env(agenticdriving_dict)
    ad_cfg = resolve_agenticdriving_config()

    merged_vlm_dict: Dict[str, Any] = {}
    for cfg in (
        agent_cfg.get("vlm"),
        planner_cfg.get("vlm"),
        agenticdriving_dict.get("vlm"),
        vlm_dict,
    ):
        if isinstance(cfg, dict):
            merged_vlm_dict.update(cfg)

    selector_name = str(ad_cfg.selector).strip().lower()
    vlm_enabled = bool(ad_cfg.enabled and selector_name == "vlm")
    agent._vlm_selector_cfg = build_vlm_selector_config_from_dict(
        merged_vlm_dict,
        enabled=vlm_enabled,
        python_bin=str(planner_cfg.get("python_bin", sys.executable)),
    )
    if vlm_enabled:
        raise ValueError("Only the dedicated Claude recovery selector is supported")
    agent._vlm_selector = DummyPlanSelector()
    LOG.info(
        "AgenticDriving recovery=%s selector=dummy (VLM disabled)",
        bool(ad_cfg.enabled),
    )


def build_recovery_reason_config_from_dict(
    recovery_reason_dict: Optional[Dict[str, Any]],
    *,
    python_bin: str = "",
) -> "RecoveryReasonConfig":
    from agenticdriving.vlm.recovery_reason import RecoveryReasonConfig

    raw = dict(recovery_reason_dict or {})
    fields_by_name = {
        field.name: field
        for field in RecoveryReasonConfig.__dataclass_fields__.values()
    }
    kwargs: Dict[str, Any] = {}
    for name, field in fields_by_name.items():
        if name not in raw:
            continue
        kwargs[name] = _coerce_vlm_value(raw[name], field.default)
    if python_bin and not kwargs.get("python_bin"):
        kwargs["python_bin"] = str(python_bin)
    return RecoveryReasonConfig(**kwargs)


def build_semantic_takeover_config_from_dict(
    raw: Optional[Dict[str, Any]],
):
    """Build the conservative asynchronous A/E semantic-takeover config."""
    from dataclasses import fields

    from agenticdriving.vlm.semantic_takeover import SemanticTakeoverConfig

    raw = dict(raw or {})
    kwargs: Dict[str, Any] = {}
    for field in fields(SemanticTakeoverConfig):
        if field.name in raw:
            kwargs[field.name] = _coerce_vlm_value(raw[field.name], field.default)
    return SemanticTakeoverConfig(**kwargs)


def build_stop_sign_config_from_dict(raw: Optional[Dict[str, Any]]):
    from dataclasses import fields

    from agenticdriving.vlm.stop_sign import StopSignConfig

    raw = dict(raw or {})
    kwargs: Dict[str, Any] = {}
    for field in fields(StopSignConfig):
        if field.name in raw:
            kwargs[field.name] = _coerce_vlm_value(raw[field.name], field.default)
    return StopSignConfig(**kwargs)


def setup_recovery_explainer(
    agent,
    recovery_reason_dict: Optional[Dict[str, Any]] = None,
    *,
    planner_cfg: Optional[Dict[str, Any]] = None,
) -> None:
    """Wire the standalone "explain why recovery triggered" VLM call.

    Independent of ``agenticdriving.selector`` (dummy vs vlm): this only
    explains the verifier's decision at the instant recovery triggers, so it
    is useful even when the main selector stays on plain argmax/dummy. Off by
    default; enable via the yaml ``recovery_reason:`` block (``enabled: true``).
    """
    agent._recovery_explainer = None
    agent._semantic_takeover_manager = None
    agent._stop_sign_manager = None
    planner_cfg = dict(planner_cfg or {})
    agent_cfg = getattr(agent, "_cfg", None) or {}
    agenticdriving_dict = dict(agent_cfg.get("agenticdriving") or {})

    merged_dict: Dict[str, Any] = {}
    for cfg in (
        agent_cfg.get("recovery_reason"),
        planner_cfg.get("recovery_reason"),
        agenticdriving_dict.get("recovery_reason"),
        # Preferred name for the recovery ablations: the VLM directly selects
        # a primitive and does not need to summarize a reason first.
        agenticdriving_dict.get("recovery_selector"),
        recovery_reason_dict,
    ):
        if isinstance(cfg, dict):
            merged_dict.update(cfg)

    cfg = build_recovery_reason_config_from_dict(
        merged_dict,
        python_bin=str(planner_cfg.get("python_bin", sys.executable)),
    )
    semantic_cfg = build_semantic_takeover_config_from_dict(
        agenticdriving_dict.get("semantic_takeover")
    )
    stop_sign_cfg = build_stop_sign_config_from_dict(
        agenticdriving_dict.get("stop_sign")
    )
    disagreement_enabled = bool(
        ((agent_cfg.get("perception") or {}).get("disagreement_vlm") or {}).get("enabled", False)
    )
    # Either capability (explain the trigger, and/or pick the recovery primitive)
    # or the sparse semantic gate turns the shared VLM on.
    if not (
        cfg.enabled
        or cfg.select_primitive
        or cfg.select_sequence
        or cfg.select_recovery_plan
        or semantic_cfg.enabled
        or stop_sign_cfg.enabled
        or disagreement_enabled
    ):
        LOG.info("Recovery-reason, semantic-takeover, and stop-sign VLM disabled")
        return

    from agenticdriving.vlm.recovery_reason import RecoveryReasonExplainer
    from agenticdriving.vlm.semantic_takeover import AsyncSemanticTakeoverManager
    from agenticdriving.vlm.stop_sign import AsyncStopSignManager

    agent._recovery_explainer = RecoveryReasonExplainer(
        cfg,
        agent._output_dir,
        extra_active=semantic_cfg.enabled or stop_sign_cfg.enabled or disagreement_enabled,
    )
    agent._recovery_explainer.preload()
    if semantic_cfg.enabled:
        agent._semantic_takeover_manager = AsyncSemanticTakeoverManager(
            semantic_cfg,
            checker=agent._recovery_explainer,
            output_dir=agent._output_dir,
            logger=LOG,
        )
    if stop_sign_cfg.enabled:
        agent._stop_sign_manager = AsyncStopSignManager(
            stop_sign_cfg,
            checker=agent._recovery_explainer,
            output_dir=agent._output_dir,
            logger=LOG,
        )
    LOG.info(
        "Shared recovery VLM enabled (explain=%s select_primitive=%s semantic_takeover=%s "
        "stop_sign=%s) "
        "backend=%s model=%s",
        cfg.enabled, cfg.select_primitive, semantic_cfg.enabled, stop_sign_cfg.enabled,
        cfg.backend, cfg.model_id,
    )
    if cfg.tool_rule_blocked_escape_enabled:
        LOG.info(
            "Persistent-blockage tool rule ON: after %.1fs choose first "
            "verified tool from %s",
            cfg.tool_rule_blocked_escape_after_sec,
            tuple(cfg.tool_rule_blocked_escape_primitives),
        )
    if semantic_cfg.enabled:
        LOG.info(
            "Semantic takeover A/E gate ON: min_check=%.1fs confidence>=%.2f "
            "confirm=%s takeover_cooldown=%.1fs max_per_episode=%d",
            semantic_cfg.min_check_interval_sec,
            semantic_cfg.confidence_threshold,
            semantic_cfg.require_confirmation,
            semantic_cfg.semantic_takeover_cooldown_sec,
            semantic_cfg.max_semantic_takeovers_per_episode,
        )
    if stop_sign_cfg.enabled:
        LOG.info(
            "Stop-sign VLM gate ON: lookahead=%.1fm checks=%s confidence>=%.2f "
            "negative_cap=%.1fm/s@%.1fm hold=%.2fs stop_speed<%.2fm/s",
            stop_sign_cfg.lookahead_m,
            tuple(float(item) for item in stop_sign_cfg.check_distances_m),
            stop_sign_cfg.confidence_threshold,
            stop_sign_cfg.negative_speed_cap_mps,
            stop_sign_cfg.negative_speed_cap_start_m,
            stop_sign_cfg.hold_sec,
            stop_sign_cfg.stop_speed_threshold_mps,
        )


def _plan_step1_speed(plan: np.ndarray, plan_dt: float) -> float:
    """Speed implied by the first plan waypoint (m/s), for logging only."""
    plan = np.asarray(plan, dtype=np.float32)
    if plan.ndim != 2 or len(plan) == 0 or plan_dt <= 0.0:
        return 0.0
    return float(np.linalg.norm(plan[0, :2])) / float(plan_dt)




def _cfg_float(cfg: Dict[str, Any], key: str, default: float) -> float:
    try:
        return float(cfg.get(key, default))
    except (TypeError, ValueError):
        return float(default)










def _record_applied_control(
    agent,
    info: Dict[str, Any],
    *,
    control_mode: str,
    acc_cmd: Optional[float] = None,
    plan_step1_speed: Optional[float] = None,
    vehicle_ctrl: Optional[carla.VehicleControl] = None,
    ack_ctrl: Optional[carla.VehicleAckermannControl] = None,
    steer_norm: Optional[float] = None,
) -> None:
    """Stash commanded control + previous-tick pedals for JSON diagnostics."""
    prev = getattr(agent, "_previous_pedals", None) or {}
    payload: Dict[str, Any] = {
        "control_mode": control_mode,
        "acc_cmd": float(acc_cmd) if acc_cmd is not None else None,
        "plan_step1_speed": (
            float(plan_step1_speed) if plan_step1_speed is not None else None
        ),
        "ego_velo": float(info.get("ego_velo", 0.0)),
        "applied_throttle": prev.get("throttle"),
        "applied_brake": prev.get("brake"),
        "applied_steer": prev.get("steer"),
    }
    if vehicle_ctrl is not None:
        payload.update({
            "throttle": float(vehicle_ctrl.throttle),
            "brake": float(vehicle_ctrl.brake),
            "steer": float(vehicle_ctrl.steer),
        })
    if ack_ctrl is not None:
        payload.update({
            "ack_speed": float(ack_ctrl.speed),
            "ack_acceleration": float(ack_ctrl.acceleration),
            "ack_steer_rad": float(ack_ctrl.steer),
            "ack_steer_speed": float(ack_ctrl.steer_speed),
            "ack_jerk": float(ack_ctrl.jerk),
            "steer": float(steer_norm) if steer_norm is not None else 0.0,
            "throttle": None,
            "brake": None,
        })
    agent._last_applied_control = payload


def agent_brake_control(agent) -> carla.VehicleControl:
    _clear_ackermann_control(agent)
    return brake_control()


def update_stop_sign_control(agent, ego_state: Dict[str, Any]) -> bool:
    """Advance the HD-map/VLM stop-sign gate for the current simulator tick."""
    manager = getattr(agent, "_stop_sign_manager", None)
    if manager is None or not manager.enabled:
        return False

    info = ego_state["info"]
    junction = None
    carla_map = CarlaDataProvider.get_map()
    if carla_map is not None:
        ego_pos = np.asarray(info.get("ego_pos"), dtype=np.float64).reshape(-1)
        if ego_pos.size >= 2:
            location = carla.Location(
                x=float(ego_pos[0]),
                y=float(ego_pos[1]),
                z=float(ego_pos[2]) if ego_pos.size >= 3 else 0.0,
            )
            ego_waypoint, _ = driving_waypoint_at(
                carla_map,
                location,
                driving_lane_type=carla.LaneType.Driving,
            )
            from agenticdriving.vlm.stop_sign import find_upcoming_junction

            junction = find_upcoming_junction(
                ego_waypoint,
                lookahead_m=float(manager.cfg.lookahead_m),
                waypoint_step_m=float(manager.cfg.waypoint_step_m),
            )

    if junction is not None:
        info["stop_sign_junction"] = {
            "key": junction.key,
            "distance_m": float(junction.distance_m),
            "is_current": bool(junction.is_current),
        }
    else:
        info["stop_sign_junction"] = None

    should_brake = manager.step(
        frame_index=int(getattr(agent, "_frame_index", 0)),
        obs=ego_state["obs"],
        info=info,
        speed_mps=float(ego_state.get("speed_mps", info.get("ego_velo", 0.0)) or 0.0),
        junction=junction,
    )
    info["stop_sign_control"] = {
        "state": manager.state,
        "braking": bool(should_brake),
        "last_result": manager.last_result,
    }
    return bool(should_brake)


def stop_sign_brake_control(agent, info: Dict[str, Any]) -> carla.VehicleControl:
    """Return the final hard-brake override and keep control diagnostics honest."""
    ctrl = agent_brake_control(agent)
    _record_applied_control(
        agent,
        info,
        control_mode="stop_sign_hard_brake",
        vehicle_ctrl=ctrl,
    )
    if isinstance(getattr(agent, "_last_applied_control", None), dict):
        manager = getattr(agent, "_stop_sign_manager", None)
        agent._last_applied_control.update({
            "stop_sign_override": True,
            "stop_sign_state": getattr(manager, "state", None),
        })
    return ctrl


def apply_stop_sign_speed_cap(
    agent,
    info: Dict[str, Any],
    ctrl,
):
    """Limit a negative/pending Stop approach without forcing a full stop."""
    manager = getattr(agent, "_stop_sign_manager", None)
    cap = None if manager is None else manager.speed_cap_mps
    if cap is None or not hasattr(ctrl, "throttle") or not hasattr(ctrl, "brake"):
        return ctrl

    speed_mps = float(info.get("ego_velo", 0.0) or 0.0)
    cap = max(0.0, float(cap))
    overspeed = speed_mps - cap
    if overspeed > 0.0:
        ctrl.throttle = 0.0
        ctrl.brake = max(
            float(ctrl.brake),
            float(np.clip(0.15 + overspeed / 3.0, 0.15, 1.0)),
        )
    elif speed_mps >= max(0.0, cap - 0.5):
        ctrl.throttle = 0.0
    else:
        # Preserve useful approach speed below the cap. A 0.35 clamp was too
        # weak against rolling resistance and pulled field runs down to roughly
        # 3.5--4 m/s; 0.60 still suppresses a full-throttle pulse while allowing
        # the nominal controller to settle near the intended 6 m/s cap.
        ctrl.throttle = min(float(ctrl.throttle), 0.60)

    if isinstance(getattr(agent, "_last_applied_control", None), dict):
        agent._last_applied_control.update({
            "throttle": float(ctrl.throttle),
            "brake": float(ctrl.brake),
            "steer": float(ctrl.steer),
            "stop_sign_speed_cap_applied": True,
            "stop_sign_speed_cap_mps": cap,
        })
    return ctrl




def apply_plan_control(
    agent,
    selected_plan: Optional[np.ndarray],
    info: Dict[str, Any],
    selected_score_raw: Optional[float],
    selected_source: str,
) -> carla.VehicleControl:
    if selected_plan is None or len(selected_plan) == 0:
        fallback = agent_brake_control(agent)
        _record_applied_control(agent, info, control_mode="vehicle", vehicle_ctrl=fallback)
        return fallback

    agent._previous_selected_plan = np.asarray(
        selected_plan, dtype=np.float32
    ).copy()
    agent._previous_selected_pose = agent._info_to_pose(info)
    agent._previous_selected_score = selected_score_raw
    agent._previous_selected_timestamp = float(info.get("timestamp", 0.0))
    agent._previous_selected_source = selected_source

    # Plan -> control law selected by carla.control.mode (fail2drive_pid /
    # our_pid / ackermann). Imported lazily so the control package can import
    # this helper module at its own import time without a cycle.
    import control

    return control.dispatch(
        agent, selected_plan, info, selected_score_raw, selected_source
    )


def resolve_recovery(
    planner_cfg: Dict[str, Any], prefixes: tuple[str, ...]
):
    from agenticdriving.planners.fallback_rule_based.recovery_config import (
        resolve_recovery_config,
    )

    return resolve_recovery_config(
        planner_python_bin=str(
            planner_cfg.get("python_bin", sys.executable)
        ),
        prefixes=prefixes,
    )

def create_learned_planner_selector(
    agent,
    *,
    recovery_cfg: Any,
    current_source_name: str,
    learned_default_source: str,
    plain_source: str,
    score_fallback_key: str,
    planner_log_name: str,
    strict_learned_argmax_lookup: bool,
    q_key_prefix: bool,
) -> Any:
    from agenticdriving.config import resolve_agenticdriving_config
    from agenticdriving.scorer.planner_selection import LearnedPlannerSelector

    return LearnedPlannerSelector(
        vlm_selector=agent._vlm_selector,
        runtime_name=planner_log_name.lower(),
        agenticdriving_cfg=resolve_agenticdriving_config(),
        vlm_cfg=agent._vlm_selector_cfg,
        recovery_cfg=recovery_cfg,
        current_source_name=current_source_name,
        learned_default_source=learned_default_source,
        plain_source=plain_source,
        score_fallback_key=score_fallback_key,
        planner_log_name=planner_log_name,
        strict_learned_argmax_lookup=strict_learned_argmax_lookup,
        q_key_prefix=q_key_prefix,
        logger=LOG,
    )


def build_lane_tier_probe():
    """(N, 3) world points -> float[N] DAC tiers (PDMS DAC term).

    Tiered drivable-area membership (NAVSIM's tiered-multiplier pattern — NC
    uses 0/0.5/1 there — applied to DAC):

      1.0  point is inside the Driving/Parking corridor per Fail2Drive's
           outside-lane rule (exact port of
           OutsideRouteLanesTest._is_outside_driving_lanes,
           atomic_criteria.py:1107-1127: distance to the nearest
           Driving/Parking lane center <= lane_width/2 + 0.5m,
           atomic_criteria.py:996). NOTE: nearest-center over ALL driving
           lanes — adjacent/opposite-lane use is tier 1.0 by design.
      0.5  outside the corridor but still on a vehicle-plausible road
           surface: exact containment (project_to_road=False) on a
           Shoulder / Median / Border / Bidirectional / Stop / Biking lane.
           Tactically tolerable (e.g. skirting a parked blocker), penalised.
      0.0  everything else — off the mapped road, or Sidewalk/Rail/
           Restricted (pedestrian space is deliberately NOT tier 0.5).

    Fails open: without a map every point is tier 1.0.

    The probe also exposes ``probe.details(points)`` for the PDMS LK term:
    per-sample planar deviation from the nearest Driving-lane centerline
    (``get_waypoint`` snaps the point onto the lane center, so the distance
    to the snapped location IS the lateral deviation — Hydra-MDP++'s
    min-over-nearby-lane-segments d_i, with the candidate set instantiated
    as CARLA's nearest driving lane), plus ``lk_skip`` marking junction
    samples (NavSim v2 disables LK on intersections; ``is_junction`` of the
    snapped waypoint is the CARLA analog) and samples with no driving lane.
    Both outputs come from the same waypoint queries — no extra map calls.
    """
    allowed_out_distance = 0.5  # OutsideRouteLanesTest.ALLOWED_OUT_DISTANCE
    surface_lane_types = (
        carla.LaneType.Shoulder | carla.LaneType.Median | carla.LaneType.Border
        | carla.LaneType.Bidirectional | carla.LaneType.Stop | carla.LaneType.Biking
    )

    def details(points_world: np.ndarray) -> Dict[str, np.ndarray]:
        carla_map = CarlaDataProvider.get_map()
        points_world = np.asarray(points_world, dtype=np.float64)
        n = len(points_world)
        tiers = np.ones(n, dtype=np.float64)
        lat_dev_m = np.zeros(n, dtype=np.float64)
        lane_width_m = np.full(n, np.nan, dtype=np.float64)
        lk_skip = np.zeros(n, dtype=bool)
        # World xy of the nearest Driving-lane centerline snap (NaN when no
        # driving lane / no map); consumed by the recovery road-context prompt.
        lane_center_xy = np.full((n, 2), np.nan, dtype=np.float64)
        if carla_map is None:
            lk_skip[:] = True
            return {"tiers": tiers, "lat_dev_m": lat_dev_m,
                    "lane_width_m": lane_width_m, "lk_skip": lk_skip,
                    "lane_center_xy": lane_center_xy}
        for i, point in enumerate(points_world):
            loc = carla.Location(
                x=float(point[0]), y=float(point[1]),
                z=float(point[2]) if len(point) > 2 else 0.0,
            )
            driving_wp = carla_map.get_waypoint(loc, lane_type=carla.LaneType.Driving)
            parking_wp = carla_map.get_waypoint(loc, lane_type=carla.LaneType.Parking)

            # LK inputs. Planar (xy) distance, not loc.distance: the 3D form
            # the DAC tier uses would inflate deviation with the z gap on
            # elevated/banked roads, and LK's 0.5 m limit is a BEV quantity.
            if driving_wp is None:
                lk_skip[i] = True
            else:
                wloc = driving_wp.transform.location
                lat_dev_m[i] = math.hypot(loc.x - wloc.x, loc.y - wloc.y)
                lane_width_m[i] = float(driving_wp.lane_width)
                lane_center_xy[i] = (float(wloc.x), float(wloc.y))
                if driving_wp.is_junction:
                    lk_skip[i] = True

            driving_distance = (
                loc.distance(driving_wp.transform.location)
                if driving_wp is not None else float("inf")
            )
            parking_distance = (
                loc.distance(parking_wp.transform.location)
                if parking_wp is not None else float("inf")
            )

            if driving_distance >= parking_distance:
                distance, lane_width = parking_distance, parking_wp.lane_width
            else:
                distance, lane_width = driving_distance, driving_wp.lane_width

            if distance <= (lane_width / 2 + allowed_out_distance):
                tiers[i] = 1.0
                continue

            surface_wp = carla_map.get_waypoint(
                loc, project_to_road=False, lane_type=surface_lane_types
            )
            tiers[i] = 0.5 if surface_wp is not None else 0.0
        return {"tiers": tiers, "lat_dev_m": lat_dev_m,
                "lane_width_m": lane_width_m, "lk_skip": lk_skip,
                "lane_center_xy": lane_center_xy}

    def probe(points_world: np.ndarray) -> np.ndarray:
        return details(points_world)["tiers"]

    probe.details = details
    return probe


def build_ground_probe():
    """(N, 3) world points -> bool[N]: True when physical ground exists below.

    Consumed by the orchestrator's recovery footprint check to tell
    unmapped-but-drivable ground (construction dirt, tier 0.0 in the lane
    probe yet perfectly flat) from actual VOID (unguarded elevated-road edges,
    where tier 0.0 means a 25 km drop). Casts CARLA's gravity-direction
    ground_projection from 2 m above each point; no hit within the search
    depth = nothing to drive on. Fails open (True) without a world."""
    search_depth_m = 12.0

    def probe(points_world: np.ndarray) -> np.ndarray:
        world = CarlaDataProvider.get_world()
        pts = np.asarray(points_world, dtype=np.float64)
        if world is None:
            return np.ones(len(pts), dtype=bool)
        out = []
        for p in pts:
            z = float(p[2]) if len(p) > 2 else 0.0
            loc = carla.Location(x=float(p[0]), y=float(p[1]), z=z + 2.0)
            try:
                hit = world.ground_projection(loc, search_depth_m)
            except Exception:
                out.append(True)
                continue
            out.append(hit is not None)
        return np.asarray(out, dtype=bool)

    return probe


def build_ep_reference_factory(agent, verifier_cfg):
    """Lazily build the PDMS EP route reference from the leaderboard route.

    Deferred to first verify because ``set_global_plan`` may run after setup.
    The expected pace is ``ep_expected_speed_mps`` from the yaml ``verifier:``
    section (CARLA routes carry no human timing to infer it from).
    """
    pace = float(verifier_cfg.get("ep_expected_speed_mps", 5.0))

    def factory():
        from agenticdriving.verifiers import EgoProgressReference

        plan = getattr(agent, "_global_plan_world_coord", None)
        if not plan:
            LOG.warning("PDMS EP: no global plan available; EP term disabled")
            return None
        route_xy = []
        for entry, _option in plan:
            loc = getattr(entry, "location", entry)
            route_xy.append((float(loc.x), float(loc.y)))
        if len(route_xy) < 2:
            return None
        return EgoProgressReference(route_xy, pace_speed_mps=pace)

    return factory


def resolve_end2end_topk(scorer_sel: Dict[str, Any]) -> int:
    """Top-K of the End2End->PDMS branch, from ``scorer.end2end_topk``.

    Default 30 (the historical hard-coded value; absent/zero/None yield it
    unchanged). Overridable so planners with small candidate pools (e.g.
    bridgedrive's 60) can PDMS-score every proposal.
    """
    return max(1, int(scorer_sel.get("end2end_topk", 30) or 30))


def build_orchestrator(agent, *, planner_service, recovery_cfg, orchestrator_cls=None):
    """Wire planner + selector + verifier into the per-frame Orchestrator.

    Verifier type/threshold come from the ``verifier:`` section of the agent
    config yaml (``type``: none|dummy|ttc|pdms|geometric); the
    AGENTICDRIVING_VERIFIER_TYPE / _VERIFIER_THRESHOLD env vars are the
    fallback when the yaml omits it. Default: no verifier, which keeps the
    pre-verifier behaviour bit-for-bit.
    """
    from agenticdriving.config import resolve_agenticdriving_config
    from agenticdriving.orchestrator import Orchestrator
    from agenticdriving.scorer.scoring import (
        InstructionComplianceScorer,
        PlannerNativeScorer,
        RuleBasedPDMSScorer,
    )
    from agenticdriving.verifiers import create_verifier, resolve_verifier_threshold

    ad_cfg = resolve_agenticdriving_config()
    agent_cfg = getattr(agent, "_cfg", None) or {}
    verifier_cfg = dict(agent_cfg.get("verifier") or {})
    # DAC/LK always take the HD map (map context is
    # permitted ground truth; the sensed ROAD-mask DAC experiment was
    # deleted).
    lane_tier_probe = build_lane_tier_probe()
    verifier = create_verifier(
        ad_cfg,
        plan_dt_sec=float(getattr(agent, "_plan_dt_sec", 0.5)),
        verifier_cfg=verifier_cfg,
        lane_tier_probe=lane_tier_probe,
        ep_reference_factory=build_ep_reference_factory(agent, verifier_cfg),
    )
    if verifier is not None:
        # Retained for diagnostics/hold compatibility. The shared hard HD-map
        # gate no longer permits unmapped ground even when this raycast finds a
        # physical surface beneath it.
        verifier._ground_probe = build_ground_probe()
    verifier_threshold = resolve_verifier_threshold(ad_cfg, verifier_cfg)
    control_cfg = _control_cfg(agent)
    pedestrian_ttc_brake_hold_sec = float(
        control_cfg.get("pedestrian_ttc_brake_hold_sec", 1.0)
    )
    pedestrian_ttc_brake_distance_m = float(
        control_cfg.get("pedestrian_caution_distance_m", 30.0)
    )
    pedestrian_ttc_brake_lateral_m = float(
        control_cfg.get("pedestrian_caution_lateral_m", 6.0)
    )
    if verifier is not None:
        LOG.info(
            "Trajectory verifier enabled: type=%s threshold=%.2f (source=%s) agent_forecast=%s",
            type(verifier).__name__,
            verifier_threshold,
            "yaml" if verifier_cfg.get("type") is not None else "env",
            verifier_cfg.get("agent_forecast_mode", "cv"),
        )
        LOG.info(
            "Pedestrian TTC hard brake: hold=%.2fs range=%.1fm/±%.1fm "
            "(VRU progress retiming disabled)",
            pedestrian_ttc_brake_hold_sec,
            pedestrian_ttc_brake_distance_m,
            pedestrian_ttc_brake_lateral_m,
        )
    instruction_scorer = (
        InstructionComplianceScorer() if ad_cfg.instruction_scorer_enabled else None
    )

    # Ranking scorers: yaml ``scorer:`` block selects which candidate scorers
    # feed the argmax. Both default on; they must not both be off (with neither
    # there is nothing to rank by). When only one is on, only that one scores.
    #   scorer:
    #     rule_based: true   # PDMS rule-based per-candidate scorer
    #     end2end:    true   # planner-native end2end scorer
    scorer_sel = dict(agent_cfg.get("scorer") or {})
    use_rule_based = coerce_bool(scorer_sel.get("rule_based", True), default=True)
    use_end2end = coerce_bool(scorer_sel.get("end2end", True), default=True)
    branch_keys_present = any(
        key in scorer_sel
        for key in ("end2end_pdms_branch", "target_region_pdms_branch")
    )
    if branch_keys_present:
        end2end_pdms_branch = coerce_bool(
            scorer_sel.get("end2end_pdms_branch", False)
        )
        target_region_pdms_branch = coerce_bool(
            scorer_sel.get("target_region_pdms_branch", False)
        )
    else:
        # Preserve the legacy interaction: both scorers on means both branches
        # run; single-scorer modes retain their old all-candidate/native paths.
        end2end_pdms_branch = bool(use_end2end and use_rule_based)
        target_region_pdms_branch = bool(use_end2end and use_rule_based)
    # Proposal geometry is intentionally code-defined. End2End sends its native
    # Top-30 to PDMS. In parallel, the rule-based branch splits the complete
    # endpoint range into three equal-width radial layers with 1/3/5 angular
    # sectors, then randomly samples from every one of the nine regions. The
    # leaderboard target point is not used by this branch.
    region_sector_counts = (1, 3, 5)
    region_count = sum(int(value) for value in region_sector_counts)
    end2end_topk = resolve_end2end_topk(scorer_sel)
    # PDMS-as-gate selection (default off): PDMS filters, the planner's native
    # ranking picks among survivors. Gate threshold defaults to the verifier
    # threshold when unset.
    pdms_gate_native_argmax = coerce_bool(
        scorer_sel.get("pdms_gate_native_argmax", False)
    )
    pdms_gate_threshold_raw = scorer_sel.get("pdms_gate_threshold")
    pdms_gate_threshold = (
        None if pdms_gate_threshold_raw is None else float(pdms_gate_threshold_raw)
    )
    region_bearing = (-20.0, 20.0)
    deprecated_sample_count = scorer_sel.get("target_region_max_candidates")
    deprecated_random_seed = scorer_sel.get("target_region_random_seed")
    region_samples_per_region = max(
        1,
        int(
            scorer_sel.get(
                "region_samples_per_region",
                3 if deprecated_sample_count is None else deprecated_sample_count,
            )
            or 3
        ),
    )
    region_random_seed = int(
        scorer_sel.get(
            "region_random_seed",
            0 if deprecated_random_seed is None else deprecated_random_seed,
        )
        or 0
    )
    if deprecated_sample_count is not None or deprecated_random_seed is not None:
        LOG.warning(
            "scorer.target_region_max_candidates/target_region_random_seed are "
            "deprecated; use region_samples_per_region/region_random_seed"
        )
    ignored_region_keys = sorted(
        key
        for key in (
            "region_radii_m",
            "region_sector_counts",
            "topk_per_region",
            "region_bearing_deg",
        )
        if key in scorer_sel
    )
    if ignored_region_keys:
        LOG.warning(
            "Ignoring deprecated scorer region YAML keys %s; radial layers are "
            "dynamic equal-width and the target-region 1/3/5, +/-20deg layout is fixed",
            ignored_region_keys,
        )
    if not use_rule_based and not use_end2end:
        raise ValueError(
            "scorer.rule_based and scorer.end2end cannot both be false; "
            "at least one ranking scorer must be enabled"
        )
    if end2end_pdms_branch and not (use_end2end and use_rule_based):
        raise ValueError(
            "scorer.end2end_pdms_branch requires scorer.end2end=true and "
            "scorer.rule_based=true"
        )
    if target_region_pdms_branch and not use_rule_based:
        raise ValueError(
            "scorer.target_region_pdms_branch requires scorer.rule_based=true"
        )

    end2end_scorer = PlannerNativeScorer() if use_end2end else None

    # Rule-based PDMS evaluation uses a dedicated PDMSVerifier built via the
    # same create_verifier path as the final verifier, but with debug off and
    # TTC debounce disabled. With both scorers enabled it evaluates two
    # independent batches: End2End's native Top-30 and stratified random
    # samples from all nine regions. Each branch proposes its PDMS argmax.
    rule_based_scorer = None
    if use_rule_based:
        scorer_cfg = dict(verifier_cfg)
        scorer_cfg["type"] = "pdms"
        scorer_cfg["debug"] = False
        scorer_cfg["ttc_debounce_steps"] = 1
        pdms_for_scoring = create_verifier(
            ad_cfg,
            plan_dt_sec=float(getattr(agent, "_plan_dt_sec", 0.5)),
            verifier_cfg=scorer_cfg,
            lane_tier_probe=lane_tier_probe,
            ep_reference_factory=build_ep_reference_factory(agent, verifier_cfg),
        )
        rule_based_scorer = RuleBasedPDMSScorer(pdms_for_scoring)

    LOG.info(
        "Round-1 scoring: end2end=%s rule_based=%s instruction=%s "
        "end2end_topk=%d radii=dynamic_equal_width sectors=%s regions=%d "
        "bearing=[%.1f, %.1f] branches=(end2end_pdms=%s target_region_pdms=%s)",
        "planner_native" if end2end_scorer is not None else "disabled",
        "pdms" if rule_based_scorer is not None else "disabled",
        "instruction_compliance" if instruction_scorer is not None else "disabled",
        end2end_topk,
        [int(value) for value in region_sector_counts],
        region_count,
        float(region_bearing[0]),
        float(region_bearing[1]),
        end2end_pdms_branch,
        target_region_pdms_branch,
    )
    LOG.info(
        "All-region PDMS random sampling ON: regions=%d samples_per_region=%d "
        "max_total=%d seed=%d",
        region_count,
        region_samples_per_region,
        region_count * region_samples_per_region,
        region_random_seed,
    )

    # commit-and-hold recovery knobs (yaml recovery block; default off).
    recovery_dict = get_agenticdriving_recovery_cfg(agent_cfg)
    commit_and_hold = coerce_bool(recovery_dict.get("commit_and_hold", False))
    random_order = coerce_bool(recovery_dict.get("random_order", True), default=True)
    recovery_seed = recovery_dict.get("seed")
    # tier-0.5 hardening knobs (yaml recovery block; all default off so the
    # frozen t043 configs stay bit-identical). See Orchestrator.__init__.
    tier_veto = coerce_bool(recovery_dict.get("primitive_tier_veto", False))
    recovery_hd_map_endpoint_only = coerce_bool(
        recovery_dict.get("hd_map_endpoint_only", False)
    )
    full_plan_dac = coerce_bool(recovery_dict.get("primitive_full_plan_dac", False))
    hold_dac_guard = coerce_bool(recovery_dict.get("hold_dac_guard", False))
    hold_dac_guard_ticks = int(recovery_dict.get("hold_dac_guard_ticks", 3) or 3)
    bypass_return_nc_guard = coerce_bool(
        recovery_dict.get("bypass_return_nc_guard", False)
    )
    # Ceiling on the exclusive lane-bypass wait. <= 0 keeps the old unbounded
    # behaviour; see Orchestrator.lane_bypass_wait_timeout_sec.
    lane_bypass_wait_timeout_sec = float(
        recovery_dict.get("lane_bypass_wait_timeout_sec", 30.0)
    )
    allow_failed_primitive_escalation = coerce_bool(
        recovery_dict.get("allow_failed_primitive_escalation", True),
        default=True,
    )
    normal_ep_gate = coerce_bool(verifier_cfg.get("normal_ep_gate", False))
    normal_ep_min = float(verifier_cfg.get("normal_ep_min", 0.05))
    normal_ep_gate_steps = int(verifier_cfg.get("normal_ep_gate_steps", 3) or 3)
    stuck_trigger_dict = recovery_dict.get("stuck_trigger")
    if not isinstance(stuck_trigger_dict, dict):
        stuck_trigger_dict = {}
    recovery_enabled = coerce_bool(recovery_dict.get("enabled", False))
    stuck_recovery_enabled = recovery_enabled and coerce_bool(
        stuck_trigger_dict.get("enabled", True), default=True
    )
    stuck_recovery_timeout_sec = float(
        stuck_trigger_dict.get("timeout_sec", 10.0)
    )
    stuck_recovery_position_tolerance_m = float(
        stuck_trigger_dict.get("position_tolerance_m", 0.5)
    )
    recovery_vlm_reuse_sec = float(
        recovery_dict.get("vlm_reuse_sec", 0.0)
    )
    failure_memory_dict = recovery_dict.get("failure_memory")
    if not isinstance(failure_memory_dict, dict):
        failure_memory_dict = {}
    failure_memory_enabled = coerce_bool(
        failure_memory_dict.get("enabled", False)
    )
    failure_memory_clear_after_loop1_sec = float(
        failure_memory_dict.get("clear_after_loop1_sec", 1.0)
    )
    failure_memory_max_events = int(
        failure_memory_dict.get("max_events", 64) or 64
    )
    if commit_and_hold:
        LOG.info(
            "Recovery commit-and-hold ON: verify-each primitive (random=%s), "
            "commit first pass for its fixed action duration (open-loop hold)",
            random_order,
        )
    if tier_veto or full_plan_dac or hold_dac_guard:
        LOG.info(
            "Recovery tier-0.5 hardening ON: primitive_tier_veto=%s "
            "primitive_full_plan_dac=%s hold_dac_guard=%s (ticks=%d) "
            "hd_map_endpoint_only=%s",
            tier_veto, full_plan_dac, hold_dac_guard, hold_dac_guard_ticks,
            recovery_hd_map_endpoint_only,
        )
    if bypass_return_nc_guard:
        LOG.info(
            "Lane-bypass live-actor NC guard ON: pedestrians are checked over "
            "every exclusive-bypass phase; route-lane vehicles join over the "
            "return blend"
        )
    if normal_ep_gate:
        LOG.info(
            "Loop-1 EP gate ON: reject after %d consecutive EP < %.3f steps",
            normal_ep_gate_steps,
            normal_ep_min,
        )
    if stuck_recovery_enabled:
        LOG.info(
            "Stuck recovery trigger ON: force recovery after %.2fs within %.2fm",
            stuck_recovery_timeout_sec,
            stuck_recovery_position_tolerance_m,
        )
    if recovery_vlm_reuse_sec > 0.0:
        LOG.info(
            "Continuous-recovery VLM answer reuse ON: %.2f simulator seconds",
            recovery_vlm_reuse_sec,
        )
    if failure_memory_enabled:
        LOG.info(
            "Recovery failure memory ON: clear after %.2fs continuously in "
            "Loop 1 (max_events=%d)",
            failure_memory_clear_after_loop1_sec,
            failure_memory_max_events,
        )

    return (Orchestrator if orchestrator_cls is None else orchestrator_cls)(
        planner=planner_service,
        selector=agent._learned_selector,
        end2end_scorer=end2end_scorer,
        instruction_scorer=instruction_scorer,
        rule_based_scorer=rule_based_scorer,
        scorer_region_sector_counts=tuple(
            int(value) for value in region_sector_counts
        ),
        scorer_end2end_topk=end2end_topk,
        scorer_region_samples_per_region=region_samples_per_region,
        scorer_region_random_seed=region_random_seed,
        scorer_region_min_bearing_deg=float(region_bearing[0]),
        scorer_region_max_bearing_deg=float(region_bearing[1]),
        scorer_end2end_pdms_branch=end2end_pdms_branch,
        scorer_target_region_pdms_branch=target_region_pdms_branch,
        scorer_pdms_gate_native_argmax=pdms_gate_native_argmax,
        scorer_pdms_gate_threshold=pdms_gate_threshold,
        verifier=verifier,
        lane_tier_probe=lane_tier_probe,
        verifier_threshold=verifier_threshold,
        pedestrian_ttc_brake_hold_sec=pedestrian_ttc_brake_hold_sec,
        pedestrian_ttc_brake_distance_m=pedestrian_ttc_brake_distance_m,
        pedestrian_ttc_brake_lateral_m=pedestrian_ttc_brake_lateral_m,
        recovery_explainer=getattr(agent, "_recovery_explainer", None),
        semantic_takeover_manager=getattr(agent, "_semantic_takeover_manager", None),
        recovery_cfg=recovery_cfg,
        recovery_commit_and_hold=commit_and_hold,
        recovery_random_order=random_order,
        recovery_seed=None if recovery_seed is None else int(recovery_seed),
        recovery_tier_veto=tier_veto,
        recovery_hd_map_endpoint_only=recovery_hd_map_endpoint_only,
        recovery_full_plan_dac=full_plan_dac,
        recovery_hold_dac_guard=hold_dac_guard,
        recovery_hold_dac_guard_ticks=hold_dac_guard_ticks,
        recovery_bypass_return_nc_guard=bypass_return_nc_guard,
        lane_bypass_wait_timeout_sec=lane_bypass_wait_timeout_sec,
        recovery_allow_failed_primitive_escalation=allow_failed_primitive_escalation,
        recovery_tie_break_order=[
            str(n) for n in (recovery_dict.get("tie_break_order") or ())
        ],
        recovery_tabu_exempt=[
            str(n) for n in (recovery_dict.get("tabu_exempt") or ())
        ],
        recovery_rank_without_ep=coerce_bool(
            recovery_dict.get("rank_without_ep", False)
        ),
        normal_ep_gate=normal_ep_gate,
        normal_ep_min=normal_ep_min,
        normal_ep_gate_steps=normal_ep_gate_steps,
        stuck_recovery_enabled=stuck_recovery_enabled,
        stuck_recovery_timeout_sec=stuck_recovery_timeout_sec,
        stuck_recovery_position_tolerance_m=(
            stuck_recovery_position_tolerance_m
        ),
        recovery_vlm_reuse_sec=recovery_vlm_reuse_sec,
        failure_memory_enabled=failure_memory_enabled,
        failure_memory_clear_after_loop1_sec=failure_memory_clear_after_loop1_sec,
        failure_memory_max_events=failure_memory_max_events,
        failure_memory_guidance_mode=str(
            (recovery_dict.get("failure_memory") or {}).get(
                "guidance_mode", "avoid"
            )
        ),
        logger=LOG,
    )

def _transform_cached_local_plan(
    agent,
    cached_plan: np.ndarray,
    cached_pose: np.ndarray,
    current_info: Dict[str, Any],
    planner_name: str = "DrivoR",
) -> np.ndarray:
    """Move a cached ego-local plan from its original ego frame to the current one."""
    current_pose = agent._info_to_pose(current_info)
    try:
        points_world = agent._local_plan_to_world(cached_plan, cached_pose)
        return agent._world_points_to_current_local(points_world, current_pose)
    except Exception:
        LOG.exception(
            "Failed to transform cached %s plan; forcing fresh inference",
            planner_name,
        )
        return np.zeros((0, 2), dtype=np.float32)


def _trim_cached_local_plan(
    agent, plan: np.ndarray, planner_key: str = "sparsedrivev2"
) -> np.ndarray:
    """Drop stale cached waypoints before reusing a local ego-frame plan."""
    plan = np.asarray(plan, dtype=np.float32)
    if plan.ndim == 1:
        plan = plan.reshape(1, -1)
    if plan.ndim != 2 or plan.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float32)

    plan = plan[:, :2]
    finite_mask = np.isfinite(plan).all(axis=1)
    plan = plan[finite_mask]
    if len(plan) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    behind_tol = float(
        getattr(agent, f"_{planner_key}_replan_behind_tolerance_m", 0.25)
    )
    eligible = np.flatnonzero(plan[:, 1] >= -behind_tol)
    if len(eligible) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    start_idx = int(eligible[np.argmin(np.linalg.norm(plan[eligible], axis=1))])
    trimmed = plan[start_idx:]

    min_forward = float(
        getattr(agent, f"_{planner_key}_replan_trim_min_forward_m", 0.05)
    )
    front = np.flatnonzero(trimmed[:, 1] >= min_forward)
    if len(front) > 0 and int(front[0]) > 0:
        trimmed = trimmed[int(front[0]):]
    if len(trimmed) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    first = trimmed[0]
    max_start_dist = float(
        getattr(agent, f"_{planner_key}_replan_max_start_distance_m", 3.0)
    )
    max_start_lat = float(
        getattr(agent, f"_{planner_key}_replan_max_start_lateral_m", 2.5)
    )
    if max_start_dist > 0.0 and float(np.linalg.norm(first)) > max_start_dist:
        return np.zeros((0, 2), dtype=np.float32)
    if max_start_lat > 0.0 and abs(float(first[0])) > max_start_lat:
        return np.zeros((0, 2), dtype=np.float32)

    return trimmed.astype(np.float32, copy=False)


def setup_cameras_and_recording(agent):
    carla_cfg = agent._carla_cfg
    attach_legacy = coerce_bool(carla_cfg.get("attach_camera", False))
    vlm_enabled = bool(
        getattr(getattr(agent, "_vlm_selector_cfg", None), "enabled", False)
        or getattr(agent, "_semantic_takeover_manager", None) is not None
    )
    agent._camera_specs = resolve_camera_specs(
        carla_cfg,
        attach_legacy=attach_legacy,
        planner_type=agent._planner_type,
        vlm_enabled=vlm_enabled,
        perception_source=str(
            getattr(getattr(agent, "_perception", None), "source", "ground_truth")
        ),
    )

    rec_cfg = carla_cfg.get("recording") or {}
    agent._recording_enabled = coerce_bool(rec_cfg.get("enabled", False))
    agent._recording_save_frames = coerce_bool(
        rec_cfg.get("save_frames", True), default=True
    )
    # (`save_video` / `save_front_video` were parsed here for years but had
    # no consumers anywhere -- dropped.)
    agent._recording_save_grid_video = coerce_bool(
        rec_cfg.get("save_grid_video", True), default=True
    )
    # Side-by-side front-camera + top-down overlay video (front_topdown.mp4).
    # Preferred key: save_front_topdown_video. Legacy aliases still honored:
    # save_combined_video, and the ancient save_topdown_video.
    agent._recording_save_combined_video = coerce_bool(
        rec_cfg.get(
            "save_front_topdown_video",
            rec_cfg.get(
                "save_combined_video",
                rec_cfg.get("save_topdown_video", False),
            ),
        ),
        default=False,
    )
    agent._recording_fps = float(rec_cfg.get("fps", 20.0))
    # Keep the encoded FPS unchanged while retaining only every Nth simulator
    # frame. frame_stride=4 therefore produces a 4x-speed video with roughly
    # one quarter of the frames and no timestamp/FPS rewriting afterwards.
    agent._recording_frame_stride = max(1, int(rec_cfg.get("frame_stride", 1)))
    agent._recording_frame_ext = str(
        rec_cfg.get("frame_format", "jpg")
    ).lstrip(".")
    # Encode quality tiers for the inline h264 encode; explicit crf/preset
    # keys override the tier. "low" mirrors the throwaway-test-video role the
    # old mp4v writer filled.
    _QUALITY_TIERS = {
        "low": (28, "veryfast"),
        "med": (23, "fast"),
        "high": (18, "slow"),
    }
    tier = str(rec_cfg.get("video_quality", "low")).strip().lower()
    tier_crf, tier_preset = _QUALITY_TIERS.get(tier, _QUALITY_TIERS["low"])
    agent._recording_video_crf = int(rec_cfg.get("crf", tier_crf))
    agent._recording_video_preset = str(rec_cfg.get("preset", tier_preset))
    # Encode-time downscale. None -> keep the historical wrapper default
    # (0.5, see agenticdriving_agent.py); the "high" tier implies full res
    # unless the config says otherwise.
    _scale_default = 1.0 if tier == "high" else None
    _scale_raw = rec_cfg.get("video_scale", _scale_default)
    agent._recording_video_scale = (
        None if _scale_raw is None else float(_scale_raw)
    )
    # Streaming is the default: frames are written immediately to a fragmented
    # MP4, so completed fragments remain playable after an abrupt stop. Two
    # x264 threads per route keeps eight-way runs from oversubscribing the CPU.
    agent._recording_stream_video = coerce_bool(
        rec_cfg.get("stream_video", True), default=True
    )
    agent._recording_video_threads = max(
        1, int(rec_cfg.get("video_threads", os.environ.get("AGENTIC_VIDEO_THREADS", 2)))
    )
    agent._recording_video_queue_size = max(
        1, int(rec_cfg.get("video_queue_size", 2))
    )
    # Per-tick dump of the composed front_topdown frame (HUD included) so the
    # video can be re-encoded at any quality in post, and survives crashes.
    # Preferred key: save_front_topdown_frames (alias: save_combined_frames).
    agent._recording_save_combined_frames = coerce_bool(
        rec_cfg.get(
            "save_front_topdown_frames",
            rec_cfg.get("save_combined_frames", False),
        ),
        default=False,
    )
    # Standalone video of the recording-only camera (<name>.mp4, e.g.
    # chase_back.mp4): clean footage with no HUD/BEV pane -- the page-hero
    # format. Preferred: save_video/save_frames INSIDE the recording_camera
    # block, next to the mount they belong to. Legacy aliases:
    # recording.save_rec_video / save_rec_frames.
    _rec_cam_cfg = carla_cfg.get("recording_camera") or {}
    agent._recording_save_rec_video = coerce_bool(
        _rec_cam_cfg.get(
            "save_video", rec_cfg.get("save_rec_video", False)
        ),
        default=False,
    )
    agent._recording_save_rec_frames = coerce_bool(
        _rec_cam_cfg.get(
            "save_frames", rec_cfg.get("save_rec_frames", False)
        ),
        default=False,
    )

    # ------------------------------------------------------------------
    # SIMPLE MODE -- `recording.camera:` picks one of the footage ablations
    # and derives every camera/artifact detail from it + video_quality.
    # The granular keys above remain the advanced/legacy layer; when
    # `camera:` is set it is authoritative and overrides them.
    #   camera: front_topdown  -> HUD dual-view (the historical video).
    #                             high tier: 1080x1080 BEV + native 1920x1080
    #                             front pane; low/med: legacy 800/rig look.
    #   camera: front          -> clean 1920x1080 ego-front (front_1080.mp4)
    #   camera: chase_back     -> third-person chase (chase_back.mp4)
    #   camera: grid           -> 6-cam rig mosaic (grid.mp4). Rig resolution
    #                             is planner-frozen, so high tier only buys
    #                             the crf18 encode + frame dump here.
    # high tier also dumps per-tick frames for post re-encode; low/med
    # skip frames (throwaway test videos).
    # ------------------------------------------------------------------
    camera_mode = str(rec_cfg.get("camera", "")).strip().lower()
    agent._recording_hd_topdown = False
    agent._recording_save_grid_frames = False
    if camera_mode:
        hi = tier == "high"
        agent._recording_save_frames = False
        agent._recording_save_grid_video = False
        if camera_mode in ("front_topdown", "fronttop"):
            agent._recording_save_combined_video = True
            agent._recording_save_combined_frames = hi
            agent._recording_save_rec_video = False
            agent._recording_save_rec_frames = False
            if hi:
                agent._recording_hd_topdown = True
                _rec_cam_cfg = {"name": "front_1080"}
        elif camera_mode in ("front", "front_1080"):
            agent._recording_save_combined_video = False
            agent._recording_save_combined_frames = False
            agent._recording_save_rec_video = True
            agent._recording_save_rec_frames = hi
            _rec_cam_cfg = {"name": "front_1080"}
        elif camera_mode in ("chase", "chase_back"):
            agent._recording_save_combined_video = False
            agent._recording_save_combined_frames = False
            agent._recording_save_rec_video = True
            agent._recording_save_rec_frames = hi
            _rec_cam_cfg = {"name": "chase_back"}
        elif camera_mode == "grid":
            agent._recording_save_combined_video = False
            agent._recording_save_combined_frames = False
            agent._recording_save_rec_video = False
            agent._recording_save_rec_frames = False
            agent._recording_save_grid_video = True
            agent._recording_save_grid_frames = hi
            _rec_cam_cfg = {}
        else:
            raise ValueError(
                f"recording.camera={camera_mode!r}: choose "
                "front_topdown | front | chase_back | grid"
            )
    agent._recording_camera_cfg = _rec_cam_cfg

    dir_name = str(rec_cfg.get("dir_name", "recordings"))
    agent._recording_dir = agent._output_dir / dir_name
    agent._grid_video_buffer: List[np.ndarray] = []
    # Legacy fallback buffers are used only when streaming ffmpeg is missing or
    # fails to start. Normal runs write each frame directly to fragmented MP4.
    agent._combined_video_buffer: List[np.ndarray] = []
    agent._rec_video_buffer: List[np.ndarray] = []
    agent._streaming_video_writers: Dict[str, StreamingVideoWriter] = {}
    agent._recording_finalized = False
    agent._predictions_dir = resolve_predictions_dir(agent)
    agent._topdown_camera_spec = None
    agent._topdown_pixels_per_meter = 10.0

    wants_topdown = (
        agent._recording_save_combined_video
        or agent._predictions_dir is not None
    )
    if wants_topdown:
        agent._topdown_camera_spec = build_topdown_camera_spec(carla_cfg)
        if agent._recording_hd_topdown:
            # Simple-mode high tier: full-1080p BEV pane regardless of the
            # (legacy) carla.topdown_camera block.
            agent._topdown_camera_spec["width"] = 1080
            agent._topdown_camera_spec["height"] = 1080
        agent._topdown_pixels_per_meter = topdown_pixels_per_meter(
            int(agent._topdown_camera_spec["width"]),
            float(agent._topdown_camera_spec["fov"]),
            float(agent._topdown_camera_spec["z"]),
        )
        if not any(
            spec["id"] == TOPDOWN_CAMERA_ID for spec in agent._camera_specs
        ):
            agent._camera_specs.append(agent._topdown_camera_spec)

    if agent._recording_enabled and not agent._camera_specs:
        agent._camera_specs = [_build_camera_spec("CAM_FRONT", carla_cfg)]
        LOG.warning(
            "Recording enabled but no cameras configured; attaching CAM_FRONT"
        )

    # Recording-only high-res front camera. Kept OUT of _camera_specs so it
    # never enters obs["rgb"]; declared to CARLA separately in sensors().
    agent._recording_camera_spec = (
        build_recording_camera_spec(
            agent._recording_camera_cfg, agent._camera_specs
        )
        if agent._recording_enabled
        else None
    )

    if agent._recording_enabled:
        agent._recording_dir.mkdir(parents=True, exist_ok=True)
        if agent._recording_save_frames:
            for spec in agent._camera_specs:
                (agent._recording_dir / spec["id"]).mkdir(exist_ok=True)
        LOG.info(
            "Recording enabled -> %s (cameras=%s, front_topdown=%s, rec_cam=%s, "
            "predictions=%s, fps=%.1f, frame_stride=%d, speedup=%dx)",
            agent._recording_dir,
            [spec["id"] for spec in agent._camera_specs],
            agent._recording_save_combined_video,
            (agent._recording_camera_spec or {}).get("id"),
            agent._predictions_dir,
            agent._recording_fps,
            agent._recording_frame_stride,
            agent._recording_frame_stride,
        )
    elif agent._camera_specs:
        LOG.info(
            "Cameras attached (no recording): %s",
            [spec["id"] for spec in agent._camera_specs],
        )

def maybe_record_frames(agent, ego_state: Dict[str, Any]) -> None:
    if not getattr(agent, "_recording_enabled", False):
        return

    obs_rgb = ego_state.get("obs", {}).get("rgb", {})
    if not obs_rgb:
        return

    frame_idx = agent._frame_index
    if agent._recording_save_frames:
        for cam_id, rgb in obs_rgb.items():
            if cam_id == TOPDOWN_CAMERA_ID:
                continue
            frame_path = (
                agent._recording_dir
                / cam_id
                / f"{frame_idx:05d}.{agent._recording_frame_ext}"
            )
            frame_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(frame_path),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            )

    if agent._recording_save_grid_video and len(obs_rgb) > 1:
        grid_rgb = {
            cam_id: rgb
            for cam_id, rgb in obs_rgb.items()
            if cam_id != TOPDOWN_CAMERA_ID
        }
        if len(grid_rgb) > 1:
            grid = compose_grid_frame(grid_rgb, VIDEO_GRID_LAYOUT)
            if grid is not None:
                if not record_streaming_video_frame(
                    agent, "grid", "grid.mp4", grid
                ):
                    agent._grid_video_buffer.append(grid)
                if agent._recording_save_frames or getattr(
                    agent, "_recording_save_grid_frames", False
                ):
                    grid_dir = agent._recording_dir / "grid"
                    grid_dir.mkdir(exist_ok=True)
                    grid_path = (
                        grid_dir
                        / f"{frame_idx:05d}.{agent._recording_frame_ext}"
                    )
                    cv2.imwrite(
                        str(grid_path),
                        cv2.cvtColor(grid, cv2.COLOR_RGB2BGR),
                    )

    # Standalone recording-camera video (<name>.mp4): clean footage, no
    # HUD. Buffered JPEG-encoded (raw 1080p is ~6 MB/frame in RAM).
    rec_rgb = ego_state.get("rec_front_rgb")
    if rec_rgb is not None and getattr(
        agent, "_recording_save_rec_video", False
    ):
        rec_spec = getattr(agent, "_recording_camera_spec", None) or {}
        rec_name = rec_spec.get("name", "rec_view")
        streamed = record_streaming_video_frame(
            agent, "recording_camera", f"{rec_name}.mp4", rec_rgb
        )
        save_frames = getattr(agent, "_recording_save_rec_frames", False)
        rec_bgr = None
        if not streamed or save_frames:
            rec_bgr = cv2.cvtColor(rec_rgb, cv2.COLOR_RGB2BGR)
        if not streamed:
            ok, enc = cv2.imencode(".jpg", rec_bgr)
            if ok:
                agent._rec_video_buffer.append(enc)
        if save_frames:
            rec_dir = agent._recording_dir / rec_spec.get(
                "name", RECORDING_CAMERA_ID
            )
            rec_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(
                    rec_dir
                    / f"{frame_idx:05d}.{agent._recording_frame_ext}"
                ),
                rec_bgr,
            )


def _recording_env_float(name: str, default: Optional[float]) -> Optional[float]:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        LOG.warning("Ignoring invalid %s=%r", name, raw)
        return default


def record_streaming_video_frame(
    agent,
    key: str,
    filename: str,
    frame_rgb: np.ndarray,
    *,
    fps: Optional[float] = None,
) -> bool:
    """Append a frame to a lazily-created crash-tolerant streaming MP4."""
    if not video_frame_is_due(
        int(getattr(agent, "_frame_index", 0)),
        int(getattr(agent, "_recording_frame_stride", 1)),
    ):
        # Intentional skip counts as handled so callers do not append the same
        # frame to their non-streaming fallback buffers.
        return True
    if not getattr(agent, "_recording_stream_video", False):
        return False
    writers = getattr(agent, "_streaming_video_writers", None)
    if writers is None:
        return False
    writer = writers.get(key)
    if writer is None:
        try:
            crf = int(_recording_env_float(
                "AGENTIC_VIDEO_CRF",
                float(getattr(agent, "_recording_video_crf", 28)),
            ))
            configured_scale = getattr(agent, "_recording_video_scale", None)
            # Preserve the legacy encoder's low/med default: an unspecified
            # scale means half resolution, while the high tier stores 1.0.
            default_scale = 0.5 if configured_scale is None else configured_scale
            scale = _recording_env_float(
                "AGENTIC_VIDEO_SCALE",
                default_scale,
            )
            preset = os.environ.get(
                "AGENTIC_VIDEO_PRESET",
                str(getattr(agent, "_recording_video_preset", "veryfast")),
            )
            writer = StreamingVideoWriter(
                agent._recording_dir / filename,
                fps=(
                    max(1.0, float(fps))
                    if fps is not None
                    else max(1.0, float(getattr(agent, "_recording_fps", 20.0)))
                ),
                crf=crf,
                preset=preset,
                scale=scale,
                threads=int(getattr(agent, "_recording_video_threads", 2)),
                queue_size=int(getattr(agent, "_recording_video_queue_size", 2)),
                logger=LOG,
            )
            writers[key] = writer
            LOG.info("Streaming video started: %s", writer.path)
        except Exception:
            LOG.exception("Failed to start streaming video %s", filename)
            return False
    try:
        return writer.append(frame_rgb)
    except Exception:
        LOG.exception("Failed to append streaming video frame to %s", writer.path)
        return False

def write_rgb_video(
    out_path: Path,
    frames: List[np.ndarray],
    fps: float,
    crf: int = 28,
    preset: str = "veryfast",
    scale: Optional[float] = None,
) -> None:
    """Encode RGB frames as browser-playable h264 (yuv420p, +faststart).

    Pipes raw frames to ffmpeg; falls back to the legacy cv2/mp4v writer when
    no ffmpeg binary is on PATH so recording never hard-fails.

    NOTE: adapters/carla_agent/agenticdriving_agent.py monkeypatches this
    symbol with an imageio_ffmpeg-based wrapper of the same signature; keep
    the two signatures in sync."""
    if not frames:
        return
    height, width = frames[0].shape[:2]
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is not None:
        if scale is not None and 0.0 < float(scale) < 1.0:
            # Encode-time downscale (even dims enforced by trunc(.../2)*2).
            vf = (
                f"scale=trunc(iw*{float(scale)}/2)*2:"
                f"trunc(ih*{float(scale)}/2)*2,format=yuv420p"
            )
        else:
            # yuv420p requires even dims; crop a stray row/col if present.
            even_w, even_h = width - (width % 2), height - (height % 2)
            vf = f"crop={even_w}:{even_h}:0:0,format=yuv420p"
        cmd = [
            ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", f"{fps:g}", "-i", "-",
            "-vf", vf,
            "-c:v", "libx264", "-crf", str(int(crf)), "-preset", str(preset),
            "-movflags", "+faststart", "-an", str(out_path),
        ]
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            for frame in frames:
                proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            proc.stdin.close()
            if proc.wait() == 0:
                return
            LOG.warning(
                "ffmpeg encode failed for %s (rc=%s); falling back to mp4v",
                out_path, proc.returncode,
            )
        except (OSError, BrokenPipeError, ValueError) as exc:
            LOG.warning(
                "ffmpeg encode failed for %s (%s); falling back to mp4v",
                out_path, exc,
            )
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        LOG.warning("VideoWriter failed for %s", out_path)
        return
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()

def finalize_recording(agent) -> None:
    if getattr(agent, "_recording_finalized", False):
        return
    agent._recording_finalized = True
    if not getattr(agent, "_recording_enabled", False):
        return

    fps = max(1.0, float(getattr(agent, "_recording_fps", 20.0)))
    crf = int(getattr(agent, "_recording_video_crf", 28))
    preset = str(getattr(agent, "_recording_video_preset", "veryfast"))
    scale = getattr(agent, "_recording_video_scale", None)

    streaming_writers = getattr(agent, "_streaming_video_writers", {})
    for key, writer in list(streaming_writers.items()):
        if writer.close():
            LOG.info(
                "Finalized streaming video %s: %s (%d frames)",
                key,
                writer.path,
                writer.frames_written,
            )
    streaming_writers.clear()

    tfpp_frames = getattr(agent, "_tfpp_view_frames", None)
    if tfpp_frames:
        # frames may be JPEG-compressed buffers (full-res composer) or raw arrays
        decoded = [
            cv2.cvtColor(cv2.imdecode(f, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            if getattr(f, "ndim", 3) == 1 else f
            for f in tfpp_frames
        ]
        # planner helpers sharing this buffer name their own video and declare
        # their capture cadence (tfpp composes every 2nd tick -> div 2 default;
        # simlingo_helper records every tick and sets div 1)
        view_name = getattr(agent, "_planner_view_name", "tfpp_view")
        view_fps_div = float(getattr(agent, "_planner_view_fps_div", 2))
        tfpp_path = agent._recording_dir / f"{view_name}.mp4"
        write_rgb_video(tfpp_path, decoded, fps / view_fps_div, crf=crf, preset=preset, scale=scale)
        LOG.info("Wrote tfpp view video: %s (%d frames)", tfpp_path, len(decoded))

    if getattr(agent, "_recording_save_grid_video", False) and agent._grid_video_buffer:
        grid_path = agent._recording_dir / "grid.mp4"
        write_rgb_video(grid_path, agent._grid_video_buffer, fps, crf=crf, preset=preset, scale=scale)
        LOG.info(
            "Wrote grid video: %s (%d frames)",
            grid_path,
            len(agent._grid_video_buffer),
        )

    combined_buffer = getattr(agent, "_combined_video_buffer", [])
    if getattr(agent, "_recording_save_combined_video", False) and combined_buffer:
        combined_path = agent._recording_dir / "front_topdown.mp4"
        write_rgb_video(
            combined_path,
            _decode_jpeg_buffer(combined_buffer),
            fps,
            crf=crf,
            preset=preset,
            scale=scale,
        )
        LOG.info(
            "Wrote dual-view (front+topdown) video: %s (%d frames)",
            combined_path,
            len(combined_buffer),
        )

    rec_buffer = getattr(agent, "_rec_video_buffer", [])
    if getattr(agent, "_recording_save_rec_video", False) and rec_buffer:
        _rec_spec = getattr(agent, "_recording_camera_spec", None) or {}
        rec_path = agent._recording_dir / (
            f"{_rec_spec.get('name', 'rec_view')}.mp4"
        )
        write_rgb_video(
            rec_path,
            _decode_jpeg_buffer(rec_buffer),
            fps,
            crf=crf,
            preset=preset,
            scale=scale,
        )
        LOG.info(
            "Wrote recording-camera video: %s (%d frames)",
            rec_path,
            len(rec_buffer),
        )


def _decode_jpeg_buffer(buffer: List[np.ndarray]) -> List[np.ndarray]:
    """Decode a list of cv2.imencode'd JPEG frames back to RGB arrays.

    Tolerates raw HxWx3 arrays too (legacy callers), passing them through."""
    frames = []
    for item in buffer:
        if getattr(item, "ndim", 3) == 1:
            frames.append(
                cv2.cvtColor(
                    cv2.imdecode(item, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
                )
            )
        else:
            frames.append(item)
    return frames

# ------------------------------------------------------------------
# sensors() — what CARLA should attach to the ego vehicle
# ------------------------------------------------------------------

def brake_control() -> carla.VehicleControl:
    ctrl = carla.VehicleControl()
    ctrl.brake = 1.0
    return ctrl


def bridgedrive_feed_needed(agent) -> bool:
    """True when a BridgeDrive worker consumes this agent's sensor stream:
    as the planner, or as the judge perception source (Mode A)."""
    perception = getattr(agent, "_perception", None)
    return getattr(perception, "source", None) == "bridgedrive"


def get_ego_state(agent, hero, input_data, timestamp):
    capture_previous_pedals(agent, hero)
    transform = hero.get_transform()
    velocity = hero.get_velocity()
    speed_mps = math.sqrt(
        velocity.x**2 + velocity.y**2 + velocity.z**2
    )
    accel = hero.get_acceleration()
    forward = np.array([
        math.cos(math.radians(transform.rotation.pitch))
        * math.cos(math.radians(transform.rotation.yaw)),
        math.cos(math.radians(transform.rotation.pitch))
        * math.sin(math.radians(transform.rotation.yaw)),
        math.sin(math.radians(transform.rotation.pitch)),
    ])
    accel_fwd = float(
        np.dot([accel.x, accel.y, accel.z], forward)
    )
    # Match the IMU convention used by the Bench2Drive SparseDriveV2 data
    # converter: local acceleration [forward, -right, up] including gravity,
    # and negated local angular velocity in rad/s.
    ego2world = carla_transform_matrix(
        [transform.location.x, transform.location.y, transform.location.z],
        np.radians([
            transform.rotation.roll,
            transform.rotation.pitch,
            transform.rotation.yaw,
        ]),
    )
    rotation_world_from_ego = ego2world[:3, :3]
    accel_world = np.array([accel.x, accel.y, accel.z + 9.81], dtype=np.float64)
    accel_local_carla = rotation_world_from_ego.T @ accel_world
    ego_accel = np.array(
        [accel_local_carla[0], -accel_local_carla[1], accel_local_carla[2]],
        dtype=np.float32,
    )
    angular_velocity = hero.get_angular_velocity()
    angular_world = np.radians(
        [angular_velocity.x, angular_velocity.y, angular_velocity.z]
    )
    angular_local_carla = rotation_world_from_ego.T @ angular_world
    ego_angular_velocity = (-angular_local_carla).astype(np.float32)
    command = get_route_command(agent)

    obs: Dict[str, Any] = {"rgb": {}}
    # Raw half-sweep lidar for planners that consume it (TransFuser++);
    # the planner helper wraps it with pose/prev-sweep context.
    if "lidar" in input_data:
        obs["lidar_points"] = np.array(input_data["lidar"][1])
    front_rgb = None
    topdown_rgb = None
    for spec in getattr(agent, "_camera_specs", []):
        cam_id = spec["id"]
        rgb = rgb_from_input_data(input_data, cam_id)
        if rgb is not None:
            obs["rgb"][cam_id] = rgb
            if cam_id == "CAM_FRONT":
                front_rgb = rgb
            if cam_id == TOPDOWN_CAMERA_ID:
                topdown_rgb = rgb

    # Planner-specific extra cameras (e.g. TransFuser++'s 'rgb_front') are not
    # part of the recording rig; add them to obs only, without touching the
    # front/topdown video detection above.
    for spec in getattr(agent, "_extra_sensor_specs", []):
        if spec.get("type") == "sensor.camera.rgb":
            rgb = rgb_from_input_data(input_data, spec["id"])
            if rgb is not None:
                obs["rgb"][spec["id"]] = rgb

    # UniAD's reference Bench2Drive agent consumes the raw leaderboard sensor
    # tuples (including GPS/IMU and the semantic BEV camera).  Forward a
    # planner-scoped copy through the subprocess instead of trying to recreate
    # GPS/IMU conventions from the already-normalized ``info`` dictionary.

    if bridgedrive_feed_needed(agent):
        from agenticdriving.planners.bridgedrive.carla_inputs import (
            WORKER_FORWARDED_IDS,
        )

        # Raw sensor payloads for the LEAD worker, in leaderboard
        # (frame, data) form. "speed" aliases the built-in speedometer.
        # Judge-only duty (perception.source=bridgedrive, other planner)
        # forwards the reduced rig; the worker black-fills absent cameras.
        bd_input = {}
        for sensor_id in WORKER_FORWARDED_IDS:
            source_id = "speedometer" if sensor_id == "speed" else sensor_id
            if source_id in input_data:
                bd_input[sensor_id] = input_data[source_id]
        obs["bd_input_data"] = bd_input


    bbox = hero.bounding_box
    ego_box = np.array([
        transform.location.x,
        transform.location.y,
        transform.location.z,
        2.0 * bbox.extent.x,
        2.0 * bbox.extent.y,
        2.0 * bbox.extent.z,
        math.radians(transform.rotation.yaw),
    ], dtype=np.float32)

    info = {
        "timestamp": float(timestamp),
        "ego_pos": np.array([
            transform.location.x,
            transform.location.y,
            transform.location.z,
        ], dtype=np.float32),
        "ego_rot": np.radians(np.array([
            transform.rotation.roll,
            transform.rotation.pitch,
            transform.rotation.yaw,
        ], dtype=np.float32)),
        "ego_velo": float(speed_mps),
        "accelerate": float(accel_fwd),
        "ego_accel": ego_accel,
        "ego_angular_velocity": ego_angular_velocity,
        "ego_steer": float(
            getattr(hero.get_control(), "steer", agent._last_steer)
        ),
        "command": command,
        "ego_box": ego_box,
        "task_instruction": {
            0: "right", 1: "left", 2: "straight"
        }.get(command, "straight"),
        "cam_params": build_cam_params(agent),
    }

    # Recovery bypasses need authoritative lane topology before interpreting
    # nearby traffic.  In particular, an empty set of *oncoming* actors must
    # not make a same-direction adjacent lane look like a clear opposing lane.
    info["adjacent_lane_flows"] = {
        "left": {"flow": "unknown", "lane_key": None},
        "right": {"flow": "unknown", "lane_key": None},
    }
    carla_map = CarlaDataProvider.get_map()
    ego_waypoint, ego_waypoint_projected = driving_waypoint_at(
        carla_map,
        transform.location,
        driving_lane_type=carla.LaneType.Driving,
    )
    info["ego_waypoint_projected"] = bool(ego_waypoint_projected)
    # Report only the transitions. The condition persists for as long as the
    # ego straddles a lane edge, and one line per tick at 20 Hz would bury the
    # rest of the episode log.
    if bool(ego_waypoint_projected) != bool(
        getattr(agent, "_ego_waypoint_projected_last", False)
    ):
        LOG.warning(
            "ego is %s a Driving lane; HD-map topology is now anchored to %s",
            "off" if ego_waypoint_projected else "back on",
            "the nearest lane centre" if ego_waypoint_projected else "its own lane",
        )
        agent._ego_waypoint_projected_last = bool(ego_waypoint_projected)
    if ego_waypoint is not None:
        try:
            info["adjacent_lane_flows"] = carla_adjacent_lane_flows(
                ego_waypoint,
                driving_lane_type=carla.LaneType.Driving,
                ego_transform=transform,
            )
        except (RuntimeError, TypeError):
            LOG.debug("CARLA adjacent-lane topology lookup failed", exc_info=True)

    # SparseDriveV2 receives the CARLA RoadOption and approximately five-metre
    # near route point used by the official Bench2Drive agent. Keep the generic
    # coarse command above for the rest of AgenticDriving.
    if getattr(agent, "_planner_type", None) == "sparsedrivev2":
        target_point, road_option = sparsedrivev2_route_condition(agent, info)
        info["target_point"] = target_point
        info["sparsedrive_command"] = road_option

    if bridgedrive_feed_needed(agent):
        # Previously *applied* control, forwarded to the LEAD worker's Kalman
        # motion model (standalone feeds its own suggestion; ours feeds what
        # was actually actuated — see runtime.py's deviation note). Correct
        # for judge-only duty too: the Kalman input is the executed control,
        # whichever planner computed it.
        prev_ctrl = hero.get_control()
        info["bd_prev_control"] = {
            "steer": float(getattr(prev_ctrl, "steer", 0.0)),
            "throttle": float(getattr(prev_ctrl, "throttle", 0.0)),
            "brake": float(getattr(prev_ctrl, "brake", 1.0)),
        }


    publish_junction_route_heading_context(
        agent,
        info,
        ego_waypoint,
        info.get("target_point"),
    )

    # Recording-only camera frame (never merged into obs["rgb"]).
    rec_spec = getattr(agent, "_recording_camera_spec", None)
    rec_front_rgb = (
        rgb_from_input_data(input_data, rec_spec["id"])
        if rec_spec is not None
        else None
    )

    return {
        "info": info,
        "obs": obs,
        "front_rgb": front_rgb,
        "rec_front_rgb": rec_front_rgb,
        "topdown_rgb": topdown_rgb,
        "speed_mps": speed_mps,
        "transform": transform,
    }

def get_privileged_info(agent, info: Dict[str, Any]) -> List[Dict[str, Any]]:
    world = CarlaDataProvider.get_world()
    if world is None:
        return []
    carla_map = CarlaDataProvider.get_map()

    ego = find_hero(agent)
    ego_loc = ego.get_location() if ego else None
    if ego_loc is None:
        # Without an ego reference the radius and vertical filters below would
        # silently switch off — and PedestriansOnRoad-style scenarios park
        # physics-disabled substitute walkers at the ego's own (x, y) tens of
        # meters underground (pedestrian_on_road.py:57-68), which the planar
        # OBB sweep would then treat as a permanent collision. Fall back to
        # the pose snapshot taken for this frame so the filters always apply.
        ego_pos = info.get("ego_pos")
        if ego_pos is not None:
            ego_loc = carla.Location(
                x=float(ego_pos[0]), y=float(ego_pos[1]), z=float(ego_pos[2])
            )
    timestamp = float(info.get("timestamp", 0.0))
    nearby: List[Dict[str, Any]] = []
    perception_cfg = dict(
        (getattr(agent, "_cfg", None) or {}).get("perception") or {}
    )
    privileged_agent_radius_m = max(
        1.0,
        float(
            perception_cfg.get(
                "privileged_agent_radius_m",
                PRIVILEGED_AGENT_RADIUS_M,
            )
        ),
    )

    for actor in world.get_actors():
        type_id = actor.type_id
        # vehicles + walkers are the dynamic agents; static.prop.* are the
        # obstacle props that CustomObstacles-style scenarios deliberately spawn
        # (guitar case, warning signs, cones, construction barriers...). Map
        # decorations (streetlights, baked-in signage) are EnvironmentObjects,
        # not actors, so they never surface here — while traffic lights / stop
        # signs (traffic.*) and the ego's own sensors (sensor.*) *are* actors but
        # must not count as collision obstacles. Restricting to static.prop keeps
        # the deliberately-spawned obstacles and nothing else.
        is_dynamic = "vehicle" in type_id or "walker" in type_id
        is_static_prop = type_id.startswith("static.prop")
        if not is_dynamic and not is_static_prop:
            continue
        # Self-exclusion must also hold on the ego-pos fallback path (ego is
        # None there, so an id comparison alone would let the hero through as
        # a zero-distance "vehicle" obstacle).
        if ego is not None and actor.id == ego.id:
            continue
        if actor.attributes.get("role_name") == "hero":
            continue

        loc = actor.get_location()
        actor_tf = actor.get_transform()
        bbox = actor.bounding_box
        actor_rotation_rad = np.radians(
            [
                actor_tf.rotation.roll,
                actor_tf.rotation.pitch,
                actor_tf.rotation.yaw,
            ]
        )
        bbox_rotation_rad = np.radians(
            [bbox.rotation.roll, bbox.rotation.pitch, bbox.rotation.yaw]
        )
        # Use the same actor x bbox-local transform that CARLA's
        # bounding_box.get_world_vertices() uses for the green GT overlay. For
        # normal dynamic actors bbox_rotation is identity; static props expose
        # their +/-90-degree mesh alignment here.
        bbox_center_world, bbox_heading_world = carla_bbox_world_pose(
            [
                actor_tf.location.x,
                actor_tf.location.y,
                actor_tf.location.z,
            ],
            actor_rotation_rad,
            [bbox.location.x, bbox.location.y, bbox.location.z],
            bbox_rotation_rad,
        )
        bbox_yaw_offset_deg = math.degrees(
            math.atan2(
                math.sin(float(bbox_rotation_rad[2])),
                math.cos(float(bbox_rotation_rad[2])),
            )
        )
        if ego_loc is not None:
            dist = math.sqrt(
                (bbox_center_world[0] - ego_loc.x) ** 2
                + (bbox_center_world[1] - ego_loc.y) ** 2
            )
            if dist > privileged_agent_radius_m:
                continue
            if (
                abs(bbox_center_world[2] - ego_loc.z)
                > PRIVILEGED_AGENT_MAX_VERT_SEP_M
            ):
                continue  # off-plane phantom (e.g. walker parked under the map)

        vel = actor.get_velocity()
        if "walker" in type_id:
            object_type = "pedestrian"
        elif is_static_prop:
            object_type = "static"  # -> COLLISION_PENALTIES["static"] downstream
        else:
            object_type = "vehicle"

        entry = {
            "agent_id": str(actor.id),
            "actor_type_id": str(type_id),
            "timestamp": timestamp,
            "agent_pos_world": [
                float(bbox_center_world[0]),
                float(bbox_center_world[1]),
                float(bbox_center_world[2]),
            ],
            "agent_heading": float(bbox_heading_world),
            "bbox_yaw_offset_deg": float(bbox_yaw_offset_deg),
            "agent_velo": math.sqrt(
                vel.x ** 2 + vel.y ** 2 + vel.z ** 2
            ),
            "agent_vel_vec": [float(vel.x), float(vel.y)],
            "agent_extent": [
                float(2.0 * bbox.extent.x),
                float(2.0 * bbox.extent.y),
            ],
            "object_type": object_type,
            "perception_source": "ground_truth_actor",
            "actor_origin_world": [
                float(loc.x),
                float(loc.y),
                float(loc.z),
            ],
        }
        # Publish the authoritative HD-map lane membership with every dynamic
        # actor.  The recovery traffic gate can then use one lane-topology test
        # at every distance instead of guessing from an ego-aligned lateral
        # corridor on bends.
        if object_type in {"vehicle", "pedestrian"} and carla_map is not None:
            try:
                actor_waypoint = carla_map.get_waypoint(
                    loc,
                    project_to_road=False,
                    lane_type=carla.LaneType.Driving,
                )
                actor_lane_key = carla_waypoint_lane_key(actor_waypoint)
                # Keep lane-query failure distinct from a successful lookup
                # that says the actor is outside every Driving lane.  The
                # pedestrian verifier may soften prediction only for the
                # latter; missing/unknown membership remains conservative.
                entry["agent_on_driving_lane"] = bool(
                    actor_lane_key is not None
                )
                if actor_lane_key is not None:
                    entry["agent_lane_key"] = list(actor_lane_key)
            except (RuntimeError, TypeError):
                LOG.debug(
                    "CARLA actor lane lookup failed for actor %s",
                    actor.id,
                    exc_info=True,
                )
        # Preserve CARLA's exact bbox geometry for recording/debug. PDMS consumes
        # the equivalent compact center/heading/extent fields above; drawing both
        # representations makes any future transform regression immediately clear.
        try:
            bbox_vertices = bbox.get_world_vertices(actor_tf)
            entry["bbox_world_corners"] = [
                [float(vertex.x), float(vertex.y), float(vertex.z)]
                for vertex in bbox_vertices
            ]
            entry["bbox_local_location"] = [
                float(bbox.location.x),
                float(bbox.location.y),
                float(bbox.location.z),
            ]
            entry["bbox_local_rotation"] = [
                float(bbox.rotation.roll),
                float(bbox.rotation.pitch),
                float(bbox.rotation.yaw),
            ]
        except Exception:
            # Some CARLA actor wrappers do not expose get_world_vertices; the
            # red compact PDMS box remains available as a fallback visualization.
            pass
        # Current control snapshot (vehicles only): input for the verifier's
        # optional held-action bicycle forecast (agent_forecast_mode:
        # "bicycle"). Additive — the default constant-velocity path ignores it.
        if object_type == "vehicle":
            try:
                ctrl = actor.get_control()
                entry["agent_control"] = [
                    float(ctrl.steer), float(ctrl.throttle), float(ctrl.brake)
                ]
            except Exception:
                pass
        nearby.append(entry)


    return nearby

def build_cam_params(agent):
    cam_params: Dict[str, Dict[str, Any]] = {}
    specs = getattr(agent, "_camera_specs", None)
    if not specs:
        specs = [_build_camera_spec("CAM_FRONT", agent._carla_cfg)]

    for spec in specs:
        w = int(spec["width"])
        h = int(spec["height"])
        fov_rad = math.radians(float(spec["fov"]))
        front2cam = np.eye(4, dtype=np.float32)
        front2cam[0, 3] = float(spec["x"])
        front2cam[1, 3] = float(spec["y"])
        front2cam[2, 3] = float(spec["z"])
        intrinsic = {
            "H": h,
            "W": w,
            "cx": w / 2.0,
            "cy": h / 2.0,
            "fovx": fov_rad,
            "fovy": fov_rad,
        }
        single_cam = {
            "intrinsic": intrinsic,
            "front2cam": front2cam,
            "v2c": front2cam.copy(),
            "l2c": front2cam.copy(),
        }
        cam_params[spec["id"]] = single_cam
    return cam_params

def get_route_command(agent) -> int:
    if not agent._global_plan_world_coord:
        return 2
    start = max(0, agent._route_cursor)
    end = min(len(agent._global_plan_world_coord), start + 20)
    for _, option in agent._global_plan_world_coord[start:end]:
        name = str(getattr(option, "name", option)).upper()
        if "LEFT" in name and "CHANGE" not in name:
            return 1
        if "RIGHT" in name and "CHANGE" not in name:
            return 0
    return 2


def publish_junction_route_heading_context(
    agent,
    info: Dict[str, Any],
    ego_waypoint,
    target_point,
) -> None:
    """Publish the HD-map junction gate and route direction for selection.

    Junction presence comes only from CARLA waypoints. Direction comes from
    the already-computed Leaderboard/HD-map route target, so legal left and
    right turns are compared with their intended exit rather than the ego's
    incoming-road tangent.
    """
    enabled = bool(
        getattr(agent, "_junction_route_heading_filter_enabled", True)
    )
    lookahead_m = max(
        0.0,
        float(getattr(agent, "_junction_route_heading_lookahead_m", 12.0)),
    )
    max_angle_deg = max(
        0.0,
        min(
            180.0,
            float(
                getattr(
                    agent, "_junction_route_heading_max_angle_deg", 60.0
                )
            ),
        ),
    )
    is_current = bool(
        ego_waypoint is not None
        and getattr(ego_waypoint, "is_junction", False)
    )
    distance_m = 0.0 if is_current else None
    if enabled and ego_waypoint is not None and not is_current:
        step_m = min(2.0, lookahead_m) if lookahead_m > 0.0 else 0.0
        distance = step_m
        while distance > 0.0 and distance <= lookahead_m + 1e-6:
            try:
                successors = list(ego_waypoint.next(float(distance)))
            except Exception:
                break
            if any(
                successor is not None
                and bool(getattr(successor, "is_junction", False))
                for successor in successors
            ):
                distance_m = float(distance)
                break
            distance += step_m

    route_heading = None
    try:
        candidate = np.asarray(target_point, dtype=np.float64).reshape(-1)[:2]
        if (
            candidate.size >= 2
            and np.isfinite(candidate).all()
            and float(np.linalg.norm(candidate)) >= 0.5
        ):
            route_heading = candidate.astype(float).tolist()
    except (TypeError, ValueError):
        pass

    active = bool(
        enabled
        and distance_m is not None
        and float(distance_m) <= lookahead_m + 1e-6
    )
    info["hd_map_junction"] = {
        "is_current": is_current,
        "distance_m": distance_m,
        "lookahead_m": lookahead_m,
    }
    info["junction_route_heading_filter"] = {
        "enabled": enabled,
        "active": active,
        "is_current": is_current,
        "junction_distance_m": distance_m,
        "lookahead_m": lookahead_m,
        "max_angle_deg": max_angle_deg,
        "route_heading_local_xy": route_heading,
        "route_target_source": info.get("target_point_source"),
    }


def sparsedrivev2_route_condition(
    agent, info: Dict[str, Any], lookahead_m: float = 5.0
) -> Tuple[Optional[List[float]], int]:
    """Return the normal route condition or an HD-map lane-reentry condition.

    NORMAL operation retains SparseDriveV2's historical nearest-route-vertex
    waypoint exactly. Re-entry starts only after CARLA's HD map confirms that
    the ego occupies a lane adjacent to the route lane for several ticks. This
    detects Loop-1 lane changes without confusing curved route geometry for
    lateral motion.
    """
    plan = getattr(agent, "_global_plan_world_coord", None)
    if not plan:
        return None, 4

    points: List[np.ndarray] = []
    options: List[Any] = []
    for entry, option in plan:
        loc = getattr(entry, "location", entry)
        try:
            points.append(np.array([float(loc.x), float(loc.y)], dtype=np.float64))
            options.append(option)
        except Exception:
            continue
    if not points:
        return None, 4

    normal_lookahead_m = max(
        0.0,
        float(getattr(agent, "_sparsedrivev2_route_lookahead_m", lookahead_m)),
    )
    target, road_option = route_condition_from_polyline(
        points,
        options,
        info["ego_pos"],
        info["ego_rot"],
        normal_lookahead_m,
    )

    info["route_distance_m"] = None
    info["route_lane_key"] = None
    info["ego_lane_key"] = None
    info["route_lane_relation"] = None
    info["route_reentry_active"] = False
    info["route_reentry_primitive"] = None
    info["route_reentry_command_override"] = False
    info["route_reentry_side"] = None
    info["route_reentry_trigger_reason"] = None
    info["route_target_lookahead_m"] = normal_lookahead_m
    info["target_point_source"] = "route"

    projection = route_projection_from_polyline(
        points,
        info["ego_pos"],
        segment_hint=getattr(
            agent, "_sparsedrivev2_route_segment_index", None
        ),
        search_back=int(
            getattr(agent, "_sparsedrivev2_route_projection_search_back", 20)
        ),
        search_ahead=int(
            getattr(agent, "_sparsedrivev2_route_projection_search_ahead", 200)
        ),
    )
    if projection is None:
        return target, road_option

    route_projection, route_distance_m, segment_index = projection
    agent._sparsedrivev2_route_segment_index = segment_index
    info["route_distance_m"] = route_distance_m

    # The historical nearest-vertex target can be behind the ego when route
    # samples are sparse, in which case it returns None and SparseDriveV2
    # silently substitutes command_target_point() == [0, 5]. That point is
    # attached to the vehicle heading and is the source of the apparent
    # "waypoint rotating with the car" behaviour. Preserve the historical
    # result when valid, but fall back to an ego-progress route projection.
    if target is None:
        target, road_option = projected_route_condition_from_polyline(
            points,
            options,
            info["ego_pos"],
            info["ego_rot"],
            normal_lookahead_m,
            segment_hint=segment_index,
            search_back=int(
                getattr(agent, "_sparsedrivev2_route_projection_search_back", 20)
            ),
            search_ahead=int(
                getattr(agent, "_sparsedrivev2_route_projection_search_ahead", 200)
            ),
        )
        if target is not None:
            info["target_point_source"] = "route_projection"

    now = float(info.get("timestamp", 0.0) or 0.0)
    active = bool(getattr(agent, "_sparsedrivev2_route_reentry_active", False))

    if not bool(getattr(agent, "_sparsedrivev2_route_reentry_enabled", True)):
        return target, road_option

    carla_map = CarlaDataProvider.get_map()
    route_waypoint = None
    ego_waypoint = None
    lane_relation = None
    if carla_map is not None:
        ego = np.asarray(info["ego_pos"], dtype=np.float64).reshape(-1)
        ego_loc = carla.Location(
            x=float(ego[0]),
            y=float(ego[1]),
            z=float(ego[2]) if ego.size >= 3 else 0.0,
        )
        # get_ego_state() already published info["ego_waypoint_projected"] for
        # this same pose; it stays the single writer of that flag.
        ego_waypoint, _ = driving_waypoint_at(
            carla_map,
            ego_loc,
            driving_lane_type=carla.LaneType.Driving,
        )

    def lane_key(waypoint):
        if waypoint is None:
            return None
        try:
            return [
                int(waypoint.road_id),
                int(waypoint.section_id),
                int(waypoint.lane_id),
            ]
        except (AttributeError, TypeError, ValueError):
            return None

    def hd_map_target(waypoint, lookahead_m):
        """Return a forward ego-frame target on one HD-map lane centerline."""
        if waypoint is None:
            return None
        try:
            candidates = list(waypoint.next(float(lookahead_m)))
        except Exception:
            return None
        key = lane_key(waypoint)
        for candidate in candidates:
            if candidate is None or lane_key(candidate) != key:
                continue
            if bool(getattr(candidate, "is_junction", False)):
                continue
            transform = getattr(candidate, "transform", None)
            loc = getattr(transform, "location", None)
            if loc is None:
                continue
            world = np.array([float(loc.x), float(loc.y), float(ego[2]), 1.0])
            local = np.linalg.inv(
                carla_transform_matrix(info["ego_pos"], info["ego_rot"])
            ) @ world
            right, forward = float(local[1]), float(local[0])
            if forward > 0.0:
                return [right, forward]
        return None

    max_adjacent_lanes = int(
        getattr(agent, "_sparsedrivev2_route_reentry_max_adjacent_lanes", 2)
    )
    # Do not query the route lane at ``route_projection``: on a curve, the
    # polyline chord can lie several metres inside the true lane and snap to an
    # adjacent lane. Query actual Leaderboard route samples around the current
    # progress instead, then choose the nearest sample whose HD-map lane has a
    # valid route/left/right topology relation to the ego lane.
    if (
        carla_map is not None
        and ego_waypoint is not None
        and not bool(getattr(ego_waypoint, "is_junction", False))
    ):
        search_points = max(
            1,
            int(getattr(agent, "_sparsedrivev2_route_reentry_map_search_points", 12)),
        )
        start = max(0, segment_index - search_points)
        end = min(len(points), segment_index + search_points + 2)
        candidates = []
        for idx in range(start, end):
            route_loc = carla.Location(
                x=float(points[idx][0]),
                y=float(points[idx][1]),
                z=float(ego[2]) if ego.size >= 3 else 0.0,
            )
            try:
                candidate = carla_map.get_waypoint(
                    route_loc,
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving,
                )
            except (RuntimeError, TypeError):
                continue
            if candidate is None or bool(getattr(candidate, "is_junction", False)):
                continue
            relation = carla_route_lane_relation(
                candidate,
                ego_waypoint,
                max_adjacent_lanes=max_adjacent_lanes,
                driving_lane_type=carla.LaneType.Driving,
            )
            if relation is None:
                continue
            distance_sq = (
                (float(points[idx][0]) - float(ego[0])) ** 2
                + (float(points[idx][1]) - float(ego[1])) ** 2
            )
            candidates.append((distance_sq, idx, candidate, relation))
        if candidates:
            _, _, route_waypoint, lane_relation = min(candidates, key=lambda x: x[0])

    route_lane_key = lane_key(route_waypoint)
    ego_lane_key = lane_key(ego_waypoint)
    info["route_lane_key"] = route_lane_key
    info["ego_lane_key"] = ego_lane_key
    info["route_lane_relation"] = lane_relation

    # Anchor HD-map lookahead at the ego's current longitudinal progress. The
    # selected ``route_waypoint`` above originates from a sparse Leaderboard
    # route sample and may be tens of metres away; using it directly caused a
    # configured 5 m target to appear 70+ m ahead in both planner input and BEV.
    route_waypoint_at_ego = carla_route_waypoint_at_ego_progress(
        route_waypoint,
        ego_waypoint,
        lane_relation,
        max_adjacent_lanes=max_adjacent_lanes,
        driving_lane_type=carla.LaneType.Driving,
    )

    # Anchor bypass topology to the route lane, not blindly to whichever lane
    # the learned planner currently occupies.  If Loop 1 has already drifted
    # into the opposing lane, ``lane_bypass_left`` still means "use the
    # route's left adjacent lane and return to the route"; otherwise resolving
    # left/right from the borrowed waypoint can point at a second lane or the
    # sidewalk.  The centerlines remain expressed in the current vehicle frame.
    if route_waypoint_at_ego is not None:
        try:
            ego_pos = np.asarray(info["ego_pos"], dtype=np.float64).reshape(-1)
            ego_rot = np.asarray(info["ego_rot"], dtype=np.float64).reshape(-1)
            current_transform = carla.Transform(
                carla.Location(
                    x=float(ego_pos[0]),
                    y=float(ego_pos[1]),
                    z=float(ego_pos[2]) if ego_pos.size >= 3 else 0.0,
                ),
                carla.Rotation(
                    roll=math.degrees(float(ego_rot[0])) if ego_rot.size >= 1 else 0.0,
                    pitch=math.degrees(float(ego_rot[1])) if ego_rot.size >= 2 else 0.0,
                    yaw=math.degrees(float(ego_rot[2])) if ego_rot.size >= 3 else 0.0,
                ),
            )
            info["adjacent_lane_flows"] = carla_adjacent_lane_flows(
                route_waypoint_at_ego,
                driving_lane_type=carla.LaneType.Driving,
                ego_transform=current_transform,
            )
            info["adjacent_lane_flows_anchor"] = "route_lane"
        except (RuntimeError, TypeError, ValueError, IndexError):
            LOG.debug(
                "CARLA route-anchored adjacent-lane lookup failed",
                exc_info=True,
            )

    normal_hd_target = hd_map_target(route_waypoint_at_ego, normal_lookahead_m)
    if normal_hd_target is not None:
        target = normal_hd_target
        info["target_point_source"] = "hd_map"

    if active and lane_relation == "route":
        settled = int(
            getattr(agent, "_sparsedrivev2_route_reentry_settled_ticks", 0)
        ) + 1
        agent._sparsedrivev2_route_reentry_settled_ticks = settled
        required_ticks = max(
            1,
            int(getattr(agent, "_sparsedrivev2_route_reentry_completion_ticks", 4)),
        )
        if settled >= required_ticks:
            completed_side = getattr(
                agent, "_sparsedrivev2_route_reentry_side", None
            )
            _clear_sparsedrivev2_route_reentry(agent)
            LOG.info(
                "SparseDriveV2 route re-entry completed from %s "
                "(route_lane=%s ego_lane=%s)",
                completed_side,
                route_lane_key,
                ego_lane_key,
            )
            return target, road_option
    elif active:
        agent._sparsedrivev2_route_reentry_settled_ticks = 0

    if not active:
        if lane_relation not in {"left", "right"}:
            agent._sparsedrivev2_route_reentry_trigger_ticks_seen = 0
            agent._sparsedrivev2_route_reentry_pending_side = None
            return target, road_option
        pending_side = getattr(
            agent, "_sparsedrivev2_route_reentry_pending_side", None
        )
        if pending_side != lane_relation:
            agent._sparsedrivev2_route_reentry_pending_side = lane_relation
            agent._sparsedrivev2_route_reentry_trigger_ticks_seen = 0
        trigger_ticks_seen = int(
            getattr(agent, "_sparsedrivev2_route_reentry_trigger_ticks_seen", 0)
        ) + 1
        agent._sparsedrivev2_route_reentry_trigger_ticks_seen = trigger_ticks_seen
        trigger_ticks_required = max(
            1,
            int(getattr(agent, "_sparsedrivev2_route_reentry_trigger_ticks", 3)),
        )
        if trigger_ticks_seen < trigger_ticks_required:
            return target, road_option
        source = str(getattr(agent, "_previous_selected_source", "") or "")
        primitive = _recovery_primitive_from_source(source)
        side = lane_relation
        agent._sparsedrivev2_route_reentry_active = True
        agent._sparsedrivev2_route_reentry_settled_ticks = 0
        agent._sparsedrivev2_route_reentry_pending_side = None
        agent._sparsedrivev2_route_reentry_primitive = primitive
        agent._sparsedrivev2_route_reentry_side = side
        agent._sparsedrivev2_route_reentry_trigger_reason = "hd_map_adjacent_lane"
        agent._sparsedrivev2_route_reentry_started_at = now
        active = True
        LOG.info(
            "SparseDriveV2 route re-entry triggered by HD map: side=%s "
            "route_lane=%s ego_lane=%s source=%s",
            side,
            route_lane_key,
            ego_lane_key,
            source or "unknown",
        )

    started_at_raw = getattr(
        agent, "_sparsedrivev2_route_reentry_started_at", now
    )
    started_at = now if started_at_raw is None else float(started_at_raw)
    max_duration_sec = max(
        0.0,
        float(getattr(agent, "_sparsedrivev2_route_reentry_max_duration_sec", 15.0)),
    )
    if max_duration_sec > 0.0 and now - started_at > max_duration_sec:
        expired_side = getattr(agent, "_sparsedrivev2_route_reentry_side", None)
        _clear_sparsedrivev2_route_reentry(agent)
        LOG.info(
            "SparseDriveV2 route re-entry expired after %.1fs from %s "
            "(route_lane=%s ego_lane=%s relation=%s)",
            now - started_at,
            expired_side,
            route_lane_key,
            ego_lane_key,
            lane_relation,
        )
        return target, road_option

    reentry_lookahead_m = max(
        normal_lookahead_m,
        float(getattr(agent, "_sparsedrivev2_route_reentry_lookahead_m", 12.0)),
    )
    reentry_target, _ = projected_route_condition_from_polyline(
        points,
        options,
        info["ego_pos"],
        info["ego_rot"],
        reentry_lookahead_m,
        segment_hint=segment_index,
        search_back=int(
            getattr(agent, "_sparsedrivev2_route_projection_search_back", 20)
        ),
        search_ahead=int(
            getattr(agent, "_sparsedrivev2_route_projection_search_ahead", 200)
        ),
    )

    if reentry_target is not None:
        target = reentry_target

    # Once HD-map lane topology has confirmed a lane departure, anchor the
    # re-entry target to the *route lane's HD-map centerline*.  The old
    # polyline projection is only a geometric chord through sparse route
    # samples; on a bend (and especially when the ego is in the adjacent lane)
    # that chord can project to a point many metres off the drivable surface.
    # CARLA's Waypoint.next() follows the lane centerline through the curve.
    if route_waypoint_at_ego is not None:
        hd_target = hd_map_target(route_waypoint_at_ego, reentry_lookahead_m)
        if hd_target is not None:
            target = hd_target
            info["target_point_source"] = "hd_map"
    command_override = False
    # Preserve explicit junction and Leaderboard lane-change commands. Only
    # strengthen the ambiguous LANEFOLLOW case that causes the borrowed-lane
    # linger. Bench2Drive RoadOption: 5=CHANGELANELEFT, 6=CHANGELANERIGHT.
    if road_option == 4 and target is not None:
        reentry_side = str(
            getattr(agent, "_sparsedrivev2_route_reentry_side", "") or ""
        )
        if reentry_side == "right":
            # Ego departed to the route's right; return left.
            road_option = 5
            command_override = True
        elif reentry_side == "left":
            # Ego departed to the route's left; return right.
            road_option = 6
            command_override = True

    info["route_reentry_active"] = True
    info["route_reentry_primitive"] = getattr(
        agent, "_sparsedrivev2_route_reentry_primitive", None
    )
    info["route_reentry_side"] = getattr(
        agent, "_sparsedrivev2_route_reentry_side", None
    )
    info["route_reentry_trigger_reason"] = getattr(
        agent, "_sparsedrivev2_route_reentry_trigger_reason", None
    )
    info["route_reentry_command_override"] = command_override
    info["route_target_lookahead_m"] = reentry_lookahead_m
    return target, road_option


def _recovery_primitive_from_source(source: str) -> Optional[str]:
    """Extract a stable primitive id from recovery/recovery-hold sources."""
    source = str(source or "")
    for prefix in ("recovery_hold:", "recovery:"):
        if source.startswith(prefix):
            primitive = source[len(prefix):].strip()
            return primitive or None
    return None


def _clear_sparsedrivev2_route_reentry(agent) -> None:
    agent._sparsedrivev2_route_reentry_active = False
    agent._sparsedrivev2_route_reentry_settled_ticks = 0
    agent._sparsedrivev2_route_reentry_trigger_ticks_seen = 0
    agent._sparsedrivev2_route_reentry_pending_side = None
    agent._sparsedrivev2_route_reentry_primitive = None
    agent._sparsedrivev2_route_reentry_side = None
    agent._sparsedrivev2_route_reentry_trigger_reason = None
    agent._sparsedrivev2_route_reentry_started_at = None




def find_hero(agent):
    world = CarlaDataProvider.get_world()
    if world is None:
        return None
    for actor in world.get_actors():
        if actor.attributes.get("role_name") == "hero":
            return actor
    return None


from sparsedrivev2_helper import (  # noqa: E402
    setup_sparsedrivev2,
    run_step_sparsedrivev2,
)
