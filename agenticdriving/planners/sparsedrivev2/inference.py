#!/usr/bin/env python3
"""SparseDriveV2 Bench2Drive inference from native CARLA observations.

This runs the **bench2drive** branch of SparseDriveV2: the mmdet3d-plugin model
(``projects/mmdet3d_plugin``) built from ``projects/configs/sparsedrive_stage2.py``
and loaded from ``sparsedrive_small_b2d_stage2.pth``. It reproduces the closed-loop
per-frame path of ``leaderboard/team_code/sparsedrive_b2d_agent.py`` (``run_step``):
build the ``results`` dict, run the repo's own ``test_pipeline`` + ``collate``,
forward the model, and read the planning trajectory from ``img_bbox``. We do NOT
re-implement any model math.

Unlike the NAVSIM variant this branch is:
  * the exact six-camera Bench2Drive CARLA sensor rig,
  * config/registry driven (mmcv ``Config`` + mmdet ``build_detector``),
  * temporal (an InstanceBank propagates memory across frames using ``T_global``).

Env contract (set by launch.sh / the CARLA helper):
  SPARSEDRIVEV2_REPO_ROOT  -> third_party/SparseDriveV2   (bench2drive branch)
  SPARSEDRIVEV2_CONFIG     -> projects/configs/sparsedrive_stage2.py (repo-relative or abs)
  SPARSEDRIVEV2_CKPT       -> ../../checkpoints/sparsedrivev2/sparsedrive_small_b2d_stage2.pth
  SPARSEDRIVEV2_DEVICE     -> cuda / cuda:N / cpu
  SPARSEDRIVEV2_TEMPORAL   -> 1 (default) to feed ego-pose T_global; 0 to disable

Runs in the dedicated ``sparsedrive_b2d`` conda env (torch2.0.1+cu118, mmcv-full
1.7.2, mmdet 2.28.2, compiled deformable_aggregation op).
"""
from __future__ import annotations

import contextlib
import copy
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .carla_inputs import (
    CAMERAS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    JPEG_QUALITY,
    MODEL_FRAME_RATE,
    carla_pose_to_lidar2global,
    coarse_command_to_road_option,
    command_onehot,
)

LOG = logging.getLogger("sparsedrivev2_inference")

DEFAULT_CAM_MAP: Dict[str, str] = {c: c for c in CAMERAS}

# Bench2Drive rig geometry: raw nuScenes-style 1600x900 images fed at 0.44 scale.
RESIZE_SCALE = 0.44
DEFAULT_CONFIG_REL = "projects/configs/sparsedrive_stage2.py"
DEFAULT_CKPT_REL = "../../checkpoints/sparsedrivev2/sparsedrive_small_b2d_stage2.pth"


def _ensure_repo_on_path() -> str:
    repo_root = os.environ.get("SPARSEDRIVEV2_REPO_ROOT", "").strip()
    if not repo_root:
        raise RuntimeError("SPARSEDRIVEV2_REPO_ROOT must be set in environment")
    repo_root = str(Path(repo_root).resolve())
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    return repo_root


# ---- hardcoded Bench2Drive calibration (verbatim from sparsedrive_b2d_agent.py) ----
def _b2d_calibration() -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    lidar2cam = {
        "CAM_FRONT": np.array([[1., 0., 0., 0.], [0., 0., 1., 0.], [0., -1., 0., 0.], [0., -0.24, -1.19, 1.]]),
        "CAM_FRONT_RIGHT": np.array([[0.57357644, 0., 0.81915204, 0.], [-0.81915204, 0., 0.57357644, 0.], [0., -1., 0., 0.], [0.22517331, -0.24, -0.82909407, 1.]]),
        "CAM_FRONT_LEFT": np.array([[0.57357644, 0., -0.81915204, 0.], [0.81915204, 0., 0.57357644, 0.], [0., -1., 0., 0.], [-0.22517331, -0.24, -0.82909407, 1.]]),
        "CAM_BACK": np.array([[-1., 0., 1.22464680e-16, 0.], [-1.22464680e-16, 0., -1., 0.], [0., -1., 0., 0.], [-1.97168135e-16, -0.24, -1.61, 1.]]),
        "CAM_BACK_LEFT": np.array([[-0.34202014, 0., -0.93969262, 0.], [0.93969262, 0., -0.34202014, 0.], [0., -1., 0., 0.], [-0.25388956, -0.24, -0.49288953, 1.]]),
        "CAM_BACK_RIGHT": np.array([[-0.34202014, 0., 0.93969262, 0.], [-0.93969262, 0., -0.34202014, 0.], [0., -1., 0., 0.], [0.25388956, -0.24, -0.49288953, 1.]]),
    }
    K_std = np.array([[1142.51841, 0., 800.], [0., 1142.51841, 450.], [0., 0., 1.]])
    K_back = np.array([[560.16603057, 0., 800.], [0., 560.16603057, 450.], [0., 0., 1.]])
    cam_intrinsic = {c: (K_back.copy() if c == "CAM_BACK" else K_std.copy()) for c in CAMERAS}
    lidar2img = {}
    for key, value in cam_intrinsic.items():
        tf = np.eye(3)
        tf[:2, :2] *= RESIZE_SCALE
        intrinsic = tf @ value
        cam_intrinsic[key] = intrinsic
        viewpad = np.eye(4)
        viewpad[:3, :3] = intrinsic
        lidar2img[key] = viewpad @ lidar2cam[key].T
    return lidar2cam, cam_intrinsic, lidar2img


def make_command_onehot(command: int) -> np.ndarray:
    """CARLA/Bench2Drive RoadOption value 1..6 -> six-way one-hot."""
    return command_onehot(command)


def command_target_point(command: int) -> np.ndarray:
    """Five-metre CARLA near-point fallback in ego ``[right, forward]``."""
    mapping = {
        1: np.array([-3.0, 4.0], dtype=np.float32),  # LEFT
        2: np.array([3.0, 4.0], dtype=np.float32),   # RIGHT
    }
    return mapping.get(int(command), np.array([0.0, 5.0], dtype=np.float32))


def _float_list(value: Any) -> List[float]:
    if value is None:
        return []
    try:
        return np.asarray(value, dtype=np.float64).astype(float).tolist()
    except Exception:
        return []


class SparseDriveV2Inference:
    """Loads the Bench2Drive SparseDrive stage2 model once and runs its native
    forward per frame, returning an ego-local CARLA trajectory."""

    def __init__(
        self,
        repo_root: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        config_path: Optional[str] = None,
        cam_map: Optional[Dict[str, str]] = None,
    ) -> None:
        self.repo_root = str(Path(repo_root).resolve()) if repo_root else _ensure_repo_on_path()
        _ensure_repo_on_path()
        root = Path(self.repo_root)
        cfg_env = config_path or os.environ.get("SPARSEDRIVEV2_CONFIG", "").strip() or DEFAULT_CONFIG_REL
        self.config_path = str(Path(cfg_env) if os.path.isabs(cfg_env) else root / cfg_env)
        ckpt_env = checkpoint_path or os.environ.get("SPARSEDRIVEV2_CKPT", "").strip() or DEFAULT_CKPT_REL
        self.checkpoint_path = str(Path(ckpt_env) if os.path.isabs(ckpt_env) else root / ckpt_env)
        self.device_str = device or os.environ.get("SPARSEDRIVEV2_DEVICE", "cuda")
        self.use_temporal = os.environ.get("SPARSEDRIVEV2_TEMPORAL", "1").strip() not in ("0", "false", "False")
        self.cam_map = dict(cam_map or DEFAULT_CAM_MAP)

        self._torch = None
        self._cv2 = None
        self._model = None
        self._device = None
        self._pipeline = None
        self._data_aug_conf = None
        self._aug_config = None
        self._lidar2cam, self._cam_intrinsic, self._lidar2img = _b2d_calibration()
        self.num_poses = 6
        self.plan_dt = 0.5
        self._frame_idx = 0
        self.last_debug: Dict[str, Any] = {}

    @contextlib.contextmanager
    def _in_repo(self):
        """Temporarily chdir to the repo root so the config's relative anchor
        paths (``data/kmeans/*``, loaded at model-build AND lazily at decode)
        resolve, then restore CWD — the in-process CARLA agent shares this
        process and must not have its working directory silently changed."""
        prev = os.getcwd()
        os.chdir(self.repo_root)
        try:
            yield
        finally:
            os.chdir(prev)

    # ---- setup -------------------------------------------------------------
    def setup(self) -> None:
        import cv2
        import torch

        from mmcv import Config
        from mmcv.runner import load_checkpoint
        from mmcv.parallel import MMDataParallel
        from mmdet.models import build_detector
        from mmdet.datasets.pipelines import Compose

        self._torch = torch
        self._cv2 = cv2

        with self._in_repo():
            cfg = Config.fromfile(self.config_path)
            # Register the mmdet3d_plugin modules (models/datasets/pipelines).
            if cfg.get("plugin", False) and cfg.get("plugin_dir"):
                import importlib
                module_path = cfg.plugin_dir.rstrip("/").replace("/", ".")
                importlib.import_module(module_path)

            model = build_detector(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
            load_checkpoint(model, self.checkpoint_path, map_location="cpu", strict=True)

        # CUDA_VISIBLE_DEVICES (set by launch.sh) already masks to the target GPU,
        # so device index 0 is correct inside the subprocess.
        gpu_rank = 0
        model = MMDataParallel(model, device_ids=[gpu_rank])
        model.eval()
        self._model = model
        self._device = next(model.module.parameters()).device

        pipeline = [t for t in cfg.test_pipeline
                    if t["type"] not in ("LoadMultiViewImageFromFilesInCeph", "LoadMultiViewImageFromFiles")]
        self._pipeline = Compose(pipeline)
        self._data_aug_conf = cfg.data_aug_conf
        self._aug_config = self._build_aug_config()
        self.num_poses = int(cfg.get("lon_fut_ts", 6))
        LOG.info(
            "SparseDriveV2 (bench2drive) loaded: cfg=%s ckpt=%s device=%s temporal=%s num_poses=%d",
            self.config_path, self.checkpoint_path, self._device, self.use_temporal, self.num_poses,
        )

    def _build_aug_config(self) -> Dict[str, Any]:
        conf = self._data_aug_conf
        H = 900 * RESIZE_SCALE
        W = 1600 * RESIZE_SCALE
        fH, fW = conf["final_dim"]
        resize = max(fH / H, fW / W)
        resize_dims = (int(W * resize), int(H * resize))
        newW, newH = resize_dims
        crop_h = int((1 - np.mean(conf["bot_pct_lim"])) * newH) - fH
        crop_w = int(max(0, newW - fW) / 2)
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        return {"resize": resize, "resize_dims": resize_dims, "crop": crop,
                "flip": False, "rotate": 0, "rotate_3d": 0}

    # ---- per-frame ---------------------------------------------------------
    def _build_results(self, obs: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, Any]:
        cv2 = self._cv2
        rgb_map = obs.get("rgb", {}) if isinstance(obs, dict) else {}
        imgs: List[np.ndarray] = []
        for cam in CAMERAS:
            carla_cam = self.cam_map.get(cam)
            if carla_cam is None or carla_cam not in rgb_map:
                raise KeyError(
                    f"missing CARLA camera {carla_cam!r} for SparseDrive camera {cam!r}"
                )
            image_rgb = np.asarray(rgb_map[carla_cam])[..., :3].astype(np.uint8)
            if image_rgb.shape[:2] != (IMAGE_HEIGHT, IMAGE_WIDTH):
                # BEVFormer perception shares the same physical Bench2Drive
                # mounts but requests the detector's native 1600x900 frames.
                # Downsample those shared sensors for SparseDriveV2 instead of
                # attaching a second six-camera rig (the leaderboard limit is 8).
                image_rgb = cv2.resize(
                    image_rgb,
                    (IMAGE_WIDTH, IMAGE_HEIGHT),
                    interpolation=cv2.INTER_AREA,
                )
            # The upstream agent passes OpenCV BGR into a pipeline configured
            # with to_rgb=True. Preserve that contract and its quality-20 JPEG
            # domain matching exactly.
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            ok, encoded = cv2.imencode(
                ".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            )
            if not ok:
                raise RuntimeError(f"JPEG encoding failed for {carla_cam}")
            image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            imgs.append(image_bgr)

        results: Dict[str, Any] = {}
        # The temporal InstanceBank estimates speed from (inter-frame ego
        # displacement / timestamp delta). The original code hardcoded one model
        # step = 1/MODEL_FRAME_RATE (0.1s), assuming the model is called at 10 Hz
        # like the reference agent's 10 Hz CARLA. But this rig runs CARLA at 20 Hz
        # and calls process() every tick (~0.05s real), so a hardcoded 0.1s makes
        # the model clock run ~2x real time and halves every inferred speed. Feed
        # the real CARLA sim time instead so the labelled dt matches reality,
        # regardless of simulator rate. Set AGENTICDRIVING_SD_REAL_TIMESTAMP=0 to
        # restore the legacy hardcoded-dt behavior (for A/B comparison).
        if os.environ.get("AGENTICDRIVING_SD_REAL_TIMESTAMP", "1") != "0":
            results["timestamp"] = float(
                info.get("timestamp", self._frame_idx / MODEL_FRAME_RATE)
            )
        else:
            results["timestamp"] = self._frame_idx / MODEL_FRAME_RATE
        results["img"] = imgs
        results["lidar2img"] = [self._lidar2img[c] for c in CAMERAS]
        results["lidar2cam"] = [self._lidar2cam[c] for c in CAMERAS]
        results["cam_intrinsic"] = [copy.deepcopy(self._cam_intrinsic[c]) for c in CAMERAS]

        # CARLA pose is [world_x, world_y, world_z] and
        # [roll, pitch, yaw]. Convert it exactly as the B2D data converter does.
        if self.use_temporal:
            ego_pos = np.asarray(info.get("ego_pos", [0.0, 0.0, 0.0]), dtype=np.float64).ravel()
            ego_rot = np.asarray(info.get("ego_rot", [0.0, 0.0, 0.0]), dtype=np.float64).ravel()
            if ego_pos.size != 3 or ego_rot.size != 3:
                raise ValueError("CARLA ego_pos and ego_rot must both contain three values")
            results["lidar2global"] = carla_pose_to_lidar2global(ego_pos, ego_rot)
        else:
            results["lidar2global"] = np.eye(4)

        speed = float(info.get("ego_velo", 0.0) or 0.0)
        command = info.get("sparsedrive_command")
        if command is None:
            command = coarse_command_to_road_option(info.get("command", -1))
        ego_status = np.zeros(10, dtype=np.float32)
        accel = np.asarray(info.get("ego_accel", []), dtype=np.float32).reshape(-1)
        angular_velocity = np.asarray(
            info.get("ego_angular_velocity", []), dtype=np.float32
        ).reshape(-1)
        if accel.size == 3:
            ego_status[:3] = accel
        else:
            ego_status[0] = float(info.get("accelerate", 0.0) or 0.0)
        if angular_velocity.size == 3:
            ego_status[3:6] = angular_velocity
        ego_status[6:9] = np.array([speed, 0.0, 0.0], dtype=np.float32)
        results["ego_status"] = ego_status
        results["gt_ego_fut_cmd"] = make_command_onehot(command)
        # Prefer the real route waypoint supplied by the CARLA helper
        # (ego-frame [x_right, y_forward]); fall back to the coarse
        # command-synthesised point when no route target is available.
        tp = info.get("target_point")
        tp = np.asarray(tp, dtype=np.float32) if tp is not None else None
        if tp is None or tp.shape != (2,) or not bool(np.all(np.isfinite(tp))):
            tp = command_target_point(command)
            self._tp_source = "command"
        else:
            self._tp_source = str(info.get("target_point_source", "route"))
        results["tp_near"] = tp
        results["tp_far"] = tp
        results["route"] = np.zeros((0, 2), dtype=np.float32)

        stacked = np.stack(results["img"], axis=-1)
        results["img_shape"] = stacked.shape
        results["ori_shape"] = stacked.shape
        results["pad_shape"] = stacked.shape
        results["aug_config"] = self._aug_config
        return results

    # Surface the whole final 64 path x 16 velocity vocabulary. The pipeline
    # runs two independent proposal branches over these 1024 modes: native
    # End2End Top-30 and all trajectories in the leaderboard target-point
    # region. Both batches are ranked by PDMS before their winners compete.
    TOP_K = 1024

    def process(self, obs: Dict[str, Any], info: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Run one native Bench2Drive SparseDrive forward.

        Returns (proposals[N,T,2], scores[N]) in CARLA ego-local coordinates
        ([x_right, y_forward]), where N == min(TOP_K, num_modes). For the V13
        head this is all 1024 final modes, ordered by the model's post-rescore
        score (proposals[0] is the model's own argmax winner). Falls back to the
        single ``traj_final`` (N==1) if the decoder did not expose candidates.
        """
        torch = self._torch
        from mmcv.parallel.collate import collate as mm_collate_to_batch_form

        results = self._build_results(obs, info)
        model_timestamp = float(results["timestamp"])
        results = self._pipeline(results)
        batch = mm_collate_to_batch_form([results], samples_per_gpu=1)
        for key, data in list(batch.items()):
            if key != "img_metas" and torch.is_tensor(data):
                batch[key] = data.to(self._device)

        # CWD guard: the planning decoder lazily np.load()s the vel_seq anchor by
        # its config-relative path on the first forward.
        with torch.no_grad(), self._in_repo():
            output = self._model(**batch)
        self._frame_idx += 1

        img_bbox = output[0]["img_bbox"]
        # traj_final is the combined space-time plan: [T=6, 2] at 0.5 s spacing,
        # already in ego [x_right, y_forward]. It is
        # the model's argmax candidate — kept as the rank-0 proposal below.
        traj = img_bbox["traj_final"].detach().cpu().numpy().astype(np.float32)  # [T, 2]

        # Top-K candidates: the V13LatLonTrajDecoder now exposes every mode's
        # trajectory (traj_all [N, T, 2]) and its post-rescore score (traj_score
        # [N], collision/rescore-masked modes already zeroed). Rank by score and
        # keep the best TOP_K; proposals[0] == traj_final (the argmax winner).
        traj_all_t = img_bbox.get("traj_all")
        traj_score_t = img_bbox.get("traj_score")
        candidate_lat = None
        candidate_lon = None
        if traj_all_t is not None and traj_score_t is not None:
            traj_all = traj_all_t.detach().cpu().numpy().astype(np.float32)     # [N, T, 2]
            traj_score = traj_score_t.detach().cpu().numpy().astype(np.float32)  # [N]
            k = int(min(self.TOP_K, traj_all.shape[0]))
            top_idx = np.argsort(-traj_score)[:k]
            proposals = traj_all[top_idx].astype(np.float32)  # [k, T, 2]
            scores = traj_score[top_idx].astype(np.float32)   # [k]

            # Keep the native lateral path and longitudinal speed head aligned
            # with every returned top-K trajectory. V13 flattens its
            # [num_lat, num_lon] grid in row-major order, so each flattened
            # trajectory index maps back to (lat_idx, lon_idx) via divmod.
            # The controller can then track the candidate AgenticDriving
            # actually selected instead of silently using the model argmax's
            # lat_reg_final/lon_reg_final heads.
            lat_all_t = img_bbox.get("lat_reg")
            lon_all_t = img_bbox.get("lon_reg")
            if lat_all_t is not None and lon_all_t is not None:
                lat_all = lat_all_t.detach().cpu().numpy().astype(np.float32)
                lon_all = lon_all_t.detach().cpu().numpy().astype(np.float32)
                if lat_all.ndim >= 3 and lon_all.ndim >= 2 and len(lon_all) > 0:
                    lat_idx = top_idx // len(lon_all)
                    lon_idx = top_idx % len(lon_all)
                    if np.all(lat_idx < len(lat_all)):
                        candidate_lat = lat_all[lat_idx]
                        candidate_lon = lon_all[lon_idx]
        else:
            # Decoder did not expose candidates (e.g. a different decode path):
            # fall back to the single argmax trajectory, N == 1.
            proposals = traj[None, ...].astype(np.float32)  # [1, T, 2]
            scores = np.array([1.0], dtype=np.float32)

        lat = img_bbox.get("lat_reg_final")
        lon = img_bbox.get("lon_reg_final")
        self.last_debug = {
            "planner": "sparsedrivev2_b2d",
            "coordinate_note": {"raw": "traj_final [x_right, y_forward] ego", "carla_local": "[x_right, y_forward]"},
            "plan_dt_sec": self.plan_dt,
            "input_info": {
                "carla_timestamp": float(info.get("timestamp", 0.0) or 0.0),
                "model_timestamp": model_timestamp,
                "coarse_command": int(command) if (command := info.get("command", -1)) is not None else None,
                "road_option": int(
                    info.get(
                        "sparsedrive_command",
                        coarse_command_to_road_option(info.get("command", -1)),
                    )
                ),
                "command_onehot": results["gt_ego_fut_cmd"].tolist(),
                "target_point": _float_list(results["tp_near"]),
                "target_point_source": getattr(self, "_tp_source", "command"),
                "route_distance_m": info.get("route_distance_m"),
                "route_lane_key": info.get("route_lane_key"),
                "ego_lane_key": info.get("ego_lane_key"),
                "route_lane_relation": info.get("route_lane_relation"),
                "route_target_lookahead_m": info.get("route_target_lookahead_m"),
                "route_reentry_active": bool(info.get("route_reentry_active", False)),
                "route_reentry_primitive": info.get("route_reentry_primitive"),
                "route_reentry_side": info.get("route_reentry_side"),
                "route_reentry_trigger_reason": info.get(
                    "route_reentry_trigger_reason"
                ),
                "route_reentry_command_override": bool(
                    info.get("route_reentry_command_override", False)
                ),
                "ego_velo_mps": float(info.get("ego_velo", 0.0) or 0.0),
                "ego_accel": _float_list(info.get("ego_accel", [])),
                "ego_angular_velocity": _float_list(info.get("ego_angular_velocity", [])),
                "ego_pos": _float_list(info.get("ego_pos", [])),
                "ego_rot": _float_list(info.get("ego_rot", [])),
                "temporal": self.use_temporal,
            },
            "lat_reg_final": (lat.detach().cpu().numpy().astype(float).tolist() if lat is not None else None),
            "lon_reg_final_speed_mps": (lon.detach().cpu().numpy().astype(float).tolist() if lon is not None else None),
            "traj_final_xy": traj.astype(float).tolist(),
            "carla_local_plan_xy": proposals[0].astype(float).tolist(),
            "candidate_plans_xy": proposals.astype(float).tolist(),
            "candidate_lat_reg": (
                candidate_lat.astype(float).tolist()
                if candidate_lat is not None
                else None
            ),
            "candidate_lon_reg_speed_mps": (
                candidate_lon.astype(float).tolist()
                if candidate_lon is not None
                else None
            ),
            "num_candidates": int(proposals.shape[0]),
            "candidate_scores": scores.astype(float).tolist(),
        }
        return proposals, scores
