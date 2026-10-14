"""Subprocess services for SparseDriveV2 planning and BridgeDrive perception."""
from __future__ import annotations

import os
import pickle
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from agenticdriving.planners.base import PlannerResult, PlannerService








class SparseDriveV2SubprocessCarlaService(PlannerService):
    """SparseDriveV2 inference in its dedicated Python env.

    The CARLA leaderboard agent must run in the CARLA/fail2drive Python, while
    SparseDriveV2's Bench2Drive branch needs mmcv/mmdet from ``sparsedrive_b2d``.
    This service keeps the CARLA side in-process and sends only obs/info to a
    long-lived SparseDriveV2 worker subprocess.
    """

    history_frames: int = 4

    def __init__(self, agent, *, cfg: Dict[str, Any]) -> None:
        self._agent = agent
        self._cfg = dict(cfg)
        self._proc: Optional[subprocess.Popen[bytes]] = None
        self._stderr_handle = None

    def _worker_env(self) -> Dict[str, str]:
        repo_root = str(self._cfg["repo_root"])
        agentic_root = os.environ.get("AGENTICDRIVING_ROOT", "")
        env = os.environ.copy()
        env["SPARSEDRIVEV2_REPO_ROOT"] = repo_root
        env["SPARSEDRIVEV2_CKPT"] = str(self._cfg["checkpoint"])
        env["SPARSEDRIVEV2_CONFIG"] = str(
            self._cfg.get("config", "projects/configs/sparsedrive_stage2.py")
        )
        # inference.py always builds the model on local device index 0 and relies
        # on CUDA_VISIBLE_DEVICES to pin which *physical* GPU that is. The per-shard
        # config gives `device: cuda:<gpu>` (nvidia-smi index, following the CARLA
        # slot), so translate it into a CUDA_VISIBLE_DEVICES mask and hand the worker
        # cuda:0. Without this mask every worker sees all GPUs and piles onto GPU0.
        device = str(self._cfg.get("device", "cuda")).strip()
        if device.startswith("cuda:"):
            env["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]
            env["SPARSEDRIVEV2_DEVICE"] = "cuda:0"
        else:
            env["SPARSEDRIVEV2_DEVICE"] = device  # "cuda" (inherit mask) or "cpu"
        env.setdefault("SPARSEDRIVEV2_TEMPORAL", "1")
        pythonpath = [p for p in (agentic_root, repo_root, env.get("PYTHONPATH", "")) if p]
        env["PYTHONPATH"] = ":".join(pythonpath)
        return env

    def _ensure_proc(self) -> subprocess.Popen[bytes]:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc
        if self._proc is not None:
            raise RuntimeError(f"SparseDriveV2 worker exited with code {self._proc.returncode}")

        python_bin = str(self._cfg.get("python_bin", sys.executable))
        repo_root = str(self._cfg["repo_root"])
        log_dir = Path(
            self._cfg.get(
                "worker_log_dir",
                Path(os.environ.get("AGENTICDRIVING_ROOT", ".")) / "output" / "logs",
            )
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "sparsedrivev2_worker.log"
        self._stderr_handle = open(log_path, "ab", buffering=0)
        self._proc = subprocess.Popen(
            [python_bin, "-m", "agenticdriving.planners.sparsedrivev2.worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_handle,
            cwd=repo_root,
            env=self._worker_env(),
        )
        return self._proc

    @staticmethod
    def _write_message(pipe, message: Any) -> None:
        payload = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
        pipe.write(struct.pack("<Q", len(payload)))
        pipe.write(payload)
        pipe.flush()

    @staticmethod
    def _read_message(pipe) -> Any:
        header = pipe.read(8)
        if len(header) != 8:
            raise EOFError("SparseDriveV2 worker closed stdout")
        size = struct.unpack("<Q", header)[0]
        payload = bytearray()
        while len(payload) < size:
            chunk = pipe.read(size - len(payload))
            if not chunk:
                raise EOFError("SparseDriveV2 worker closed stdout mid-message")
            payload.extend(chunk)
        return pickle.loads(payload)

    def process(
        self,
        obs: Dict[str, Any],
        info: Dict[str, Any],
        info_history: Sequence[Dict[str, Any]],
        extra: Dict[str, Any],
    ) -> PlannerResult:
        proc = self._ensure_proc()
        if proc.stdin is None or proc.stdout is None:
            raise RuntimeError("SparseDriveV2 worker pipes are unavailable")
        self._write_message(proc.stdin, {"obs": obs, "info": info})
        response = self._read_message(proc.stdout)
        if not isinstance(response, dict):
            raise RuntimeError(f"Unexpected SparseDriveV2 worker response: {type(response)}")
        if response.get("error"):
            raise RuntimeError(str(response["error"]))

        proxy = getattr(self._agent, "_sparsedrive", None)
        if proxy is not None:
            proxy.last_debug = dict(response.get("last_debug") or {})
            proxy.num_poses = int(response.get("num_poses") or proxy.num_poses)
        if hasattr(self._agent, "_sparsedrive_output_num_poses"):
            self._agent._sparsedrive_output_num_poses = int(
                response.get("num_poses") or self._agent._sparsedrive_output_num_poses
            )

        return PlannerResult(
            proposals=np.asarray(response["proposals"], dtype=np.float32),
            scores=np.asarray(response["scores"], dtype=np.float32),
        )

    def finalize(self) -> None:
        proc = self._proc
        if proc is not None:
            try:
                if proc.stdin is not None and proc.poll() is None:
                    self._write_message(proc.stdin, {"message": "Done"})
                    proc.stdin.close()
            except Exception:
                pass



class BridgeDrivePerceptionCarlaService:
    """BridgeDrive detection_ft in its dedicated Python environment.

    Same isolation pattern as SparseDriveV2SubprocessCarlaService: the CARLA
    side stays in-process; a long-lived worker (LEAD venv, torch 2.5) hosts
    the model + LEAD preprocessing. Extra wrinkle: the worker needs a one-time
    episode init (global route plan + town) before the per-tick stream, sent
    lazily before the first tick.
    """

    history_frames: int = 4

    def __init__(self, agent, *, cfg: Dict[str, Any]) -> None:
        self._agent = agent
        self._cfg = dict(cfg)
        self._proc: Optional[subprocess.Popen[bytes]] = None
        self._stderr_handle = None
        self._initialized = False

    def _worker_env(self) -> Dict[str, str]:
        lead_root = str(self._cfg.get("lead_root", str(Path(__file__).resolve().parents[2] / "third_party/LEAD")))
        agentic_root = os.environ.get("AGENTICDRIVING_ROOT", "")
        env = os.environ.copy()
        leaderboard_root = str(
            self._cfg.get("leaderboard_root")
            or (
                f"{lead_root}/3rd_party/leaderboard"
                if Path(f"{lead_root}/3rd_party/leaderboard").is_dir()
                else f"{lead_root}/3rd_party/fail2drive/leaderboard"
            )
        )
        scenario_runner_root = str(
            self._cfg.get("scenario_runner_root")
            or (
                f"{lead_root}/3rd_party/scenario_runner"
                if Path(f"{lead_root}/3rd_party/scenario_runner").is_dir()
                else f"{lead_root}/3rd_party/fail2drive/scenario_runner"
            )
        )
        inherited_carla_root = env.get("CARLA_ROOT", "")
        carla_python_root = str(
            self._cfg.get("carla_python_root")
            or (
                f"{inherited_carla_root}/PythonAPI/carla"
                if inherited_carla_root
                else f"{lead_root}/3rd_party/CARLA_F2D/PythonAPI/carla"
            )
        )
        env["BRIDGEDRIVE_CHECKPOINT_DIR"] = str(self._cfg["checkpoint_dir"])
        # Env contract of LEAD's runner script; ClosedLoopConfig hard-fails on
        # unknown keys so these strings must track LEAD's config surface.
        env.setdefault(
            "LEAD_CLOSED_LOOP_CONFIG",
            "steer_modality=route throttle_modality=target_speed "
            "brake_modality=target_speed step_num="
            f"{int(self._cfg.get('step_num', 20))} diffusion_speed=False",
        )
        env.setdefault("LEAD_TRAINING_CONFIG", "diffusion_speed=False")
        env["LEAD_PROJECT_ROOT"] = lead_root
        # ffmpeg presence is checked by SensorAgent.setup.
        ffmpeg_dir = str(self._cfg.get("ffmpeg_dir", ""))
        env["PATH"] = ":".join(p for p in (ffmpeg_dir, env.get("PATH", "")) if p)
        # Same CUDA translation as the SparseDriveV2 service: LEAD hardcodes
        # cuda:0 and relies on CUDA_VISIBLE_DEVICES for physical placement.
        device = str(self._cfg.get("device", "cuda")).strip()
        if device.startswith("cuda:"):
            env["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]
        pythonpath = [
            p
            for p in (
                agentic_root,
                lead_root,
                leaderboard_root,
                scenario_runner_root,
                carla_python_root,
                env.get("PYTHONPATH", ""),
            )
            if p
        ]
        env["PYTHONPATH"] = ":".join(pythonpath)
        return env

    def _ensure_proc(self) -> subprocess.Popen[bytes]:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc
        if self._proc is not None:
            raise RuntimeError(
                f"BridgeDrive worker exited with code {self._proc.returncode}"
            )
        python_bin = str(self._cfg["python_bin"])
        log_dir = Path(
            self._cfg.get(
                "worker_log_dir",
                Path(os.environ.get("AGENTICDRIVING_ROOT", ".")) / "output" / "logs",
            )
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        self._stderr_handle = open(log_dir / "bridgedrive_worker.log", "ab", buffering=0)
        self._proc = subprocess.Popen(
            [python_bin, "-m", "agenticdriving.planners.bridgedrive.worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_handle,
            cwd=os.environ.get("AGENTICDRIVING_ROOT", "."),
            env=self._worker_env(),
        )
        return self._proc

    _write_message = staticmethod(SparseDriveV2SubprocessCarlaService._write_message)
    _read_message = staticmethod(SparseDriveV2SubprocessCarlaService._read_message)

    def _serialized_plans(self):
        """Leaderboard plans -> plain data (carla.Transform / RoadOption do
        not pickle); the worker rebuilds them (runtime.deserialize_global_plans)."""
        gps = [
            (dict(pos), int(getattr(cmd, "value", cmd)))
            for pos, cmd in (self._agent._global_plan or [])
        ]
        world = []
        for transform, cmd in self._agent._global_plan_world_coord or []:
            loc, rot = transform.location, transform.rotation
            world.append(
                (
                    (float(loc.x), float(loc.y), float(loc.z)),
                    (float(rot.pitch), float(rot.yaw), float(rot.roll)),
                    int(getattr(cmd, "value", cmd)),
                )
            )
        return gps, world

    def _ensure_initialized(self, proc: subprocess.Popen[bytes]) -> None:
        if self._initialized:
            return
        from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

        town = CarlaDataProvider.get_map().name.split("/")[-1]
        gps, world = self._serialized_plans()
        self._write_message(
            proc.stdin,
            {
                "init": {
                    "global_plan": gps,
                    "global_plan_world_coord": world,
                    "town": town,
                    "checkpoint_dir": str(self._cfg["checkpoint_dir"]),
                }
            },
        )
        response = self._read_message(proc.stdout)
        if not (isinstance(response, dict) and response.get("ready")):
            raise RuntimeError(f"BridgeDrive worker init failed: {response!r}")
        self._initialized = True



    finalize = SparseDriveV2SubprocessCarlaService.finalize

    def _round_trip(self, message: Dict[str, Any]) -> Dict[str, Any]:
        proc = self._ensure_proc()
        if proc.stdin is None or proc.stdout is None:
            raise RuntimeError("BridgeDrive perception worker pipes unavailable")
        self._ensure_initialized(proc)
        started = time.perf_counter()
        self._write_message(proc.stdin, message)
        response = self._read_message(proc.stdout)
        self.last_round_trip_sec = float(time.perf_counter() - started)
        if not isinstance(response, dict):
            raise RuntimeError(
                f"Unexpected BridgeDrive perception response: {type(response)}"
            )
        if response.get("error"):
            raise RuntimeError(str(response["error"]))
        return response

    @staticmethod
    def _payload(obs: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "obs": {"input_data": obs.get("bd_input_data") or {}},
            "info": {"bd_prev_control": info.get("bd_prev_control")},
        }

    def detect(
        self, obs: Dict[str, Any], info: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Tick + full forward; returns the harvest or None (warm-up tick)."""
        response = self._round_trip(self._payload(obs, info))
        return (response.get("last_debug") or {}).get("perception")

    def tick(self, obs: Dict[str, Any], info: Dict[str, Any]) -> None:
        """Stateful preprocessing only (no forward) — the stride filler."""
        self._round_trip({"tick": self._payload(obs, info)})
