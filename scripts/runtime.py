"""Release preflight and evaluator time-limit handling."""
from __future__ import annotations
import json

import os

from pathlib import Path

import shutil

import sys

from omegaconf import OmegaConf

from agenticdriving.release import prepare_config

def check_runtime(cfg, root):
    """Validate all three environments before starting a CARLA process."""
    import subprocess
    sd = cfg["sparsedrivev2"]
    bd = cfg["perception"]["bridgedrive"]
    failures = []
    for command in ("nvidia-smi", "vulkaninfo", "ffmpeg", "setsid", "timeout"):
        if not shutil.which(command):
            failures.append(f"Missing command: {command}")
    probes = [
        ("Evaluator", os.environ["EVALUATOR_PYTHON_BIN"], root,
         [root, Path(os.environ["CARLA_ROOT"]) / "PythonAPI/carla",
          Path(os.environ["FAIL2DRIVE_ROOT"]) / "leaderboard",
          Path(os.environ["FAIL2DRIVE_ROOT"]) / "scenario_runner"],
         "import carla, cv2, PIL, imageio, imageio_ffmpeg, omegaconf, yaml, scipy, shapely; "
         "from adapters.carla_agent.agenticdriving_agent import AgenticDrivingCarlaAgent; "
         "from leaderboard.scenarios.scenario_manager import ScenarioManager"),
        ("SparseDriveV2", os.environ["SPARSEDRIVEV2_PYTHON_BIN"], Path(sd["repo_root"]),
         [root, sd["repo_root"]],
         "import torch, mmcv, mmdet; import projects.mmdet3d_plugin; "
         "from agenticdriving.planners.sparsedrivev2.inference import SparseDriveV2Inference; "
         "assert torch.cuda.is_available(), 'CUDA is unavailable'"),
        ("BridgeDrive", os.environ["BRIDGEDRIVE_PYTHON_BIN"], root,
         [root, root / "detection", bd["lead_root"], bd["leaderboard_root"],
          bd["scenario_runner_root"], bd["carla_python_root"]],
         "import torch, sklearn, joblib; "
         "from lead.inference.sensor_agent_bridgedrive import SensorAgent; "
         "from closed_loop_detection import ClosedLoopDetection; "
         "assert torch.cuda.is_available(), 'CUDA is unavailable'"),
    ]
    for label, python, cwd, paths, code in probes:
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(str(p) for p in paths)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["LEAD_PROJECT_ROOT"] = bd["lead_root"]
        try:
            result = subprocess.run([python, "-c", code], cwd=cwd, env=env,
                                    capture_output=True, text=True, timeout=120)
            if result.returncode:
                detail = result.stderr.strip()
                if label == "SparseDriveV2" and "deformable_aggregation" in detail:
                    detail = "Build the SparseDriveV2 CUDA extensions (docs/quickstart.md).\n" + detail
                failures.append(f"{label}:\n{detail}")
        except (OSError, subprocess.TimeoutExpired) as exc:
            failures.append(f"{label}: {exc}")
    if failures:
        raise RuntimeError("Runtime preflight failed:\n" + "\n".join(failures))


def prepare_main():
    root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.to_container(OmegaConf.load(os.environ["CONFIG"]), resolve=True)
    prepare_config(cfg, root)
    sd = cfg["sparsedrivev2"]
    bridge = cfg["perception"]["bridgedrive"]
    required = [Path(sd["repo_root"]) / sd["config"], Path(sd["repo_root"]) / sd["checkpoint"],
                Path(bridge["checkpoint_dir"]) / "model_BridgeDrive_m2_k60_0030.pth",
                Path(bridge["checkpoint_dir"]) / "config.json",
                Path(bridge["lead_root"]) / "lead/inference/sensor_agent_bridgedrive.py",
                Path(os.environ["CARLA_ROOT"]) / "CarlaUE4.sh",
                Path(os.environ["FAIL2DRIVE_ROOT"]) / "start_carla.sh",
                Path(os.environ["FAIL2DRIVE_ROOT"]) / "leaderboard/leaderboard/leaderboard_evaluator.py"]
    for line in (root / "checkpoints/SHA256SUMS").read_text().splitlines():
        _, path = line.split(maxsplit=1)
        required.append(root / path.strip())
    for group in ("pedestrian", "static"):
        selection = json.loads((root / "checkpoints/detection_ft" / group / "selection.json").read_text())
        for key in ("threshold", "bundle_hashes"):
            if key not in selection:
                raise ValueError(f"{group}/selection.json is missing {key}")
        if group == "pedestrian" and not selection.get("original_checkpoint_sha256"):
            raise ValueError("pedestrian/selection.json is missing original_checkpoint_sha256")
        required.extend(root / "checkpoints/detection_ft" / group / name for name in selection["bundle_hashes"])
    names = json.loads(Path(os.environ["EPISODES_JSON"]).read_text())["episodes"]
    limit = int(os.environ.get("MAX_ROUTES", "0"))
    if limit < 0 or not names or len(names) != len(set(names)):
        raise ValueError("Require unique nonempty episodes and MAX_ROUTES >= 0")
    names = names[:limit] if limit else names
    routes = Path(os.environ.get("ROUTES_DIR", str(Path(os.environ["FAIL2DRIVE_ROOT"]) / "fail2drive_split")))
    required.extend(routes / (name + ".xml") for name in names)
    missing = [str(p) for p in required if not p.is_file()]
    for key in ("EVALUATOR_PYTHON_BIN", "SPARSEDRIVEV2_PYTHON_BIN", "BRIDGEDRIVE_PYTHON_BIN"):
        if not shutil.which(os.environ[key]):
            missing.append(key + " (executable)")
    if missing:
        raise FileNotFoundError("Missing runtime dependencies:\n" + "\n".join(missing))
    sim_limit = float(os.environ["AGENTICDRIVING_SIM_TIME_LIMIT_SEC"])
    if not math.isfinite(sim_limit) or sim_limit < 0:
        raise ValueError("AGENTICDRIVING_SIM_TIME_LIMIT_SEC must be finite and >= 0")
    check_runtime(cfg, root)
    print(json.dumps({"pipeline": "SparseDriveV2 + BridgeDrive detection_ft + claude-opus-5",
                      "episodes": len(names), "recovery_query_budget": 15,
                      "output": os.environ["OUT_DIR"], "full_detection_camera_rig": True}, indent=2))
    if sys.argv[1] == "--dry-run":
        return
    if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        raise ValueError("Set ANTHROPIC_API_KEY before execution")
    output = Path(os.environ["OUT_DIR"])
    output.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(OmegaConf.create(cfg), output / "config.yaml")
    (output / "episodes_requested.json").write_text(json.dumps({"episodes": names}, indent=2))


import math



import runpy


def install_sim_limit(manager_class, game_time, limit_sec, marker_path):
    if not math.isfinite(limit_sec) or limit_sec <= 0:
        raise ValueError("Simulator time limit must be finite and positive")
    original_tick = manager_class._tick_scenario

    def limited_tick(self):
        elapsed = float(game_time.get_time()) - float(self.start_game_time)
        if self._running and elapsed >= limit_sec - 1e-6:
            self._running = False
            if os.environ.get("AGENTICDRIVING_SIM_LIMIT_AS_TIMEOUT") == "1":
                self.scenario.timeout_node.timeout = True
            criteria = []
            for criterion in self.scenario.get_criteria():
                criteria.append({
                    "name": criterion.name,
                    "status": criterion.test_status,
                    "actual_value": criterion.actual_value,
                    "events": [{
                        "type": str(event.get_type()),
                        "message": event.get_message(),
                    } for event in criterion.events],
                })
            marker = {
                "reason": "diagnostic_sim_time_limit",
                "limit_sec": limit_sec,
                "elapsed_sim_sec": elapsed,
                "full_route_evaluation": False,
                "criteria_at_cutoff": criteria,
            }
            marker_path.parent.mkdir(parents=True, exist_ok=True)
            marker_path.write_text(json.dumps(marker, indent=2, default=str) + "\n")
            print(f"[sim-limit] reached {elapsed:.3f}s / {limit_sec:.3f}s; "
                  "finalizing partial episode and collision records", flush=True)
            return
        return original_tick(self)

    manager_class._tick_scenario = limited_tick

def evaluate_main():
    evaluator = Path(sys.argv[1]).resolve()
    args = sys.argv[2:]
    checkpoint = None
    for i, arg in enumerate(args):
        if arg == "--checkpoint":
            checkpoint = args[i + 1]
        elif arg.startswith("--checkpoint="):
            checkpoint = arg.split("=", 1)[1]
    if checkpoint is None:
        raise ValueError("--checkpoint is required to record the diagnostic cutoff")
    limit = float(os.environ["AGENTICDRIVING_SIM_TIME_LIMIT_SEC"])
    sys.path.insert(0, str(evaluator.parent))
    from leaderboard.scenarios.scenario_manager import ScenarioManager
    from srunner.scenariomanager.timer import GameTime

    if os.environ.get("BRIDGEDRIVE_FINETUNED_DETECTION") == "1":
        # Local research rig: SparseDrive's six cameras, the specialist's
        # three calibrated cameras, and the existing recording camera.
        # This changes only this evaluator process, not the external benchmark.
        from leaderboard.autoagents import agent_wrapper
        agent_wrapper.SENSORS_LIMITS = dict(agent_wrapper.SENSORS_LIMITS)
        agent_wrapper.SENSORS_LIMITS["sensor.camera.rgb"] = 10

    if limit > 0:
        install_sim_limit(ScenarioManager, GameTime, limit,
                          Path(checkpoint).with_suffix(".sim_limit.json"))
    print(f"[sim-limit] enabled: {limit:.3f} simulator seconds", flush=True)
    sys.argv = [str(evaluator), *args]
    runpy.run_path(str(evaluator), run_name="__main__")

import argparse



import tempfile

from typing import Any

INFRACTION_KEYS = (
    "collisions_layout",
    "collisions_pedestrian",
    "collisions_vehicle",
    "red_light",
    "stop_infraction",
    "outside_route_lanes",
    "min_speed_infractions",
    "yield_emergency_vehicle_infractions",
    "scenario_timeouts",
    "route_dev",
    "vehicle_blocked",
    "route_timeout",
)

LABELS = [
    "Avg. driving score",
    "Avg. route completion",
    "Avg. infraction penalty",
    "Collisions with pedestrians",
    "Collisions with vehicles",
    "Collisions with layout",
    "Red lights infractions",
    "Stop sign infractions",
    "Off-road infractions",
    "Route deviations",
    "Route timeouts",
    "Agent blocked",
    "Yield emergency vehicles infractions",
    "Scenario timeouts",
    "Min speed infractions",
]

def _load(path: str) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}

def _atomic_write(path: str, data: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".timeout-result-", suffix=".json", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=4)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

def finalize(path: str, route_name: str, timeout_sec: int) -> None:
    data = _load(path)
    checkpoint = data.setdefault("_checkpoint", {})
    records = checkpoint.setdefault("records", [])
    timeout_message = f"Episode exceeded the {timeout_sec}s wall-clock limit."
    status = "Failed - Agent timed out (wall-clock limit)"
    simulation_event = _load(path + ".simulation_timeout.json")
    if float(simulation_event.get("limit_seconds", 0)) > 0:
        limit = float(simulation_event["limit_seconds"])
        timeout_message = f"Episode exceeded the {limit:g}s simulation-time trial limit."
        status = "Failed - Agent timed out (simulation-time trial limit)"
        data["simulation_timeout"] = simulation_event
    fallback_game = float(simulation_event.get("observed_sim_seconds", 0))
    fallback_system = float(simulation_event.get("elapsed_wall_seconds", timeout_sec))

    if records and records[0].get("scores", {}).get("score_composed") is not None:
        record = records[0]
        infractions = record.setdefault("infractions", {})
        for key in INFRACTION_KEYS:
            infractions.setdefault(key, [])
        if timeout_message not in infractions["route_timeout"]:
            infractions["route_timeout"].append(timeout_message)
        record["num_infractions"] = sum(len(infractions[key]) for key in INFRACTION_KEYS)
        record["status"] = status
        route_score = float(record.get("scores", {}).get("score_route") or 0.0)
        penalty_score = float(record.get("scores", {}).get("score_penalty") or 0.0)
        composed_score = float(record.get("scores", {}).get("score_composed") or 0.0)
        duration_game = float(record.get("meta", {}).get("duration_game") or 0.0)
        duration_system = float(record.get("meta", {}).get("duration_system") or timeout_sec)
        route_length = float(record.get("meta", {}).get("route_length") or 0.0)
    else:
        infractions = {key: [] for key in INFRACTION_KEYS}
        infractions["route_timeout"] = [timeout_message]
        record = {
            "index": 0,
            "route_id": f"{route_name}_rep0",
            "status": status,
            "num_infractions": 1,
            "infractions": infractions,
            "scores": {"score_route": 0.0, "score_penalty": 1.0, "score_composed": 0.0},
            "meta": {"route_length": 0.0, "duration_game": fallback_game, "duration_system": fallback_system},
        }
        checkpoint["records"] = [record]
        route_score, penalty_score, composed_score = 0.0, 1.0, 0.0
        duration_game, duration_system, route_length = fallback_game, fallback_system, 0.0

    global_infractions = {key: 0.0 for key in INFRACTION_KEYS}
    global_infractions["route_timeout"] = 1.0
    checkpoint["global_record"] = {
        "index": -1,
        "route_id": -1,
        "status": "Failed",
        "infractions": global_infractions,
        "scores_mean": {
            "score_composed": composed_score,
            "score_route": route_score,
            "score_penalty": penalty_score,
        },
        "scores_std_dev": {"score_composed": 0, "score_route": 0, "score_penalty": 0},
        "meta": {
            "total_length": route_length,
            "duration_game": duration_game,
            "duration_system": duration_system,
            "exceptions": [[record.get("route_id", f"{route_name}_rep0"), 0, status]],
        },
    }
    checkpoint["progress"] = [1, 1]
    data["entry_status"] = "Finished"
    data["eligible"] = True
    data["values"] = [
        str(composed_score), str(route_score), str(penalty_score),
        "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0",
        "1.0", "0.0", "0.0", "0.0", "0.0",
    ]
    data["labels"] = LABELS
    data.setdefault("sensors", [])
    _atomic_write(path, data)

def timeout_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--route-name", required=True)
    parser.add_argument("--timeout-sec", required=True, type=int)
    args = parser.parse_args()
    finalize(args.result, args.route_name, args.timeout_sec)
    print(f"[timeout] finalized {args.result}")

if __name__ == "__main__":
    command = sys.argv.pop(1)
    {"prepare": prepare_main, "evaluate": evaluate_main, "timeout": timeout_main}[command]()
