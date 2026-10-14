"""Portable configuration for the single released inference pipeline."""
import os
from pathlib import Path


def prepare_config(cfg, root):
    root = Path(root).resolve()
    if "sparsedrivev2" not in cfg:
        raise ValueError("Release requires SparseDriveV2")
    unsupported = {"uniad", "bridgedrive", "transfuser", "simlingo", "rap", "drivor", "primitives"} & cfg.keys()
    if unsupported:
        raise ValueError(f"Unsupported planner configuration: {sorted(unsupported)}")
    perception = cfg["perception"]
    selector = cfg["agenticdriving"]["recovery_selector"]
    if perception.get("source") != "bridgedrivev2":
        raise ValueError("Release requires BridgeDriveV2 sensor detection")
    if selector.get("backend") != "anthropic" or selector.get("model_id") != "claude-opus-5":
        raise ValueError("Release requires Anthropic claude-opus-5")
    if cfg["verifier"].get("type") != "pdms":
        raise ValueError("Release requires verifier.type=pdms")
    if (perception.get("disagreement_vlm") or {}).get("enabled"):
        raise ValueError("Release supports stationary_vru_review only")
    bridge = perception["bridgedrive"]
    if not bridge.get("full_camera_rig"):
        raise ValueError("The detection_ft specialists require the full camera rig")
    if cfg["carla"]["control"]["mode"] != "sparsedrivev2_pid":
        raise ValueError("Release requires sparsedrivev2_pid")
    for node, keys in [(cfg["sparsedrivev2"], ["repo_root"]),
                       (bridge, ["lead_root", "leaderboard_root", "scenario_runner_root", "carla_python_root", "checkpoint_dir"]),
                       (cfg["carla"], ["output_dir"])]:
        for key in keys:
            path = Path(node[key]).expanduser()
            node[key] = str(path if path.is_absolute() else root / path)
    os.environ["AGENTICDRIVING_ROOT"] = str(root)
    os.environ["BRIDGEDRIVE_FINETUNED_DETECTION"] = "1"
    os.environ["BRIDGEDRIVE_DETECTION_CODE"] = str(root / "detection")
    os.environ.setdefault("AGENTICDRIVING_RUN_DIR", cfg["carla"]["output_dir"])
