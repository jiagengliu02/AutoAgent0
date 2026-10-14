"""Calibrated object crops and the optional shared-backend VLM call."""
from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile

import cv2
import numpy as np

from agenticdriving.planners.sparsedrivev2.carla_inputs import carla_transform_matrix
from agenticdriving.prompts.perception_box import build_perception_box_prompt
from agenticdriving.vlm.parsing import try_parse_json


def render_box_evidence(agent, images, specs, info):
    """Use actual sensor mount rotations, CARLA axes and the current ego pose.

    Agent extents are full lengths. World coordinates are projected into the
    current image even when the matched box originated on the previous tick.
    """
    center = np.asarray(agent["agent_pos_world"], dtype=float)
    length, width = np.asarray(agent["agent_extent"][:2]) * .5
    height = float(agent.get("agent_height", 1.5)) * .5
    offsets = np.asarray([[x, y, z] for x in (-length, length)
                          for y in (-width, width) for z in (-height, height)])
    yaw = float(agent.get("agent_heading", 0))
    c, s = math.cos(yaw), math.sin(yaw)
    offsets[:, :2] = offsets[:, :2] @ np.array([[c, s], [-s, c]])
    corners = np.column_stack([center + offsets, np.ones(8)])
    ego2world = carla_transform_matrix(info["ego_pos"], info["ego_rot"])
    choices = []
    for name, frame in images.items():
        spec = specs.get(name)
        if not spec:
            continue
        mount = carla_transform_matrix([spec.get(k, 0) for k in ("x", "y", "z")],
                                      np.radians([spec.get(k, 0) for k in ("roll", "pitch", "yaw")]))
        camera = (np.linalg.inv(ego2world @ mount) @ corners.T).T[:, :3]
        # CARLA camera: x forward, y right, z up; optical: x right, y down, z forward.
        if not np.isfinite(camera).all() or np.any(camera[:, 0] <= .1):
            continue
        frame = np.asarray(frame, dtype=np.uint8)
        h, w = frame.shape[:2]
        focal = w / (2 * math.tan(math.radians(float(spec["fov"])) / 2))
        uv = np.column_stack([focal*camera[:, 1]/camera[:, 0]+w/2,
                              -focal*camera[:, 2]/camera[:, 0]+h/2])
        low, high = uv.min(axis=0), uv.max(axis=0)
        left, top = np.maximum(np.floor(low), 0).astype(int)
        right, bottom = np.minimum(np.ceil(high), [w-1, h-1]).astype(int)
        if right-left < 6 or bottom-top < 6:
            continue
        visibility = (right-left)*(bottom-top) / max(1, np.prod(high-low))
        if visibility < .5:
            continue
        overlay = frame.copy()
        cv2.rectangle(overlay, (left, top), (right, bottom), (255, 220, 0), 2)
        cv2.putText(overlay, "REVIEW", (left, max(14, top-5)), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (255, 220, 0), 1)
        pad = max(12, int(max(right-left, bottom-top)*.25))
        crop = frame[max(0, top-pad):min(h, bottom+pad+1), max(0, left-pad):min(w, right+pad+1)].copy()
        choices.append(((right-left)*(bottom-top), name, overlay, crop))
    if not choices:
        return None
    _, name, overlay, crop = max(choices, key=lambda row: row[0])
    return name, overlay, crop


def check_perception_box(host, *, candidate, camera_images, camera_specs, info,
                         request_id, debug_dir, max_new_tokens, timeout_sec,
                         save_debug_artifacts):
    visual = render_box_evidence(candidate["agent"], camera_images, camera_specs, info)
    if visual is None:
        return {"decision": "uncertain", "error": "candidate_not_visible_in_calibrated_camera"}
    selector = host._ensure_selector()
    if selector is None:
        return {"decision": "uncertain", "error": "selector_unavailable"}
    camera_name, overlay, crop = visual
    prompt = build_perception_box_prompt(candidate)
    debug_dir = Path(debug_dir)
    if save_debug_artifacts:
        debug_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="perception_box_") as temporary:
        folder = debug_dir if save_debug_artifacts else Path(temporary)
        paths = [folder / f"{request_id}_{kind}.jpg" for kind in ("scene", "crop")]
        for path, rgb in zip(paths, (overlay, crop)):
            if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                raise IOError(f"Could not write {path}")
        inference = {}
        try:
            inference = host._infer_optional_prompt(selector, image_paths=paths, prompt=prompt,
                                                    max_new_tokens=max_new_tokens, timeout_sec=timeout_sec)
            parsed = inference.get("parsed_output") or try_parse_json(str(inference.get("raw_output", "")))
            if not isinstance(parsed, dict):
                raise ValueError("invalid perception box JSON")
            decision, confidence = parsed.get("decision"), parsed.get("confidence")
            allowed = {"keep", "uncertain", "static" if candidate["kind"] == "stationary_vru" else "drop"}
            if (decision not in allowed or type(confidence) not in (int, float)
                    or not math.isfinite(confidence) or not 0 <= confidence <= 1):
                raise ValueError("invalid perception box decision/confidence")
            result = {"decision": decision, "confidence": confidence,
                      "reasoning": str(parsed.get("reasoning", "")), "error": inference.get("error")}
        except Exception as exc:
            result = {"decision": "uncertain", "error": str(exc)}
        host._record_vlm_summary(task="perception_box_check", frame_index=int(info.get("frame_index", 0)),
                                timestamp=float(info["timestamp"]),
                                input_fields={"candidate": candidate, "camera": camera_name, "request_id": request_id},
                                inference_result=inference, accepted_result=result,
                                error=result.get("error"), image_paths=paths)
        if save_debug_artifacts:
            (folder / f"{request_id}.json").write_text(json.dumps(
                {"candidate": candidate, "prompt": prompt, "inference": inference, "result": result},
                indent=2, default=str), encoding="utf-8")
        return result
