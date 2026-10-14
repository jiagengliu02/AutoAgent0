"""Leaderboard entry point for AgenticDriving.

The leaderboard loads this file as ``--agent``. It:
  - puts adapters/carla_agent/, the repo root, and AGENTICDRIVING_ROOT on sys.path;
  - applies the in-memory leaderboard hooks before any route starts;
  - exposes ``AgenticDrivingCarlaAgent`` / ``get_entry_point``;
  - encodes recordings as H.264 so they play in VSCode/browsers.

Trajectory tracking uses the SparseDriveV2 PID controller in control.py.
"""
from pathlib import Path
import os
import sys

import numpy as np

_AGENT_DIR = Path(__file__).resolve().parent
# This file now lives at <repo>/adapters/carla_agent/, so the repo root — which
# holds the `agenticdriving` pipeline package + `leaderboard_hooks` — is two
# levels up. Env var wins so the tree can be relocated freely.
_REPO_ROOT = _AGENT_DIR.parent.parent
_AGENTICDRIVING_ROOT = Path(os.environ.get("AGENTICDRIVING_ROOT", str(_REPO_ROOT)))

for _path in (_AGENT_DIR, _REPO_ROOT, _AGENTICDRIVING_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# Re-apply the two leaderboard behaviours AgenticDriving needs, in-memory, so the
# fail2drive checkout on PYTHONPATH stays pristine. Must run before any route
# starts (this module is imported by the leaderboard evaluator before it builds
# scenarios).
import leaderboard_hooks  # noqa: E402
leaderboard_hooks.apply()

import agenticdriving_carla_helper as _helper  # noqa: E402

from agenticdriving_carla import AgenticDrivingCarlaAgent, get_entry_point  # noqa: E402


_ORIGINAL_WRITE_RGB_VIDEO = _helper.write_rgb_video


def _video_env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _write_rgb_video_h264(out_path, frames, fps, crf=None, preset=None, scale=None):
    """Encode recordings as H.264 so they play in VSCode/browsers (mp4v does not).

    Size/quality controls, in precedence order:
      1. env vars (explicit operator override): AGENTIC_VIDEO_CRF,
         AGENTIC_VIDEO_SCALE, AGENTIC_VIDEO_PRESET.
      2. kwargs passed by finalize_recording from the run config
         (carla.recording.video_quality / crf / preset / video_scale).
      3. legacy defaults: crf 28, scale 0.5, preset veryfast.
    18=near-lossless CRF, 23=default, 28=noticeably smaller. scale is an
    encode-time downscale only (buffers keep full resolution); 1.0 disables.
    """
    if not frames:
        return
    try:
        import imageio_ffmpeg

        crf = int(_video_env_float(
            "AGENTIC_VIDEO_CRF", 28 if crf is None else crf
        ))
        scale = _video_env_float(
            "AGENTIC_VIDEO_SCALE", 0.5 if scale is None else scale
        )
        preset = os.environ.get(
            "AGENTIC_VIDEO_PRESET", "veryfast" if preset is None else preset
        )

        height, width = frames[0].shape[:2]
        output_params = ["-crf", str(crf), "-preset", preset]
        # Default: let imageio pad odd dims to even (needed by yuv420p).
        macro_block_size = 2
        if 0.0 < scale < 1.0:
            # Our own scale filter already forces even output via trunc(.../2)*2.
            # imageio would otherwise inject a second -vf for odd input and the two
            # -vf options would collide, so disable its rounding here.
            output_params += [
                "-vf",
                f"scale=trunc(iw*{scale}/2)*2:trunc(ih*{scale}/2)*2",
            ]
            macro_block_size = 1

        writer = imageio_ffmpeg.write_frames(
            str(out_path),
            (width, height),
            fps=float(fps),
            codec="libx264",
            pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p",
            macro_block_size=macro_block_size,
            output_params=output_params,
        )
        writer.send(None)
        try:
            for frame in frames:
                writer.send(np.ascontiguousarray(frame, dtype=np.uint8))
        finally:
            writer.close()
    except Exception:
        _helper.LOG.exception(
            "H.264 encode failed for %s; falling back to mp4v", out_path
        )
        _write_rgb_video_mp4v_scaled(out_path, frames, fps, scale=scale)


def _write_rgb_video_mp4v_scaled(out_path, frames, fps, crf=None, preset=None, scale=None):
    """mp4v fallback (no imageio_ffmpeg). Can't set CRF, but still honours
    the scale (env var over kwarg) by downscaling frames so the file shrinks."""
    scale = _video_env_float(
        "AGENTIC_VIDEO_SCALE", 0.5 if scale is None else scale
    )
    if not (0.0 < scale < 1.0):
        _ORIGINAL_WRITE_RGB_VIDEO(out_path, frames, fps)
        return
    import cv2

    h0, w0 = frames[0].shape[:2]
    w = max(2, (int(w0 * scale) // 2) * 2)
    h = max(2, (int(h0 * scale) // 2) * 2)
    scaled = [
        cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in frames
    ]
    _ORIGINAL_WRITE_RGB_VIDEO(out_path, scaled, fps)


_helper.write_rgb_video = _write_rgb_video_h264

__all__ = ["AgenticDrivingCarlaAgent", "get_entry_point"]
