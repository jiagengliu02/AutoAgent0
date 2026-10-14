"""Crash-tolerant, per-frame video recording for CARLA agents.

The writer keeps one ffmpeg process alive and emits fragmented MP4.  Every
encoded frame is its own MP4 fragment and packets are flushed immediately, so
an interrupted run leaves a playable file containing all complete fragments.
"""

from __future__ import annotations

import logging
import queue
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Optional

import numpy as np


def video_frame_is_due(frame_index: int, frame_stride: int) -> bool:
    """Return True for frames retained by fixed-stride speed-up recording."""
    stride = max(1, int(frame_stride))
    return int(frame_index) % stride == 0


class StreamingVideoWriter:
    """Write RGB frames to a crash-tolerant fragmented MP4 in the background."""

    _STOP = object()

    def __init__(
        self,
        path: Path,
        fps: float,
        *,
        crf: int = 28,
        preset: str = "veryfast",
        scale: Optional[float] = None,
        threads: int = 2,
        queue_size: int = 2,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin is None:
            raise FileNotFoundError("ffmpeg is required for streaming video")

        self.path = Path(path)
        self.fps = max(1.0, float(fps))
        self.crf = int(crf)
        self.preset = str(preset)
        self.scale = None if scale is None else float(scale)
        self.threads = max(1, int(threads))
        self.logger = logger or logging.getLogger(__name__)
        self._ffmpeg_bin = ffmpeg_bin
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, int(queue_size)))
        self._process: Optional[subprocess.Popen] = None
        self._worker: Optional[threading.Thread] = None
        self._shape = None
        self._error: Optional[BaseException] = None
        self._closed = False
        self.frames_written = 0

    def _command(self, width: int, height: int) -> list[str]:
        if self.scale is not None and 0.0 < self.scale < 1.0:
            video_filter = (
                f"scale=trunc(iw*{self.scale}/2)*2:"
                f"trunc(ih*{self.scale}/2)*2,format=yuv420p"
            )
        else:
            even_width = width - (width % 2)
            even_height = height - (height % 2)
            video_filter = (
                f"crop={even_width}:{even_height}:0:0,format=yuv420p"
            )

        keyframe_interval = max(1, int(round(self.fps)))
        return [
            self._ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{self.fps:g}",
            "-i",
            "pipe:0",
            "-vf",
            video_filter,
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            str(self.crf),
            "-preset",
            self.preset,
            "-threads",
            str(self.threads),
            "-tune",
            "zerolatency",
            "-g",
            str(keyframe_interval),
            "-keyint_min",
            str(keyframe_interval),
            "-sc_threshold",
            "0",
            "-bf",
            "0",
            "-movflags",
            "+empty_moov+default_base_moof+frag_every_frame",
            "-flush_packets",
            "1",
            "-f",
            "mp4",
            str(self.path),
        ]

    def _start(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        self._shape = (height, width, 3)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._process = subprocess.Popen(
            self._command(width, height),
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._worker = threading.Thread(
            target=self._run,
            name=f"video-{self.path.stem}",
            daemon=True,
        )
        self._worker.start()

    def _run(self) -> None:
        process = self._process
        if process is None or process.stdin is None:
            return
        try:
            while True:
                item = self._queue.get()
                if item is self._STOP:
                    break
                process.stdin.write(item.tobytes())
                process.stdin.flush()
                self.frames_written += 1
        except (BrokenPipeError, OSError, ValueError) as exc:
            self._error = exc
        finally:
            try:
                process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            return_code = process.wait()
            stderr = b""
            if process.stderr is not None:
                stderr = process.stderr.read()
                process.stderr.close()
            if return_code != 0 and self._error is None:
                message = stderr.decode("utf-8", errors="replace").strip()
                self._error = RuntimeError(
                    f"ffmpeg exited with status {return_code}: {message}"
                )

    def append(self, frame: np.ndarray) -> bool:
        """Queue one RGB uint8 frame, blocking only if the small queue is full."""
        if self._closed or self._error is not None:
            return False
        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"expected HxWx3 RGB frame, got {array.shape}")
        array = np.ascontiguousarray(array, dtype=np.uint8)
        if self._process is None:
            self._start(array)
        if array.shape != self._shape:
            self.logger.warning(
                "Skipping video frame with changed shape for %s: %s != %s",
                self.path,
                array.shape,
                self._shape,
            )
            return False

        # Copy because CARLA/model buffers may be reused after this call.
        queued = array.copy()
        while self._error is None and not self._closed:
            try:
                self._queue.put(queued, timeout=0.25)
                return True
            except queue.Full:
                continue
        return False

    def close(self, timeout: float = 30.0) -> bool:
        """Flush queued frames and close the MP4; safe to call repeatedly."""
        if self._closed:
            return self._error is None
        self._closed = True
        worker = self._worker
        if worker is None:
            return True

        while worker.is_alive():
            try:
                self._queue.put(self._STOP, timeout=0.25)
                break
            except queue.Full:
                if self._error is not None:
                    break
        worker.join(timeout=max(0.0, float(timeout)))
        if worker.is_alive():
            self.logger.error("Timed out finalizing streaming video %s", self.path)
            process = self._process
            if process is not None and process.poll() is None:
                process.terminate()
            worker.join(timeout=5.0)
        if self._error is not None:
            self.logger.error("Streaming video failed for %s: %s", self.path, self._error)
            return False
        return not worker.is_alive()
