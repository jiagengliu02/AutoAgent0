from __future__ import annotations

import fcntl
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable


def clear_vlm_debug_artifacts(debug_dir: Path) -> None:
    patterns = [
        "frame_*_candidates.jpg",
        "frame_*_candidates_*.jpg",
        "frame_*_gate_*.jpg",
        "frame_*_planner_gate_*.jpg",
        "frame_*_result.json",
        "frame_*_planner_gate.json",
    ]
    for pattern in patterns:
        for stale_path in debug_dir.glob(pattern):
            stale_path.unlink(missing_ok=True)


def append_jsonl(path: Path, record: Dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as wf:
        wf.write(json.dumps(record) + "\n")


def append_episode_vlm_summary(
    output_dir: Path,
    call: Dict[str, Any],
    *,
    file_name: str = "vlm_summary.json",
) -> Path:
    """Append one readable VLM call to an episode-level JSON document.

    The recovery selector and asynchronous semantic checker can finish on
    different threads. A small advisory lock plus atomic replace keeps the JSON
    valid even when their writes overlap.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / file_name
    lock_path = output_dir / f".{file_name}.lock"
    with lock_path.open("a", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            if summary_path.exists():
                try:
                    document = json.loads(summary_path.read_text(encoding="utf-8"))
                except Exception:
                    document = {}
            else:
                document = {}
            calls = document.get("calls")
            if not isinstance(calls, list):
                calls = []
            entry = dict(call)
            entry.setdefault("call_id", len(calls) + 1)
            entry.setdefault(
                "recorded_at_utc",
                datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            )
            calls.append(entry)
            counts: Dict[str, int] = {}
            for item in calls:
                task = str(item.get("task") or "unknown")
                counts[task] = counts.get(task, 0) + 1
            document = {
                "schema_version": 1,
                "description": (
                    "Chronological, human-readable summary of every VLM answer "
                    "recorded for this episode. Input fields are separated from "
                    "the raw and parsed answer."
                ),
                "episode_output_dir": str(output_dir),
                "total_vlm_calls": len(calls),
                "calls_by_task": counts,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(
                    timespec="milliseconds"
                ),
                "calls": calls,
            }
            temp_path = output_dir / (
                f".{file_name}.tmp.{os.getpid()}.{threading.get_ident()}"
            )
            temp_path.write_text(
                json.dumps(document, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            os.replace(temp_path, summary_path)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    return summary_path


def write_debug_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def unlink_paths(paths: Iterable[Path]) -> None:
    for path in paths:
        path.unlink(missing_ok=True)


__all__ = [
    "append_jsonl",
    "append_episode_vlm_summary",
    "clear_vlm_debug_artifacts",
    "unlink_paths",
    "write_debug_json",
]
