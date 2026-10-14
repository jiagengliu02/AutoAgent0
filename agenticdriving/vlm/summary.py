"""Episode-level consolidation of saved VLM call artifacts."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

from agenticdriving.vlm.debug import append_episode_vlm_summary


_FRAME_RE = re.compile(r"frame_(\d+)")


def _task_for(path: Path) -> Optional[str]:
    text = str(path)
    if "/semantic_takeover/" in text:
        return "semantic_route_check"
    if "/recovery_explain/" in text and path.name.endswith("_recovery_reason.json"):
        return "recovery_failure_explanation"
    if "/recovery_select/" in text and path.name.endswith("_recovery_sequence.json"):
        return "recovery_sequence_selection"
    if "/recovery_select/" in text and path.name.endswith("_recovery_select.json"):
        return "recovery_primitive_selection"
    return None


def _frame_index(path: Path) -> Optional[int]:
    match = _FRAME_RE.search(path.name)
    return None if match is None else int(match.group(1))


def _accepted_result(task: str, parsed: Any) -> Dict[str, Any]:
    parsed_dict = dict(parsed) if isinstance(parsed, dict) else {}
    if task == "recovery_sequence_selection":
        steps = list(parsed_dict.get("steps") or [])
        return {
            "steps": steps,
            "first_action_executed_now": steps[0] if steps else None,
            "remaining_actions_saved_as_intent": steps[1:],
            "reasoning": parsed_dict.get("reasoning"),
            "confidence": parsed_dict.get("confidence"),
        }
    return parsed_dict


def _prompt_input_summary(task: str, prompt: Any, path: Path) -> Dict[str, Any]:
    if not isinstance(prompt, str):
        return {}
    result: Dict[str, Any] = {}
    route_match = re.search(r'(?:Route objective|Route):\s*"([^"]+)"', prompt)
    if route_match:
        result["route_instruction"] = route_match.group(1)
    failure_match = re.search(r"^- Failure:\s*(.+)$", prompt, flags=re.MULTILINE)
    if failure_match:
        result["failure_reason"] = failure_match.group(1).strip()
    trigger_match = re.search(
        r'triggered by:\s*"([^"]+)"', prompt, flags=re.IGNORECASE
    )
    if trigger_match:
        result["trigger"] = trigger_match.group(1)
    elif task == "semantic_route_check":
        frame_match = re.match(r"frame_\d+_(.+)", path.stem)
        if frame_match:
            result["trigger"] = frame_match.group(1)
    candidate_match = re.search(
        r"^- candidate_0\s*\|\s*(.+)$", prompt, flags=re.MULTILINE
    )
    if candidate_match:
        result["selected_trajectory_summary"] = candidate_match.group(1).strip()
    return result


def summarize_episode(episode_dir: Path, *, overwrite: bool = False) -> Path:
    """Create ``vlm_summary.json`` from all saved calls in one episode."""
    episode_dir = Path(episode_dir).resolve()
    summary_path = episode_dir / "vlm_summary.json"
    if summary_path.exists() and not overwrite:
        return summary_path
    if overwrite:
        summary_path.unlink(missing_ok=True)

    artifacts = []
    for path in episode_dir.rglob("frame_*.json"):
        task = _task_for(path)
        if task is not None:
            artifacts.append((path.stat().st_mtime, path, task))
    artifacts.sort(key=lambda item: (item[0], str(item[1])))

    for _mtime, path, task in artifacts:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        inference = payload.get("inference_result")
        if not isinstance(inference, dict):
            continue
        excluded = {
            "raw_output",
            "parsed_output",
            "prompt",
            "elapsed_sec",
            "token_usage",
            "error",
        }
        input_fields = {
            key: value for key, value in inference.items() if key not in excluded
        }
        input_fields = {
            **_prompt_input_summary(task, inference.get("prompt"), path),
            **input_fields,
        }
        error = inference.get("error")
        append_episode_vlm_summary(
            episode_dir,
            {
                "task": task,
                "frame_index": _frame_index(path),
                "simulation_timestamp_sec": input_fields.pop("timestamp", None),
                "status": "error" if error else "ok",
                "input": input_fields,
                "answer": {
                    "raw_text": inference.get("raw_output"),
                    "parsed_json": inference.get("parsed_output"),
                    "accepted_result": _accepted_result(
                        task, inference.get("parsed_output")
                    ),
                },
                "runtime": {
                    "elapsed_sec": inference.get("elapsed_sec", 0.0),
                    "token_usage": inference.get("token_usage") or {},
                    "error": error,
                },
                "artifacts": {"result_json": str(path)},
            },
        )
    if not summary_path.exists():
        summary_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "description": (
                        "Chronological, human-readable summary of every VLM "
                        "answer recorded for this episode."
                    ),
                    "episode_output_dir": str(episode_dir),
                    "total_vlm_calls": 0,
                    "calls_by_task": {},
                    "calls": [],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episode_dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(summarize_episode(args.episode_dir, overwrite=args.overwrite))


if __name__ == "__main__":
    main()
