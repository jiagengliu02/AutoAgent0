"""No-VLM selector used by the AgenticDriving recovery loop."""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np


def _empty_token_usage() -> Dict[str, int]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


class DummyPlanSelector:
    """Selector facade with the same public methods as ``VLMPlanSelector``.

    It never calls a vision-language model. The default critique asks the
    recovery loop to build the revised candidate pool, and the scorer randomly
    picks one candidate from that pool.
    """

    def __init__(self, *, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(seed)
        self._timeline_records: list[Dict[str, object]] = []

    def preload(self) -> None:
        return None

    def finalize(self) -> None:
        return None

    def _random_index(self, candidate_rows: Sequence[Dict[str, object]]) -> int:
        if not candidate_rows:
            raise ValueError("DummyPlanSelector requires at least one candidate")
        return int(self._rng.integers(0, len(candidate_rows)))

    def _record_timeline(self, record: Dict[str, object]) -> Dict[str, object]:
        self._timeline_records.append(record)
        return record

    def critique_agenticdriving_candidate(
        self,
        *,
        frame_index: int,
        camera_images: Dict[str, Any],
        info: Dict[str, object],
        candidate_row: Dict[str, object],
        stage: str,
        previous_feedback: Optional[str] = None,
    ) -> Dict[str, object]:
        timestamp = float(info.get("timestamp", 0.0))
        is_default_stage = str(stage).strip().lower() == "default"
        accepted = not is_default_stage
        reasoning = (
            "dummy_selector_requests_random_recovery"
            if is_default_stage
            else "dummy_selector_accepts_random_recovery_candidate"
        )
        timeline_record = self._record_timeline(
            {
                "frame_index": int(frame_index),
                "timestamp": timestamp,
                "execution_mode": f"agenticdriving_{stage}_dummy_critic",
                "candidate_count": 1,
                "selected_source": candidate_row.get("source"),
                "agenticdriving_stage": stage,
                "agenticdriving_tool": "request_critique",
                "agenticdriving_critique_accepted": accepted,
                "agenticdriving_critique_error": None,
                "vlm_elapsed_sec": 0.0,
                "scoring_invoked": False,
                "q_invoked_vlm": False,
                "vlm_failed": False,
                "error": None,
            }
        )
        return {
            "stage": stage,
            "agenticdriving_critique_accepted": accepted,
            "agenticdriving_critique_rejected": not accepted,
            "agenticdriving_critique_severity_score": 1.0 if is_default_stage else 0.0,
            "agenticdriving_critique_corrective_action": "random_recovery",
            "agenticdriving_critique_confidence": None,
            "agenticdriving_critique_reasoning": reasoning,
            "agenticdriving_critique_error": None,
            "agenticdriving_critique_elapsed_sec": 0.0,
            "agenticdriving_critique_token_usage": _empty_token_usage(),
            "latency_timeline_record": timeline_record,
            "error": None,
        }

    def score_agenticdriving_candidates(
        self,
        *,
        frame_index: int,
        camera_images: Dict[str, Any],
        info: Dict[str, object],
        candidate_rows: Sequence[Dict[str, object]],
        default_selected_index: int,
        default_selected_source: str,
        critique_reason: Optional[str] = None,
        corrective_action: Optional[str] = None,
        stage: str = "revised",
    ) -> Dict[str, object]:
        selected_candidate_index = self._random_index(candidate_rows)
        selected_row = dict(candidate_rows[selected_candidate_index])
        selected_source = "agenticdriving_dummy_random"
        timestamp = float(info.get("timestamp", 0.0))
        timeline_record = self._record_timeline(
            {
                "frame_index": int(frame_index),
                "timestamp": timestamp,
                "execution_mode": f"agenticdriving_{stage}_dummy_random",
                "candidate_count": len(candidate_rows),
                "selected_source": selected_source,
                "selected_candidate_index": selected_candidate_index,
                "selected_candidate_source": selected_row.get("source"),
                "selected_proposal_index": selected_row.get("proposal_index"),
                "agenticdriving_stage": stage,
                "agenticdriving_tool": "select_final_actions",
                "vlm_elapsed_sec": 0.0,
                "scoring_invoked": False,
                "q_invoked_vlm": False,
                "vlm_failed": False,
                "error": None,
            }
        )
        return {
            "selected_candidate_row": selected_row,
            "selected_source": selected_source,
            "selected_path_reasoning": "Dummy selector randomly selected a recovery candidate.",
            "execution_mode": f"agenticdriving_{stage}_dummy_random",
            "dummy_selected_idx": selected_candidate_index,
            "vlm_candidate_index": selected_candidate_index,
            "vlm_confidence": None,
            "vlm_reasoning": None,
            "vlm_elapsed_sec": 0.0,
            "vlm_error": None,
            "vlm_candidate_count": len(candidate_rows),
            "vlm_q_valid": False,
            "vlm_q_candidate_scores": None,
            "vlm_q_best_candidate_index": selected_candidate_index,
            "vlm_q_score_gap_top2": None,
            "adaptive_replan_decision": "agenticdriving_dummy_random",
            "latency_timeline_record": timeline_record,
            "vlm_failed": False,
            "scoring_invoked": False,
            "scoring_token_usage": _empty_token_usage(),
            "error": None,
        }

    def maybe_select(
        self,
        frame_index: int,
        camera_images: Dict[str, Any],
        info: Dict[str, object],
        candidate_rows: Sequence[Dict[str, object]],
        default_selected_index: int,
        default_selected_source: str,
        force_scoring: bool = False,
        intervention_corrective_action_override: Optional[str] = None,
        execution_mode_label: Optional[str] = None,
    ) -> Dict[str, object]:
        return self.score_agenticdriving_candidates(
            frame_index=frame_index,
            camera_images=camera_images,
            info=info,
            candidate_rows=candidate_rows,
            default_selected_index=default_selected_index,
            default_selected_source=default_selected_source,
            stage=execution_mode_label or "dummy",
        )

    def maybe_select_planner(
        self,
        *,
        frame_index: int,
        camera_images: Dict[str, Any],
        info: Dict[str, object],
        learned_candidate_rows: Sequence[Dict[str, object]],
        rule_based_candidate_rows: Sequence[Dict[str, object]],
    ) -> Dict[str, object]:
        planners = []
        if learned_candidate_rows:
            planners.append("learned")
        if rule_based_candidate_rows:
            planners.append("rule_based")
        selected_planner = (
            planners[self._random_index([{"source": item} for item in planners])]
            if planners
            else "learned"
        )
        return {
            "selected_planner": selected_planner,
            "confidence": None,
            "reasoning": "dummy_selector_random_planner",
            "elapsed_sec": 0.0,
            "error": None,
            "timed_out": False,
            "prompt_char_count": 0,
            "image_count": 0,
            "token_usage": _empty_token_usage(),
        }
