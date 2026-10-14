"""Base interface shared by trajectory selection strategies."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class LearnedPlannerSelection:
    selected_row: Dict[str, object]
    selected_plan: np.ndarray
    selected_idx: Optional[int]
    selected_score: float
    selected_score_raw: float
    selected_source: str
    selected_planner: str
    candidate_rows: list[Dict[str, object]]
    planner_gate_result: Dict[str, object]
    default_selected_index: int
    default_selected_source: str
    selection_debug: Dict[str, object]


@dataclass
class SelectionOutcome:
    """What a selection strategy produces for one control step."""

    plan_payload: Dict[str, object]
    selected_plan: np.ndarray
    selected_score_raw: float
    selected_source: str
    advance_frame: bool


class BaseSelectionStrategy(ABC):
    """Picks one trajectory from planner proposals for one control step."""

    def __init__(self, selector: Any) -> None:
        self.selector = selector

    @abstractmethod
    def select(
        self,
        *,
        proposals: np.ndarray,
        scores: np.ndarray,
        output_num_poses: int,
        obs: Dict[str, Any],
        info: Dict[str, Any],
        info_history: Sequence[Dict[str, Any]],
        privileged_info: Optional[Any] = None,
    ) -> SelectionOutcome:
        raise NotImplementedError

    @staticmethod
    def _disabled_planner_gate_result() -> Dict[str, object]:
        return {
            "selected_planner": "learned",
            "confidence": None,
            "reasoning": "planner_gate_disabled",
            "elapsed_sec": 0.0,
            "error": None,
            "timed_out": False,
            "prompt_char_count": 0,
            "image_count": 0,
        }

    @staticmethod
    def _default_selection_for_family(
        *,
        candidate_rows: Sequence[Dict[str, object]],
        selected_planner: str,
        scores: np.ndarray,
        learned_source_name: str,
        learned_default_source: str,
        strict_learned_argmax_lookup: bool = False,
    ) -> tuple[int, str]:
        if selected_planner == "rule_based":
            return (
                int(max(range(len(candidate_rows)), key=lambda idx: float(candidate_rows[idx].get("proposal_score", 0.0)))),
                "fallback_rule_based_argmax",
            )

        best_idx = int(np.argmax(scores))
        for idx, row in enumerate(candidate_rows):
            if row.get("source") == learned_source_name and row.get("proposal_index") == best_idx:
                return int(idx), learned_default_source
            if not strict_learned_argmax_lookup and row.get("proposal_index") is not None:
                try:
                    if int(row.get("proposal_index")) == best_idx:
                        return int(idx), learned_default_source
                except Exception:
                    pass

        if strict_learned_argmax_lookup:
            raise StopIteration(f"Could not find learned argmax row for proposal_index={best_idx}")
        return 0, learned_default_source
