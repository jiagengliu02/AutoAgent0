"""Argmax trajectory selection strategy."""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

from agenticdriving.scorer.selection_strategies.base import (
    BaseSelectionStrategy,
    SelectionOutcome,
)


class ArgmaxStrategy(BaseSelectionStrategy):
    """Behavior-preserving non-VLM path: pick the highest-scoring proposal."""

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
        sel = self.selector
        best_idx = int(np.argmax(scores))
        selected_plan = np.asarray(proposals[best_idx], dtype=np.float32)
        selected_score = float(scores[best_idx])
        plan_payload = sel._build_plan_payload(
            proposals,
            scores,
            output_num_poses,
            selected_idx=best_idx,
            selected_source=sel.plain_source,
            topk=10,
        )
        return SelectionOutcome(
            plan_payload=plan_payload,
            selected_plan=selected_plan,
            selected_score_raw=selected_score,
            selected_source=sel.current_source_name,
            advance_frame=False,
        )
