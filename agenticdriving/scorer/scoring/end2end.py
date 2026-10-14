"""End2end (learned) trajectory scorer.

Target design: a single shared learned scorer for all planners, defaulting to
SparseDriveV2's vocabulary-scoring head (``TrajectoryHead`` in the
SparseDriveV2 repo: path_mlp -> topk -> vel_mlp -> topk -> traj_mlp over the
kmeans anchor vocabulary, conditioned on camera+status features). To plug it
in here:
  1. expose the scored top-K vocabulary from
     ``planners/sparsedrivev2/inference.py`` (today it returns only the
     argmax pick with a hardcoded score of 1.0), and
  2. score foreign proposals (RAP / DrivoR) by nearest-anchor lookup in the
     trajectory vocabulary.

Until that lands, each planner self-scores (RAP score head, DrivoR extracted
scores) and :class:`PlannerNativeScorer` simply passes those through, which
preserves current behaviour while giving the orchestrator a uniform
"End2end Scorer" slot.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from agenticdriving.scorer.scoring.base import BaseScorer


class PlannerNativeScorer(BaseScorer):
    """Pass-through of the generating planner's own scores."""

    name = "planner_native"

    def score(
        self,
        proposals: np.ndarray,
        *,
        native_scores: Optional[np.ndarray] = None,
        info: Optional[Dict[str, Any]] = None,
        obs: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        n = len(np.asarray(proposals))
        if native_scores is None:
            return np.zeros(n, dtype=np.float32)
        scores = np.asarray(native_scores, dtype=np.float32)
        if len(scores) != n:
            raise ValueError(f"native_scores length {len(scores)} != proposals {n}")
        return scores
