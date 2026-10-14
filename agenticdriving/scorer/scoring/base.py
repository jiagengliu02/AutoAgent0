"""Base interface for per-candidate trajectory scorers.

Scoring vs selection (the two sibling concepts in the ``scorer`` package):

- ``scoring/`` (this subpackage): composable per-candidate scorers. Each takes
  the N proposals of one frame and returns N scores. Several run together per
  frame (e.g. end2end score gated by the rule-based instruction mask) and
  their combined output *feeds* the selection.
- ``selection_strategies/``: mutually exclusive end-to-end selection flows
  (argmax / recovery loop). Exactly one runs per ``select`` call and turns
  candidates + scores into a single chosen trajectory.

These are the "Rule-based scorer" / "End2end Scorer" boxes of the pipeline
diagram; strategies are an implementation detail below them.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np


class BaseScorer(ABC):
    """Scores every candidate proposal of one frame.

    ``proposals``: ``[N, T, 2]`` CARLA ego-local coordinates. Returns ``[N]``
    scores (higher is better). ``native_scores`` carries the planner's own
    scores for scorers that pass them through or refine them.
    """

    #: short label used in debug payloads.
    name: str = "scorer"

    @abstractmethod
    def score(
        self,
        proposals: np.ndarray,
        *,
        native_scores: Optional[np.ndarray] = None,
        info: Optional[Dict[str, Any]] = None,
        obs: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Return per-candidate scores ``[N]`` for this frame."""
