"""Rule-based (PDMS) per-candidate scorer.

The per-candidate twin of the PDMS *verifier*
(``agenticdriving/verifiers/pdms.py``): where the verifier scores the single
selected plan in the verify/recover stage, this scorer runs the SAME
``NC * DAC * weighted_avg(TTC, EP)`` form over each candidate batch supplied by
the orchestrator.

It wraps a :class:`~agenticdriving.verifiers.pdms.PDMSVerifier` rather than
re-deriving the metric, so the two stay bit-for-bit identical. The wrapped
instance is dedicated to scoring (own state, debug off, TTC debounce disabled):
debounce is a temporal decision-layer concept about *consecutive verifier
steps in time*, which has no meaning across the independent candidates of one
frame, so each proposal is scored with the instantaneous PDMS.

When both learned and rule-based scorers are active, this scorer is called for
two independent proposal batches: the learned scorer's native Top-K and all
trajectories in the layered region containing the leaderboard target point.
Each branch proposes its PDMS argmax; the higher-PDMS proposal is then passed
to the final verifier.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from agenticdriving.scorer.scoring.base import BaseScorer
from agenticdriving.verifiers.pdms import PDMSVerifier


class RuleBasedPDMSScorer(BaseScorer):
    """Score the supplied candidate batch with the PDMS trajectory verifier.

    ``info`` must carry the same live state the verifier consumes
    (``ego_pos`` / ``ego_rot`` / ``ego_velo`` and, for the NC/TTC terms,
    ``privileged_agents``); the orchestrator's scoring stage populates it.
    """

    name = "rule_based_pdms"

    def __init__(self, verifier: PDMSVerifier) -> None:
        self._verifier = verifier

    @property
    def last_endpoint_tiers(self):
        """HD-map endpoint tiers aligned with the most recent score batch."""
        return getattr(self._verifier, "last_batch_endpoint_tiers", None)

    def score(
        self,
        proposals: np.ndarray,
        *,
        native_scores: Optional[np.ndarray] = None,
        info: Optional[Dict[str, Any]] = None,
        obs: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        proposals = np.asarray(proposals, dtype=np.float32)
        n = len(proposals)
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        info = info or {}
        batch_score = getattr(self._verifier, "score_batch", None)
        if callable(batch_score):
            return np.asarray(batch_score(proposals, info), dtype=np.float32)
        scores = np.empty(n, dtype=np.float32)
        for idx in range(n):
            scores[idx] = float(self._verifier.score(proposals[idx], info))
        return scores
