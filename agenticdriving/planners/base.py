"""SparseDriveV2 planner service interface and trajectory output."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Sequence

import numpy as np



@dataclass
class PlannerResult:
    """Pure inference output of one planning step.

    ``proposals``: ``np.ndarray`` of shape ``[N, T, 2]`` in CARLA ego-local
    coordinates. ``scores``: ``np.ndarray`` of shape ``[N]``.
    """

    proposals: np.ndarray
    scores: np.ndarray


class PlannerService(ABC):
    """Planner service used by the CARLA orchestrator."""

    #: ego-history frames the model expects (sizes the runtime's info deque).
    history_frames: int = 4

    def setup(self) -> None:
        """Load models / heavy resources. Called once before the read loop."""

    @abstractmethod
    def process(
        self,
        obs: Dict[str, Any],
        info: Dict[str, Any],
        info_history: Sequence[Dict[str, Any]],
        extra: Dict[str, Any],
    ) -> PlannerResult:
        """Run inference for the current frame.

        ``info_history`` is ordered oldest->newest, padded to ``history_frames``.
        ``extra`` carries auxiliary inputs such as ``privileged_info`` (which only
        the rule-based planner consumes; learned planners ignore it).
        """

    def finalize(self) -> None:
        """Release resources. Called once when the loop exits (even on error)."""
