"""Base class + result structure for trajectory verifiers.

Ported from AutoAgent0's ``autoagent0/verifiers/base.py`` (interface) and the
``VerificationResult`` structure of its PDMS verifier. A verifier scores the
selected trajectory (local plan ``[T, 2]`` in CARLA ego-local coordinates,
``[x_right, y_forward]``) before it is turned into a control action. The
orchestrator rejects trajectories whose score falls below its threshold and
recovers by regenerating the plan with the rule-based planner.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class VerificationResult:
    """Hard gates (0/1 vetoes) and soft terms (graded [0, 1] scores).

    ``score = gate_product() * soft_weighted_average()`` mirrors the two-tier
    PDMS form ``NC * DAC * weighted_avg(TTC, Comfort, EP)`` used by AutoAgent0,
    so future verifiers can add terms without changing the aggregation.
    """

    gates: Dict[str, float] = field(default_factory=dict)
    soft: Dict[str, float] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)

    def gate_product(self) -> float:
        product = 1.0
        for value in self.gates.values():
            product *= float(value)
        return product

    def soft_weighted_average(self) -> float:
        if not self.soft:
            return 1.0
        denom = sum(self.weights.get(key, 1.0) for key in self.soft)
        if denom <= 0:
            return 1.0
        return sum(self.weights.get(key, 1.0) * float(value) for key, value in self.soft.items()) / denom

    def score(self) -> float:
        return self.gate_product() * self.soft_weighted_average()

    @property
    def vetoed(self) -> bool:
        """True iff any hard gate is fully tripped (value 0.0)."""
        return any(float(value) == 0.0 for value in self.gates.values())

    def to_debug_dict(self) -> Dict[str, object]:
        return {
            "gates": dict(self.gates),
            "soft": dict(self.soft),
            "score": self.score(),
            "vetoed": self.vetoed,
        }


class BaseVerifier(ABC):
    """Scores a planned trajectory before it is turned into a control action.

    Subclasses implement :meth:`verify`; :meth:`score` is the scalar view the
    orchestrator thresholds on (higher is safer / more trustworthy).
    """

    #: set by :meth:`verify` for callers that want the term breakdown.
    last_result: Optional[VerificationResult] = None

    @abstractmethod
    def verify(self, trajectory: Any, current_info: Optional[Dict[str, Any]] = None) -> VerificationResult:
        """Verify ``trajectory`` (local plan ``[T, 2]``) against ``current_info``."""

    def score(self, trajectory: Any, current_info: Optional[Dict[str, Any]] = None) -> float:
        result = self.verify(trajectory, current_info)
        self.last_result = result
        return result.score()
