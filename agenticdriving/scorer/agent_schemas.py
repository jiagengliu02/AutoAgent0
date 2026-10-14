from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class SceneContext:
    """Normalized scene metadata exposed to AgenticDriving components."""

    frame_index: int
    route_instruction: str
    timestamp: float = 0.0
    task_instruction: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OrchestratorDecision:
    """Structured orchestrator decision for trace/debug output."""

    decision_type: str
    selected_source: Optional[str] = None
    selected_planner: Optional[str] = None
    selected_candidate_index: Optional[int] = None
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DesignChangeRequest:
    """Request issued after critic rejection to expand/revise candidates."""

    reason: str
    corrective_action: Optional[str]
    candidate_budget: int
    learned_budget: int
    rule_based_budget: int
    allocation_strategy: str
    include_learned: bool = True
    include_rule_based: bool = True
    routing_mode: str = "normal"


@dataclass(frozen=True)
class FrameUncertainty:
    """Per-frame uncertainty signals used to route the design-change budget."""

    intra_learned_m: float
    cross_family_m: float
    mode_count: int
    routing_zone: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VerifierResult:
    """Verifier result. Phase 1 is passive and always accepts."""

    accepted: bool
    mode: str = "passive"
    rejection_reason: Optional[str] = None
    checks: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class SemanticVerifierResult:
    """Post-selection semantic route/track check from the VLM verifier."""

    accepted: bool
    mode: str = "semantic_vlm"
    rejection_reason: Optional[str] = None
    confidence: Optional[float] = None
    checks: Dict[str, Any] = field(default_factory=dict)

    def to_debug_dict(self) -> Dict[str, Any]:
        return {
            "accepted": self.accepted,
            "mode": self.mode,
            "rejection_reason": self.rejection_reason,
            "confidence": self.confidence,
            "checks": self.checks,
        }


@dataclass(frozen=True)
class AgentStepTrace:
    """Debug-only trace joining designer, orchestrator, and verifier state."""

    schema_version: str
    scene: SceneContext
    designer: Dict[str, Any]
    orchestrator: OrchestratorDecision
    verifier: VerifierResult
    semantic_verifier: Optional[SemanticVerifierResult] = None
    previous_verifier_feedback: Optional[Dict[str, Any]] = None

    def to_debug_dict(self) -> Dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "scene": {
                "frame_index": self.scene.frame_index,
                "route_instruction": self.scene.route_instruction,
                "timestamp": self.scene.timestamp,
                "task_instruction": self.scene.task_instruction,
                "metadata": self.scene.metadata,
            },
            "designer": self.designer,
            "orchestrator": {
                "decision_type": self.orchestrator.decision_type,
                "selected_source": self.orchestrator.selected_source,
                "selected_planner": self.orchestrator.selected_planner,
                "selected_candidate_index": self.orchestrator.selected_candidate_index,
                "confidence": self.orchestrator.confidence,
                "reasoning": self.orchestrator.reasoning,
                "metadata": self.orchestrator.metadata,
            },
            "verifier": {
                "accepted": self.verifier.accepted,
                "mode": self.verifier.mode,
                "rejection_reason": self.verifier.rejection_reason,
                "checks": self.verifier.checks,
            },
            "previous_verifier_feedback": self.previous_verifier_feedback or {},
        }
        if self.semantic_verifier is not None:
            payload["semantic_verifier"] = self.semantic_verifier.to_debug_dict()
        return payload
