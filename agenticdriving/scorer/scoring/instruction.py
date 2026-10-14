"""Rule-based scorer: route-instruction (left/straight/right) compliance mask.

fail2drive's route carries RoadOptions which the CARLA adapter condenses into
a discrete command (``get_route_command``: 0=right, 1=left, 2=straight) that
is already fed *into* the planner (e.g. RAP's command one-hot). This scorer
feeds the SAME instruction to the selection side: each proposal's geometric
direction (start->end endpoint bearing) is classified and scored by whether
it complies with the commanded direction.

The bearing classifier is ported from AutoAgent0's
the original geometric route check, with the same thresholds.

Scores are meant to be used as a multiplicative gate over normalized end2end
scores: match -> 1.0, ambiguous/unknown -> neutral 0.5, contradiction -> 0.0.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

from agenticdriving.scorer.scoring.base import BaseScorer

STRAIGHT_HEADING_DEG = 10.0
TURN_HEADING_DEG = 15.0
MIN_FORWARD_PROGRESS_M = 0.3

#: CARLA adapter coarse-command encoding.
COMMAND_TO_DIRECTION = {0: "right", 1: "left", 2: "straight"}


def normalize_turn_label(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if "straight" in text or text in {"go straight", "forward", "continue"}:
        return "straight"
    if "left" in text:
        return "left"
    if "right" in text:
        return "right"
    return None


def instruction_to_turn_direction(
    *,
    route_instruction: Optional[str] = None,
    command: Optional[object] = None,
) -> Optional[str]:
    """Map route text or the discrete command to left/straight/right."""
    label = normalize_turn_label(route_instruction)
    if label is not None:
        return label
    if command is None:
        return None
    try:
        return COMMAND_TO_DIRECTION.get(int(command))
    except (TypeError, ValueError):
        return normalize_turn_label(command)


def classify_local_plan_endpoint_bearing(
    local_plan: np.ndarray,
    *,
    straight_heading_deg: float = STRAIGHT_HEADING_DEG,
    turn_heading_deg: float = TURN_HEADING_DEG,
    min_forward_progress_m: float = MIN_FORWARD_PROGRESS_M,
) -> str:
    """Classify a CARLA ego-local plan [T, 2] by start->end endpoint bearing.

    ``atan2(delta_x_right, delta_y_forward) > 0`` means right, ``< 0`` left.
    Returns "straight" | "left" | "right" | "unknown".
    """
    plan = np.asarray(local_plan, dtype=np.float32)
    if plan.ndim != 2 or plan.shape[1] < 2 or len(plan) < 2:
        return "unknown"
    delta_x = float(plan[-1, 0] - plan[0, 0])
    delta_y = float(plan[-1, 1] - plan[0, 1])
    if delta_y < min_forward_progress_m:
        return "unknown"
    heading_deg = math.degrees(math.atan2(delta_x, delta_y))
    if abs(heading_deg) <= straight_heading_deg:
        return "straight"
    if heading_deg >= turn_heading_deg:
        return "right"
    if heading_deg <= -turn_heading_deg:
        return "left"
    return "unknown"


class InstructionComplianceScorer(BaseScorer):
    """Per-candidate gate: does the proposal's geometry follow the instruction?"""

    name = "instruction_compliance"

    def __init__(
        self,
        *,
        match_score: float = 1.0,
        unknown_score: float = 0.5,
        mismatch_score: float = 0.0,
    ) -> None:
        self.match_score = float(match_score)
        self.unknown_score = float(unknown_score)
        self.mismatch_score = float(mismatch_score)

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
        info = info or {}
        target = instruction_to_turn_direction(
            route_instruction=info.get("task_instruction"),
            command=info.get("command"),
        )
        if target is None:
            # No usable instruction: neutral, gate nothing.
            return np.ones(n, dtype=np.float32)

        scores = np.empty(n, dtype=np.float32)
        for idx in range(n):
            direction = classify_local_plan_endpoint_bearing(proposals[idx])
            if direction == "unknown":
                scores[idx] = self.unknown_score
            elif direction == target:
                scores[idx] = self.match_score
            else:
                scores[idx] = self.mismatch_score
        return scores
