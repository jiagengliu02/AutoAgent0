from __future__ import annotations

from typing import Dict, Optional, Sequence

from agenticdriving.adapters.carla.context import describe_vlm_camera_inputs
from agenticdriving.scorer.candidates import format_candidate_text


def build_semantic_verifier_prompt(
    candidate_row: Dict[str, object],
    route_instruction: str,
    *,
    task_target_hint: Optional[str] = None,
    previous_feedback: Optional[str] = None,
    trigger_reason: Optional[str] = None,
    camera_order: Sequence[str] = ("CAM_FRONT",),
) -> str:
    candidate_text = format_candidate_text([candidate_row])
    camera_line_1, camera_line_2 = describe_vlm_camera_inputs(camera_order)
    task_target_guidance = ""
    if task_target_hint:
        task_target_guidance = f"""
- Task target: "{task_target_hint}".
- Treat the task target as additional route context when judging on-track behavior.
""".strip()
    feedback_guidance = ""
    if previous_feedback:
        feedback_guidance = f"""
- Previous semantic verifier feedback: "{previous_feedback}".
- Use it only if it is still relevant in the current frame.
""".strip()
    trigger_guidance = ""
    if trigger_reason:
        trigger_guidance = f"""
- This sparse asynchronous check was triggered by: "{trigger_reason}".
- The deterministic high-frequency controller remains active while you inspect
  this snapshot.  Recommend takeover only for a severe, persistent-looking,
  recovery-relevant semantic failure.
""".strip()

    return f"""
You are the AgenticDriving Semantic Verifier.

Inputs:
- Visual context: {camera_line_1}
- Camera interpretation: {camera_line_2}
- Route objective: "{route_instruction}"
- Selected trajectory: one proposed short-horizon path drawn on the front image

Important:
- The colored overlay is the proposed future path from the current ego pose.
- The vehicle has not yet executed that path.
- Judge whether the proposed path is on-track with the route objective.

Task:
- Decide whether the selected trajectory is on-track with the route objective.
- Judge both the vehicle's visible current road placement and the proposed
  short-horizon path.
- Focus on severe route alignment, road-departure, wrong-way, and
  turn-direction failures.
- Also reject when a nearby, clearly unreasonable obstacle lies inside and
  materially blocks the ego's current drivable corridor. Do not treat normal
  vehicles, pedestrians, traffic controls, or harmless roadside objects as
  unreasonable obstacles.
- Do not reject merely because another path might look smoother.
- Do not reject for a mild lane-centering error, a normal pre-turn approach,
  temporary obstacle avoidance, uncertainty, or an issue that recovery cannot
  correct.
- When visual evidence is ambiguous, keep on_track=true and use
  reason_code="UNCERTAIN".
- Return only valid JSON.

Reasoning:
- Keep reasoning short and causal.
- Mention visible road/lane/intersection evidence and why the path is or is not on-track.

{task_target_guidance}
{feedback_guidance}
{trigger_guidance}

Selected trajectory:
{candidate_text}

Return ONLY valid JSON in this exact schema:
{{
  "on_track": <true or false>,
  "confidence": <float between 0.0 and 1.0>,
  "severity": <"none", "minor", or "severe">,
  "reason_code": <"ALIGNED", "WRONG_TURN", "OFF_ROUTE", "ROAD_DEPARTURE", "WRONG_WAY", "UNEXPECTED_ROAD_OBSTACLE", or "UNCERTAIN">,
  "takeover_recommended": <true or false>,
  "reasoning": "<short causal explanation>"
}}
""".strip()
