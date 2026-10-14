from __future__ import annotations

from typing import Sequence

from agenticdriving.adapters.carla.context import describe_vlm_camera_inputs


def build_stop_sign_prompt(
    route_instruction: str,
    *,
    junction_distance_m: float,
    camera_order: Sequence[str] = ("CAM_FRONT",),
) -> str:
    """Build the dedicated, single-purpose stop-sign classification prompt."""
    camera_line_1, _ = describe_vlm_camera_inputs(camera_order)
    return f"""
You are the AgenticDriving Stop Sign Verifier.

Inputs:
- Visual context: {camera_line_1}
- The ego vehicle is approaching a mapped intersection about
  {float(junction_distance_m):.1f} metres ahead.
- Planned route direction at the intersection: "{route_instruction}".

Your only task:
- Decide whether the ego vehicle must stop before entering this intersection
  because of a visible STOP sign controlling the ego's approach lane.

Rules:
- Count a red octagonal STOP sign facing the ego approach, including one that
  is partly obscured, damaged, altered, or visible near an image edge but is
  still recognisable as the stop control for this approach.
- A sign may be on either side of the ego lane.
- Ignore STOP signs controlling a side road, signs facing away from the ego,
  the back of a sign, and signs beyond the intersection for another approach.
- Do not infer a STOP sign merely because an intersection exists.
- Set must_stop=true only when stop_sign_visible=true and
  applies_to_ego_lane=true.
- Return only valid JSON. Keep reasoning to one short sentence.

Return ONLY valid JSON in this exact schema:
{{
  "stop_sign_visible": <true or false>,
  "applies_to_ego_lane": <true or false>,
  "must_stop": <true or false>,
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<short visual explanation>"
}}
""".strip()
