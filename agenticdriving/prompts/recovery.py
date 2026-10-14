from __future__ import annotations

import json
from typing import Dict, Optional, Sequence

from agenticdriving.adapters.carla.context import describe_vlm_camera_inputs


RECOVERY_PLAN_EXCLUDED_PRIMITIVES = frozenset(
    {"back_bypass_left", "back_bypass_right"}
)


RECOVERY_PLAN_SYSTEM_PROMPT = """You are a driving recovery planner.

Use only exact primitive IDs supplied in the user prompt. Never output or
recommend an action that is absent from the supplied action list.
Return only valid JSON."""


def recovery_plan_bypass_first_system_prompt(bypass_action: str) -> str:
    """Return the common plan contract; bypass policy lives in the user prompt."""
    return RECOVERY_PLAN_SYSTEM_PROMPT


RECOVERY_PLAN_BYPASS_FIRST_SYSTEM_PROMPT = (
    recovery_plan_bypass_first_system_prompt("lane_bypass_left")
)


# Short natural-language description of each recovery primitive, so the VLM
# reasons about what a candidate DOES rather than parsing its raw label.
_PRIMITIVE_DESCRIPTIONS: Dict[str, str] = {
    "brake": "come to a controlled stop in the current lane",
    "wait": "remain stopped and wait for the blockage to clear",
    "creep": "crawl forward slowly (~1.2 m/s) to nudge past a close obstacle",
    "lane_left": "change one lane to the left",
    "lane_right": "change one lane to the right",
    "lane_bypass_left": (
        "temporarily borrow the adjacent driving lane on the left, pass the "
        "static blockage, and return to the original lane"
    ),
    "lane_bypass_right": (
        "temporarily borrow the adjacent driving lane on the right, pass the "
        "static blockage, and return to the original lane"
    ),
    "swerve_left": "briefly steer left around an obstacle, then realign",
    "swerve_right": "briefly steer right around an obstacle, then realign",
    "reverse": "back up straight to open space behind the ego",
    "back_left": "reverse while steering, repositioning to the left-rear",
    "back_right": "reverse while steering, repositioning to the right-rear",
    "back_bypass_left": "reverse straight, then drive forward around the blockage on the left",
    "back_bypass_right": "reverse straight, then drive forward around the blockage on the right",
    "three_point": "perform a three-point turn to reorient the ego",
}


def _format_road_context_lines(road_context: Optional[Dict[str, object]]) -> str:
    """Render the road-context facts as prompt input lines.

    ``road_context`` is the orchestrator's dict (see
    ``Orchestrator._recovery_road_context``): ``ego_on_corridor`` (bool),
    ``nearest_lane_distance_m`` / ``nearest_lane_direction`` (present only
    when the ego is off the corridor and a driving lane was found). Returns
    "" when ``road_context`` is None, so callers that never pass it produce
    byte-identical prompts to the stack without this feature.
    """
    if not road_context:
        return ""
    # Exposure caution: key is set by the orchestrator only when
    # sequence_exposure_guidance is on and the ego sits laterally off its
    # route line while on drivable road (straddling / opposite lane).
    exposure_line = (
        "\n- CAUTION: you are stopped in or across a live traffic lane, exposed"
        "\n  to moving vehicles; remaining stopped here is dangerous. Prefer"
        "\n  actions that clear this lane (advance out of it, or back toward"
        "\n  your own lane) over brake or wait."
        if road_context.get("exposed_position")
        else ""
    )
    if road_context.get("ego_on_corridor", True):
        return "\n- Ego position: on a driving lane." + exposure_line
    lines = [
        "\n- Ego position: OFF the drivable corridor. Priority: return to a "
        "driving lane; obstacle clearance is secondary."
    ]
    distance = road_context.get("nearest_lane_distance_m")
    direction = road_context.get("nearest_lane_direction")
    if distance is not None and direction is not None:
        lines.append(
            f"\n- Nearest driving lane: ~{float(distance):.0f} m, direction {direction}."
        )
    return "".join(lines) + exposure_line


def _format_route_waypoint_lines(
    route_waypoint: Optional[Dict[str, float]],
) -> str:
    """Render one ego-frame route waypoint for recovery selection."""
    if not route_waypoint:
        return ""
    try:
        forward_m = float(route_waypoint["forward_m"])
        right_m = float(route_waypoint["right_m"])
        distance_m = float(route_waypoint["distance_m"])
        bearing_deg = float(route_waypoint["bearing_deg"])
    except (KeyError, TypeError, ValueError):
        return ""
    return (
        "\n- Next route waypoint (ego frame): "
        f"forward={forward_m:.2f} m, right={right_m:.2f} m, "
        f"distance={distance_m:.2f} m, bearing={bearing_deg:+.1f} deg. "
        "Positive right/bearing means ego-right; negative means ego-left."
        "\n- Preserve a feasible path toward this waypoint after clearing the "
        "immediate hazard."
    )


def _describe_plan_direction(candidate_row: Dict[str, object]) -> str:
    """Classify the rejected plan as left, center, or right.

    Candidate coordinates are ``[x_right, y_forward]`` in the ego frame, so
    negative lateral displacement is left and positive displacement is right.
    """
    s = candidate_row.get("summary", {}) or {}
    dx = s.get("delta_x")
    if dx is None:
        return "center"
    dx = float(dx)
    if dx < -0.5:
        return "left"
    if dx > 0.5:
        return "right"
    return "center"


# Compact verifier definitions used in the recovery-reason prompt.
_GATE_MEANINGS: Dict[str, str] = {
    "nc": "collision safety; 0 = collision veto",
    "dac": "drivable-area compliance; 0 = off-road veto, not collision",
}
_SOFT_MEANINGS: Dict[str, str] = {
    "ttc": "collision-time margin; lower = less margin",
    "ep": "forward progress; lower = less progress",
    "lk": "lane keeping; lower = more deviation",
}


def _format_verifier_terms(
    gates: Dict[str, float],
    soft: Dict[str, float],
) -> str:
    lines = []
    for name, value in gates.items():
        meaning = _GATE_MEANINGS.get(name)
        meaning_txt = f"  [{meaning}]" if meaning else ""
        flag = " <- TRIPPED (hard veto)" if float(value) <= 0.0 else ""
        lines.append(f"- gate {name} = {float(value):.3f}{flag}{meaning_txt}")
    for name, value in soft.items():
        meaning = _SOFT_MEANINGS.get(name)
        meaning_txt = f"  [{meaning}]" if meaning else ""
        flag = " <- LOW" if float(value) < 0.5 else ""
        lines.append(f"- soft {name} = {float(value):.3f}{flag}{meaning_txt}")
    return "\n".join(lines) if lines else "(no per-term breakdown available)"


def build_recovery_reason_prompt(
    candidate_row: Dict[str, object],
    *,
    verifier_score: float,
    verifier_threshold: float,
    verifier_gates: Dict[str, float],
    verifier_soft: Dict[str, float],
    camera_order: Sequence[str] = ("CAM_FRONT",),
    road_context: Optional[Dict[str, object]] = None,
) -> str:
    direction_text = _describe_plan_direction(candidate_row)
    camera_line_1, _camera_line_2 = describe_vlm_camera_inputs(camera_order)
    verifier_terms_text = _format_verifier_terms(verifier_gates, verifier_soft)
    road_context_lines = _format_road_context_lines(road_context)
    off_corridor = bool(road_context) and not road_context.get("ego_on_corridor", True)
    # Off-corridor, the "look ahead" instruction misdirects the model: the
    # obstacle ahead is a symptom, being off the road is the situation.
    surface_task_line = (
        "- The ego is off the drivable corridor: state where the drivable road is relative to the ego, not just the visible obstacle ahead."
        if off_corridor
        else "- Pay particular attention to the road surface directly ahead of the ego, in its immediate forward path."
    )

    return f"""
You are an autonomous-driving incident analyst. The vehicle's safety verifier
just rejected the currently selected normal trajectory. This does NOT imply
that every forward path, adjacent lane, or bypass is blocked; another corridor
may still be safe. Explain, in plain language, why THIS trajectory was rejected.

Inputs:
- Visual context: {camera_line_1}
- Rejected trajectory direction: "{direction_text}" (left, center, or right).{road_context_lines}

Verifier result:
- overall score = {verifier_score:.3f} (threshold = {verifier_threshold:.3f}; below threshold => rejected)
{verifier_terms_text}

Task:
- Using the image(s) together with the verifier term breakdown, explain the root cause of why the selected trajectory was rejected.
- Treat a gate value of 0 as the root cause: `nc` means collision; `dac` means off-road. Never swap them.
- A low soft term indicates graded risk, not a hard veto.
- Ground the explanation in what is visible: the specific obstacle responsible and where it sits relative to the ego vehicle and its forward path.
- Distinguish "the selected/current-lane path is blocked" from "the whole roadway is blocked". Do not claim that no maneuver is possible when an adjacent lane or visible bypass remains open.
{surface_task_line}
- Keep it short: identifying the specific object, actor, or road boundary causing the rejection.

Return ONLY valid JSON in this exact schema:
{{
  "reason": "<short causal explanation of why the selected trajectory was rejected>"
}}
""".strip()


def _format_recovery_candidate_text(candidate_rows: Sequence[Dict[str, object]]) -> str:
    lines = []
    for row in candidate_rows:
        name = str(row.get("primitive_name", "?"))
        desc = _PRIMITIVE_DESCRIPTIONS.get(name, "recovery maneuver")
        s = row.get("summary", {}) or {}
        # Deliberately do not expose nominal verifier scores. The VLM proposes a
        # maneuver/parameter from scene geometry; the concrete rollout is then
        # checked by the verifier before execution.
        lateral_range = row.get("lateral_distance_range_m")
        lateral_text = ""
        if lateral_range is not None:
            range_label = (
                "pdms_threshold_safe_lateral_distance_m"
                if row.get("lateral_distance_range_preverified")
                else "tunable_lateral_distance_m"
            )
            lateral_text = (
                f" | {range_label}=[{float(lateral_range[0]):.2f},"
                f"{float(lateral_range[1]):.2f}]"
            )
        # Road-context arm only (see Orchestrator._candidate_ends_on_driving_lane);
        # absent -> no column, so default prompts are unchanged. Graded v2 adds
        # distance_to_lane_m on off-lane rows (on-lane rows show the bare flag:
        # the number carries no signal there and "yes | 1.4m" reads as a
        # contradiction) and tags the argmin when nothing reaches the lane.
        road_flag = row.get("ends_on_driving_lane")
        if road_flag is None:
            road_text = ""
        else:
            road_text = f" | ends_on_driving_lane={'yes' if road_flag else 'no'}"
            distance = row.get("distance_to_lane_m")
            if not road_flag and distance is not None:
                road_text += f" | distance_to_lane_m={float(distance):.1f}"
                if row.get("_closest_to_lane"):
                    road_text += " | closest_to_lane=yes"
        lines.append(
            f"- candidate_{row['candidate_index']} | "
            f"maneuver={name} ({desc}) | "
            f"end={s.get('end')} | forward_progress_m={s.get('forward_progress_m')} | "
            f"path_length_m={s.get('path_length_m')}{lateral_text}{road_text}"
        )
    return "\n".join(lines)


def build_recovery_primitive_selection_prompt(
    candidate_rows: Sequence[Dict[str, object]],
    route_instruction: str,
    *,
    failure_reason: Optional[str],
    previous_attempts: Optional[Sequence[str]] = None,
    failure_memory: Optional[str] = None,
    camera_order: Sequence[str] = ("CAM_FRONT",),
    verifier_score: Optional[float] = None,
    verifier_threshold: Optional[float] = None,
    verifier_gates: Optional[Dict[str, float]] = None,
    verifier_soft: Optional[Dict[str, float]] = None,
    road_context: Optional[Dict[str, object]] = None,
    route_waypoint: Optional[Dict[str, float]] = None,
) -> str:
    """Prompt for picking a recovery primitive and optional lateral distance.

    ``candidate_rows`` must already carry ``candidate_index``, ``primitive_name``
    and ``summary``. In B5, preverified parameter ranges make this a one-shot
    choice; the concrete trajectory is checked afterward only as a safety
    assertion and is never returned to the model as retry context.
    ``failure_reason`` is optional legacy context. Table-2 ablations leave it
    empty and make the model reason directly from the raw scene plus the PDMS
    score and metric breakdown. No trajectories or markers are drawn; candidates are
    conveyed purely as the numeric list below.
    """
    camera_line_1, _camera_line_2 = describe_vlm_camera_inputs(camera_order)
    # Header index list stays in ascending order even when the v2 block below
    # reorders the rendered rows.
    candidate_index_list = ", ".join(str(row["candidate_index"]) for row in candidate_rows)
    failure_reason_text = (
        failure_reason.strip()
        if isinstance(failure_reason, str) and failure_reason.strip()
        else "(no prior explanation available; infer the hazard from the images)"
    )
    retry_text = ""
    if previous_attempts:
        retry_text = "\nPrevious rejected selections:\n" + "\n".join(
            f"- {attempt}" for attempt in previous_attempts
        )
    recovery_process_text = (
        failure_memory.strip()
        if isinstance(failure_memory, str) and failure_memory.strip()
        else "(no prior recovery attempt in this episode)"
    )
    gate_text = (
        ", ".join(f"{key}={float(value):.2f}" for key, value in (verifier_gates or {}).items())
        or "unavailable"
    )
    soft_text = (
        ", ".join(f"{key}={float(value):.2f}" for key, value in (verifier_soft or {}).items())
        or "unavailable"
    )
    score_text = (
        "unavailable"
        if verifier_score is None
        else f"{float(verifier_score):.3f}"
    )
    threshold_text = (
        "unavailable"
        if verifier_threshold is None
        else f"{float(verifier_threshold):.3f}"
    )
    road_context_lines = _format_road_context_lines(road_context)
    route_waypoint_lines = _format_route_waypoint_lines(route_waypoint)
    off_corridor = bool(road_context) and not road_context.get("ego_on_corridor", True)
    has_distances = any(
        row.get("distance_to_lane_m") is not None for row in candidate_rows
    )
    road_rule_line = ""
    if off_corridor:
        road_rule_line = (
            "\n- The ego is off the corridor: strongly prefer candidates with ends_on_driving_lane=yes."
        )
        if has_distances:
            road_rule_line += (
                " Among ends_on_driving_lane=no candidates, prefer the smallest distance_to_lane_m,"
                " and choose lateral_distance_m to close the remaining gap when the range allows."
            )
            # Presentation-level v2 signals; candidate_index labels are preserved
            # so selected_index mapping is unaffected. (1) order rows
            # best-return-first (VLM choices are order-biased); (2) when NO
            # on-lane candidate exists — the saturated ep-1023 regime — tag the
            # argmin so the numeric comparison becomes a lookup. Advisory only:
            # visual judgment can still override (e.g. a barrier on the
            # straight-line path to the lane).
            def _order_key(row):
                if row.get("ends_on_driving_lane"):
                    return (0, 0.0)
                d = row.get("distance_to_lane_m")
                return (1, float(d)) if d is not None else (2, 0.0)

            candidate_rows = sorted(candidate_rows, key=_order_key)
            if not any(row.get("ends_on_driving_lane") for row in candidate_rows):
                with_d = [
                    row for row in candidate_rows
                    if row.get("distance_to_lane_m") is not None
                ]
                if with_d:
                    closest = min(
                        with_d, key=lambda row: float(row["distance_to_lane_m"])
                    )
                    candidate_rows = [
                        {**row, "_closest_to_lane": row is closest}
                        for row in candidate_rows
                    ]
                    road_rule_line += " The closest is marked closest_to_lane=yes."
    # Rendered AFTER the v2 block so its reordering/tagging is reflected.
    candidate_text = _format_recovery_candidate_text(candidate_rows)

    return f"""
You are an autonomous-driving recovery planner. The vehicle's normal trajectory
was just rejected by the safety verifier and the car must execute a fallback
recovery maneuver. Choose a maneuver and, when offered, a lateral displacement.
The resulting trajectory will be checked by the safety verifier before execution.
A previously rejected choice must be revised rather than repeated.

Inputs:
- Visual context: {camera_line_1}
- The image(s) show the raw driving scene only; no trajectories or markers are drawn on them.
- Route objective: "{route_instruction}"{route_waypoint_lines}{road_context_lines}
- The candidate recovery maneuvers are described numerically below (they are NOT drawn on the image).

PDMS verifier context (the total score fell below the recovery threshold):
- Score: {score_text}
- Threshold: {threshold_text}
- Gate metrics: {gate_text}
- Soft diagnostics: {soft_text}

Optional prior explanation (normally disabled for this ablation):
{failure_reason_text}

Compact recovery process context (only the previous decision outcome and actions
to avoid in the current unresolved episode):
{recovery_process_text}
{retry_text}

Candidate recovery maneuvers ({len(candidate_rows)} options, indices: {candidate_index_list}):
{candidate_text}

Task (choose a direction, then a maneuver):
- Reason directly from the scene, PDMS score context, and available candidates.
- Choose exactly ONE direction represented by at least one listed candidate, then choose the best candidate moving in that direction. Do not propose an unavailable direction or maneuver.
- For a candidate with a lateral-distance range, choose one value inside its inclusive range. A `pdms_threshold_safe_lateral_distance_m` range was preverified; do not request another range or retry.
- For every other candidate, return `lateral_distance_m` as null.
- Prefer maneuvers that clear the obstacle/conflict AND keep the ego moving productively.
- Only choose "stop" or a reversing direction when the images and the failure reason indicate no forward or lateral maneuver is appropriate.{road_rule_line}

Output rules:
- Return ONLY valid JSON with exactly the seven fields below.
- Put the decision fields first. Always emit `selected_index` and `maneuver`.
- Keep `direction_analysis` to at most 20 words: compare only the relevant available directions.
- Keep `reasoning` to at most 20 words: state only why the selected maneuver resolves the hazard.
- Do not repeat the full scene description, failure reason, route objective, or candidate geometry.

Return this exact schema:
{{
  "chosen_direction": "<one of: forward | left | right | back | stop>",
  "selected_index": <int, one of: {candidate_index_list}>,
  "maneuver": "<the maneuver name of the candidate you picked; must match selected_index and the chosen_direction>",
  "lateral_distance_m": <float inside the selected candidate range, otherwise null>,
  "direction_analysis": "<20 words or fewer>",
  "reasoning": "<20 words or fewer>",
  "confidence": <float between 0 and 1>
}}
""".strip()


def build_recovery_sequence_prompt(
    route_instruction: str,
    *,
    action_specs: Dict[str, Dict[str, object]],
    max_steps: int,
    failure_reason: Optional[str],
    first_action_names: Optional[Sequence[str]] = None,
    failure_memory: Optional[str] = None,
    sequence_context: Optional[Dict[str, object]] = None,
    camera_order: Sequence[str] = ("CAM_FRONT",),
    verifier_score: Optional[float] = None,
    verifier_threshold: Optional[float] = None,
    verifier_gates: Optional[Dict[str, float]] = None,
    verifier_soft: Optional[Dict[str, float]] = None,
    road_context: Optional[Dict[str, object]] = None,
    route_waypoint: Optional[Dict[str, float]] = None,
    first_action_evaluations: Optional[Dict[str, Dict[str, object]]] = None,
    previous_attempts: Optional[Sequence[str]] = None,
    memory_guidance: str = "avoid",
) -> str:
    """Prompt the VLM for a bounded sequence of whitelisted recovery actions.

    The normal-plan verifier result explains why recovery was entered. Separate
    per-primitive PDMS results tell the VLM which recovery actions have already
    passed deterministic safety checks, so it can prefer useful progress over
    a needlessly conservative stop.
    """
    camera_line_1, _camera_line_2 = describe_vlm_camera_inputs(camera_order)
    failure_reason_text = (
        failure_reason.strip()
        if isinstance(failure_reason, str) and failure_reason.strip()
        else "(infer the hazard directly from the images and verifier context)"
    )
    memory_text = (
        failure_memory.strip()
        if isinstance(failure_memory, str) and failure_memory.strip()
        else "(no prior recovery attempt in this episode)"
    )
    sequence_context_text = (
        json.dumps(sequence_context, ensure_ascii=False, separators=(",", ":"))
        if sequence_context
        else "(no previous sequence intent)"
    )
    road_context_lines = _format_road_context_lines(road_context)
    route_waypoint_lines = _format_route_waypoint_lines(route_waypoint)
    action_lines = []
    for name, spec in action_specs.items():
        fields = [
            f"fixed_duration={float(spec['fixed_duration_sec']):.1f}s"
        ]
        if spec.get("target_speed_mps") is not None:
            speed = spec["target_speed_mps"]
            fields.append(f"target_speed_mps=[{float(speed[0]):.1f},{float(speed[1]):.1f}]")
        if spec.get("lateral_distance_m") is not None:
            lateral = spec["lateral_distance_m"]
            fields.append(
                f"lateral_distance_m=[{float(lateral[0]):.1f},{float(lateral[1]):.1f}]"
            )
        action_lines.append(
            f"- {name}: {spec.get('description', '')}; " + "; ".join(fields)
        )
    action_text = "\n".join(action_lines)
    first_actions = (
        list(action_specs)
        if first_action_names is None
        else [
            str(name)
            for name in first_action_names
            if str(name) in action_specs
        ]
    )
    first_action_text = ", ".join(first_actions) or "(none)"
    evaluation_lines = []
    for name in action_specs:
        evaluation = dict((first_action_evaluations or {}).get(name) or {})
        if not evaluation:
            continue
        score = evaluation.get("pdms_score")
        threshold = evaluation.get("pdms_threshold")
        eligible = bool(evaluation.get("eligible", False))
        gates = dict(evaluation.get("gates") or {})
        gate_text = " ".join(
            f"{key}={float(gates[key]):.2f}"
            for key in ("nc", "dac")
            if key in gates
        )
        rejection_reasons = [
            str(reason) for reason in evaluation.get("rejection_reasons", [])
        ]
        fields = [
            f"PDMS={float(score):.3f}" if score is not None else "PDMS=unknown",
            "PASS" if eligible else "REJECTED",
        ]
        if threshold is not None:
            fields.append(f"threshold={float(threshold):.3f}")
        if gate_text:
            fields.append(gate_text)
        if rejection_reasons:
            fields.append("reasons=" + ",".join(rejection_reasons))
        evaluation_lines.append(f"- {name}: " + "; ".join(fields))
    evaluation_text = (
        "\nFirst-action deterministic safety results:\n"
        + "\n".join(evaluation_lines)
        + "\nPASS means the concrete primitive already cleared PDMS and road-safety checks."
        if evaluation_lines
        else ""
    )
    productive_passes = [
        name
        for name in first_actions
        if name not in {"brake", "wait"}
        and bool(
            ((first_action_evaluations or {}).get(name) or {}).get(
                "eligible", False
            )
        )
    ]
    productive_text = ", ".join(productive_passes) or "(none)"
    retry_text = ""
    if previous_attempts:
        retry_text = (
            "\nRejected choices from this selection cycle:\n"
            + "\n".join(f"- {item}" for item in previous_attempts)
            + "\nDo not repeat a rejected choice; correct it using the feedback.\n"
        )

    def _example_step(action: str) -> Dict[str, object]:
        spec = action_specs[action]
        speed = spec.get("target_speed_mps")
        lateral = spec.get("lateral_distance_m")
        return {
            "action": action,
            "target_speed_mps": (
                None
                if speed is None
                else round((float(speed[0]) + float(speed[1])) / 2.0, 2)
            ),
            "lateral_distance_m": (
                None
                if lateral is None
                else round((float(lateral[0]) + float(lateral[1])) / 2.0, 2)
            ),
        }

    patterns = []
    pattern_candidates = [
        ("bypass right and return", ("swerve_right", "creep", "swerve_left")),
        ("bypass left and return", ("swerve_left", "creep", "swerve_right")),
        ("create rear-right space, then advance", ("back_right", "creep")),
        ("create rear-left space, then advance", ("back_left", "creep")),
        ("create straight-back space (heading preserved), then advance", ("creep_back", "creep")),
        ("stop, then wait", ("brake", "wait")),
    ]
    for label, actions in pattern_candidates:
        if (
            len(actions) <= max(1, int(max_steps))
            and actions[0] in first_actions
            and all(action in action_specs for action in actions)
        ):
            patterns.append((label, actions))
    # Memory guidance rule: "avoid" keeps the shipped instruction verbatim;
    # "productivity" pairs with the parameterized attempts ledger -- prefer
    # untried actions or new parameters over zero-progress repeats.
    memory_rule_line = (
        "- Avoid failed actions in the history unless the scene changed."
        if memory_guidance != "productivity"
        else "- Prefer untried actions or new parameters over attempts that "
        "bought no route progress; do not repeat an exact rejected attempt."
    )
    # Menu-aware guidance (ported x10/x10v2 exemplars; the shift/veer variants
    # were dropped with those atoms). Each line renders only when the named
    # actions are on the menu, so every other arm keeps byte-identical prompts.
    creep_back_offered = "creep_back" in action_specs
    back_steer_offered = (
        "back_left" in action_specs or "back_right" in action_specs
    )
    forward_lateral_offered = any(
        name in action_specs
        for name in (
            "swerve_left", "swerve_right", "back_bypass_left",
            "back_bypass_right", "three_point", "lane_left", "lane_right",
            "lane_bypass_left", "lane_bypass_right",
        )
    )
    menu_guidance = ""
    back_options = []
    if creep_back_offered:
        back_options.append("creep_back (straight back, heading preserved)")
    if back_steer_offered:
        back_options.append(
            "back_left/back_right (back away while angling the nose toward a gap)"
        )
    if back_options and not forward_lateral_offered:
        # MRM menu: no forward-lateral atom is offered, so the exemplar must
        # not recommend one. Recovery's job is space-making; the nominal
        # planner drives the bypass once there is room. With an obstacle
        # ahead, the steered backs are encouraged as first-class options, not
        # just straight backing: they buy the same space AND posture.
        menu_guidance += (
            "\n- To escape a blockage, create space by backing up: "
            + ", or ".join(back_options)
            + ";"
            "\n  the normal planner will drive around the obstacle once there is room."
        )
    if back_steer_offered:
        # Teach the reverse-steer mirror so the model aims the correct way
        # (the nose swings OPPOSITE the drift while reversing).
        menu_guidance += (
            "\n- To aim at a bypass gap while retreating: back_right backs away"
            "\n  and angles the nose LEFT (for a gap on the left); back_left"
            "\n  angles the nose RIGHT (for a gap on the right). When the"
            "\n  obstacle sits to one side, prefer the back that angles your"
            "\n  nose toward the open side over backing straight."
        )
    _recurrence_names = [
        name
        for name in ("creep_back", "back_left", "back_right")
        if name in action_specs
    ]
    brake_recurrence = (
        " If braking has already recurred without progress,"
        "\n  prefer creating space with "
        + "/".join(_recurrence_names)
        + " over waiting again."
        if _recurrence_names
        else ""
    )
    pattern_text = ""
    if patterns:
        pattern_lines = [
            f"- {label}: {' -> '.join(actions)}"
            for label, actions in patterns
        ]
        example_actions = patterns[0][1]
        example_json = json.dumps(
            {
                "steps": [_example_step(action) for action in example_actions],
                "reasoning": "Execute the ordered recovery pattern and finish in a safer route-aligned state.",
                "confidence": 0.9,
            },
            indent=2,
        )
        pattern_text = (
            "\nUseful multi-step patterns (only actions available in this call):\n"
            + "\n".join(pattern_lines)
            + "\n\nValid multi-step JSON example:\n"
            + example_json
            + "\n"
        )

    return f"""
Select ONE safe, short recovery sequence from the driving images.

{camera_line_1}
- Route: "{route_instruction}"{route_waypoint_lines}{road_context_lines}
- Failure: {failure_reason_text}

Recovery history:
{memory_text}

Previous sequence execution state:
{sequence_context_text}

Actions (exact ids; ranges are inclusive):
{action_text}
{evaluation_text}
Productive first actions that passed deterministic safety checks: {productive_text}
{pattern_text}
{retry_text}

Rules:
- Return 1 to {int(max_steps)} ordered actions, never an empty list.
- The orchestrator executes only the first returned action, then returns to Loop 1. Later actions are saved as intent for the next recovery decision.
- Use the previous sequence as context: keep its remaining intent when still safe, or replace it when the new images require a different plan.
- Prefer a productive multi-step pattern when the scene supports the follow-up actions.{menu_guidance}
- First action: one of {first_action_text}. Later actions: any listed action.
- Action durations are fixed by the system. Do not output duration_sec.
- Use exact action ids and valid parameter ranges. Unused parameter fields must be null.
{memory_rule_line}
- Treat PDMS as a safety/feasibility signal: among visually appropriate PASS actions, prefer a route-aligned action that makes progress.
- When one or more productive first actions are listed as PASS and the image supports them, do not return a stop-only brake/wait sequence and do not claim that no maneuver is safe.
- Use brake or wait as the whole sequence only when no passing movement is visually safe, or an immediate collision requires stopping first.{brake_recurrence}
- If no safe movement exists, output brake or wait as a step; never return empty steps.

Return only valid JSON:
{{
  "steps": [
    {{
      "action": "<available action name>",
      "target_speed_mps": <float or null>,
      "lateral_distance_m": <float or null>
    }}
  ],
  "reasoning": "<25 words or fewer>",
  "confidence": <float between 0 and 1>
}}
""".strip()


def _format_recovery_plan_previous_result(
    sequence_context: Optional[Dict[str, object]],
) -> str:
    if not sequence_context:
        return (
            "- Earlier attempts, if any, are listed in [Recovery memory].\n"
            "- No remaining plan is active."
        )
    completed = list(sequence_context.get("completed_steps") or [])
    in_flight = sequence_context.get("in_flight_step")
    remaining = list(sequence_context.get("remaining_steps") or [])
    last_action = in_flight or (completed[-1] if completed else None)
    lines = []
    if isinstance(last_action, dict):
        lines.append(
            "- Previous action: "
            + str(last_action.get("action") or "unknown")
            + "."
        )
    else:
        lines.append("- Earlier attempts, if any, are listed in [Recovery memory].")
    outcome = sequence_context.get("last_outcome")
    if outcome:
        lines.append(f"- Previous outcome: {outcome}.")
    lines.append(
        f"- Remaining saved actions: {len(remaining)}."
        if remaining
        else "- No remaining plan is active."
    )
    return "\n".join(lines)


def _filter_recovery_plan_untried_actions(
    failure_memory: str,
    safe_names: Sequence[str],
) -> str:
    """Keep ``Untried:`` memory entries inside the current safe menu.

    Recovery memory is assembled before the structured VLM menu is narrowed,
    so its compact ``Untried:`` line can otherwise leak action ids that failed
    the current PDMS/menu gate. Other history lines remain byte-for-byte intact.
    """
    safe = {str(name) for name in safe_names}
    filtered_lines = []
    for line in failure_memory.splitlines():
        stripped = line.lstrip()
        if not stripped.startswith("Untried:"):
            # Older recovery events may contain the retired VLM parameter
            # name. Keep the diagnostic value without leaking that field back
            # into the new prompt/output contract.
            filtered_lines.append(
                line.replace("lateral_distance_m", "internal_geometry_m")
            )
            continue
        indent = line[: len(line) - len(stripped)]
        raw_names = stripped.partition(":")[2].strip()
        names = [name.strip() for name in raw_names.split(",")]
        kept = [name for name in names if name in safe]
        filtered_lines.append(
            f"{indent}Untried: {', '.join(kept) if kept else 'none'}"
        )
    return "\n".join(filtered_lines)


def build_recovery_plan_prompt(
    route_instruction: str,
    *,
    action_specs: Dict[str, Dict[str, object]],
    first_action_names: Optional[Sequence[str]] = None,
    failure_memory: Optional[str] = None,
    sequence_context: Optional[Dict[str, object]] = None,
    camera_order: Sequence[str] = ("CAM_FRONT",),
    road_context: Optional[Dict[str, object]] = None,
    previous_attempts: Optional[Sequence[str]] = None,
    first_action_evaluations: Optional[Dict[str, Dict[str, object]]] = None,
    bypass_first: bool = False,
    bypass_action: str = "lane_bypass_left",
    temporal_image_age_sec: Optional[float] = None,
) -> str:
    """Build the concise probabilistic primitive-ranking prompt.

    The caller supplies only PDMS-passed primitive ids in ``first_action_names``.
    The model states the visible blockage, chooses an escape direction, and
    assigns probabilities to a variable-size subset of that safe menu. Geometry
    parameters are system-owned. In temporal-tree mode the scorer has already
    searched and cached the winning execution profile before the VLM call.
    """
    camera_line_1, _camera_line_2 = describe_vlm_camera_inputs(camera_order)
    if (
        temporal_image_age_sec is not None
        and float(temporal_image_age_sec) > 0.0
        and tuple(camera_order) == ("CAM_FRONT",)
    ):
        camera_line_1 = (
            "Two front-facing driving images in chronological order: the first "
            f"was captured {float(temporal_image_age_sec):.2f} seconds before "
            "the second; the second image is the current scene. Compare the "
            "same blocking object across both images to decide whether it is "
            "parked/static or ordinary moving traffic."
        )
    road_context_lines = _format_road_context_lines(road_context)
    safe_names = (
        list(action_specs)
        if first_action_names is None
        else [str(name) for name in first_action_names if str(name) in action_specs]
    )
    safe_names = [
        name for name in safe_names if name not in RECOVERY_PLAN_EXCLUDED_PRIMITIVES
    ]
    bypass_action = str(bypass_action).strip()
    bypass_side = "right" if bypass_action.endswith("_right") else "left"
    bypass_side_upper = bypass_side.upper()
    bypass_field = f"needs_lane_bypass_{bypass_side}"
    regular_safe_names = (
        [name for name in safe_names if name != bypass_action]
        if bypass_first
        else safe_names
    )
    action_lines = []
    temporal_scorecards_present = False
    for name in regular_safe_names:
        spec = action_specs[name]
        action_lines.append(f"- {name}\n  {spec.get('description', 'recovery maneuver')}.")
        speed = spec.get("target_speed_mps")
        if speed is not None:
            action_lines.append(
                f"  target_speed_mps: {float(speed[0]):.1f} to {float(speed[1]):.1f}.\n"
                "  The system chooses trajectory geometry internally."
            )
        else:
            action_lines.append(
                "  target_speed_mps must be null.\n"
                "  The system chooses trajectory geometry internally."
            )
        if name == "back_left":
            action_lines.append("  This angles the vehicle nose toward the right.")
        elif name == "back_right":
            action_lines.append("  This angles the vehicle nose toward the left.")
        evaluation = dict((first_action_evaluations or {}).get(name) or {})
        temporal = dict(evaluation.get("temporal_search") or {})
        if temporal:
            temporal_scorecards_present = True
            profile = dict(temporal.get("best_profile") or {})
            action_lines.append(
                "  Temporal-tree scorer: "
                f"best={float(temporal.get('best_score', 0.0)):.3f}; "
                f"robust={float(temporal.get('robust_score', 0.0)):.3f}; "
                f"feasible={int(temporal.get('feasible_count', 0))}/"
                f"{int(temporal.get('expanded_count', 0))}; "
                f"winning_profile={json.dumps(profile, ensure_ascii=False, separators=(',', ':'))}."
            )
            action_lines.append(
                "  The winning profile is already cached and verified. Copy its "
                "target_speed_mps into the JSON only when that field is present; "
                "otherwise use null. Do not choose new numerical parameters."
            )
            start_delay = profile.get("start_delay_sec")
            if start_delay is not None and float(start_delay) > 0.0:
                action_lines.append(
                    "  This cached profile brakes/remains stopped for "
                    f"{float(start_delay):.1f} s before beginning the maneuver."
                )
    action_text = "\n".join(action_lines) or "(no safe primitive candidate)"

    if bypass_first and bypass_action in action_specs:
        bypass_spec = action_specs[bypass_action]
        bypass_evaluation = dict(
            (first_action_evaluations or {}).get(bypass_action) or {}
        )
        bypass_temporal = dict(
            bypass_evaluation.get("temporal_search") or {}
        )
        bypass_scorecard = ""
        if bypass_temporal:
            temporal_scorecards_present = True
            bypass_scorecard = (
                "\n  Temporal-tree scorer: "
                f"best={float(bypass_temporal.get('best_score', 0.0)):.3f}; "
                f"robust={float(bypass_temporal.get('robust_score', 0.0)):.3f}; "
                f"feasible={int(bypass_temporal.get('feasible_count', 0))}/"
                f"{int(bypass_temporal.get('expanded_count', 0))}; "
                "the winning execution profile is cached by the system."
            )
        bypass_text = (
            f"- {bypass_action}\n"
            f"  {bypass_spec.get('description', 'complete side-lane bypass maneuver')}.\n"
            f"  This is one complete primitive: borrow the adjacent {bypass_side} "
            "driving lane, pass the blockage, "
            "return to the original lane.\n"
            "  target_speed_mps must be null.\n"
            "  Its current traffic/topology execution gate is intentionally not "
            f"part of this decision.{bypass_scorecard}"
        )
    else:
        # Do not leak a topology-filtered action back into the prompt.  The
        # caller normally disables bypass-first mode when this happens; the
        # empty fallback is a second line of defense for direct prompt users.
        bypass_text = ""

    previous_result = _format_recovery_plan_previous_result(sequence_context)
    memory_text = (
        _filter_recovery_plan_untried_actions(
            str(failure_memory).strip(), safe_names
        )
        if isinstance(failure_memory, str) and failure_memory.strip()
        else "(no prior recovery memory)"
    )
    retry_text = (
        "\n\n[Rejected outputs from this selection cycle]\n"
        + "\n".join(f"- {item}" for item in previous_attempts)
        if previous_attempts
        else ""
    )
    if bypass_first:
        return f"""
{camera_line_1}

[Driving context]
- Intended route direction: {route_instruction}.{road_context_lines}
- The obstacle and open driving space must be inferred from the image.

[Previous recovery result]
{previous_result}

[Recovery memory]
{memory_text}

[Lane bypass decision]
Decide whether the vehicle must pass the blocking object
by temporarily borrowing the adjacent driving lane on the {bypass_side_upper},
passing the blockage, and returning to the original route lane. The adjacent
lane may carry same-direction or opposing traffic; that distinction must not
change the necessity decision.

{bypass_text}

Set {bypass_field} to true only when this complete {bypass_side}-side
maneuver is necessary in the visible scene and the blockage will not clear by
simply waiting for the object ahead to continue normally. Do not use the
current presence or direction of traffic in the adjacent lane to answer false:
traffic only controls when execution may start.

Use this concrete rule:
- true for a localized static blockage in the route lane when an adjacent
  driving lane provides a path around it and back to the route;
- false when a wall/barrier closes the whole road with no driving path around
  it: stop and wait for the road to reopen, even if this takes many seconds;
- false when the current route lane remains usable without this maneuver,
  ordinary moving traffic ahead can reasonably clear it by continuing, or a
  one-way lane change without returning is the correct route-following action.
Do not require the adjacent lane to look empty or even be fully visible in the
image. The system, not you, will determine its traffic direction and find a
safe execution time after a true answer.
Roadside parked cars are not route-lane blockages. A car maintaining its place
in front of ego across two images may be moving at a similar speed; this is
not evidence that it is parked. Prefer brake/wait for ordinary following traffic.
A wall can display a photograph of cars, trees or an open road. Treat the solid
panel as the obstacle, not the scene printed on it. Do not creep toward a wall
or select a lateral maneuver through it. Waiting safely is useful progress.

- If {bypass_field} is true: return exactly one candidate,
  {bypass_action}, with probability 1.0 and escape_direction "{bypass_side}". The
  system will lock this intent, brake and
  wait if necessary, then commit that primitive to completion when its safety
  gate opens. Do not combine it with lane changes, swerves, SparseDrive output,
  or any other recovery action while waiting or executing.
- If {bypass_field} is false: never return {bypass_action}; choose
  only from the regular safe primitives below.

[Regular safe primitive candidates]
Only used when {bypass_field} is false:

{action_text}
{retry_text}

[Task]
1. Return {bypass_field} as a JSON boolean. It means specifically
   "must temporarily use the adjacent lane on the {bypass_side}, pass the
   blockage, and return to the original route lane".
2. Write front_blockage as one short sentence explaining the visible reason the
   path directly ahead cannot be used.
3. Select escape_direction: forward, left, right, back, or stop.
4. Apply the mutually exclusive policy above:
   - true => exactly [{bypass_action}] with probability 1.0;
   - false => {"exactly 1 regular candidate" if temporal_scorecards_present else "0 to 5 regular candidates"} and no {bypass_action}.

If no regular primitive is visibly appropriate when the bypass decision is
false, select escape_direction "stop" and return "candidates": [].

[Output format]
Return only valid JSON using this structure:
{{
  "{bypass_field}": true,
  "front_blockage": "one short sentence based only on visible evidence",
  "escape_direction": "{bypass_side}",
  "candidates": [
    {{
      "primitive": {{
        "action": "{bypass_action}",
        "target_speed_mps": null
      }},
      "probability": 1.0,
      "reasoning": "the fixed route-lane blockage must eventually be passed"
    }}
  ]
}}

Fill target_speed_mps only for a regular action that supports it; otherwise use
null. Output JSON only; do not use Markdown fences.
""".strip()

    return f"""
{camera_line_1}

[Driving context]
- Intended route direction: {route_instruction}.{road_context_lines}
- The obstacle and open driving space must be inferred from the image.

[Previous recovery result]
{previous_result}

[Recovery memory]
{memory_text}

[Safe primitive candidates]
Only the following preverified primitives may be selected:

{action_text}
{retry_text}

[Task]
First, write front_blockage as one short sentence explaining the visible reason
the path directly ahead cannot be used.
Cars parked beside a usable lane do not block that lane. Follow moving traffic
with brake/wait when needed. If a wall closes the road, stop and wait for it to
clear; do not repeatedly creep toward it or steer through it. A wall may carry
a photograph of an open road, cars or trees: the printed scene is not free space.

Second, select escape_direction:
- forward: cautious forward motion can clear the visible conflict.
- left: the safest open bypass space is on the left.
- right: the safest open bypass space is on the right.
- back: the vehicle must first create space behind it.
- stop: no movement is currently safe.

Third, return {"exactly 1 primitive candidate" if temporal_scorecards_present else "0 to 5 alternative primitive candidates"} for the next recovery
action. For each candidate, provide one supplied primitive and its probability.
Only use exact primitive IDs listed under [Safe primitive candidates]. Never
output an action that is not listed there.
{"The scorer has already searched and verified each listed primitive's temporal execution tree. Select semantically among those cached winning profiles; do not invent or modify numerical parameters." if temporal_scorecards_present else ""}
If none of the listed primitives is consistent with the visible road and open
space, select escape_direction "stop" and return "candidates": [].

[Output format]
Return only valid JSON using this structure:
{{
  "front_blockage": "one short sentence based only on visible evidence",
  "escape_direction": "forward, left, right, back, or stop",
  "candidates": [
    {{
    "primitive": {{
      "action": "one supplied primitive ID",
        "target_speed_mps": null
      }},
      "probability": 1.0,
      "reasoning": "short reason"
    }}
  ]
}}

Fill target_speed_mps only for an action that supports it; otherwise use null.
Output JSON only; do not use Markdown fences.
""".strip()
