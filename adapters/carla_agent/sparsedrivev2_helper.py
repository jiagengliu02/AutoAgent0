"""CARLA-agent integration for the SparseDriveV2 planner.

``setup_sparsedrivev2`` loads SparseDriveV2's own model once, wraps it behind the
shared ``PlannerService`` interface, and wires it into the AgenticDriving
orchestrator. ``run_step_sparsedrivev2`` then lets the orchestrator drive
generate -> select -> verify -> recover before the selected trajectory enters the
shared plan -> CARLA control path.

Config block (top-level ``sparsedrivev2`` key in the agent yaml):

    sparsedrivev2:
      repo_root: third_party/SparseDriveV2                  # bench2drive branch
      config: projects/configs/sparsedrive_stage2.py         # repo-relative or abs
      checkpoint: ckpt/sparsedrive_small_b2d_stage2.pth
      device: cuda:7
      python_bin: /path/to/sparsedrivev2/bin/python

Needs the ``full_6cam`` rig (bench2drive SparseDrive is a 6-camera surround
model) and the dedicated ``sparsedrive_b2d`` env (mmcv-full 1.7.2 / mmdet 2.28.2
/ compiled deformable_aggregation op).
"""
from __future__ import annotations

import logging
import os
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np

LOG = logging.getLogger(__name__)

_SPARSEDRIVEV2_MODEL_SOURCES = frozenset(
    {"sparsedrivev2", "sparsedrivev2_argmax"}
)
_SPARSEDRIVEV2_CANDIDATE_DEBUG_KEYS = frozenset(
    {
        "candidate_plans_xy",
        "candidate_lat_reg",
        "candidate_lon_reg_speed_mps",
        "candidate_scores",
        "candidate_original_indices",
        "num_model_candidates",
        "num_candidates",
        # These describe the model's native/default trajectory, not necessarily
        # the trajectory ultimately selected by the scorer. ``selected_plan``
        # is the single source of truth in selected-only prediction records.
        "lat_reg_final",
        "lon_reg_final_speed_mps",
        "traj_final_xy",
        "carla_local_plan_xy",
    }
)


def _visualization_candidate_indices(
    scoring_debug,
    proposal_count,
    *,
    selected_idx=None,
):
    """Return drawable model candidates after the HD-map hard filter.

    The scorer shortlist describes which trajectories were evaluated, while
    ``hd_map_hard_filter.rejected_indices`` records the subset whose final
    target lies outside the driving corridor. Those rejected trajectories must
    not remain visible in the top-down overlay or prediction debug payload.
    """
    proposal_count = max(0, int(proposal_count))
    shortlist_indices = scoring_debug.get("shortlist_indices")
    if shortlist_indices is not None:
        indices = np.asarray(shortlist_indices, dtype=np.int64).reshape(-1)
        indices = indices[(indices >= 0) & (indices < proposal_count)]
    else:
        indices = np.arange(proposal_count, dtype=np.int64)

    hd_map_filter = scoring_debug.get("hd_map_hard_filter") or {}
    if bool(hd_map_filter.get("active")):
        rejected = np.asarray(
            hd_map_filter.get("rejected_indices", []), dtype=np.int64
        ).reshape(-1)
        if len(rejected):
            indices = indices[~np.isin(indices, rejected)]
        # Do not resurrect proposal 0 when the map rejected every drawable
        # candidate. The selected stationary/recovery plan is rendered through
        # ``selected_plan`` separately; the unsafe model lines stay absent.
        return indices

    if len(indices) == 0 and proposal_count > 0:
        fallback_idx = 0 if selected_idx is None else int(selected_idx)
        if 0 <= fallback_idx < proposal_count:
            indices = np.asarray([fallback_idx], dtype=np.int64)
    return indices


def _route_target_viz_metadata(info):
    """Compact metadata shown beside SparseDriveV2's top-down target point."""
    return {
        "road_option": info.get("sparsedrive_command"),
        "route_distance_m": info.get("route_distance_m"),
        "route_lane_key": info.get("route_lane_key"),
        "ego_lane_key": info.get("ego_lane_key"),
        "route_lane_relation": info.get("route_lane_relation"),
        "route_target_lookahead_m": info.get("route_target_lookahead_m"),
        "target_point_source": info.get("target_point_source"),
        "route_reentry_active": bool(info.get("route_reentry_active", False)),
        "route_reentry_primitive": info.get("route_reentry_primitive"),
        "route_reentry_side": info.get("route_reentry_side"),
        "route_reentry_trigger_reason": info.get("route_reentry_trigger_reason"),
        "route_reentry_command_override": bool(
            info.get("route_reentry_command_override", False)
        ),
    }


def _target_point_for_visualization(agent, info, planner_debug, selected_source):
    """Return a target point expressed in the current ego frame.

    A model-selection tick may use the target recorded in that inference's
    debug payload because it is the exact input seen by SparseDriveV2. Recovery
    and commit-hold ticks do not run the model, so ``last_debug`` is stale and
    must not override the route target recomputed from the current ego pose.
    """
    candidates = []
    if str(selected_source) in _SPARSEDRIVEV2_MODEL_SOURCES:
        if isinstance(planner_debug, dict):
            input_info = planner_debug.get("input_info")
            if isinstance(input_info, dict):
                candidates.append(input_info.get("target_point"))
    candidates.append(info.get("target_point"))

    for target_point in candidates:
        try:
            target_point_arr = np.asarray(
                target_point, dtype=np.float32
            ).reshape(-1)
        except (TypeError, ValueError):
            continue
        if (
            target_point_arr.size >= 2
            and np.isfinite(target_point_arr[:2]).all()
        ):
            current = target_point_arr[:2].astype(float).tolist()
            agent._sparsedrivev2_last_target_point_input = current
            return current

    # An old ego-frame point cannot be re-used safely after the vehicle moves
    # or rotates. Hiding the marker is preferable to drawing it at a false
    # world location.
    return None


def _planner_debug_for_prediction(planner_debug, selected_source):
    """Return source-appropriate planner debug for a prediction record.

    ``last_debug`` belongs to the most recent SparseDriveV2 model inference. A
    recovery/hold result is produced without a new model inference, so attaching
    the complete previous payload would serialize stale 1024-candidate arrays on
    every control tick. Keep only planner identity for non-model selections; the
    caller adds the current selection/recovery metadata below.
    """
    if not isinstance(planner_debug, dict) or not planner_debug:
        return None
    if str(selected_source) in _SPARSEDRIVEV2_MODEL_SOURCES:
        return dict(planner_debug)
    return {
        "planner": str(planner_debug.get("planner", "sparsedrivev2_b2d")),
    }


def _selected_only_prediction_debug(planner_debug):
    """Drop candidate/native-trajectory arrays from persisted planner debug."""
    if not isinstance(planner_debug, dict):
        return None
    return {
        key: value
        for key, value in planner_debug.items()
        if key not in _SPARSEDRIVEV2_CANDIDATE_DEBUG_KEYS
    }


def setup_sparsedrivev2(agent) -> None:
    from agenticdriving_carla_helper import (
        apply_recovery_env,
        build_orchestrator,
        create_learned_planner_selector,
        get_agenticdriving_recovery_cfg,
        init_selection_state,
        resolve_recovery,
        setup_recovery_explainer,
        setup_vlm_selector,
    )

    sd_cfg = agent._cfg["sparsedrivev2"]
    repo_root = str(sd_cfg["repo_root"])
    ckpt = str(sd_cfg["checkpoint"])
    config = str(sd_cfg.get("config", "projects/configs/sparsedrive_stage2.py"))
    device = str(sd_cfg.get("device", "cuda"))

    # Env parity with launch.sh (used by the subprocess path).
    os.environ["SPARSEDRIVEV2_REPO_ROOT"] = repo_root
    os.environ["SPARSEDRIVEV2_CKPT"] = ckpt
    os.environ["SPARSEDRIVEV2_CONFIG"] = config
    os.environ["SPARSEDRIVEV2_DEVICE"] = device
    os.environ["SPARSEDRIVEV2_PYTHON_BIN"] = str(
        sd_cfg.get("python_bin", sys.executable)
    )

    repo_root_path = str(Path(repo_root).resolve())
    if repo_root_path not in sys.path:
        sys.path.insert(0, repo_root_path)

    from agenticdriving.adapters.carla.geometry import (
        info_to_pose,
        local_plan_to_world,
        world_points_to_current_local,
    )
    from services import SparseDriveV2SubprocessCarlaService

    # Keep the CARLA evaluator in its CARLA-compatible Python and run
    # SparseDriveV2 in the dedicated mmcv/mmdet env configured by python_bin.
    agent._sparsedrive = SimpleNamespace(
        last_debug={},
        num_poses=int(sd_cfg.get("output_num_poses", 8)),
    )
    # Shared control path (apply_plan_control) needs this to derive ego pose.
    agent._info_to_pose = info_to_pose
    # Needed by the async replan cache to reproject a cached plan onto the live
    # ego pose (_transform_cached_local_plan) on the control ticks between plans.
    agent._local_plan_to_world = local_plan_to_world
    agent._world_points_to_current_local = world_points_to_current_local

    # SparseDriveV2 emits poses at its trajectory interval (0.5 s).
    agent._plan_dt_sec = 0.5
    agent._sparsedrive_output_num_poses = int(agent._sparsedrive.num_poses)
    agent._sparsedrivev2_selected_candidate_idx = None
    agent._sparsedrivev2_recovery_states = None
    agent._sparsedrivev2_last_target_point_input = None
    agent._sparsedrivev2_last_loop1_lane_change_ts = None
    agent._sparsedrive_service = SparseDriveV2SubprocessCarlaService(agent, cfg=sd_cfg)

    _setup_async_replan(agent, sd_cfg)
    _setup_route_reentry(agent, sd_cfg)

    recovery_cfg = get_agenticdriving_recovery_cfg(agent._cfg)
    apply_recovery_env(
        recovery_cfg,
        str(sd_cfg.get("python_bin", sys.executable)),
        ("PLANNER_RULE_BASED_", "SPARSEDRIVEV2_RULE_BASED_"),
    )
    agent._sparsedrivev2_recovery = resolve_recovery(
        sd_cfg, ("PLANNER_RULE_BASED_", "SPARSEDRIVEV2_RULE_BASED_")
    )

    setup_vlm_selector(agent, planner_cfg=sd_cfg)
    setup_recovery_explainer(agent, planner_cfg=sd_cfg)

    agent._learned_selector = create_learned_planner_selector(
        agent,
        recovery_cfg=agent._sparsedrivev2_recovery,
        current_source_name="sparsedrivev2",
        learned_default_source="sparsedrivev2_argmax",
        plain_source="sparsedrivev2",
        score_fallback_key="proposal_score",
        planner_log_name="SparseDriveV2",
        strict_learned_argmax_lookup=True,
        q_key_prefix=False,
    )
    LOG.info(
        "SparseDriveV2 planner subprocess configured (device=%s, num_poses=%d, python=%s)",
        device, agent._sparsedrive_output_num_poses,
        os.environ["SPARSEDRIVEV2_PYTHON_BIN"],
    )
    init_selection_state(agent)

    from agenticdriving.planners.sparsedrivev2.recovery import SparseDriveOrchestrator

    agent._orchestrator = build_orchestrator(
        agent,
        planner_service=agent._sparsedrive_service,
        recovery_cfg=agent._sparsedrivev2_recovery,
        orchestrator_cls=SparseDriveOrchestrator,
    )


def _setup_route_reentry(agent, sd_cfg) -> None:
    """Configure HD-map lane-topology route re-entry.

    Defaults are intentionally local to SparseDriveV2: no Leaderboard route is
    mutated, and other planners keep their existing target-point behaviour.
    """
    agent._sparsedrivev2_route_lookahead_m = max(
        0.0, float(sd_cfg.get("route_lookahead_m", 5.0))
    )
    agent._sparsedrivev2_route_reentry_enabled = bool(
        sd_cfg.get("route_reentry_enabled", True)
    )
    agent._sparsedrivev2_route_reentry_lookahead_m = max(
        agent._sparsedrivev2_route_lookahead_m,
        float(sd_cfg.get("route_reentry_lookahead_m", 12.0)),
    )
    agent._sparsedrivev2_route_reentry_trigger_ticks = max(
        1, int(sd_cfg.get("route_reentry_trigger_ticks", 3))
    )
    agent._sparsedrivev2_route_reentry_max_adjacent_lanes = max(
        1, int(sd_cfg.get("route_reentry_max_adjacent_lanes", 2))
    )
    agent._sparsedrivev2_route_reentry_map_search_points = max(
        1, int(sd_cfg.get("route_reentry_map_search_points", 12))
    )
    agent._sparsedrivev2_route_reentry_completion_ticks = max(
        1, int(sd_cfg.get("route_reentry_completion_ticks", 4))
    )
    agent._sparsedrivev2_route_reentry_max_duration_sec = max(
        0.0, float(sd_cfg.get("route_reentry_max_duration_sec", 15.0))
    )
    agent._junction_route_heading_filter_enabled = bool(
        sd_cfg.get("junction_route_heading_filter_enabled", True)
    )
    agent._junction_route_heading_lookahead_m = max(
        0.0, float(sd_cfg.get("junction_route_heading_lookahead_m", 12.0))
    )
    agent._junction_route_heading_max_angle_deg = max(
        0.0,
        min(
            180.0,
            float(sd_cfg.get("junction_route_heading_max_angle_deg", 60.0)),
        ),
    )
    agent._sparsedrivev2_route_projection_search_back = max(
        0, int(sd_cfg.get("route_projection_search_back", 20))
    )
    agent._sparsedrivev2_route_projection_search_ahead = max(
        1, int(sd_cfg.get("route_projection_search_ahead", 200))
    )
    agent._sparsedrivev2_route_reentry_active = False
    agent._sparsedrivev2_route_reentry_settled_ticks = 0
    agent._sparsedrivev2_route_reentry_trigger_ticks_seen = 0
    agent._sparsedrivev2_route_reentry_pending_side = None
    agent._sparsedrivev2_route_reentry_primitive = None
    agent._sparsedrivev2_route_reentry_side = None
    agent._sparsedrivev2_route_reentry_trigger_reason = None
    agent._sparsedrivev2_route_reentry_started_at = None
    agent._sparsedrivev2_route_segment_index = None
    LOG.info(
        "SparseDriveV2 route re-entry: enabled=%s normal=%.1fm return=%.1fm "
        "HD-map adjacent-lane trigger=%d ticks complete=%d ticks "
        "max_adjacent=%d map_search=%d",
        agent._sparsedrivev2_route_reentry_enabled,
        agent._sparsedrivev2_route_lookahead_m,
        agent._sparsedrivev2_route_reentry_lookahead_m,
        agent._sparsedrivev2_route_reentry_trigger_ticks,
        agent._sparsedrivev2_route_reentry_completion_ticks,
        agent._sparsedrivev2_route_reentry_max_adjacent_lanes,
        agent._sparsedrivev2_route_reentry_map_search_points,
    )


def run_step_sparsedrivev2(agent, ego_state):
    from agenticdriving_carla_helper import (
        agent_brake_control,
        apply_plan_control,
        build_step_viz_payload,
    )
    from perception_helper import resolve_step_agents

    obs = ego_state["obs"]
    info = ego_state["info"]

    # Async pipeline: replan runs at replan_frequency (< control_frequency), so on
    # the control ticks between two plans we reuse the last selected plan instead
    # of calling the orchestrator again. Returns None when a fresh replan is due.
    cached_ctrl = _try_run_cached_sparsedrivev2_plan(agent, obs, info)
    if cached_ctrl is not None:
        return cached_ctrl

    privileged_agents = None
    if (
        agent._sparsedrivev2_recovery.enabled
        and agent._sparsedrivev2_recovery.include_privileged_info
    ) or (
        agent._orchestrator.verifier_enabled
        or agent._orchestrator.candidate_pdms_enabled
    ):
        try:
            privileged_agents = resolve_step_agents(agent, obs, info)
        except Exception:
            LOG.exception("SparseDriveV2 perception unavailable; braking this frame")
            return agent_brake_control(agent)

    try:
        step_result = agent._orchestrator.step(
            obs=obs,
            info=info,
            info_history=list(agent._info_history),
            privileged_info=privileged_agents,
            frame_index=agent._frame_index,
        )
    except Exception:
        LOG.exception("SparseDriveV2 planning step failed")
        return agent_brake_control(agent)

    selected_plan = step_result.selected_plan
    selected_score_raw = step_result.selected_score_raw
    selected_source = step_result.selected_source
    recovery_states = step_result.plan_payload.get("fallback_states")
    if recovery_states is None:
        agent._sparsedrivev2_recovery_states = None
    else:
        states_arr = np.asarray(recovery_states, dtype=np.float64)
        agent._sparsedrivev2_recovery_states = (
            states_arr[:, :2]
            if states_arr.ndim == 2
            and states_arr.shape[0] > 0
            and states_arr.shape[1] >= 2
            and np.isfinite(states_arr[:, :2]).all()
            else None
        )
    # The SparseDriveV2 PID uses this to retrieve the lateral/speed heads aligned
    # with the exact top-K proposal selected by the generic orchestrator. Recovery
    # payload indices refer to a different candidate pool, so never expose them as
    # model proposal indices.
    agent._sparsedrivev2_selected_candidate_idx = (
        step_result.plan_payload.get("selected_idx")
        if str(selected_source) in {"sparsedrivev2", "sparsedrivev2_argmax"}
        else None
    )
    output_num_poses = agent._sparsedrive_output_num_poses

    if selected_plan is None or len(selected_plan) == 0:
        return agent_brake_control(agent)

    scoring_debug = step_result.debug.get("scoring", {})
    verifier_metrics = {
        **(step_result.debug.get("verifier", {}).get("gates", {}) or {}),
        **(step_result.debug.get("verifier", {}).get("soft", {}) or {}),
    }
    # Draw/store the union of the two PDMS batches (End2End Top-30 plus every
    # trajectory in the target-point region). The planner still returns all
    # 1024 modes, but serializing all of them each tick would obscure the overlay
    # and balloon prediction JSON files.
    viz_indices = _visualization_candidate_indices(
        scoring_debug,
        len(step_result.proposals),
        selected_idx=step_result.plan_payload.get("selected_idx"),
    )
    viz_proposals = np.asarray(step_result.proposals)[viz_indices]
    viz_scores = np.asarray(step_result.scores)[viz_indices]

    # Finite display scores aligned with the original proposal array. In the
    # parallel flow only the two branch winners carry their PDMS values.
    combined_scores = scoring_debug.get("combined_scores")
    combined_scores_arr = (
        np.asarray(combined_scores, dtype=np.float32)
        if combined_scores is not None else None
    )
    if combined_scores_arr is not None and len(combined_scores_arr) == len(step_result.proposals):
        combined_scores_arr = combined_scores_arr[viz_indices]
    agent._step_viz = build_step_viz_payload(
        selected_plan=selected_plan,
        selected_source=selected_source,
        selected_score=selected_score_raw,
        proposals=viz_proposals,
        scores=viz_scores,
        output_num_poses=output_num_poses,
        # SparseDriveV2 already emits CARLA ego-local [right, forward] plans.
        proposals_already_local=True,
        # Per-step recovery strategy for the top-down overlay's top-left label.
        recovered=step_result.recovered,
        verifier_passed=step_result.verifier_passed,
        verifier_score=step_result.verifier_score,
        fallback_primitive=step_result.plan_payload.get("fallback_primitive"),
        fallback_mode=step_result.plan_payload.get("fallback_mode"),
        # PDMS sub-metrics {nc, dac, ttc, ep} (gates + soft terms).
        verifier_metrics=verifier_metrics,
        # This is the replan tick: it draws every candidate, so keep the selected
        # line thin here. The reuse ticks that follow draw it bold (default).
        selected_bold=False,
        candidate_color_scores=combined_scores_arr,
    )
    planner_debug = getattr(getattr(agent, "_sparsedrive", None), "last_debug", None)
    agent._step_viz["sparsedrivev2_target_point"] = (
        _target_point_for_visualization(
            agent, info, planner_debug, selected_source
        )
    )
    agent._step_viz["sparsedrivev2_target_point_meta"] = (
        _route_target_viz_metadata(info)
    )
    planner_debug_for_record = _planner_debug_for_prediction(
        planner_debug, selected_source
    )
    if planner_debug_for_record is not None:
        # Candidate-specific native lateral/speed heads stay complete on
        # agent._sparsedrive.last_debug for the controller. Only the copy placed
        # in visualization/prediction output is shortened to the regional set.
        for key in (
            "candidate_plans_xy",
            "candidate_lat_reg",
            "candidate_lon_reg_speed_mps",
            "candidate_scores",
        ):
            values = planner_debug_for_record.get(key)
            if isinstance(values, list) and len(values) == len(step_result.proposals):
                planner_debug_for_record[key] = [values[int(i)] for i in viz_indices]
        planner_debug_for_record["num_model_candidates"] = int(len(step_result.proposals))
        planner_debug_for_record["num_candidates"] = int(len(viz_indices))
        planner_debug_for_record["candidate_original_indices"] = viz_indices.tolist()

        # Keep compact full-pool region telemetry even though the heavyweight
        # candidate arrays above are shortened to the evaluated visualization set.
        # This makes it possible to tune radial bounds from all 1024 model modes
        # after a run without serializing all 1024 trajectories into every JSON.
        endpoint_radii = np.asarray(
            scoring_debug.get("endpoint_radii_m", []), dtype=np.float64
        ).reshape(-1)
        endpoint_bearings = np.asarray(
            scoring_debug.get("endpoint_bearings_deg", []), dtype=np.float64
        ).reshape(-1)
        finite_radii = endpoint_radii[np.isfinite(endpoint_radii)]
        finite_bearings = endpoint_bearings[np.isfinite(endpoint_bearings)]
        per_region_indices = scoring_debug.get("per_region_indices") or []
        if len(finite_radii) or per_region_indices:
            radius_percentiles = (
                {
                    f"p{percentile:02d}": float(
                        np.percentile(finite_radii, percentile)
                    )
                    for percentile in (1, 5, 10, 25, 33, 50, 67, 75, 90, 95, 99)
                }
                if len(finite_radii)
                else {}
            )
            bearing_percentiles = (
                {
                    f"p{percentile:02d}": float(
                        np.percentile(finite_bearings, percentile)
                    )
                    for percentile in (1, 5, 25, 50, 75, 95, 99)
                }
                if len(finite_bearings)
                else {}
            )
            planner_debug_for_record["region_selection"] = {
                "radial_bounds_m": scoring_debug.get("region_radii_m", []),
                "radial_bounds_mode": scoring_debug.get("region_radii_mode"),
                "sectors_per_layer": scoring_debug.get(
                    "region_sector_counts", []
                ),
                "end2end_topk": scoring_debug.get("end2end_topk"),
                "target_point_region_id": scoring_debug.get(
                    "target_point_region_id"
                ),
                "region_sampling_mode": scoring_debug.get(
                    "region_sampling_mode"
                ),
                "region_samples_per_region": scoring_debug.get(
                    "region_samples_per_region"
                ),
                "sampled_per_region_counts": scoring_debug.get(
                    "sampled_per_region_counts", []
                ),
                "pdms_traj_counts": scoring_debug.get("pdms_traj_counts", {}),
                "pdms_unique_traj_count": scoring_debug.get(
                    "pdms_unique_traj_count", 0
                ),
                "end2end_proposal_index": scoring_debug.get(
                    "end2end_proposal_index"
                ),
                "end2end_proposal_pdms": scoring_debug.get(
                    "end2end_proposal_pdms"
                ),
                "rule_based_proposal_index": scoring_debug.get(
                    "rule_based_proposal_index"
                ),
                "rule_based_proposal_pdms": scoring_debug.get(
                    "rule_based_proposal_pdms"
                ),
                "model_candidate_count": int(len(step_result.proposals)),
                "shortlist_count": int(len(viz_indices)),
                "per_region_candidate_counts": [
                    int(len(indices)) for indices in per_region_indices
                ],
                "endpoint_radius_percentiles_m": radius_percentiles,
                "endpoint_bearing_percentiles_deg": bearing_percentiles,
            }
        agent._step_viz["planner_debug"] = {
            **planner_debug_for_record,
            "selection": {
                "selected_source": str(selected_source),
                "selected_scorer": step_result.plan_payload.get("selected_scorer"),
                "selected_candidate_original_idx": step_result.plan_payload.get("selected_idx"),
                "selected_score": (
                    None if selected_score_raw is None else float(selected_score_raw)
                ),
                "verifier_score": step_result.verifier_score,
                "verifier_passed": step_result.verifier_passed,
                "verifier_recovered": step_result.recovered,
            },
        }

    # Candidate trajectories remain in _step_viz for the live top-down overlay,
    # but prediction JSONs persist only the actual selected trajectory plus
    # compact scorer/verifier metadata. This avoids duplicating a candidate pool
    # that is neither executed nor needed to reconstruct the chosen control.
    agent._step_viz["_prediction_selected_only"] = True
    agent._step_viz["_prediction_planner_debug"] = (
        _selected_only_prediction_debug(agent._step_viz.get("planner_debug"))
    )

    # Surface the recovery-reason VLM output (explain()) onto the top-down
    # overlay / front_topdown video. Set only on the tick a recovery event fires
    # (recovery_reason lands in step_result.debug then); the sticky-viz group in
    # record_post_step_visualization carries it across the commit-and-hold reuse
    # ticks that follow, and clears it on the next recovery-free verifier run.
    recovery_reason = step_result.debug.get("recovery_reason")
    if isinstance(recovery_reason, dict) and recovery_reason.get("reason"):
        agent._step_viz["vlm_reason"] = {
            "reason": recovery_reason.get("reason"),
            "primary_cause": recovery_reason.get("primary_cause"),
            "confidence": recovery_reason.get("confidence"),
        }

    _cache_sparsedrivev2_selected_plan(
        agent, selected_plan, info, selected_score_raw, selected_source
    )
    control_info = dict(info)
    control_info["verifier_metrics"] = verifier_metrics
    control_info["privileged_agents"] = privileged_agents
    ctrl = apply_plan_control(
        agent, selected_plan, control_info, selected_score_raw, selected_source
    )
    if isinstance(getattr(agent, "_last_applied_control", None), dict):
        agent._last_applied_control.update(
            {
                "sparsedrivev2_replan_cached": False,
                "sparsedrivev2_controls_since_replan": 1,
                "sparsedrivev2_replan_every_n": int(
                    getattr(agent, "_sparsedrivev2_replan_every_n", 1)
                ),
                "verifier_score": step_result.verifier_score,
                "verifier_passed": step_result.verifier_passed,
                "verifier_recovered": step_result.recovered,
            }
        )
    return ctrl


def _setup_async_replan(agent, sd_cfg) -> None:
    """Wire the async replan/control cadence from the yaml.

    ``control_frequency`` (Hz) is the CARLA control-tick rate (``run_step`` is
    called once per sim tick). ``replan_frequency`` (Hz) is how often we actually
    call the planner + scorer. With control=20 Hz and replan=1 Hz we run the
    planner once, then reuse that plan for ``control_frequency / replan_frequency``
    = 20 control ticks (reprojecting it to the live ego pose each tick) before the
    next replan. ``replan_frequency >= control_frequency`` (or absent) keeps the
    legacy every-tick behaviour.

    The verifier is deliberately NOT on this cadence: every reuse tick re-verifies
    the reprojected plan (see ``_verify_cached_sparsedrivev2_plan``), so it runs at
    ``control_frequency``. A failing verdict parks the reuse counter at ``every_n``,
    which makes the next tick take the full replan path.
    """
    control_frequency = max(1e-6, float(sd_cfg.get("control_frequency", 20.0)))
    replan_frequency = float(sd_cfg.get("replan_frequency", control_frequency))
    replan_frequency = min(max(replan_frequency, 1e-6), control_frequency)

    agent._sparsedrivev2_control_frequency = control_frequency
    agent._sparsedrivev2_replan_frequency = replan_frequency
    # Control ticks per plan, e.g. round(20 / 2) = 10. 1 disables caching.
    agent._sparsedrivev2_replan_every_n = max(
        1, int(round(control_frequency / replan_frequency))
    )
    agent._sparsedrivev2_controls_since_replan = 0
    agent._sparsedrivev2_cached_selected_plan = None
    agent._sparsedrivev2_cached_selected_pose = None
    agent._sparsedrivev2_cached_selected_score = None
    agent._sparsedrivev2_cached_selected_source = "sparsedrivev2_argmax"

    # Reprojection/trim knobs, keyed as ``_sparsedrivev2_replan_*`` so the shared
    # _trim_cached_local_plan(planner_key="sparsedrivev2") picks them up.
    agent._sparsedrivev2_replan_min_forward_m = max(
        0.0, float(sd_cfg.get("replan_min_forward_m", 2.0))
    )
    agent._sparsedrivev2_replan_trim_min_forward_m = float(
        sd_cfg.get("replan_trim_min_forward_m", 0.05)
    )
    agent._sparsedrivev2_replan_behind_tolerance_m = max(
        0.0, float(sd_cfg.get("replan_behind_tolerance_m", 0.25))
    )
    agent._sparsedrivev2_replan_max_start_distance_m = max(
        0.0, float(sd_cfg.get("replan_max_start_distance_m", 3.0))
    )
    agent._sparsedrivev2_replan_max_start_lateral_m = max(
        0.0, float(sd_cfg.get("replan_max_start_lateral_m", 2.5))
    )
    if agent._sparsedrivev2_replan_every_n > 1:
        LOG.info(
            "SparseDriveV2 async pipeline: control=%.1fHz replan=%.1fHz "
            "-> %d control ticks per plan",
            control_frequency,
            replan_frequency,
            agent._sparsedrivev2_replan_every_n,
        )


def _try_run_cached_sparsedrivev2_plan(agent, obs, info):
    """Reuse the last selected plan on a control tick between two replans.

    Returns a control dict when the cached plan is still valid for this tick, or
    None when a fresh replan is due (or caching is disabled).
    """
    from agenticdriving_carla_helper import (
        _transform_cached_local_plan,
        _trim_cached_local_plan,
        agent_brake_control,
        apply_plan_control,
        build_step_viz_payload,
    )

    # A committed recovery carries a time-indexed signed state sequence. Let the
    # orchestrator advance it every control tick; reusing the cached first slice
    # would freeze the maneuver in phase 1 and prevent reverse/forward changes.
    orchestrator = getattr(agent, "_orchestrator", None)
    if orchestrator is not None and orchestrator.recovery_hold_active:
        return None

    every_n = int(getattr(agent, "_sparsedrivev2_replan_every_n", 1))
    if every_n <= 1:
        return None

    cached_plan = getattr(agent, "_sparsedrivev2_cached_selected_plan", None)
    cached_pose = getattr(agent, "_sparsedrivev2_cached_selected_pose", None)
    since = int(getattr(agent, "_sparsedrivev2_controls_since_replan", 0))
    # A fresh replan is only allowed at the cadence boundary (since >= every_n),
    # not because a reused plan looks stale -> the planner runs exactly once every
    # ``every_n`` control ticks (strict cadence).
    if cached_plan is None or cached_pose is None or since <= 0 or since >= every_n:
        return None

    # This is a reuse tick: it counts toward the ``every_n`` budget no matter what
    # the reprojected plan looks like, so a short/stale plan can never pull the
    # next replan earlier than the fixed cadence.
    agent._sparsedrivev2_controls_since_replan = since + 1

    selected_plan = _transform_cached_local_plan(
        agent,
        np.asarray(cached_plan, dtype=np.float32),
        cached_pose,
        info,
        planner_name="SparseDriveV2",
    )
    selected_plan = _trim_cached_local_plan(
        agent, selected_plan, planner_key="sparsedrivev2"
    )

    selected_score_raw = getattr(agent, "_sparsedrivev2_cached_selected_score", None)
    selected_source = str(
        getattr(agent, "_sparsedrivev2_cached_selected_source", "sparsedrivev2_argmax")
    )
    cached_source = f"{selected_source}_cached"
    agent._sparsedrivev2_recovery_states = None

    if len(selected_plan) == 0:
        # Reprojected plan is fully behind the ego -> nothing to track this tick.
        # Hold (brake) but stay on the fixed cadence; do NOT replan early.
        if isinstance(getattr(agent, "_last_applied_control", None), dict):
            agent._last_applied_control.update(
                {
                    "sparsedrivev2_replan_cached": True,
                    "sparsedrivev2_replan_cache_empty": True,
                    "sparsedrivev2_controls_since_replan": since + 1,
                    "sparsedrivev2_replan_every_n": every_n,
                }
            )
        return agent_brake_control(agent)

    # Verifier runs at control_frequency, not replan_frequency: the reprojected
    # cached plan is re-verified on EVERY reuse tick, so a plan that goes unsafe
    # between two replans is caught within one control tick instead of after the
    # full every_n-tick window.
    verdict = _verify_cached_sparsedrivev2_plan(agent, selected_plan, obs, info)

    if getattr(verdict, "perception_failed", False):
        agent._sparsedrivev2_controls_since_replan = every_n
        return agent_brake_control(agent)

    agent._step_viz = build_step_viz_payload(
        selected_plan=selected_plan,
        selected_source=cached_source,
        selected_score=selected_score_raw,
        proposals=np.asarray(selected_plan, dtype=np.float32)[None, ...],
        scores=np.asarray(
            [0.0 if selected_score_raw is None else float(selected_score_raw)],
            dtype=np.float32,
        ),
        output_num_poses=agent._sparsedrive_output_num_poses,
        proposals_already_local=True,
        verifier_passed=verdict.passed,
        verifier_score=verdict.score,
        verifier_metrics=verdict.metrics,
    )
    agent._step_viz["sparsedrivev2_target_point"] = (
        _target_point_for_visualization(
            agent, info, None, cached_source
        )
    )
    agent._step_viz["sparsedrivev2_target_point_meta"] = (
        _route_target_viz_metadata(info)
    )
    agent._step_viz["_prediction_selected_only"] = True
    agent._step_viz["_prediction_planner_debug"] = None
    control_info = dict(info)
    control_info["verifier_metrics"] = dict(verdict.metrics or {})
    control_info["privileged_agents"] = verdict.privileged_agents
    ctrl = apply_plan_control(
        agent, selected_plan, control_info, selected_score_raw, cached_source
    )

    # A failed verify breaks the strict cadence: park the counter at every_n so
    # the NEXT tick fails the `since < every_n` gate above and runs the full
    # orchestrator step (generate -> select -> verify -> recover). This tick still
    # tracks the cached plan -- control has to emit something -- so the early
    # replan lands one tick later.
    controls_since = since + 1
    if verdict.score is not None and not verdict.passed:
        controls_since = every_n
        agent._sparsedrivev2_controls_since_replan = every_n
        LOG.info(
            "SparseDriveV2 async: cached-plan verify %.3f < %.3f at tick %d/%d "
            "-> forcing early replan",
            verdict.score, verdict.threshold, since + 1, every_n,
        )

    if isinstance(getattr(agent, "_last_applied_control", None), dict):
        agent._last_applied_control.update(
            {
                "sparsedrivev2_replan_cached": True,
                "sparsedrivev2_controls_since_replan": controls_since,
                "sparsedrivev2_replan_every_n": every_n,
                "verifier_score": verdict.score,
                "verifier_passed": verdict.passed,
                "verifier_recovered": False,
                "sparsedrivev2_replan_forced_early": (
                    verdict.score is not None and not verdict.passed
                ),
            }
        )
    return ctrl


def _verify_cached_sparsedrivev2_plan(agent, selected_plan, obs, info):
    """Score a reused (reprojected) plan with the gating verifier.

    Returns ``SimpleNamespace(score, passed, threshold, metrics)``; ``score`` is
    None when no verdict was produced (no verifier configured, a committed
    recovery maneuver is holding, or the verifier raised), in which case the
    caller leaves the replan cadence untouched.

    Deliberately calls ``verifier.score()`` alone and never
    ``recovery_tracker.observe_verifier()``: the tracker settles *pending
    recovery attempts* against one verdict per attempt, so feeding it the
    every-tick verdicts would settle an attempt up to every_n times over.
    Recovery bookkeeping stays exclusively on the orchestrator's replan path.
    """
    from perception_helper import resolve_step_agents

    blank = SimpleNamespace(
        score=None,
        passed=True,
        threshold=None,
        metrics={},
        privileged_agents=None,
    )

    orch = getattr(agent, "_orchestrator", None)
    if orch is None or not orch.verifier_enabled:
        return blank
    # Commit-and-hold is open-loop by definition; don't cut its window short.
    if orch.recovery_hold_active:
        return blank

    verify_info = dict(info)
    try:
        verify_info["privileged_agents"] = resolve_step_agents(agent, obs, info)
    except Exception:
        LOG.exception("SparseDriveV2 cached-plan perception unavailable")
        return SimpleNamespace(
            score=0.0,
            passed=False,
            threshold=float(orch.verifier_threshold),
            metrics={},
            privileged_agents=None,
            perception_failed=True,
        )
    try:
        score = float(orch.verifier.score(selected_plan, verify_info))
    except Exception:
        LOG.exception("SparseDriveV2 async: cached-plan verify failed")
        return blank

    threshold = float(orch.verifier_threshold)
    result = getattr(orch.verifier, "last_result", None)
    metrics = {}
    if result is not None:
        metrics = {**(result.gates or {}), **(result.soft or {})}
    return SimpleNamespace(
        score=score,
        passed=orch._verifier_accepts(score),
        threshold=threshold,
        metrics=metrics,
        privileged_agents=verify_info.get("privileged_agents"),
    )


def _cache_sparsedrivev2_selected_plan(
    agent, selected_plan, info, selected_score_raw, selected_source
) -> None:
    """Stash a fresh plan so the next replan_every_n-1 control ticks can reuse it."""
    if int(getattr(agent, "_sparsedrivev2_replan_every_n", 1)) <= 1:
        return
    agent._sparsedrivev2_cached_selected_plan = np.asarray(
        selected_plan, dtype=np.float32
    ).copy()
    agent._sparsedrivev2_cached_selected_pose = agent._info_to_pose(info)
    agent._sparsedrivev2_cached_selected_score = selected_score_raw
    agent._sparsedrivev2_cached_selected_source = str(selected_source)
    # This fresh plan produced exactly one control this tick.
    agent._sparsedrivev2_controls_since_replan = 1
