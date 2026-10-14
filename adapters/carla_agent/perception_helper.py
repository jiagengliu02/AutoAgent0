"""BridgeDrive detection_ft observations for PDMS and recovery."""
from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np


LOG = logging.getLogger(__name__)
SOURCES = {"bridgedrivev2"}
_SOURCE_ALIASES = {
    # Public YAML name for the strict BridgeDrive aux-head perception path.
    # Keep the existing internal name so older worker/protocol code remains
    # backward compatible; ``configured_source`` below retains the strictness
    # distinction from the legacy, optionally-privileged ``bridgedrive`` mode.
    "bridgedrivev2": "bridgedrive",
}
_STRICT_NONPRIVILEGED_SOURCE = "bridgedrivev2"
_OBJECT_GT_CONFIG_KEYS = (
    "record_ground_truth",
    "fuse_privileged_static_props",
    "replace_sensed_static_props_with_privileged",
    "fuse_privileged_environment_objects",
    "include_environment_objects",
)
# Sources whose detections come from sensors (vs the simulator): they share
# the stale-cache/advance/fusion/shadow machinery below.
_SENSOR_SOURCES = {"bridgedrive"}
_SENSOR_SOURCE_LABELS = {"bridgedrive": "BridgeDrive"}
_STATIONARY_MEMORY_SPEED_MPS = 0.3
_STATIONARY_MEMORY_DEFAULT_MIN_FRAMES = 3
_STATIONARY_MEMORY_DEFAULT_MATCH_DISTANCE_M = 2.0
_STATIONARY_MEMORY_DEFAULT_MAX_AGE_SEC = 2.0
_STATIONARY_MEMORY_MIN_PHYSICAL_EXTENT_M = 0.5
_STATIONARY_MEMORY_MAX_GEOMETRY_WEIGHT = 9
_RADAR_CONFIRMATION_DEFAULT_MAX_AGE_SEC = 0.8
_BEVFORMER_DEFAULT_MAX_RANGE_M = 50.0


def _json_default(value: Any) -> Any:
    """Serialize NumPy sensor values without losing scalar precision."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return float(value)


def _default_max_range_m(source: str) -> float:
    if source == "bridgedrive":
        from agenticdriving.planners.bridgedrive.judge_adapter import (
            BRIDGEDRIVE_FULL_BEV_RANGE_M,
        )

        return float(BRIDGEDRIVE_FULL_BEV_RANGE_M)
    return _BEVFORMER_DEFAULT_MAX_RANGE_M


class PerceptionUnavailable(RuntimeError):
    """Raised when strict sensor perception cannot provide a usable frame."""


# The one judge row that needs more than a single key: the full sensors-only
# bundle. Every other row is already one word (perception.source).
_JUDGE_MODE_EXPANSIONS = {
    "bridgedrive_nonprivileged": {
        "perception": {
            # HD-map exemption: drivable-area and
            # lane-keeping context always comes from the ground-truth HD map
            # (the ROAD-semantic DAC experiment was deleted outright), so
            # the sensed part of the row is OBJECT perception only. The
            # semantic-head statics feed (occupancy_static_agents) is a free
            # per-config choice, default OFF. The preset pins the strict
            # no-object-GT bundle while leaving that sensor-only choice open.
            "source": "bridgedrivev2",
            "record_ground_truth": False,
            "fuse_privileged_static_props": False,
            "replace_sensed_static_props_with_privileged": False,
            "fuse_privileged_environment_objects": False,
            "include_environment_objects": False,
            "semantic_pedestrian_agents": True,
        },
        # No verifier overrides: with the HD map allowed for DAC/LK, scoring
        # keeps the full 12-weight form and the standard threshold.
    },
}


def expand_judge_mode(cfg: Dict[str, Any]) -> None:
    """Expand the optional top-level ``judge_mode`` preset in place.

    A row is either a named mode or a hand-built flag set, never a blend: an
    explicit key that CONTRADICTS the preset is an error (equal restatement
    is fine). Absent/none = no-op, so all existing configs are untouched."""
    mode = cfg.get("judge_mode")
    if mode in (None, "", "none"):
        return
    mode = str(mode).strip().lower()
    expansion = _JUDGE_MODE_EXPANSIONS.get(mode)
    if expansion is None:
        raise ValueError(
            f"judge_mode={mode!r}; expected one of "
            f"{sorted(_JUDGE_MODE_EXPANSIONS)} (or omit the key)"
        )
    applied = []
    for block, keys in expansion.items():
        section = cfg.setdefault(block, {})
        for key, value in keys.items():
            if key in section and section[key] != value:
                raise ValueError(
                    f"judge_mode={mode} sets {block}.{key}={value!r} but the "
                    f"config explicitly sets {section[key]!r}; remove one"
                )
            section[key] = value
            applied.append(f"{block}.{key}={value}")
    LOG.info("judge_mode=%s expanded: %s", mode, " ".join(applied))




def setup_perception(agent) -> None:
    cfg = dict((getattr(agent, "_cfg", None) or {}).get("perception") or {})
    configured_source = str(cfg.get("source", "ground_truth")).strip().lower()
    if configured_source not in SOURCES:
        raise ValueError(
            f"Release requires perception.source=bridgedrivev2, got {configured_source!r}"
        )
    strict_no_object_ground_truth = (
        configured_source == _STRICT_NONPRIVILEGED_SOURCE
    )
    if strict_no_object_ground_truth:
        forbidden = [
            key for key in _OBJECT_GT_CONFIG_KEYS if bool(cfg.get(key, False))
        ]
        if forbidden:
            raise ValueError(
                "perception.source='bridgedrivev2' forbids simulator object "
                "ground truth; disable "
                + ", ".join(f"perception.{key}" for key in forbidden)
            )
    source = _SOURCE_ALIASES.get(configured_source, configured_source)
    vru_review_enabled = bool(cfg.get("stationary_vru_review", False))

    bridgedrive_mode = "service"
    bd_cfg = dict(cfg["bridgedrive"])
    bd_cfg.pop("mode", None)
    for key in ("checkpoint_dir", "python_bin"):
        if not bd_cfg.get(key):
            raise ValueError(f"perception.bridgedrive.{key} is required")
    if not bd_cfg.get("full_camera_rig"):
        raise ValueError("detection_ft requires the complete BridgeDrive camera rig")
    bd_cfg.setdefault("worker_log_dir", str(Path(agent._output_dir) / "logs"))
    from services import BridgeDrivePerceptionCarlaService
    service = BridgeDrivePerceptionCarlaService(agent, cfg=bd_cfg)
    from agenticdriving.planners.bridgedrive.carla_inputs import BRIDGEDRIVE_SENSOR_SPECS
    extras = list(getattr(agent, "_extra_sensor_specs", []) or [])
    existing_ids = {spec.get("id") for spec in extras}
    extras.extend(dict(spec) for spec in BRIDGEDRIVE_SENSOR_SPECS
                  if spec["type"] != "sensor.speedometer" and spec["id"] not in existing_ids)
    agent._extra_sensor_specs = extras

    agent._perception = SimpleNamespace(
        source=source,
        configured_source=configured_source,
        strict_no_object_ground_truth=strict_no_object_ground_truth,
        hd_map_lane_annotations=(
            bool(cfg.get("hd_map_lane_annotations", True))
            and source in _SENSOR_SOURCES
        ),
        service=service,
        score_threshold=float(cfg.get("score_threshold", 0.35)),
        max_range_m=float(cfg.get("max_range_m", _default_max_range_m(source))),
        inference_stride=max(1, int(cfg.get("inference_stride", 5))),
        max_staleness_sec=float(cfg.get("max_staleness_sec", 1.0)),
        jpeg_quality=int(cfg.get("jpeg_quality", 20)),
        scene_token=str(getattr(agent, "_route_name", None) or agent._output_dir.name),
        last_detection_ts=None,
        last_agents=None,
        cache_frame_index=None,
        cache_agents=None,
        bridgedrive_mode=bridgedrive_mode,
        # Erosion default 0: cell-level erosion deletes isolated objects up
        # to 2 cells (0.5 m) wide — gnome-sized. Speckle rejection happens at
        # COMPONENT level instead (occupancy_min_component_cells).
        occupancy_erosion_cells=int(cfg.get("occupancy_erosion_cells", 0)),
        occupancy_min_component_cells=int(
            cfg.get("occupancy_min_component_cells", 2)
        ),
        occupancy_static_agents=bool(cfg.get("occupancy_static_agents", False)),
        semantic_pedestrian_agents=(
            bool(cfg.get("semantic_pedestrian_agents", False))
            and source == "bridgedrive"
        ),
        semantic_pedestrian_min_component_cells=max(
            1, int(cfg.get("semantic_pedestrian_min_component_cells", 2))
        ),
        semantic_pedestrian_duplicate_distance_m=max(
            0.0, float(cfg.get("semantic_pedestrian_duplicate_distance_m", 2.0))
        ),
        last_semantic_pedestrian_agents=None,
        last_mask_static_agents=None,
        last_occupancy=None,
        last_box_class_stats=None,
        last_bev_class_counts=None,
        # Optional sensor-only double check for directly-forward BridgeDrive
        # vehicle/static boxes.  CARLA radar confirms the visible near face
        # and lateral span; it never queries actor or environment-object GT.
        radar_bbox_enabled=(
            bool(cfg.get("radar_bbox_enabled", False))
            and source == "bridgedrive"
        ),
        radar_bbox_max_distance_m=max(
            0.0, float(cfg.get("radar_bbox_max_distance_m", 35.0))
        ),
        radar_bbox_lateral_limit_m=max(
            0.0, float(cfg.get("radar_bbox_lateral_limit_m", 8.0))
        ),
        radar_bbox_max_heading_error_deg=max(
            0.0, float(cfg.get("radar_bbox_max_heading_error_deg", 45.0))
        ),
        radar_bbox_association_margin_m=max(
            0.0, float(cfg.get("radar_bbox_association_margin_m", 1.5))
        ),
        radar_bbox_min_points=max(1, int(cfg.get("radar_bbox_min_points", 5))),
        radar_bbox_min_unique_sensors=max(
            1, int(cfg.get("radar_bbox_min_unique_sensors", 1))
        ),
        radar_bbox_quantile=min(
            0.45, max(0.0, float(cfg.get("radar_bbox_quantile", 0.1)))
        ),
        radar_bbox_longitudinal_padding_m=max(
            0.0, float(cfg.get("radar_bbox_longitudinal_padding_m", 0.15))
        ),
        radar_bbox_lateral_padding_m=max(
            0.0, float(cfg.get("radar_bbox_lateral_padding_m", 0.25))
        ),
        radar_bbox_blend=min(
            1.0, max(0.0, float(cfg.get("radar_bbox_blend", 0.6)))
        ),
        radar_bbox_min_extent_scale=max(
            0.1, float(cfg.get("radar_bbox_min_extent_scale", 0.7))
        ),
        radar_bbox_max_extent_scale=max(
            1.0, float(cfg.get("radar_bbox_max_extent_scale", 1.3))
        ),
        radar_bbox_max_center_shift_m=max(
            0.0, float(cfg.get("radar_bbox_max_center_shift_m", 0.75))
        ),
        radar_bbox_gate_stationary_memory=(
            bool(cfg.get("radar_bbox_gate_stationary_memory", True))
            and bool(cfg.get("radar_bbox_enabled", False))
            and source == "bridgedrive"
        ),
        radar_bbox_gate_vehicle_stop=(
            bool(cfg.get("radar_bbox_gate_vehicle_stop", True))
            and bool(cfg.get("radar_bbox_enabled", False))
            and source == "bridgedrive"
        ),
        radar_bbox_confirmation_max_age_sec=max(
            0.0,
            float(
                cfg.get(
                    "radar_bbox_confirmation_max_age_sec",
                    _RADAR_CONFIRMATION_DEFAULT_MAX_AGE_SEC,
                )
            ),
        ),
        last_radar_bbox_stats=None,
        last_radar_bbox_diagnostics=None,
        # The LiDARs already run every simulator tick.  Within twenty metres,
        # use their dense non-ground returns as the final bounded geometry
        # refinement after radar; vision remains the semantic detector.
        lidar_bbox_enabled=(
            bool(cfg.get("lidar_bbox_enabled", False))
            and source == "bridgedrive"
        ),
        lidar_bbox_max_distance_m=max(
            0.0, float(cfg.get("lidar_bbox_max_distance_m", 20.0))
        ),
        lidar_bbox_lateral_limit_m=max(
            0.0, float(cfg.get("lidar_bbox_lateral_limit_m", 8.0))
        ),
        lidar_bbox_longitudinal_association_margin_m=max(
            0.0,
            float(
                cfg.get("lidar_bbox_longitudinal_association_margin_m", 1.0)
            ),
        ),
        lidar_bbox_lateral_association_margin_m=max(
            0.0,
            float(cfg.get("lidar_bbox_lateral_association_margin_m", 0.75)),
        ),
        lidar_bbox_min_points=max(1, int(cfg.get("lidar_bbox_min_points", 12))),
        lidar_bbox_static_min_points=cfg.get("lidar_bbox_static_min_points"),
        lidar_bbox_static_min_vertical_span_m=cfg.get("lidar_bbox_static_min_vertical_span_m"),
        lidar_bbox_static_ground_clearance_m=cfg.get("lidar_bbox_static_ground_clearance_m"),
        lidar_bbox_min_vertical_span_m=max(
            0.0, float(cfg.get("lidar_bbox_min_vertical_span_m", 0.25))
        ),
        lidar_bbox_ground_clearance_m=max(
            0.0, float(cfg.get("lidar_bbox_ground_clearance_m", 0.18))
        ),
        lidar_bbox_max_object_height_m=max(
            0.1, float(cfg.get("lidar_bbox_max_object_height_m", 4.0))
        ),
        lidar_bbox_quantile=min(
            0.35, max(0.0, float(cfg.get("lidar_bbox_quantile", 0.05)))
        ),
        lidar_bbox_longitudinal_padding_m=max(
            0.0, float(cfg.get("lidar_bbox_longitudinal_padding_m", 0.10))
        ),
        lidar_bbox_lateral_padding_m=max(
            0.0, float(cfg.get("lidar_bbox_lateral_padding_m", 0.20))
        ),
        lidar_bbox_blend=min(
            1.0, max(0.0, float(cfg.get("lidar_bbox_blend", 0.85)))
        ),
        lidar_bbox_min_extent_scale=max(
            0.1, float(cfg.get("lidar_bbox_min_extent_scale", 0.4))
        ),
        lidar_bbox_max_extent_scale=max(
            1.0, float(cfg.get("lidar_bbox_max_extent_scale", 1.25))
        ),
        lidar_bbox_max_center_shift_m=max(
            0.0, float(cfg.get("lidar_bbox_max_center_shift_m", 1.5))
        ),
        lidar_bbox_max_near_edge_shift_m=max(
            0.0, float(cfg.get("lidar_bbox_max_near_edge_shift_m", 3.0))
        ),
        lidar_bbox_rebuild_static_longitudinal_bounds=bool(
            cfg.get("lidar_bbox_rebuild_static_longitudinal_bounds", True)
        ),
        # BridgeDrive's class-5 head can emit repeated decoder-floor boxes
        # (0.34375 m square in 1020). Inside the LiDAR validation range, do
        # not expose such fragments unless a physical sensor confirms them.
        lidar_bbox_gate_degenerate_static=(
            bool(cfg.get("lidar_bbox_gate_degenerate_static", True))
            and bool(cfg.get("lidar_bbox_enabled", False))
            and source == "bridgedrive"
        ),
        lidar_bbox_gate_unconfirmed_static=(
            bool(cfg.get("lidar_bbox_gate_unconfirmed_static", False))
            and bool(cfg.get("lidar_bbox_enabled", False))
            and source == "bridgedrive"
        ),
        lidar_bbox_gate_static_memory=(
            bool(cfg.get("lidar_bbox_gate_static_memory", False))
            and bool(cfg.get("lidar_bbox_enabled", False))
            and source == "bridgedrive"
        ),
        lidar_bbox_min_static_extent_m=max(
            0.0, float(cfg.get("lidar_bbox_min_static_extent_m", 0.5))
        ),
        lidar_front_static_agents=(
            bool(cfg.get("lidar_front_static_agents", False))
            and bool(cfg.get("lidar_bbox_enabled", False))
            and source == "bridgedrive"
        ),
        last_lidar_bbox_stats=None,
        last_lidar_bbox_diagnostics=None,
        last_filtered_static_sensor_gate_agents=None,
        last_raw_boxes=None,
        last_radar_boxes=None,
        last_lidar_boxes=None,
        # BridgeDrive detections are world-frame snapshots.  Keep a separate
        # static-object memory so a confirmed stationary obstacle survives
        # later perception dropouts instead of disappearing from NC/TTC.
        stationary_memory_enabled=(
            bool(cfg.get("stationary_memory_enabled", True))
            and source == "bridgedrive"
        ),
        stationary_memory_min_frames=max(
            1,
            int(
                cfg.get(
                    "stationary_memory_min_frames",
                    _STATIONARY_MEMORY_DEFAULT_MIN_FRAMES,
                )
            ),
        ),
        stationary_memory_match_distance_m=max(
            0.1,
            float(
                cfg.get(
                    "stationary_memory_match_distance_m",
                    _STATIONARY_MEMORY_DEFAULT_MATCH_DISTANCE_M,
                )
            ),
        ),
        stationary_memory_max_age_sec=max(
            0.0,
            float(
                cfg.get(
                    "stationary_memory_max_age_sec",
                    _STATIONARY_MEMORY_DEFAULT_MAX_AGE_SEC,
                )
            ),
        ),
        stationary_memory=[],
        stationary_memory_next_id=0,
        telemetry_path=(
            str(Path(agent._output_dir) / "perception_telemetry.jsonl")
            if source == "bridgedrive" else None
        ),
        telemetry_fh=None,
    )
    if vru_review_enabled:

        checker = getattr(agent, "_recovery_explainer", None)
        if checker is None:
            # The primitives planner does not normally initialize the shared
            # recovery backend. This also supports perception-only VLM use.
            from agenticdriving_carla_helper import setup_recovery_explainer
            setup_recovery_explainer(agent)
            checker = agent._recovery_explainer
        if vru_review_enabled:
            from agenticdriving.perception.stationary_vru import StationaryVruReview
            agent._perception.stationary_vru_review = StationaryVruReview(
                checker=checker, output_dir=agent._output_dir)
    LOG.info("BridgeDrive detection_ft: score>=%.2f range<=%.1fm, stationary_vru_review=%s",
             agent._perception.score_threshold, agent._perception.max_range_m, vru_review_enabled)


def object_ground_truth_allowed(agent) -> bool:
    return False


def _query_hd_map_lane(agent_pos_world) -> tuple[Optional[bool], Optional[list]]:
    """Exact Driving-lane membership for a detected world position.

    This consults only the permitted HD map.  It never searches CARLA actors
    and never associates a detection with a simulator object identity.
    """
    try:
        import carla
        from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
        from agenticdriving.planners.sparsedrivev2.carla_inputs import (
            carla_waypoint_lane_key,
        )

        carla_map = CarlaDataProvider.get_map()
        if carla_map is None:
            return None, None
        pos = list(agent_pos_world or (0.0, 0.0, 0.0))
        while len(pos) < 3:
            pos.append(0.0)
        waypoint = carla_map.get_waypoint(
            carla.Location(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])),
            project_to_road=False,
            lane_type=carla.LaneType.Driving,
        )
        lane_key = carla_waypoint_lane_key(waypoint)
        return lane_key is not None, (
            list(lane_key) if lane_key is not None else None
        )
    except (ImportError, RuntimeError, TypeError, ValueError):
        LOG.debug("HD-map lane lookup failed for sensed object", exc_info=True)
        return None, None


def _annotate_agents_from_hd_map(
    agents: List[Dict[str, Any]],
    *,
    lane_query=None,
) -> List[Dict[str, Any]]:
    """Attach map-only lane metadata to sensed vehicles and pedestrians."""
    lane_query = lane_query or _query_hd_map_lane
    annotated: List[Dict[str, Any]] = []
    for original in agents:
        item = dict(original)
        if item.get("object_type") in {"vehicle", "pedestrian"}:
            on_driving_lane, lane_key = lane_query(item.get("agent_pos_world"))
            if on_driving_lane is not None:
                item["agent_on_driving_lane"] = bool(on_driving_lane)
                if lane_key is not None:
                    item["agent_lane_key"] = list(lane_key)
                else:
                    item.pop("agent_lane_key", None)
        annotated.append(item)
    return annotated


def _advance_agents(
    agents: List[Dict[str, Any]], source_ts: float, target_ts: float
) -> List[Dict[str, Any]]:
    dt = max(0.0, float(target_ts) - float(source_ts))
    advanced: List[Dict[str, Any]] = []
    for original in agents:
        agent = dict(original)
        pos = list(agent.get("agent_pos_world") or (0.0, 0.0, 0.0))
        vel = list(agent.get("agent_vel_vec") or (0.0, 0.0))
        while len(pos) < 3:
            pos.append(0.0)
        while len(vel) < 2:
            vel.append(0.0)
        pos[0] = float(pos[0]) + float(vel[0]) * dt
        pos[1] = float(pos[1]) + float(vel[1]) * dt
        agent["agent_pos_world"] = [float(pos[0]), float(pos[1]), float(pos[2])]
        agent["timestamp"] = float(target_ts)
        # Cached memory already has an observation age; extrapolation is not
        # a new detection and must not make a stale obstacle fresh again.
        agent["perception_age_sec"] = float(original.get("perception_age_sec", 0.0) or 0.0) + dt
        advanced.append(agent)
    return advanced


def _agent_xy(agent: Dict[str, Any]) -> tuple[float, float]:
    pos = list(agent.get("agent_pos_world") or (0.0, 0.0))
    while len(pos) < 2:
        pos.append(0.0)
    return float(pos[0]), float(pos[1])


def _agent_match_radius(agent: Dict[str, Any]) -> float:
    extent = list(agent.get("agent_extent") or (0.0, 0.0))
    while len(extent) < 2:
        extent.append(0.0)
    # ``agent_extent`` stores full length/width.  Half of the box diagonal is a
    # conservative circle around it and works across BEVFormer/GT yaw noise.
    return max(0.5, 0.5 * math.hypot(float(extent[0]), float(extent[1])))


def _is_stationary_memory_candidate(agent: Dict[str, Any]) -> bool:
    """Whether an agent is eligible for persistent static-object memory."""
    # A current LiDAR surface is collision evidence, not a tracked object.
    # Averaging unrelated slices freezes car rears and road-edge returns.
    if agent.get("perception_source") == "bridgedrive_lidar_corridor":
        return False
    object_type = str(agent.get("object_type", ""))
    if object_type not in {"static", "vehicle"}:
        return False
    extent = list(agent.get("agent_extent") or (0.0, 0.0))
    while len(extent) < 2:
        extent.append(0.0)
    # Decoder-floor vehicle and class-5 boxes have no meaningful physical
    # footprint. In 1020 the 0.34375 m class-5 fragments persisted for dozens
    # of harvests and stretched lane-bypass clearance despite zero LiDAR/radar
    # support. A sensor-confirmed small prop is physical, however: retain it
    # through brief occlusion just like a larger static obstacle.
    if min(abs(float(extent[0])), abs(float(extent[1]))) < (
        _STATIONARY_MEMORY_MIN_PHYSICAL_EXTENT_M
    ):
        return bool(object_type == "static" and (
            agent.get("lidar_bbox_confirmed") or agent.get("radar_bbox_confirmed")
        ))
    if object_type == "static":
        return True
    speed = abs(float(agent.get("agent_velo", 0.0) or 0.0))
    if speed <= _STATIONARY_MEMORY_SPEED_MPS:
        return True
    vel = agent.get("agent_vel_vec") or ()
    try:
        speed = math.hypot(float(vel[0]), float(vel[1]))
    except (IndexError, TypeError, ValueError):
        pass
    return speed <= _STATIONARY_MEMORY_SPEED_MPS


def _filter_unconfirmed_static_agents(
    perception, agents: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Drop close class-5 boxes that require but lack sensor confirmation.

    The broad gate is intended for custom-obstacle suites where class-5 model
    boxes are proposals rather than trustworthy geometry.  The narrower
    decoder-floor gate remains available independently for other configs.
    """
    perception.last_filtered_static_sensor_gate_agents = []
    gate_all = bool(
        getattr(perception, "lidar_bbox_gate_unconfirmed_static", False)
    )
    gate_degenerate = bool(
        getattr(perception, "lidar_bbox_gate_degenerate_static", False)
    )
    if not gate_all and not gate_degenerate:
        return agents
    min_extent = float(
        getattr(perception, "lidar_bbox_min_static_extent_m", 0.5)
    )
    kept = []
    for agent in agents:
        extent = list(agent.get("agent_extent") or (0.0, 0.0))
        while len(extent) < 2:
            extent.append(0.0)
        is_degenerate = min(abs(float(extent[0])), abs(float(extent[1]))) < (
            min_extent
        )
        sensor_confirmed = bool(
            agent.get("lidar_bbox_confirmed") or agent.get("radar_bbox_confirmed")
        )
        should_gate = bool(gate_all or (gate_degenerate and is_degenerate))
        if (
            agent.get("object_type") == "static"
            and agent.get("lidar_bbox_eligible")
            and should_gate
            and not sensor_confirmed
        ):
            perception.last_filtered_static_sensor_gate_agents.append(
                {
                    "agent_id": agent.get("agent_id"),
                    "detection_class": agent.get("detection_class"),
                    "detection_score": agent.get("detection_score"),
                    "agent_pos_world": agent.get("agent_pos_world"),
                    "agent_extent": agent.get("agent_extent"),
                    "lidar_bbox_confirmed": agent.get("lidar_bbox_confirmed"),
                    "radar_bbox_confirmed": agent.get("radar_bbox_confirmed"),
                }
            )
            continue
        kept.append(agent)
    return kept


def _box_corners_world(agent: Dict[str, Any]) -> np.ndarray:
    """Return the four xy corners of an agent's full-size oriented bbox."""
    pos = list(agent.get("agent_pos_world") or (0.0, 0.0))
    while len(pos) < 2:
        pos.append(0.0)
    extent = list(agent.get("agent_extent") or (1.0, 1.0))
    while len(extent) < 2:
        extent.append(1.0)
    half_len = max(0.025, float(extent[0]) / 2.0)
    half_width = max(0.025, float(extent[1]) / 2.0)
    heading = float(agent.get("agent_heading", 0.0) or 0.0)
    c, s = math.cos(heading), math.sin(heading)
    local = np.asarray(
        [
            [-half_len, -half_width],
            [-half_len, half_width],
            [half_len, half_width],
            [half_len, -half_width],
        ],
        dtype=np.float64,
    )
    rotation = np.asarray([[c, -s], [s, c]], dtype=np.float64)
    return local @ rotation.T + np.asarray(pos[:2], dtype=np.float64)


def _points_to_anchor(
    points: np.ndarray, anchor_center: np.ndarray, anchor_heading: float
) -> np.ndarray:
    c, s = math.cos(anchor_heading), math.sin(anchor_heading)
    delta = np.asarray(points, dtype=np.float64) - anchor_center.reshape(1, 2)
    return delta @ np.asarray([[c, -s], [s, c]], dtype=np.float64)


def _points_from_anchor(
    points: np.ndarray, anchor_center: np.ndarray, anchor_heading: float
) -> np.ndarray:
    c, s = math.cos(anchor_heading), math.sin(anchor_heading)
    return np.asarray(points, dtype=np.float64) @ np.asarray(
        [[c, s], [-s, c]], dtype=np.float64
    ) + anchor_center.reshape(1, 2)


def _stationary_memory_agent(
    track: Dict[str, Any], now: float, *, current: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Materialize a track's detection-averaged bbox as a normal judge agent."""
    base = dict(current or track["agent"])
    if base.get("occupancy_parts") is not None:
        # A region is a set of world-space occupied cells, not a rigid box.
        # Keep the latest shape (also on missed frames) without re-boxing it.
        base.update(agent_id=track["track_id"], timestamp=float(now),
                    perception_age_sec=max(0., float(now)-float(track["last_seen_ts"])),
                    stationary_memory_confirmed=bool(track["confirmed"]),
                    stationary_memory_hits=int(track["hits"]),
                    stationary_memory_last_seen_ts=float(track["last_seen_ts"]),
                    stationary_memory_source="bridgedrive_stationary_memory")
        return base
    anchor_center = np.asarray(track["anchor_center"], dtype=np.float64)
    anchor_heading = float(track["anchor_heading"])
    min_local = np.asarray(track["min_local"], dtype=np.float64)
    max_local = np.asarray(track["max_local"], dtype=np.float64)
    center_local = (min_local + max_local) / 2.0
    center_world = _points_from_anchor(
        center_local.reshape(1, 2), anchor_center, anchor_heading
    )[0]
    base_pos = list(base.get("agent_pos_world") or (0.0, 0.0, 0.0))
    while len(base_pos) < 3:
        base_pos.append(0.0)
    base["agent_id"] = track["track_id"]
    base["agent_pos_world"] = [
        float(center_world[0]), float(center_world[1]), float(base_pos[2])
    ]
    base["agent_heading"] = anchor_heading
    base["agent_extent"] = [
        max(0.05, float(max_local[0] - min_local[0])),
        max(0.05, float(max_local[1] - min_local[1])),
    ]
    base["agent_velo"] = 0.0
    base["agent_vel_vec"] = [0.0, 0.0]
    base["timestamp"] = float(now)
    base["perception_age_sec"] = max(0.0, float(now) - float(track["last_seen_ts"]))
    base["stationary_memory_confirmed"] = bool(track["confirmed"])
    base["stationary_memory_hits"] = int(track["hits"])
    base["stationary_memory_last_seen_ts"] = float(track["last_seen_ts"])
    base["stationary_memory_source"] = "bridgedrive_stationary_memory"
    return base


def _stationary_memory_track_is_fresh(
    perception, track: Dict[str, Any], now: float
) -> bool:
    max_age = float(
        getattr(perception, "stationary_memory_max_age_sec", float("inf"))
    )
    return float(now) - float(track["last_seen_ts"]) <= max_age


def _update_stationary_memory(
    perception, agents: List[Dict[str, Any]], now: float
) -> List[Dict[str, Any]]:
    """Update static tracks and append confirmed tracks missing this frame.

    Track geometry is accumulated in the first detection's local frame, so
    ego motion and small heading jitter do not make the averaged bbox drift.
    Only actual matched detections contribute to the running boundary average;
    frames where the object is not detected leave both the hit count and bbox
    unchanged. Matching is one-to-one per harvest; detections are never merged
    merely because they occur in the same frame.
    """
    if not getattr(perception, "stationary_memory_enabled", False):
        return agents
    memory = [
        track
        for track in (getattr(perception, "stationary_memory", []) or [])
        if _stationary_memory_track_is_fresh(perception, track, now)
    ]
    match_distance = float(perception.stationary_memory_match_distance_m)
    used_tracks = set()
    replacements: Dict[int, Dict[str, Any]] = {}

    def nearest_track(agent: Dict[str, Any], *, allow_used: bool = False):
        ax, ay = _agent_xy(agent)
        best = None
        for index, track in enumerate(memory):
            if (agent.get("occupancy_parts") is not None) != (
                track["agent"].get("occupancy_parts") is not None
            ):
                continue
            if not allow_used and index in used_tracks:
                continue
            tx, ty = _agent_xy(track["agent"])
            distance = math.hypot(ax - tx, ay - ty)
            limit = max(
                match_distance,
                0.5 * _agent_match_radius(agent)
                + 0.5 * _agent_match_radius(track["agent"]),
            )
            if distance <= limit and (best is None or distance < best[0]):
                best = (distance, index)
        return best

    # A previously stationary object that is now clearly moving should not be
    # kept forever as a phantom obstacle.
    invalidated = set()
    for agent in agents:
        if (
            str(agent.get("object_type")) != "vehicle"
            or _is_stationary_memory_candidate(agent)
        ):
            continue
        match = nearest_track(agent, allow_used=True)
        if match is not None:
            invalidated.add(match[1])
    if invalidated:
        memory = [track for index, track in enumerate(memory) if index not in invalidated]
        used_tracks.clear()

    for agent_index, agent in enumerate(agents):
        if not _is_stationary_memory_candidate(agent):
            continue
        match = nearest_track(agent)
        sensor_required = bool(
            agent.get("stationary_memory_radar_required")
            or agent.get("stationary_memory_lidar_required")
        )
        sensor_confirmed = bool(
            agent.get("radar_bbox_confirmed")
            or agent.get("lidar_bbox_confirmed")
        )
        if sensor_required and not sensor_confirmed:
            # An unconfirmed forward box remains available as a fresh visual
            # detection, but it may neither create nor refresh persistent
            # stationary memory. If it overlaps an existing confirmed track,
            # replace the raw duplicate with that track until its TTL expires.
            if match is not None:
                _distance, track_index = match
                used_tracks.add(track_index)
                track = memory[track_index]
                if track.get("confirmed"):
                    replacements[agent_index] = _stationary_memory_agent(
                        track, now
                    )
            continue
        if match is None:
            center = np.asarray(_agent_xy(agent), dtype=np.float64)
            heading = float(agent.get("agent_heading", 0.0) or 0.0)
            corners_local = _points_to_anchor(
                _box_corners_world(agent), center, heading
            )
            track = {
                "track_id": (
                    f"bd_stationary_{int(perception.stationary_memory_next_id)}"
                ),
                "hits": 1,
                "confirmed": False,
                "last_seen_ts": float(now),
                "anchor_center": center.tolist(),
                "anchor_heading": heading,
                "min_local": corners_local.min(axis=0).tolist(),
                "max_local": corners_local.max(axis=0).tolist(),
                "agent": dict(agent),
            }
            perception.stationary_memory_next_id += 1
            memory.append(track)
            track_index = len(memory) - 1
        else:
            _distance, track_index = match
            track = memory[track_index]
            anchor_center = np.asarray(track["anchor_center"], dtype=np.float64)
            anchor_heading = float(track["anchor_heading"])
            corners_local = _points_to_anchor(
                _box_corners_world(agent), anchor_center, anchor_heading
            )
            previous_hits = int(track["hits"])
            next_hits = previous_hits + 1
            if agent.get("occupancy_parts") is not None:
                # Association only; the latest exact parts live in track.agent.
                pass
            elif agent.get("lidar_bbox_full_extent_confirmed"):
                # A dense two-boundary LiDAR measurement is newer and more
                # geometric than the historical BridgeDrive average. Replace
                # the stored static bounds so an early delayed far edge cannot
                # keep lengthening lane-bypass clearance for dozens of frames.
                track["min_local"] = corners_local.min(axis=0).tolist()
                track["max_local"] = corners_local.max(axis=0).tolist()
            else:
                # Keep smoothing, but do not let hundreds of old detections
                # override fresh geometry after an occluding wall disappears.
                geometry_weight = min(previous_hits, _STATIONARY_MEMORY_MAX_GEOMETRY_WEIGHT)
                track["min_local"] = (
                    (
                        np.asarray(track["min_local"], dtype=np.float64)
                        * geometry_weight
                        + corners_local.min(axis=0)
                    )
                    / (geometry_weight + 1)
                ).tolist()
                track["max_local"] = (
                    (
                        np.asarray(track["max_local"], dtype=np.float64)
                        * geometry_weight
                        + corners_local.max(axis=0)
                    )
                    / (geometry_weight + 1)
                ).tolist()
            track["hits"] = next_hits
            track["last_seen_ts"] = float(now)
            track["agent"] = dict(agent)
        # Fresh dense near-face geometry must not be expanded back toward the
        # ego by the historical average. Clip only that longitudinal boundary;
        # leave the far side and lateral smoothing intact.
        face_sign = int(agent.get("lidar_near_face_sign", 0) or 0)
        heading_cos = math.cos(float(agent.get("agent_heading", 0.0)) - track["anchor_heading"])
        if face_sign and abs(heading_cos) > 0.99:
            lower = np.asarray(track["min_local"], dtype=np.float64)
            upper = np.asarray(track["max_local"], dtype=np.float64)
            if face_sign * heading_cos < 0:
                lower[0] = min(upper[0] - 0.05, max(lower[0], corners_local[:, 0].min()))
            else:
                upper[0] = max(lower[0] + 0.05, min(upper[0], corners_local[:, 0].max()))
            track["min_local"], track["max_local"] = lower.tolist(), upper.tolist()
        used_tracks.add(track_index)
        track = memory[track_index]
        track["confirmed"] = bool(
            str(agent.get("perception_source", ""))
            == "bridgedrive_bev_occupancy"
            or int(track["hits"]) >= int(perception.stationary_memory_min_frames)
        )
        if track["confirmed"]:
            replacements[agent_index] = _stationary_memory_agent(
                track, now, current=agent
            )

    output = [replacements.get(index, dict(agent)) for index, agent in enumerate(agents)]
    for index, track in enumerate(memory):
        if track.get("confirmed") and index not in used_tracks:
            output.append(_stationary_memory_agent(track, now))
    perception.stationary_memory = memory
    return output


def _confirmed_stationary_memory_agents(perception, now: float) -> List[Dict[str, Any]]:
    agents = [
        _stationary_memory_agent(track, now)
        for track in (getattr(perception, "stationary_memory", []) or [])
        if track.get("confirmed")
        and _stationary_memory_track_is_fresh(perception, track, now)
    ]
    return agents




def _ingest_bridgedrive_harvest(
    perception,
    harvest: Dict[str, Any],
    info: Dict[str, Any],
    capture: Optional[Dict[str, Any]] = None,
    radar_input: Optional[Dict[str, Any]] = None,
    lidar_input: Optional[Dict[str, Any]] = None,
    *,
    consumed_frame_index: Optional[int] = None,
    worker_round_trip_sec: Optional[float] = None,
) -> None:
    """Fresh worker harvest -> world-frame agents + capture-pose occupancy.

    ``capture`` pins the ego pose/clock the harvest's sensors were captured
    at. Service mode ingests on the same tick (capture == current info);
    piggyback consumes the planner's harvest on the NEXT tick's scoring pass,
    so the planner service stashes the capture pose alongside it — converting
    with the current pose would shift every box by one tick of ego motion."""
    from agenticdriving.planners.bridgedrive import judge_adapter

    capture = capture or info
    ego_pos = capture["ego_pos"]
    ego_rot = capture["ego_rot"]
    now = float(capture.get("timestamp", 0.0) or 0.0)
    raw_boxes_value = harvest.get("boxes_vehicle_system")
    raw_boxes = np.asarray(
        [] if raw_boxes_value is None else raw_boxes_value,
        dtype=np.float64,
    ).reshape(-1, 9)
    perception.last_raw_boxes = raw_boxes.tolist()
    boxes_for_judge = raw_boxes
    radar_diagnostics = {}
    radar_eligible_indices = set()
    perception.last_radar_bbox_stats = None
    perception.last_radar_bbox_diagnostics = None
    if getattr(perception, "radar_bbox_enabled", False):
        from agenticdriving.planners.bridgedrive.radar_bbox import (
            box_is_eligible_for_radar_validation,
            radar_points_to_ego,
            refine_forward_boxes_with_radar,
        )

        radar_points = radar_points_to_ego(radar_input)
        radar_eligible_indices = {
            index
            for index, box in enumerate(raw_boxes)
            if box_is_eligible_for_radar_validation(
                box,
                score_threshold=perception.score_threshold,
                max_distance_m=perception.radar_bbox_max_distance_m,
                lateral_limit_m=perception.radar_bbox_lateral_limit_m,
                max_heading_error_deg=(
                    perception.radar_bbox_max_heading_error_deg
                ),
            )
        }
        boxes_for_judge, radar_diagnostics, radar_stats = (
            refine_forward_boxes_with_radar(
                raw_boxes,
                radar_points,
                score_threshold=perception.score_threshold,
                max_distance_m=perception.radar_bbox_max_distance_m,
                lateral_limit_m=perception.radar_bbox_lateral_limit_m,
                max_heading_error_deg=(
                    perception.radar_bbox_max_heading_error_deg
                ),
                association_margin_m=(
                    perception.radar_bbox_association_margin_m
                ),
                min_points=perception.radar_bbox_min_points,
                min_unique_sensors=(
                    perception.radar_bbox_min_unique_sensors
                ),
                quantile=perception.radar_bbox_quantile,
                longitudinal_padding_m=(
                    perception.radar_bbox_longitudinal_padding_m
                ),
                lateral_padding_m=perception.radar_bbox_lateral_padding_m,
                blend=perception.radar_bbox_blend,
                min_extent_scale=perception.radar_bbox_min_extent_scale,
                max_extent_scale=perception.radar_bbox_max_extent_scale,
                max_center_shift_m=(
                    perception.radar_bbox_max_center_shift_m
                ),
            )
        )
        perception.last_radar_bbox_stats = radar_stats
        perception.last_radar_bbox_diagnostics = radar_diagnostics
    perception.last_radar_boxes = np.asarray(boxes_for_judge).tolist()

    lidar_diagnostics = {}
    lidar_eligible_indices = set()
    perception.last_lidar_bbox_stats = None
    perception.last_lidar_bbox_diagnostics = None
    if getattr(perception, "lidar_bbox_enabled", False):
        from agenticdriving.planners.bridgedrive.lidar_bbox import (
            box_is_eligible_for_lidar_validation,
            front_corridor_static_agent,
            lidar_points_to_ego,
            refine_close_boxes_with_lidar,
        )

        lidar_points = lidar_points_to_ego(lidar_input)
        lidar_eligible_indices = {
            index
            for index, box in enumerate(boxes_for_judge)
            if box_is_eligible_for_lidar_validation(
                box,
                score_threshold=perception.score_threshold,
                max_distance_m=perception.lidar_bbox_max_distance_m,
                lateral_limit_m=perception.lidar_bbox_lateral_limit_m,
            )
        }
        boxes_for_judge, lidar_diagnostics, lidar_stats = (
            refine_close_boxes_with_lidar(
                boxes_for_judge,
                lidar_points,
                score_threshold=perception.score_threshold,
                max_distance_m=perception.lidar_bbox_max_distance_m,
                lateral_limit_m=perception.lidar_bbox_lateral_limit_m,
                longitudinal_association_margin_m=(
                    perception.lidar_bbox_longitudinal_association_margin_m
                ),
                lateral_association_margin_m=(
                    perception.lidar_bbox_lateral_association_margin_m
                ),
                min_points=perception.lidar_bbox_min_points,
                min_vertical_span_m=perception.lidar_bbox_min_vertical_span_m,
                ground_clearance_m=perception.lidar_bbox_ground_clearance_m,
                static_min_points=getattr(perception, "lidar_bbox_static_min_points", None),
                static_min_vertical_span_m=getattr(perception, "lidar_bbox_static_min_vertical_span_m", None),
                static_ground_clearance_m=getattr(perception, "lidar_bbox_static_ground_clearance_m", None),
                max_object_height_m=perception.lidar_bbox_max_object_height_m,
                quantile=perception.lidar_bbox_quantile,
                longitudinal_padding_m=(
                    perception.lidar_bbox_longitudinal_padding_m
                ),
                lateral_padding_m=perception.lidar_bbox_lateral_padding_m,
                blend=perception.lidar_bbox_blend,
                min_extent_scale=perception.lidar_bbox_min_extent_scale,
                max_extent_scale=perception.lidar_bbox_max_extent_scale,
                max_center_shift_m=perception.lidar_bbox_max_center_shift_m,
                max_near_edge_shift_m=(
                    perception.lidar_bbox_max_near_edge_shift_m
                ),
                rebuild_static_longitudinal_bounds=(
                    getattr(
                        perception,
                        "lidar_bbox_rebuild_static_longitudinal_bounds",
                        True,
                    )
                ),
            )
        )
        perception.last_lidar_bbox_stats = lidar_stats
        perception.last_lidar_bbox_diagnostics = lidar_diagnostics
    perception.last_lidar_boxes = np.asarray(boxes_for_judge).tolist()

    perception.last_agents = judge_adapter.boxes_to_judge_agents(
        boxes_for_judge,
        ego_pos,
        ego_rot,
        now,
        score_threshold=perception.score_threshold,
        score_threshold_by_class=harvest.get("score_threshold_by_class"),
        max_range_m=perception.max_range_m,
    )
    if getattr(perception, "lidar_front_static_agents", False):
        front_static = front_corridor_static_agent(
            lidar_points,
            ego_pos,
            ego_rot,
            now,
            known_boxes=boxes_for_judge,
            score_threshold=perception.score_threshold,
            max_distance_m=min(20.0, perception.lidar_bbox_max_distance_m),
        )
        if front_static is not None:
            perception.last_agents.append(front_static)
    if radar_eligible_indices:
        for item in perception.last_agents:
            agent_id = str(item.get("agent_id", ""))
            if not agent_id.startswith("bd_"):
                continue
            try:
                box_index = int(agent_id.split("_", 1)[1])
            except (TypeError, ValueError):
                continue
            if box_index not in radar_eligible_indices:
                continue
            item["radar_bbox_eligible"] = True
            item["radar_bbox_confirmed"] = False
            item["stationary_memory_radar_required"] = bool(
                getattr(perception, "radar_bbox_gate_stationary_memory", False)
            )
            item["vehicle_bbox_stop_radar_required"] = bool(
                getattr(perception, "radar_bbox_gate_vehicle_stop", False)
            )
            item["radar_bbox_confirmation_max_age_sec"] = float(
                getattr(
                    perception,
                    "radar_bbox_confirmation_max_age_sec",
                    _RADAR_CONFIRMATION_DEFAULT_MAX_AGE_SEC,
                )
            )
            diagnostic = radar_diagnostics.get(box_index)
            if diagnostic is not None:
                item.update(diagnostic)
                item["bbox_validation_source"] = "carla_radar"
                item["radar_bbox_confirmation_ts"] = float(now)
    if lidar_eligible_indices:
        for item in perception.last_agents:
            agent_id = str(item.get("agent_id", ""))
            if not agent_id.startswith("bd_"):
                continue
            try:
                box_index = int(agent_id.split("_", 1)[1])
            except (TypeError, ValueError):
                continue
            if box_index not in lidar_eligible_indices:
                continue
            item["lidar_bbox_eligible"] = True
            item["lidar_bbox_confirmed"] = False
            diagnostic = lidar_diagnostics.get(box_index)
            if diagnostic is not None:
                item.update(diagnostic)
                previous_source = item.get("bbox_validation_source")
                item["bbox_validation_source"] = (
                    "carla_radar+lidar"
                    if previous_source == "carla_radar"
                    else "carla_lidar"
                )
                item["lidar_bbox_confirmation_ts"] = float(now)
    if getattr(perception, "lidar_bbox_gate_static_memory", False):
        for item in perception.last_agents:
            if item.get("object_type") == "static":
                item["stationary_memory_lidar_required"] = True
    perception.last_agents = _filter_unconfirmed_static_agents(
        perception, perception.last_agents
    )
    class_stats = {}
    for row in raw_boxes:
        clazz = str(int(row[7]))
        score = float(row[8])
        stats = class_stats.setdefault(
            clazz, {"count": 0, "accepted": 0, "max_score": 0.0}
        )
        stats["count"] += 1
        stats["accepted"] += int(score >= harvest.get("score_threshold_by_class", {}).get(int(clazz), perception.score_threshold))
        stats["max_score"] = max(stats["max_score"], score)
    perception.last_box_class_stats = class_stats
    perception.last_detection_ts = now
    bev_argmax = harvest.get("bev_argmax")
    if bev_argmax is not None:
        values, counts = np.unique(np.asarray(bev_argmax), return_counts=True)
        perception.last_bev_class_counts = {
            str(int(value)): int(count)
            for value, count in zip(values, counts)
        }
    else:
        perception.last_bev_class_counts = None
    perception.last_semantic_pedestrian_agents = (
        judge_adapter.bev_semantic_pedestrian_agents(
            bev_argmax,
            ego_pos,
            ego_rot,
            now,
            min_cells=perception.semantic_pedestrian_min_component_cells,
            max_range_m=perception.max_range_m,
        )
        if bev_argmax is not None and perception.semantic_pedestrian_agents
        else []
    )
    perception.last_agents = judge_adapter.merge_semantic_pedestrian_backstop(
        perception.last_agents,
        perception.last_semantic_pedestrian_agents,
        duplicate_center_m=perception.semantic_pedestrian_duplicate_distance_m,
    )
    perception.last_occupancy = (
        judge_adapter.pack_occupancy(
            bev_argmax,
            ego_pos,
            ego_rot,
            now,
            erosion_cells=perception.occupancy_erosion_cells,
        )
        if bev_argmax is not None
        else None
    )
    # Mask components are always computed (they appear in the shadow log for
    # validation against GT) but join the live agent list — and therefore
    # NC/TTC, recovery reasons, failure memory, identically to every other
    # static obstacle — only behind the flag.
    perception.last_mask_static_agents = (
        judge_adapter.occupancy_static_agents(
            perception.last_occupancy,
            min_cells=perception.occupancy_min_component_cells,
            preserve_shape=getattr(perception, "occupancy_preserve_shape", False),
        )
        if perception.last_occupancy is not None
        else []
    )
    if perception.occupancy_static_agents:
        perception.last_agents = (
            list(perception.last_agents) + perception.last_mask_static_agents
        )
    perception.last_agents = _update_stationary_memory(
        perception, list(perception.last_agents), now
    )
    perception.cache_frame_index = None  # invalidate the per-frame agent cache
    _record_sensor_telemetry(
        perception,
        capture,
        consumed_frame_index=consumed_frame_index,
        worker_round_trip_sec=worker_round_trip_sec,
    )


def _record_sensor_telemetry(
    perception,
    capture: Dict[str, Any],
    *,
    consumed_frame_index: Optional[int],
    worker_round_trip_sec: Optional[float],
) -> None:
    """Append sensor-only diagnostics; this file must never contain GT actors."""
    path = getattr(perception, "telemetry_path", None)
    if not path:
        return
    if perception.telemetry_fh is None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        perception.telemetry_fh = open(path, "a", buffering=1)
    detected_agents = list(getattr(perception, "last_agents", None) or [])
    pedestrians = [
        {
            "agent_id": item.get("agent_id"),
            "perception_source": item.get("perception_source"),
            "detection_class": item.get("detection_class"),
            "detection_score": item.get("detection_score"),
            "semantic_classes": item.get("semantic_classes"),
            "occupancy_cells": item.get("occupancy_cells"),
            "agent_pos_world": item.get("agent_pos_world"),
            "agent_velo": item.get("agent_velo"),
        }
        for item in detected_agents
        if item.get("object_type") == "pedestrian"
        and item.get("perception_source") in {
            "bridgedrive_centernet",
            "bridgedrive_bev_semantic_vru",
        }
    ]
    record = {
        "schema": "bridgedrive_sensor_only_v2",
        "wall_time_unix": time.time(),
        "capture_frame_index": capture.get("frame_index"),
        "consumed_frame_index": consumed_frame_index,
        "sim_timestamp": float(capture.get("timestamp", 0.0) or 0.0),
        "ego_pos": capture.get("ego_pos"),
        "ego_rot": capture.get("ego_rot"),
        "worker_round_trip_sec": (
            None if worker_round_trip_sec is None else float(worker_round_trip_sec)
        ),
        "box_class_stats": dict(perception.last_box_class_stats or {}),
        "accepted_agent_counts": {
            kind: sum(1 for item in detected_agents if item.get("object_type") == kind)
            for kind in ("vehicle", "pedestrian", "static")
        },
        "stationary_memory": {
            "enabled": bool(getattr(perception, "stationary_memory_enabled", False)),
            "min_frames": int(getattr(perception, "stationary_memory_min_frames", 0)),
            "confirmed_count": len(
                [
                    track
                    for track in (getattr(perception, "stationary_memory", []) or [])
                    if track.get("confirmed")
                    and _stationary_memory_track_is_fresh(
                        perception,
                        track,
                        float(capture.get("timestamp", 0.0) or 0.0),
                    )
                ]
            ),
        },
        "stationary_memory_agents": _confirmed_stationary_memory_agents(
            perception, float(capture.get("timestamp", 0.0) or 0.0)
        ),
        "pedestrian_detections": pedestrians,
        "pedestrian_detection_counts_by_source": {
            source: sum(
                1 for item in pedestrians if item.get("perception_source") == source
            )
            for source in (
                "bridgedrive_centernet",
                "bridgedrive_bev_semantic_vru",
            )
        },
        "bev_class_counts": dict(perception.last_bev_class_counts or {}),
        "radar_bbox_validation": dict(
            getattr(perception, "last_radar_bbox_stats", None) or {}
        ),
        "lidar_bbox_validation": dict(
            getattr(perception, "last_lidar_bbox_stats", None) or {}
        ),
        "filtered_static_sensor_gate_agents": list(
            getattr(perception, "last_filtered_static_sensor_gate_agents", None)
            or []
        ),
        "bbox_stages": {
            "raw": getattr(perception, "last_raw_boxes", None) or [],
            "radar": getattr(perception, "last_radar_boxes", None) or [],
            "lidar": getattr(perception, "last_lidar_boxes", None) or [],
        },
        "final_agents": [
            {
                key: item.get(key)
                for key in (
                    "agent_id",
                    "object_type",
                    "detection_class",
                    "detection_score",
                    "agent_pos_world",
                    "agent_heading",
                    "agent_extent",
                    "stationary_memory_confirmed",
                    "stationary_memory_hits",
                    "bbox_validation_source",
                    "radar_bbox_confirmed",
                    "lidar_bbox_confirmed",
                    "lidar_bbox_full_extent_confirmed",
                    "perception_source",
                    "perception_age_sec",
                    "agent_velo",
                    "agent_vel_vec",
                    "lidar_center_local",
                    "lidar_point_count",
                )
            }
            for item in detected_agents
        ],
        "radar_confirmed_agents": [
            {
                "agent_id": item.get("agent_id"),
                "object_type": item.get("object_type"),
                "agent_pos_world": item.get("agent_pos_world"),
                "agent_extent": item.get("agent_extent"),
                "radar_point_count": item.get("radar_point_count"),
                "radar_sensor_count": item.get("radar_sensor_count"),
                "radar_original_extent": item.get("radar_original_extent"),
                "radar_refined_extent": item.get("radar_refined_extent"),
                "radar_center_shift_m": item.get("radar_center_shift_m"),
                "radar_near_edge_error_m": item.get("radar_near_edge_error_m"),
            }
            for item in detected_agents
            if item.get("radar_bbox_confirmed")
        ],
        "lidar_confirmed_agents": [
            {
                "agent_id": item.get("agent_id"),
                "object_type": item.get("object_type"),
                "agent_pos_world": item.get("agent_pos_world"),
                "agent_extent": item.get("agent_extent"),
                "lidar_point_count": item.get("lidar_point_count"),
                "lidar_sensor_count": item.get("lidar_sensor_count"),
                "lidar_vertical_span_m": item.get("lidar_vertical_span_m"),
                "lidar_original_extent": item.get("lidar_original_extent"),
                "lidar_refined_extent": item.get("lidar_refined_extent"),
                "lidar_center_shift_m": item.get("lidar_center_shift_m"),
                "lidar_near_edge_error_m": item.get("lidar_near_edge_error_m"),
                "lidar_far_edge_error_m": item.get("lidar_far_edge_error_m"),
                "lidar_measured_longitudinal_bounds": item.get(
                    "lidar_measured_longitudinal_bounds"
                ),
                "lidar_measured_longitudinal_span_m": item.get(
                    "lidar_measured_longitudinal_span_m"
                ),
                "lidar_bbox_full_extent_confirmed": item.get(
                    "lidar_bbox_full_extent_confirmed"
                ),
            }
            for item in detected_agents
            if item.get("lidar_bbox_confirmed")
        ],
    }
    perception.telemetry_fh.write(json.dumps(record, default=_json_default) + "\n")


def pump_bridgedrive_perception(
    agent, obs: Dict[str, Any], info: Dict[str, Any]
) -> None:
    """Per-tick worker feed for Mode A (bridgedrive judge, other planner).

    MUST run every simulator tick: the hosted SensorAgent's tick() is
    stateful (Kalman ego filter, LiDAR sweep queue), so skipping frames
    silently corrupts the worker's preprocessing state. The FORWARD strides
    (inference_stride): detect() on stride frames (or until the first
    harvest lands), tick() in between — cached detections advance via the
    shared staleness machinery, like BEVFormer's stride. Piggyback mode is a
    no-op here (the planner's own cadence feeds it)."""
    perception = getattr(agent, "_perception", None)
    if (
        perception is None
        or perception.source != "bridgedrive"
        or perception.bridgedrive_mode != "service"
    ):
        return
    frame_index = int(getattr(agent, "_frame_index", 0))
    run_forward = (
        perception.last_agents is None
        or frame_index % perception.inference_stride == 0
    )
    try:
        if run_forward:
            harvest = perception.service.detect(obs, info)
            if harvest is not None:
                _ingest_bridgedrive_harvest(
                    perception,
                    harvest,
                    info,
                    radar_input=(obs.get("bd_input_data") or {}),
                    lidar_input=(obs.get("bd_input_data") or {}),
                    consumed_frame_index=frame_index,
                    worker_round_trip_sec=getattr(
                        perception.service, "last_round_trip_sec", None
                    ),
                )
        else:
            perception.service.tick(obs, info)
    except Exception:
        LOG.exception("BridgeDrive perception tick failed")


def resolve_step_agents(
    agent, obs: Dict[str, Any], info: Dict[str, Any]
) -> List[Dict[str, Any]]:
    perception = agent._perception

    label = _SENSOR_SOURCE_LABELS.get(perception.source, perception.source)
    frame_index = int(getattr(agent, "_frame_index", 0))
    if perception.cache_frame_index == frame_index:
        return list(perception.cache_agents or [])

    now = float(info.get("timestamp", 0.0) or 0.0)
    if perception.last_agents is None or perception.last_detection_ts is None:
        raise PerceptionUnavailable(f"{label} has not produced its first frame")
    age = now - float(perception.last_detection_ts)
    if age > perception.max_staleness_sec:
        persistent = _confirmed_stationary_memory_agents(perception, now)
        if not persistent:
            raise PerceptionUnavailable(
                f"{label} detections are stale ({age:.2f}s > "
                f"{perception.max_staleness_sec:.2f}s)"
            )
        # Do not keep stale dynamic detections alive just because a static
        # memory exists.  Once the normal sensor cache expires, only confirmed
        # stationary objects are safe to retain.
        agents = persistent
    else:
        agents = _advance_agents(perception.last_agents, perception.last_detection_ts, now)
    review = getattr(perception, "stationary_vru_review", None)
    if review is not None:
        review.update(agents, sensor_timestamp=perception.last_detection_ts, info=info,
                      brake_held=getattr(getattr(agent, "_orchestrator", None),
                                         "_pedestrian_ttc_brake_hold", None) is not None)
        agents = review.apply(
            agents, obs=obs, info={**info, "frame_index": frame_index},
            camera_specs=(list(getattr(agent, "_camera_specs", []) or [])
                          + list(getattr(agent, "_extra_sensor_specs", []) or [])),
            max_staleness_sec=perception.max_staleness_sec)
        info["stationary_vru_review"] = {"busy": review.busy, "applied": list(review.last_applied)}
    if perception.hd_map_lane_annotations:
        agents = _annotate_agents_from_hd_map(agents)
    perception.cache_frame_index = frame_index
    perception.cache_agents = list(agents)

    return agents






def finalize_perception(agent) -> None:
    perception = getattr(agent, "_perception", None)
    if perception is None:
        return
    review = getattr(perception, "stationary_vru_review", None)
    if review is not None:
        review.finalize()
    if getattr(perception, "telemetry_fh", None) is not None:
        try:
            perception.telemetry_fh.close()
        except Exception:
            pass
        perception.telemetry_fh = None
    if perception.service is not None:
        perception.service.finalize()
