# adapters/carla_agent/agenticdriving_carla.py

from __future__ import annotations
import sys
import logging
import os
from collections import deque
from pathlib import Path
from omegaconf import OmegaConf

# This file lives at <repo>/adapters/carla_agent/, so the repo root that holds
# the `agenticdriving` pipeline package is two levels up.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Root that holds the `agenticdriving` pipeline package (== repo root here).
_AGENTICDRIVING_ROOT = Path(os.environ.get("AGENTICDRIVING_ROOT", str(_REPO_ROOT)))
if str(_AGENTICDRIVING_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENTICDRIVING_ROOT))

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track

from agenticdriving_carla_helper import (
    EGO_HISTORY_FRAMES,
    apply_stop_sign_speed_cap,
    agent_brake_control,
    finalize_recording,
    find_hero,
    get_ego_state,
    maybe_record_frames,
    record_post_step_visualization,
    run_step_sparsedrivev2,
    setup_cameras_and_recording,
    setup_sparsedrivev2,
    stop_sign_brake_control,
    update_stop_sign_control,
    resolve_scenario_output_dir,
    _control_cfg,
    _control_mode,
)
from perception_helper import (
    finalize_perception,
    pump_bridgedrive_perception,
    setup_perception,
)

LOG = logging.getLogger(__name__)


def get_entry_point():
    return "AgenticDrivingCarlaAgent"


class AgenticDrivingCarlaAgent(AutonomousAgent):

    def set_route_context(self, route_name: str, repetition_index: int = 0) -> None:
        """Optional hook the leaderboard may call before setup() for each route."""
        self._route_name = str(route_name)
        self._repetition_index = int(repetition_index)

    def _resolve_route_context(self):
        """(route_name, repetition_index) for the current route.

        Pristine fail2drive does not call set_route_context; leaderboard_hooks captures
        it from the RouteScenario instead. Prefer an explicit set_route_context, then
        fall back to the hook, then to defaults.
        """
        name = getattr(self, "_route_name", None)
        rep = int(getattr(self, "_repetition_index", 0))
        if name is None:
            try:
                import leaderboard_hooks

                hook_name, hook_rep = leaderboard_hooks.get_route_context()
                if hook_name is not None:
                    name, rep = hook_name, hook_rep
            except Exception:
                pass
        return name, rep

    def setup(self, path_to_conf_file):
        self.track = Track.SENSORS
        raw_cfg = OmegaConf.load(path_to_conf_file)
        self._cfg = OmegaConf.to_container(raw_cfg, resolve=True)
        # Optional judge_mode preset expands into perception/verifier keys
        # before anything reads them (conflicting explicit keys error out).
        from perception_helper import expand_judge_mode

        expand_judge_mode(self._cfg)
        self._carla_cfg = self._cfg.get("carla", {})

        from agenticdriving.release import prepare_config
        prepare_config(self._cfg, _REPO_ROOT)
        self._planner_type = "sparsedrivev2"
        LOG.info("AgenticDriving: planner_type=%s", self._planner_type)

        base_output_dir = Path(
            self._carla_cfg.get("output_dir", "/tmp/fail2drive_agenticdriving")
        )
        route_name, repetition_index = self._resolve_route_context()
        self._output_dir = resolve_scenario_output_dir(
            self._carla_cfg,
            base_output_dir=base_output_dir,
            planner_type=self._planner_type,
            scenario_name=route_name,
            repetition_index=repetition_index,
            config_path=path_to_conf_file,
        )

        self._route_cursor = 0
        self._frame_index = 0
        self._last_steer = 0.0
        self._info_history = deque(maxlen=EGO_HISTORY_FRAMES)
        self._vlm_selector = None
        self._vlm_selector_cfg = None

        setup_sparsedrivev2(self)

        LOG.info(
            "AgenticDriving: control mode=%s cfg=%s",
            _control_mode(self),
            _control_cfg(self),
        )

        setup_perception(self)
        setup_cameras_and_recording(self)

    def sensors(self):
        sensors = []
        for spec in getattr(self, "_camera_specs", []):
            sensors.append({
                "type": spec["type"],
                "x": spec["x"],
                "y": spec["y"],
                "z": spec["z"],
                "roll": spec["roll"],
                "pitch": spec["pitch"],
                "yaw": spec["yaw"],
                "width": spec["width"],
                "height": spec["height"],
                "fov": spec["fov"],
                "id": spec["id"],
            })

        # Planner-specific extra sensors (e.g. TransFuser++ front camera +
        # lidar) ride alongside the configured camera rig.
        for spec in getattr(self, "_extra_sensor_specs", []):
            sensors.append(dict(spec))

        # Recording-only high-res camera (carla.recording_camera).
        # Deliberately not part of _camera_specs: its frames are routed to
        # the video composite only, never into the planner's obs["rgb"].
        # "name" is our artifact label, not a CARLA sensor key -- strip it.
        rec_spec = getattr(self, "_recording_camera_spec", None)
        if rec_spec is not None:
            sensors.append(
                {k: v for k, v in rec_spec.items() if k != "name"}
            )

        sensors.append({
            "type": "sensor.speedometer",
            "reading_frequency": 25,
            "id": "speedometer",
        })

        return sensors

    def run_step(self, input_data, timestamp):
        hero = find_hero(self)
        if hero is None:
            return agent_brake_control(self)

        ego_state = get_ego_state(self, hero, input_data, timestamp)
        self._info_history.append(ego_state["info"])
        while len(self._info_history) < EGO_HISTORY_FRAMES:
            self._info_history.appendleft(
                dict(self._info_history[0])
            )

        self._route_cursor = min(
            self._route_cursor + 1,
            max(0, len(self._global_plan_world_coord) - 1),
        )

        maybe_record_frames(self, ego_state)

        # Mode-A judge perception (bridgedrive source, other planner): the
        # hosted LEAD agent's tick() is stateful, so its worker must be fed
        # on EVERY simulator tick, independent of replan/scoring cadence.
        pump_bridgedrive_perception(self, ego_state["obs"], ego_state["info"])

        # The stop-sign gate is planner-independent. It launches one async VLM
        # request when HD-map lookahead first sees a junction and advances the
        # braking/hold latch from simulator time and measured ego speed.
        stop_sign_brake = update_stop_sign_control(self, ego_state)

        ctrl = run_step_sparsedrivev2(self, ego_state)

        if stop_sign_brake:
            ctrl = stop_sign_brake_control(self, ego_state["info"])
        else:
            ctrl = apply_stop_sign_speed_cap(
                self,
                ego_state["info"],
                ctrl,
            )

        record_post_step_visualization(self, ego_state)
        self._frame_index += 1
        return ctrl

    def destroy(self):
        stop_sign_manager = getattr(self, "_stop_sign_manager", None)
        if stop_sign_manager is not None:
            try:
                stop_sign_manager.finalize()
            except Exception:
                pass

        semantic_manager = getattr(self, "_semantic_takeover_manager", None)
        if semantic_manager is not None:
            try:
                semantic_manager.finalize()
            except Exception:
                pass

        # Flush streaming/fallback footage before shutting down auxiliary workers. This
        # ordering is especially important on an external wall-clock timeout:
        # the timeout grace period is finite, and preserving the episode video
        # takes priority over optional perception-service cleanup.
        try:
            finalize_recording(self)
        except Exception:
            LOG.exception("Failed to finalize recording")

        try:
            finalize_perception(self)
        except Exception:
            LOG.exception("Failed to finalize perception")

        vlm_selector = getattr(self, "_vlm_selector", None)
        if vlm_selector is not None:
            try:
                vlm_selector.finalize()
            except Exception:
                pass
