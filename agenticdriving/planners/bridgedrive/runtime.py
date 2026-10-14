"""BridgeDrive detection_ft runtime hosted by the dedicated LEAD worker."""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

# BridgeDrive worker configuration defaults; applied only
# when the service did not set them (ClosedLoopConfig hard-fails on unknown
# keys, so this string must track LEAD's config surface).
_DEFAULT_CLOSED_LOOP_CONFIG = (
    "steer_modality=route throttle_modality=target_speed "
    "brake_modality=target_speed step_num=20 diffusion_speed=False"
)
_DEFAULT_TRAINING_CONFIG = "diffusion_speed=False"


def deserialize_global_plans(
    plan_gps: Any, plan_world: Any
) -> Tuple[list, list]:
    """Rebuild leaderboard-shaped plans from the pipe's plain-data form.

    The agent side serializes ``_global_plan`` as [(gps_dict, int_cmd)] and
    ``_global_plan_world_coord`` as [((x, y, z), (pitch, yaw, roll),
    int_cmd)] because carla.Transform / RoadOption do not pickle. LEAD's
    consumers need exactly: gps dicts with lat/lon/z + RoadOption cmds
    (route_planner.set_route) and transforms with .location.x/y/z
    (find_gps_ref).
    """
    import carla
    from agents.navigation.local_planner import RoadOption

    gps = [(dict(pos), RoadOption(int(cmd))) for pos, cmd in plan_gps]
    world = []
    for loc, rot, cmd in plan_world:
        transform = carla.Transform(
            carla.Location(x=float(loc[0]), y=float(loc[1]), z=float(loc[2])),
            carla.Rotation(
                pitch=float(rot[0]), yaw=float(rot[1]), roll=float(rot[2])
            ),
        )
        world.append((transform, RoadOption(int(cmd))))
    return gps, world


class BridgeDriveRuntime:
    """Owns the hosted SensorAgent; one instance per worker process."""

    def __init__(self, checkpoint_dir: str) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.agent = None
        self._town: Optional[str] = None
        self._first_tick = True
        self._detection = None

    # -- episode lifecycle ---------------------------------------------------

    def init_episode(
        self,
        global_plan: Any,
        global_plan_world_coord: Any,
        town: str,
    ) -> None:
        os.environ.setdefault("LEAD_CLOSED_LOOP_CONFIG", _DEFAULT_CLOSED_LOOP_CONFIG)
        os.environ.setdefault("LEAD_TRAINING_CONFIG", _DEFAULT_TRAINING_CONFIG)

        import lead
        from lead.inference.sensor_agent_bridgedrive import SensorAgent

        # Portability: the checkpoint's config.json pins plan_anchor_path as an
        # absolute path from the training box. Override it via the env-config
        # merge (applied on top of the json) with the anchor shipped inside
        # whatever LEAD clone this worker actually imported, so replication
        # needs no config.json edit.
        anchor = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(lead.__file__))),
            "anchor_utils", "anchor_data", "lead_cp_kmeans_60_10.npy",
        )
        if os.path.isfile(anchor) and " " not in anchor:
            tc = os.environ.get("LEAD_TRAINING_CONFIG", "")
            if "plan_anchor_path=" not in tc:
                os.environ["LEAD_TRAINING_CONFIG"] = (
                    f"{tc} plan_anchor_path={anchor}".strip()
                )

        self._town = str(town)
        agent = SensorAgent("localhost", 0)  # AutonomousAgent.__init__ only
        # Same content the leaderboard installed on our agent (serialized to
        # plain data for the pipe, rebuilt here); LEAD's RoutePlanner and
        # find_gps_ref consume exactly these (base_agent.py:24-80).
        plan_gps, plan_world = deserialize_global_plans(
            global_plan, global_plan_world_coord
        )
        agent._global_plan = plan_gps
        agent._global_plan_world_coord = plan_world
        agent.setup(self.checkpoint_dir)
        # _init() is CARLA-coupled and skipped; nothing below needs _vehicle
        # or _world (town is forwarded, video_recorder uses are hasattr-
        # guarded, weather is the simulator's business on the agent side).
        self._wrap_trajectory_heads(agent)
        self.agent = agent
        import sys
        sys.path.insert(0, os.environ["BRIDGEDRIVE_DETECTION_CODE"])
        from closed_loop_detection import ClosedLoopDetection
        self._detection = ClosedLoopDetection(agent)
        self._first_tick = True

    def _wrap_trajectory_heads(self, agent: Any) -> None:
        """Route TrajectoryHead calls through forward_test_all_modes,
        stashing (all_modes, cls_logits) and returning the gathered winner so
        the surrounding LEAD code runs unchanged."""

        for net in agent.closed_loop_inference.nets:
            head = net.planning_decoder.trajectory_head
            orig_forward = head.forward

            def wrapped(
                ego_query=None,
                bev_feature=None,
                bev_spatial_shape=None,
                gt_checkpoints=None,
                global_img=None,
                _head=head,
                _orig=orig_forward,
                **kwargs,
            ):
                if _head.training or gt_checkpoints is not None:
                    return _orig(
                        ego_query=ego_query,
                        bev_feature=bev_feature,
                        bev_spatial_shape=bev_spatial_shape,
                        gt_checkpoints=gt_checkpoints,
                        global_img=global_img,
                        **kwargs,
                    )
                x0, _, _ = _head.forward_test_all_modes(
                    ego_query, bev_feature, bev_spatial_shape, global_img)
                return x0

            head.forward = wrapped


    # -- perception harvest (judge feed) -------------------------------------

    def _harvest_perception(self, prediction: Any) -> Dict[str, Any]:
        """Aux-head products for the PDMS judge, from the forward that just ran.

        Consume LEAD's final ensemble prediction, not the first network's raw
        CenterNet top-k tensor.  This preserves LEAD's confidence filter and
        cross-model NMS and avoids forwarding several competing classes for
        one physical object.  BEV semantic is likewise the ensembled output.
        Values cross the worker boundary as fp32/uint8 arrays.
        """
        boxes = np.zeros((0, 9), dtype=np.float32)
        final_boxes = prediction.pred_bounding_box_vehicle_system or []
        if final_boxes:
            boxes = np.asarray(
                [
                    [
                        box.x,
                        box.y,
                        box.w,
                        box.h,
                        box.yaw,
                        box.velocity,
                        box.brake,
                        box.clazz,
                        box.score,
                    ]
                    for box in final_boxes
                ],
                dtype=np.float32,
            ).reshape(-1, 9)

        bev_argmax = None
        if prediction.pred_bev_semantic is not None:
            bev_argmax = (
                prediction.pred_bev_semantic.float()
                .argmax(1)[0]
                .cpu()
                .numpy()
                .astype(np.uint8)
            )
        return {"boxes_vehicle_system": boxes, "bev_argmax": bev_argmax}

    # -- per tick --------------------------------------------------------------

    def _tick_agent(
        self, obs: Dict[str, Any], info: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Shared stateful prelude: step increment, applied-control feed,
        agent.tick() (Kalman, LiDAR sweep queue, target points). Returns the
        tick_data, or None on the warm-up first tick (mirrors standalone
        run_step's first call: full brake, tick only, no forward)."""
        import carla

        agent = self.agent
        assert agent is not None, "init_episode must run before ticking"
        input_data = obs["input_data"]
        if self._detection is not None:
            # Validate and capture real sensors before preprocessing mutates them
            # or fills missing cameras. The frozen specialist requires the full rig.
            self._detection.observe(input_data)
        agent.step += 1

        if self._first_tick:
            agent.control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
            agent.tick(input_data)
            self._first_tick = False
            return None

        prev = info.get("bd_prev_control") or {}
        agent.control = carla.VehicleControl(
            steer=float(prev.get("steer", 0.0)),
            throttle=float(prev.get("throttle", 0.0)),
            brake=float(prev.get("brake", 1.0)),
        )
        return agent.tick(input_data)

    def tick_only(
        self, obs: Dict[str, Any], info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Judge-duty stride support: keep the stateful preprocessing honest
        on EVERY simulator tick while the expensive forward (backbone + aux
        heads + 20-step diffusion decoder) runs only on detect() frames."""
        tick_data = self._tick_agent(obs, info)
        return {"ticked": True, "warmup": tick_data is None}

    def process(
        self, obs: Dict[str, Any], info: Dict[str, Any]
    ) -> Dict[str, Any]:
        import torch

        agent = self.agent
        tick_data = self._tick_agent(obs, info)
        if tick_data is None:
            return {"first_tick": True}

        # --- transcribed from sensor_agent_bridgedrive.py:498-532 ---------
        # (single substitution: town from the forwarded name instead of
        # self._world.get_map(); input-log block dropped — worker-side I/O)
        device = agent.device
        input_data_tensors = {
            "rgb": torch.Tensor(tick_data["rgb"]).to(device, dtype=torch.float32)[None],
            "rasterized_lidar": torch.Tensor(tick_data["rasterized_lidar"]).to(
                device, dtype=torch.float32
            ),
            "target_point_previous": torch.Tensor(tick_data["target_point_previous"])
            .to(device, dtype=torch.float32)
            .view(1, 2),
            "target_point": torch.Tensor(tick_data["target_point"])
            .to(device, dtype=torch.float32)
            .view(1, 2),
            "target_point_next": torch.Tensor(tick_data["target_point_next"])
            .to(device, dtype=torch.float32)
            .view(1, 2),
            "speed": torch.Tensor([tick_data["speed"]])
            .to(device, dtype=torch.float32)
            .view(1),
            "command": torch.Tensor(tick_data["command"])
            .to(device, dtype=torch.float32)
            .view(1, 6),
            "next_command": torch.Tensor(tick_data["next_command"])
            .to(device, dtype=torch.float32)
            .view(1, 6),
            "town": np.array([self._town]),
        }
        if agent.training_config.use_radars and "radar" in tick_data:
            input_data_tensors["radar"] = torch.Tensor(tick_data["radar"]).to(
                device, dtype=torch.float32
            )[None]
        # -------------------------------------------------------------------

        prediction = agent.closed_loop_inference.forward(data=input_data_tensors)
        perception = self._harvest_perception(prediction)
        return {"perception": self._detection.apply(perception, input_data_tensors)}
