"""BridgeDrive sensor rig and frame constants for the CARLA agent.

The 12-sensor rig below is FROZEN output of LEAD's ``av_sensor_setup(config,
lidar=True, radar=True, sensor_agent=True, perturbate=False, ...)`` evaluated
against the shipped checkpoint config
(checkpoints/bridgedrive/config.json), dumped
from LEAD @ bridgedrive_f2d/92ffbe92. Frozen here so the agent
process never imports LEAD; the worker (which does import LEAD) is the only
LEAD-coupled component. Do not hand-edit values — regenerate from LEAD if the
checkpoint ever changes.

LiDAR/radar physics (range, channels, rotation rate, ...) are force-set by the
fail2drive evaluator itself (agent_wrapper._preprocess_sensor_spec) and cannot
be influenced from these specs.
"""
from __future__ import annotations

from typing import Any, Dict, List

BRIDGEDRIVE_SENSOR_SPECS: List[Dict[str, Any]] = [
    {"type": "sensor.camera.rgb", "x": 0.1, "y": -0.35, "z": 2.25,
     "roll": 0.0, "pitch": 0.0, "yaw": -54.5,
     "width": 384, "height": 384, "fov": 60, "id": "rgb_1"},
    {"type": "sensor.camera.rgb", "x": 0.35, "y": 0.0, "z": 2.25,
     "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
     "width": 384, "height": 384, "fov": 60, "id": "rgb_2"},
    {"type": "sensor.camera.rgb", "x": 0.1, "y": 0.35, "z": 2.25,
     "roll": 0.0, "pitch": 0.0, "yaw": 54.5,
     "width": 384, "height": 384, "fov": 60, "id": "rgb_3"},
    {"type": "sensor.lidar.ray_cast", "x": 0.0, "y": 0.0, "z": 2.5,
     "roll": 0.0, "pitch": 0.0, "yaw": -90.0, "id": "lidar1"},
    {"type": "sensor.lidar.ray_cast", "x": 0.0, "y": 0.0, "z": 2.5,
     "roll": 0.0, "pitch": 0.0, "yaw": -270.0, "id": "lidar2"},
    {"type": "sensor.other.radar", "x": 2.6, "y": 0, "z": 0.6,
     "roll": 0.0, "pitch": 0.0, "yaw": -45.0,
     "horizontal_fov": 90, "vertical_fov": 0.1, "id": "radar1"},
    {"type": "sensor.other.radar", "x": 2.6, "y": 0, "z": 0.6,
     "roll": 0.0, "pitch": 0.0, "yaw": 45.0,
     "horizontal_fov": 90, "vertical_fov": 0.1, "id": "radar2"},
    {"type": "sensor.other.radar", "x": -2.6, "y": 0, "z": 0.6,
     "roll": 0.0, "pitch": 0.0, "yaw": 135,
     "horizontal_fov": 90, "vertical_fov": 0.1, "id": "radar3"},
    {"type": "sensor.other.radar", "x": -2.6, "y": 0, "z": 0.6,
     "roll": 0.0, "pitch": 0.0, "yaw": 225,
     "horizontal_fov": 90, "vertical_fov": 0.1, "id": "radar4"},
    {"type": "sensor.other.imu", "x": 0.0, "y": 0.0, "z": 0.0,
     "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
     "sensor_tick": 0.05, "id": "imu"},
    {"type": "sensor.other.gnss", "x": 0.0, "y": 0.0, "z": 0.0,
     "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
     "sensor_tick": 0.01, "id": "gps"},
    {"type": "sensor.speedometer", "reading_frequency": 20, "id": "speed"},
]

#: input_data entries piped verbatim to the worker each tick.
WORKER_FORWARDED_IDS = tuple(spec["id"] for spec in BRIDGEDRIVE_SENSOR_SPECS)
