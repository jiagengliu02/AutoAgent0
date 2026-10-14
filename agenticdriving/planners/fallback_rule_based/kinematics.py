"""Ego-frame forward integrator for the fallback recovery planner.

The recovery hook must return a single trajectory of ``[x_right, y_forward]``
poses in the CARLA ego-local frame (origin at the ego at planning time). The atomic
actions configure a *signed* target speed (``+`` forward, ``-`` reverse) and a
front-wheel deflection (steering) angle on a :class:`VehicleSim`; the executor
advances it one ``dt`` per generator ``yield`` to materialize the path.

A simple kinematic-bicycle model is used so that:

* reversing (negative speed) walks the path backward along the heading, and
* a steering angle turns the vehicle, with the yaw direction naturally inverting
  in reverse (back up with the wheel right -> the nose swings the other way).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

import numpy as np

# Front-wheel deflection is clamped so a single maneuver stays physically plausible.
MAX_STEER_RAD = math.radians(35.0)
# Effective wheelbase (m) for the bicycle yaw model.
DEFAULT_WHEELBASE_M = 2.7


@dataclass
class VehicleSim:
    """Integrate ego motion in the CARLA ego-local frame (x=right, y=forward).

    ``heading`` is the chassis yaw (rad) measured from +y (forward), positive
    turning toward +x (right). ``speed`` is signed: positive drives forward,
    negative reverses. Each :meth:`step` advances one ``dt`` and appends one
    ``[x_right, y_forward]`` pose.
    """

    # Waypoint spacing (s) of the materialized plan, not the CARLA tick; see
    # DrivingContext.dt (primitives.py) for who must agree with this value.
    dt: float = 0.5
    max_accel_mps2: float = 2.0
    max_decel_mps2: float = 3.0
    wheelbase_m: float = DEFAULT_WHEELBASE_M

    speed: float = 0.0          # signed: + forward, - reverse
    target_speed: float = 0.0   # signed target the speed eases toward
    steer_rad: float = 0.0      # commanded front-wheel deflection (+ = right)

    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    poses: List[List[float]] = field(default_factory=list)
    # Per-step signed state after each step: [speed, steer_rad]. Used to emit the
    # exact (acc, steer_rate) controls for open-loop execution (see control_sequence).
    states: List[List[float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Remember the starting signed speed / steer so control diffs are anchored
        # to the true initial state (the ego state at planning time).
        self._start_speed = float(self.speed)
        self._start_steer = float(self.steer_rad)

    def step(self) -> None:
        # Speed eases toward the signed target; accelerate vs. brake limit picked
        # by whether the target magnitude is growing or shrinking.
        prev_speed = self.speed
        dv = self.target_speed - self.speed
        rate = self.max_accel_mps2 if abs(self.target_speed) >= abs(self.speed) else self.max_decel_mps2
        max_dv = rate * self.dt
        self.speed = self.speed + max(-max_dv, min(max_dv, dv))

        # Integrate position over the step with the average of the pre-/post-update
        # speed (trapezoidal). Using the end-of-step speed would collapse a step that
        # brakes to a full stop to zero displacement -- so a low-speed brake produced
        # an all-zero trajectory even though the vehicle still rolls forward while
        # slowing. The average keeps that sub-step motion.
        avg_speed = 0.5 * (prev_speed + self.speed)

        # Kinematic-bicycle yaw rate; the sign of speed makes reverse turns invert.
        steer = max(-MAX_STEER_RAD, min(MAX_STEER_RAD, self.steer_rad))
        self.heading += (avg_speed / max(self.wheelbase_m, 1e-3)) * math.tan(steer) * self.dt

        self.x += avg_speed * math.sin(self.heading) * self.dt
        self.y += avg_speed * math.cos(self.heading) * self.dt
        self.poses.append([self.x, self.y])
        self.states.append([self.speed, steer])

    def trajectory(self, num_poses: int) -> np.ndarray:
        """Return the path as ``[num_poses, 2]``, padding with the last pose."""
        pts = list(self.poses) or [[0.0, 0.0]]
        while len(pts) < num_poses:
            pts.append(list(pts[-1]))
        return np.asarray(pts[:num_poses], dtype=np.float32)

    def state_sequence(self, num_poses: int) -> np.ndarray:
        """Return one signed ``[speed, steer_rad]`` state per plan pose.

        This is dt-invariant (physical speed in m/s, steer angle in rad). The
        recovery controller uses it as the phase clock for target speed and gear,
        so rows must stay exactly aligned with :meth:`trajectory`. If a primitive
        generator finishes before ``num_poses``, its trajectory is padded at the
        final pose and the matching states are padded with a stationary command.
        Repeated live states are deliberately retained: e.g. a lane change may
        cruise straight at the same speed for several final poses, and dropping
        those rows would end its phase early.
        """
        states = [list(s) for s in self.states[:num_poses]]
        while len(states) < num_poses:
            states.append([0.0, 0.0])
        return np.asarray(states, dtype=np.float32)
