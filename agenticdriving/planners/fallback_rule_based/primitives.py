"""Driving primitives (the "APIs") for the fallback recovery planner.

LaMPilot-style: a small set of documented primitives that the VLM composes into
a recovery maneuver, plus a :data:`PRIMITIVE_MENU` of standalone maneuvers used
when the VLM is disabled (random selection).

The primitives come in three tiers:

1. **Low-level setters** -- ``set_target_speed`` / ``set_steering`` configure the
   signed speed and front-wheel deflection without advancing time.
2. **Basic atomic actions** -- ``go_forward`` / ``reverse`` / ``emergency_stop`` /
   ``hold`` / ``wait``. Each is a *generator* that sets its targets and then
   ``yield``s once per timestep for ``duration`` seconds, so its parameters
   (target speed m/s, duration s, steering deflection deg) are exactly what the
   VLM fills in. Reversing (``reverse``) is a first-class atom.
3. **Action combinations** -- ``back_up_and_steer`` / ``three_point_turn`` /
   ``back_then_bypass`` / ``swerve_and_stop`` / ``change_lane`` /
   ``lane_bypass`` chain the atoms with ``yield from`` to express common
   recovery maneuvers in one call.

Every primitive mutates a shared :class:`PrimitiveRuntime` whose
:class:`VehicleSim` the executor steps forward (one step per yielded tick).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional

import numpy as np

from agenticdriving.planners.fallback_rule_based.kinematics import MAX_STEER_RAD, VehicleSim


# Baseline front-wheel deflection (deg) for turning primitives: a moderate,
# physically comfortable steer angle used as the default for lane changes,
# nudges, and recovery turns. Sharper maneuvers scale it up (e.g. a K-turn uses
# 2x, an intersection turn 1.5x, a u-turn a dedicated 34 deg).
DEFAULT_STEER_DEG = 15.0

# System-owned pace for the short-exposure adjacent-lane bypass. Provider-side
# availability gating imports these constants so the gap calculation and the
# trajectory it authorizes cannot silently disagree about ego speed/acceleration.
# Keep the bypass brisk without turning its fixed-duration S-curves into an
# excessively long longitudinal excursion.  Ten metres per second clears a
# close blocker roughly one 0.5 s planning row earlier than the previous
# 8 m/s default, shortening exposure to traffic in the borrowed lane while
# retaining the trackable acceleration and lateral phase timings below.
LANE_BYPASS_TARGET_SPEED_MPS = 10.0
# Match the recovery PID's physical acceleration ceiling.  Planning at 7 m/s^2
# made the steering phase start while the real CARLA ego was still far below
# the simulated entry speed, so tracking lag could carry the remaining plan
# into the obstacle and trip the mid-hold NC guard.
LANE_BYPASS_ACCEL_MPS2 = 3.5
LANE_BYPASS_ENTRY_SPEED_MPS = 3.0
LANE_BYPASS_LANE_WIDTH_M = 3.0
LANE_BYPASS_LANE_CHANGE_DURATION_SEC = 1.5
# CARLA's real steering rack does not reverse the wheel angle instantaneously.
# A single 0.5 s counter-steer left Route 21 at roughly -39 degrees of heading
# and carried the ego across the borrowed lane onto the far-side sidewalk.
# Two 0.5 s waypoints make the counter-steer trackable before the configured
# obstacle-clearance decision begins.
LANE_BYPASS_COUNTER_STEER_DURATION_SEC = 1.0
LANE_BYPASS_RETURN_PHASE_DURATION_SEC = 0.5
LANE_BYPASS_PASS_DURATION_SEC = 0.0
LANE_BYPASS_OBSTACLE_CLEARANCE_M = 8.0
# Give the real vehicle a full second of straight, in-lane travel after the
# return S-curve.  A single 0.5 s row left only a repeated endpoint behind it;
# SparseDrive's feedback then chased that point with an abrupt opposite-lock
# command exactly as the hold ended.  Two straight rows provide a real heading
# reference while the steering rack recenters.
LANE_BYPASS_SETTLE_DURATION_SEC = 1.0
# Time from starting the first S-curve until the return S-curve is complete.
# The final straight settle is already back in the original lane.
LANE_BYPASS_BORROW_TIME_SEC = (
    LANE_BYPASS_LANE_CHANGE_DURATION_SEC
    + LANE_BYPASS_COUNTER_STEER_DURATION_SEC
    + 2.0 * LANE_BYPASS_RETURN_PHASE_DURATION_SEC
    + LANE_BYPASS_PASS_DURATION_SEC
)

# Deceleration (m/s^2, ~0.8 g) used for EVERY brake-to-stop. There is a single
# brake atom (``emergency_stop``); it always decelerates this hard -- well above
# the sim's cruising ``max_decel_mps2`` (~3) -- so the ego snaps to a halt
# instead of coasting into whatever is ahead. No softer stop variant exists.
HARD_DECEL_MPS2 = 8.0

# VLM-tunable net lateral-distance envelopes for parameterized recovery
# maneuvers.  These bounds are intentionally conservative with respect to the
# bicycle-model rollout and the fixed recovery horizons below: values near the
# upper edge remain reachable without demanding steering beyond MAX_STEER_RAD.
RECOVERY_LATERAL_DISTANCE_RANGES_M: Dict[str, tuple[float, float]] = {
    "swerve": (0.5, 2.5),
    "back": (0.3, 1.5),
    "back_bypass": (1.0, 3.0),
}


def recovery_lateral_distance_range(
    primitive_name: str,
) -> Optional[tuple[float, float]]:
    """Return the tunable lateral-distance range for a recovery primitive."""
    name = str(primitive_name).strip().lower()
    if name.startswith("back_bypass_"):
        return RECOVERY_LATERAL_DISTANCE_RANGES_M["back_bypass"]
    if name.startswith("swerve_"):
        return RECOVERY_LATERAL_DISTANCE_RANGES_M["swerve"]
    if name.startswith("back_"):
        return RECOVERY_LATERAL_DISTANCE_RANGES_M["back"]
    return None


def validate_recovery_lateral_distance(
    primitive_family: str,
    lateral_m: float,
) -> float:
    """Validate and normalize a requested absolute lateral displacement."""
    family = str(primitive_family).strip().lower()
    if family not in RECOVERY_LATERAL_DISTANCE_RANGES_M:
        raise ValueError(f"unsupported recovery lateral family: {primitive_family!r}")
    value = abs(float(lateral_m))
    lower, upper = RECOVERY_LATERAL_DISTANCE_RANGES_M[family]
    if not (lower <= value <= upper):
        raise ValueError(
            f"{family} lateral_m={value:.3f} outside [{lower:.3f}, {upper:.3f}]"
        )
    return value


@dataclass
class DrivingContext:
    """Per-frame scene summary the primitives and prompt read from."""

    ego_speed_mps: float = 0.0
    route_command: str = "straight"  # 'left' | 'right' | 'straight'
    task_target_hint: Optional[str] = None
    front_vehicle_distance_m: Optional[float] = None
    # None means the caller is not using scene-gated code generation (for
    # example fixed primitive mode). Codegen/recovery callers set explicit
    # booleans so unavailable adjacent-lane directions cannot be invoked.
    lane_bypass_left_available: Optional[bool] = None
    lane_bypass_right_available: Optional[bool] = None
    # Stationary wait until the selected measured traffic gap is usable.  Once
    # the delay expires, steering starts immediately on the first moving tick;
    # there is no straight acceleration lead-in.
    lane_bypass_left_entry_delay_sec: float = 0.0
    lane_bypass_right_entry_delay_sec: float = 0.0
    lane_bypass_obstacle_clearance_m: float = LANE_BYPASS_OBSTACLE_CLEARANCE_M
    # primitive tuning (from the fallback yaml)
    # NOT the CARLA tick: time spacing between consecutive plan waypoints. Must
    # match what the plan's consumer assumes (agent._plan_dt_sec / verifier
    # timestep). The recovery-ladder provider (provider.py) relies on this
    # default and its consumers assume 0.5 s; the primitives planner overrides
    # it from config (0.05) and propagates the value to _plan_dt_sec itself.

    dt: float = 0.5
    max_accel_mps2: float = 2.0
    max_decel_mps2: float = 3.0
    lateral_nudge_m: float = 1.5
    default_target_speed_mps: float = 8.0   # forward cruise fallback (~29 km/h)
    reverse_speed_mps: float = 2.0      # default magnitude used when reversing
    default_steer_deg: float = DEFAULT_STEER_DEG  # default deflection for turning atoms


def _agent_local_positions(privileged_info: Any) -> List[np.ndarray]:
    """Best-effort extraction of agent positions in the ego-local frame.

    ``privileged_info`` is an opaque payload from ``get_agent_privileged_info``;
    we accept a few common shapes and degrade to ``[]`` on anything else so the
    perception primitives never raise.
    """
    agents = privileged_info
    if isinstance(privileged_info, dict):
        agents = privileged_info.get("agents") or privileged_info.get("objects") or []
    if agents is None:
        return []
    out: List[np.ndarray] = []
    try:
        for agent in agents:
            pos = None
            if isinstance(agent, dict):
                pos = (
                    agent.get("position")
                    or agent.get("center")
                    or agent.get("translation")
                    or agent.get("xy")
                )
            elif isinstance(agent, (list, tuple, np.ndarray)):
                pos = agent
            if pos is None:
                continue
            arr = np.asarray(pos, dtype=np.float32).reshape(-1)
            if arr.size >= 2:
                out.append(arr[:2])
    except Exception:
        return []
    return out


def front_vehicle_distance(privileged_info: Any, lane_half_width_m: float = 1.75) -> Optional[float]:
    """Forward distance (m) to the nearest agent roughly ahead, ``None`` if none.

    Assumes agent positions are already ego-local ``[x_right, y_forward]``; agents
    behind the ego (``y <= 0``) or outside the ego lane corridor are ignored.
    """
    best: Optional[float] = None
    for xy in _agent_local_positions(privileged_info):
        x_right, y_forward = float(xy[0]), float(xy[1])
        if y_forward <= 0.0 or abs(x_right) > lane_half_width_m:
            continue
        if best is None or y_forward < best:
            best = y_forward
    return best


# A yielded tick is just a sentinel; the executor advances the sim one ``dt``
# every time a tick is produced. Value is unused.
_TICK = None


class PrimitiveRuntime:
    """Holds the :class:`VehicleSim` + context and exposes the bound APIs.

    The sim starts at the current ego speed driving straight, so a maneuver that
    only steers still keeps moving.
    """

    def __init__(self, ctx: DrivingContext) -> None:
        self.ctx = ctx
        self.sim = VehicleSim(
            dt=ctx.dt,
            max_accel_mps2=ctx.max_accel_mps2,
            max_decel_mps2=ctx.max_decel_mps2,
            speed=max(0.0, ctx.ego_speed_mps),
            target_speed=max(0.0, ctx.ego_speed_mps),
        )

    # --- helpers ----------------------------------------------------------
    def _ticks(self, duration: float) -> int:
        """Number of sim steps that approximate ``duration`` seconds (>= 1)."""
        return max(1, int(round(abs(float(duration)) / max(self.sim.dt, 1e-6))))

    def _stop_duration(self) -> float:
        """Seconds to brake the *current* signed speed to a full stop.

        Sized from the live sim speed and the decel limit (+ a small margin) so a
        reverse leg that follows actually nets backward instead of being consumed
        bleeding off forward speed. A reverse atom started while still rolling
        forward at, say, 5 m/s spends its whole horizon decelerating and never
        moves backward -- prepend ``emergency_stop(self._stop_duration())`` first.
        """
        return abs(self.sim.speed) / max(self.sim.max_decel_mps2, 1e-3) + 0.2

    def autopilot(self):
        """One timestep tick. ``yield autopilot()`` to advance the vehicle."""
        return _TICK

    # --- Low-level setters (no time advance) ------------------------------
    def set_target_speed(self, target_speed: float) -> None:
        """Set the signed target speed in m/s (+ forward, - reverse)."""
        self.sim.target_speed = float(target_speed)

    def set_steering(self, steer_deg: float) -> None:
        """Set the front-wheel deflection angle in degrees (+ right, - left)."""
        self.sim.steer_rad = math.radians(float(steer_deg))

    # --- Basic atomic actions (generators) --------------------------------
    def go_forward(
        self,
        target_speed: Optional[float] = 8.0,
        duration: float = 2.0,
        steer_deg: float = 0.0,
    ) -> Iterator[Any]:
        """Drive FORWARD at ``target_speed`` m/s for ``duration`` s.

        ``steer_deg`` deflects the wheels (+ right / - left); 0 goes straight.
        """
        v = self.ctx.default_target_speed_mps if target_speed is None else abs(float(target_speed))
        self.set_target_speed(abs(v))
        self.set_steering(steer_deg)
        for _ in range(self._ticks(duration)):
            yield self.autopilot()

    def go_forward_distance(
        self,
        target_speed: Optional[float] = 8.0,
        distance_m: float = 0.0,
        steer_deg: float = 0.0,
    ) -> Iterator[Any]:
        """Drive forward until the materialized trajectory covers ``distance_m``.

        Unlike a fixed-duration straight leg, this keeps the planned longitudinal
        clearance stable when the primitive enters the leg below target speed.
        The executor advances ``self.sim`` after every yielded tick, so the loop
        measures the same kinematic rollout that is handed to the verifier.
        """
        distance = max(0.0, float(distance_m))
        if distance <= 0.0:
            return
        v = self.ctx.default_target_speed_mps if target_speed is None else abs(float(target_speed))
        self.set_target_speed(abs(v))
        self.set_steering(steer_deg)
        start_x = float(self.sim.x)
        start_y = float(self.sim.y)
        while (
            math.hypot(float(self.sim.x) - start_x, float(self.sim.y) - start_y)
            + 1e-6
            < distance
        ):
            yield self.autopilot()

    def reverse(
        self,
        target_speed: Optional[float] = 2.0,
        duration: float = 1.5,
        steer_deg: float = 0.0,
    ) -> Iterator[Any]:
        """REVERSE (back up) at ``target_speed`` m/s magnitude for ``duration`` s.

        ``steer_deg`` deflects the wheels while reversing (+ right / - left).
        """
        v = self.ctx.reverse_speed_mps if target_speed is None else abs(float(target_speed))
        self.set_target_speed(-abs(v))
        self.set_steering(steer_deg)
        for _ in range(self._ticks(duration)):
            yield self.autopilot()

    def hold(self, duration: float = 1.0) -> Iterator[Any]:
        """Keep the current speed and steering for ``duration`` s."""
        for _ in range(self._ticks(duration)):
            yield self.autopilot()

    def wait(self, duration: float = 1.0) -> Iterator[Any]:
        """Remain stationary (speed 0, wheels straight) for ``duration`` s."""
        self.set_target_speed(0.0)
        self.set_steering(0.0)
        for _ in range(self._ticks(duration)):
            yield self.autopilot()

    # --- Extra atomic actions (forward-only; cruise / intersection driving) ---
    # These extend the recovery-biased atoms above with the longitudinal and
    # turning building blocks a VLM needs to *drive* (not just recover). All are
    # forward-only so the CARLA waypoint PIDs can track them: a reverse segment
    # is executed as forward motion by a norm-based speed tracker (see
    # control/our_pid.py), so keep reversing to the dedicated reverse atoms.
    def cruise(self, target_speed: Optional[float] = 8.0, duration: float = 3.0) -> Iterator[Any]:
        """Hold a steady FORWARD speed straight ahead for ``duration`` s.

        Like ``go_forward`` but with a longer default horizon; the plain
        "keep driving down the lane" building block.
        """
        yield from self.go_forward(target_speed=target_speed, duration=duration, steer_deg=0.0)

    def creep(self, target_speed: float = 1.5, duration: float = 1.5) -> Iterator[Any]:
        """Inch FORWARD at a low crawl speed (occluded merges, tight approaches)."""
        yield from self.go_forward(target_speed=target_speed, duration=duration, steer_deg=0.0)

    def slow_to(self, target_speed: float, duration: float = 2.0) -> Iterator[Any]:
        """Decelerate to a nonzero FORWARD ``target_speed`` and keep rolling.

        Unlike ``emergency_stop`` this eases toward a positive cruising speed
        rather than a full stop.
        """
        self.set_target_speed(abs(float(target_speed)))
        self.set_steering(0.0)
        for _ in range(self._ticks(duration)):
            yield self.autopilot()

    def coast(self, duration: float = 1.5) -> Iterator[Any]:
        """Ease off the throttle: decay toward a low crawl without braking hard."""
        crawl = min(abs(self.sim.speed), 1.0)
        self.set_target_speed(crawl)
        self.set_steering(0.0)
        for _ in range(self._ticks(duration)):
            yield self.autopilot()

    def emergency_stop(self, duration: float = 0.8) -> Iterator[Any]:
        """Brake to a full stop -- the ONE and ONLY brake atom.

        Every "stop" routes through here: it forces a hard ``HARD_DECEL_MPS2``
        (~0.8 g) deceleration for the braking horizon regardless of the sim's
        cruising ``max_decel_mps2``, so the ego snaps to a halt instead of
        coasting. There is no softer brake variant.
        """
        self.set_target_speed(0.0)
        self.set_steering(0.0)
        saved_decel = self.sim.max_decel_mps2
        self.sim.max_decel_mps2 = max(saved_decel, HARD_DECEL_MPS2)
        try:
            for _ in range(self._ticks(duration)):
                yield self.autopilot()
        finally:
            self.sim.max_decel_mps2 = saved_decel

    def turn(
        self,
        direction: str = "left",
        steer_deg: Optional[float] = DEFAULT_STEER_DEG,
        target_speed: Optional[float] = 6.0,
        duration: float = 2.5,
    ) -> Iterator[Any]:
        """Sustained FORWARD turn (held deflection) toward ``'left'``/``'right'``.

        For actually rounding a corner; contrast ``change_lane`` (steer out then
        back) and ``follow_route`` (gentle route-command nudge).
        """
        deg = self.ctx.default_steer_deg if steer_deg is None else abs(float(steer_deg))
        sign = -1.0 if str(direction).lower().startswith("l") else 1.0
        yield from self.go_forward(target_speed=target_speed, duration=duration, steer_deg=sign * deg)

    def nudge(
        self,
        direction: str = "left",
        duration: float = 1.5,
        target_speed: Optional[float] = 8.0,
    ) -> Iterator[Any]:
        """Small in-lane lateral correction: brief gentle steer, then straighten."""
        deg = 0.5 * self.ctx.default_steer_deg
        sign = -1.0 if str(direction).lower().startswith("l") else 1.0
        half = max(0.1, 0.5 * float(duration))
        yield from self.go_forward(target_speed=target_speed, duration=half, steer_deg=sign * deg)
        yield from self.go_forward(target_speed=target_speed, duration=half, steer_deg=-sign * deg)

    # --- Action combinations (composite generators) -----------------------
    def back_up_and_steer(
        self,
        reverse_speed: Optional[float] = 2.0,
        reverse_duration: float = 2.0,
        steer_deg: Optional[float] = DEFAULT_STEER_DEG,
        settle_duration: float = 1.5,
        lateral_m: Optional[float] = None,
        direction: Optional[str] = None,
    ) -> Iterator[Any]:
        """Reverse while steering, then brake to a stop.

        When ``lateral_m`` is supplied, size the steering angle so the complete
        maneuver ends approximately that far to ``direction`` (``left`` or
        ``right``).  The accepted recovery range is 0.3--1.5 m.  The legacy
        signed ``steer_deg`` interface remains available when ``lateral_m`` is
        omitted.
        """
        if lateral_m is not None:
            target_lateral = validate_recovery_lateral_distance("back", lateral_m)
            sign = -1.0 if str(direction or "left").lower().startswith("l") else 1.0
            run_speed = self.ctx.reverse_speed_mps if reverse_speed is None else abs(float(reverse_speed))
            deg = self._steer_deg_for_lateral(
                target_lateral,
                lambda sim, d: self._run_reverse_reposition(
                    sim,
                    sign,
                    d,
                    run_speed,
                    reverse_duration,
                    settle_duration,
                ),
                seed_speed=0.0,
            )
            deg *= sign
        else:
            deg = self.ctx.default_steer_deg if steer_deg is None else float(steer_deg)
        # Come to a stop before reversing: you cannot back up while still rolling
        # forward, and a reverse leg started at speed only bleeds off that speed.
        yield from self.emergency_stop(duration=self._stop_duration())
        yield from self.reverse(target_speed=reverse_speed, duration=reverse_duration, steer_deg=deg)
        yield from self.emergency_stop(duration=settle_duration)

    def three_point_turn(
        self,
        steer_deg: Optional[float] = 2.0 * DEFAULT_STEER_DEG,
        forward_speed: Optional[float] = 4.0,
        reverse_speed: Optional[float] = 2.0,
        seg_duration: float = 2.5,
    ) -> Iterator[Any]:
        """K-turn / U-turn: forward-turn, reverse-turn, forward-turn."""
        deg = (self.ctx.default_steer_deg * 2.0) if steer_deg is None else float(steer_deg)
        # Stop at each reversal point (a real 3-point turn is stop-reverse-stop):
        # otherwise the reverse leg is spent decelerating the forward-turn speed
        # and never actually backs up.
        yield from self.go_forward(target_speed=forward_speed, duration=seg_duration, steer_deg=+abs(deg))
        yield from self.emergency_stop(duration=self._stop_duration())
        yield from self.reverse(target_speed=reverse_speed, duration=seg_duration, steer_deg=-abs(deg))
        yield from self.emergency_stop(duration=self._stop_duration())
        yield from self.go_forward(target_speed=forward_speed, duration=seg_duration, steer_deg=+abs(deg))

    def back_then_bypass(
        self,
        reverse_duration: float = 2.0,
        reverse_speed: Optional[float] = 3.0,
        steer_deg: Optional[float] = 35.0,
        direction: str = "left",
        settle_duration: float = 0.5,
        forward_speed: Optional[float] = 4.0,
        bypass_duration: float = 1.5,
        straighten_duration: float = 1.0,
        lateral_m: Optional[float] = None,
    ) -> Iterator[Any]:
        """Back up straight, then drive forward around the chosen side.

        The complete maneuver is deliberately sized to the recovery provider's
        12-pose / 6-second materialization window at 0.5 s spacing:

        1. hard-stop the incoming forward motion;
        2. reverse straight to create room;
        3. stop before changing travel direction;
        4. drive forward and steer toward ``direction``;
        5. counter-steer so the final heading is approximately straight.

        ``direction='left'`` and ``'right'`` are exposed as separate recovery
        candidates (``back_bypass_left`` / ``back_bypass_right``).
        """
        sign = -1.0 if str(direction).lower().startswith("l") else 1.0
        if lateral_m is not None:
            target_lateral = validate_recovery_lateral_distance(
                "back_bypass", lateral_m
            )
            # The original 1.5 s steer-out + 1.0 s counter-steer profile is
            # well aligned up to roughly 1.5 m. Larger requested offsets need
            # more of the fixed 6 s horizon for the outward arc, so use a
            # 2.0 s + 0.5 s split and a stronger counter-steer. The stop and
            # 2 s reverse phases stay unchanged.
            if target_lateral > 1.5:
                effective_bypass_duration = 2.0
                effective_straighten_duration = 0.5
                counter_steer_ratio = 1.0
            else:
                effective_bypass_duration = bypass_duration
                effective_straighten_duration = straighten_duration
                counter_steer_ratio = 0.63
            reverse_v = self.ctx.reverse_speed_mps if reverse_speed is None else abs(float(reverse_speed))
            forward_v = self.ctx.default_target_speed_mps if forward_speed is None else abs(float(forward_speed))
            deg = self._steer_deg_for_lateral(
                target_lateral,
                lambda sim, d: self._run_back_bypass(
                    sim,
                    sign,
                    d,
                    reverse_v,
                    reverse_duration,
                    settle_duration,
                    forward_v,
                    effective_bypass_duration,
                    effective_straighten_duration,
                    counter_steer_ratio,
                ),
            )
        else:
            deg = 35.0 if steer_deg is None else abs(float(steer_deg))
            effective_bypass_duration = bypass_duration
            effective_straighten_duration = straighten_duration
            counter_steer_ratio = 0.63
        # One second at the hard 8 m/s^2 brake is enough to stop the normal
        # recovery entry range (<= 8 m/s). Giving the saved half-second to the
        # reverse leg makes the backing motion unmistakable instead of barely
        # crossing behind the start pose. At the 5 m/s visualization speed the
        # old profile reached only y=-0.75 m; this one reaches about y=-2.75 m.
        yield from self.emergency_stop(duration=1.0)
        yield from self.reverse(
            target_speed=reverse_speed,
            duration=reverse_duration,
            steer_deg=0.0,
        )
        yield from self.emergency_stop(duration=settle_duration)
        yield from self.go_forward(
            target_speed=forward_speed,
            duration=effective_bypass_duration,
            steer_deg=sign * deg,
        )
        # The return phase runs at the higher, already-built forward speed, so a
        # smaller counter-steer cancels the heading accumulated while accelerating
        # through the first phase. 0.63 gives ~zero final heading for the default
        # 1.5 s out / 1.0 s return profile in VehicleSim.
        yield from self.go_forward(
            target_speed=forward_speed,
            duration=effective_straighten_duration,
            steer_deg=-sign * counter_steer_ratio * deg,
        )

    def creep_back(
        self,
        target_speed: Optional[float] = 1.2,
        duration: float = 3.0,
    ) -> Iterator[Any]:
        """Stop, then back STRAIGHT up -- heading-preserving space-making.

        The steered backs (back_left/right) rotate the nose while retreating;
        this is the mirror of creep: create room without spending pose. The
        1.0 s leading stop halts the <=8 m/s recovery entry range so the leg
        nets backward even when entered rolling (same rationale as
        back_then_bypass's stop phase).
        """
        v = 1.2 if target_speed is None else abs(float(target_speed))
        yield from self.emergency_stop(duration=1.0)
        yield from self.reverse(
            target_speed=v,
            duration=max(0.5, float(duration) - 1.0),
            steer_deg=0.0,
        )

    def swerve_and_stop(
        self,
        steer_deg: Optional[float] = DEFAULT_STEER_DEG,
        swerve_duration: float = 1.2,
        settle_duration: float = 1.5,
        target_speed: Optional[float] = 8.0,
        lateral_m: Optional[float] = None,
        direction: Optional[str] = None,
    ) -> Iterator[Any]:
        """Steer aside briefly to dodge, then brake to a stop.

        Pass ``lateral_m`` (with ``direction`` ``'left'``/``'right'``) to target a
        fixed dodge offset regardless of speed: the steer angle is sized from the
        current speed so a fixed-angle swerve does not scale with v^2 and veer off
        the road at speed. ``steer_deg`` (signed: + right / - left) is used only
        when ``lateral_m`` is ``None``.
        """
        if lateral_m is not None:
            lateral_m = validate_recovery_lateral_distance("swerve", lateral_m)
            sign = -1.0 if str(direction or "right").lower().startswith("l") else 1.0
            # Hold the current speed through the dodge (then brake) so the lateral
            # shift comes from steering, not from accelerating through the swerve.
            run_speed = max(0.0, abs(self.sim.speed))
            deg = self._steer_deg_for_lateral(
                lateral_m,
                lambda sim, d: self._run_swerve(sim, sign, d, run_speed, swerve_duration, settle_duration),
            )
            yield from self.go_forward(target_speed=run_speed, duration=swerve_duration, steer_deg=sign * deg)
            yield from self.emergency_stop(duration=settle_duration)
            return
        deg = self.ctx.default_steer_deg if steer_deg is None else float(steer_deg)
        yield from self.go_forward(target_speed=target_speed, duration=swerve_duration, steer_deg=deg)
        yield from self.emergency_stop(duration=settle_duration)

    def _fresh_sim(self, speed: Optional[float] = None) -> VehicleSim:
        """A throwaway sim seeded with the *live* ego motion state.

        Used to pre-simulate a maneuver (e.g. to size a steer angle) without
        perturbing the real sim the executor is stepping. Pass ``speed`` to seed a
        specific *steady* speed (speed == target, so the pre-sim does not ease):
        used when the real maneuver is preceded by a lead-in that brings the ego up
        to that speed, so both must model the same constant-speed profile.
        """
        s = self.sim.speed if speed is None else float(speed)
        t = self.sim.target_speed if speed is None else float(speed)
        return VehicleSim(
            dt=self.sim.dt,
            max_accel_mps2=self.sim.max_accel_mps2,
            max_decel_mps2=self.sim.max_decel_mps2,
            speed=s,
            target_speed=t,
            wheelbase_m=self.sim.wheelbase_m,
        )

    def _rapid_lane_shift_steers(
        self,
        lateral_m: float,
        target_speed: float,
        outward_duration: float,
        counter_duration: Optional[float] = None,
    ) -> tuple[float, float]:
        """Size an accelerating S-curve and return ``(out_deg, return_deg)``.

        During acceleration the second phase travels farther, so mirroring the
        same steering angle would over-correct. We first measure the longitudinal
        distance covered by each phase, analytically choose the smaller return
        angle that cancels yaw, then binary-search the outward angle for the
        requested lane-width displacement. Both ordinary lane changes and the
        adjacent-lane bypass use this immediate-steer profile.
        """
        outward_ticks = self._ticks(outward_duration)
        counter_ticks = self._ticks(
            outward_duration if counter_duration is None else counter_duration
        )
        v = abs(float(target_speed))
        distance_probe = self._fresh_sim()
        distance_probe.target_speed = v
        distance_probe.steer_rad = 0.0
        for _ in range(outward_ticks):
            distance_probe.step()
        first_distance = abs(float(distance_probe.y))
        first_end_y = float(distance_probe.y)
        for _ in range(counter_ticks):
            distance_probe.step()
        second_distance = abs(float(distance_probe.y) - first_end_y)
        distance_ratio = first_distance / max(second_distance, 1e-6)
        cap_deg = math.degrees(MAX_STEER_RAD)

        def return_deg_for(out_deg: float) -> float:
            # Bicycle yaw is proportional to distance * tan(steer).  Scaling in
            # tangent space (rather than degrees) makes the two phase yaw changes
            # cancel even when the vehicle accelerates between them.
            value = math.degrees(
                math.atan(math.tan(math.radians(out_deg)) * distance_ratio)
            )
            return max(0.0, min(value, cap_deg))

        def displacement(out_deg: float) -> float:
            sim = self._fresh_sim()
            sim.target_speed = v
            sim.steer_rad = math.radians(out_deg)
            for _ in range(outward_ticks):
                sim.step()
            sim.steer_rad = math.radians(-return_deg_for(out_deg))
            for _ in range(counter_ticks):
                sim.step()
            return abs(float(sim.x))

        target = abs(float(lateral_m))
        if displacement(cap_deg) <= target:
            outward = cap_deg
        else:
            lo, hi = 0.0, cap_deg
            for _ in range(24):
                mid = 0.5 * (lo + hi)
                if displacement(mid) < target:
                    lo = mid
                else:
                    hi = mid
            outward = 0.5 * (lo + hi)
        return outward, return_deg_for(outward)

    def _rapid_lane_shift(
        self,
        direction: str,
        lateral_m: float,
        target_speed: float,
        outward_duration: float,
        counter_duration: Optional[float] = None,
    ) -> Iterator[Any]:
        """Execute an immediate accelerating one-lane S-curve."""
        sign = -1.0 if str(direction).lower().startswith("l") else 1.0
        outward_deg, return_deg = self._rapid_lane_shift_steers(
            lateral_m=lateral_m,
            target_speed=target_speed,
            outward_duration=outward_duration,
            counter_duration=counter_duration,
        )
        yield from self.go_forward(
            target_speed=target_speed,
            duration=outward_duration,
            steer_deg=sign * outward_deg,
        )
        yield from self.go_forward(
            target_speed=target_speed,
            duration=(
                outward_duration
                if counter_duration is None
                else counter_duration
            ),
            steer_deg=-sign * return_deg,
        )

    def _run_swerve(
        self, sim: VehicleSim, sign: float, deg: float, run_speed: float,
        swerve_duration: float, settle_duration: float,
    ) -> float:
        """Play swerve-aside-then-brake on ``sim``; return net lateral.

        Mirrors :meth:`swerve_and_stop`: hold ``sign*deg`` at ``run_speed`` for
        ``swerve_duration``, then brake to a stop (wheels straight). The ego keeps
        drifting laterally while braking, so the settle phase is included in the
        measured shift.
        """
        v = abs(float(run_speed))
        sim.target_speed = v
        sim.steer_rad = math.radians(sign * deg)
        for _ in range(self._ticks(swerve_duration)):
            sim.step()
        sim.target_speed = 0.0
        sim.steer_rad = 0.0
        # Mirror emergency_stop's HARD decel during the settle brake. Using the
        # sim's gentler cruising max_decel here made the pre-sim coast (and keep
        # drifting laterally) far longer than the real hard stop, overestimating the
        # net lateral -> the sized angle undershot (~1.35 m for a 2 m request). Match
        # the real decel so the predicted shift equals what emergency_stop delivers.
        saved_decel = sim.max_decel_mps2
        sim.max_decel_mps2 = max(saved_decel, HARD_DECEL_MPS2)
        for _ in range(self._ticks(settle_duration)):
            sim.step()
        sim.max_decel_mps2 = saved_decel
        return sim.x

    def _hard_stop_sim(self, sim: VehicleSim, duration: float) -> None:
        """Apply the same hard-stop profile as :meth:`emergency_stop` in a probe."""
        sim.target_speed = 0.0
        sim.steer_rad = 0.0
        saved_decel = sim.max_decel_mps2
        sim.max_decel_mps2 = max(saved_decel, HARD_DECEL_MPS2)
        for _ in range(self._ticks(duration)):
            sim.step()
        sim.max_decel_mps2 = saved_decel

    def _run_reverse_reposition(
        self,
        sim: VehicleSim,
        sign: float,
        deg: float,
        reverse_speed: float,
        reverse_duration: float,
        settle_duration: float,
    ) -> float:
        """Probe a parameterized back-left/right maneuver and return net x."""
        sim.target_speed = -abs(float(reverse_speed))
        sim.steer_rad = math.radians(sign * abs(float(deg)))
        for _ in range(self._ticks(reverse_duration)):
            sim.step()
        self._hard_stop_sim(sim, settle_duration)
        return sim.x

    def _run_back_bypass(
        self,
        sim: VehicleSim,
        sign: float,
        deg: float,
        reverse_speed: float,
        reverse_duration: float,
        settle_duration: float,
        forward_speed: float,
        bypass_duration: float,
        straighten_duration: float,
        counter_steer_ratio: float,
    ) -> float:
        """Probe the complete back-then-bypass profile and return net x."""
        self._hard_stop_sim(sim, 1.0)
        sim.target_speed = -abs(float(reverse_speed))
        sim.steer_rad = 0.0
        for _ in range(self._ticks(reverse_duration)):
            sim.step()
        self._hard_stop_sim(sim, settle_duration)
        sim.target_speed = abs(float(forward_speed))
        sim.steer_rad = math.radians(sign * abs(float(deg)))
        for _ in range(self._ticks(bypass_duration)):
            sim.step()
        sim.steer_rad = math.radians(
            -sign * float(counter_steer_ratio) * abs(float(deg))
        )
        for _ in range(self._ticks(straighten_duration)):
            sim.step()
        return sim.x

    def _steer_deg_for_lateral(
        self, lateral_m: float, run_fn: "Callable[[VehicleSim, float], float]", ref_deg: float = 4.0,
        seed_speed: Optional[float] = None,
    ) -> float:
        """Steer deflection (deg) whose maneuver nets ~``lateral_m`` at the CURRENT speed.

        ``run_fn(sim, deg) -> net_lateral`` plays the maneuver at deflection ``deg``
        on a throwaway sim seeded with the live speed. A fixed-angle steer shifts
        the ego by ~``(v^2/W)*delta*T^2`` -- it grows with speed, so a constant
        angle undershoots the target when slow and overshoots (off the road) when
        fast. The net shift is ~linear in the angle for a fixed speed profile, so
        we pre-simulate one reference angle (accel easing included) and scale to
        hit ``lateral_m``. Clamped to the model's max steer -- at very low speed the
        target may be steer-limited within the horizon and the recovery loop simply
        re-issues the maneuver to finish.
        """
        x_ref = abs(run_fn(self._fresh_sim(seed_speed), ref_deg))
        cap_deg = math.degrees(MAX_STEER_RAD)
        if x_ref < 1e-6:
            return cap_deg
        return max(0.0, min(ref_deg * abs(float(lateral_m)) / x_ref, cap_deg))

    def change_lane(
        self,
        direction: str = "left",
        target_speed: Optional[float] = 8.0,
        steer_deg: Optional[float] = DEFAULT_STEER_DEG,
        seg_duration: float = 1.5,
        lateral_m: Optional[float] = None,
        straighten_duration: float = 0.0,
        total_duration: Optional[float] = None,
    ) -> Iterator[Any]:
        """Shift one lane with an immediate accelerating S-curve.

        Pass ``lateral_m`` (e.g. ~3.5 for one lane) to target a fixed lateral
        offset. Steering starts on the first generated waypoint; there is no
        straight acceleration lead-in. Because the ego accelerates throughout the
        maneuver, the counter-steer is sized in tangent space from the distance
        covered by each phase instead of simply mirroring the outward angle. This
        keeps the final heading straight while allowing a stopped/crawling ego to
        use the full 35-degree steering limit immediately when needed.

        ``steer_deg`` is used only when ``lateral_m`` is ``None`` (legacy
        fixed-angle behavior). ``straighten_duration`` or spare
        ``total_duration`` may append a centered-wheel exit leg, but the recovery
        lane primitives intentionally use two 1.5 s phases and finish in 3 s.
        """
        sign = -1.0 if str(direction).lower().startswith("l") else 1.0
        exit_speed = (
            self.ctx.default_target_speed_mps
            if target_speed is None
            else abs(float(target_speed))
        )
        if lateral_m is not None:
            yield from self._rapid_lane_shift(
                direction=direction,
                lateral_m=lateral_m,
                target_speed=exit_speed,
                outward_duration=seg_duration,
                counter_duration=seg_duration,
            )
        else:
            deg = self.ctx.default_steer_deg if steer_deg is None else abs(float(steer_deg))
            yield from self.go_forward(
                target_speed=exit_speed,
                duration=seg_duration,
                steer_deg=sign * deg,
            )
            yield from self.go_forward(
                target_speed=exit_speed,
                duration=seg_duration,
                steer_deg=-sign * deg,
            )
        exit_duration = max(0.0, float(straighten_duration))
        if total_duration is not None:
            exit_duration = max(
                exit_duration,
                float(total_duration) - 2.0 * float(seg_duration),
            )
        if exit_duration > 0.0:
            yield from self.go_forward(
                target_speed=exit_speed,
                duration=exit_duration,
                steer_deg=0.0,
            )

    def follow_route(self, target_speed: Optional[float] = 8.0, duration: float = 3.0) -> Iterator[Any]:
        """Steer gently along the planned route (left / right / straight)."""
        if self.ctx.route_command == "left":
            yield from self.go_forward(target_speed=target_speed, duration=duration, steer_deg=-self.ctx.default_steer_deg)
        elif self.ctx.route_command == "right":
            yield from self.go_forward(target_speed=target_speed, duration=duration, steer_deg=+self.ctx.default_steer_deg)
        else:
            yield from self.go_forward(target_speed=target_speed, duration=duration, steer_deg=0.0)

    def u_turn(
        self,
        direction: str = "left",
        target_speed: Optional[float] = 6.0,
        steer_deg: Optional[float] = 34.0,
        duration: float = 0.0,
    ) -> Iterator[Any]:
        """Continuous 180-degree turn in one FORWARD sweep (needs open road).

        Steers hard toward ``direction`` and drives forward long enough to
        reverse heading. Use ``three_point_turn`` instead when boxed in (that one
        reverses). Being forward-only, this one is trackable by the CARLA PIDs.

        When ``duration`` is 0 (the default) it is sized from the bicycle model so
        the sweep is ~180 deg at the chosen speed and deflection; pass a positive
        value to override.
        """
        deg = 34.0 if steer_deg is None else abs(float(steer_deg))
        sign = -1.0 if str(direction).lower().startswith("l") else 1.0
        v = self.ctx.default_target_speed_mps if target_speed is None else abs(float(target_speed))
        if duration is None or duration <= 0.0:
            # Bicycle yaw rate = (v / L) * tan(delta); time to sweep pi radians.
            yaw_rate = (max(v, 0.5) / max(self.sim.wheelbase_m, 1e-3)) * math.tan(math.radians(deg))
            duration = math.pi / max(yaw_rate, 1e-3)
        yield from self.go_forward(target_speed=v, duration=duration, steer_deg=sign * deg)

    def stop_and_go(
        self,
        stop_duration: float = 1.5,
        wait_duration: float = 1.0,
        go_speed: Optional[float] = 8.0,
        go_duration: float = 3.0,
    ) -> Iterator[Any]:
        """Brake to a stop, wait, then accelerate away (stop signs / red lights)."""
        yield from self.emergency_stop(duration=stop_duration)
        yield from self.wait(duration=wait_duration)
        yield from self.go_forward(target_speed=go_speed, duration=go_duration, steer_deg=0.0)

    def yield_then_go(
        self,
        slow_speed: float = 1.5,
        slow_duration: float = 1.5,
        go_speed: Optional[float] = 8.0,
        go_duration: float = 3.0,
    ) -> Iterator[Any]:
        """Slow to a crawl (as if yielding), then accelerate once clear."""
        yield from self.slow_to(target_speed=slow_speed, duration=slow_duration)
        yield from self.go_forward(target_speed=go_speed, duration=go_duration, steer_deg=0.0)

    def overtake(
        self,
        direction: str = "left",
        target_speed: Optional[float] = 8.0,
        steer_deg: Optional[float] = DEFAULT_STEER_DEG,
        seg_duration: float = 2.0,
        pass_duration: float = 3.0,
    ) -> Iterator[Any]:
        """Forward-only bypass: change lane out, drive past, change lane back.

        The forward-only analogue of ``back_then_bypass`` (which reverses first).
        Built from two speed-sized ``change_lane`` calls (out to the adjacent lane,
        then back) with a straight pass between them, so it nets ~0 lateral and is
        handed back pointing straight ahead. A fixed ``steer_deg`` was geometrically
        broken here: at cruise a held 15 deg swings the heading ~90 deg per 2 s leg
        (verified: ~24 m off-lane, not a lane change), so the deflection is now sized
        from the current speed to hit one lane width. ``steer_deg`` is accepted for
        back-compat but ignored (kept so existing / VLM-generated calls don't break).
        """
        opp = "right" if str(direction).lower().startswith("l") else "left"
        yield from self.change_lane(direction=direction, target_speed=target_speed,
                                    lateral_m=3.5, seg_duration=seg_duration, straighten_duration=0.5)
        yield from self.go_forward(target_speed=target_speed, duration=pass_duration, steer_deg=0.0)
        yield from self.change_lane(direction=opp, target_speed=target_speed,
                                    lateral_m=3.5, seg_duration=seg_duration, straighten_duration=0.5)

    def lane_bypass(
        self,
        direction: str = "left",
        target_speed: float = LANE_BYPASS_TARGET_SPEED_MPS,
        lane_width_m: float = LANE_BYPASS_LANE_WIDTH_M,
        lane_change_duration: float = LANE_BYPASS_LANE_CHANGE_DURATION_SEC,
        counter_steer_duration: float = LANE_BYPASS_COUNTER_STEER_DURATION_SEC,
        return_phase_duration: float = LANE_BYPASS_RETURN_PHASE_DURATION_SEC,
        pass_duration: float = LANE_BYPASS_PASS_DURATION_SEC,
        obstacle_clearance_m: Optional[float] = None,
        settle_duration: float = LANE_BYPASS_SETTLE_DURATION_SEC,
        accel_mps2: float = LANE_BYPASS_ACCEL_MPS2,
        entry_speed_mps: float = LANE_BYPASS_ENTRY_SPEED_MPS,
    ) -> Iterator[Any]:
        """Rapidly borrow an adjacent driving lane, clear an obstacle, and return.

        This is the short-exposure counterpart to :meth:`overtake`.  It boosts
        the rollout acceleration, accelerates through two speed-aware S-curves,
        spends only ``pass_duration`` fully in the adjacent lane, and finishes
        centered on the original lane with zero steering.  It is intentionally
        aggressive and should only be selected after the target corridor has
        passed topology, collision, and TTC verification.

        ``direction`` is the side of the borrowed lane relative to the ego.
        The target lane may carry same-direction or opposing traffic. The
        default 10 m/s target
        is about 36 km/h; at the recovery planner's 0.5 s waypoint spacing the
        borrowed-lane portion takes about 2.5 s.  When the measured safe gap is
        still approaching, the generated profile waits stationary.  As soon as
        it moves, it steers into the adjacent lane while accelerating.
        """
        is_left = str(direction).lower().startswith("l")
        allowed = (
            self.ctx.lane_bypass_left_available
            if is_left
            else self.ctx.lane_bypass_right_available
        )
        if allowed is False:
            raise ValueError(
                f"lane_bypass direction={direction!r} is not available "
                "for the current measured target-lane traffic gap"
            )
        v = abs(float(target_speed))
        lateral = abs(float(lane_width_m))
        phase_duration = max(float(lane_change_duration), self.sim.dt)
        counter_duration = max(float(counter_steer_duration), self.sim.dt)
        return_duration = max(float(return_phase_duration), self.sim.dt)
        pass_duration = max(0.0, float(pass_duration))
        clearance_distance = max(
            0.0,
            float(
                self.ctx.lane_bypass_obstacle_clearance_m
                if obstacle_clearance_m is None
                else obstacle_clearance_m
            ),
        )
        settle_duration = max(0.0, float(settle_duration))
        opp = "right" if is_left else "left"
        entry_delay = max(
            0.0,
            float(
                self.ctx.lane_bypass_left_entry_delay_sec
                if is_left
                else self.ctx.lane_bypass_right_entry_delay_sec
            ),
        )

        saved_accel = self.sim.max_accel_mps2
        self.sim.max_accel_mps2 = max(saved_accel, abs(float(accel_mps2)))
        try:
            # Preserve the public parameter for generated-code compatibility,
            # but do not spend distance reaching it before steering.  A traffic-
            # scheduled delay is stationary; the first moving tick is already
            # part of the outward S-curve.
            _ = entry_speed_mps
            if entry_delay > 0.0:
                yield from self.wait(duration=entry_delay)
            bypass_start_x = float(self.sim.x)
            bypass_start_y = float(self.sim.y)
            yield from self._rapid_lane_shift(
                direction=direction,
                lateral_m=lateral,
                target_speed=v,
                outward_duration=phase_duration,
                counter_duration=counter_duration,
            )
            if pass_duration > 0.0:
                yield from self.go_forward(
                    target_speed=v,
                    duration=pass_duration,
                    steer_deg=0.0,
                )
            # The configured clearance is the obstacle's total longitudinal
            # extent from the start of the bypass, not an extra straight leg
            # after reaching the borrowed lane.  A short obstacle is normally
            # already cleared by the outward S-curve itself.
            outward_progress = math.hypot(
                float(self.sim.x) - bypass_start_x,
                float(self.sim.y) - bypass_start_y,
            )
            remaining_clearance = max(0.0, clearance_distance - outward_progress)
            if remaining_clearance > 0.0:
                yield from self.go_forward_distance(
                    target_speed=v,
                    distance_m=remaining_clearance,
                    steer_deg=0.0,
                )
            yield from self._rapid_lane_shift(
                direction=opp,
                lateral_m=lateral,
                target_speed=v,
                outward_duration=return_duration,
                counter_duration=return_duration,
            )
            if settle_duration > 0.0:
                yield from self.go_forward(
                    target_speed=v,
                    duration=settle_duration,
                    steer_deg=0.0,
                )
        finally:
            self.sim.max_accel_mps2 = saved_accel

    def turn_at_intersection(
        self,
        direction: str = "left",
        approach_speed: float = 3.0,
        turn_speed: Optional[float] = 5.0,
        steer_deg: Optional[float] = 1.5 * DEFAULT_STEER_DEG,
        turn_duration: float = 3.0,
    ) -> Iterator[Any]:
        """Slow on approach, execute a sustained turn, then straighten out."""
        deg = (self.ctx.default_steer_deg * 1.5) if steer_deg is None else abs(float(steer_deg))
        sign = -1.0 if str(direction).lower().startswith("l") else 1.0
        yield from self.slow_to(target_speed=approach_speed, duration=1.0)
        yield from self.go_forward(target_speed=turn_speed, duration=turn_duration, steer_deg=sign * deg)
        yield from self.go_forward(target_speed=turn_speed, duration=1.0, steer_deg=0.0)

    def pull_over(
        self,
        direction: str = "right",
        target_speed: Optional[float] = 8.0,
        steer_deg: Optional[float] = DEFAULT_STEER_DEG,
        seg_duration: float = 2.0,
        settle_duration: float = 2.0,
    ) -> Iterator[Any]:
        """Change lane toward the shoulder, then brake to a stop.

        The shoulder shift is sized from the current speed (via ``change_lane``) to
        land ~one lane over regardless of speed. A fixed ``steer_deg`` scaled the
        offset with v^2 (verified: ~14.8 m of lateral at 5 m/s -- off the road, not a
        pull-over), so ``steer_deg`` is accepted for back-compat but ignored.
        """
        yield from self.change_lane(direction=direction, target_speed=target_speed,
                                    lateral_m=3.0, seg_duration=seg_duration, straighten_duration=0.0)
        yield from self.emergency_stop(duration=settle_duration)

    # # --- Perception APIs --------------------------------------------------
    # def get_ego_speed(self) -> float:
    #     return float(self.ctx.ego_speed_mps)

    # def detect_front_vehicle(self) -> bool:
    #     return self.ctx.front_vehicle_distance_m is not None

    # def distance_to_front_vehicle(self) -> float:
    #     d = self.ctx.front_vehicle_distance_m
    #     return float(d) if d is not None else 1e9


# API names exposed to VLM-generated code (kept in sync with apis_doc.py).
API_NAMES = (
    # low-level setters
    "set_target_speed",
    "set_steering",
    "autopilot",
    # basic atomic actions
    "go_forward",
    "reverse",
    "hold",
    "wait",
    # extra atomic actions (forward-only cruise / intersection driving)
    "cruise",
    "creep",
    "slow_to",
    "coast",
    "emergency_stop",
    "turn",
    "nudge",
    # action combinations
    "back_up_and_steer",
    "three_point_turn",
    "back_then_bypass",
    "creep_back",
    "swerve_and_stop",
    "change_lane",
    "follow_route",
    "u_turn",
    "stop_and_go",
    "yield_then_go",
    "overtake",
    "lane_bypass",
    "turn_at_intersection",
    "pull_over",
    # # perception
    # "get_ego_speed",
    # "detect_front_vehicle",
    # "distance_to_front_vehicle",
)


def build_api_namespace(runtime: PrimitiveRuntime) -> Dict[str, Callable[..., Any]]:
    """Bind the primitive APIs to ``runtime`` for restricted execution."""
    return {name: getattr(runtime, name) for name in API_NAMES}


# Standalone maneuvers for the no-VLM (random) path. Each is plain source run
# through the same executor as VLM output; the executor's step cap stops it, so
# these are safe. They ``yield from`` the parameterized atoms / combinations.
def _menu_entry(fn_name: str, call: str) -> str:
    return f"def {fn_name}():\n    yield from {call}\n"


PRIMITIVE_MENU = [
    ("emergency_stop", _menu_entry("come_to_stop", "emergency_stop(duration=2.0)")),
    ("reverse", _menu_entry("back_up", "reverse(target_speed=2.0, duration=2.0)")),
    ("back_up_and_steer", _menu_entry("back_up_and_steer_left", "back_up_and_steer(reverse_duration=2.0, steer_deg=-15.0)")),
    # Speed-aware, matching the CANDIDATE_LADDER entries: a fixed steer angle
    # integrates to a v^2-scaled shift (change_lane(direction=...) alone held the
    # 15 deg default -> ~15 m off the road at cruise). lateral_m sizes the angle
    # from the current speed so the dodge / lane change lands a fixed offset.
    ("swerve_and_stop", _menu_entry("swerve_right_and_stop", "swerve_and_stop(direction='right', lateral_m=2.0, swerve_duration=1.2)")),
    ("change_lane", _menu_entry("change_lane_left", "change_lane(direction='left', lateral_m=3.5, seg_duration=1.5, total_duration=3.0)")),
    ("follow_route", _menu_entry("follow_the_route", "follow_route(duration=3.0)")),
    ("hold", _menu_entry("hold_then_stop", "hold(duration=1.0)")),
]


# Concrete recovery maneuvers, one per (primitive, direction) instance, keyed by
# a stable id. Ordered safest -> most aggressive; consumers either take the
# ladder head (argmax recovery), reorder it from failure memory
# (planner._ordered_candidates), or hand the whole set to the VLM to re-rank.
CANDIDATE_LADDER = [
    ("brake",       "def m():\n    yield from emergency_stop(duration=2.0)\n"),
    ("wait",        "def m():\n    yield from wait(duration=2.0)\n"),
    # Inchworm escape for "blocked by something a few meters ahead" (standing
    # pedestrian / construction chokepoints): 1.2 m/s for 2 s = ~2.4 m, sized to
    # the NC gate horizon (2 s) so the WHOLE materialized plan is covered by the
    # collision check -- the hold exhausts its progress and re-verifies before
    # the next inch. Passes the verifier while the obstacle is > ~3 m out and
    # goes back to a clean stop once it is not, where a lane change / bypass
    # would either fail NC outright or lunge. Without it those scenes deadlock
    # into brake-forever -> vehicle_blocked (eps 1011/1017).
    ("creep",       "def m():\n    yield from creep(target_speed=1.2, duration=2.0)\n"),
    # Backward mirror of creep: ~2.4 m straight back, heading preserved.
    # Fully inside the NC gate horizon, so each inchworm is re-verified
    # before the next; unlike back_left/right it does not rotate the nose.
    ("creep_back",  "def m():\n    yield from creep_back(target_speed=1.2, duration=3.0)\n"),
    # Target a fixed ~2 m dodge (with the brake) rather than a fixed steer angle:
    # a fixed steer_deg=8.0 swerve scaled with v^2 (~0.8 m at 2 m/s to ~9 m off the
    # road at 12 m/s). lateral_m sizes the angle from the current speed so the dodge
    # is consistent and stays near the corridor. Smaller than the ~3.5 m lane change.
    ("swerve_left", "def m():\n    yield from swerve_and_stop(direction='left', lateral_m=2.0, swerve_duration=1.2)\n"),
    ("swerve_right","def m():\n    yield from swerve_and_stop(direction='right', lateral_m=2.0, swerve_duration=1.2)\n"),
    ("back_left",   "def m():\n    yield from back_up_and_steer(reverse_duration=2.0, direction='left', lateral_m=0.75)\n"),
    ("back_right",  "def m():\n    yield from back_up_and_steer(reverse_duration=2.0, direction='right', lateral_m=0.75)\n"),
    ("back_bypass_left",  "def m():\n    yield from back_then_bypass(direction='left', lateral_m=1.5)\n"),
    ("back_bypass_right", "def m():\n    yield from back_then_bypass(direction='right', lateral_m=1.5)\n"),
    ("three_point", "def m():\n    yield from three_point_turn()\n"),
    # Target a fixed one-lane (~3.5 m) offset rather than a fixed steer angle:
    # change_lane sizes the deflection from the current speed so the ego lands on
    # the adjacent lane at any speed. A fixed angle scaled with v^2 -- undershoot
    # when slow, off-the-road overshoot when fast (steer_deg=6.0 hit ~9.6 m).
    ("lane_left",   "def m():\n    yield from change_lane(direction='left', lateral_m=3.5, seg_duration=1.5, total_duration=3.0)\n"),
    ("lane_right",  "def m():\n    yield from change_lane(direction='right', lateral_m=3.5, seg_duration=1.5, total_duration=3.0)\n"),
    # Aggressive last-resort forward bypass for a static blockage. The verifier
    # must confirm that the borrowed adjacent corridor is safe whether its flow
    # is same-direction or opposing; the maneuver then returns to the original
    # lane in one fixed profile.
    ("lane_bypass_left",  "def m():\n    yield from lane_bypass(direction='left')\n"),
    ("lane_bypass_right", "def m():\n    yield from lane_bypass(direction='right')\n"),
]

_SWERVE_KEYS = {"swerve_left", "swerve_right"}

# Below this speed a swerve cannot dodge anything, so it is dropped from the
# ladder. swerve_and_stop deliberately holds the *current* speed through the
# dodge (it must not accelerate into the obstacle it is avoiding), so once an
# earlier rung -- typically brake -- has parked the ego, the rollout integrates
# to a zero-length trajectory. Downstream that is worse than useless: the PID
# reads target_speed 0 and never opens the throttle, while the steer angle is an
# atan2 over ~1e-4 m waypoints, i.e. pure float noise that saturates to +-1 and
# flips sign every few ticks. Dropping the candidate lets the ladder fall through
# to a reverse maneuver instead of committing its full hold window to a no-op.
SWERVE_MIN_SPEED_MPS = 1.0


def ladder_for_speed(ego_speed_mps: float) -> List[tuple]:
    """:data:`CANDIDATE_LADDER` minus maneuvers that cannot work at this speed."""
    try:
        speed = abs(float(ego_speed_mps))
    except (TypeError, ValueError):
        return list(CANDIDATE_LADDER)
    if speed >= SWERVE_MIN_SPEED_MPS:
        return list(CANDIDATE_LADDER)
    return [(k, c) for k, c in CANDIDATE_LADDER if k not in _SWERVE_KEYS]
