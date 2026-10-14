"""TTC (time-to-collision) trajectory verifier for the CARLA path.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from agenticdriving.adapters.carla.geometry import info_to_pose, world_points_to_current_local
from agenticdriving.verifiers.base import BaseVerifier, VerificationResult

LOG = logging.getLogger(__name__)

DEFAULT_PROBE_HORIZONS_SEC = (0.5, 1.0)

# Fail2Drive CollisionTest at-fault gate: contacts with ego speed below this
ATFAULT_EGO_SPEED_EPSILON_MPS = 0.01

# Fail2Drive multiplicative collision penalties per actor type
COLLISION_PENALTIES: Dict[str, float] = {
    "pedestrian": 0.50,
    "vehicle": 0.60,
    "static": 0.65,
}

# VRU classes for the PDMS vru_ttc_veto (see PDMSVerifier). The CARLA adapter
# maps every ``walker.*`` blueprint to "pedestrian", so animals are covered.
VRU_OBJECT_TYPES = frozenset({"pedestrian"})

# A measured-near-zero pedestrian stays at its measured pose for hard NC/TTC
# checks, regardless of the HD-map lane label. Lane membership and actor
# heading are too noisy to justify inventing motion in a hard safety gate: a
# stationary construction worker labelled ``on_driving_lane=True`` otherwise
# gets projected several metres along its facing direction and can veto a path
# that never intersects its real box. Once measured motion exceeds this
# threshold, the configured walking-speed floor applies again.
STATIONARY_WALKER_SPEED_MPS = 0.3

# A vehicle below this measured speed is treated as parked/stopped for the
# optional lateral door-envelope margin.  The margin changes only the current
# measured bbox; it never invents longitudinal or lateral motion.
STATIONARY_VEHICLE_SPEED_MPS = 0.3


def gating_vru_conflict(collisions: Sequence[Dict[str, Any]]) -> bool:
    """True iff any in-horizon collision in ``collisions`` involves a VRU.

    Records outside the veto horizon are advisories (``beyond_gate_horizon``)
    and never gate, so they are skipped here too.
    """
    return any(
        str(collision.get("object_type")) in VRU_OBJECT_TYPES
        and not collision.get("beyond_gate_horizon", False)
        for collision in collisions
    )


class Obstacle(NamedTuple):
    """A privileged agent mapped into the ego-local (right, forward) frame."""

    center: np.ndarray
    direction: np.ndarray
    half_len: float
    half_width: float
    velocity: np.ndarray
    agent_id: str
    object_type: str
    pos_world: List[float]
    #: Optional held-action kinematic-bicycle rollout (agent_forecast_mode
    #: "bicycle"): positions [K, 2] / heading unit vectors [K, 2] in the ego
    #: local frame at times k * future_dt. None -> constant-velocity fallback.
    future_pos: Optional[np.ndarray] = None
    future_dir: Optional[np.ndarray] = None
    future_dt: float = 0.05


# Kinematic bicycle model, World-on-Rails-tuned CARLA constants — the same
# parameter set as carla_garage/SimLingo team_code (kinematic_bicycle_model.py,
# agent_simlingo.py bicycle_model_forward). Used only for OTHER-vehicle
# forecasting under agent_forecast_mode="bicycle"; the ego plan is never
# rolled through this (the plan polyline + retiming stays the ego forecast).
_BICYCLE_FRONT_WB = -0.090769015
_BICYCLE_REAR_WB = 1.4178275
_BICYCLE_STEER_GAIN = 0.36848336
_BICYCLE_BRAKE_ACCEL = -4.952399
_BICYCLE_THROT_ACCEL = 0.5633837


def bicycle_forecast_world(
    pos_xy: np.ndarray,
    heading_rad: float,
    speed_mps: float,
    control: Sequence[float],
    horizon_sec: float,
    dt: float = 0.05,
):
    """Held-action bicycle rollout in the raw CARLA world frame.

    ``control`` is the vehicle's CURRENT (steer, throttle, brake) from the
    privileged snapshot, held constant over the horizon — the PDM-Lite
    autopilot's forecast_other_vehicles assumption. World-frame on purpose:
    the WoR math is CARLA-native (left-handed frame, steer sign), so rolling
    here and converting to ego-local afterwards reuses the exact transforms
    the constant-velocity path already trusts.

    Returns ``(positions [K, 2], headings [K])`` with row k at t = k * dt.
    """
    steer = float(control[0])
    throttle = float(control[1])
    brake = float(control[2]) >= 0.5
    n = max(2, int(np.ceil(max(0.0, float(horizon_sec)) / dt)) + 1)
    xs = np.empty(n)
    ys = np.empty(n)
    hs = np.empty(n)
    x, y = float(pos_xy[0]), float(pos_xy[1])
    h = float(heading_rad)
    v = max(0.0, float(speed_mps))
    wheel = _BICYCLE_STEER_GAIN * steer
    beta = np.arctan(_BICYCLE_REAR_WB / (_BICYCLE_FRONT_WB + _BICYCLE_REAR_WB) * np.tan(wheel))
    accel = _BICYCLE_BRAKE_ACCEL if brake else _BICYCLE_THROT_ACCEL * throttle
    for i in range(n):
        xs[i], ys[i], hs[i] = x, y, h
        x += v * np.cos(h + beta) * dt
        y += v * np.sin(h + beta) * dt
        h += v / _BICYCLE_REAR_WB * np.sin(beta) * dt
        v = max(0.0, v + accel * dt)
    return np.column_stack([xs, ys]), hs


def _obb_overlap(
    c1: np.ndarray, u1: np.ndarray, hl1: float, hw1: float,
    c2: np.ndarray, u2: np.ndarray, hl2: float, hw2: float,
) -> bool:
    """Separating-axis test for two oriented 2D boxes.

    ``c``: center, ``u``: unit heading axis (half-length ``hl`` along it),
    ``hw``: half-width along the perpendicular.
    """
    v1 = np.array([-u1[1], u1[0]], dtype=np.float64)
    v2 = np.array([-u2[1], u2[0]], dtype=np.float64)
    delta = c2 - c1
    for axis in (u1, v1, u2, v2):
        dist = abs(float(np.dot(delta, axis)))
        r1 = hl1 * abs(float(np.dot(u1, axis))) + hw1 * abs(float(np.dot(v1, axis)))
        r2 = hl2 * abs(float(np.dot(u2, axis))) + hw2 * abs(float(np.dot(v2, axis)))
        if dist > r1 + r2:
            return False
    return True


def _traj_headings(traj: np.ndarray) -> np.ndarray:
    """Unit motion-direction per trajectory point; holds the previous heading
    through near-stationary segments (initial heading = ego forward)."""
    headings = np.zeros_like(traj)
    prev = np.array([0.0, 1.0], dtype=np.float64)  # (right, forward): forward
    deltas = np.diff(traj, axis=0)
    for i in range(len(traj)):
        d = deltas[min(i, len(deltas) - 1)] if len(deltas) > 0 else np.zeros(2)
        norm = float(np.linalg.norm(d))
        if norm > 1e-3:
            prev = d / norm
        headings[i] = prev
    return headings


def _retreat_overlap_exit_mask(traj, headings, speeds, half_len, half_width, obstacles):
    """Exempt only the shrinking initial contact of a slow, straight retreat.

    Full separation must occur within one metre in the supplied trajectory;
    waiting beyond the scoring horizon cannot stand in for an actual escape.
    Other objects, subsequent contacts and VRUs retain their usual checks.
    """
    mask = np.zeros((len(traj), len(obstacles)), dtype=bool)
    if abs(float(speeds[0])) > 0.3 or not np.any(traj[:, 1] < -1e-4):
        return mask
    for column, obstacle in enumerate(obstacles):
        if (obstacle.object_type not in {"vehicle", "static"}
                or np.linalg.norm(obstacle.velocity) > 1e-3
                or obstacle.future_pos is not None
                or obstacle.center[1] <= 0.0):
            continue
        u = obstacle.direction
        v = np.array([-u[1], u[0]])
        axes = np.array([[0., 1.], [1., 0.], u, v])
        radii = (half_len * np.abs(axes[:, 1]) + half_width * np.abs(axes[:, 0])
                 + obstacle.half_len * np.abs(axes @ u)
                 + obstacle.half_width * np.abs(axes @ v))
        initial_gap = np.max(np.abs(axes @ obstacle.center) - radii)
        if not -0.3 <= initial_gap <= 0.0:
            continue
        gaps = np.max(np.abs((obstacle.center - traj) @ axes.T) - radii, axis=1)
        separated = np.flatnonzero(gaps > 1e-6)
        if not len(separated):
            continue
        end = int(separated[0])
        prefix = traj[:end + 1]
        deltas = np.diff(np.vstack([np.zeros(2), prefix]), axis=0)
        improvement = np.diff(np.r_[initial_gap, gaps[:end + 1]])
        # Constant lateral position also avoids a rotation changing which
        # corner penetrates while a scalar penetration depth seems to improve.
        if (np.max(np.abs(prefix[:, 0])) > 1e-3
                or np.max(np.abs(headings[:end + 1, 0])) > 1e-3
                or np.any(deltas[:, 1] > 1e-6)
                or np.linalg.norm(prefix[-1]) > 1.0
                or np.max(speeds[:end + 1]) > 1.5
                or np.any(improvement < -1e-6)
                or np.any((deltas[:, 1] < -1e-4) & (improvement <= 1e-6))):
            continue
        mask[:end, column] = True
    return mask


def _traj_headings_batch(traj: np.ndarray) -> np.ndarray:
    """Batched ``_traj_headings`` over ``(B, T, 2)`` trajectories."""
    traj = np.asarray(traj, dtype=np.float64)
    batch_size, sample_count, _ = traj.shape
    headings = np.empty_like(traj)
    previous = np.zeros((batch_size, 2), dtype=np.float64)
    previous[:, 1] = 1.0
    deltas = np.diff(traj, axis=1)
    for sample_idx in range(sample_count):
        delta = (
            deltas[:, min(sample_idx, deltas.shape[1] - 1), :]
            if deltas.shape[1]
            else np.zeros((batch_size, 2), dtype=np.float64)
        )
        norms = np.linalg.norm(delta, axis=1)
        moving = norms > 1e-3
        previous[moving] = delta[moving] / norms[moving, None]
        headings[:, sample_idx, :] = previous
    return headings


def _batch_obb_overlap(
    ego_centers: np.ndarray,
    ego_directions: np.ndarray,
    ego_half_len: float,
    ego_half_width: float,
    obstacle_centers: np.ndarray,
    obstacle_directions: np.ndarray,
    obstacle_half_lens: np.ndarray,
    obstacle_half_widths: np.ndarray,
) -> np.ndarray:
    """Broadcasted SAT overlap for every ``(candidate, time, obstacle)``."""
    ego_u = ego_directions[:, :, None, :]
    ego_v = np.stack([-ego_u[..., 1], ego_u[..., 0]], axis=-1)
    obs_u = obstacle_directions[None, :, :, :]
    obs_v = np.stack([-obs_u[..., 1], obs_u[..., 0]], axis=-1)
    delta = obstacle_centers[None, :, :, :] - ego_centers[:, :, None, :]
    obs_hl = obstacle_half_lens[None, None, :]
    obs_hw = obstacle_half_widths[None, None, :]

    overlap = np.ones(delta.shape[:-1], dtype=bool)
    for axis in (ego_u, ego_v, obs_u, obs_v):
        distance = np.abs(np.sum(delta * axis, axis=-1))
        ego_radius = (
            float(ego_half_len) * np.abs(np.sum(ego_u * axis, axis=-1))
            + float(ego_half_width) * np.abs(np.sum(ego_v * axis, axis=-1))
        )
        obstacle_radius = (
            obs_hl * np.abs(np.sum(obs_u * axis, axis=-1))
            + obs_hw * np.abs(np.sum(obs_v * axis, axis=-1))
        )
        overlap &= distance <= ego_radius + obstacle_radius
    return overlap


def a(
    traj: np.ndarray,
    current_speed_mps: float,
    accel_limit_mps2: float,
    timestep: float,
) -> np.ndarray:
    """Longitudinal feasibility re-timing of a plan polyline. 
    This helper keeps the plan's PATH but moves row k to the arc
    length actually reachable by t_k from the measured current speed under a
    constant acceleration limit:

        s_max(t_k) = v0 * t_k + 0.5 * a_max * t_k^2
        row_k      = point at arc length min(s_plan(k), s_max(t_k))

    Properties: exact no-op whenever the plan's own pacing is achievable
    (s_plan <= s_max everywhere); at low speed the far samples bunch along the
    reachable prefix, so distant geometry can no longer trip NC/TTC/DAC within
    the horizon; the index-as-time contract downstream becomes physically
    honest.
    """
    traj = np.asarray(traj, dtype=np.float64)
    if traj.ndim != 2 or len(traj) < 2 or accel_limit_mps2 is None:
        return traj
    a_max = float(accel_limit_mps2)
    if a_max <= 0.0:
        return traj
    seg_len = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    if cum[-1] <= 1e-9:
        return traj
    v0 = max(0.0, float(current_speed_mps or 0.0))
    t = np.arange(len(traj), dtype=np.float64) * float(timestep)
    s_new = np.minimum(cum, v0 * t + 0.5 * a_max * t * t)
    x = np.interp(s_new, cum, traj[:, 0])
    y = np.interp(s_new, cum, traj[:, 1])
    return np.column_stack([x, y])


def retime_plan_polyline(
    traj: np.ndarray,
    current_speed_mps: float,
    accel_limit_mps2: float,
    timestep: float,
    brake_limit_mps2: Optional[float] = None,
) -> np.ndarray:
    """Longitudinal feasibility re-timing of a plan polyline.

    The checks below treat row k as "the ego at t = k*timestep" (obstacle
    projection at ``idx * timestep``, gate horizons, per-sample speeds from
    ``diff/timestep``). Raw planner output frequently violates that timetable:
    e.g. a stationary ego handed a plan with 7.6 m point spacing "reaches"
    45 m in 3 s. This helper keeps the plan's PATH but moves row k to the arc
    length actually reachable by t_k from the measured current speed under a
    constant acceleration limit:

        s_max(t_k) = v0 * t_k + 0.5 * a_max * t_k^2
        row_k      = point at arc length min(s_plan(k), s_max(t_k))

    Properties: exact no-op whenever the plan's own pacing is achievable
    (s_plan <= s_max everywhere); at low speed the far samples bunch along the
    reachable prefix, so distant geometry can no longer trip NC/TTC/DAC within
    the horizon; the index-as-time contract downstream becomes physically
    honest. If braking requires travel beyond a short plan endpoint, the last
    path segment is extrapolated so the safety check cannot assume an instant
    stop at the endpoint.
    """
    traj = np.asarray(traj, dtype=np.float64)
    if traj.ndim != 2 or len(traj) < 2 or accel_limit_mps2 is None:
        return traj
    a_max = float(accel_limit_mps2)
    if a_max <= 0.0:
        return traj
    seg_len = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    if cum[-1] <= 1e-9:
        return traj
    v0 = max(0.0, float(current_speed_mps or 0.0))
    t = np.arange(len(traj), dtype=np.float64) * float(timestep)
    s_max = v0 * t + 0.5 * a_max * t * t
    if brake_limit_mps2 is None:
        s_min = np.zeros_like(t)
    else:
        b_max = float(brake_limit_mps2)
        if b_max <= 0.0:
            raise ValueError(
                f"brake_limit_mps2 must be > 0, got {brake_limit_mps2!r}"
            )
        stop_t = v0 / b_max
        s_min = np.where(
            t <= stop_t,
            v0 * t - 0.5 * b_max * t * t,
            0.5 * v0 * stop_t,
        )
    s_new = np.minimum(np.maximum(cum, s_min), s_max)
    xy = np.column_stack(
        [np.interp(s_new, cum, traj[:, dim]) for dim in range(2)]
    )
    beyond = s_new > cum[-1]
    if np.any(beyond):
        last_nonzero = np.flatnonzero(seg_len > 1e-9)
        if len(last_nonzero):
            direction = traj[last_nonzero[-1] + 1, :2] - traj[last_nonzero[-1], :2]
            direction /= np.linalg.norm(direction)
            xy[beyond] = traj[-1, :2] + (s_new[beyond] - cum[-1])[:, None] * direction
    return xy


class TTCVerifier(BaseVerifier):
    """NC + TTC gate over privileged obstacles, pure numpy, CARLA-frame aware."""

    def __init__(
        self,
        timestep: float = 0.5,
        probe_horizons_sec: Sequence[float] = DEFAULT_PROBE_HORIZONS_SEC,
        nc_gate_horizon_sec: Optional[float] = None,
        ttc_gate_horizon_sec: Optional[float] = None,
        retime_accel_limit_mps2: Optional[float] = None,
        retime_brake_limit_mps2: Optional[float] = None,
        min_walker_speed_mps: Optional[float] = None,
        pedestrian_gate_horizon_sec: Optional[float] = None,
        stationary_vehicle_lateral_margin_m: float = 0.0,
        agent_forecast_mode: str = "cv",
        gate_horizon_stop_conditioned: bool = False,
        gate_horizon_reaction_sec: float = 0.15,
    ) -> None:
        # Obstacle forecast fidelity for NC/TTC projections:
        #   "cv"      — constant-velocity straight line (legacy default; all
        #               campaign runs before 2026-07-22 used this)
        #   "bicycle" — VEHICLES with a privileged agent_control snapshot get a
        #               held-action kinematic-bicycle rollout (WoR constants);
        #               pedestrians (walking-pace floor) and static props stay CV.
        if agent_forecast_mode not in ("cv", "bicycle"):
            raise ValueError(
                f"agent_forecast_mode must be 'cv' or 'bicycle', got {agent_forecast_mode!r}"
            )
        self._agent_forecast_mode = str(agent_forecast_mode)
        self._timestep = float(timestep)
        self._probe_horizons_sec = tuple(float(t) for t in probe_horizons_sec)
        self._nc_gate_horizon_sec = (
            None if nc_gate_horizon_sec is None else float(nc_gate_horizon_sec)
        )
        self._ttc_gate_horizon_sec = (
            None if ttc_gate_horizon_sec is None else float(ttc_gate_horizon_sec)
        )
        # Stop-conditioned veto horizon. A FIXED horizon silently stops being a
        # safety gate once the ego is fast enough that braking takes longer than
        # the horizon: the conflict is classed "beyond gate" (advisory) at the
        # exact moment braking would have had to start. Measured on the c5
        # parameter set (a=3.5, b=4.0, horizon 1.5 s), margin between the first
        # veto and the stopping distance for a 1.4 m/s crossing pedestrian:
        #     6 m/s -> +3.94 m | 8 -> +1.66 | 10 -> +0.07 | 12 -> -2.31
        # i.e. the gate is already worthless at 10 m/s and actively too late at
        # 12. Raising the fixed horizon instead would re-admit the far-tip false
        # positives the 2026-07-16 retiming memo removed, so scale it with the
        # ego's own stopping time: horizon >= v0/b + reaction.
        self._gate_horizon_stop_conditioned = bool(gate_horizon_stop_conditioned)
        self._gate_horizon_reaction_sec = max(0.0, float(gate_horizon_reaction_sec))
        #: stopping-time floor for this verify() call; 0.0 disables the floor.
        self._stop_horizon_floor_sec = 0.0
        if retime_accel_limit_mps2 is not None and float(retime_accel_limit_mps2) <= 0.0:
            raise ValueError(f"retime_accel_limit_mps2 must be > 0, got {retime_accel_limit_mps2!r}")
        self._retime_accel_limit_mps2 = (
            None if retime_accel_limit_mps2 is None else float(retime_accel_limit_mps2)
        )
        if retime_brake_limit_mps2 is not None and float(retime_brake_limit_mps2) <= 0.0:
            raise ValueError(f"retime_brake_limit_mps2 must be > 0, got {retime_brake_limit_mps2!r}")
        self._retime_brake_limit_mps2 = (
            None if retime_brake_limit_mps2 is None else float(retime_brake_limit_mps2)
        )
        # Walking-speed floor for pedestrians that are already measurably
        # moving. Truly stationary pedestrians keep their measured zero
        # velocity; possible step-out uncertainty belongs in the controller's
        # soft caution policy, not in this hard bbox/TTC projection.
        self._min_walker_speed_mps = (
            None if min_walker_speed_mps is None else float(min_walker_speed_mps)
        )
        # Class-dependent veto horizon: collisions with pedestrians use this
        # (typically LONGER) horizon for both the NC and TTC gates; vehicles
        # and static obstacles keep nc/ttc_gate_horizon_sec. None = no
        # per-class override (legacy).
        self._pedestrian_gate_horizon_sec = (
            None if pedestrian_gate_horizon_sec is None else float(pedestrian_gate_horizon_sec)
        )
        self._stationary_vehicle_lateral_margin_m = max(
            0.0, float(stationary_vehicle_lateral_margin_m)
        )
        self.last_result: Optional[VerificationResult] = None
        #: collision detail of the last verify() call, for diagnostics:
        #: {"nc": [collision dict, ...], "ttc": [...]} (see _sweep_collisions).
        self.last_collisions: Dict[str, List[Dict[str, Any]]] = {"nc": [], "ttc": []}
        # The two Round-1 proposal branches share one simulator snapshot. Keep
        # their expensive local obstacle conversion/bicycle rollout for the
        # second batch; the timestamp prevents reuse across control ticks.
        self._batch_obstacle_cache_key = None
        self._batch_obstacle_cache: Optional[List[Obstacle]] = None


    def _all_pass(self) -> VerificationResult:
        self.last_collisions = {"nc": [], "ttc": []}
        result = VerificationResult(
            gates={"nc": 1.0}, soft={"ttc": 1.0}, weights={"ttc": 1.0}
        )
        self.last_result = result
        return result

    def _stop_time_floor_sec(self, current_speed_mps: float) -> float:
        """Seconds the ego needs to stop from ``current_speed_mps``, plus reaction.

        Zero when the feature is off or no brake limit is configured, which
        leaves every horizon exactly as it was.
        """
        if not self._gate_horizon_stop_conditioned:
            return 0.0
        brake = self._retime_brake_limit_mps2
        if brake is None or brake <= 0.0:
            return 0.0
        return abs(float(current_speed_mps)) / float(brake) + self._gate_horizon_reaction_sec

    def _gate_horizon_for(self, collision: Dict[str, Any], default_horizon: Optional[float]) -> Optional[float]:
        """Veto horizon for one collision: pedestrian override, else default.

        Widened to the ego's stopping time when ``gate_horizon_stop_conditioned``
        is on, so the gate never expires before the vehicle can physically stop.
        """
        if (
            self._pedestrian_gate_horizon_sec is not None
            and collision.get("object_type") == "pedestrian"
        ):
            horizon = self._pedestrian_gate_horizon_sec
        else:
            horizon = default_horizon
        if horizon is None:
            return None  # already an unlimited veto; nothing to widen
        return max(float(horizon), self._stop_horizon_floor_sec)

    @staticmethod
    def _obstacles_to_local(
        current_info: Dict[str, Any],
        privileged_agents: List[Dict[str, Any]],
        min_walker_speed_mps: Optional[float] = None,
        stationary_vehicle_lateral_margin_m: float = 0.0,
        agent_forecast_mode: str = "cv",
        forecast_horizon_sec: float = 0.0,
    ) -> List[Obstacle]:
        """Map privileged agents into the ego-local (right, forward) frame."""
        pose = info_to_pose(current_info)
        rot = np.asarray(pose[:3, :3], dtype=np.float64)
        obstacles = []
        for agent in privileged_agents:
            pos_world = np.asarray(agent.get("agent_pos_world", (0.0, 0.0, 0.0)), dtype=np.float64)
            center = world_points_to_current_local(pos_world[None, :], pose)[0].astype(np.float64)

            heading = float(agent.get("agent_heading", 0.0))
            dir_world = np.array([np.cos(heading), np.sin(heading), 0.0], dtype=np.float64)
            dir_local3 = rot.T @ dir_world
            # (right, forward) = (body Y, body X); matches world_points_to_current_local.
            dir_local = np.array([dir_local3[1], dir_local3[0]], dtype=np.float64)
            norm = float(np.linalg.norm(dir_local))
            dir_local = dir_local / norm if norm > 1e-6 else np.array([0.0, 1.0])

            vel_vec = agent.get("agent_vel_vec") or (0.0, 0.0)
            vel_world = np.array([float(vel_vec[0]), float(vel_vec[1]), 0.0], dtype=np.float64)
            vel_local3 = rot.T @ vel_world
            vel_local = np.array([vel_local3[1], vel_local3[0]], dtype=np.float64)
            actor_speed = float(np.linalg.norm(vel_local))
            stationary_walker = (
                str(agent.get("object_type", "vehicle")) == "pedestrian"
                and actor_speed < STATIONARY_WALKER_SPEED_MPS
            )
            if (
                min_walker_speed_mps is not None
                and str(agent.get("object_type", "vehicle")) == "pedestrian"
                and actor_speed < float(min_walker_speed_mps)
                and not stationary_walker
            ):
                # Floor an already-moving pedestrian to walking pace along its
                # facing. A truly stationary pedestrian retains measured zero
                # velocity, so hard TTC checks the ego sweep against its real
                # current bbox instead of an invented future crossing.
                vel_local = dir_local * float(min_walker_speed_mps)

            extent = agent.get("agent_extent") or (1.0, 1.0)
            half_len = max(0.05, float(extent[0]) / 2.0)
            half_width = max(0.05, float(extent[1]) / 2.0)
            if (
                str(agent.get("object_type", "vehicle")) == "vehicle"
                and actor_speed < STATIONARY_VEHICLE_SPEED_MPS
            ):
                # CARLA's actor.bounding_box covers the closed vehicle body,
                # not an opened door.  Expand only the lateral extent of a
                # measured-stationary vehicle so candidate selection leaves a
                # door-width clearance.  Position and velocity stay measured.
                half_width += max(0.0, float(stationary_vehicle_lateral_margin_m))

            # Held-action bicycle rollout: vehicles only, and only when the
            # snapshot carries a control triple (older pickles/replays fall
            # back to constant velocity). Rolled in the world frame, then
            # mapped through the same pose transforms as the current state.
            future_pos = future_dir = None
            future_dt = 0.05
            control = agent.get("agent_control")
            if (
                agent_forecast_mode == "bicycle"
                and str(agent.get("object_type", "vehicle")) == "vehicle"
                and control is not None
                and forecast_horizon_sec > 0.0
            ):
                speed = float(agent.get("agent_velo", 0.0) or 0.0)
                pos_w, head_w = bicycle_forecast_world(
                    pos_world[:2], heading, speed, control, forecast_horizon_sec, future_dt
                )
                pos3 = np.column_stack([pos_w, np.full(len(pos_w), pos_world[2])])
                future_pos = world_points_to_current_local(pos3, pose)[:, :2].astype(np.float64)
                dirs_w = np.column_stack([np.cos(head_w), np.sin(head_w), np.zeros(len(head_w))])
                dirs_l3 = dirs_w @ rot  # == (rot.T @ dirs_w.T).T
                future_dir = np.column_stack([dirs_l3[:, 1], dirs_l3[:, 0]])
                norms = np.linalg.norm(future_dir, axis=1, keepdims=True)
                future_dir = np.where(norms > 1e-6, future_dir / np.maximum(norms, 1e-12), [0.0, 1.0])

            obstacles.append(Obstacle(
                center=center,
                direction=dir_local,
                half_len=half_len,
                half_width=half_width,
                velocity=vel_local,
                agent_id=str(agent.get("agent_id", "?")),
                object_type=str(agent.get("object_type", "vehicle")),
                pos_world=[float(v) for v in pos_world],
                future_pos=future_pos,
                future_dir=future_dir,
                future_dt=future_dt,
            ))
        return obstacles

    def _sweep_collisions(
        self,
        traj: np.ndarray,
        headings: np.ndarray,
        speeds: np.ndarray,
        ego_half_len: float,
        ego_half_width: float,
        obstacles: List[Obstacle],
        obstacle_time_offset_sec: Optional[float],
    ) -> List[Dict[str, Any]]:
        """All at-fault collisions of the swept ego box, first hit per obstacle.

        With ``obstacle_time_offset_sec`` set, each obstacle is projected with
        constant velocity to ``idx * timestep + offset`` (time-aligned probe);
        with ``None`` obstacles are held static at their current pose.

        Fail2Drive at-fault gate: samples where the ego's (planned) speed is
        below ``ATFAULT_EGO_SPEED_EPSILON_MPS`` are skipped — the reference
        CollisionTest never counts contacts while the ego is stationary
        (atomic_criteria.py:384-385).
        """
        hit_ids = set()
        collisions: List[Dict[str, Any]] = []
        exiting = _retreat_overlap_exit_mask(
            traj, headings, speeds, ego_half_len, ego_half_width, obstacles
        )
        for idx in range(len(traj)):
            if float(speeds[idx]) < ATFAULT_EGO_SPEED_EPSILON_MPS:
                continue
            t_abs = None if obstacle_time_offset_sec is None else idx * self._timestep + obstacle_time_offset_sec
            for column, obstacle in enumerate(obstacles):
                if obstacle.agent_id in hit_ids or exiting[idx, column]:
                    continue
                if t_abs is None:
                    obstacle_center, obstacle_dir = obstacle.center, obstacle.direction
                elif obstacle.future_pos is not None and obstacle.future_dir is not None:
                    # bicycle rollout lookup (position AND heading evolve)
                    k = min(int(round(t_abs / obstacle.future_dt)), len(obstacle.future_pos) - 1)
                    obstacle_center, obstacle_dir = obstacle.future_pos[k], obstacle.future_dir[k]
                else:
                    obstacle_center = obstacle.center + obstacle.velocity * t_abs
                    obstacle_dir = obstacle.direction
                if _obb_overlap(
                    traj[idx], headings[idx], ego_half_len, ego_half_width,
                    obstacle_center, obstacle_dir, obstacle.half_len, obstacle.half_width,
                ):
                    hit_ids.add(obstacle.agent_id)
                    collisions.append({
                        "agent_id": obstacle.agent_id,
                        "object_type": obstacle.object_type,
                        "pos_world": obstacle.pos_world,
                        "sample_idx": int(idx),
                        "distance_m": float(np.linalg.norm(obstacle.center)),
                        "penalty": COLLISION_PENALTIES.get(obstacle.object_type, COLLISION_PENALTIES["static"]),
                        # which projection produced this hit — audit trail for
                        # the cv/bicycle A/B in verifier_debug.jsonl
                        "forecast": "bicycle" if obstacle.future_pos is not None else "cv",
                    })
        return collisions

    @staticmethod
    def _obstacle_states_at_times(
        obstacles: List[Obstacle], times_sec: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Obstacle centers/headings at all requested times."""
        times_sec = np.asarray(times_sec, dtype=np.float64).reshape(-1)
        centers = np.empty((len(times_sec), len(obstacles), 2), dtype=np.float64)
        directions = np.empty_like(centers)
        for obstacle_idx, obstacle in enumerate(obstacles):
            if obstacle.future_pos is not None and obstacle.future_dir is not None:
                future_indices = np.asarray(
                    [
                        min(
                            int(round(float(t) / obstacle.future_dt)),
                            len(obstacle.future_pos) - 1,
                        )
                        for t in times_sec
                    ],
                    dtype=np.int64,
                )
                centers[:, obstacle_idx, :] = obstacle.future_pos[future_indices]
                directions[:, obstacle_idx, :] = obstacle.future_dir[future_indices]
            else:
                centers[:, obstacle_idx, :] = (
                    obstacle.center[None, :]
                    + times_sec[:, None] * obstacle.velocity[None, :]
                )
                directions[:, obstacle_idx, :] = obstacle.direction[None, :]
        return centers, directions

    def _batch_gate_mask(
        self,
        obstacles: List[Obstacle],
        conflict_times_sec: np.ndarray,
        default_horizon_sec: Optional[float],
    ) -> np.ndarray:
        """Whether each ``(time, obstacle)`` collision belongs to the gate."""
        conflict_times_sec = np.asarray(conflict_times_sec, dtype=np.float64)
        horizons = np.empty(len(obstacles), dtype=np.float64)
        for obstacle_idx, obstacle in enumerate(obstacles):
            horizon = (
                self._pedestrian_gate_horizon_sec
                if (
                    self._pedestrian_gate_horizon_sec is not None
                    and obstacle.object_type == "pedestrian"
                )
                else default_horizon_sec
            )
            horizons[obstacle_idx] = (
                np.inf if horizon is None
                else max(float(horizon), self._stop_horizon_floor_sec)
            )
        return conflict_times_sec[:, None] <= horizons[None, :]

    def verify_batch(
        self,
        trajectories: Any,
        current_info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return batched ``(NC, TTC, TTC_VRU)`` arrays without state writes.

        Candidate scoring evaluates independent trajectories from one simulator
        frame. Their obstacle snapshot and bicycle forecasts are identical, so
        build those once and broadcast the SAT collision checks over the whole
        ``(candidate, time, obstacle)`` tensor.

        ``TTC_VRU`` is 0.0 for candidates whose TTC=0 was caused by a VRU, i.e.
        the batched form of ``gating_vru_conflict(last_collisions["ttc"])``.
        Like the scalar path it reports the FIRST probe horizon that trips TTC
        and stops there, so a candidate already resolved by a vehicle is never
        re-flagged as a VRU conflict at a later probe.
        """
        plans = np.asarray(trajectories, dtype=np.float64)
        if plans.ndim != 3:
            raise ValueError(
                f"trajectories must have shape (B, T, 2), got {plans.shape}"
            )
        batch_size = len(plans)
        if batch_size == 0:
            empty = np.zeros(0, dtype=np.float64)
            return empty, empty.copy(), empty.copy()
        if plans.shape[1] < 2 or current_info is None:
            ones = np.ones(batch_size, dtype=np.float64)
            return ones, ones.copy(), ones.copy()

        privileged_agents = current_info.get("privileged_agents") or []
        ego_box = current_info.get("ego_box")
        if not privileged_agents or ego_box is None:
            ones = np.ones(batch_size, dtype=np.float64)
            return ones, ones.copy(), ones.copy()

        ego_box = np.asarray(ego_box, dtype=np.float64)
        ego_half_len = max(0.05, float(ego_box[3]) / 2.0)
        ego_half_width = max(0.05, float(ego_box[4]) / 2.0)
        max_probe = max(self._probe_horizons_sec) if self._probe_horizons_sec else 0.0
        forecast_horizon_sec = (plans.shape[1] - 1) * self._timestep + max_probe
        obstacle_cache_key = (
            float(current_info.get("timestamp", 0.0) or 0.0),
            id(privileged_agents),
            int(plans.shape[1]),
            float(forecast_horizon_sec),
            tuple(
                np.asarray(current_info.get("ego_pos", ()), dtype=np.float64)
                .reshape(-1)
                .tolist()
            ),
            tuple(
                np.asarray(current_info.get("ego_rot", ()), dtype=np.float64)
                .reshape(-1)
                .tolist()
            ),
        )
        if obstacle_cache_key == self._batch_obstacle_cache_key:
            obstacles = self._batch_obstacle_cache or []
        else:
            obstacles = self._obstacles_to_local(
                current_info,
                list(privileged_agents),
                min_walker_speed_mps=self._min_walker_speed_mps,
                stationary_vehicle_lateral_margin_m=(
                    self._stationary_vehicle_lateral_margin_m
                ),
                agent_forecast_mode=self._agent_forecast_mode,
                forecast_horizon_sec=forecast_horizon_sec,
            )
            self._batch_obstacle_cache_key = obstacle_cache_key
            self._batch_obstacle_cache = obstacles
        if not obstacles:
            ones = np.ones(batch_size, dtype=np.float64)
            return ones, ones.copy(), ones.copy()

        traj = np.concatenate(
            [np.zeros((batch_size, 1, 2), dtype=np.float64), plans[:, :, :2]],
            axis=1,
        )
        raw_speeds = np.linalg.norm(np.diff(traj, axis=1), axis=2) / self._timestep
        current_speed = float(
            current_info.get(
                "ego_velo", raw_speeds[0, 0] if raw_speeds.size else 0.0
            )
            or 0.0
        )
        self._stop_horizon_floor_sec = self._stop_time_floor_sec(current_speed)
        if self._retime_accel_limit_mps2 is not None:
            traj = np.stack(
                [
                    retime_plan_polyline(
                        candidate,
                        current_speed,
                        self._retime_accel_limit_mps2,
                        self._timestep,
                        self._retime_brake_limit_mps2,
                    )
                    for candidate in traj
                ],
                axis=0,
            )

        headings = _traj_headings_batch(traj)
        planned_speeds = np.linalg.norm(np.diff(traj, axis=1), axis=2) / self._timestep
        speeds = np.concatenate(
            [
                np.full((batch_size, 1), current_speed, dtype=np.float64),
                planned_speeds,
            ],
            axis=1,
        )
        sample_times = np.arange(traj.shape[1], dtype=np.float64) * self._timestep
        obstacle_half_lens = np.asarray(
            [obstacle.half_len for obstacle in obstacles], dtype=np.float64
        )
        obstacle_half_widths = np.asarray(
            [obstacle.half_width for obstacle in obstacles], dtype=np.float64
        )

        obstacle_centers, obstacle_directions = self._obstacle_states_at_times(
            obstacles, sample_times
        )
        nc_overlap = _batch_obb_overlap(
            traj,
            headings,
            ego_half_len,
            ego_half_width,
            obstacle_centers,
            obstacle_directions,
            obstacle_half_lens,
            obstacle_half_widths,
        )
        for candidate in range(batch_size):
            nc_overlap[candidate] &= ~_retreat_overlap_exit_mask(
                traj[candidate], headings[candidate], speeds[candidate],
                ego_half_len, ego_half_width, obstacles,
            )
        nc_overlap &= speeds[:, :, None] >= ATFAULT_EGO_SPEED_EPSILON_MPS
        nc_overlap &= self._batch_gate_mask(
            obstacles, sample_times, self._nc_gate_horizon_sec
        )[None, :, :]
        nc = (~np.any(nc_overlap, axis=(1, 2))).astype(np.float64)

        velocities = np.diff(traj, axis=1) / self._timestep
        velocities = (
            np.concatenate([velocities[:, :1, :], velocities], axis=1)
            if velocities.shape[1]
            else np.zeros_like(traj)
        )
        ttc = np.ones(batch_size, dtype=np.float64)
        ttc_vru = np.ones(batch_size, dtype=np.float64)
        vru_columns = np.asarray(
            [obstacle.object_type in VRU_OBJECT_TYPES for obstacle in obstacles],
            dtype=bool,
        )
        unresolved = np.ones(batch_size, dtype=bool)
        for probe_sec in self._probe_horizons_sec:
            if not np.any(unresolved):
                break
            probed = traj + velocities * float(probe_sec)
            obstacle_centers, obstacle_directions = self._obstacle_states_at_times(
                obstacles, sample_times + float(probe_sec)
            )
            overlap = _batch_obb_overlap(
                probed,
                headings,
                ego_half_len,
                ego_half_width,
                obstacle_centers,
                obstacle_directions,
                obstacle_half_lens,
                obstacle_half_widths,
            )
            for candidate in range(batch_size):
                overlap[candidate] &= ~_retreat_overlap_exit_mask(
                    probed[candidate], headings[candidate], speeds[candidate],
                    ego_half_len, ego_half_width, obstacles,
                )
            overlap &= speeds[:, :, None] >= ATFAULT_EGO_SPEED_EPSILON_MPS
            overlap &= self._batch_gate_mask(
                obstacles,
                sample_times + float(probe_sec),
                self._ttc_gate_horizon_sec,
            )[None, :, :]
            collided = np.any(overlap, axis=(1, 2))
            if vru_columns.any():
                # Mask by ``unresolved`` first: a candidate whose TTC was
                # already zeroed by a vehicle at an earlier probe is settled,
                # matching the scalar path's break-on-first-tripping-probe.
                ttc_vru[
                    unresolved & np.any(overlap[:, :, vru_columns], axis=(1, 2))
                ] = 0.0
            ttc[collided] = 0.0
            unresolved &= ~collided
        return nc, ttc, ttc_vru


    def verify(self, trajectory: Any, current_info: Optional[Dict[str, Any]] = None) -> VerificationResult:
        if trajectory is None or current_info is None:
            return self._all_pass()
        plan = np.asarray(trajectory, dtype=np.float64)
        if plan.ndim != 2 or len(plan) < 2:
            return self._all_pass()

        privileged_agents = current_info.get("privileged_agents") or []
        ego_box = current_info.get("ego_box")
        if not privileged_agents or ego_box is None:
            return self._all_pass()

        ego_box = np.asarray(ego_box, dtype=np.float64)
        ego_half_len = max(0.05, float(ego_box[3]) / 2.0)
        ego_half_width = max(0.05, float(ego_box[4]) / 2.0)

        max_probe = max(self._probe_horizons_sec) if self._probe_horizons_sec else 0.0
        obstacles = self._obstacles_to_local(
            current_info, list(privileged_agents),
            min_walker_speed_mps=self._min_walker_speed_mps,
            stationary_vehicle_lateral_margin_m=(
                self._stationary_vehicle_lateral_margin_m
            ),
            agent_forecast_mode=self._agent_forecast_mode,
            forecast_horizon_sec=(len(plan) - 1) * self._timestep + max_probe,
        )

        # Ego samples: current pose prepended, motion-direction headings.
        traj = np.concatenate([np.zeros((1, 2), dtype=np.float64), plan[:, :2]], axis=0)
        raw_speeds = np.linalg.norm(np.diff(traj, axis=0), axis=1) / self._timestep
        current_speed = float(current_info.get("ego_velo", raw_speeds[0] if len(raw_speeds) else 0.0))
        self._stop_horizon_floor_sec = self._stop_time_floor_sec(current_speed)
        # Feasibility re-timing: after this, "row k is the ego at k*timestep"
        # (assumed by the obstacle projection and gate horizons below) is
        # physically true rather than the planner's claim. No-op when the
        # plan's own pacing is achievable or the knob is off (None).
        if self._retime_accel_limit_mps2 is not None:
            traj = retime_plan_polyline(
                traj,
                current_speed,
                self._retime_accel_limit_mps2,
                self._timestep,
                self._retime_brake_limit_mps2,
            )
        headings = _traj_headings(traj)

        planned_speeds = np.linalg.norm(np.diff(traj, axis=0), axis=1) / self._timestep
        speeds = np.concatenate([[current_speed], planned_speeds])

        nc_collisions = self._sweep_collisions(
            traj, headings, speeds, ego_half_len, ego_half_width, obstacles,
            obstacle_time_offset_sec=0.0,
        )
        # NAVSIM binary gate: any at-fault collision inside the gate horizon
        # vetoes the plan; later conflicts are advisory (see __init__ note).
        gating = []
        for collision in nc_collisions:
            conflict_t = collision["sample_idx"] * self._timestep
            horizon = self._gate_horizon_for(collision, self._nc_gate_horizon_sec)
            beyond = horizon is not None and conflict_t > horizon
            collision["conflict_time_sec"] = float(conflict_t)
            collision["beyond_gate_horizon"] = bool(beyond)
            if not beyond:
                gating.append(collision)
        nc = 0.0 if gating else 1.0

        # TTC probes: shift ego forward by velocity * t, obstacles time-aligned.
        ttc = 1.0
        ttc_collisions: List[Dict[str, Any]] = []
        velocities = np.diff(traj, axis=0) / self._timestep
        velocities = np.vstack([velocities[:1], velocities]) if len(velocities) else np.zeros_like(traj)
        for t_probe in self._probe_horizons_sec:
            probed = traj + velocities * t_probe
            ttc_collisions = self._sweep_collisions(
                probed, headings, speeds, ego_half_len, ego_half_width, obstacles,
                obstacle_time_offset_sec=t_probe,
            )
            gating_ttc = []
            for collision in ttc_collisions:
                conflict_t = collision["sample_idx"] * self._timestep + t_probe
                horizon = self._gate_horizon_for(collision, self._ttc_gate_horizon_sec)
                beyond = horizon is not None and conflict_t > horizon
                collision["conflict_time_sec"] = float(conflict_t)
                collision["beyond_gate_horizon"] = bool(beyond)
                if not beyond:
                    gating_ttc.append(collision)
            if gating_ttc:
                ttc = 0.0
                break

        self.last_collisions = {"nc": nc_collisions, "ttc": ttc_collisions}
        result = VerificationResult(gates={"nc": nc}, soft={"ttc": ttc}, weights={"ttc": 1.0})
        self.last_result = result
        return result
