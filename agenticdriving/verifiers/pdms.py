"""PDMS trajectory verifier for the CARLA path.

    PDMS = NC * DAC * (5*TTC + 5*EP + 2*LK) / 10

Deviations from published NavSim PDMS:
  - DAC is adapted for CARLA (no NavSim drivable-area polygons exist here),
    selectable via ``dac_mode``: "tiered" (default) or "binary".
    The two published precedents disagree: NavSim's DAC is binary but over
    the WHOLE paved surface (ROADBLOCK/INTERSECTION/DRIVABLE_AREA/CARPARK
    union — opposite lanes and arguably shoulders are compliant; corridor
    adherence is v2's separate LK/DDC metrics), while Fail2Drive has no plan
    gate at all (OutsideRouteLanesTest: center-point vs any Driving/Parking
    lane, graded %-of-route penalty). Our tiers: 1.0 = exact port of F2D's
    lane rule (opposite lane compliant by design), 0.5 = paved non-driving
    surface (NavSim-compliant, F2D-penalized — the 0/0.5/1 tier pattern is
    NavSim's own, from its NC term), 0.0 = surface both precedents reject.
    "binary" clamps 0.5 -> 0.0 (stricter than BOTH precedents) for ablation.
    Empirically the mode DOES matter (2026-07-14 3-route probe): tier 0.5
    fired 0x on PedestriansOnRoad/RightConstruction but 561x in HardBrake's
    post-brake phase (shoulder-skirting plans, score 0.5 -> pass at t=0.35;
    binary would veto all of them). F2D intended solutions never need it
    (blocked scenarios despawn after a wait, srunner roadblocked.py;
    construction bypass is the opposite lane, tier 1.0).
    See _drivable_area_compliance and build_lane_tier_probe.
  - Comfort ("c") is dropped from the soft weights (published form is
    ttc=5, c=2, ep=5). See PDMS_SOFT_WEIGHTS below to restore.
  - LK (lane keeping) soft term, weight 2: a hybrid of the two published
    variants (each ingredient cited, the combination is ours).
    Hydra-MDP++ (arXiv 2503.12820) invented the metric as the min perpendicular
    distance d_i to "nearby lane segments" (candidate set undefined in the
    paper, code unreleased) checked instantaneously at every step; NavSim v2
    (pdm_scorer.py _calculate_lane_keeping) revised it: deviation must exceed
    0.5 m CONTINUOUSLY for 2.0 s (consecutive samples, counter resets on a
    compliant sample), intersections exempt (skipped WITHOUT resetting the
    counter), binary, weight 2 in the EPDMS soft avg. We take Hydra's
    nearest-lane reference. Deviation is bounded by ~lane_width/2 at a lane boundary (the
    snap flips to the next center), so LK owns the 0.5..~1.75 m in-corridor
    band (edge-hugging, straddling, weaving) while DAC owns the corridor edge.
    With the 2 s window a decisive lane change (<~2 s in the >0.5 m band)
    passes; lazy drifts and line-riding fail. See _lane_keeping.
  - VRU TTC veto (``vru_ttc_veto``, default off): TTC=0 caused by a
    pedestrian-class agent additionally trips a hard ``ttc_vru`` gate, making
    PDMS = NC * DAC * TTC_VRU * weighted_avg(...). Without it a lone TTC=0
    scores 0.583, above every threshold we run, so a pedestrian conflict can
    only trigger recovery once NC also trips — by which point the ego is
    typically inside its own stopping distance. See PDMSVerifier.__init__.
  - Possible change would be If TTC=0 probe noise causes recovery churn, ablate
    in the DECISION layer instead: reject only when TTCg=0 persists for k
    consecutive verifier steps (debounce), and/or tune verifier_threshold —
"""
from __future__ import annotations

import atexit
import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from agenticdriving.adapters.carla.geometry import info_to_pose, local_plan_to_world
from agenticdriving.verifiers.base import BaseVerifier, VerificationResult
from agenticdriving.verifiers.ego_progress import EgoProgressReference
# from agenticdriving.verifiers.kinematics import is_comfortable  # comfort removed from aggregation
from agenticdriving.verifiers.ttc import DEFAULT_PROBE_HORIZONS_SEC, TTCVerifier, gating_vru_conflict, retime_plan_polyline

LOG = logging.getLogger(__name__)

# PDMS is now NC * DAC * (5*TTC + 5*EP + 2*LK)/12. Restore comfort by
# re-adding "c" below and re-enabling the comfort block in verify().
PDMS_SOFT_WEIGHTS = {"ttc": 5.0, "ep": 5.0, "lk": 2.0}

LaneTierProbe = Callable[[np.ndarray], np.ndarray]
EPReferenceFactory = Callable[[], Optional[EgoProgressReference]]


def resample_plan_timestep(
    plan: np.ndarray,
    *,
    input_timestep_sec: float,
    sample_timestep_sec: float,
) -> np.ndarray:
    """Densify a local plan onto the verifier's sampling timestep.

    Planner row ``k`` represents time ``(k + 1) * input_timestep_sec``.  PDMS
    may use a finer collision-checking grid without changing that planner
    contract: prepend the current ego origin at ``t=0`` and linearly
    interpolate the path onto ``sample_timestep_sec`` while preserving the
    original horizon and endpoint.
    """
    plan = np.asarray(plan, dtype=np.float64)
    if plan.ndim != 2 or len(plan) < 2:
        return plan
    input_dt = float(input_timestep_sec)
    sample_dt = float(sample_timestep_sec)
    if input_dt <= 0.0 or sample_dt <= 0.0:
        raise ValueError(
            "PDMS timesteps must be positive, got "
            f"input={input_timestep_sec!r}, sample={sample_timestep_sec!r}"
        )
    if abs(input_dt - sample_dt) <= 1e-9:
        return plan

    horizon = len(plan) * input_dt
    sample_times = np.arange(sample_dt, horizon + 1e-9, sample_dt, dtype=np.float64)
    if len(sample_times) == 0 or sample_times[-1] < horizon - 1e-9:
        sample_times = np.append(sample_times, horizon)
    else:
        sample_times[-1] = horizon

    source_times = np.arange(len(plan) + 1, dtype=np.float64) * input_dt
    source_plan = np.concatenate(
        [np.zeros((1, plan.shape[1]), dtype=np.float64), plan], axis=0
    )
    sampled = np.column_stack(
        [
            np.interp(sample_times, source_times, source_plan[:, dim])
            for dim in range(plan.shape[1])
        ]
    )
    return sampled


class PDMSVerifier(BaseVerifier):
    """PDMS = NC * DAC * weighted_avg(TTC, Comfort[, EP]) from live CARLA state."""

    def __init__(
        self,
        timestep: float = 0.5,
        input_timestep_sec: Optional[float] = None,
        probe_horizons_sec: Sequence[float] = DEFAULT_PROBE_HORIZONS_SEC,
        nc_gate_horizon_sec: Optional[float] = None,
        ttc_gate_horizon_sec: Optional[float] = None,
        dac_gate_horizon_sec: Optional[float] = None,
        dac_exit_grace: bool = False,
        retime_accel_limit_mps2: Optional[float] = None,
        retime_brake_limit_mps2: Optional[float] = None,
        min_walker_speed_mps: Optional[float] = None,
        pedestrian_gate_horizon_sec: Optional[float] = None,
        stationary_vehicle_lateral_margin_m: float = 0.0,
        ep_on_retimed: bool = False,
        lk_deviation_limit_m: Optional[float] = 0.5,
        lk_horizon_window_sec: float = 2.0,
        hd_map_endpoint_only: bool = False,
        lane_tier_probe: Optional[LaneTierProbe] = None,
        ep_reference: Optional[EgoProgressReference] = None,
        ep_reference_factory: Optional[EPReferenceFactory] = None,
        dac_mode: str = "tiered",
        ttc_debounce_steps: int = 1,
        vru_ttc_veto: bool = False,
        agent_forecast_mode: str = "cv",
        gate_horizon_stop_conditioned: bool = False,
        gate_horizon_reaction_sec: float = 0.15,
        debug: bool = False,
        debug_log_path: Optional[str] = None,
        debug_vru_radius_m: Optional[float] = None,
    ) -> None:
        if dac_mode not in ("tiered", "binary"):
            raise ValueError(f"dac_mode must be 'tiered' or 'binary', got {dac_mode!r}")
        self._dac_mode = dac_mode
        if int(ttc_debounce_steps) < 1:
            raise ValueError(f"ttc_debounce_steps must be >= 1, got {ttc_debounce_steps!r}")
        self._ttc_debounce_steps = int(ttc_debounce_steps)
        self._ttc_zero_streak = 0
        self._last_ttc_raw: Optional[float] = None
        # VRU TTC veto: promote a pedestrian-caused TTC=0 from a soft term to
        # a hard gate. With the published weights a lone TTC=0 only pulls PDMS
        # down to (5*0 + 5*1 + 2*1)/12 = 0.583, which sits ABOVE every
        # threshold we run (0.35 / 0.43) — so a TTC=0 on a pedestrian could
        # never trigger recovery on its own, and only crossed the threshold
        # when EP happened to be low too. Measured on
        # c5_gated_gain1.3_full200_0819: 1087 logged seven consecutive
        # TTC=0 frames on a walker closing from 16.3 m, each scored 0.583
        # (pass), and the ego stayed at full throttle until NC tripped with
        # too little room to stop. Vetoing now hands those frames to the
        # orchestrator's timed pedestrian hard-brake latch; it deliberately
        # bypasses both progress retiming and maneuver recovery.
        # Applied to the DEBOUNCED ttc, so ttc_debounce_steps still governs
        # how many consecutive frames a conflict must persist.
        self._vru_ttc_veto = bool(vru_ttc_veto)
        self._timestep = float(timestep)
        self._input_timestep_sec = float(
            timestep if input_timestep_sec is None else input_timestep_sec
        )
        if self._timestep <= 0.0 or self._input_timestep_sec <= 0.0:
            raise ValueError(
                "PDMS timesteps must be positive, got "
                f"input={self._input_timestep_sec!r}, sample={self._timestep!r}"
            )
        self._ttc = TTCVerifier(
            timestep=timestep,
            probe_horizons_sec=probe_horizons_sec,
            nc_gate_horizon_sec=nc_gate_horizon_sec,
            ttc_gate_horizon_sec=ttc_gate_horizon_sec,
            retime_accel_limit_mps2=retime_accel_limit_mps2,
            retime_brake_limit_mps2=retime_brake_limit_mps2,
            min_walker_speed_mps=min_walker_speed_mps,
            pedestrian_gate_horizon_sec=pedestrian_gate_horizon_sec,
            stationary_vehicle_lateral_margin_m=(
                stationary_vehicle_lateral_margin_m
            ),
            agent_forecast_mode=agent_forecast_mode,
            gate_horizon_stop_conditioned=gate_horizon_stop_conditioned,
            gate_horizon_reaction_sec=gate_horizon_reaction_sec,
        )
        # DAC veto horizon (mirrors nc/ttc_gate_horizon_sec): only samples in
        # the first dac_gate_horizon_sec of (re-timed) plan time can zero DAC;
        # later tiers stay visible in _last_dac_tiers / debug as advisories.
        self._dac_gate_horizon_sec = (
            None if dac_gate_horizon_sec is None else float(dac_gate_horizon_sec)
        )
        # Exit grace: see _drivable_area_compliance. Default off (mentor-merge
        # safe); flip via yaml verifier.dac_exit_grace.
        self._dac_exit_grace = bool(dac_exit_grace)
        self._last_dac_exit: Optional[Dict[str, Any]] = None
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
        # EP semantics: False (default) = EP scores the RAW plan ("intended
        # route progress") even when the safety terms use the re-timed one.
        # The 0.35-threshold calibration ("one soft defect passes only with
        # strong EP") assumes intent semantics; flipping this to True without
        # a state-conditioned EP reference biases EP low whenever the ego is slow
        self._ep_on_retimed = bool(ep_on_retimed)
        self._last_retime: Optional[Dict[str, Any]] = None
        # LK knobs (header note). None limit = LK off. The term also silently
        # stays off when the injected probe has no .details() (stale helper) —
        # soft["lk"] is simply never set, and the weighted average ignores it.
        if lk_deviation_limit_m is not None and float(lk_deviation_limit_m) <= 0.0:
            raise ValueError(f"lk_deviation_limit_m must be > 0, got {lk_deviation_limit_m!r}")
        if float(lk_horizon_window_sec) <= 0.0:
            raise ValueError(f"lk_horizon_window_sec must be > 0, got {lk_horizon_window_sec!r}")
        self._lk_deviation_limit_m = (
            None if lk_deviation_limit_m is None else float(lk_deviation_limit_m)
        )
        self._lk_horizon_window_sec = float(lk_horizon_window_sec)
        # Fast mode for high-cardinality candidate ranking: preserve the
        # collision/TTC trajectory checks, but evaluate HD-map DAC/LK only at
        # the final target point. This reduces a Bx31 map probe to Bx1 for the
        # standard six-pose, 0.1 s verifier configuration.
        self._hd_map_endpoint_only = bool(hd_map_endpoint_only)
        self._last_lk: Optional[Dict[str, Any]] = None
        self._lane_tier_probe = lane_tier_probe
        self._last_dac_tiers: Optional[np.ndarray] = None
        self.last_batch_endpoint_tiers: Optional[np.ndarray] = None
        self._ep_reference = ep_reference
        self._ep_reference_factory = ep_reference_factory
        self.last_result: Optional[VerificationResult] = None
        self._debug = bool(debug)
        # Diagnostic: per-frame dump of every VRU inside this radius, whether or
        # not it tripped a term. _emit_debug otherwise only records actors for
        # terms < 1.0, so a frame that scores 1.000 leaves no evidence of what
        # the verifier was actually looking at -- which is exactly the state the
        # 2026-08-19 low-speed pedestrian collisions land in (score 1.000,
        # throttle 1.00, contact). None = off (no cost on campaign runs).
        self._debug_vru_radius_m = (
            None if debug_vru_radius_m is None else float(debug_vru_radius_m)
        )
        self._debug_log_path = debug_log_path or "verifier_debug.jsonl"
        self._debug_fh = None
        self._step_index = 0
        self._episode_stats: Dict[str, Any] = {"steps": 0, "zero_steps": 0, "zeroed_by": {}}
        if self._debug:
            atexit.register(self._write_episode_summary)

    def _actor_debug_entry(self, collision: Dict[str, Any], current_info: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich a TTCVerifier collision record with z / on-road context."""
        entry = dict(collision)
        pos = collision.get("pos_world") or [0.0, 0.0, 0.0]
        ego_pos = current_info.get("ego_pos")
        if ego_pos is not None:
            entry["dz_vs_ego_m"] = float(pos[2]) - float(ego_pos[2])
        if self._lane_tier_probe is not None:
            try:
                tier = float(np.asarray(self._lane_tier_probe(np.asarray([pos])), dtype=np.float64)[0])
                entry["lane_tier"] = tier
                entry["on_road_surface"] = bool(tier > 0.0)
            except Exception:
                entry["on_road_surface"] = None
        return entry

    def _write_debug_record(self, record: Dict[str, Any]) -> None:
        if self._debug_fh is None:
            os.makedirs(os.path.dirname(os.path.abspath(self._debug_log_path)), exist_ok=True)
            self._debug_fh = open(self._debug_log_path, "a", buffering=1)
        self._debug_fh.write(json.dumps(record, default=float) + "\n")

    def _write_episode_summary(self) -> None:
        if not self._debug or self._episode_stats["steps"] == 0:
            return
        try:
            summary = {"kind": "episode_summary", **self._episode_stats}
            self._write_debug_record(summary)
            print("PDMS episode summary: %s" % json.dumps(summary, default=float))
        except Exception:
            LOG.exception("PDMS debug: failed to write episode summary")

    def _nearby_vru(self, current_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Every pedestrian-class agent within debug_vru_radius_m, ego-local.

        Answers the question the aggregate score cannot: was the actor in
        ``privileged_agents`` at all, and where did the verifier think it was?
        """
        out: List[Dict[str, Any]] = []
        agents = current_info.get("privileged_agents") or []
        radius = float(self._debug_vru_radius_m or 0.0)
        try:
            obstacles = self._ttc._obstacles_to_local(
                current_info,
                [a for a in agents
                 if str(a.get("object_type", "")).startswith("pedestrian")],
                min_walker_speed_mps=None,
                agent_forecast_mode="cv",
                forecast_horizon_sec=0.0,
            )
        except Exception:
            LOG.exception("PDMS debug: VRU dump failed")
            return out
        for ob in obstacles:
            dist = float(np.linalg.norm(ob.center))
            if dist > radius:
                continue
            out.append({
                "agent_id": ob.agent_id,
                "object_type": ob.object_type,
                "dist_m": round(dist, 2),
                # ego-local (right, forward): forward>0 is ahead of the ego
                "right_m": round(float(ob.center[0]), 2),
                "forward_m": round(float(ob.center[1]), 2),
                "speed_mps": round(float(np.linalg.norm(ob.velocity)), 2),
                "half_len_m": round(ob.half_len, 2),
                "half_width_m": round(ob.half_width, 2),
                "pos_world": ob.pos_world,
            })
        out.sort(key=lambda r: r["dist_m"])
        return out

    def _emit_debug(self, result: VerificationResult, current_info: Dict[str, Any]) -> None:
        submetrics = {**result.gates, **result.soft}
        zeroed_by = sorted(name for name, value in submetrics.items() if float(value) == 0.0)
        actors: List[Dict[str, Any]] = []
        for term in ("nc", "ttc"):
            if float(submetrics.get(term, 1.0)) < 1.0:
                for collision in self._ttc.last_collisions.get(term, []):
                    actors.append({"term": term, **self._actor_debug_entry(collision, current_info)})

        stats = self._episode_stats
        stats["steps"] += 1
        if result.score() == 0.0:
            stats["zero_steps"] += 1
        for name in zeroed_by:
            stats["zeroed_by"][name] = stats["zeroed_by"].get(name, 0) + 1

        record = {
            "kind": "step",
            "step": self._step_index,
            "frame_index": current_info.get("frame_index"),
            "timestamp": current_info.get("timestamp"),
            "score": result.score(),
            "gates": dict(result.gates),
            "soft": dict(result.soft),
            "ttc_raw": self._last_ttc_raw,
            "ttc_zero_streak": self._ttc_zero_streak,
            "ttc_debounced": bool(
                self._last_ttc_raw == 0.0 and float(result.soft.get("ttc", 1.0)) != 0.0
            ),
            "zeroed_by": zeroed_by,
            "retime": self._last_retime,
            "actors": actors,
            "ego_speed_mps": current_info.get("ego_velo"),
            "ego_pos": [float(v) for v in current_info.get("ego_pos", [])],
        }
        if self._last_dac_tiers is not None:
            tiers = self._last_dac_tiers
            record["dac_tiers"] = {
                "corridor": int((tiers == 1.0).sum()),
                "road_surface": int((tiers == 0.5).sum()),
                "off_road": int((tiers == 0.0).sum()),
            }
        if self._last_dac_exit is not None:
            record["dac_exit"] = dict(self._last_dac_exit)
        if self._last_lk is not None:
            record["lk"] = dict(self._last_lk)
        if self._debug_vru_radius_m is not None:
            record["vru"] = self._nearby_vru(current_info)
        try:
            self._write_debug_record(record)
        except Exception:
            LOG.exception("PDMS debug: failed to write step record")

        terms = " ".join(
            "%s=%.2f" % (name.upper(), float(submetrics[name]))
            for name in ("nc", "dac", "ttc", "c", "ep", "lk") if name in submetrics
        )
        actor_str = "".join(
            " | %s %s(%s) dz=%.1fm d=%.1fm %s" % (
                a["term"], a.get("object_type"), a.get("agent_id"),
                a.get("dz_vs_ego_m", float("nan")), a.get("distance_m", float("nan")),
                {True: "on-road", False: "OFF-ROAD", None: "road?"}[a.get("on_road_surface")],
            ) for a in actors
        )
        print("PDMS=%.3f | %s | zeroed_by=%s%s" % (result.score(), terms, zeroed_by, actor_str))
        self._step_index += 1

    def _all_pass(self) -> VerificationResult:
        # No plan evaluated: break any running TTC=0 streak (debounce counts
        # consecutive *evaluated* verifier steps).
        self._ttc_zero_streak = 0
        self._last_ttc_raw = None
        gates = {"nc": 1.0, "dac": 1.0}
        if self._vru_ttc_veto:
            gates["ttc_vru"] = 1.0
        result = VerificationResult(
            gates=gates,
            soft={"ttc": 1.0},  # "c" removed from aggregation (see PDMS_SOFT_WEIGHTS)
            weights=dict(PDMS_SOFT_WEIGHTS),
        )
        self.last_result = result
        return result

    def _resolve_ep_reference(self) -> Optional[EgoProgressReference]:
        if self._ep_reference is None and self._ep_reference_factory is not None:
            try:
                self._ep_reference = self._ep_reference_factory()
            except Exception:
                LOG.exception("EP reference factory failed; EP term disabled")
            self._ep_reference_factory = None
        return self._ep_reference

    def _drivable_area_compliance(
        self, world_traj: np.ndarray, tiers: Optional[np.ndarray] = None
    ) -> float:
        """DAC as a TIERED gate: the minimum lane tier touched by the plan.

        Per-sample tiers from the injected probe (see build_lane_tier_probe):
        1.0 = Driving/Parking corridor, 0.5 = non-driving road surface
        (shoulder/median/...), 0.0 = off-road or sidewalk. ``tiers`` may be
        passed precomputed (verify() shares one probe.details() pass between
        DAC and LK) — the probe is only queried when it is not.
        """
        if tiers is None:
            tiers = self._lane_tier_probe(world_traj)
        tiers = np.asarray(tiers, dtype=np.float64)
        self._last_dac_tiers = tiers
        self._last_dac_exit = None
        if len(tiers) == 0:
            return 1.0
        # Exit grace (dac_exit_grace): when the ego is ALREADY in violation
        # (sample 0 is the current position; tier < 1), forgive the leading
        # samples that are no worse than the current tier until the plan FIRST
        # returns to the corridor — punish entering/deepening violations,
        # never exiting them.
        if self._dac_exit_grace and tiers[0] < 1.0:
            exit_candidates = np.nonzero(tiers >= 1.0)[0]
            if len(exit_candidates) > 0:
                exit_idx = int(exit_candidates[0])
                if (tiers[:exit_idx] >= tiers[0]).all():
                    tiers = tiers.copy()
                    tiers[:exit_idx] = 1.0
                    self._last_dac_exit = {
                        "current_tier": float(self._last_dac_tiers[0]),
                        "exit_sample": exit_idx,
                    }
        # Gate horizon: only samples within the first dac_gate_horizon_sec of
        # plan time can veto (sample k is at t = k*timestep — honest time when
        # re-timing is on). Later tiers remain in _last_dac_tiers for debug.
        gate_tiers = tiers
        if self._dac_gate_horizon_sec is not None:
            n_gate = int(np.floor(self._dac_gate_horizon_sec / self._timestep)) + 1
            if n_gate < len(tiers):
                gate_tiers = tiers[:n_gate]
        tier = float(gate_tiers.min())
        if self._dac_mode == "binary":
            # NavSim-style hard gate: anything off the driving corridor vetoes.
            # (_last_dac_tiers keeps the raw tiers either way, so debug logs
            # still show what a tiered DAC would have said.)
            return 1.0 if tier == 1.0 else 0.0
        return tier

    def _lane_keeping(self, lane_details: Dict[str, np.ndarray]) -> float:
        """LK soft term: NavSim v2's _calculate_lane_keeping loop, Hydra-MDP++'s
        nearest-lane reference (see the header's LK note for the full lineage).

        Binary per plan: 0.0 iff the planar deviation from the nearest
        Driving-lane centerline exceeds lk_deviation_limit_m for at least
        ceil(lk_horizon_window_sec / timestep) CONSECUTIVE samples (4 at the
        0.5 s plan dt); a compliant sample resets the streak. Junction samples
        and samples with no driving lane are skipped WITHOUT resetting —
        NavSim's `continue`, so a violation window may span a junction.
        Stateless across verifier steps (the streak lives within one plan), so
        the scorer's per-candidate use needs no debounce-style reset.
        """
        devs = np.asarray(lane_details["lat_dev_m"], dtype=np.float64)
        skip = np.asarray(lane_details["lk_skip"], dtype=bool)
        required = max(1, int(np.ceil(self._lk_horizon_window_sec / self._timestep)))
        lk = 1.0
        streak = 0
        max_streak = 0
        for dev, skipped in zip(devs, skip):
            if skipped:
                continue
            if dev > self._lk_deviation_limit_m:
                streak += 1
                max_streak = max(max_streak, streak)
                if streak >= required:
                    lk = 0.0
            else:
                streak = 0
        self._last_lk = {
            "deviations_m": [round(float(d), 2) for d in devs],
            "skipped_samples": int(skip.sum()),
            "exceed_samples": int(((devs > self._lk_deviation_limit_m) & ~skip).sum()),
            "max_streak": int(max_streak),
            "required_streak": int(required),
        }
        return lk

    def _drivable_area_compliance_batch(self, tiers: np.ndarray) -> np.ndarray:
        """Vectorized-score counterpart of ``_drivable_area_compliance``."""
        tiers = np.asarray(tiers, dtype=np.float64)
        if tiers.ndim != 2:
            raise ValueError(f"tiers must have shape (B, T), got {tiers.shape}")
        if tiers.shape[1] == 0:
            return np.ones(len(tiers), dtype=np.float64)
        checked = tiers.copy()
        if self._dac_exit_grace:
            for row in checked:
                if row[0] >= 1.0:
                    continue
                exit_candidates = np.nonzero(row >= 1.0)[0]
                if len(exit_candidates) > 0:
                    exit_idx = int(exit_candidates[0])
                    if np.all(row[:exit_idx] >= row[0]):
                        row[:exit_idx] = 1.0
        if self._dac_gate_horizon_sec is not None:
            n_gate = int(np.floor(self._dac_gate_horizon_sec / self._timestep)) + 1
            checked = checked[:, : max(1, n_gate)]
        score = np.min(checked, axis=1)
        if self._dac_mode == "binary":
            score = np.where(score == 1.0, 1.0, 0.0)
        return score.astype(np.float64)

    def _lane_keeping_batch(self, lane_details: Dict[str, np.ndarray]) -> np.ndarray:
        """Batched LK state machine over candidate trajectories."""
        deviations = np.asarray(lane_details["lat_dev_m"], dtype=np.float64)
        skipped = np.asarray(lane_details["lk_skip"], dtype=bool)
        required = max(
            1, int(np.ceil(self._lk_horizon_window_sec / self._timestep))
        )
        streak = np.zeros(len(deviations), dtype=np.int64)
        failed = np.zeros(len(deviations), dtype=bool)
        for sample_idx in range(deviations.shape[1]):
            active = ~skipped[:, sample_idx]
            exceeded = deviations[:, sample_idx] > self._lk_deviation_limit_m
            streak = np.where(
                active,
                np.where(exceeded, streak + 1, 0),
                streak,
            )
            failed |= streak >= required
        return np.where(failed, 0.0, 1.0).astype(np.float64)

    def _endpoint_lane_keeping_batch(
        self, lane_details: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Endpoint-only LK used by the fast HD-map mode."""
        deviations = np.asarray(lane_details["lat_dev_m"], dtype=np.float64)
        skipped = np.asarray(lane_details["lk_skip"], dtype=bool)
        if deviations.ndim == 2:
            deviations = deviations[:, -1]
            skipped = skipped[:, -1]
        return np.where(
            skipped | (deviations <= self._lk_deviation_limit_m), 1.0, 0.0
        ).astype(np.float64)

    def _endpoint_lane_keeping(
        self, lane_details: Dict[str, np.ndarray]
    ) -> float:
        """Scalar counterpart of :meth:`_endpoint_lane_keeping_batch`."""
        deviations = np.asarray(lane_details["lat_dev_m"], dtype=np.float64)
        skipped = np.asarray(lane_details["lk_skip"], dtype=bool)
        deviation = float(deviations[-1])
        skip = bool(skipped[-1])
        self._last_lk = {
            "endpoint_deviation_m": round(deviation, 2),
            "endpoint_skipped": skip,
            "endpoint_only": True,
        }
        return float(skip or deviation <= self._lk_deviation_limit_m)

    @staticmethod
    def _reshape_unique_probe_values(
        values: Any,
        inverse: np.ndarray,
        batch_size: int,
        sample_count: int,
    ) -> np.ndarray:
        values = np.asarray(values)
        restored = values[inverse]
        return restored.reshape(
            (batch_size, sample_count) + tuple(values.shape[1:])
        )

    def score_batch(
        self,
        trajectories: Any,
        current_info: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Score independent same-frame trajectories in one NumPy batch.

        This path is intentionally stateless: it is for Round-1 candidate
        ranking only. The final verifier and recovery keep using ``score()`` so
        debounce, diagnostics, and recovery bookkeeping remain byte-for-byte on
        their existing scalar path.
        """
        plans = np.asarray(trajectories, dtype=np.float64)
        if plans.ndim != 3:
            raise ValueError(
                f"trajectories must have shape (B, T, 2), got {plans.shape}"
            )
        batch_size = len(plans)
        self.last_batch_endpoint_tiers = None
        if batch_size == 0:
            return np.zeros(0, dtype=np.float32)
        if current_info is None or plans.shape[1] < 2:
            return np.ones(batch_size, dtype=np.float32)
        # Temporal debounce is a final-decision concept and the candidate
        # scorer config fixes it to one. Fall back defensively if this method is
        # accidentally used with a stateful verifier configuration.
        if self._ttc_debounce_steps != 1:
            return np.asarray(
                [self.score(plan, current_info) for plan in plans],
                dtype=np.float32,
            )

        sampled = np.stack(
            [
                resample_plan_timestep(
                    plan,
                    input_timestep_sec=self._input_timestep_sec,
                    sample_timestep_sec=self._timestep,
                )
                for plan in plans
            ],
            axis=0,
        )
        nc, ttc, ttc_vru = self._ttc.verify_batch(sampled, current_info)
        traj = np.concatenate(
            [
                np.zeros((batch_size, 1, 2), dtype=np.float64),
                sampled[:, :, :2],
            ],
            axis=1,
        )
        traj_checked = traj
        if self._retime_accel_limit_mps2 is not None:
            current_speed = float(current_info.get("ego_velo") or 0.0)
            traj_checked = np.stack(
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

        world_checked = None
        world_ep = None
        try:
            pose = info_to_pose(current_info)
            origin = np.asarray(pose[:3, 3], dtype=np.float32)
            right = np.asarray(pose[:3, 1], dtype=np.float32)
            forward = np.asarray(pose[:3, 0], dtype=np.float32)

            def to_world(local: np.ndarray) -> np.ndarray:
                local = np.asarray(local, dtype=np.float32)
                return np.asarray(
                    origin[None, None, :]
                    + local[:, :, 0, None] * right[None, None, :]
                    + local[:, :, 1, None] * forward[None, None, :],
                    dtype=np.float32,
                )

            world_checked = to_world(traj_checked)
            world_ep = world_checked if self._ep_on_retimed else to_world(traj)
        except Exception:
            LOG.exception(
                "PDMS batch: failed to build world-frame plans; skipping DAC/EP"
            )

        dac = np.ones(batch_size, dtype=np.float64)
        lane_details = None
        if self._lane_tier_probe is not None and world_checked is not None:
            details_fn = (
                getattr(self._lane_tier_probe, "details", None)
                if self._lk_deviation_limit_m is not None
                else None
            )
            try:
                probe_world = (
                    world_checked[:, -1:, :]
                    if self._hd_map_endpoint_only
                    else world_checked
                )
                flat_world = probe_world.reshape(-1, probe_world.shape[-1])
                unique_world, inverse = np.unique(
                    flat_world, axis=0, return_inverse=True
                )
                if details_fn is not None:
                    unique_details = details_fn(unique_world)
                    lane_details = {
                        key: self._reshape_unique_probe_values(
                            value,
                            inverse,
                            batch_size,
                            probe_world.shape[1],
                        )
                        for key, value in unique_details.items()
                    }
                    tiers = np.asarray(lane_details["tiers"], dtype=np.float64)
                else:
                    unique_tiers = self._lane_tier_probe(unique_world)
                    tiers = self._reshape_unique_probe_values(
                        unique_tiers,
                        inverse,
                        batch_size,
                        probe_world.shape[1],
                    )
                dac = self._drivable_area_compliance_batch(tiers)
                self.last_batch_endpoint_tiers = np.asarray(
                    tiers[:, -1], dtype=np.float64
                )
            except Exception:
                LOG.exception(
                    "PDMS batch: lane tier probe failed; DAC passes this frame"
                )
                lane_details = None

        soft_terms: Dict[str, np.ndarray] = {"ttc": ttc}
        ep_reference = self._resolve_ep_reference()
        if ep_reference is not None and world_ep is not None:
            horizon_s = (traj.shape[1] - 1) * self._timestep
            batch_ep = getattr(ep_reference, "ego_progress_batch", None)
            if callable(batch_ep):
                soft_terms["ep"] = np.asarray(
                    batch_ep(world_ep[:, :, :2], horizon_s), dtype=np.float64
                )
            else:
                soft_terms["ep"] = np.asarray(
                    [
                        ep_reference.ego_progress(candidate[:, :2], horizon_s)
                        for candidate in world_ep
                    ],
                    dtype=np.float64,
                )
        if lane_details is not None:
            try:
                soft_terms["lk"] = (
                    self._endpoint_lane_keeping_batch(lane_details)
                    if self._hd_map_endpoint_only
                    else self._lane_keeping_batch(lane_details)
                )
            except Exception:
                LOG.exception("PDMS batch: LK term failed; LK omitted")

        numerator = np.zeros(batch_size, dtype=np.float64)
        denominator = 0.0
        for name, values in soft_terms.items():
            weight = float(PDMS_SOFT_WEIGHTS.get(name, 1.0))
            numerator += weight * np.asarray(values, dtype=np.float64)
            denominator += weight
        soft_score = (
            numerator / denominator
            if denominator > 0.0
            else np.ones(batch_size, dtype=np.float64)
        )
        # The veto has to apply here too, not just in verify(): B5 parameter
        # preverification thresholds these batch scores directly (and falls
        # back to the scalar score() when the shapes differ), so a batch-only
        # exemption would let it commit a swerve variant that the very next
        # verify() vetoes.
        gates = nc * dac * ttc_vru if self._vru_ttc_veto else nc * dac
        return (gates * soft_score).astype(np.float32)

    def verify(self, trajectory: Any, current_info: Optional[Dict[str, Any]] = None) -> VerificationResult:
        if trajectory is None or current_info is None:
            return self._all_pass()
        plan = np.asarray(trajectory, dtype=np.float64)
        if plan.ndim != 2 or len(plan) < 2:
            return self._all_pass()
        plan = resample_plan_timestep(
            plan,
            input_timestep_sec=self._input_timestep_sec,
            sample_timestep_sec=self._timestep,
        )

        # NC gate + TTC soft term: delegate to the ported OBB sweep.
        ttc_result = self._ttc.verify(plan, current_info)
        nc = float(ttc_result.gates.get("nc", 1.0))
        ttc = float(ttc_result.soft.get("ttc", 1.0))
        # TTC debounce (header note: "reject only when TTCg=0 persists for k
        # consecutive verifier steps"): a TTC=0 probe hit only enters the
        # aggregation once it has persisted for ttc_debounce_steps consecutive
        # verifier steps; shorter streaks are suppressed to 1.0. The raw value
        # stays visible in the debug stream (ttc_raw / ttc_zero_streak).
        # ttc_debounce_steps=1 (default) preserves original behaviour.
        self._last_ttc_raw = ttc
        if ttc == 0.0:
            self._ttc_zero_streak += 1
            if self._ttc_zero_streak < self._ttc_debounce_steps:
                ttc = 1.0
        else:
            self._ttc_zero_streak = 0
        # See __init__: a surviving TTC=0 whose gating conflict is a VRU
        # becomes a hard gate instead of one soft term among three.
        ttc_vru = (
            0.0
            if (
                self._vru_ttc_veto
                and ttc == 0.0
                and gating_vru_conflict(self._ttc.last_collisions.get("ttc", ()))
            )
            else 1.0
        )

        # Ego samples with the current pose prepended (plan is ego-local).
        traj = np.concatenate([np.zeros((1, 2)), plan[:, :2]], axis=0)
        # Feasibility re-timing for the safety terms (DAC here; NC/TTC do the
        # same inside TTCVerifier with the shared knob). EP keeps the raw plan
        # unless ep_on_retimed is set — see __init__ note.
        traj_checked = traj
        self._last_retime = None
        if self._retime_accel_limit_mps2 is not None:
            v0 = float(current_info.get("ego_velo") or 0.0)
            traj_checked = retime_plan_polyline(
                traj,
                v0,
                self._retime_accel_limit_mps2,
                self._timestep,
                self._retime_brake_limit_mps2,
            )
            raw_reach = float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum())
            checked_reach = float(np.linalg.norm(np.diff(traj_checked, axis=0), axis=1).sum())
            self._last_retime = {
                "v0_mps": v0,
                "raw_reach_m": round(raw_reach, 3),
                "checked_reach_m": round(checked_reach, 3),
                "clamped": bool(not np.allclose(traj_checked, traj, atol=1e-6)),
            }

        # Comfort removed from the aggregation for now (see PDMS_SOFT_WEIGHTS
        # note). Restore by uncommenting (and the import at the top):
        # headings = _traj_headings(traj)
        # yaw = np.arctan2(headings[:, 1], headings[:, 0])
        # comfort = is_comfortable(np.column_stack([traj, yaw]), self._timestep)

        # World-frame polyline for DAC / EP (needs ego_pos / ego_rot in info).
        world_traj = None      # re-timed polyline (DAC)
        world_traj_ep = None   # EP polyline (raw plan unless ep_on_retimed)
        try:
            pose = info_to_pose(current_info)
            world_traj = local_plan_to_world(traj_checked, pose)
            if self._ep_on_retimed or traj_checked is traj:
                world_traj_ep = world_traj
            else:
                world_traj_ep = local_plan_to_world(traj, pose)
        except Exception:
            LOG.exception("PDMS: failed to build world-frame plan; skipping DAC/EP")

        dac = 1.0
        self._last_dac_tiers = None
        self._last_lk = None
        lane_details = None
        if self._lane_tier_probe is not None and world_traj is not None:
            # One probe pass serves both DAC (tiers) and LK (deviations) when
            # the probe exposes .details(); a plain tier probe keeps the old
            # DAC-only path and LK stays silently off (fail-open).
            details_fn = (
                getattr(self._lane_tier_probe, "details", None)
                if self._lk_deviation_limit_m is not None else None
            )
            try:
                probe_world = (
                    world_traj[-1:, :]
                    if self._hd_map_endpoint_only
                    else world_traj
                )
                if details_fn is not None:
                    lane_details = details_fn(probe_world)
                    dac = self._drivable_area_compliance(
                        probe_world, tiers=lane_details["tiers"]
                    )
                else:
                    dac = self._drivable_area_compliance(probe_world)
            except Exception:
                LOG.exception("PDMS: lane tier probe failed; DAC passes this frame")
                dac = 1.0
                lane_details = None

        soft = {"ttc": ttc}  # "c": comfort — removed from aggregation for now
        ep_reference = self._resolve_ep_reference()
        if ep_reference is not None and world_traj_ep is not None:
            # traj[0] is the prepended current pose, so the plan spans
            # (len - 1) * timestep seconds.
            horizon_s = (len(traj) - 1) * self._timestep
            soft["ep"] = ep_reference.ego_progress(world_traj_ep[:, :2], horizon_s)
        if lane_details is not None:
            try:
                soft["lk"] = (
                    self._endpoint_lane_keeping(lane_details)
                    if self._hd_map_endpoint_only
                    else self._lane_keeping(lane_details)
                )
            except Exception:
                LOG.exception("PDMS: LK term failed; LK omitted this frame")
                self._last_lk = None

        gates = {"nc": nc, "dac": dac}
        if self._vru_ttc_veto:
            # Only present when the knob is on, so debug records / the
            # "nc=.. dac=.." step log stay byte-identical for existing runs.
            gates["ttc_vru"] = ttc_vru
        result = VerificationResult(
            gates=gates,
            soft=soft,
            weights=dict(PDMS_SOFT_WEIGHTS),
        )
        self.last_result = result
        if self._debug:
            try:
                self._emit_debug(result, current_info)
            except Exception:
                LOG.exception("PDMS debug: instrumentation failed (run unaffected)")
        return result
