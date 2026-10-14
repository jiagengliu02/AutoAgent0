"""Road-context and geometric safety checks for recovery candidates."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from agenticdriving.adapters.carla.geometry import (
    info_to_pose,
    local_plan_to_world,
    world_points_to_current_local,
)


class RecoverySafetyMixin:
    """Methods shared by recovery selection and hold execution."""
    def _hd_map_lane_tier_probe(self):
        """Return the shared map probe, independent of verifier enablement."""
        probe = getattr(self, "_lane_tier_probe", None)
        if probe is not None:
            return probe
        return getattr(getattr(self, "verifier", None), "_lane_tier_probe", None)

    def _ego_on_penalized_surface(self, info: Dict[str, Any]) -> bool:
        """True when the ego's own position sits on a lane-tier <= 0.5 surface.

        Exit grace for the tier-0.5 hardening checks: when the ego is already
        on a shoulder-class (or worse) surface, escaping maneuvers necessarily
        traverse tier <= 0.5 ground, so the hardening vetoes are suspended and
        plan acceptance falls back to the verifier (whose dac_exit_grace
        handles the same situation). Fails open (False -> checks stay active)
        only when the probe or the ego position is unavailable."""
        probe = self._hd_map_lane_tier_probe()
        if probe is None:
            return False
        try:
            pos = np.asarray(info.get("ego_pos"), dtype=np.float64).reshape(1, -1)
            if pos.shape[1] == 2:
                pos = np.hstack([pos, np.zeros((1, 1))])
            tier = float(np.asarray(probe(pos[:, :3]), dtype=np.float64)[0])
        except Exception:
            return False
        return tier <= 0.5

    _ROAD_CONTEXT_SECTORS = (
        "ahead", "front-right", "right", "rear-right",
        "behind", "rear-left", "left", "front-left",
    )

    def _road_context_enabled(self) -> bool:
        """True when the road-context arm is on (recovery_reason.road_context)."""
        return (
            self.recovery_explainer is not None
            and bool(getattr(self.recovery_explainer.cfg, "road_context", False))
        )

    def _recovery_road_context(self, info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Road-context facts for the recovery prompts (None when unavailable).

        ``ego_on_corridor``: lane tier 1.0 at the ego's own position. When the
        ego is off the corridor and the probe finds a nearest driving-lane
        centerline, adds ``nearest_lane_distance_m`` and an 8-sector
        ``nearest_lane_direction`` in the ego frame. Fails open (None) without
        a probe/map — prompts then render exactly as the non-road-context arm.
        """
        if not self._road_context_enabled():
            return None
        probe = self._hd_map_lane_tier_probe()
        details = getattr(probe, "details", None)
        if probe is None or details is None:
            return None
        try:
            pos = np.asarray(info.get("ego_pos"), dtype=np.float64).reshape(1, -1)
            if pos.shape[1] == 2:
                pos = np.hstack([pos, np.zeros((1, 1))])
            d = details(pos[:, :3])
            tier = float(np.asarray(d["tiers"], dtype=np.float64)[0])
        except Exception:
            self.log.debug("road-context probe failed", exc_info=True)
            return None
        context: Dict[str, Any] = {
            "ego_lane_tier": tier,
            "ego_on_corridor": tier >= 1.0,
        }
        if context["ego_on_corridor"]:
            return context
        lane_center = np.asarray(
            d.get("lane_center_xy", np.full((1, 2), np.nan)), dtype=np.float64
        )[0]
        if not np.all(np.isfinite(lane_center)):
            return context
        try:
            target_world = np.array(
                [[lane_center[0], lane_center[1], float(pos[0, 2])]], dtype=np.float64
            )
            local = world_points_to_current_local(target_world, info_to_pose(info))[0]
        except Exception:
            self.log.debug("road-context ego-frame conversion failed", exc_info=True)
            return context
        x_right, y_forward = float(local[0]), float(local[1])
        distance = float(np.hypot(x_right, y_forward))
        # Bearing 0 = straight ahead, positive clockwise (to the right);
        # 45-degree sectors centered on the 8 compass labels.
        bearing_deg = float(np.degrees(np.arctan2(x_right, y_forward)))
        sector = int(np.floor((bearing_deg + 22.5) / 45.0)) % 8
        context["nearest_lane_distance_m"] = round(distance, 1)
        context["nearest_lane_direction"] = self._ROAD_CONTEXT_SECTORS[sector]
        return context

    def _candidate_ends_on_driving_lane(
        self, traj_local, info: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Road relationship of the candidate rollout's endpoint (None = unknown).

        Returns ``{"ends_on_driving_lane": bool}``, plus — in the graded v2 arm
        (``recovery_reason.road_context_distance``) — ``"distance_to_lane_m"``
        (endpoint's planar distance to the nearest driving-lane centerline) for
        off-lane endpoints, so the VLM can rank candidates when none reaches
        the lane. Map/geometry context only — deliberately NOT a verifier score
        (the prompt's no-scores design principle). For parameterized primitives
        the caller materializes the nominal rollout, so the values reflect the
        midpoint parameter.
        """
        results = self._candidate_ends_on_driving_lane_batch([traj_local], info)
        return results[0] if results else None

    def _candidate_ends_on_driving_lane_batch(
        self, trajectories: Sequence[np.ndarray], info: Dict[str, Any]
    ) -> List[Optional[Dict[str, Any]]]:
        """Return endpoint road flags with one shared HD-map query."""
        count = len(trajectories)
        if not self._road_context_enabled():
            return [None] * count
        probe = self._hd_map_lane_tier_probe()
        if probe is None:
            return [None] * count
        try:
            endpoints = np.asarray(
                [np.asarray(traj, dtype=np.float32)[-1] for traj in trajectories],
                dtype=np.float32,
            )
            world = np.asarray(
                local_plan_to_world(endpoints, info_to_pose(info)), dtype=np.float64
            )
        except Exception:
            return [None] * count
        if world.ndim != 2 or len(world) == 0 or world.shape[1] < 2:
            return [None] * count
        if world.shape[1] == 2:
            world = np.hstack([world, np.zeros((len(world), 1))])
        graded = bool(
            getattr(self.recovery_explainer.cfg, "road_context_distance", False)
        )
        details = getattr(probe, "details", None)
        try:
            if graded and details is not None:
                d = details(world[:, :3])
                tiers = np.asarray(d["tiers"], dtype=np.float64)
                lk_skip = np.asarray(d["lk_skip"], dtype=bool)
                deviations = np.asarray(d["lat_dev_m"], dtype=np.float64)
                flags: List[Optional[Dict[str, Any]]] = []
                for idx, tier in enumerate(tiers):
                    flag: Dict[str, Any] = {
                        "ends_on_driving_lane": float(tier) >= 1.0
                    }
                    if tier < 1.0 and not bool(lk_skip[idx]):
                        flag["distance_to_lane_m"] = round(float(deviations[idx]), 1)
                    flags.append(flag)
                return flags
            tiers = np.asarray(probe(world[:, :3]), dtype=np.float64)
        except Exception:
            return [None] * count
        return [
            {"ends_on_driving_lane": float(tier) >= 1.0} for tier in tiers
        ]

    def _primitive_tier_gate_fails(self, traj_local, info: Dict[str, Any]) -> bool:
        """Full-maneuver DAC-tier check for primitive trials.

        The verifier's DAC gate binarizes only the first dac_gate_horizon_sec
        of a plan; a 12-pose primitive committed at creep speed is therefore
        gated on centimeters of its head. This probes the CENTERLINE of every
        materialized pose and fails the trial when any sample lands on
        0 < tier <= 0.5 (Shoulder/Median/Border/... — paved but
        Fail2Drive-penalized). Tier 0.0 is deliberately NOT handled here: it
        stays with _footprint_offroad's ground-probe logic so the
        construction-dirt bypass allowance is preserved. Suspended while the
        ego itself is on a penalized surface (exit grace). Fails open."""
        # The endpoint-only recovery mode has already applied its complete
        # configured map policy through _footprint_offroad. Avoid re-probing
        # the centerline here.
        if bool(getattr(self, "recovery_hd_map_endpoint_only", False)):
            return False
        probe = self._hd_map_lane_tier_probe()
        if probe is None:
            return False
        try:
            ego_pose = info_to_pose(info)
            world = np.asarray(
                local_plan_to_world(
                    np.asarray(traj_local, dtype=np.float32), ego_pose
                ),
                dtype=np.float64,
            )
        except Exception:
            return False
        if world.ndim != 2 or len(world) == 0 or world.shape[1] < 2:
            return False
        if world.shape[1] == 2:
            world = np.hstack([world, np.zeros((len(world), 1))])
        try:
            tiers = np.asarray(probe(world), dtype=np.float64)
        except Exception:
            return False
        penalized = (tiers > 0.0) & (tiers <= 0.5)
        if not penalized.any():
            return False
        return not self._ego_on_penalized_surface(info)

    def _footprint_offroad(
        self,
        traj_local,
        info: Dict[str, Any],
        *,
        allow_tier_half_crossing: bool = False,
    ) -> bool:
        """Hard HD-map corridor gate for a complete vehicle trajectory.

        Every sampled center point and both footprint sides must remain in a
        Driving/Parking corridor (lane tier 1.0). Shoulder/median surfaces
        (tier 0.5) and unmapped/off-road surfaces (tier 0.0) are both rejected,
        even when physical ground exists there. Recovery paths use this strict
        gate; the normal Loop-1 hot path uses the endpoint-only batch gate.

        ``allow_tier_half_crossing`` is retained only for caller compatibility;
        the hard gate has no bypass exception. Without an HD-map probe the
        check fails open because corridor membership is unknowable."""
        rejected = self._footprint_offroad_batch(
            np.asarray(traj_local, dtype=np.float32)[None, ...],
            info,
            allow_tier_half_crossing=allow_tier_half_crossing,
        )
        return bool(rejected[0]) if len(rejected) else False

    def _verified_trajectory_offroad(
        self,
        traj_local,
        info: Dict[str, Any],
        *,
        allow_tier_half_crossing: bool = False,
    ) -> bool:
        """Reuse the verifier's endpoint DAC instead of querying the map twice."""
        if bool(getattr(self, "recovery_hd_map_endpoint_only", False)):
            result = getattr(getattr(self, "verifier", None), "last_result", None)
            gates = getattr(result, "gates", None)
            if gates is not None and "dac" in gates:
                return float(gates["dac"]) < 1.0
        if allow_tier_half_crossing:
            return self._footprint_offroad(
                traj_local,
                info,
                allow_tier_half_crossing=True,
            )
        return self._footprint_offroad(traj_local, info)

    def _endpoint_offroad_batch(
        self,
        trajectories,
        info: Dict[str, Any],
    ) -> np.ndarray:
        """Check only each trajectory's final target point in one map probe.

        This is the fast Loop-1 gate: one point per shortlisted proposal, with
        exact duplicate endpoints collapsed before probing. Recovery paths use
        the stricter full-trajectory footprint gate below.
        """
        plans = np.asarray(trajectories, dtype=np.float32)
        if plans.ndim == 2:
            plans = plans[None, ...]
        if plans.ndim != 3:
            return np.zeros(0, dtype=bool)

        rejected = np.zeros(len(plans), dtype=bool)
        if len(plans) == 0 or plans.shape[1] == 0:
            return rejected

        probe = self._hd_map_lane_tier_probe()
        if probe is None:
            return rejected
        try:
            endpoints_world = np.asarray(
                local_plan_to_world(plans[:, -1, :], info_to_pose(info)),
                dtype=np.float64,
            )
            if endpoints_world.shape[1] == 2:
                endpoints_world = np.hstack(
                    [endpoints_world, np.zeros((len(endpoints_world), 1))]
                )
            unique_points, inverse = np.unique(
                endpoints_world[:, :3], axis=0, return_inverse=True
            )
            tiers = np.asarray(probe(unique_points), dtype=np.float64)
            rejected = tiers[inverse] < 1.0
        except Exception:
            self.log.debug("endpoint HD-map batch probe failed", exc_info=True)
            return np.zeros(len(plans), dtype=bool)
        return np.asarray(rejected, dtype=bool)

    def _footprint_offroad_batch(
        self,
        trajectories,
        info: Dict[str, Any],
        *,
        allow_tier_half_crossing: bool = False,
    ) -> np.ndarray:
        """Check a batch of trajectories with one HD-map probe invocation.

        This strict recovery-path helper concatenates every center/footprint
        sample, removes exact duplicates, and invokes the lane probe once.
        The scalar wrapper also routes through here for consistent semantics.

        ``allow_tier_half_crossing`` remains a compatibility-only argument;
        the hard corridor gate rejects every tier below 1.0.
        """
        plans = np.asarray(trajectories, dtype=np.float32)
        if plans.ndim == 2:
            plans = plans[None, ...]
        if plans.ndim != 3:
            return np.zeros(0, dtype=bool)

        if bool(getattr(self, "recovery_hd_map_endpoint_only", False)):
            return self._endpoint_offroad_batch(plans, info)

        rejected = np.zeros(len(plans), dtype=bool)
        if len(plans) == 0:
            return rejected

        probe = self._hd_map_lane_tier_probe()
        if probe is None:
            return rejected
        try:
            ego_pose = info_to_pose(info)
        except Exception:
            return rejected

        sample_batches = []
        sample_owners = []
        for plan_index, traj_local in enumerate(plans):
            try:
                world = np.asarray(
                    local_plan_to_world(traj_local, ego_pose), dtype=np.float64
                )
            except Exception:
                continue
            if world.ndim != 2 or len(world) == 0 or world.shape[1] < 2:
                continue
            if world.shape[1] == 2:
                world = np.hstack([world, np.zeros((len(world), 1))])

            # Probe segments as well as native poses so a sparse curved plan
            # cannot cut across a lane boundary between endpoints.
            if len(world) > 1:
                dense = [world[0]]
                for start, end in zip(world[:-1], world[1:]):
                    steps = max(
                        1,
                        int(
                            np.ceil(
                                np.linalg.norm(end[:2] - start[:2]) / 0.5
                            )
                        ),
                    )
                    dense.extend(
                        start + (end - start) * (index / steps)
                        for index in range(1, steps + 1)
                    )
                world = np.asarray(dense, dtype=np.float64)

            d = np.diff(world[:, :2], axis=0)
            dirs = (
                np.asarray([ego_pose[:2, 0]], dtype=np.float64)
                if len(d) == 0
                else np.vstack([d, d[-1:]])
            )
            norms = np.linalg.norm(dirs, axis=1, keepdims=True)
            valid = norms[:, 0] > 1e-6
            for index in range(1, len(dirs)):
                if not valid[index] and valid[index - 1]:
                    dirs[index] = dirs[index - 1]
                    valid[index] = True
            for index in range(len(dirs) - 2, -1, -1):
                if not valid[index] and valid[index + 1]:
                    dirs[index] = dirs[index + 1]
                    valid[index] = True
            if not valid.all():
                dirs[~valid] = np.asarray(ego_pose[:2, 0], dtype=np.float64)
            norms = np.linalg.norm(dirs, axis=1, keepdims=True)
            dirs = dirs / np.maximum(norms, 1e-9)
            lateral = np.stack([-dirs[:, 1], dirs[:, 0]], axis=1)

            half_width_m = 1.2
            edges = []
            for sign in (1.0, -1.0):
                off = world.copy()
                off[:, :2] = world[:, :2] + sign * half_width_m * lateral
                edges.append(off)
            samples = np.vstack([world, *edges])
            sample_batches.append(samples)
            sample_owners.append(
                np.full(len(samples), plan_index, dtype=np.int64)
            )

        if not sample_batches:
            return rejected

        all_samples = np.vstack(sample_batches)
        owners = np.concatenate(sample_owners)
        try:
            unique_samples, inverse = np.unique(
                all_samples, axis=0, return_inverse=True
            )
            unique_tiers = np.asarray(probe(unique_samples), dtype=np.float64)
            tiers = unique_tiers[inverse]
        except Exception:
            return rejected
        outside = tiers < 1.0
        for plan_index in np.unique(owners[outside]):
            plan_mask = owners == plan_index
            plan_outside = outside & plan_mask
            rejected[int(plan_index)] = True
            self.log.info(
                "HD-map hard filter: %d/%d center/footprint samples leave "
                "Driving/Parking corridors -> veto",
                int(plan_outside.sum()), int(plan_mask.sum()),
            )
        return rejected

    def _route_station(self, info: Dict[str, Any]) -> Optional[float]:
        """Ego's arc-length station (m) along the verifier's route reference.

        Returns None when no EP reference is available (no route / non-PDMS
        verifier); the no-progress tabu is disabled in that case."""
        resolve = getattr(self.verifier, "_resolve_ep_reference", None)
        if resolve is None:
            return None
        try:
            ref = resolve()
            if ref is None:
                return None
            ego_xy = np.asarray(info["ego_pos"], dtype=np.float64)[:2]
            return float(ref.station(ego_xy))
        except Exception:
            return None
