"""Commit-and-hold recovery lifecycle.

The mixin keeps the public Orchestrator method surface stable while isolating
world-anchored maneuver execution from the per-frame orchestration flow.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from agenticdriving.planners.fallback_rule_based.provider import (
    RECOVERY_PLAN_DT_SEC,
    recovery_action_fixed_duration_sec,
)

if TYPE_CHECKING:
    from agenticdriving.orchestrator import StepResult


class RecoveryHoldMixin:
    """Begin, advance, guard, and finish committed recovery maneuvers."""

    def _recovery_hold_intended_tier(self, hold, prog, ahead, ego_tier):
        """Preserve the default DAC reference; planners may align its time."""
        return float(ahead[0]) if len(ahead) else ego_tier

    def _recovery_hold_safety_plan(self, trajectory):
        """Keep the existing guard geometry unless a planner specializes it."""
        plan = np.asarray(trajectory, dtype=np.float32)
        if plan.ndim == 2 and plan.shape[1] >= 2:
            origin = np.zeros((1, plan.shape[1]), dtype=np.float32)
            if len(plan) == 0 or np.linalg.norm(plan[0, :2]) > 1e-3:
                plan = np.vstack([origin, plan])
        return plan

    # ------------------------------------------------- commit-and-hold recovery
    # Ladder key of the guaranteed last-resort maneuver. It is never offered
    # among the normal commit-and-hold trials; instead it is forced when every
    # other primitive fails the verifier, so recovery never hands control back
    # to loop 1 while the situation is unresolved.
    _RECOVERY_BRAKE_PRIMITIVE = "brake"

    # Anti-deadlock: after this many CONSECUTIVE forced-brake commits with the
    # ego parked (see _recover_verify_first_pass), commit the best collision-free
    # primitive even though it failed the verifier threshold. 2 commits at the
    # Consecutive forced-brake commits trigger the first escape attempt.
    _RECOVERY_ESCALATE_AFTER = 2

    # No-progress tabu (see _recover_verify_first_pass): a committed primitive
    # that advanced the route station by less than MIN_GAIN by the next commit
    # cycle is excluded from the following TABU_CYCLES cycles.
    _RECOVERY_TABU_MIN_GAIN_M = 2.0
    _RECOVERY_TABU_CYCLES = 3

    def _settle_lane_bypass_intent(
        self,
        hold: Dict[str, Any],
        *,
        completed: bool,
    ) -> None:
        """Clear a completed bypass latch, or retain it after an abort."""
        intent = getattr(self, "_lane_bypass_intent", None)
        if intent is None or str(intent.get("action")) != str(hold.get("name")):
            return
        if completed and bool(hold.get("preserve_lane_bypass_intent", False)):
            intent["committed"] = False
            intent["last_gate"] = "internal_retreat_complete"
            self.log.info(
                "latched lane bypass '%s' internal retreat completed; "
                "keeping the latch for the forward bypass",
                hold.get("name"),
            )
            return
        if completed:
            self._lane_bypass_intent = None
            self.log.info(
                "latched lane bypass '%s' completed; normal planning may resume",
                hold.get("name"),
            )
        else:
            intent["committed"] = False
            intent["last_gate"] = "hold_aborted_retry"

    def _lane_bypass_return_guard_context(
        self,
        *,
        name: str,
        info: Dict[str, Any],
        committed: "np.ndarray",
        states: Any,
    ) -> tuple:
        """Locate the bypass's return blend and the lane it merges back into.

        The exclusive bypass runs open-loop (see ``exclusive_lane_bypass``), and
        its admission gate only ever looks at the BORROWED lane: both
        ``lane_bypass_gap_assessment``'s target-lane membership test and
        ``_lane_bypass_bbox_clearance_m``'s ``speed > 2`` / ``front <= 0``
        filters drop every moving vehicle in the route lane. The return leg
        therefore merges back into a slot nothing has checked. On the c5 200-
        route run that produced 4 of the 9 holds whose first post-hold PDMS
        tick read NC=0 with a vehicle 4.5-5.1 m dead ahead (routes 27/80/
        1028/1080); 3 of those routes then collided.

        Returns ``(return_prog, source_lane_keys)``, or ``(None, None)`` when
        the return phase or the route-lane identity cannot be established --
        the guard then stays off and the hold behaves exactly as before.

        ``return_prog`` is read off the committed plan's own geometry rather
        than plumbed down from ``_solve_...``'s ``return_start_step``: the
        materializer hands back a bare ``(trajectory, states)`` pair shared with
        every codegen primitive, so widening it would ripple through unrelated
        call sites. The lateral profile is exact for this purpose -- it holds
        the borrowed centreline at full magnitude until the return blend starts,
        so the LAST row still at the plateau is the return row.
        """
        direction = str(name)[len("lane_bypass_"):]
        descriptor = (info.get("adjacent_lane_flows") or {}).get(direction)
        if not isinstance(descriptor, dict):
            return None, None
        source_lane_keys = set()
        for value in descriptor.get("source_lane_keys") or ():
            try:
                items = tuple(int(item) for item in value)
            except (TypeError, ValueError):
                continue
            if len(items) == 3:
                source_lane_keys.add(items)
        if not source_lane_keys:
            return None, None

        try:
            plan = np.asarray(committed, dtype=np.float64)
            lateral = np.abs(plan[:, 0])
        except (TypeError, ValueError, IndexError):
            return None, None
        if lateral.size < 2:
            return None, None
        # Ignore the executor padding rows ([speed, steer] == [0, 0]) that the
        # materializer appends to reach output_num_poses; they repeat the final
        # pose and would drag the plateau search past the real maneuver.
        last_active = lateral.size - 1
        if states is not None:
            try:
                state_array = np.asarray(states, dtype=np.float64)
                active_rows = np.flatnonzero(
                    np.any(np.abs(state_array[:, :2]) > 1e-4, axis=1)
                )
                if active_rows.size:
                    last_active = min(last_active, int(active_rows[-1]))
            except (TypeError, ValueError, IndexError):
                pass
        lateral = lateral[: last_active + 1]
        if lateral.size < 2:
            return None, None
        peak = float(lateral.max())
        # Below half a metre no lane was actually borrowed (the retreat phase,
        # or a degenerate plan). There is no return blend to guard.
        if peak < 0.5:
            return None, None
        plateau = np.flatnonzero(lateral >= 0.95 * peak)
        return_prog = int(plateau[-1])
        if return_prog <= 0 or return_prog >= last_active:
            # The plan never comes back inside its active rows; guarding a
            # "return" that does not exist would just re-enable the blanket
            # per-tick NC recheck this exemption exists to avoid.
            return None, None
        return return_prog, frozenset(source_lane_keys)

    def _lane_bypass_guard_agents(
        self,
        hold: Dict[str, Any],
        privileged_info: List[Dict[str, Any]],
        *,
        include_route_lane_vehicles: bool,
    ) -> Optional[List[Dict[str, Any]]]:
        """Live actors that must remain guarded during an exclusive bypass.

        Pedestrians are checked over every phase, irrespective of lane
        membership. A walker may be stationary when the bypass is committed
        and begin crossing during the open-loop entry/pass phase. Static props
        are checked throughout the maneuver, including the intended route
        blocker. Vehicles remain excluded until the return blend, then only
        route-lane vehicles are added because that merge slot was not covered
        by the entry gate.

        ``None`` means "nothing to check": the caller then skips the verifier
        call entirely instead of scoring against an empty obstacle set.
        """
        keys = hold.get("lane_bypass_source_lane_keys") or ()
        selected: List[Dict[str, Any]] = []
        for agent in privileged_info:
            if not isinstance(agent, dict):
                continue
            object_type = str(agent.get("object_type", "vehicle"))
            if object_type == "pedestrian":
                selected.append(agent)
                continue
            if object_type == "static":
                selected.append(agent)
                continue
            # Detector boxes for parked props/walls can arrive as vehicles;
            # once stationary memory confirms them they are static obstacles
            # and must remain guarded during the entry/pass phases too.
            if object_type == "vehicle" and agent.get("stationary_memory_confirmed"):
                selected.append(agent)
                continue
            if object_type != "vehicle" or not include_route_lane_vehicles:
                continue
            raw_key = agent.get("agent_lane_key")
            if raw_key is None:
                continue
            try:
                agent_key = tuple(int(item) for item in raw_key)
            except (TypeError, ValueError):
                continue
            if agent_key in keys:
                selected.append(agent)
        return selected or None

    def _recovery_emergency_brake_result(
        self,
        hold: Dict[str, Any],
        *,
        info: Dict[str, Any],
        guard: str,
        details: Dict[str, Any],
    ) -> StepResult:
        """Latch and return a controller-ready minimum-risk brake maneuver.

        Ending a hold and returning ``None`` lets :meth:`Orchestrator.step`
        enter recovery selection before it produces a new control.  That is
        acceptable for ordinary replanning, but it wastes the only useful
        reaction tick after a live collision guard fires.  A zero-speed signed
        recovery state is understood by every CARLA adapter as an explicit
        brake command.

        This is deliberately a real fixed-duration hold rather than the old
        one-tick placeholder.  An emergency stop must not hand control back to
        Loop 1 on the next frame merely because the zero-speed brake trajectory
        still overlaps the obstacle that caused the emergency.
        """
        duration = recovery_action_fixed_duration_sec(
            self._RECOVERY_BRAKE_PRIMITIVE
        )
        duration = 1.0 if duration is None else float(duration)
        plan_dt = float(RECOVERY_PLAN_DT_SEC)
        plan_len = max(2, int(np.ceil(duration / max(plan_dt, 1e-3))))
        plan = np.zeros((plan_len, 2), dtype=np.float32)
        states = np.zeros((plan_len, 2), dtype=np.float32)
        try:
            world_plan = np.asarray(
                self._local_plan_to_world(plan, self._info_to_pose(info)),
                dtype=np.float32,
            )
        except Exception:
            # Projection should be available whenever a normal recovery hold
            # was active.  Keep braking fail-safe even with incomplete pose
            # metadata by anchoring the stop at the supplied ego position.
            ego_pos = np.asarray(
                info.get("ego_pos", (0.0, 0.0, 0.0)), dtype=np.float32
            ).reshape(-1)
            if ego_pos.size < 3:
                ego_pos = np.pad(ego_pos, (0, 3 - ego_pos.size))
            world_plan = np.repeat(ego_pos[None, :3], plan_len, axis=0)

        self._recovery_hold = {
            "name": self._RECOVERY_BRAKE_PRIMITIVE,
            "start_ts": float(info.get("timestamp", 0.0)),
            "execute_seconds": duration,
            "prog": 0,
            "output_num_poses": int(hold.get("output_num_poses", plan_len)),
            "world_plan": world_plan,
            "states": states,
            "plan_dt_sec": plan_dt,
            "phase_start_prog": 0,
            "start_dac_tier": 1.0,
            "dac_exit_reached": True,
            "lane_borrowed_lane_seen": False,
            "lane_bypass_route_reentry_streak": 0,
            "exclusive_lane_bypass": False,
            "lane_bypass_return_prog": None,
            "lane_bypass_source_lane_keys": None,
            "phase_start_ego_pos": (
                None
                if info.get("ego_pos") is None
                else np.asarray(info["ego_pos"], dtype=np.float64).reshape(-1)[:3]
            ),
            "sequence_step": None,
            "sequence_context": self._recovery_sequence_context(),
            "emergency_guard": str(guard),
        }
        payload = {
            "selected_plan": plan,
            "selected_score": 0.0,
            "selected_source": "recovery:brake",
            "fallback_primitive": self._RECOVERY_BRAKE_PRIMITIVE,
            "fallback_mode": "hold",
            "fallback_states": states,
            "fallback_execute_seconds": duration,
            "emergency_guard": str(guard),
        }
        return self._make_step_result(
            selected_plan=plan,
            selected_score_raw=0.0,
            selected_source="recovery:brake",
            plan_payload=payload,
            proposals=plan[np.newaxis, ...],
            scores=np.asarray([0.0], dtype=np.float32),
            verifier_score=0.0,
            verifier_passed=False,
            recovered=True,
            debug={
                "recovery_hold": {
                    "primitive": str(hold.get("name", "unknown")),
                    "emergency_brake": True,
                    "guard": str(guard),
                    **dict(details),
                }
            },
        )

    def _step_recovery_hold(
        self,
        info: Dict[str, Any],
        privileged_info: Optional[List[Dict[str, Any]]],
    ) -> Optional[StepResult]:
        """Play the committed maneuver OPEN-LOOP for one tick.

        The maneuver is generated ONCE at commit time and anchored in the world
        frame. Each tick we re-express it in the CURRENT ego frame and advance a
        progress index along it, so the LATER phases (a lane change's straighten-
        out, a K-turn's reverse leg, ...) actually reach the controller. The old
        path regenerated the primitive from live state every tick, which always
        restarted at phase 1 -- so multi-phase maneuvers never completed.
        """
        hold = self._recovery_hold
        assert hold is not None
        exclusive_lane_bypass = bool(
            hold.get("exclusive_lane_bypass", False)
        )
        brake_hold = (
            str(hold.get("name", "")) == self._RECOVERY_BRAKE_PRIMITIVE
        )
        # The hold is governed only by SIM time, so its duration is independent
        # of control/replan frequency.
        now_ts = float(info.get("timestamp", 0.0))
        elapsed = now_ts - float(hold.get("start_ts", now_ts))
        time_up = elapsed >= float(hold.get("execute_seconds", 0.0))

        world_plan = hold.get("world_plan")
        try:
            pose = self._info_to_pose(info)
            local = np.asarray(
                self._world_points_to_current_local(world_plan, pose), dtype=np.float32
            )
        except Exception:
            self.log.exception("commit-hold: could not project committed maneuver; ending hold")
            self._record_failure_memory(
                info,
                event_type="recovery_execution",
                problem=getattr(self, "_pending_recovery_reason", None),
                method=str(hold.get("name", "unknown")),
                outcome="projection_failed",
                details={"elapsed_sec": elapsed},
            )
            self._finish_sequence_hold(
                hold, outcome="projection_failed", completed=False
            )
            self._settle_lane_bypass_intent(hold, completed=False)
            self._recovery_hold = None
            return None

        # Advance on the primitive's own time grid. Nearest-point progress can
        # deadlock at stops and direction changes, so simulation time remains
        # the normal phase clock. At a gear boundary, however, do not advance
        # out of a moving phase that physically never happened: episode 1011
        # exposed a reverse phase with throttle=1 and ~zero displacement that
        # time-jumped into forward motion with every target still behind ego.
        plan_dt = max(float(hold.get("plan_dt_sec", RECOVERY_PLAN_DT_SEC)), 1e-3)
        timed_prog = int(np.floor(max(elapsed, 0.0) / plan_dt + 1e-9))

        states = hold.get("states")
        ego_pos_raw = info.get("ego_pos")
        ego_pos = None
        if ego_pos_raw is not None:
            try:
                ego_pos = np.asarray(ego_pos_raw, dtype=np.float64).reshape(-1)[:3]
                if ego_pos.size < 2 or not np.isfinite(ego_pos).all():
                    ego_pos = None
            except (TypeError, ValueError):
                ego_pos = None

        if states is not None and ego_pos is not None:
            state_array = np.asarray(states, dtype=np.float64)
            state_count = min(len(state_array), len(local), len(world_plan))
            phase_start = int(hold.get("phase_start_prog", 0))
            phase_start = int(np.clip(phase_start, 0, max(state_count - 1, 0)))
            if hold.get("phase_start_ego_pos") is None:
                hold["phase_start_ego_pos"] = ego_pos.copy()

            def _motion_sign(value: float) -> int:
                return -1 if value < -1e-4 else (1 if value > 1e-4 else 0)

            if state_count > 0:
                phase_sign = _motion_sign(float(state_array[phase_start, 0]))
                boundary = phase_start + 1
                while (
                    boundary < state_count
                    and _motion_sign(float(state_array[boundary, 0])) == phase_sign
                ):
                    boundary += 1

                if timed_prog >= boundary and boundary < state_count:
                    if phase_sign != 0:
                        phase_origin = np.asarray(
                            hold["phase_start_ego_pos"], dtype=np.float64
                        )
                        dims = min(phase_origin.size, ego_pos.size, world_plan.shape[1])
                        planned_endpoint = np.asarray(
                            world_plan[boundary - 1], dtype=np.float64
                        )
                        planned_delta = (
                            planned_endpoint[:dims] - phase_origin[:dims]
                        )
                        expected_displacement = float(
                            np.linalg.norm(planned_delta)
                        )
                        actual_delta = ego_pos[:dims] - phase_origin[:dims]
                        actual_displacement = (
                            float(np.dot(actual_delta, planned_delta))
                            / max(expected_displacement, 1e-6)
                        )
                        required_displacement = min(
                            1.0, max(0.35, 0.35 * expected_displacement)
                        )
                        phase_stalled = (
                            expected_displacement >= 0.5
                            and actual_displacement < required_displacement
                        )
                        if phase_stalled:
                            blocked_since = hold.setdefault(
                                "phase_blocked_since_ts", now_ts
                            )
                            # Keep sending the final command from the uncompleted
                            # phase, never a stop/forward command whose geometry
                            # assumes the missing displacement.
                            timed_prog = min(timed_prog, boundary - 1)
                            if now_ts - float(blocked_since) >= 1.0:
                                self._record_failure_memory(
                                    info,
                                    event_type="recovery_execution",
                                    problem=getattr(self, "_pending_recovery_reason", None),
                                    method=str(hold.get("name", "unknown")),
                                    outcome="phase_stalled_abort",
                                    details={
                                        "boundary": boundary,
                                        "actual_displacement_m": actual_displacement,
                                        "required_displacement_m": required_displacement,
                                        "elapsed_sec": elapsed,
                                    },
                                )
                                self._finish_sequence_hold(
                                    hold,
                                    outcome="phase_stalled_abort",
                                    completed=False,
                                )
                                self._settle_lane_bypass_intent(
                                    hold, completed=False
                                )
                                self._recovery_hold = None
                                self.log.warning(
                                    "commit-hold: '%s' stalled before motion "
                                    "boundary %d (moved %.2fm, need %.2fm); "
                                    "aborting hold and replanning now",
                                    hold["name"], boundary,
                                    actual_displacement, required_displacement,
                                )
                                return None
                        else:
                            hold.pop("phase_blocked_since_ts", None)
                            hold["phase_start_prog"] = boundary
                            hold["phase_start_ego_pos"] = ego_pos.copy()
                    else:
                        # Stop/gear-settle phases have no displacement contract.
                        hold.pop("phase_blocked_since_ts", None)
                        hold["phase_start_prog"] = boundary
                        hold["phase_start_ego_pos"] = ego_pos.copy()

        action_complete = timed_prog >= len(local)
        prog = min(max(int(hold["prog"]), timed_prog), max(len(local) - 1, 0))
        hold["prog"] = prog
        remaining = local[prog:]

        # Run for at most the configured SIM-time budget. A shorter maneuver may
        # finish earlier once every trajectory phase has consumed its interval.
        if action_complete or time_up:
            if action_complete and not time_up:
                reason = "action complete"
                outcome = "action_complete"
            else:
                reason = "time budget elapsed (%.2fs/%.2fs)" % (
                    elapsed, float(hold.get("execute_seconds", 0.0))
                )
                outcome = "time_budget_elapsed"
            self._record_failure_memory(
                info,
                event_type="recovery_execution",
                problem=getattr(self, "_pending_recovery_reason", None),
                method=str(hold.get("name", "unknown")),
                outcome=outcome,
                details={
                    "progress": prog,
                    "plan_length": len(local),
                    "elapsed_sec": elapsed,
                },
            )
            self._finish_sequence_hold(hold, outcome=outcome, completed=True)
            self._settle_lane_bypass_intent(hold, completed=True)
            self._recovery_hold = None
            self.log.info(
                "commit-hold: '%s' complete [%s] (progress %d/%d); resuming normal loop",
                hold["name"], reason, prog, len(local),
            )
            return None

        traj = np.asarray(remaining[: hold["output_num_poses"]], dtype=np.float32)
        if len(traj) == 1:
            # Keep the final pose controllable until its interval expires;
            # SparseDrive's PID expects at least two waypoints.
            traj = np.repeat(traj, 2, axis=0)

        # Return-leg NC guard for the otherwise exempt exclusive bypass
        # (recovery.bypass_return_nc_guard, default off). The exemption below
        # exists because a blanket per-tick recheck aborts the maneuver the
        # instant steering begins, when the plan intentionally grazes the
        # static obstacle. That reasoning covers the entry and pass phases
        # only: once the plan starts blending back, the ego is merging into a
        # slot no admission gate ever looked at. Restricted to the route lane's
        # selected live actors (see _lane_bypass_guard_agents) so the
        # Every live static prop remains a hard NC obstacle during bypass.
        pedestrian_guard_active = False
        return_guard_active = False
        if (
            exclusive_lane_bypass
            and getattr(self, "recovery_bypass_return_nc_guard", False)
            and not brake_hold
        ):
            pedestrian_guard_active = True
            return_prog = hold.get("lane_bypass_return_prog")
            if return_prog is not None and prog >= int(return_prog):
                return_guard_active = True

        # Mid-hold layered PDMS guard. Compute the complete verifier result on
        # every control tick, but enforce only the hard NC gate here. Using the
        # aggregate PDMS score would incorrectly kill reverse/stop/lane-change
        # recovery because EP, LK, and TTC are deliberately soft terms.
        #
        # The safety trajectory starts at the REAL ego pose (local origin),
        # then joins the remaining world-anchored primitive. Without that first
        # segment the verifier only checks the ideal committed centerline; a
        # tracking-lagged ego can physically hit an obstacle while the detached
        # ideal path still clears it (RouteScenario_1028).
        #
        # This guard intentionally has no speed floor for moving maneuvers. A
        # low-speed collision is still a collision, and NC=0 must be fail-safe
        # even for an intentional close-pass maneuver. Brake is the sole
        # exemption: aborting the minimum-risk action because its stopping path
        # overlaps the hazard creates a one-frame brake/replan oscillation and
        # can re-enable throttle before the vehicle has stopped.
        ego_speed = abs(float(info.get("ego_velo", 0.0)))
        guard_agents: Optional[List[Dict[str, Any]]] = None
        if (
            not brake_hold
            and self.verifier is not None
            and privileged_info is not None
        ):
            if not exclusive_lane_bypass:
                guard_agents = privileged_info
            elif pedestrian_guard_active:
                guard_agents = self._lane_bypass_guard_agents(
                    hold,
                    privileged_info,
                    include_route_lane_vehicles=return_guard_active,
                )
        if guard_agents is not None:
            verify_info = dict(info)
            verify_info["privileged_agents"] = guard_agents
            safety_traj = self._recovery_hold_safety_plan(traj)
            prev_debug = getattr(self.verifier, "_debug", None)
            if prev_debug is not None:
                self.verifier._debug = False
            guard_error = None
            try:
                guard_score = float(self.verifier.score(safety_traj, verify_info))
                last = getattr(self.verifier, "last_result", None)
                nc = float((last.gates or {}).get("nc", 1.0)) if last else 1.0
                gates = dict(getattr(last, "gates", {}) or {}) if last else {}
                soft = dict(getattr(last, "soft", {}) or {}) if last else {}
            except Exception as exc:
                # A failed safety verifier must fail closed. Continuing a
                # committed maneuver with unknown collision state is riskier
                # than a conservative one-tick stop and retry.
                guard_score = 0.0
                nc = 0.0
                gates = {"nc": 0.0}
                soft = {}
                guard_error = str(exc)
            finally:
                if prev_debug is not None:
                    self.verifier._debug = prev_debug
            if nc <= 0.0:
                guard_name = (
                    "nc_return"
                    if return_guard_active
                    else (
                        "nc_pedestrian"
                        if pedestrian_guard_active
                        else "nc"
                    )
                )
                guard_outcome = (
                    "nc_return_guard_abort"
                    if return_guard_active
                    else (
                        "nc_pedestrian_guard_abort"
                        if pedestrian_guard_active
                        else "nc_guard_abort"
                    )
                )
                guard_details = {
                    "progress": prog,
                    "plan_length": len(local),
                    "elapsed_sec": elapsed,
                    "ego_speed_mps": ego_speed,
                    "pdms_score": guard_score,
                    "gates": gates,
                    "soft": soft,
                }
                if return_guard_active:
                    guard_details["guard_scope"] = "bypass_return_route_lane"
                    guard_details["return_prog"] = int(
                        hold.get("lane_bypass_return_prog", -1)
                    )
                    guard_details["route_lane_agents"] = len(guard_agents)
                elif pedestrian_guard_active:
                    guard_details["guard_scope"] = "bypass_all_phase_pedestrians"
                    guard_details["pedestrian_agents"] = len(guard_agents)
                if guard_error is not None:
                    guard_details["verifier_error"] = guard_error
                self._record_failure_memory(
                    info,
                    event_type="recovery_execution",
                    problem=getattr(self, "_pending_recovery_reason", None),
                    method=str(hold.get("name", "unknown")),
                    outcome=guard_outcome,
                    details=guard_details,
                )
                self._finish_sequence_hold(
                    hold, outcome=guard_outcome, completed=False
                )
                self._settle_lane_bypass_intent(hold, completed=False)
                self._recovery_hold = None
                if return_guard_active:
                    self.log.warning(
                        "commit-hold: route-lane NC gate tripped on the return "
                        "leg of '%s' (progress %d/%d >= return %s, %.2fs in, "
                        "ego=%.2fm/s, %d route-lane actor(s)); emergency "
                        "braking now",
                        hold["name"], prog, len(local),
                        hold.get("lane_bypass_return_prog"), elapsed,
                        ego_speed, len(guard_agents),
                    )
                elif pedestrian_guard_active:
                    self.log.warning(
                        "commit-hold: static/pedestrian NC gate tripped during the "
                        "entry/pass phase of '%s' (progress %d/%d, %.2fs in, "
                        "ego=%.2fm/s, %d guarded actor(s)); emergency braking now",
                        hold["name"], prog, len(local), elapsed,
                        ego_speed, len(guard_agents),
                    )
                else:
                    self.log.warning(
                        "commit-hold: NC gate tripped mid-hold on '%s' (progress "
                        "%d/%d, %.2fs in, ego=%.2fm/s); emergency braking now",
                        hold["name"], prog, len(local), elapsed,
                        ego_speed,
                    )
                return self._recovery_emergency_brake_result(
                    hold,
                    info=info,
                    guard=guard_name,
                    details=guard_details,
                )

        # Mid-hold DAC-tier guard (recovery_hold_dac_guard, default off).
        # Probe the REAL ego position as well as the next anchored poses. The
        # commit-time full-plan check only validates the intended centerline;
        # tracking error can still carry the vehicle into tier 0 (observed on
        # 1031's back_bypass_left). Tier-0 ego/lookahead violations are guarded
        # at every speed. Tier 0.5 lookahead retains the >2 m/s speed gate to
        # preserve low-speed squeeze-past behavior. A hold that STARTED outside
        # the corridor is allowed to improve/exit under dac_exit_grace; it is
        # aborted only if it worsens, or re-enters a violation after reaching
        # tier 1.
        if (
            self.recovery_hold_dac_guard
            and not exclusive_lane_bypass
            and not brake_hold
        ):
            probe = getattr(self.verifier, "_lane_tier_probe", None)
            if probe is not None:
                try:
                    ego_pos = np.asarray(
                        info.get("ego_pos"), dtype=np.float64
                    ).reshape(1, -1)
                    if ego_pos.shape[1] == 2:
                        ego_pos = np.hstack([ego_pos, np.zeros((1, 1))])
                    look = np.asarray(
                        world_plan[prog : prog + 2], dtype=np.float64
                    )
                    tiers = np.asarray(
                        probe(np.vstack([ego_pos[:, :3], look])), dtype=np.float64
                    )
                    ego_tier = float(tiers[0])
                    ahead = tiers[1:]
                    intended_tier = self._recovery_hold_intended_tier(
                        hold, prog, ahead, ego_tier)
                    start_tier = float(hold.get("start_dac_tier", 1.0))
                    if start_tier <= 0.5:
                        if ego_tier >= 1.0:
                            hold["dac_exit_reached"] = True
                        dac_violation = bool(
                            ego_tier < start_tier
                            # The committed centerline has already improved to
                            # a better tier at the current progress index, but
                            # the real ego has not followed it. This is the
                            # tracking-divergence case that let 1031 finish a
                            # nominal exit hold while remaining tier 0.
                            or intended_tier > ego_tier
                            or (
                                hold.get("dac_exit_reached", False)
                                and ego_tier <= 0.5
                            )
                        )
                    else:
                        ego_violation = ego_tier <= 0.5
                        ahead_offroad = bool((ahead <= 0.0).any())
                        ahead_surface = bool(
                            ego_speed > 2.0
                            and ((ahead > 0.0) & (ahead <= 0.5)).any()
                        )
                        dac_violation = bool(
                            ego_violation or ahead_offroad or ahead_surface
                        )
                except Exception:
                    ego_tier, intended_tier, start_tier, dac_violation = (
                        1.0, 1.0, 1.0, False
                    )
                if (
                    dac_violation
                    and str(hold.get("name", "")).startswith("lane_bypass_")
                    and states is not None
                ):
                    remaining_states = np.asarray(states, dtype=np.float32)[prog:]
                    # During the compact S-curve, the real footprint can briefly
                    # straddle a map-tier seam even though the committed centerline
                    # is tier 1.  Do not abort in the middle of its already-planned
                    # counter-steer/return; require the violation to persist after
                    # all remaining lateral correction has finished.
                    if (
                        remaining_states.ndim == 2
                        and remaining_states.shape[1] >= 2
                        and np.any(np.abs(remaining_states[:, 1]) > 1e-4)
                    ):
                        dac_violation = False
                if dac_violation:
                    hold["dac_streak"] = int(hold.get("dac_streak", 0)) + 1
                else:
                    hold["dac_streak"] = 0
                if hold["dac_streak"] >= self.recovery_hold_dac_guard_ticks:
                    self._record_failure_memory(
                        info,
                        event_type="recovery_execution",
                        problem=getattr(self, "_pending_recovery_reason", None),
                        method=str(hold.get("name", "unknown")),
                        outcome="dac_guard_abort",
                        details={
                            "progress": prog,
                            "plan_length": len(local),
                            "elapsed_sec": elapsed,
                            "dac_streak": int(hold["dac_streak"]),
                            "ego_tier": ego_tier,
                            "intended_tier": intended_tier,
                            "start_dac_tier": start_tier,
                        },
                    )
                    self._finish_sequence_hold(
                        hold, outcome="dac_guard_abort", completed=False
                    )
                    self._settle_lane_bypass_intent(
                        hold, completed=False
                    )
                    self._recovery_hold = None
                    self.log.warning(
                        "commit-hold: DAC-tier guard tripped mid-hold on '%s' "
                        "(ego_tier=%.1f intended_tier=%.1f start_tier=%.1f, "
                        "violation for %d ticks, "
                        "progress %d/%d, %.2fs in); aborting hold and "
                        "replanning now",
                        hold["name"], ego_tier, intended_tier, start_tier,
                        self.recovery_hold_dac_guard_ticks,
                        prog, len(local), elapsed,
                    )
                    return None

        payload = {
            "selected_plan": traj,
            "selected_score": None,
            "selected_source": f"recovery_hold:{hold['name']}",
            "fallback_primitive": hold["name"],
            "fallback_mode": "hold",
        }
        if hold.get("sequence_step"):
            payload["fallback_sequence_step"] = dict(hold["sequence_step"])
            payload["recovery_sequence_context"] = self._recovery_sequence_context()
        if states is not None:
            remaining_states = np.asarray(states, dtype=np.float32)[prog:]
            state_slice = remaining_states[: len(traj)]
            if len(state_slice) == 1:
                state_slice = np.repeat(state_slice, 2, axis=0)
            payload["fallback_states"] = state_slice
        seconds_left = max(0.0, float(hold.get("execute_seconds", 0.0)) - elapsed)
        self.log.info(
            "commit-hold: playing '%s' progress=%d/%d (%.2fs left)",
            hold["name"], prog, len(local), seconds_left,
        )
        return self._make_step_result(
            selected_plan=traj,
            selected_score_raw=None,
            selected_source=str(payload["selected_source"]),
            plan_payload=payload,
            proposals=traj[np.newaxis, ...],
            scores=np.array([1.0], dtype=np.float32),
            verifier_score=None,
            verifier_passed=True,
            recovered=True,
            debug={"recovery_hold": {
                "primitive": hold["name"], "progress": prog, "of": len(local),
                "elapsed_sec": round(elapsed, 3),
                "seconds_left": round(seconds_left, 3),
                "sequence_context": self._recovery_sequence_context(),
            }},
        )

    def _begin_commit_and_hold(
        self,
        result: StepResult,
        *,
        obs: Dict[str, Any],
        info: Dict[str, Any],
        privileged_info: Optional[List[Dict[str, Any]]],
        output_num_poses: int,
        verifier_score: float,
        allow_vlm_selection: bool = True,
        trigger_reason: Optional[str] = None,
        fail_closed: bool = True,
    ) -> StepResult:
        """Try each primitive in random order, verify each, and commit the first
        that passes for its fixed sim-time duration.

        When nothing passes the behaviour depends on WHY recovery was entered:

        * the verifier REJECTED the incoming plan (``verifier_score`` below
          threshold) -- the plan is known-unsafe, so brake. Returning ``result``
          here used to hand the rejected plan straight back to the controller;
          on routes 70/1077 of the 2026-08-18 run that produced 22 consecutive
          PDMS=0.000 ticks (NC=0, pedestrian 4.8 m ahead) driven at throttle
          1.00 into the collision.
        * a forced trigger fired while the verifier PASSED the plan (stuck
          timer, normal-EP gate) -- the plan is safe, merely unproductive, so
          keep it and retry next tick. Braking here would deepen the very
          deadlock the trigger exists to escape.

        ``fail_closed=False`` opts out for callers that already supply a brake
        as ``result`` and must re-evaluate every tick (the latched lane-bypass
        wait, which is polling for its execution window).
        """
        closure_wait = self._road_closure_wait_if_needed(info, privileged_info)
        if closure_wait is not None:
            return closure_wait
        chosen = self._recover_verify_first_pass(
            obs=obs,
            info=info,
            privileged_info=privileged_info,
            output_num_poses=output_num_poses,
            allow_vlm_selection=allow_vlm_selection,
        )
        if chosen is None:
            plan_was_rejected = float(verifier_score) < float(self.verifier_threshold)
            brake_instead = bool(fail_closed and plan_was_rejected)
            self._record_failure_memory(
                info,
                event_type="recovery_selection",
                problem=getattr(self, "_pending_recovery_reason", None),
                outcome=(
                    "no_primitive_available_braking"
                    if brake_instead
                    else "no_primitive_available"
                ),
                details={
                    "trigger_verifier_score": verifier_score,
                    "verifier_threshold": self.verifier_threshold,
                    "plan_was_rejected": plan_was_rejected,
                    "fail_closed": bool(fail_closed),
                },
            )
            if brake_instead:
                self.log.warning(
                    "verifier score %.3f < %.3f and no recovery primitive was "
                    "available (%s); braking instead of keeping the rejected "
                    "plan",
                    verifier_score,
                    self.verifier_threshold,
                    trigger_reason or "verifier rejection",
                )
                return self._recovery_emergency_brake_result(
                    {
                        "name": "no_primitive_available",
                        "output_num_poses": int(output_num_poses),
                    },
                    info=info,
                    guard="no_primitive_available",
                    details={
                        "trigger_verifier_score": float(verifier_score),
                        "verifier_threshold": float(self.verifier_threshold),
                        "trigger_reason": trigger_reason,
                    },
                )
            if trigger_reason:
                self.log.warning(
                    "%s; no recovery primitive was available; keeping learned "
                    "plan (verifier passed it: %.3f >= %.3f)",
                    trigger_reason, verifier_score, self.verifier_threshold,
                )
            else:
                self.log.warning(
                    "no primitive passed the verifier; keeping learned plan "
                    "and retrying next tick (verifier %.3f >= %.3f)",
                    verifier_score, self.verifier_threshold,
                )
            return result

        name, payload, prim_score = chosen
        if payload.get("fallback_mode") == "single_tick":
            # A verifier-failed brake is only a one-tick safety action. It is
            # deliberately NOT a successful commit-and-hold: keep the failure
            # episode/memory active and ask the VLM to select again next cycle.
            result.selected_plan = np.asarray(payload["selected_plan"], dtype=np.float32)
            result.selected_score_raw = prim_score
            result.selected_source = str(payload["selected_source"])
            result.plan_payload = payload
            result.recovered = True
            self._record_failure_memory(
                info,
                event_type="recovery_selection",
                problem=getattr(self, "_pending_recovery_reason", None),
                method=name,
                outcome="safe_brake_one_tick_retry",
                details={
                    "verifier_score": prim_score,
                    "verifier_threshold": self.verifier_threshold,
                },
            )
            self.log.warning(
                "commit-hold: no primitive passed PDMS; applying '%s' for one "
                "tick only and retrying VLM selection next cycle",
                name,
            )
            return result
        committed = np.asarray(payload["selected_plan"], dtype=np.float32)
        # Anchor the committed maneuver in the world frame ONCE, so the hold can
        # play it open-loop to completion (see _step_recovery_hold) instead of
        # regenerating -- and thus re-issuing only its first phase -- every tick.
        try:
            world_plan = np.asarray(
                self._local_plan_to_world(committed, self._info_to_pose(info)), dtype=np.float32
            )
        except Exception:
            self.log.exception(
                "commit-hold: could not world-anchor '%s'; executing one tick without a hold",
                name,
            )
            self._record_failure_memory(
                info,
                event_type="recovery_execution",
                problem=getattr(self, "_pending_recovery_reason", None),
                method=name,
                outcome="world_anchor_failed_one_tick_execution",
                details={"verifier_score": prim_score},
            )
            world_plan = None
        if world_plan is not None and len(world_plan) >= 2:
            sequence_steps = list(payload.get("fallback_sequence") or [])
            sequence_step = None
            if sequence_steps:
                manager = self._sequence_action_manager()
                if manager.original_steps != sequence_steps:
                    manager.select(
                        sequence_steps,
                        reasoning=None,
                        confidence=None,
                    )
                sequence_step = manager.start_first()
            action_execute_seconds = payload.get("fallback_execute_seconds")
            if action_execute_seconds is not None:
                try:
                    action_execute_seconds = float(action_execute_seconds)
                except (TypeError, ValueError):
                    action_execute_seconds = None
                if action_execute_seconds is not None and (
                    not np.isfinite(action_execute_seconds)
                    or action_execute_seconds <= 0.0
                ):
                    action_execute_seconds = None
            if action_execute_seconds is None:
                action_execute_seconds = recovery_action_fixed_duration_sec(
                    str(payload.get("fallback_primitive") or name)
                )
            # The lane-bypass primitive has a scene-dependent entry delay. Its
            # old fixed eight-second budget could truncate the final zero-steer
            # settle row when a long approaching gap required waiting first.
            # Derive the exact materialized duration from the last active state;
            # rows after it are executor padding ([speed, steer] == [0, 0]).
            if str(name).startswith("lane_bypass_"):
                lane_bypass_states = payload.get("fallback_states")
                if lane_bypass_states is not None:
                    try:
                        state_array = np.asarray(
                            lane_bypass_states, dtype=np.float64
                        )
                        active_rows = np.flatnonzero(
                            np.any(np.abs(state_array[:, :2]) > 1e-4, axis=1)
                        )
                        if active_rows.size:
                            action_execute_seconds = (
                                int(active_rows[-1]) + 1
                            ) * RECOVERY_PLAN_DT_SEC
                    except (TypeError, ValueError, IndexError):
                        pass
            # Return-leg guard context. Captured HERE, at commit, because the
            # descriptor is rebuilt from the LIVE ego waypoint every tick: once
            # the bypass has entered the borrowed lane, source_lane_keys would
            # resolve to the borrowed lane instead of the route lane the
            # maneuver has to merge back into.
            lane_bypass_return_prog = None
            lane_bypass_source_lane_keys = None
            if str(name).startswith("lane_bypass_"):
                (
                    lane_bypass_return_prog,
                    lane_bypass_source_lane_keys,
                ) = self._lane_bypass_return_guard_context(
                    name=name,
                    info=info,
                    committed=committed,
                    states=payload.get("fallback_states"),
                )
            if action_execute_seconds is None:
                # Unknown/custom recovery actions fall back to their generated
                # trajectory horizon, not to the old global six-second cap.
                action_execute_seconds = max(
                    RECOVERY_PLAN_DT_SEC,
                    len(world_plan) * RECOVERY_PLAN_DT_SEC,
                )
            start_dac_tier = 1.0
            probe = getattr(self.verifier, "_lane_tier_probe", None)
            if probe is not None and info.get("ego_pos") is not None:
                try:
                    ego_pos = np.asarray(info["ego_pos"], dtype=np.float64).reshape(1, -1)
                    if ego_pos.shape[1] == 2:
                        ego_pos = np.hstack([ego_pos, np.zeros((1, 1))])
                    start_dac_tier = float(
                        np.asarray(probe(ego_pos[:, :3]), dtype=np.float64)[0]
                    )
                except Exception:
                    start_dac_tier = 1.0
            # A multi-step VLM output is an intent queue. Commit only its first
            # primitive, then return to Loop 1; a later recovery selection sees
            # the completed/remaining queue and can keep or revise it.
            self._recovery_hold = {
                "name": name,
                "start_ts": float(info.get("timestamp", 0.0)),
                "execute_seconds": float(action_execute_seconds),
                "prog": 0,
                "output_num_poses": int(output_num_poses),
                "world_plan": world_plan,
                "states": (
                    None
                    if payload.get("fallback_states") is None
                    else np.asarray(payload["fallback_states"], dtype=np.float32)
                ),
                "plan_dt_sec": RECOVERY_PLAN_DT_SEC,
                "phase_start_prog": 0,
                "start_dac_tier": start_dac_tier,
                "dac_exit_reached": start_dac_tier >= 1.0,
                "lane_borrowed_lane_seen": (
                    str(name).startswith("lane_bypass_")
                    and info.get("route_lane_relation") in {"left", "right"}
                ),
                "lane_bypass_route_reentry_streak": 0,
                # The VLM first decides whether the fixed blockage requires
                # borrowing the adjacent lane. After the live provider and
                # commit-time NC/DAC gates open a safe entry window, this
                # direct primitive is intentionally one-shot: per-tick PDMS
                # re-evaluation would see the cones being passed and the
                # center-line crossing, then abort the maneuver halfway.
                "exclusive_lane_bypass": bool(
                    (
                        payload.get("recovery_needs_lane_bypass", False)
                        or payload.get(
                            "recovery_needs_lane_bypass_left", False
                        )
                        or payload.get(
                            "recovery_needs_lane_bypass_right", False
                        )
                    )
                    and payload.get("recovery_direct_primitive", False)
                    and str(name).startswith("lane_bypass_")
                ),
                "lane_bypass_internal_phase": payload.get(
                    "lane_bypass_internal_phase"
                ),
                "lane_bypass_return_prog": lane_bypass_return_prog,
                "lane_bypass_source_lane_keys": lane_bypass_source_lane_keys,
                "preserve_lane_bypass_intent": bool(
                    payload.get("preserve_lane_bypass_intent", False)
                ),
                "phase_start_ego_pos": (
                    None
                    if info.get("ego_pos") is None
                        else np.asarray(info["ego_pos"], dtype=np.float64).reshape(-1)[:3]
                ),
                "sequence_step": sequence_step,
                "sequence_context": self._recovery_sequence_context(),
            }
            if self._recovery_hold["exclusive_lane_bypass"] and getattr(
                self, "recovery_bypass_return_nc_guard", False
            ):
                if lane_bypass_return_prog is None:
                    self.log.warning(
                        "commit-hold: '%s' return-leg NC guard unavailable "
                        "(no route-lane identity or no return blend in the "
                        "committed plan); all-phase pedestrian guard remains "
                        "armed, but route-lane vehicles cannot be added",
                        name,
                    )
                else:
                    self.log.info(
                        "commit-hold: '%s' all-phase pedestrian guard armed; "
                        "route-lane vehicle guard joins at progress %d "
                        "(%.2fs into the maneuver), route lanes=%s",
                        name,
                        lane_bypass_return_prog,
                        lane_bypass_return_prog * RECOVERY_PLAN_DT_SEC,
                        sorted(lane_bypass_source_lane_keys),
                    )
            intent = getattr(self, "_lane_bypass_intent", None)
            if intent is not None and str(intent.get("action")) == str(name):
                intent["committed"] = True
            self._record_failure_memory(
                info,
                event_type="recovery_committed",
                problem=getattr(self, "_pending_recovery_reason", None),
                method=name,
                outcome="commit_and_hold_started",
                details={
                    "verifier_score": prim_score,
                    "execute_seconds": float(action_execute_seconds),
                    "parameters": dict(payload.get("fallback_parameters") or {}),
                },
            )
            if trigger_reason:
                self.log.warning(
                    "%s; committing primitive '%s' (verifier %.3f); playing it "
                    "open-loop for up to %.2fs",
                    trigger_reason, name, prim_score,
                    float(action_execute_seconds),
                )
            else:
                self.log.warning(
                    "verifier score %.3f < %.3f; committing primitive '%s' "
                    "(verifier %.3f); playing it open-loop for up to %.2fs",
                    verifier_score, self.verifier_threshold, name, prim_score,
                    float(action_execute_seconds),
                )
        else:
            one_tick_outcome = (
                "world_anchor_failed_one_tick_execution"
                if world_plan is None
                else "world_plan_too_short_one_tick_execution"
            )
            if payload.get("fallback_sequence"):
                manager = self._sequence_action_manager()
                sequence_steps = list(payload.get("fallback_sequence") or [])
                if manager.original_steps != sequence_steps:
                    manager.select(
                        sequence_steps,
                        reasoning=None,
                        confidence=None,
                    )
                manager.start_first()
                manager.finish_current(one_tick_outcome, completed=False)
            if world_plan is not None:
                self._record_failure_memory(
                    info,
                    event_type="recovery_execution",
                    problem=getattr(self, "_pending_recovery_reason", None),
                    method=name,
                    outcome="world_plan_too_short_one_tick_execution",
                    details={"verifier_score": prim_score},
                )
            if trigger_reason:
                self.log.warning(
                    "%s; executing primitive '%s' (verifier %.3f) for one tick "
                    "(no world anchor)",
                    trigger_reason, name, prim_score,
                )
            else:
                self.log.warning(
                    "verifier score %.3f < %.3f; executing primitive '%s' "
                    "(verifier %.3f) for one tick (no world anchor)",
                    verifier_score, self.verifier_threshold, name, prim_score,
                )
        result.selected_plan = committed
        result.selected_score_raw = prim_score
        result.selected_source = str(payload["selected_source"])
        result.plan_payload = payload
        result.recovered = True
        return result
