"""Deterministic temporal-profile trees for recovery primitives.

The simulator remains frozen while these profiles are enumerated.  Each leaf
is a complete execution schedule that the orchestrator materializes and scores;
the VLM sees only the resulting primitive-level scorecard and selects one of
the already verified winning leaves.
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


# Treat tiny floating-point differences in already-passed safety metrics as
# equivalent. Lane-bypass EP rewards extra forward progress, so total PDMS is
# deliberately not used to decide whether a slower return is "safer".
LANE_BYPASS_SAFETY_EQUIVALENCE_TOLERANCE = 1e-3


@dataclass
class TemporalSearchLeaf:
    """One fully materialized and verifier-scored temporal profile."""

    profile: Dict[str, Any]
    score: float
    accepted: bool
    trajectory: Optional[np.ndarray] = None
    states: Optional[np.ndarray] = None
    gates: Optional[Dict[str, float]] = None
    soft: Optional[Dict[str, float]] = None
    failure_flags: Optional[List[str]] = None


@dataclass
class TemporalSearchResult:
    """Primitive-level reduction of a temporal search tree."""

    action: str
    leaves: List[TemporalSearchLeaf]
    best_leaf: Optional[TemporalSearchLeaf]
    best_score: float
    robust_score: float
    feasible_count: int
    expanded_count: int

    @property
    def eligible(self) -> bool:
        return self.best_leaf is not None

    @property
    def feasible_ratio(self) -> float:
        if self.expanded_count <= 0:
            return 0.0
        return float(self.feasible_count) / float(self.expanded_count)

    def scorecard(self) -> Dict[str, Any]:
        accepted_scores = [
            float(leaf.score) for leaf in self.leaves if leaf.accepted
        ]
        best_profile = (
            _json_safe(dict(self.best_leaf.profile))
            if self.best_leaf is not None
            else None
        )
        best_gates = (
            _json_safe(dict(self.best_leaf.gates or {}))
            if self.best_leaf is not None
            else {}
        )
        best_soft = (
            _json_safe(dict(self.best_leaf.soft or {}))
            if self.best_leaf is not None
            else {}
        )
        result = {
            "eligible": bool(self.eligible),
            "selection_policy": (
                "fastest_safety_equivalent_return"
                if str(self.action).startswith("lane_bypass_")
                else "highest_score"
            ),
            "best_score": float(self.best_score),
            "peak_score": max(accepted_scores) if accepted_scores else 0.0,
            "robust_score": float(self.robust_score),
            "feasible_count": int(self.feasible_count),
            "expanded_count": int(self.expanded_count),
            "feasible_ratio": float(self.feasible_ratio),
            "best_profile": best_profile,
            "best_gates": best_gates,
            "best_soft": best_soft,
        }
        if str(self.action).startswith("lane_bypass_"):
            result["candidate_summaries"] = [
                {
                    "profile_id": leaf.profile.get("profile_id"),
                    "score": float(leaf.score),
                    "accepted": bool(leaf.accepted),
                    "return_complete_seconds": leaf.profile.get(
                        "return_complete_seconds"
                    ),
                    "execute_seconds": leaf.profile.get("execute_seconds"),
                    "return_steps": leaf.profile.get("return_steps"),
                    "settle_steps": leaf.profile.get("settle_steps"),
                    "gates": _json_safe(dict(leaf.gates or {})),
                    "soft": _json_safe(dict(leaf.soft or {})),
                    "failure_flags": list(leaf.failure_flags or ()),
                }
                for leaf in self.leaves
            ]
        return result


def _json_safe(value: Any) -> Any:
    """Convert NumPy/scorer values into strict JSON-compatible scalars."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _unique_floats(values: Iterable[float]) -> Tuple[float, ...]:
    result: List[float] = []
    for raw in values:
        value = float(raw)
        if not math.isfinite(value):
            continue
        if not any(abs(value - existing) <= 1e-9 for existing in result):
            result.append(value)
    return tuple(result)


def _center_first(values: Sequence[float], nominal: float) -> Tuple[float, ...]:
    return tuple(
        sorted(
            _unique_floats(values),
            key=lambda value: (abs(float(value) - float(nominal)), float(value)),
        )
    )


def _bounded_product(
    axes: Sequence[Tuple[str, Sequence[Any]]],
    *,
    budget: int,
) -> List[Dict[str, Any]]:
    """Breadth-like traversal of a small parameter tree, capped at ``budget``.

    Values on every axis are nominal-first. Sorting complete leaves by their
    summed axis rank visits the nominal path first, then one-axis deviations,
    instead of exhausting the deepest axis before trying another speed.
    """
    limit = max(1, int(budget))
    if not axes:
        return [{}]
    indexed_axes = [tuple(enumerate(values)) for _name, values in axes]
    ranked: List[Tuple[Tuple[Any, ...], Dict[str, Any]]] = []
    for combination in itertools.product(*indexed_axes):
        indices = tuple(int(item[0]) for item in combination)
        profile = {
            axes[position][0]: combination[position][1]
            for position in range(len(axes))
        }
        rank = (sum(indices), max(indices), indices)
        ranked.append((rank, profile))
    ranked.sort(key=lambda item: item[0])
    return [profile for _rank, profile in ranked[:limit]]


def build_temporal_profile_tree(
    *,
    action: str,
    spec: Mapping[str, Any],
    budget: int,
    duration_scales: Sequence[float] = (0.75, 1.0, 1.25),
    start_delays_sec: Sequence[float] = (0.0, 0.5, 1.0, 1.5),
    lateral_range: Optional[Tuple[float, float]] = None,
    bypass_target_speeds_mps: Sequence[float] = (6.0, 8.0, 10.0, 12.0),
    bypass_accels_mps2: Sequence[float] = (3.5, 4.5),
    bypass_total_steps: Sequence[int] = (8, 9, 10, 12),
    bypass_entry_steps: Sequence[int] = (2, 3, 4),
    bypass_return_steps: Sequence[int] = (2, 3),
    bypass_settle_steps: Sequence[int] = (1, 2),
) -> List[Dict[str, Any]]:
    """Return deterministic temporal leaves ordered nominal-first.

    The axes are deliberately phase-oriented for lane bypasses.  Other actions
    search their exposed speed/lateral parameter together with a bounded action
    duration and a short pre-action stop interval. Brake and wait retain their
    system-owned duration, while lane bypasses retain their traffic-derived
    entry delay and latch behavior.
    """
    action = str(action)
    limit = max(1, int(budget))
    fixed_duration = float(spec.get("fixed_duration_sec", 1.0))

    if action.startswith("lane_bypass_"):
        profiles: List[Dict[str, Any]] = [{"profile_kind": "legacy_auto"}]
        speed_values = _center_first(bypass_target_speeds_mps, 10.0)
        accel_values = _center_first(bypass_accels_mps2, 3.5)
        total_values = tuple(
            sorted(
                {max(1, int(v)) for v in bypass_total_steps},
                key=lambda v: (abs(v - 9), v),
            )
        )
        entry_values = tuple(
            sorted(
                {max(1, int(v)) for v in bypass_entry_steps},
                key=lambda v: (abs(v - 3), v),
            )
        )
        return_values = tuple(
            sorted(
                {max(1, int(v)) for v in bypass_return_steps},
                key=lambda v: (abs(v - 2), v),
            )
        )
        settle_values = tuple(
            sorted(
                {max(0, int(v)) for v in bypass_settle_steps},
                key=lambda v: (abs(v - 1), v),
            )
        )
        axes = [
            ("target_speed_mps", speed_values),
            ("accel_mps2", accel_values),
            ("total_steps", total_values),
            ("entry_steps", entry_values),
            ("return_steps", return_values),
            ("settle_steps", settle_values),
        ]
        nominal = {
            "target_speed_mps": speed_values[0],
            "accel_mps2": accel_values[0],
            "total_steps": total_values[0],
            "entry_steps": entry_values[0],
            "return_steps": return_values[0],
            "settle_steps": settle_values[0],
        }
        seeded: List[Dict[str, Any]] = [dict(nominal)]
        for axis_name, values in axes:
            for value in values[1:]:
                candidate = dict(nominal)
                candidate[axis_name] = value
                seeded.append(candidate)
        candidates = seeded + _bounded_product(
            axes,
            budget=max(limit * 4, limit),
        )
        seen = set()
        for profile in candidates:
            if (
                int(profile["entry_steps"])
                + int(profile["return_steps"])
                + int(profile["settle_steps"])
                > int(profile["total_steps"])
            ):
                continue
            key = tuple(profile[name] for name, _values in axes)
            if key in seen:
                continue
            seen.add(key)
            profile["profile_kind"] = "phase_tree"
            profiles.append(profile)
            if len(profiles) >= limit:
                break
        return profiles[:limit]

    if action in {"brake", "wait", "three_point"}:
        return [
            {
                "profile_kind": "fixed",
                "duration_sec": fixed_duration,
                "target_speed_mps": None,
                "lateral_distance_m": None,
            }
        ]

    duration_values = _center_first(
        (fixed_duration * float(scale) for scale in duration_scales),
        fixed_duration,
    )
    axes: List[Tuple[str, Sequence[Any]]] = [
        ("duration_sec", duration_values),
    ]
    speed_range = spec.get("target_speed_mps")
    if speed_range is not None:
        low, high = map(float, speed_range)
        axes.append(
            (
                "target_speed_mps",
                _center_first((low, 0.5 * (low + high), high), 0.5 * (low + high)),
            )
        )
    else:
        axes.append(("target_speed_mps", (None,)))
    effective_lateral = lateral_range or spec.get("lateral_distance_m")
    if effective_lateral is not None:
        low, high = map(float, effective_lateral)
        axes.append(
            (
                "lateral_distance_m",
                _center_first((low, 0.5 * (low + high), high), 0.5 * (low + high)),
            )
        )
    else:
        axes.append(("lateral_distance_m", (None,)))
    valid_delays: List[float] = []
    for raw_delay in start_delays_sec:
        try:
            delay = float(raw_delay)
        except (TypeError, ValueError):
            continue
        if math.isfinite(delay) and delay >= 0.0:
            valid_delays.append(delay)
    delay_values = _center_first(valid_delays, 0.0)
    axes.append(("start_delay_sec", delay_values or (0.0,)))
    nominal = {name: values[0] for name, values in axes}
    seeded: List[Dict[str, Any]] = [dict(nominal)]
    for axis_name, values in axes:
        for value in values[1:]:
            candidate = dict(nominal)
            candidate[axis_name] = value
            seeded.append(candidate)
    candidates = seeded + _bounded_product(
        axes,
        budget=max(limit * 4, limit),
    )
    profiles: List[Dict[str, Any]] = []
    seen = set()
    for profile in candidates:
        key = tuple(profile[name] for name, _values in axes)
        if key in seen:
            continue
        seen.add(key)
        profile["profile_kind"] = "action_tree"
        profiles.append(profile)
        if len(profiles) >= limit:
            break
    return profiles


def reduce_temporal_search(
    action: str,
    leaves: Sequence[TemporalSearchLeaf],
) -> TemporalSearchResult:
    accepted = [leaf for leaf in leaves if leaf.accepted]
    accepted.sort(key=lambda leaf: float(leaf.score), reverse=True)
    best = accepted[0] if accepted else None
    if best is not None and str(action).startswith("lane_bypass_"):
        safety_equivalent = list(accepted)

        def gate_floor(leaf: TemporalSearchLeaf) -> float:
            values = []
            for value in (leaf.gates or {}).values():
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(numeric):
                    values.append(numeric)
            return min(values) if values else 1.0

        def soft_metric(leaf: TemporalSearchLeaf, name: str) -> float:
            try:
                numeric = float((leaf.soft or {}).get(name, 1.0))
            except (TypeError, ValueError):
                return 1.0
            return numeric if math.isfinite(numeric) else 1.0

        # All leaves here already passed the verifier threshold and footprint/
        # DAC checks. Preserve the best remaining hard-gate, TTC, and lane-
        # keeping quality; within that safety-equivalent set, do not let EP's
        # preference for a longer endpoint delay the return to the route lane.
        for metric in (
            gate_floor,
            lambda leaf: soft_metric(leaf, "ttc"),
            lambda leaf: soft_metric(leaf, "lk"),
        ):
            best_metric = max(metric(leaf) for leaf in safety_equivalent)
            safety_equivalent = [
                leaf
                for leaf in safety_equivalent
                if metric(leaf)
                >= best_metric - LANE_BYPASS_SAFETY_EQUIVALENCE_TOLERANCE
            ]

        def fast_return_key(leaf: TemporalSearchLeaf) -> Tuple[float, ...]:
            profile = leaf.profile
            try:
                return_complete = float(profile["return_complete_seconds"])
            except (KeyError, TypeError, ValueError):
                return_complete = math.inf
            try:
                execute_seconds = float(profile["execute_seconds"])
            except (KeyError, TypeError, ValueError):
                execute_seconds = math.inf
            try:
                return_steps = float(profile.get("return_steps", math.inf))
            except (TypeError, ValueError):
                return_steps = math.inf
            try:
                settle_steps = float(profile.get("settle_steps", math.inf))
            except (TypeError, ValueError):
                settle_steps = math.inf
            return (
                return_complete,
                execute_seconds,
                return_steps,
                settle_steps,
                -float(leaf.score),
            )

        best = min(safety_equivalent, key=fast_return_key)
    scores = np.asarray([float(leaf.score) for leaf in accepted], dtype=np.float64)
    robust = float(np.quantile(scores, 0.25)) if scores.size else 0.0
    return TemporalSearchResult(
        action=str(action),
        leaves=list(leaves),
        best_leaf=best,
        best_score=float(best.score) if best is not None else 0.0,
        robust_score=robust,
        feasible_count=len(accepted),
        expanded_count=len(leaves),
    )


__all__ = [
    "TemporalSearchLeaf",
    "TemporalSearchResult",
    "build_temporal_profile_tree",
    "reduce_temporal_search",
]
