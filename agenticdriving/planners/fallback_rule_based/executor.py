"""Restricted execution of code that composes the driving primitives.

LaMPilot drives an LLM-written generator step by step; we do the same but cap the
steps at ``output_num_poses`` to materialize exactly one recovery trajectory. The
code runs with an empty ``__builtins__`` plus a short allowlist, so it cannot
import modules, open files, or touch attributes — any failure degrades to ``None``.
"""
from __future__ import annotations

import logging
import types
from typing import Optional

import numpy as np

from agenticdriving.planners.fallback_rule_based.primitives import (
    API_NAMES,
    PrimitiveRuntime,
    build_api_namespace,
)

LOG = logging.getLogger("fallback_rule_based.executor")

# Minimal builtins LLM code may legitimately need; deliberately excludes
# __import__, open, eval, exec, getattr, etc.
_SAFE_BUILTINS = {
    "range": range,
    "len": len,
    "min": min,
    "max": max,
    "abs": abs,
    "round": round,
    "float": float,
    "int": int,
    "bool": bool,
    "enumerate": enumerate,
    "True": True,
    "False": False,
    "None": None,
}


def _find_entry_function(namespace: dict):
    """Return the single user-defined function (ignoring the bound API names)."""
    candidates = [
        value
        for name, value in namespace.items()
        if isinstance(value, types.FunctionType) and name not in API_NAMES
    ]
    return candidates[0] if len(candidates) == 1 else (candidates[-1] if candidates else None)


def run_codegen(code_str: str, runtime: PrimitiveRuntime, output_num_poses: int,
                name: Optional[str] = None):
    """Execute ``code_str`` against ``runtime`` and return ``(traj, states)``.

    ``traj`` is the ``[T, 2]`` pose trajectory and ``states`` is the matching
    signed ``[K, 2]`` ``(speed, steer_rad)`` sequence (for driving the maneuver
    open-loop, reverse segments included). Supports both LaMPilot-style generator
    functions (``yield autopilot()`` per step) and plain setter functions; either
    way the sim is advanced exactly ``output_num_poses`` steps. Returns ``None`` on
    any error.

    ``name`` is the primitive key (when known). Most legacy names starting with
    ``"back"`` are preceded by a quick emergency stop so the ego is stationary
    before their reverse leg. ``back_then_bypass`` is excluded because it owns its
    complete stop -> reverse -> stop -> forward sequence and is exactly sized to
    the recovery window; prepending another stop would truncate its forward phase.
    """
    if not code_str or not code_str.strip():
        return None
    try:
        namespace = build_api_namespace(runtime)
        namespace["__builtins__"] = dict(_SAFE_BUILTINS)
        exec(compile(code_str, "<fallback_rule_based>", "exec"), namespace)  # noqa: S102
    except Exception:
        LOG.warning("fallback codegen failed to compile/exec", exc_info=True)
        return None

    entry = _find_entry_function(namespace)
    if entry is None:
        LOG.warning("fallback codegen produced no entry function")
        return None

    def _drive(gen, steps: int) -> int:
        """Advance ``gen`` (a per-tick generator) up to ``steps`` sim steps.

        Returns the number of steps actually consumed (a generator that stops
        early leaves the remainder for the caller)."""
        used = 0
        while used < steps:
            try:
                next(gen)
            except StopIteration:
                break
            runtime.sim.step()
            used += 1
        return used

    try:
        steps_left = int(output_num_poses)
        # ``back*`` maneuvers must come to a full stop before their reverse leg:
        # a reverse started while still rolling forward only bleeds off forward
        # speed and never backs up. Prepend a QUICK, hard brake (emergency_stop,
        # short horizon) -- distinct from the maneuver's own gradual settle-brake
        # -- so the ego is stationary before the maneuver body runs. Shares the
        # fixed output_num_poses budget with the maneuver.
        name_text = "" if name is None else str(name)
        bypass_owns_stop = name_text.startswith("back_bypass") or name_text.startswith(
            "back_then_bypass"
        )
        if name_text.startswith("back") and not bypass_owns_stop:
            estop = namespace.get("emergency_stop")
            if callable(estop):
                estop_gen = estop()
                if isinstance(estop_gen, types.GeneratorType):
                    # ``emergency_stop`` self-applies the hard (~0.8 g) decel, so
                    # this pre-stop snaps the ego stationary in a few ticks before
                    # the reverse leg -- no manual decel override needed here.
                    steps_left -= _drive(estop_gen, steps_left)

        result = entry()
        if isinstance(result, types.GeneratorType):
            _drive(result, steps_left)
        else:
            # Plain setter function: configure once, then drive the full horizon.
            for _ in range(steps_left):
                runtime.sim.step()
    except Exception:
        LOG.warning("fallback codegen raised during execution", exc_info=True)
        return None

    traj = runtime.sim.trajectory(output_num_poses)
    states = runtime.sim.state_sequence(output_num_poses)
    if not np.all(np.isfinite(traj)) or not np.all(np.isfinite(states)):
        return None
    return traj, states
