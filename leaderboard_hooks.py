"""Route context and the 10-RGB research rig for the release evaluator."""
from __future__ import annotations

import importlib
import logging

LOG = logging.getLogger(__name__)

# (route_name, repetition_index) of the RouteScenario currently being built/run.
_CURRENT_ROUTE = {"name": None, "rep": 0}


def get_route_context():
    """Return (route_name, repetition_index) for the current route, or (None, 0)."""
    return _CURRENT_ROUTE["name"], int(_CURRENT_ROUTE["rep"] or 0)


def _patch_route_context() -> bool:
    try:
        rs = importlib.import_module("leaderboard.scenarios.route_scenario")
    except Exception as exc:  # pragma: no cover
        LOG.warning("leaderboard_hooks: route-context patch skipped (%s)", exc)
        return False

    cls = getattr(rs, "RouteScenario", None)
    if cls is None or getattr(cls.__init__, "_agenticdriving_patched", False):
        return False

    _orig_init = cls.__init__

    def __init__(self, *args, **kwargs):
        cfg = kwargs.get("config")
        if cfg is None and len(args) >= 2:
            # RouteScenario.__init__(self, world, config, ...) -> config is args[1]
            cfg = args[1]
        if cfg is not None:
            _CURRENT_ROUTE["name"] = getattr(cfg, "name", None)
            _CURRENT_ROUTE["rep"] = getattr(cfg, "repetition_index", 0)
        return _orig_init(self, *args, **kwargs)

    __init__._agenticdriving_patched = True
    cls.__init__ = __init__
    return True




_APPLIED = False


def apply() -> None:
    """Apply release evaluator hooks. Idempotent."""
    global _APPLIED
    if _APPLIED:
        return
    rc = _patch_route_context()
    from leaderboard.autoagents import agent_wrapper
    agent_wrapper.SENSORS_LIMITS = dict(agent_wrapper.SENSORS_LIMITS)
    agent_wrapper.SENSORS_LIMITS["sensor.camera.rgb"] = 10
    _APPLIED = True
    LOG.info("Release hooks applied: route_context=%s RGB limit=10", rc)
