"""Review persistent VRU stops without bypassing the emergency brake."""
import math

from .disagreement import DisagreementConfig, DisagreementManager


class StationaryVruReview(DisagreementManager):
    def __init__(self, *, checker, output_dir):
        super().__init__(DisagreementConfig(enabled=True, window_sec=10.,
                         match_distance_m=.5), checker=checker, output_dir=output_dir)
        self.debug_dir = self.debug_dir.parent / "stationary_vru_review"
        self.stop_since = None

    def update(self, agents, *, sensor_timestamp, info, brake_held):
        now = float(info["timestamp"])
        if not brake_held or abs(float(info.get("ego_velo", 0))) > .2:
            self.stop_since = None
        elif self.stop_since is None or now < self.stop_since:
            self.stop_since = now
        if now - self.last_observation > 1.0 or sensor_timestamp < self.last_observation:
            self.tracks.clear()
            self._epoch += 1
        yaw = float(info["ego_rot"][2])
        c, s = math.cos(yaw), math.sin(yaw)
        candidates, moving = [], []
        for a in agents:
            if not str(a.get("object_type", "")).startswith(("pedestrian", "walker")):
                continue
            if abs(float(a.get("agent_velo", 0))) > .2:
                moving.append(a)
                continue
            dx, dy = (float(a["agent_pos_world"][i]) - float(info["ego_pos"][i]) for i in (0, 1))
            if 0 < dx*c + dy*s < 30 and abs(-dx*s + dy*c) < 6:
                candidates.append({"kind": "stationary_vru", "agent": a})
        self.observe(candidates, sensor_timestamp, agreements=moving)
        for tr in self.tracks.values():
            if not tr.active:
                tr.verdict = None
                tr.hits.clear()
                tr.version += 1  # A departed target cannot inherit an in-flight answer.

    def _ready(self, track, now):
        return (self.stop_since is not None and now - self.stop_since >= 5.
                and super()._ready(track, now))
