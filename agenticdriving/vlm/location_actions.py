"""Remember ineffective recovery choices until the ego leaves the location."""
from collections import Counter
import math


class LocationActionMemory:
    def __init__(self, repeat_limit=0, reset_distance_m=3.0):
        self.limit = max(0, int(repeat_limit))
        self.reset_distance = float(reset_distance_m)
        if not math.isfinite(self.reset_distance) or self.reset_distance <= .5:
            raise ValueError("recovery_location_reset_distance_m must exceed 0.5m")
        self.anchor = self.position = self.pending = None
        self.timestamp = None
        self.counts = Counter()

    def reset(self):
        self.anchor = self.pending = None
        self.counts.clear()

    def observe(self, info):
        if not self.limit:
            return
        pos = tuple(float(x) for x in info.get("ego_pos", [])[:2])
        now = float(info.get("timestamp", 0))
        if len(pos) != 2 or not all(math.isfinite(x) for x in (*pos, now)):
            self.position = None
            return
        if (self.timestamp is not None and now < self.timestamp
                or self.anchor is not None and math.dist(pos, self.anchor) >= self.reset_distance):
            self.reset()
        self.position, self.timestamp = pos, now
        if self.pending and math.dist(pos, self.pending[1]) >= .5:
            self.pending = None  # This attempt moved; do not count it as ineffective.

    def exclusions(self, info):
        self.observe(info)
        if self.position is not None and self.pending:
            self.counts[self.pending[0]] += 1
            self.pending = None
        return {name for name, count in self.counts.items() if count >= self.limit} if self.limit else set()

    def record(self, action):
        if self.limit and self.position is not None:
            if self.anchor is None:
                self.anchor = self.position
            self.pending = (action, self.position)
