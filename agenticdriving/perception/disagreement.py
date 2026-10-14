"""Optional, simulator-independent tracking and arbitration of sensor disagreements.

Only fresh sensor harvests call observe(). Control ticks may call apply() as
often as needed without increasing the five-in-five-seconds evidence count.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
import math
import logging
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DisagreementConfig:
    enabled: bool = False
    window_sec: float = 5.0
    min_disagreements: int = 5
    match_distance_m: float = 1.0
    result_max_age_sec: float = 5.0
    verdict_ttl_sec: float = 10.0
    retry_interval_sec: float = 5.0
    global_cooldown_sec: float = 5.0
    confidence_threshold: float = 0.8
    timeout_sec: float = 12.0
    max_new_tokens: int = 160
    max_tracks: int = 256
    save_debug_artifacts: bool = True

    def __post_init__(self):
        for key in ("window_sec", "match_distance_m", "result_max_age_sec",
                    "verdict_ttl_sec", "retry_interval_sec", "global_cooldown_sec", "timeout_sec"):
            if not math.isfinite(float(getattr(self, key))) or getattr(self, key) <= 0:
                raise ValueError(f"disagreement_vlm.{key} must be positive and finite")
        for key in ("min_disagreements", "max_new_tokens", "max_tracks"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"disagreement_vlm.{key} must be a positive integer")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("disagreement_vlm.confidence_threshold must be in [0, 1]")


def same_object(a, b, radius):
    """Conservative spatial association; bd_N is a row index, not a track ID."""
    pa, pb = np.asarray(a["agent_pos_world"]), np.asarray(b["agent_pos_world"])
    if np.linalg.norm(pa[:2] - pb[:2]) > radius or abs(pa[2] - pb[2]) > 1.5:
        return False
    ea, eb = sorted(a["agent_extent"][:2]), sorted(b["agent_extent"][:2])
    # A fixed one-metre radius alone merges neighboring small cones.
    geometry_radius = .5 * min(math.hypot(*ea), math.hypot(*eb)) + .25
    if np.linalg.norm(pa[:2] - pb[:2]) > min(radius, geometry_radius):
        return False
    return all(min(x, y) / max(x, y, 0.01) >= 0.25 for x, y in zip(ea, eb))


@dataclass
class _Track:
    key: int
    candidate: dict
    anchor: dict
    hits: deque = field(default_factory=deque)
    last_seen: float = 0.0
    active: bool = True
    version: int = 0
    last_request: float = -math.inf
    verdict: dict | None = None


class DisagreementManager:
    def __init__(self, cfg: DisagreementConfig, *, checker: Any, output_dir: Path):
        self.cfg, self.checker = cfg, checker
        self.debug_dir = Path(output_dir) / "perception_disagreement"
        self.tracks: dict[int, _Track] = {}
        self.last_observation = -math.inf
        self._last_request_timestamp = -math.inf
        self._next_key = 0
        self._epoch = 0
        self._thread = None
        self._completed = None
        self._lock = threading.Lock()
        self._closed = False
        self.last_applied = []

    @property
    def busy(self):
        return (self._thread is not None and self._thread.is_alive()) or self._completed is not None

    def _record(self, record):
        if self.cfg.save_debug_artifacts:
            from agenticdriving.vlm.debug import append_jsonl
            try:
                self.debug_dir.mkdir(parents=True, exist_ok=True)
                append_jsonl(self.debug_dir / "events.jsonl", record)
            except OSError:
                logging.getLogger(__name__).exception("Could not record perception disagreement")

    def observe(self, candidates, timestamp, *, agreements=()):
        if not self.cfg.enabled or self._closed:
            return
        now = float(timestamp)
        if not math.isfinite(now):
            return
        if now < self.last_observation - 1e-6:
            self.tracks.clear()
            self._epoch += 1
            self._last_request_timestamp = -math.inf
        elif abs(now - self.last_observation) <= 1e-6:
            return
        self.last_observation = now
        for key, tr in list(self.tracks.items()):
            if now - tr.last_seen > self.cfg.window_sec + 1e-6:
                del self.tracks[key]
                continue
            tr.active = False
            while tr.hits and now - tr.hits[0] > self.cfg.window_sec + 1e-6:
                tr.hits.popleft()
            if any(same_object(tr.anchor, a, self.cfg.match_distance_m) for a in agreements):
                # New agreement invalidates an old cancellation, including an
                # in-flight answer, but the requested rolling count is not a
                # consecutive-frame count: earlier disagreements still count.
                tr.verdict = None
                tr.version += 1
        used = set()
        unique = []
        for candidate in candidates:
            if any(np.allclose(c["agent"]["agent_pos_world"], candidate["agent"]["agent_pos_world"], atol=.05, rtol=0)
                   and np.allclose(c["agent"]["agent_extent"], candidate["agent"]["agent_extent"], atol=.05, rtol=0)
                   for c in unique):
                continue
            unique.append(candidate)
            options = [tr for tr in self.tracks.values()
                       if tr.key not in used
                       and same_object(tr.anchor, candidate["agent"], self.cfg.match_distance_m)]
            if options:
                tr = min(options, key=lambda t: np.linalg.norm(
                    np.asarray(t.anchor["agent_pos_world"][:2]) - candidate["agent"]["agent_pos_world"][:2]))
            else:
                if len(self.tracks) >= self.cfg.max_tracks:
                    oldest = min(self.tracks, key=lambda k: self.tracks[k].last_seen)
                    del self.tracks[oldest]
                tr = _Track(self._next_key, deepcopy(candidate), deepcopy(candidate["agent"]))
                self.tracks[tr.key] = tr
                self._next_key += 1
            used.add(tr.key)
            tr.active, tr.last_seen = True, now
            tr.candidate = deepcopy(candidate)
            tr.hits.append(now)
            self._record({"kind": "observation", "timestamp": now, "track_id": tr.key,
                          "disagreement": candidate["kind"], "count": len(tr.hits),
                          "agent": candidate["agent"]})

    def _run(self, request):
        started = time.monotonic()
        try:
            result = self.checker.check_perception_box(
                candidate=request["candidate"], camera_images=request["images"],
                camera_specs=request["camera_specs"], info=request["info"],
                request_id=request["request_id"], debug_dir=self.debug_dir,
                max_new_tokens=self.cfg.max_new_tokens, timeout_sec=self.cfg.timeout_sec,
                save_debug_artifacts=self.cfg.save_debug_artifacts)
        except Exception as exc:
            result = {"decision": "uncertain", "error": str(exc)}
        if not isinstance(result, dict):
            result = {"decision": "uncertain", "error": "invalid_checker_result"}
        if time.monotonic() - started > self.cfg.timeout_sec:
            result = {"decision": "uncertain", "error": "timeout"}
        with self._lock:
            if not self._closed:
                self._completed = (request, result)

    def _poll(self, now):
        with self._lock:
            completed, self._completed = self._completed, None
        if completed is None:
            return
        request, result = completed
        tr = self.tracks.get(request["track_id"])
        confidence = result.get("confidence")
        accepted = bool(
            tr and tr.active and request["epoch"] == self._epoch
            and request["version"] == tr.version
            and 0 <= now - request["timestamp"] <= self.cfg.result_max_age_sec
            and same_object(tr.candidate["agent"], request["candidate"]["agent"], self.cfg.match_distance_m)
            and not result.get("error") and result.get("decision") in (
                {"keep", "static"} if tr.candidate["kind"] == "stationary_vru" else {"keep", "drop"})
            and type(confidence) in (int, float) and math.isfinite(confidence)
            and self.cfg.confidence_threshold <= confidence <= 1)
        if accepted:
            tr.verdict = {**result, "timestamp": now}
        self._record({"kind": "result", "timestamp": now, "track_id": request["track_id"],
                      "accepted": accepted, "result": result})

    def apply(self, agents, *, obs, info, camera_specs, max_staleness_sec=1.0):
        """Poll/submit without waiting; apply verdicts only to fresh candidates."""
        self.last_applied = []
        if not self.cfg.enabled or self._closed:
            return agents
        now = float(info.get("timestamp", 0.0))
        self._poll(now)
        live = [tr for tr in self.tracks.values() if tr.active
                and 0 <= now - tr.last_seen <= max_staleness_sec + 1e-6]
        for tr in self.tracks.values():
            while tr.hits and now - tr.hits[0] > self.cfg.window_sec + 1e-6:
                tr.hits.popleft()
            if tr.verdict and now - tr.verdict["timestamp"] >= self.cfg.verdict_ttl_sec:
                tr.verdict = None
        eligible = [tr for tr in live if not tr.verdict
                    and self._ready(tr, now)
                    and now - tr.last_request >= self.cfg.retry_interval_sec]
        cooldown_elapsed = now - self._last_request_timestamp >= self.cfg.global_cooldown_sec
        if eligible and cooldown_elapsed and not self.busy and self.checker is not None:
            # Reuse the episode's VLM without piling up work behind another
            # optional stop-sign/semantic query.
            if not getattr(self.checker, "optional_async_busy", False):
                tr = min(eligible, key=lambda t: (t.last_request, t.key))
                specs = {s["id"]: s for s in camera_specs if s.get("type") == "sensor.camera.rgb"
                         and abs(float(s.get("pitch", 0))) < 45}
                images = {k: np.asarray(v).copy() for k, v in (obs.get("rgb") or {}).items() if k in specs}
                if images:
                    tr.last_request = now
                    # Global simulation-time cooldown, including failed or
                    # unprojectable requests and requests for other objects.
                    self._last_request_timestamp = now
                    request = {"track_id": tr.key, "epoch": self._epoch, "version": tr.version,
                               "timestamp": now, "request_id": f"{self._epoch}_{tr.key}_{now:.6f}",
                               "candidate": deepcopy(tr.candidate), "images": images,
                               "camera_specs": deepcopy(specs),
                               "info": {**{k: deepcopy(info[k]) for k in ("timestamp", "ego_pos", "ego_rot")},
                                        "frame_index": info.get("frame_index", 0)}}
                    self._record({"kind": "request", "timestamp": now, "track_id": tr.key,
                                  "count": len(tr.hits), "candidate": tr.candidate})
                    self._thread = threading.Thread(target=self._run, args=(request,), daemon=True)
                    self._thread.start()
        output = list(agents)
        live_keys = {tr.key for tr in live}
        decisions = live + [tr for tr in self.tracks.values() if tr.key not in live_keys
                            and tr.verdict and tr.verdict["decision"] == "drop"]
        for tr in decisions:
            if not tr.verdict:
                continue
            agent = tr.candidate["agent"]
            def matches_target(a):
                # Do not let stationary memory resurrect a cancelled box on
                # a transient no-detection frame. A fresh, unrelated object
                # must not inherit an inactive track's old decision.
                return (same_object(agent, a, self.cfg.match_distance_m)
                        and (tr.key in live_keys or a.get("stationary_memory_confirmed")
                             or float(a.get("perception_age_sec") or 0) > 0))
            matches = [a for a in output if matches_target(a)]
            decision = tr.verdict["decision"]
            annotation = {"vlm_box_decision": decision, "vlm_box_track_id": tr.key,
                          "vlm_box_confidence": tr.verdict["confidence"]}
            if tr.candidate["kind"] == "stationary_vru":
                # A semantic correction never deletes an obstacle or restores a
                # missing detection. Preserve its complete collision geometry.
                output = [{**a, **annotation, "object_type": "static"}
                          if decision == "static" and any(a is m for m in matches)
                          and str(a.get("object_type", "")).startswith(("pedestrian", "walker"))
                          and abs(float(a.get("agent_velo", 0))) <= .2 else a for a in output]
            elif decision == "drop":
                output = [a for a in output if not matches_target(a)]
            elif matches:
                output = [{**a, **annotation} if any(a is m for m in matches) else a for a in output]
            else:
                # In particular, restore a vision-only box removed by the
                # existing static sensor gate. Never forge LiDAR confirmation.
                output.append({**deepcopy(agent), **annotation,
                               "agent_id": f"vlm_box_{tr.key}",
                               "perception_age_sec": max(0.0, now - tr.last_seen)})
            self.last_applied.append({"track_id": tr.key, "decision": decision,
                                      "matched_agents": len(matches)})
            self._record({"kind": "application", "timestamp": now, "track_id": tr.key,
                          "decision": decision,
                          "action": ("removed" if matches else "rejected_proposal") if decision == "drop"
                                    else ("retained" if matches else "added"),
                          "matched_agent_ids": [a.get("agent_id") for a in matches],
                          "candidate": agent})
        return output

    def _ready(self, track, now):
        return len(track.hits) >= self.cfg.min_disagreements

    def finalize(self):
        self._closed = True
        self._epoch += 1
        self.tracks.clear()
        with self._lock:
            self._completed = None
