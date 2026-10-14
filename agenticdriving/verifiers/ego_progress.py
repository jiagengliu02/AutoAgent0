"""Ego-progress (EP) reference for the PDMS verifier.

Ported from AutoAgent0's ``autoagent0/verifiers/ego_progress.py``. Two pace
sources are supported:

- ``frame_dt`` (CARLA adapter): the route polyline is a logged human
  drive sampled every ``frame_dt`` seconds, so expected progress over a
  horizon is how far the human advanced in that time.
- ``pace_speed_mps`` (CARLA): leaderboard routes carry no timing, so expected
  progress is ``pace_speed_mps * horizon``, saturated at the remaining route
  length so the escape hatch still kicks in near the route end.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class EgoProgressReference:
    """Route polyline + pace lookup backing the EP soft term.

    Parameters
    ----------
    route_xy : (M, 2) array
        Reference route in the same world frame as the verifier's world-frame
        plan polyline.
    frame_dt : float, optional
        Seconds between consecutive ``route_xy`` poses (logged-drive pace).
    pace_speed_mps : float, optional
        Constant expected speed; alternative to ``frame_dt`` when the route
        carries no timing. Exactly one of the two must be set.
    progress_threshold : float
        Achievable-progress floor in meters below which EP is 1.0.
    """

    def __init__(
        self,
        route_xy,
        frame_dt: Optional[float] = None,
        pace_speed_mps: Optional[float] = None,
        progress_threshold: float = 5.0,
    ):
        if (frame_dt is None) == (pace_speed_mps is None):
            raise ValueError("Set exactly one of frame_dt / pace_speed_mps")
        route_xy = np.asarray(route_xy, dtype=np.float64).reshape(-1, 2)
        self._xy = route_xy
        segments = np.diff(route_xy, axis=0)
        seg_len = np.linalg.norm(segments, axis=1)
        self._stations = np.concatenate([[0.0], np.cumsum(seg_len)])
        self._frame_dt = None if frame_dt is None else float(frame_dt)
        self._pace_speed = None if pace_speed_mps is None else float(pace_speed_mps)
        self._threshold = float(progress_threshold)

    def station(self, point) -> float:
        """Arc-length station (meters from route start) of ``point``'s closest
        projection onto the route polyline."""
        return float(self.stations(np.asarray(point, dtype=np.float64)[None, :])[0])

    def stations(self, points) -> np.ndarray:
        """Vectorized :meth:`station` for ``(N, 2)`` world-frame points."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if len(points) == 0:
            return np.zeros(0, dtype=np.float64)
        a, b = self._xy[:-1], self._xy[1:]
        ab = b - a
        seg_len2 = np.einsum("ij,ij->i", ab, ab)
        delta = points[:, None, :] - a[None, :, :]
        t = np.einsum("nsi,si->ns", delta, ab) / np.maximum(
            seg_len2[None, :], 1e-12
        )
        t = np.clip(np.where(seg_len2 > 0, t, 0.0), 0.0, 1.0)
        proj = a[None, :, :] + t[:, :, None] * ab[None, :, :]
        residual = points[:, None, :] - proj
        d2 = np.einsum("nsi,nsi->ns", residual, residual)
        indices = np.argmin(d2, axis=1)
        rows = np.arange(len(points))
        return self._stations[indices] + t[rows, indices] * np.sqrt(
            seg_len2[indices]
        )

    def _nearest_vertex(self, point) -> int:
        p = np.asarray(point, dtype=np.float64)
        d2 = np.einsum("ij,ij->i", self._xy - p, self._xy - p)
        return int(np.argmin(d2))

    def expected_progress(self, point, horizon_s: float) -> float:
        """Meters the reference pace advances over ``horizon_s`` starting from
        the route pose nearest to ``point``; saturates at the route end."""
        if self._pace_speed is not None:
            remaining = float(self._stations[-1]) - self.station(point)
            return float(min(self._pace_speed * horizon_s, max(remaining, 0.0)))
        j = self._nearest_vertex(point)
        k = max(1, int(round(horizon_s / self._frame_dt)))
        end = min(j + k, len(self._stations) - 1)
        return float(self._stations[end] - self._stations[j])

    def ego_progress(self, planned_xy, horizon_s: float) -> float:
        """EP in [0, 1] for a world-frame plan polyline over ``horizon_s``."""
        planned_xy = np.asarray(planned_xy, dtype=np.float64)
        if len(self._xy) < 2 or planned_xy.shape[0] < 2:
            return 1.0
        progress = max(self.station(planned_xy[-1]) - self.station(planned_xy[0]), 0.0)
        expected = self.expected_progress(planned_xy[0], horizon_s)
        if expected < self._threshold:
            return 1.0
        return float(np.clip(progress / expected, 0.0, 1.0))

    def ego_progress_batch(self, planned_xy, horizon_s: float) -> np.ndarray:
        """Vectorized EP for ``(B, T, 2)`` world-frame trajectories."""
        planned_xy = np.asarray(planned_xy, dtype=np.float64)
        if planned_xy.ndim != 3:
            raise ValueError(
                f"planned_xy must have shape (B, T, 2), got {planned_xy.shape}"
            )
        batch_size = len(planned_xy)
        if len(self._xy) < 2 or planned_xy.shape[1] < 2:
            return np.ones(batch_size, dtype=np.float64)

        starts = planned_xy[:, 0, :2]
        ends = planned_xy[:, -1, :2]
        start_stations = self.stations(starts)
        end_stations = self.stations(ends)
        progress = np.maximum(end_stations - start_stations, 0.0)

        if self._pace_speed is not None:
            remaining = np.maximum(float(self._stations[-1]) - start_stations, 0.0)
            expected = np.minimum(self._pace_speed * float(horizon_s), remaining)
        else:
            delta = self._xy[None, :, :] - starts[:, None, :]
            d2 = np.einsum("nvi,nvi->nv", delta, delta)
            nearest = np.argmin(d2, axis=1)
            step_count = max(1, int(round(float(horizon_s) / self._frame_dt)))
            end_indices = np.minimum(nearest + step_count, len(self._stations) - 1)
            expected = self._stations[end_indices] - self._stations[nearest]

        scores = np.ones(batch_size, dtype=np.float64)
        valid = expected >= self._threshold
        scores[valid] = np.clip(
            progress[valid] / np.maximum(expected[valid], 1e-12), 0.0, 1.0
        )
        return scores
