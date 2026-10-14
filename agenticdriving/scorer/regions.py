"""Layered polar regions used to diversify learned trajectory proposals.

The default layout has three radial layers containing ``1 + 3 + 5 = 9``
angular regions. Radial boundaries are recomputed every frame from the full
proposal pool: ``[0, max_endpoint_radius]`` is split into three equal-width
bands. This keeps a short-horizon prediction frame from collapsing into one
region merely because a fixed boundary was calibrated on longer trajectories.

Candidates outside the nominal bearing fan are clipped into the corresponding
edge region so viable hard-turn modes are not discarded.

Coordinates follow the CARLA convention used by the rest of AgenticDriving:
``[x_right, y_forward]``. Positive bearing is to the right.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


DEFAULT_SECTORS_PER_LAYER = (1, 3, 5)
DEFAULT_TOPK_PER_REGION = 3
DEFAULT_MIN_BEARING_DEG = -45.0
DEFAULT_MAX_BEARING_DEG = 45.0
MIN_DYNAMIC_OUTER_RADIUS_M = 3e-3






@dataclass(frozen=True)
class RandomPerRegionSelection:
    """Random proposal samples drawn independently from every layered region."""

    indices: np.ndarray
    candidate_region_ids: np.ndarray
    candidate_layer_ids: np.ndarray
    endpoint_bearings_deg: np.ndarray
    endpoint_radii_m: np.ndarray
    radial_bounds_m: np.ndarray
    per_region_indices: Tuple[np.ndarray, ...]
    sampled_per_region_indices: Tuple[np.ndarray, ...]


def endpoint_polar_coordinates(proposals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return endpoint bearing in degrees and endpoint radius in metres."""

    proposals = np.asarray(proposals, dtype=np.float32)
    if proposals.ndim != 3 or proposals.shape[-1] < 2:
        raise ValueError(f"proposals must have shape [N,T,2+], got {proposals.shape}")
    if proposals.shape[1] == 0:
        empty = np.full(len(proposals), np.nan, dtype=np.float32)
        return empty.copy(), empty
    endpoints = proposals[:, -1, :2]
    bearings = np.degrees(np.arctan2(endpoints[:, 0], endpoints[:, 1]))
    radii = np.linalg.norm(endpoints, axis=1)
    return bearings.astype(np.float32), radii.astype(np.float32)


def _validated_layout(
    radial_bounds_m: Sequence[float],
    sectors_per_layer: Sequence[int],
) -> tuple[np.ndarray, tuple[int, ...]]:
    bounds = np.asarray(radial_bounds_m, dtype=np.float32).reshape(-1)
    sectors = tuple(int(value) for value in sectors_per_layer)
    if len(bounds) == 0 or len(bounds) != len(sectors):
        raise ValueError("radial bounds and sectors-per-layer must have equal nonzero length")
    if not np.all(np.isfinite(bounds)) or np.any(bounds <= 0.0):
        raise ValueError("radial bounds must be finite and positive")
    if len(bounds) > 1 and not np.all(np.diff(bounds) > 0.0):
        raise ValueError("radial bounds must be strictly increasing")
    if any(value <= 0 for value in sectors):
        raise ValueError("every layer must contain at least one angular region")
    return bounds, sectors


def dynamic_equal_radial_bounds(
    proposals: np.ndarray,
    *,
    layer_count: int = 3,
) -> np.ndarray:
    """Split the current proposal endpoint extent into equal-width layers.

    The range always starts at zero because endpoint radius is non-negative.
    Non-finite endpoints do not influence the boundary calculation. A tiny
    positive fallback extent keeps the boundaries valid for empty/all-zero
    proposal pools while still assigning zero-radius trajectories inward.
    """

    layer_count = int(layer_count)
    if layer_count <= 0:
        raise ValueError("layer_count must be positive")

    _, radii = endpoint_polar_coordinates(proposals)
    finite_radii = radii[np.isfinite(radii)]
    outer_radius = (
        float(np.max(finite_radii)) if len(finite_radii) else 0.0
    )
    outer_radius = max(outer_radius, MIN_DYNAMIC_OUTER_RADIUS_M)
    return np.linspace(
        outer_radius / layer_count,
        outer_radius,
        layer_count,
        dtype=np.float32,
    )


def assign_layered_forward_region_ids(
    proposals: np.ndarray,
    *,
    radial_bounds_m: Optional[Sequence[float]] = None,
    sectors_per_layer: Sequence[int] = DEFAULT_SECTORS_PER_LAYER,
    min_bearing_deg: float = DEFAULT_MIN_BEARING_DEG,
    max_bearing_deg: float = DEFAULT_MAX_BEARING_DEG,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assign finite endpoints to the configured radial/angular region.

    Region IDs are layer-major. With the default layout, region 0 is the whole
    inner layer, regions 1--3 are the middle layer from left to right, and
    regions 4--8 are the outer layer from left to right.
    """

    sectors = tuple(int(value) for value in sectors_per_layer)
    if radial_bounds_m is None:
        radial_bounds_m = dynamic_equal_radial_bounds(
            proposals,
            layer_count=len(sectors),
        )
    bounds, sectors = _validated_layout(radial_bounds_m, sectors)
    min_bearing_deg = float(min_bearing_deg)
    max_bearing_deg = float(max_bearing_deg)
    if not max_bearing_deg > min_bearing_deg:
        raise ValueError("max_bearing_deg must be greater than min_bearing_deg")

    bearings, radii = endpoint_polar_coordinates(proposals)
    region_ids = np.full(len(bearings), -1, dtype=np.int32)
    layer_ids = np.full(len(bearings), -1, dtype=np.int32)
    finite = np.isfinite(bearings) & np.isfinite(radii)
    if not np.any(finite):
        return region_ids, layer_ids, bearings, radii

    # A point exactly on a boundary belongs to the inner adjacent layer.
    layers = np.searchsorted(bounds, radii[finite], side="left")
    layers = np.clip(layers, 0, len(sectors) - 1).astype(np.int32)
    layer_ids[finite] = layers

    upper = np.nextafter(np.float32(max_bearing_deg), np.float32(min_bearing_deg))
    clipped_bearings = np.clip(bearings[finite], min_bearing_deg, upper)
    offsets = np.cumsum((0, *sectors[:-1]), dtype=np.int32)
    finite_indices = np.flatnonzero(finite)
    fan_width = max_bearing_deg - min_bearing_deg
    for layer_id, sector_count in enumerate(sectors):
        in_layer = layers == layer_id
        if not np.any(in_layer):
            continue
        local = np.floor(
            (clipped_bearings[in_layer] - min_bearing_deg)
            / (fan_width / sector_count)
        ).astype(np.int32)
        local = np.clip(local, 0, sector_count - 1)
        region_ids[finite_indices[in_layer]] = offsets[layer_id] + local

    return region_ids, layer_ids, bearings, radii




def select_random_per_layered_region(
    proposals: np.ndarray,
    *,
    samples_per_region: int = DEFAULT_TOPK_PER_REGION,
    rng: Optional[np.random.Generator] = None,
    radial_bounds_m: Optional[Sequence[float]] = None,
    sectors_per_layer: Sequence[int] = DEFAULT_SECTORS_PER_LAYER,
    min_bearing_deg: float = DEFAULT_MIN_BEARING_DEG,
    max_bearing_deg: float = DEFAULT_MAX_BEARING_DEG,
) -> RandomPerRegionSelection:
    """Randomly sample up to ``samples_per_region`` proposals from all regions.

    Unlike :func:`select_target_point_layered_region`, this selection does not
    use the leaderboard target point and never restricts PDMS evaluation to one
    target region.  All ``1 + 3 + 5 = 9`` layered regions participate.  Empty
    regions simply contribute no trajectories, so the default evaluates at
    most 27 proposals.
    """

    proposals = np.asarray(proposals, dtype=np.float32)
    samples_per_region = int(samples_per_region)
    if samples_per_region <= 0:
        raise ValueError("samples_per_region must be positive")
    if rng is None:
        rng = np.random.default_rng()

    sectors = tuple(int(value) for value in sectors_per_layer)
    if radial_bounds_m is None:
        radial_bounds_m = dynamic_equal_radial_bounds(
            proposals,
            layer_count=len(sectors),
        )
    bounds, sectors = _validated_layout(radial_bounds_m, sectors)
    region_ids, layer_ids, bearings, radii = assign_layered_forward_region_ids(
        proposals,
        radial_bounds_m=bounds,
        sectors_per_layer=sectors,
        min_bearing_deg=min_bearing_deg,
        max_bearing_deg=max_bearing_deg,
    )

    per_region = []
    sampled_per_region = []
    sampled = []
    for region_id in range(sum(sectors)):
        candidates = np.asarray(
            np.flatnonzero(region_ids == region_id), dtype=np.int64
        )
        per_region.append(candidates)
        if len(candidates) > samples_per_region:
            chosen = np.asarray(
                rng.choice(
                    candidates,
                    size=samples_per_region,
                    replace=False,
                ),
                dtype=np.int64,
            )
            # Preserve full-pool index order after the random draw so equal
            # PDMS ties remain stable and easy to inspect.
            chosen = np.sort(chosen)
        else:
            chosen = candidates.copy()
        sampled_per_region.append(chosen)
        sampled.extend(chosen.tolist())

    return RandomPerRegionSelection(
        indices=np.asarray(sampled, dtype=np.int64),
        candidate_region_ids=region_ids,
        candidate_layer_ids=layer_ids,
        endpoint_bearings_deg=bearings,
        endpoint_radii_m=radii,
        radial_bounds_m=bounds,
        per_region_indices=tuple(per_region),
        sampled_per_region_indices=tuple(sampled_per_region),
    )
