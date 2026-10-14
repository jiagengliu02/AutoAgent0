"""Pure-numpy adapter: BridgeDrive aux-head outputs -> PDMS judge inputs.

Maps the two harvested perception products onto the judge's existing
contracts so `perception.source: bridgedrive` can feed the acceptance path
with the same agent-dict schema ground truth and BEVFormer already use:

1. CenterNet box rows -> perception agent dicts (dynamics for NC/TTC).
2. BEV-semantic WALKER/BIKER masks -> connected-component pedestrian
   backstops when the CenterNet head misses a VRU.
3. BEV-semantic OBSTACLE mask -> connected-component static agent dicts
   (unboxed clutter joins the same NC/TTC pipeline behind its own flag).

Source-verified conventions:
- Box rows are (x, y, w, h, yaw, velocity, brake, class, score) in LEAD's
  vehicle system: x forward / y right, meters. w/h are CARLA HALF extents
  (extent.x = half length, extent.y = half width; stored unscaled at
  lead/data_loader/carla_dataset_utils.py:274-276). yaw is actor_yaw -
  ego_yaw in radians. velocity is the actor's SIGNED ABSOLUTE forward speed
  along its own heading (dot(world velocity, heading vector);
  lead/expert/expert.py:2734-2753) -- not ego-relative; reversing is
  negative.
- The BEV-semantic argmax grid is (320, 384) = (y, x) over x in [-32, 64] m
  forward, y in [-40, 40] m right at 0.25 m cells (pixels_per_meter=4.0,
  checkpoint config); WALKER/OBSTACLE/BIKER are classes 5/6/9
  (TransfuserBEVSemanticClass).

No torch, no LEAD imports: unit-testable everywhere. Confidence filtering
happens here; the knob's default (0.3) IS LEAD's own inference default
(bb_confidence_threshold) — changing it is a documented deviation, not a
tuning surface. No NMS anywhere on this path: LEAD's closed-loop applies
none, and the faithful transcription forwards duplicates as-is.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np

from agenticdriving.planners.sparsedrivev2.carla_inputs import (
    carla_transform_matrix,
)

# TransfuserBoundingBoxClass (lead/common/constants.py:76-86)
BOX_CLASSES_VEHICLE = frozenset({0, 4, 6})  # VEHICLE, SPECIAL, PARKING
BOX_CLASSES_PEDESTRIAN = frozenset({1, 7})  # WALKER, BIKER
BOX_CLASSES_STATIC = frozenset({5})  # OBSTACLE
# TRAFFIC_LIGHT=2 / STOP_SIGN=3 dropped, mirroring the ground-truth feed's
# traffic.* actor exclusion.

# TransfuserBEVSemanticClass (lead/common/constants.py:127-136)
BEV_WALKER_CLASS = 5
BEV_OBSTACLE_CLASS = 6
BEV_BIKER_CLASS = 9
BEV_PEDESTRIAN_CLASSES = frozenset({BEV_WALKER_CLASS, BEV_BIKER_CLASS})

# Aux-BEV window (checkpoint config: min/max_x/y_meter, pixels_per_meter=4.0
# at the label layer; the decoded raster was measured at 0.25 m cells).
BEV_MIN_X_M = -32.0
BEV_MAX_X_M = 64.0
BEV_MIN_Y_M = -40.0
BEV_MAX_Y_M = 40.0
BEV_CELL_M = 0.25
BEV_SHAPE = (320, 384)  # (rows=y, cols=x)
# Cover the complete native BridgeDrive BEV rectangle.  The previous 50 m
# radius discarded valid predictions in the front 50--64 m strip and in the
# outer corners even though those cells are present in the trained head.
BRIDGEDRIVE_FULL_BEV_RANGE_M = math.hypot(
    max(abs(BEV_MIN_X_M), abs(BEV_MAX_X_M)),
    max(abs(BEV_MIN_Y_M), abs(BEV_MAX_Y_M)),
)

PERCEPTION_SOURCE = "bridgedrive_centernet"
SEMANTIC_PEDESTRIAN_SOURCE = "bridgedrive_bev_semantic_vru"
STATIONARY_SPEED_MPS = 0.3
STATIC_DUPLICATE_CENTER_M = 0.5
STATIC_DUPLICATE_EXTENT_M = 0.75
STATIC_DUPLICATE_YAW_RAD = 0.3


def _pose_matrix(ego_pos, ego_rot) -> np.ndarray:
    return carla_transform_matrix(ego_pos, ego_rot)


def boxes_to_judge_agents(
    boxes_vehicle_system: np.ndarray,
    ego_pos,
    ego_rot,
    timestamp: float,
    *,
    score_threshold: float = 0.3,
    score_threshold_by_class=None,
    max_range_m: float = BRIDGEDRIVE_FULL_BEV_RANGE_M,
) -> List[Dict[str, Any]]:
    """CenterNet rows [K, 9] (vehicle system, meters) -> judge agent dicts.

    Output rows carry the same fields the acceptance path reads from the
    ground-truth/BEVFormer feeds (agent_pos_world, agent_heading,
    agent_vel_vec, agent_extent, object_type, ...). agent_lane_key is
    intentionally absent: lane membership stays on the privileged proposal
    path (split-feeds design).
    """
    boxes = np.asarray(boxes_vehicle_system, dtype=np.float64).reshape(-1, 9)
    pose = _pose_matrix(ego_pos, ego_rot)
    ego_yaw = float(np.asarray(ego_rot, dtype=np.float64).reshape(-1)[2])

    # CenterNet can emit the same stationary object twice with competing
    # VEHICLE and OBSTACLE labels.  Keeping both is not harmless here: the
    # false dynamic copy enters adjacent-lane flow checks and can permanently
    # veto an otherwise safe static-obstacle bypass.  If a near-identical,
    # accepted static row exists, suppress only the stationary dynamic copy.
    # Moving vehicles are never affected.
    accepted_static_rows = [
        row
        for row in boxes
        if row[8] >= score_threshold and int(row[7]) in BOX_CLASSES_STATIC
    ]

    def overlaps_static_duplicate(row: np.ndarray) -> bool:
        if int(row[7]) not in BOX_CLASSES_VEHICLE:
            return False
        if abs(float(row[5])) > STATIONARY_SPEED_MPS:
            return False
        for static in accepted_static_rows:
            if math.hypot(float(row[0] - static[0]), float(row[1] - static[1])) > (
                STATIC_DUPLICATE_CENTER_M
            ):
                continue
            if max(abs(float(row[2] - static[2])), abs(float(row[3] - static[3]))) > (
                STATIC_DUPLICATE_EXTENT_M
            ):
                continue
            yaw_error = abs(
                math.atan2(
                    math.sin(float(row[4] - static[4])),
                    math.cos(float(row[4] - static[4])),
                )
            )
            yaw_error = min(yaw_error, abs(math.pi - yaw_error))
            if yaw_error <= STATIC_DUPLICATE_YAW_RAD:
                return True
        return False

    agents: List[Dict[str, Any]] = []
    for index, row in enumerate(boxes):
        x, y, half_len, half_wid, yaw, velocity, _brake, clazz, score = row
        if score < (score_threshold_by_class or {}).get(int(clazz), score_threshold):
            continue
        # LEAD/CenterNet pads absent detections with a high-confidence box at
        # approximately (-32, -40) and near-zero/negative extents.  Passing
        # this sentinel through as a vehicle creates a fake obstacle in PDMS
        # (and can poison TTC/NC decisions), so reject invalid geometry before
        # any world-frame conversion.  Real actors are comfortably larger than
        # this floor; the threshold also catches quantized ~1 mm padding boxes.
        if (
            not np.isfinite([x, y, half_len, half_wid, yaw, velocity, score]).all()
            or half_len < 0.05
            or half_wid < 0.05
        ):
            continue
        if math.hypot(x, y) > max_range_m:
            continue
        if overlaps_static_duplicate(row):
            continue
        clazz = int(clazz)
        if clazz in BOX_CLASSES_VEHICLE:
            object_type = "vehicle"
        elif clazz in BOX_CLASSES_PEDESTRIAN:
            object_type = "pedestrian"
        elif clazz in BOX_CLASSES_STATIC:
            object_type = "static"
        else:
            continue

        # Vehicle system (x fwd, y right) == CARLA ego-local axes: transform
        # the box center with the full actor-to-world matrix (z from the ego
        # plane; boxes are 2D).
        world = pose @ np.array([x, y, 0.0, 1.0], dtype=np.float64)
        heading_world = math.atan2(
            math.sin(ego_yaw + yaw), math.cos(ego_yaw + yaw)
        )
        # Label convention: signed absolute forward speed along the actor's
        # own heading, so the world velocity vector is v * heading direction
        # (a reversing actor keeps its heading and gets a negative v).
        vel_vec = [
            float(velocity) * math.cos(heading_world),
            float(velocity) * math.sin(heading_world),
        ]
        agents.append(
            {
                "agent_id": f"bd_{index}",
                "timestamp": float(timestamp),
                "agent_pos_world": [
                    float(world[0]),
                    float(world[1]),
                    float(world[2]),
                ],
                "agent_heading": float(heading_world),
                "agent_velo": abs(float(velocity)),
                "agent_vel_vec": vel_vec,
                # GT publishes [2*extent.x, 2*extent.y] = full length/width;
                # the head predicts CARLA half extents, so double them.
                "agent_extent": [
                    float(2.0 * half_len),
                    float(2.0 * half_wid),
                ],
                "object_type": object_type,
                "perception_source": PERCEPTION_SOURCE,
                "detection_score": float(score),
                "detection_class": clazz,
            }
        )
    return agents


def _erode_once(mask: np.ndarray) -> np.ndarray:
    padded = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=bool)
    padded[1:-1, 1:-1] = mask
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            out &= padded[
                1 + dr : 1 + dr + mask.shape[0],
                1 + dc : 1 + dc + mask.shape[1],
            ]
    return out


def erode_mask(mask: np.ndarray, cells: int = 1) -> np.ndarray:
    """Binary erosion with a 3x3 structuring element, `cells` iterations.

    The anti-false-positive knob: a single-cell blob (0.25 m) cannot veto a
    trajectory; only >= (2*cells+1)-cell-wide obstacle regions survive.
    """
    out = np.asarray(mask, dtype=bool)
    for _ in range(max(0, int(cells))):
        out = _erode_once(out)
    return out


def pack_occupancy(
    bev_argmax: np.ndarray,
    ego_pos,
    ego_rot,
    timestamp: float,
    *,
    obstacle_class: int = BEV_OBSTACLE_CLASS,
    erosion_cells: int = 1,
) -> Dict[str, Any]:
    """BEV-semantic argmax grid -> packed capture-pose occupancy record.

    ~15 KB/tick packed; consumed by occupancy_static_agents (components ->
    NC/TTC statics). Geometry is anchored to the CAPTURE pose: the mask is
    valid from where it was seen, never re-registered to the current pose.
    """
    grid = np.asarray(bev_argmax)
    if grid.shape != BEV_SHAPE:
        raise ValueError(f"expected {BEV_SHAPE} argmax grid, got {grid.shape}")
    mask = erode_mask(grid == int(obstacle_class), erosion_cells)
    return {
        "packed": np.packbits(mask),
        "shape": list(mask.shape),
        "min_x_m": BEV_MIN_X_M,
        "min_y_m": BEV_MIN_Y_M,
        "cell_m": BEV_CELL_M,
        "erosion_cells": int(erosion_cells),
        "obstacle_cells": int(mask.sum()),
        "ego_pos": [float(v) for v in np.asarray(ego_pos).reshape(-1)[:3]],
        "ego_rot": [float(v) for v in np.asarray(ego_rot).reshape(-1)[:3]],
        "timestamp": float(timestamp),
    }


def unpack_occupancy(occupancy: Dict[str, Any]) -> np.ndarray:
    shape = tuple(int(v) for v in occupancy["shape"])
    packed = np.asarray(occupancy["packed"], dtype=np.uint8)
    flat = np.unpackbits(packed, count=shape[0] * shape[1])
    return flat.reshape(shape).astype(bool)


def _label_components(mask: np.ndarray) -> list:
    """4-connected components of a bool grid -> list of (rows, cols) index
    arrays. Iterative flood fill; the eroded OBSTACLE mask is sparse."""
    remaining = mask.copy()
    components = []
    rows_all, cols_all = np.nonzero(remaining)
    for seed_row, seed_col in zip(rows_all, cols_all):
        if not remaining[seed_row, seed_col]:
            continue
        stack = [(int(seed_row), int(seed_col))]
        remaining[seed_row, seed_col] = False
        cells = []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (
                    0 <= nr < remaining.shape[0]
                    and 0 <= nc < remaining.shape[1]
                    and remaining[nr, nc]
                ):
                    remaining[nr, nc] = False
                    stack.append((nr, nc))
        component = np.asarray(cells, dtype=np.int64)
        components.append((component[:, 0], component[:, 1]))
    return components


def _component_bounds(
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    cell_m: float,
    min_x_m: float,
    min_y_m: float,
    pad_m: float = 0.0,
) -> tuple[float, float, float, float]:
    """Inclusive raster component -> metric cell-boundary AABB."""
    x_lo = min_x_m + cols.min() * cell_m - pad_m
    x_hi = min_x_m + (cols.max() + 1) * cell_m + pad_m
    y_lo = min_y_m + rows.min() * cell_m - pad_m
    y_hi = min_y_m + (rows.max() + 1) * cell_m + pad_m
    return float(x_lo), float(x_hi), float(y_lo), float(y_hi)


def bev_semantic_pedestrian_agents(
    bev_argmax: np.ndarray,
    ego_pos,
    ego_rot,
    timestamp: float,
    *,
    pedestrian_classes=BEV_PEDESTRIAN_CLASSES,
    min_cells: int = 2,
    max_components: int = 64,
    max_range_m: float = BRIDGEDRIVE_FULL_BEV_RANGE_M,
    min_extent_m: float = 0.6,
) -> List[Dict[str, Any]]:
    """Semantic WALKER/BIKER components -> conservative pedestrian agents.

    These zero-velocity boxes are a class-specific VRU backstop, not generic
    semantic occupancy. CenterNet remains the primary source because it
    supplies heading and velocity; callers should remove semantic components
    that overlap a CenterNet pedestrian before publishing the combined list.
    """
    grid = np.asarray(bev_argmax)
    if grid.shape != BEV_SHAPE:
        raise ValueError(f"expected {BEV_SHAPE} argmax grid, got {grid.shape}")
    classes = frozenset(int(value) for value in pedestrian_classes)
    if not classes:
        return []
    mask = np.isin(grid, tuple(classes))
    components = _label_components(mask)
    components = [rc for rc in components if len(rc[0]) >= max(1, int(min_cells))]
    if not components:
        return []
    components.sort(key=lambda rc: -len(rc[0]))
    components = components[:max(0, int(max_components))]

    pose = _pose_matrix(ego_pos, ego_rot)
    ego_yaw = float(np.asarray(ego_rot, dtype=np.float64).reshape(-1)[2])
    timestamp = float(timestamp)
    minimum_extent = max(0.0, float(min_extent_m))
    agents: List[Dict[str, Any]] = []
    for index, (rows, cols) in enumerate(components):
        x_lo, x_hi, y_lo, y_hi = _component_bounds(
            rows,
            cols,
            cell_m=BEV_CELL_M,
            min_x_m=BEV_MIN_X_M,
            min_y_m=BEV_MIN_Y_M,
        )
        center_local = ((x_lo + x_hi) / 2.0, (y_lo + y_hi) / 2.0)
        if math.hypot(*center_local) > float(max_range_m):
            continue
        world = pose @ np.array(
            [center_local[0], center_local[1], 0.0, 1.0], dtype=np.float64
        )
        semantic_classes = sorted(
            int(value) for value in np.unique(grid[rows, cols]) if int(value) in classes
        )
        agents.append(
            {
                "agent_id": f"bdsemvru_{index}",
                "timestamp": timestamp,
                "agent_pos_world": [
                    float(world[0]),
                    float(world[1]),
                    float(world[2]),
                ],
                "agent_heading": ego_yaw,
                "agent_velo": 0.0,
                "agent_vel_vec": [0.0, 0.0],
                "agent_extent": [
                    max(minimum_extent, float(x_hi - x_lo)),
                    max(minimum_extent, float(y_hi - y_lo)),
                ],
                "object_type": "pedestrian",
                "perception_source": SEMANTIC_PEDESTRIAN_SOURCE,
                "semantic_classes": semantic_classes,
                "occupancy_cells": int(len(rows)),
            }
        )
    return agents


def merge_semantic_pedestrian_backstop(
    primary_agents: List[Dict[str, Any]],
    semantic_agents: List[Dict[str, Any]],
    *,
    duplicate_center_m: float = 2.0,
) -> List[Dict[str, Any]]:
    """Append semantic VRUs unless a primary pedestrian already covers them."""
    primary = list(primary_agents)
    primary_pedestrians = [
        item for item in primary if item.get("object_type") == "pedestrian"
    ]

    def radius(item: Dict[str, Any]) -> float:
        extent = list(item.get("agent_extent") or (0.6, 0.6))
        while len(extent) < 2:
            extent.append(0.6)
        return max(0.3, 0.5 * math.hypot(float(extent[0]), float(extent[1])))

    merged = list(primary)
    for semantic in semantic_agents:
        sx, sy = semantic.get("agent_pos_world", (0.0, 0.0))[:2]
        semantic_radius = radius(semantic)
        duplicate = False
        for detected in primary_pedestrians:
            dx, dy = detected.get("agent_pos_world", (0.0, 0.0))[:2]
            threshold = max(
                float(duplicate_center_m),
                semantic_radius + radius(detected),
            )
            if math.hypot(float(sx) - float(dx), float(sy) - float(dy)) <= threshold:
                duplicate = True
                break
        if not duplicate:
            merged.append(semantic)
    return merged


def occupancy_static_agents(
    occupancy: Dict[str, Any],
    *,
    min_cells: int = 2,
    max_components: int = 64,
    preserve_shape: bool = False,
) -> List[Dict[str, Any]]:
    """OBSTACLE-mask components -> synthetic static agent dicts for NC/TTC.

    The ingrained statics path: each 4-connected component becomes one
    zero-velocity static box that rides the SAME privileged_agents pipeline
    as every other obstacle — same NC gate, same recovery reasons, same
    failure-memory semantics. Boxes are axis-aligned in the CAPTURE ego
    frame (heading = capture ego yaw); bounds are re-dilated by
    erosion_cells when the record was eroded.

    False-positive filtering is COMPONENT-level (``min_cells``), not
    cell-level erosion: measured, 1-cell erosion deletes any isolated object
    up to 2 cells (0.5 m) wide — small props vanish from the mask entirely.
    Erosion
    filters by local thickness; ``min_cells`` filters by object size, so a
    2-cell gnome survives while 1-cell speckle still dies. Pair with
    erosion_cells=0 at pack time. Components beyond ``max_components``
    (largest first) are dropped — count survivors against
    ``occupancy['obstacle_cells']`` when auditing, never assume full
    coverage."""
    mask = unpack_occupancy(occupancy)
    components = _label_components(mask)
    components = [rc for rc in components if len(rc[0]) >= max(1, int(min_cells))]
    if not components:
        return []
    components.sort(key=lambda rc: -len(rc[0]))
    components = components[:max_components]

    cell = float(occupancy["cell_m"])
    min_x = float(occupancy["min_x_m"])
    min_y = float(occupancy["min_y_m"])
    pad = float(occupancy.get("erosion_cells", 0)) * cell
    pose = _pose_matrix(occupancy["ego_pos"], occupancy["ego_rot"])
    ego_yaw = float(np.asarray(occupancy["ego_rot"]).reshape(-1)[2])
    timestamp = float(occupancy.get("timestamp", 0.0) or 0.0)

    agents: List[Dict[str, Any]] = []
    for index, (rows, cols) in enumerate(components):
        # cell (r, c) spans x in [min_x + c*cell, +cell), y likewise
        x_lo, x_hi, y_lo, y_hi = _component_bounds(
            rows,
            cols,
            cell_m=cell,
            min_x_m=min_x,
            min_y_m=min_y,
            pad_m=pad,
        )
        center_local = ((x_lo + x_hi) / 2.0, (y_lo + y_hi) / 2.0)
        world = pose @ np.array(
            [center_local[0], center_local[1], 0.0, 1.0], dtype=np.float64
        )
        agents.append(
            {
                "agent_id": f"bdocc_{index}",
                "timestamp": timestamp,
                "agent_pos_world": [
                    float(world[0]),
                    float(world[1]),
                    float(world[2]),
                ],
                # axis-aligned in the capture frame: heading = capture yaw,
                # extent = [x-span (length), y-span (width)]
                "agent_heading": ego_yaw,
                "agent_velo": 0.0,
                "agent_vel_vec": [0.0, 0.0],
                "agent_extent": [float(x_hi - x_lo), float(y_hi - y_lo)],
                "object_type": "static",
                "perception_source": "bridgedrive_bev_occupancy",
                "occupancy_cells": int(len(rows)),
            }
        )
        if preserve_shape:
            from .occupancy_geometry import world_parts

            # Outer box is ONLY an association envelope. The caller expands
            # these exact world-frame parts after object-memory association.
            agents[-1]['occupancy_parts'] = world_parts(
                rows, cols, pose=pose, heading=ego_yaw, cell=cell,
                min_x=min_x, min_y=min_y, pad=pad,
            )
    return agents
