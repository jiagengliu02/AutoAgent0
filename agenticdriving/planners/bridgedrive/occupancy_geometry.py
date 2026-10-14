"""Lossless rectangular cover of BEV regions; world geometry survives memory."""
from __future__ import annotations

import numpy as np


def rectangular_cover(rows, cols):
    """Merge identical occupied row runs, never fill an unoccupied cell."""
    active = {}
    rectangles = []
    for row in range(int(rows.min()), int(rows.max()) + 2):
        occupied = np.sort(cols[rows == row])
        runs = []
        if len(occupied):
            for run in np.split(occupied, np.flatnonzero(np.diff(occupied) > 1) + 1):
                runs.append((int(run[0]), int(run[-1]) + 1))
        current = set(runs)
        for span in list(active):
            if span not in current:
                rectangles.append((active.pop(span), row, *span))
        for span in runs:
            active.setdefault(span, row)
    return rectangles


def world_parts(rows, cols, *, pose, heading, cell, min_x, min_y, pad):
    parts = []
    for r0, r1, c0, c1 in rectangular_cover(rows, cols):
        x0, x1 = min_x+c0*cell-pad, min_x+c1*cell+pad
        y0, y1 = min_y+r0*cell-pad, min_y+r1*cell+pad
        world = pose @ np.array([(x0+x1)/2, (y0+y1)/2, 0., 1.])
        parts.append(dict(agent_pos_world=world[:3].tolist(), agent_heading=heading,
                          agent_extent=[x1-x0, y1-y0],
                          occupancy_cells=(r1-r0)*(c1-c0)))
    return parts
