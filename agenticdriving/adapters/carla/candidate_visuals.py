"""CARLA candidate visualization styles."""
from __future__ import annotations

import colorsys
from dataclasses import dataclass
from typing import Tuple


CARRY_PREVIOUS_SOURCE = "carry_prev"
CARRY_PREVIOUS_COLOR_BGR = (0, 0, 0)
CARRY_PREVIOUS_COLOR_NAME = "black"

_BASE_COLOR_NAMES = (
    "red",
    "orange",
    "amber",
    "yellow",
    "lime",
    "green",
    "teal",
    "cyan",
    "sky",
    "blue",
    "indigo",
    "violet",
    "magenta",
    "rose",
)
_GOLDEN_RATIO_STEP = 0.3819660112501051
_START_HUE = 0.58


@dataclass(frozen=True)
class CandidateVisualStyle:
    color_bgr: Tuple[int, int, int]
    color_name: str


def _rank_to_bgr(rank: int) -> Tuple[int, int, int]:
    hue = (_START_HUE + rank * _GOLDEN_RATIO_STEP) % 1.0
    saturation = 0.95 if rank % 2 == 0 else 0.78
    value = 1.0 if (rank // 2) % 2 == 0 else 0.88
    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
    return (
        int(round(blue * 255)),
        int(round(green * 255)),
        int(round(red * 255)),
    )


def _rank_to_name(rank: int) -> str:
    hue = (_START_HUE + rank * _GOLDEN_RATIO_STEP) % 1.0
    hue_idx = int(round(hue * len(_BASE_COLOR_NAMES))) % len(_BASE_COLOR_NAMES)
    return f"{_BASE_COLOR_NAMES[hue_idx]}_{rank + 1}"


def get_candidate_visual_style(source: str, non_carry_rank: int) -> CandidateVisualStyle:
    if source == CARRY_PREVIOUS_SOURCE:
        return CandidateVisualStyle(
            color_bgr=CARRY_PREVIOUS_COLOR_BGR,
            color_name=CARRY_PREVIOUS_COLOR_NAME,
        )
    return CandidateVisualStyle(
        color_bgr=_rank_to_bgr(non_carry_rank),
        color_name=_rank_to_name(non_carry_rank),
    )
