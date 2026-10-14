from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from agenticdriving.planners.fallback_rule_based.recovery_env import (
    RULE_BASED_ENV_DEFAULTS,
    RULE_BASED_ENV_FIELD_NAMES,
    get_prefixed_rule_based_env_value,
)


LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecoveryConfig:
    enabled: bool
    python_bin: str
    topk: int
    include_privileged_info: bool
    source_name: str
    allowed_primitives: Tuple[str, ...] = ()
    lane_bypass_min_gap_m: float = 0.0
    lane_bypass_clearance_buffer_m: float = 1.0
    lane_bypass_required_time_sec: float = 3.5
    lane_bypass_min_time_gap_sec: float = 4.5
    obstacle_clearance_m: float = 8.0


def _coerce_env_value(raw_value: Any, default_value: Any) -> Any:
    if isinstance(default_value, bool):
        return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default_value, int) and not isinstance(default_value, bool):
        return int(raw_value)
    if isinstance(default_value, float):
        return float(raw_value)
    return str(raw_value)


def resolve_recovery_config(
    *,
    planner_python_bin: str = "",
    prefixes: Tuple[str, ...] = ("PLANNER_RULE_BASED_", "RAP_RULE_BASED_"),
) -> RecoveryConfig:
    values: Dict[str, Any] = {}
    for suffix, field_name in RULE_BASED_ENV_FIELD_NAMES.items():
        default_value = RULE_BASED_ENV_DEFAULTS[suffix]
        if suffix == "PYTHON_BIN":
            default_value = planner_python_bin
        raw_value = get_prefixed_rule_based_env_value(
            suffix,
            default=default_value,
            prefixes=prefixes,
        )
        values[field_name] = _coerce_env_value(raw_value, default_value)

    topk = max(1, int(values["topk"]))
    allowed_primitives = tuple(
        name.strip()
        for name in str(values.get("allowed_primitives", "")).split(",")
        if name.strip()
    )
    # Catch stale/renamed ladder ids once at startup instead of on every
    # recovery tick. get_primitive_proposals_and_scores() drops unknown ids
    # defensively, but a config that names maneuvers which no longer exist is
    # almost always a mistake worth shouting about before the route starts.
    if allowed_primitives:
        from agenticdriving.planners.fallback_rule_based.primitives import (
            CANDIDATE_LADDER,
        )

        known = {name for name, _code in CANDIDATE_LADDER}
        unknown = sorted(set(allowed_primitives) - known)
        if unknown:
            LOG.error(
                "recovery.allowed_primitives names %d unknown primitive(s): "
                "%s. They will be ignored; valid ladder ids are: %s",
                len(unknown),
                ", ".join(unknown),
                ", ".join(name for name, _code in CANDIDATE_LADDER),
            )
    return RecoveryConfig(
        enabled=bool(values["enabled"]),
        python_bin=str(values["python_bin"]),
        topk=topk,
        include_privileged_info=bool(values["include_privileged_info"]),
        source_name=str(values["source_name"] or "rule_based"),
        allowed_primitives=allowed_primitives,
        lane_bypass_min_gap_m=max(
            0.0, float(values["lane_bypass_min_gap_m"])
        ),
        lane_bypass_clearance_buffer_m=max(
            0.0, float(values["lane_bypass_clearance_buffer_m"])
        ),
        lane_bypass_required_time_sec=max(
            0.0, float(values["lane_bypass_required_time_sec"])
        ),
        lane_bypass_min_time_gap_sec=max(
            0.0, float(values["lane_bypass_min_time_gap_sec"])
        ),
        obstacle_clearance_m=max(0.0, float(values["obstacle_clearance_m"])),
    )
