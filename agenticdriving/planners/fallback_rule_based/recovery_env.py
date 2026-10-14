from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Tuple


RULE_BASED_ENV_DEFAULTS: Dict[str, Any] = {
    "ENABLED": False,
    "PYTHON_BIN": "",
    "TOPK": 3,
    "INCLUDE_PRIVILEGED_INFO": True,
    "SOURCE_NAME": "rule_based",
    # Comma-separated stable ids from primitives.CANDIDATE_LADDER. Empty means
    # the complete ladder. YAML lists are normalized by build_prefixed_rule_based_env.
    "ALLOWED_PRIMITIVES": "",
    "LANE_BYPASS_MIN_GAP_M": 0.0,
    "LANE_BYPASS_CLEARANCE_BUFFER_M": 1.0,
    "LANE_BYPASS_REQUIRED_TIME_SEC": 3.5,
    "LANE_BYPASS_MIN_TIME_GAP_SEC": 4.5,
    "OBSTACLE_CLEARANCE_M": 8.0,
}


RULE_BASED_ENV_FIELD_NAMES: Dict[str, str] = {
    "ENABLED": "enabled",
    "PYTHON_BIN": "python_bin",
    "TOPK": "topk",
    "INCLUDE_PRIVILEGED_INFO": "include_privileged_info",
    "SOURCE_NAME": "source_name",
    "ALLOWED_PRIMITIVES": "allowed_primitives",
    "LANE_BYPASS_MIN_GAP_M": "lane_bypass_min_gap_m",
    "LANE_BYPASS_CLEARANCE_BUFFER_M": "lane_bypass_clearance_buffer_m",
    "LANE_BYPASS_REQUIRED_TIME_SEC": "lane_bypass_required_time_sec",
    "LANE_BYPASS_MIN_TIME_GAP_SEC": "lane_bypass_min_time_gap_sec",
    "OBSTACLE_CLEARANCE_M": "obstacle_clearance_m",
}


def build_prefixed_rule_based_env(
    rule_based_cfg: Any,
    *,
    planner_python_bin: str = "",
    prefixes: Iterable[str] = ("PLANNER_RULE_BASED_", "RAP_RULE_BASED_"),
) -> Dict[str, Any]:
    env_values: Dict[str, Any] = {}
    for suffix, field_name in RULE_BASED_ENV_FIELD_NAMES.items():
        default_value = RULE_BASED_ENV_DEFAULTS[suffix]
        if suffix == "PYTHON_BIN":
            default_value = planner_python_bin
        value = rule_based_cfg.get(field_name, default_value)
        if field_name == "allowed_primitives" and isinstance(value, (list, tuple, set)):
            value = ",".join(str(item).strip() for item in value if str(item).strip())
        for prefix in prefixes:
            env_values[f"{prefix}{suffix}"] = value
    return env_values


def get_prefixed_rule_based_env_value(
    suffix: str,
    *,
    default: Any = None,
    prefixes: Tuple[str, ...] = ("PLANNER_RULE_BASED_", "RAP_RULE_BASED_"),
) -> Any:
    for prefix in prefixes:
        value = os.environ.get(f"{prefix}{suffix}")
        if value is not None:
            return value
    return default
