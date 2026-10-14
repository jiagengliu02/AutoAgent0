from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Tuple


AGENTICDRIVING_ENV_DEFAULTS: Dict[str, Any] = {
    "ENABLED": False,
    "SELECTOR": "dummy",
    "REDESIGN_CANDIDATE_BUDGET": 10,
    "MAX_REDESIGN_ATTEMPTS": 1,
    "FALLBACK_MODE": "hold",
    "UNCERTAINTY_ENABLED": True,
    "UNCERTAINTY_T_INTRA": 0.20,
    "UNCERTAINTY_T_CROSS": 2.40,
    "UNCERTAINTY_MODE_K_MAX": 3,
    "UNCERTAINTY_MODE_COUNT_HIGH": 3,
    "UNCERTAINTY_HORIZON_STEPS": 0,
    "UNCERTAINTY_KMEANS_BACKEND": "auto",
    "VERIFIER_TYPE": "pdms",
    "VERIFIER_THRESHOLD": 0.5,
    "INSTRUCTION_SCORER_ENABLED": True,
}

AGENTICDRIVING_ENV_FIELD_NAMES: Dict[str, str] = {
    "ENABLED": "enabled",
    "SELECTOR": "selector",
    "REDESIGN_CANDIDATE_BUDGET": "redesign_candidate_budget",
    "MAX_REDESIGN_ATTEMPTS": "max_redesign_attempts",
    "FALLBACK_MODE": "fallback_mode",
    "UNCERTAINTY_ENABLED": "uncertainty_enabled",
    "UNCERTAINTY_T_INTRA": "uncertainty_t_intra",
    "UNCERTAINTY_T_CROSS": "uncertainty_t_cross",
    "UNCERTAINTY_MODE_K_MAX": "uncertainty_mode_k_max",
    "UNCERTAINTY_MODE_COUNT_HIGH": "uncertainty_mode_count_high",
    "UNCERTAINTY_HORIZON_STEPS": "uncertainty_horizon_steps",
    "UNCERTAINTY_KMEANS_BACKEND": "uncertainty_kmeans_backend",
    "VERIFIER_TYPE": "verifier_type",
    "VERIFIER_THRESHOLD": "verifier_threshold",
    "INSTRUCTION_SCORER_ENABLED": "instruction_scorer_enabled",
}


@dataclass(frozen=True)
class AgenticDrivingConfig:
    enabled: bool = False
    selector: str = "dummy"
    redesign_candidate_budget: int = 10
    max_redesign_attempts: int = 1
    fallback_mode: str = "hold"
    uncertainty_enabled: bool = True
    uncertainty_t_intra: float = 0.20
    uncertainty_t_cross: float = 2.40
    uncertainty_mode_k_max: int = 3
    uncertainty_mode_count_high: int = 3
    uncertainty_horizon_steps: int = 0
    uncertainty_kmeans_backend: str = "auto"
    verifier_type: str = "pdms"
    verifier_threshold: float = 0.5
    instruction_scorer_enabled: bool = True


def _coerce_env_value(raw_value: Any, default_value: Any) -> Any:
    if isinstance(default_value, bool):
        return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default_value, int) and not isinstance(default_value, bool):
        return int(raw_value)
    if isinstance(default_value, float):
        return float(raw_value)
    return str(raw_value)




def get_prefixed_agenticdriving_env_value(
    suffix: str,
    *,
    default: Any = None,
    prefixes: Tuple[str, ...] = ("AGENTICDRIVING_",),
) -> Any:
    for prefix in prefixes:
        value = os.environ.get(f"{prefix}{suffix}")
        if value is not None:
            return value
    return default


def resolve_agenticdriving_config(
    *,
    prefixes: Tuple[str, ...] = ("AGENTICDRIVING_",),
) -> AgenticDrivingConfig:
    values: Dict[str, Any] = {}
    for suffix, field_name in AGENTICDRIVING_ENV_FIELD_NAMES.items():
        default_value = AGENTICDRIVING_ENV_DEFAULTS[suffix]
        raw_value = get_prefixed_agenticdriving_env_value(
            suffix,
            default=default_value,
            prefixes=prefixes,
        )
        values[field_name] = _coerce_env_value(raw_value, default_value)
    values["redesign_candidate_budget"] = max(1, int(values["redesign_candidate_budget"]))
    values["max_redesign_attempts"] = max(0, int(values["max_redesign_attempts"]))
    values["uncertainty_mode_k_max"] = max(1, int(values["uncertainty_mode_k_max"]))
    values["uncertainty_mode_count_high"] = max(2, int(values["uncertainty_mode_count_high"]))
    values["uncertainty_horizon_steps"] = max(0, int(values["uncertainty_horizon_steps"]))
    values["uncertainty_t_intra"] = float(values["uncertainty_t_intra"])
    values["uncertainty_t_cross"] = float(values["uncertainty_t_cross"])
    values["verifier_type"] = str(values["verifier_type"]).strip().lower()
    values["verifier_threshold"] = float(values["verifier_threshold"])
    values["selector"] = str(values["selector"]).strip().lower()
    if values["selector"] == "random":
        values["selector"] = "dummy"
    if values["selector"] not in {"dummy", "vlm"}:
        raise ValueError(
            f"Unsupported AgenticDriving selector={values['selector']!r}; "
            "expected 'dummy' or 'vlm'"
        )
    return AgenticDrivingConfig(**values)
