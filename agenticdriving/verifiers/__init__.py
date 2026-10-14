"""PDMS trajectory verification for the released pipeline."""
from __future__ import annotations
from typing import Any, Callable, Dict, Optional
from .base import BaseVerifier, VerificationResult
from .ego_progress import EgoProgressReference
from .pdms import PDMSVerifier

def resolve_verifier_threshold(agenticdriving_cfg: Any, verifier_cfg: Optional[Dict[str, Any]] = None) -> float:
    """Reject-below threshold: yaml ``verifier.threshold`` wins over the env var."""
    if verifier_cfg and verifier_cfg.get("threshold") is not None:
        return float(verifier_cfg["threshold"])
    return float(getattr(agenticdriving_cfg, "verifier_threshold", 0.5))


def create_verifier(agenticdriving_cfg: Any, *, plan_dt_sec: float = 0.5,
                    verifier_cfg: Optional[Dict[str, Any]] = None,
                    lane_tier_probe: Optional[Callable] = None,
                    ep_reference_factory: Optional[Callable] = None) -> BaseVerifier:
    verifier_cfg = dict(verifier_cfg or {})
    if verifier_cfg.get("type", "pdms") != "pdms":
        raise ValueError("Release requires verifier.type=pdms")
    import os
    debug = bool(verifier_cfg.get(
        "debug", os.environ.get("AGENTICDRIVING_VERIFIER_DEBUG", "") not in ("", "0", "false")
    ))
    sample_timestep_sec = float(
        verifier_cfg.get("sample_timestep_sec", plan_dt_sec)
    )
    return PDMSVerifier(
        timestep=sample_timestep_sec,
        input_timestep_sec=float(plan_dt_sec),
        probe_horizons_sec=tuple(verifier_cfg.get("probe_horizons_sec", (0.5, 1.0))),
        nc_gate_horizon_sec=verifier_cfg.get("nc_gate_horizon_sec"),
        lane_tier_probe=lane_tier_probe,
        ep_reference_factory=ep_reference_factory,
        dac_mode=str(verifier_cfg.get("dac_mode", "tiered")),
        ttc_debounce_steps=int(verifier_cfg.get("ttc_debounce_steps", 1)),
        vru_ttc_veto=bool(verifier_cfg.get("vru_ttc_veto", False)),
        agent_forecast_mode=str(verifier_cfg.get("agent_forecast_mode", "cv")),
        gate_horizon_stop_conditioned=bool(
            verifier_cfg.get("gate_horizon_stop_conditioned", False)
        ),
        gate_horizon_reaction_sec=float(
            verifier_cfg.get("gate_horizon_reaction_sec", 0.15)
        ),
        ttc_gate_horizon_sec=verifier_cfg.get("ttc_gate_horizon_sec"),
        dac_gate_horizon_sec=verifier_cfg.get("dac_gate_horizon_sec"),
        dac_exit_grace=bool(verifier_cfg.get("dac_exit_grace", False)),
        retime_accel_limit_mps2=verifier_cfg.get("plan_retime_a_max"),
        retime_brake_limit_mps2=verifier_cfg.get("plan_retime_brake_max"),
        min_walker_speed_mps=verifier_cfg.get("min_walker_speed_mps"),
        pedestrian_gate_horizon_sec=verifier_cfg.get("pedestrian_gate_horizon_sec"),
        stationary_vehicle_lateral_margin_m=float(
            verifier_cfg.get("stationary_vehicle_lateral_margin_m", 0.0)
        ),
        ep_on_retimed=bool(verifier_cfg.get("ep_on_retimed", False)),
        lk_deviation_limit_m=verifier_cfg.get("lk_deviation_limit_m", 0.5),
        lk_horizon_window_sec=float(verifier_cfg.get("lk_horizon_window_sec", 2.0)),
        hd_map_endpoint_only=bool(
            verifier_cfg.get("hd_map_endpoint_only", False)
        ),
        debug=debug,
        debug_vru_radius_m=verifier_cfg.get("debug_vru_radius_m"),
        debug_log_path=verifier_cfg.get(
            "debug_log", os.environ.get("AGENTICDRIVING_VERIFIER_DEBUG_LOG") or None
        ),
    )
