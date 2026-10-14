"""Round-one candidate scoring for the per-frame orchestrator.

The orchestrator owns control flow; this module owns the independent End2End
and all-region PDMS proposal branches and returns one aligned score vector.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np

from agenticdriving.scorer.regions import select_random_per_layered_region
from agenticdriving.scorer.scoring import BaseScorer
from agenticdriving.adapters.carla.geometry import normalize_scores
from agenticdriving.orchestration.directional_options import (
    junction_route_heading_angles_deg,
    junction_route_heading_violation_mask,
    leftward_proposal_mask,
)


@dataclass(frozen=True)
class CandidateScoringConfig:
    region_sector_counts: Sequence[int]
    end2end_topk: int
    region_samples_per_region: int
    region_random_seed: int
    region_min_bearing_deg: float
    region_max_bearing_deg: float
    end2end_pdms_branch: bool
    target_region_pdms_branch: bool
    # PDMS-as-gate selection (default off = shipped argmax behaviour): PDMS
    # only filters candidates below the gate threshold; among survivors the
    # planner's native ranking picks the winner. Restores the standalone
    # planner's choice whenever it is PDMS-acceptable, so soft-term pricing
    # (TTC 5/12 vs LK 2/12) cannot select lane-crossing dodges the planner
    # ranked deep. When no candidate passes the gate, behaviour is unchanged
    # (PDMS-argmax winners stand and the sub-threshold score triggers the
    # verifier's recovery path as before).
    pdms_gate_native_argmax: bool = False
    pdms_gate_threshold: float = 0.0


@dataclass(frozen=True)
class CandidateScoringResult:
    scores: np.ndarray
    debug: Dict[str, Any]


class CandidateScoringPipeline:
    """Score planner proposals while preserving their original index space."""

    def __init__(
        self,
        *,
        end2end_scorer: Optional[BaseScorer],
        instruction_scorer: Optional[BaseScorer],
        rule_based_scorer: Optional[BaseScorer],
        config: CandidateScoringConfig,
        logger: logging.Logger,
    ) -> None:
        self.end2end_scorer = end2end_scorer
        self.instruction_scorer = instruction_scorer
        self.rule_based_scorer = rule_based_scorer
        self.config = config
        self.log = logger
        self._region_rng = np.random.default_rng(
            int(config.region_random_seed)
        )

    def score(
        self,
        *,
        proposals: np.ndarray,
        native_scores: np.ndarray,
        obs: Dict[str, Any],
        info: Dict[str, Any],
        privileged_info: Any,
    ) -> CandidateScoringResult:
        """Return the selector score vector and structured scoring diagnostics."""
        cfg = self.config
        debug: Dict[str, Any] = {}
        # Strict A/B path: execute exactly the planner's native argmax.  This
        # early return intentionally bypasses PDMS, instruction masking, and
        # downstream route-direction filters so the comparison isolates the
        # candidate-selection policy itself.
        if os.environ.get("BRIDGEDRIVE_NATIVE_ONLY", "0") == "1":
            n = len(np.asarray(proposals))
            scores = np.full(n, -np.inf, dtype=np.float32)
            if n:
                native = np.asarray(native_scores, dtype=np.float32).reshape(-1)
                scores[0] = float(native[0]) if len(native) else 0.0
            debug.update(
                {
                    "interaction": "bridgedrive_native_only",
                    "native_only": True,
                    "native_winner_index": 0 if n else None,
                    "combined_scores": scores.tolist(),
                }
            )
            return CandidateScoringResult(scores=scores, debug=debug)
        score_info = dict(info)
        score_info["privileged_agents"] = privileged_info

        e2e_scores = None
        if self.end2end_scorer is not None:
            e2e_scores = np.asarray(
                self.end2end_scorer.score(
                    proposals,
                    native_scores=native_scores,
                    info=score_info,
                    obs=obs,
                ),
                dtype=np.float32,
            )
            debug["end2end_scorer"] = self.end2end_scorer.name

        instruction_scores = None
        if self.instruction_scorer is not None:
            instruction_scores = np.asarray(
                self.instruction_scorer.score(
                    proposals,
                    native_scores=native_scores,
                    info=score_info,
                    obs=obs,
                ),
                dtype=np.float32,
            )
            debug["instruction_scores"] = instruction_scores.tolist()
            debug["instruction_scorer"] = self.instruction_scorer.name

        pdms_branch_active = self.rule_based_scorer is not None and (
            cfg.end2end_pdms_branch or cfg.target_region_pdms_branch
        )
        if pdms_branch_active:
            combined_scores = self._score_pdms_branches(
                proposals=proposals,
                native_scores=native_scores,
                e2e_scores=e2e_scores,
                instruction_scores=instruction_scores,
                obs=obs,
                info=info,
                score_info=score_info,
                debug=debug,
            )
        elif self.rule_based_scorer is not None and e2e_scores is None:
            rule_scores = np.asarray(
                self.rule_based_scorer.score(
                    proposals,
                    native_scores=native_scores,
                    info=score_info,
                    obs=obs,
                ),
                dtype=np.float32,
            )
            combined_scores = rule_scores
            if instruction_scores is not None:
                allowed = instruction_scores > 0.0
                if np.any(allowed):
                    combined_scores = np.where(allowed, rule_scores, -np.inf)
            debug["interaction"] = "rule_based_only"
            debug["rule_based_scores"] = rule_scores.tolist()
            debug["rule_based_scorer"] = self.rule_based_scorer.name
        else:
            assert e2e_scores is not None
            base_ranking = normalize_scores(e2e_scores) + 1e-3
            combined_scores = e2e_scores
            gated = (
                base_ranking * instruction_scores
                if instruction_scores is not None
                else None
            )
            if gated is not None:
                if float(np.max(gated)) > 0.0:
                    combined_scores = gated
                else:
                    self.log.warning(
                        "%s masked all candidates; falling back to ungated ranking",
                        self.instruction_scorer.name,
                    )
            debug["interaction"] = "end2end_only"

        # RouteScenario 11 policy: when the route-lane HD map says the
        # immediate left lane is opposing and the immediate right lane is
        # same-flow, do not let a left-going Loop-1 proposal survive into
        # selection.  Keep proposal indices stable for planner/debug payloads.
        blocked_left = leftward_proposal_mask(proposals, info)
        if np.any(blocked_left):
            combined_scores = np.asarray(combined_scores, dtype=np.float32).copy()
            combined_scores[blocked_left] = -np.inf
            allowed = ~blocked_left
            if np.any(allowed) and not np.any(np.isfinite(combined_scores[allowed])):
                allowed_indices = np.flatnonzero(allowed)
                allowed_native = np.asarray(native_scores)[allowed_indices]
                finite_native = np.isfinite(allowed_native)
                local_idx = (
                    int(np.argmax(np.where(finite_native, allowed_native, -np.inf)))
                    if np.any(finite_native)
                    else 0
                )
                combined_scores[int(allowed_indices[local_idx])] = 0.0
            debug["hd_map_leftward_filter"] = {
                "active": True,
                "blocked_indices": np.flatnonzero(blocked_left).tolist(),
                "reason": "left_opposing_right_same_direction",
            }

        # At or immediately before an HD-map junction, a proposal whose net
        # heading points more than the configured angle away from the route
        # target can only take the wrong branch or turn back.  Mask it before
        # argmax; the orchestrator already substitutes a stationary plan and
        # enters recovery when no finite Loop-1 proposal remains.
        blocked_junction_heading = junction_route_heading_violation_mask(
            proposals, info
        )
        if np.any(blocked_junction_heading):
            combined_scores = np.asarray(
                combined_scores, dtype=np.float32
            ).copy()
            combined_scores[blocked_junction_heading] = -np.inf
            context = info.get("junction_route_heading_filter") or {}
            proposal_angles = [
                junction_route_heading_angles_deg(plan, info)
                for plan in proposals
            ]
            debug["junction_route_heading_filter"] = {
                "active": True,
                "blocked_indices": np.flatnonzero(
                    blocked_junction_heading
                ).tolist(),
                "angles_deg": [
                    max(
                        (
                            angle
                            for angle in angles.values()
                            if angle is not None
                        ),
                        default=None,
                    )
                    for angles in proposal_angles
                ],
                "full_angles_deg": [
                    angles.get("full") for angles in proposal_angles
                ],
                "tail_angles_deg": [
                    angles.get("tail") for angles in proposal_angles
                ],
                "max_angle_deg": float(context.get("max_angle_deg", 60.0)),
                "reason": "trajectory_route_angle_exceeds_limit",
            }

        finite_combined = np.asarray(combined_scores, dtype=np.float32)
        if not np.all(np.isfinite(finite_combined)):
            finite_combined = np.where(
                np.isfinite(finite_combined), finite_combined, 0.0
            )
        debug["combined_scores"] = finite_combined.tolist()
        return CandidateScoringResult(
            scores=np.asarray(combined_scores, dtype=np.float32),
            debug=debug,
        )

    def _score_pdms_branches(
        self,
        *,
        proposals: np.ndarray,
        native_scores: np.ndarray,
        e2e_scores: Optional[np.ndarray],
        instruction_scores: Optional[np.ndarray],
        obs: Dict[str, Any],
        info: Dict[str, Any],
        score_info: Dict[str, Any],
        debug: Dict[str, Any],
    ) -> np.ndarray:
        cfg = self.config
        assert self.rule_based_scorer is not None

        end2end_indices = np.zeros(0, dtype=np.int64)
        if cfg.end2end_pdms_branch:
            assert e2e_scores is not None
            finite_e2e = np.flatnonzero(np.isfinite(e2e_scores))
            if len(finite_e2e):
                end2end_order = np.argsort(-e2e_scores[finite_e2e], kind="stable")
                end2end_indices = finite_e2e[
                    end2end_order[: cfg.end2end_topk]
                ].astype(np.int64)
            elif len(proposals):
                end2end_indices = np.asarray([0], dtype=np.int64)

        regional_selection = None
        regional_candidate_count = 0
        rule_based_indices = np.zeros(0, dtype=np.int64)
        if cfg.target_region_pdms_branch:
            regional_selection = select_random_per_layered_region(
                proposals,
                samples_per_region=cfg.region_samples_per_region,
                rng=self._region_rng,
                sectors_per_layer=cfg.region_sector_counts,
                min_bearing_deg=cfg.region_min_bearing_deg,
                max_bearing_deg=cfg.region_max_bearing_deg,
            )
            regional_candidate_count = sum(
                len(indices) for indices in regional_selection.per_region_indices
            )
            rule_based_indices = regional_selection.indices

        # Both branches use the same stateless PDMS scorer. Evaluate their
        # exact union once so overlapping proposals and the HD-map probe are
        # never repeated between End2End Top-K and regional sampling.
        evaluated_indices = np.unique(
            np.concatenate((end2end_indices, rule_based_indices))
        ).astype(np.int64)
        evaluated_scores = np.asarray(
            self.rule_based_scorer.score(
                proposals[evaluated_indices],
                native_scores=native_scores[evaluated_indices],
                info=score_info,
                obs=obs,
            ),
            dtype=np.float32,
        )
        self._validate_score_count(
            evaluated_scores, evaluated_indices, "combined"
        )
        score_by_index = {
            int(index): float(score)
            for index, score in zip(evaluated_indices, evaluated_scores)
        }
        end2end_pdms_scores = np.asarray(
            [score_by_index[int(index)] for index in end2end_indices],
            dtype=np.float32,
        )
        rule_based_pdms_scores = np.asarray(
            [score_by_index[int(index)] for index in rule_based_indices],
            dtype=np.float32,
        )

        end2end_winner_idx, end2end_winner_pdms = self._branch_winner(
            end2end_indices,
            end2end_pdms_scores,
            instruction_scores,
            "End2End",
        )
        rule_based_winner_idx, rule_based_winner_pdms = self._branch_winner(
            rule_based_indices,
            rule_based_pdms_scores,
            instruction_scores,
            "rule-based",
        )

        selection_scores = np.full(len(proposals), -np.inf, dtype=np.float32)
        if end2end_winner_idx is not None:
            selection_scores[end2end_winner_idx] = end2end_winner_pdms
        if rule_based_winner_idx is not None:
            selection_scores[rule_based_winner_idx] = max(
                selection_scores[rule_based_winner_idx],
                float(rule_based_winner_pdms),
            )
        if cfg.pdms_gate_native_argmax and len(evaluated_indices):
            gate_eligible = evaluated_indices[
                evaluated_scores >= float(cfg.pdms_gate_threshold)
            ]
            if instruction_scores is not None and len(gate_eligible):
                compliant = instruction_scores[gate_eligible] > 0.0
                if np.any(compliant):
                    gate_eligible = gate_eligible[compliant]
            if len(gate_eligible):
                gate_native = np.asarray(native_scores)[gate_eligible]
                finite_native = np.isfinite(gate_native)
                if np.any(finite_native):
                    gate_local = int(
                        np.argmax(np.where(finite_native, gate_native, -np.inf))
                    )
                    gate_winner = int(gate_eligible[gate_local])
                    selection_scores = np.full(
                        len(proposals), -np.inf, dtype=np.float32
                    )
                    selection_scores[gate_winner] = score_by_index[gate_winner]
                    self.log.info(
                        "PDMS gate-native selection: %d/%d candidates passed "
                        "gate %.3f; native argmax -> index %d (pdms %.3f, "
                        "pdms argmax would pick %s)",
                        len(gate_eligible),
                        len(evaluated_indices),
                        float(cfg.pdms_gate_threshold),
                        gate_winner,
                        score_by_index[gate_winner],
                        end2end_winner_idx,
                    )
                    debug["pdms_gate_native_argmax"] = {
                        "gate_threshold": float(cfg.pdms_gate_threshold),
                        "passed": len(gate_eligible),
                        "evaluated": len(evaluated_indices),
                        "winner": gate_winner,
                        "winner_pdms": score_by_index[gate_winner],
                        "pdms_argmax_winner": end2end_winner_idx,
                    }
        if not np.any(np.isfinite(selection_scores)) and len(proposals):
            fallback_pool = np.arange(len(proposals), dtype=np.int64)
            if instruction_scores is not None:
                compliant = instruction_scores > 0.0
                if np.any(compliant):
                    fallback_pool = np.flatnonzero(compliant)
            fallback_scores = native_scores[fallback_pool]
            finite = np.isfinite(fallback_scores)
            fallback_local = (
                int(np.argmax(np.where(finite, fallback_scores, -np.inf)))
                if np.any(finite)
                else 0
            )
            selection_scores[int(fallback_pool[fallback_local])] = 0.0

        self.log.info(
            "PDMS proposal batch: unique=%d end2end=%d (winner=%s score=%s) "
            "rule_based=%d (sampled_from=%d regions=%d per_region=%d "
            "winner=%s score=%s)",
            len(evaluated_indices),
            len(end2end_indices),
            end2end_winner_idx,
            end2end_winner_pdms,
            len(rule_based_indices),
            regional_candidate_count,
            sum(cfg.region_sector_counts),
            cfg.region_samples_per_region,
            rule_based_winner_idx,
            rule_based_winner_pdms,
        )

        if cfg.end2end_pdms_branch and cfg.target_region_pdms_branch:
            interaction = "parallel_end2end_topk_and_all_regions_pdms_argmax"
        elif cfg.end2end_pdms_branch:
            interaction = "end2end_topk_pdms_argmax"
        else:
            interaction = "all_regions_pdms_argmax"
        debug.update(
            {
                "interaction": interaction,
                "end2end_pdms_branch_enabled": cfg.end2end_pdms_branch,
                "target_region_pdms_branch_enabled": cfg.target_region_pdms_branch,
                "all_regions_pdms_branch_enabled": cfg.target_region_pdms_branch,
                "end2end_topk": cfg.end2end_topk,
                "end2end_candidate_indices": end2end_indices.tolist(),
                "end2end_candidate_native_scores": (
                    [] if e2e_scores is None else e2e_scores[end2end_indices].tolist()
                ),
                "end2end_pdms_scores": end2end_pdms_scores.tolist(),
                "end2end_pdms_traj_count": int(len(end2end_indices)),
                "end2end_proposal_index": end2end_winner_idx,
                "end2end_proposal_pdms": end2end_winner_pdms,
                "rule_based_candidate_indices": rule_based_indices.tolist(),
                "rule_based_candidate_count_before_sampling": (
                    regional_candidate_count
                ),
                "rule_based_candidate_random_sample_limit": (
                    cfg.region_samples_per_region
                ),
                "region_sampling_mode": "random_per_region",
                "region_samples_per_region": cfg.region_samples_per_region,
                "rule_based_pdms_scores": rule_based_pdms_scores.tolist(),
                "rule_based_pdms_traj_count": int(len(rule_based_indices)),
                "rule_based_proposal_index": rule_based_winner_idx,
                "rule_based_proposal_pdms": rule_based_winner_pdms,
                "pdms_traj_counts": {
                    "end2end": int(len(end2end_indices)),
                    "rule_based": int(len(rule_based_indices)),
                },
                "pdms_unique_traj_count": int(len(evaluated_indices)),
                "shortlist_indices": evaluated_indices.tolist(),
            }
        )
        endpoint_tiers = getattr(
            self.rule_based_scorer, "last_endpoint_tiers", None
        )
        if endpoint_tiers is not None:
            endpoint_tiers = np.asarray(endpoint_tiers, dtype=np.float64)
            if len(endpoint_tiers) == len(evaluated_indices):
                debug["shortlist_endpoint_tiers"] = endpoint_tiers.tolist()
        if regional_selection is not None:
            debug.update(
                {
                    "num_regions": sum(cfg.region_sector_counts),
                    "region_radii_m": regional_selection.radial_bounds_m.tolist(),
                    "region_radii_mode": "dynamic_equal_width",
                    "region_sector_counts": list(cfg.region_sector_counts),
                    "region_min_bearing_deg": cfg.region_min_bearing_deg,
                    "region_max_bearing_deg": cfg.region_max_bearing_deg,
                    "candidate_region_ids": (
                        regional_selection.candidate_region_ids.tolist()
                    ),
                    "candidate_layer_ids": (
                        regional_selection.candidate_layer_ids.tolist()
                    ),
                    "endpoint_bearings_deg": (
                        regional_selection.endpoint_bearings_deg.tolist()
                    ),
                    "endpoint_radii_m": (
                        regional_selection.endpoint_radii_m.tolist()
                    ),
                    "per_region_indices": [
                        indices.tolist()
                        for indices in regional_selection.per_region_indices
                    ],
                    "sampled_per_region_indices": [
                        indices.tolist()
                        for indices in regional_selection.sampled_per_region_indices
                    ],
                    "sampled_per_region_counts": [
                        int(len(indices))
                        for indices in regional_selection.sampled_per_region_indices
                    ],
                    # Explicit null keeps older telemetry readers compatible.
                    "target_point_region_id": None,
                }
            )
        debug["rule_based_scorer"] = self.rule_based_scorer.name
        return selection_scores

    def _branch_winner(
        self,
        indices: np.ndarray,
        pdms_scores: np.ndarray,
        instruction_scores: Optional[np.ndarray],
        branch_name: str,
    ) -> tuple[Optional[int], Optional[float]]:
        if len(indices) == 0:
            return None, None
        eligible = np.arange(len(indices), dtype=np.int64)
        if instruction_scores is not None:
            compliant = instruction_scores[indices] > 0.0
            if np.any(compliant):
                eligible = np.flatnonzero(compliant)
            else:
                self.log.warning(
                    "%s masked all %s proposals; keeping PDMS argmax",
                    self.instruction_scorer.name,
                    branch_name,
                )
        eligible_scores = pdms_scores[eligible]
        finite = np.isfinite(eligible_scores)
        if not np.any(finite):
            return None, None
        best_local = int(
            eligible[np.argmax(np.where(finite, eligible_scores, -np.inf))]
        )
        return int(indices[best_local]), float(pdms_scores[best_local])

    @staticmethod
    def _validate_score_count(
        pdms_scores: np.ndarray,
        indices: np.ndarray,
        branch_name: str,
    ) -> None:
        if len(pdms_scores) != len(indices):
            raise ValueError(
                "PDMS scorer returned %d scores for %d %s proposals"
                % (len(pdms_scores), len(indices), branch_name)
            )
