from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import numpy as np


PlanConverter = Callable[[np.ndarray], np.ndarray]
DefaultTrajectoryProvider = Callable[[int], Sequence[np.ndarray]]


def build_carla_plan_payload(
    *,
    proposals: np.ndarray,
    scores: np.ndarray,
    output_num_poses: int,
    plan_converter: PlanConverter,
    selected_idx: Optional[int] = None,
    selected_source: str,
    selection_debug: Optional[Dict[str, object]] = None,
    selected_plan_override: Optional[np.ndarray] = None,
    selected_score_override: Optional[float] = None,
    candidate_pool_rows: Optional[Sequence[Dict[str, object]]] = None,
    topk: int,
    default_source_name: str,
    default_trajectory_provider: Optional[DefaultTrajectoryProvider] = None,
) -> Dict[str, object]:
    """Build the CARLA plan payload shared by learned planner adapters."""

    topk = max(1, min(int(topk), int(len(scores))))
    top_indices = np.argsort(scores)[-topk:][::-1]
    if selected_idx is None and selected_plan_override is None:
        selected_idx = int(top_indices[0])
    else:
        selected_idx = None if selected_idx is None else int(selected_idx)

    if selected_plan_override is not None:
        selected_plan = np.asarray(selected_plan_override, dtype=np.float32)
        selected_score = float(selected_score_override) if selected_score_override is not None else None
    else:
        assert selected_idx is not None
        selected_traj = proposals[selected_idx, :output_num_poses]
        selected_plan = plan_converter(selected_traj)
        selected_score = float(scores[selected_idx])

    if candidate_pool_rows is not None:
        candidate_pool_plans = [
            np.asarray(row["local_plan"], dtype=np.float32).tolist()
            for row in candidate_pool_rows
        ]
        candidate_pool_execution_plans = [
            np.asarray(row.get("execution_plan", row["local_plan"]), dtype=np.float32).tolist()
            for row in candidate_pool_rows
        ]
        candidate_pool_scores = [float(row.get("proposal_score", 0.0)) for row in candidate_pool_rows]
        candidate_pool_q_scores = [
            None if row.get("q_score") is None else float(row["q_score"])
            for row in candidate_pool_rows
        ]
        candidate_pool_sources = [str(row.get("source", default_source_name)) for row in candidate_pool_rows]
        candidate_pool_proposal_indices = [
            None if row.get("proposal_index") is None else int(row["proposal_index"])
            for row in candidate_pool_rows
        ]
    else:
        candidate_pool_plans = [
            plan_converter(proposals[idx, :output_num_poses]).tolist()
            for idx in top_indices
        ]
        candidate_pool_scores = [float(scores[idx]) for idx in top_indices]
        candidate_pool_execution_plans = list(candidate_pool_plans)
        candidate_pool_q_scores = [None for _ in top_indices]
        candidate_pool_sources = [default_source_name for _ in top_indices]
        candidate_pool_proposal_indices = [int(idx) for idx in top_indices]

    default_overlay_plans = None
    default_overlay_sources = None
    if (
        default_trajectory_provider is not None
        and bool(selection_debug and selection_debug.get("display_default_trajectories"))
    ):
        default_overlay_plans = [
            np.asarray(traj, dtype=np.float32).tolist()
            for traj in default_trajectory_provider(output_num_poses)
        ]
        default_overlay_sources = [f"default_fallback_{idx}" for idx in range(len(default_overlay_plans))]

    payload: Dict[str, object] = {
        "selected_idx": selected_idx,
        "selected_score": selected_score,
        "selected_source": selected_source,
        "selected_plan": selected_plan,
        "topk_indices": [int(idx) for idx in top_indices],
        "topk_scores": [float(scores[idx]) for idx in top_indices],
        "topk_plans": [
            plan_converter(proposals[idx, :output_num_poses]).tolist()
            for idx in top_indices
        ],
        "candidate_pool_plans": candidate_pool_plans,
        "candidate_pool_execution_plans": candidate_pool_execution_plans,
        "candidate_pool_scores": candidate_pool_scores,
        "candidate_pool_q_scores": candidate_pool_q_scores,
        "candidate_pool_sources": candidate_pool_sources,
        "candidate_pool_proposal_indices": candidate_pool_proposal_indices,
        "default_overlay_plans": default_overlay_plans,
        "default_overlay_sources": default_overlay_sources,
    }
    if selection_debug:
        payload.update(selection_debug)
    return payload
