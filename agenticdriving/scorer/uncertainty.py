"""Frame-level uncertainty signals for the AgenticDriving recovery loop.

Three trajectory-based signals computed once per frame:

1. ``compute_intra_learned_disagreement`` — score-weighted spread of the
   learned planner's proposal distribution. Algorithms for Validation
   (Kochenderfer et al., 2026), section 12.2.1 Eq 12.2 framing (output
   uncertainty over a predicted distribution). The learned planner emits M
   ranked proposals per frame; their score-weighted variance around the
   weighted mean captures how peaked vs. spread the model's preference is.
2. ``compute_cross_family_disagreement`` — nearest-neighbor distance from
   the best learned trajectory to the rule-based candidate set.
   Algorithms for Validation section 12.2.3 Eq 12.4 nonconformity framing.
3. ``compute_mode_count`` — silhouette-selected k from K-Means on the
   learned pool. Detects bimodal disagreement that scalar L2 spread blurs.
   Backend swappable: ``flashlib`` (preferred when available) or
   ``sklearn`` fallback.

``classify_routing_zone`` combines the three scalars into the four-zone
routing decision consumed by ``build_design_change_request``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


ROUTING_ZONE_NORMAL = "normal"
ROUTING_ZONE_LEAN_RULE_BASED = "lean_rule_based"
ROUTING_ZONE_RULE_BASED_FALLBACK = "rule_based_fallback"


def _valid_plan_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    key: str = "local_plan",
) -> Sequence[Dict[str, Any]]:
    """Filter rows that carry a usable (T, 2+) trajectory array."""

    valid = []
    for row in rows:
        plan = row.get(key)
        if plan is None:
            continue
        arr = np.asarray(plan, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 2:
            continue
        valid.append(row)
    return valid


def _trajectories_to_array(
    rows: Sequence[Dict[str, Any]],
    *,
    key: str = "local_plan",
    horizon_steps: Optional[int] = None,
) -> np.ndarray:
    plans = []
    for row in rows:
        plan = row.get(key)
        if plan is None:
            continue
        arr = np.asarray(plan, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 2:
            continue
        plans.append(arr[:, :2])
    if not plans:
        return np.zeros((0, 0, 2), dtype=np.float32)
    horizon = min(p.shape[0] for p in plans)
    if horizon_steps and horizon_steps > 0:
        horizon = min(horizon, int(horizon_steps))
    if horizon == 0:
        return np.zeros((0, 0, 2), dtype=np.float32)
    return np.stack([p[:horizon] for p in plans], axis=0)


def _row_score(row: Dict[str, Any], key: str = "proposal_score") -> Optional[float]:
    value = row.get(key)
    if value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(score):
        return None
    return score


def _softmax_weights(scores: Sequence[float], *, temperature: float = 1.0) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        return arr
    tau = max(float(temperature), 1e-9)
    shifted = (arr - arr.max()) / tau
    exp = np.exp(shifted)
    total = exp.sum()
    if not np.isfinite(total) or total <= 0.0:
        return np.full(arr.shape, 1.0 / arr.size, dtype=np.float64)
    return exp / total


def compute_intra_learned_disagreement(
    learned_rows: Sequence[Dict[str, Any]],
    *,
    horizon_steps: Optional[int] = None,
    score_key: str = "proposal_score",
    score_temperature: float = 1.0,
) -> Dict[str, Any]:
    """Score-weighted spread of the learned planner's proposal distribution.

    The learned planner emits M ranked trajectory proposals per frame, each
    with a ``proposal_score``. The signal computes the score-weighted
    variance of waypoint positions around the score-weighted mean
    trajectory and aggregates as mean-over-time RMS deviation in meters.

    Score weighting matches §12.2.1 framing — the model's output is a
    learned distribution over candidates; we measure how peaked vs.
    spread that distribution is in trajectory space.

    Fallback: when any row is missing a finite ``proposal_score``, all
    candidates are weighted uniformly and the result reduces to the
    unweighted variance-around-mean. ``weighting`` in the returned dict
    indicates which path was taken.
    """

    valid_rows = _valid_plan_rows(learned_rows)
    traj = _trajectories_to_array(valid_rows, horizon_steps=horizon_steps)
    M, T, _ = traj.shape
    if M < 2 or T == 0:
        return {
            "disagreement_m": 0.0,
            "member_count": int(M),
            "horizon": int(T),
            "weighting": "uniform",
        }

    raw_scores = [_row_score(r, key=score_key) for r in valid_rows]
    if any(score is None for score in raw_scores):
        weights = np.full(M, 1.0 / M, dtype=np.float64)
        weighting = "uniform_missing_scores"
    else:
        weights = _softmax_weights(raw_scores, temperature=score_temperature)
        weighting = "softmax"

    mean_traj = np.einsum("m,mtd->td", weights.astype(traj.dtype), traj)
    deviations_sq = np.sum((traj - mean_traj[None, :, :]) ** 2, axis=2)
    variance_per_t = np.einsum("m,mt->t", weights.astype(deviations_sq.dtype), deviations_sq)
    variance_per_t = np.clip(variance_per_t, 0.0, None)
    rms_per_t = np.sqrt(variance_per_t)
    disagreement = float(rms_per_t.mean())

    return {
        "disagreement_m": disagreement,
        "member_count": int(M),
        "horizon": int(T),
        "weighting": weighting,
        "effective_member_count": float(1.0 / float(np.sum(weights ** 2))) if weights.size else 0.0,
    }


def compute_cross_family_disagreement(
    best_learned_row: Dict[str, Any],
    rule_based_rows: Sequence[Dict[str, Any]],
    *,
    horizon_steps: Optional[int] = None,
) -> Dict[str, float]:
    """Nearest-neighbor distance from best learned to rule-based reference set.

    §12.2.3 nonconformity score: an outlier-from-prediction-set measure where
    the rule-based set acts as the prediction set. Min aggregation captures
    the asymmetric trust assumption (one close rule-based neighbor is enough
    to say the learned pick is not an outlier).
    """

    if best_learned_row is None or not rule_based_rows:
        return {
            "disagreement_m": 0.0,
            "rule_based_count": 0,
            "horizon": 0,
        }

    all_rows = [best_learned_row, *rule_based_rows]
    traj = _trajectories_to_array(all_rows, horizon_steps=horizon_steps)
    if traj.shape[0] < 2 or traj.shape[1] == 0:
        return {
            "disagreement_m": 0.0,
            "rule_based_count": int(max(0, traj.shape[0] - 1)),
            "horizon": int(traj.shape[1]),
        }

    learned = traj[0]
    rule_based = traj[1:]
    per_step_distance = np.linalg.norm(rule_based - learned[None, :, :], axis=2)
    per_rule_distance = per_step_distance.mean(axis=1)
    disagreement = float(per_rule_distance.min())

    return {
        "disagreement_m": disagreement,
        "rule_based_count": int(rule_based.shape[0]),
        "horizon": int(traj.shape[1]),
    }


def _kmeans_sklearn(X: np.ndarray, k: int) -> Tuple[np.ndarray, str]:
    from sklearn.cluster import KMeans

    model = KMeans(n_clusters=k, n_init=10, random_state=0)
    labels = model.fit_predict(X)
    return labels.astype(np.int64), "sklearn"


def _kmeans_flashlib(X: np.ndarray, k: int) -> Tuple[np.ndarray, str]:
    import flashlib  # type: ignore[import-not-found]

    kmeans_op = getattr(flashlib, "KMeans", None) or getattr(flashlib, "kmeans", None)
    if kmeans_op is None:
        raise ImportError("flashlib does not expose a KMeans/kmeans entry point")
    result = kmeans_op(X, n_clusters=k)
    if isinstance(result, tuple):
        labels = result[0]
    elif hasattr(result, "labels"):
        labels = result.labels
    elif hasattr(result, "labels_"):
        labels = result.labels_
    else:
        labels = result
    labels_np = np.asarray(labels).astype(np.int64)
    return labels_np, "flashlib"


def _run_kmeans(X: np.ndarray, k: int, backend: str) -> Tuple[np.ndarray, str]:
    if backend == "sklearn":
        return _kmeans_sklearn(X, k)
    if backend == "flashlib":
        return _kmeans_flashlib(X, k)
    try:
        return _kmeans_flashlib(X, k)
    except Exception:
        return _kmeans_sklearn(X, k)


def _silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    if unique.size < 2 or unique.size >= X.shape[0]:
        return 0.0
    from sklearn.metrics import silhouette_score

    try:
        return float(silhouette_score(X, labels, metric="euclidean"))
    except Exception:
        return 0.0


SILHOUETTE_MIN_FOR_MULTIMODAL = 0.25


def compute_mode_count(
    learned_rows: Sequence[Dict[str, Any]],
    *,
    k_max: int = 3,
    horizon_steps: Optional[int] = None,
    backend: str = "auto",
    silhouette_min: float = SILHOUETTE_MIN_FOR_MULTIMODAL,
) -> Dict[str, Any]:
    """Silhouette-selected mode count k in {1, ..., k_max} on learned plans.

    Catches multimodal disagreement that scalar L2 spread blurs. M < 4 -> 1
    (silhouette undefined). A k > 1 is only returned if its silhouette
    score clears ``silhouette_min`` to suppress spurious noise-driven
    clusters. Ties tie-break toward smaller k.
    """

    traj = _trajectories_to_array(learned_rows, horizon_steps=horizon_steps)
    M, T, _ = traj.shape
    if M < 4 or T == 0:
        return {
            "mode_count": 1,
            "silhouette_per_k": {1: 0.0},
            "backend_used": "none",
            "member_count": int(M),
        }

    X = traj.reshape(M, -1)
    silhouette_per_k: Dict[int, float] = {1: 0.0}
    best_k = 1
    best_score = float(silhouette_min)
    backend_used = backend

    for k in range(2, max(2, int(k_max)) + 1):
        if k >= M:
            break
        try:
            labels, used = _run_kmeans(X, k, backend)
        except Exception:
            continue
        backend_used = used
        if np.unique(labels).size < k:
            silhouette_per_k[k] = 0.0
            continue
        score = _silhouette(X, labels)
        silhouette_per_k[k] = score
        if score > best_score + 1e-9:
            best_score = score
            best_k = k

    return {
        "mode_count": int(best_k),
        "silhouette_per_k": silhouette_per_k,
        "backend_used": backend_used,
        "member_count": int(M),
    }


def classify_routing_zone(
    intra_disagreement_m: float,
    cross_disagreement_m: float,
    mode_count: int,
    *,
    t_intra: float,
    t_cross: float,
    mode_count_high: int = 3,
) -> str:
    """Combine the three uncertainty signals into a routing zone (v2, danger-quadrant framing).

    ``t_intra`` and ``t_cross`` are interpreted as **ceilings**: a signal is
    "danger" when it falls *at or below* its threshold. This matches the
    empirical finding on the drivor calibration corpus (~7.8k frames) that
    ``future_unsafe`` is enriched ~2.6x in the low-intra AND low-cross
    quadrant — i.e., when the learned planner is confident and agrees with
    rule-based, yet the scene is about to become unsafe. High uncertainty on
    either axis is near or below base rate, so it no longer triggers routing
    on its own.

    - rule_based_fallback: (intra_danger AND cross_danger) OR mode_count >= mode_count_high
    - lean_rule_based:     exactly one of {intra_danger, cross_danger} OR mode_count == 2
    - normal:              everything else
    """

    intra_danger = float(intra_disagreement_m) <= float(t_intra)
    cross_danger = float(cross_disagreement_m) <= float(t_cross)
    modes = int(mode_count)
    modes_fallback = modes >= int(mode_count_high)
    modes_lean = modes >= 2 and not modes_fallback

    if (intra_danger and cross_danger) or modes_fallback:
        return ROUTING_ZONE_RULE_BASED_FALLBACK
    if intra_danger or cross_danger or modes_lean:
        return ROUTING_ZONE_LEAN_RULE_BASED
    return ROUTING_ZONE_NORMAL
