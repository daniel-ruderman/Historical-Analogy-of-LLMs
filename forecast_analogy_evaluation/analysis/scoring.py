"""Scoring functions for forecast evaluation.

Implements the metrics defined in analysis_spec.md.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from analysis.schemas import CLIP_BOUNDS, ForecastRecord


def clip_probability(p: float, bounds: Tuple[float, float] = CLIP_BOUNDS) -> float:
    lo, hi = bounds
    return max(lo, min(hi, p))


def brier_score(p: float, outcome: int, bounds: Tuple[float, float] = CLIP_BOUNDS) -> float:
    p_c = clip_probability(p, bounds)
    y = float(outcome)
    return (p_c - y) ** 2


def log_loss(p: float, outcome: int, bounds: Tuple[float, float] = CLIP_BOUNDS) -> float:
    p_c = clip_probability(p, bounds)
    y = float(outcome)
    return -(y * math.log(p_c) + (1 - y) * math.log(1 - p_c))


def paired_differences(
    forecasts: Sequence[ForecastRecord],
    condition_a: str,
    condition_b: str,
    metric: str = "brier",
) -> List[Dict]:
    """Compute per-question paired metric differences: A − B.

    Negative mean difference means condition_a is better (lower loss).
    """
    by_q: Dict[str, Dict[str, ForecastRecord]] = {}
    for f in forecasts:
        if f.status != "ok" or f.outcome is None:
            continue
        by_q.setdefault(f.question_id, {})[f.condition] = f

    rows = []
    for qid, cond_map in by_q.items():
        fa = cond_map.get(condition_a)
        fb = cond_map.get(condition_b)
        if fa is None or fb is None:
            continue
        if metric == "brier":
            ma = fa.brier()
            mb = fb.brier()
        elif metric == "log_loss":
            ma = fa.log_loss()
            mb = fb.log_loss()
        else:
            raise ValueError(f"Unknown metric: {metric}")
        if ma is None or mb is None:
            continue
        sample = fa  # for metadata
        rows.append({
            "question_id": qid,
            "condition_a": condition_a,
            "condition_b": condition_b,
            "metric": metric,
            "value_a": ma,
            "value_b": mb,
            "difference": ma - mb,
            "round_id": sample.round_id,
            "cluster_id": sample.cluster_id,
            "outcome": sample.outcome,
        })
    return rows


def cluster_bootstrap_ci(
    paired_rows: Sequence[Dict],
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict:
    """Cluster-bootstrap CI for mean paired difference.

    Clusters on (round_id, cluster_id). Returns mean, CI, and one-sided p-value.
    """
    import random

    if not paired_rows:
        return {"mean": None, "ci_low": None, "ci_high": None, "p_value": None, "n": 0}

    rng = random.Random(seed)
    clusters: Dict[str, List[Dict]] = {}
    for row in paired_rows:
        key = f"{row.get('round_id', '')}:{row.get('cluster_id', '')}"
        clusters.setdefault(key, []).append(row)

    cluster_keys = list(clusters.keys())
    n_clusters = len(cluster_keys)
    diffs = [r["difference"] for r in paired_rows]
    observed_mean = sum(diffs) / len(diffs)

    boot_means: List[float] = []
    for _ in range(n_bootstrap):
        sampled_keys = [rng.choice(cluster_keys) for _ in range(n_clusters)]
        boot_diffs = []
        for k in sampled_keys:
            boot_diffs.extend(r["difference"] for r in clusters[k])
        if boot_diffs:
            boot_means.append(sum(boot_diffs) / len(boot_diffs))

    boot_means.sort()
    alpha = 1 - ci
    lo_idx = int(alpha / 2 * len(boot_means))
    hi_idx = int((1 - alpha / 2) * len(boot_means))
    p_value = sum(1 for m in boot_means if m >= 0) / len(boot_means)

    return {
        "mean": observed_mean,
        "ci_low": boot_means[lo_idx],
        "ci_high": boot_means[min(hi_idx, len(boot_means) - 1)],
        "p_value": p_value,
        "n": len(paired_rows),
        "n_clusters": n_clusters,
    }


def coverage_by_condition(forecasts: Sequence[ForecastRecord]) -> Dict[str, float]:
    """Fraction of questions with status=ok per condition."""
    totals: Dict[str, int] = {}
    ok_counts: Dict[str, int] = {}
    for f in forecasts:
        totals[f.condition] = totals.get(f.condition, 0) + 1
        if f.status == "ok":
            ok_counts[f.condition] = ok_counts.get(f.condition, 0) + 1
    return {
        cond: ok_counts.get(cond, 0) / totals[cond]
        for cond in totals
    }
