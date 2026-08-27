"""Summarize resolved forecasts with Brier plots and paired contrasts.

Creates:
- results/<run_id>/resolved_summary.json
- results/<run_id>/<output_name>
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt

from analysis.aggregate import load_forecasts
from analysis.scoring import brier_score, paired_differences

ROOT = Path(__file__).resolve().parent.parent


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    idx = max(0, min(len(sorted_values) - 1, int(round((len(sorted_values) - 1) * q))))
    return sorted_values[idx]


def _bootstrap_ci(values: Sequence[float], n_boot: int = 5000, seed: int = 42) -> Tuple[float, float]:
    import random

    if not values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    means: List[float] = []
    n = len(values)
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(_mean(sample))
    means.sort()
    return _quantile(means, 0.025), _quantile(means, 0.975)


def _sign_test_two_sided(values: Sequence[float]) -> float:
    nonzero = [v for v in values if abs(v) > 1e-12]
    n = len(nonzero)
    if n == 0:
        return 1.0
    k = sum(1 for v in nonzero if v > 0)
    low = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(low + 1)) / (2**n)
    return min(1.0, 2 * tail)


def _attach_outcomes(records, outcomes: Dict[str, int]):
    """Return copies with outcomes filled from id -> {0,1} map."""
    out = []
    for rec in records:
        outcome = rec.outcome
        if outcome is None and rec.question_id in outcomes:
            outcome = outcomes[rec.question_id]
        # Rebuild via dict to avoid mutating source objects in place unexpectedly.
        data = rec.to_dict()
        data["outcome"] = outcome
        from analysis.schemas import ForecastRecord

        out.append(ForecastRecord.from_dict(data))
    return out


def summarize_resolved(
    run_id: str,
    output_name: str = "resolved_summary_plots.png",
    outcomes: Optional[Dict[str, int]] = None,
) -> dict:
    run_dir = ROOT / "runs" / run_id
    results_dir = ROOT / "results" / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    records = load_forecasts(run_dir)
    if outcomes:
        records = _attach_outcomes(records, outcomes)

    scored = [r for r in records if r.status == "ok" and r.outcome is not None]
    if not scored:
        raise ValueError(f"No resolved forecasts with outcomes in run {run_id}")

    by_cond: Dict[str, List[float]] = {}
    for rec in scored:
        by_cond.setdefault(rec.condition, []).append(brier_score(rec.p_yes, int(rec.outcome)))

    primary_rows = paired_differences(scored, "historical_analogy", "plain", metric="brier")
    secondary_rows = paired_differences(
        scored, "historical_analogy", "matched_deliberation", metric="brier"
    )
    d_primary = [r["difference"] for r in primary_rows]
    d_secondary = [r["difference"] for r in secondary_rows]
    p_primary = _sign_test_two_sided(d_primary)
    p_secondary = _sign_test_two_sided(d_secondary)
    ci_primary = _bootstrap_ci(d_primary)
    ci_secondary = _bootstrap_ci(d_secondary)

    summary = {
        "run_id": run_id,
        "n_scored_records": len(scored),
        "n_questions_primary": len(primary_rows),
        "by_condition": {
            cond: {
                "n": len(vals),
                "mean_brier": _mean(vals),
                "median_brier": _quantile(sorted(vals), 0.5),
            }
            for cond, vals in by_cond.items()
        },
        "paired_brier_differences": {
            "historical_minus_plain": {
                "n": len(d_primary),
                "mean": _mean(d_primary) if d_primary else None,
                "ci_low": ci_primary[0] if d_primary else None,
                "ci_high": ci_primary[1] if d_primary else None,
                "p_value_sign_test_two_sided": p_primary,
                "note": "Negative mean favors historical_analogy",
            },
            "historical_minus_matched_deliberation": {
                "n": len(d_secondary),
                "mean": _mean(d_secondary) if d_secondary else None,
                "ci_low": ci_secondary[0] if d_secondary else None,
                "ci_high": ci_secondary[1] if d_secondary else None,
                "p_value_sign_test_two_sided": p_secondary,
                "note": "Negative mean favors historical_analogy",
            },
        },
    }

    order = ["plain", "matched_deliberation", "historical_analogy"]
    labels = [c for c in order if c in by_cond]
    data = [by_cond[c] for c in labels]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    ax0, ax1 = axes
    if data:
        parts = ax0.violinplot(data, showmeans=True, showmedians=True)
        for body in parts["bodies"]:
            body.set_alpha(0.35)
        ax0.set_xticks(range(1, len(labels) + 1), labels, rotation=20)
    ax0.set_title("Brier score distribution by arm")
    ax0.set_ylabel("Brier (lower is better)")
    ax0.set_xlabel("Condition")

    contrasts = [
        ("hist - plain", d_primary, ci_primary, p_primary),
        ("hist - deliberation", d_secondary, ci_secondary, p_secondary),
    ]
    xs = [1, 2]
    means = [_mean(c[1]) if c[1] else float("nan") for c in contrasts]
    lows = [means[i] - contrasts[i][2][0] if contrasts[i][1] else float("nan") for i in range(2)]
    highs = [contrasts[i][2][1] - means[i] if contrasts[i][1] else float("nan") for i in range(2)]
    ax1.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    ax1.errorbar(xs, means, yerr=[lows, highs], fmt="o-", capsize=6, linewidth=2)
    ax1.set_xticks(xs, [c[0] for c in contrasts], rotation=15)
    ax1.set_title("Paired Brier delta with 95% bootstrap CI")
    ax1.set_ylabel("Delta Brier (neg = analogy better)")
    for i, (_, values, _, pval) in enumerate(contrasts):
        y = means[i] if values else 0.0
        ax1.text(xs[i], y + 0.01, f"p={pval:.3g}", ha="center", va="bottom")

    out_png = results_dir / output_name
    fig.savefig(out_png, dpi=140)
    plt.close(fig)

    out_json = results_dir / "resolved_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote summary: {out_json}")
    print(f"Wrote plots:   {out_png}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot resolved-run Brier summary")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-name", default="resolved_summary_plots.png")
    args = parser.parse_args()
    summary = summarize_resolved(args.run_id, args.output_name)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
