"""Aggregate analysis runner.

Reads forecasts from a run directory, computes paired effects, and writes
results/ files as specified in analysis_spec.md.

Usage:
    python -m analysis.aggregate --run-id pilot_001 --config configs/retrospective_pilot.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

from analysis.schemas import ForecastRecord
from analysis.scoring import cluster_bootstrap_ci, coverage_by_condition, paired_differences

ROOT = Path(__file__).resolve().parent.parent


def load_forecasts(run_dir: Path) -> List[ForecastRecord]:
    path = run_dir / "forecasts.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No forecasts at {path}")
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(ForecastRecord.from_dict(json.loads(line)))
    return records


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_paired_effects(rows: List[Dict], out_path: Path) -> None:
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_aggregate(run_id: str, config_path: Path) -> Dict[str, Any]:
    run_dir = ROOT / "runs" / run_id
    results_dir = ROOT / "results" / run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    analysis_cfg = config.get("analysis", {})
    forecasts = load_forecasts(run_dir)

    primary = analysis_cfg.get("primary_contrast", ["historical_analogy", "plain"])
    secondary = analysis_cfg.get("secondary_contrast", ["historical_analogy", "matched_deliberation"])
    n_boot = analysis_cfg.get("bootstrap_samples", 10_000)

    all_paired: List[Dict] = []
    summary: Dict[str, Any] = {"run_id": run_id, "contrasts": {}}

    for label, (cond_a, cond_b) in [("primary_practical", primary), ("secondary_mechanistic", secondary)]:
        for metric in ("brier", "log_loss"):
            rows = paired_differences(forecasts, cond_a, cond_b, metric=metric)
            all_paired.extend(rows)
            key = f"{label}_{metric}"
            summary["contrasts"][key] = cluster_bootstrap_ci(rows, n_bootstrap=n_boot)
            summary["contrasts"][key]["condition_a"] = cond_a
            summary["contrasts"][key]["condition_b"] = cond_b
            summary["contrasts"][key]["analysis_type"] = (
                "confirmatory" if label == "primary_practical" and metric == "brier" else
                "confirmatory_secondary" if label == "secondary_mechanistic" and metric == "brier" else
                "exploratory"
            )

    summary["coverage"] = coverage_by_condition(forecasts)
    summary["n_forecasts"] = len(forecasts)

    write_paired_effects(all_paired, results_dir / "paired_effects.csv")
    with (results_dir / "aggregate_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate forecast evaluation results")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", default="configs/retrospective_pilot.yaml")
    args = parser.parse_args(argv)

    config_path = ROOT / args.config
    summary = run_aggregate(args.run_id, config_path)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
