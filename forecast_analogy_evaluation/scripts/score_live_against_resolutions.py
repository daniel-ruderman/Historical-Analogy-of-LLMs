#!/usr/bin/env python3
"""Score live forecasts against a published resolution set and plot Brier results."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.resolved_summary_plots import summarize_resolved  # noqa: E402
from runner.forecastbench import BASE_URL, _is_binary_resolved  # noqa: E402


def load_binary_outcomes(resolution_date: str) -> Dict[str, int]:
    raw_dir = ROOT / "data" / "raw" / "forecastbench" / "resolution_sets"
    path = raw_dir / f"{resolution_date}_resolution_set.json"
    if not path.exists() or path.stat().st_size == 0:
        url = f"{BASE_URL}/datasets/resolution_sets/{resolution_date}_resolution_set.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, path)  # noqa: S310
    data = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, int] = {}
    for item in data.get("resolutions", []):
        qid = item.get("id")
        if isinstance(qid, str) and _is_binary_resolved(item):
            out[qid] = int(float(item["resolved_to"]))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resolution-date", default="2026-08-02")
    parser.add_argument("--output-name", default="resolved_early_summary_plots.png")
    args = parser.parse_args()

    outcomes = load_binary_outcomes(args.resolution_date)
    print(f"Binary outcomes available: {len(outcomes)}")
    summary = summarize_resolved(
        args.run_id,
        output_name=args.output_name,
        outcomes=outcomes,
    )
    # Rename JSON to avoid overwriting full-run resolved_summary.json
    results_dir = ROOT / "results" / args.run_id
    src = results_dir / "resolved_summary.json"
    dst = results_dir / "resolved_early_summary.json"
    if src.exists():
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
