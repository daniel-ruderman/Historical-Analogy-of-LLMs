"""Tests for forecast evaluation scoring."""

import json
import tempfile
from pathlib import Path

from analysis.aggregate import run_aggregate
from analysis.schemas import ForecastRecord
from analysis.scoring import brier_score, cluster_bootstrap_ci, paired_differences


def test_brier_score():
    assert brier_score(0.7, 1) == (0.7 - 1) ** 2
    assert brier_score(0.7, 0) == 0.7 ** 2


def test_paired_differences():
    forecasts = [
        ForecastRecord("q1", "historical_analogy", 0.7, outcome=1),
        ForecastRecord("q1", "plain", 0.5, outcome=1),
        ForecastRecord("q2", "historical_analogy", 0.3, outcome=0),
        ForecastRecord("q2", "plain", 0.6, outcome=0),
    ]
    rows = paired_differences(forecasts, "historical_analogy", "plain")
    assert len(rows) == 2
    assert rows[0]["question_id"] == "q1"


def test_cluster_bootstrap():
    rows = [{"difference": -0.01, "round_id": "r1", "cluster_id": "c1"}] * 50
    result = cluster_bootstrap_ci(rows, n_bootstrap=100, seed=0)
    assert result["n"] == 50
    assert result["mean"] == -0.01


def test_aggregate_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run_dir = root / "runs" / "test_run"
        run_dir.mkdir(parents=True)
        config_path = root / "config.yaml"
        config_path.write_text(
            "analysis:\n"
            "  primary_contrast: [historical_analogy, plain]\n"
            "  secondary_contrast: [historical_analogy, matched_deliberation]\n"
            "  bootstrap_samples: 50\n",
            encoding="utf-8",
        )
        forecasts = [
            {"question_id": "q1", "condition": "historical_analogy", "p_yes": 0.8,
             "outcome": 1, "round_id": "r1", "cluster_id": "c1", "status": "ok"},
            {"question_id": "q1", "condition": "plain", "p_yes": 0.5,
             "outcome": 1, "round_id": "r1", "cluster_id": "c1", "status": "ok"},
            {"question_id": "q1", "condition": "matched_deliberation", "p_yes": 0.6,
             "outcome": 1, "round_id": "r1", "cluster_id": "c1", "status": "ok"},
        ]
        with (run_dir / "forecasts.jsonl").open("w", encoding="utf-8") as f:
            for row in forecasts:
                f.write(json.dumps(row) + "\n")

        # Patch ROOT by calling internals with modified paths
        from analysis import aggregate as agg_mod
        old_root = agg_mod.ROOT
        agg_mod.ROOT = root
        try:
            summary = run_aggregate("test_run", config_path)
            assert summary["n_forecasts"] == 3
            assert "primary_practical_brier" in summary["contrasts"]
        finally:
            agg_mod.ROOT = old_root
