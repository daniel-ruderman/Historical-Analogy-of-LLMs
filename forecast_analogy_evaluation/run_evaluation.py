"""Run the historical-analogy forecast evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.aggregate import run_aggregate
from runner.engine import EvaluationEngine, _load_analysis_packets, _read_completed_keys
from runner.forecastbench import build_eligible_questions, load_round, sample_questions


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    return cfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run forecast analogy evaluation")
    parser.add_argument("--config", default="configs/retrospective_pilot.yaml")
    parser.add_argument("--run-id", default=None, help="Run identifier (default: auto)")
    parser.add_argument("--round-date", default="2024-07-21",
                        help="ForecastBench round date (YYYY-MM-DD)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max questions (overrides config sampling)")
    parser.add_argument("--smoke", action="store_true",
                        help="Reduced pipeline settings; implies --limit 2 if unset")
    parser.add_argument("--dry-run", action="store_true",
                        help="Mock LLM — no Ollama/GPU required")
    parser.add_argument("--resume", action="store_true",
                        help="Skip completed question×condition pairs")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Skip generation; only run aggregate analysis")
    parser.add_argument("--arms", nargs="*", default=None,
                        help="Override arms (default: main arms from config)")
    parser.add_argument(
        "--source-categories",
        nargs="+",
        choices=["market", "dataset"],
        default=None,
        help="Restrict to these source categories (default: both)",
    )
    parser.add_argument(
        "--dataset-question-limit",
        type=int,
        default=None,
        help="If set with dataset category, sample at most this many dataset questions",
    )
    parser.add_argument("--seed", type=int, default=None,
                        help="Override sampling seed")
    args = parser.parse_args(argv)

    config_path = ROOT / args.config
    config = load_config(config_path)

    run_id = args.run_id or datetime.now(timezone.utc).strftime("pilot_%Y%m%d_%H%M%S")
    run_dir = ROOT / config["output"]["runs_dir"] / run_id

    if args.analyze_only:
        print(f"Analyzing run {run_id}...")
        summary = run_aggregate(run_id, config_path)
        print(json.dumps(summary, indent=2))
        return 0

    limit = args.limit
    if args.smoke and limit is None:
        limit = 2

    print(f"Run ID: {run_id}")
    print(f"Config: {config_path}")
    print(f"Round:  {args.round_date}")
    print(f"Smoke:  {args.smoke}  Dry-run: {args.dry_run}  Limit: {limit}")
    print(f"Categories: {args.source_categories or ['market', 'dataset']}")

    raw_dir = ROOT / "data" / "raw" / "forecastbench"
    questions_raw, resolutions = load_round(args.round_date, raw_dir)
    eligible = build_eligible_questions(questions_raw, resolutions, args.round_date)
    print(f"Eligible binary questions in round: {len(eligible)}")

    if args.source_categories:
        want = set(args.source_categories)
        eligible = [q for q in eligible if q.source_category in want]
        print(f"After category filter: {len(eligible)}")

    sampling = config["benchmark"].get("sampling", {})
    seed = args.seed if args.seed is not None else sampling.get("seed", 42)

    if (
        args.dataset_question_limit is not None
        and args.source_categories == ["dataset"]
        and limit is None
    ):
        sampled = sample_questions(
            eligible,
            mode="stratified_sample",
            n_questions=args.dataset_question_limit,
            strata={"dataset": args.dataset_question_limit},
            seed=seed,
        )
    elif args.source_categories == ["market"] and limit is None:
        sampled = sample_questions(
            eligible,
            mode="full_round",
            n_questions=len(eligible),
            seed=seed,
        )
    elif limit is not None:
        if args.source_categories and len(args.source_categories) == 1:
            cat = args.source_categories[0]
            sampled = sample_questions(
                eligible,
                mode="stratified_sample",
                n_questions=limit,
                strata={cat: limit},
                seed=seed,
            )
        else:
            sampled = sample_questions(
                eligible,
                mode="stratified_sample",
                n_questions=limit,
                strata={"market": limit // 2 + limit % 2, "dataset": limit // 2},
                seed=seed,
            )
    else:
        sampled = sample_questions(
            eligible,
            mode=sampling.get("mode", "stratified_sample"),
            n_questions=int(sampling.get("n_questions", 100)),
            strata=sampling.get("strata"),
            seed=seed,
        )
    print(f"Selected {len(sampled)} questions")

    arms = args.arms or config["arms"]["main"]
    if args.smoke:
        arms = [a for a in arms if a in config["arms"]["main"]]

    engine = EvaluationEngine(config, run_id, dry_run=args.dry_run)
    engine.write_manifest(len(sampled), arms, args.smoke)

    done = _read_completed_keys(run_dir / "forecasts.jsonl") if args.resume else set()
    cached_packets = _load_analysis_packets(run_dir / "analysis_packets.jsonl") if args.resume else {}

    for i, q in enumerate(sampled, 1):
        print(f"\n[{i}/{len(sampled)}] {q.question_id} ({q.source_category})")
        print(f"  Q: {q.question[:100]}...")
        engine.run_question(
            q, arms, smoke=args.smoke, done=done, cached_packets=cached_packets,
        )

    # Mark manifest complete
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["status"] = "complete"
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Register run
    registry = ROOT / "data" / "manifests" / "run_registry.jsonl"
    registry.parent.mkdir(parents=True, exist_ok=True)
    with registry.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "run_id": run_id,
            "stage": config["study"]["stage"],
            "config_path": str(config_path),
            "started_at": manifest.get("started_at") if manifest_path.exists() else "",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "model": config["model"]["llm"],
            "n_questions": len(sampled),
            "status": "complete",
            "notes": f"smoke={args.smoke} dry_run={args.dry_run}",
        }) + "\n")

    print("\n=== AGGREGATE ANALYSIS ===")
    summary = run_aggregate(run_id, config_path)
    print(json.dumps(summary, indent=2))

    # Update experiment log reminder
    print(f"\nDone. Artifacts: {run_dir}")
    print("Update docs/experiment_log.md with this run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
