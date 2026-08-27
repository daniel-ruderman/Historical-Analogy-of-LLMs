"""Generate ForecastBench submission files for a live round.

Produces one JSON file per configured arm/slot and keeps raw run artifacts under
``runs/<run_id>/`` for provenance.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.schemas import ForecastRecord
from runner.engine import EvaluationEngine, _load_analysis_packets, _read_completed_keys
from runner.forecastbench import build_live_questions, load_live_round


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    return cfg


def slugify_filename_component(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip())
    slug = slug.strip(".-")
    return slug or "submission"


def load_forecasts(path: Path) -> List[ForecastRecord]:
    records: List[ForecastRecord] = []
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(ForecastRecord.from_dict(json.loads(line)))
    return records


def filter_live_items(
    items: Sequence,
    *,
    categories: List[str],
    dataset_question_limit: int | None,
    dataset_horizons_per_question: int | None,
    seed: int,
) -> List:
    want = set(categories)
    picked: List = []

    if "market" in want:
        picked.extend([q for q in items if q.source_category == "market"])

    if "dataset" in want:
        dataset_items = [q for q in items if q.source_category == "dataset"]
        by_qid: Dict[str, List] = {}
        for q in dataset_items:
            by_qid.setdefault(q.question_id, []).append(q)

        qids = list(by_qid.keys())
        rng = random.Random(seed)
        rng.shuffle(qids)
        if dataset_question_limit is not None:
            qids = qids[: max(0, dataset_question_limit)]

        for qid in qids:
            horizons = list(by_qid[qid])
            horizons.sort(key=lambda x: x.resolution_date or "")
            if dataset_horizons_per_question is not None:
                horizons = horizons[: max(0, dataset_horizons_per_question)]
            picked.extend(horizons)

    return picked


def build_submission_payload(
    *,
    organization: str,
    model_name: str,
    model_organization: str,
    question_set_name: str,
    forecasts: List[ForecastRecord],
    include_reasoning: bool,
) -> Dict[str, Any]:
    rows = []
    for rec in forecasts:
        rows.append({
            "id": rec.question_id,
            "source": rec.source,
            "forecast": rec.p_yes_clipped,
            "resolution_date": rec.resolution_date,
            "reasoning": rec.rationale if include_reasoning else None,
        })
    return {
        "organization": organization,
        "model": model_name,
        "model_organization": model_organization,
        "question_set": question_set_name,
        "forecasts": rows,
    }


def write_submission_files(
    *,
    run_dir: Path,
    round_date: str,
    organization: str,
    model_organization: str,
    model_name_root: str,
    question_set_name: str,
    slots: Dict[str, int],
    forecasts: List[ForecastRecord],
    include_reasoning: bool,
) -> List[Path]:
    out_dir = run_dir / "submissions"
    out_dir.mkdir(parents=True, exist_ok=True)

    by_arm: Dict[str, List[ForecastRecord]] = {}
    for rec in forecasts:
        if rec.status == "ok":
            by_arm.setdefault(rec.condition, []).append(rec)

    created: List[Path] = []
    org_slug = slugify_filename_component(organization)
    for arm, slot in sorted(slots.items(), key=lambda kv: kv[1]):
        arm_forecasts = by_arm.get(arm, [])
        payload = build_submission_payload(
            organization=organization,
            model_name=f"{model_name_root} ({arm})",
            model_organization=model_organization,
            question_set_name=question_set_name,
            forecasts=arm_forecasts,
            include_reasoning=include_reasoning,
        )
        out_path = out_dir / f"{round_date}.{org_slug}.{slot}.json"
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        created.append(out_path)
    return created


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate ForecastBench submission files")
    parser.add_argument("--config", default="configs/prospective_forecastbench.yaml")
    parser.add_argument("--round-date", required=True, help="ForecastBench round date YYYY-MM-DD")
    parser.add_argument("--run-id", default=None, help="Run identifier (default: auto)")
    parser.add_argument("--organization", required=True, help="ForecastBench organization field")
    parser.add_argument("--model-organization", default=None,
                        help="ForecastBench model_organization field (defaults to organization)")
    parser.add_argument("--model-name-root", default=None,
                        help="Base model label; arm name is appended automatically")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional cap on expanded forecast items for rehearsals")
    parser.add_argument("--arms", nargs="*", default=None,
                        help="Override arms (default: main arms from config)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume an interrupted live generation run")
    parser.add_argument("--dry-run", action="store_true",
                        help="Mock LLM for schema rehearsal")
    parser.add_argument("--include-reasoning", action="store_true",
                        help="Include rationale strings in output files")
    parser.add_argument(
        "--source-categories",
        nargs="+",
        choices=["market", "dataset"],
        default=["market", "dataset"],
        help="Question categories to include (default: both).",
    )
    parser.add_argument(
        "--dataset-question-limit",
        type=int,
        default=None,
        help="Limit number of unique dataset question IDs (sampled by seed).",
    )
    parser.add_argument(
        "--dataset-horizons-per-question",
        type=int,
        default=None,
        help="Keep only first N sorted horizons per selected dataset question.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    args = parser.parse_args(argv)

    config_path = ROOT / args.config
    config = load_config(config_path)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("live_%Y%m%d_%H%M%S")
    run_dir = ROOT / config["output"]["runs_dir"] / run_id

    raw_dir = ROOT / "data" / "raw" / "forecastbench"
    question_set = load_live_round(args.round_date, raw_dir)
    items = build_live_questions(question_set)
    items = filter_live_items(
        items,
        categories=args.source_categories,
        dataset_question_limit=args.dataset_question_limit,
        dataset_horizons_per_question=args.dataset_horizons_per_question,
        seed=args.seed,
    )
    if args.limit is not None:
        items = items[: args.limit]

    arms = args.arms or config["arms"]["main"]
    engine = EvaluationEngine(config, run_id, dry_run=args.dry_run)
    engine.write_manifest(len(items), arms, smoke=False)

    done = _read_completed_keys(run_dir / "forecasts.jsonl") if args.resume else set()
    cached_packets = _load_analysis_packets(run_dir / "analysis_packets.jsonl") if args.resume else {}

    print(f"Run ID: {run_id}")
    print(f"Question set: {question_set.get('question_set')}")
    print(f"Expanded forecast items: {len(items)}")
    print(f"Arms: {', '.join(arms)}")
    print(f"Categories: {','.join(args.source_categories)}")
    if args.dataset_question_limit is not None:
        print(f"Dataset question limit: {args.dataset_question_limit}")
    if args.dataset_horizons_per_question is not None:
        print(f"Dataset horizons/question: {args.dataset_horizons_per_question}")

    for i, q in enumerate(items, 1):
        print(f"\n[{i}/{len(items)}] {q.question_id} ({q.source_category})")
        if q.resolution_date:
            print(f"  Resolution date: {q.resolution_date}")
        print(f"  Q: {q.question[:100]}...")
        engine.run_question(
            q,
            arms,
            smoke=False,
            done=done,
            cached_packets=cached_packets,
        )

    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["status"] = "complete"
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    model_organization = args.model_organization or args.organization
    model_name_root = args.model_name_root or config["model"]["llm"]
    slots = dict(config.get("benchmark", {}).get("submission", {}).get("slots", {}))
    missing_slots = [arm for arm in arms if arm not in slots]
    if missing_slots:
        raise SystemExit(f"Missing submission slots in config for arms: {missing_slots}")

    created = write_submission_files(
        run_dir=run_dir,
        round_date=args.round_date,
        organization=args.organization,
        model_organization=model_organization,
        model_name_root=model_name_root,
        question_set_name=str(question_set.get("question_set") or f"{args.round_date}-llm.json"),
        slots={arm: slots[arm] for arm in arms},
        forecasts=load_forecasts(run_dir / "forecasts.jsonl"),
        include_reasoning=args.include_reasoning,
    )

    registry = ROOT / "data" / "manifests" / "run_registry.jsonl"
    registry.parent.mkdir(parents=True, exist_ok=True)
    with registry.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "run_id": run_id,
            "stage": config["study"]["stage"],
            "config_path": str(config_path),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "model": model_name_root,
            "n_questions": len(items),
            "status": "complete",
            "notes": f"live_submission dry_run={args.dry_run}",
        }) + "\n")

    print(f"\nDone. Raw artifacts: {run_dir}")
    print("Submission files:")
    for path in created:
        print(f"  - {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
