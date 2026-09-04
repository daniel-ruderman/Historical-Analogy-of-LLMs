#!/usr/bin/env python3
"""Report resolution status for a ForecastBench round.

For resolved historical rounds, summarizes the official resolution set.
For live rounds without a published resolution file, cross-matches question
IDs against the newest available resolution set(s).
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runner.forecastbench import (  # noqa: E402
    BASE_URL,
    _is_binary_resolved,
    _source_category,
)

API_CONTENTS = (
    "https://api.github.com/repos/forecastingresearch/"
    "forecastbench-datasets/contents/datasets/resolution_sets?ref=main"
)


def _get_json(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "forecast-analogy-eval"})
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def _try_get_json(url: str) -> Optional[Any]:
    try:
        return _get_json(url)
    except Exception as exc:  # noqa: BLE001
        print(f"MISSING/ERROR {url}: {exc}")
        return None


def _list_resolution_names() -> List[str]:
    data = _get_json(API_CONTENTS)
    names = sorted(
        x["name"]
        for x in data
        if isinstance(x, dict) and str(x.get("name", "")).endswith(".json")
    )
    return names


def _load_resolutions(round_date: str) -> Optional[Dict[str, dict]]:
    url = f"{BASE_URL}/datasets/resolution_sets/{round_date}_resolution_set.json"
    data = _try_get_json(url)
    if not data:
        return None
    out: Dict[str, dict] = {}
    for item in data.get("resolutions", []):
        qid = item.get("id")
        if isinstance(qid, str):
            out[qid] = item
    return out


def _load_questions(round_date: str) -> List[dict]:
    url = f"{BASE_URL}/datasets/question_sets/{round_date}-llm.json"
    data = _get_json(url)
    return list(data.get("questions", []))


def _summarize_with_resolutions(
    questions: List[dict],
    resolutions: Dict[str, dict],
    *,
    label: str,
) -> Dict[str, Any]:
    by_source = Counter()
    by_cat = Counter()
    binary_resolved = 0
    resolved_any = 0
    unresolved = 0
    missing_in_res_file = 0
    examples: List[Dict[str, Any]] = []

    for q in questions:
        qid = q.get("id")
        if not isinstance(qid, str):
            continue
        if q.get("combination_of") not in (None, "N/A"):
            continue
        source = str(q.get("source") or "")
        cat = _source_category(source)
        if cat == "other":
            continue
        res = resolutions.get(qid)
        if res is None:
            missing_in_res_file += 1
            continue
        if res.get("resolved"):
            resolved_any += 1
            by_source[source] += 1
            by_cat[cat] += 1
            if _is_binary_resolved(res):
                binary_resolved += 1
                if len(examples) < 15:
                    examples.append({
                        "id": qid,
                        "source": source,
                        "resolved_to": res.get("resolved_to"),
                        "question": str(q.get("question") or "")[:120],
                    })
        else:
            unresolved += 1

    return {
        "label": label,
        "n_eligible_questions": sum(
            1
            for q in questions
            if isinstance(q.get("id"), str)
            and q.get("combination_of") in (None, "N/A")
            and _source_category(str(q.get("source") or "")) in {"market", "dataset"}
        ),
        "resolved_any": resolved_any,
        "binary_resolved": binary_resolved,
        "unresolved_in_file": unresolved,
        "missing_from_resolution_file": missing_in_res_file,
        "by_category": dict(by_cat),
        "by_source": dict(by_source),
        "examples": examples,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round-date", required=True)
    parser.add_argument(
        "--fallback-resolution-dates",
        nargs="*",
        default=None,
        help="Optional resolution files to cross-match against if round file missing",
    )
    args = parser.parse_args()

    names = _list_resolution_names()
    print(f"Published resolution sets: {len(names)}")
    print("Newest:")
    for name in names[-12:]:
        print(f"  {name}")

    questions = _load_questions(args.round_date)
    print(f"\nQuestion set {args.round_date}: {len(questions)} raw questions")

    own = _load_resolutions(args.round_date)
    report: Dict[str, Any] = {
        "round_date": args.round_date,
        "own_resolution_set_published": own is not None,
        "newest_published_resolution_set": names[-1] if names else None,
    }

    if own is not None:
        summary = _summarize_with_resolutions(
            questions, own, label=f"{args.round_date}_resolution_set"
        )
        report["summary"] = summary
        print(json.dumps(report, indent=2))
        return 0

    print(f"\nNo official {args.round_date}_resolution_set.json yet.")
    fallback_dates = args.fallback_resolution_dates
    if not fallback_dates:
        # Cross-match against the few newest published resolution files.
        fallback_dates = [
            n.replace("_resolution_set.json", "")
            for n in names[-5:]
        ]

    combined: Dict[str, dict] = {}
    used_files: List[str] = []
    for d in fallback_dates:
        res = _load_resolutions(d)
        if res is None:
            continue
        used_files.append(d)
        for qid, item in res.items():
            # Prefer a later file's entry if the same id appears twice.
            combined[qid] = item

    summary = _summarize_with_resolutions(
        questions,
        combined,
        label=f"cross_match:{','.join(used_files)}",
    )
    report["cross_matched_resolution_files"] = used_files
    report["summary"] = summary
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
