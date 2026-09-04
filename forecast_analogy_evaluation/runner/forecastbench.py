"""Load ForecastBench question and resolution sets."""

from __future__ import annotations

import json
import random
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

MARKET_SOURCES = frozenset({"manifold", "metaculus", "kalshi", "polymarket"})
DATASET_SOURCES = frozenset({"acled", "fred", "wikipedia", "yahoo", "dbnomics", "ecb"})

BASE_URL = (
    "https://raw.githubusercontent.com/forecastingresearch/forecastbench-datasets/main"
)


@dataclass
class BenchmarkQuestion:
    question_id: str
    question: str
    background: str
    resolution_criteria: str
    forecast_timestamp: str
    outcome: Optional[int]
    source: str
    source_category: str  # market | dataset
    round_id: str
    cluster_id: str
    resolution_date: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "background": self.background,
            "resolution_criteria": self.resolution_criteria,
            "forecast_timestamp": self.forecast_timestamp,
            "outcome": self.outcome,
            "source": self.source,
            "source_category": self.source_category,
            "round_id": self.round_id,
            "cluster_id": self.cluster_id,
            "resolution_date": self.resolution_date,
        }


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)  # noqa: S310


def _round_id_from_question_file(name: str) -> str:
    # e.g. 2024-07-21-llm.json -> 2024-07-21
    return name.replace("-llm.json", "").replace("-human.json", "")


def load_round(
    round_date: str,
    raw_dir: Path,
    *,
    market_only: bool = False,
) -> tuple[List[dict], Dict[str, dict]]:
    """Return (questions, resolutions_by_id) for one ForecastBench round."""
    q_name = f"{round_date}-llm.json"
    r_name = f"{round_date}_resolution_set.json"
    q_path = raw_dir / "question_sets" / q_name
    r_path = raw_dir / "resolution_sets" / r_name
    _download(f"{BASE_URL}/datasets/question_sets/{q_name}", q_path)
    _download(f"{BASE_URL}/datasets/resolution_sets/{r_name}", r_path)

    with q_path.open(encoding="utf-8") as f:
        q_data = json.load(f)
    with r_path.open(encoding="utf-8") as f:
        r_data = json.load(f)

    resolutions: Dict[str, dict] = {}
    for item in r_data.get("resolutions", []):
        qid = item.get("id")
        if isinstance(qid, str):
            resolutions[qid] = item

    questions = q_data.get("questions", [])
    if market_only:
        questions = [q for q in questions if q.get("source") in MARKET_SOURCES]
    return questions, resolutions


def load_live_round(round_date: str, raw_dir: Path) -> dict[str, Any]:
    """Return the unresolved question set for a live ForecastBench round."""
    q_name = f"{round_date}-llm.json"
    q_path = raw_dir / "question_sets" / q_name
    _download(f"{BASE_URL}/datasets/question_sets/{q_name}", q_path)
    with q_path.open(encoding="utf-8") as f:
        return json.load(f)


def _is_binary_resolved(res: dict) -> bool:
    if not res or not res.get("resolved"):
        return False
    val = res.get("resolved_to")
    if val is None:
        return False
    if isinstance(val, (list, dict)):
        return False
    try:
        fval = float(val)
    except (TypeError, ValueError):
        return False
    return fval in (0.0, 1.0)


def _source_category(source: str) -> str:
    if source in MARKET_SOURCES:
        return "market"
    if source in DATASET_SOURCES:
        return "dataset"
    return "other"


def _forecast_timestamp(round_id: str) -> str:
    return f"{round_id}T00:00:00Z"


def _render_live_question_text(
    template: str,
    *,
    forecast_due_date: str,
    resolution_date: Optional[str],
) -> str:
    rendered = template.replace("{forecast_due_date}", forecast_due_date)
    if resolution_date is not None:
        rendered = rendered.replace("{resolution_date}", resolution_date)
    return rendered


def build_eligible_questions(
    questions: Sequence[dict],
    resolutions: Dict[str, dict],
    round_id: str,
) -> List[BenchmarkQuestion]:
    eligible: List[BenchmarkQuestion] = []
    forecast_ts = f"{round_id}T12:00:00Z"

    for q in questions:
        if q.get("combination_of") not in (None, "N/A"):
            continue
        qid = q.get("id")
        if not qid:
            continue
        res = resolutions.get(qid)
        if not _is_binary_resolved(res):
            continue
        source = str(q.get("source") or "")
        category = _source_category(source)
        if category == "other":
            continue
        outcome = int(float(res["resolved_to"]))
        cluster = qid
        combo = q.get("combination_of")
        if isinstance(combo, list) and combo:
            cluster = combo[0] if isinstance(combo[0], str) else qid

        eligible.append(BenchmarkQuestion(
            question_id=qid,
            question=str(q.get("question") or ""),
            background=str(q.get("background") or q.get("source_intro") or ""),
            resolution_criteria=str(q.get("resolution_criteria") or ""),
            forecast_timestamp=forecast_ts,
            outcome=outcome,
            source=source,
            source_category=category,
            round_id=round_id,
            cluster_id=cluster,
        ))
    return eligible


def build_live_questions(question_set: dict[str, Any]) -> List[BenchmarkQuestion]:
    """Expand a live question set into forecastable items.

    Market questions yield one item with ``resolution_date=None``.
    Dataset questions yield one item per listed resolution date.
    """
    round_id = str(question_set.get("forecast_due_date") or "")
    forecast_ts = _forecast_timestamp(round_id)
    items: List[BenchmarkQuestion] = []

    for q in question_set.get("questions", []):
        qid = q.get("id")
        if not isinstance(qid, str) or not qid:
            continue
        if q.get("combination_of") not in (None, "N/A"):
            continue
        source = str(q.get("source") or "")
        category = _source_category(source)
        if category == "other":
            continue

        base_kwargs = dict(
            question_id=qid,
            background=str(q.get("background") or q.get("source_intro") or ""),
            resolution_criteria=str(q.get("resolution_criteria") or ""),
            forecast_timestamp=forecast_ts,
            outcome=None,
            source=source,
            source_category=category,
            round_id=round_id,
            cluster_id=qid,
        )

        raw_question = str(q.get("question") or "")
        if category == "market":
            items.append(BenchmarkQuestion(
                question=_render_live_question_text(
                    raw_question,
                    forecast_due_date=round_id,
                    resolution_date=None,
                ),
                resolution_date=None,
                **base_kwargs,
            ))
            continue

        resolution_dates = q.get("resolution_dates")
        if not isinstance(resolution_dates, list):
            continue
        for resolution_date in resolution_dates:
            if not isinstance(resolution_date, str) or not resolution_date:
                continue
            items.append(BenchmarkQuestion(
                question=_render_live_question_text(
                    raw_question,
                    forecast_due_date=round_id,
                    resolution_date=resolution_date,
                ),
                resolution_date=resolution_date,
                **base_kwargs,
            ))
    return items


def sample_questions(
    questions: Sequence[BenchmarkQuestion],
    *,
    mode: str,
    n_questions: int,
    strata: Optional[Dict[str, int]] = None,
    seed: int = 42,
) -> List[BenchmarkQuestion]:
    if mode == "full_round":
        return list(questions)
    rng = random.Random(seed)
    if not strata:
        pool = list(questions)
        rng.shuffle(pool)
        return pool[:n_questions]

    by_cat: Dict[str, List[BenchmarkQuestion]] = {}
    for q in questions:
        by_cat.setdefault(q.source_category, []).append(q)

    picked: List[BenchmarkQuestion] = []
    for cat, count in strata.items():
        pool = list(by_cat.get(cat, []))
        rng.shuffle(pool)
        picked.extend(pool[:count])
    rng.shuffle(picked)
    return picked[:n_questions] if n_questions else picked
