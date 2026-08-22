"""Orchestration for the automatic evaluation.

Responsibilities (the metric itself lives in
:mod:`gemini_baselines.evaluation_mds`, which is our faithful port of the
paper's ``evaluation.py``):

* run each method over a dataset (or reuse previously saved answers),
* score every answer with the paper's MDS procedure,
* write per-example rows incrementally so a run can be resumed,
* aggregate into the paper's results-table layout and write CSV.

Nothing here changes the metric.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from gemini_baselines.evaluation_mds import (
    DIMENSION_WEIGHTS,
    DIMENSIONS,
    LITERAL_THRESHOLD,
    EvaluationContext,
    component_scores,
    pass_1_single,
    score_sample_detailed,
)
from hal.config import REPO_ROOT, get_settings
from hal.io_utils import load_dataset, read_jsonl
from hal.retry import ProviderError

from .methods import METHOD_LABELS, GenerationConfig, MethodRunner, canonical

RESULTS_ROOT = REPO_ROOT / "results"
RESULTS_DIR = RESULTS_ROOT / "automatic_evaluation"   # fallback when no model is known
GENERATIONS_DIRNAME = "generations"


def model_slug(model: str) -> str:
    """Turn a model id into a filesystem-safe folder fragment.

        gemma-4-31b-it        -> gemma_4_31b
        gemma-4-26b-a4b-it    -> gemma_4_26b_a4b
        gemini-3.5-flash-lite -> gemini_3_5_flash_lite
        llama3.1:8b-instruct  -> llama3_1_8b

    The trailing instruction-tuned marker is dropped: it says nothing about
    which model produced the numbers, and every model we use is instruct-tuned.
    """
    import re

    slug = (model or "unknown").strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug).strip("_")
    for suffix in ("_it", "_instruct", "_instruction_tuned"):
        if slug.endswith(suffix):
            slug = slug[: -len(suffix)]
            break
    return slug or "unknown"


def default_output_dir(generation_model: str,
                       evaluation_model: Optional[str] = None) -> Path:
    """``results/automatic_evaluation_<model>`` for the model that answered.

    Results from different models are different experimental conditions and
    must not share a directory. When a *different* model did the judging, its
    slug is appended too, so re-judging the same answers cannot silently
    overwrite the original scores.
    """
    name = f"automatic_evaluation_{model_slug(generation_model)}"
    if evaluation_model and model_slug(evaluation_model) != model_slug(generation_model):
        name += f"__judge_{model_slug(evaluation_model)}"
    return RESULTS_ROOT / name

# The columns of the paper's automatic-evaluation table.
COMPONENT_COLUMNS = [
    "TAbs", "TLit", "TAll",
    "BAbs", "BLit", "BAll",
    "PAbs", "PLit", "PAll",
    "RAbs", "RLit", "RAll",
    "MDS",
]


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def is_quota_error(exc: BaseException) -> bool:
    """Whether an exception looks like a provider quota / rate-limit refusal."""
    message = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in message for marker in
               ("429", "resource_exhausted", "resource exhausted", "quota",
                "rate limit", "ratelimit", "too many requests"))


class QuotaExhausted(RuntimeError):
    """Raised to stop a run cleanly when the provider refuses further calls."""


def _is_daily_quota(message: str) -> bool:
    lowered = message.lower()
    return "perday" in lowered.replace(" ", "") or "requests per day" in lowered


def _print_quota_notice(exc: BaseException, dataset: str, method: str,
                        index: int) -> None:
    """Explain a quota stop clearly, including what did NOT get evaluated."""
    message = str(exc)
    daily = _is_daily_quota(message)
    print(f"\n  !! Gemini refused further requests (quota / rate limit).")
    print(f"     Stopped at: method={method}, example index={index}, dataset={dataset}")
    if daily:
        print("     This looks like the PER-DAY free-tier cap, not a short burst "
              "limit,\n"
              "     so waiting a minute will not help -- it resets on Google's "
              "daily schedule.")
    else:
        print("     This looks like a short-term rate limit; waiting a little "
              "may be enough.")
    print("     Everything already completed has been saved to disk.")
    print("     Continue later with the SAME command plus --resume, e.g.:")
    print(f"       py examples/run_automatic_evaluation.py --dataset {dataset} "
          f"--methods all --resume")
    print("     Tip: methods run in order, so a quota stop always costs the LAST "
          "ones.\n"
          "     To make sure a specific method gets evaluated, run it first/alone:\n"
          f"       py examples/run_automatic_evaluation.py --dataset {dataset} "
          "--methods agentic")


@dataclass
class EvaluationConfig:
    datasets: List[str] = field(default_factory=lambda: ["popular"])
    methods: List[str] = field(default_factory=list)
    limit: Optional[int] = None
    smoke: bool = False
    resume: bool = False
    output_dir: Path = RESULTS_DIR
    evaluation_model: Optional[str] = None
    use_cache: bool = True
    verbose: bool = False
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    def run_tag(self) -> str:
        return "smoke" if self.smoke else "full"


def _detailed_path(config: EvaluationConfig, dataset: str) -> Path:
    suffix = "_smoke" if config.smoke else ""
    return config.output_dir / f"{dataset}_detailed{suffix}.jsonl"


def _summary_path(config: EvaluationConfig, dataset: str) -> Path:
    suffix = "_smoke" if config.smoke else ""
    return config.output_dir / f"automatic_evaluation_{dataset}{suffix}.csv"


def _generations_path(config: EvaluationConfig, dataset: str, method: str) -> Path:
    suffix = "_smoke" if config.smoke else ""
    return (config.output_dir / GENERATIONS_DIRNAME /
            f"{dataset}_{method}{suffix}.jsonl")


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    """Append one record and flush, so an interrupted run keeps its results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def dedupe_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One row per (dataset, method, example), keeping the most recent.

    The detailed file is append-only, so re-running a dataset without
    ``--resume`` writes a second row for examples that were already scored.
    Averaging over both would weight those examples twice, so every consumer of
    the rows collapses duplicates first. The last occurrence wins: a later run
    supersedes an earlier one.
    """
    latest: Dict[Tuple[Any, Any, Any], Dict[str, Any]] = {}
    for row in rows:
        latest[(row.get("dataset"), row.get("method"), row.get("index"))] = row
    return list(latest.values())


def load_detailed(path: Path) -> List[Dict[str, Any]]:
    """All rows of a detailed jsonl file, duplicates collapsed."""
    if not path.exists():
        return []
    return dedupe_rows(read_jsonl(path))


def load_completed(path: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """Existing successful rows, keyed by ``(method, index)`` -- for --resume."""
    if not path.exists():
        return {}
    done: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("status") == "ok":
            done[(row.get("method", ""), int(row.get("index", -1)))] = row
    return done


def load_saved_generations(path: Path) -> Dict[int, Dict[str, Any]]:
    """Previously saved method answers, keyed by example index."""
    if not path.exists():
        return {}
    return {int(row["index"]): row for row in read_jsonl(path)
            if row.get("index") is not None}


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------
class AutomaticEvaluation:
    """Generate (or load) analogies and score them with the paper's metric."""

    def __init__(self, config: EvaluationConfig,
                 evaluation_context: Optional[EvaluationContext] = None,
                 method_runner: Optional[MethodRunner] = None):
        self.config = config
        self.settings = get_settings()
        self.context = evaluation_context or EvaluationContext.build(
            model=config.evaluation_model, verbose=config.verbose,
            use_cache=config.use_cache,
        )
        self.methods = method_runner or MethodRunner(config.generation)

    # -- metadata ---------------------------------------------------------
    def metadata(self, dataset: str, method: str) -> Dict[str, Any]:
        """Reproducibility metadata. Never contains an API key."""
        generation = self.config.generation
        return {
            "dataset": dataset,
            "method": method,
            "method_label": METHOD_LABELS[method],
            "generation_model": generation.model or self.settings.model_for("baseline"),
            "evaluation_model": self.context.llm.model,
            "llm_provider": self.settings.llm_provider,
            "embedding_model": (self.settings.embedding_model
                                if method in ("direct_retrieval", "twostage_retrieval")
                                else None),
            # The full agentic configuration, not just the round count: two runs
            # of the same model at different settings are different conditions
            # and their averages must not be pooled.
            "refinement_rounds": (generation.refinement_rounds
                                  if method == "agentic" else None),
            "max_candidates": (generation.max_candidates
                               if method == "agentic" else None),
            "react_max_steps": (generation.react_max_steps
                                if method == "agentic" else None),
            "critique_top_n": (generation.critique_top_n
                               if method == "agentic" else None),
            "pool_limit": (generation.pool_limit
                           if method in ("direct_retrieval", "twostage_retrieval")
                           else None),
            "mode": self.config.run_tag(),
            "timestamp": utc_now(),
            "alpha": LITERAL_THRESHOLD,
            "dimension_weights": dict(DIMENSION_WEIGHTS),
        }

    # -- one example ------------------------------------------------------
    def evaluate_example(self, dataset: str, index: int, event: Dict[str, Any],
                         method: str,
                         generation: Optional[Dict[str, Any]] = None
                         ) -> Dict[str, Any]:
        """Produce (or reuse) an answer for one example and score it."""
        row: Dict[str, Any] = {
            "dataset": dataset,
            "index": index,
            "input_event": event.get("event_name", ""),
            "reference_event": event.get("target_event"),
            "event_type": event.get("event_type"),
            **self.metadata(dataset, method),
        }

        # --- 1. get the answer (generate, or reuse a saved one) ----------
        if generation is None:
            try:
                generation = self.methods.run(method, event)
            except ProviderError as exc:
                row.update(status="error", error=f"generation: {exc}")
                if is_quota_error(exc):
                    raise QuotaExhausted(str(exc)) from exc
                return row
        row["predicted_analogy"] = generation.get("analogy_event", "")
        row["candidate"] = generation.get("candidate", [])
        extra = generation.get("extra") or {}
        if extra:
            # Agentic metadata is recorded but never affects the score.
            row["method_metadata"] = extra

        # --- 2. Pass@1 (popular set only, reference answer available) ----
        if event.get("target_event"):
            try:
                row["pass_1"] = bool(pass_1_single(
                    event["target_event"], row["predicted_analogy"], self.context))
            except Exception as exc:  # a search failure must not kill the run
                row["pass_1"] = None
                row["pass_1_error"] = str(exc)

        # --- 3. MDS ------------------------------------------------------
        sample = {
            "event_name": event.get("event_name", ""),
            "event_intro": event.get("event_intro", ""),
            "analogy_event": row["predicted_analogy"],
        }
        try:
            scored, status = score_sample_detailed(sample, self.context)
        except ProviderError as exc:
            row.update(status="error", error=f"evaluation: {exc}")
            if is_quota_error(exc):
                raise QuotaExhausted(str(exc)) from exc
            return row

        if scored is None:
            # Not a scoring failure we can fake a number for: record why.
            row.update(status=status)
            return row

        row.update(component_scores(scored["score"]))
        row["raw_score"] = scored["score"]
        row["input_dimensions"] = scored["input_dimensions"]
        row["analogy_dimensions"] = scored["analogy_dimensions"]
        row["status"] = "ok"
        return row

    # -- one dataset ------------------------------------------------------
    def run_dataset(self, dataset: str, evaluate: bool = True,
                    generate: bool = True) -> List[Dict[str, Any]]:
        config = self.config
        events = load_dataset(dataset)
        if config.limit:
            events = events[: config.limit]

        detailed_path = _detailed_path(config, dataset)
        completed = load_completed(detailed_path) if config.resume else {}
        if completed:
            print(f"  resuming: {len(completed)} example/method results already done")

        rows: List[Dict[str, Any]] = list(completed.values())
        methods = config.methods or list(METHOD_LABELS)
        stopped = False

        for method in methods:
            method = canonical(method)
            saved = {} if generate else load_saved_generations(
                _generations_path(config, dataset, method))
            if not generate and not saved:
                print(f"  [{method}] no saved generations found -- skipping "
                      f"({_generations_path(config, dataset, method)})")
                continue

            print(f"\n  [{METHOD_LABELS[method]}] {len(events)} example(s)")
            for index, event in enumerate(events):
                if (method, index) in completed:
                    continue
                label = event.get("event_name", "?")
                try:
                    generation_row = None
                    if not generate:
                        record = saved.get(index)
                        if record is None:
                            print(f"    [{index}] {label}: no saved answer -- skipped")
                            continue
                        generation_row = {
                            "analogy_event": record.get("analogy_event", ""),
                            "candidate": record.get("candidate", []),
                            "extra": record.get("extra", {}),
                        }
                    elif config.resume:
                        # A previously saved answer avoids regenerating.
                        record = load_saved_generations(
                            _generations_path(config, dataset, method)).get(index)
                        if record is not None:
                            generation_row = {
                                "analogy_event": record.get("analogy_event", ""),
                                "candidate": record.get("candidate", []),
                                "extra": record.get("extra", {}),
                            }

                    if generate and generation_row is None:
                        generation_row = self.methods.run(method, event)
                        _append_jsonl(_generations_path(config, dataset, method), {
                            "dataset": dataset, "index": index,
                            "event_name": event.get("event_name", ""),
                            "timestamp": utc_now(),
                            **generation_row,
                        })

                    if not evaluate:
                        print(f"    [{index}] {label} -> "
                              f"{generation_row.get('analogy_event') or '(none)'}")
                        continue

                    row = self.evaluate_example(dataset, index, event, method,
                                                generation=generation_row)
                except QuotaExhausted as exc:
                    _print_quota_notice(exc, dataset, method, index)
                    stopped = True
                    break
                except ProviderError as exc:
                    row = {"dataset": dataset, "index": index, "method": method,
                           "status": "error", "error": str(exc),
                           "input_event": event.get("event_name", ""),
                           **self.metadata(dataset, method)}
                    if is_quota_error(exc):
                        _append_jsonl(detailed_path, row)
                        _print_quota_notice(exc, dataset, method, index)
                        stopped = True
                        break

                if evaluate:
                    _append_jsonl(detailed_path, row)
                    rows.append(row)
                    self._print_example(index, label, row)
            if stopped:
                break

        if evaluate:
            print(f"\n  detailed rows -> {detailed_path}")
        return dedupe_rows(rows)

    def _print_example(self, index: int, label: str, row: Dict[str, Any]) -> None:
        if row.get("status") != "ok":
            print(f"    [{index}] {label}: {row.get('status')} "
                  f"{row.get('error', '')}".rstrip())
            return
        pass_1 = row.get("pass_1")
        suffix = "" if pass_1 is None else f" pass@1={'yes' if pass_1 else 'no'}"
        print(f"    [{index}] {label} -> {row['predicted_analogy']} "
              f"(MDS {row['MDS']:.3f}{suffix})")


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------
def _condition_key(row: Dict[str, Any]) -> str:
    """Identify the experimental condition a row was produced under.

    Two rows only belong in the same average if the model that answered, the
    model that judged, and the pipeline settings all match.
    """
    parts = [str(row.get("generation_model")), str(row.get("evaluation_model"))]
    for field in ("refinement_rounds", "max_candidates", "react_max_steps",
                  "critique_top_n"):
        value = row.get(field)
        if value is not None:
            parts.append(f"{field}={value}")
    return "|".join(parts)


def aggregate(rows: Sequence[Dict[str, Any]], dataset: str,
              methods: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """Average the component columns per method, as the paper's table does.

    ``methods`` is the list that was *requested*. Any requested method with no
    results still gets a row (``n_evaluated = 0``), so a method that never ran
    -- because the quota stopped the run, for example -- is visible instead of
    silently vanishing from the table.
    """
    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for row in dedupe_rows(rows):
        if row.get("dataset") != dataset:
            continue
        by_method.setdefault(row.get("method", "?"), []).append(row)

    # Without an explicit request list, report only the methods that produced
    # rows. With one, every requested method gets a row so a method that never
    # ran shows up as NOT RUN instead of disappearing.
    if methods is None:
        wanted = [m for m in METHOD_LABELS if m in by_method]
    else:
        wanted = [m for m in methods if m in METHOD_LABELS]
        wanted += [m for m in by_method if m in METHOD_LABELS and m not in wanted]

    summary: List[Dict[str, Any]] = []
    for method in METHOD_LABELS:
        if method not in wanted:
            continue
        method_rows = by_method.get(method)
        if not method_rows:
            summary.append({
                "method": method,
                "method_label": METHOD_LABELS[method],
                "n_evaluated": 0,
                "n_attempted": 0,
                "Pass@1": None,
                "status_counts": {"not run": 0},
                **{column: None for column in COMPONENT_COLUMNS},
            })
            continue
        scored = [r for r in method_rows if r.get("status") == "ok"]
        entry: Dict[str, Any] = {
            "method": method,
            "method_label": METHOD_LABELS[method],
            "n_evaluated": len(scored),
            "n_attempted": len(method_rows),
        }
        for column in COMPONENT_COLUMNS:
            values = [float(r[column]) for r in scored if column in r]
            entry[column] = round(sum(values) / len(values), 4) if values else None

        # Pass@1 is defined over the samples that have a reference answer, and
        # counts a missing/unscorable answer as a miss -- as in the original,
        # which divides by len(dataset).
        with_reference = [r for r in method_rows if r.get("reference_event")]
        if with_reference:
            hits = sum(1 for r in with_reference if r.get("pass_1") is True)
            entry["Pass@1"] = round(hits / len(with_reference), 4)
        else:
            entry["Pass@1"] = None

        statuses: Dict[str, int] = {}
        for row in method_rows:
            statuses[row.get("status", "?")] = statuses.get(row.get("status", "?"), 0) + 1
        entry["status_counts"] = statuses

        # Averaging across different generation settings (or models) would pool
        # incomparable conditions into one number. Surface it rather than hide it.
        entry["mixed_conditions"] = sorted({
            _condition_key(row) for row in method_rows
        }) if len({_condition_key(row) for row in method_rows}) > 1 else []
        entry["evaluation_model"] = method_rows[0].get("evaluation_model")
        entry["generation_model"] = method_rows[0].get("generation_model")
        summary.append(entry)
    return summary


def write_summary_csv(summary: Sequence[Dict[str, Any]], path: Path,
                      include_pass_1: bool) -> Path:
    """Write the aggregate table: one row per method, metrics as columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = (["method", "method_label", "n_evaluated", "n_attempted"]
               + COMPONENT_COLUMNS
               + (["Pass@1"] if include_pass_1 else [])
               + ["generation_model", "evaluation_model"])
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for entry in summary:
            writer.writerow(entry)
    return path


def print_summary_table(summary: Sequence[Dict[str, Any]], dataset: str,
                        include_pass_1: bool, smoke: bool = False) -> None:
    """Print a readable table in the shape of the paper's results table."""
    columns = COMPONENT_COLUMNS + (["Pass@1"] if include_pass_1 else [])
    header = f"{'Method':<28}" + "".join(f"{c:>8}" for c in columns) + f"{'n':>6}"
    title = f"AUTOMATIC EVALUATION -- {dataset.upper()}"
    if smoke:
        title += "   [SMOKE RESULTS -- NOT RESEARCH RESULTS]"
    print("\n" + "=" * len(header))
    print(title)
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for entry in summary:
        line = f"{entry['method_label']:<28}"
        for column in columns:
            value = entry.get(column)
            line += f"{value:>8.3f}" if isinstance(value, (int, float)) else f"{'-':>8}"
        line += f"{entry['n_evaluated']:>6}"
        if entry["n_evaluated"] == 0:
            line += "   <-- NOT RUN"
        print(line)
    print("-" * len(header))
    print("TAbs/TLit/TAll = Topic abstract (1-4) / literal (Jaccard) / "
          "abstract*max(alpha-literal,0)")
    print("MDS = weighted sum over dimensions; "
          f"weights {DIMENSION_WEIGHTS}, alpha={LITERAL_THRESHOLD}")
    model = next((e.get("evaluation_model") for e in summary
                  if e.get("evaluation_model")), None)
    if model:
        print(f"evaluation model: {model}")

    # Anything incomplete must be impossible to miss: an absent or short row
    # is not a result, it is missing data.
    expected = max((e["n_attempted"] for e in summary), default=0)
    incomplete = [e for e in summary
                  if e["n_evaluated"] == 0 or e["n_attempted"] < expected]
    unscored = [e for e in summary
                if e["n_attempted"] and e["n_evaluated"] < e["n_attempted"]]
    mixed = [e for e in summary if e.get("mixed_conditions")]
    if incomplete or unscored or mixed:
        print("\nWARNING -- this table is incomplete:")
        for entry in incomplete:
            if entry["n_evaluated"] == 0:
                print(f"  * {entry['method_label']}: did not run at all "
                      "(no results). Re-run with --resume.")
            else:
                print(f"  * {entry['method_label']}: only "
                      f"{entry['n_attempted']}/{expected} examples were attempted.")
        for entry in summary:
            for condition in entry.get("mixed_conditions", []):
                print(f"  * {entry['method_label']}: rows come from MORE THAN ONE "
                      f"condition -- {condition}")
        for entry in unscored:
            skipped = entry["n_attempted"] - entry["n_evaluated"]
            reasons = {k: v for k, v in entry.get("status_counts", {}).items()
                       if k != "ok"}
            print(f"  * {entry['method_label']}: {skipped} of "
                  f"{entry['n_attempted']} answers could not be scored {reasons}")
        print("  Averages are computed only over the examples that were scored, "
              "so methods with different n are not directly comparable.")
