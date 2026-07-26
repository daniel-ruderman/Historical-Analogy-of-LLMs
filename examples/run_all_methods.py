"""Run every method on a real example from the repository's datasets.

Six original-paper baselines (through our provider abstraction) plus our agentic
pipeline, on the same input event, with clearly labelled output.

Quick check (few API calls, small event pool, 1 refinement round)::

    python examples/run_all_methods.py --smoke

Full run on one real example::

    python examples/run_all_methods.py --dataset popular --index 0

Only our pipeline::

    python examples/run_all_methods.py --methods agentic --dataset popular --index 0

Offline demo with fake providers -- no API key, no network::

    python examples/run_all_methods.py --dry-run

Cost note: the retrieval methods must embed the event pool once (658 events).
Vectors are cached on disk, so only the first run pays for it; ``--smoke``
embeds only ``--pool-limit`` events (default 40).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hal.config import get_settings, load_settings, set_settings  # noqa: E402
from hal.io_utils import (  # noqa: E402
    DATASETS,
    configure_stdout,
    read_jsonl,
    write_jsonl,
)

BASELINE_METHODS = [
    ("direct_retrieval", "DIRECT RETRIEVAL"),
    ("twostage_retrieval", "TWO-STAGE RETRIEVAL"),
    ("direct_generation", "DIRECT GENERATION"),
    ("twostage_generation", "TWO-STAGE GENERATION"),
    ("summary_generation", "GENERATION WITH SUMMARIZING"),
    ("reflection_generation", "SELF-REFLECTION"),
]
ALL_METHOD_KEYS = [key for key, _ in BASELINE_METHODS] + ["agentic"]
RETRIEVAL_METHODS = {"direct_retrieval", "twostage_retrieval"}


def banner(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def truncate(text: str, limit: int = 400) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def bullets(items: List[str], indent: str = "    - ", limit: int = 6) -> None:
    for item in items[:limit]:
        print(f"{indent}{truncate(str(item), 300)}")


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------
def run_baselines(event: Dict[str, Any], args, methods: List[str]) -> Dict[str, Any]:
    from gemini_baselines.common import BaselineContext

    results: Dict[str, Any] = {}
    needs_embeddings = any(m in RETRIEVAL_METHODS for m in methods)
    context = BaselineContext.build(model=args.model, with_embeddings=needs_embeddings,
                                    verbose=args.verbose)
    pool = None
    if needs_embeddings:
        from gemini_baselines.direct_retrieval import get_pool

        print(f"\n(embedding the event pool: "
              f"{args.pool_limit if args.pool_limit else 'all 658'} events, cached on disk)")
        pool = get_pool(context, limit=args.pool_limit, verbose=True)

    for key, title in BASELINE_METHODS:
        if key not in methods:
            continue
        banner(title)
        started = time.time()
        try:
            if key == "direct_retrieval":
                from gemini_baselines import direct_retrieval

                row = direct_retrieval.run(event, context, pool=pool)
            elif key == "twostage_retrieval":
                from gemini_baselines import twostage_retrieval

                row = twostage_retrieval.run(event, context, pool=pool)
            else:
                from gemini_baselines import get_method

                row = get_method(key)(dict(event), context)
        except Exception as exc:
            print(f"  ! {type(exc).__name__}: {exc}")
            results[key] = {"error": f"{type(exc).__name__}: {exc}"}
            continue

        elapsed = time.time() - started
        print(f"  analogy event : {row.get('analogy_event') or '(none)'}")
        candidates = row.get("candidate") or []
        for index, candidate_set in enumerate(candidates):
            label = "candidate set" if len(candidates) == 1 else f"candidate set {index + 1}"
            print(f"  {label} : {', '.join(map(str, candidate_set))}")
        print(f"  ({elapsed:.1f}s)")
        results[key] = row
    return results


# --------------------------------------------------------------------------
# Our agentic pipeline
# --------------------------------------------------------------------------
def run_agentic(event: Dict[str, Any], args) -> Dict[str, Any]:
    from agentic_pipeline import AgenticAnalogyPipeline

    banner("OUR AGENTIC PIPELINE")
    pipeline = AgenticAnalogyPipeline.build(
        refinement_rounds=args.rounds,
        max_candidates=args.max_candidates,
        react_max_steps=args.react_steps,
        critique_top_n=args.critique_top_n,
        verbose=args.verbose,
    )
    print(f"  rounds={pipeline.config.refinement_rounds} "
          f"max_candidates={pipeline.config.max_candidates} "
          f"react_steps={pipeline.config.react_max_steps}")
    started = time.time()
    result = pipeline.run(event)
    elapsed = time.time() - started

    print("\n  -- initial candidates (Generate/Search agent) --")
    for candidate in result.initial_candidates:
        flag = {True: "verified", False: "UNVERIFIED", None: "unchecked"}[
            candidate.event.verified]
        print(f"    * {candidate.name} [{flag}] (confidence {candidate.confidence:.2f})")
        if candidate.rationale:
            print(f"      rationale: {truncate(candidate.rationale, 220)}")

    for round_data in result.rounds:
        print(f"\n  -- refinement round {round_data.index} --")
        print("    Critic feedback:")
        for critique in round_data.critiques:
            scores = ", ".join(f"{k}={v:g}" for k, v in critique.dimension_scores.items())
            print(f"      * {critique.candidate}: overall {critique.overall_score:g}/4"
                  f" -> {critique.recommendation}"
                  + (f" ({scores})" if scores else ""))
            if critique.summary:
                print(f"        {truncate(critique.summary, 260)}")
            if critique.important_differences:
                print(f"        differences: "
                      f"{truncate('; '.join(critique.important_differences), 260)}")
        print("    Anti-Analogy counterexamples:")
        for report in round_data.anti_analogies:
            print(f"      * {report.candidate}: {report.verdict} "
                  f"(robustness {report.robustness:.2f})")
            for counter in report.counterexamples[:3]:
                print(f"        - {counter.event_name}: "
                      f"{truncate(counter.divergence, 200)}")
        print("    Revisions:")
        for revision in round_data.revisions:
            arrow = (f" (was {revision.previous_candidate})"
                     if revision.previous_candidate else "")
            print(f"      * {revision.action}: {revision.candidate}{arrow}")
        print(f"    Candidates after round {round_data.index}: "
              f"{', '.join(c.name for c in round_data.candidates_after)}")

    print("\n  -- Final Judge ranking --")
    for row in result.ranking.ranking:
        print(f"    {row.rank}. {row.event_name} (score {row.score:g})")
        if row.reason:
            print(f"       reason: {truncate(row.reason, 260)}")
        if row.weaknesses:
            print(f"       weaknesses: {truncate('; '.join(row.weaknesses), 260)}")
    if result.ranking.notes:
        print(f"    notes: {truncate(result.ranking.notes, 200)}")

    print("\n  -- Final Summarizer --")
    print(f"    winning analogy: {result.analogy_event}")
    if result.explanation:
        print(f"    explanation: {truncate(result.explanation, 900)}")
    if result.similarities:
        print("    similarities:")
        bullets(result.similarities, "      - ")
    if result.differences:
        print("    differences:")
        bullets(result.differences, "      - ")
    if result.counterexamples:
        print("    counterexamples:")
        bullets(result.counterexamples, "      - ")
    if result.limitations:
        print("    limitations:")
        bullets(result.limitations, "      - ")
    if result.why_ranked_first:
        print(f"    why it ranked first: {truncate(result.why_ranked_first, 400)}")
    if result.errors:
        print("    non-fatal issues:")
        bullets(result.errors, "      - ")
    print(f"\n  ({elapsed:.1f}s)")
    return result.to_output_row()


# --------------------------------------------------------------------------
# Offline demo providers
# --------------------------------------------------------------------------
def install_dry_run_providers() -> None:
    """Register scripted fake providers so the script runs with no API key."""
    from hal.providers.factory import register_embedding, register_llm, register_search
    from hal.providers.mock import (
        MockEmbeddingProvider,
        MockLLMProvider,
        make_mock_wikipedia,
    )
    from hal.io_utils import load_event_pool

    pages = {row["history_event_text"]: row["history_intro_text"]
             for row in load_event_pool()}
    search = make_mock_wikipedia([(k, v) for k, v in pages.items()])

    def responder(prompt: str) -> str:
        # Enough structure for every parser in the project to succeed.
        # Match the opening role sentence: prompts also mention the other roles.
        if "You are the Generate/Search agent" in prompt:
            return (
                '{"thought": "propose candidates", "action": "finish", "result": '
                '{"candidates": [{"event_name": "French Revolution", '
                '"description": "Revolution in France from 1789.", '
                '"rationale": "Popular uprising against an entrenched regime.", '
                '"mapping": {"topic": "revolution", "background": "economic crisis", '
                '"process": "mass protest", "result": "regime change"}, '
                '"confidence": 0.7, "action": "keep"}, '
                '{"event_name": "Revolutions of 1848", '
                '"description": "A wave of revolutions across Europe.", '
                '"rationale": "A regional cascade of uprisings.", '
                '"mapping": {"topic": "revolutionary wave"}, "confidence": 0.6, '
                '"action": "keep"}]}}'
            )
        if "You are the Critic agent" in prompt:
            return (
                '{"thought": "assess", "action": "finish", "result": '
                '{"dimension_scores": {"topic": 3, "background": 3, "process": 3, '
                '"result": 2}, "structural_correspondence": "Both are cascading '
                'uprisings against entrenched regimes.", '
                '"important_differences": ["Different media environment"], '
                '"weak_assumptions": ["Assumes comparable state capacity"], '
                '"factual_problems": [], "surface_level": false, '
                '"overall_score": 3, "recommendation": "keep", '
                '"summary": "Reasonable structural match, weakest on outcomes."}}'
            )
        if "You are the Anti-Analogy agent" in prompt:
            return (
                '{"thought": "look for divergent cases", "action": "finish", "result": '
                '{"counterexamples": [{"event_name": "Revolutions of 1848", '
                '"description": "Most 1848 revolutions were reversed.", '
                '"divergence": "Similar uprisings ended in restoration, not reform."}], '
                '"failure_modes": ["Expecting durable democratisation"], '
                '"robustness": 0.5, "verdict": "weakened", '
                '"summary": "The pattern often reverses within a few years."}}'
            )
        if "You are the Final Judge" in prompt:
            return (
                '{"ranking": [{"rank": 1, "event_name": "Revolutions of 1848", '
                '"score": 7.5, "reason": "Closest cascade structure.", '
                '"weaknesses": ["Different outcomes"]}, '
                '{"rank": 2, "event_name": "French Revolution", "score": 6.0, '
                '"reason": "Single-country case.", "weaknesses": ["Scale differs"]}], '
                '"notes": "The top two were close."}'
            )
        if "You are the Final Summarizer" in prompt:
            return (
                '{"explanation": "The comparison highlights how regional waves of '
                'uprising spread and then stall.", '
                '"similarities": ["Cascade across neighbouring states"], '
                '"differences": ["Different communication technology"], '
                '"counterexamples": ["1848: most gains were reversed"], '
                '"limitations": ["Do not infer inevitable restoration"], '
                '"why_ranked_first": "It matches the cascade structure more closely."}'
            )
        if "Python list" in prompt or "list format" in prompt:
            return '["French Revolution","Revolutions of 1848","Cold War"]'
        if "summarize it into four parts" in prompt:
            return ("1. Summary: A political upheaval. 2. Background: Long-standing "
                    "grievances. 3. Process: Mass mobilisation. 4. Result: Regime change.")
        if "Reflection" in prompt or "Final Answer" in prompt:
            return "Thought: The candidate set is adequate.\n\nFinal Answer:\nFrench Revolution"
        return "French Revolution"

    register_llm("mock", lambda settings, role, model: MockLLMProvider(
        responses=responder, model=model or f"mock-{role}"))
    register_embedding("mock", lambda settings, model: MockEmbeddingProvider(
        model=model or "mock-embedding"))
    register_search("mock", lambda settings: search)
    set_settings(load_settings(llm_provider="mock", embedding_provider="mock",
                               search_provider="mock", cache_enabled=False))


# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the paper's baselines and our agentic pipeline on one example"
    )
    parser.add_argument("--dataset", default="popular",
                        choices=["popular", "general"],
                        help="which dataset from dataset/ to use")
    parser.add_argument("--index", type=int, default=0,
                        help="index of the event inside the dataset")
    parser.add_argument("--methods", default="all",
                        help="'all', 'baselines', 'agentic', or a comma-separated list: "
                             + ",".join(ALL_METHOD_KEYS))
    parser.add_argument("--rounds", type=int, default=None,
                        help="refinement rounds for our pipeline (default REFINEMENT_ROUNDS)")
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--react-steps", type=int, default=None)
    parser.add_argument("--critique-top-n", type=int, default=None,
                        help="only critique the N most confident candidates")
    parser.add_argument("--model", default=None, help="override LLM_MODEL for every role")
    parser.add_argument("--provider", default=None, help="override LLM_PROVIDER")
    parser.add_argument("--pool-limit", type=int, default=None,
                        help="embed only the first N pool events (retrieval methods)")
    parser.add_argument("--smoke", action="store_true",
                        help="fast/cheap check: small pool, 1 round, 3 candidates")
    parser.add_argument("--full", action="store_true",
                        help="explicit full run: entire 658-event pool, configured rounds")
    parser.add_argument("--dry-run", action="store_true",
                        help="use fake providers (no API key, no network)")
    parser.add_argument("--output", default=None, help="write all results to this .jsonl")
    parser.add_argument("--verbose", action="store_true",
                        help="show agent tool calls as they happen")
    args = parser.parse_args()
    configure_stdout()

    if args.smoke and not args.full:
        args.pool_limit = args.pool_limit or 40
        args.rounds = 1 if args.rounds is None else args.rounds
        args.max_candidates = args.max_candidates or 3
        args.react_steps = args.react_steps or 1
        args.critique_top_n = args.critique_top_n or 2
    if args.full:
        args.pool_limit = None

    if args.dry_run:
        install_dry_run_providers()
    else:
        overrides = {}
        if args.provider:
            overrides["llm_provider"] = args.provider
        if args.model:
            overrides["llm_model"] = args.model
            overrides["role_models"] = {}
        set_settings(load_settings(**overrides))

    settings = get_settings()
    if not args.dry_run and settings.llm_provider == "gemini" and not settings.api_key:
        print("ERROR: GEMINI_API_KEY is not set.\n"
              "  Copy .env.example to .env and add your key, or export GEMINI_API_KEY.\n"
              "  To try the plumbing without a key, run with --dry-run.")
        sys.exit(1)

    selection = args.methods.strip().lower()
    if selection == "all":
        methods = list(ALL_METHOD_KEYS)
    elif selection == "baselines":
        methods = [key for key, _ in BASELINE_METHODS]
    else:
        methods = [m.strip() for m in selection.split(",") if m.strip()]
        unknown = [m for m in methods if m not in ALL_METHOD_KEYS]
        if unknown:
            parser.error(f"unknown method(s): {unknown}. Known: {ALL_METHOD_KEYS}")

    testset = read_jsonl(DATASETS[args.dataset])
    if not 0 <= args.index < len(testset):
        parser.error(f"--index must be in [0, {len(testset) - 1}] for '{args.dataset}'")
    event = testset[args.index]

    banner("INPUT EVENT")
    print(f"  dataset      : {args.dataset} (index {args.index}, {len(testset)} events)")
    print(f"  event_name   : {event['event_name']}")
    print(f"  event_intro  : {truncate(event['event_intro'], 600)}")
    if event.get("target_event"):
        print(f"  reference    : {event['target_event']}  (popular-set gold answer)")
    if event.get("event_type"):
        print(f"  event_type   : {event['event_type']}")
    print(f"\n  configuration:\n  {settings.describe()}")
    if args.smoke:
        print("  MODE: smoke (reduced pool / rounds / candidates)")
    if args.dry_run:
        print("  MODE: dry-run (fake providers -- results are not research output)")

    results: Dict[str, Any] = {}
    baseline_keys = [m for m in methods if m != "agentic"]
    if baseline_keys:
        results.update(run_baselines(event, args, baseline_keys))
    if "agentic" in methods:
        results["agentic"] = run_agentic(event, args)

    banner("SUMMARY")
    for key in ALL_METHOD_KEYS:
        if key not in results:
            continue
        row = results[key]
        answer = row.get("error") or row.get("analogy_event") or "(none)"
        print(f"  {key:<22} -> {answer}")
    if event.get("target_event"):
        print(f"  {'reference answer':<22} -> {event['target_event']}")

    if args.output:
        rows = []
        for key, row in results.items():
            if isinstance(row, dict) and "error" not in row:
                enriched = dict(row)
                enriched["method"] = key
                rows.append(enriched)
        write_jsonl(args.output, rows)
        print(f"\nWrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
