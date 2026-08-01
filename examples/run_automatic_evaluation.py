"""Automatic evaluation of every implemented method.

Scores the analogies produced by the six provider-neutral baselines and by our
agentic pipeline, using the original paper's automatic evaluation
(Multi-Dimensional Similarity, plus Pass@1 on the popular set).

Tiny check first (a couple of examples, reduced generation settings):

    py examples/run_automatic_evaluation.py --dataset popular --methods agentic --smoke

Full runs:

    py examples/run_automatic_evaluation.py --dataset popular --methods all
    py examples/run_automatic_evaluation.py --dataset general --methods all
    py examples/run_automatic_evaluation.py --dataset all     --methods all

Resume after a quota interruption (skips example/method pairs already scored,
and reuses answers already generated):

    py examples/run_automatic_evaluation.py --dataset general --methods all --resume

Score answers that were generated earlier, without calling the methods again:

    py examples/run_automatic_evaluation.py --dataset popular --methods all --evaluate

Generate answers only, and score them later:

    py examples/run_automatic_evaluation.py --dataset popular --methods all --generate

Offline check with fake providers (no API key, no quota):

    py examples/run_automatic_evaluation.py --dataset popular --methods all --dry-run

Results are written to results/automatic_evaluation/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from automatic_evaluation import (  # noqa: E402
    METHOD_LABELS,
    AutomaticEvaluation,
    EvaluationConfig,
    GenerationConfig,
    aggregate,
    expand_methods,
    print_summary_table,
    write_summary_csv,
)
from automatic_evaluation.runner import (  # noqa: E402
    RESULTS_DIR,
    _detailed_path,
    _summary_path,
)
from hal.config import get_settings, load_settings, set_settings  # noqa: E402
from hal.io_utils import configure_stdout  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Automatic evaluation (paper MDS + Pass@1) of all methods",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default="popular",
                        choices=["popular", "general", "all"],
                        help="which dataset to evaluate")
    parser.add_argument("--methods", nargs="+", default=["all"],
                        help="'all', 'baselines', or names: "
                             + ", ".join(METHOD_LABELS))
    parser.add_argument("--limit", type=int, default=None,
                        help="evaluate only the first N examples")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny check: 2 examples, small pool, 1 round, "
                             "3 candidates. NOT research results.")
    parser.add_argument("--resume", action="store_true",
                        help="skip example/method pairs already scored and reuse "
                             "answers already generated")
    parser.add_argument("--generate", action="store_true",
                        help="only produce and save answers, do not score them")
    parser.add_argument("--evaluate", action="store_true",
                        help="only score previously saved answers, do not generate")
    parser.add_argument("--evaluation-model", default=None,
                        help="model for the evaluation LLM "
                             "(default: EVALUATION_MODEL, else LLM_MODEL)")
    parser.add_argument("--generation-model", default=None,
                        help="model used to PRODUCE the analogies (kept separate "
                             "from the evaluation model)")
    parser.add_argument("--rounds", type=int, default=None,
                        help="refinement rounds for the agentic pipeline")
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--react-steps", type=int, default=None)
    parser.add_argument("--critique-top-n", type=int, default=None)
    parser.add_argument("--pool-limit", type=int, default=None,
                        help="embed only the first N pool events (retrieval methods)")
    parser.add_argument("--output-dir", default=str(RESULTS_DIR))
    parser.add_argument("--no-cache", action="store_true",
                        help="disable the evaluation cache (more API calls)")
    parser.add_argument("--dry-run", action="store_true",
                        help="use fake providers: no API key, no network, no quota")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    configure_stdout()
    args = parse_args(argv)

    if args.dry_run:
        from examples.run_all_methods import install_dry_run_providers

        install_dry_run_providers()

    if args.generation_model:
        set_settings(load_settings(llm_model=args.generation_model, role_models={}))
    settings = get_settings()

    if not args.dry_run and settings.llm_provider == "gemini" and not settings.api_key:
        print("ERROR: GEMINI_API_KEY is not set.\n"
              "  Put your key in .env (see .env.example), or use --dry-run to check\n"
              "  the plumbing without an API key.")
        return 1

    # --generate / --evaluate select a phase; neither flag means do both.
    generate = not args.evaluate
    evaluate = not args.generate

    if args.smoke:
        args.limit = args.limit or 2
        args.pool_limit = args.pool_limit or 40
        args.rounds = 1 if args.rounds is None else args.rounds
        args.max_candidates = args.max_candidates or 3
        args.react_steps = args.react_steps or 1
        args.critique_top_n = args.critique_top_n or 2

    datasets: List[str] = (["popular", "general"] if args.dataset == "all"
                           else [args.dataset])
    methods = expand_methods(args.methods)

    config = EvaluationConfig(
        datasets=datasets,
        methods=methods,
        limit=args.limit,
        smoke=args.smoke,
        resume=args.resume,
        output_dir=Path(args.output_dir),
        evaluation_model=args.evaluation_model,
        use_cache=not args.no_cache,
        verbose=args.verbose,
        generation=GenerationConfig(
            pool_limit=args.pool_limit,
            refinement_rounds=args.rounds,
            max_candidates=args.max_candidates,
            react_max_steps=args.react_steps,
            critique_top_n=args.critique_top_n,
            model=args.generation_model,
            verbose=args.verbose,
        ),
    )

    evaluation = AutomaticEvaluation(config)

    print("=" * 72)
    print("AUTOMATIC EVALUATION" + ("  [SMOKE MODE -- not research results]"
                                    if args.smoke else ""))
    print("=" * 72)
    print(f"  datasets         : {', '.join(datasets)}")
    print(f"  methods          : {', '.join(methods)}")
    print(f"  generation model : {config.generation.model or settings.model_for('baseline')}")
    print(f"  evaluation model : {evaluation.context.llm.model}")
    print(f"  phase            : "
          f"{'generate only' if not evaluate else ('evaluate saved answers' if not generate else 'generate + evaluate')}")
    print(f"  resume           : {args.resume}")
    print(f"  output           : {config.output_dir}")

    exit_code = 0
    for dataset in datasets:
        print("\n" + "-" * 72)
        print(f"DATASET: {dataset}")
        print("-" * 72)
        rows = evaluation.run_dataset(dataset, evaluate=evaluate, generate=generate)
        if not evaluate:
            print(f"  answers saved under {config.output_dir / 'generations'}")
            continue

        summary = aggregate(rows, dataset, methods=methods)
        if not summary:
            print("  nothing was scored.")
            exit_code = 1
            continue
        include_pass_1 = any(entry.get("Pass@1") is not None for entry in summary)
        print_summary_table(summary, dataset, include_pass_1, smoke=args.smoke)
        csv_path = write_summary_csv(summary, _summary_path(config, dataset),
                                     include_pass_1)
        print(f"\n  aggregate table -> {csv_path}")
        print(f"  per-example rows -> {_detailed_path(config, dataset)}")

    stats = evaluation.context
    print(f"\nevaluation LLM calls: {stats.llm_calls} "
          f"(cache hits: {stats.cache_hits})")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
