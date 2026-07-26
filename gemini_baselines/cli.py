"""Shared command line plumbing for the baseline scripts.

Each baseline keeps the original interface (``--model``, ``--testset``) and the
original output format (one jsonl row per input event, containing
``analogy_event`` and ``candidate``).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict, List, Optional

from hal.config import get_settings, load_settings, set_settings
from hal.io_utils import DATASETS, configure_stdout, read_jsonl, write_jsonl

from .common import BaselineContext


def build_parser(description: str, needs_llm: bool = True) -> argparse.ArgumentParser:
    configure_stdout()
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--testset", required=True, type=str,
        help="'popular', 'general' or a path to a .jsonl file with "
             "'event_name'/'event_intro' keys",
    )
    if needs_llm:
        parser.add_argument("--model", type=str, default=None,
                            help="model id (defaults to LLM_MODEL / BASELINE_MODEL)")
        parser.add_argument("--provider", type=str, default=None,
                            help="LLM provider (defaults to LLM_PROVIDER, e.g. 'gemini')")
    parser.add_argument("--output", type=str, default="output.jsonl")
    parser.add_argument("--limit", type=int, default=None,
                        help="only process the first N events")
    parser.add_argument("--verbose", action="store_true")
    return parser


def resolve_testset(name_or_path: str) -> List[Dict]:
    if name_or_path in DATASETS:
        return read_jsonl(DATASETS[name_or_path])
    return read_jsonl(name_or_path)


def apply_provider_overrides(args) -> None:
    """Let ``--provider``/``--model`` override the environment configuration."""
    overrides = {}
    provider = getattr(args, "provider", None)
    if provider:
        overrides["llm_provider"] = provider
    if overrides:
        set_settings(load_settings(**overrides))


def run_over_dataset(
    method: Callable[[Dict, BaselineContext], Dict],
    args,
    context: BaselineContext,
    progress: bool = True,
) -> List[Dict]:
    """Apply ``method`` to every event, writing rows in the original format."""
    testset = resolve_testset(args.testset)
    if args.limit:
        testset = testset[: args.limit]
    output_path = Path(args.output)
    if output_path.exists():
        output_path.unlink()
    rows = []
    for index, event in enumerate(testset, start=1):
        if progress:
            print(f"[{index}/{len(testset)}] {event.get('event_name', '?')}")
        try:
            row = method(dict(event), context)
        except Exception as exc:  # keep going; record the failure
            row = dict(event)
            row["analogy_event"] = ""
            row["candidate"] = []
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"    ! {row['error']}")
        rows.append(row)
        write_jsonl(output_path, [row], mode="a+")
    print(f"\nWrote {len(rows)} rows to {output_path}")
    return rows


def make_context(args, needs_embeddings: bool = False,
                 needs_llm: bool = True) -> BaselineContext:
    apply_provider_overrides(args)
    settings = get_settings(refresh=True)
    return BaselineContext.build(
        model=getattr(args, "model", None),
        settings=settings,
        with_embeddings=needs_embeddings,
        verbose=getattr(args, "verbose", False),
    ) if needs_llm else BaselineContext.build(
        settings=settings, with_embeddings=needs_embeddings,
        verbose=getattr(args, "verbose", False),
    )


def optional_int(value: Optional[str]) -> Optional[int]:
    return int(value) if value not in (None, "") else None
