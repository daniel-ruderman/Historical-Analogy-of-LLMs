"""Registry mapping a method name to the ALREADY IMPLEMENTED algorithm.

This module contains **no** analogy-producing logic of its own. Each entry
calls the existing implementation in ``gemini_baselines/`` or
``agentic_pipeline/`` and normalises its return value into a common shape:

    {"analogy_event": str, "candidate": [...], "extra": {...}}

so the evaluator can score every method through the same code path.

    method implementation  ->  produces analogy  ->  evaluator  ->  scores
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

# Canonical keys match examples/run_all_methods.py. The alternative spellings
# used in some notes are accepted as aliases.
METHOD_LABELS: Dict[str, str] = {
    "direct_retrieval": "Direct Retrieval",
    "twostage_retrieval": "Two-stage Retrieval",
    "direct_generation": "Direct Generation",
    "twostage_generation": "Two-stage Generation",
    "summary_generation": "Generation with Summarizing",
    "reflection_generation": "Self-reflection",
    "agentic": "Our Agentic Pipeline",
}

ALIASES: Dict[str, str] = {
    "two_stage_retrieval": "twostage_retrieval",
    "two_stage_generation": "twostage_generation",
    "summarizing": "summary_generation",
    "self_reflection": "reflection_generation",
    "agentic_pipeline": "agentic",
}

BASELINE_METHODS = [
    "direct_retrieval", "twostage_retrieval", "direct_generation",
    "twostage_generation", "summary_generation", "reflection_generation",
]
ALL_METHODS = BASELINE_METHODS + ["agentic"]
RETRIEVAL_METHODS = {"direct_retrieval", "twostage_retrieval"}


def canonical(name: str) -> str:
    """Resolve an alias to the canonical method key."""
    key = (name or "").strip().lower()
    key = ALIASES.get(key, key)
    if key not in METHOD_LABELS:
        raise ValueError(
            f"Unknown method {name!r}. Known: {sorted(METHOD_LABELS)} "
            f"(aliases: {sorted(ALIASES)})"
        )
    return key


def expand_methods(names: List[str]) -> List[str]:
    """Expand ``all`` / ``baselines`` / aliases into canonical method keys."""
    if not names:
        return list(ALL_METHODS)
    out: List[str] = []
    for raw in names:
        for token in str(raw).replace(",", " ").split():
            token = token.strip().lower()
            if token == "all":
                out.extend(ALL_METHODS)
            elif token in ("baselines", "baseline"):
                out.extend(BASELINE_METHODS)
            else:
                out.append(canonical(token))
    seen = set()
    ordered = []
    for key in out:
        if key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


@dataclass
class GenerationConfig:
    """How the *generation* side should be run (not the evaluation side)."""

    pool_limit: Optional[int] = None      # retrieval: embed only N pool events
    refinement_rounds: Optional[int] = None
    max_candidates: Optional[int] = None
    react_max_steps: Optional[int] = None
    critique_top_n: Optional[int] = None
    model: Optional[str] = None           # override the generation model only
    verbose: bool = False


@dataclass
class MethodRunner:
    """Lazily builds the shared objects the methods need, then runs them."""

    config: GenerationConfig = field(default_factory=GenerationConfig)
    _context: Any = None
    _pool: Any = None
    _pipeline: Any = None

    # -- shared resources -------------------------------------------------
    def baseline_context(self):
        if self._context is None:
            from gemini_baselines.common import BaselineContext

            self._context = BaselineContext.build(
                model=self.config.model,
                with_embeddings=True,
                verbose=self.config.verbose,
            )
        return self._context

    def event_pool(self):
        """Embed the event pool once and share it across retrieval methods."""
        if self._pool is None:
            from gemini_baselines.direct_retrieval import get_pool

            self._pool = get_pool(self.baseline_context(),
                                  limit=self.config.pool_limit,
                                  verbose=self.config.verbose)
        return self._pool

    def pipeline(self):
        if self._pipeline is None:
            from agentic_pipeline import AgenticAnalogyPipeline

            self._pipeline = AgenticAnalogyPipeline.build(
                refinement_rounds=self.config.refinement_rounds,
                max_candidates=self.config.max_candidates,
                react_max_steps=self.config.react_max_steps,
                critique_top_n=self.config.critique_top_n,
                verbose=self.config.verbose,
            )
        return self._pipeline

    # -- running ----------------------------------------------------------
    def run(self, method: str, event: Dict[str, Any]) -> Dict[str, Any]:
        """Produce one analogy with ``method``; returns a normalised record."""
        method = canonical(method)
        if method == "agentic":
            return self._run_agentic(event)
        return self._run_baseline(method, event)

    def _run_baseline(self, method: str, event: Dict[str, Any]) -> Dict[str, Any]:
        from gemini_baselines import get_method

        context = self.baseline_context()
        run = get_method(method)
        if method in RETRIEVAL_METHODS:
            row = run(dict(event), context, pool=self.event_pool())
        else:
            row = run(dict(event), context)
        return {
            "analogy_event": (row.get("analogy_event") or "").strip(),
            "candidate": row.get("candidate") or [],
            "extra": {},
        }

    def _run_agentic(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """The pipeline's answer is the FINAL winning analogy, nothing else.

        Intermediate candidates are kept as metadata for inspection only; they
        never reach the metric, so the agentic method is scored by exactly the
        same procedure as the baselines.
        """
        result = self.pipeline().run(dict(event))
        ranking = [
            {"rank": row.rank, "event_name": row.event_name, "score": row.score}
            for row in result.ranking.ranking
        ]
        return {
            "analogy_event": (result.analogy_event or "").strip(),
            "candidate": [[c.name for c in result.initial_candidates]]
            + [[c.name for c in r.candidates_after] for r in result.rounds],
            "extra": {
                "final_winning_candidate": result.analogy_event,
                "refinement_rounds": len(result.rounds),
                "judge_ranking": ranking,
                "final_candidates": [c.name for c in result.final_candidates],
                "pipeline_errors": list(result.errors),
            },
        }
