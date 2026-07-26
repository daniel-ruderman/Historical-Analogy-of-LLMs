"""Agent 2 -- the Critic agent.

Evaluates every candidate analogy along the paper's four dimensions plus the
deeper structural questions: does the causal skeleton really map, which
differences matter, which assumptions are weak, is anything factually wrong, and
is the analogy mostly surface-level?

It is a ReAct-style agent: it may search before criticising a candidate.  Its
output is structured (:class:`~hal.schemas.Critique`) so the Generate/Search
agent can act on it in the next round.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from hal.json_utils import get_float, get_list, get_str
from hal.providers.base import LLMProvider, SearchProvider
from hal.schemas import DIMENSIONS, CandidateAnalogy, Critique, HistoricalEvent

from . import prompts
from .react import ReActAgent, ReActOutcome

RECOMMENDATIONS = ("keep", "revise", "replace")


class CriticAgent:
    """Structured criticism of candidate analogies."""

    name = "critic"

    def __init__(self, llm: LLMProvider, search: Optional[SearchProvider] = None,
                 max_steps: int = 3, top_k: int = 4, verbose: bool = False):
        self.llm = llm
        self.verbose = verbose
        self._react = ReActAgent(llm, search, max_steps=max_steps, top_k=top_k,
                                 verbose=verbose, name=self.name)

    def critique_one(self, event: HistoricalEvent, candidate: CandidateAnalogy
                     ) -> Tuple[Critique, ReActOutcome]:
        prompt = prompts.CRITIC.format(
            event=event.text(), candidate=_describe_candidate(candidate)
        )
        outcome = self._react.run(prompt)
        return self._parse(candidate, outcome), outcome

    def critique_all(self, event: HistoricalEvent,
                     candidates: Sequence[CandidateAnalogy]) -> List[Critique]:
        critiques = []
        for candidate in candidates:
            if self.verbose:
                print(f"    [critic] {candidate.name}")
            critique, _ = self.critique_one(event, candidate)
            critiques.append(critique)
        return critiques

    # -- internals --------------------------------------------------------
    def _parse(self, candidate: CandidateAnalogy, outcome: ReActOutcome) -> Critique:
        critique = Critique(candidate=candidate.name, evidence=list(outcome.evidence))
        result = outcome.result if isinstance(outcome.result, dict) else None
        if result is None:
            # Malformed output must not abort a run: emit a neutral critique that
            # tells the Generate/Search agent nothing was established.
            critique.summary = (
                "The Critic could not produce a usable assessment for this candidate "
                f"({outcome.error or 'unparseable output'})."
            )
            critique.overall_score = 0.0
            critique.recommendation = "keep"
            return critique

        scores = result.get("dimension_scores")
        if isinstance(scores, dict):
            for dimension in DIMENSIONS:
                value = get_float(scores, dimension, default=0.0)
                if value:
                    critique.dimension_scores[dimension] = _clamp(value, 1.0, 4.0)
        critique.structural_correspondence = get_str(result, "structural_correspondence")
        critique.important_differences = get_list(result, "important_differences",
                                                  "differences")
        critique.weak_assumptions = get_list(result, "weak_assumptions", "assumptions")
        critique.factual_problems = get_list(result, "factual_problems",
                                             "evidence_problems")
        critique.surface_level = bool(result.get("surface_level", False))
        critique.summary = get_str(result, "summary", "assessment")

        overall = get_float(result, "overall_score", "score", default=0.0)
        if not overall and critique.dimension_scores:
            overall = sum(critique.dimension_scores.values()) / len(critique.dimension_scores)
        critique.overall_score = _clamp(overall, 0.0, 4.0)

        recommendation = get_str(result, "recommendation", "action").lower()
        critique.recommendation = (
            recommendation if recommendation in RECOMMENDATIONS else "keep"
        )
        if candidate.event.verified is False and not critique.factual_problems:
            critique.factual_problems.append(
                "The candidate could not be verified in the knowledge base."
            )
        return critique


def _describe_candidate(candidate: CandidateAnalogy) -> str:
    verified = {True: "verified", False: "NOT VERIFIED in the knowledge base",
                None: "not checked"}[candidate.event.verified]
    lines = [f"{candidate.name} [{verified}]",
             f"description: {candidate.event.description[:1200]}",
             f"proposed rationale: {candidate.rationale}"]
    if candidate.mapping:
        mapping = "; ".join(f"{k}: {v}" for k, v in candidate.mapping.items())
        lines.append(f"claimed correspondence: {mapping}")
    return "\n".join(lines)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


describe_candidate = _describe_candidate
