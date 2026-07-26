"""Final Summarizer -- explains the winning analogy.

A plain LLM call (no tools, no autonomous loop).  It produces something useful
to a person studying the analogy: what the comparison illuminates, which
structural similarities and differences matter, which counterexamples to keep in
mind, where the analogy should not be pushed, and why it beat the runner-up.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from hal.json_utils import get_list, get_str, parse_json_object
from hal.providers.base import LLMProvider
from hal.schemas import (
    AntiAnalogyReport,
    CandidateAnalogy,
    Critique,
    FinalAnalogyResult,
    HistoricalEvent,
    JudgeRanking,
)

from . import prompts
from .critic_agent import describe_candidate


class FinalSummarizer:
    """Turns the winning candidate into a readable explanation."""

    name = "final_summarizer"

    def __init__(self, llm: LLMProvider, verbose: bool = False):
        self.llm = llm
        self.verbose = verbose

    def summarize(self, event: HistoricalEvent, winner: CandidateAnalogy,
                  ranking: JudgeRanking, critique: Optional[Critique] = None,
                  anti_analogy: Optional[AntiAnalogyReport] = None) -> dict:
        alternatives = "\n".join(
            f"{row.rank}. {row.event_name} -- {row.reason}"
            for row in ranking.ranking[1:6]
        ) or "(no alternatives)"
        judge_reason = ranking.winner.reason if ranking.winner else ""

        prompt = prompts.FINAL_SUMMARIZER.format(
            event=event.text(),
            winner=describe_candidate(winner),
            judge_reason=judge_reason or "(not given)",
            alternatives=alternatives,
            critique=critique.as_feedback() if critique else "(no critique)",
            anti_analogy=anti_analogy.as_feedback() if anti_analogy
            else "(no counterexamples)",
        )
        raw = self.llm.generate(prompt, json_output=True) or ""
        data = parse_json_object(raw) or {}
        if not data:
            # Fall back to the structured material we already have.
            return _fallback(winner, critique, anti_analogy, ranking, raw)
        return {
            "explanation": get_str(data, "explanation", "summary"),
            "similarities": get_list(data, "similarities", "important_similarities"),
            "differences": get_list(data, "differences", "important_differences"),
            "counterexamples": get_list(data, "counterexamples"),
            "limitations": get_list(data, "limitations"),
            "why_ranked_first": get_str(data, "why_ranked_first", "why_first"),
        }

    def apply(self, result: FinalAnalogyResult, critique: Optional[Critique] = None,
              anti_analogy: Optional[AntiAnalogyReport] = None) -> FinalAnalogyResult:
        """Run the summarizer and fill the fields of ``result`` in place."""
        if result.winning_candidate is None:
            return result
        summary = self.summarize(result.input_event, result.winning_candidate,
                                 result.ranking, critique, anti_analogy)
        result.explanation = summary.get("explanation", "")
        result.similarities = summary.get("similarities", [])
        result.differences = summary.get("differences", [])
        result.counterexamples = summary.get("counterexamples", [])
        result.limitations = summary.get("limitations", [])
        result.why_ranked_first = summary.get("why_ranked_first", "")
        return result


def _fallback(winner: CandidateAnalogy, critique: Optional[Critique],
              anti_analogy: Optional[AntiAnalogyReport], ranking: JudgeRanking,
              raw: str) -> dict:
    similarities: List[str] = []
    if winner.rationale:
        similarities.append(winner.rationale)
    similarities.extend(f"{k}: {v}" for k, v in winner.mapping.items())
    differences = list(critique.important_differences) if critique else []
    limitations = list(critique.weak_assumptions) if critique else []
    counterexamples = (
        [f"{c.event_name} -- {c.divergence}" for c in anti_analogy.counterexamples]
        if anti_analogy else []
    )
    explanation = (raw.strip()[:1500] if raw.strip()
                   else (critique.summary if critique else winner.rationale))
    return {
        "explanation": explanation,
        "similarities": similarities,
        "differences": differences,
        "counterexamples": counterexamples,
        "limitations": limitations,
        "why_ranked_first": ranking.winner.reason if ranking.winner else "",
    }
