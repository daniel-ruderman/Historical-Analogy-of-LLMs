"""Final Judge -- the ranking stage after the iterative refinement loop.

**This is not an agent.**  It has no tools, no ReAct loop and no ability to
propose new candidates.  It is a single evaluation call that receives the
refined candidates together with the Critic's feedback, the Anti-Analogy
counterexamples and the evidence collected during the run, and returns a
ranking.

If the ranking cannot be parsed, a deterministic fallback ranks the candidates
from the structured Critic/Anti-Analogy scores, so a run always produces a
result.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from hal.json_utils import get_float, get_list, get_str, parse_json_object
from hal.providers.base import LLMProvider
from hal.schemas import (
    AntiAnalogyReport,
    CandidateAnalogy,
    Critique,
    Evidence,
    HistoricalEvent,
    JudgeRanking,
    RankedCandidate,
)

from . import prompts
from .generate_search_agent import format_candidates, join_sections

MAX_EVIDENCE_LINES = 25


class FinalJudge:
    """Ranks the refined candidate analogies."""

    name = "final_judge"

    def __init__(self, llm: LLMProvider, verbose: bool = False):
        self.llm = llm
        self.verbose = verbose

    def rank(self, event: HistoricalEvent, candidates: Sequence[CandidateAnalogy],
             critiques: Sequence[Critique],
             anti_analogies: Sequence[AntiAnalogyReport],
             evidence: Optional[Sequence[Evidence]] = None) -> JudgeRanking:
        if not candidates:
            return JudgeRanking(ranking=[], notes="no candidates to rank")

        prompt = prompts.FINAL_JUDGE.format(
            event=event.text(),
            candidates=format_candidates(candidates),
            critiques=join_sections([c.as_feedback() for c in critiques],
                                    "(no critiques)"),
            anti_analogies=join_sections([a.as_feedback() for a in anti_analogies],
                                         "(no counterexamples)"),
            evidence=_format_evidence(evidence or []),
        )
        raw = self.llm.generate(prompt, json_output=True) or ""
        ranking = self._parse(raw, candidates)
        if not ranking.ranking:
            ranking = heuristic_ranking(candidates, critiques, anti_analogies)
            ranking.notes = (
                "Fallback ranking computed from Critic and Anti-Analogy scores "
                "(the judge's output could not be parsed)."
            )
        return ranking

    # -- internals --------------------------------------------------------
    def _parse(self, raw: str, candidates: Sequence[CandidateAnalogy]) -> JudgeRanking:
        data = parse_json_object(raw)
        if not data:
            return JudgeRanking()
        rows = data.get("ranking")
        if not isinstance(rows, list):
            return JudgeRanking()

        known = {c.name.lower(): c.name for c in candidates}
        ranked: List[RankedCandidate] = []
        seen = set()
        for item in rows:
            if isinstance(item, str):
                item = {"event_name": item}
            if not isinstance(item, dict):
                continue
            name = get_str(item, "event_name", "candidate", "name", "event")
            if not name:
                continue
            name = known.get(name.lower(), name)
            if name.lower() in seen:
                continue
            seen.add(name.lower())
            ranked.append(RankedCandidate(
                rank=int(get_float(item, "rank", default=len(ranked) + 1)),
                event_name=name,
                reason=get_str(item, "reason", "justification", "explanation"),
                weaknesses=get_list(item, "weaknesses", "weakness", "limitations"),
                score=get_float(item, "score", default=0.0),
            ))
        # Append candidates the judge forgot, so the ranking stays complete.
        for candidate in candidates:
            if candidate.name.lower() not in seen:
                ranked.append(RankedCandidate(
                    rank=len(ranked) + 1, event_name=candidate.name,
                    reason="not ranked by the judge", score=0.0,
                ))
        for position, row in enumerate(ranked, start=1):
            row.rank = position
        return JudgeRanking(ranking=ranked, notes=get_str(data, "notes"))


def heuristic_ranking(candidates: Sequence[CandidateAnalogy],
                      critiques: Sequence[Critique],
                      anti_analogies: Sequence[AntiAnalogyReport]) -> JudgeRanking:
    """Deterministic ranking from the structured feedback.

    Used as a fallback and available for ablations ("how much does the LLM judge
    add over the structured scores?").  Score =
    ``critic_overall (1-4) * robustness (0-1)``, penalised for unverified
    candidates and surface-level analogies.
    """
    critique_by_name = {c.candidate.lower(): c for c in critiques}
    anti_by_name = {a.candidate.lower(): a for a in anti_analogies}

    scored = []
    for candidate in candidates:
        critique = critique_by_name.get(candidate.name.lower())
        anti = anti_by_name.get(candidate.name.lower())
        base = critique.overall_score if critique and critique.overall_score else 2.0
        robustness = anti.robustness if anti else 0.5
        score = base * (0.5 + 0.5 * robustness)
        if candidate.event.verified is False:
            score *= 0.5
        if critique and critique.surface_level:
            score *= 0.7
        if anti and anti.verdict == "undermined":
            score *= 0.6
        weaknesses: List[str] = []
        if critique:
            weaknesses.extend(critique.important_differences[:2])
            weaknesses.extend(critique.factual_problems[:1])
        if anti:
            weaknesses.extend(
                f"counterexample: {c.event_name}" for c in anti.counterexamples[:2]
            )
        reason_parts = []
        if critique and critique.structural_correspondence:
            reason_parts.append(critique.structural_correspondence)
        if anti:
            reason_parts.append(f"anti-analogy verdict: {anti.verdict}")
        scored.append((score, candidate.name, weaknesses, " ".join(reason_parts)))

    scored.sort(key=lambda row: row[0], reverse=True)
    ranking = [
        RankedCandidate(rank=index, event_name=name, reason=reason,
                        weaknesses=weaknesses, score=round(score, 3))
        for index, (score, name, weaknesses, reason) in enumerate(scored, start=1)
    ]
    return JudgeRanking(ranking=ranking, notes="heuristic ranking")


def rank_candidates(event: HistoricalEvent, candidates: Sequence[CandidateAnalogy],
                    critiques: Sequence[Critique],
                    anti_analogies: Sequence[AntiAnalogyReport],
                    llm: Optional[LLMProvider] = None,
                    evidence: Optional[Sequence[Evidence]] = None) -> JudgeRanking:
    """Functional entry point for the Final Judge."""
    if llm is None:
        return heuristic_ranking(candidates, critiques, anti_analogies)
    return FinalJudge(llm).rank(event, candidates, critiques, anti_analogies, evidence)


def _format_evidence(evidence: Sequence[Evidence]) -> str:
    if not evidence:
        return "(no external evidence was retrieved)"
    lines = []
    seen = set()
    for item in evidence:
        key = (item.title, item.query)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- {item.title} (query: {item.query}): {item.snippet[:200]}")
        if len(lines) >= MAX_EVIDENCE_LINES:
            break
    return "\n".join(lines)
