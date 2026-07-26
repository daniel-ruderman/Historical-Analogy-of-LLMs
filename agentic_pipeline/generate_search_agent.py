"""Agent 1 -- the Generate/Search agent.

Proposes 5-10 candidate analogies for the input event and, in later rounds,
revises the set using the Critic's and the Anti-Analogy agent's feedback: it may
keep, revise, replace or drop candidates, and search for better alternatives.

It is a ReAct-style agent: it can query the search backend (Wikipedia by
default) before committing to a candidate.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from hal.json_utils import get_float, get_list, get_str
from hal.providers.base import LLMProvider, SearchProvider
from hal.schemas import (
    AntiAnalogyReport,
    CandidateAnalogy,
    CandidateRevision,
    Critique,
    HistoricalEvent,
)

from . import prompts
from .react import ReActAgent, ReActOutcome

VALID_ACTIONS = ("keep", "revise", "replace", "add", "drop")


class GenerateSearchAgent:
    """The candidate-proposing / candidate-revising agent."""

    name = "generate_search"

    def __init__(self, llm: LLMProvider, search: Optional[SearchProvider] = None,
                 max_candidates: int = 8, min_candidates: int = 5,
                 max_steps: int = 4, top_k: int = 4, verify: bool = True,
                 verbose: bool = False):
        self.llm = llm
        self.search = search
        self.max_candidates = max(1, max_candidates)
        self.min_candidates = min(min_candidates, self.max_candidates)
        self.verify = verify
        self.verbose = verbose
        self._react = ReActAgent(llm, search, max_steps=max_steps, top_k=top_k,
                                 verbose=verbose, name=self.name)

    # -- public API -------------------------------------------------------
    def propose(self, event: HistoricalEvent) -> Tuple[List[CandidateAnalogy], ReActOutcome]:
        """Initial candidate set."""
        prompt = prompts.GENERATE_INITIAL.format(
            event=event.text(), n_candidates=self.max_candidates
        )
        outcome = self._react.run(prompt)
        candidates, _ = self._parse_candidates(outcome, round_index=0)
        for candidate in candidates:
            candidate.status = "new"
        return self._verify(candidates), outcome

    def revise(self, event: HistoricalEvent, candidates: Sequence[CandidateAnalogy],
               critiques: Sequence[Critique], anti_analogies: Sequence[AntiAnalogyReport],
               round_index: int
               ) -> Tuple[List[CandidateAnalogy], List[CandidateRevision], ReActOutcome]:
        """Revised candidate set for refinement round ``round_index``."""
        prompt = prompts.GENERATE_REVISE.format(
            event=event.text(),
            candidates=_format_candidates(candidates),
            critiques=_join([c.as_feedback() for c in critiques],
                            "(no critiques available)"),
            anti_analogies=_join([a.as_feedback() for a in anti_analogies],
                                 "(no counterexamples available)"),
            n_candidates=self.max_candidates,
        )
        outcome = self._react.run(prompt)
        revised, meta = self._parse_candidates(outcome, round_index=round_index)
        if not revised:
            # Keep the previous set rather than losing the round's work.
            return list(candidates), [], outcome

        previous_names = {c.name.lower() for c in candidates}
        revisions: List[CandidateRevision] = []
        for candidate in revised:
            info = meta.get(candidate.name.lower(), {})
            action = candidate.status if candidate.status in VALID_ACTIONS else None
            if action is None:
                action = "keep" if candidate.name.lower() in previous_names else "add"
            candidate.status = action
            revisions.append(CandidateRevision(
                round_index=round_index,
                action=action,
                candidate=candidate.name,
                previous_candidate=info.get("previous_candidate", ""),
                reason=info.get("change_reason") or (
                    candidate.rationale if action != "keep" else ""
                ),
            ))
        # Record candidates that disappeared from the set.
        new_names = {c.name.lower() for c in revised}
        for old in candidates:
            if old.name.lower() not in new_names:
                revisions.append(CandidateRevision(
                    round_index=round_index, action="drop", candidate=old.name,
                    reason="not retained in the revised candidate set",
                ))
        return self._verify(revised), revisions, outcome

    # -- internals --------------------------------------------------------
    def _parse_candidates(self, outcome: ReActOutcome, round_index: int
                          ) -> Tuple[List[CandidateAnalogy], Dict[str, Dict[str, str]]]:
        """Validate the model's JSON into candidates + per-candidate revision info."""
        if not outcome.ok or not isinstance(outcome.result, dict):
            return [], {}
        raw = outcome.result.get("candidates")
        if not isinstance(raw, list):
            # Some models return the list directly under another key.
            for value in outcome.result.values():
                if isinstance(value, list) and value and isinstance(value[0], dict):
                    raw = value
                    break
        if not isinstance(raw, list):
            return [], {}

        candidates: List[CandidateAnalogy] = []
        meta: Dict[str, Dict[str, str]] = {}
        seen = set()
        for item in raw:
            if isinstance(item, str):
                item = {"event_name": item}
            if not isinstance(item, dict):
                continue
            name = get_str(item, "event_name", "name", "event", "title")
            if not name or name.lower() in seen:
                continue
            seen.add(name.lower())
            mapping = item.get("mapping")
            mapping = {str(k): str(v) for k, v in mapping.items()} \
                if isinstance(mapping, dict) else {}
            action = get_str(item, "action", "status").lower()
            candidate = CandidateAnalogy(
                event=HistoricalEvent(
                    name=name,
                    description=get_str(item, "description", "event_intro", "summary"),
                    source="agent",
                ),
                rationale=get_str(item, "rationale", "change_reason", "reason"),
                mapping=mapping,
                confidence=min(max(get_float(item, "confidence", default=0.5), 0.0), 1.0),
                status=action if action in VALID_ACTIONS else "new",
                origin_round=round_index,
                evidence=list(outcome.evidence),
            )
            candidates.append(candidate)
            meta[name.lower()] = {
                "previous_candidate": get_str(item, "previous_candidate", "replaces"),
                "change_reason": get_str(item, "change_reason", "reason"),
            }
            if len(candidates) >= self.max_candidates:
                break
        return candidates, meta

    def _verify(self, candidates: List[CandidateAnalogy]) -> List[CandidateAnalogy]:
        """Confirm each candidate exists in the knowledge base (hallucination filter).

        Like the paper, unverifiable events are flagged; unlike the paper's
        two-stage generation we keep them (marked ``verified=False``) so that the
        Critic can comment on them and the Final Judge can penalise them.
        """
        if not self.verify or self.search is None:
            return candidates
        for candidate in candidates:
            page = self.search.get_page(candidate.name)
            if page is None:
                resolve = getattr(self.search, "resolve", None)
                page = resolve(candidate.name) if resolve else None
            if page is None:
                candidate.event.verified = False
                if self.verbose:
                    print(f"      [{self.name}] unverified candidate: {candidate.name}")
                continue
            candidate.event.verified = True
            candidate.event.url = page.url
            if not candidate.event.description:
                candidate.event.description = page.content[:1200]
        return candidates


def _format_candidates(candidates: Sequence[CandidateAnalogy]) -> str:
    if not candidates:
        return "(no candidates yet)"
    lines = []
    for index, candidate in enumerate(candidates, start=1):
        verified = {True: "verified", False: "NOT VERIFIED", None: "unchecked"}[
            candidate.event.verified
        ]
        lines.append(
            f"{index}. {candidate.name} [{verified}]\n"
            f"   description: {candidate.event.description[:400]}\n"
            f"   rationale: {candidate.rationale}"
        )
    return "\n".join(lines)


def _join(chunks: Sequence[str], empty: str) -> str:
    chunks = [c for c in chunks if c and c.strip()]
    return "\n\n".join(chunks) if chunks else empty


format_candidates = _format_candidates
join_sections = _join
