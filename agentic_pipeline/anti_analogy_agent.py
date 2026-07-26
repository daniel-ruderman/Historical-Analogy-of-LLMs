"""Agent 3 -- the Anti-Analogy agent.

Actively tries to break each candidate: it searches for historically similar
cases that produced a different or opposite outcome, for counterexamples, and
for evidence that the analogy would mislead.  This is the component that stops
the system from simply accumulating support for its first idea.

Its findings (:class:`~hal.schemas.AntiAnalogyReport`) go back to the
Generate/Search agent, which can weaken, modify or replace a candidate, and to
the Final Judge, which weighs robustness when ranking.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from hal.json_utils import get_float, get_list, get_str
from hal.providers.base import LLMProvider, SearchProvider
from hal.schemas import AntiAnalogyReport, CandidateAnalogy, CounterExample, HistoricalEvent

from . import prompts
from .critic_agent import describe_candidate
from .react import ReActAgent, ReActOutcome

VERDICTS = ("holds", "weakened", "undermined")


class AntiAnalogyAgent:
    """Counterexample search for candidate analogies."""

    name = "anti_analogy"

    def __init__(self, llm: LLMProvider, search: Optional[SearchProvider] = None,
                 max_steps: int = 3, top_k: int = 4, verify: bool = True,
                 verbose: bool = False):
        self.llm = llm
        self.search = search
        self.verify = verify
        self.verbose = verbose
        self._react = ReActAgent(llm, search, max_steps=max_steps, top_k=top_k,
                                 verbose=verbose, name=self.name)

    def investigate_one(self, event: HistoricalEvent, candidate: CandidateAnalogy
                        ) -> Tuple[AntiAnalogyReport, ReActOutcome]:
        prompt = prompts.ANTI_ANALOGY.format(
            event=event.text(), candidate=describe_candidate(candidate)
        )
        outcome = self._react.run(prompt)
        return self._parse(candidate, outcome), outcome

    def investigate_all(self, event: HistoricalEvent,
                        candidates: Sequence[CandidateAnalogy]) -> List[AntiAnalogyReport]:
        reports = []
        for candidate in candidates:
            if self.verbose:
                print(f"    [anti-analogy] {candidate.name}")
            report, _ = self.investigate_one(event, candidate)
            reports.append(report)
        return reports

    # -- internals --------------------------------------------------------
    def _parse(self, candidate: CandidateAnalogy, outcome: ReActOutcome
               ) -> AntiAnalogyReport:
        report = AntiAnalogyReport(candidate=candidate.name,
                                   evidence=list(outcome.evidence))
        result = outcome.result if isinstance(outcome.result, dict) else None
        if result is None:
            report.summary = (
                "The Anti-Analogy agent could not produce a usable report "
                f"({outcome.error or 'unparseable output'})."
            )
            report.verdict = "holds"
            report.robustness = 0.5
            return report

        raw_counters = result.get("counterexamples")
        if isinstance(raw_counters, list):
            for item in raw_counters:
                if isinstance(item, str):
                    item = {"event_name": item}
                if not isinstance(item, dict):
                    continue
                name = get_str(item, "event_name", "name", "event", "title")
                if not name:
                    continue
                counter = CounterExample(
                    event_name=name,
                    description=get_str(item, "description", "summary"),
                    divergence=get_str(item, "divergence", "difference", "outcome"),
                )
                if self.verify and self.search is not None:
                    page = self.search.get_page(name)
                    if page is not None:
                        counter.url = page.url
                        if not counter.description:
                            counter.description = page.snippet
                report.counterexamples.append(counter)

        report.failure_modes = get_list(result, "failure_modes", "risks")
        report.summary = get_str(result, "summary", "assessment")
        report.robustness = min(max(get_float(result, "robustness", default=0.5), 0.0), 1.0)
        verdict = get_str(result, "verdict").lower()
        if verdict not in VERDICTS:
            # Derive a verdict from the evidence when the model omitted it.
            if report.robustness < 0.35 or len(report.counterexamples) >= 3:
                verdict = "undermined"
            elif report.counterexamples:
                verdict = "weakened"
            else:
                verdict = "holds"
        report.verdict = verdict
        return report
