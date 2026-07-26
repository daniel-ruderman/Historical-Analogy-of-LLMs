"""Our agentic historical-analogy pipeline.

    Input event / analogy prompt
              |
              v
      Generate/Search agent          -> 5-10 candidate analogies
              |
    +---------+-------------------------------+
    |  ITERATIVE REFINEMENT LOOP (1-3 rounds) |
    |                                         |
    |   Critic agent        evaluates each    |
    |   Anti-Analogy agent  challenges each   |
    |            |                            |
    |            v                            |
    |   Generate/Search agent revises the set |
    +---------+-------------------------------+
              |
              v  refined candidates + critiques + counterexamples + evidence
        Final Judge          (ranking component -- NOT an agent)
              |
              v
        Final Summarizer     (explains the winning analogy)

Configuration: ``REFINEMENT_ROUNDS`` (default 2), ``MAX_CANDIDATES``,
``REACT_MAX_STEPS``, ``SEARCH_TOP_K`` and the per-role model variables.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from hal.config import Settings, get_settings, load_settings, set_settings
from hal.providers.base import LLMProvider, SearchProvider
from hal.schemas import (
    AntiAnalogyReport,
    CandidateAnalogy,
    Critique,
    Evidence,
    FinalAnalogyResult,
    HistoricalEvent,
    RefinementRound,
)

from .anti_analogy_agent import AntiAnalogyAgent
from .critic_agent import CriticAgent
from .final_judge import FinalJudge
from .final_summarizer import FinalSummarizer
from .generate_search_agent import GenerateSearchAgent


@dataclass
class PipelineConfig:
    """Knobs of one pipeline run (defaults come from the environment)."""

    refinement_rounds: int = 2
    max_candidates: int = 8
    react_max_steps: int = 4
    search_top_k: int = 4
    critique_top_n: Optional[int] = None   # only critique the N most confident
    verbose: bool = False

    @classmethod
    def from_settings(cls, settings: Optional[Settings] = None, **overrides
                      ) -> "PipelineConfig":
        settings = settings or get_settings()
        config = cls(
            refinement_rounds=settings.refinement_rounds,
            max_candidates=settings.max_candidates,
            react_max_steps=settings.react_max_steps,
            search_top_k=settings.search_top_k,
        )
        for key, value in overrides.items():
            if value is not None and hasattr(config, key):
                setattr(config, key, value)
        return config


class AgenticAnalogyPipeline:
    """The full pipeline. Components are injectable, which keeps it testable."""

    def __init__(self,
                 generate_agent: Optional[GenerateSearchAgent] = None,
                 critic_agent: Optional[CriticAgent] = None,
                 anti_analogy_agent: Optional[AntiAnalogyAgent] = None,
                 final_judge: Optional[FinalJudge] = None,
                 final_summarizer: Optional[FinalSummarizer] = None,
                 config: Optional[PipelineConfig] = None,
                 search: Optional[SearchProvider] = None,
                 settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.config = config or PipelineConfig.from_settings(self.settings)
        self.search = search
        self.generate_agent = generate_agent
        self.critic_agent = critic_agent
        self.anti_analogy_agent = anti_analogy_agent
        self.final_judge = final_judge
        self.final_summarizer = final_summarizer
        if any(component is None for component in (
                generate_agent, critic_agent, anti_analogy_agent,
                final_judge, final_summarizer)):
            self._build_missing()

    # -- construction -----------------------------------------------------
    @classmethod
    def build(cls, *, settings: Optional[Settings] = None,
              search: Optional[SearchProvider] = None,
              **config_overrides) -> "AgenticAnalogyPipeline":
        settings = settings or get_settings()
        config = PipelineConfig.from_settings(settings, **config_overrides)
        return cls(config=config, search=search, settings=settings)

    def _build_missing(self) -> None:
        from hal.providers.factory import get_llm, get_search_provider

        settings = self.settings
        if self.search is None:
            self.search = get_search_provider(settings=settings)
        config = self.config
        if self.generate_agent is None:
            self.generate_agent = GenerateSearchAgent(
                get_llm(role="generator", settings=settings), self.search,
                max_candidates=config.max_candidates,
                min_candidates=settings.min_candidates,
                max_steps=config.react_max_steps, top_k=config.search_top_k,
                verbose=config.verbose,
            )
        if self.critic_agent is None:
            self.critic_agent = CriticAgent(
                get_llm(role="critic", settings=settings), self.search,
                max_steps=max(1, config.react_max_steps - 1), top_k=config.search_top_k,
                verbose=config.verbose,
            )
        if self.anti_analogy_agent is None:
            self.anti_analogy_agent = AntiAnalogyAgent(
                get_llm(role="anti_analogy", settings=settings), self.search,
                max_steps=max(1, config.react_max_steps - 1), top_k=config.search_top_k,
                verbose=config.verbose,
            )
        if self.final_judge is None:
            self.final_judge = FinalJudge(get_llm(role="judge", settings=settings),
                                          verbose=config.verbose)
        if self.final_summarizer is None:
            self.final_summarizer = FinalSummarizer(
                get_llm(role="summarizer", settings=settings), verbose=config.verbose
            )

    # -- the pipeline -----------------------------------------------------
    def run(self, event) -> FinalAnalogyResult:
        """Run the full pipeline for one input event."""
        if isinstance(event, dict):
            event = HistoricalEvent.from_dataset_row(event)
        config = self.config
        result = FinalAnalogyResult(input_event=event)
        evidence: List[Evidence] = []

        # --- Generate/Search: initial candidate set ----------------------
        self._log(f"[generate] proposing up to {config.max_candidates} candidates")
        candidates, outcome = self.generate_agent.propose(event)
        evidence.extend(outcome.evidence)
        if outcome.error:
            result.errors.append(f"generate(initial): {outcome.error}")
        if not candidates:
            result.errors.append("no candidates were produced")
            return result
        result.initial_candidates = list(candidates)
        self._log(f"[generate] {len(candidates)} candidates: "
                  f"{', '.join(c.name for c in candidates)}")

        # --- iterative refinement loop -----------------------------------
        critiques: List[Critique] = []
        anti_analogies: List[AntiAnalogyReport] = []
        reviewed_names: List[str] = []
        for round_index in range(1, max(0, config.refinement_rounds) + 1):
            self._log(f"[round {round_index}] critic + anti-analogy on "
                      f"{len(candidates)} candidates")
            reviewed = self._select_for_review(candidates)
            reviewed_names = [c.name.lower() for c in reviewed]

            critiques = self.critic_agent.critique_all(event, reviewed)
            anti_analogies = self.anti_analogy_agent.investigate_all(event, reviewed)
            for critique in critiques:
                evidence.extend(critique.evidence)
            for report in anti_analogies:
                evidence.extend(report.evidence)

            candidates, revisions, outcome = self.generate_agent.revise(
                event, candidates, critiques, anti_analogies, round_index
            )
            evidence.extend(outcome.evidence)
            if outcome.error:
                result.errors.append(f"generate(round {round_index}): {outcome.error}")

            result.rounds.append(RefinementRound(
                index=round_index,
                critiques=critiques,
                anti_analogies=anti_analogies,
                revisions=revisions,
                candidates_after=list(candidates),
            ))
            self._log(f"[round {round_index}] revised set: "
                      f"{', '.join(c.name for c in candidates)}")

        # The judge must see feedback about the *final* candidate set.  Review
        # again only when the last round actually changed it -- re-reviewing an
        # unchanged set would just burn API quota.
        reviewed = self._select_for_review(candidates)
        if [c.name.lower() for c in reviewed] != reviewed_names:
            self._log("[review] final candidate set changed -- reviewing it")
            critiques = self.critic_agent.critique_all(event, reviewed)
            anti_analogies = self.anti_analogy_agent.investigate_all(event, reviewed)
            for critique in critiques:
                evidence.extend(critique.evidence)
            for report in anti_analogies:
                evidence.extend(report.evidence)

        result.final_candidates = list(candidates)

        # --- Final Judge (ranking component, not an agent) ---------------
        self._log("[judge] ranking candidates")
        result.ranking = self.final_judge.rank(event, candidates, critiques,
                                               anti_analogies, evidence)
        winner_row = result.ranking.winner
        if winner_row is None:
            result.errors.append("the Final Judge returned an empty ranking")
            return result
        result.analogy_event = winner_row.event_name
        result.winning_candidate = _find_candidate(candidates, winner_row.event_name)
        self._log(f"[judge] winner: {result.analogy_event}")

        # --- Final Summarizer --------------------------------------------
        if result.winning_candidate is not None:
            critique = _find_by_candidate(critiques, result.analogy_event)
            anti = _find_by_candidate(anti_analogies, result.analogy_event)
            self._log("[summarizer] explaining the winning analogy")
            self.final_summarizer.apply(result, critique, anti)
        return result

    # -- helpers ----------------------------------------------------------
    def _select_for_review(self, candidates: Sequence[CandidateAnalogy]
                           ) -> List[CandidateAnalogy]:
        """Optionally review only the top-N candidates (saves API calls)."""
        top_n = self.config.critique_top_n
        if not top_n or top_n >= len(candidates):
            return list(candidates)
        ordered = sorted(candidates, key=lambda c: c.confidence, reverse=True)
        return ordered[:top_n]

    def _log(self, message: str) -> None:
        if self.config.verbose:
            print(f"  {message}")


def _find_candidate(candidates: Sequence[CandidateAnalogy], name: str
                    ) -> Optional[CandidateAnalogy]:
    for candidate in candidates:
        if candidate.name.lower() == (name or "").lower():
            return candidate
    return candidates[0] if candidates else None


def _find_by_candidate(items: Sequence[Any], name: str) -> Optional[Any]:
    for item in items:
        if getattr(item, "candidate", "").lower() == (name or "").lower():
            return item
    return None


def run_pipeline(event, **kwargs) -> FinalAnalogyResult:
    """Convenience wrapper: build a pipeline from the environment and run it."""
    return AgenticAnalogyPipeline.build(**kwargs).run(event)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main() -> None:
    from hal.io_utils import DATASETS, configure_stdout, read_jsonl, write_jsonl

    configure_stdout()

    parser = argparse.ArgumentParser(description="Our agentic historical-analogy pipeline")
    parser.add_argument("--dataset", default="popular",
                        help="'popular', 'general', or a path to a .jsonl file")
    parser.add_argument("--index", type=int, default=None,
                        help="run a single event by index (default: the whole set)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None,
                        help="refinement rounds (default: REFINEMENT_ROUNDS)")
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--react-steps", type=int, default=None,
                        help="max tool calls per agent invocation")
    parser.add_argument("--critique-top-n", type=int, default=None,
                        help="only review the N most confident candidates")
    parser.add_argument("--model", type=str, default=None,
                        help="model for every role (overrides LLM_MODEL)")
    parser.add_argument("--provider", type=str, default=None)
    parser.add_argument("--search", type=str, default=None,
                        help="search provider: wikipedia | none")
    parser.add_argument("--output", type=str, default="agentic_output.jsonl")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    overrides = {}
    if args.provider:
        overrides["llm_provider"] = args.provider
    if args.model:
        overrides["llm_model"] = args.model
        overrides["role_models"] = {}
    if args.search:
        overrides["search_provider"] = args.search
    if overrides:
        set_settings(load_settings(**overrides))
    settings = get_settings()

    path = DATASETS.get(args.dataset, args.dataset)
    testset = read_jsonl(path)
    if args.index is not None:
        testset = [testset[args.index]]
    elif args.limit:
        testset = testset[: args.limit]

    pipeline = AgenticAnalogyPipeline.build(
        settings=settings,
        refinement_rounds=args.rounds,
        max_candidates=args.max_candidates,
        react_max_steps=args.react_steps,
        critique_top_n=args.critique_top_n,
        verbose=not args.quiet,
    )
    print(f"configuration:\n  {settings.describe()}")
    print(f"  rounds={pipeline.config.refinement_rounds} "
          f"max_candidates={pipeline.config.max_candidates}\n")

    rows: List[Dict[str, Any]] = []
    for position, event in enumerate(testset, start=1):
        print(f"[{position}/{len(testset)}] {event.get('event_name', '?')}")
        result = pipeline.run(event)
        print(f"  => analogy: {result.analogy_event}\n")
        rows.append(result.to_output_row())
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
