"""Our agentic pipeline: agents, refinement loop, Final Judge, Final Summarizer.

Everything runs against a scripted mock LLM -- no API calls.
"""

from __future__ import annotations

import json

import pytest

from agentic_pipeline import (
    AgenticAnalogyPipeline,
    AntiAnalogyAgent,
    CriticAgent,
    FinalJudge,
    FinalSummarizer,
    GenerateSearchAgent,
    PipelineConfig,
    heuristic_ranking,
    rank_candidates,
)
from agentic_pipeline.react import ReActAgent
from hal.providers.mock import MockLLMProvider
from hal.schemas import (
    AntiAnalogyReport,
    CandidateAnalogy,
    CounterExample,
    Critique,
    HistoricalEvent,
)

EVENT = HistoricalEvent(name="Arab Spring",
                        description="A wave of uprisings across the Arab world.")

CANDIDATES_JSON = {
    "candidates": [
        {"event_name": "Revolutions of 1848", "description": "A European wave.",
         "rationale": "A cascade of uprisings across neighbouring states.",
         "mapping": {"topic": "revolutionary wave"}, "confidence": 0.8},
        {"event_name": "French Revolution", "description": "France, 1789.",
         "rationale": "Popular uprising against an entrenched regime.",
         "confidence": 0.6},
    ]
}
CRITIQUE_JSON = {
    "dimension_scores": {"topic": 4, "background": 3, "process": 3, "result": 2},
    "structural_correspondence": "Both are cascading uprisings.",
    "important_differences": ["Different communication technology"],
    "weak_assumptions": ["Assumes comparable state capacity"],
    "factual_problems": [],
    "surface_level": False,
    "overall_score": 3,
    "recommendation": "keep",
    "summary": "Solid structural match, weakest on outcomes.",
}
ANTI_JSON = {
    "counterexamples": [
        {"event_name": "Revolutions of 1848", "description": "Reversed within years.",
         "divergence": "Similar uprisings ended in restoration."}
    ],
    "failure_modes": ["Expecting durable democratisation"],
    "robustness": 0.4,
    "verdict": "weakened",
    "summary": "The pattern frequently reverses.",
}
JUDGE_JSON = {
    "ranking": [
        {"rank": 1, "event_name": "Revolutions of 1848", "score": 8,
         "reason": "Closest cascade structure.", "weaknesses": ["Outcomes differ"]},
        {"rank": 2, "event_name": "French Revolution", "score": 5,
         "reason": "Single country.", "weaknesses": ["Scale"]},
    ],
    "notes": "Close call.",
}
SUMMARY_JSON = {
    "explanation": "The comparison shows how regional waves spread and stall.",
    "similarities": ["Cascade across neighbouring states"],
    "differences": ["Different media environment"],
    "counterexamples": ["1848: gains were reversed"],
    "limitations": ["Do not infer inevitable restoration"],
    "why_ranked_first": "It matches the cascade structure best.",
}


def finish(result: dict) -> str:
    return json.dumps({"thought": "ready", "action": "finish", "result": result})


def scripted_llm(**overrides) -> MockLLMProvider:
    """A mock LLM that answers according to which component is prompting it."""
    payloads = {
        "generate": finish(CANDIDATES_JSON),
        "critic": finish(CRITIQUE_JSON),
        "anti": finish(ANTI_JSON),
        "judge": json.dumps(JUDGE_JSON),
        "summarizer": json.dumps(SUMMARY_JSON),
    }
    payloads.update(overrides)

    # Dispatch on the role sentence that opens each prompt -- prompts also
    # *mention* the other roles, so a looser check would misroute.
    markers = [
        ("You are the Generate/Search agent", "generate"),
        ("You are the Critic agent", "critic"),
        ("You are the Anti-Analogy agent", "anti"),
        ("You are the Final Judge", "judge"),
        ("You are the Final Summarizer", "summarizer"),
    ]

    def responder(prompt: str) -> str:
        for marker, key in markers:
            if marker in prompt:
                return payloads[key]
        return "{}"

    return MockLLMProvider(responses=responder)


def build_pipeline(llm=None, search=None, **config):
    llm = llm or scripted_llm()
    settings_config = PipelineConfig(refinement_rounds=1, max_candidates=3,
                                     react_max_steps=2, **config)
    return AgenticAnalogyPipeline(
        generate_agent=GenerateSearchAgent(llm, search, max_candidates=3, max_steps=2),
        critic_agent=CriticAgent(llm, search, max_steps=1),
        anti_analogy_agent=AntiAnalogyAgent(llm, search, max_steps=1),
        final_judge=FinalJudge(llm),
        final_summarizer=FinalSummarizer(llm),
        config=settings_config,
        search=search,
    )


# --- ReAct loop -----------------------------------------------------------
def test_react_agent_searches_then_finishes(mock_search):
    llm = MockLLMProvider(responses=[
        json.dumps({"thought": "need evidence", "action": "search",
                    "query": "revolution wave europe"}),
        finish({"answer": "done"}),
    ])
    agent = ReActAgent(llm, mock_search, max_steps=2)
    outcome = agent.run("task")
    assert outcome.ok
    assert outcome.result == {"answer": "done"}
    assert [s.action for s in outcome.steps] == ["search", "finish"]
    assert outcome.steps[0].query == "revolution wave europe"
    assert mock_search.queries == ["revolution wave europe"]
    assert outcome.evidence  # search results are recorded as evidence


def test_react_agent_respects_the_tool_budget(mock_search):
    llm = MockLLMProvider(responses=[json.dumps(
        {"thought": "search again", "action": "search", "query": "revolution"})])
    outcome = ReActAgent(llm, mock_search, max_steps=2).run("task")
    assert not outcome.ok
    assert "budget" in outcome.error
    assert len([s for s in outcome.steps if s.action == "search"]) == 2


def test_react_agent_reports_unparseable_output(mock_search):
    llm = MockLLMProvider(responses=["I would rather not answer in JSON."])
    outcome = ReActAgent(llm, mock_search, max_steps=1).run("task")
    assert not outcome.ok
    assert outcome.result is None
    assert outcome.error


def test_react_logs_contain_no_long_reasoning(mock_search):
    llm = MockLLMProvider(responses=[
        json.dumps({"thought": "x" * 5000, "action": "search", "query": "q"}),
        finish({"ok": True}),
    ])
    outcome = ReActAgent(llm, mock_search, max_steps=2).run("task")
    assert all(len(step.rationale) <= 240 for step in outcome.steps)


# --- Generate/Search agent ------------------------------------------------
def test_generate_agent_proposes_and_verifies_candidates(mock_search):
    agent = GenerateSearchAgent(scripted_llm(), mock_search, max_candidates=3)
    candidates, outcome = agent.propose(EVENT)
    assert [c.name for c in candidates] == ["Revolutions of 1848", "French Revolution"]
    assert all(c.event.verified for c in candidates)
    assert candidates[0].confidence == 0.8


def test_generate_agent_flags_unverifiable_candidates(mock_search):
    llm = scripted_llm(generate=finish({"candidates": [
        {"event_name": "Battle of Nowhere", "description": "invented"}]}))
    agent = GenerateSearchAgent(llm, mock_search, max_candidates=3)
    candidates, _ = agent.propose(EVENT)
    assert candidates[0].event.verified is False


def test_generate_agent_caps_the_candidate_count(mock_search):
    many = {"candidates": [{"event_name": f"Event {i}"} for i in range(12)]}
    agent = GenerateSearchAgent(scripted_llm(generate=finish(many)), mock_search,
                                max_candidates=5, verify=False)
    candidates, _ = agent.propose(EVENT)
    assert len(candidates) == 5


def test_generate_agent_revision_records_actions(mock_search):
    revised = {"candidates": [
        {"event_name": "Revolutions of 1848", "action": "keep",
         "change_reason": "survived criticism"},
        {"event_name": "Prague Spring", "action": "replace",
         "previous_candidate": "French Revolution",
         "change_reason": "counterexample undermined the previous candidate"},
    ]}
    llm = scripted_llm(generate=finish(revised))
    agent = GenerateSearchAgent(llm, mock_search, max_candidates=3, verify=False)
    previous = [CandidateAnalogy(event=HistoricalEvent(name="Revolutions of 1848")),
                CandidateAnalogy(event=HistoricalEvent(name="French Revolution"))]
    candidates, revisions, _ = agent.revise(EVENT, previous, [], [], round_index=1)
    actions = {r.candidate: r.action for r in revisions}
    assert actions["Revolutions of 1848"] == "keep"
    assert actions["Prague Spring"] == "replace"
    assert actions["French Revolution"] == "drop"   # disappeared from the set
    replacement = next(r for r in revisions if r.candidate == "Prague Spring")
    assert replacement.previous_candidate == "French Revolution"


def test_generate_agent_keeps_previous_set_on_bad_output(mock_search):
    llm = scripted_llm(generate="not json at all")
    agent = GenerateSearchAgent(llm, mock_search, max_candidates=3, verify=False)
    previous = [CandidateAnalogy(event=HistoricalEvent(name="Cold War"))]
    candidates, revisions, outcome = agent.revise(EVENT, previous, [], [], 1)
    assert [c.name for c in candidates] == ["Cold War"]
    assert revisions == []
    assert outcome.error


# --- Critic agent ---------------------------------------------------------
def test_critic_returns_structured_feedback(mock_search):
    critic = CriticAgent(scripted_llm(), mock_search, max_steps=1)
    candidate = CandidateAnalogy(event=HistoricalEvent(name="Revolutions of 1848"))
    critique, _ = critic.critique_one(EVENT, candidate)
    assert critique.candidate == "Revolutions of 1848"
    assert critique.dimension_scores == {"topic": 4.0, "background": 3.0,
                                         "process": 3.0, "result": 2.0}
    assert critique.overall_score == 3.0
    assert critique.recommendation == "keep"
    assert critique.important_differences == ["Different communication technology"]


def test_critic_degrades_gracefully_on_malformed_output(mock_search):
    critic = CriticAgent(scripted_llm(critic="nonsense"), mock_search, max_steps=1)
    critique, _ = critic.critique_one(
        EVENT, CandidateAnalogy(event=HistoricalEvent(name="X")))
    assert critique.recommendation == "keep"
    assert "could not produce" in critique.summary


def test_critic_notes_unverified_candidates(mock_search):
    critic = CriticAgent(scripted_llm(), mock_search, max_steps=1)
    candidate = CandidateAnalogy(
        event=HistoricalEvent(name="Battle of Nowhere", verified=False))
    critique, _ = critic.critique_one(EVENT, candidate)
    assert any("verif" in problem.lower() for problem in critique.factual_problems)


def test_critic_derives_overall_score_from_dimensions(mock_search):
    payload = dict(CRITIQUE_JSON)
    payload.pop("overall_score")
    critic = CriticAgent(scripted_llm(critic=finish(payload)), mock_search, max_steps=1)
    critique, _ = critic.critique_one(
        EVENT, CandidateAnalogy(event=HistoricalEvent(name="X")))
    assert critique.overall_score == pytest.approx(3.0)


# --- Anti-Analogy agent ---------------------------------------------------
def test_anti_analogy_returns_counterexamples(mock_search):
    agent = AntiAnalogyAgent(scripted_llm(), mock_search, max_steps=1)
    report, _ = agent.investigate_one(
        EVENT, CandidateAnalogy(event=HistoricalEvent(name="Revolutions of 1848")))
    assert report.verdict == "weakened"
    assert report.robustness == pytest.approx(0.4)
    assert report.counterexamples[0].event_name == "Revolutions of 1848"
    assert report.counterexamples[0].url  # verified against the knowledge base


def test_anti_analogy_infers_a_verdict_when_missing(mock_search):
    payload = dict(ANTI_JSON)
    payload.pop("verdict")
    payload["robustness"] = 0.2
    agent = AntiAnalogyAgent(scripted_llm(anti=finish(payload)), mock_search, max_steps=1)
    report, _ = agent.investigate_one(
        EVENT, CandidateAnalogy(event=HistoricalEvent(name="X")))
    assert report.verdict == "undermined"


def test_anti_analogy_handles_malformed_output(mock_search):
    agent = AntiAnalogyAgent(scripted_llm(anti="???"), mock_search, max_steps=1)
    report, _ = agent.investigate_one(
        EVENT, CandidateAnalogy(event=HistoricalEvent(name="X")))
    assert report.counterexamples == []
    assert "could not produce" in report.summary


# --- Final Judge (not an agent) -------------------------------------------
def test_final_judge_ranks_candidates():
    judge = FinalJudge(scripted_llm())
    candidates = [CandidateAnalogy(event=HistoricalEvent(name="Revolutions of 1848")),
                  CandidateAnalogy(event=HistoricalEvent(name="French Revolution"))]
    ranking = judge.rank(EVENT, candidates, [], [])
    assert [row.event_name for row in ranking.ranking] == ["Revolutions of 1848",
                                                           "French Revolution"]
    assert ranking.winner.rank == 1
    assert ranking.winner.weaknesses == ["Outcomes differ"]


def test_final_judge_has_no_tools():
    judge = FinalJudge(scripted_llm())
    assert not hasattr(judge, "search")
    assert not hasattr(judge, "_react")


def test_final_judge_completes_a_partial_ranking():
    partial = {"ranking": [{"rank": 1, "event_name": "French Revolution"}]}
    judge = FinalJudge(scripted_llm(judge=json.dumps(partial)))
    candidates = [CandidateAnalogy(event=HistoricalEvent(name="French Revolution")),
                  CandidateAnalogy(event=HistoricalEvent(name="Cold War"))]
    ranking = judge.rank(EVENT, candidates, [], [])
    assert [row.event_name for row in ranking.ranking] == ["French Revolution",
                                                           "Cold War"]
    assert [row.rank for row in ranking.ranking] == [1, 2]


def test_final_judge_falls_back_to_structured_scores():
    judge = FinalJudge(scripted_llm(judge="I cannot rank these."))
    candidates = [CandidateAnalogy(event=HistoricalEvent(name="Weak")),
                  CandidateAnalogy(event=HistoricalEvent(name="Strong"))]
    critiques = [Critique(candidate="Weak", overall_score=1.5, surface_level=True),
                 Critique(candidate="Strong", overall_score=4.0)]
    anti = [AntiAnalogyReport(candidate="Weak", robustness=0.1, verdict="undermined"),
            AntiAnalogyReport(candidate="Strong", robustness=0.9, verdict="holds")]
    ranking = judge.rank(EVENT, candidates, critiques, anti)
    assert ranking.winner.event_name == "Strong"
    assert "Fallback" in ranking.notes


def test_heuristic_ranking_penalises_unverified_candidates():
    verified = CandidateAnalogy(event=HistoricalEvent(name="Real", verified=True))
    invented = CandidateAnalogy(event=HistoricalEvent(name="Invented", verified=False))
    critiques = [Critique(candidate="Real", overall_score=3.0),
                 Critique(candidate="Invented", overall_score=3.0)]
    ranking = heuristic_ranking([invented, verified], critiques, [])
    assert ranking.winner.event_name == "Real"


def test_rank_candidates_without_an_llm_is_deterministic():
    candidates = [CandidateAnalogy(event=HistoricalEvent(name="A")),
                  CandidateAnalogy(event=HistoricalEvent(name="B"))]
    critiques = [Critique(candidate="A", overall_score=2.0),
                 Critique(candidate="B", overall_score=4.0)]
    ranking = rank_candidates(EVENT, candidates, critiques, [])
    assert ranking.winner.event_name == "B"


# --- Final Summarizer -----------------------------------------------------
def test_final_summarizer_produces_all_sections():
    from hal.schemas import FinalAnalogyResult, JudgeRanking, RankedCandidate

    result = FinalAnalogyResult(
        input_event=EVENT,
        analogy_event="Revolutions of 1848",
        winning_candidate=CandidateAnalogy(
            event=HistoricalEvent(name="Revolutions of 1848")),
        ranking=JudgeRanking(ranking=[
            RankedCandidate(rank=1, event_name="Revolutions of 1848", reason="best"),
            RankedCandidate(rank=2, event_name="French Revolution"),
        ]),
    )
    FinalSummarizer(scripted_llm()).apply(result)
    assert result.explanation.startswith("The comparison shows")
    assert result.similarities and result.differences
    assert result.counterexamples and result.limitations
    assert result.why_ranked_first


def test_final_summarizer_falls_back_on_bad_json():
    from hal.schemas import FinalAnalogyResult, JudgeRanking, RankedCandidate

    result = FinalAnalogyResult(
        input_event=EVENT, analogy_event="X",
        winning_candidate=CandidateAnalogy(
            event=HistoricalEvent(name="X"), rationale="shared structure",
            mapping={"topic": "same kind of event"}),
        ranking=JudgeRanking(ranking=[RankedCandidate(rank=1, event_name="X",
                                                      reason="only option")]),
    )
    critique = Critique(candidate="X", important_differences=["scale"],
                        weak_assumptions=["stability"])
    anti = AntiAnalogyReport(candidate="X", counterexamples=[
        CounterExample(event_name="Y", divergence="opposite outcome")])
    FinalSummarizer(scripted_llm(summarizer="sorry, prose only")).apply(result, critique,
                                                                        anti)
    assert result.similarities == ["shared structure", "topic: same kind of event"]
    assert result.differences == ["scale"]
    assert result.counterexamples == ["Y -- opposite outcome"]


# --- the whole pipeline ---------------------------------------------------
def test_pipeline_runs_end_to_end(mock_search):
    pipeline = build_pipeline(search=mock_search)
    result = pipeline.run({"event_name": "Arab Spring", "event_intro": "uprisings"})

    assert result.analogy_event == "Revolutions of 1848"
    assert len(result.initial_candidates) == 2
    assert len(result.rounds) == 1
    round_one = result.rounds[0]
    assert round_one.critiques and round_one.anti_analogies
    assert round_one.candidates_after
    assert result.ranking.winner.event_name == "Revolutions of 1848"
    assert result.explanation
    assert result.winning_candidate is not None


def test_pipeline_round_count_is_configurable(mock_search):
    for rounds in (0, 1, 3):
        pipeline = build_pipeline(search=mock_search)
        pipeline.config.refinement_rounds = rounds
        result = pipeline.run(EVENT)
        assert len(result.rounds) == rounds
        assert result.analogy_event  # a result is produced either way


def test_pipeline_feeds_critique_and_counterexamples_back_to_the_generator(mock_search):
    llm = scripted_llm()
    pipeline = build_pipeline(llm=llm, search=mock_search)
    pipeline.run(EVENT)
    revise_prompts = [p for p in llm.prompts if "Critic feedback" in p]
    assert revise_prompts, "the generator was never asked to revise"
    prompt = revise_prompts[0]
    assert "Solid structural match" in prompt              # critic summary
    assert "Similar uprisings ended in restoration" in prompt  # counterexample
    assert "Anti-Analogy findings" in prompt


def test_pipeline_judge_sees_critiques_and_counterexamples(mock_search):
    llm = scripted_llm()
    build_pipeline(llm=llm, search=mock_search).run(EVENT)
    judge_prompt = next(p for p in llm.prompts if "Final Judge" in p)
    assert "Critic feedback" in judge_prompt
    assert "Anti-Analogy findings" in judge_prompt
    assert "Supporting evidence" in judge_prompt


def test_pipeline_survives_a_failing_generator(mock_search):
    pipeline = build_pipeline(llm=scripted_llm(generate="no json"), search=mock_search)
    result = pipeline.run(EVENT)
    assert result.analogy_event == ""
    assert result.errors


def test_pipeline_survives_component_failures(mock_search):
    llm = scripted_llm(critic="broken", anti="broken", judge="broken",
                       summarizer="broken")
    result = build_pipeline(llm=llm, search=mock_search).run(EVENT)
    assert result.analogy_event  # heuristic ranking still produced a winner
    assert result.ranking.ranking


def test_pipeline_output_row_is_evaluation_compatible(mock_search):
    result = build_pipeline(search=mock_search).run(
        {"event_name": "Arab Spring", "event_intro": "uprisings"})
    row = result.to_output_row()
    assert row["event_name"] == "Arab Spring"
    assert row["analogy_event"] == "Revolutions of 1848"
    assert isinstance(row["candidate"], list)


def test_pipeline_works_without_a_search_backend():
    result = build_pipeline(search=None).run(EVENT)
    assert result.analogy_event == "Revolutions of 1848"


def test_critique_top_n_limits_reviewed_candidates(mock_search):
    llm = scripted_llm()
    pipeline = build_pipeline(llm=llm, search=mock_search, critique_top_n=1)
    result = pipeline.run(EVENT)
    assert len(result.rounds[0].critiques) == 1
    assert result.rounds[0].critiques[0].candidate == "Revolutions of 1848"
