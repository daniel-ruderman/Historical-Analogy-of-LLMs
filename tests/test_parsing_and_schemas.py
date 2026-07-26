"""Tolerant parsing of LLM output, and the structured schemas."""

from __future__ import annotations

from hal.json_utils import (
    as_str_list,
    get_float,
    get_list,
    get_str,
    parse_json_array,
    parse_json_object,
)
from hal.schemas import (
    AntiAnalogyReport,
    CandidateAnalogy,
    CounterExample,
    Critique,
    EventDimensions,
    FinalAnalogyResult,
    HistoricalEvent,
    JudgeRanking,
    RankedCandidate,
)


# --- JSON parsing ---------------------------------------------------------
def test_plain_json_object():
    assert parse_json_object('{"a": 1}') == {"a": 1}


def test_json_inside_markdown_fence():
    text = 'Here you go:\n```json\n{"event_name": "Spanish flu"}\n```\nHope that helps!'
    assert parse_json_object(text)["event_name"] == "Spanish flu"


def test_json_with_surrounding_prose():
    text = 'Sure. {"rank": 1, "event_name": "Cold War"} That is my answer.'
    assert parse_json_object(text)["event_name"] == "Cold War"


def test_json_object_with_nested_braces_in_strings():
    text = '{"note": "a } brace inside", "n": 2}'
    assert parse_json_object(text)["n"] == 2


def test_trailing_comma_is_repaired():
    assert parse_json_object('{"a": 1, "b": 2,}') == {"a": 1, "b": 2}


def test_python_list_literal_is_accepted():
    text = "['Spanish flu pandemic', \"Asian flu pandemic\"]"
    assert parse_json_array(text) == ["Spanish flu pandemic", "Asian flu pandemic"]


def test_malformed_output_returns_none_instead_of_raising():
    assert parse_json_object("I am afraid I cannot do that.") is None
    assert parse_json_array("no list here") is None


def test_as_str_list_handles_dicts_and_strings():
    assert as_str_list([{"event_name": "A"}, "B", {"title": "C"}, 5]) == ["A", "B", "C"]


def test_field_getters_are_forgiving():
    data = {"score": "about 3 points", "name": "", "event_name": "X",
            "weaknesses": ["a", "b"]}
    assert get_float(data, "score") == 3.0
    assert get_str(data, "name", "event_name") == "X"
    assert get_list(data, "missing", "weaknesses") == ["a", "b"]
    assert get_float(data, "missing", default=1.5) == 1.5


# --- schemas --------------------------------------------------------------
def test_event_dimensions_accepts_topic_or_summary():
    assert EventDimensions.from_dict({"summary": "s"}).topic == "s"
    assert EventDimensions.from_dict({"topic": "t"}).topic == "t"
    assert EventDimensions.from_dict(None).is_empty()


def test_historical_event_from_both_dataset_schemas():
    analogy_row = {"event_name": "Arab Spring", "event_intro": "protests"}
    pool_row = {"history_event_text": "World War I", "history_intro_text": "a war",
                "url": "http://example.org"}
    assert HistoricalEvent.from_dataset_row(analogy_row).name == "Arab Spring"
    pool_event = HistoricalEvent.from_dataset_row(pool_row)
    assert pool_event.name == "World War I"
    assert pool_event.description == "a war"


def test_candidate_analogy_roundtrip():
    candidate = CandidateAnalogy(
        event=HistoricalEvent(name="Spanish flu", description="a pandemic"),
        rationale="both are global outbreaks", confidence=0.8, status="keep",
    )
    restored = CandidateAnalogy.from_dict(candidate.to_dict())
    assert restored.name == "Spanish flu"
    assert restored.confidence == 0.8
    assert restored.status == "keep"


def test_critique_feedback_is_actionable_text():
    critique = Critique(
        candidate="Spanish flu",
        dimension_scores={"topic": 4, "result": 2},
        important_differences=["different medical capability"],
        surface_level=True, overall_score=3.0, recommendation="revise",
        summary="Strong on topic, weak on outcomes.",
    )
    feedback = critique.as_feedback()
    assert "Spanish flu" in feedback
    assert "revise" in feedback
    assert "different medical capability" in feedback
    assert "surface-level" in feedback


def test_anti_analogy_feedback_lists_counterexamples():
    report = AntiAnalogyReport(
        candidate="Revolutions of 1848",
        counterexamples=[CounterExample(event_name="Prague Spring",
                                        divergence="suppressed by invasion")],
        verdict="weakened", robustness=0.4,
    )
    feedback = report.as_feedback()
    assert "Prague Spring" in feedback
    assert "suppressed by invasion" in feedback
    assert "weakened" in feedback


def test_final_result_serialises_to_the_original_output_format():
    result = FinalAnalogyResult(
        input_event=HistoricalEvent(name="Arab Spring", description="protests"),
        analogy_event="Revolutions of 1848",
        initial_candidates=[CandidateAnalogy(event=HistoricalEvent(name="A"))],
        ranking=JudgeRanking(ranking=[RankedCandidate(rank=1, event_name="B")]),
    )
    row = result.to_output_row()
    # evaluation.py expects exactly these keys
    assert set(["event_name", "event_intro", "analogy_event"]).issubset(row)
    assert row["analogy_event"] == "Revolutions of 1848"
    assert row["candidate"][0] == ["A"]
    assert result.to_dict()["ranking"]["ranking"][0]["event_name"] == "B"
