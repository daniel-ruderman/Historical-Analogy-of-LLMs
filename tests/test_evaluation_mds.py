"""The provider-neutral MDS evaluation keeps the paper's methodology."""

from __future__ import annotations

import pytest

from gemini_baselines.evaluation_mds import (
    DIMENSION_WEIGHTS,
    DIMENSIONS,
    LITERAL_THRESHOLD,
    EvaluationContext,
    abstract_similarity,
    extract_features,
    jacc,
    mds_from_scores,
    multi_dimensional_similarity,
    pass_1,
)
from hal.providers.mock import MockLLMProvider
from hal.text_similarity import jaccard, tokenizer_backend

FOUR_DIM = ("1. Summary: A wave of uprisings. 2. Background: Economic stagnation. "
            "3. Process: Protests spread. 4. Result: Some regimes fell.")


def context_with(responses, mock_wiki):
    return EvaluationContext(llm=MockLLMProvider(responses=responses), wiki=mock_wiki)


# --- constants ------------------------------------------------------------
def test_paper_constants_are_unchanged():
    assert DIMENSIONS == ("topic", "background", "process", "result")
    assert DIMENSION_WEIGHTS == {"topic": 0.5, "background": 1.0,
                                 "process": 2.0, "result": 2.0}
    assert LITERAL_THRESHOLD == 0.35


# --- literal similarity (algorithmic, no LLM) -----------------------------
def test_jaccard_is_deterministic_and_llm_free():
    assert jacc("the cold war", "the cold war") == 1.0
    assert jacc("cold war tension", "pandemic influenza outbreak") == 0.0
    assert 0 < jacc("global pandemic outbreak", "global influenza outbreak") < 1


def test_jaccard_ignores_stopwords():
    assert jaccard("the war", "a war") == 1.0


def test_tokenizer_backend_is_reported():
    assert tokenizer_backend() in ("nltk", "regex")


# --- MDS formula ----------------------------------------------------------
def test_mds_formula_matches_the_paper():
    scores = {d: {"abstract_level": 4, "literal_level": 0.0} for d in DIMENSIONS}
    expected = sum(DIMENSION_WEIGHTS[d] * 4 * LITERAL_THRESHOLD for d in DIMENSIONS)
    assert mds_from_scores(scores) == pytest.approx(expected)


def test_high_literal_similarity_is_penalised_to_zero():
    scores = {d: {"abstract_level": 4, "literal_level": 0.9} for d in DIMENSIONS}
    assert mds_from_scores(scores) == 0.0


def test_literal_similarity_reduces_the_score_gradually():
    low = {d: {"abstract_level": 3, "literal_level": 0.1} for d in DIMENSIONS}
    high = {d: {"abstract_level": 3, "literal_level": 0.3} for d in DIMENSIONS}
    assert mds_from_scores(low) > mds_from_scores(high) > 0


def test_process_and_result_dominate_the_weighted_sum():
    topic_only = {d: {"abstract_level": 4 if d == "topic" else 1,
                      "literal_level": 0.0} for d in DIMENSIONS}
    result_only = {d: {"abstract_level": 4 if d == "result" else 1,
                       "literal_level": 0.0} for d in DIMENSIONS}
    assert mds_from_scores(result_only) > mds_from_scores(topic_only)


# --- LLM-judged abstract similarity ---------------------------------------
def test_abstract_similarity_parses_a_bare_score(mock_wiki):
    assert abstract_similarity("a", "b", context_with(["3"], mock_wiki)) == 3


def test_abstract_similarity_parses_a_verbose_answer(mock_wiki):
    context = context_with(["Score: 2 -- the topics differ."], mock_wiki)
    assert abstract_similarity("a", "b", context) == 2


def test_abstract_similarity_clamps_out_of_range_scores(mock_wiki):
    assert abstract_similarity("a", "b", context_with(["9"], mock_wiki)) == 4


def test_abstract_similarity_survives_unusable_output(mock_wiki):
    assert abstract_similarity("a", "b", context_with(["no idea"], mock_wiki)) == 1


def test_abstract_similarity_goes_through_the_provider(mock_wiki):
    context = context_with(["3"], mock_wiki)
    abstract_similarity("text one", "text two", context)
    prompt = context.llm.prompts[0]
    assert "sentence-level analogy scoring robot" in prompt  # the paper's prompt
    assert "text one" in prompt and "text two" in prompt


# --- feature extraction ---------------------------------------------------
def test_extract_features_fills_the_four_dimensions(mock_wiki, sample_event):
    context = context_with([FOUR_DIM], mock_wiki)
    event = extract_features(dict(sample_event), context)
    assert event["topic"].strip() == "A wave of uprisings."
    assert event["result"].strip() == "Some regimes fell."


def test_extract_features_uses_the_input_event_as_example(mock_wiki, sample_event):
    context = context_with([FOUR_DIM, FOUR_DIM], mock_wiki)
    input_event = extract_features(dict(sample_event), context)
    extract_features({"event_name": "Cold War", "event_intro": "tension"}, context,
                     input_example=input_event)
    assert "Arab Spring" in context.llm.prompts[1]  # the example is in the prompt


# --- end to end -----------------------------------------------------------
def test_multi_dimensional_similarity_aggregates(mock_wiki):
    other = ("1. Summary: A European revolutionary cascade. "
             "2. Background: Monarchical repression. "
             "3. Process: Barricades in many capitals. "
             "4. Result: Restoration followed.")

    def responder(prompt: str) -> str:
        if "event summary robot" not in prompt:
            return "3"
        # the analogy event must get a *different* summary, otherwise the
        # literal-similarity penalty correctly drives MDS to zero
        return other if "Revolutions of 1848" in prompt else FOUR_DIM

    context = EvaluationContext(llm=MockLLMProvider(responses=responder), wiki=mock_wiki)
    testset = [{"event_name": "Arab Spring", "event_intro": "uprisings",
                "analogy_event": "Revolutions of 1848"}]
    abstract, literal, overall, scored = multi_dimensional_similarity(
        testset, context, progress=False)
    assert all(abstract[d] == 3 for d in DIMENSIONS)
    assert set(literal) == set(DIMENSIONS)
    assert overall["all"] > 0
    assert scored[0]["mds"] == pytest.approx(overall["all"])


def test_samples_without_a_resolvable_analogy_are_skipped(mock_wiki):
    context = context_with([FOUR_DIM, "3"], mock_wiki)
    testset = [{"event_name": "X", "event_intro": "y", "analogy_event": "Nonexistent"}]
    _, _, overall, scored = multi_dimensional_similarity(testset, context,
                                                         progress=False)
    assert scored == []
    assert overall["all"] == 0.0


def test_pass_1_matches_reference_answers(mock_wiki):
    context = context_with([], mock_wiki)
    testset = [
        {"target_event": "Cold War", "analogy_event": "Cold War"},
        {"target_event": "Cold War", "analogy_event": "Spanish flu"},
    ]
    assert pass_1(testset, context) == 0.5
