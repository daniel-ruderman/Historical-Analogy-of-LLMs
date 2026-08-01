"""The provider-neutral baselines, driven by a scripted mock LLM."""

from __future__ import annotations

import pytest

from gemini_baselines import METHODS, get_method
from gemini_baselines.common import (
    BaselineContext,
    event_analysis,
    four_dimension_text,
    get_candidate_details,
    parse_candidate_list,
    split_four_dimensions,
)
from hal.providers.mock import MockLLMProvider

FOUR_DIM_OUTPUT = (
    "1. Summary: A wave of anti-government uprisings. "
    "2. Background: Economic stagnation and corruption. "
    "3. Process: Protests spread from country to country. "
    "4. Result: Some regimes fell, others repressed the protests."
)


def context_with(responses, mock_wiki, offline_settings, embeddings=None):
    return BaselineContext(
        llm=MockLLMProvider(responses=responses),
        wiki=mock_wiki,
        embeddings=embeddings,
        settings=offline_settings,
    )


# --- shared helpers -------------------------------------------------------
def test_four_dimension_split_matches_the_original_markers():
    parts = split_four_dimensions(FOUR_DIM_OUTPUT)
    assert parts["summary"].strip() == "A wave of anti-government uprisings."
    assert parts["background"].strip() == "Economic stagnation and corruption."
    assert parts["process"].strip() == "Protests spread from country to country."
    assert parts["result"].startswith("Some regimes fell")


def test_four_dimension_split_tolerates_markdown_numbering():
    parts = split_four_dimensions(
        "**1. Topic:** upheaval\n**2. Background:** grievances\n"
        "**3. Process:** mobilisation\n**4. Result:** change"
    )
    assert parts["summary"] == "upheaval"
    assert parts["result"] == "change"


def test_candidate_list_parsing_uses_the_repair_prompt(mock_wiki, offline_settings):
    llm = MockLLMProvider(responses=[
        'Here are the events: "Spanish flu", "Cold War"',      # unparseable
        '["Spanish flu","Cold War"]',                           # repair prompt output
    ])
    context = BaselineContext(llm=llm, wiki=mock_wiki, settings=offline_settings)
    assert parse_candidate_list(llm.generate("x"), context) == ["Spanish flu", "Cold War"]
    assert llm.calls == 2  # original answer + one repair call


def test_wikipedia_verification_drops_hallucinated_events(mock_wiki, offline_settings):
    context = BaselineContext(llm=MockLLMProvider(), wiki=mock_wiki,
                              settings=offline_settings)
    details = get_candidate_details(
        ["Spanish flu", "Battle of Fictional Ridge", "Cold War"], context
    )
    assert [d["event_name"] for d in details] == ["Spanish flu", "Cold War"]


def test_event_analysis_adds_four_dimensions(mock_wiki, offline_settings, sample_event):
    context = context_with([FOUR_DIM_OUTPUT], mock_wiki, offline_settings)
    event = event_analysis(dict(sample_event), context)
    assert event["background"].strip() == "Economic stagnation and corruption."
    assert "Arab Spring" in four_dimension_text(event)


# --- generation baselines -------------------------------------------------
def test_direct_generation_returns_a_single_event(mock_wiki, offline_settings,
                                                  sample_event):
    from gemini_baselines import direct_generation

    context = context_with(["Revolutions of 1848\nextra text"], mock_wiki,
                           offline_settings)
    row = direct_generation.run(sample_event, context)
    assert row["analogy_event"] == "Revolutions of 1848"
    assert row["candidate"] == []
    assert context.llm.calls == 1  # exactly one LLM call, as in the paper


def test_two_stage_generation_verifies_then_selects(mock_wiki, offline_settings,
                                                    sample_event):
    from gemini_baselines import twostage_generation

    context = context_with([
        '["French Revolution","Made Up Event","Revolutions of 1848"]',
        "Revolutions of 1848",
    ], mock_wiki, offline_settings)
    row = twostage_generation.run(sample_event, context)
    assert row["analogy_event"] == "Revolutions of 1848"
    assert row["candidate"] == [["French Revolution", "Made Up Event",
                                 "Revolutions of 1848"]]
    # The unverifiable candidate never reaches the selection prompt.
    assert "Made Up Event" not in context.llm.prompts[-1]


def test_two_stage_generation_survives_empty_verified_set(mock_wiki, offline_settings,
                                                          sample_event):
    from gemini_baselines import twostage_generation

    context = context_with(['["Entirely Fictional Event"]'], mock_wiki, offline_settings)
    row = twostage_generation.run(sample_event, context)
    assert row["analogy_event"] == ""


def test_summary_generation_uses_four_dimensions(mock_wiki, offline_settings,
                                                 sample_event):
    from gemini_baselines import summary_generation

    context = context_with({
        "summarize it into four parts": FOUR_DIM_OUTPUT,
        "output 10 historical events": '["French Revolution","Revolutions of 1848"]',
        "find the best event": "French Revolution",
    }, mock_wiki, offline_settings)
    row = summary_generation.run(sample_event, context)
    assert row["analogy_event"] == "French Revolution"
    selection_prompt = context.llm.prompts[-1]
    assert "Economic stagnation" in selection_prompt  # structured summaries, not raw text


def test_self_reflection_iterates_then_answers(mock_wiki, offline_settings, sample_event):
    from gemini_baselines import reflection_generation

    responses = [
        FOUR_DIM_OUTPUT,                                   # event_analysis (input)
        '["French Revolution","Cold War"]',                # generator, round 1
        FOUR_DIM_OUTPUT, FOUR_DIM_OUTPUT,                  # analysis of candidates
        "Thought: these are not close enough.\n\nReflection: focus on regional waves "
        "of uprising.",                                    # reflector -> reflect
        '["Revolutions of 1848"]',                         # generator, round 2
        FOUR_DIM_OUTPUT,                                   # analysis of candidate
        "Thought: this matches.\n\nFinal Answer:\nRevolutions of 1848",
    ]
    context = context_with(responses, mock_wiki, offline_settings)
    row = reflection_generation.run(sample_event, context)
    assert row["analogy_event"] == "Revolutions of 1848"
    assert row["candidate"] == [["French Revolution", "Cold War"],
                                ["Revolutions of 1848"]]


def test_self_reflection_conversation_memory_format():
    memory = reflection_memory()
    memory.save("Input Event: x", '["A"]')
    memory.save("focus on B", '["B"]')
    assert memory.buffer == (
        'Input: Input Event: x\nOutput: ["A"]\nInput: focus on B\nOutput: ["B"]'
    )


def test_self_reflection_reflection_loop_is_capped(mock_wiki, offline_settings,
                                                   sample_event):
    from gemini_baselines import reflection_generation

    # A model that always reflects would loop forever in the original code.
    always_reflect = MockLLMProvider(responses=lambda prompt: (
        FOUR_DIM_OUTPUT if "summarize it into four parts" in prompt
        else '["French Revolution"]' if "getting historical analogies events" in prompt
        else "Thought: not good.\n\nReflection: try again."
    ))
    context = BaselineContext(llm=always_reflect, wiki=mock_wiki,
                              settings=offline_settings)
    row = reflection_generation.run(sample_event, context, max_reflections=2)
    assert len(row["candidate"]) == 3  # initial set + 2 capped reflections


def reflection_memory():
    from gemini_baselines.reflection_generation import ConversationMemory

    return ConversationMemory()


# --- retrieval baselines --------------------------------------------------
@pytest.fixture
def small_pool(mock_embeddings):
    pool = [
        {"history_event_text": "Revolutions of 1848",
         "history_intro_text": "A wave of anti-government uprisings and protests "
                               "spreading across Europe in 1848.",
         "url": "u1"},
        {"history_event_text": "Cold War",
         "history_intro_text": "Geopolitical tension between superpowers with proxy "
                               "wars and an arms race.",
         "url": "u2"},
    ]
    for row in pool:
        row["embeddings"] = mock_embeddings.embed(row["history_intro_text"])
    return pool


def test_pool_rows_without_a_description_still_get_embeddable_text():
    """7 of the 658 pool rows have an empty description; Gemini rejects empty input."""
    from gemini_baselines.common import EMPTY_EVENT_PLACEHOLDER, pool_embedding_text

    assert pool_embedding_text({"history_intro_text": "a real description"}) == \
        "a real description"
    # blank description -> fall back to the title
    assert pool_embedding_text({"history_intro_text": "  ",
                                "history_event_text": "Day of the Dead"}) == \
        "Day of the Dead"
    # both blank -> placeholder, so the row stays in the pool
    assert pool_embedding_text({"history_intro_text": "",
                                "history_event_text": ""}) == EMPTY_EVENT_PLACEHOLDER
    assert pool_embedding_text({}) == EMPTY_EVENT_PLACEHOLDER


def test_real_event_pool_has_no_unembeddable_rows():
    from gemini_baselines.common import pool_embedding_text
    from hal.io_utils import load_event_pool

    pool = load_event_pool()
    assert len(pool) == 658
    assert all(pool_embedding_text(row).strip() for row in pool)


def test_direct_retrieval_picks_the_nearest_pool_event(small_pool, mock_embeddings,
                                                       offline_settings, sample_event):
    from gemini_baselines import direct_retrieval

    context = BaselineContext(embeddings=mock_embeddings, settings=offline_settings)
    row = direct_retrieval.run(sample_event, context, pool=small_pool)
    assert row["analogy_event"] == "Revolutions of 1848"
    assert row["candidate"] == []


def test_two_stage_retrieval_passes_top_k_to_the_llm(small_pool, mock_embeddings,
                                                     mock_wiki, offline_settings,
                                                     sample_event):
    from gemini_baselines import twostage_retrieval

    llm = MockLLMProvider(responses=["Cold War"])
    context = BaselineContext(llm=llm, wiki=mock_wiki, embeddings=mock_embeddings,
                              settings=offline_settings)
    row = twostage_retrieval.run(sample_event, context, pool=small_pool)
    assert row["analogy_event"] == "Cold War"
    assert row["candidate"] == [["Revolutions of 1848", "Cold War"]]
    assert "Optional Historical Events:" in llm.prompts[0]


def test_retrieval_excludes_the_input_event_itself(small_pool, mock_embeddings,
                                                   offline_settings):
    from gemini_baselines import direct_retrieval

    context = BaselineContext(embeddings=mock_embeddings, settings=offline_settings)
    event = {"event_name": "Revolutions of 1848",
             "event_intro": "A wave of anti-government uprisings across Europe."}
    row = direct_retrieval.run(event, context, pool=small_pool)
    assert row["analogy_event"] == "Cold War"


# --- registry -------------------------------------------------------------
def test_all_six_paper_methods_are_registered():
    assert set(METHODS) == {
        "direct_retrieval", "twostage_retrieval", "direct_generation",
        "twostage_generation", "summary_generation", "reflection_generation",
    }
    for name in METHODS:
        assert callable(get_method(name))
