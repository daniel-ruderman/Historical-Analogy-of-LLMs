"""Recovering the event name from a generation baseline's reply.

The paper's generation prompts are completion-style: they demonstrate a layout
and stop mid-pattern, expecting the model to continue with the event name alone.
Completion models do that; chat-tuned models restart the layout and put the
answer in the slot several lines down. These tests pin both behaviours, using
replies captured verbatim from qwen3:8b.

Offline: no provider, no network.
"""

from __future__ import annotations

import pytest

from gemini_baselines.common import clean_answer, extract_analogy_answer


# --- the paper's own format must be untouched -----------------------------
@pytest.mark.parametrize("reply", [
    "Spanish flu",
    "  Spanish flu  ",
    "Spanish flu.",
    "Historical Analogies Events: Spanish flu",
    "Answer: Spanish flu",
])
def test_a_terse_answer_is_unchanged(reply):
    """A completion model's reply parses exactly as it always did."""
    assert extract_analogy_answer(reply) == clean_answer(reply)
    assert extract_analogy_answer(reply) == "Spanish flu"


# --- replies captured from qwen3:8b ---------------------------------------
QWEN_CORRECT = """    ==== Answer
    Input Event:
    Russian Revolution

    Historical Analogies Events:
    French Revolution"""

QWEN_MARKDOWN = """**Input Event:**
Trump administration migrant detentions

**Historical Analogies Events:**
**Mexican Revolution**

**==== Answer**
**Input Event:**
Trump administration migrant detentions

**Historical Analogies Events:**
Mexican Revolution"""

QWEN_PARROT = """    ==== Answer
    Input Event:
    Arab Spring

    Historical Analogies Events:
    Spanish flu

    ==== Analogy
    The Arab Spring can be analogized to the Spanish flu in terms of their
    transformative and disruptive impact on societies."""


def test_recovers_the_answer_from_the_template_slot():
    # Before the fix this returned "==== Answer".
    assert extract_analogy_answer(QWEN_CORRECT) == "French Revolution"


def test_recovers_an_answer_wrapped_in_markdown():
    # Before the fix this returned "**Input Event:**".
    assert extract_analogy_answer(QWEN_MARKDOWN) == "Mexican Revolution"


def test_keeps_a_genuinely_bad_answer():
    """Parroting the prompt's one-shot example is a real method failure.

    It must be recorded as the answer it is, not laundered into a miss.
    """
    assert extract_analogy_answer(QWEN_PARROT) == "Spanish flu"


def test_reads_the_twostage_sentence_slot():
    reply = ("Among the options, the most appropriate one to use as an analogy "
             "for the Arab Spring is Revolutions of 1989")
    assert extract_analogy_answer(reply) == "Revolutions of 1989"


# --- shapes captured from the choice prompts ------------------------------
QWEN_TWOSTAGE = """The most appropriate one to use as an analogy for the **Arab Spring** \
is the **French Revolution**.

### Reasoning:

Both the **Arab Spring** and the **French Revolution** represent movements."""

QWEN_SUMMARY = """Based on the input event **"Arab Spring"** and the optional historical \
events provided, the **best historical analogy** is:

### ✅ **French Revolution**

---

### **Reasoning:**

Both represent massive uprisings against authoritarian regimes."""


def test_emphasis_inside_the_line_is_removed():
    # Stripping only the ends would leave "the **French Revolution**".
    assert extract_analogy_answer(QWEN_TWOSTAGE) == "the French Revolution"


def test_a_lead_in_sentence_is_skipped():
    # The reply has no marker at all: line one is a lead-in ending in ":" and
    # the name sits under a heading, behind an emoji.
    assert extract_analogy_answer(QWEN_SUMMARY) == "French Revolution"


@pytest.mark.parametrize("reply, expected", [
    ("### **Cold War**", "Cold War"),
    ("- Cold War", "Cold War"),
    ("> Cold War", "Cold War"),
    ("**Final Answer:** Cold War", "Cold War"),
    ("✅ Cold War", "Cold War"),
])
def test_decoration_is_stripped(reply, expected):
    assert extract_analogy_answer(reply) == expected


# --- scaffolding must never be returned as an event -----------------------
@pytest.mark.parametrize("reply", [
    "==== Answer",
    "**Input Event:**",
    "Input Event:",
    "==== Answer the following questions using the format given above",
    "Historical Analogies Events:",
    "",
    "   ",
])
def test_scaffolding_never_becomes_an_analogy(reply):
    """An empty answer becomes ``no_analogy`` and drops out of the averages.

    Returning the scaffolding instead would hand the judge a string like
    "==== Answer" to score as though it were a real analogy.
    """
    assert extract_analogy_answer(reply) == ""


def test_a_marker_with_only_scaffolding_after_it_yields_nothing():
    reply = "Historical Analogies Events:\n\n==== Answer\nInput Event:"
    assert extract_analogy_answer(reply) == ""


def test_event_names_are_not_mistaken_for_scaffolding():
    # "Case Blue" starts with a template-ish word but is a real operation.
    assert extract_analogy_answer("Case Blue") == "Case Blue"
    assert extract_analogy_answer("- **Cold War**") == "Cold War"


# --- hedged answers -------------------------------------------------------
# qwen3 answers the Self-reflection prompt with the event name wrapped in
# qualifications. All five strings below are verbatim from the popular run of
# 2026-08-23, where they cost reflection_generation 5 of its 20 examples.
HEDGED = [
    ("Suez Crisis (with reservations) or Berlin Blockade (with reservations), "
     "but none are ideal. A better analogy would be a nuclear standoff between "
     "superpowers with secret diplomacy and avoidance of war.", "Suez Crisis"),
    ("1998 U.S. embassy bombings in Africa (with the caveat that a more precise "
     "analogy might require a similar large-scale, ideologically driven attack)",
     "1998 U.S. embassy bombings in Africa"),
    ("Digital Revolution (with the caveat that it is a partial analogy and a "
     "more precise historical event would be better suited)", "Digital Revolution"),
]


@pytest.mark.parametrize("reply, expected", HEDGED)
def test_a_hedged_answer_keeps_only_the_event(reply, expected):
    assert extract_analogy_answer(reply) == expected


def test_an_abbreviation_is_not_read_as_a_sentence_end():
    """"1998 U.S. embassy bombings" must not be cut to "1998 U.S"."""
    assert extract_analogy_answer("1998 U.S. embassy bombings in Africa") == \
        "1998 U.S. embassy bombings in Africa"


@pytest.mark.parametrize("reply", [
    "None of the provided events are suitable analogies for space colonization.",
    "No suitable analogy exists among the candidates.",
    "There is no good match here.",
    "I cannot identify a suitable analogy.",
])
def test_an_explicit_refusal_is_recorded_as_no_answer(reply):
    """A refusal is an answer of "none", not a name we failed to parse.

    Salvaging a name out of it would invent an answer the method never gave.
    """
    assert extract_analogy_answer(reply) == ""


@pytest.mark.parametrize("title", [
    "Cannabis Act (Canada)",
    "Estado Novo (Portugal)",
    "Border campaign (Irish Republican Army)",
    "U.S. internment of Japanese Americans during World War II",
    "1998 United States embassy bombings",
])
def test_real_titles_survive_the_hedge_stripping(title):
    """Only *hedging* parentheticals are removed -- disambiguators are titles."""
    assert extract_analogy_answer(title) == title
