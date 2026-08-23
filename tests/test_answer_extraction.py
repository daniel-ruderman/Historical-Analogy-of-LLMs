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
