"""Shared helpers for the provider-neutral re-implementations of the paper.

The research procedure is unchanged; only the API layer differs.  Every helper
here corresponds to a function in ``framework/`` -- the docstrings name it.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from hal.config import Settings, get_settings
from hal.io_utils import read_jsonl, write_jsonl  # noqa: F401  (re-exported)
from hal.json_utils import as_str_list, parse_json_array
from hal.providers.base import EmbeddingProvider, LLMProvider, SearchProvider
from hal.wiki import WikipediaHelper

from . import prompts

DIMENSION_KEYS = ("summary", "background", "process", "result")


@dataclass
class BaselineContext:
    """Everything a baseline needs: an LLM, Wikipedia, and settings.

    Passing this around (instead of module-level globals as in the original
    scripts) is what lets the same code run against any provider and lets the
    tests run without network access.
    """

    llm: Optional[LLMProvider] = None
    wiki: Optional[WikipediaHelper] = None
    embeddings: Optional[EmbeddingProvider] = None
    settings: Settings = field(default_factory=get_settings)
    verbose: bool = False

    @classmethod
    def build(cls, *, model: Optional[str] = None, role: str = "baseline",
              settings: Optional[Settings] = None,
              search_provider: Optional[SearchProvider] = None,
              with_embeddings: bool = False,
              verbose: bool = False) -> "BaselineContext":
        from hal.providers.factory import (
            get_embedding_provider,
            get_llm,
            get_search_provider,
        )

        settings = settings or get_settings()
        llm = get_llm(role=role, model=model, settings=settings)
        provider = search_provider or get_search_provider(settings=settings)
        wiki = WikipediaHelper(provider, settings.wiki_max_chars)
        embeddings = get_embedding_provider(settings=settings) if with_embeddings else None
        return cls(llm=llm, wiki=wiki, embeddings=embeddings, settings=settings,
                   verbose=verbose)

    # -- llm_predict() of the original scripts ---------------------------
    def predict(self, text: str, stop: Optional[Sequence[str]] = None,
                temperature: Optional[float] = None) -> str:
        if self.llm is None:
            raise ValueError("This method needs an LLM provider (context.llm)")
        response = self.llm.generate(
            text,
            stop=list(stop) if stop else None,
            temperature=self.settings.temperature if temperature is None else temperature,
        )
        if self.verbose:
            print(f"    [llm {self.llm.model}] {len((response or '').strip())} chars")
        return response or ""

    def log(self, message: str) -> None:
        if self.verbose:
            print(f"    {message}")


# --------------------------------------------------------------------------
# Candidate list parsing
# --------------------------------------------------------------------------
def parse_candidate_list(text: str, context: BaselineContext) -> List[str]:
    """``ast.literal_eval`` with the original repository's repair prompt.

    ``twostage_generation.get_candidate`` tries ``ast.literal_eval`` first and,
    on failure, asks the LLM to rewrite the answer as a Python list.  We keep
    both steps and add a purely local JSON/array extraction as a last resort so
    that a malformed answer degrades instead of crashing the run.
    """
    text = (text or "").strip()
    try:
        value = ast.literal_eval(text)
        items = as_str_list(value)
        if items:
            return items
    except Exception:
        pass

    repaired = context.predict(prompts.CANDIDATE_LIST_REPAIR.format(text=text))
    try:
        items = as_str_list(ast.literal_eval(repaired.strip()))
        if items:
            return items
    except Exception:
        pass

    for source in (repaired, text):
        items = as_str_list(parse_json_array(source))
        if items:
            return items
    return []


# --------------------------------------------------------------------------
# Four-dimension summarisation (event_analysis in summary/reflection methods)
# --------------------------------------------------------------------------
def split_four_dimensions(output: str) -> Dict[str, str]:
    """Slice ``1. Summary: ... 4. Result: ...`` exactly like the original code."""
    output = output or ""
    markers = ("1. Summary: ", "2. Background: ", "3. Process: ", "4. Result: ")
    if all(marker in output for marker in markers):
        i1, i2, i3, i4 = (output.find(m) for m in markers)
        return {
            "summary": output[i1 + len(markers[0]):i2],
            "background": output[i2 + len(markers[1]):i3],
            "process": output[i3 + len(markers[2]):i4],
            "result": output[i4 + len(markers[3]):],
        }
    # Tolerant fallback: numbered lines with different spacing/markdown.
    import re

    result = {key: "" for key in DIMENSION_KEYS}
    keys = "Summary|Topic|Background|Process|Result"
    pattern = re.compile(
        rf"(?:^|\n)[\s*]*(?:\d\.)?[\s*]*({keys})[\s*]*:[\s*]*(.+?)"
        rf"(?=(?:\n[\s*]*(?:\d\.)?[\s*]*(?:{keys})[\s*]*:)|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    for match in pattern.finditer(output):
        key = match.group(1).lower()
        key = "summary" if key == "topic" else key
        if key in result and not result[key]:
            result[key] = match.group(2).strip()
    if not any(result.values()):
        result["summary"] = output.strip()
    return result


def event_analysis(event: Dict[str, Any], context: BaselineContext) -> Dict[str, Any]:
    """``event_analysis`` of summary_generation.py / reflection_generation.py.

    Adds ``summary``/``background``/``process``/``result`` keys to ``event``.
    """
    output = context.predict(
        prompts.EVENT_ANALYSIS.format(
            event=f"{event['event_name']}: {event['event_intro']}"
        )
    )
    event.update(split_four_dimensions(output))
    return event


def four_dimension_text(event: Dict[str, Any]) -> str:
    """``"name: summary background process result"`` as the original composes it."""
    return (
        f"{event['event_name']}: {event.get('summary', '')} "
        f"{event.get('background', '')} {event.get('process', '')} "
        f"{event.get('result', '')}"
    )


# --------------------------------------------------------------------------
# Wikipedia verification of generated candidates
# --------------------------------------------------------------------------
def get_candidate_details(
    candidates: Sequence[str],
    context: BaselineContext,
    analyze: bool = False,
) -> List[Dict[str, Any]]:
    """``get_candidate_details``: keep only candidates that exist on Wikipedia.

    The original wraps ``wikipedia.summary`` in ``try/except: continue``, which
    silently drops hallucinated events.  ``analyze=True`` additionally runs the
    four-dimension summarisation, as summary/reflection generation do.
    """
    if context.wiki is None:
        raise ValueError("This method needs a Wikipedia provider (context.wiki)")
    details: List[Dict[str, Any]] = []
    for name in candidates:
        intro = context.wiki.summary(name)
        if not intro:
            context.log(f"[wiki] no page for {name!r} -- dropped")
            continue
        candidate_event = {"event_name": name, "event_intro": intro}
        if analyze:
            candidate_event = event_analysis(candidate_event, context)
        details.append(candidate_event)
    return details


# --------------------------------------------------------------------------
# Event pool + embeddings (retrieval methods)
# --------------------------------------------------------------------------
EMPTY_EVENT_PLACEHOLDER = "(no description available)"


def pool_embedding_text(row: Dict[str, Any]) -> str:
    """Text to embed for one event-pool row.

    Normally ``history_intro_text``, exactly as the original repository embeds
    it. Seven of the 658 pool rows have an empty description (and one also has
    an empty title); OpenAI's endpoint accepts an empty input, but the Gemini
    endpoint rejects it with ``400 ... contains an empty Part``. So we fall back
    to the event title, and finally to a placeholder, which keeps all 658 rows
    in the pool instead of silently changing its size.
    """
    intro = (row.get("history_intro_text") or "").strip()
    if intro:
        return intro
    title = (row.get("history_event_text") or "").strip()
    return title or EMPTY_EVENT_PLACEHOLDER


def load_pool_with_embeddings(
    context: BaselineContext,
    limit: Optional[int] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> List[Dict[str, Any]]:
    """Return the event pool with an ``embeddings`` key on every row.

    The original repository ships pre-computed OpenAI vectors
    (``similarity_embeddings.jsonl``).  We compute them with the configured
    :class:`~hal.providers.base.EmbeddingProvider` instead and cache them on
    disk keyed by provider/model, so the pool is embedded only once.
    """
    from hal.io_utils import load_event_pool

    if context.embeddings is None:
        raise ValueError("This method needs an embedding provider (context.embeddings)")
    pool = load_event_pool(limit=limit)
    texts = [pool_embedding_text(row) for row in pool]
    batch_size = 32
    vectors: List[List[float]] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        vectors.extend(context.embeddings.embed_batch(chunk))
        if progress:
            progress(min(start + batch_size, len(texts)), len(texts))
    for row, vector in zip(pool, vectors):
        row["embeddings"] = vector
    return pool


def rank_pool_events(event: Dict[str, Any], pool: List[Dict[str, Any]],
                     context: BaselineContext) -> List[Dict[str, Any]]:
    """``get_similar_events``: rank the pool by similarity to the input event.

    Same algorithm as the original: embed ``event_intro``, score every pool
    event, exclude the input event itself, sort descending.
    """
    from hal.vector import cosine_similarity

    query = context.embeddings.embed(event["event_intro"])
    scored = [
        (row, cosine_similarity(row["embeddings"], query))
        for row in pool
        if row["history_event_text"] != event["event_name"]
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [row for row, _ in scored]


def output_row(event: Dict[str, Any], analogy_event: str,
               candidate: Optional[List[Any]] = None) -> Dict[str, Any]:
    """Build an output record in the original repository's jsonl format."""
    row = dict(event)
    row.pop("embeddings", None)
    row["analogy_event"] = analogy_event
    row["candidate"] = candidate if candidate is not None else []
    return row


def clean_answer(text: str) -> str:
    """Trim an LLM answer that should be a single event name."""
    text = (text or "").strip()
    for prefix in ("Historical Analogies Events:", "Answer:", "Final Answer:"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    return text.split("\n")[0].strip().strip(".").strip()
