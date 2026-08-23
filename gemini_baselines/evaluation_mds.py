"""Provider-neutral port of ``evaluation.py`` (Pass@1 and MDS).

Methodology and constants are the paper's, unchanged:

* four dimensions ``D = {topic, background, process, result}``;
* **abstract similarity** ``sim_Abs`` -- an LLM scores each dimension 1-4 with
  the paper's grading prompt (GPT-4 originally; any configured provider here);
* **literal similarity** ``sim_Lit`` -- NLTK tokenization + stop-word removal +
  Jaccard.  Purely algorithmic: no LLM is involved, by design;
* ``MDS = sum_d w_d * sim_Abs(D_I^d, D_H^d) * max(alpha - sim_Lit(D_I^d, D_H^d), 0)``
  with ``w = {topic: 0.5, background: 1, process: 2, result: 2}`` and
  ``alpha = 0.35``;
* **Pass@1** -- the Wikipedia title sets of the produced analogy and of the
  reference ``target_event`` must intersect.

The only substitutions are infrastructural: the judge LLM goes through
:class:`~hal.providers.base.LLMProvider` and Wikipedia goes through a
:class:`~hal.providers.base.SearchProvider`.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from hal.config import get_settings, load_settings, set_settings
from hal.io_utils import configure_stdout, read_jsonl, write_jsonl
from hal.providers.base import LLMProvider
from hal.text_similarity import jaccard, tokenizer_backend
from hal.wiki import WikipediaHelper

from . import prompts
from .common import split_four_dimensions

# --- constants from the paper (Sec. 3.5 / Appendix) -----------------------
DIMENSIONS = ("topic", "background", "process", "result")
DIMENSION_WEIGHTS = {"topic": 0.5, "background": 1.0, "process": 2.0, "result": 2.0}
LITERAL_THRESHOLD = 0.35  # alpha


# Bump when a prompt below changes, so cached values are never reused across
# different prompts. Cached entries are also keyed by the evaluation model.
PROMPT_VERSION = "paper-v1"


@dataclass
class EvaluationContext:
    llm: LLMProvider
    wiki: WikipediaHelper
    temperature: float = 0.0
    verbose: bool = False
    cache: Optional[Any] = None      # hal.cache.JsonCache; None disables caching
    llm_calls: int = 0
    cache_hits: int = 0
    # How each 1-4 judgement was recovered from the judge's reply, keyed by the
    # ``how`` of :func:`parse_abstract_score`. Auditing only -- these counts
    # never enter the metric.
    score_parses: Dict[str, int] = field(default_factory=dict)

    @classmethod
    def build(cls, model: Optional[str] = None, verbose: bool = False,
              search_provider=None, use_cache: bool = True) -> "EvaluationContext":
        from hal.providers.factory import get_llm, get_search_provider

        settings = get_settings()
        llm = get_llm(role="evaluation", model=model, settings=settings)
        provider = search_provider or get_search_provider(settings=settings)
        cache = None
        if use_cache and settings.cache_enabled:
            from hal.cache import JsonCache

            cache = JsonCache(settings.cache_dir, "evaluation_mds", enabled=True)
        return cls(llm=llm, wiki=WikipediaHelper(provider, settings.wiki_max_chars),
                   temperature=settings.evaluation_temperature, verbose=verbose,
                   cache=cache)

    def predict(self, prompt: str) -> str:
        self.llm_calls += 1
        return self.llm.generate(prompt, temperature=self.temperature) or ""

    # -- caching ---------------------------------------------------------
    def cache_key(self, kind: str, *parts: str) -> str:
        """Key an entry by kind + evaluation model + prompt version + inputs.

        Values produced by a different model or a different prompt therefore
        never collide. Caching is an efficiency measure only: a cache hit
        returns exactly what the same call would have returned.
        """
        import hashlib

        payload = "\x1f".join([kind, self.llm.model or "", PROMPT_VERSION, *parts])
        return f"{kind}:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"

    def cached(self, kind: str, parts: Sequence[str], compute):
        if self.cache is None:
            return compute()
        key = self.cache_key(kind, *parts)
        hit = self.cache.get(key)
        if hit is not None:
            self.cache_hits += 1
            return hit
        value = compute()
        self.cache.set(key, value)
        return value


# --------------------------------------------------------------------------
# Dimension summarisation  (extract_features in evaluation.py)
# --------------------------------------------------------------------------
def extract_features(event: Dict[str, Any], context: EvaluationContext,
                     input_example: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Summarise an event into the four dimensions.

    Like the original, when ``input_example`` is given (the already-summarised
    input event) it is used as the in-context example, so that both events are
    summarised in a comparable style.
    """
    event_text = f"{event['event_name']}: {event['event_intro']}"
    if input_example is None:
        prompt = prompts.EVAL_EXTRACT_FEATURES.format(event=event_text)
    else:
        prompt = prompts.EVAL_EXTRACT_FEATURES_WITH_EXAMPLE.format(
            event_name=input_example["event_name"],
            event_intro=input_example["event_intro"],
            event_summary=input_example["topic"],
            event_background=input_example["background"],
            event_process=input_example["process"],
            event_result=input_example["result"],
            event=event_text,
        )
    # Cached on the exact prompt, so a repeated event (or a repeated
    # input-example style) is summarised only once per evaluation model.
    parts = context.cached(
        "extract_features", (prompt,),
        lambda: split_four_dimensions(context.predict(prompt)),
    )
    # evaluation.py stores the "summary" part under the key "topic".
    event["topic"] = parts["summary"]
    event["background"] = parts["background"]
    event["process"] = parts["process"]
    event["result"] = parts["result"]
    return event


# --------------------------------------------------------------------------
# Abstract similarity (LLM judge, 1-4)  /  literal similarity (Jaccard)
# --------------------------------------------------------------------------
class ScoreParseError(ValueError):
    """The judge's reply carried no recoverable 1-4 score.

    Raised instead of inventing a number: a dimension that cannot be judged
    makes the whole sample unscored, and the runner reports it with a status.
    """


# The judge's reply is read in three steps, tried in this order. The first is
# the original's own behaviour and the only step a bare-digit reply ever
# reaches, so for the format ``evaluation.py`` was written against -- GPT-4
# continuing the prompt's trailing "Score:" -- this returns exactly what the
# original returned. The later steps matter only for chat-tuned judges that
# wrap the digit in prose: qwen3:8b answers "**Score: 3**" followed by a
# paragraph of reasoning, and never emits a bare digit at all.
_SCORE_ANCHOR = re.compile(r"(?i)\bscore\b\W{0,12}(\d+)")
_SCORE_IN_RANGE = re.compile(r"\b([1-4])\b")


def parse_abstract_score(result: str) -> Tuple[int, str]:
    """Recover the 1-4 judgement from a judge reply.

    Returns ``(score, how)`` with ``how`` one of ``"bare"``, ``"anchored"`` or
    ``"scan"``; raises :class:`ScoreParseError` when the reply contains no
    score at all.

    Deliberately *not* the original's ``re.search(r"\\d+", ...)``. That takes the
    first digit run anywhere in the reply, which in a prose answer is as likely
    to come from "Description 1" -- a phrase this very prompt teaches the model
    to use -- or from a year as from the verdict. Anchoring on the "Score:" cue
    first, and only then on a standalone 1-4, stops a paragraph of reasoning we
    discard from silently deciding the metric.
    """
    text = (result or "").strip()
    try:
        return int(text), "bare"
    except ValueError:
        pass
    match = _SCORE_ANCHOR.search(text)
    if match:
        return int(match.group(1)), "anchored"
    match = _SCORE_IN_RANGE.search(text)
    if match:
        return int(match.group(1)), "scan"
    raise ScoreParseError(f"no 1-4 score in judge reply: {text[:200]!r}")


def abstract_similarity(text1: str, text2: str, context: EvaluationContext) -> int:
    """The paper's 1-4 abstract-similarity judgement.

    Two robustness differences from the original, neither of which changes the
    result for a well-formed bare-digit answer:

    * the digit is located by :func:`parse_abstract_score` rather than by
      "first digit anywhere in the reply";
    * an unreadable reply raises :class:`ScoreParseError` instead of scoring 1,
      so the sample is reported unscored rather than silently counted as a bad
      analogy. Inventing a 1 there would both fabricate a judgement and drag
      the average down.

    An out-of-range score is still clamped to [1, 4]; the original warns and
    returns it unclamped.
    """
    prompt = prompts.EVAL_ABSTRACT_SIMILARITY.format(text1=text1, text2=text2)
    result = context.cached("abstract_similarity", (text1, text2),
                            lambda: context.predict(prompt))
    score, how = parse_abstract_score(result)
    context.score_parses[how] = context.score_parses.get(how, 0) + 1
    if score > 4:
        print("error score")
    return max(1, min(4, score))


# ``jacc`` in the original; kept algorithmic (no LLM).
jacc = jaccard


# --------------------------------------------------------------------------
# Pass@1
# --------------------------------------------------------------------------
def pass_1_single(target_event: str, analogy_event: str,
                  context: EvaluationContext) -> bool:
    """Whether one produced analogy counts as a Pass@1 hit.

    Exactly the original rule: run a Wikipedia *search* for the reference name
    and for the produced name, and count a hit when the two result sets
    intersect. This is deliberately not raw string equality -- it accepts
    aliases and alternative titles, as the original does.
    """
    target_events = context.wiki.search_titles(target_event or "")
    analogy_events = context.wiki.search_titles(analogy_event or "")
    return any(e in analogy_events for e in target_events)


def pass_1(dataset: List[Dict[str, Any]], context: EvaluationContext) -> float:
    """Fraction of samples whose analogy matches the reference ``target_event``.

    As in the original: compare the Wikipedia title sets of both names.
    """
    hits = sum(
        1 for data in dataset
        if pass_1_single(data.get("target_event", ""), data.get("analogy_event", ""),
                         context)
    )
    return hits / len(dataset) if dataset else 0.0


# --------------------------------------------------------------------------
# Multi-dimensional similarity
# --------------------------------------------------------------------------
def mds_from_scores(scores: Dict[str, Dict[str, float]]) -> float:
    """Apply the paper's MDS formula to per-dimension scores of one sample."""
    total = 0.0
    for dimension in DIMENSIONS:
        abstract_level = scores[dimension]["abstract_level"]
        literal_level = scores[dimension]["literal_level"]
        if literal_level >= LITERAL_THRESHOLD:
            literal_level = 0.0
        else:
            literal_level = LITERAL_THRESHOLD - literal_level
        total += DIMENSION_WEIGHTS[dimension] * abstract_level * literal_level
    return total


def score_sample_detailed(data: Dict[str, Any], context: EvaluationContext
                          ) -> Tuple[Optional[Dict[str, Any]], str]:
    """Score one output row, returning ``(row, status)``.

    ``status`` is ``"ok"``, ``"no_analogy"`` (the method produced no answer),
    ``"unresolved_event"`` (the named event has no entry in the knowledge base,
    which is where ``evaluation.py`` raises inside ``wiki()``) or
    ``"unparseable_score"`` (the judge replied with nothing we can read a 1-4
    out of). Distinguishing them lets the runner report *why* a sample was
    skipped instead of silently dropping it -- or, worse, scoring it anyway.
    """
    analogy_name = (data.get("analogy_event") or "").strip()
    if not analogy_name:
        return None, "no_analogy"
    input_event = extract_features(dict(data), context)
    intro = context.wiki.summary(analogy_name)
    if not intro:
        return None, "unresolved_event"
    analog_event = extract_features(
        {"event_name": analogy_name, "event_intro": intro}, context,
        input_example=input_event,
    )
    scores: Dict[str, Dict[str, float]] = {}
    for dimension in DIMENSIONS:
        try:
            abstract_level = abstract_similarity(
                input_event[dimension], analog_event[dimension], context
            )
        except ScoreParseError:
            # One unreadable dimension makes the sample's MDS undefined: the
            # formula sums over all four. Report it rather than guess a value.
            return None, "unparseable_score"
        scores[dimension] = {
            "abstract_level": abstract_level,
            "literal_level": jacc(input_event[dimension], analog_event[dimension]),
        }
    row = dict(data)
    row["score"] = scores
    row["mds"] = mds_from_scores(scores)
    row["input_dimensions"] = {d: input_event[d] for d in DIMENSIONS}
    row["analogy_dimensions"] = {d: analog_event[d] for d in DIMENSIONS}
    return row, "ok"


def score_sample(data: Dict[str, Any], context: EvaluationContext
                 ) -> Optional[Dict[str, Any]]:
    """Score one output row; ``None`` when the analogy event cannot be resolved."""
    row, _status = score_sample_detailed(data, context)
    return row


def component_scores(scores: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Flatten one sample's scores into the paper's table columns.

    ``TAbs``/``TLit``/``TAll`` for Topic and likewise ``B``/``P``/``R``.
    ``*Abs`` is the 1-4 LLM judgement, ``*Lit`` the Jaccard similarity and
    ``*All`` the per-dimension product ``abstract * max(alpha - literal, 0)``
    -- exactly ``overall_score[d]`` in the original ``evaluation.py``, i.e.
    *without* the dimension weight. ``MDS`` is the weighted sum.
    """
    letters = {"topic": "T", "background": "B", "process": "P", "result": "R"}
    flat: Dict[str, float] = {}
    for dimension in DIMENSIONS:
        letter = letters[dimension]
        abstract_level = float(scores[dimension]["abstract_level"])
        literal_level = float(scores[dimension]["literal_level"])
        adjusted = max(LITERAL_THRESHOLD - literal_level, 0.0)
        flat[f"{letter}Abs"] = abstract_level
        flat[f"{letter}Lit"] = literal_level
        flat[f"{letter}All"] = abstract_level * adjusted
    flat["MDS"] = mds_from_scores(scores)
    return flat


def multi_dimensional_similarity(testset: List[Dict[str, Any]],
                                 context: EvaluationContext,
                                 progress: bool = True
                                 ) -> Tuple[Dict[str, float], Dict[str, float],
                                            Dict[str, float], List[Dict[str, Any]]]:
    """Aggregate abstract / literal / overall scores exactly as the original."""
    scored: List[Dict[str, Any]] = []
    for index, data in enumerate(testset, start=1):
        if progress:
            print(f"[{index}/{len(testset)}] {data.get('event_name', '?')} "
                  f"-> {data.get('analogy_event', '?')}")
        row = score_sample(data, context)
        if row is None:
            if progress:
                print("    ! skipped (no resolvable analogy event)")
            continue
        scored.append(row)

    abstract_score = {d: 0.0 for d in DIMENSIONS}
    literal_score = {d: 0.0 for d in DIMENSIONS}
    overall_score = {d: 0.0 for d in DIMENSIONS}
    overall_score["all"] = 0.0
    if not scored:
        return abstract_score, literal_score, overall_score, scored

    n = len(scored)
    for data in scored:
        per_dimension = {}
        for dimension in DIMENSIONS:
            abstract_level = data["score"][dimension]["abstract_level"]
            literal_level = data["score"][dimension]["literal_level"]
            abstract_score[dimension] += abstract_level / n
            literal_score[dimension] += literal_level / n
            adjusted = max(LITERAL_THRESHOLD - literal_level, 0.0)
            value = abstract_level * adjusted
            overall_score[dimension] += value / n
            per_dimension[dimension] = value
        overall_score["all"] += sum(
            DIMENSION_WEIGHTS[d] * per_dimension[d] for d in DIMENSIONS
        ) / n
    return abstract_score, literal_score, overall_score, scored


def main() -> None:
    configure_stdout()
    parser = argparse.ArgumentParser(
        description="Multi-dimensional similarity (MDS) evaluation, provider-neutral"
    )
    parser.add_argument("--testset", required=True, type=str,
                        help="jsonl with 'event_name', 'event_intro', 'analogy_event'")
    parser.add_argument("--model", type=str, default=None,
                        help="judge model (defaults to EVALUATION_MODEL / LLM_MODEL)")
    parser.add_argument("--provider", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pass1", action="store_true",
                        help="also compute Pass@1 (needs 'target_event', popular set)")
    parser.add_argument("--output", type=str, default=None,
                        help="write per-sample scores to this jsonl file")
    args = parser.parse_args()

    if args.provider:
        set_settings(load_settings(llm_provider=args.provider))

    testset = read_jsonl(args.testset)
    if args.limit:
        testset = testset[: args.limit]
    context = EvaluationContext.build(model=args.model, verbose=True)

    print(f"judge model: {context.llm.model} | literal-similarity tokenizer: "
          f"{tokenizer_backend()}")
    abstract_score, literal_score, overall_score, scored = multi_dimensional_similarity(
        testset, context
    )
    print(f"\nabstract score: {abstract_score}")
    print(f"literal score: {literal_score}")
    print(f"overall multi-dimensional similarity: {overall_score}")
    if context.score_parses:
        # A judge that stops answering with a bare digit is not an error, but it
        # is a change worth seeing: everything but "bare" was recovered from
        # prose. See parse_abstract_score.
        print(f"judgement formats: {dict(sorted(context.score_parses.items()))}")
    if args.pass1:
        print(f"pass@1: {pass_1(testset, context)}")
    if args.output:
        write_jsonl(args.output, scored)
        print(f"per-sample scores written to {args.output}")


if __name__ == "__main__":
    main()
