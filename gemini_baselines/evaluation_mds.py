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
from typing import Any, Dict, List, Optional, Tuple

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


@dataclass
class EvaluationContext:
    llm: LLMProvider
    wiki: WikipediaHelper
    temperature: float = 0.0
    verbose: bool = False

    @classmethod
    def build(cls, model: Optional[str] = None, verbose: bool = False,
              search_provider=None) -> "EvaluationContext":
        from hal.providers.factory import get_llm, get_search_provider

        settings = get_settings()
        llm = get_llm(role="evaluation", model=model, settings=settings)
        provider = search_provider or get_search_provider(settings=settings)
        return cls(llm=llm, wiki=WikipediaHelper(provider, settings.wiki_max_chars),
                   temperature=settings.evaluation_temperature, verbose=verbose)

    def predict(self, prompt: str) -> str:
        return self.llm.generate(prompt, temperature=self.temperature) or ""


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
    parts = split_four_dimensions(context.predict(prompt))
    # evaluation.py stores the "summary" part under the key "topic".
    event["topic"] = parts["summary"]
    event["background"] = parts["background"]
    event["process"] = parts["process"]
    event["result"] = parts["result"]
    return event


# --------------------------------------------------------------------------
# Abstract similarity (LLM judge, 1-4)  /  literal similarity (Jaccard)
# --------------------------------------------------------------------------
def abstract_similarity(text1: str, text2: str, context: EvaluationContext) -> int:
    """The paper's 1-4 abstract-similarity judgement.

    Robustness difference from the original: an out-of-range score is clamped to
    [1, 4] (the original warns but returns it), and an unparseable answer scores
    1 instead of raising. Both keep a batch run alive without changing the
    metric for well-formed answers.
    """
    result = context.predict(prompts.EVAL_ABSTRACT_SIMILARITY.format(text1=text1,
                                                                     text2=text2))
    try:
        score = int((result or "").strip())
    except Exception:
        match = re.search(r"\d+", result or "")
        if not match:
            return 1
        score = int(match.group())
    if score > 4:
        print("error score")
    return max(1, min(4, score))


# ``jacc`` in the original; kept algorithmic (no LLM).
jacc = jaccard


# --------------------------------------------------------------------------
# Pass@1
# --------------------------------------------------------------------------
def pass_1(dataset: List[Dict[str, Any]], context: EvaluationContext) -> float:
    """Fraction of samples whose analogy matches the reference ``target_event``.

    As in the original: compare the Wikipedia title sets of both names.
    """
    hits = 0
    for data in dataset:
        target_events = context.wiki.search_titles(data.get("target_event", ""))
        analogy_events = context.wiki.search_titles(data.get("analogy_event", ""))
        if any(e in analogy_events for e in target_events):
            hits += 1
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


def score_sample(data: Dict[str, Any], context: EvaluationContext
                 ) -> Optional[Dict[str, Any]]:
    """Score one output row; ``None`` when the analogy event cannot be resolved."""
    input_event = extract_features(dict(data), context)
    analogy_name = (data.get("analogy_event") or "").strip()
    if not analogy_name:
        return None
    intro = context.wiki.summary(analogy_name)
    if not intro:
        return None
    analog_event = extract_features(
        {"event_name": analogy_name, "event_intro": intro}, context,
        input_example=input_event,
    )
    scores: Dict[str, Dict[str, float]] = {}
    for dimension in DIMENSIONS:
        scores[dimension] = {
            "abstract_level": abstract_similarity(
                input_event[dimension], analog_event[dimension], context
            ),
            "literal_level": jacc(input_event[dimension], analog_event[dimension]),
        }
    row = dict(data)
    row["score"] = scores
    row["mds"] = mds_from_scores(scores)
    return row


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
    if args.pass1:
        print(f"pass@1: {pass_1(testset, context)}")
    if args.output:
        write_jsonl(args.output, scored)
        print(f"per-sample scores written to {args.output}")


if __name__ == "__main__":
    main()
