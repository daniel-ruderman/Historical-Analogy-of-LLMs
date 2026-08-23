"""Two-stage Generation -- provider-neutral port of
``framework/generation-based/twostage_generation.py``.

Method (unchanged):

1. one LLM call proposes **10** candidate events (Python-list format; the
   original's repair prompt is reused when parsing fails);
2. every candidate is verified through Wikipedia and replaced by its summary --
   candidates without a page are dropped (hallucination filter);
3. a second LLM call selects the most appropriate candidate.

Only the API layer differs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .cli import build_parser, make_context, run_over_dataset
from .common import (
    BaselineContext,
    extract_analogy_answer,
    get_candidate_details,
    output_row,
    parse_candidate_list,
)
from . import prompts

METHOD_NAME = "Two-stage Generation"
N_CANDIDATES = 10  # the paper's candidate count for this method


def get_candidate(event: Dict[str, Any], context: BaselineContext) -> List[str]:
    """``get_candidate`` of the original script (10 candidates)."""
    candidate_text = context.predict(
        prompts.TWOSTAGE_GET_CANDIDATE.format(
            event=f"{event['event_name']}\n{event['event_intro']}"
        )
    )
    context.log(f"[candidates] {candidate_text.strip()[:200]}")
    return parse_candidate_list(candidate_text, context)


def llm_choice(event: Dict[str, Any], candidate: List[Dict[str, Any]],
               context: BaselineContext) -> str:
    """``llm_choice`` of the original script."""
    return context.predict(
        prompts.TWOSTAGE_CHOICE.format(
            input_event=f"{event['event_name']}: {event['event_intro']}",
            candidate_events="\n".join(
                f"{e['event_name']}: {e['event_intro']}" for e in candidate
            ),
            input_name=event["event_name"],
        ),
    )


def run(event: Dict[str, Any], context: Optional[BaselineContext] = None) -> Dict[str, Any]:
    context = context or BaselineContext.build()
    candidate = get_candidate(event, context)
    candidate_with_details = get_candidate_details(candidate, context)
    if not candidate_with_details:
        # Nothing survived Wikipedia verification; the original would crash on
        # an empty candidate list, we return an empty answer instead.
        return output_row(event, "", candidate=[candidate])
    answer = llm_choice(event, candidate_with_details, context)
    return output_row(event, extract_analogy_answer(answer), candidate=[candidate])


def main() -> None:
    parser = build_parser(f"{METHOD_NAME} (provider-neutral)")
    args = parser.parse_args()
    context = make_context(args)
    run_over_dataset(run, args, context)


if __name__ == "__main__":
    main()
