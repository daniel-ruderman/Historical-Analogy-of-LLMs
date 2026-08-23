"""Generation with Summarizing -- provider-neutral port of
``framework/generation-based/summary_generation.py``.

Method (unchanged):

1. the input event is summarised into the paper's four dimensions
   (**summary/topic, background, process, result**);
2. the LLM proposes **10** candidates from the four-dimension description;
3. each candidate is verified through Wikipedia *and* summarised into the same
   four dimensions;
4. the LLM selects the best analogy by comparing the structured summaries
   instead of the raw descriptions.

Only the API layer differs.

Note: the original ``summary_generation.py`` does not import ``llm_tools``, so
running it as-is raises ``NameError`` -- this port fixes the missing import
implicitly by routing through the provider abstraction. No prompt or step was
changed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .cli import build_parser, make_context, run_over_dataset
from .common import (
    BaselineContext,
    extract_analogy_answer,
    event_analysis,
    four_dimension_text,
    get_candidate_details,
    output_row,
    parse_candidate_list,
)
from . import prompts

METHOD_NAME = "Generation with Summarizing"
N_CANDIDATES = 10


def get_candidate(event: Dict[str, Any], context: BaselineContext) -> List[str]:
    """``get_candidate``: candidates proposed from the four-dimension summary."""
    candidate_text = context.predict(
        prompts.SUMMARY_GET_CANDIDATE.format(event=four_dimension_text(event))
    )
    context.log(f"[candidates] {candidate_text.strip()[:200]}")
    return parse_candidate_list(candidate_text, context)


def llm_choice(event: Dict[str, Any], candidate: List[Dict[str, Any]],
               context: BaselineContext) -> str:
    """``llm_choice``: selection over four-dimension summaries."""
    return context.predict(
        prompts.SUMMARY_CHOICE.format(
            input_event=four_dimension_text(event),
            candidate_events="\n".join(four_dimension_text(e) for e in candidate),
        ),
    )


def run(event: Dict[str, Any], context: Optional[BaselineContext] = None) -> Dict[str, Any]:
    context = context or BaselineContext.build()
    event = event_analysis(dict(event), context)
    candidate = get_candidate(event, context)
    candidate_with_details = get_candidate_details(candidate, context, analyze=True)
    if not candidate_with_details:
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
