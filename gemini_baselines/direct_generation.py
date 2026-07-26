"""Direct Generation -- provider-neutral port of
``framework/generation-based/direct_generation.py``.

Method (unchanged): a single LLM call receives ``event_name`` + ``event_intro``
with the paper's one-shot prompt and returns one analogous historical event.
``stop=['\\n']`` keeps the answer to a single line, as in the original.

Only the API layer differs: ``llm_predict`` goes through
:class:`~hal.providers.base.LLMProvider` (Gemini by default) instead of
``ChatOpenAI``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .cli import build_parser, make_context, run_over_dataset
from .common import BaselineContext, clean_answer, output_row
from . import prompts

METHOD_NAME = "Direct Generation"


def get_analogy(event: Dict[str, Any], context: BaselineContext) -> str:
    """``get_analogy`` of the original script."""
    return context.predict(
        prompts.DIRECT_GENERATION.format(
            event=f"{event['event_name']}\n{event['event_intro']}"
        ),
        stop=["\n"],
    )


def run(event: Dict[str, Any], context: Optional[BaselineContext] = None) -> Dict[str, Any]:
    context = context or BaselineContext.build()
    answer = get_analogy(event, context)
    return output_row(event, clean_answer(answer), candidate=[])


def main() -> None:
    parser = build_parser(f"{METHOD_NAME} (provider-neutral)")
    args = parser.parse_args()
    context = make_context(args)
    run_over_dataset(run, args, context)


if __name__ == "__main__":
    main()
