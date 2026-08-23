"""Direct Generation -- provider-neutral port of
``framework/generation-based/direct_generation.py``.

Method (unchanged): a single LLM call receives ``event_name`` + ``event_intro``
with the paper's one-shot prompt and returns one analogous historical event.
Only the API layer differs: ``llm_predict`` goes through
:class:`~hal.providers.base.LLMProvider` (Gemini by default) instead of
``ChatOpenAI``.

Deviation from the original, forced by chat-tuned models (see PROJECT.md): the
original passes ``stop=['\\n']`` to keep a completion model's answer to one
line. A chat-tuned model opens by restating the prompt's layout, so that stop
truncates the reply to ``"==== Answer"`` and the answer is never generated at
all. The stop is dropped and :func:`~gemini_baselines.common.extract_analogy_answer`
takes the name from the slot the prompt defines. The prompt itself is untouched.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .cli import build_parser, make_context, run_over_dataset
from .common import BaselineContext, extract_analogy_answer, output_row
from . import prompts

METHOD_NAME = "Direct Generation"


def get_analogy(event: Dict[str, Any], context: BaselineContext) -> str:
    """``get_analogy`` of the original script."""
    return context.predict(
        prompts.DIRECT_GENERATION.format(
            event=f"{event['event_name']}\n{event['event_intro']}"
        ),
    )


def run(event: Dict[str, Any], context: Optional[BaselineContext] = None) -> Dict[str, Any]:
    context = context or BaselineContext.build()
    answer = get_analogy(event, context)
    return output_row(event, extract_analogy_answer(answer), candidate=[])


def main() -> None:
    parser = build_parser(f"{METHOD_NAME} (provider-neutral)")
    args = parser.parse_args()
    context = make_context(args)
    run_over_dataset(run, args, context)


if __name__ == "__main__":
    main()
