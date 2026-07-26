"""Self-reflection Framework -- provider-neutral port of
``framework/generation-based/reflection_generation.py``.

Method (unchanged): two LLM modules cooperate.

* **Candidate Generator** -- proposes **5** candidates from the input event's
  four-dimension description.  It is a *conversational* chain: previous turns
  (and the reflections) stay in its context, so it can revise its own list.
* **Answer Reflector** -- either emits ``Final Answer: <event>`` or a
  ``Reflection: ...`` that tells the generator how to change the candidate set.

Loop: generate 5 candidates -> verify each through Wikipedia + summarise into
four dimensions -> reflect; while the reflector answers with a ``Reflection``,
feed it back to the generator and try again; finally force a ``Final Answer``.

The original builds the conversation with LangChain's ``LLMChain`` +
``ConversationBufferMemory`` (``human_prefix="Input"``, ``ai_prefix="Output"``).
:class:`ConversationMemory` below reproduces that buffer format exactly, so the
prompt the model sees is the same, without the LangChain dependency.

Difference from the original: a safety cap (``max_reflections``, default 5) on
the reflection loop, because the original ``while 'Reflection' in choice`` loop
can never terminate if the model always reflects. The paper reports that
reflection triggers in about 10% of cases, so the cap does not normally bind.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .cli import build_parser, make_context, run_over_dataset
from .common import (
    BaselineContext,
    event_analysis,
    four_dimension_text,
    get_candidate_details,
    output_row,
    parse_candidate_list,
)
from . import prompts

METHOD_NAME = "Self-reflection Framework"
N_CANDIDATES = 5           # the paper's candidate-set size for self-reflection
DEFAULT_WARMUP_ROUNDS = 0  # `warm_up_rounds = 0` in the original script
DEFAULT_MAX_REFLECTIONS = 5


@dataclass
class ConversationMemory:
    """LangChain ``ConversationBufferMemory`` buffer format, reimplemented.

    ``ConversationBufferMemory(human_prefix="Input", ai_prefix="Output")``
    renders as ``"Input: <input>\\nOutput: <output>"`` per turn, turns joined by
    newlines. Only the ``input`` variable is stored (``input_key="input"``).
    """

    human_prefix: str = "Input"
    ai_prefix: str = "Output"
    turns: List[Tuple[str, str]] = field(default_factory=list)

    def save(self, user_input: str, output: str) -> None:
        self.turns.append((user_input, output))

    @property
    def buffer(self) -> str:
        return "\n".join(
            f"{self.human_prefix}: {user}\n{self.ai_prefix}: {output}"
            for user, output in self.turns
        )


class CandidateGenerator:
    """The conversational candidate-proposing module."""

    def __init__(self, context: BaselineContext):
        self.context = context
        self.memory = ConversationMemory()

    def predict(self, input_type: str, user_input: str) -> str:
        prompt = prompts.REFLECTION_GET_CANDIDATE.format(
            chat_history=self.memory.buffer,
            input_type=input_type,
            input=user_input,
        )
        output = self.context.predict(prompt)
        self.memory.save(user_input, output)
        return output


def llm_choice(event: Dict[str, Any], candidate: List[Dict[str, Any]],
               context: BaselineContext, warm_up: bool = False,
               thought: str = "") -> str:
    """``llm_choice``: the Answer Reflector (warm-up variant only reflects)."""
    template = prompts.REFLECTION_WARMUP if warm_up else prompts.REFLECTION_CHOICE
    return context.predict(
        template.format(
            input_event=four_dimension_text(event),
            candidate_events="\n".join(four_dimension_text(e) for e in candidate),
            thought=thought,
        ),
        stop=["Input Event:"],
    )


def historical_analogy(event_dict: Dict[str, Any], context: BaselineContext,
                       warm_up_rounds: int = DEFAULT_WARMUP_ROUNDS,
                       max_reflections: int = DEFAULT_MAX_REFLECTIONS
                       ) -> Tuple[str, List[List[str]]]:
    """``historical_analogy`` of the original script.

    Returns ``(final_answer, candidate_sets_per_iteration)``.
    """
    generator = CandidateGenerator(context)
    event = event_analysis(dict(event_dict), context)

    raw = generator.predict("Input Event", four_dimension_text(event))
    candidate_set = parse_candidate_list(raw, context)
    candidate_history: List[List[str]] = [candidate_set]
    context.log(f"[candidates] {candidate_set}")

    candidate = get_candidate_details(candidate_set, context, analyze=True)
    if warm_up_rounds > 0:
        choice = llm_choice(event, candidate, context, warm_up=True)
        warm_up_rounds -= 1
    else:
        choice = llm_choice(event, candidate, context)

    reflections = 0
    while "Reflection" in choice and reflections < max_reflections:
        reflections += 1
        reflection_text = choice[choice.find("Reflection:") + len("Reflection:"):]
        context.log(f"[reflection {reflections}] {reflection_text.strip()[:200]}")
        raw = generator.predict("Reflection", reflection_text)
        candidate_set = parse_candidate_list(raw, context)
        candidate_history.append(candidate_set)
        candidate = get_candidate_details(candidate_set, context, analyze=True)
        if warm_up_rounds > 0:
            choice = llm_choice(event, candidate, context, warm_up=True)
            warm_up_rounds -= 1
        else:
            choice = llm_choice(event, candidate, context)

    if "Final Answer:" not in choice:
        choice += "\n\nFinal Answer:"
        choice += llm_choice(event, candidate, context, thought=choice)

    answer = choice[choice.find("Final Answer:") + len("Final Answer:"):].strip()
    return answer.split("\n")[0].strip(), candidate_history


def run(event: Dict[str, Any], context: Optional[BaselineContext] = None,
        warm_up_rounds: int = DEFAULT_WARMUP_ROUNDS,
        max_reflections: int = DEFAULT_MAX_REFLECTIONS) -> Dict[str, Any]:
    context = context or BaselineContext.build()
    answer, candidate_history = historical_analogy(
        event, context, warm_up_rounds=warm_up_rounds, max_reflections=max_reflections
    )
    return output_row(event, answer, candidate=candidate_history)


def main() -> None:
    parser = build_parser(f"{METHOD_NAME} (provider-neutral)")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP_ROUNDS,
                        help="forced reflection rounds before answering (paper: 0/2/5/10)")
    parser.add_argument("--max-reflections", type=int, default=DEFAULT_MAX_REFLECTIONS)
    args = parser.parse_args()
    context = make_context(args)

    def method(event: Dict[str, Any], ctx: BaselineContext) -> Dict[str, Any]:
        return run(event, ctx, warm_up_rounds=args.warmup,
                   max_reflections=args.max_reflections)

    run_over_dataset(method, args, context)


if __name__ == "__main__":
    main()
