"""Two-stage Retrieval -- provider-neutral port of
``framework/retrieval-based/twostage_retrieval.py``.

Method (unchanged): retrieve the **top-10** most similar events from the event
pool by cosine similarity, then ask the LLM to pick the most appropriate
analogy from that candidate set (the paper's selection prompt).

Provider differences are the same as for Direct Retrieval (Gemini embeddings
instead of OpenAI ``text-embedding-3-small``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .cli import build_parser, make_context, run_over_dataset
from .common import BaselineContext, extract_analogy_answer, output_row
from .direct_retrieval import get_pool, get_similar_events
from . import prompts

METHOD_NAME = "Two-stage Retrieval"
TOP_K = 10  # the paper's candidate-set size for retrieval


def llm_choice(event: Dict[str, Any], candidate: List[Dict[str, Any]],
               context: BaselineContext) -> str:
    """``llm_choice`` of the original script (no stop sequence, as in the original)."""
    return context.predict(
        prompts.RETRIEVAL_CHOICE.format(
            input_event=f"{event['event_name']}: {event['event_intro']}",
            candidate_events="\n".join(
                f"{e['history_event_text']}: {e['history_intro_text']}" for e in candidate
            ),
        )
    )


def run(event: Dict[str, Any], context: Optional[BaselineContext] = None,
        pool: Optional[List[Dict[str, Any]]] = None,
        pool_limit: Optional[int] = None, top_k: int = TOP_K) -> Dict[str, Any]:
    context = context or BaselineContext.build(with_embeddings=True)
    pool = pool if pool is not None else get_pool(context, limit=pool_limit,
                                                  verbose=context.verbose)
    ranked = get_similar_events(event, context, pool)
    candidate = ranked[:top_k]
    answer = llm_choice(event, candidate, context) if candidate else ""
    return output_row(
        event,
        extract_analogy_answer(answer),
        candidate=[[e["history_event_text"] for e in candidate]],
    )


def main() -> None:
    parser = build_parser(f"{METHOD_NAME} (provider-neutral)")
    parser.add_argument("--pool-limit", type=int, default=None,
                        help="only embed the first N pool events (development)")
    args = parser.parse_args()
    context = make_context(args, needs_embeddings=True)
    pool = get_pool(context, limit=args.pool_limit, verbose=True)

    def method(event: Dict[str, Any], ctx: BaselineContext) -> Dict[str, Any]:
        return run(event, ctx, pool=pool)

    run_over_dataset(method, args, context)


if __name__ == "__main__":
    main()
