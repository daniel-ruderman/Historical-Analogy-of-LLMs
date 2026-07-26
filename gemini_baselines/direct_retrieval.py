"""Direct Retrieval -- provider-neutral port of
``framework/retrieval-based/direct_retrieval.py``.

Method (unchanged): embed the input event's description, embed every event of
the 658-event Google Arts & Culture pool, and return the pool event with the
highest cosine similarity (excluding the input event itself). No LLM involved.

Provider difference: the original uses OpenAI ``text-embedding-3-small`` and
ships pre-computed vectors in ``dataset/similarity_embeddings-example.jsonl``.
We compute the vectors with the configured ``EmbeddingProvider``
(``gemini-embedding-001`` by default) and cache them on disk, keyed by
provider/model. The *algorithm* -- cosine similarity over description
embeddings, top-1 -- is identical; the numbers are not comparable across
embedding models.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .cli import build_parser, make_context, run_over_dataset
from .common import BaselineContext, load_pool_with_embeddings, output_row, rank_pool_events

METHOD_NAME = "Direct Retrieval"

_POOL_CACHE: Dict[int, List[Dict[str, Any]]] = {}


def get_pool(context: BaselineContext, limit: Optional[int] = None,
             verbose: bool = False) -> List[Dict[str, Any]]:
    """Load + embed the event pool once per process (vectors are cached on disk)."""
    key = limit or 0
    if key not in _POOL_CACHE:
        def progress(done: int, total: int) -> None:
            if verbose:
                print(f"    embedding event pool: {done}/{total}")

        _POOL_CACHE[key] = load_pool_with_embeddings(context, limit=limit,
                                                     progress=progress)
    return _POOL_CACHE[key]


def get_similar_events(event: Dict[str, Any], context: BaselineContext,
                       pool: Optional[List[Dict[str, Any]]] = None
                       ) -> List[Dict[str, Any]]:
    """``get_similar_events`` of the original script."""
    pool = pool if pool is not None else get_pool(context)
    return rank_pool_events(event, pool, context)


def run(event: Dict[str, Any], context: Optional[BaselineContext] = None,
        pool: Optional[List[Dict[str, Any]]] = None,
        pool_limit: Optional[int] = None) -> Dict[str, Any]:
    context = context or BaselineContext.build(with_embeddings=True)
    pool = pool if pool is not None else get_pool(context, limit=pool_limit,
                                                  verbose=context.verbose)
    ranked = get_similar_events(event, context, pool)
    result = ranked[0]["history_event_text"] if ranked else ""
    return output_row(event, result, candidate=[])


def main() -> None:
    parser = build_parser(f"{METHOD_NAME} (provider-neutral)", needs_llm=False)
    parser.add_argument("--pool-limit", type=int, default=None,
                        help="only embed the first N pool events (development)")
    args = parser.parse_args()
    context = make_context(args, needs_embeddings=True, needs_llm=False)
    pool = get_pool(context, limit=args.pool_limit, verbose=True)

    def method(event: Dict[str, Any], ctx: BaselineContext) -> Dict[str, Any]:
        return run(event, ctx, pool=pool)

    run_over_dataset(method, args, context)


if __name__ == "__main__":
    main()
