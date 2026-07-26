"""Provider-neutral (Gemini-by-default) re-implementations of the paper's methods.

The original implementation under ``framework/`` is untouched and remains the
reference.  Each module here mirrors one original script one-to-one:

======================  =================================================
this package            original file
======================  =================================================
direct_retrieval        framework/retrieval-based/direct_retrieval.py
twostage_retrieval      framework/retrieval-based/twostage_retrieval.py
direct_generation       framework/generation-based/direct_generation.py
twostage_generation     framework/generation-based/twostage_generation.py
summary_generation      framework/generation-based/summary_generation.py
reflection_generation   framework/generation-based/reflection_generation.py
evaluation_mds          evaluation.py
prompts                 all prompt templates, copied verbatim
======================  =================================================

These are **baselines**: they must stay faithful.  Do not add critics,
anti-analogies or extra search here -- that is what ``agentic_pipeline/`` is for.
"""

from typing import Callable, Dict

METHODS: Dict[str, str] = {
    "direct_retrieval": "Direct Retrieval",
    "twostage_retrieval": "Two-stage Retrieval",
    "direct_generation": "Direct Generation",
    "twostage_generation": "Two-stage Generation",
    "summary_generation": "Generation with Summarizing",
    "reflection_generation": "Self-reflection Framework",
}


def get_method(name: str) -> Callable:
    """Return the ``run`` callable of a baseline module by name."""
    import importlib

    if name not in METHODS:
        raise ValueError(f"Unknown baseline {name!r}. Known: {sorted(METHODS)}")
    module = importlib.import_module(f"{__name__}.{name}")
    return module.run


__all__ = ["METHODS", "get_method"]
