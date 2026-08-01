"""Automatic evaluation of every implemented method.

Uses the original paper's methodology (Multi-Dimensional Similarity + Pass@1),
ported in :mod:`gemini_baselines.evaluation_mds`, with the LLM-based steps
routed through our configurable provider (``EVALUATION_MODEL``, which inherits
``LLM_MODEL``) instead of GPT-4.

    automatic_evaluation.methods  -- calls the existing method implementations
    automatic_evaluation.runner   -- generate / score / resume / aggregate

The original ``evaluation.py`` and ``framework/`` are untouched.
"""

from .methods import (
    ALL_METHODS,
    BASELINE_METHODS,
    METHOD_LABELS,
    GenerationConfig,
    MethodRunner,
    canonical,
    expand_methods,
)
from .runner import (
    COMPONENT_COLUMNS,
    RESULTS_DIR,
    AutomaticEvaluation,
    EvaluationConfig,
    QuotaExhausted,
    aggregate,
    is_quota_error,
    print_summary_table,
    write_summary_csv,
)

__all__ = [
    "METHOD_LABELS", "ALL_METHODS", "BASELINE_METHODS", "GenerationConfig",
    "MethodRunner", "canonical", "expand_methods",
    "AutomaticEvaluation", "EvaluationConfig", "QuotaExhausted",
    "aggregate", "write_summary_csv", "print_summary_table",
    "COMPONENT_COLUMNS", "RESULTS_DIR", "is_quota_error",
]
