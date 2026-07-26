"""hal -- Historical Analogy Lab.

Shared infrastructure for our extension of "Past Meets Present: Creating
Historical Analogy with Large Language Models".

This package contains *no research method*.  It only provides the pieces that
every method needs:

* :mod:`hal.config`      -- environment-driven configuration (models, rounds, ...)
* :mod:`hal.providers`   -- LLMProvider / EmbeddingProvider / SearchProvider
* :mod:`hal.schemas`     -- structured objects passed between components
* :mod:`hal.cache`       -- on-disk cache for embeddings and Wikipedia pages
* :mod:`hal.wiki`        -- Wikipedia access (verification + search)
* :mod:`hal.json_utils`  -- tolerant parsing of LLM JSON output
* :mod:`hal.io_utils`    -- jsonl helpers and dataset paths
* :mod:`hal.text_similarity` -- Jaccard / literal similarity (no LLM involved)

The original paper implementation under ``framework/`` and ``evaluation.py`` is
left untouched; it is kept as the reference implementation.
"""

__all__ = ["config", "providers", "schemas"]

__version__ = "0.1.0"
