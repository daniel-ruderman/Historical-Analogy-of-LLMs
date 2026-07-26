"""Shared, **tracked** project defaults.

This file is committed to Git, so every collaborator gets the same research
defaults from a ``git pull`` -- nobody has to copy values into their own `.env`.

    .env                = local secrets and optional personal overrides (NOT tracked)
    project_defaults.py = shared non-secret project defaults          (tracked)

Resolution order used by :mod:`hal.config`:

    1. an environment variable (including anything loaded from `.env`)
    2. ``PROJECT_DEFAULTS`` below
    3. the code-level fallback in ``hal/config.py``

To change a default for the whole team, edit the value here and commit it.
To change it only for yourself, set the environment variable (or put it in
`.env`) -- that always wins.

**Never add a secret here.** ``SECRET_KEYS`` is enforced at import time.
"""

from __future__ import annotations

from typing import Dict, Optional

# Keys that must never have a tracked default: they are per-developer secrets
# and belong in a local `.env` only.
SECRET_KEYS = frozenset({
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
})

# ---------------------------------------------------------------------------
# Shared project defaults (non-secret). Keys are environment-variable names.
# ---------------------------------------------------------------------------
PROJECT_DEFAULTS: Dict[str, str] = {
    # --- providers ---------------------------------------------------------
    "LLM_PROVIDER": "gemini",
    "EMBEDDING_PROVIDER": "gemini",
    "SEARCH_PROVIDER": "wikipedia",

    # --- models ------------------------------------------------------------
    # The generation model and the embedding model are configured
    # independently: changing one does not imply changing the other.
    "LLM_MODEL": "gemini-2.5-flash",
    "EMBEDDING_MODEL": "gemini-embedding-001",

    # Per-role models are intentionally absent: each role falls back to
    # LLM_MODEL. To pin one for the whole team, add it here, e.g.
    #     "JUDGE_MODEL": "gemini-2.5-flash",
    # Recognised keys: GENERATOR_MODEL, CRITIC_MODEL, ANTI_ANALOGY_MODEL,
    # JUDGE_MODEL, SUMMARIZER_MODEL, BASELINE_MODEL, EVALUATION_MODEL.

    # Which Gemini generation surface to use: auto | models | interactions
    "GEMINI_API_SURFACE": "auto",

    # --- our agentic pipeline ---------------------------------------------
    "REFINEMENT_ROUNDS": "2",
    "MAX_CANDIDATES": "8",
    "MIN_CANDIDATES": "5",
    "REACT_MAX_STEPS": "4",
    "SEARCH_TOP_K": "4",

    # --- generation settings ----------------------------------------------
    # The paper runs its baselines at temperature 0.1 (Sec. 5.1).
    "LLM_TEMPERATURE": "0.1",
    # The original evaluation.py uses an effectively greedy judge.
    "EVALUATION_TEMPERATURE": "0.0",
    "MAX_OUTPUT_TOKENS": "2048",

    # --- quota / robustness ------------------------------------------------
    "MAX_RETRIES": "5",
    "RETRY_BASE_DELAY": "2.0",
    "RETRY_MAX_DELAY": "60.0",
    "REQUEST_DELAY": "0.0",

    # --- caching (never part of a research result) -------------------------
    "CACHE_ENABLED": "true",
    # Relative paths are resolved against the repository root, so the cache
    # location does not depend on the current working directory.
    "CACHE_DIR": ".hal_cache",

    # --- misc --------------------------------------------------------------
    "WIKI_LANG": "en",
}


def _validate() -> None:
    """Fail loudly if a secret ever gets a tracked default."""
    leaked = SECRET_KEYS & set(PROJECT_DEFAULTS)
    if leaked:
        raise RuntimeError(
            "Secrets must never have a tracked default; remove "
            f"{sorted(leaked)} from PROJECT_DEFAULTS and put them in a local "
            ".env instead."
        )


_validate()


def get_project_default(name: str) -> Optional[str]:
    """Tracked default for ``name``, or ``None`` when there is none."""
    if name in SECRET_KEYS:
        return None
    value = PROJECT_DEFAULTS.get(name)
    if value is None:
        return None
    value = str(value).strip()
    return value or None
