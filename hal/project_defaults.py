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
    # Generation runs on a LOCAL model server (Ollama); see the local settings
    # below and section 9 of CLAUDE.md. Embeddings deliberately stay on the
    # Gemini API -- they are a separate setting, only the two retrieval
    # baselines use them, and keeping them fixed preserves comparability with
    # every earlier result. That means GEMINI_API_KEY is still required for
    # direct_retrieval / twostage_retrieval, but for nothing else.
    "LLM_PROVIDER": "local",
    "EMBEDDING_PROVIDER": "gemini",
    "SEARCH_PROVIDER": "wikipedia",

    # --- models ------------------------------------------------------------
    # The generation model and the embedding model are configured
    # independently: changing one does not imply changing the other.
    #
    # qwen3:8b for EVERY role, including the judge -- one model keeps the
    # experimental condition clean. Requires `ollama pull qwen3:8b`.
    #
    # Chosen 2026-08-22 from a head-to-head on all 20 popular examples
    # (agentic method, reduced settings, judged by gemma-4-31b-it):
    #     qwen3:8b     MDS 4.002   Pass@1 0.30   scored 20/20
    #     llama3.1:8b  MDS 3.824   Pass@1 0.10   scored 20/20
    #     gemma-4-31b  MDS 4.110   Pass@1 0.29   scored 17/20  (API, full settings)
    # The three are close; qwen3 wins on Pass@1 and edges MDS, and both local
    # models completed every example where the 31B API model lost three.
    # qwen3 is ~35% slower than llama3.1 and produced one self-analogy in 20.
    "LLM_MODEL": "qwen3:8b",
    "EMBEDDING_MODEL": "gemini-embedding-001",   # Ollama models serve no embeddings

    # Per-role models are intentionally absent: each role falls back to
    # LLM_MODEL, so the judge is qwen3:8b too. To pin one, add it here, e.g.
    #     "JUDGE_MODEL": "qwen3:8b",
    # Recognised keys: GENERATOR_MODEL, CRITIC_MODEL, ANTI_ANALOGY_MODEL,
    # JUDGE_MODEL, SUMMARIZER_MODEL, BASELINE_MODEL, EVALUATION_MODEL.
    # Per-role PROVIDERS exist too (GENERATOR_PROVIDER, EVALUATION_PROVIDER,
    # ...), which is how you would keep an API judge while generating locally.

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
    # Reasoning models spend output tokens on internal thinking BEFORE the
    # answer, and this budget covers both. Measured: Gemma 4 uses ~300 thinking
    # tokens even for a one-line answer, so a small budget yields an empty
    # response. 4096 leaves room for thinking plus the long JSON the agents emit.
    "MAX_OUTPUT_TOKENS": "4096",

    # --- quota / robustness ------------------------------------------------
    "MAX_RETRIES": "5",
    "RETRY_BASE_DELAY": "2.0",
    "RETRY_MAX_DELAY": "60.0",
    "REQUEST_DELAY": "0.0",

    # --- local model server (used when LLM_PROVIDER=local) -----------------
    # Ollama by default. `ollama` is the only API style that lets us set the
    # context window per request, which is why it is the default.
    "LOCAL_BASE_URL": "http://localhost:11434",
    "LOCAL_API_STYLE": "ollama",          # ollama | openai
    # MEASURED, not guessed. One full-settings agentic example on qwen3:8b
    # (140 calls) gave prompt sizes: mean 1265, p90 1473, max 6572 tokens.
    # Ollama's own default is 2k-4k and TRUNCATES SILENTLY -- it drops the START
    # of the prompt and answers anyway, so the model loses the task and the input
    # event while still seeing the trailing "reply with JSON" instruction. The
    # result is well-formed JSON about the wrong thing. hal/providers/local.py
    # now checks prompt_eval_count after every call and raises instead.
    #
    # Context costs VRAM. Measured on an 8 GB card (RTX 2070 SUPER, qwen3:8b Q4):
    #     8192  -> 6.2 GB, 100% on GPU
    #    10240  -> 6.9 GB,  92% on GPU
    #    16384  -> 7.8 GB,  80% on GPU  (~45% slower per call)
    # 10240 clears the measured 6572-token worst case with ~3.6k to spare while
    # keeping nearly all layers on the GPU. Raise it only if the guard fires.
    "LOCAL_NUM_CTX": "10240",
    # Keep the model loaded between calls: the agentic pipeline makes dozens of
    # calls per example and reloading would dominate the runtime.
    "LOCAL_KEEP_ALIVE": "30m",
    # Reasoning models think before answering. Measured on qwen3:8b: 20-24 s
    # with thinking vs 6 s without, for the same answer -- 3-4x the wall time
    # across a run that makes tens of thousands of calls. Off by default;
    # set true to test whether thinking improves analogy quality.
    "LOCAL_THINK": "false",
    "LOCAL_TIMEOUT": "600",

    # --- caching (never part of a research result) -------------------------
    "CACHE_ENABLED": "true",
    # Relative paths are resolved against the repository root, so the cache
    # location does not depend on the current working directory.
    "CACHE_DIR": ".hal_cache",

    # --- misc --------------------------------------------------------------
    "WIKI_LANG": "en",
    # Seconds to wait before each MediaWiki request. The popular run of
    # 2026-08-22 issued thousands of unpaced Wikipedia calls over ~10 hours and
    # was throttled so persistently that 1048 lookups failed even after three
    # retries each -- and, because failures were cached, every one of those
    # events became permanently "nonexistent" (see PROJECT.md). This paces the
    # calls to ~5/s. Deliberately separate from REQUEST_DELAY, which also paces
    # the LLM: a local Ollama needs no pacing at all.
    "WIKI_REQUEST_DELAY": "0.2",
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
