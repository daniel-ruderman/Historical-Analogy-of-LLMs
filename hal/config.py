"""Configuration.

Everything that could reasonably change between experiments lives here, so that
the *method* code never mentions a concrete provider or model name.

Values are resolved in this order:

    1. an environment variable (including anything loaded from a local `.env`)
    2. :data:`hal.project_defaults.PROJECT_DEFAULTS` -- tracked, shared with the
       team through Git
    3. the code-level fallback constants below

So a collaborator who clones the repository gets the project's defaults
automatically and only has to supply the one thing that must stay local: their
own ``GEMINI_API_KEY``. Secrets never have a tracked default.

The important variables (see ``.env.example``)::

    GEMINI_API_KEY                      (secret -- local `.env` only)
    LLM_PROVIDER / LLM_MODEL
    EMBEDDING_PROVIDER / EMBEDDING_MODEL
    SEARCH_PROVIDER
    GENERATOR_MODEL / CRITIC_MODEL / ANTI_ANALOGY_MODEL / JUDGE_MODEL /
    SUMMARIZER_MODEL / BASELINE_MODEL / EVALUATION_MODEL
    REFINEMENT_ROUNDS / MAX_CANDIDATES

Per-role model variables default to ``LLM_MODEL`` when unset, so a single
variable is enough to switch the whole project to another model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from .project_defaults import SECRET_KEYS, get_project_default

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset"

# Roles that may optionally use a different model.
ROLES = (
    "generator",       # Generate/Search agent
    "critic",          # Critic agent
    "anti_analogy",    # Anti-Analogy agent
    "judge",           # Final Judge (not an agent)
    "summarizer",      # Final Summarizer
    "baseline",        # original-paper baseline methods
    "evaluation",      # MDS abstract-similarity judging
)

_ROLE_PROVIDER_ENV = {
    "generator": "GENERATOR_PROVIDER",
    "critic": "CRITIC_PROVIDER",
    "anti_analogy": "ANTI_ANALOGY_PROVIDER",
    "judge": "JUDGE_PROVIDER",
    "summarizer": "SUMMARIZER_PROVIDER",
    "baseline": "BASELINE_PROVIDER",
    "evaluation": "EVALUATION_PROVIDER",
}

_ROLE_ENV = {
    "generator": "GENERATOR_MODEL",
    "critic": "CRITIC_MODEL",
    "anti_analogy": "ANTI_ANALOGY_MODEL",
    "judge": "JUDGE_MODEL",
    "summarizer": "SUMMARIZER_MODEL",
    "baseline": "BASELINE_MODEL",
    "evaluation": "EVALUATION_MODEL",
}

# Code-level fallbacks. The values actually used come from
# hal/project_defaults.py (tracked) unless an environment variable overrides
# them; these constants only matter if a key is removed from PROJECT_DEFAULTS.
# The generation model and the embedding model are independent settings.
DEFAULT_LLM_PROVIDER = get_project_default("LLM_PROVIDER") or "local"
DEFAULT_EMBEDDING_PROVIDER = get_project_default("EMBEDDING_PROVIDER") or "gemini"
DEFAULT_SEARCH_PROVIDER = get_project_default("SEARCH_PROVIDER") or "wikipedia"
DEFAULT_LLM_MODEL = get_project_default("LLM_MODEL") or "qwen3:8b"
DEFAULT_EMBEDDING_MODEL = get_project_default("EMBEDDING_MODEL") or "gemini-embedding-001"

# The paper (Sec. 5.1) runs the baselines with temperature 0.1.
DEFAULT_TEMPERATURE = 0.1
# evaluation.py in the original repository uses an (effectively) greedy GPT-4.
DEFAULT_EVAL_TEMPERATURE = 0.0


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    """Resolve ``name``: environment -> tracked project default -> ``default``."""
    value = os.environ.get(name)
    if value is not None:
        value = value.strip()
        if value:
            return value
    tracked = get_project_default(name)
    if tracked is not None:
        return tracked
    return default


def resolve(name: str, default: Optional[str] = None) -> Optional[str]:
    """Public form of the resolution order (useful in tests and scripts)."""
    return _env(name, default)


def config_source(name: str) -> str:
    """Where ``name`` currently comes from: ``env`` / ``project`` / ``fallback``."""
    if (os.environ.get(name) or "").strip():
        return "env"
    if get_project_default(name) is not None:
        return "project"
    return "fallback"


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(_env(name, str(default))))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(str(_env(name, str(default))))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    return raw.lower() in ("1", "true", "yes", "on")


def load_dotenv(path: Optional[Path] = None) -> None:
    """Load ``.env`` into ``os.environ`` without overriding existing values.

    Uses ``python-dotenv`` when available and falls back to a tiny parser so
    that the project also runs in a bare interpreter.
    """
    path = Path(path) if path is not None else REPO_ROOT / ".env"
    try:  # pragma: no cover - depends on the environment
        from dotenv import load_dotenv as _load

        _load(dotenv_path=str(path), override=False)
        return
    except Exception:
        pass
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class Settings:
    """A snapshot of the configuration."""

    # --- providers -------------------------------------------------------
    llm_provider: str = DEFAULT_LLM_PROVIDER
    llm_model: str = DEFAULT_LLM_MODEL
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_dimensions: Optional[int] = None
    search_provider: str = DEFAULT_SEARCH_PROVIDER
    role_models: Dict[str, str] = field(default_factory=dict)
    # Optional per-role provider. Lets the model that PRODUCES an analogy live
    # somewhere different from the model that JUDGES it -- e.g. generate on a
    # local server while keeping the API judge, so MDS stays comparable with
    # results scored earlier.
    role_providers: Dict[str, str] = field(default_factory=dict)

    # --- generation ------------------------------------------------------
    temperature: float = DEFAULT_TEMPERATURE
    evaluation_temperature: float = DEFAULT_EVAL_TEMPERATURE
    max_output_tokens: int = 2048

    # --- our agentic pipeline -------------------------------------------
    refinement_rounds: int = 2
    max_candidates: int = 8
    min_candidates: int = 5
    react_max_steps: int = 4
    search_top_k: int = 4

    # --- robustness / quota ---------------------------------------------
    max_retries: int = 5
    retry_base_delay: float = 2.0
    retry_max_delay: float = 60.0
    request_delay: float = 0.0

    # --- caching ---------------------------------------------------------
    cache_enabled: bool = True
    cache_dir: Path = REPO_ROOT / ".hal_cache"

    # --- local model server (LLM_PROVIDER=local) -------------------------
    local_base_url: str = "http://localhost:11434"
    local_api_style: str = "ollama"     # ollama | openai
    local_num_ctx: int = 10240          # context window; too small = silent truncation
    local_keep_alive: str = "30m"       # keep the model resident between calls
    local_think: str = "false"          # true | false | auto (model default)
    local_timeout: float = 600.0

    # --- misc ------------------------------------------------------------
    wiki_lang: str = "en"
    wiki_max_chars: int = 4096  # evaluation.py truncates summaries at 4096
    # Seconds to wait before each MediaWiki call. Separate from REQUEST_DELAY
    # because that one also paces the LLM, and a local Ollama needs no pacing.
    wiki_request_delay: float = 0.2
    api_key: Optional[str] = None
    gemini_api_surface: str = "auto"  # auto | models | interactions

    def model_for(self, role: str) -> str:
        """Return the model configured for ``role`` (defaults to ``llm_model``)."""
        return self.role_models.get(role) or self.llm_model

    def provider_for(self, role: str) -> str:
        """Return the provider for ``role`` (defaults to ``llm_provider``)."""
        return self.role_providers.get(role) or self.llm_provider

    def describe(self) -> str:
        roles = ", ".join(f"{r}={self.model_for(r)}" for r in ROLES)
        return (
            f"llm={self.llm_provider}:{self.llm_model} "
            f"embed={self.embedding_provider}:{self.embedding_model} "
            f"search={self.search_provider}\n  roles: {roles}"
        )


def _cache_dir() -> Path:
    """Cache directory; relative values resolve against the repository root.

    That keeps the tracked ``CACHE_DIR`` default (``.hal_cache``) pointing at
    the same place regardless of the current working directory.
    """
    raw = _env("CACHE_DIR", ".hal_cache") or ".hal_cache"
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path


def load_settings(**overrides) -> Settings:
    """Build :class:`Settings` from the environment, applying ``overrides``."""
    load_dotenv()

    llm_model = _env("LLM_MODEL", DEFAULT_LLM_MODEL)
    role_models = {}
    for role, env_name in _ROLE_ENV.items():
        value = _env(env_name)
        if value:
            role_models[role] = value

    role_providers = {}
    for role, env_name in _ROLE_PROVIDER_ENV.items():
        value = _env(env_name)
        if value:
            role_providers[role] = value

    dims = _env("EMBEDDING_DIMENSIONS")
    settings = Settings(
        llm_provider=_env("LLM_PROVIDER", DEFAULT_LLM_PROVIDER),
        llm_model=llm_model,
        embedding_provider=_env("EMBEDDING_PROVIDER", DEFAULT_EMBEDDING_PROVIDER),
        embedding_model=_env("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        embedding_dimensions=int(dims) if dims else None,
        search_provider=_env("SEARCH_PROVIDER", DEFAULT_SEARCH_PROVIDER),
        role_models=role_models,
        role_providers=role_providers,
        temperature=_env_float("LLM_TEMPERATURE", DEFAULT_TEMPERATURE),
        evaluation_temperature=_env_float(
            "EVALUATION_TEMPERATURE", DEFAULT_EVAL_TEMPERATURE
        ),
        max_output_tokens=_env_int("MAX_OUTPUT_TOKENS", 2048),
        refinement_rounds=_env_int("REFINEMENT_ROUNDS", 2),
        max_candidates=_env_int("MAX_CANDIDATES", 8),
        min_candidates=_env_int("MIN_CANDIDATES", 5),
        react_max_steps=_env_int("REACT_MAX_STEPS", 4),
        search_top_k=_env_int("SEARCH_TOP_K", 4),
        max_retries=_env_int("MAX_RETRIES", 5),
        retry_base_delay=_env_float("RETRY_BASE_DELAY", 2.0),
        retry_max_delay=_env_float("RETRY_MAX_DELAY", 60.0),
        request_delay=_env_float("REQUEST_DELAY", 0.0),
        cache_enabled=_env_bool("CACHE_ENABLED", True),
        cache_dir=_cache_dir(),
        local_base_url=_env("LOCAL_BASE_URL", "http://localhost:11434"),
        local_api_style=_env("LOCAL_API_STYLE", "ollama"),
        local_num_ctx=_env_int("LOCAL_NUM_CTX", 10240),
        local_keep_alive=_env("LOCAL_KEEP_ALIVE", "30m"),
        local_think=_env("LOCAL_THINK", "false"),
        local_timeout=_env_float("LOCAL_TIMEOUT", 600.0),
        wiki_lang=_env("WIKI_LANG", "en"),
        wiki_request_delay=_env_float("WIKI_REQUEST_DELAY", 0.2),
        api_key=_env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY"),
        gemini_api_surface=_env("GEMINI_API_SURFACE", "auto"),
    )
    if overrides:
        from dataclasses import replace

        settings = replace(settings, **overrides)
    return settings


_SETTINGS: Optional[Settings] = None


def get_settings(refresh: bool = False) -> Settings:
    """Process-wide settings singleton."""
    global _SETTINGS
    if _SETTINGS is None or refresh:
        _SETTINGS = load_settings()
    return _SETTINGS


def set_settings(settings: Settings) -> None:
    """Install a :class:`Settings` instance (used by CLIs and tests)."""
    global _SETTINGS
    _SETTINGS = settings
