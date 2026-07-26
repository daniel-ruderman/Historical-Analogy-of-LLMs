"""Shared tracked defaults vs. local `.env` secrets.

Covers the collaboration contract:

    hal/project_defaults.py  = tracked, shared through Git
    .env / environment       = local secrets and personal overrides, and always wins
    secrets                  = never have a tracked default
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hal import config as config_module
from hal.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MODEL,
    REPO_ROOT,
    Settings,
    config_source,
    load_settings,
    resolve,
)
from hal.project_defaults import (
    PROJECT_DEFAULTS,
    SECRET_KEYS,
    _validate,
    get_project_default,
)

ROLE_ENV_VARS = [
    "GENERATOR_MODEL", "CRITIC_MODEL", "ANTI_ANALOGY_MODEL", "JUDGE_MODEL",
    "SUMMARIZER_MODEL", "BASELINE_MODEL", "EVALUATION_MODEL",
]


# --- the tracked defaults -------------------------------------------------
def test_team_default_model_is_tracked_in_the_repository():
    assert PROJECT_DEFAULTS["LLM_MODEL"] == "gemini-2.5-flash"
    assert PROJECT_DEFAULTS["LLM_PROVIDER"] == "gemini"


def test_tracked_defaults_apply_without_any_env_var():
    """A fresh clone with no model values in `.env` still gets the team default."""
    settings = load_settings()
    assert settings.llm_provider == "gemini"
    assert settings.llm_model == "gemini-2.5-flash"
    assert settings.embedding_provider == "gemini"
    assert settings.embedding_model == "gemini-embedding-001"
    assert settings.refinement_rounds == 2
    assert settings.max_candidates == 8
    assert config_source("LLM_MODEL") == "project"


def test_every_role_defaults_to_the_tracked_model():
    settings = load_settings()
    for role in ("generator", "critic", "anti_analogy", "judge", "summarizer",
                 "baseline", "evaluation"):
        assert settings.model_for(role) == "gemini-2.5-flash"


def test_code_level_fallbacks_agree_with_the_tracked_defaults():
    assert DEFAULT_LLM_MODEL == PROJECT_DEFAULTS["LLM_MODEL"]
    assert DEFAULT_EMBEDDING_MODEL == PROJECT_DEFAULTS["EMBEDDING_MODEL"]
    assert Settings().llm_model == "gemini-2.5-flash"


def test_generation_and_embedding_models_are_independent(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "some-other-llm")
    settings = load_settings()
    assert settings.llm_model == "some-other-llm"
    assert settings.embedding_model == "gemini-embedding-001"  # unchanged


# --- local overrides win --------------------------------------------------
def test_environment_overrides_the_tracked_default(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "some-other-model")
    assert load_settings().llm_model == "some-other-model"
    assert config_source("LLM_MODEL") == "env"


def test_dotenv_values_override_tracked_defaults(tmp_path, monkeypatch):
    """`.env` reaches the settings the same way a real environment variable does."""
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_MODEL=model-from-dotenv\n", encoding="utf-8")
    config_module.load_dotenv(env_file)
    try:
        assert load_settings().llm_model == "model-from-dotenv"
    finally:
        monkeypatch.delenv("LLM_MODEL", raising=False)


def test_role_specific_override_still_works(monkeypatch):
    monkeypatch.setenv("JUDGE_MODEL", "judge-only-model")
    settings = load_settings()
    assert settings.model_for("judge") == "judge-only-model"
    assert settings.model_for("critic") == "gemini-2.5-flash"
    assert settings.model_for("generator") == "gemini-2.5-flash"


@pytest.mark.parametrize("env_var", ROLE_ENV_VARS)
def test_each_role_env_var_is_honoured(monkeypatch, env_var):
    monkeypatch.setenv(env_var, f"model-for-{env_var}")
    settings = load_settings()
    assert f"model-for-{env_var}" in settings.role_models.values()


def test_blank_env_var_falls_back_to_the_tracked_default(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "   ")
    assert load_settings().llm_model == "gemini-2.5-flash"


def test_numeric_and_boolean_overrides(monkeypatch):
    monkeypatch.setenv("REFINEMENT_ROUNDS", "3")
    monkeypatch.setenv("CACHE_ENABLED", "false")
    settings = load_settings()
    assert settings.refinement_rounds == 3
    assert settings.cache_enabled is False


def test_resolution_order_is_env_then_project_then_fallback(monkeypatch):
    assert resolve("LLM_MODEL") == "gemini-2.5-flash"          # project
    monkeypatch.setenv("LLM_MODEL", "env-model")
    assert resolve("LLM_MODEL") == "env-model"                  # env wins
    assert resolve("NOT_A_TRACKED_KEY", "code-fallback") == "code-fallback"
    assert config_source("NOT_A_TRACKED_KEY") == "fallback"


# --- secrets stay local ---------------------------------------------------
def test_no_secret_has_a_tracked_default():
    for key in SECRET_KEYS:
        assert key not in PROJECT_DEFAULTS
        assert get_project_default(key) is None


def test_adding_a_secret_to_the_tracked_defaults_is_rejected(monkeypatch):
    monkeypatch.setitem(PROJECT_DEFAULTS, "GEMINI_API_KEY", "AIza-not-a-real-key")
    with pytest.raises(RuntimeError, match="Secrets must never have a tracked default"):
        _validate()


def test_api_key_must_come_from_the_local_environment(monkeypatch):
    assert load_settings().api_key is None      # nothing tracked provides it
    monkeypatch.setenv("GEMINI_API_KEY", "local-key-from-env")
    assert load_settings().api_key == "local-key-from-env"


def test_google_api_key_is_accepted_as_an_alternative(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "local-google-key")
    assert load_settings().api_key == "local-google-key"


def test_no_api_key_literal_appears_in_tracked_files():
    """Guard against a real key ever being committed."""
    import re

    pattern = re.compile(r"AIza[0-9A-Za-z_\-]{30,}|sk-[A-Za-z0-9]{30,}")
    roots = ["hal", "gemini_baselines", "agentic_pipeline", "examples", "tests"]
    files = [p for root in roots for p in (REPO_ROOT / root).rglob("*.py")]
    files += [REPO_ROOT / ".env.example", REPO_ROOT / "PROJECT.md"]
    for path in files:
        if path.exists():
            assert not pattern.search(path.read_text(encoding="utf-8")), path


def test_env_example_documents_the_secret_without_a_value():
    text = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    assert "GEMINI_API_KEY=" in text
    # the key line must be an empty placeholder
    line = next(l for l in text.splitlines()
                if l.strip().startswith("GEMINI_API_KEY="))
    assert line.strip() == "GEMINI_API_KEY="
    # non-secret defaults are documented as optional (commented out)
    assert "# LLM_MODEL=gemini-2.5-flash" in text


def test_env_is_git_ignored():
    ignored = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    entries = {line.strip() for line in ignored}
    assert ".env" in entries
    assert ".env.example" not in entries   # the template must stay tracked


# --- cache dir ------------------------------------------------------------
def test_relative_cache_dir_resolves_against_the_repository_root():
    assert load_settings().cache_dir == REPO_ROOT / ".hal_cache"


def test_absolute_cache_dir_is_respected(monkeypatch, tmp_path):
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "elsewhere"))
    assert load_settings().cache_dir == Path(tmp_path / "elsewhere")
