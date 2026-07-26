"""Test fixtures. No test in this directory performs a real API call."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hal.config import Settings, set_settings  # noqa: E402
from hal.providers.mock import (  # noqa: E402
    MockEmbeddingProvider,
    MockLLMProvider,
    MockSearchProvider,
)
from hal.wiki import WikipediaHelper  # noqa: E402


@pytest.fixture(autouse=True)
def offline_settings(tmp_path, monkeypatch):
    """Every test runs with mock providers, caching in a temp dir, no API key."""
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY",
                 "LLM_PROVIDER", "LLM_MODEL", "EMBEDDING_PROVIDER", "EMBEDDING_MODEL",
                 "REFINEMENT_ROUNDS", "MAX_CANDIDATES", "GENERATOR_MODEL",
                 "CRITIC_MODEL", "ANTI_ANALOGY_MODEL", "JUDGE_MODEL",
                 "SUMMARIZER_MODEL", "BASELINE_MODEL", "EVALUATION_MODEL"):
        monkeypatch.delenv(name, raising=False)
    settings = Settings(
        llm_provider="mock",
        llm_model="mock-model",
        embedding_provider="mock",
        embedding_model="mock-embedding",
        search_provider="mock",
        cache_dir=tmp_path / "cache",
        refinement_rounds=1,
        max_candidates=3,
        react_max_steps=2,
    )
    set_settings(settings)
    yield settings


@pytest.fixture
def wiki_pages():
    return {
        "Spanish flu": "The 1918-1920 influenza pandemic killed tens of millions "
                       "worldwide and disrupted societies across the globe.",
        "French Revolution": "A period of political upheaval in France beginning in "
                             "1789 that overthrew the monarchy.",
        "Revolutions of 1848": "A wave of revolutions across Europe in 1848, most of "
                               "which were reversed within two years.",
        "Cold War": "A period of geopolitical tension between the United States and "
                    "the Soviet Union after the Second World War.",
    }


@pytest.fixture
def mock_search(wiki_pages):
    return MockSearchProvider(dict(wiki_pages))


@pytest.fixture
def mock_wiki(mock_search):
    return WikipediaHelper(mock_search, max_chars=4096)


@pytest.fixture
def mock_embeddings():
    return MockEmbeddingProvider()


@pytest.fixture
def sample_event():
    return {
        "event_name": "Arab Spring",
        "event_intro": "A series of anti-government protests, uprisings and armed "
                       "rebellions that spread across much of the Arab world in the "
                       "early 2010s, beginning in Tunisia.",
        "target_event": "Revolutions of 1848",
    }


def make_llm(responses=None, **kwargs) -> MockLLMProvider:
    return MockLLMProvider(responses=responses, **kwargs)
