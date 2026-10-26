"""Experience RAG retrieval must never change behaviour when the library is empty.

The library ships empty and only fills up as evolution batches promote entries,
so the absent-index path is the common case, not an edge case.
"""

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import utils  # noqa: E402


@pytest.fixture(autouse=True)
def reset_experience_cache():
    """The cache latches after the first load, so clear it between tests."""
    utils.EXPERIENCE_DB_CACHE = None
    utils._EXPERIENCE_DB_LOADED = False
    yield
    utils.EXPERIENCE_DB_CACHE = None
    utils._EXPERIENCE_DB_LOADED = False


def test_missing_index_returns_no_results(monkeypatch, tmp_path):
    """An absent index degrades to [] rather than raising."""
    monkeypatch.setattr(
        utils, "load_experience_db", lambda config=None: None
    )
    assert utils.retrieve_experience("reactingFoam mixture") == []


def test_missing_index_does_not_load_an_embedding_model(monkeypatch):
    """Absence must be decided by a path check, before any model download."""
    def explode(*args, **kwargs):
        raise AssertionError("embedding model must not be constructed")

    monkeypatch.setattr(utils, "get_embedding_model", explode)
    monkeypatch.setattr(
        utils.Path, "exists", lambda self: False, raising=False
    )
    assert utils.load_experience_db() is None


def test_search_failure_is_swallowed(monkeypatch):
    """A corrupt index must not take the run down with it."""
    class Broken:
        def similarity_search_with_score(self, *args, **kwargs):
            raise RuntimeError("corrupt index")

    monkeypatch.setattr(utils, "load_experience_db", lambda config=None: Broken())
    assert utils.retrieve_experience("anything") == []


def test_empty_guidance_renders_as_empty_string():
    """Empty guidance keeps prompts byte-identical to their pre-feature form."""
    assert utils.format_experience_guidance([]) == ""
    assert utils.format_experience_guidance([{"full_content": ""}]) == ""


def test_guidance_requires_applicability():
    """The prompt must require checking conditions before applying guidance."""
    rendered = utils.format_experience_guidance(
        [{"full_content": "use coefficientWilkeMultiComponentMixture"}]
    )
    assert "<experience_guidance>" in rendered
    assert "</experience_guidance>" in rendered
    assert "applicab" in rendered.lower()
    assert "coefficientWilkeMultiComponentMixture" in rendered


def test_retrieval_is_cached_after_first_call(monkeypatch):
    """load_experience_db must run once, not once per case."""
    calls = []

    def counting_load(config=None):
        calls.append(1)
        return None

    monkeypatch.setattr(utils, "load_experience_db", counting_load)
    utils.retrieve_experience("first")
    utils.retrieve_experience("second")
    assert len(calls) == 1
