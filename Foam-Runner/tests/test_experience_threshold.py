"""Guidance must be near the query, not merely the nearest thing in the library.

FAISS returns the k nearest neighbours unconditionally, so before a ceiling
existed the library behaved as "always hit": once any entry was promoted, every
query received guidance however unrelated. The score is an L2 distance under
langchain's default EUCLIDEAN_DISTANCE strategy, so lower is closer and the
ceiling is a maximum.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import utils  # noqa: E402


class _Doc:
    def __init__(self, **metadata):
        self.metadata = metadata
        self.page_content = ""


class _ScoredDb:
    """Returns caller-supplied (doc, distance) pairs, nearest first."""

    def __init__(self, scored):
        self.scored = scored

    def similarity_search_with_score(self, query, k):
        return self.scored[:k]


def _entry(entry_id):
    return _Doc(
        entry_id=entry_id,
        full_content=f"guidance body for {entry_id}",
        trigger_type="mixture_type_selection",
        solver="reactingFoam",
        fork="foundation",
        status="active",
    )


@pytest.fixture(autouse=True)
def reset_cache():
    utils.EXPERIENCE_DB_CACHE = None
    utils._EXPERIENCE_DB_LOADED = False
    yield
    utils.EXPERIENCE_DB_CACHE = None
    utils._EXPERIENCE_DB_LOADED = False


def _install(monkeypatch, scored):
    monkeypatch.setattr(
        utils, "load_experience_db", lambda config=None: _ScoredDb(scored)
    )


def test_default_ceiling_is_a_distance_not_a_similarity():
    """Guard the direction: a similarity cutoff here would invert the filter."""
    assert utils.EXPERIENCE_MAX_DISTANCE > 0.5


def test_the_one_measured_real_hit_still_passes(monkeypatch):
    """The only end-to-end verified retrieval scored 0.354; it must survive."""
    _install(monkeypatch, [(_entry("exp-real"), 0.354)])

    results = utils.retrieve_experience("reactingFoam mixture", solver="reactingFoam", fork="foundation")

    assert [r["entry_id"] for r in results] == ["exp-real"]


def test_distant_match_is_dropped(monkeypatch):
    _install(monkeypatch, [(_entry("exp-far"), 1.4)])

    assert utils.retrieve_experience("something unrelated") == []


def test_exactly_at_the_ceiling_is_kept(monkeypatch):
    _install(monkeypatch, [(_entry("exp-edge"), utils.EXPERIENCE_MAX_DISTANCE)])

    results = utils.retrieve_experience("borderline", solver="reactingFoam", fork="foundation")

    assert [r["entry_id"] for r in results] == ["exp-edge"]


def test_scan_stops_at_the_first_entry_beyond_the_ceiling(monkeypatch):
    """Results are ordered nearest-first, so the rest can only be further."""
    _install(
        monkeypatch,
        [
            (_entry("exp-near"), 0.2),
            (_entry("exp-far"), 1.1),
            (_entry("exp-farther"), 1.9),
        ],
    )

    results = utils.retrieve_experience("query", solver="reactingFoam", fork="foundation")

    assert [r["entry_id"] for r in results] == ["exp-near"]


def test_caller_may_override_the_ceiling(monkeypatch):
    _install(monkeypatch, [(_entry("exp-far"), 1.4)])

    assert utils.retrieve_experience("query", solver="reactingFoam", fork="foundation", max_distance=2.0)
    assert utils.retrieve_experience("query", solver="reactingFoam", fork="foundation", max_distance=0.1) == []


def test_empty_library_is_unaffected(monkeypatch):
    """The threshold must not change the cold-start path at all."""
    monkeypatch.setattr(utils, "load_experience_db", lambda config=None: None)

    assert utils.retrieve_experience("anything") == []
    assert utils.format_experience_guidance([]) == ""


def test_a_dropped_match_injects_no_guidance(monkeypatch):
    """End to end: a far match must leave the prompt byte-identical."""
    _install(monkeypatch, [(_entry("exp-far"), 1.4)])

    rendered = utils.format_experience_guidance(
        utils.retrieve_experience("unrelated query")
    )

    assert rendered == ""
