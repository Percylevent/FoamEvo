"""Retrieval filters and injection points.

Pure similarity is not sufficient here: guidance written for the other OpenFOAM
fork is wrong rather than merely less relevant, and a retired entry must stop
influencing runs without being deleted.
"""

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import utils  # noqa: E402


class _Doc:
    def __init__(self, **metadata):
        self.metadata = metadata
        self.page_content = ""


class _Db:
    """Returns a fixed candidate set so filtering is what is under test."""

    def __init__(self, docs):
        self.docs = docs
        self.requested_k = None

    def similarity_search_with_score(self, query, k):
        self.requested_k = k
        return [(d, 0.1 * (i + 1)) for i, d in enumerate(self.docs)]


def _entry(entry_id, solver="reactingFoam", fork="foundation", status="active"):
    return _Doc(
        entry_id=entry_id,
        full_content=f"guidance body for {entry_id}",
        trigger_type="mixture_type_selection",
        solver=solver,
        fork=fork,
        status=status,
    )


@pytest.fixture(autouse=True)
def reset_cache():
    utils.EXPERIENCE_DB_CACHE = None
    utils._EXPERIENCE_DB_LOADED = False
    yield
    utils.EXPERIENCE_DB_CACHE = None
    utils._EXPERIENCE_DB_LOADED = False


def _install(monkeypatch, docs):
    db = _Db(docs)
    monkeypatch.setattr(utils, "load_experience_db", lambda config=None: db)
    return db


def test_wrong_solver_is_dropped(monkeypatch):
    _install(monkeypatch, [_entry("exp-a", solver="simpleFoam")])
    assert utils.retrieve_experience("q", solver="reactingFoam") == []


def test_wrong_fork_is_dropped(monkeypatch):
    _install(monkeypatch, [_entry("exp-a", fork="esi")])
    assert utils.retrieve_experience("q", fork="foundation") == []


def test_any_opts_out_of_both_filters(monkeypatch):
    _install(monkeypatch, [_entry("exp-generic", solver="any", fork="any")])
    hits = utils.retrieve_experience("q", solver="reactingFoam", fork="foundation")
    assert [h["entry_id"] for h in hits] == ["exp-generic"]


def test_retired_and_superseded_never_retrieved(monkeypatch):
    _install(monkeypatch, [
        _entry("exp-retired", status="retired"),
        _entry("exp-superseded", status="superseded"),
        _entry("exp-live", status="active"),
    ])
    hits = utils.retrieve_experience("q", solver="reactingFoam", fork="foundation", topk=5)
    assert [h["entry_id"] for h in hits] == ["exp-live"]


def test_status_filter_applies_even_without_solver_or_fork(monkeypatch):
    """Retiring an entry must work regardless of how retrieval is called."""
    _install(monkeypatch, [_entry("exp-retired", status="retired")])
    assert utils.retrieve_experience("q") == []


def test_missing_status_is_treated_as_active(monkeypatch):
    """Entries written before status existed must keep working."""
    doc = _Doc(entry_id="exp-old", full_content="body", trigger_type="t")
    _install(monkeypatch, [doc])
    assert [h["entry_id"] for h in utils.retrieve_experience("q")] == ["exp-old"]


def test_recall_is_wider_than_topk(monkeypatch):
    """Filtered-out entries must not consume the topk budget."""
    db = _install(monkeypatch, [_entry("exp-a")])
    utils.retrieve_experience("q", solver="reactingFoam", fork="foundation", topk=2)
    assert db.requested_k >= 8, db.requested_k


def test_topk_is_still_respected_after_filtering(monkeypatch):
    _install(monkeypatch, [_entry(f"exp-{i}") for i in range(10)])
    assert len(utils.retrieve_experience("q", solver="reactingFoam", fork="foundation", topk=3)) == 3


def test_unknown_context_rejects_restricted_entries(monkeypatch):
    """Unknown context must not satisfy a declared restriction."""
    _install(monkeypatch, [
        _entry("exp-a", solver="simpleFoam", fork="esi"),
        _entry("exp-b", solver="reactingFoam", fork="foundation"),
    ])
    assert utils.retrieve_experience("q", topk=5) == []


def test_entry_id_is_logged_not_just_trigger_type(monkeypatch, capsys):
    """Several entries can share a trigger_type, so only the id identifies one."""
    _install(monkeypatch, [_entry("exp-abc1234")])
    utils.retrieve_experience("q")
    assert "exp-abc1234" in capsys.readouterr().out


def test_a_config_without_embedding_settings_degrades(monkeypatch, capsys):
    """Retrieval is additive; a config-like object must not abort the case."""
    assert utils.load_experience_db(SimpleNamespace()) is None
    assert "no embedding_model" in capsys.readouterr().out


def test_load_failure_never_propagates(monkeypatch):
    def boom(config=None):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(utils, "load_experience_db", boom)
    assert utils.retrieve_experience("q") == []


def test_injection_points_accept_guidance():
    """All three prompt builders must take the parameter, or injection is dead."""
    from services.input_writer import initial_write
    from services.review import review_error_logs

    assert "experience_guidance" in inspect.signature(initial_write).parameters
    assert "experience_guidance" in inspect.signature(review_error_logs).parameters


def test_guidance_defaults_to_empty_everywhere():
    """Every injection point must default to no guidance, so an empty library
    leaves prompts byte-identical to their pre-feature form."""
    from services.input_writer import initial_write
    from services.review import review_error_logs

    for fn in (initial_write, review_error_logs):
        assert inspect.signature(fn).parameters["experience_guidance"].default == ""
