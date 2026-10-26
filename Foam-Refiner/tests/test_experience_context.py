from foam_evo.execution.planner_context import build_planner_context
from foam_evo.experience.store import ExperienceStore


def test_active_experience_entries_are_visible_to_planner(tmp_path):
    (tmp_path / "database").mkdir()
    store = ExperienceStore(tmp_path)
    entry = store.add(
        {
            "trigger_type": "mesh topology",
            "solver": "rhoSimpleFoam",
            "guidance": "Reuse the tutorial block decomposition.",
        },
        source_batch="batch-a",
    )
    context = build_planner_context(worktree=tmp_path)
    assert entry in context
    assert "Reuse the tutorial block decomposition" not in context
