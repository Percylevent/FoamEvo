from __future__ import annotations

from types import SimpleNamespace

import pytest

import nodes.input_writer_node as input_writer_module
import nodes.reviewer_node as reviewer_module
from nodes.review_limit_node import review_limit_node
from router_func import route_after_reviewer, route_after_runner
from services.input_writer import render_allrun


def test_allrun_is_rendered_with_standard_openfoam_logs() -> None:
    script = render_allrun(["blockMesh", "checkMesh -allTopology", "shallowWaterFoam"])

    assert script.startswith("#!/bin/sh\n")
    assert '. "$WM_PROJECT_DIR/bin/tools/RunFunctions"' in script
    assert "runApplication blockMesh" in script
    assert "runApplication checkMesh -allTopology" in script
    assert "runApplication shallowWaterFoam" in script


@pytest.mark.parametrize(
    "command",
    [
        "shallowWaterFoam || exit 1",
        "blockMesh > log.blockMesh",
        "runApplication blockMesh",
        "blockMesh\nshallowWaterFoam",
    ],
)
def test_allrun_rejects_commands_that_bypass_its_logging_contract(command: str) -> None:
    with pytest.raises(ValueError):
        render_allrun([command])


def test_max_review_loop_one_executes_exactly_one_rewrite(
    monkeypatch,
    tmp_path,
) -> None:
    events = ["initial_input_writer", "local_runner"]
    state = {
        "case_dir": str(tmp_path),
        "config": SimpleNamespace(max_loop=1),
        "error_logs": [{"evidence_id": "err-1", "excerpt": "fatal"}],
        "foamfiles": {},
        "history_text": [],
        "input_writer_mode": "initial",
        "loop_count": 0,
        "requires_visualization": False,
        "review_analysis": None,
        "rewrite_plan": None,
        "user_requirement": "test",
        "dir_structure": {},
    }

    assert route_after_runner(state) == "reviewer"
    events.append("reviewer")

    monkeypatch.setattr(
        reviewer_module,
        "review_error_logs",
        lambda **kwargs: ("replace PISO with PIMPLE", ["review"]),
    )
    monkeypatch.setattr(
        reviewer_module,
        "generate_rewrite_plan",
        lambda **kwargs: {
            "target_files": [{"file": "system/fvSolution"}],
        },
    )
    reviewer_output = reviewer_module.reviewer_node(state)
    state.update(reviewer_output)

    # The reviewer asking for a rewrite is what spends the budget, so that a
    # rewrite returning the file byte-identical does not consume a loop.
    assert state["loop_count"] == 1
    assert route_after_reviewer(state) == "input_writer"

    monkeypatch.setattr(
        input_writer_module,
        "rewrite_files",
        lambda **kwargs: {
            "dir_structure": {},
            "foamfiles": {},
            "error_logs": [],
            "write_results": [
                {"file": "system/fvSolution", "status": "modified"},
            ],
            "changed_files": ["system/fvSolution"],
            "rewrite_changed": True,
        },
    )
    monkeypatch.setattr(
        input_writer_module,
        "convert_case_to_esi_if_needed",
        lambda *args: None,
    )
    monkeypatch.setattr(
        input_writer_module,
        "scan_case_directory",
        lambda *args: {},
    )
    monkeypatch.setattr(
        input_writer_module,
        "read_case_foamfiles",
        lambda *args: {},
    )

    events.append("rewrite_input_writer")
    state.update(input_writer_module._rewrite_mode(state))
    assert state["loop_count"] == 1
    assert state["last_rewrite_changed"] is True

    events.append("local_runner")
    state["error_logs"] = [{"evidence_id": "err-2", "excerpt": "still failing"}]
    assert route_after_runner(state) == "review_limit"
    events.append("review_limit")
    state.update(review_limit_node(state))
    assert state["termination_reason"] == "max_review_loop_reached"
    assert events == [
        "initial_input_writer",
        "local_runner",
        "reviewer",
        "rewrite_input_writer",
        "local_runner",
        "review_limit",
    ]


def test_repeated_error_does_not_end_the_run_before_the_budget() -> None:
    """A stalled signature is recorded but must not cut the rewrite budget short.

    The stall check used to return END here, and it did so *before* the budget
    comparison, making the remaining loops unreachable rather than merely
    unused: counterFlowFlame2D runs stopped at loop 9 of 25. The progress
    fields are still projected for analysis, they just no longer decide when to
    stop.
    """

    state = {
        "config": SimpleNamespace(max_loop=25),
        "error_logs": [{"evidence_id": "err-1", "excerpt": "same failure"}],
        "loop_count": 9,
        "termination_reason": "stalled_review_loop",
        "error_signature": "a" * 64,
        "repeated_error_count": 2,
        "requires_visualization": False,
    }

    assert route_after_runner(state) == "reviewer"


def test_exhausted_budget_still_routes_to_review_limit() -> None:
    state = {
        "config": SimpleNamespace(max_loop=25),
        "error_logs": [{"evidence_id": "err-1", "excerpt": "still failing"}],
        "loop_count": 25,
        "termination_reason": "stalled_review_loop",
        "requires_visualization": False,
    }

    assert route_after_runner(state) == "review_limit"


def test_infrastructure_counter_is_declared_in_graph_state() -> None:
    from utils import GraphState

    assert "consecutive_infrastructure_failures" in GraphState.__annotations__


def test_persisted_infrastructure_counter_early_stops_runner() -> None:
    state = {
        "config": SimpleNamespace(max_loop=25, max_no_progress_loops=2),
        "error_logs": [{"source": "infrastructure", "excerpt": "exit 127"}],
        "consecutive_infrastructure_failures": 2,
        "loop_count": 1,
        "requires_visualization": False,
    }
    assert route_after_runner(state) == "review_limit"
