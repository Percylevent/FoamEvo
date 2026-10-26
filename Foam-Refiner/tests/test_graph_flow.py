from pathlib import Path

from foam_evo.evaluation.matrix import CaseOutcome, EvaluationMatrix
from foam_evo.graph.batch_graph import BatchGraphServices, build_batch_graph
from foam_evo.models import BatchRequest, BatchStatus, SourceTestResult


def _request(**limits):
    return BatchRequest(
        batch_id="batch-test",
        repo_root=Path("/tmp/repo"),
        base_commit="base",
        case_ids=["case-a"],
        limits=limits,
    )


def _matrix(improved: bool):
    return EvaluationMatrix(
        (
            CaseOutcome(
                "case-a",
                improved,
                "ok" if improved else "failed",
                baseline_checks=({"check_id": "mesh", "passed": False},),
                baseline_failed_checks=("mesh",),
                baseline_passed=False,
                candidate_checks=({"check_id": "mesh", "passed": improved},),
            ),
        )
    )


def test_failed_implementer_is_retried_within_code_budget():
    calls = {"implement": 0}

    def implement(state):
        calls["implement"] += 1
        return {
            "implementer_path": f"impl-{calls['implement']}",
            "implementation_ok": calls["implement"] == 2,
            "implementation_error": "transport failure" if calls["implement"] == 1 else "",
        }

    services = BatchGraphServices(
        prepare=lambda req: {},
        locate=lambda state: {"locator_path": "locator.md"},
        plan=lambda state: {"plan_path": "plan.md", "plan_exit": "CODE_ONLY"},
        review_plan=lambda state: {"plan_review_path": "review.md", "plan_review_verdict": "APPROVE"},
        implement=implement,
        review_code=lambda state: {"code_review_path": "code-review.md", "code_review_verdict": "APPROVE"},
        source_tests=lambda state: SourceTestResult(passed=True),
        replay=lambda state: _matrix(True),
        review_iteration=lambda state: {"iteration_review_path": "iteration.md", "iteration_verdict": "CONVERGED"},
        finalize=lambda state: __import__("foam_evo.models", fromlist=["BatchResult"]).BatchResult(
            batch_id="batch-test", status=state["status"], base_commit="base"
        ),
    )
    result = build_batch_graph(services).invoke({"request": _request(max_code_rounds=2)})["result"]
    assert result.status == BatchStatus.CANDIDATE_READY
    assert calls["implement"] == 2


def test_replay_failure_returns_to_locator_and_carries_feedback():
    calls = {"locate": 0, "replay": 0}
    locator_states = []

    def locate(state):
        calls["locate"] += 1
        locator_states.append(dict(state))
        return {"locator_path": f"locator-{calls['locate']}.md"}

    def replay(state):
        calls["replay"] += 1
        return _matrix(calls["replay"] == 2)

    services = BatchGraphServices(
        prepare=lambda req: {},
        locate=locate,
        plan=lambda state: {"plan_path": "plan.md", "plan_exit": "CODE_ONLY"},
        review_plan=lambda state: {"plan_review_path": "review.md", "plan_review_verdict": "APPROVE"},
        implement=lambda state: {"implementer_path": "impl.md", "implementation_ok": True},
        review_code=lambda state: {"code_review_path": "code-review.md", "code_review_verdict": "APPROVE"},
        source_tests=lambda state: SourceTestResult(passed=True),
        replay=replay,
        review_iteration=lambda state: {
            "iteration_review_path": "iteration.md",
            "iteration_verdict": "CONVERGED" if state["replay_matrix"].gate_passed else "NEED_MORE_WORK",
        },
        finalize=lambda state: __import__("foam_evo.models", fromlist=["BatchResult"]).BatchResult(
            batch_id="batch-test", status=state["status"], base_commit="base"
        ),
    )
    result = build_batch_graph(services).invoke({"request": _request(max_iterations=2)})["result"]
    assert result.status == BatchStatus.CANDIDATE_READY
    assert calls == {"locate": 2, "replay": 2}
    assert "case-a" in locator_states[1]["replay_failure_summary"]


def test_outer_plan_and_code_limits_complete_without_graph_cutoff():
    from foam_evo.graph.runner import BatchExecution
    from foam_evo.models import BatchResult
    calls = {"locate": 0, "plan": 0, "implement": 0, "replay": 0}
    def locate(state):
        calls["locate"] += 1
        return {}
    def plan(state):
        calls["plan"] += 1
        return {"plan_exit": "CODE_ONLY"}
    def implement(state):
        calls["implement"] += 1
        return {"implementation_ok": True}
    def replay(state):
        calls["replay"] += 1
        return _matrix(calls["replay"] == 3)
    services = BatchGraphServices(
        prepare=lambda req: {}, locate=locate, plan=plan,
        review_plan=lambda state: {"plan_review_verdict": "APPROVE" if state["plan_round"] == 3 else "REJECT_COMPLETENESS"},
        implement=implement,
        review_code=lambda state: {"code_review_verdict": "APPROVE"},
        source_tests=lambda state: SourceTestResult(passed=state["code_round"] == 3),
        replay=replay,
        review_iteration=lambda state: {"iteration_verdict": "CONVERGED" if state["replay_matrix"].gate_passed else "NEED_MORE_WORK"},
        finalize=lambda state: BatchResult(batch_id="batch-test", status=state["status"], base_commit="base"))
    result = BatchExecution(services=services).run(_request(max_iterations=3,max_plan_rounds=3,max_code_rounds=3))
    assert result.status == BatchStatus.CANDIDATE_READY, result
    assert calls == {"locate": 3, "plan": 9, "implement": 9, "replay": 3}
