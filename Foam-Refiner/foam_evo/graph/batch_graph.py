"""LangGraph orchestration for one isolated batch runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from ..evaluation.matrix import EvaluationMatrix
from ..models import BatchRequest, BatchResult, BatchStatus, SourceTestResult


class BatchGraphNotReady(RuntimeError):
    pass


class BatchState(TypedDict, total=False):
    request: BatchRequest
    status: BatchStatus
    iteration: int
    plan_round: int
    code_round: int
    branch_name: str
    worktree_path: str
    locator_path: str
    plan_path: str
    plan_exit: str | None
    plan_review_path: str
    plan_review_verdict: str
    implementer_path: str
    implementation_ok: bool
    implementation_error: str
    changed_files: list[str]
    code_review_path: str
    code_review_verdict: str
    source_test_result: SourceTestResult
    replay_matrix: EvaluationMatrix
    replay_failure_summary: str
    iteration_review_path: str
    iteration_verdict: str
    result: BatchResult
    error: str


@dataclass(frozen=True)
class BatchGraphServices:
    prepare: Callable[[BatchRequest], dict[str, Any]] | None = None
    locate: Callable[[BatchState], dict[str, Any]] | None = None
    plan: Callable[[BatchState], dict[str, Any]] | None = None
    review_plan: Callable[[BatchState], dict[str, Any]] | None = None
    implement: Callable[[BatchState], dict[str, Any]] | None = None
    review_code: Callable[[BatchState], dict[str, Any]] | None = None
    source_tests: Callable[[BatchState], SourceTestResult] | None = None
    replay: Callable[[BatchState], EvaluationMatrix] | None = None
    review_iteration: Callable[[BatchState], dict[str, Any]] | None = None
    finalize: Callable[[BatchState], BatchResult] | None = None


def _required(fn: Callable | None, name: str) -> Callable:
    if fn is None:
        raise BatchGraphNotReady(f"BatchGraph service is not configured: {name}")
    return fn


def build_batch_graph(services: BatchGraphServices | None = None):
    services = services or BatchGraphServices()
    graph = StateGraph(BatchState)

    def prepare(state: BatchState) -> dict[str, Any]:
        data = _required(services.prepare, "prepare")(state["request"])
        return {
            **data,
            "status": BatchStatus.LOCATING,
            "iteration": 1,
            "plan_round": 0,
            "code_round": 0,
        }

    def locate(state: BatchState) -> dict[str, Any]:
        return {**_required(services.locate, "locate")(state), "status": BatchStatus.PLANNING, "plan_round": 0}

    def plan(state: BatchState) -> dict[str, Any]:
        data = _required(services.plan, "plan")(state)
        status = BatchStatus.NO_EVOLUTION if data.get("plan_exit") == "NO_EVALUATION" else BatchStatus.PLANNING
        return {**data, "status": status, "plan_round": state.get("plan_round", 0) + 1}

    def review_plan(state: BatchState) -> dict[str, Any]:
        return {**_required(services.review_plan, "review_plan")(state), "status": BatchStatus.PLANNING}

    def implement(state: BatchState) -> dict[str, Any]:
        data = _required(services.implement, "implement")(state)
        return {**data, "status": BatchStatus.IMPLEMENTING, "code_round": state.get("code_round", 0) + 1}

    def review_code(state: BatchState) -> dict[str, Any]:
        return {**_required(services.review_code, "review_code")(state), "status": BatchStatus.REVIEWING}

    def source_tests(state: BatchState) -> dict[str, Any]:
        result = _required(services.source_tests, "source_tests")(state)
        if not isinstance(result, SourceTestResult):
            raise TypeError("source_tests service must return SourceTestResult")
        return {"source_test_result": result, "status": BatchStatus.TESTING if result.passed else BatchStatus.FAILED}

    def replay(state: BatchState) -> dict[str, Any]:
        matrix = _required(services.replay, "replay")(state)
        if not isinstance(matrix, EvaluationMatrix):
            raise TypeError("replay service must return EvaluationMatrix")
        failed = [f"{item.case_id}: {item.reason}" for item in matrix.outcomes if not item.passed]
        failure_summary = "; ".join(failed)
        if not failure_summary and not matrix.gate_passed:
            failure_summary = "replay matrix did not improve baseline"
        return {
            "replay_matrix": matrix,
            "replay_failure_summary": failure_summary,
            "status": BatchStatus.REVIEWING,
        }

    def review_iteration(state: BatchState) -> dict[str, Any]:
        data = _required(services.review_iteration, "review_iteration")(state)
        verdict = data.get("iteration_verdict", "AMBIGUOUS")
        matrix = state.get("replay_matrix")
        # The reviewer supplies strategy, but only the deterministic evaluator
        # can make a candidate eligible for promotion.
        if verdict == "CONVERGED" and isinstance(matrix, EvaluationMatrix) and matrix.gate_passed:
            status = BatchStatus.CANDIDATE_READY
        else:
            status = BatchStatus.FAILED
        return {**data, "status": status}

    def advance_iteration(state: BatchState) -> dict[str, Any]:
        return {
            "iteration": state.get("iteration", 1) + 1,
            "plan_round": 0,
            "code_round": 0,
            "status": BatchStatus.LOCATING,
        }

    def finalize(state: BatchState) -> dict[str, Any]:
        result = _required(services.finalize, "finalize")(state)
        return {"result": result, "status": result.status}

    def route_plan(state: BatchState) -> str:
        if state.get("plan_exit") == "NO_EVALUATION":
            state["status"] = BatchStatus.NO_EVOLUTION
            return "finish"
        if state.get("plan_exit") is None and state.get("plan_round", 0) >= state["request"].limits.max_plan_rounds:
            state["status"] = BatchStatus.FAILED
            return "finish"
        return "review_plan"

    def route_plan_review(state: BatchState) -> str:
        verdict = state.get("plan_review_verdict", "AMBIGUOUS")
        if verdict == "APPROVE" and state.get("plan_exit") in {"CODE_ONLY", "KNOWLEDGE_ONLY", "CODE_AND_KNOWLEDGE"}:
            return "implement"
        if state.get("plan_round", 0) < state["request"].limits.max_plan_rounds:
            return "plan"
        state["status"] = BatchStatus.FAILED
        return "finish"

    def route_implementation(state: BatchState) -> str:
        if state.get("implementation_ok"):
            return "review_code"
        if state.get("code_round", 0) < state["request"].limits.max_code_rounds:
            return "implement"
        state["status"] = BatchStatus.FAILED
        return "finish"

    def route_code_review(state: BatchState) -> str:
        if state.get("code_review_verdict") == "APPROVE":
            return "source_tests"
        if state.get("code_round", 0) < state["request"].limits.max_code_rounds:
            return "implement"
        state["status"] = BatchStatus.FAILED
        return "finish"

    def route_tests(state: BatchState) -> str:
        result = state.get("source_test_result")
        if isinstance(result, SourceTestResult) and result.passed:
            return "replay"
        if state.get("code_round", 0) < state["request"].limits.max_code_rounds:
            return "implement"
        state["status"] = BatchStatus.FAILED
        return "finish"

    def route_iteration(state: BatchState) -> str:
        if state.get("status") == BatchStatus.CANDIDATE_READY:
            return "finish"
        if state.get("iteration", 1) < state["request"].limits.max_iterations:
            return "advance"
        return "finish"

    graph.add_node("prepare", prepare)
    graph.add_node("locate", locate)
    graph.add_node("plan", plan)
    graph.add_node("review_plan", review_plan)
    graph.add_node("implement", implement)
    graph.add_node("review_code", review_code)
    graph.add_node("source_tests", source_tests)
    graph.add_node("replay", replay)
    graph.add_node("review_iteration", review_iteration)
    graph.add_node("advance_iteration", advance_iteration)
    graph.add_node("finish", finalize)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "locate")
    graph.add_edge("locate", "plan")
    graph.add_conditional_edges("plan", route_plan, {"review_plan": "review_plan", "finish": "finish"})
    graph.add_conditional_edges("review_plan", route_plan_review, {"plan": "plan", "implement": "implement", "finish": "finish"})
    graph.add_conditional_edges("implement", route_implementation, {"implement": "implement", "review_code": "review_code", "finish": "finish"})
    graph.add_conditional_edges("review_code", route_code_review, {"implement": "implement", "source_tests": "source_tests", "finish": "finish"})
    graph.add_conditional_edges("source_tests", route_tests, {"implement": "implement", "replay": "replay", "finish": "finish"})
    graph.add_edge("replay", "review_iteration")
    graph.add_conditional_edges("review_iteration", route_iteration, {"advance": "advance_iteration", "finish": "finish"})
    graph.add_edge("advance_iteration", "locate")
    graph.add_edge("finish", END)
    return graph.compile()
