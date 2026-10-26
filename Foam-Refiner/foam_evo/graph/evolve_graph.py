"""Global selection/clustering graph and isolated batch fan-out."""

from __future__ import annotations

import operator
import re
from typing import Annotated, TypedDict

from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph

from ..models import BatchRequest, BatchResult, EvolveRequest
from ..selection.clustering import Cluster


class EvolveState(TypedDict, total=False):
    request: EvolveRequest
    base_commit: str
    record_id: str
    selected_cases: list
    selected_case_ids: list[str]
    clusters: list[Cluster]
    batch_requests: list[BatchRequest]
    batch_request: BatchRequest
    batch_results: Annotated[list[BatchResult], operator.add]
    errors: Annotated[list[str], operator.add]


def build_evolve_graph(*, record_selector, clusterer, batch_runner):
    """Build the parent graph.

    ``batch_runner`` is a read-only factory. Its ``run`` call constructs a new
    ``BatchRuntime`` inside each Send branch, so the graph never shares mutable
    per-batch state.
    """

    graph = StateGraph(EvolveState)

    def select(state: EvolveState) -> dict:
        base, record_id, cases = record_selector.select(state["request"])
        return {
            "base_commit": base,
            "record_id": record_id,
            "selected_cases": cases,
            "selected_case_ids": [case.case_id for case in cases],
        }

    def cluster(state: EvolveState) -> dict:
        clusters = clusterer.cluster(state["selected_cases"], state["request"])
        return {"clusters": clusters}

    def make_requests(state: EvolveState) -> dict:
        request = state["request"]
        batch_requests = [
            BatchRequest(
                batch_id=_batch_id(cluster.cluster_id, index),
                repo_root=request.repo_root,
                base_commit=state["base_commit"],
                regression_suite=request.regression_suite,
                case_ids=list(cluster.case_ids),
                cluster_reason=cluster.reason,
                limits=request.batch_limits,
                record_id=state.get("record_id"),
                evo_record_root=request.evo_record_root / str(state.get("record_id", "record")),
            )
            for index, cluster in enumerate(state["clusters"], start=1)
        ]
        return {"batch_requests": batch_requests}

    def dispatch(state: EvolveState):
        return [Send("run_batch", {"batch_request": item}) for item in state["batch_requests"]]

    def run_batch(state: EvolveState) -> dict:
        request = state["batch_request"]
        try:
            result = batch_runner.run(request)
        except Exception as exc:
            result = BatchResult(
                batch_id=request.batch_id,
                status="failed",
                base_commit=request.base_commit,
                summary=f"{type(exc).__name__}: {exc}",
                error=f"{type(exc).__name__}: {exc}",
            )
        update: dict = {"batch_results": [result]}
        if result.status not in {"candidate_ready", "no_evolution"}:
            update["errors"] = [f"{result.batch_id}: {result.error or result.summary}"]
        return update

    graph.add_node("select", select)
    graph.add_node("cluster", cluster)
    graph.add_node("make_requests", make_requests)
    graph.add_node("run_batch", run_batch)
    graph.add_edge(START, "select")
    graph.add_edge("select", "cluster")
    graph.add_edge("cluster", "make_requests")
    graph.add_conditional_edges("make_requests", dispatch)
    graph.add_edge("run_batch", END)
    return graph.compile()


def _batch_id(raw: str, index: int) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", str(raw)).strip("-") or f"cluster-{index:03d}"
    return f"{cleaned}-{index:03d}"
