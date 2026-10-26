"""Independent graph for explicit candidate branch integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from ..models import MergePlan, MergeRequest, MergeResult


class MergeState(TypedDict, total=False):
    request: MergeRequest
    base_commit: str
    candidate_diffs: dict[str, str]
    merge_plan: MergePlan
    merge_result: MergeResult
    merged_commit: str


@dataclass
class MergeGraphServices:
    inspect: Callable[[MergeRequest], tuple[str, dict[str, str]]]
    analyze: Callable[[str, dict[str, str]], MergePlan]
    merge: Callable[[MergeRequest, str, MergePlan], MergeResult]
    gates: Callable[[MergeState], None]
    commit: Callable[[MergeState], str]


def build_merge_graph(services: MergeGraphServices):
    """Build the merge-only graph; EvolveGraph never invokes this graph."""

    graph = StateGraph(MergeState)

    def inspect(state: MergeState) -> dict:
        base, diffs = services.inspect(state["request"])
        return {"base_commit": base, "candidate_diffs": diffs}

    def analyze(state: MergeState) -> dict:
        return {
            "merge_plan": services.analyze(
                state["base_commit"], state["candidate_diffs"]
            )
        }

    def merge(state: MergeState) -> dict:
        effective = state["request"].model_copy(
            update={"candidate_branches": state["merge_plan"].merge_order}
        )
        return {
            "merge_result": services.merge(
                effective, state["base_commit"], state["merge_plan"]
            )
        }

    def route_merge(state: MergeState) -> str:
        return "gates" if state["merge_result"].status == "READY" else "finish"

    def gates(state: MergeState) -> dict:
        services.gates(state)
        return {}

    def commit(state: MergeState) -> dict:
        return {"merged_commit": services.commit(state)}

    def finish(state: MergeState) -> dict:
        return {}

    graph.add_node("inspect", inspect)
    graph.add_node("analyze", analyze)
    graph.add_node("merge", merge)
    graph.add_node("gates", gates)
    graph.add_node("commit", commit)
    graph.add_node("finish", finish)
    graph.add_edge(START, "inspect")
    graph.add_edge("inspect", "analyze")
    graph.add_edge("analyze", "merge")
    graph.add_conditional_edges(
        "merge", route_merge, {"gates": "gates", "finish": "finish"}
    )
    graph.add_edge("gates", "commit")
    graph.add_edge("commit", "finish")
    graph.add_edge("finish", END)
    return graph.compile()
