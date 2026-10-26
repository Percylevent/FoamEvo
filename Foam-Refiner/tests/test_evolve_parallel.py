import time
from pathlib import Path
from types import SimpleNamespace

from foam_evo.graph.evolve_graph import build_evolve_graph
from foam_evo.models import BatchStatus, EvolveRequest, BatchResult
from foam_evo.selection.clustering import Cluster


class Selector:
    def select(self, request):
        return "base", "abc12", [SimpleNamespace(case_id="a"), SimpleNamespace(case_id="b")]


class Clusterer:
    def cluster(self, cases, request):
        return [Cluster("first", ["a"], "mesh"), Cluster("second", ["b"], "solver")]


class Runner:
    def __init__(self):
        self.seen = []

    def run(self, request):
        self.seen.append((request.batch_id, tuple(request.case_ids)))
        time.sleep(0.02)
        return BatchResult(batch_id=request.batch_id, status=BatchStatus.FAILED, base_commit=request.base_commit)


def test_send_branches_construct_independent_batch_requests():
    runner = Runner()
    state = build_evolve_graph(
        record_selector=Selector(), clusterer=Clusterer(), batch_runner=runner
    ).invoke(
        {"request": EvolveRequest(repo_root=Path("/tmp/repo"), record_id="abc12")},
        config={"max_concurrency": 2},
    )
    assert sorted(runner.seen) == [("first-001", ("a",)), ("second-002", ("b",))]
    assert sorted(item.batch_id for item in state["batch_results"]) == ["first-001", "second-002"]
    assert len(state["errors"]) == 2
