import json
import subprocess
from pathlib import Path

import foam_evo.cli as foam_evo_cli
from foam_evo.execution.batch_runtime import BatchRuntimeFactory
from foam_evo.graph.evolve_graph import build_evolve_graph
from foam_evo.models import BatchLimits, BatchRequest, BatchStatus, EvolveRequest
from foam_evo.providers.base import ProviderResult
from foam_evo.roles.markdown_runner import RoleRunner
from foam_evo.selection.clustering import GlobalClusterer
from foam_evo.selection.records import RecordSelector
from foam_evo.evaluation.protection import freeze_suite


class FakeMarkdownProvider:
    """A deterministic role provider that still exercises every engine edge."""

    def run(self, *, prompt, cwd=None, artifact_dir=None, timeout_s=None):
        title = prompt.splitlines()[0].lower() if prompt.splitlines() else ""
        if "global clusterer" in title:
            markdown = "## Cluster cluster-001\nCases: case-a\n\n## Cluster cluster-002\nCases: case-b\n"
        elif "locator" in title:
            markdown = "# Locator\n\nShared root cause: the mesh check is not repaired.\n"
        elif "planner" in title:
            markdown = "# Plan\n\nImplement the shared mesh fix and run replay.\n\n## Exit\nCODE_ONLY\n"
        elif "plan reviewer" in title:
            markdown = "# Plan review\n\n## Verdict\nAPPROVE\n"
        elif "implementer" in title:
            assert cwd is not None
            source = Path(cwd) / "src" / "mesh_fix.py"
            source.write_text("FIXED = True\n", encoding="utf-8")
            markdown = "# Implementer\n\nApplied the mesh fix.\n"
        elif "code reviewer" in title:
            markdown = "# Code review\n\n## Verdict\nAPPROVE\n"
        elif "iteration reviewer" in title:
            markdown = "# Iteration review\n\n## Verdict\nCONVERGED\n"
        else:
            raise AssertionError(f"unexpected role prompt: {title}")
        return ProviderResult(markdown=markdown)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, text=True, capture_output=True, check=True
    )
    return result.stdout.strip()


def _fixture(tmp_path: Path, case_ids=("case-a", "case-b")):
    repo = tmp_path / "foam-runner"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Foam-Refiner tests")
    _git(repo, "config", "user.email", "tests@example.invalid")
    (repo / "database" / "experience" / "entries").mkdir(parents=True)
    (repo / "database" / "experience" / "entries" / ".gitkeep").write_text("", encoding="utf-8")
    (repo / "database/faiss").mkdir()
    (repo / "database/faiss/fixture.index").write_bytes(b"frozen-tutorial-index")
    (repo / "src").mkdir()
    (repo / "src" / "__init__.py").write_text("\n", encoding="utf-8")
    (repo / "foambench_main.py").write_text("# fixture\n", encoding="utf-8")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_fixture.py").write_text(
        "def test_fixture_source():\n    assert True\n", encoding="utf-8"
    )
    (repo / "src" / "record_cli.py").write_text(
        """
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('command')
parser.add_argument('--task', required=True)
parser.add_argument('--output-dir', required=True)
parser.add_argument('--case-id')
args = parser.parse_args()
success = (args.case_id == 'protected' or (Path(__file__).parent / 'mesh_fix.py').exists()) and not (args.case_id == 'protected' and (Path(__file__).parent / 'regress.py').exists())
output = Path(args.output_dir)
output.mkdir(parents=True, exist_ok=True)
(output / 'evaluation.json').write_text(json.dumps({
    'schema_version': 2,
    'passed': success,
    'checks': [{'check_id': 'mesh', 'passed': success}],
}) + '\\n')
raise SystemExit(0 if success else 1)
""".lstrip(),
        encoding="utf-8",
    )
    (repo / ".gitignore").write_text('database/faiss/\ndatabase/experience/index.jsonl\ndatabase/experience/faiss/\n', encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "baseline")
    base = _git(repo, "rev-parse", "HEAD")

    record = tmp_path / "record" / "abc12"
    (record / "cases").mkdir(parents=True)
    (record / "manifest.json").write_text(
        json.dumps({"schema_version": 2, "record_id": "abc12", "status": "completed"}),
        encoding="utf-8",
    )
    source_cases = {}
    for case_id in (*case_ids, "protected"):
        case = record / "cases" / case_id
        (case / "inputs").mkdir(parents=True)
        (case / "inputs" / "prompt.txt").write_text(
            f"Repair {case_id}.\n", encoding="utf-8"
        )
        task = tmp_path / "tasks" / case_id / "task.json"
        task.parent.mkdir(parents=True, exist_ok=True)
        (task.parent / "prompt.txt").write_text(
            f"Repair {case_id}.\n", encoding="utf-8"
        )
        task.write_text(
            json.dumps({"task_id": case_id, "prompt_path": str(task.parent / "prompt.txt")}),
            encoding="utf-8",
        )
        (case / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "case_id": case_id,
                    "status": "completed",
                    "passed": case_id == "protected",
                    "task_path": str(task),
                }
            ),
            encoding="utf-8",
        )
        (case / "evaluation.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "passed": case_id == "protected",
                    "failure_summary": "mesh check failed",
                    "checks": [{"check_id": "mesh", "passed": case_id == "protected"}],
                }
            ),
            encoding="utf-8",
        )
        (case / "stdout.log").write_text("mesh failure\n", encoding="utf-8")
        source_cases[case_id] = case
    refs = RecordSelector(repo).load(EvolveRequest(repo_root=repo, base_ref="main", record_root=record.parent, record_id="abc12"), include_successes=True)
    freeze_suite(repo=repo, base=base, sources={x.case_id:x for x in refs}, allowed_ids=[x.case_id for x in refs], path=tmp_path / "suite.json", workers=2)
    return repo, base, record.parent, source_cases


def _limits(**overrides):
    values = {"max_iterations": 1, "max_plan_rounds": 1, "max_code_rounds": 1, "replay_workers": 2}
    values.update(overrides)
    return BatchLimits(**values)


def test_single_batch_e2e_creates_candidate_branch(tmp_path):
    repo, base, _, source_cases = _fixture(tmp_path, ("case-a",))
    request = BatchRequest(
        batch_id="batch-single",
        repo_root=repo,
        base_commit=base,
        case_ids=["case-a"],
        limits=_limits(),
        regression_suite=tmp_path / "suite.json",
        evo_record_root=tmp_path / "evorecord" / "abc12" / "batches",
    )
    result = BatchRuntimeFactory(
        repo_root=repo,
        provider=FakeMarkdownProvider(),
        source_cases=source_cases,
        artifact_root=tmp_path / "evorecord" / "abc12" / "batches",
        worktree_root=tmp_path / "worktrees",
    ).run(request)
    assert result.status == BatchStatus.CANDIDATE_READY
    assert result.candidate_commit
    assert result.branch_name == "foam-evo/batch/batch-single"
    assert (tmp_path / "evorecord" / "abc12" / "batches" / "batch-single" / "iterations" / "iter-001" / "replay" / "matrix.json").is_file()


def test_parallel_evolve_uses_isolated_batch_runtimes(tmp_path):
    repo, _, record_root, source_cases = _fixture(tmp_path)
    selector = RecordSelector(repo)
    request = EvolveRequest(
        repo_root=repo,
        base_ref="main",
        record_root=record_root,
        record_id="abc12",
        max_cases_per_batch=1,
        parallel_workers=2,
        batch_limits=_limits(),
        regression_suite=tmp_path / "suite.json",
        evo_record_root=tmp_path / "evorecord",
    )
    provider = FakeMarkdownProvider()
    factory = BatchRuntimeFactory(
        repo_root=repo,
        provider=provider,
        source_cases=source_cases,
        artifact_root=tmp_path / "evorecord" / "abc12" / "batches",
        worktree_root=tmp_path / "worktrees",
    )
    state = build_evolve_graph(
        record_selector=selector,
        clusterer=GlobalClusterer(
            RoleRunner(provider), artifact_dir=tmp_path / "evorecord" / "abc12" / "clustering"
        ),
        batch_runner=factory,
    ).invoke({"request": request}, config={"max_concurrency": 2})
    results = sorted(state["batch_results"], key=lambda item: item.batch_id)
    assert [item.status for item in results] == [BatchStatus.CANDIDATE_READY, BatchStatus.CANDIDATE_READY]
    assert {item.branch_name for item in results} == {
        "foam-evo/batch/cluster-001-001",
        "foam-evo/batch/cluster-002-002",
    }
    assert all(item.candidate_commit for item in results)


def test_cli_evolve_runs_complete_external_record_flow(tmp_path, monkeypatch):
    repo, base, record_root, _ = _fixture(tmp_path, ("case-a", "case-b"))
    monkeypatch.setattr(foam_evo_cli, "configured_provider", lambda: FakeMarkdownProvider())
    exit_code = foam_evo_cli.main(
        [
            "evolve",
            "--regression-suite", str(tmp_path / "suite.json"),
            "--repo-root",
            str(repo),
            "--base-ref",
            "main",
            "--record-root",
            str(record_root),
            "--record-id",
            "abc12",
            "--evo-record-root",
            str(tmp_path / "evorecord"),
            "--worktree-root",
            str(tmp_path / "worktrees"),
            "--max-cases-per-batch",
            "1",
            "--parallel-workers",
            "2",
            "--replay-workers",
            "2",
            "--max-iterations",
            "1",
            "--max-plan-rounds",
            "1",
            "--max-code-rounds",
            "1",
        ]
    )
    assert exit_code == 0
    manifest = json.loads((tmp_path / "evorecord" / "abc12" / "manifest.json").read_text())
    assert manifest["batch_count"] == 2
    assert manifest["candidate_count"] == 2
    assert _git(repo, "rev-parse", base) == base
