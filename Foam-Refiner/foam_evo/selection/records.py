"""Select failed cases from the new external Foam-Runner record layout."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
import re

from ..models import CaseRef, EvolveRequest


class RecordSelectionError(RuntimeError):
    """Raised when an external record cannot be used for evolution."""


class RecordSelector:
    def __init__(self, repo_root: str | Path) -> None:
        self.repo = Path(repo_root).expanduser().resolve()
        self.excluded_cases: list[dict[str, str]] = []

    def select(self, request: EvolveRequest) -> tuple[str, str, list[CaseRef]]:
        base = self._resolve_base(request.base_ref)
        if request.regression_suite:
            from ..evaluation.protection import suite_cases
            suite, refs = suite_cases(request.regression_suite)
            if suite["base_commit"] != base:
                raise RecordSelectionError("regression suite belongs to a different base")
            wanted = set(request.case_ids or suite["failure_ids"])
            if not wanted or not wanted <= set(suite["failure_ids"]):
                raise RecordSelectionError("requested failures are outside the frozen suite")
            records = [refs[x] for x in sorted(wanted)]
        else:
            records = self.load(request)
        if not records:
            if self.excluded_cases:
                details = "; ".join(
                    f"{item['case_id']}: {item['reason']}" for item in self.excluded_cases
                )
                raise RecordSelectionError("failed cases are not evolvable: " + details)
            raise RecordSelectionError("no failed cases were found in the record root")
        record_ids = {item.record_id for item in records}
        if len(record_ids) != 1:
            raise RecordSelectionError(
                "one evolve invocation must use one source record; select a record_id"
            )
        record_id = next(iter(record_ids))
        return base, record_id, records

    def load(self, request: EvolveRequest, *, include_successes: bool = False) -> list[CaseRef]:
        self.excluded_cases = []
        root = request.record_root.expanduser().resolve()
        if not root.is_dir():
            raise RecordSelectionError(f"record root does not exist: {root}")
        wanted_record = request.record_id
        if wanted_record is not None and not re.fullmatch(r"[a-z0-9]{5}", wanted_record):
            raise RecordSelectionError("record_id must be a five-character lowercase token")
        wanted_cases = set(request.case_ids or [])
        output: list[CaseRef] = []
        record_dirs = [root / wanted_record] if wanted_record else sorted(root.iterdir())
        for record in record_dirs:
            if not record.is_dir() or not (record / "manifest.json").is_file():
                continue
            record_manifest = _read_json(record / "manifest.json")
            record_id = str(record_manifest.get("record_id") or record.name)
            harness_results = _read_harness_results(record / "harness" / "results.jsonl")
            case_root = record / "cases"
            for case_dir in sorted(case_root.iterdir() if case_root.is_dir() else []):
                if not case_dir.is_dir():
                    continue
                manifest = _read_json(case_dir / "manifest.json")
                if not manifest:
                    harness = harness_results.get(case_dir.name)
                    if not harness or bool(harness.get("passed", True)):
                        continue
                    manifest = {"case_id": case_dir.name, "status": "completed", "passed": False}
                evaluation_path = case_dir / "evaluation.json"
                evaluation = _read_json(evaluation_path) if evaluation_path.is_file() else {}
                passed = bool(evaluation.get("passed", manifest.get("passed", False)))
                status = str(manifest.get("status", "unknown")).lower()
                # A workflow may exit normally while deterministic evaluation
                # still fails (for example a missing solver artifact). Such a
                # case is evolvable too; only prepared/running records are
                # excluded from the failed-case set.
                if (passed and not include_successes) or status in {"created", "prepared", "running"}:
                    continue
                if wanted_cases and str(manifest.get("case_id") or case_dir.name) not in wanted_cases:
                    continue
                # Foam-Runner records keep the immutable task in the benchmark
                # tree and record its absolute path in the case manifest. Do
                # not require a duplicate task.json inside the run artifact.
                task_value = str(case_dir / "task.json") if (case_dir / "task.json").is_file() else manifest.get("task_path")
                if task_value:
                    task_path = Path(str(task_value)).expanduser()
                elif request.benchmark_root:
                    task_path = request.benchmark_root / "tasks" / case_dir.name / "task.json"
                else:
                    task_path = case_dir / "task.json"
                if not task_path.is_file():
                    self._exclude(record_id, case_dir.name, "missing task.json referenced by failed case")
                    continue
                prompt = case_dir / "inputs" / "prompt.txt"
                if not prompt.is_file():
                    prompt = task_path.parent / "prompt.txt"
                if not prompt.is_file():
                    self._exclude(record_id, case_dir.name, "missing prompt for failed case")
                    continue
                if evaluation.get("schema_version") != 2 or not isinstance(evaluation.get("checks"), list):
                    self._exclude(record_id, case_dir.name, "missing or obsolete evaluation schema v2")
                    continue
                output.append(
                    CaseRef(
                        record_id=record_id,
                        case_id=str(manifest.get("case_id") or case_dir.name),
                        record_path=record,
                        case_path=case_dir,
                        task_path=task_path,
                        prompt_path=prompt if prompt.is_file() else None,
                        evaluation_path=evaluation_path if evaluation_path.is_file() else None,
                        baseline_commit=record_manifest.get("baseline_commit"),
                    )
                )
        return output

    def _exclude(self, record_id: str, case_id: str, reason: str) -> None:
        self.excluded_cases.append({"record_id": record_id, "case_id": case_id, "reason": reason})

    def _resolve_base(self, ref: str) -> str:
        result = subprocess.run(
            ["git", "rev-parse", f"{ref}^{{commit}}"],
            cwd=self.repo,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode:
            raise RecordSelectionError(result.stderr.strip() or f"cannot resolve base ref: {ref}")
        return result.stdout.strip()


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_harness_results(path: Path) -> dict[str, dict]:
    output: dict[str, dict] = {}
    if not path.is_file():
        return output
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return output
    for line in lines:
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if not isinstance(value, dict):
            continue
        slug = value.get("slug") or (Path(str(value["case_dir"])).name if value.get("case_dir") else None)
        if slug:
            output[str(slug)] = value
    return output
