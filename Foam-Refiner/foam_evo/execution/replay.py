"""Replay frozen Foam-Runner prompts against one candidate worktree."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from ..evaluation.matrix import CaseOutcome, EvaluationMatrix, validate_comparable_checks
from ..models import CaseRef
from .worker_pool import CaseJob, CaseWorkerPool


class ReplayError(RuntimeError):
    """Raised when a replay input or process is inconsistent."""


class FoamAgentReplay:
    def __init__(
        self,
        *,
        source_cases: dict[str, Path | CaseRef],
        timeout_s: float = 10800.0,
        replay_workers: int = 1,
    ) -> None:
        self.source_cases = dict(source_cases)
        self.timeout_s = timeout_s
        self.replay_workers = replay_workers

    def baseline_matrix(self, case_ids: list[str]) -> EvaluationMatrix:
        """Build the deterministic matrix represented by the source record."""
        outcomes: list[CaseOutcome] = []
        for case_id in case_ids:
            source_ref = self.source_cases.get(case_id)
            if source_ref is None:
                outcomes.append(CaseOutcome(case_id, False, "source case is not available", infrastructure_blocked=True))
                continue
            source = _case_path(source_ref)
            checks, error = _load_evaluation(source_ref, source)
            failed = tuple(str(item["check_id"]) for item in checks if item.get("passed") is False)
            outcomes.append(CaseOutcome(
                case_id=case_id,
                passed=not failed and error is None,
                reason=error or "source evaluation",
                baseline_failed_checks=failed,
                baseline_checks=checks,
                baseline_passed=not failed,
                candidate_checks=checks,
                comparison_error=error,
            ))
        return EvaluationMatrix.baseline_only(tuple(outcomes))

    def run_case(self, *, case_id: str, worktree: str | Path, output_root: str | Path) -> CaseOutcome:
        root = Path(worktree).resolve()
        outputs = Path(output_root).resolve()
        outputs.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not case_id or Path(case_id).is_absolute() or ".." in Path(case_id).parts or "\\" in case_id:
            raise ReplayError("invalid case identifier")
        if not (outputs / case_id).resolve().is_relative_to(outputs):
            raise ReplayError("case output escapes replay root")
        source_ref = self.source_cases.get(case_id)
        if source_ref is None:
            raise ReplayError(f"source case is not available: {case_id}")
        source = _case_path(source_ref)
        prompt = _prompt_path(source_ref, source)
        if not prompt.is_file():
            raise ReplayError(f"frozen prompt is missing for {case_id}")
        prompt_copy = outputs / case_id / "prompt.txt"
        prompt_copy.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        shutil.copy2(prompt, prompt_copy)
        result_dir = outputs / case_id / "output"
        result_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
        task = _load_task(source_ref, source)
        # Reuse Foam-Runner's single-case primitive. The replay coordinator
        # supplies parallelism; the agent process remains deliberately unaware
        # of benchmark/replay orchestration.
        task_copy = outputs / case_id / "task.json"
        task_path = _task_path(source_ref, source)
        if task_path.is_file():
            task_payload = json.loads(task_path.read_text(encoding="utf-8"))
            if isinstance(task_payload, dict):
                task_payload["prompt_path"] = str(prompt_copy)
            task_copy.write_text(
                json.dumps(task_payload, ensure_ascii=True, indent=2) + "\n",
                encoding="utf-8",
            )
        else:
            task_payload = dict(task)
            task_payload["prompt_path"] = str(prompt_copy)
            task_copy.write_text(json.dumps(task_payload, ensure_ascii=True, indent=2), encoding="utf-8")
        command = [
            sys.executable,
            "-m",
            "src.record_cli",
            "run-case",
            "--task",
            str(task_copy),
            "--output-dir",
            str(result_dir),
            # Keep the immutable case identity in the per-case manifest and
            # correction summary. Without this explicit flag record_cli uses
            # the nested directory name ("output"), which makes aggregated
            # first-success and correction-round statistics ambiguous.
            "--case-id",
            case_id,
        ]
        max_review_loop = (task.get("limits") or {}).get("max_review_loop")
        if isinstance(max_review_loop, int) and not isinstance(max_review_loop, bool):
            command.extend(["--max-review-loop", str(max_review_loop)])
        timed_out = False
        try:
            result = subprocess.run(
                command,
                cwd=root,
                env={**os.environ, "PYTHONPATH": str(root / "src")},
                text=True,
                capture_output=True,
                timeout=self.timeout_s,
                check=False,
            )
            stdout, stderr, return_code = result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            return_code = None
        (outputs / case_id / "stdout.log").write_text(stdout, encoding="utf-8")
        (outputs / case_id / "stderr.log").write_text(stderr, encoding="utf-8")
        baseline_checks, baseline_error = _load_evaluation(source_ref, source)
        candidate_checks, candidate_error = _load_evaluation_path(result_dir / "evaluation.json")
        correction_rounds, first_success_round, attempted_rounds = _load_correction_summary(
            result_dir / "correction_rounds.json"
        )
        comparison_error = baseline_error or candidate_error
        if timed_out or return_code not in (0, 1) or (not candidate_error and return_code != (0 if all(c["passed"] for c in candidate_checks) else 1)):
            comparison_error = comparison_error or f"replay process unavailable: exit={return_code}, timeout={timed_out}"
        if not comparison_error:
            comparison_error = validate_comparable_checks(baseline_checks, candidate_checks)
        failed = [str(item["check_id"]) for item in candidate_checks if item.get("passed") is False]
        passed = bool(candidate_checks) and not failed and not candidate_error and not timed_out and return_code == 0
        if comparison_error:
            reason = f"evaluation contract unavailable: {comparison_error}"
        elif passed:
            reason = "candidate replay passed"
        else:
            reason = "failed checks: " + ", ".join(failed or ["unknown"])
        infrastructure = _infrastructure_blocked(candidate_checks) or timed_out or return_code is None
        return CaseOutcome(
            case_id=case_id,
            passed=passed,
            reason=reason,
            baseline_failed_checks=tuple(str(item["check_id"]) for item in baseline_checks if item.get("passed") is False),
            baseline_checks=baseline_checks,
            baseline_passed=bool(baseline_checks) and not baseline_error and not any(item.get("passed") is False for item in baseline_checks),
            candidate_checks=candidate_checks,
            infrastructure_blocked=infrastructure,
            comparison_error=comparison_error,
            correction_rounds=correction_rounds,
            first_success_round=first_success_round,
            attempted_rounds=attempted_rounds,
        )

    def run(self, *, case_ids: list[str], worktree: str | Path, output_root: str | Path) -> EvaluationMatrix:
        return self.run_matrix(case_ids=case_ids, worktree=worktree, output_root=output_root)

    def run_matrix(
        self,
        *,
        case_ids: list[str],
        worktree: str | Path,
        output_root: str | Path,
        workers: int | None = None,
        workflow_version: str = "candidate",
        history: EvaluationMatrix | None = None,
    ):
        """Run all cases in independent directories and return their matrix."""

        def run(job: CaseJob[str]) -> CaseOutcome:
            try:
                return self.run_case(
                    case_id=job.payload,
                    worktree=worktree,
                    output_root=output_root,
                )
            except ReplayError as exc:
                return CaseOutcome(job.case_id, False, str(exc), infrastructure_blocked=True)
            except Exception as exc:
                return CaseOutcome(
                    job.case_id,
                    False,
                    f"replay {job.case_id} infrastructure error: {type(exc).__name__}: {exc}",
                    infrastructure_blocked=True,
                )

        pool = CaseWorkerPool(workers or self.replay_workers)
        jobs = [CaseJob(case_id=case_id, payload=case_id) for case_id in case_ids]
        current = EvaluationMatrix(
            tuple(pool.run(jobs, run)),
            workflow_versions=("baseline", workflow_version),
            current_workflow_version=workflow_version,
        )
        if history is not None:
            return history.append_version(workflow_version, current.outcomes)
        return current


def _case_path(source: Path | CaseRef) -> Path:
    return source.case_path.expanduser().resolve() if isinstance(source, CaseRef) else Path(source).expanduser().resolve()


def _task_path(source_ref: Path | CaseRef, case_path: Path) -> Path:
    return source_ref.task_path.expanduser().resolve() if isinstance(source_ref, CaseRef) else case_path / "task.json"


def _prompt_path(source_ref: Path | CaseRef, case_path: Path) -> Path:
    if isinstance(source_ref, CaseRef) and source_ref.prompt_path is not None:
        return source_ref.prompt_path.expanduser().resolve()
    return case_path / "inputs" / "prompt.txt"


def _load_task(source_ref: Path | CaseRef, case_path: Path) -> dict:
    path = _task_path(source_ref, case_path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _load_evaluation(source_ref: Path | CaseRef, case_path: Path) -> tuple[tuple[dict, ...], str | None]:
    path = (
        source_ref.evaluation_path.expanduser().resolve()
        if isinstance(source_ref, CaseRef) and source_ref.evaluation_path is not None
        else case_path / "evaluation.json"
    )
    return _load_evaluation_path(path)


def _load_evaluation_path(path: Path) -> tuple[tuple[dict, ...], str | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return (), f"cannot read evaluation.json: {exc}"
    if not isinstance(value, dict):
        return (), "evaluation.json must be an object"
    if value.get("schema_version") != 2:
        return (), "evaluation schema_version must be 2"
    checks = value.get("checks")
    if not isinstance(checks, list):
        return (), "evaluation checks must be a list"
    normalized = tuple(item for item in checks if isinstance(item, dict) and item.get("check_id"))
    if len(normalized) != len(checks):
        return normalized, "evaluation contains invalid checks"
    error = validate_comparable_checks(normalized, normalized)
    if error:
        return normalized, error
    if type(value.get("passed")) is not bool or value["passed"] != all(x["passed"] for x in normalized):
        return normalized, "overall result is missing or inconsistent with checks"
    return normalized, None


def _infrastructure_blocked(checks: tuple[dict, ...]) -> bool:
    return any(
        item.get("check_id") == "infrastructure_healthy" and item.get("passed") is False
        for item in checks
    )


def _load_correction_summary(path: Path) -> tuple[tuple[dict, ...], int | None, int]:
    """Load Foam-Runner's per-correction record without making it mandatory.

    Older case records have no correction summary; replay remains compatible
    with them while new runs expose the round history in the matrix.
    """

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return (), None, 0
    if not isinstance(value, dict):
        return (), None, 0
    rounds = tuple(item for item in value.get("rounds", []) if isinstance(item, dict))
    first = value.get("first_success_round")
    if not isinstance(first, int):
        first = None
    attempted = value.get("attempted_rounds", len(rounds))
    try:
        attempted = max(0, int(attempted))
    except (TypeError, ValueError):
        attempted = len(rounds)
    return rounds, first, attempted
