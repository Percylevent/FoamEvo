"""Canonical deterministic evaluation for one Foam-Runner case record.

The record writer owns this contract. Foam-Refiner consumes the persisted matrix
for both baseline and replay instead of comparing unrelated checks.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping


EVALUATION_SCHEMA_VERSION = 2

_END = re.compile(r"^\s*End\s*$", re.MULTILINE)
_FATAL = re.compile(r"(?im)^-->\s*FOAM FATAL(?: IO)? ERROR:|^ERROR:\s*")
_TRACEBACK = re.compile(r"(?m)^Traceback \(most recent call last\):")
_INFRASTRUCTURE = (
    re.compile(r"(?i)error code:\s*5\d\d"),
    re.compile(r"(?i)error code:\s*429|too many requests|ratelimiterror"),
    re.compile(r"(?i)upstream account|proxy read timeout|overloaded_error"),
)
_FAILURE_LINES = re.compile(
    r"(?i)FOAM FATAL|Traceback \(most recent call last\)|<workflow_error>|"
    r"max_review_loop_reached|ended without convergence|error code:\s*(?:429|5\d\d)|"
    r"inconsistent number of faces|arc point.*lies on the line"
)


def evaluate_case(
    *,
    case_dir: str | Path,
    return_code: int | None,
    stdout: str,
    stderr: str,
    task: Mapping[str, Any] | None = None,
    timed_out: bool = False,
) -> dict[str, Any]:
    """Return the versioned evaluator contract persisted with every case."""

    root = Path(case_dir).resolve()
    task = task or {}
    expected = task.get("expected") if isinstance(task.get("expected"), Mapping) else {}
    workflow = stdout + "\n" + stderr
    # A correction workflow can encounter a transient runner failure before a
    # later correction succeeds. Evaluate infrastructure health from the final
    # attempt; the complete history remains in workflow.log and correction
    # artifacts for diagnosis and statistics.
    final_attempt = _last_runner_attempt(workflow)
    checks: list[dict[str, Any]] = []

    process_details: list[str] = []
    if timed_out:
        process_details.append("process_timeout")
    elif return_code != 0:
        process_details.append(f"process_exit_{return_code}")
    checks.append(_check("process_completed", not process_details, process_details))

    infrastructure = [pattern.pattern for pattern in _INFRASTRUCTURE if pattern.search(final_attempt)]
    checks.append(_check("infrastructure_healthy", not infrastructure, infrastructure))

    converged = (
        "<workflow_end>Workflow completed successfully!</workflow_end>" in stdout
        and "<workflow_error>" not in workflow
        and "max_review_loop_reached" not in workflow
        and "ended without convergence" not in workflow.lower()
    )
    checks.append(_check(
        "workflow_converged",
        converged,
        [] if converged else ["workflow_success_marker_missing_or_error_present"],
    ))

    expects_openfoam = bool(expected.get("must_run_openfoam", True))
    allrun_ok = not expects_openfoam or (root / "Allrun").is_file()
    checks.append(_check("allrun_exists", allrun_ok, [] if allrun_ok else ["missing:Allrun"]))

    required_missing: list[str] = []
    required_files = expected.get("required_files", [])
    if isinstance(required_files, list):
        for relative in sorted(str(item) for item in required_files):
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts or not (root / path).is_file():
                required_missing.append(relative)
    checks.append(_check("required_files_exist", not required_missing, required_missing))

    log_failures: list[str] = []
    if expects_openfoam:
        logs = sorted(path for path in root.glob("log.*") if path.is_file())
        if not logs:
            log_failures.append("missing_solver_logs")
        for path in logs:
            content = path.read_text(encoding="utf-8", errors="replace")
            if _FATAL.search(content):
                log_failures.append(f"fatal:{path.name}")
            if _TRACEBACK.search(content):
                log_failures.append(f"traceback:{path.name}")
            if not _END.search(content):
                log_failures.append(f"missing_end:{path.name}")
        solver = expected.get("solver")
        if isinstance(solver, str) and solver and not (root / f"log.{solver}").is_file():
            log_failures.append(f"missing_expected_log:log.{solver}")
    checks.append(_check("solver_completed", not log_failures, log_failures))

    visualization_ok = (
        not expected.get("must_generate_visualization")
        or (root / "visualization.png").is_file()
    )
    checks.append(_check(
        "visualization_generated",
        visualization_ok,
        [] if visualization_ok else ["missing:visualization.png"],
    ))

    requirements = expected.get("requirements")
    if isinstance(requirements, Mapping):
        requirement_failures: list[str] = []
        for relative in requirements.get("required_files", []):
            path = Path(str(relative))
            if path.is_absolute() or ".." in path.parts or not (root / path).is_file():
                requirement_failures.append(f"missing:{relative}")
        for relative in requirements.get("forbidden_files", []):
            path = Path(str(relative))
            if not path.is_absolute() and ".." not in path.parts and (root / path).exists():
                requirement_failures.append(f"forbidden_present:{relative}")
        checks.append(_check("task_expectations_met", not requirement_failures, requirement_failures))

    passed = all(bool(item["passed"]) for item in checks)
    failed = [item for item in checks if not item["passed"]]
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "passed": passed,
        "infrastructure_blocked": bool(infrastructure),
        "checks": checks,
        "failure_summary": _failure_summary(failed),
    }


def build_failure_packet(
    *, case_dir: str | Path, evaluation: Mapping[str, Any], stdout: str, stderr: str
) -> dict[str, Any]:
    """Build a compact first-look diagnosis without replacing detailed logs."""

    root = Path(case_dir).resolve()
    failed_checks = [
        str(item.get("check_id"))
        for item in evaluation.get("checks", [])
        if isinstance(item, Mapping) and item.get("passed") is False
    ]
    evidence: list[dict[str, str]] = []
    for path in _diagnostic_paths(root):
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        matches = [index for index, line in enumerate(lines) if _FAILURE_LINES.search(line)]
        if matches:
            start = matches[0]
            excerpt = "\n".join(line.strip() for line in lines[start:start + 30])
            evidence.append({"artifact": path.name, "excerpt": excerpt[:1200]})
        if len(evidence) >= 6:
            break
    if not evidence:
        fallback = (stderr or stdout).strip().splitlines()
        if fallback:
            evidence.append({"artifact": "process_output", "excerpt": "\n".join(fallback[-8:])[:1200]})
    return {
        "schema_version": 1,
        "failure_class": failed_checks[0] if failed_checks else "unknown_failure",
        "trigger_reason": evaluation.get("failure_summary") or "case did not satisfy the evaluator",
        "failed_checks": failed_checks,
        "evidence": evidence,
        "available_artifacts": [path.name for path in _diagnostic_paths(root)],
    }


def _check(check_id: str, passed: bool, details: list[str]) -> dict[str, Any]:
    return {"check_id": check_id, "passed": passed, "details": details}


def _failure_summary(failed: list[dict[str, Any]]) -> str | None:
    if not failed:
        return None
    first = failed[0]
    details = first.get("details") or []
    detail = str(details[0]) if details else "failed"
    return f"{first['check_id']}: {detail}"


def _last_runner_attempt(workflow: str) -> str:
    """Return the latest local/HPC runner block and following output."""

    pattern = re.compile(
        r"<(?P<tag>runner|hpc_runner)>\s*(?P<body>.*?)\s*</(?P=tag)>",
        re.DOTALL,
    )
    matches = list(pattern.finditer(workflow))
    if not matches:
        return workflow
    latest = matches[-1]
    # Provider failures can happen after a runner completes (for example in
    # the reviewer call), so preserve the tail after the latest runner too.
    return latest.group("body") + "\n" + workflow[latest.end():]


def _diagnostic_paths(root: Path) -> list[Path]:
    fixed = [
        root / "stdout.log", root / "stderr.log", root / "workflow.log", root / "review.log",
        root / "Allrun.err", root / "Allrun.out",
    ]
    return [path for path in fixed if path.is_file()] + sorted(root.glob("log.*"))
