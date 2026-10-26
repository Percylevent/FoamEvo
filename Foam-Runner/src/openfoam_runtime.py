"""Local OpenFOAM command execution and deterministic error extraction."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping


@dataclass
class CommandResult:
    returncode: int | None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False


def execute_script(script_path: str, stdout_path: str, stderr_path: str,
                   working_dir: str, timeout_s: int) -> CommandResult:
    """Run an Allrun/SLURM script with inherited OpenFOAM environment."""
    try:
        completed = subprocess.run(
            ["bash", script_path], cwd=working_dir, text=True,
            capture_output=True, timeout=timeout_s, check=False,
        )
        result = CommandResult(completed.returncode, completed.stdout or "", completed.stderr or "")
    except subprocess.TimeoutExpired as exc:
        result = CommandResult(None, exc.stdout or "", exc.stderr or "", timed_out=True)
    Path(stdout_path).write_text(result.stdout, encoding="utf-8")
    Path(stderr_path).write_text(result.stderr, encoding="utf-8")
    return result


_FATAL = re.compile(r"(?im)FOAM FATAL(?: IO)? ERROR:|^ERROR:|Traceback \(most recent call last\)|<workflow_error>")
_TRANSIENT = re.compile(r"(?i)error code:\s*(?:429|5\d\d)|too many requests|ratelimiterror|overloaded_error")


def detect_execution_errors(*, case_dir: str, command: CommandResult | None = None,
                            expected: Mapping[str, Any] | None = None) -> list[dict[str, str]]:
    root = Path(case_dir)
    expected = expected or {}
    evidence: list[dict[str, str]] = []

    def add(source: str, excerpt: str) -> None:
        excerpt = "\n".join(
            " ".join(line.strip().split())
            for line in excerpt.strip().splitlines()
        )[:1200]
        if not excerpt:
            return
        digest = hashlib.sha256(f"{source}\n{excerpt}".encode()).hexdigest()[:12]
        evidence.append({"evidence_id": f"err-{digest}", "source": source, "excerpt": excerpt})

    if command is not None:
        if command.timed_out:
            add("process", "Allrun timed out")
        if command.returncode not in (None, 0):
            add("process", f"Allrun exited with code {command.returncode}: {command.stderr or command.stdout}")

    for path in [root / "Allrun.err", root / "Allrun.out", *sorted(root.glob("log.*"))]:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        fatal = _fatal_excerpt(text)
        if fatal:
            add(path.name, fatal)
        elif path.name == "Allrun.err" and text.strip():
            add(path.name, text[-1200:])

    logs = sorted(root.glob("log.*"))
    if expected.get("must_run_openfoam", True) and not logs:
        add("logs", "no solver logs were produced")
    for path in logs:
        text = path.read_text(encoding="utf-8", errors="replace")
        if not re.search(r"(?m)^\s*End\s*$", text):
            add(path.name, "solver log is missing the End marker")
    solver = expected.get("solver")
    if isinstance(solver, str) and solver and not (root / f"log.{solver}").is_file():
        add("logs", f"missing expected solver log: log.{solver}")
    return evidence


def _fatal_excerpt(text: str, max_lines: int = 30) -> str:
    """Return a fatal marker with the diagnostic lines that follow it."""
    match = _FATAL.search(text)
    if not match:
        return ""
    return "\n".join(text[match.start():].splitlines()[:max_lines])


def format_error_logs(evidence: list[Any]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for item in evidence:
        if isinstance(item, Mapping):
            result.append({str(k): str(v) for k, v in item.items()})
        else:
            text = str(item)
            digest = hashlib.sha256(text.encode()).hexdigest()[:12]
            result.append({"evidence_id": f"err-{digest}", "source": "unknown", "excerpt": text})
    return result


def update_error_progress(state: Mapping[str, Any], error_logs: list[Any], *, max_no_progress_loops: int = 2) -> dict[str, Any]:
    previous = str(state.get("error_signature") or "")
    signature = hashlib.sha256(json.dumps(error_logs, sort_keys=True, default=str).encode()).hexdigest() if error_logs else None
    repeated = int(state.get("repeated_error_count") or 0) + 1 if signature and signature == previous else 0
    history = list(state.get("error_signature_history") or [])
    if signature:
        history.append(signature)
    has_infrastructure_error = any(
        isinstance(item, Mapping) and item.get("source") == "infrastructure"
        for item in error_logs
    )
    previous_infrastructure = int(state.get("consecutive_infrastructure_failures") or 0)
    consecutive_infrastructure = previous_infrastructure + 1 if has_infrastructure_error else 0
    return {
        "error_signature": signature,
        "error_signature_history": history[-10:],
        "repeated_error_count": repeated,
        "consecutive_infrastructure_failures": consecutive_infrastructure,
    }
