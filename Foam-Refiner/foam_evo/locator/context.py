"""Materialize a progressive, read-only Locator context."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from ..models import CaseRef
from ..artifacts.store import ArtifactStore


class LocatorContextError(RuntimeError):
    pass


_LOG_NAMES = {
    "stdout.log",
    "stderr.log",
    "workflow.log",
    "review.log",
    "Allrun.err",
    "Allrun.out",
}


def build_locator_context(
    *, source_cases: dict[str, Path | CaseRef], case_ids: list[str], root: str | Path
) -> Path:
    """Copy complete case inputs plus diagnostic artifacts into *root*.

    ``case-index.json`` is the only content placed in the initial Locator
    prompt.  Every copied path is available to Claude's native Read/Grep/Bash
    tools for progressive inspection.
    """

    destination = Path(root).resolve()
    context = destination / "context"
    context.mkdir(mode=0o700, parents=True, exist_ok=True)
    index: list[dict] = []
    for position, case_id in enumerate(case_ids, start=1):
        reference = source_cases.get(case_id)
        if reference is None:
            raise LocatorContextError(f"source case is unavailable: {case_id}")
        source = _case_path(reference)
        if not source.is_dir():
            raise LocatorContextError(f"source case is unavailable: {case_id}")
        case_root = context / "cases" / f"case-{position:03d}"
        case_root.mkdir(mode=0o700, parents=True, exist_ok=True)

        copied: dict[str, str] = {}
        # A record can contain both the canonical ``inputs/`` tree and
        # generated-case input directories (0/, constant/, system/). Copy all
        # of them when present so Locator never loses evidence from a mixed or
        # partially failed run.
        for relative in ("inputs", "0", "constant", "system", "Allrun"):
            candidate = source / relative
            if candidate.exists():
                _copy(candidate, case_root / relative)
                copied[f"input:{relative}"] = _relative(case_root / relative, destination)

        for relative, target_name in _metadata_files(reference, source):
            source_file = Path(relative)
            if not source_file.is_file():
                continue
            target = case_root / target_name
            _copy(source_file, target)
            copied[target_name] = _relative(target, destination)

        for log in sorted(source.glob("log.*")):
            if log.is_file():
                target = case_root / "artifacts" / log.name
                _copy(log, target)
                copied[f"log:{log.name}"] = _relative(target, destination)

        # Keep any additional top-level diagnostic text, while excluding large
        # generated directories that are not useful for first-pass location.
        for child in sorted(source.iterdir()):
            if child.name in {"inputs", "0", "constant", "system", "Allrun"} or child.name.startswith("log."):
                continue
            if child.is_file() and child.suffix.lower() in {".log", ".err", ".out", ".txt"}:
                target = case_root / "artifacts" / child.name
                _copy(child, target)
                copied[f"artifact:{child.name}"] = _relative(target, destination)

        evaluation = _json(_explicit(reference, "evaluation") or source / "evaluation.json")
        task = _json(_explicit(reference, "task") or source / "task.json")
        failed_checks = [
            str(item.get("check_id"))
            for item in evaluation.get("checks", [])
            if isinstance(item, dict) and item.get("passed") is False and item.get("check_id")
        ]
        summary = {
            "case_id": case_id,
            "task_id": task.get("task_id"),
            "passed": evaluation.get("passed"),
            "failure_summary": _short(evaluation.get("failure_summary")),
            "failed_checks": failed_checks[:40],
            "check_count": len(evaluation.get("checks", [])) if isinstance(evaluation.get("checks"), list) else 0,
            "files": sorted(copied),
        }
        (case_root / "summary.md").write_text(_summary_markdown(summary), encoding="utf-8")
        copied["summary"] = _relative(case_root / "summary.md", destination)
        index.append({
            "case_id": case_id,
            "case_dir": _relative(case_root, destination),
            "summary": summary,
            "files": copied,
        })
    ArtifactStore(destination).write_json("context/case-index.json", index)
    return destination


def _metadata_files(reference: Path | CaseRef, source: Path):
    values: list[tuple[Path, str]] = []
    explicit_task = _explicit(reference, "task")
    explicit_eval = _explicit(reference, "evaluation")
    explicit_prompt = _explicit(reference, "prompt")
    for path, name in (
        (explicit_task or source / "task.json", "task.json"),
        (explicit_eval or source / "evaluation.json", "evaluation.json"),
        (explicit_prompt or source / "inputs" / "prompt.txt", "inputs/prompt.txt"),
        (source / "result.json", "result.json"),
        (source / "manifest.json", "manifest.json"),
    ):
        values.append((Path(path), name))
    for name in _LOG_NAMES:
        values.append((source / name, f"artifacts/{name}"))
    return values


def _explicit(reference: Path | CaseRef, name: str) -> Path | None:
    if isinstance(reference, CaseRef):
        return {
            "task": reference.task_path,
            "evaluation": reference.evaluation_path,
            "prompt": reference.prompt_path,
        }.get(name)
    return None


def _case_path(reference: Path | CaseRef) -> Path:
    return reference.case_path.expanduser().resolve() if isinstance(reference, CaseRef) else Path(reference).expanduser().resolve()


def _copy(source: Path, target: Path) -> None:
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True)
    else:
        shutil.copy2(source, target)


def _relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root)).replace("\\", "/")


def _json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _short(value: object, limit: int = 2000) -> object:
    return value[:limit] if isinstance(value, str) else value


def _summary_markdown(summary: dict) -> str:
    lines = [f"# Case {summary['case_id']}", "", "## Summary", ""]
    for key in ("task_id", "passed", "failure_summary", "check_count"):
        lines.append(f"- {key}: {summary.get(key)}")
    lines.extend(["", "## Failed checks", ""])
    lines.extend(f"- {item}" for item in summary.get("failed_checks", []))
    if not summary.get("failed_checks"):
        lines.append("- (none recorded)")
    lines.extend(["", "## Available files", ""])
    lines.extend(f"- `{item}`" for item in summary.get("files", []))
    return "\n".join(lines) + "\n"
