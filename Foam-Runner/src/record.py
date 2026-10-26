"""Primitive record and case artifact operations for Foam-Runner."""

from __future__ import annotations

import json
import secrets
import string
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any


_TOKEN_ALPHABET = string.ascii_lowercase + string.digits
_TOKEN_RE = re.compile(r"^[a-z0-9]{5}$")


class RecordError(RuntimeError):
    """Raised when a record or case directory cannot be created."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_record(root: str | Path, *, metadata: dict[str, Any] | None = None) -> Path:
    """Create and return a new five-character record directory."""

    root_path = Path(root).expanduser().resolve()
    root_path.mkdir(mode=0o700, parents=True, exist_ok=True)
    for _ in range(100):
        token = "".join(secrets.choice(_TOKEN_ALPHABET) for _ in range(5))
        record = root_path / token
        try:
            record.mkdir(mode=0o700)
        except FileExistsError:
            continue
        (record / "cases").mkdir(mode=0o700)
        manifest = {
            "schema_version": 2,
            "record_id": token,
            "status": "created",
            "created_at": utc_now(),
            "case_count": 0,
            "completed_count": 0,
            "passed_count": 0,
            "failed_count": 0,
            **(metadata or {}),
        }
        _write_json(record / "manifest.json", manifest)
        return record
    raise RecordError(f"could not allocate a unique record token under {root_path}")


def case_directory(record: str | Path, case_id: str) -> Path:
    """Return a validated case directory below a record."""

    if not case_id or case_id in {".", ".."} or "/" in case_id or "\\" in case_id:
        raise RecordError(f"invalid case id: {case_id!r}")
    record_path = Path(record).expanduser().resolve()
    result = (record_path / "cases" / case_id).resolve()
    if not result.is_relative_to((record_path / "cases").resolve()):
        raise RecordError(f"case path escapes record: {case_id!r}")
    return result


def prepare_case(record: str | Path, case_id: str, task_path: str | Path) -> Path:
    """Create a case directory and copy its task and prompt inputs."""

    task = Path(task_path).expanduser().resolve()
    if not task.is_file():
        raise RecordError(f"task file does not exist: {task}")
    target = case_directory(record, case_id)
    target.mkdir(mode=0o700, parents=True, exist_ok=False)
    (target / "inputs").mkdir(mode=0o700)
    payload = json.loads(task.read_text(encoding="utf-8"))
    _write_json(target / "task.json", payload)
    prompt_path = payload.get("prompt_path") if isinstance(payload, dict) else None
    if prompt_path:
        prompt = Path(prompt_path).expanduser()
        if not prompt.is_absolute():
            prompt = task.parent / prompt
        if prompt.is_file():
            (target / "inputs" / "prompt.txt").write_text(
                prompt.read_text(encoding="utf-8"), encoding="utf-8"
            )
            payload["prompt_path"] = str((target / "inputs" / "prompt.txt").resolve())
            _write_json(target / "task.json", payload)
    _write_json(
        target / "manifest.json",
        {
            "schema_version": 2,
            "record_id": Path(record).name,
            "case_id": case_id,
            "status": "prepared",
            "created_at": utc_now(),
            "task_path": str(task),
        },
    )
    return target


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
