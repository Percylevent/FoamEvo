"""External evolution artifact layout keyed by the source record token."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any


_RECORD_ID = re.compile(r"^[a-z0-9]{5}$")


class EvoRecordError(RuntimeError):
    """Raised when an evolution record cannot be initialized."""


class EvoRecordStore:
    """Create an external, append-only-by-convention evolution workspace."""

    def __init__(self, root: str | Path = "evorecord") -> None:
        self.root = Path(root).expanduser().resolve()

    def open(self, record_id: str, *, source_record: str | Path, base_ref: str) -> Path:
        if not _RECORD_ID.fullmatch(record_id):
            raise EvoRecordError(f"invalid source record id: {record_id!r}")
        path = (self.root / record_id).resolve()
        if not path.is_relative_to(self.root):
            raise EvoRecordError("evolution record escapes configured root")
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        (path / "batches").mkdir(mode=0o700, exist_ok=True)
        manifest = path / "manifest.json"
        if not manifest.exists():
            self.write_json(
                path / "manifest.json",
                {
                    "schema_version": 1,
                    "record_id": record_id,
                    "source_record": str(Path(source_record).expanduser().resolve()),
                    "base_ref": base_ref,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "status": "created",
                },
            )
        self.write_json(
            path / "source-record.json",
            {
                "record_id": record_id,
                "path": str(Path(source_record).expanduser().resolve()),
            },
        )
        return path

    def update(self, record_id: str, **fields: Any) -> Path:
        """Persist coordinator status without touching the source record."""
        path = (self.root / record_id).resolve()
        manifest = path / "manifest.json"
        if not path.is_relative_to(self.root) or not manifest.is_file():
            raise EvoRecordError(f"evolution record does not exist: {record_id}")
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise EvoRecordError(f"invalid evolution manifest: {manifest}") from exc
        payload.update(fields)
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        return self.write_json(manifest, payload)

    @staticmethod
    def write_json(path: Path, value: Any) -> Path:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=True, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        return path
