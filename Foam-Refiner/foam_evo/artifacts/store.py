"""Atomic JSON/text artifacts; no workflow semantics live here."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


class ArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)

    def _target(self, relative: str | Path) -> Path:
        target = (self.root / relative).resolve()
        if not target.is_relative_to(self.root):
            raise ValueError(f"artifact path escapes root: {relative}")
        return target

    def write_json(self, relative: str | Path, value: Any) -> Path:
        payload = json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
        return self.write_text(relative, payload)

    def write_text(self, relative: str | Path, value: str) -> Path:
        target = self._target(relative)
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(value)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return target

    def sha256(self, relative: str | Path) -> str:
        return hashlib.sha256(self._target(relative).read_bytes()).hexdigest()
