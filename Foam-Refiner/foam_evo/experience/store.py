"""Validated lifecycle manager for Foam-Runner's experience library."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path


class ExperienceError(RuntimeError):
    pass


_SLUG = re.compile(r"[^a-z0-9]+")
_ENTRY_ID = re.compile(r"^exp-[0-9a-f]{7}$")
_GUIDANCE = re.compile(r"<guidance>\n(.*?)\n</guidance>", re.DOTALL)
_ACTIVE = "active"


def _entry_id(trigger_type: str, solver: str, fork: str, version: str, body: str) -> str:
    digest = hashlib.sha256(
        f"{trigger_type}\n{solver}\n{fork}\n{version}\n{body.strip()}".encode()
    ).hexdigest()
    return f"exp-{digest[:7]}"


def _one_line(value: object, *, default: str = "") -> str:
    if value is None:
        return default
    return " ".join(str(value).strip().splitlines())


def valid_conditions(conditions):
    if not isinstance(conditions, dict):
        return False
    def scalar(value):
        return (isinstance(value, (str, int, float, bool)) and str(value).strip() != ""
                and (not isinstance(value, float) or math.isfinite(value)))
    for key, declared in conditions.items():
        if not isinstance(key, str) or not key.strip():
            return False
        if isinstance(declared, dict):
            if not declared or not set(declared) <= {"min", "max"}:
                return False
            try:
                if any(isinstance(x, bool) or not math.isfinite(float(x)) for x in declared.values()):
                    return False
                if float(declared.get("min", "-inf")) > float(declared.get("max", "inf")):
                    return False
            except (TypeError, ValueError):
                return False
        elif isinstance(declared, list):
            if not declared or not all(scalar(x) for x in declared):
                return False
        elif not scalar(declared):
            return False
    return True


class ExperienceStore:
    """Create and transition content-addressed experience entries.

    ``delete`` is logical: inactive Markdown stays available for review while
    Foam-Runner retrieval filters it out. The ledger has one canonical row per ID.
    """

    def __init__(self, repo_root: str | Path) -> None:
        self.repo_root = Path(repo_root).resolve()
        if not (self.repo_root / "database").is_dir():
            raise ExperienceError("candidate is not a Foam-Runner checkout")
        self.root = self.repo_root / "database" / "experience"
        self.entries = self.root / "entries"
        self.index = self.root / "index.jsonl"

    def add(self, proposal: dict, *, source_batch: str) -> str:
        fields = self._proposal_fields(proposal, require_complete=True)
        entry_id = _entry_id(fields["trigger_type"], fields["solver"], fields["fork"], fields["version"], fields["guidance"] + json.dumps(fields.get("conditions", {}), sort_keys=True))
        existing = self._row(entry_id, required=False)
        if existing is not None:
            if existing.get("status", _ACTIVE) != _ACTIVE:
                raise ExperienceError(f"{entry_id} already exists with status {existing.get('status')}; use update or provide revised guidance")
            return entry_id
        row = self._make_row(entry_id, fields, status=_ACTIVE, source_batch=source_batch)
        self._persist(entry_id, self._render(row, fields["guidance"]), row)
        return entry_id

    def update(self, entry_id: str, proposal: dict, *, source_batch: str) -> str:
        """Update an active entry in place while keeping its stable ID."""
        row, body = self._entry(entry_id)
        self._require_status(row, {_ACTIVE}, "update")
        fields = {"trigger_type": row["trigger_type"], "solver": row["solver"], "fork": row.get("fork", "any"), "version": row.get("version", ""), "keywords": row.get("keywords", ""), "guidance": body, "conditions": row.get("conditions", {})}
        fields.update(self._proposal_fields(proposal, require_complete=False))
        fields = self._validate_fields(fields)
        updated = self._make_row(entry_id, fields, status=_ACTIVE, source_batch=source_batch)
        reason = _one_line(proposal.get("reason"))
        if reason:
            updated["lifecycle_reason"] = reason
        self._persist(entry_id, self._render(updated, fields["guidance"]), updated)
        return entry_id

    def retire(self, entry_id: str, *, source_batch: str, reason: str) -> str:
        return self._set_status(entry_id, "retired", source_batch=source_batch, reason=reason)

    def supersede(self, entry_id: str, replacement_id: str, *, source_batch: str, reason: str) -> str:
        if entry_id == replacement_id:
            raise ExperienceError("an experience entry cannot supersede itself")
        replacement = self._row(replacement_id)
        self._require_status(replacement, {_ACTIVE}, "use as a replacement")
        return self._set_status(entry_id, "superseded", source_batch=source_batch, reason=reason, superseded_by=replacement_id)

    def delete(self, entry_id: str, *, source_batch: str, reason: str) -> str:
        """Logically delete an entry without removing its reviewable source."""
        return self._set_status(entry_id, "deleted", source_batch=source_batch, reason=reason)

    def rebuild_index(self, *, timeout_s: float = 900.0) -> bool:
        indexer = self.repo_root / "database" / "script" / "init_experience_db.py"
        if not indexer.is_file():
            return False
        result = subprocess.run([sys.executable, str(indexer)], cwd=self.repo_root, text=True, capture_output=True, timeout=timeout_s, check=False)
        if result.returncode:
            raise ExperienceError(result.stderr.strip() or result.stdout.strip() or "experience index rebuild failed")
        return True

    def prepare_for_replay(self, *, timeout_s: float = 900.0) -> dict:
        """Synchronize all experience assets before a candidate replay.

        Experience Markdown is the source of truth.  The JSONL ledger and the
        FAISS directory are derived runtime assets, so an Implementer cannot
        accidentally make replay use a stale index by hand-editing the ledger.
        This method is deliberately called before *every* replay iteration,
        including iterations where no knowledge file changed.
        """

        rows = self.rebuild_ledger()
        faiss_rebuilt = self.rebuild_index(timeout_s=timeout_s)
        return {
            "entries": len(rows),
            "ledger": str(self.index),
            "ledger_sha256": hashlib.sha256(self.index.read_bytes()).hexdigest()
            if self.index.is_file() else None,
            "faiss_rebuilt": bool(faiss_rebuilt),
            "faiss_root": str(self.root / "faiss"),
        }

    def rebuild_ledger(self) -> list[dict]:
        """Rebuild ``index.jsonl`` from the checked-in Markdown entries.

        The previous ledger is used only for lifecycle metadata that is not
        represented in the Markdown source.  Invalid entries fail loudly so a
        replay never silently ignores a malformed knowledge change.
        """

        previous = {str(row.get("id")): row for row in self._rows() if row.get("id")}
        rows: list[dict] = []
        if self.entries.is_dir():
            for path in sorted(self.entries.glob("exp-*.md")):
                text = path.read_text(encoding="utf-8")
                match = _GUIDANCE.search(text)
                if match is None:
                    raise ExperienceError(f"experience guidance is malformed: {path.name}")
                entry_id = path.stem
                self._validate_entry_id(entry_id)
                fields = {
                    "trigger_type": self._field(text, "trigger_type"),
                    "solver": self._field(text, "solver"),
                    "fork": self._field(text, "openfoam_fork") or self._field(text, "fork") or "any",
                    "version": self._field(text, "openfoam_version") or self._field(text, "version"),
                    "keywords": self._field(text, "keywords"),
                    "guidance": match.group(1).strip(),
                    "conditions": json.loads(self._field(text, "conditions_json") or "{}"),
                }
                fields = self._validate_fields(fields)
                old = previous.get(entry_id, {})
                source_batch = self._field(text, "evolution_batch") or old.get("source_batch") or "imported"
                row = self._make_row(entry_id, fields, status=self._field(text, "status") or old.get("status", _ACTIVE), source_batch=source_batch)
                for key in ("lifecycle_reason", "superseded_by"):
                    if old.get(key):
                        row[key] = old[key]
                    value = self._field(text, key)
                    if value:
                        row[key] = value
                rows.append(row)
        self._write_rows(rows)
        return rows

    @staticmethod
    def _field(text: str, name: str) -> str:
        match = re.search(rf"(?im)^[ \t]*{re.escape(name)}[ \t]*:[ \t]*([^\r\n]*)[ \t]*$", text)
        return match.group(1).strip() if match else ""

    def _set_status(self, entry_id: str, status: str, *, source_batch: str, reason: str, superseded_by: str | None = None) -> str:
        row, body = self._entry(entry_id)
        current = row.get("status", _ACTIVE)
        if current == "deleted":
            raise ExperienceError(f"cannot change deleted experience {entry_id}")
        if status in {"retired", "superseded"}:
            self._require_status(row, {_ACTIVE}, status)
        clean_reason = _one_line(reason)
        if not clean_reason:
            raise ExperienceError(f"{status} requires a reason")
        updated = dict(row)
        updated.update({"status": status, "source_batch": self._source_batch(source_batch), "lifecycle_reason": clean_reason})
        updated.pop("superseded_by", None)
        if superseded_by is not None:
            updated["superseded_by"] = superseded_by
        self._persist(entry_id, self._render(updated, body), updated)
        return entry_id

    def _entry(self, entry_id: str) -> tuple[dict, str]:
        row = self._row(entry_id)
        path = self._entry_path(entry_id, row)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ExperienceError(f"experience source is unavailable: {entry_id}") from exc
        match = _GUIDANCE.search(text)
        if match is None:
            raise ExperienceError(f"experience guidance is malformed: {entry_id}")
        return row, match.group(1).strip()

    def _row(self, entry_id: str, *, required: bool = True) -> dict | None:
        self._validate_entry_id(entry_id)
        rows = [row for row in self._rows() if row.get("id") == entry_id]
        if not rows:
            if required:
                raise ExperienceError(f"unknown experience entry: {entry_id}")
            return None
        return dict(rows[-1])

    def _entry_path(self, entry_id: str, row: dict) -> Path:
        if row.get("path") != f"entries/{entry_id}.md":
            raise ExperienceError(f"unsafe experience path for {entry_id}")
        return self.root / row["path"]

    @staticmethod
    def _require_status(row: dict, allowed: set[str], operation: str) -> None:
        status = str(row.get("status", _ACTIVE)).lower()
        if status not in allowed:
            raise ExperienceError(f"cannot {operation} experience {row.get('id')}: status is {status}")

    @staticmethod
    def _validate_entry_id(entry_id: str) -> None:
        if not _ENTRY_ID.fullmatch(str(entry_id)):
            raise ExperienceError(f"invalid experience entry id: {entry_id}")

    @staticmethod
    def _source_batch(source_batch: str) -> str:
        value = _one_line(source_batch)
        if not value:
            raise ExperienceError("source_batch is required")
        return value

    def _proposal_fields(self, proposal: dict, *, require_complete: bool) -> dict:
        fields = {}
        for name in ("trigger_type", "solver", "fork", "version", "keywords", "guidance"):
            value = proposal.get(name)
            if value is not None:
                fields[name] = str(value).strip()
        if "conditions" in proposal:
            fields["conditions"] = proposal["conditions"]
        if require_complete:
            fields.setdefault("fork", "any")
            fields.setdefault("version", "")
            fields.setdefault("keywords", "")
            return self._validate_fields(fields)
        return fields

    @staticmethod
    def _validate_fields(fields: dict[str, str]) -> dict[str, str]:
        trigger = _SLUG.sub("_", fields.get("trigger_type", "").lower()).strip("_")
        solver = _one_line(fields.get("solver"))
        fork = _one_line(fields.get("fork"), default="any") or "any"
        version = _one_line(fields.get("version"))
        keywords = _one_line(fields.get("keywords"))
        body = fields.get("guidance", "").strip()
        if not trigger:
            raise ExperienceError("trigger_type must contain at least one alphanumeric character")
        if not solver or not body:
            raise ExperienceError("experience solver and guidance are required")
        if fork not in {"any", "foundation", "esi"}:
            raise ExperienceError("experience fork must be any, foundation, or esi")
        if any(marker in body for marker in ("</guidance>", "<source>", "<experience_end>")):
            raise ExperienceError("experience guidance contains a reserved format marker")
        conditions = fields.get("conditions", {})
        if not valid_conditions(conditions):
            raise ExperienceError("conditions must use nonempty scalar/list values or finite ordered min/max bounds")
        return {"conditions": conditions, "trigger_type": trigger, "solver": solver, "fork": fork, "version": version, "keywords": keywords, "guidance": body}

    def _make_row(self, entry_id: str, fields: dict[str, str], *, status: str, source_batch: str) -> dict:
        return {"conditions": fields.get("conditions", {}), "id": entry_id, "path": f"entries/{entry_id}.md", "trigger_type": fields["trigger_type"], "solver": fields["solver"], "fork": fields["fork"], "version": fields["version"], "keywords": fields["keywords"], "status": status, "source_batch": self._source_batch(source_batch), "body_sha256": hashlib.sha256(fields["guidance"].encode()).hexdigest()}

    def _persist(self, entry_id: str, text: str, row: dict) -> None:
        self.entries.mkdir(parents=True, exist_ok=True)
        path = self.entries / f"{entry_id}.md"
        previous_text = path.read_text(encoding="utf-8") if path.exists() else None
        previous_index = self.index.read_bytes() if self.index.exists() else None
        rows = self._replace_row(self._rows(), row)
        try:
            self._write_text_atomic(path, text)
            self._write_rows(rows)
        except Exception:
            if previous_text is None:
                path.unlink(missing_ok=True)
            else:
                path.write_text(previous_text, encoding="utf-8")
            if previous_index is None:
                self.index.unlink(missing_ok=True)
            else:
                self.index.write_bytes(previous_index)
            raise

    @staticmethod
    def _replace_row(rows: list[dict], replacement: dict) -> list[dict]:
        result, replaced = [], False
        for row in rows:
            if row.get("id") == replacement["id"]:
                if not replaced:
                    result.append(replacement)
                    replaced = True
            else:
                result.append(row)
        if not replaced:
            result.append(replacement)
        return result

    def _write_rows(self, rows: list[dict]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._write_text_atomic(self.index, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))

    @staticmethod
    def _write_text_atomic(path: Path, text: str) -> None:
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)

    def _rows(self) -> list[dict]:
        if not self.index.exists():
            return []
        rows = []
        for line in self.index.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ExperienceError("experience index rows must be JSON objects")
                rows.append(value)
        return rows

    @staticmethod
    def _render(row: dict, body: str) -> str:
        lines = ["<experience_begin>", "<index>", f"trigger_type: {row['trigger_type']}", f"solver: {row['solver']}", f"openfoam_fork: {row.get('fork', 'any')}", f"openfoam_version: {row.get('version', '')}", f"keywords: {row.get('keywords', '')}", "conditions_json: " + json.dumps(row.get("conditions", {}), sort_keys=True), "</index>", "<guidance>", *(line.rstrip() for line in body.splitlines()), "</guidance>", "<source>", f"entry_id: {row['id']}", f"status: {row.get('status', _ACTIVE)}", f"evolution_batch: {row['source_batch']}"]
        if row.get("superseded_by"):
            lines.append(f"superseded_by: {row['superseded_by']}")
        if row.get("lifecycle_reason"):
            lines.append(f"lifecycle_reason: {row['lifecycle_reason']}")
        lines.extend(["</source>", "<experience_end>"])
        return "\n".join(lines) + "\n"
