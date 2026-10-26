"""Small orientation context for Planner; source inspection remains native."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def build_planner_context(*, worktree: str | Path, max_bytes: int = 24000) -> str:
    root = Path(worktree).resolve()
    try:
        tree = subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=root,
            text=True,
            stderr=subprocess.STDOUT,
        ).splitlines()
    except (OSError, subprocess.CalledProcessError):
        tree = []
    source_files = [item for item in tree if _source_like(item)]
    try:
        diff = subprocess.check_output(
            ["git", "diff", "--no-ext-diff", "--stat"], cwd=root, text=True, stderr=subprocess.STDOUT
        )
    except (OSError, subprocess.CalledProcessError):
        diff = ""
    experience = _experience_index(root)
    text = (
        f"Candidate worktree: {root}\n\n"
        "Use Claude native Read/Grep/Bash to inspect the actual files and active experience Markdown; this is only an orientation index.\n"
        "SOURCE FILES:\n" + "\n".join(source_files[:400]) + "\n\n"
        "CURRENT DIFF STAT:\n" + (diff or "(clean)") + "\n\n"
        "ACTIVE EXPERIENCE INDEX:\n" + json.dumps(experience, ensure_ascii=True, indent=2)
    )
    return text[:max_bytes]


def _source_like(path: str) -> bool:
    parts = Path(path).parts
    if any(part in {".git", "__pycache__", ".pytest_cache", "node_modules"} for part in parts):
        return False
    return Path(path).suffix.lower() in {".py", ".pyi", ".yaml", ".yml", ".toml", ".json", ".sh", ".cfg", ".txt", ".md"}


def _experience_index(root: Path) -> list[dict]:
    path = root / "database" / "experience" / "index.jsonl"
    rows: list[dict] = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if isinstance(value, dict) and str(value.get("status", "active")).lower() == "active":
                rows.append({key: value.get(key) for key in ("id", "trigger_type", "solver", "fork", "version", "keywords", "path")})
    # The ledger is a generated convenience index and may be absent from a
    # clean checkout. Markdown entries are the source of truth, so expose a
    # compact metadata view directly from their <index>/<source> sections.
    if not rows:
        entries = root / "database" / "experience" / "entries"
        entry_paths = sorted(entries.glob("exp-*.md")) if entries.is_dir() else []
        for entry in entry_paths:
            text = entry.read_text(encoding="utf-8", errors="replace")
            if "<source>" in text and "status: active" not in text:
                continue
            values: dict[str, str] = {
                "id": entry.stem,
                "path": str(entry.relative_to(root / "database" / "experience")),
            }
            for key, aliases in {
                "trigger_type": ("trigger_type",),
                "solver": ("solver",),
                "fork": ("openfoam_fork", "fork"),
                "version": ("openfoam_version", "version"),
                "keywords": ("keywords",),
            }.items():
                for alias in aliases:
                    marker = f"{alias}:"
                    match = next(
                        (line for line in text.splitlines() if line.strip().lower().startswith(marker)),
                        None,
                    )
                    if match:
                        values[key] = match.split(":", 1)[1].strip()
                        break
            rows.append(values)
    return rows[-100:]
