"""Collect source changes while ignoring Foam-Runner runtime products."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CandidateDiff:
    changed_files: tuple[str, ...]
    patch: str
    patch_sha256: str


class DiffCollectionError(ValueError):
    """Raised when git cannot describe the candidate worktree."""


def collect_candidate_diff(
    worktree: str | Path,
    parent_commit: str,
) -> CandidateDiff:
    root = Path(worktree).resolve()
    names = _git(root, "diff", "--name-only", parent_commit, "--").splitlines()
    untracked = _git(root, "ls-files", "--others", "--exclude-standard").splitlines()
    changed = tuple(sorted(
        name
        for name in set(names + untracked)
        if not _is_generated_runtime_path(name)
    ))
    tracked = [name for name in changed if name in names]
    untracked_changed = [name for name in changed if name in untracked]
    patch_parts = [
        _git(root, "diff", "--binary", parent_commit, "--", *tracked)
        if tracked
        else ""
    ]
    for name in untracked_changed:
        patch_parts.append(_git_diff_untracked(root, name))
    patch = "".join(patch_parts)
    return CandidateDiff(changed, patch, hashlib.sha256(patch.encode()).hexdigest())


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True, check=False)
    if result.returncode:
        raise DiffCollectionError(result.stderr.strip() or "git diff command failed")
    return result.stdout


def _is_generated_runtime_path(name: str) -> bool:
    """Exclude Foam-Runner runtime/index products from candidate source diffs."""

    normalized = name.replace("\\", "/").lstrip("./")
    return (
        normalized.startswith(("record/", "evorecord/", "runs/", "output/"))
        or normalized in {"database/experience/index.jsonl", "database/experience/validations.jsonl"}
        or normalized.startswith(("database/faiss/", "database/experience/faiss/", "database/foamgpt/data/"))
        or normalized.endswith((".faiss", ".index"))
    )


def _git_diff_untracked(root: Path, name: str) -> str:
    result = subprocess.run(
        ["git", "diff", "--no-index", "--binary", "--", "/dev/null", name],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode not in {0, 1}:
        raise DiffCollectionError(result.stderr.strip() or f"cannot diff untracked file: {name}")
    return result.stdout
