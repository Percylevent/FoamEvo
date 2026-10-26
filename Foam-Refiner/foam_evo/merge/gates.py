"""Post-merge experience and source gates."""

from __future__ import annotations

import shutil
import subprocess
import os
import shlex
import sys
import tempfile
from pathlib import Path


class MergeGateError(RuntimeError):
    pass


def rebuild_experience(worktree: str | Path, *, timeout_s: float = 1800.0) -> None:
    root = Path(worktree).resolve()
    # Foam-Runner's indexer writes all generated FAISS stores below
    # database/faiss.  Experience JSON entries remain tracked; only this
    # derived tree is replaced during the merge gate.
    faiss = root / "database" / "faiss"
    backup = faiss.with_name("faiss.foam-evo-backup")
    if backup.exists():
        shutil.rmtree(backup)
    if faiss.exists():
        shutil.copytree(faiss, backup)
        shutil.rmtree(faiss)
    try:
        command_text = os.getenv("FOAM_EVO_REINDEX_COMMAND")
        if command_text:
            command = shlex.split(command_text)
        else:
            indexer = root / "database" / "script" / "init_experience_db.py"
            if not indexer.is_file():
                # Retrieval is deliberately optional in older Foam-Runner
                # checkouts.  The tracked Markdown remains valid without a
                # generated FAISS derivative.
                if backup.exists():
                    if faiss.exists():
                        shutil.rmtree(faiss)
                    backup.rename(faiss)
                return
            command = [sys.executable, str(indexer)]
        result = subprocess.run(command, cwd=root, text=True, capture_output=True, timeout=timeout_s, check=False)
        if result.returncode != 0:
            raise MergeGateError(result.stderr.strip() or "experience reindex failed")
        if backup.exists():
            shutil.rmtree(backup)
    except Exception:
        if faiss.exists():
            shutil.rmtree(faiss)
        if backup.exists():
            backup.rename(faiss)
        raise


def run_full_tests(worktree: str | Path, *, timeout_s: float = 3600.0) -> None:
    from ..gates.source import run_source_tests
    result = run_source_tests(worktree, timeout_s=timeout_s)
    if not result.passed:
        raise MergeGateError(result.output)
