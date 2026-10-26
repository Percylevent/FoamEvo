"""Git worktree and per-batch runtime materialization."""

from __future__ import annotations

import subprocess
import tempfile
import hashlib
from dataclasses import dataclass
from pathlib import Path

from ..models import BatchRequest
from .assets import materialize_runtime_assets


class WorktreeError(RuntimeError):
    """Raised when a candidate worktree cannot be prepared safely."""


@dataclass(frozen=True)
class CandidateWorktree:
    batch_id: str
    branch_name: str
    path: Path
    base_commit: str


class WorktreeService:
    def __init__(self, repo_root: str | Path, *, root: str | Path | None = None) -> None:
        self.repo = Path(repo_root).resolve()
        self.root = Path(root or (Path(tempfile.gettempdir()) / "foam-evo-worktrees")).resolve()

    def _git(self, *args: str, cwd: Path | None = None) -> str:
        result = subprocess.run(
            ["git", *args], cwd=cwd or self.repo, text=True,
            capture_output=True, check=False,
        )
        if result.returncode:
            raise WorktreeError(result.stderr.strip() or "git command failed")
        return result.stdout.strip()

    def prepare(self, request: BatchRequest) -> CandidateWorktree:
        resolved_base = self._git("rev-parse", f"{request.base_commit}^{{commit}}")
        if resolved_base != request.base_commit:
            raise WorktreeError("base commit did not resolve to the requested SHA")
        if not request.batch_id.replace("-", "").replace("_", "").isalnum():
            raise WorktreeError(f"invalid batch id: {request.batch_id}")
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        path = (self.root / request.batch_id).resolve()
        if path.exists():
            raise WorktreeError(f"candidate worktree already exists: {path}")
        branch = f"foam-evo/batch/{request.batch_id}"
        result = subprocess.run(
            ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"],
            cwd=self.repo,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            raise WorktreeError(f"candidate branch already exists: {branch}")
        self._git("worktree", "add", "-b", branch, str(path), request.base_commit)
        try:
            if self._git("rev-parse", "HEAD", cwd=path) != request.base_commit:
                raise WorktreeError("candidate HEAD does not match base commit")
            materialize_runtime_assets(self.repo, path)
            return CandidateWorktree(request.batch_id, branch, path, request.base_commit)
        except Exception:
            subprocess.run(["git", "worktree", "remove", "--force", str(path)], cwd=self.repo)
            subprocess.run(["git", "branch", "-D", branch], cwd=self.repo, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            raise

    def snapshot(self, path: Path) -> dict[str, str]:
        status = self._git("status", "--porcelain=v1", "--untracked-files=all", cwd=path)
        diff = self._git("diff", "--binary", cwd=path)
        return {
            "head": self._git("rev-parse", "HEAD", cwd=path),
            "status_sha256": hashlib.sha256(status.encode()).hexdigest(),
            "diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
        }

    def remove(self, candidate: CandidateWorktree, *, keep_branch: bool = True) -> None:
        self._git("worktree", "remove", "--force", str(candidate.path))
        if not keep_branch:
            self._git("branch", "-D", candidate.branch_name)

    def cleanup(self, batch_id: str, *, keep_branch: bool = True) -> None:
        path = (self.root / batch_id).resolve()
        if not path.is_relative_to(self.root):
            raise WorktreeError(f"invalid batch id: {batch_id}")
        if path.is_dir():
            self._git("worktree", "remove", "--force", str(path))
        if not keep_branch:
            self._git("branch", "-D", f"foam-evo/batch/{batch_id}")
