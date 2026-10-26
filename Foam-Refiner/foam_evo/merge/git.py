"""Git workspace for explicit candidate-branch integration."""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from ..models import MergeRequest, MergeResult


class MergeGitError(RuntimeError):
    pass


class MergeGitService:
    def __init__(self, repo_root: str | Path, *, worktree_root: str | Path | None = None) -> None:
        self.repo = Path(repo_root).resolve()
        self.root = Path(worktree_root or (Path(tempfile.gettempdir()) / "foam-evo-merges")).resolve()

    def _git(self, *args: str, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(["git", *args], cwd=cwd or self.repo, text=True, capture_output=True, check=False)
        if check and result.returncode:
            raise MergeGitError(result.stderr.strip() or "git command failed")
        return result

    def inspect(self, request: MergeRequest) -> tuple[str, dict[str, str]]:
        base = self._git("rev-parse", f"{request.base_ref}^{{commit}}").stdout.strip()
        diffs: dict[str, str] = {}
        for branch in request.candidate_branches:
            tip = self._git("rev-parse", f"{branch}^{{commit}}").stdout.strip()
            if tip == base:
                raise MergeGitError(f"candidate branch has no changes: {branch}")
            diff = self._git("diff", "--binary", f"{base}..{tip}").stdout
            if not diff:
                raise MergeGitError(f"candidate branch diff is empty: {branch}")
            diffs[branch] = diff
        return base, diffs

    def merge(
        self,
        request: MergeRequest,
        *,
        base_commit: str,
        merge_id: str,
        semantic_merger=None,
        artifact_dir: Path | None = None,
    ) -> MergeResult:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,100}", merge_id):
            raise MergeGitError(f"invalid merge id: {merge_id}")
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        branch = f"foam-evo/merge/{merge_id}"
        path = (self.root / merge_id).resolve()
        if path.exists():
            raise MergeGitError(f"merge worktree already exists: {path}")
        if self._git("show-ref", "--verify", "--quiet", f"refs/heads/{branch}", check=False).returncode == 0:
            raise MergeGitError(f"merge branch already exists: {branch}")
        self._git("worktree", "add", "-b", branch, str(path), base_commit)
        try:
            if semantic_merger is not None:
                prompt = f"""Integrate candidate branches {list(request.candidate_branches)} into this fresh worktree.
Read every branch diff from the base commit `{base_commit}`. Use git merge or
cherry-pick where useful, resolve conflicts semantically by editing files, run
focused tests, and commit the completed integration. Preserve improvements
from all candidates. Do not modify the main worktree.
"""
                semantic_merger.run(
                    prompt=prompt,
                    cwd=path,
                    artifact_dir=artifact_dir or (self.root / merge_id / "claude"),
                )
            else:
                for candidate in request.candidate_branches:
                    result = self._git("merge", "--no-ff", "-m", f"Merge Foam-Refiner candidate {candidate}", candidate, cwd=path, check=False)
                    if result.returncode:
                        conflicts = self._git("diff", "--name-only", "--diff-filter=U", cwd=path).stdout.splitlines()
                        return MergeResult(status="CONFLICT", base_commit=base_commit, merge_branch=branch, conflicted_files=conflicts, error=result.stderr.strip())
            unresolved = self._git("diff", "--name-only", "--diff-filter=U", cwd=path).stdout.splitlines()
            if unresolved:
                return MergeResult(status="CONFLICT", base_commit=base_commit, merge_branch=branch, conflicted_files=unresolved, error="unresolved merge conflicts")
            if self._git("rev-parse", "HEAD", cwd=path).stdout.strip() == base_commit and not self._git("status", "--porcelain", cwd=path).stdout:
                return MergeResult(status="FAILED", base_commit=base_commit, merge_branch=branch, error="integration produced no changes")
            return MergeResult(status="READY", base_commit=base_commit, merge_branch=branch)
        except Exception:
            raise

    def worktree_for(self, merge_id: str) -> Path:
        path = (self.root / merge_id).resolve()
        if not path.is_relative_to(self.root) or not path.is_dir():
            raise MergeGitError(f"merge worktree does not exist: {merge_id}")
        return path

    def commit_ready_merge(self, *, merge_id: str, message: str) -> str:
        path = self.worktree_for(merge_id)
        unresolved = self._git("diff", "--name-only", "--diff-filter=U", cwd=path).stdout.splitlines()
        if unresolved:
            raise MergeGitError("cannot commit unresolved conflicts: " + ", ".join(unresolved))
        if self._git("status", "--porcelain", cwd=path).stdout:
            self._git("add", "-A", cwd=path)
            self._git("commit", "-m", message, cwd=path)
        return self._git("rev-parse", "HEAD", cwd=path).stdout.strip()

    def changed_files(self, *, base_commit: str, merge_id: str) -> list[str]:
        return self._git("diff", "--name-only", f"{base_commit}..HEAD", cwd=self.worktree_for(merge_id)).stdout.splitlines()

    def cleanup(self, merge_id: str, *, keep_branch: bool = True) -> None:
        path = (self.root / merge_id).resolve()
        if not path.is_relative_to(self.root):
            raise MergeGitError(f"invalid merge id: {merge_id}")
        if path.is_dir():
            self._git("worktree", "remove", "--force", str(path))
        if not keep_branch:
            self._git("branch", "-D", f"foam-evo/merge/{merge_id}", check=False)
