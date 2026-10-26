"""Isolated runtime for one Foam-Refiner batch.

Every LangGraph ``Send`` creates one instance.  A runtime owns its candidate
worktree, role runner, replay pool and artifact directory; no mutable object is
shared with another batch.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..artifacts.store import ArtifactStore
from ..evaluation.matrix import EvaluationMatrix, CaseOutcome
from ..evaluation.protection import (suite_cases, evaluate_suite, read_sealed, seal,
    source_fingerprint, assets_fingerprint, write_json, git, ProtectionError)
from ..experience.store import ExperienceStore
from ..gates.diff import collect_candidate_diff
from ..gates.source import run_source_tests
from ..graph.batch_graph import BatchGraphServices
from ..locator.context import build_locator_context
from ..locator.investigator import LocatorInvestigator
from ..models import BatchRequest, BatchResult, BatchStatus, CaseRef, SourceTestResult
from ..providers.base import MarkdownProvider
from ..roles.markdown_protocol import (
    AMBIGUOUS,
    parse_code_reviewer_verdict,
    parse_iteration_verdict,
    parse_plan_exit,
    parse_plan_reviewer_verdict,
)
from ..roles.markdown_runner import RoleRunner
from ..worktree.service import CandidateWorktree, WorktreeService
from .planner_context import build_planner_context
from .replay import FoamAgentReplay


@dataclass
class BatchRuntime:
    request: BatchRequest
    worktrees: WorktreeService
    candidate: CandidateWorktree
    artifact_root: Path
    source_cases: dict[str, Path | CaseRef]
    provider: MarkdownProvider | None
    replay_runner: FoamAgentReplay | None
    role_runner: RoleRunner | None
    evaluation_history: EvaluationMatrix | None = None

    @classmethod
    def create(
        cls,
        request: BatchRequest,
        *,
        worktrees: WorktreeService,
        provider: MarkdownProvider | None,
        source_cases: dict[str, Path | CaseRef] | None,
        artifact_root: str | Path | None,
    ) -> "BatchRuntime":
        if request.regression_suite is None:
            raise ProtectionError("candidate development requires a frozen regression suite")
        suite, protected_sources = suite_cases(request.regression_suite)
        if suite["base_commit"] != request.base_commit or not set(request.case_ids) <= set(suite["failure_ids"]):
            raise ProtectionError("batch is outside the frozen base or failure scope")
        source_cases = protected_sources
        candidate = worktrees.prepare(request)
        try:
            root = Path(artifact_root or request.evo_record_root or "/tmp/foam-evo-artifacts").resolve() / request.batch_id
            root.mkdir(mode=0o700, parents=True, exist_ok=True)
            cases = dict(source_cases or {})
            replay = (
                FoamAgentReplay(
                    source_cases=cases,
                    replay_workers=request.limits.replay_workers,
                )
                if cases
                else None
            )
            baseline_matrix = replay.baseline_matrix(request.case_ids) if replay is not None else None
            runtime = cls(
                request=request,
                worktrees=worktrees,
                candidate=candidate,
                artifact_root=root,
                source_cases=cases,
                provider=provider,
                replay_runner=replay,
                role_runner=RoleRunner(provider) if provider is not None else None,
                evaluation_history=baseline_matrix,
            )
            runtime._write("request.json", request.model_dump(mode="json"))
            if cases:
                build_locator_context(source_cases=cases, case_ids=request.case_ids, root=root)
                if baseline_matrix is not None:
                    runtime._write("baseline-matrix.json", baseline_matrix.to_dict())
                    runtime._write("evaluation-matrix.json", baseline_matrix.to_dict())
            return runtime
        except Exception:
            try:
                worktrees.remove(candidate, keep_branch=False)
            except Exception:
                pass
            raise

    def as_graph_services(self) -> BatchGraphServices:
        return BatchGraphServices(
            prepare=self.prepare,
            locate=self.locate,
            plan=self.plan,
            review_plan=self.review_plan,
            implement=self.implement,
            review_code=self.review_code,
            source_tests=self.source_tests,
            replay=self.replay,
            review_iteration=self.review_iteration,
            finalize=self.finalize,
        )

    def prepare(self, request: BatchRequest) -> dict[str, Any]:
        # Worktree/context are materialized by create(); this node is retained
        # so the graph has an explicit beginning and can record its paths.
        return {"branch_name": self.candidate.branch_name, "worktree_path": str(self.candidate.path)}

    def locate(self, state: dict[str, Any]) -> dict[str, Any]:
        iteration = int(state.get("iteration", 1))
        target = self._iteration_dir(iteration) / "locator"
        failure = self._failure_context(state)
        if self.role_runner is None or not self.source_cases:
            # Direct ``batch`` may be intentionally run without an LLM when a
            # human supplied the cluster reason. This is explicit, not a
            # silent global-cluster fallback.
            text = "# Locator diagnosis\n\n" + (self.request.cluster_reason or "No LLM locator configured.") + "\n"
            self._write_relative(target / "locator.md", text)
            return {"locator_path": str(target / "locator.md")}
        try:
            document = LocatorInvestigator(self.role_runner).run(
                context_root=self.artifact_root,
                artifact_dir=target,
                batch_id=self.request.batch_id,
                candidate_worktree=self.candidate.path,
                failure_context=failure,
            )
            return {"locator_path": str(document.path)}
        except Exception as exc:
            # Keep the normal planning/retry budget in control while exposing
            # a concrete provider failure to Planner and the next iteration.
            output = target / "locator.md"
            self._write_relative(output, f"# Locator failure\n\n{type(exc).__name__}: {exc}\n")
            return {"locator_path": str(output), "locator_error": f"{type(exc).__name__}: {exc}"}

    def plan(self, state: dict[str, Any]) -> dict[str, Any]:
        iteration = int(state.get("iteration", 1))
        round_no = int(state.get("plan_round", 0)) + 1
        target = self._iteration_dir(iteration) / "plan" / f"round-{round_no:02d}"
        output = target / "plan.md"
        if self.role_runner is None:
            text = "# Plan\n\nNo model configured for this direct batch.\n\n## Exit\nNO_EVALUATION\n"
            self._write_relative(output, text)
            return {"plan_path": str(output), "plan_exit": "NO_EVALUATION"}
        prompt = self._planner_prompt(state, output)
        try:
            document = self.role_runner.run(
                role="planner",
                prompt=prompt,
                artifact_dir=target,
                output_name="plan.md",
                cwd=self.candidate.path,
            )
            exit_token = parse_plan_exit(document.markdown)
            if exit_token is None:
                self._write_relative(target / "protocol-error.md", "Missing or ambiguous ## Exit token.\n")
            return {"plan_path": str(document.path), "plan_exit": exit_token}
        except Exception as exc:
            self._write_relative(output, f"# Plan role failure\n\n{type(exc).__name__}: {exc}\n")
            return {"plan_path": str(output), "plan_exit": None, "plan_error": f"{type(exc).__name__}: {exc}"}

    def review_plan(self, state: dict[str, Any]) -> dict[str, Any]:
        iteration = int(state.get("iteration", 1))
        round_no = int(state.get("plan_round", 1))
        target = self._iteration_dir(iteration) / "plan" / f"round-{round_no:02d}" / "review"
        output = target / "plan-reviewer.md"
        if self.role_runner is None:
            text = "# Plan review\n\n## Verdict\nAPPROVE\n"
            self._write_relative(output, text)
            return {"plan_review_path": str(output), "plan_review_verdict": "APPROVE"}
        plan_path = Path(state["plan_path"])
        prompt = f"""# Foam-Refiner Plan Reviewer — batch {self.request.batch_id}

Read the complete plan at `{plan_path}`, the Locator artifact at
`{state.get('locator_path', '(missing)')}`, the selected case summaries and
the actual candidate source tree. Audit that the plan addresses the shared
failure, names concrete implementation and test work, explains cross-case
impact, and includes a realistic test matrix. Do not require an allowed-files
list: the Implementer may make any justified source or experience change.

Write your full review to `{output}`. The machine-readable decision must use
this exact final block (replace the token with exactly one valid alternative):

## Verdict
APPROVE

Valid alternatives are `REJECT_ARCHITECTURAL` and `REJECT_COMPLETENESS`.
Do not write `### Verdict`, `## Verdict Justification`, or put two decisions
on one line, for example:

### Verdict
APPROVE

## Verdict Justification
APPROVE or REJECT_COMPLETENESS

If you need to explain the decision, put the explanation before the final
`## Verdict` section (for example under `## Analysis`). End the document with
one and only one `## Verdict` block.
Previous review feedback, if any:
{self._read(state.get('plan_review_path'))}
"""
        try:
            document = self.role_runner.run(
                role="plan-reviewer",
                prompt=prompt,
                artifact_dir=target,
                output_name="plan-reviewer.md",
                cwd=self.candidate.path,
            )
            verdict = parse_plan_reviewer_verdict(document.markdown)
        except Exception as exc:
            self._write_relative(output, f"# Plan reviewer failure\n\n{type(exc).__name__}: {exc}\n")
            verdict = AMBIGUOUS
        return {"plan_review_path": str(output), "plan_review_verdict": verdict}

    def implement(self, state: dict[str, Any]) -> dict[str, Any]:
        iteration = int(state.get("iteration", 1))
        round_no = int(state.get("code_round", 0)) + 1
        target = self._iteration_dir(iteration) / "code" / f"round-{round_no:02d}"
        output = target / "implementer.md"
        if self.role_runner is None:
            self._write_relative(output, "# Implementer\n\nNo Foam-Refiner provider configured.\n")
            return {"implementer_path": str(output), "implementation_ok": False, "implementation_error": "provider unavailable"}
        prompt = f"""# Foam-Refiner Implementer — batch {self.request.batch_id}

Work in the candidate checkout `{self.candidate.path}`. Read the approved plan
at `{state.get('plan_path')}`, its review at `{state.get('plan_review_path')}`,
and the Locator/case evidence. Apply the plan using native Read/Grep/Bash and
editing tools. Preserve useful changes already present from prior code rounds.
You may change source, tests, and Foam-Runner experience Markdown entries when
the plan justifies it; do not hand-edit generated `database/experience/index.jsonl`
or any FAISS files. The engine rebuilds the ledger and FAISS before every
replay, so Markdown entries are the source of truth. Run focused tests. Do not
change git history (no commit, reset, checkout, merge, or rebase); the engine
records the candidate after review.

Previous code review:
{self._read(state.get('code_review_path'))}
Previous implementation result:
{self._read(state.get('implementer_path'))}
Previous replay failure:
{state.get('replay_failure_summary', '(none)')}
Previous source-test result:
{json.dumps(state.get('source_test_result').model_dump(mode='json') if state.get('source_test_result') is not None else None, ensure_ascii=True)}

Write a concise implementation report to `{output}` after editing. Use this
shape:

## Changes
Files changed and the behavior addressed.

## Tests
Commands actually run and their results.

## Remaining Issues
Any unresolved failure or follow-up.

Do not add `## Verdict` or `## Exit`; the engine routes from the source tree,
review artifacts, tests, and replay evaluator rather than this prose report.
"""
        try:
            document = self.role_runner.run(
                role="implementer",
                prompt=prompt,
                artifact_dir=target,
                output_name="implementer.md",
                cwd=self.candidate.path,
            )
            after = self._diff()
            changed = list(after.changed_files)
            if not changed:
                return {
                    "implementer_path": str(document.path),
                    "implementation_ok": False,
                    "implementation_error": "implementer produced no candidate changes",
                    "changed_files": [],
                }
            return {"implementer_path": str(document.path), "implementation_ok": True, "changed_files": changed}
        except Exception as exc:
            self._write_relative(output, f"# Implementer failure\n\n{type(exc).__name__}: {exc}\n")
            return {"implementer_path": str(output), "implementation_ok": False, "implementation_error": f"{type(exc).__name__}: {exc}"}

    def review_code(self, state: dict[str, Any]) -> dict[str, Any]:
        iteration = int(state.get("iteration", 1))
        round_no = int(state.get("code_round", 1))
        target = self._iteration_dir(iteration) / "code" / f"round-{round_no:02d}" / "review"
        output = target / "code-reviewer.md"
        if self.role_runner is None:
            self._write_relative(output, "# Code review\n\n## Verdict\nAPPROVE\n")
            return {"code_review_path": str(output), "code_review_verdict": "APPROVE"}
        diff = self._diff().patch
        prompt = f"""# Foam-Refiner Code Reviewer — batch {self.request.batch_id}

Read the approved plan `{state.get('plan_path')}`, the implementation report
`{state.get('implementer_path')}`, the candidate source tree, and the complete
candidate diff below. Check that the actual change addresses the common root
cause, does not regress unrelated cases, and includes focused tests or a clear
reason tests are not applicable. Review experience Markdown changes as content
and verify that generated ledger and FAISS assets are rebuilt before replay;
do not approve a hand-edited generated index as a source change.

Candidate diff:
{diff[-50000:]}

Write the full review to `{output}`. End with this exact block, replacing the
token with exactly one valid choice:

## Verdict
APPROVE

The only other valid choice is `REJECT_IMPL`. Do not write `### Verdict`,
`## Verdict Justification`, or combine choices, for example:

### Verdict
APPROVE

## Verdict Justification
APPROVE and REJECT_IMPL

Put review reasoning before the final control block, then finish with one and
only one `## Verdict` block. On rejection, list concrete corrections so the
next Implementer round can act on them.
"""
        try:
            document = self.role_runner.run(
                role="code-reviewer",
                prompt=prompt,
                artifact_dir=target,
                output_name="code-reviewer.md",
                cwd=self.candidate.path,
            )
            verdict = parse_code_reviewer_verdict(document.markdown)
        except Exception as exc:
            self._write_relative(output, f"# Code reviewer failure\n\n{type(exc).__name__}: {exc}\n")
            verdict = AMBIGUOUS
        return {"code_review_path": str(output), "code_review_verdict": verdict}

    def source_tests(self, state: dict[str, Any]) -> SourceTestResult:
        result = run_source_tests(self.candidate.path, timeout_s=self.request.limits.source_test_timeout_s)
        target = self._iteration_dir(int(state.get("iteration", 1))) / "source-tests"
        self._write("/".join(target.relative_to(self.artifact_root).parts) + "/result.json", result.model_dump(mode="json"))
        return result

    def replay(self, state: dict[str, Any]) -> EvaluationMatrix:
        iteration = int(state.get("iteration", 1))
        target = self._iteration_dir(iteration) / "replay"
        try:
            if (self.candidate.path / "database").is_dir():
                experience_status = ExperienceStore(self.candidate.path).prepare_for_replay()
            else:
                # Minimal test/legacy checkouts may not carry Foam-Runner's
                # optional experience database at all. That is an empty RAG,
                # not a replay infrastructure failure.
                experience_status = {"available": False, "entries": 0, "faiss_rebuilt": False}
        except Exception as exc:
            experience_status = {
                "error": f"{type(exc).__name__}: {exc}",
                "replay_blocked": True,
            }
            self._write_relative(target / "experience-error.md", experience_status["error"] + "\n")
        self._write(
            "/".join(target.relative_to(self.artifact_root).parts) + "/experience.json",
            experience_status,
        )
        workflow_version = f"iter-{iteration:03d}"
        if self.replay_runner is None:
            matrix = EvaluationMatrix(tuple())
        elif experience_status.get("replay_blocked"):
            # Preserve the baseline/candidate contract while making the reason
            # explicit in every selected case. No candidate is promoted from a
            # replay that did not have a valid experience environment.
            baseline = self.replay_runner.baseline_matrix(self.request.case_ids)
            blocked = tuple(
                type(item)(
                    case_id=item.case_id,
                    passed=False,
                    reason="experience replay preparation failed: " + experience_status["error"],
                    baseline_failed_checks=item.baseline_failed_checks,
                    baseline_checks=item.baseline_checks,
                    candidate_checks=(),
                    baseline_passed=item.baseline_passed,
                    infrastructure_blocked=True,
                    comparison_error=experience_status["error"],
                )
                for item in baseline.outcomes
            )
            matrix = self.evaluation_history.append_version(workflow_version, blocked) if self.evaluation_history else EvaluationMatrix(
                blocked,
                workflow_versions=("baseline", workflow_version),
                current_workflow_version=workflow_version,
            )
        else:
            try:
                _commit_all(self.candidate.path, f"{self.request.batch_id}-iter-{iteration}")
                tested_source = source_fingerprint(self.candidate.path)
                tested_assets = assets_fingerprint(self.candidate.path)
                tested_commit = git(self.candidate.path, "rev-parse", "HEAD")
                tests = run_source_tests(self.candidate.path, timeout_s=self.request.limits.source_test_timeout_s)
                self._write_relative(target / "source-tests.json", json.dumps(tests.model_dump(mode="json")))
                if not tests.passed:
                    raise ProtectionError("source tests failed on committed candidate")
                if tested_source != source_fingerprint(self.candidate.path) or tested_assets != assets_fingerprint(self.candidate.path) or tested_commit != git(self.candidate.path, "rev-parse", "HEAD"):
                    raise ProtectionError("source tests changed the committed replay state")
                matrix, evidence = evaluate_suite(
                    repo=self.candidate.path, suite_path=self.request.regression_suite,
                    failure_ids=self.request.case_ids, output=target,
                    workers=self.request.limits.replay_workers)
                evidence = seal({**evidence, "source_tests_passed": True,
                                 "plan_approved": state.get("plan_review_verdict") == "APPROVE",
                                 "code_approved": state.get("code_review_verdict") == "APPROVE"})
                write_json(target / "validation.json", evidence)
            except Exception as exc:
                matrix = EvaluationMatrix((CaseOutcome("validation", False, str(exc),
                    comparison_error=str(exc), infrastructure_blocked=True),))
        if self.evaluation_history is not None:
            matrix = self.evaluation_history.append_version(workflow_version, matrix.outcomes)
        self.evaluation_history = matrix
        self._write("/".join(target.relative_to(self.artifact_root).parts) + "/matrix.json", matrix.to_dict())
        self._write("evaluation-matrix.json", matrix.to_dict())
        return matrix

    def review_iteration(self, state: dict[str, Any]) -> dict[str, Any]:
        iteration = int(state.get("iteration", 1))
        matrix = state.get("replay_matrix")
        target = self._iteration_dir(iteration) / "reviewer"
        output = target / "reviewer.md"
        if self.role_runner is None:
            verdict = "CONVERGED" if isinstance(matrix, EvaluationMatrix) and matrix.gate_passed else "NEED_MORE_WORK"
            self._write_relative(output, f"# Iteration review\n\n## Verdict\n{verdict}\n")
            return {"iteration_review_path": str(output), "iteration_verdict": verdict}
        prompt = f"""# Foam-Refiner Iteration Reviewer — batch {self.request.batch_id}, iteration {iteration}

Read the plan, implementation and code review artifacts under
`{self._iteration_dir(iteration)}`, the replay matrix at
`{self._iteration_dir(iteration) / 'replay' / 'matrix.json'}`, and the per-case
replay artifacts. Decide whether the common failure improved without
regression. The deterministic matrix gate is authoritative: do not claim
CONVERGED if it is not comparable and materially improved.

Write the review to `{output}`. End with this exact block, replacing the token
with exactly one valid choice:

## Verdict
CONVERGED

Valid alternatives are `NEED_MORE_WORK`, `FUNDAMENTAL_LIMIT_MODEL`, and
`FUNDAMENTAL_LIMIT_ARCHITECTURE`. Do not write `### Verdict`,
`## Verdict Justification`, or multiple choices, for example:

### Verdict
CONVERGED

## Verdict Justification
CONVERGED or NEED_MORE_WORK

Put the analysis before the final control block and finish with one and only
one `## Verdict` block.
Previous iteration review:
{self._read(state.get('iteration_review_path'))}
"""
        try:
            document = self.role_runner.run(
                role="iteration-reviewer",
                prompt=prompt,
                artifact_dir=target,
                output_name="reviewer.md",
                cwd=self.candidate.path,
            )
            verdict = parse_iteration_verdict(document.markdown)
        except Exception as exc:
            self._write_relative(output, f"# Iteration reviewer failure\n\n{type(exc).__name__}: {exc}\n")
            verdict = AMBIGUOUS
        return {"iteration_review_path": str(output), "iteration_verdict": verdict}

    def finalize(self, state: dict[str, Any]) -> BatchResult:
        status = state.get("status", BatchStatus.FAILED)
        if status not in {BatchStatus.CANDIDATE_READY, BatchStatus.NO_EVOLUTION, BatchStatus.FAILED}:
            status = BatchStatus.FAILED
        candidate_commit: str | None = None
        error: str | None = state.get("error")
        if status == BatchStatus.CANDIDATE_READY:
            try:
                diff = self._diff()
                if not diff.changed_files:
                    raise ValueError("candidate has no changes to commit")
                validation = self._iteration_dir(int(state.get("iteration", 1))) / "replay" / "validation.json"
                evidence = read_sealed(validation)
                if state.get("iteration_verdict") != "CONVERGED" or not evidence.get("source_tests_passed"):
                    raise ProtectionError("candidate has not completed acceptance")
                if not evidence.get("plan_approved") or not evidence.get("code_approved"):
                    raise ProtectionError("candidate reviews were not approved")
                candidate_commit = git(self.candidate.path, "rev-parse", "HEAD")
                if (candidate_commit != evidence["tested_commit"] or
                    source_fingerprint(self.candidate.path) != evidence["source_fingerprint"] or
                    assets_fingerprint(self.candidate.path) != evidence["assets_fingerprint"]):
                    raise ProtectionError("candidate changed after replay")
                write_json(validation, seal({**evidence, "iteration_verdict": "CONVERGED"}))
            except Exception as exc:
                status = BatchStatus.FAILED
                candidate_commit = None
                error = f"{type(exc).__name__}: {exc}"
        branch = self.candidate.branch_name if status == BatchStatus.CANDIDATE_READY else None
        if status != BatchStatus.CANDIDATE_READY:
            try:
                self.worktrees.remove(self.candidate, keep_branch=False)
            except Exception as exc:
                error = error or f"candidate cleanup failed: {type(exc).__name__}: {exc}"
        summary = (
            "candidate branch ready"
            if status == BatchStatus.CANDIDATE_READY
            else state.get("replay_failure_summary")
            or state.get("implementation_error")
            or "batch did not produce a candidate"
        )
        result = BatchResult(
            batch_id=self.request.batch_id,
            status=status,
            base_commit=self.request.base_commit,
            branch_name=branch,
            candidate_commit=candidate_commit,
            validation_path=str(validation) if candidate_commit and status == BatchStatus.CANDIDATE_READY else None,
            summary=str(summary),
            error=error,
        )
        result_path = self._write("result.json", result.model_dump(mode="json"))
        result.result_path = str(result_path)
        self._write("result.json", result.model_dump(mode="json"))
        return result

    def cleanup(self) -> None:
        try:
            self.worktrees.remove(self.candidate, keep_branch=False)
        except Exception:
            pass

    def _iteration_dir(self, iteration: int) -> Path:
        return self.artifact_root / "iterations" / f"iter-{iteration:03d}"

    def _write(self, relative: str, value: Any) -> Path:
        return ArtifactStore(self.artifact_root).write_json(relative, value)

    def _write_relative(self, path: Path, value: str) -> Path:
        return ArtifactStore(self.artifact_root).write_text(str(path.resolve().relative_to(self.artifact_root)), value)

    def _diff(self):
        return collect_candidate_diff(self.candidate.path, self.request.base_commit)

    def _read(self, path: object, limit: int = 16000) -> str:
        if not path:
            return "(none)"
        try:
            return Path(str(path)).read_text(encoding="utf-8", errors="replace")[-limit:]
        except OSError:
            return "(unavailable)"

    def _planner_prompt(self, state: dict[str, Any], output: Path) -> str:
        context = build_planner_context(worktree=self.candidate.path)
        return f"""# Foam-Refiner Planner — batch {self.request.batch_id}, iteration {state.get('iteration', 1)}

Read the Locator investigation at `{state.get('locator_path')}` and inspect
the selected case index `{self.artifact_root / 'context' / 'case-index.json'}`.
Use native Read/Grep/Bash to verify the diagnosis against the candidate source
tree `{self.candidate.path}` and the actual Foam-Runner inputs. The orientation
context below is not a substitute for source inspection:

{context}

Write a complete implementation plan to `{output}`. Explain the shared
cross-case impact, the concrete files/subsystems to inspect or change, and a
test matrix covering the selected cases and relevant unit/integration checks.
Choose exactly one semantic exit and put it in this exact final block:

## Exit
CODE_ONLY

Replace `CODE_ONLY` with exactly one of `KNOWLEDGE_ONLY`,
`CODE_AND_KNOWLEDGE`, or `NO_EVALUATION`. Do not write `### Exit`,
`## Exit: CODE_ONLY`, put two choices on one line, or mention a choice only in
prose, for example:

### Exit
CODE_ONLY

## Exit: CODE_ONLY

## Exit
CODE_ONLY or CODE_AND_KNOWLEDGE

Keep the plan reasoning before this final control block and emit exactly one
`## Exit` block.
Do not emit a machine-readable role schema. If knowledge
is part of the plan, describe the experience Markdown/index changes for the
Implementer; it will be reviewed like any other diff.

Previous replay matrix:
{self._read(self._iteration_dir(int(state.get('iteration', 1)) - 1) / 'replay' / 'matrix.json')}
Previous plan-review feedback:
{self._read(state.get('plan_review_path'))}
Previous code-review feedback:
{self._read(state.get('code_review_path'))}
Previous replay failure:
{state.get('replay_failure_summary', '(none)')}
"""

    def _failure_context(self, state: dict[str, Any]) -> str:
        matrix = state.get("replay_matrix")
        return json.dumps(
            {
                "iteration": state.get("iteration", 1),
                "replay_failure_summary": state.get("replay_failure_summary"),
                "replay_matrix": matrix.to_dict() if isinstance(matrix, EvaluationMatrix) else None,
                "plan_review": self._read(state.get("plan_review_path")),
                "code_review": self._read(state.get("code_review_path")),
            },
            ensure_ascii=True,
        )


class BatchRuntimeFactory:
    """Read-only dependency holder that constructs isolated runtimes."""

    def __init__(
        self,
        *,
        repo_root: str | Path,
        provider: MarkdownProvider | None = None,
        source_cases: dict[str, Path | CaseRef] | None = None,
        artifact_root: str | Path | None = None,
        worktree_root: str | Path | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.provider = provider
        self.source_cases = dict(source_cases or {})
        self.artifact_root = Path(artifact_root).resolve() if artifact_root else None
        self.worktree_root = Path(worktree_root).resolve() if worktree_root else None

    def run(self, request: BatchRequest) -> BatchResult:
        worktrees = WorktreeService(self.repo_root, root=self.worktree_root)
        runtime: BatchRuntime | None = None
        try:
            runtime = BatchRuntime.create(
                request,
                worktrees=worktrees,
                provider=self.provider,
                source_cases=self.source_cases,
                artifact_root=self.artifact_root,
            )
            from ..graph.runner import BatchExecution

            result = BatchExecution(services=runtime.as_graph_services()).run(request)
            if result.status not in {BatchStatus.CANDIDATE_READY, BatchStatus.NO_EVOLUTION}:
                runtime.cleanup()
            return result
        except Exception as exc:
            if runtime is not None:
                runtime.cleanup()
            return BatchResult(
                batch_id=request.batch_id,
                status=BatchStatus.FAILED,
                base_commit=request.base_commit,
                summary=f"{type(exc).__name__}: {exc}",
                error=f"{type(exc).__name__}: {exc}",
            )


def _commit_all(path: Path, batch_id: str) -> str:
    subprocess.run(["git", "add", "-A"], cwd=path, check=True, capture_output=True, text=True)
    # Foam-Runner creates these products while running tests or rebuilding RAG.
    # They are deliberately outside the candidate source contract even when a
    # checkout's historical .gitignore predates one of the patterns.
    subprocess.run(
        [
            "git",
            "reset",
            "--",
            "record",
            "evorecord",
            "runs",
            "output",
            "database/faiss",
            "database/experience/faiss",
            "database/experience/index.jsonl",
            "database/experience/validations.jsonl",
            "database/foamgpt/data",
        ],
        cwd=path,
        check=False,
        capture_output=True,
        text=True,
    )
    staged = subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=path, text=True).splitlines()
    if not staged:
        # Claude is asked not to commit, but accepting an already committed
        # candidate keeps the artifact protocol independent of that prose.
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
    subprocess.run(
        ["git", "commit", "-m", f"Foam-Refiner batch {batch_id}"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
