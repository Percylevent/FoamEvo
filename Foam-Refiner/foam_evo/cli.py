"""Foam-Refiner command line entry point."""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from .artifacts.evo_record import EvoRecordStore
from .execution.batch_runtime import BatchRuntimeFactory
from .graph.evolve_graph import build_evolve_graph
from .merge.service import MergeService, adopt_validation
from .evaluation.protection import freeze_suite, suite_cases
from .models import BatchLimits, BatchRequest, EvolveRequest, MergeRequest
from .providers.factory import configured_provider
from .selection.clustering import GlobalClusterer
from .selection.records import RecordSelector
from .worktree.service import WorktreeService


def _limits(args: argparse.Namespace) -> BatchLimits:
    return BatchLimits(
        max_iterations=args.max_iterations,
        max_plan_rounds=args.max_plan_rounds,
        max_code_rounds=args.max_code_rounds,
        source_test_timeout_s=args.source_test_timeout_s,
        replay_workers=args.replay_workers,
    )


def _evolve_args(parser: argparse.ArgumentParser, *, parallel: bool) -> None:
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--base-ref", default="main")
    parser.add_argument("--regression-suite", type=Path, required=True)
    parser.add_argument("--record-root", type=Path, default=Path("record"))
    parser.add_argument("--evo-record-root", type=Path, default=Path("evorecord"))
    parser.add_argument("--worktree-root", type=Path, default=None)
    parser.add_argument("--record-id", default=None)
    parser.add_argument("--case-id", action="append", dest="case_ids", default=None)
    parser.add_argument("--max-cases-per-batch", type=int, default=10)
    if parallel:
        parser.add_argument("--parallel-workers", type=int, default=1)
    parser.add_argument("--replay-workers", type=int, default=1)
    parser.add_argument("--benchmark-root", type=Path, default=None)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--max-plan-rounds", type=int, default=3)
    parser.add_argument("--max-code-rounds", type=int, default=3)
    parser.add_argument("--source-test-timeout-s", type=float, default=1800.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="foam-refiner")
    sub = parser.add_subparsers(dest="command", required=True)
    batch = sub.add_parser("batch", help="evolve one explicitly selected case batch")
    _evolve_args(batch, parallel=False)
    evolve = sub.add_parser("evolve", help="cluster failures and evolve batches in parallel")
    _evolve_args(evolve, parallel=True)

    merge = sub.add_parser("merge", help="explicitly integrate candidate branches")
    merge.add_argument("--repo-root", type=Path, required=True)
    merge.add_argument("--base-ref", default="main")
    merge.add_argument("--branch", action="append", dest="branches", required=True)
    merge.add_argument("--merge-id", default=None)
    merge.add_argument("--candidate-evidence", type=Path, action="append", required=True)
    merge.add_argument("--replay-workers", type=int, default=1)
    merge.add_argument("--adopt", action="store_true")
    freeze = sub.add_parser("freeze-suite", help="freeze evolution inputs and refresh baseline replay")
    freeze.add_argument("--repo-root", type=Path, required=True)
    freeze.add_argument("--base-ref", default="main")
    freeze.add_argument("--record-root", type=Path, required=True)
    freeze.add_argument("--record-id", required=True)
    freeze.add_argument("--benchmark-root", type=Path)
    freeze.add_argument("--evolution-cases", type=Path, required=True, help="JSON list of declared evolution case IDs")
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--replay-workers", type=int, default=1)
    adopt = sub.add_parser("adopt", help="promote a validated integration and update retention")
    adopt.add_argument("--repo-root", type=Path, required=True)
    adopt.add_argument("--base-ref", default="main")
    adopt.add_argument("--validation", type=Path, required=True)
    cleanup = sub.add_parser("cleanup")
    cleanup.add_argument("--repo-root", type=Path, required=True)
    cleanup.add_argument("--kind", choices=["batch", "merge"], required=True)
    cleanup.add_argument("--id", required=True, dest="artifact_id")
    cleanup.add_argument("--delete-branch", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "freeze-suite":
        try:
            request = EvolveRequest(repo_root=args.repo_root.resolve(), base_ref=args.base_ref,
                record_root=args.record_root.resolve(), record_id=args.record_id,
                benchmark_root=args.benchmark_root)
            selector = RecordSelector(request.repo_root)
            sources = selector.load(request, include_successes=True)
            allowed = json.loads(args.evolution_cases.read_text(encoding="utf-8"))
            if not isinstance(allowed, list) or not all(isinstance(x, str) for x in allowed):
                raise ValueError("evolution-cases must contain a JSON string list")
            value = freeze_suite(repo=request.repo_root, base=selector._resolve_base(args.base_ref),
                sources={x.case_id: x for x in sources}, allowed_ids=allowed,
                path=args.output.resolve(), workers=args.replay_workers)
            print(json.dumps({"suite": str(args.output.resolve()), "digest": value["digest"]}))
            return 0
        except Exception as exc:
            print(f"freeze-suite failed: {exc}")
            return 1
    if args.command == "adopt":
        try:
            commit = adopt_validation(args.repo_root, args.base_ref, args.validation)
            print(json.dumps({"adopted_commit": commit}))
            return 0
        except Exception as exc:
            print(f"adoption blocked: {exc}")
            return 1
    if args.command == "cleanup":
        try:
            if args.kind == "batch":
                WorktreeService(args.repo_root).cleanup(args.artifact_id, keep_branch=not args.delete_branch)
            else:
                from .merge.git import MergeGitService

                MergeGitService(args.repo_root).cleanup(args.artifact_id, keep_branch=not args.delete_branch)
        except Exception as exc:
            print(f"foam-refiner cleanup error: {type(exc).__name__}: {exc}")
            return 1
        return 0

    if args.command == "merge":
        request = MergeRequest(repo_root=args.repo_root, base_ref=args.base_ref, candidate_branches=args.branches,
            candidate_evidence=args.candidate_evidence, replay_workers=args.replay_workers, adopt=args.adopt)
        merge_id = args.merge_id or f"merge-{uuid.uuid4().hex[:12]}"
        try:
            result = MergeService(request.repo_root).run(request, merge_id=merge_id)
        except Exception as exc:
            print(f"foam-refiner merge error: {type(exc).__name__}: {exc}")
            return 1
        print(json.dumps(result.model_dump(mode="json"), indent=2))
        return 0 if result.status in {"VALIDATED", "ADOPTED"} else 1

    request = EvolveRequest(
        repo_root=args.repo_root.resolve(),
        base_ref=args.base_ref,
        regression_suite=args.regression_suite.resolve(),
        record_root=args.record_root.resolve(),
        record_id=args.record_id,
        case_ids=args.case_ids,
        benchmark_root=args.benchmark_root.resolve() if args.benchmark_root else None,
        max_cases_per_batch=args.max_cases_per_batch,
        parallel_workers=getattr(args, "parallel_workers", 1),
        batch_limits=_limits(args),
        evo_record_root=args.evo_record_root.resolve(),
    )
    try:
        selector = RecordSelector(request.repo_root)
        base, record_id, cases = selector.select(request)
        _, source_cases = suite_cases(request.regression_suite)
        evo_root = EvoRecordStore(request.evo_record_root).open(
            record_id,
            source_record=request.record_root / record_id,
            base_ref=request.base_ref,
        )
        provider = configured_provider()
        factory = BatchRuntimeFactory(
            repo_root=request.repo_root,
            provider=provider,
            source_cases=source_cases,
            artifact_root=evo_root / "batches",
            worktree_root=args.worktree_root.resolve() if args.worktree_root else None,
        )
        if args.command == "batch":
            batch_request = BatchRequest(
                batch_id=f"batch-{uuid.uuid4().hex[:12]}",
                repo_root=request.repo_root,
                base_commit=base,
                regression_suite=request.regression_suite,
                case_ids=[case.case_id for case in cases],
                limits=request.batch_limits,
                record_id=record_id,
                evo_record_root=evo_root / "batches",
            )
            result = factory.run(batch_request)
            EvoRecordStore(request.evo_record_root).update(
                record_id,
                status="completed" if result.status.value in {"candidate_ready", "no_evolution"} else "failed",
                batch_count=1,
                candidate_count=int(result.status.value == "candidate_ready"),
            )
            print(json.dumps(result.model_dump(mode="json"), indent=2))
            return 0 if result.status.value in {"candidate_ready", "no_evolution"} else 1

        graph = build_evolve_graph(
            record_selector=selector,
            clusterer=GlobalClusterer(provider_runner(provider), artifact_dir=evo_root / "clustering"),
            batch_runner=factory,
        )
        result_state = graph.invoke({"request": request}, config={"max_concurrency": request.parallel_workers})
        results = result_state.get("batch_results", [])
        errors = result_state.get("errors", [])
        EvoRecordStore(request.evo_record_root).update(
            record_id,
            status="completed" if not errors else "failed",
            batch_count=len(results),
            candidate_count=sum(item.status.value == "candidate_ready" for item in results),
        )
        print(json.dumps({"base_commit": result_state.get("base_commit"), "batches": [item.model_dump(mode="json") for item in results], "errors": errors}, indent=2))
        return 0 if not errors else 1
    except Exception as exc:
        print(f"foam-refiner error: {type(exc).__name__}: {exc}")
        return 2


def provider_runner(provider):
    """Create a role runner without sharing mutable state with batch runtimes."""
    from .roles.markdown_runner import RoleRunner

    return RoleRunner(provider)


if __name__ == "__main__":
    raise SystemExit(main())
