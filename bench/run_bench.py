#!/usr/bin/env python3
"""Run FoamBench cases through Foam-Runner's single-case CLI with N workers.

Run as a non-root user: OpenFOAM's runtime code compilation (`dynamicCode`)
rejects root execution. In a container with an `openfoam` account, use:

    docker exec -u openfoam <container> bash -l

The record root must be writable by the benchmark user so artifacts can be created with restricted permissions.


Each case is one `python -m src.record_cli run-case` subprocess. The harness
creates one record token and each case lands in
`<record>/<token>/cases/<case-id>/` with its own inputs, logs and evaluation.

Parallelism is safe because each worker receives a separate case directory;
the coordinator is the only writer of aggregate JSONL/manifest files. OpenFOAM
cases here are all single-core.

Each invocation creates one five-character record token. Replays and later
evolution select that record explicitly; restarting a benchmark creates a new
record rather than mutating an older one.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

_WRITE_LOCK = threading.Lock()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"[{now()}] {message}", flush=True)


def read_evaluation(case_dir: Path) -> dict:
    """Read the case verdict produced by Foam-Runner."""
    path = case_dir / "evaluation.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def token_usage(case_dir: Path) -> dict:
    """Sum recorded LLM token usage for one run, if telemetry is present."""
    events = case_dir / "events.jsonl"
    calls = prompt_tokens = completion_tokens = 0
    if not events.is_file():
        return {}
    try:
        with events.open(encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("event_type") != "llm_call":
                    continue
                calls += 1
                prompt_tokens += int(event.get("prompt_tokens") or 0)
                completion_tokens += int(event.get("completion_tokens") or 0)
    except OSError:
        return {}
    return {
        "llm_calls": calls,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


def run_case(case: dict, args, log_dir: Path, record_dir: Path) -> dict:
    """Run one case to completion and return its summary row."""
    slug = case["slug"]
    started = time.monotonic()
    started_at = now()
    stdout_path = log_dir / f"{slug}.log"

    case_dir = record_dir / "cases" / slug
    task_path = case.get("_resolved_task_path", case["task_path"])
    command = [
        sys.executable, "-m", "src.record_cli", "run-case",
        "--task", task_path, "--output-dir", str(case_dir),
    ]
    # Keep the whole stream per case so a failure can be diagnosed without
    # re-running it.
    try:
        with stdout_path.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                command,
                cwd=args.repo_root,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=args.case_timeout_s,
                check=False,
            )
        exit_code = completed.returncode
        timed_out = False
    except subprocess.TimeoutExpired:
        exit_code = None
        timed_out = True

    evaluation = read_evaluation(case_dir)
    row = {
        "slug": slug,
        "case_id": case["case_id"],
        "suite": case.get("suite", "basic"),
        "family": case["family"],
        "record_id": record_dir.name,
        "case_dir": str(case_dir),
        "passed": bool(evaluation.get("passed")),
        "failure_summary": evaluation.get("failure_summary"),
        "cli_exit_code": exit_code,
        "harness_timed_out": timed_out,
        "duration_s": round(time.monotonic() - started, 1),
        "started_at": started_at,
        "finished_at": now(),
        "log_path": str(stdout_path),
    }
    row.update(token_usage(case_dir))
    if not evaluation:
        row["note"] = "no evaluation.json; see log_path"
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, help="path to a benchmark cases.json manifest")
    parser.add_argument("--repo-root", default="Foam-Runner")
    parser.add_argument("--record-root", default="record", help="root for one run record")
    parser.add_argument("--output", default=None, help="optional harness log directory")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument(
        "--case-timeout-s",
        type=float,
        default=14400,
        help="harness-side ceiling per case, above the task's own timeout",
    )
    parser.add_argument("--only-family", default=None)
    parser.add_argument("--only-suite", default=None, choices=["basic", "advanced"])
    parser.add_argument(
        "--only-slugs",
        default=None,
        help="JSON file holding a list of slugs to run; for targeted reruns",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    # Refuse to produce results that are silently invalid: as root, every
    # dynamicCode-dependent solver aborts and those cases look like model
    # failures. Better to stop here than to spend hours of tokens on garbage.
    if os.geteuid() == 0:
        print(
            "refusing to run as root: OpenFOAM dynamicCode aborts for root, so "
            "solvers that compile templates at runtime (reactingFoam and "
            "friends) would fail for environmental reasons and be recorded as "
            "benchmark failures.\n"
            "Re-run as: docker exec -u openfoam <container> bash -l",
            file=sys.stderr,
        )
        return 2

    repo_path = Path(args.repo_root).resolve()
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))
    from src.record import create_record
    record_dir = create_record(
        args.record_root,
        metadata={"benchmark_cases": str(Path(args.cases).resolve()), "worker_count": args.workers},
    )
    record_manifest = record_dir / "manifest.json"
    out = Path(args.output).resolve() if args.output else record_dir / "harness"
    log_dir = out / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    results_path = out / "results.jsonl"

    cases_path = Path(args.cases).resolve()
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    for case in cases:
        task_path = Path(case["task_path"])
        if not task_path.is_absolute():
            task_path = cases_path.parent / task_path
        case["_resolved_task_path"] = str(task_path.resolve())
    if args.only_family:
        cases = [c for c in cases if c["family"] == args.only_family]
    if args.only_suite:
        cases = [c for c in cases if c.get("suite", "basic") == args.only_suite]
    if args.only_slugs:
        wanted = set(json.loads(Path(args.only_slugs).read_text(encoding="utf-8")))
        cases = [c for c in cases if c["slug"] in wanted]
        missing = wanted - {c["slug"] for c in cases}
        if missing:
            print(f"warning: slugs not in cases.json: {sorted(missing)}", file=sys.stderr)
    # Every invocation owns a fresh five-character record.  Reading an old
    # results file here would silently mix records and make manifest counts
    # incorrect; resume is deliberately explicit via --record-id in the
    # higher-level coordinator, not an implicit benchmark behavior.
    pending = list(cases)
    if args.limit is not None:
        pending = pending[: args.limit]

    log(f"total={len(cases)} pending={len(pending)} workers={args.workers}")
    if not pending:
        log("nothing to do")
        return 0

    completed = 0
    passed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_case, case, args, log_dir, record_dir): case for case in pending
        }
        for future in as_completed(futures):
            case = futures[future]
            try:
                row = future.result()
            except Exception as exc:  # keep the sweep alive; record and move on
                row = {
                    "slug": case["slug"],
                    "case_id": case["case_id"],
                    "suite": case.get("suite", "basic"),
                    "family": case["family"],
                    "passed": False,
                    "harness_error": f"{type(exc).__name__}: {exc}",
                    "finished_at": now(),
                }
            with _WRITE_LOCK:
                with results_path.open("a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n"
                    )
                    stream.flush()
                    os.fsync(stream.fileno())
            completed += 1
            passed += 1 if row.get("passed") else 0
            log(
                f"[{completed}/{len(pending)}] {row['slug']} "
                f"passed={row.get('passed')} "
                f"duration={row.get('duration_s')}s "
                f"running_pass_rate={passed}/{completed}"
            )

    manifest = json.loads(record_manifest.read_text(encoding="utf-8"))
    manifest.update({
        "status": "completed",
        "case_count": len(cases),
        "completed_count": completed,
        "passed_count": passed,
        "failed_count": completed - passed,
        "finished_at": now(),
    })
    record_manifest.write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    (record_dir / "summary.json").write_text(
        json.dumps({"record_id": record_dir.name, "total": len(cases), "completed": completed, "passed": passed, "failed": completed - passed}, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    log(f"finished: {passed}/{completed} passed in this session")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
