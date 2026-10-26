"""CLI primitives for creating records and running one Foam-Runner case.

Each invocation owns one case directory. The directory is written even when
configuration, task parsing, or the workflow itself fails so later evolution
never has to guess whether an absent artifact means a passing case.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import traceback

_SRC_ROOT = Path(__file__).resolve().parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from config import Config
from evaluation import build_failure_packet, evaluate_case
from main import main as run_workflow
from record import case_directory, create_record, utc_now


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="foam-runner")
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("record-create")
    create.add_argument("--root", default=None)

    run = sub.add_parser("run-case")
    run.add_argument("--task", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, default=None)
    run.add_argument("--record-root", type=Path, default=None)
    run.add_argument("--record-id", default=None)
    run.add_argument("--case-id", default=None)
    run.add_argument("--custom-mesh-path", default=None)
    run.add_argument("--reuse-generated-dir", default=None)
    run.add_argument("--max-review-loop", type=int, default=None)

    args = parser.parse_args(argv)
    if args.command == "record-create":
        root = args.root or Config().run_directory
        print(create_record(root).name)
        return 0
    return _run_case(args)


def _run_case(args: argparse.Namespace) -> int:
    """Run a case and persist the same versioned evaluator contract every time."""

    record_path, case_path = _resolve_case_path(args)
    case_path.mkdir(mode=0o700, parents=True, exist_ok=True)
    (case_path / "inputs").mkdir(mode=0o700, exist_ok=True)

    case_id = args.case_id or case_path.name
    task_path = args.task.expanduser().resolve()
    payload: dict | None = None
    stdout = io.StringIO()
    stderr = io.StringIO()
    status = "completed"
    error: str | None = None

    # Write this first: every later error updates a known case artifact.
    _write_json(case_path / "manifest.json", {
        "schema_version": 2,
        "record_id": record_path.name if record_path else None,
        "case_id": case_id,
        "status": "running",
        "passed": False,
        "task_path": str(task_path),
        "started_at": utc_now(),
    })

    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            parsed = json.loads(task_path.read_text(encoding="utf-8"))
            if not isinstance(parsed, dict):
                raise ValueError("task JSON must be an object")
            payload = parsed
            _write_json(case_path / "task.json", payload)

            prompt_location = payload.get("prompt_path")
            if not isinstance(prompt_location, str) or not prompt_location:
                raise ValueError("task has no prompt_path")
            prompt_file = Path(prompt_location).expanduser()
            if not prompt_file.is_absolute():
                prompt_file = task_path.parent / prompt_file
            if not prompt_file.is_file():
                raise FileNotFoundError(f"task has no readable prompt_path: {prompt_file}")
            prompt = prompt_file.read_text(encoding="utf-8")
            (case_path / "inputs" / "prompt.txt").write_text(prompt, encoding="utf-8")
            # Archived tasks must remain replayable after leaving the input tree.
            payload["prompt_path"] = str((case_path / "inputs" / "prompt.txt").resolve())
            _write_json(case_path / "task.json", payload)

            config = Config()
            config.case_dir = str(case_path)
            context = payload.get("cfd_context", {})
            if not isinstance(context, dict):
                raise ValueError("cfd_context must be a JSON object")
            config.experience_context = context
            if args.reuse_generated_dir:
                config.reuse_generated_dir = args.reuse_generated_dir
            if args.max_review_loop is not None:
                if args.max_review_loop < 1:
                    raise ValueError("--max-review-loop must be at least 1")
                config.max_loop = args.max_review_loop
            run_workflow(prompt, config, args.custom_mesh_path)
        except BaseException as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()

    stdout_text = stdout.getvalue()
    stderr_text = stderr.getvalue()
    (case_path / "stdout.log").write_text(stdout_text, encoding="utf-8")
    (case_path / "stderr.log").write_text(stderr_text, encoding="utf-8")
    correction_summary = _write_correction_summary(
        case_path,
        case_id=case_id,
        status=status,
        error=error,
        stdout=stdout_text,
        stderr=stderr_text,
    )

    evaluation = evaluate_case(
        case_dir=case_path,
        return_code=0 if status == "completed" else 1,
        stdout=stdout_text,
        stderr=stderr_text,
        task=payload,
    )
    evaluation["correction_history"] = correction_summary
    passed = bool(evaluation["passed"])
    _write_json(case_path / "result.json", {
        "status": status,
        "passed": passed,
        "case_id": case_id,
        "error": error,
        "correction_history": correction_summary,
    })
    _write_json(case_path / "evaluation.json", evaluation)
    if not passed:
        _write_json(
            case_path / "failure_packet.json",
            build_failure_packet(
                case_dir=case_path,
                evaluation=evaluation,
                stdout=stdout_text,
                stderr=stderr_text,
            ),
        )
    _write_json(case_path / "manifest.json", {
        "schema_version": 2,
        "record_id": record_path.name if record_path else None,
        "case_id": case_id,
        "status": status,
        "passed": passed,
        "task_path": str(task_path),
        "completed_at": utc_now(),
        "correction_history": correction_summary,
    })
    print(json.dumps({"case_dir": str(case_path), "passed": passed}, ensure_ascii=True))
    return 0 if passed else 1


def _resolve_case_path(args: argparse.Namespace) -> tuple[Path | None, Path]:
    if args.output_dir:
        return None, args.output_dir.expanduser().resolve()
    if not args.record_id or not args.case_id:
        raise SystemExit("run-case requires --output-dir or --record-id and --case-id")
    root = Path(args.record_root or Config().run_directory).expanduser().resolve()
    record_path = root / args.record_id
    return record_path, case_directory(record_path, args.case_id)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _write_correction_summary(
    case_path: Path,
    *,
    case_id: str,
    status: str,
    error: str | None,
    stdout: str,
    stderr: str,
) -> dict:
    """Normalize append-only runner events into one case-level artifact."""

    events: list[dict] = []
    event_path = case_path / "corrections" / "rounds.jsonl"
    if event_path.is_file():
        for line in event_path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if isinstance(value, dict) and isinstance(value.get("round"), int):
                events.append(value)
    # A planner/provider failure can happen before the first runner node. Keep
    # an explicit round rather than making the absence of a file look like an
    # unattempted case.
    if not events:
        events = [{
            "schema_version": 1,
            "round": 0,
            "attempt": 1,
            "runner": "workflow",
            "recorded_at": utc_now(),
            "passed": False,
            "error_count": 1 if (error or status != "completed") else 0,
            "errors": [error] if error else [],
            "allrun_exists": (case_path / "Allrun").is_file(),
            "solver_logs": [],
        }]
    # Keep the last event for a round if a retrying runner emitted duplicate
    # records; the per-round JSON files remain available for forensic review.
    by_round = {int(item["round"]): item for item in events}
    rounds = [by_round[key] for key in sorted(by_round)]
    first_success = next((int(item["round"]) for item in rounds if item.get("passed")), None)
    summary = {
        "schema_version": 1,
        "case_id": case_id,
        "rounds": rounds,
        "attempted_rounds": len(rounds),
        "correction_count": max(0, len(rounds) - 1),
        "first_success_round": first_success,
        "first_success": first_success is not None,
        "final_round": rounds[-1]["round"],
        "final_passed": bool(rounds[-1].get("passed")),
        "workflow_status": status,
        "workflow_error": error,
    }
    _write_json(case_path / "corrections" / "correction_rounds.json", summary)
    _write_json(case_path / "correction_rounds.json", summary)
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
