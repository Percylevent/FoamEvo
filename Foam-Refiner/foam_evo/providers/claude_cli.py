"""Thin Claude Code CLI adapter.

The adapter intentionally knows nothing about Foam-Refiner role schemas.  Claude
uses its native Read/Grep/Bash/Write tools and the caller consumes the Markdown
artifact it leaves behind.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

from .base import ProviderResult

DEFAULT_ROLE_TIMEOUT_S = 1800.0


class ClaudeCliError(RuntimeError):
    """Raised when a Claude process cannot produce a role artifact."""


class ClaudeCliRunner:
    def __init__(
        self,
        *,
        executable: str = "claude",
        model: str | None = None,
        timeout_s: float = DEFAULT_ROLE_TIMEOUT_S,
    ) -> None:
        self.executable = executable
        self.model = model or os.getenv("FOAM_EVO_MODEL_VERSION") or os.getenv("ANTHROPIC_MODEL")
        self.timeout_s = timeout_s

    def run(
        self,
        *,
        prompt: str,
        cwd: Path | None = None,
        artifact_dir: Path,
        timeout_s: float | None = None,
    ) -> ProviderResult:
        artifact_dir = Path(artifact_dir).resolve()
        artifact_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        started = time.monotonic()
        command = _command(self.executable, self.model)
        try:
            process = subprocess.Popen(
                command,
                cwd=str(cwd) if cwd is not None else None,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                start_new_session=True,
            )
            stdout, stderr = process.communicate(input=prompt, timeout=timeout_s or self.timeout_s)
        except FileNotFoundError as exc:
            _write_attempt(artifact_dir, prompt, "", str(exc), None, started, False)
            raise ClaudeCliError("Claude CLI executable was not found") from exc
        except subprocess.TimeoutExpired as exc:
            _terminate_group(process)
            stdout, stderr = process.communicate()
            _write_attempt(artifact_dir, prompt, stdout, stderr, process.returncode, started, True)
            raise ClaudeCliError(f"Claude CLI timed out after {timeout_s or self.timeout_s}s") from exc
        _write_attempt(artifact_dir, prompt, stdout, stderr, process.returncode, started, False)
        markdown = _extract_cli_text(stdout)
        if process.returncode != 0:
            raise ClaudeCliError(stderr.strip() or markdown or f"Claude CLI exited with {process.returncode}")
        return ProviderResult(markdown=markdown, return_code=process.returncode, stdout=stdout, stderr=stderr)


def _command(executable: str, model: str | None) -> list[str]:
    # No settings directory or API variables are injected here.  The caller's
    # environment is intentionally inherited, so ANTHROPIC_* is authoritative.
    command = [
        executable,
        "--print",
        "--output-format",
        "stream-json",
        "--verbose",
        "--dangerously-skip-permissions",
        "--disable-slash-commands",
    ]
    if model:
        command.extend(["--model", model])
    return command


def _terminate_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                pass


def _extract_cli_text(raw: str) -> str:
    """Extract the final assistant Markdown from Claude's stream-json output.

    Claude emits several assistant events while it is using tools.  Foam-Refiner keeps
    the last assistant text block, rather than concatenating every intermediate
    thought; concatenation can duplicate prose and accidentally introduce a
    control token from an earlier event.  The ``result`` event is retained only
    as a fallback for providers that omit assistant events entirely.
    """

    last_assistant = ""
    last_result = ""
    for line in raw.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        # Some wrappers put the stream event under one or two ``event`` keys.
        for _ in range(2):
            nested = event.get("event")
            if not isinstance(nested, dict):
                break
            event = nested
        message = event.get("message") or event
        if event.get("type") == "assistant" or (
            isinstance(message, dict) and message.get("role") == "assistant"
        ):
            content = message.get("content", []) if isinstance(message, dict) else []
            if not isinstance(content, list):
                continue
            text = "\n\n".join(
                str(block.get("text", ""))
                for block in content
                if isinstance(block, dict) and block.get("type") == "text" and block.get("text")
            ).strip()
            if text:
                last_assistant = text
        elif event.get("type") == "result" and isinstance(event.get("result"), str):
            result = event["result"].strip()
            if result:
                last_result = result
    return last_assistant or last_result or raw.strip()


def _write_attempt(
    root: Path,
    prompt: str,
    stdout: str,
    stderr: str,
    returncode: int | None,
    started: float,
    timed_out: bool,
) -> None:
    (root / "prompt.md").write_text(prompt, encoding="utf-8")
    (root / "stdout.log").write_text(stdout, encoding="utf-8")
    (root / "stderr.log").write_text(stderr, encoding="utf-8")
    (root / "invocation.json").write_text(
        json.dumps(
            {
                "returncode": returncode,
                "duration_s": time.monotonic() - started,
                "timed_out": timed_out,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
