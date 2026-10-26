"""Construct Foam-Refiner's Claude Markdown provider."""

from __future__ import annotations

from .claude_cli import ClaudeCliRunner


def configured_provider() -> ClaudeCliRunner:
    return ClaudeCliRunner()
