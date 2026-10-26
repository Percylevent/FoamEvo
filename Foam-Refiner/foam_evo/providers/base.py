"""Provider primitives for Markdown based Foam-Refiner roles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class ProviderResult:
    markdown: str
    return_code: int = 0
    stdout: str = ""
    stderr: str = ""


class MarkdownProvider(Protocol):
    def run(
        self,
        *,
        prompt: str,
        cwd: Path | None = None,
        artifact_dir: Path,
        timeout_s: float | None = None,
    ) -> ProviderResult:
        ...
