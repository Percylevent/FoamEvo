"""Run one free-form Markdown role and materialize its artifact."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..providers.base import MarkdownProvider, ProviderResult


@dataclass(frozen=True)
class RoleDocument:
    role: str
    path: Path
    markdown: str
    provider_result: ProviderResult


class RoleRunner:
    """Per-runtime role runner; it stores no session or cross-batch state."""

    def __init__(self, provider: MarkdownProvider) -> None:
        self.provider = provider

    def run(
        self,
        *,
        role: str,
        prompt: str,
        artifact_dir: str | Path,
        output_name: str,
        cwd: str | Path | None = None,
    ) -> RoleDocument:
        root = Path(artifact_dir).resolve()
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        output = root / output_name
        result = self.provider.run(
            prompt=prompt,
            cwd=Path(cwd).resolve() if cwd is not None else None,
            artifact_dir=root,
        )
        # A role is asked to use Write, but the final assistant message is a
        # deterministic fallback for models that merely print their document.
        if not output.is_file() or not output.read_text(encoding="utf-8", errors="replace").strip():
            output.write_text(result.markdown.rstrip() + "\n", encoding="utf-8")
        markdown = output.read_text(encoding="utf-8", errors="replace")
        (root / "role.json").write_text(
            json.dumps(
                {"role": role, "artifact": str(output), "return_code": result.return_code},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return RoleDocument(role=role, path=output, markdown=markdown, provider_result=result)
