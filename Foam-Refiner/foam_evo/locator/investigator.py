"""Progressive, Markdown based Locator role."""

from __future__ import annotations

import json
from pathlib import Path

from ..roles.markdown_runner import RoleDocument, RoleRunner


class LocatorInvestigationError(RuntimeError):
    pass


class LocatorInvestigator:
    def __init__(self, runner: RoleRunner) -> None:
        self.runner = runner

    def run(
        self,
        *,
        context_root: str | Path,
        artifact_dir: str | Path,
        batch_id: str = "default",
        candidate_worktree: str | Path | None = None,
        failure_context: str = "",
    ) -> RoleDocument:
        root = Path(context_root).resolve()
        index_path = root / "context" / "case-index.json"
        try:
            cases = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise LocatorInvestigationError(f"invalid Locator context: {index_path}") from exc
        if not isinstance(cases, list) or not cases:
            raise LocatorInvestigationError("Locator context contains no cases")
        summaries = []
        for item in cases:
            if not isinstance(item, dict) or not item.get("case_id") or not isinstance(item.get("summary"), dict):
                raise LocatorInvestigationError("case-index contains an invalid case summary")
            summaries.append({"case_id": item["case_id"], "summary": item["summary"], "case_dir": item.get("case_dir")})
        output = Path(artifact_dir).resolve() / "locator.md"
        prompt = _prompt(root, candidate_worktree, summaries, failure_context, output, batch_id)
        return self.runner.run(
            role="locator",
            prompt=prompt,
            artifact_dir=Path(artifact_dir),
            output_name="locator.md",
            cwd=candidate_worktree,
        )


def _prompt(
    root: Path,
    source: str | Path | None,
    summaries: list[dict],
    failure_context: str,
    output: Path,
    batch_id: str,
) -> str:
    return f"""# Foam-Refiner Locator — batch {batch_id}

Diagnose one repairable root cause shared by the selected failed Foam-Runner
cases. This is an investigation, not a code change and not a new clustering
decision.

Start with the compact index and per-case summaries:
- `{root / 'context' / 'case-index.json'}`
- `{root / 'context' / 'cases'}`

The index is intentionally small. Use Claude's native `Read`, `Grep`, `find`
and `Bash` tools to progressively inspect only the relevant case inputs and
diagnostic files. The complete `inputs/` tree and copied `evaluation.json`,
`stdout.log`, `workflow.log`, `review.log`, `Allrun.err`, and `log.*` files are
under each case directory. Search for failing check names and concrete error
signatures before reading large logs. Inspect the candidate source tree at
`{Path(source).resolve() if source else '(unavailable)'}` when source evidence
is needed.

Selected summaries (not a replacement for reading evidence):
{json.dumps(summaries, ensure_ascii=True, indent=2)}

Previous iteration/replay feedback:
{failure_context or '(none)'}

Write the complete investigation to `{output}` using ordinary Markdown. A
useful shape is:

## Root Cause
One concise, evidence-backed statement.

## Shared Evidence
- `case-id/path/to/artifact`: exact observation

## Planner Implications
What the next role should investigate or change.

This is an investigation report, not a routing decision. Do not add `## Exit`
or `## Verdict`, do not invent evidence that is not present in a case artifact,
and do not edit the candidate tree.
"""
