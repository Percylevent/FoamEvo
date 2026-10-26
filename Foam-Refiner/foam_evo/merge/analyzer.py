"""Optional Markdown merge-plan analyst."""

from __future__ import annotations

import re
from pathlib import Path

from ..models import MergePlan
from ..roles.markdown_runner import RoleRunner


class MergeAnalysisError(RuntimeError):
    pass


class MergeAnalyzer:
    def __init__(self, runner: RoleRunner) -> None:
        self.runner = runner

    def analyze(self, *, base_commit: str, candidate_diffs: dict[str, str], artifact_dir: str | Path, batch_id: str = "merge") -> MergePlan:
        output = Path(artifact_dir).resolve() / "merge-plan.md"
        prompt = f"""Review all candidate branch diffs for Foam-Runner relative to `{base_commit}`.
Write a Markdown analysis to `{output}`. Use this exact machine-readable
section, listing every branch exactly once:

## Merge order
- branch-a
- branch-b

Replace the example names with the supplied branch names. Explain semantic
conflicts before or after that section, but do not use `### Merge order`, put
branches only in prose, omit a branch, or list one twice. For example, these
are invalid:

### Merge order
- branch-a

## Merge order
branch-a and branch-b should be merged

""" + "\n\n".join(f"## Branch {name}\n{diff[:100000]}" for name, diff in candidate_diffs.items())
        document = self.runner.run(role="merge-analyzer", prompt=prompt, artifact_dir=Path(artifact_dir), output_name="merge-plan.md")
        section = document.markdown.split("## Merge order", 1)
        order: list[str] = []
        if len(section) == 2:
            body = section[1].split("\n## ", 1)[0]
            for line in body.splitlines():
                match = re.match(r"\s*(?:[-*]\s*)?([^\s`]+)", line)
                if match and match.group(1) in candidate_diffs:
                    order.append(match.group(1))
        if set(order) != set(candidate_diffs) or len(order) != len(candidate_diffs):
            raise MergeAnalysisError("merge plan must include every candidate branch exactly once")
        return MergePlan(merge_order=order, rationale="Markdown merge analysis")
