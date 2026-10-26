"""Global failed-case clustering before LangGraph fan-out."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..models import CaseRef, EvolveRequest
from ..roles.markdown_protocol import parse_cluster_cases
from ..roles.markdown_runner import RoleRunner
from .partition import validate_clusters


class ClusteringError(RuntimeError):
    pass


@dataclass(frozen=True)
class Cluster:
    cluster_id: str
    case_ids: list[str]
    reason: str


class GlobalClusterer:
    """Use one Markdown role to partition cases using evaluator summaries."""

    def __init__(self, runner: RoleRunner, *, artifact_dir: str | Path) -> None:
        self.runner = runner
        self.artifact_dir = Path(artifact_dir).resolve()

    def cluster(self, cases: list[CaseRef], request: EvolveRequest) -> list[Cluster]:
        if not cases:
            raise ClusteringError("cannot cluster an empty case set")
        summaries = [_case_summary(case) for case in cases]
        output = self.artifact_dir / "clustering.md"
        prompt = f"""# Foam-Refiner Global Clusterer

Partition the selected failed Foam-Runner cases into disjoint groups that can
share one repair investigation. This is only coarse classification; do not
diagnose a specific code fix. Every case must occur exactly once and each group
must contain at most {request.max_cases_per_batch} cases.

Use `evaluation.json` summaries below as the deterministic input. Write a
short rationale in each cluster section. The only machine-read control format
is:

## Cluster cluster-001
Cases: case-a, case-b
Rationale: one short coarse classification reason

Repeat that exact shape for every group, replacing the id and case names. Do
not use `### Cluster`, put `Cases:` only inside prose, omit a case from the
machine-readable line, or list the same case in two groups. For example, these
are not valid control sections:

### Cluster cluster-001
- Cases: case-a, case-b

## Cluster cluster-001
These cases are case-a and case-b.

Repeat one `## Cluster <id>` section per group. Do not put a case in prose only.
Write the complete document to `{output}`.

Case summaries:
{json.dumps(summaries, ensure_ascii=True, indent=2)}
"""
        try:
            document = self.runner.run(
                role="clusterer",
                prompt=prompt,
                artifact_dir=self.artifact_dir,
                output_name="clustering.md",
            )
        except Exception as exc:
            raise ClusteringError(f"clusterer invocation failed: {type(exc).__name__}: {exc}") from exc
        parsed = parse_cluster_cases(document.markdown)
        if not parsed:
            raise ClusteringError("clustering.md contains no ## Cluster sections")
        clusters = [Cluster(cluster_id=cid, case_ids=ids, reason=_cluster_reason(document.markdown, cid)) for cid, ids in parsed]
        try:
            validate_clusters(
                expected_case_ids=[case.case_id for case in cases],
                clusters=clusters,
                max_cases_per_batch=request.max_cases_per_batch,
            )
        except Exception as exc:
            raise ClusteringError(str(exc)) from exc
        return clusters


def _case_summary(case: CaseRef) -> dict:
    evaluation = _read_json(case.evaluation_path or case.case_path / "evaluation.json")
    checks = evaluation.get("checks") if isinstance(evaluation.get("checks"), list) else []
    failed = [
        str(item.get("check_id"))
        for item in checks
        if isinstance(item, dict) and item.get("passed") is False and item.get("check_id")
    ]
    return {
        "case_id": case.case_id,
        "passed": evaluation.get("passed"),
        "failure_summary": _short(evaluation.get("failure_summary")),
        "failed_checks": failed[:40],
        "check_count": len(checks),
    }


def _cluster_reason(markdown: str, cluster_id: str) -> str:
    section = markdown.split(f"## Cluster {cluster_id}", 1)
    if len(section) == 2:
        body = section[1].split("\n## ", 1)[0]
        lines = [line.strip() for line in body.splitlines() if line.strip() and not line.lower().startswith("cases:")]
        return " ".join(lines)[:2000] or "shared failure classification"
    return "shared failure classification"


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _short(value: object, limit: int = 2000) -> object:
    if isinstance(value, str):
        return value[:limit]
    if isinstance(value, list):
        return value[:40]
    return value
