"""Persistent Foam-Refiner evaluation matrix.

The evaluator still compares a baseline with the current candidate exactly as
before.  In addition, every replay is retained on a workflow-version axis so
the persisted keypoint matrix is genuinely three-dimensional:

``case -> check identifier -> workflow version -> check payload``.

Keeping the historical axis here (rather than in an ad-hoc report) makes a
batch replay auditable and lets callers calculate first-success/iteration
statistics without losing earlier candidates.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping


MATRIX_SCHEMA_VERSION = 3
BASELINE_VERSION = "baseline"


@dataclass(frozen=True)
class CaseOutcome:
    case_id: str
    passed: bool
    reason: str
    baseline_failed_checks: tuple[str, ...] = ()
    candidate_checks: tuple[dict, ...] = ()
    baseline_checks: tuple[dict, ...] = ()
    baseline_passed: bool | None = None
    infrastructure_blocked: bool = False
    comparison_error: str | None = None
    workflow_version: str | None = None
    correction_rounds: tuple[dict, ...] = ()
    first_success_round: int | None = None
    attempted_rounds: int = 0

    @property
    def comparable(self) -> bool:
        if self.comparison_error:
            return False
        baseline_ids = _check_ids(self.baseline_checks)
        candidate_ids = _check_ids(self.candidate_checks)
        return validate_comparable_checks(self.baseline_checks, self.candidate_checks) is None


@dataclass(frozen=True)
class EvaluationMatrix:
    """Current comparison plus all workflow versions seen by the batch.

    ``outcomes`` intentionally remains the current baseline/candidate view so
    existing improvement gates do not change.  ``keypoint_matrix`` is the
    durable historical representation and is populated automatically for old
    callers that still construct ``EvaluationMatrix(outcomes)`` directly.
    """

    outcomes: tuple[CaseOutcome, ...]
    workflow_versions: tuple[str, ...] = ()
    keypoint_matrix: dict[str, dict[str, dict[str, dict[str, Any]]]] = field(default_factory=dict)
    current_workflow_version: str | None = None
    baseline_workflow_version: str = BASELINE_VERSION

    def __post_init__(self) -> None:
        versions = list(self.workflow_versions)
        matrix = deepcopy(self.keypoint_matrix)
        current = self.current_workflow_version

        # Preserve backwards compatibility with the original two-column API.
        if not versions and self.outcomes:
            versions = [BASELINE_VERSION, "candidate"]
            current = "candidate"
        elif not versions and current:
            versions = [current]
        if current is None and versions:
            current = versions[-1]
        if current and current not in versions:
            versions.append(current)
        if BASELINE_VERSION not in versions and self.outcomes:
            versions.insert(0, BASELINE_VERSION)

        for item in self.outcomes:
            _store_checks(matrix, item.case_id, BASELINE_VERSION, item.baseline_checks)
            candidate_version = item.workflow_version or current
            if candidate_version:
                _store_checks(matrix, item.case_id, candidate_version, item.candidate_checks)
                if candidate_version not in versions:
                    versions.append(candidate_version)

        object.__setattr__(self, "workflow_versions", tuple(dict.fromkeys(versions)))
        object.__setattr__(self, "keypoint_matrix", matrix)
        object.__setattr__(self, "current_workflow_version", current)

    @classmethod
    def baseline_only(cls, outcomes: tuple[CaseOutcome, ...]) -> "EvaluationMatrix":
        """Build a matrix containing only the immutable source workflow."""

        return cls(
            outcomes=outcomes,
            workflow_versions=(BASELINE_VERSION,),
            current_workflow_version=BASELINE_VERSION,
            baseline_workflow_version=BASELINE_VERSION,
        )

    def append_version(
        self,
        workflow_version: str,
        outcomes: tuple[CaseOutcome, ...],
    ) -> "EvaluationMatrix":
        """Return a copy with one replay version appended to the 3-D matrix."""

        version = str(workflow_version).strip()
        if not version or version == BASELINE_VERSION:
            raise ValueError("workflow_version must be a non-empty non-baseline value")
        history = deepcopy(self.keypoint_matrix)
        for item in outcomes:
            _store_checks(history, item.case_id, version, item.candidate_checks)
        return EvaluationMatrix(
            outcomes=tuple(_with_workflow_version(item, version) for item in outcomes),
            workflow_versions=tuple(dict.fromkeys((*self.workflow_versions, version))),
            keypoint_matrix=history,
            current_workflow_version=version,
            baseline_workflow_version=self.baseline_workflow_version,
        )

    @property
    def passed(self) -> bool:
        return bool(self.outcomes) and all(item.passed for item in self.outcomes)

    @property
    def comparable(self) -> bool:
        return (bool(self.outcomes) and len({x.case_id for x in self.outcomes}) == len(self.outcomes)
                and all(item.comparable for item in self.outcomes))

    @property
    def comparison_errors(self) -> tuple[str, ...]:
        return tuple(
            f"{item.case_id}: {item.comparison_error or 'evaluation checks are not comparable'}"
            for item in self.outcomes if not item.comparable
        )

    @property
    def success_rate(self) -> float:
        return (sum(item.passed for item in self.outcomes) / len(self.outcomes)) if self.outcomes else 0.0

    @property
    def baseline_success_rate(self) -> float:
        if not self.outcomes:
            return 0.0
        return sum(_baseline_passed(item) for item in self.outcomes) / len(self.outcomes)

    @property
    def keypoint_score(self) -> int:
        if not self.comparable:
            return 0
        return sum(sum(bool(check.get("passed")) for check in item.candidate_checks) for item in self.outcomes)

    @property
    def baseline_keypoint_score(self) -> int:
        if not self.comparable:
            return 0
        return sum(sum(bool(check.get("passed")) for check in item.baseline_checks) for item in self.outcomes)

    @property
    def repaired_checks(self) -> int:
        """Count failing baseline checks that pass under the same contract."""

        if not self.comparable:
            return 0
        repaired = 0
        for item in self.outcomes:
            candidate = {str(check["check_id"]): bool(check.get("passed")) for check in item.candidate_checks}
            repaired += sum(
                not bool(check.get("passed")) and candidate[str(check["check_id"])]
                for check in item.baseline_checks
            )
        return repaired

    @property
    def improved(self) -> bool:
        return self.comparable and (
            not self.regressions
            and (
                self.success_rate > self.baseline_success_rate
                or self.keypoint_score > self.baseline_keypoint_score
                or self.repaired_checks > 0
            )
        )

    @property
    def regressions(self) -> tuple[str, ...]:
        """Cases/checks that passed in baseline but fail in the current replay."""

        if not self.comparable:
            return ()
        rows: list[str] = []
        for item in self.outcomes:
            if _baseline_passed(item) and not item.passed:
                rows.append(f"{item.case_id}:overall_success")
            candidate = {str(c.get("check_id")): bool(c.get("passed")) for c in item.candidate_checks}
            for check in item.baseline_checks:
                check_id = str(check.get("check_id"))
                if bool(check.get("passed")) and not candidate.get(check_id, False):
                    rows.append(f"{item.case_id}:{check_id}")
        return tuple(rows)

    @property
    def infrastructure_blocked(self) -> bool:
        return any(
            item.infrastructure_blocked or "infrastructure" in item.reason.lower()
            for item in self.outcomes
        )

    @property
    def gate_passed(self) -> bool:
        """Require a comparable, non-infrastructure, material improvement."""

        return bool(self.outcomes) and self.comparable and not self.infrastructure_blocked and self.improved

    def to_dict(self) -> dict:
        return {
            "schema_version": MATRIX_SCHEMA_VERSION,
            "passed": self.passed,
            "total": len(self.outcomes),
            "passed_count": sum(item.passed for item in self.outcomes),
            "comparable": self.comparable,
            "comparison_errors": list(self.comparison_errors),
            "success_rate": self.success_rate,
            "baseline_success_rate": self.baseline_success_rate,
            "keypoint_score": self.keypoint_score,
            "baseline_keypoint_score": self.baseline_keypoint_score,
            "repaired_checks": self.repaired_checks,
            "regressions": list(self.regressions),
            "improved": self.improved,
            "gate_passed": self.gate_passed,
            "workflow_versions": list(self.workflow_versions),
            "baseline_workflow_version": self.baseline_workflow_version,
            "current_workflow_version": self.current_workflow_version,
            "keypoint_matrix": self.keypoint_matrix,
            "outcomes": [_outcome_dict(item) for item in self.outcomes],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvaluationMatrix":
        """Load a persisted matrix, accepting both schema 2 and schema 3."""

        raw_outcomes = value.get("outcomes", [])
        outcomes: list[CaseOutcome] = []
        for raw in raw_outcomes if isinstance(raw_outcomes, list) else []:
            if not isinstance(raw, Mapping):
                continue
            outcomes.append(CaseOutcome(
                case_id=str(raw.get("case_id", "")),
                passed=bool(raw.get("passed", False)),
                reason=str(raw.get("reason", "")),
                baseline_failed_checks=tuple(str(x) for x in raw.get("baseline_failed_checks", []) or []),
                baseline_checks=tuple(x for x in raw.get("baseline_checks", []) if isinstance(x, dict)),
                candidate_checks=tuple(x for x in raw.get("candidate_checks", []) if isinstance(x, dict)),
                baseline_passed=raw.get("baseline_passed"),
                infrastructure_blocked=bool(raw.get("infrastructure_blocked", False)),
                comparison_error=raw.get("comparison_error"),
                workflow_version=raw.get("workflow_version"),
                correction_rounds=tuple(x for x in raw.get("correction_rounds", []) if isinstance(x, dict)),
                first_success_round=raw.get("first_success_round"),
                attempted_rounds=int(raw.get("attempted_rounds", 0) or 0),
            ))
        matrix = value.get("keypoint_matrix", {})
        return cls(
            outcomes=tuple(outcomes),
            workflow_versions=tuple(str(x) for x in value.get("workflow_versions", []) or []),
            keypoint_matrix=deepcopy(matrix) if isinstance(matrix, dict) else {},
            current_workflow_version=value.get("current_workflow_version"),
            baseline_workflow_version=str(value.get("baseline_workflow_version", BASELINE_VERSION)),
        )


def validate_comparable_checks(
    baseline_checks: tuple[dict, ...], candidate_checks: tuple[dict, ...]
) -> str | None:
    """Return a diagnostic if evaluator matrices cannot be compared safely."""

    if any(type(c.get("passed")) is not bool for c in (*baseline_checks, *candidate_checks)):
        return "check results must be Boolean"
    baseline_ids = _check_ids(baseline_checks)
    candidate_ids = _check_ids(candidate_checks)
    if not baseline_ids:
        return "baseline evaluation has no valid checks"
    if not candidate_ids:
        return "candidate evaluation has no valid checks"
    if len(baseline_ids) != len(baseline_checks):
        return "baseline evaluation has duplicate or invalid check IDs"
    if len(candidate_ids) != len(candidate_checks):
        return "candidate evaluation has duplicate or invalid check IDs"
    if baseline_ids != candidate_ids:
        missing = sorted(baseline_ids - candidate_ids)
        extra = sorted(candidate_ids - baseline_ids)
        detail = []
        if missing:
            detail.append("missing=" + ",".join(missing))
        if extra:
            detail.append("extra=" + ",".join(extra))
        return "evaluation check IDs differ (" + "; ".join(detail) + ")"
    return None


def write_matrix(path: str | Path, matrix: EvaluationMatrix) -> None:
    target = Path(path)
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    target.write_text(json.dumps(matrix.to_dict(), ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _check_ids(checks: tuple[dict, ...]) -> set[str]:
    return {
        str(item.get("check_id")) for item in checks
        if isinstance(item, dict) and isinstance(item.get("check_id"), str) and item["check_id"]
    }


def _store_checks(
    matrix: dict[str, dict[str, dict[str, dict[str, Any]]]],
    case_id: str,
    version: str,
    checks: tuple[dict, ...],
) -> None:
    case_row = matrix.setdefault(str(case_id), {})
    for check in checks:
        if not isinstance(check, Mapping) or not check.get("check_id"):
            continue
        check_row = case_row.setdefault(str(check["check_id"]), {})
        check_row[str(version)] = deepcopy(dict(check))


def _with_workflow_version(item: CaseOutcome, version: str) -> CaseOutcome:
    return CaseOutcome(
        case_id=item.case_id,
        passed=item.passed,
        reason=item.reason,
        baseline_failed_checks=item.baseline_failed_checks,
        candidate_checks=item.candidate_checks,
        baseline_checks=item.baseline_checks,
        baseline_passed=item.baseline_passed,
        infrastructure_blocked=item.infrastructure_blocked,
        comparison_error=item.comparison_error,
        workflow_version=version,
        correction_rounds=item.correction_rounds,
        first_success_round=item.first_success_round,
        attempted_rounds=item.attempted_rounds,
    )


def _outcome_dict(item: CaseOutcome) -> dict:
    return {
        "case_id": item.case_id,
        "passed": item.passed,
        "reason": item.reason,
        "baseline_failed_checks": list(item.baseline_failed_checks),
        "baseline_checks": list(item.baseline_checks),
        "baseline_passed": item.baseline_passed,
        "candidate_checks": list(item.candidate_checks),
        "infrastructure_blocked": item.infrastructure_blocked,
        "comparison_error": item.comparison_error,
        "comparable": item.comparable,
        "workflow_version": item.workflow_version,
        "correction_rounds": list(item.correction_rounds),
        "first_success_round": item.first_success_round,
        "attempted_rounds": item.attempted_rounds,
    }


def _baseline_passed(item: CaseOutcome) -> bool:
    if item.baseline_passed is not None:
        return item.baseline_passed
    return not item.baseline_failed_checks
