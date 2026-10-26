"""Small configuration and result models for Foam-Refiner.

Role documents deliberately do not use these models. Roles communicate through
Markdown files; only the deterministic engine boundary (requests, process
results and final summaries) is typed here.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class BatchStatus(str, Enum):
    CREATED = "created"
    PREPARING = "preparing"
    LOCATING = "locating"
    PLANNING = "planning"
    IMPLEMENTING = "implementing"
    TESTING = "testing"
    REPLAYING = "replaying"
    REVIEWING = "reviewing"
    CANDIDATE_READY = "candidate_ready"
    NO_EVOLUTION = "no_evolution"
    STALLED = "stalled"
    FAILED = "failed"
    INFRASTRUCTURE_BLOCKED = "infrastructure_blocked"


class CaseRef(Model):
    """One case in an external Foam-Runner record."""

    record_id: str
    case_id: str
    record_path: Path
    case_path: Path
    task_path: Path
    prompt_path: Path | None = None
    evaluation_path: Path | None = None
    baseline_commit: str | None = None


class BatchLimits(Model):
    max_iterations: int = Field(default=3, ge=1)
    max_plan_rounds: int = Field(default=3, ge=1)
    max_code_rounds: int = Field(default=3, ge=1)
    source_test_timeout_s: float = Field(default=1800.0, gt=0)
    replay_workers: int = Field(default=1, ge=1)


class BatchRequest(Model):
    batch_id: str = Field(min_length=1)
    repo_root: Path
    base_commit: str = Field(min_length=1)
    case_ids: list[str] = Field(min_length=1)
    regression_suite: Path | None = None
    cluster_reason: str | None = None
    limits: BatchLimits = Field(default_factory=BatchLimits)
    record_id: str | None = None
    evo_record_root: Path | None = None


class SourceTestResult(Model):
    passed: bool
    return_code: int | None = None
    command: list[str] = Field(default_factory=list)
    output: str = ""
    timed_out: bool = False


class BatchResult(Model):
    batch_id: str
    status: BatchStatus
    base_commit: str
    branch_name: str | None = None
    candidate_commit: str | None = None
    validation_path: str | None = None
    result_path: str | None = None
    summary: str = ""
    error: str | None = None


class EvolveRequest(Model):
    repo_root: Path
    base_ref: str = "main"
    regression_suite: Path | None = None
    case_ids: list[str] | None = None
    max_cases_per_batch: int = Field(default=10, ge=1)
    parallel_workers: int = Field(default=1, ge=1)
    batch_limits: BatchLimits = Field(default_factory=BatchLimits)
    record_root: Path = Path("record")
    record_id: str | None = None
    benchmark_root: Path | None = None
    evo_record_root: Path = Path("evorecord")


class EvolveResult(Model):
    base_commit: str
    batches: list[BatchResult] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class MergeRequest(Model):
    repo_root: Path
    base_ref: str = "main"
    candidate_branches: list[str] = Field(min_length=1)
    candidate_evidence: list[Path] = Field(default_factory=list)
    replay_workers: int = Field(default=1, ge=1)
    adopt: bool = False


class MergeResult(Model):
    status: str
    base_commit: str
    merge_branch: str | None = None
    merged_commit: str | None = None
    validation_path: str | None = None
    adopted_commit: str | None = None
    conflicted_files: list[str] = Field(default_factory=list)
    result_path: str | None = None
    error: str | None = None


class MergePlan(Model):
    merge_order: list[str] = Field(min_length=1)
    rationale: str = Field(min_length=1)
    likely_conflicts: list[str] = Field(default_factory=list)
