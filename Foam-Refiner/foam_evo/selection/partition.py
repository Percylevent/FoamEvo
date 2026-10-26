"""Deterministic validation for the global Markdown cluster partition."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable


class PartitionError(ValueError):
    pass


def validate_clusters(
    *, expected_case_ids: Iterable[str], clusters: Iterable[object], max_cases_per_batch: int
) -> list[object]:
    expected = list(expected_case_ids)
    if len(expected) != len(set(expected)):
        raise PartitionError("input case IDs must be unique")
    if max_cases_per_batch < 1:
        raise PartitionError("max_cases_per_batch must be positive")
    result = list(clusters)
    if not result:
        raise PartitionError("clustering produced no groups")
    ids: list[str] = []
    cluster_ids: list[str] = []
    for cluster in result:
        cluster_id = str(getattr(cluster, "cluster_id", "")).strip()
        case_ids = [str(item).strip() for item in getattr(cluster, "case_ids", [])]
        if not cluster_id or cluster_id in cluster_ids:
            raise PartitionError("cluster IDs must be non-empty and unique")
        cluster_ids.append(cluster_id)
        if not case_ids:
            raise PartitionError(f"cluster {cluster_id} contains no cases")
        if len(case_ids) > max_cases_per_batch:
            raise PartitionError(f"cluster {cluster_id} exceeds batch limit")
        if len(case_ids) != len(set(case_ids)):
            raise PartitionError(f"cluster {cluster_id} contains duplicate cases")
        unknown = sorted(set(case_ids) - set(expected))
        if unknown:
            raise PartitionError(f"cluster {cluster_id} contains unknown cases: {', '.join(unknown)}")
        ids.extend(case_ids)
    duplicate = sorted(case_id for case_id, count in Counter(ids).items() if count > 1)
    if duplicate:
        raise PartitionError("cases assigned to multiple clusters: " + ", ".join(duplicate))
    uncovered = sorted(set(expected) - set(ids))
    if uncovered:
        raise PartitionError("uncovered cases: " + ", ".join(uncovered))
    return result
