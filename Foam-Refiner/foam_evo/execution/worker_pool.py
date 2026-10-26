"""Small coordinator for independent case subprocesses.

Workers must only write their own case directory. The coordinator collects
results and leaves aggregate persistence to its caller, so the same primitive
can be used by benchmark runs and candidate replays.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Generic, Iterable, TypeVar


T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class CaseJob(Generic[T]):
    case_id: str
    payload: T


class CaseWorkerPool(Generic[T, R]):
    """Run independent case jobs with bounded concurrency."""

    def __init__(self, workers: int = 1) -> None:
        if workers < 1:
            raise ValueError("workers must be at least one")
        self.workers = workers

    def run(
        self,
        jobs: Iterable[CaseJob[T]],
        worker: Callable[[CaseJob[T]], R],
    ) -> list[R]:
        pending = list(jobs)
        if not pending:
            return []
        results: dict[str, R] = {}
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {executor.submit(worker, job): job for job in pending}
            for future in as_completed(futures):
                job = futures[future]
                results[job.case_id] = future.result()
        return [results[job.case_id] for job in pending]
