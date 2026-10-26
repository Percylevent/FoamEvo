"""Run a compiled single-batch graph."""

from __future__ import annotations

from ..models import BatchRequest, BatchResult, BatchStatus
from .batch_graph import BatchGraphServices, build_batch_graph


class BatchExecution:
    def __init__(self, *, services: BatchGraphServices | None) -> None:
        self.services = services

    def run(self, request: BatchRequest) -> BatchResult:
        if self.services is None:
            return BatchResult(
                batch_id=request.batch_id,
                status=BatchStatus.FAILED,
                base_commit=request.base_commit,
                summary="BatchGraph services are not configured",
                error="BatchGraphNotReady",
            )
        try:
            limits = request.limits
            steps = 4 + limits.max_iterations * (4 + 2 * limits.max_plan_rounds + 3 * limits.max_code_rounds)
            state = build_batch_graph(self.services).invoke(
                {"request": request}, config={"recursion_limit": steps})
            result = state.get("result")
            if isinstance(result, BatchResult):
                return result
            return BatchResult(
                batch_id=request.batch_id,
                status=BatchStatus.FAILED,
                base_commit=request.base_commit,
                summary=state.get("error", "batch graph finished without a result"),
                error=state.get("error"),
            )
        except Exception as exc:
            return BatchResult(
                batch_id=request.batch_id,
                status=BatchStatus.FAILED,
                base_commit=request.base_commit,
                summary=f"{type(exc).__name__}: {exc}",
                error=f"{type(exc).__name__}: {exc}",
            )
