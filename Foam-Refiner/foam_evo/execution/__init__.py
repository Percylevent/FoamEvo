"""Concrete BatchGraph service wiring."""
from .batch_runtime import BatchRuntime, BatchRuntimeFactory

__all__ = ["BatchRuntime", "BatchRuntimeFactory"]
