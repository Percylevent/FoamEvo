# Namespace package for service-layer wrappers
from utils import LLMService
from config import Config


class _LazyLLMService:
    """Create the configured LLM client only when a service actually invokes it."""

    def __init__(self) -> None:
        self._service: LLMService | None = None

    def _get(self) -> LLMService:
        if self._service is None:
            self._service = LLMService(Config())
        return self._service

    def __getattr__(self, name: str):
        return getattr(self._get(), name)


global_llm_service = _LazyLLMService()
