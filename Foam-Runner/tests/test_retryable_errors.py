"""Retry transient gateway failures while rejecting persistent client errors."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils import LLMService  # noqa: E402


class _StructuredOutput(BaseModel):
    answer: str


def test_openai_structured_calls_use_default_method() -> None:
    class Structured:
        def invoke(self, messages):
            return _StructuredOutput(answer="pong")

    class FakeLLM:
        def __init__(self):
            self.kwargs = None

        def with_structured_output(self, schema, **kwargs):
            assert schema is _StructuredOutput
            self.kwargs = kwargs
            return Structured()

        def get_num_tokens(self, text):
            return len(text)

    service = object.__new__(LLMService)
    service.model_provider = "openai"
    service.llm = FakeLLM()
    service.total_calls = 0
    service.total_prompt_tokens = 0
    service.total_completion_tokens = 0
    service.total_tokens = 0
    service.failed_calls = 0
    service.retry_count = 0
    result = service.invoke("ping", pydantic_obj=_StructuredOutput, max_retries=0)
    assert result.answer == "pong"
    assert service.llm.kwargs == {}


def _bare_service(llm, *, structured_output_retries=2):
    service = object.__new__(LLMService)
    service.model_provider = "openai"
    service._config = type(
        "Config",
        (),
        {"structured_output_retries": structured_output_retries},
    )()
    service.llm = llm
    service.total_calls = 0
    service.total_prompt_tokens = 0
    service.total_completion_tokens = 0
    service.total_tokens = 0
    service.failed_calls = 0
    service.retry_count = 0
    return service


def test_structured_output_none_is_retried() -> None:
    class Structured:
        calls = 0

        def invoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                return None
            return _StructuredOutput(answer="pong")

    structured = Structured()

    class FakeLLM:
        def with_structured_output(self, schema, **kwargs):
            return structured

        def get_num_tokens(self, text):
            return len(text)

    service = _bare_service(FakeLLM())
    result = service.invoke("ping", pydantic_obj=_StructuredOutput, max_retries=0)

    assert result.answer == "pong"
    assert structured.calls == 2
    assert service.retry_count == 1


def test_pydantic_validation_error_is_retried() -> None:
    invalid = None
    try:
        _StructuredOutput.model_validate({"answer": ["not", "a", "string"]})
    except ValidationError as exc:
        invalid = exc
    assert invalid is not None

    class Structured:
        calls = 0

        def invoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                raise invalid
            return _StructuredOutput(answer="pong")

    structured = Structured()

    class FakeLLM:
        def with_structured_output(self, schema, **kwargs):
            return structured

        def get_num_tokens(self, text):
            return len(text)

    service = _bare_service(FakeLLM())
    result = service.invoke("ping", pydantic_obj=_StructuredOutput, max_retries=0)

    assert result.answer == "pong"
    assert structured.calls == 2


def test_token_count_falls_back_when_provider_tokenizer_is_unavailable() -> None:
    class FakeLLM:
        def get_num_tokens(self, text):
            raise OSError("tokenizer is not installed")

    service = _bare_service(FakeLLM())

    assert service._get_num_tokens("hello world") > 0


class _Detector:
    """Exercise the classifier without constructing a real LLM client."""

    _RETRYABLE_STATUS_CODES = LLMService._RETRYABLE_STATUS_CODES
    _RETRYABLE_MESSAGE_MARKERS = LLMService._RETRYABLE_MESSAGE_MARKERS
    _is_throttling_error = LLMService._is_throttling_error


def classify(error: Exception) -> bool:
    return _Detector()._is_throttling_error(error)


class _StatusError(Exception):
    def __init__(self, status_code: int, message: str = "boom") -> None:
        super().__init__(message)
        self.status_code = status_code


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504, 520, 522, 524])
def test_transient_status_codes_are_retryable(status: int) -> None:
    assert classify(_StatusError(status)) is True


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_client_errors_are_not_retryable(status: int) -> None:
    """A bad request will fail identically on retry; failing fast is correct."""
    assert classify(_StatusError(status)) is False


def test_relay_524_message_body_is_retryable() -> None:
    """The exact shape seen from an OpenAI-compatible relay during the sweep."""
    message = (
        "Error code: 524 - {'error': {'message': 'The origin web server did not "
        "return a complete response within the 120-second Proxy Read Timeout "
        "window.', 'type': 'bad_response_status_code'}}"
    )
    assert classify(Exception(message)) is True


def test_plain_400_message_body_is_not_retryable() -> None:
    assert classify(Exception("Error code: 400 - invalid request")) is False


def test_named_sdk_exceptions_are_retryable() -> None:
    for name in ("RateLimitError", "APITimeoutError", "InternalServerError"):
        error = type(name, (Exception,), {})("upstream trouble")
        assert classify(error) is True, name


def test_legacy_throttling_signals_still_match() -> None:
    assert classify(Exception("ThrottlingException: slow down")) is True
    assert classify(Exception("Too many tokens, please wait")) is True


def test_unrelated_error_is_not_retryable() -> None:
    assert classify(ValueError("solver diverged")) is False


UPSTREAM_OUTAGE_MESSAGE = (
    "Error code: 400 - {'error': {'message': 'The requested model is not "
    "available for the configured upstream account'}}"
)


def test_upstream_account_outage_is_retryable() -> None:
    """Recognize transient upstream outages even when a relay returns 400."""
    assert classify(Exception(UPSTREAM_OUTAGE_MESSAGE)) is True


def test_upstream_account_outage_is_retryable_with_status_attribute() -> None:
    error = _StatusError(400, UPSTREAM_OUTAGE_MESSAGE)
    assert classify(error) is True


def test_upstream_marker_is_case_insensitive() -> None:
    assert classify(Exception("UPSTREAM ACCOUNT temporarily unavailable")) is True


def test_plain_client_error_stays_fast_fail() -> None:
    """The message-body carve-out must not reopen the whole 400 class."""
    assert classify(_StatusError(400, "invalid_request_error: bad schema")) is False
