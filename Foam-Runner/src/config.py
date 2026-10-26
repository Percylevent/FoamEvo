"""Runtime configuration for Foam-Runner.

Foam-Runner is configured by environment variables. Runtime records may write
a non-sensitive ``runenv.yaml`` description, but that file is never read as
configuration and no repository-local YAML file is required.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path


class ConfigError(RuntimeError):
    """Raised when an environment configuration value is invalid."""


def _env(name: str, default):
    value = os.getenv(name)
    return default if value is None or not value.strip() else value.strip()


def _env_int(name: str, default: int) -> int:
    value = _env(name, str(default))
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be an integer") from exc


def _env_float(name: str, default: float) -> float:
    value = _env(name, str(default))
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be a number") from exc


@dataclass
class Config:
    max_loop: int = 25
    max_no_progress_loops: int = 2
    batchsize: int = 10
    searchdocs: int = 10
    run_times: int = 1
    database_path: str = str(Path(__file__).resolve().parent.parent / "database")
    run_directory: str = ""
    case_dir: str = ""
    max_time_limit: int = 3600
    recursion_limit: int = 100
    input_writer_generation_mode: str = "sequential_dependency"
    reuse_generated_dir: str = ""
    model_provider: str = "openai"
    model_version: str = "gpt-4o"
    temperature: float = 1.0
    reasoning_effort: str = "max"
    http_timeout: int = 300
    structured_output_retries: int = 2
    openfoam_fork: str = "foundation"
    openfoam_version: str = ""
    experience_context: dict = field(default_factory=dict)
    embedding_provider: str = "huggingface"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_style: str = "chat_completions"
    anthropic_api_key: str = ""
    anthropic_base_url: str = "https://api.anthropic.com"
    embedding_api_key: str = ""
    embedding_base_url: str = "https://api.openai.com/v1"

    def __post_init__(self) -> None:
        """Apply the documented FOAM_AGENT_* environment configuration."""

        # FOAMAGENT_* is the upstream project's spelling. Keep it as an alias
        # while preferring this fork's documented FOAM_AGENT_* variables.
        self.model_provider = str(
            _env(
                "FOAM_AGENT_MODEL_PROVIDER",
                _env("FOAMAGENT_MODEL_PROVIDER", self.model_provider),
            )
        )
        self.model_version = str(
            _env(
                "FOAM_AGENT_MODEL_VERSION",
                _env("FOAMAGENT_MODEL_VERSION", self.model_version),
            )
        )
        self.openai_base_url = str(_env("FOAM_AGENT_OPENAI_BASE_URL", self.openai_base_url))
        self.openai_api_key = str(_env("FOAM_AGENT_OPENAI_API_KEY", self.openai_api_key))
        self.anthropic_base_url = str(
            _env(
                "FOAM_AGENT_ANTHROPIC_BASE_URL",
                _env("ANTHROPIC_BASE_URL", self.anthropic_base_url),
            )
        )
        self.anthropic_api_key = str(
            _env(
                "FOAM_AGENT_ANTHROPIC_API_KEY",
                _env("ANTHROPIC_API_KEY", self.anthropic_api_key),
            )
        )
        self.embedding_provider = str(_env("FOAM_AGENT_EMBEDDING_PROVIDER", self.embedding_provider))
        self.embedding_model = str(_env("FOAM_AGENT_EMBEDDING_MODEL", self.embedding_model))
        self.openfoam_fork = str(_env("FOAM_AGENT_OPENFOAM_FORK", self.openfoam_fork)).lower()
        self.openfoam_version = str(_env("FOAM_RUNNER_OPENFOAM_VERSION", os.getenv("WM_PROJECT_VERSION", self.openfoam_version)))
        self.run_directory = str(_env("FOAM_AGENT_RUN_ROOT", self.run_directory or "record"))
        self.temperature = _env_float("FOAM_AGENT_TEMPERATURE", self.temperature)
        self.http_timeout = _env_int("FOAM_AGENT_HTTP_TIMEOUT", self.http_timeout)
        self.structured_output_retries = _env_int(
            "FOAM_AGENT_STRUCTURED_OUTPUT_RETRIES",
            self.structured_output_retries,
        )

        if self.openfoam_fork not in {"foundation", "esi"}:
            raise ConfigError("FOAM_AGENT_OPENFOAM_FORK must be foundation or esi")
        if self.embedding_provider not in {"openai", "huggingface", "ollama"}:
            raise ConfigError("FOAM_AGENT_EMBEDDING_PROVIDER is not supported")
        if self.http_timeout <= 0:
            raise ConfigError("FOAM_AGENT_HTTP_TIMEOUT must be positive")
        if self.structured_output_retries < 0:
            raise ConfigError("FOAM_AGENT_STRUCTURED_OUTPUT_RETRIES must be non-negative")

        print(f"<config>model_provider={self.model_provider} (environment/default)</config>")
        print(f"<config>model_version={self.model_version} (environment/default)</config>")
        print(f"<config>embedding_provider={self.embedding_provider} (environment/default)</config>")
        print(f"<config>embedding_model={self.embedding_model} (environment/default)</config>")
        print(f"<config>openfoam_fork={self.openfoam_fork} (environment/default)</config>")
        print(f"<config>run_root={self.run_directory} (environment/default)</config>")
