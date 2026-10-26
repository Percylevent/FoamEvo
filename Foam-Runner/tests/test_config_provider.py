from config import Config


def test_upstream_provider_environment_aliases(monkeypatch) -> None:
    monkeypatch.delenv("FOAM_AGENT_MODEL_PROVIDER", raising=False)
    monkeypatch.delenv("FOAM_AGENT_MODEL_VERSION", raising=False)
    monkeypatch.setenv("FOAMAGENT_MODEL_PROVIDER", "anthropic")
    monkeypatch.setenv("FOAMAGENT_MODEL_VERSION", "claude-sonnet-4-6")

    config = Config()

    assert config.model_provider == "anthropic"
    assert config.model_version == "claude-sonnet-4-6"


def test_local_provider_environment_names_take_precedence(monkeypatch) -> None:
    monkeypatch.setenv("FOAMAGENT_MODEL_PROVIDER", "anthropic")
    monkeypatch.setenv("FOAMAGENT_MODEL_VERSION", "claude-sonnet-4-6")
    monkeypatch.setenv("FOAM_AGENT_MODEL_PROVIDER", "openai")
    monkeypatch.setenv("FOAM_AGENT_MODEL_VERSION", "gpt-5.5")

    config = Config()

    assert config.model_provider == "openai"
    assert config.model_version == "gpt-5.5"
