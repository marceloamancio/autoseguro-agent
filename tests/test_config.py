"""Testes do loader de config (autoseguro.config).

Regras cobertas:
- Fail-fast: sem ANTHROPIC_API_KEY, load_config() levanta erro claro e o valor
  da chave NUNCA aparece na mensagem de erro (nem existe pra vazar, no caso ausente).
- Com a env setada, a config carrega com os defaults documentados no wish
  (modelo, QUOTE_API_URL, timeouts/retries de resiliência) e permite override por env.
- Quando a chave está presente, seu valor não deve vazar na representação da config
  (str/repr), só a variável/flag de que existe.
"""

import pytest

from autoseguro.config import ConfigError, load_config


def test_missing_api_key_raises_clear_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(ConfigError) as exc_info:
        load_config()

    message = str(exc_info.value)
    assert "ANTHROPIC_API_KEY" in message
    # mensagem deve orientar a definir a variável, não ecoar nenhum valor de chave
    assert "sk-ant" not in message.lower()


def test_missing_api_key_error_never_leaks_a_value(monkeypatch):
    # Mesmo que outras envs estejam presentes, a ausência da chave nao deve nunca
    # incluir um valor de chave (obviamente nao ha valor pra vazar aqui, mas
    # garantimos que a mensagem so cita o NOME da variavel).
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("QUOTE_API_URL", "http://localhost:8000")

    with pytest.raises(ConfigError) as exc_info:
        load_config()

    assert "ANTHROPIC_API_KEY" in str(exc_info.value)


def test_loads_with_defaults_when_api_key_present(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-test-key")
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    monkeypatch.delenv("QUOTE_API_URL", raising=False)
    monkeypatch.delenv("QUOTE_TIMEOUT_S", raising=False)
    monkeypatch.delenv("QUOTE_MAX_RETRIES", raising=False)
    monkeypatch.delenv("QUOTE_BACKOFF_BASE_S", raising=False)
    monkeypatch.delenv("QUOTE_DEADLINE_S", raising=False)
    monkeypatch.delenv("QUOTE_CB_FAILURE_THRESHOLD", raising=False)
    monkeypatch.delenv("QUOTE_CB_RESET_S", raising=False)

    config = load_config()

    assert config.anthropic_model == "claude-sonnet-5"
    assert config.quote_api_url == "http://localhost:8000"
    assert config.quote_timeout_s == 9.0
    assert config.quote_max_retries == 3
    assert config.quote_backoff_base_s == 0.5
    assert config.quote_deadline_s == 25.0
    assert config.quote_cb_failure_threshold == 5
    assert config.quote_cb_reset_s == 30.0


def test_allows_overriding_defaults_via_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-test-key")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-haiku-custom")
    monkeypatch.setenv("QUOTE_API_URL", "http://quote-api:9000")
    monkeypatch.setenv("QUOTE_TIMEOUT_S", "12")
    monkeypatch.setenv("QUOTE_MAX_RETRIES", "5")

    config = load_config()

    assert config.anthropic_model == "claude-haiku-custom"
    assert config.quote_api_url == "http://quote-api:9000"
    assert config.quote_timeout_s == 12.0
    assert config.quote_max_retries == 5


def test_api_key_value_never_appears_in_repr_or_str(monkeypatch):
    secret = "sk-ant-super-secret-value-12345"
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)

    config = load_config()

    # o valor real da chave deve estar disponível para uso pelo SDK...
    assert config.anthropic_api_key == secret
    # ...mas nunca deve vazar em repr()/str() da config (ex.: logs acidentais)
    assert secret not in repr(config)
    assert secret not in str(config)
