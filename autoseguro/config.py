"""Carregamento de configuração via variáveis de ambiente.

Fail-fast: se `ANTHROPIC_API_KEY` não estiver definida, `load_config()` levanta
`ConfigError` com uma mensagem clara instruindo como definir a variável. O valor
da chave NUNCA é impresso, logado ou incluído em mensagens de erro/repr — apenas
o nome da variável ausente é citado.

Demais variáveis têm defaults alinhados às decisões do desafio (ver DECISOES.md,
Q2 e Q5): modelo default `claude-sonnet-5`, `QUOTE_API_URL` apontando pro
quote-service local, e thresholds de resiliência (timeout = SLOW_SECONDS + 1s,
retries, backoff, deadline total e circuit breaker leve).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

# Carrega variáveis de um .env local (se existir) para o ambiente do processo.
# Não sobrescreve variáveis já definidas no ambiente (ex.: exportadas pelo shell/CI).
load_dotenv()


class ConfigError(RuntimeError):
    """Erro de configuração — nunca deve carregar valores de segredos na mensagem."""


@dataclass(frozen=True)
class Config:
    """Configuração do agente, carregada uma única vez no boot.

    `anthropic_api_key` fica disponível para uso (ex.: passado ao SDK), mas é
    excluído de `repr()`/`str()` via `field(repr=False)` para nunca vazar em
    logs, tracebacks ou prints acidentais da própria config.
    """

    anthropic_api_key: str = field(repr=False)
    anthropic_model: str
    quote_api_url: str
    quote_timeout_s: float
    quote_max_retries: int
    quote_backoff_base_s: float
    quote_deadline_s: float
    quote_cb_failure_threshold: int
    quote_cb_reset_s: float

    def __str__(self) -> str:  # pragma: no cover - trivial, coberto indiretamente
        return self.__repr__()


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def load_config() -> Config:
    """Carrega e valida a configuração a partir do ambiente.

    Levanta `ConfigError` se `ANTHROPIC_API_KEY` estiver ausente ou vazia —
    fail-fast, sem nunca imprimir/logar o valor da chave.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ConfigError(
            "ANTHROPIC_API_KEY não está definida. Defina a variável de ambiente "
            "ANTHROPIC_API_KEY com a sua chave da Anthropic (ex.: `export "
            "ANTHROPIC_API_KEY=...` ou via arquivo .env — veja .env.example) "
            "antes de rodar o agente."
        )

    return Config(
        anthropic_api_key=api_key,
        anthropic_model=os.getenv("ANTHROPIC_MODEL") or "claude-sonnet-5",
        quote_api_url=os.getenv("QUOTE_API_URL") or "http://localhost:8000",
        quote_timeout_s=_get_float("QUOTE_TIMEOUT_S", 9.0),
        quote_max_retries=_get_int("QUOTE_MAX_RETRIES", 3),
        quote_backoff_base_s=_get_float("QUOTE_BACKOFF_BASE_S", 0.5),
        quote_deadline_s=_get_float("QUOTE_DEADLINE_S", 25.0),
        quote_cb_failure_threshold=_get_int("QUOTE_CB_FAILURE_THRESHOLD", 5),
        quote_cb_reset_s=_get_float("QUOTE_CB_RESET_S", 30.0),
    )
