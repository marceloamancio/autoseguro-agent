"""P0-6 — coerência deadline × retries (`autoseguro.config`).

Furo: com os defaults de produção antigos (`quote_max_retries=3`,
`quote_timeout_s=9.0`, `quote_deadline_s=25.0`), o orçamento total de
tentativas é `(3+1)*9 = 36s`, maior que o deadline total de `25s` — o
deadline corta a chamada bem antes do agente esgotar os retries
pretendidos, tornando `quote_max_retries` uma promessa vazia.

Os defaults de produção devem satisfazer
`(quote_max_retries + 1) * quote_timeout_s <= quote_deadline_s`. Quando a
combinação vier do ambiente (override) e ficar incoerente, `load_config`
deve logar um aviso (nunca imprimir segredo) em vez de falhar
silenciosamente.
"""

from __future__ import annotations

import logging

from autoseguro.config import load_config


def test_production_defaults_keep_retry_budget_within_deadline(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-test-key")
    for var in (
        "QUOTE_TIMEOUT_S",
        "QUOTE_MAX_RETRIES",
        "QUOTE_DEADLINE_S",
    ):
        monkeypatch.delenv(var, raising=False)

    config = load_config()

    total_retry_budget = (config.quote_max_retries + 1) * config.quote_timeout_s
    assert total_retry_budget <= config.quote_deadline_s, (
        "FURO: orçamento total de tentativas "
        f"(({config.quote_max_retries}+1)*{config.quote_timeout_s}="
        f"{total_retry_budget}) excede o deadline total "
        f"({config.quote_deadline_s}) -- o deadline corta a chamada antes do "
        "agente esgotar os retries pretendidos."
    )


def test_incoherent_env_combo_is_flagged_by_a_warning_log(monkeypatch, caplog):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-test-key")
    # Combinação deliberadamente incoerente: (3+1)*9 = 36 > 25.
    monkeypatch.setenv("QUOTE_TIMEOUT_S", "9")
    monkeypatch.setenv("QUOTE_MAX_RETRIES", "3")
    monkeypatch.setenv("QUOTE_DEADLINE_S", "25")

    with caplog.at_level(logging.WARNING, logger="autoseguro.config"):
        config = load_config()

    # load_config nao deve silenciosamente sobrescrever um override explicito
    # do ambiente -- so avisar.
    assert config.quote_timeout_s == 9.0
    assert config.quote_max_retries == 3
    assert config.quote_deadline_s == 25.0

    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "FURO: combinação incoerente vinda do ambiente não gerou nenhum aviso"
    assert any("deadline" in msg.lower() for msg in warnings)
    # nunca deve vazar a chave nos logs
    assert all("sk-ant" not in msg for msg in warnings)


def test_coherent_env_combo_does_not_warn(monkeypatch, caplog):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-test-key")
    monkeypatch.setenv("QUOTE_TIMEOUT_S", "9")
    monkeypatch.setenv("QUOTE_MAX_RETRIES", "2")
    monkeypatch.setenv("QUOTE_DEADLINE_S", "30")

    with caplog.at_level(logging.WARNING, logger="autoseguro.config"):
        load_config()

    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert not warnings
