"""P0-4 — `httpx.AsyncClient` sem `timeout=` (`quote_client.py`, ~linha 154).

Os testes "oficiais" de `quote_client.py` (`tests/test_quote_client.py`) usam
só `httpx.MockTransport`, que NUNCA passa pelo enforcement de timeout do
httpcore (o handler roda direto, sem timer real de rede) -- então eles não
conseguem provar (nem desmentir) o comportamento real do timeout default do
`httpx.AsyncClient`. Aqui usamos o `fake_quote` (um uvicorn de verdade, socket
TCP real) para provar o furo e a correção.

Design documentado em `quote_client.py`: "Timeout por tentativa =
quote_timeout_s (default 9s = SLOW_SECONDS + 1): suficiente pra capturar a
chamada lenta de 8s do mock, tratando-a como cotação válida". Isso pressupõe
que o único timeout em jogo é o `asyncio.wait_for(..., timeout=quote_timeout_s)`
de `_send`. Só que `self._client = httpx.AsyncClient(base_url=...,
transport=transport)` nunca recebia `timeout=` -- então o httpx aplicava o
SEU default (`httpx._config.DEFAULT_TIMEOUT_CONFIG`, 5s), que é MENOR que 9s
e dispara primeiro como `httpx.ReadTimeout` na conexão real.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from autoseguro.quote_client import QuoteClient, QuoteResult, QuoteUnavailable

from doubles import make_config

PAYLOAD = {"plano_id": "essencial", "idade": 35, "veiculo_ano": 2020, "cep": "01000-000"}


@pytest.mark.asyncio
async def test_asyncclient_uses_configured_timeout():
    """Prova estrutural (sem rede): o `httpx.AsyncClient` interno criado por
    `QuoteClient.__init__` deve receber `timeout=` construído a partir de
    `config.quote_timeout_s` -- nunca o default da lib.
    """
    config = make_config(quote_timeout_s=9.0)
    client = QuoteClient(config)
    try:
        effective_timeout = client._client.timeout  # httpx.Timeout interno
        assert effective_timeout.read == config.quote_timeout_s, (
            "FURO: o timeout de leitura do httpx.AsyncClient nao bate com "
            f"quote_timeout_s={config.quote_timeout_s} (veio "
            f"{effective_timeout.read} -- provavelmente o default da lib, "
            "5s). QuoteClient.__init__ nao repassa timeout= ao construtor do "
            "httpx.AsyncClient."
        )
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_slow_call_within_timeout_returns_quote(fake_quote):
    """Contra um backend real que demora 8s (mas responde), com
    `quote_timeout_s=9` configurado, a chamada deveria devolver uma
    `QuoteResult` -- nunca `QuoteUnavailable`. Hoje o timeout default de 5s do
    httpx corta a conexão antes dos 9s pretendidos.
    """
    fake_quote.set_mode("slow_then_normal", slow_seconds=8.0)

    config = make_config(
        quote_api_url=fake_quote.base_url,
        quote_timeout_s=9.0,
        quote_max_retries=1,
        quote_backoff_base_s=0.1,
        quote_deadline_s=25.0,
        quote_cb_failure_threshold=5,
        quote_cb_reset_s=30.0,
    )
    client = QuoteClient(config)
    try:
        result = await client.cotar(
            {"plano_id": "essencial", "idade": 35, "veiculo_ano": 2020, "cep": "01000-000"}
        )
    finally:
        await client.aclose()

    assert isinstance(result, QuoteResult)


# --- P0-5: 429/408 entram no retry (nao dao break na 1a tentativa) ---------


@pytest.mark.asyncio
async def test_429_consumes_full_retry_budget(fake_quote):
    """Hoje `cotar` so trata `>=500` como infra retryavel -- 429 (rate limit,
    tipicamente transitorio) cai no ramo "status inesperado" e desiste na
    1a tentativa. Deve consumir o orcamento inteiro de tentativas
    (`quote_max_retries + 1`), igual a um 503.
    """
    fake_quote.set_mode("fail_status", fail_status=429)

    config = make_config(
        quote_api_url=fake_quote.base_url,
        quote_timeout_s=1.0,
        quote_max_retries=2,
        quote_backoff_base_s=0.02,
        quote_deadline_s=5.0,
        quote_cb_failure_threshold=100,
        quote_cb_reset_s=30.0,
    )
    client = QuoteClient(config)
    try:
        with pytest.raises(QuoteUnavailable):
            await client.cotar(PAYLOAD)
    finally:
        await client.aclose()

    stats = fake_quote.stats()
    assert stats["quote_calls"] == config.quote_max_retries + 1, (
        "FURO: 429 nao consumiu o orcamento de tentativas "
        f"(esperado {config.quote_max_retries + 1}, veio {stats['quote_calls']})"
    )


@pytest.mark.asyncio
async def test_408_consumes_full_retry_budget(fake_quote):
    """Mesma prova do P0-5 para 408 (Request Timeout)."""
    fake_quote.set_mode("fail_status", fail_status=408)

    config = make_config(
        quote_api_url=fake_quote.base_url,
        quote_timeout_s=1.0,
        quote_max_retries=2,
        quote_backoff_base_s=0.02,
        quote_deadline_s=5.0,
        quote_cb_failure_threshold=100,
        quote_cb_reset_s=30.0,
    )
    client = QuoteClient(config)
    try:
        with pytest.raises(QuoteUnavailable):
            await client.cotar(PAYLOAD)
    finally:
        await client.aclose()

    stats = fake_quote.stats()
    assert stats["quote_calls"] == config.quote_max_retries + 1, (
        "FURO: 408 nao consumiu o orcamento de tentativas "
        f"(esperado {config.quote_max_retries + 1}, veio {stats['quote_calls']})"
    )


# --- P0-7: breaker half-open via canary no /quote, nao via /health --------


@pytest.mark.asyncio
async def test_breaker_probes_quote_not_health_and_reopens_when_quote_still_down(fake_quote):
    """Depois do reset do breaker, a sonda de meia-abertura deve decidir pelo
    `/quote` (canary) -- nunca pelo `/health` (que fica sempre `up` no
    `fake_quote` por padrao). Com `/quote` ainda caido (500), o breaker deve
    REABRIR, nao fechar so porque `/health` respondeu 200.
    """
    fake_quote.set_mode("fail_status", fail_status=500)

    config = make_config(
        quote_api_url=fake_quote.base_url,
        quote_timeout_s=1.0,
        quote_max_retries=0,
        quote_backoff_base_s=0.01,
        quote_deadline_s=5.0,
        quote_cb_failure_threshold=1,
        quote_cb_reset_s=0.05,
    )
    client = QuoteClient(config)
    try:
        # 1) abre o breaker (threshold=1).
        with pytest.raises(QuoteUnavailable):
            await client.cotar(PAYLOAD)

        quote_calls_before = fake_quote.stats()["quote_calls"]

        # 2) espera passar o reset_s -- a proxima chamada deve sondar via
        # canary real no /quote (que ainda esta caido), nunca via /health.
        await asyncio.sleep(0.08)
        with pytest.raises(QuoteUnavailable):
            await client.cotar(PAYLOAD)

        stats = fake_quote.stats()
        assert stats["quote_calls"] == quote_calls_before + 1, (
            "a sonda de meia-abertura deveria ter batido no /quote (canary)"
        )
        assert stats["health_calls"] == 0, (
            "FURO: o breaker sondou /health em vez de decidir pelo /quote real"
        )

        # 3) o breaker deve ter reaberto -- a proxima chamada imediata deve
        # fast-fail sem bater no /quote de novo.
        with pytest.raises(QuoteUnavailable) as exc_info:
            await client.cotar(PAYLOAD)

        stats = fake_quote.stats()
        assert stats["quote_calls"] == quote_calls_before + 1  # nao incrementou
        assert exc_info.value.reason == "circuit_breaker_aberto"
    finally:
        await client.aclose()
