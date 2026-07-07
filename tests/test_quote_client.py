"""Testes do cliente resiliente da `/quote` (autoseguro.quote_client).

Sem rede real: usa `httpx.MockTransport` com handlers determinísticos (às
vezes assíncronos, pra simular a chamada lenta sem bloquear o event loop).
Cobre a política de resiliência do Group B / DEC-7 (Q5):
- 5xx e timeout/erro de transporte -> retry com backoff; 422/400 -> nunca.
- Chamada lenta > timeout -> tratada como timeout -> retry.
- Circuit breaker: abre após N falhas seguidas (fast-fail sem bater no
  endpoint) e fecha sondando /health após o reset.
- Sucesso -> objeto tipado (`QuoteResult`) com os campos da resposta.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from autoseguro.config import Config
from autoseguro.quote_client import (
    CotacaoRecusada,
    PayloadInvalido,
    QuoteClient,
    QuoteResult,
    QuoteUnavailable,
)

PAYLOAD = {"plano_id": "essencial", "idade": 35, "veiculo_ano": 2020, "cep": "01000-000"}

SUCCESS_BODY = {
    "plano_id": "essencial",
    "plano_nome": "Essencial",
    "premio_mensal": 119.90,
    "franquia": 4500,
    "coberturas": ["colisao", "roubo", "furto"],
    "multiplicadores": {"faixa_etaria": 1.0, "idade_veiculo": 1.0, "regiao": 1.0},
    "carencia": {"coberturas": ["roubo", "furto"], "dias": 30, "observacao": "obs"},
    "moeda": "BRL",
}


def make_config(**overrides) -> Config:
    defaults = dict(
        anthropic_api_key="sk-ant-fake-test-key",
        anthropic_model="claude-sonnet-5",
        quote_api_url="http://quote-service.test",
        quote_timeout_s=0.2,
        quote_max_retries=3,
        quote_backoff_base_s=0.01,
        quote_deadline_s=5.0,
        quote_cb_failure_threshold=5,
        quote_cb_reset_s=0.1,
    )
    defaults.update(overrides)
    return Config(**defaults)


def json_response(status_code: int, body: dict) -> httpx.Response:
    return httpx.Response(status_code, json=body)


class CallCounter:
    """Conta chamadas por path, pra afirmar quantas vezes o endpoint foi batido."""

    def __init__(self):
        self.calls: list[str] = []

    def record(self, request: httpx.Request) -> None:
        self.calls.append(request.url.path)

    def count(self, path: str) -> int:
        return sum(1 for p in self.calls if p == path)


def client_with_handler(config: Config, handler) -> QuoteClient:
    transport = httpx.MockTransport(handler)
    return QuoteClient(config, transport=transport)


# --- Sucesso ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_success_returns_typed_quote_result():
    counter = CallCounter()

    def handler(request: httpx.Request) -> httpx.Response:
        counter.record(request)
        return json_response(200, SUCCESS_BODY)

    client = client_with_handler(make_config(), handler)
    try:
        result = await client.cotar(PAYLOAD)
    finally:
        await client.aclose()

    assert isinstance(result, QuoteResult)
    assert result.plano_id == "essencial"
    assert result.premio_mensal == 119.90
    assert result.franquia == 4500
    assert result.coberturas == ["colisao", "roubo", "furto"]
    assert result.carencia["dias"] == 30
    assert result.moeda == "BRL"
    assert result.multiplicadores["faixa_etaria"] == 1.0
    assert result.primeiro_pagamento_pro_rata is None
    assert counter.count("/quote") == 1


@pytest.mark.asyncio
async def test_success_parses_optional_pro_rata_field():
    body = dict(SUCCESS_BODY)
    body["primeiro_pagamento_pro_rata"] = {
        "dias_no_mes": 30,
        "dias_cobrados": 15,
        "valor_primeiro_pagamento": 59.95,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(200, body)

    client = client_with_handler(make_config(), handler)
    try:
        result = await client.cotar(PAYLOAD)
    finally:
        await client.aclose()

    assert result.primeiro_pagamento_pro_rata == {
        "dias_no_mes": 30,
        "dias_cobrados": 15,
        "valor_primeiro_pagamento": 59.95,
    }


# --- 5xx: retry ---------------------------------------------------------


@pytest.mark.asyncio
async def test_5xx_retries_then_succeeds():
    counter = CallCounter()

    def handler(request: httpx.Request) -> httpx.Response:
        counter.record(request)
        if counter.count("/quote") < 3:
            return json_response(503, {"error": "upstream_unavailable"})
        return json_response(200, SUCCESS_BODY)

    client = client_with_handler(make_config(quote_max_retries=3), handler)
    try:
        result = await client.cotar(PAYLOAD)
    finally:
        await client.aclose()

    assert result.premio_mensal == 119.90
    assert counter.count("/quote") == 3


@pytest.mark.asyncio
async def test_5xx_exhausts_retries_and_raises_quote_unavailable():
    counter = CallCounter()

    def handler(request: httpx.Request) -> httpx.Response:
        counter.record(request)
        return json_response(500, {"error": "upstream_unavailable"})

    client = client_with_handler(
        make_config(quote_max_retries=2, quote_cb_failure_threshold=100), handler
    )
    try:
        with pytest.raises(QuoteUnavailable) as exc_info:
            await client.cotar(PAYLOAD)
    finally:
        await client.aclose()

    # 1 tentativa inicial + 2 retries = 3 chamadas ao endpoint.
    assert counter.count("/quote") == 3
    assert exc_info.value.attempts == 3
    assert exc_info.value.context["payload"] == PAYLOAD
    assert "500" in exc_info.value.reason or "esgotou" in exc_info.value.reason


# --- 422 / 400: sem retry, erro de negócio -------------------------------


@pytest.mark.asyncio
async def test_422_raises_cotacao_recusada_without_retry():
    counter = CallCounter()

    def handler(request: httpx.Request) -> httpx.Response:
        counter.record(request)
        return json_response(422, {"error": "cotacao_recusada", "motivo": "Idade acima do limite."})

    client = client_with_handler(make_config(quote_max_retries=3), handler)
    try:
        with pytest.raises(CotacaoRecusada) as exc_info:
            await client.cotar(PAYLOAD)
    finally:
        await client.aclose()

    assert exc_info.value.motivo == "Idade acima do limite."
    assert counter.count("/quote") == 1


@pytest.mark.asyncio
async def test_400_raises_payload_invalido_without_retry():
    counter = CallCounter()

    def handler(request: httpx.Request) -> httpx.Response:
        counter.record(request)
        return json_response(400, {"error": "payload_invalido", "detalhe": "'idade'"})

    client = client_with_handler(make_config(quote_max_retries=3), handler)
    try:
        with pytest.raises(PayloadInvalido) as exc_info:
            await client.cotar(PAYLOAD)
    finally:
        await client.aclose()

    assert exc_info.value.detalhe == "'idade'"
    assert counter.count("/quote") == 1


# --- Timeout / chamada lenta ----------------------------------------------


@pytest.mark.asyncio
async def test_slow_call_beyond_timeout_is_treated_as_timeout_and_retried():
    counter = CallCounter()

    async def handler(request: httpx.Request) -> httpx.Response:
        counter.record(request)
        if counter.count("/quote") == 1:
            # Simula a chamada lenta (QUOTE_SLOW_SECONDS) além do timeout
            # configurado por tentativa -> deve ser cortada e contar como
            # falha de infra retryable.
            await asyncio.sleep(0.5)
            return json_response(200, SUCCESS_BODY)
        return json_response(200, SUCCESS_BODY)

    client = client_with_handler(
        make_config(quote_timeout_s=0.05, quote_backoff_base_s=0.01), handler
    )
    try:
        result = await client.cotar(PAYLOAD)
    finally:
        await client.aclose()

    assert result.premio_mensal == 119.90
    assert counter.count("/quote") == 2


@pytest.mark.asyncio
async def test_always_slow_call_exhausts_retries_as_quote_unavailable():
    counter = CallCounter()

    async def handler(request: httpx.Request) -> httpx.Response:
        counter.record(request)
        await asyncio.sleep(0.5)
        return json_response(200, SUCCESS_BODY)  # nunca alcançado a tempo

    client = client_with_handler(
        make_config(
            quote_timeout_s=0.05,
            quote_max_retries=1,
            quote_backoff_base_s=0.01,
            quote_cb_failure_threshold=100,
        ),
        handler,
    )
    try:
        with pytest.raises(QuoteUnavailable) as exc_info:
            await client.cotar(PAYLOAD)
    finally:
        await client.aclose()

    assert counter.count("/quote") == 2  # 1 inicial + 1 retry
    assert "timeout" in exc_info.value.reason


# --- Deadline total ---------------------------------------------------------


@pytest.mark.asyncio
async def test_deadline_exceeded_raises_quote_unavailable_before_all_retries():
    counter = CallCounter()

    def handler(request: httpx.Request) -> httpx.Response:
        counter.record(request)
        return json_response(503, {"error": "upstream_unavailable"})

    client = client_with_handler(
        make_config(
            quote_max_retries=10,
            quote_backoff_base_s=0.05,
            quote_deadline_s=0.03,
            quote_cb_failure_threshold=100,
        ),
        handler,
    )
    try:
        with pytest.raises(QuoteUnavailable) as exc_info:
            await client.cotar(PAYLOAD)
    finally:
        await client.aclose()

    # O deadline curto deve cortar bem antes das 11 tentativas possíveis.
    assert counter.count("/quote") < 11
    assert exc_info.value.reason in {"deadline_excedido"} or "esgotou" in exc_info.value.reason


# --- Circuit breaker ---------------------------------------------------------


@pytest.mark.asyncio
async def test_circuit_breaker_opens_and_fast_fails_without_hitting_endpoint():
    counter = CallCounter()

    def handler(request: httpx.Request) -> httpx.Response:
        counter.record(request)
        if request.url.path == "/health":
            return json_response(200, {"status": "ok"})
        return json_response(500, {"error": "upstream_unavailable"})

    config = make_config(
        quote_max_retries=0,
        quote_cb_failure_threshold=2,
        quote_cb_reset_s=60.0,
    )
    client = client_with_handler(config, handler)
    try:
        with pytest.raises(QuoteUnavailable):
            await client.cotar(PAYLOAD)
        with pytest.raises(QuoteUnavailable):
            await client.cotar(PAYLOAD)

        assert counter.count("/quote") == 2  # breaker ainda fechado nas 2 primeiras

        # Terceira chamada: breaker deve estar aberto -> fast-fail, sem bater
        # no endpoint (reset_s=60s é grande demais pra já ter passado).
        with pytest.raises(QuoteUnavailable) as exc_info:
            await client.cotar(PAYLOAD)

        assert exc_info.value.reason == "circuit_breaker_aberto"
        assert counter.count("/quote") == 2  # não incrementou
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_circuit_breaker_closes_after_reset_when_health_probe_ok():
    counter = CallCounter()

    def handler(request: httpx.Request) -> httpx.Response:
        counter.record(request)
        if request.url.path == "/health":
            return json_response(200, {"status": "ok"})
        if counter.count("/quote") <= 2:
            return json_response(500, {"error": "upstream_unavailable"})
        return json_response(200, SUCCESS_BODY)

    config = make_config(
        quote_max_retries=0,
        quote_cb_failure_threshold=2,
        quote_cb_reset_s=0.05,
    )
    client = client_with_handler(config, handler)
    try:
        with pytest.raises(QuoteUnavailable):
            await client.cotar(PAYLOAD)
        with pytest.raises(QuoteUnavailable):
            await client.cotar(PAYLOAD)

        # Breaker aberto agora; chamada imediata deve fast-fail sem sondar.
        with pytest.raises(QuoteUnavailable):
            await client.cotar(PAYLOAD)
        assert counter.count("/health") == 0

        # Espera passar o reset_s e tenta de novo: deve sondar /health (ok)
        # e deixar a chamada real passar, que agora terá sucesso (200).
        await asyncio.sleep(0.06)
        result = await client.cotar(PAYLOAD)

        assert isinstance(result, QuoteResult)
        assert counter.count("/health") == 1
        assert counter.count("/quote") == 3
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_circuit_breaker_stays_open_when_health_probe_fails():
    counter = CallCounter()

    def handler(request: httpx.Request) -> httpx.Response:
        counter.record(request)
        if request.url.path == "/health":
            return json_response(503, {"status": "down"})
        return json_response(500, {"error": "upstream_unavailable"})

    config = make_config(
        quote_max_retries=0,
        quote_cb_failure_threshold=1,
        quote_cb_reset_s=0.03,
    )
    client = client_with_handler(config, handler)
    try:
        with pytest.raises(QuoteUnavailable):
            await client.cotar(PAYLOAD)  # abre o breaker (threshold=1)

        await asyncio.sleep(0.04)
        with pytest.raises(QuoteUnavailable) as exc_info:
            await client.cotar(PAYLOAD)  # sonda /health, falha, reabre

        assert exc_info.value.reason == "circuit_breaker_aberto"
        assert counter.count("/health") == 1
        assert counter.count("/quote") == 1  # a sonda não bateu em /quote
    finally:
        await client.aclose()
