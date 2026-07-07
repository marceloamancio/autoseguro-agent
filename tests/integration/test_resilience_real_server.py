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

import httpx
import pytest

from autoseguro.quote_client import QuoteClient, QuoteResult

from doubles import make_config


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
