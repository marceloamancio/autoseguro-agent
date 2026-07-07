"""Dublês compartilhados entre os testes de integração — vendorizado de
`adversarial-lab/red_team/doubles.py` (Stage 0.1) para que os testes de
integração não dependam do lab externo. Mesmo estilo dos testes originais do
`autoseguro-agent` (`tests/test_agent.py`, `tests/test_quote_client.py`).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from autoseguro.config import Config
from autoseguro.quote_client import QuoteResult


class StubExtractor:
    """Dublê síncrono do `LlmExtractorClient`. Pode devolver uma sequência de
    respostas diferentes por chamada (`responses`), pra simular o lead indo
    corrigindo/adicionando dado turno a turno — ou uma única resposta fixa
    repetida em todo turno.
    """

    def __init__(self, response: dict | None = None, responses: list[dict] | None = None):
        self._responses = responses
        self._response = response or {}
        self.calls: list[str] = []
        self._idx = 0

    def extract(self, text: str) -> dict:
        self.calls.append(text)
        if self._responses is not None:
            if self._idx < len(self._responses):
                r = self._responses[self._idx]
            else:
                r = self._responses[-1]
            self._idx += 1
            return r
        return self._response


class StubQuoteClient:
    """Dublê assíncrono do `QuoteClient`. Aceita uma sequência de resultados/
    exceções (`sequence`) pra simular respostas diferentes por chamada (ex.:
    sempre recusa por payload inválido, simulando o backend real rejeitando
    uma `data_inicio` fora do formato ISO).
    """

    def __init__(
        self,
        result: QuoteResult | None = None,
        exc: Exception | None = None,
        sequence: list[Any] | None = None,
    ):
        self._result = result
        self._exc = exc
        self._sequence = sequence
        self.calls: list[dict] = []

    async def cotar(self, payload: dict) -> QuoteResult:
        self.calls.append(payload)
        if self._sequence is not None:
            item = self._sequence[min(len(self.calls) - 1, len(self._sequence) - 1)]
            if isinstance(item, Exception):
                raise item
            return item
        if self._exc is not None:
            raise self._exc
        assert self._result is not None
        return self._result


class _StubMessagesApi:
    def __init__(self, text: str):
        self._text = text
        self.calls: list[dict] = []

    async def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(text=self._text)])


class StubLlm:
    """Dublê do cliente `AsyncAnthropic`-like — nunca chama a API real."""

    def __init__(self, text: str = "Fico à disposição!"):
        self.messages = _StubMessagesApi(text)


def make_quote(**overrides: Any) -> QuoteResult:
    defaults = dict(
        plano_id="essencial",
        plano_nome="Essencial",
        premio_mensal=119.90,
        franquia=4500.0,
        coberturas=["colisao", "roubo", "furto"],
        carencia={"coberturas": ["roubo", "furto"], "dias": 30, "observacao": "obs"},
        moeda="BRL",
        multiplicadores={"faixa_etaria": 1.0, "idade_veiculo": 1.0, "regiao": 1.0},
        primeiro_pagamento_pro_rata=None,
    )
    defaults.update(overrides)
    return QuoteResult(**defaults)


def make_config(**overrides: Any) -> Config:
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
