"""Testes do CLI REPL (autoseguro.cli) — Group F, DEC-9 (Q4).

Cobertura:
- Fluxo ponta a ponta com LLM e `QuoteClient` mockados (dublês locais, sem
  chamada real de rede/LLM) produz respostas e um `trace.jsonl` com ids de
  mensagem/cotação.
- Ack imediato ao iniciar uma cotação + nudge se a cotação demorar.
- Sem `ANTHROPIC_API_KEY`, `main()` aborta (fail-fast) sem nunca imprimir a
  chave — testado **separadamente** dos demais (não usa monkeypatch de env
  fake para essa parte).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from autoseguro import cli
from autoseguro.agent import Agent
from autoseguro.extraction import QualificationSession
from autoseguro.handoff import HandoffReason
from autoseguro.quote_client import QuoteResult
from autoseguro.tracing import Tracer


class StubExtractor:
    """Dublê síncrono do `LlmExtractorClient` (mesmo Protocol de extraction.py)."""

    def __init__(self, response: dict | None = None):
        self._response = response or {}

    def extract(self, text: str) -> dict:
        return self._response


class StubQuoteClient:
    """Dublê assíncrono do `QuoteClient` — nunca bate em rede real."""

    def __init__(self, result: QuoteResult | None = None, exc: Exception | None = None, delay: float = 0.0):
        self._result = result
        self._exc = exc
        self._delay = delay
        self.calls: list[dict] = []

    async def cotar(self, payload: dict) -> QuoteResult:
        self.calls.append(payload)
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._exc is not None:
            raise self._exc
        assert self._result is not None
        return self._result


class _StubMessagesApi:
    def __init__(self, text: str):
        self._text = text

    async def create(self, **kwargs) -> SimpleNamespace:
        return SimpleNamespace(content=[SimpleNamespace(text=self._text)])


class StubLlm:
    """Dublê do cliente `AsyncAnthropic`-like — nunca chama a API real."""

    def __init__(self, text: str = "Fico à disposição!"):
        self.messages = _StubMessagesApi(text)


def make_quote(**overrides) -> QuoteResult:
    defaults = dict(
        plano_id="completo",
        plano_nome="Completo",
        premio_mensal=209.90,
        franquia=3500.0,
        coberturas=["colisao", "roubo", "furto", "terceiros"],
        carencia={"coberturas": ["roubo", "furto"], "dias": 30, "observacao": "obs"},
        moeda="BRL",
        multiplicadores={"faixa_etaria": 1.0, "idade_veiculo": 1.0, "regiao": 1.0},
        primeiro_pagamento_pro_rata=None,
    )
    defaults.update(overrides)
    return QuoteResult(**defaults)


def _read_events(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _make_input_fn(messages: list[str]):
    it = iter(messages)

    def _input(prompt: str = "") -> str:
        return next(it)

    return _input


# ---------------------------------------------------------------------------
# Fluxo ponta a ponta com mocks -> respostas + trace
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_repl_end_to_end_with_mocks_produces_replies_and_trace(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")  # só pra passar o fail-fast do config

    quote = make_quote()
    quote_client = StubQuoteClient(result=quote)
    extractor = StubExtractor({"idade": 35, "veiculo_ano": 2008, "cep": "26703-384"})
    session = QualificationSession()
    agent = Agent(StubLlm(), quote_client, session, extractor=extractor)

    trace_path = tmp_path / "trace.jsonl"
    tracer = Tracer(path=trace_path, run_id="run-x", conversation_id="conv-x")

    input_fn = _make_input_fn(
        [
            "Tenho um Corolla 2008, 35 anos, CEP 26703-384",
            "sim, confirmo",
            "sair",
        ]
    )
    outputs: list[str] = []

    await cli.run_repl(agent, tracer, input_fn=input_fn, output_fn=outputs.append)
    tracer.close()

    joined = "\n".join(outputs)
    assert "confirmando" in joined.lower() or "correto" in joined.lower()
    assert "209.9" in joined  # cotação entregue, preço vem do mock

    events = _read_events(trace_path)
    assert events, "trace.jsonl deveria ter eventos"
    assert all("event_id" in e and e["run_id"] == "run-x" and e["conversation_id"] == "conv-x" for e in events)

    message_events = [e for e in events if e["type"] in ("message.in", "message.out")]
    assert len(message_events) >= 4  # 2 turnos reais (msg1 + confirmação), in+out cada

    quote_events = [e for e in events if e["type"] == "quote.result"]
    assert len(quote_events) == 1
    assert quote_events[0]["status"] == "success"
    assert quote_events[0]["quote_request_id"]

    decision_events = [e for e in events if e["type"] == "decision"]
    assert any(e["status"] == "resolved" for e in decision_events)


@pytest.mark.asyncio
async def test_run_repl_prints_ack_and_nudge_when_quote_is_slow(tmp_path):
    quote = make_quote()
    quote_client = StubQuoteClient(result=quote, delay=0.05)
    extractor = StubExtractor({"idade": 35, "veiculo_ano": 2008, "cep": "26703-384"})
    session = QualificationSession()
    agent = Agent(StubLlm(), quote_client, session, extractor=extractor)

    tracer = Tracer(path=tmp_path / "trace.jsonl")

    input_fn = _make_input_fn(
        [
            "Tenho um Corolla 2008, 35 anos, CEP 26703-384",
            "confirmo",
            "sair",
        ]
    )
    outputs: list[str] = []

    await cli.run_repl(
        agent,
        tracer,
        input_fn=input_fn,
        output_fn=outputs.append,
        nudge_after_s=0.01,
    )
    tracer.close()

    assert cli.ACK_MESSAGE in outputs
    assert cli.NUDGE_MESSAGE in outputs


@pytest.mark.asyncio
async def test_run_repl_logs_handoff_with_reason_code_when_quote_unavailable(tmp_path):
    from autoseguro.quote_client import QuoteUnavailable

    exc = QuoteUnavailable("esgotou_tentativas:http_503", attempts=3, context={"payload": {}})
    quote_client = StubQuoteClient(exc=exc)
    extractor = StubExtractor({"idade": 40, "veiculo_ano": 2015, "cep": "01000-000"})
    session = QualificationSession()
    agent = Agent(StubLlm(), quote_client, session, extractor=extractor)

    trace_path = tmp_path / "trace.jsonl"
    tracer = Tracer(path=trace_path)

    input_fn = _make_input_fn(
        [
            "40 anos, Onix 2015, cep 01000-000",
            "confirmo, pode cotar",
            "sair",
        ]
    )
    outputs: list[str] = []

    await cli.run_repl(agent, tracer, input_fn=input_fn, output_fn=outputs.append)
    tracer.close()

    events = _read_events(trace_path)
    handoff_events = [e for e in events if e["type"] == "handoff"]
    assert handoff_events
    assert handoff_events[0]["reason_code"] == "quote_unavailable"

    quote_events = [e for e in events if e["type"] == "quote.result"]
    assert quote_events
    assert quote_events[0]["status"] == "unavailable"
    assert quote_events[0]["attempts"] == 3

    decision_events = [e for e in events if e["type"] == "decision"]
    assert any(e["status"] == "handoff" for e in decision_events)


# ---------------------------------------------------------------------------
# 2.3 (P1-3) — CLI parseia o marcador líder de mídia e passa `media_type`
# pro `Agent.handle_turn` (antes, código morto: `MEDIA_UNREADABLE` nunca
# disparava na entrega real).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cli_media_marker_triggers_media_unreadable(tmp_path):
    quote_client = StubQuoteClient(result=make_quote())
    extractor = StubExtractor({})
    session = QualificationSession()
    agent = Agent(StubLlm(), quote_client, session, extractor=extractor)

    tracer = Tracer(path=tmp_path / "trace.jsonl")

    input_fn = _make_input_fn(["[documento] CNH_frente.pdf", "sair"])
    outputs: list[str] = []

    await cli.run_repl(agent, tracer, input_fn=input_fn, output_fn=outputs.append)
    tracer.close()

    assert agent.state.handoff is not None
    assert agent.state.handoff.reason == HandoffReason.MEDIA_UNREADABLE
    assert quote_client.calls == []  # nunca tratou o marcador como texto de qualificação


@pytest.mark.asyncio
async def test_cli_regular_text_still_qualifies_normally(tmp_path):
    quote = make_quote()
    quote_client = StubQuoteClient(result=quote)
    extractor = StubExtractor({"idade": 35, "veiculo_ano": 2008, "cep": "26703-384"})
    session = QualificationSession()
    agent = Agent(StubLlm(), quote_client, session, extractor=extractor)

    tracer = Tracer(path=tmp_path / "trace.jsonl")

    input_fn = _make_input_fn(
        ["Tenho um Corolla 2008, 35 anos, CEP 26703-384", "sim, confirmo", "sair"]
    )
    outputs: list[str] = []

    await cli.run_repl(agent, tracer, input_fn=input_fn, output_fn=outputs.append)
    tracer.close()

    assert agent.state.handoff is None
    assert len(quote_client.calls) == 1


@pytest.mark.parametrize(
    "raw, expected_text, expected_media_type",
    [
        ("[documento] CNH_frente.pdf", "CNH_frente.pdf", "document"),
        ("[imagem] foto.png", "foto.png", "image"),
        ("[foto] carro.jpg", "carro.jpg", "image"),
        ("[audio] mensagem.ogg", "mensagem.ogg", "audio"),
        ("Tenho 35 anos", "Tenho 35 anos", None),
    ],
)
def test_parse_media_marker(raw, expected_text, expected_media_type):
    text, media_type = cli.parse_media_marker(raw)

    assert text == expected_text
    assert media_type == expected_media_type


# ---------------------------------------------------------------------------
# 2.4 (P1-1) — log entregue/curado: varredura LLM em lote, fora do hot-path
# ---------------------------------------------------------------------------


def test_cure_delivered_log_calls_sweep_once_in_batch_and_masks_beyond_regex():
    calls = []

    def fake_sweep(texts, categories):
        calls.append((list(texts), list(categories)))
        return [t.replace("Fulano de Tal", "⟨NOME_TERCEIRO⟩") for t in texts]

    events = [
        {
            "type": "message.in",
            "message_body": "cpf 389.083.863-43, e Fulano de Tal confirma",
        },
        {"type": "message.out", "message_body": "sem nenhuma pii aqui"},
    ]

    cured = cli.cure_delivered_log(events, llm_client=fake_sweep)

    assert len(calls) == 1  # uma única chamada em lote pro log inteiro
    assert "⟨CPF⟩" in cured[0]["message_body"]
    assert "⟨NOME_TERCEIRO⟩" in cured[0]["message_body"]
    assert cured[1]["message_body"] == "sem nenhuma pii aqui"
    # não muta os eventos originais
    assert events[0]["message_body"] == "cpf 389.083.863-43, e Fulano de Tal confirma"


def test_cure_delivered_log_sem_llm_client_aplica_so_o_regex():
    events = [{"type": "message.in", "message_body": "cpf 389.083.863-43"}]

    cured = cli.cure_delivered_log(events)

    assert cured[0]["message_body"] == "cpf ⟨CPF⟩"


def test_cure_delivered_log_ignora_campos_de_texto_nao_configurados():
    events = [{"message_body": "cpf 389.083.863-43", "sender_name": "Fulano de Tal"}]

    cured = cli.cure_delivered_log(events, text_fields=("message_body",))

    assert cured[0]["sender_name"] == "Fulano de Tal"


def test_write_jsonl_grava_um_evento_por_linha(tmp_path):
    events = [{"a": 1}, {"b": 2}]
    path = tmp_path / "delivered.jsonl"

    cli.write_jsonl(path, events)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == events


# ---------------------------------------------------------------------------
# Fail-fast de chave (separado dos demais — sem monkeypatch de env fake)
# ---------------------------------------------------------------------------


def test_main_fails_fast_without_api_key_and_never_prints_key(monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code != 0
    captured = capsys.readouterr()
    assert "ANTHROPIC_API_KEY" in captured.err
    assert "sk-ant" not in captured.err.lower()
