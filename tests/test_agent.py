"""Testes do core do agente (autoseguro.agent) — Group E.

Cobre o fluxo conversa → qualifica → cota → decide com LLM e `QuoteClient`
mockados (dublês locais, sem chamada de rede real):
- Happy path: extração + confirmação + `/quote` 200 → cotação explicada,
  mencionando carência e pró-rata.
- `QuoteUnavailable` → handoff `QUOTE_UNAVAILABLE`, sem inventar preço.
- 422 `CotacaoRecusada` → explica a recusa, sem handoff, sem retry.
- Minimização: o agente nunca solicita CPF/e-mail/telefone/placa.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from autoseguro.agent import (
    Agent,
    FIELD_QUESTIONS,
    SYSTEM_PROMPT,
    build_missing_fields_question,
)
from autoseguro.extraction import QualificationSession
from autoseguro.handoff import HandoffReason
from autoseguro.quote_client import CotacaoRecusada, PayloadInvalido, QuoteResult, QuoteUnavailable

FORBIDDEN_WORDS = ("cpf", "e-mail", "email", "telefone", "placa")


class StubExtractor:
    """Dublê síncrono do `LlmExtractorClient` — mesmo Protocol de extraction.py."""

    def __init__(self, response: dict | None = None):
        self._response = response or {}
        self.calls: list[str] = []

    def extract(self, text: str) -> dict:
        self.calls.append(text)
        return self._response


class StubQuoteClient:
    """Dublê assíncrono do `QuoteClient` — nunca bate em rede real."""

    def __init__(self, result: QuoteResult | None = None, exc: Exception | None = None):
        self._result = result
        self._exc = exc
        self.calls: list[dict] = []

    async def cotar(self, payload: dict) -> QuoteResult:
        self.calls.append(payload)
        if self._exc is not None:
            raise self._exc
        assert self._result is not None
        return self._result


class _StubMessagesApi:
    def __init__(self, text: str):
        self._text = text
        self.calls: list[dict] = []

    async def create(self, **kwargs) -> SimpleNamespace:
        self.calls.append(kwargs)
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
        primeiro_pagamento_pro_rata={
            "dias_no_mes": 30,
            "dias_cobrados": 20,
            "valor_primeiro_pagamento": 139.93,
        },
    )
    defaults.update(overrides)
    return QuoteResult(**defaults)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_happy_path_delivers_quote_mentioning_carencia_and_pro_rata():
    quote = make_quote()
    quote_client = StubQuoteClient(result=quote)
    extractor = StubExtractor({"idade": 35, "veiculo_ano": 2008, "cep": "26703-384"})
    session = QualificationSession()
    agent = Agent(StubLlm(), quote_client, session, extractor=extractor)

    turn1 = await agent.handle_turn(
        "Tenho um Corolla 2008, 35 anos, CEP 26703-384"
    )
    assert turn1.handoff is None
    assert turn1.quote is None
    assert "correto" in turn1.reply.lower() or "confirmando" in turn1.reply.lower()

    turn2 = await agent.handle_turn("sim, confirmo")

    assert turn2.handoff is None
    assert turn2.quote is quote
    assert len(quote_client.calls) == 1
    reply_lower = turn2.reply.lower()
    assert "carência" in reply_lower or "carencia" in reply_lower
    assert "pró-rata" in reply_lower or "proporcional" in reply_lower
    assert "209.9" in turn2.reply  # o preço vem exatamente da resposta mockada


@pytest.mark.asyncio
async def test_happy_path_uses_fallback_data_inicio_when_missing():
    quote = make_quote(primeiro_pagamento_pro_rata=None)
    quote_client = StubQuoteClient(result=quote)
    extractor = StubExtractor({"idade": 40, "veiculo_ano": 2019, "cep": "01000-000"})
    session = QualificationSession()
    agent = Agent(StubLlm(), quote_client, session, extractor=extractor)

    await agent.handle_turn("40 anos, Onix 2019, cep 01000-000")
    turn = await agent.handle_turn("confirmo")

    assert turn.handoff is None
    assert quote_client.calls[0]["data_inicio"] is not None
    assert "data de início" in turn.reply.lower()


# ---------------------------------------------------------------------------
# QuoteUnavailable -> handoff
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_quote_unavailable_triggers_handoff_without_fabricating_price():
    exc = QuoteUnavailable(
        "esgotou_tentativas:http_503", attempts=3, context={"payload": {}}
    )
    quote_client = StubQuoteClient(exc=exc)
    extractor = StubExtractor({"idade": 40, "veiculo_ano": 2015, "cep": "01000-000"})
    session = QualificationSession()
    agent = Agent(StubLlm(), quote_client, session, extractor=extractor)

    await agent.handle_turn("40 anos, Onix 2015, cep 01000-000")
    turn = await agent.handle_turn("confirmo, pode cotar")

    assert turn.handoff is not None
    assert turn.handoff.reason == HandoffReason.QUOTE_UNAVAILABLE
    assert turn.quote is None
    assert "R$" not in turn.reply  # nunca inventa preço
    assert agent.state.handoff is not None
    assert len(quote_client.calls) == 1

    # Turno seguinte já está no handoff -- não reprocessa nem re-cota.
    turn_after = await agent.handle_turn("oi?")
    assert turn_after.handoff == turn.handoff
    assert len(quote_client.calls) == 1


# ---------------------------------------------------------------------------
# CotacaoRecusada (422) -> explica, sem handoff, sem retry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cotacao_recusada_explains_refusal_without_handoff_or_retry():
    exc = CotacaoRecusada("Idade fora das faixas aceitas.")
    quote_client = StubQuoteClient(exc=exc)
    extractor = StubExtractor({"idade": 80, "veiculo_ano": 2015, "cep": "01000-000"})
    session = QualificationSession()
    agent = Agent(StubLlm(), quote_client, session, extractor=extractor)

    await agent.handle_turn("80 anos, Onix 2015, cep 01000-000")
    turn = await agent.handle_turn("confirmo")

    assert turn.handoff is None
    assert "Idade fora das faixas aceitas." in turn.reply
    assert len(quote_client.calls) == 1  # sem retry
    assert agent.state.closed is True


@pytest.mark.asyncio
async def test_payload_invalido_asks_again_without_handoff():
    exc = PayloadInvalido("idade obrigatória")
    quote_client = StubQuoteClient(exc=exc)
    extractor = StubExtractor({"idade": 35, "veiculo_ano": 2015, "cep": "01000-000"})
    session = QualificationSession()
    agent = Agent(StubLlm(), quote_client, session, extractor=extractor)

    await agent.handle_turn("35 anos, Onix 2015, cep 01000-000")
    turn = await agent.handle_turn("confirmo")

    assert turn.handoff is None
    assert "idade obrigatória" in turn.reply
    assert len(quote_client.calls) == 1


# ---------------------------------------------------------------------------
# Minimização (Q3): nunca pede CPF/e-mail/telefone/placa
# ---------------------------------------------------------------------------


def test_field_questions_never_reference_forbidden_pii():
    for question in FIELD_QUESTIONS.values():
        lowered = question.lower()
        assert not any(word in lowered for word in FORBIDDEN_WORDS)

    assert set(FIELD_QUESTIONS.keys()) == {"idade", "veiculo_ano", "cep"}


def test_missing_fields_question_never_asks_forbidden_data():
    question = build_missing_fields_question(
        ["idade", "veiculo_ano", "cep"], ask_data_inicio=True
    )

    lowered = question.lower()
    assert not any(word in lowered for word in FORBIDDEN_WORDS)


@pytest.mark.asyncio
async def test_agent_never_requests_forbidden_fields_in_generated_flow():
    extractor = StubExtractor({})
    session = QualificationSession()
    agent = Agent(StubLlm(), StubQuoteClient(), session, extractor=extractor)

    turn = await agent.handle_turn("oi, quero um seguro pro meu carro")

    lowered = turn.reply.lower()
    assert not any(word in lowered for word in FORBIDDEN_WORDS)


def test_system_prompt_instructs_minimization_and_essential_fields_only():
    lowered = SYSTEM_PROMPT.lower()
    assert "nunca peça cpf" in lowered
    assert "idade" in lowered
    assert "veículo" in lowered
    assert "cep" in lowered


# ---------------------------------------------------------------------------
# Handoff determinístico embutido no fluxo do agente
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_explicit_human_request_triggers_handoff_mid_conversation():
    session = QualificationSession()
    agent = Agent(StubLlm(), StubQuoteClient(), session, extractor=StubExtractor({}))

    turn = await agent.handle_turn("quero falar com um atendente, por favor")

    assert turn.handoff is not None
    assert turn.handoff.reason == HandoffReason.EXPLICIT_REQUEST


@pytest.mark.asyncio
async def test_essential_media_message_triggers_media_unreadable_handoff():
    session = QualificationSession()
    agent = Agent(StubLlm(), StubQuoteClient(), session, extractor=StubExtractor({}))

    turn = await agent.handle_turn("", media_type="audio")

    assert turn.handoff is not None
    assert turn.handoff.reason == HandoffReason.MEDIA_UNREADABLE


@pytest.mark.asyncio
async def test_clarify_loop_exhausted_after_max_attempts_without_essential_data():
    extractor = StubExtractor({})  # nunca extrai nada
    session = QualificationSession(max_attempts=2)
    agent = Agent(StubLlm(), StubQuoteClient(), session, extractor=extractor)

    turn1 = await agent.handle_turn("oi")
    assert turn1.handoff is None

    turn2 = await agent.handle_turn("não sei bem")
    assert turn2.handoff is not None
    assert turn2.handoff.reason == HandoffReason.CLARIFY_LOOP_EXHAUSTED


@pytest.mark.asyncio
async def test_agent_error_fallback_triggers_agent_error_handoff():
    class ExplodingQuoteClient:
        async def cotar(self, payload):
            raise RuntimeError("boom")

    extractor = StubExtractor({"idade": 35, "veiculo_ano": 2015, "cep": "01000-000"})
    session = QualificationSession()
    agent = Agent(StubLlm(), ExplodingQuoteClient(), session, extractor=extractor)

    await agent.handle_turn("35 anos, Onix 2015, cep 01000-000")
    turn = await agent.handle_turn("confirmo")

    assert turn.handoff is not None
    assert turn.handoff.reason == HandoffReason.AGENT_ERROR


# ---------------------------------------------------------------------------
# LLM mockado só na conversa livre pós-cotação (nunca decide preço/handoff)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_is_used_for_free_chat_after_quote_delivered():
    quote = make_quote()
    quote_client = StubQuoteClient(result=quote)
    extractor = StubExtractor({"idade": 35, "veiculo_ano": 2008, "cep": "26703-384"})
    llm = StubLlm(text="Imagina! Qualquer coisa é só chamar.")
    session = QualificationSession()
    agent = Agent(llm, quote_client, session, extractor=extractor)

    await agent.handle_turn("Corolla 2008, 35 anos, CEP 26703-384")
    await agent.handle_turn("confirmo")
    turn = await agent.handle_turn("muito obrigado!")

    assert turn.reply == "Imagina! Qualquer coisa é só chamar."
    assert len(llm.messages.calls) == 1
    assert len(quote_client.calls) == 1  # não recotou no papo livre
