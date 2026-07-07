"""Testes do replay harness opcional (autoseguro.replay) — Group G.

Roda o `Agent` sobre 1-2 conversas com **LLM stub** e `QuoteClient` stub —
sem chave, sem rede, sem depender do dataset real do desafio (que vive fora
deste repo, em `namastex-fde-challenge/`). As conversas de teste são
sintéticas, mas seguem exatamente o esquema documentado em
`dataset/DICIONARIO.md` (mesmas colunas), para que `replay.load_conversations`/
`replay.lead_messages` operem sobre um `DataFrame` real.

Cobre:
- Caminho feliz: extração completa + confirmação + `/quote` 200 -> resultado
  com cotação, sem handoff.
- Handoff: mensagem de mídia essencial (áudio, sem transcrição) -> handoff
  `media_unreadable` no primeiro turno.
- `replay_dataset` sobre as duas conversas, com `tracer_factory` gravando um
  `trace.jsonl` por conversa em `tmp_path` — verifica que cada conversa
  produz eventos de trace.
- Se o dataset real do desafio estiver acessível no workspace, um teste
  adicional (marcado `skipif`) roda o replay sobre 2 conversas de verdade —
  não é obrigatório, só uma checagem oportunista de integração.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from autoseguro.agent import Agent
from autoseguro.extraction import QualificationSession
from autoseguro.quote_client import QuoteResult
from autoseguro.replay import (
    DEFAULT_DATASET_PATH,
    lead_messages,
    load_conversations,
    replay_conversation,
    replay_dataset,
)
from autoseguro.tracing import Tracer

_COLUMNS = [
    "conversation_id",
    "message_index",
    "timestamp",
    "sender_role",
    "sender_name",
    "message_type",
    "message_body",
    "channel",
    "conversation_outcome",
    "lead_idade_informada",
    "veiculo_texto",
]

_ROWS = [
    # conv_test_001: caminho feliz — extração completa num turno + confirmação.
    (
        "conv_test_001", 0, "2026-01-01T10:00:00", "lead", "Ana", "text",
        "Tenho 35 anos, Corolla 2008, CEP 01310-100", "whatsapp", "ganho", 35,
        "Corolla 2008",
    ),
    (
        "conv_test_001", 1, "2026-01-01T10:01:00", "vendedor", "Vendas", "text",
        "Só confirmando antes de cotar: confere?", "whatsapp", "ganho", 35,
        "Corolla 2008",
    ),
    (
        "conv_test_001", 2, "2026-01-01T10:02:00", "lead", "Ana", "text",
        "sim, confirmo", "whatsapp", "ganho", 35, "Corolla 2008",
    ),
    # conv_test_002: mídia essencial (áudio, sem transcrição) -> handoff imediato.
    (
        "conv_test_002", 0, "2026-01-01T11:00:00", "lead", "Bruno", "audio",
        "[áudio] 0:12", "whatsapp", "perdido", None, None,
    ),
]


@pytest.fixture
def conversations_df() -> pd.DataFrame:
    return pd.DataFrame(_ROWS, columns=_COLUMNS)


class StubExtractor:
    """Dublê síncrono do `LlmExtractorClient` — sempre devolve os mesmos dados."""

    def __init__(self, response: dict):
        self._response = response

    def extract(self, text: str) -> dict:
        return self._response


class StubQuoteClient:
    """Dublê assíncrono do `QuoteClient` — nunca bate em rede real."""

    def __init__(self, result: QuoteResult):
        self._result = result
        self.calls: list[dict] = []

    async def cotar(self, payload: dict) -> QuoteResult:
        self.calls.append(payload)
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
        multiplicadores={"idade": 1.0, "veiculo": 1.0},
        primeiro_pagamento_pro_rata=None,
    )
    defaults.update(overrides)
    return QuoteResult(**defaults)


def _agent_factory(conversation_id: str) -> Agent:
    """Monta um `Agent` novo e isolado por conversa (LLM/quote client stub)."""
    if conversation_id == "conv_test_001":
        extractor = StubExtractor({"idade": 35, "veiculo_ano": 2008, "cep": "01310-100"})
        quote_client = StubQuoteClient(make_quote())
    else:
        extractor = None
        quote_client = StubQuoteClient(make_quote())
    return Agent(
        StubLlm(),
        quote_client,
        QualificationSession(),
        extractor=extractor,
    )


def test_lead_messages_extracts_only_lead_rows_in_order(conversations_df: pd.DataFrame) -> None:
    messages = lead_messages(conversations_df, "conv_test_001")
    assert messages == [
        ("Tenho 35 anos, Corolla 2008, CEP 01310-100", None),
        ("sim, confirmo", None),
    ]


def test_lead_messages_maps_media_type_for_unreadable_media(
    conversations_df: pd.DataFrame,
) -> None:
    messages = lead_messages(conversations_df, "conv_test_002")
    assert messages == [("[áudio] 0:12", "audio")]


@pytest.mark.asyncio
async def test_replay_conversation_happy_path_produces_quote(
    conversations_df: pd.DataFrame,
) -> None:
    messages = lead_messages(conversations_df, "conv_test_001")
    agent = _agent_factory("conv_test_001")

    result = await replay_conversation("conv_test_001", messages, agent)

    assert result.conversation_id == "conv_test_001"
    assert result.turns == 2
    assert result.resolved is True
    assert result.handoff_reason is None
    assert result.quote is not None
    assert result.quote.plano_id == "completo"


@pytest.mark.asyncio
async def test_replay_conversation_media_triggers_handoff(
    conversations_df: pd.DataFrame,
) -> None:
    messages = lead_messages(conversations_df, "conv_test_002")
    agent = _agent_factory("conv_test_002")

    result = await replay_conversation("conv_test_002", messages, agent)

    assert result.turns == 1
    assert result.handoff_reason == "media_unreadable"
    assert result.quote is None


@pytest.mark.asyncio
async def test_replay_dataset_runs_over_two_conversations_with_trace(
    conversations_df: pd.DataFrame, tmp_path: Path
) -> None:
    def tracer_factory(conversation_id: str) -> Tracer:
        return Tracer(
            conversation_id=conversation_id,
            path=tmp_path / f"{conversation_id}.jsonl",
        )

    results = await replay_dataset(
        conversations_df,
        _agent_factory,
        concurrency=2,
        tracer_factory=tracer_factory,
    )

    assert len(results) == 2
    by_id = {r.conversation_id: r for r in results}

    happy = by_id["conv_test_001"]
    assert happy.resolved is True
    assert happy.quote is not None
    assert happy.trace_events, "trace deveria ter eventos gravados"

    handoff = by_id["conv_test_002"]
    assert handoff.handoff_reason == "media_unreadable"
    assert handoff.trace_events, "trace deveria ter eventos gravados"

    for conversation_id in by_id:
        trace_file = tmp_path / f"{conversation_id}.jsonl"
        assert trace_file.exists()
        assert trace_file.read_text(encoding="utf-8").strip() != ""


_REAL_DATASET_PATH = (
    Path(__file__).resolve().parents[2]
    / "namastex-fde-challenge"
    / "dataset"
    / "conversations.parquet"
)


@pytest.mark.skipif(
    not _REAL_DATASET_PATH.exists(),
    reason="dataset real do desafio não está disponível neste workspace",
)
@pytest.mark.asyncio
async def test_replay_over_real_dataset_sample_is_opportunistic() -> None:
    """Checagem oportunista: só roda se o dataset do desafio estiver ao lado do repo.

    Não é exigido pelo Group G (o dataset fica fora deste repo público) — só
    aproveita o ambiente de desenvolvimento quando disponível, sem quebrar o
    `uv run pytest` de quem só tem `autoseguro-agent/`.
    """
    df = load_conversations(_REAL_DATASET_PATH)
    sample_ids = list(df["conversation_id"].unique()[:2])

    def extractor_factory(_: str) -> StubExtractor:
        return StubExtractor({})

    def agent_factory(conversation_id: str) -> Agent:
        return Agent(
            StubLlm(),
            StubQuoteClient(make_quote()),
            QualificationSession(),
            extractor=extractor_factory(conversation_id),
        )

    results = await replay_dataset(df, agent_factory, conversation_ids=sample_ids)

    assert len(results) == 2
    assert {r.conversation_id for r in results} == set(sample_ids)
    assert DEFAULT_DATASET_PATH  # só para documentar o default sem usá-lo aqui
