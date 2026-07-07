"""Replay harness (**opcional**) sobre o dataset de conversas — Group G.

Não é um entregável do desafio (ver `DECISOES.md`, Q4: "replay sobre o
dataset: engenharia **opcional** (robustez/testes/adversarial), não é
entregável"). Serve para rodar o `Agent` (Group E) sobre as mensagens do
`lead` de `dataset/conversations.parquet` e produzir um resultado por
conversa — útil para achar buracos de extração/handoff em lote, antes de
gravar os logs de execução reais que o desafio pede.

Como o resto do projeto, o LLM (extractor) e o `QuoteClient` que o `Agent`
usa são **injetáveis/mockáveis**: este módulo nunca importa `anthropic` no
topo nem chama rede sozinho. Quem monta a execução real decide os clients —
em produção seria algo próximo de `cli.build_agent_from_config` (SDK real da
Anthropic + `QuoteClient` real apontando pro `quote-service` do desafio);
nos testes (`tests/test_replay.py`) só entram dublês simples, sem chave nem
rede.

Concorrência é opcional (`concurrency=1` por padrão, sequencial) — em lote
sobre as ~2.500 conversas do dataset, o circuit breaker do `QuoteClient`
(Q5) evita martelar um `/quote` real que esteja instável, mesmo com várias
conversas em paralelo.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .agent import Agent
from .handoff import HandoffReason
from .tracing import Tracer

# Caminho default relativo ao workspace (`namastex-fde-challenge` é irmão de
# `autoseguro-agent` no ambiente de desenvolvimento). Não existe dentro deste
# repo — quem rodar o replay de verdade passa o caminho local do dataset.
DEFAULT_DATASET_PATH = Path("../namastex-fde-challenge/dataset/conversations.parquet")

_MEDIA_TYPES = ("image", "audio", "document")

# Uma fábrica que monta um `Agent` novo e isolado para uma conversa (o
# `Agent` cuida de uma conversa por vez — ver docstring de `agent.Agent`).
AgentFactory = Callable[[str], Agent]

# Uma fábrica opcional que monta um `Tracer` novo por conversa (ex.: um
# `trace.jsonl` por `conversation_id` num diretório de replay).
TracerFactory = Callable[[str], Tracer]


@dataclass
class ConversationReplayResult:
    """Resultado de rodar o `Agent` sobre uma conversa do dataset."""

    conversation_id: str
    turns: int = 0
    resolved: bool = False
    quote: Any | None = None
    handoff_reason: str | None = None
    error: str | None = None
    trace_events: list[dict[str, Any]] = field(default_factory=list)


def load_conversations(path: Path | str = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    """Carrega o `conversations.parquet` e devolve ordenado por conversa/turno.

    Ordena por `conversation_id` e depois `message_index` (ver
    `DICIONARIO.md`: "linha = mensagem"; conversas se reconstroem agrupando
    por `conversation_id`, ordenando por `message_index`).
    """
    df = pd.read_parquet(path)
    return df.sort_values(["conversation_id", "message_index"]).reset_index(drop=True)


def lead_messages(df: pd.DataFrame, conversation_id: str) -> list[tuple[str, str | None]]:
    """Extrai as mensagens do `lead` de uma conversa, em ordem.

    Devolve pares `(message_body, media_type)`; `media_type` é `None` para
    `text` e o próprio tipo (`image`/`audio`/`document`) para mídia — o
    mesmo parâmetro que `Agent.handle_turn(media_type=...)` espera para
    disparar o handoff `MEDIA_UNREADABLE` (Q6) quando a mídia é essencial.
    """
    convo = df[
        (df["conversation_id"] == conversation_id) & (df["sender_role"] == "lead")
    ].sort_values("message_index")

    out: list[tuple[str, str | None]] = []
    for _, row in convo.iterrows():
        msg_type = row.get("message_type") or "text"
        media_type = msg_type if msg_type in _MEDIA_TYPES else None
        out.append((row["message_body"], media_type))
    return out


def _log_turn_outcome(tracer: Tracer, turn: Any) -> None:
    """Espelha `cli._log_quote_outcome` para o modo replay (mesmos 3 status)."""
    if turn.quote is not None:
        tracer.quote_result(
            status="success",
            plano_id=turn.quote.plano_id,
            premio_mensal=turn.quote.premio_mensal,
        )
    elif turn.handoff is not None and turn.handoff.reason == HandoffReason.QUOTE_UNAVAILABLE:
        tracer.quote_result(
            status="unavailable",
            attempts=turn.handoff.context.get("attempts"),
            reason=turn.handoff.context.get("quote_reason"),
        )
    elif turn.closed and turn.quote is None:
        tracer.quote_result(status="recusado")


async def replay_conversation(
    conversation_id: str,
    messages: list[tuple[str, str | None]],
    agent: Agent,
    *,
    tracer: Tracer | None = None,
) -> ConversationReplayResult:
    """Roda um `Agent` já montado sobre as mensagens de `lead` de uma conversa.

    Processa turno a turno via `Agent.handle_turn`, replicando o mesmo
    registro de trace que a CLI faz (`message.in`/`message.out`/
    `quote.result`/`handoff`/`decision`) quando `tracer` é passado. Para no
    primeiro handoff ou fechamento (`turn.closed`) — o resto das mensagens do
    dataset seria pós-decisão e não interessa ao replay. Qualquer exceção
    não tratada pelo `Agent` (não deveria acontecer — ver o fail-safe de
    `agent._quote_and_reply`) é capturada aqui para não derrubar o lote
    inteiro por causa de uma conversa.
    """
    result = ConversationReplayResult(conversation_id=conversation_id)

    for text, media_type in messages:
        if tracer is not None:
            tracer.message_in(text)

        try:
            turn = await agent.handle_turn(text, media_type=media_type)
        except Exception as exc:  # noqa: BLE001 - isola a falha, não derruba o lote
            result.error = repr(exc)
            break

        result.turns += 1
        if tracer is not None:
            tracer.message_out(turn.reply)
            _log_turn_outcome(tracer, turn)

        if turn.handoff is not None:
            result.handoff_reason = turn.handoff.reason.value
            if tracer is not None:
                tracer.handoff(reason_code=turn.handoff.reason.value)
                tracer.decision(status="handoff")
            break

        if tracer is not None:
            tracer.decision(status="resolved")

        if turn.quote is not None:
            result.quote = turn.quote

        if turn.closed:
            result.resolved = True
            break

    result.resolved = result.resolved or result.quote is not None
    if tracer is not None:
        result.trace_events = list(tracer.events)
    return result


async def replay_dataset(
    df: pd.DataFrame,
    agent_factory: AgentFactory,
    *,
    conversation_ids: list[str] | None = None,
    concurrency: int = 1,
    tracer_factory: TracerFactory | None = None,
) -> list[ConversationReplayResult]:
    """Roda o replay sobre várias conversas, com concorrência **opcional**.

    - `agent_factory(conversation_id)`: monta um `Agent` novo e isolado por
      conversa (estado de qualificação não vaza entre conversas).
    - `conversation_ids`: subconjunto a rodar; default é todas as conversas
      do `DataFrame` (na ordem em que aparecem).
    - `concurrency`: nº máximo de conversas em paralelo (default `1` =
      sequencial). Limitado por um `asyncio.Semaphore` — útil pra rodar o
      dataset inteiro sem martelar um `/quote` real além do que o circuit
      breaker (Q5) já limitaria por conversa individual.
    - `tracer_factory`: opcional, monta um `Tracer` por conversa (ex.: um
      arquivo `logs/replay/<conversation_id>.jsonl`); sem ele, o replay roda
      sem gravar trace em disco (só devolve `ConversationReplayResult`).
    """
    ids = conversation_ids if conversation_ids is not None else list(
        dict.fromkeys(df["conversation_id"])
    )
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _run_one(conversation_id: str) -> ConversationReplayResult:
        async with semaphore:
            messages = lead_messages(df, conversation_id)
            agent = agent_factory(conversation_id)
            tracer = tracer_factory(conversation_id) if tracer_factory is not None else None
            try:
                return await replay_conversation(
                    conversation_id, messages, agent, tracer=tracer
                )
            finally:
                if tracer is not None:
                    tracer.close()

    return await asyncio.gather(*(_run_one(cid) for cid in ids))
