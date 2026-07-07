"""CLI de chat turn-based — Group F, DEC-9 (Q4).

Design (ver `DECISOES.md`, Q4 e `WISH.md`, Group F): a interface de entrega
obrigatória é uma **CLI de terminal** sobre um **core assíncrono** — sem
porta exposta (superfície mínima), chave via env (fail-fast, `load_config`),
turno-a-turno (lê mensagem do lead, chama `Agent.handle_turn`, imprime a
resposta), gerando o `trace.jsonl` (Group F/`tracing.py`) que serve de "log
de execução completa" (ids + status por mensagem/cotação).

- **Fail-fast de chave:** `main()` chama `load_config()` antes de montar
  qualquer coisa; sem `ANTHROPIC_API_KEY`, aborta com a mensagem de
  `ConfigError` (nunca imprime o valor da chave) e código de saída != 0.
- **Ack imediato + nudge:** ao ler uma mensagem que provavelmente vai
  disparar uma chamada de cotação (`agent.state.awaiting_confirmation` já
  True, ou seja, o lead está confirmando o resumo antes de cotar), a CLI
  imprime `ACK_MESSAGE` de cara; se a chamada demorar mais que
  `nudge_after_s`, imprime `NUDGE_MESSAGE` (pode repetir enquanto durar).
- **Deps injetáveis:** `run_repl` recebe um `Agent` e um `Tracer` já
  montados (com `llm`/`quote_client` mockáveis em teste); em produção,
  `build_agent_from_config` monta a partir de `Config` (SDK real da
  Anthropic + `QuoteClient` real).
- **Log entregue/curado (2.4, P1-1):** `trace.jsonl` (hot path, por-evento)
  só passa pelo regex simples (`Tracer`/`PiiRedactor.redact_record`) — nunca
  a varredura LLM, que é cara/lenta demais pra rodar por mensagem. O log de
  execução **entregue** (`cure_delivered_log`) é uma etapa **em lote**, fora
  do hot-path, que roda depois que a conversa encerra: pega os eventos já
  mascarados pelo regex e passa **todos os textos de uma vez** por
  `PiiRedactor.redact_batch` (regex de novo, idempotente, + a varredura LLM
  em lote, via `AnthropicSweepClient` em produção). `main()` grava o
  resultado em `DELIVERED_LOG_FILENAME`, ao lado de `trace.jsonl`.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

from .agent import Agent
from .config import Config, ConfigError, load_config
from .extraction import LlmExtractorClient, QualificationSession
from .handoff import FuzzyClassifier, HandoffReason
from .pii import LlmSweepClient, PiiRedactor
from .quote_client import QuoteClient
from .tracing import DEFAULT_TEXT_FIELDS, Tracer

ACK_MESSAGE = "Deixa eu calcular sua cotação, só um instante..."
NUDGE_MESSAGE = "Ainda estou calculando sua cotação, só mais um instante..."
NUDGE_AFTER_S = 5.0

EXIT_WORDS: tuple[str, ...] = ("sair", "exit", "quit")

# 2.4 (P1-1): nome do arquivo do log de execução entregue/curado (em lote,
# fora do hot-path) -- irmão de `trace.jsonl`, mesmo `DEFAULT_LOG_DIR`.
DELIVERED_LOG_FILENAME = "delivered.jsonl"

InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]

# 2.3 (P1-3): marcador líder de mídia -- `[documento]/[imagem]/[audio]/
# [foto]/[video] <arquivo>`. O dataset do desafio só traz o marcador (sem
# transcrição), então a CLI precisa mapear pro `media_type` que
# `handoff.is_media_essential`/`for_media_unreadable` já sabem tratar --
# antes disso, o marcador era lido como texto de qualificação comum.
_MEDIA_MARKER_RE = re.compile(
    r"^\[(documento|imagem|audio|áudio|foto|video|vídeo)\]\s*(.*)$", re.IGNORECASE
)
_MEDIA_MARKER_TO_TYPE: dict[str, str] = {
    "documento": "document",
    "imagem": "image",
    "foto": "image",
    "audio": "audio",
    "áudio": "audio",
    "video": "video",
    "vídeo": "video",
}


def parse_media_marker(user_msg: str) -> tuple[str, str | None]:
    """Detecta o marcador líder de mídia e devolve `(texto, media_type)`.

    Sem marcador, devolve o texto original e `media_type=None` (mensagem de
    texto comum). Reusa `handoff.is_media_essential` pra decidir se o
    `media_type` resultante é essencial (imagem/áudio/documento) -- este
    parser só normaliza o marcador, nunca decide handoff sozinho.
    """
    match = _MEDIA_MARKER_RE.match(user_msg.strip())
    if not match:
        return user_msg, None
    marker, rest = match.groups()
    media_type = _MEDIA_MARKER_TO_TYPE.get(marker.lower())
    return rest.strip(), media_type


def build_agent_from_config(
    config: Config,
    *,
    extractor: LlmExtractorClient | None = None,
    fuzzy_classifier: FuzzyClassifier | None = None,
) -> Agent:
    """Monta o `Agent` de produção a partir da `Config` (chave/URLs reais).

    Import do SDK da Anthropic é feito aqui dentro (não no topo do módulo)
    para que os testes de `cli.py` nunca precisem da dependência real
    instanciada — só usado no caminho de produção (`main()`).
    """
    import anthropic

    llm = anthropic.AsyncAnthropic(api_key=config.anthropic_api_key)
    if extractor is None:
        # Adapter real de extração (tool-use estrito) — sem ele, a qualificação
        # nunca captura os dados e todo lead acaba em handoff. Só no caminho de
        # produção; os testes injetam um extractor mockado.
        from .anthropic_extractor import AnthropicExtractor

        extractor = AnthropicExtractor(config.anthropic_api_key, config.anthropic_model)
    if fuzzy_classifier is None:
        # Classificador fuzzy determinístico (fora de escopo / reclamação) — sem
        # ele, esses reason codes de handoff nunca disparam na CLI real.
        from .handoff import KeywordFuzzyClassifier

        fuzzy_classifier = KeywordFuzzyClassifier()
    quote_client = QuoteClient(config)
    session = QualificationSession()
    return Agent(
        llm,
        quote_client,
        session,
        extractor=extractor,
        fuzzy_classifier=fuzzy_classifier,
        model=config.anthropic_model,
    )


def cure_delivered_log(
    events: Sequence[dict[str, Any]],
    *,
    llm_client: LlmSweepClient | None = None,
    text_fields: Sequence[str] = DEFAULT_TEXT_FIELDS,
) -> list[dict[str, Any]]:
    """Cura o log de execução entregue: varredura LLM em **lote**, fora do
    hot-path (2.4, P1-1).

    Recebe os eventos já emitidos pelo `Tracer` (mascarados pelo regex
    simples, camada 1, por-evento) e passa **todos os textos configurados de
    uma vez** por `PiiRedactor.redact_batch` -- uma única chamada em lote
    (`llm_client` é o `AnthropicSweepClient` em produção; um dublê/`None` em
    teste). Nunca muta `events`; devolve cópias com os campos de texto
    substituídos pelo resultado da varredura.

    Sem `llm_client`, `redact_batch` reduz-se ao regex (idempotente sobre
    texto já mascarado) -- ou seja, é seguro chamar mesmo quando a varredura
    LLM está desligada.
    """
    positions: list[tuple[int, str]] = []
    texts: list[str] = []
    for i, event in enumerate(events):
        for field_name in text_fields:
            value = event.get(field_name)
            if isinstance(value, str):
                positions.append((i, field_name))
                texts.append(value)

    redactor = PiiRedactor(llm_client=llm_client)
    swept = redactor.redact_batch(texts)

    cured = [dict(event) for event in events]
    for (i, field_name), new_value in zip(positions, swept):
        cured[i][field_name] = new_value
    return cured


def write_jsonl(path: Path, events: Sequence[dict[str, Any]]) -> None:
    """Grava `events` como JSON Lines em `path` (mesmo formato de `trace.jsonl`)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


async def _run_turn_with_ack_and_nudge(
    agent: Agent,
    user_msg: str,
    *,
    nudge_after_s: float,
    output_fn: OutputFn,
    media_type: str | None = None,
) -> Any:
    """Roda `agent.handle_turn`, emitindo ack imediato e nudges se demorar.

    Ack imediato: se o turno provavelmente vai disparar uma chamada de
    cotação (`agent.state.awaiting_confirmation` já estava True antes deste
    turno — ou seja, o lead está respondendo à confirmação), avisa antes de
    chamar `handle_turn`. Nudge: enquanto o turno não terminar, reimprime
    `NUDGE_MESSAGE` a cada `nudge_after_s`.
    """
    if agent.state.awaiting_confirmation:
        output_fn(ACK_MESSAGE)

    task = asyncio.ensure_future(agent.handle_turn(user_msg, media_type=media_type))
    while True:
        try:
            return await asyncio.wait_for(asyncio.shield(task), timeout=nudge_after_s)
        except asyncio.TimeoutError:
            output_fn(NUDGE_MESSAGE)


def _log_quote_outcome(tracer: Tracer, turn: Any) -> None:
    """Emite `quote.result` quando o turno corresponde a um desfecho de cotação.

    Cobre os 3 status possíveis sem precisar que o `Agent` exponha a exceção
    original: `success` (turn.quote presente), `unavailable` (handoff com
    `QUOTE_UNAVAILABLE`, carrega `attempts`/`quote_reason` do contexto) e
    `recusado` (a única situação em que o agente encerra a conversa —
    `turn.closed` — sem entregar cotação nem gerar handoff, ver
    `agent.format_refusal`).
    """
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


async def run_repl(
    agent: Agent,
    tracer: Tracer,
    *,
    input_fn: InputFn = input,
    output_fn: OutputFn = print,
    nudge_after_s: float = NUDGE_AFTER_S,
    exit_words: tuple[str, ...] = EXIT_WORDS,
) -> None:
    """REPL turn-based: lê mensagem, chama `Agent.handle_turn`, traça e imprime.

    `input_fn`/`output_fn` são injetáveis para teste (dublês simples que não
    tocam stdin/stdout real). Cada turno real (não uma palavra de saída)
    gera eventos `message.in`/`message.out` e, quando aplicável,
    `quote.result`/`handoff`/`decision` no `tracer`.
    """
    output_fn("Agente AutoSeguro — digite sua mensagem (ou 'sair' para encerrar).")

    while True:
        try:
            user_msg = input_fn("Você: ")
        except EOFError:
            break

        if user_msg.strip().lower() in exit_words:
            break

        tracer.message_in(user_msg)

        text, media_type = parse_media_marker(user_msg)

        turn = await _run_turn_with_ack_and_nudge(
            agent,
            text,
            nudge_after_s=nudge_after_s,
            output_fn=output_fn,
            media_type=media_type,
        )

        tracer.message_out(turn.reply)
        _log_quote_outcome(tracer, turn)

        if turn.handoff is not None:
            tracer.handoff(reason_code=turn.handoff.reason.value)
            tracer.decision(status="handoff")
        else:
            tracer.decision(status="resolved")

        output_fn(f"Agente: {turn.reply}")


def main() -> None:
    """Ponto de entrada da CLI de produção — fail-fast de chave, sem porta exposta."""
    try:
        config = load_config()
    except ConfigError as exc:
        print(f"Erro de configuração: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    agent = build_agent_from_config(config)
    tracer = Tracer()
    try:
        asyncio.run(run_repl(agent, tracer))
    finally:
        tracer.close()

    # 2.4 (P1-1): cura o log de execução entregue em lote, fora do hot-path --
    # a varredura LLM real (`AnthropicSweepClient`) só roda aqui, nunca por
    # mensagem durante a conversa (import lazy, mesmo padrão de
    # `build_agent_from_config`/`AnthropicExtractor`).
    from .pii import AnthropicSweepClient

    sweep_client = AnthropicSweepClient(config.anthropic_api_key, config.anthropic_model)
    delivered = cure_delivered_log(tracer.events, llm_client=sweep_client)
    write_jsonl(tracer.path.parent / DELIVERED_LOG_FILENAME, delivered)


if __name__ == "__main__":
    main()
