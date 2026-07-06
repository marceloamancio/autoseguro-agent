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
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any, Callable

from .agent import Agent
from .config import Config, ConfigError, load_config
from .extraction import LlmExtractorClient, QualificationSession
from .handoff import FuzzyClassifier, HandoffReason
from .quote_client import QuoteClient
from .tracing import Tracer

ACK_MESSAGE = "Deixa eu calcular sua cotação, só um instante..."
NUDGE_MESSAGE = "Ainda estou calculando sua cotação, só mais um instante..."
NUDGE_AFTER_S = 5.0

EXIT_WORDS: tuple[str, ...] = ("sair", "exit", "quit")

InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]


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


async def _run_turn_with_ack_and_nudge(
    agent: Agent,
    user_msg: str,
    *,
    nudge_after_s: float,
    output_fn: OutputFn,
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

    task = asyncio.ensure_future(agent.handle_turn(user_msg))
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

        turn = await _run_turn_with_ack_and_nudge(
            agent, user_msg, nudge_after_s=nudge_after_s, output_fn=output_fn
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


if __name__ == "__main__":
    main()
