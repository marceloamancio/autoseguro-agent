"""Rastreabilidade (`trace.jsonl`) — Group F, DEC-9 (Q7).

Design (ver `DECISOES.md`, Q7 e `WISH.md`, Group F): `trace.jsonl` é o
artefato canônico que atende "dá pra rastrear o que aconteceu (cada mensagem/
cotação, com id e status)" — um evento por linha JSON, append-only.

Ids:
- `run_id`: uma execução (processo da CLI).
- `conversation_id`: uma conversa (uma instância de `Agent`).
- `event_id`: cada evento individual (uma mensagem, um resultado de cotação,
  uma decisão, um handoff) — sempre único.
- `quote_request_id`: cada cotação (permite linkar tentativas de uma mesma
  chamada quando o chamador quiser correlacionar).

Tipos de evento: `message.in` / `message.out` (status: recebida/enviada),
`quote.result` (status: `success` / `recusado` / `unavailable`, com
`attempts` quando disponível — ver `QuoteUnavailable.attempts`), `decision`
(status: `resolved` / `handoff`), `handoff` (com `reason_code`).

Mascaramento (Q3): **todo** evento passa por `PiiRedactor.redact_record`
antes de ser gravado — nunca grava PII em claro em disco. A implementação é
`logging` stdlib + um formatter JSON (`_JsonLineFormatter`), sem dependência
extra.

Os logs de execução **entregues** (happy-path e falha→handoff) não saem
daqui: são produzidos em lote por `cli.cure_delivered_log` a partir de
`Tracer.events`, porque a varredura LLM de PII (camada 2) roda fora do
hot-path — ver `cli.py`.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .pii import PiiRedactor

DEFAULT_LOG_DIR = Path("logs")
DEFAULT_TRACE_FILENAME = "trace.jsonl"

# Campos de texto livre que podem carregar PII espontânea do lead — mascarados
# em todo evento antes de gravar (Q3).
DEFAULT_TEXT_FIELDS: tuple[str, ...] = ("message_body", "message", "reason")


class _JsonLineFormatter(logging.Formatter):
    """Formatter stdlib: serializa `record.msg` (um dict já mascarado) como JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload = record.msg
        if not isinstance(payload, dict):
            payload = {"message": str(payload)}
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def new_id() -> str:
    """Gera um id novo (hex de um UUID4) — usado para `event_id`/`run_id`/etc."""
    return uuid.uuid4().hex


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_file_logger(name: str, path: Path) -> logging.Logger:
    """Monta um `Logger` stdlib dedicado, com um único `FileHandler` JSON.

    Cada `Tracer` usa um nome de logger próprio (baseado em `id(self)`) para
    nunca compartilhar handlers com outra instância — evita duplicar linhas
    quando vários `Tracer`s/testes rodam no mesmo processo.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(_JsonLineFormatter())
    logger.addHandler(file_handler)
    return logger


@dataclass
class Tracer:
    """Emissor de eventos `trace.jsonl` — append-only, mascarado (Q3).

    Uso típico (CLI, Group F): uma instância por conversa, injetada com
    `run_id`/`conversation_id` estáveis; cada turno chama `message_in`,
    `message_out`, e (quando aplicável) `quote_result`, `decision`/`handoff`.
    """

    run_id: str = field(default_factory=new_id)
    conversation_id: str = field(default_factory=new_id)
    path: Path = field(default_factory=lambda: DEFAULT_LOG_DIR / DEFAULT_TRACE_FILENAME)
    redactor: PiiRedactor = field(default_factory=PiiRedactor)
    text_fields: tuple[str, ...] = DEFAULT_TEXT_FIELDS

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        base_name = f"autoseguro.trace.{id(self)}"
        self._logger = _build_file_logger(base_name, self.path)
        self.events: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Núcleo: monta ids comuns, mascara e grava
    # ------------------------------------------------------------------

    def _emit(self, event: dict[str, Any]) -> dict[str, Any]:
        base: dict[str, Any] = {
            "ts": _now_iso(),
            "run_id": self.run_id,
            "conversation_id": self.conversation_id,
            "event_id": new_id(),
        }
        base.update(event)
        masked = self.redactor.redact_record(base, text_fields=self.text_fields)

        self.events.append(masked)
        self._logger.info(masked)
        return masked

    # ------------------------------------------------------------------
    # Tipos de evento
    # ------------------------------------------------------------------

    def message_in(self, text: str, **extra: Any) -> dict[str, Any]:
        """Mensagem recebida do lead — sempre carrega `event_id` próprio."""
        return self._emit(
            {"type": "message.in", "status": "recebida", "message_body": text, **extra}
        )

    def message_out(self, text: str, **extra: Any) -> dict[str, Any]:
        """Mensagem enviada ao lead — sempre carrega `event_id` próprio."""
        return self._emit(
            {"type": "message.out", "status": "enviada", "message_body": text, **extra}
        )

    def quote_result(
        self,
        *,
        status: str,
        quote_request_id: str | None = None,
        attempts: int | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        """Resultado de uma cotação: `status` ∈ {success, recusado, unavailable}.

        `quote_request_id` identifica a cotação (gerado automaticamente se
        não fornecido); `attempts` é repassado quando disponível (ex.: de
        `QuoteUnavailable.attempts`).
        """
        event: dict[str, Any] = {
            "type": "quote.result",
            "status": status,
            "quote_request_id": quote_request_id or new_id(),
        }
        if attempts is not None:
            event["attempts"] = attempts
        event.update(extra)
        return self._emit(event)

    def decision(self, *, status: str, **extra: Any) -> dict[str, Any]:
        """Decisão do turno: `status` ∈ {resolved, handoff}."""
        return self._emit({"type": "decision", "status": status, **extra})

    def handoff(self, *, reason_code: str, **extra: Any) -> dict[str, Any]:
        """Transbordo pro humano — carrega o `reason_code` auditável (Q6)."""
        return self._emit(
            {"type": "handoff", "status": "handoff", "reason_code": reason_code, **extra}
        )

    # ------------------------------------------------------------------

    def close(self) -> None:
        """Fecha o `FileHandler` — libera o arquivo (útil em testes)."""
        for handler in list(self._logger.handlers):
            handler.close()
            self._logger.removeHandler(handler)
