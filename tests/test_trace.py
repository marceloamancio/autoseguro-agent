"""Testes do emissor de trace (autoseguro.tracing) — Group F, DEC-9 (Q7).

Cobertura:
- `trace.jsonl` é append-only: cada evento carrega `run_id`/`conversation_id`/
  `event_id`; mensagens (`message.in`/`message.out`) têm `event_id` próprio e
  cotações (`quote.result`) têm `quote_request_id` + `status`.
- PII (ex.: CPF, e-mail) que aparecer no corpo de uma mensagem sai mascarada
  no trace gravado em disco — nunca em claro (Q3, via `PiiRedactor`).
- `decision` (resolved/handoff) e `handoff` (com `reason_code`) são graváveis.
"""

from __future__ import annotations

import json
from pathlib import Path

from autoseguro.tracing import Tracer


def _read_events(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_trace_jsonl_has_event_id_per_message_and_quote_id_status_per_quote(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    tracer = Tracer(path=trace_path, run_id="run-1", conversation_id="conv-1")

    tracer.message_in("35 anos, Onix 2019, cep 01000-000")
    tracer.message_out("Só confirmando: você tem 35 anos, veículo Onix 2019...")
    tracer.quote_result(status="success", quote_request_id="quote-1", attempts=1)
    tracer.decision(status="resolved")
    tracer.close()

    events = _read_events(trace_path)
    assert len(events) == 4

    message_events = [e for e in events if e["type"] in ("message.in", "message.out")]
    assert len(message_events) == 2
    assert all("event_id" in e and e["event_id"] for e in message_events)
    # event_id são únicos por evento (nenhuma colisão entre os 4 eventos)
    assert len({e["event_id"] for e in events}) == len(events)

    quote_events = [e for e in events if e["type"] == "quote.result"]
    assert len(quote_events) == 1
    assert quote_events[0]["quote_request_id"] == "quote-1"
    assert quote_events[0]["status"] == "success"
    assert quote_events[0]["attempts"] == 1

    # ids de execução/conversa constantes em todo evento do run
    assert all(e["run_id"] == "run-1" for e in events)
    assert all(e["conversation_id"] == "conv-1" for e in events)

    decision_events = [e for e in events if e["type"] == "decision"]
    assert decision_events[0]["status"] == "resolved"


def test_quote_request_id_is_auto_generated_when_not_provided(tmp_path):
    tracer = Tracer(path=tmp_path / "trace.jsonl")

    event = tracer.quote_result(status="unavailable", attempts=3)
    tracer.close()

    assert event["quote_request_id"]
    assert event["status"] == "unavailable"
    assert event["attempts"] == 3


def test_handoff_event_carries_reason_code(tmp_path):
    tracer = Tracer(path=tmp_path / "trace.jsonl")

    event = tracer.handoff(reason_code="quote_unavailable")
    tracer.decision(status="handoff")
    tracer.close()

    events = _read_events(tracer.path)
    handoff_events = [e for e in events if e["type"] == "handoff"]
    assert handoff_events[0]["reason_code"] == "quote_unavailable"
    assert event["reason_code"] == "quote_unavailable"

    decision_events = [e for e in events if e["type"] == "decision"]
    assert decision_events[0]["status"] == "handoff"


def test_trace_masks_pii_in_message_body_before_writing_to_disk(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    tracer = Tracer(path=trace_path)

    raw = "meu cpf é 123.456.789-00 e meu e-mail é lead@example.com"
    tracer.message_in(raw)
    tracer.close()

    events = _read_events(trace_path)
    assert len(events) == 1
    body = events[0]["message_body"]
    assert "123.456.789-00" not in body
    assert "lead@example.com" not in body
    assert "⟨CPF⟩" in body
    assert "⟨EMAIL⟩" in body

    # o arquivo em disco (bytes crus) também não deve conter o CPF em claro
    raw_disk = trace_path.read_text(encoding="utf-8")
    assert "123.456.789-00" not in raw_disk
    assert "lead@example.com" not in raw_disk

