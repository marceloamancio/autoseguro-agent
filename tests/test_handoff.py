"""Testes do motor de handoff (autoseguro.handoff) — Group E, DEC-8 (Q6).

Cobre a tabela de gatilhos do Q6:
- Determinísticos: `/quote` esgotada, erro inesperado, mídia essencial,
  fechamento/emissão, pedido explícito de humano, loop de esclarecimento
  esgotado.
- Fuzzy (classificador injetável/mockável): fora de escopo, dados
  contraditórios, conflito/reclamação — sempre via `classify_fuzzy`.
- "Não transbordam": recusa 422, plano fora do catálogo, objeção de preço —
  nunca geram handoff (funções `resolve_*` sempre `None`).

Nenhum teste chama LLM ou HTTP real.
"""

from __future__ import annotations

import pytest

from autoseguro.extraction import ExtractedData, QualificationSession
from autoseguro.handoff import (
    HandoffDecision,
    HandoffReason,
    classify_fuzzy,
    for_agent_error,
    for_clarify_loop_exhausted,
    for_complaint_conflict,
    for_contradictory_data,
    for_explicit_request,
    for_media_unreadable,
    for_out_of_scope,
    for_policy_issuance,
    for_quote_unavailable,
    is_media_essential,
    resolve_plan_not_in_catalog,
    resolve_price_objection,
    resolve_quote_refusal,
)
from autoseguro.quote_client import QuoteUnavailable


class StubFuzzyClassifier:
    """Dublê do classificador fuzzy — sem chamada LLM real."""

    def __init__(self, reason: HandoffReason | None):
        self._reason = reason
        self.calls: list[str] = []

    def classify(self, text: str) -> HandoffReason | None:
        self.calls.append(text)
        return self._reason


# ---------------------------------------------------------------------------
# Determinísticos
# ---------------------------------------------------------------------------


def test_quote_unavailable_triggers_quote_unavailable_reason():
    exc = QuoteUnavailable(
        "esgotou_tentativas:http_503", attempts=3, context={"payload": {"idade": 35}}
    )

    decision = for_quote_unavailable(exc)

    assert isinstance(decision, HandoffDecision)
    assert decision.reason == HandoffReason.QUOTE_UNAVAILABLE
    assert decision.context["attempts"] == 3
    assert decision.context["quote_reason"] == "esgotou_tentativas:http_503"
    assert "R$" not in decision.message  # nunca inventa preço


def test_agent_error_triggers_agent_error_reason():
    decision = for_agent_error(ValueError("algo quebrou de forma inesperada"))

    assert decision.reason == HandoffReason.AGENT_ERROR
    assert "algo quebrou" in decision.context["error"]


@pytest.mark.parametrize("media_type", ["image", "audio", "document"])
def test_media_essential_types_trigger_media_unreadable(media_type):
    assert is_media_essential(media_type) is True

    decision = for_media_unreadable(media_type)

    assert decision.reason == HandoffReason.MEDIA_UNREADABLE
    assert decision.context["media_type"] == media_type


def test_text_message_is_not_media_essential():
    assert is_media_essential(None) is False
    assert is_media_essential("text") is False


@pytest.mark.parametrize(
    "text",
    [
        "quero emitir a apólice agora",
        "como eu pago o boleto?",
        "quero contratar esse plano",
        "qual a forma de pagamento?",
    ],
)
def test_policy_issuance_keywords_trigger_handoff(text):
    decision = for_policy_issuance(text)

    assert decision is not None
    assert decision.reason == HandoffReason.POLICY_ISSUANCE


def test_regular_quote_question_does_not_trigger_policy_issuance():
    assert for_policy_issuance("quanto fica o seguro do meu carro?") is None


@pytest.mark.parametrize(
    "text",
    [
        "quero falar com um atendente",
        "posso falar com uma pessoa de verdade?",
        "chama o gerente por favor",
    ],
)
def test_explicit_human_request_triggers_handoff(text):
    decision = for_explicit_request(text)

    assert decision is not None
    assert decision.reason == HandoffReason.EXPLICIT_REQUEST


def test_regular_message_does_not_trigger_explicit_request():
    assert for_explicit_request("quero um seguro pro meu carro") is None


def test_clarify_loop_exhausted_after_max_attempts_without_essential_data():
    session = QualificationSession(max_attempts=2)
    session.attempts = 2
    session.data = ExtractedData(idade=None, veiculo_ano=2010, cep=None)

    decision = for_clarify_loop_exhausted(session)

    assert decision is not None
    assert decision.reason == HandoffReason.CLARIFY_LOOP_EXHAUSTED
    assert "idade" in decision.context["missing"]
    assert "cep" in decision.context["missing"]


def test_clarify_loop_not_exhausted_when_data_complete():
    session = QualificationSession(max_attempts=2)
    session.attempts = 5
    session.data = ExtractedData(idade=35, veiculo_ano=2010, cep="01000-000")

    assert for_clarify_loop_exhausted(session) is None


def test_clarify_loop_not_exhausted_before_max_attempts():
    session = QualificationSession(max_attempts=2)
    session.attempts = 1
    session.data = ExtractedData(idade=None, veiculo_ano=None, cep=None)

    assert for_clarify_loop_exhausted(session) is None


# ---------------------------------------------------------------------------
# Fuzzy (classificador injetável/mockável)
# ---------------------------------------------------------------------------


def test_classify_fuzzy_out_of_scope():
    classifier = StubFuzzyClassifier(HandoffReason.OUT_OF_SCOPE)

    decision = classify_fuzzy("quero fazer um sinistro do meu seguro residencial", classifier)

    assert decision is not None
    assert decision.reason == HandoffReason.OUT_OF_SCOPE
    assert classifier.calls == ["quero fazer um sinistro do meu seguro residencial"]


def test_classify_fuzzy_contradictory_data():
    classifier = StubFuzzyClassifier(HandoffReason.CONTRADICTORY_DATA)

    decision = classify_fuzzy("meu carro é de 2020 mas falei 2010 antes", classifier)

    assert decision is not None
    assert decision.reason == HandoffReason.CONTRADICTORY_DATA


def test_classify_fuzzy_complaint_conflict():
    classifier = StubFuzzyClassifier(HandoffReason.COMPLAINT_CONFLICT)

    decision = classify_fuzzy("isso é um absurdo, vou processar vocês", classifier)

    assert decision is not None
    assert decision.reason == HandoffReason.COMPLAINT_CONFLICT


def test_classify_fuzzy_returns_none_when_classifier_finds_nothing():
    classifier = StubFuzzyClassifier(None)

    assert classify_fuzzy("oi, tudo bem?", classifier) is None


def test_classify_fuzzy_returns_none_when_no_classifier_injected():
    assert classify_fuzzy("qualquer coisa", None) is None


def test_classify_fuzzy_rejects_reason_outside_fuzzy_set():
    classifier = StubFuzzyClassifier(HandoffReason.QUOTE_UNAVAILABLE)

    with pytest.raises(ValueError):
        classify_fuzzy("texto qualquer", classifier)


def test_for_out_of_scope_and_for_complaint_conflict_build_decisions_directly():
    out_of_scope = for_out_of_scope("quero cancelar minha apólice")
    complaint = for_complaint_conflict("que atendimento horrível")
    contradictory = for_contradictory_data("primeiro disse 2010, agora diz 2020")

    assert out_of_scope.reason == HandoffReason.OUT_OF_SCOPE
    assert complaint.reason == HandoffReason.COMPLAINT_CONFLICT
    assert contradictory.reason == HandoffReason.CONTRADICTORY_DATA


# ---------------------------------------------------------------------------
# "Não transbordam" — o agente resolve sozinho, nunca handoff
# ---------------------------------------------------------------------------


def test_quote_refusal_never_generates_handoff():
    assert resolve_quote_refusal("Idade fora das faixas aceitas.") is None


def test_plan_not_in_catalog_never_generates_handoff():
    assert resolve_plan_not_in_catalog("super_plano") is None


def test_price_objection_never_generates_handoff():
    assert resolve_price_objection("tem desconto? tá caro demais") is None
