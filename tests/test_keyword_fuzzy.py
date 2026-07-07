"""Testes do KeywordFuzzyClassifier (classificador fuzzy determinístico)."""

from autoseguro.handoff import (
    HandoffReason,
    KeywordFuzzyClassifier,
    classify_fuzzy,
)


def test_fora_de_escopo():
    c = KeywordFuzzyClassifier()
    assert c.classify("quero registrar um sinistro") == HandoffReason.OUT_OF_SCOPE
    assert c.classify("preciso da segunda via do boleto") == HandoffReason.OUT_OF_SCOPE
    assert c.classify("quero cancelar meu seguro") == HandoffReason.OUT_OF_SCOPE
    assert c.classify("vocês têm seguro residencial?") == HandoffReason.OUT_OF_SCOPE


def test_reclamacao_conflito():
    c = KeywordFuzzyClassifier()
    assert c.classify("isso é um absurdo, vou no procon") == HandoffReason.COMPLAINT_CONFLICT
    assert c.classify("péssimo atendimento, vou processar vocês") == HandoffReason.COMPLAINT_CONFLICT


def test_no_topico_retorna_none():
    c = KeywordFuzzyClassifier()
    assert c.classify("quero um seguro pro meu Corolla 2008") is None
    assert c.classify("tenho 35 anos, meu CEP é 01310-100") is None


def test_reclamacao_tem_prioridade_sobre_fora_de_escopo():
    c = KeywordFuzzyClassifier()
    # "cancelar" (fora de escopo) + "absurdo" (reclamação) → conflito
    assert c.classify("quero cancelar, isso é um absurdo") == HandoffReason.COMPLAINT_CONFLICT


def test_integra_com_classify_fuzzy():
    # Amarra o classificador ao pipeline de handoff (monta a HandoffDecision).
    decision = classify_fuzzy("quero abrir um sinistro", KeywordFuzzyClassifier())
    assert decision is not None
    assert decision.reason == HandoffReason.OUT_OF_SCOPE
