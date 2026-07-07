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


# ---------------------------------------------------------------------------
# P1-2 — substrings largas demais ("processo", "cobrança"/"cobranca") davam
# falso-positivo em perguntas legítimas; endurecidas para frases específicas.
# ---------------------------------------------------------------------------


def test_pergunta_sobre_processo_de_contratacao_nao_e_reclamacao():
    c = KeywordFuzzyClassifier()
    assert c.classify("qual o processo pra contratar?") is None


def test_pergunta_sobre_cobranca_mensal_nao_e_fora_de_escopo():
    c = KeywordFuzzyClassifier()
    assert c.classify("como funciona a cobrança mensal?") is None
    assert c.classify("como funciona a cobranca mensal?") is None


def test_abrir_processo_de_verdade_ainda_e_reclamacao():
    c = KeywordFuzzyClassifier()
    assert c.classify("quero abrir um processo contra vocês") == HandoffReason.COMPLAINT_CONFLICT
    assert c.classify("vou abrir processo na justiça") == HandoffReason.COMPLAINT_CONFLICT


def test_cobranca_indevida_de_verdade_ainda_e_fora_de_escopo():
    c = KeywordFuzzyClassifier()
    assert c.classify("tive uma cobrança indevida no meu cartão") == HandoffReason.OUT_OF_SCOPE
