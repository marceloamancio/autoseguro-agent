"""Testes de extração/qualificação de dados (autoseguro.extraction) — Group D, DEC-5.

Cobre:
- Extração via LLM mockado (structured outputs) de fixtures de texto livre do
  desafio ("e um Sandero 2022", "Toyota Corolla, ano 2008", "tenho 35 anos, cep
  26703-384").
- Normalização: ano com 2 ou 4 dígitos, CEP com/sem hífen, "nasci em AAAA" → idade
  coerente com o ano corrente.
- Backstop regex leve: com o LLM mockado falhando/retornando vazio, o regex ainda
  extrai ano (19xx/20xx) e CEP diretamente do texto.
- Dado essencial (idade + veiculo_ano + cep): função que aponta o que falta e sinal
  de handoff após N=2 tentativas (parametrizável).
- Validação de faixas (idade 0–200, veiculo_ano 1950–2100): fora da faixa é inválido.

Nenhum teste chama a API real da Anthropic — o "cliente LLM" é sempre um dublê
(stub) local que implementa `.extract(text) -> dict`.
"""

from __future__ import annotations

from datetime import date

import pytest

from autoseguro.extraction import (
    ExtractedData,
    QualificationSession,
    extract_once,
    normalize_ano,
    normalize_cep,
    normalize_idade,
)


class StubLlmClient:
    """Dublê do cliente LLM de structured outputs — sem chamada de rede."""

    def __init__(self, response: dict | None = None, raises: bool = False):
        self._response = response or {}
        self._raises = raises
        self.calls: list[str] = []

    def extract(self, text: str) -> dict:
        self.calls.append(text)
        if self._raises:
            raise RuntimeError("LLM indisponível (simulado)")
        return self._response


# ---------------------------------------------------------------------------
# Extração via LLM mockado (structured outputs) — fixtures do desafio
# ---------------------------------------------------------------------------


def test_extracts_veiculo_ano_from_sandero_fixture():
    llm = StubLlmClient({"veiculo_ano": 2022, "modelo": "Sandero"})

    result = extract_once("e um Sandero 2022", llm_client=llm)

    assert result.data.veiculo_ano == 2022
    assert result.data.modelo == "Sandero"
    assert result.llm_used is True
    assert llm.calls == ["e um Sandero 2022"]


def test_extracts_veiculo_ano_marca_modelo_from_corolla_fixture():
    llm = StubLlmClient({"veiculo_ano": 2008, "marca": "Toyota", "modelo": "Corolla"})

    result = extract_once("Toyota Corolla, ano 2008", llm_client=llm)

    assert result.data.veiculo_ano == 2008
    assert result.data.marca == "Toyota"
    assert result.data.modelo == "Corolla"


def test_extracts_idade_e_cep_from_fixture():
    llm = StubLlmClient({"idade": 35, "cep": "26703-384"})

    result = extract_once("tenho 35 anos, cep 26703-384", llm_client=llm)

    assert result.data.idade == 35
    assert result.data.cep == "26703-384"


# ---------------------------------------------------------------------------
# Normalização
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        (2022, 2022),
        ("2022", 2022),
        (8, 2008),
        ("08", 2008),
        (97, 1997),
        ("97", 1997),
    ],
)
def test_normalize_ano_aceita_2_ou_4_digitos(raw, expected):
    assert normalize_ano(raw) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("26703384", "26703-384"),
        ("26703-384", "26703-384"),
        ("04125060", "04125-060"),
    ],
)
def test_normalize_cep_com_ou_sem_hifen(raw, expected):
    assert normalize_cep(raw) == expected


def test_normalize_cep_invalido_retorna_none():
    assert normalize_cep("abc") is None
    assert normalize_cep("123") is None


def test_normalize_idade_aceita_inteiro_ou_string_numerica():
    assert normalize_idade(35) == 35
    assert normalize_idade("35") == 35


def test_normalize_idade_a_partir_de_nasci_em_ano():
    ano_atual = date.today().year
    esperado = ano_atual - 1989

    assert normalize_idade("nasci em 1989") == esperado


def test_extract_once_normaliza_nasci_em_do_texto_quando_llm_devolve_frase():
    # O LLM pode devolver o campo idade como a frase capturada, cabe à
    # normalização converter para um inteiro coerente com o ano corrente.
    llm = StubLlmClient({"idade": "nasci em 1989", "veiculo_ano": 2010, "cep": "01310-000"})

    result = extract_once("Nasci em 1989, o carro é de 2010", llm_client=llm)

    assert result.data.idade == date.today().year - 1989


# ---------------------------------------------------------------------------
# Backstop regex leve (ano + CEP), usado quando o LLM falha ou volta vazio
# ---------------------------------------------------------------------------


def test_backstop_regex_extrai_ano_quando_llm_falha():
    llm = StubLlmClient(raises=True)

    result = extract_once("Tenho um Onix 2015 aqui, cep 04125-060", llm_client=llm)

    assert result.llm_used is False
    assert result.data.veiculo_ano == 2015
    assert result.data.cep == "04125-060"


def test_backstop_regex_extrai_ano_e_cep_quando_llm_devolve_vazio():
    llm = StubLlmClient({})

    result = extract_once("Meu carro é 2019, moro no cep 26703384", llm_client=llm)

    assert result.data.veiculo_ano == 2019
    assert result.data.cep == "26703-384"


def test_backstop_regex_funciona_sem_llm_client_nenhum():
    result = extract_once("Carro 2013, cep 30130-000", llm_client=None)

    assert result.llm_used is False
    assert result.data.veiculo_ano == 2013
    assert result.data.cep == "30130-000"


def test_backstop_nao_inventa_dado_ausente_no_texto():
    result = extract_once("oi, tudo bem?", llm_client=None)

    assert result.data.veiculo_ano is None
    assert result.data.cep is None
    assert result.data.idade is None


# ---------------------------------------------------------------------------
# Validação de faixas — fora da faixa é inválido (não vira dado essencial)
# ---------------------------------------------------------------------------


def test_veiculo_ano_fora_da_faixa_e_invalido():
    llm = StubLlmClient({"veiculo_ano": 1800, "idade": 30, "cep": "26703-384"})

    result = extract_once("carro de 1800, tenho 30 anos, cep 26703-384", llm_client=llm)

    assert result.data.veiculo_ano is None
    assert any("veiculo_ano" in w for w in result.warnings)
    assert "veiculo_ano" in result.data.essential_missing()


def test_idade_fora_da_faixa_e_invalida():
    llm = StubLlmClient({"veiculo_ano": 2020, "idade": 250, "cep": "26703-384"})

    result = extract_once("carro 2020, 250 anos, cep 26703-384", llm_client=llm)

    assert result.data.idade is None
    assert any("idade" in w for w in result.warnings)


def test_faixas_validas_no_limite_sao_aceitas():
    assert normalize_ano(1950) == 1950
    assert normalize_ano(2100) == 2100
    assert normalize_idade(0) == 0
    assert normalize_idade(200) == 200


# ---------------------------------------------------------------------------
# Dado essencial (idade + veiculo_ano + cep) e handoff após N=2 tentativas
# ---------------------------------------------------------------------------


def test_essential_missing_aponta_campos_faltantes():
    data = ExtractedData(veiculo_ano=2020, idade=None, cep=None)

    assert data.essential_missing() == ["idade", "cep"]
    assert data.has_essential() is False


def test_essential_completo_quando_idade_veiculo_ano_e_cep_presentes():
    data = ExtractedData(veiculo_ano=2020, idade=35, cep="26703-384")

    assert data.essential_missing() == []
    assert data.has_essential() is True


def test_qualification_session_nao_sinaliza_handoff_antes_de_n_tentativas():
    session = QualificationSession(max_attempts=2)

    session.process_turn("oi, quero um seguro", llm_client=None)

    assert session.attempts == 1
    assert session.needs_handoff() is False


def test_qualification_session_sinaliza_handoff_apos_n_tentativas_sem_essencial():
    session = QualificationSession(max_attempts=2)

    session.process_turn("oi, quero um seguro", llm_client=None)
    session.process_turn("nao sei bem os dados", llm_client=None)

    assert session.attempts == 2
    assert session.needs_handoff() is True
    assert session.is_complete() is False


def test_qualification_session_completa_antes_do_limite_nao_precisa_handoff():
    # Idade não é coberta pelo backstop regex (por design — só ano/CEP são
    # "rede de segurança" leve); chega via LLM, como no fluxo real.
    llm = StubLlmClient({"idade": 35})
    session = QualificationSession(max_attempts=2)

    session.process_turn(
        "tenho 35 anos, carro 2015, cep 26703-384", llm_client=llm
    )

    assert session.is_complete() is True
    assert session.needs_handoff() is False


def test_qualification_session_acumula_dados_entre_turnos():
    session = QualificationSession(max_attempts=2)
    llm_idade = StubLlmClient({"idade": 35})

    session.process_turn("meu carro é de 2015", llm_client=None)
    session.process_turn("tenho 35 anos, cep 26703-384", llm_client=llm_idade)

    assert session.data.veiculo_ano == 2015
    assert session.data.idade == 35
    assert session.data.cep == "26703-384"
    assert session.is_complete() is True


# ---------------------------------------------------------------------------
# 2.1 — `intent` fundido na extração (zero chamada extra de LLM)
# ---------------------------------------------------------------------------


def test_extract_once_carrega_intent_do_llm():
    llm = StubLlmClient({"idade": 40, "intent": "correct"})

    result = extract_once("na verdade tenho 40 anos", llm_client=llm)

    assert result.data.intent == "correct"


def test_extract_once_default_intent_quando_llm_nao_devolve_o_campo():
    # Extração antiga (sem `intent`) ou LLM que não retornou o campo -- nunca
    # deve quebrar; cai num default neutro.
    llm = StubLlmClient({"idade": 35})

    result = extract_once("tenho 35 anos", llm_client=llm)

    assert result.data.intent == "other"


def test_extract_once_default_intent_sem_llm_client_nenhum():
    result = extract_once("carro 2013, cep 30130-000", llm_client=None)

    assert result.data.intent == "other"


def test_extract_once_normaliza_intent_desconhecido_para_other():
    llm = StubLlmClient({"intent": "algo_nao_mapeado"})

    result = extract_once("oi", llm_client=llm)

    assert result.data.intent == "other"


def test_qualification_session_process_turn_expoe_intent_do_turno_atual():
    session = QualificationSession(max_attempts=2)
    llm = StubLlmClient({"idade": 40, "intent": "correct"})

    result = session.process_turn("na verdade tenho 40 anos", llm_client=llm)

    assert result.data.intent == "correct"


def test_qualification_session_max_attempts_e_parametrizavel():
    session = QualificationSession(max_attempts=3)

    session.process_turn("oi", llm_client=None)
    session.process_turn("nao lembro", llm_client=None)

    assert session.needs_handoff() is False  # ainda não bateu N=3

    session.process_turn("desculpa, nao consigo passar os dados", llm_client=None)

    assert session.attempts == 3
    assert session.needs_handoff() is True
