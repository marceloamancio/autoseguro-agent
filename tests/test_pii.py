"""Testes do redator de PII at-rest (autoseguro.pii).

Cobertura (Group C / DEC-4, Q3):
- Recall: CPF, e-mail, telefone, placa e CEP nos formatos exatos produzidos pelo
  gerador do dataset (namastex-fde-challenge/scripts/generate_dataset.py) são
  mascarados. Também rodamos recall sobre uma amostra real do dataset sintético
  (parquet/sample.jsonl), pulando o teste se o repo irmão não estiver presente
  nesta máquina (o dataset não faz parte deste repo).
- Precisão: textos de controle sem PII (ou com números soltos que não são
  CPF/CEP) não são mascarados indevidamente.
- Varredura LLM em lote: mockada (nunca chama API real); desligada (no-op) sem
  client — roda sem ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoseguro.pii import (
    CEP_RE,
    CPF_RE,
    EMAIL_RE,
    MANDATORY_CATEGORIES,
    MARKERS,
    PLACA_RE,
    TELEFONE_RE,
    PiiRedactor,
    build_sweep_prompt,
    llm_sweep,
    redact_record,
    redact_text,
)

DATASET_PARQUET = Path(
    "/Users/marceloamancio/Desktop/nama_novo/namastex-fde-challenge/dataset/conversations.parquet"
)
DATASET_JSONL = Path(
    "/Users/marceloamancio/Desktop/nama_novo/namastex-fde-challenge/dataset/sample.jsonl"
)

KNOWN_PII_REGEXES = [CPF_RE, EMAIL_RE, TELEFONE_RE, PLACA_RE, CEP_RE]


def _no_known_pii_left(text: str) -> bool:
    return not any(rx.search(text) for rx in KNOWN_PII_REGEXES)


# ---------------------------------------------------------------------------
# Recall — formatos exatos do gerador do dataset (fonte da verdade)
# ---------------------------------------------------------------------------


class TestRecallFormatosConhecidos:
    def test_cpf_mascarado(self):
        text = "meu cpf e 389.083.863-43, pode confirmar?"
        out = redact_text(text)
        assert "389.083.863-43" not in out
        assert MARKERS["cpf"] in out
        assert _no_known_pii_left(out)

    def test_email_mascarado(self):
        text = "meu email é ursula.souza90@gmail.com pra contato"
        out = redact_text(text)
        assert "ursula.souza90@gmail.com" not in out
        assert MARKERS["email"] in out
        assert _no_known_pii_left(out)

    def test_email_com_underscore_e_sem_numero(self):
        text = "pode ser joao_silva@hotmail.com mesmo"
        out = redact_text(text)
        assert "joao_silva@hotmail.com" not in out
        assert MARKERS["email"] in out

    def test_telefone_mascarado(self):
        text = "o whats é esse mesmo +55 21 97224-2584 ok"
        out = redact_text(text)
        assert "+55 21 97224-2584" not in out
        assert MARKERS["telefone"] in out
        assert _no_known_pii_left(out)

    def test_placa_mascarada(self):
        text = "a placa é ABC1D23 se precisar"
        out = redact_text(text)
        assert "ABC1D23" not in out
        assert MARKERS["placa"] in out
        assert _no_known_pii_left(out)

    def test_cep_mascarado(self):
        text = "moro no cep 26703-384 mesmo"
        out = redact_text(text)
        assert "26703-384" not in out
        assert MARKERS["cep"] in out
        assert _no_known_pii_left(out)

    def test_mensagem_com_multiplas_pii_planta_do_gerador(self):
        # Formato real plantado pelo generate_dataset.py:
        # ", ".join(["cpf ...", "tenho X anos", "cep ..."]).capitalize()
        text = "Tenho 35 anos, cep 26703-384, cpf 389.083.863-43"
        out = redact_text(text)
        assert _no_known_pii_left(out)
        assert MARKERS["cep"] in out
        assert MARKERS["cpf"] in out
        # idade (35 anos) não é uma categoria coberta pelo regex simples —
        # deve permanecer (minimização trata isso na origem, Group E)
        assert "35 anos" in out


# ---------------------------------------------------------------------------
# Recall sobre amostra real do dataset sintético (skip se repo irmão ausente)
# ---------------------------------------------------------------------------


def _load_sample_bodies(n=60):
    if DATASET_JSONL.exists():
        bodies = []
        with DATASET_JSONL.open() as f:
            for line in f:
                bodies.append(json.loads(line)["message_body"])
        return bodies[:n]
    if DATASET_PARQUET.exists():
        pd = pytest.importorskip("pandas")
        df = pd.read_parquet(DATASET_PARQUET)
        return df["message_body"].astype(str).head(n).tolist()
    return None


@pytest.mark.skipif(
    _load_sample_bodies() is None,
    reason="dataset sintético (namastex-fde-challenge) não encontrado nesta máquina",
)
def test_recall_sobre_amostra_do_dataset():
    bodies = _load_sample_bodies(n=60)
    redacted = [redact_text(b) for b in bodies]
    leaked = [b for b in redacted if not _no_known_pii_left(b)]
    assert leaked == [], f"PII conhecida vazou após redação: {leaked}"


# ---------------------------------------------------------------------------
# Precisão — controles sem PII não devem ser mascarados
# ---------------------------------------------------------------------------


class TestPrecisao:
    @pytest.mark.parametrize(
        "text",
        [
            "quero seguro pro meu carro",
            "Toyota Corolla 2008",
            "tenho 35 anos",
            "e um Sandero 2022",
            "Show! Pelo perfil consigo o plano Premium por R$ 219,90/mes.",
            "qualquer coisa me chama",
            "nasci em 1989",
            "Ola! Vi o anuncio de voces, quanto fica o seguro?",
        ],
    )
    def test_texto_sem_pii_nao_e_alterado(self, text):
        assert redact_text(text) == text

    def test_numeros_soltos_nao_sao_cpf_nem_cep(self):
        text = "o numero de protocolo é 12345 e o pedido 6789-01"
        # "6789-01" não é CEP (precisa 5 dígitos + hífen + 3 dígitos)
        assert redact_text(text) == text


# ---------------------------------------------------------------------------
# redact_record — conveniência para registros de trace/log
# ---------------------------------------------------------------------------


def test_redact_record_mascara_campos_de_texto():
    record = {
        "conversation_id": "conv_00000",
        "sender_role": "lead",
        "message_body": "meu cpf e 389.083.863-43",
    }
    out = redact_record(record)

    assert out["message_body"] != record["message_body"]
    assert MARKERS["cpf"] in out["message_body"]
    assert out["conversation_id"] == "conv_00000"
    assert out["sender_role"] == "lead"
    # não muta o registro original
    assert record["message_body"] == "meu cpf e 389.083.863-43"


def test_redact_record_so_mascara_campos_configurados():
    record = {"message_body": "cpf 389.083.863-43", "sender_name": "Ana 389.083.863-43"}
    out = redact_record(record, text_fields=("message_body",))

    assert MARKERS["cpf"] in out["message_body"]
    assert out["sender_name"] == record["sender_name"]


# ---------------------------------------------------------------------------
# Varredura LLM em lote — mockável e desligável (Group C)
# ---------------------------------------------------------------------------


class TestLlmSweep:
    def test_sem_client_e_no_op_e_roda_sem_chave(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        texts = ["Fulano de Tal mora na Rua das Flores, 123", "sem pii nenhuma aqui"]

        out = llm_sweep(texts, client=None)

        assert out == texts

    def test_com_client_mockado_e_chamado_em_lote_uma_unica_vez(self):
        calls = []

        def fake_client(batch_texts, categories):
            calls.append((list(batch_texts), list(categories)))
            return [t.replace("Fulano de Tal", "⟨NOME_TERCEIRO⟩") for t in batch_texts]

        texts = ["Fulano de Tal ligou hoje", "outra mensagem qualquer"]
        out = llm_sweep(texts, client=fake_client)

        assert len(calls) == 1  # uma única chamada em lote, não uma por texto
        batch_arg, categories_arg = calls[0]
        assert batch_arg == texts
        for mandatory in MANDATORY_CATEGORIES:
            assert mandatory in categories_arg
        assert out[0] == "⟨NOME_TERCEIRO⟩ ligou hoje"
        assert out[1] == "outra mensagem qualquer"

    def test_pii_redactor_combina_regex_e_llm_sweep_em_lote(self):
        def fake_client(batch_texts, categories):
            return [t.replace("Fulano de Tal", "⟨NOME_TERCEIRO⟩") for t in batch_texts]

        redactor = PiiRedactor(llm_client=fake_client)
        texts = [
            "cpf 389.083.863-43, e Fulano de Tal confirma",
            "sem nenhuma pii aqui",
        ]
        out = redactor.redact_batch(texts)

        assert MARKERS["cpf"] in out[0]
        assert "⟨NOME_TERCEIRO⟩" in out[0]
        assert out[1] == "sem nenhuma pii aqui"

    def test_pii_redactor_sem_client_aplica_so_regex_e_e_no_op_pro_resto(self):
        redactor = PiiRedactor()  # sem llm_client -> desligado/no-op

        assert redactor.redact_text("389.083.863-43") == MARKERS["cpf"]
        assert redactor.redact_batch(["389.083.863-43"]) == [MARKERS["cpf"]]


def test_build_sweep_prompt_inclui_categorias_obrigatorias_e_pede_adicionais():
    prompt = build_sweep_prompt(["texto de exemplo com pii"])

    for category in MANDATORY_CATEGORIES:
        assert category in prompt
    assert "adicional" in prompt.lower()


# ---------------------------------------------------------------------------
# 2.4 (P1-1) — ampliação de recall: formatos plausíveis fora do gerador
# ---------------------------------------------------------------------------


class TestRecallVariantesPlausiveis:
    def test_cpf_cru_mascarado(self):
        text = "meu cpf e 12345678901, pode confirmar?"
        out = redact_text(text)
        assert "12345678901" not in out
        assert MARKERS["cpf"] in out

    def test_telefone_parenteses_mascarado(self):
        text = "pode me ligar no (11) 91234-5678 se precisar"
        out = redact_text(text)
        assert "91234-5678" not in out
        assert MARKERS["telefone"] in out

    def test_telefone_so_digitos_mascarado(self):
        text = "meu whats é 11912345678"
        out = redact_text(text)
        assert "11912345678" not in out
        assert MARKERS["telefone"] in out

    def test_telefone_fixo_mascarado(self):
        text = "pode ligar fixo também: (11) 3456-7890"
        out = redact_text(text)
        assert "3456-7890" not in out
        assert MARKERS["telefone"] in out

    def test_placa_antiga_mascarada(self):
        text = "a placa do carro é ABC1234, é um Gol antigo"
        out = redact_text(text)
        assert "ABC1234" not in out
        assert MARKERS["placa"] in out

    def test_cep_cru_mascarado(self):
        text = "meu cep é 01310100, moro perto da paulista"
        out = redact_text(text)
        assert "01310100" not in out
        assert MARKERS["cep"] in out

    def test_mensagem_com_todas_as_variantes_planta_do_red_team(self):
        text = (
            "cpf 12345678901, telefone (11) 91234-5678, placa ABC1234, "
            "cep 01310100, fone 11912345678"
        )
        out = redact_text(text)
        assert "12345678901" not in out
        assert "91234-5678" not in out
        assert "ABC1234" not in out
        assert "01310100" not in out
        assert "11912345678" not in out

    def test_nao_regride_precisao_existente(self):
        # 3.4 (viés de recall) não pode fazer idade/ano de veículo virarem PII.
        for text in ("tenho 35 anos", "Toyota Corolla 2008", "nasci em 1989"):
            assert redact_text(text) == text


# ---------------------------------------------------------------------------
# 2.4 (P1-1) — AnthropicSweepClient: adaptador real (client mockado, sem rede)
# ---------------------------------------------------------------------------


class _FakeSweepMessages:
    def __init__(self, response):
        self._response = response
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


class _FakeSweepClient:
    def __init__(self, response):
        self.messages = _FakeSweepMessages(response)


def _sweep_text_response(text: str):
    from types import SimpleNamespace

    block = SimpleNamespace(type="text", text=text)
    return SimpleNamespace(content=[block])


class TestAnthropicSweepClient:
    def test_chama_api_uma_unica_vez_em_lote_e_realinha_resposta_numerada(self):
        from autoseguro.pii import AnthropicSweepClient

        resp = _sweep_text_response("1. primeiro texto mascarado\n2. segundo texto mascarado")
        fake_client = _FakeSweepClient(resp)
        sweep = AnthropicSweepClient("sk-ant-fake", "claude-sonnet-5", client=fake_client)

        out = sweep(["texto 1 com pii", "texto 2 sem pii"], MANDATORY_CATEGORIES)

        assert len(fake_client.messages.calls) == 1
        assert out == ["primeiro texto mascarado", "segundo texto mascarado"]

    def test_sem_textos_e_no_op_sem_chamar_a_api(self):
        from autoseguro.pii import AnthropicSweepClient

        fake_client = _FakeSweepClient(_sweep_text_response(""))
        sweep = AnthropicSweepClient("sk-ant-fake", "m", client=fake_client)

        assert sweep([], MANDATORY_CATEGORIES) == []
        assert fake_client.messages.calls == []

    def test_resposta_fora_do_formato_numerado_devolve_textos_originais(self):
        from autoseguro.pii import AnthropicSweepClient

        resp = _sweep_text_response("resposta livre sem numeração")
        fake_client = _FakeSweepClient(resp)
        sweep = AnthropicSweepClient("sk-ant-fake", "m", client=fake_client)

        texts = ["a", "b"]
        assert sweep(texts, MANDATORY_CATEGORIES) == texts

    def test_pluga_direto_no_pii_redactor(self):
        from autoseguro.pii import AnthropicSweepClient

        resp = _sweep_text_response("1. ⟨NOME_TERCEIRO⟩ confirma o cpf ⟨CPF⟩")
        fake_client = _FakeSweepClient(resp)
        sweep = AnthropicSweepClient("sk-ant-fake", "m", client=fake_client)

        redactor = PiiRedactor(llm_client=sweep)
        out = redactor.redact_batch(["Fulano de Tal confirma o cpf 389.083.863-43"])

        assert out == ["⟨NOME_TERCEIRO⟩ confirma o cpf ⟨CPF⟩"]
