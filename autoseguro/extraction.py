"""Extração e qualificação de dados de texto livre (Group D, DEC-5 / Q3b).

Contexto (ver DECISOES.md, Q3b e `WISH.md`, Group D): o lead descreve veículo,
idade e CEP em texto livre e desorganizado ("e um Sandero 2022", "tenho 35 anos,
cep 26703-384"). A `/quote` só usa o **ano** do veículo para cotar — marca/modelo
servem só pra rapport, nunca entram no preço.

Estratégia (DEC-5): a **extração é 100% do LLM** com structured outputs (cliente
injetado/mockável — este módulo nunca chama a API real). Nenhum regex garimpa o
texto livre do lead: essa era a origem de bugs (ano pescado de dentro de um
telefone, PII virando "dado"). O que fica aqui é só **normalização de formato do
valor que o LLM devolveu** (ano 2→4 dígitos, CEP → `XXXXX-XXX`, data → ISO) e
**validação de faixas** iguais às da `/quote` (`idade` 0–200, `veiculo_ano`
1950–2100). A defesa contra alucinação do LLM é a **confirmação explícita com o
lead antes de cotar** (Group E), não um segundo extrator.

**Dado essencial** para cotar = `idade + veiculo_ano + cep`. A confirmação com o
lead antes de cotar é responsabilidade do agente (Group E) — aqui só entra a
extração/normalização/validação e o sinal de "preciso de handoff" quando, após
`N=2` tentativas (parametrizável via `QualificationSession.max_attempts`), ainda
falta dado essencial.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# Faixas válidas — espelham exatamente o schema da /quote
# (`namastex-fde-challenge/quote-service/app/main.py`, `QuoteRequest`).
IDADE_MIN, IDADE_MAX = 0, 200
VEICULO_ANO_MIN, VEICULO_ANO_MAX = 1950, 2100

# Após quantas tentativas de qualificação sem dado essencial completo o agente
# deve sinalizar handoff (DEC-5 / Q3b: "N=2"). Parametrizável por caller.
DEFAULT_MAX_ATTEMPTS = 2

_ESSENTIAL_FIELDS = ("idade", "veiculo_ano", "cep")

# Intenções reconhecidas do turno (2.1 -- fundidas no EXTRACT_TOOL,
# `anthropic_extractor.py`). "other" é o default neutro/seguro quando o
# `llm_client` não devolve o campo (extração antiga) ou devolve um valor
# fora do enum -- nunca deve derrubar a extração dos demais campos.
_VALID_INTENTS = frozenset(
    {
        "confirm",
        "correct",
        "reject",
        "requote",
        "out_of_scope",
        "complaint",
        "explicit_human",
        "provide_data",
        "other",
    }
)
_DEFAULT_INTENT = "other"

# Pivô pra desambiguar ano de 2 dígitos que o LLM devolva (ex.: "08" -> 2008,
# "97" -> 1997). yy <= _PIVOT_2_DIGIT_ANO -> 20xx, senão -> 19xx. Não garimpa
# texto: só normaliza o inteiro que o LLM já extraiu.
_PIVOT_2_DIGIT_ANO = 68

# Regex mantidos são só de FORMATAÇÃO do valor que o LLM devolveu (nunca
# garimpam a mensagem crua do lead): CEP com hífen já ok, dígitos de CEP, e
# data dd/mm/aaaa -> ISO.
_REGEX_CEP_COM_HIFEN = re.compile(r"\b\d{5}-\d{3}\b")
_REGEX_DIGITS_ONLY = re.compile(r"\D+")
_REGEX_DATA_BR = re.compile(r"^(\d{2})/(\d{2})/(\d{4})$")

# 2.2 (P1-4): limiares conservadores pra distinguir correção pequena/plausível
# (nunca contradição) de dado materialmente diferente (contradição) --
# coordena com 1.1 (35 -> 40 é correção; 35 -> 90 é contradição).
_IDADE_CONTRADICTION_DELTA = 15
_VEICULO_ANO_CONTRADICTION_DELTA = 10


class LlmExtractorClient(Protocol):
    """Interface mínima do cliente de extração via structured outputs.

    Implementações reais envolvem `AsyncAnthropic`/`Anthropic` com tool/schema
    estrito; nos testes, um dublê simples que devolve um dict já basta — este
    módulo nunca instancia nem chama a API da Anthropic.
    """

    def extract(self, text: str) -> dict[str, Any]:
        ...


@dataclass
class ExtractedData:
    """Dados extraídos e normalizados de uma (ou mais) mensagens do lead."""

    veiculo_ano: int | None = None
    idade: int | None = None
    cep: str | None = None
    marca: str | None = None
    modelo: str | None = None
    data_inicio: str | None = None
    intent: str = _DEFAULT_INTENT

    def essential_missing(self) -> list[str]:
        """Lista os campos essenciais (`idade`, `veiculo_ano`, `cep`) ausentes."""
        return [f for f in _ESSENTIAL_FIELDS if getattr(self, f) is None]

    def has_essential(self) -> bool:
        """True quando `idade`, `veiculo_ano` e `cep` já foram capturados."""
        return not self.essential_missing()


@dataclass
class ExtractionResult:
    """Resultado de uma extração pontual (uma mensagem/turno)."""

    data: ExtractedData
    llm_used: bool
    llm_raw: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    # O extractor existia mas LEVANTOU (LLM fora do ar, sem crédito, 5xx) --
    # distinto de "sem client" e de "o lead não informou nada". Sem esse
    # sinal, uma queda da API vira `clarify_loop_exhausted`, culpando o lead
    # por uma falha de infra nossa (ver `agent._handle_turn`).
    extractor_failed: bool = False
    extractor_error: Exception | None = None
    # 2.2 (P1-4): só é preenchido por `QualificationSession.process_turn`
    # (compara contra o dado essencial já acumulado na sessão) -- sempre
    # `False` num `extract_once` isolado, que não tem histórico pra comparar.
    contradiction: bool = False


def normalize_ano(raw: int | str | None) -> int | None:
    """Normaliza um ano de veículo em 2 ou 4 dígitos para 4 dígitos.

    Aceita `int` ou `str` numérica. Anos de 2 dígitos são resolvidos por um
    pivô fixo (`_PIVOT_2_DIGIT_ANO`): `yy <= pivô` vira `20yy`, senão `19yy`.
    Não aplica validação de faixa aqui — isso é responsabilidade de quem chama
    (ver `extract_once`), pra manter a normalização pura e reutilizável.
    """
    if raw is None:
        return None
    try:
        value = int(str(raw).strip())
    except (ValueError, TypeError):
        return None

    if 0 <= value <= 99:
        return 2000 + value if value <= _PIVOT_2_DIGIT_ANO else 1900 + value
    return value


def normalize_cep(raw: str | int | None) -> str | None:
    """Normaliza um CEP com ou sem hífen para o formato `XXXXX-XXX`.

    Retorna `None` quando o valor não tem exatamente 8 dígitos (CEP inválido).

    `int` (3.1 / P2-1): zero-pad para 8 dígitos (`f"{raw:08d}"`) antes de
    seguir o fluxo normal -- um CEP de alto risco começando com "0" (ex.:
    07654-321, região de São Paulo) chega do LLM como `int` sem o
    zero à esquerda (`7654321`); convertê-lo direto pra string perderia esse
    zero e o CEP viraria "faltante" (8 dígitos exigidos) sem nenhum aviso.
    """
    if raw is None:
        return None
    if isinstance(raw, int):
        text = f"{raw:08d}"
    else:
        text = str(raw).strip()
    if _REGEX_CEP_COM_HIFEN.fullmatch(text):
        return text
    digits = _REGEX_DIGITS_ONLY.sub("", text)
    if len(digits) != 8:
        return None
    return f"{digits[:5]}-{digits[5:]}"


def normalize_idade(raw: int | str | None) -> int | None:
    """Normaliza a idade que o LLM devolveu para um inteiro.

    O `EXTRACT_TOOL` pede `idade` como inteiro (o LLM já resolve "nasci em
    1989", "tenho trinta e cinco" etc. e devolve o número). Aqui só coeragimos
    o valor do LLM para `int`; **não** interpretamos frase nenhuma por regex --
    interpretar linguagem natural é trabalho do LLM, não de um regex frágil que
    quebra em adversarial. Não valida faixa aqui -- ver `extract_once`.
    """
    if raw is None:
        return None
    if isinstance(raw, int):
        return raw
    try:
        return int(str(raw).strip())
    except ValueError:
        return None


def normalize_intent(raw: str | None) -> str:
    """Normaliza a intenção do turno (2.1) para um valor do enum conhecido.

    `None`, campo ausente (extração antiga) ou um valor fora do enum sempre
    caem no default neutro `"other"` -- nunca derruba a extração dos demais
    campos.
    """
    if raw is None:
        return _DEFAULT_INTENT
    value = str(raw).strip()
    return value if value in _VALID_INTENTS else _DEFAULT_INTENT


def normalize_data_inicio(raw: str | None) -> str | None:
    """Normaliza `data_inicio` para ISO `YYYY-MM-DD` (1.2 / P0-1).

    Aceita ISO (`YYYY-MM-DD`) e `dd/mm/aaaa`; qualquer outra coisa -- data
    inexistente ("30/02/2026"), texto livre ("amanhã") ou vazio -- vira
    `None`. **Nunca** deixa uma data malformada chegar ao payload da
    `/quote` (que só aceita `date.fromisoformat`): o caller (`agent.py`) usa
    o fallback de "hoje" quando este normalizador devolve `None`.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None

    try:
        return date.fromisoformat(text).isoformat()
    except ValueError:
        pass

    match = _REGEX_DATA_BR.fullmatch(text)
    if not match:
        return None
    dd, mm, yyyy = match.groups()
    try:
        return date(int(yyyy), int(mm), int(dd)).isoformat()
    except ValueError:
        return None


def _is_contradictory(field_name: str, old_value: Any, new_value: Any) -> bool:
    """Sobrescrita materialmente diferente de um essencial (2.2 / P1-4).

    Conservador de propósito: `None` (primeiro turno) ou valor igual nunca é
    contradição -- só correções grosseiras de `idade` (Δ>15) e `veiculo_ano`
    (Δ>10). Correção pequena/plausível (ex.: 35 -> 40) não passa desses
    limiares.

    **`cep` deliberadamente NÃO é contradição** (achado da bateria
    adversarial, `b14`). O prefixo de 2 dígitos do CEP é exatamente o que a
    `/quote` usa pra decidir o agravo de região (`prefixos_alto_risco`) --
    tratar prefixo diferente como fraude significava marcar como suspeita
    justamente toda correção de CEP que muda o preço. Mudar de CEP é a
    correção mais banal do domínio (mudança de endereço, carro que fica na
    casa da mãe); a `/quote` só reprecifica. O turno cai em
    `essential_changed` e o agente re-cota de verdade.
    """
    if old_value is None or new_value is None or old_value == new_value:
        return False
    if field_name == "idade":
        return abs(new_value - old_value) > _IDADE_CONTRADICTION_DELTA
    if field_name == "veiculo_ano":
        return abs(new_value - old_value) > _VEICULO_ANO_CONTRADICTION_DELTA
    return False


def _validate_range(value: int | None, low: int, high: int) -> int | None:
    if value is None:
        return None
    if low <= value <= high:
        return value
    return None


def extract_once(
    text: str,
    llm_client: LlmExtractorClient | None = None,
) -> ExtractionResult:
    """Extrai dados de uma única mensagem/turno do lead — 100% via LLM.

    Fluxo:
    1. Chama `llm_client.extract(text)` (structured output). O LLM é o único
       extrator: nenhum regex garimpa a mensagem crua (isso quebrava em
       adversarial -- ex.: pescar um "ano" de dentro de um telefone).
    2. Normaliza o **formato** do que o LLM devolveu (ano 2→4 díg, CEP, data)
       e valida as faixas da `/quote`; valores fora da faixa viram `None` com
       um aviso em `warnings` (nunca derrubam a extração dos demais campos).

    Sem `llm_client` nada é extraído. Se o client **levantar**, nada é
    extraído e `extractor_failed=True` -- o LLM é infraestrutura tanto quanto
    a `/quote`, então sua queda vira handoff `agent_error` imediato
    (`agent._handle_turn`), nunca `clarify_loop_exhausted` culpando o lead.
    A robustez contra alucinação vem da confirmação com o lead antes de
    cotar (Group E).
    """
    raw: dict[str, Any] = {}
    llm_used = False
    extractor_failed = False
    extractor_error: Exception | None = None

    if llm_client is not None:
        try:
            raw = llm_client.extract(text) or {}
            llm_used = True
        except Exception as exc:  # LLM indisponível -- sinaliza, não engole
            raw = {}
            llm_used = False
            extractor_failed = True
            extractor_error = exc

    warnings: list[str] = []

    veiculo_ano = normalize_ano(raw.get("veiculo_ano"))
    idade = normalize_idade(raw.get("idade"))
    cep = normalize_cep(raw.get("cep"))

    validated_ano = _validate_range(veiculo_ano, VEICULO_ANO_MIN, VEICULO_ANO_MAX)
    if veiculo_ano is not None and validated_ano is None:
        warnings.append(
            f"veiculo_ano fora da faixa válida ({VEICULO_ANO_MIN}-{VEICULO_ANO_MAX}): {veiculo_ano}"
        )

    validated_idade = _validate_range(idade, IDADE_MIN, IDADE_MAX)
    if idade is not None and validated_idade is None:
        warnings.append(f"idade fora da faixa válida ({IDADE_MIN}-{IDADE_MAX}): {idade}")

    data = ExtractedData(
        veiculo_ano=validated_ano,
        idade=validated_idade,
        cep=cep,
        marca=raw.get("marca"),
        modelo=raw.get("modelo"),
        data_inicio=normalize_data_inicio(raw.get("data_inicio")),
        intent=normalize_intent(raw.get("intent")),
    )

    return ExtractionResult(
        data=data,
        llm_used=llm_used,
        llm_raw=raw,
        warnings=warnings,
        extractor_failed=extractor_failed,
        extractor_error=extractor_error,
    )


@dataclass
class QualificationSession:
    """Acumula dados extraídos ao longo de vários turnos de conversa.

    `attempts` conta **todo** turno processado (telemetria). O que decide
    handoff é `stalled_turns`: turnos **consecutivos sem progresso** (nenhum
    essencial novo -- `idade`/`veiculo_ano`/`cep` -- capturado). Um lead
    cooperativo que dá 1 dado essencial por turno nunca estoura, mesmo que
    leve mais turnos que `max_attempts` pra terminar (bug L1: contar
    tentativas totais penalizava progresso lento, não falta de cooperação).
    Quando `stalled_turns` atinge `max_attempts` (default `N=2`, DEC-5) e
    ainda falta dado essencial, `needs_handoff()` sinaliza que o agente deve
    transbordar em vez de insistir indefinidamente (ver Group E / handoff).
    """

    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    attempts: int = 0
    stalled_turns: int = 0
    data: ExtractedData = field(default_factory=ExtractedData)
    warnings: list[str] = field(default_factory=list)

    def process_turn(
        self,
        text: str,
        llm_client: LlmExtractorClient | None = None,
    ) -> ExtractionResult:
        """Processa mais uma mensagem do lead, acumulando o que for extraído.

        Campos já conhecidos de turnos anteriores são preservados quando o
        turno atual não traz um valor novo (não-`None`) para eles.

        Progresso (novo essencial capturado neste turno) zera
        `stalled_turns`; sem progresso, incrementa -- é esse contador de
        estagnação (não `attempts`) que alimenta `needs_handoff()`.
        """
        self.attempts += 1
        before_missing = set(self.data.essential_missing())
        result = extract_once(text, llm_client=llm_client)

        contradiction = False
        for f in ("veiculo_ano", "idade", "cep", "marca", "modelo", "data_inicio"):
            new_value = getattr(result.data, f)
            if new_value is not None:
                if f in _ESSENTIAL_FIELDS and _is_contradictory(
                    f, getattr(self.data, f), new_value
                ):
                    contradiction = True
                setattr(self.data, f, new_value)

        after_missing = set(self.data.essential_missing())
        if before_missing - after_missing:
            self.stalled_turns = 0
        else:
            self.stalled_turns += 1

        self.warnings.extend(result.warnings)
        result.contradiction = contradiction
        return result

    def missing_essential(self) -> list[str]:
        """Delega para `ExtractedData.essential_missing()` sobre o acumulado."""
        return self.data.essential_missing()

    def is_complete(self) -> bool:
        """True quando idade, veiculo_ano e cep já foram todos capturados."""
        return self.data.has_essential()

    def needs_handoff(self) -> bool:
        """True quando `stalled_turns` consecutivos sem progresso atingiu o
        limite e ainda falta dado essencial."""
        return self.stalled_turns >= self.max_attempts and not self.is_complete()
