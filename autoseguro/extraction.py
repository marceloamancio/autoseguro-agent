"""Extração e qualificação de dados de texto livre (Group D, DEC-5 / Q3b).

Contexto (ver DECISOES.md, Q3b e `WISH.md`, Group D): o lead descreve veículo,
idade e CEP em texto livre e desorganizado ("e um Sandero 2022", "tenho 35 anos,
cep 26703-384"). A `/quote` só usa o **ano** do veículo para cotar — marca/modelo
servem só pra rapport, nunca entram no preço.

Estratégia (DEC-5): extração via **LLM com structured outputs** (cliente
injetado/mockável — este módulo nunca chama a API real) com **normalização**
(ano 2/4 dígitos, CEP com/sem hífen, "nasci em AAAA" → idade) e **validação de
faixas** iguais às da `/quote` (`idade` 0–200, `veiculo_ano` 1950–2100). Quando o
LLM falha, está indisponível ou devolve campos vazios, um **backstop regex leve**
tenta pescar ano e CEP diretamente do texto, como rede de segurança.

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

# Pivô pra desambiguar ano de 2 dígitos: yy <= _PIVOT_2_DIGIT_ANO -> 20xx,
# senão -> 19xx. Regra comum de calendário (ex.: "08" -> 2008, "97" -> 1997).
_PIVOT_2_DIGIT_ANO = 68  # ano corrente % 100 seria mais "correto", mas um pivô
# fixo é mais previsível/testável e cobre a janela útil de veículos (1950-2100).

_REGEX_ANO = re.compile(r"\b(19\d{2}|20\d{2})\b")
_REGEX_CEP_COM_HIFEN = re.compile(r"\b\d{5}-\d{3}\b")
_REGEX_CEP_8_DIGITOS = re.compile(r"\b\d{8}\b")
# 3.2 (P2-2): data completa de nascimento ("nasci em dd/mm/aaaa") -- checada
# antes de `_REGEX_NASCI_EM` (só ano) pra calcular a idade exata respeitando
# mês/dia, em vez do erro sistemático de ±1 que o cálculo só-por-ano produz
# (grave justo na fronteira 75/76 de recusa).
_REGEX_NASCI_EM_DATA_COMPLETA = re.compile(
    r"nasci\w*\s+em\s+(\d{2})/(\d{2})/(\d{4})", re.IGNORECASE
)
_REGEX_NASCI_EM = re.compile(r"nasci\w*\s+em\s+(\d{4})", re.IGNORECASE)

# Idades onde o erro de ±1 do cálculo só-por-ano é mais grave: cruzam a
# fronteira de recusa 75/76 (ver `namastex-fde-challenge` `plans.json`/regra
# de negócio) -- só nesses valores vale logar o warning (não pra toda idade).
_IDADE_BOUNDARY_WARNING_VALUES = frozenset({75, 76})
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
    07654-321, região de São Paulo) chega do LLM/backstop como `int` sem o
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


def _age_from_birthdate(birth: date, today: date) -> int:
    """Idade exata em `today`, respeitando se o aniversário do ano já passou."""
    age = today.year - birth.year
    if (today.month, today.day) < (birth.month, birth.day):
        age -= 1
    return age


def _idade_so_por_ano(birth_year: int) -> int:
    """Cálculo aproximado de sempre (`date.today().year - birth_year`), com
    warning (3.2 / P2-2) quando o resultado cai na fronteira 75/76."""
    idade = date.today().year - birth_year
    if idade in _IDADE_BOUNDARY_WARNING_VALUES:
        logger.warning(
            "normalize_idade: idade %s calculada só a partir do ano de "
            "nascimento (%s), sem mês/dia -- erro de ±1 possível bem na "
            "fronteira 75/76 de recusa por idade",
            idade,
            birth_year,
        )
    return idade


def normalize_idade(raw: int | str | None) -> int | None:
    """Normaliza idade a partir de um inteiro, string numérica ou frase.

    Reconhece dois padrões (3.2 / P2-2):

    - **Data completa** ("nasci em dd/mm/aaaa"): calcula a idade **exata**,
      respeitando se o aniversário deste ano já passou (`_age_from_birthdate`)
      -- corrige o erro sistemático de ±1 do cálculo só-por-ano. Data
      inexistente (ex.: "31/02/1990") não derruba a extração: cai no
      fallback abaixo (só ano).
    - **Só o ano** ("nasci em aaaa"): segue o cálculo aproximado de sempre
      (`date.today().year - aaaa`), mas **loga um warning** quando a idade
      resultante cai na fronteira 75/76 (`_IDADE_BOUNDARY_WARNING_VALUES`) --
      é justo aí que o erro de ±1 pode empurrar o lead pro lado errado da
      regra de recusa por idade.

    Não valida faixa aqui — ver `extract_once`.
    """
    if raw is None:
        return None
    if isinstance(raw, int):
        return raw

    text = str(raw).strip()

    full_date_match = _REGEX_NASCI_EM_DATA_COMPLETA.search(text)
    if full_date_match:
        dd, mm, yyyy = (int(g) for g in full_date_match.groups())
        try:
            birth = date(yyyy, mm, dd)
        except ValueError:
            # Data inexistente (ex.: 31/02) -- não derruba a extração, cai no
            # fallback de só-ano (mesmo warning de fronteira se aplicável).
            return _idade_so_por_ano(yyyy)
        return _age_from_birthdate(birth, date.today())

    match = _REGEX_NASCI_EM.search(text)
    if match:
        return _idade_so_por_ano(int(match.group(1)))

    try:
        return int(text)
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


def backstop_extract(text: str) -> dict[str, Any]:
    """Rede de segurança via regex leve: ano (19xx/20xx) e CEP direto do texto.

    Usado quando o cliente LLM não está disponível, levanta exceção, ou devolve
    campos vazios para ano/CEP. Não tenta capturar `idade`/marca/modelo — esses
    dependem demais de contexto pra um regex leve ser confiável.
    """
    result: dict[str, Any] = {}

    ano_match = _REGEX_ANO.search(text)
    if ano_match:
        result["veiculo_ano"] = int(ano_match.group(1))

    cep_match = _REGEX_CEP_COM_HIFEN.search(text) or _REGEX_CEP_8_DIGITOS.search(text)
    if cep_match:
        normalized = normalize_cep(cep_match.group(0))
        if normalized:
            result["cep"] = normalized

    return result


def _is_contradictory(field_name: str, old_value: Any, new_value: Any) -> bool:
    """Sobrescrita materialmente diferente de um essencial (2.2 / P1-4).

    Conservador de propósito: `None` (primeiro turno) ou valor igual nunca é
    contradição -- só correções grosseiras dos 3 essenciais: `idade`
    Δ>15, `veiculo_ano` Δ>10, `cep` com prefixo (2 díg) diferente. Correção
    pequena/plausível (ex.: 35 -> 40) não passa desses limiares.
    """
    if old_value is None or new_value is None or old_value == new_value:
        return False
    if field_name == "idade":
        return abs(new_value - old_value) > _IDADE_CONTRADICTION_DELTA
    if field_name == "veiculo_ano":
        return abs(new_value - old_value) > _VEICULO_ANO_CONTRADICTION_DELTA
    if field_name == "cep":
        return str(old_value)[:2] != str(new_value)[:2]
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
    """Extrai dados de uma única mensagem/turno do lead.

    Fluxo:
    1. Tenta o `llm_client.extract(text)` (structured output); qualquer
       exceção ou retorno vazio é tratado como "LLM não ajudou" — sem
       propagar erro pro caller.
    2. Normaliza os campos essenciais (`veiculo_ano`, `idade`, `cep`) e valida
       as faixas da `/quote`; valores fora da faixa viram `None` com um aviso
       em `warnings` (nunca derrubam a extração dos demais campos).
    3. Preenche `veiculo_ano`/`cep` ainda ausentes com o backstop regex.
    """
    raw: dict[str, Any] = {}
    llm_used = False

    if llm_client is not None:
        try:
            raw = llm_client.extract(text) or {}
            llm_used = True
        except Exception:
            raw = {}
            llm_used = False

    warnings: list[str] = []

    veiculo_ano = normalize_ano(raw.get("veiculo_ano"))
    idade = normalize_idade(raw.get("idade"))
    cep = normalize_cep(raw.get("cep"))

    # Backstop regex: só entra em ação pros campos que o LLM não trouxe.
    if veiculo_ano is None or cep is None:
        backstop = backstop_extract(text)
        if veiculo_ano is None and "veiculo_ano" in backstop:
            veiculo_ano = backstop["veiculo_ano"]
        if cep is None and "cep" in backstop:
            cep = backstop["cep"]

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

    return ExtractionResult(data=data, llm_used=llm_used, llm_raw=raw, warnings=warnings)


@dataclass
class QualificationSession:
    """Acumula dados extraídos ao longo de vários turnos de conversa.

    Cada `process_turn` conta como uma tentativa de qualificação. Quando o
    número de tentativas atinge `max_attempts` (default `N=2`, DEC-5) e ainda
    falta dado essencial, `needs_handoff()` sinaliza que o agente deve
    transbordar em vez de insistir indefinidamente (ver Group E / handoff).
    """

    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    attempts: int = 0
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
        """
        self.attempts += 1
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
        """True quando o limite de tentativas foi atingido sem dado essencial."""
        return self.attempts >= self.max_attempts and not self.is_complete()
