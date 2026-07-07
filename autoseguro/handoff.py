"""Motor de handoff (transbordo pro humano) — Group E, DEC-8 (Q6).

Critério (ver `DECISOES.md`, Q6): transbordar **apenas** quando o humano pode
fazer algo que o agente **estruturalmente não pode** — fronteira de
**capacidade** ou **autoridade** (+ respeito ao lead / sensibilidade), nunca
conveniência. Cada transbordo carrega um `HandoffReason` auditável — nunca um
handoff "silencioso".

Tabela de gatilhos (Q6):

| Caso                                                        | Perna              | Reason                    |
|--------------------------------------------------------------|---------------------|---------------------------|
| `/quote` esgotou os retries                                   | Capacidade (infra)  | `QUOTE_UNAVAILABLE`       |
| Erro inesperado no agente                                      | Capacidade (fail-safe) | `AGENT_ERROR`          |
| Mídia essencial (áudio/imagem/doc sem transcrição)             | Capacidade          | `MEDIA_UNREADABLE`        |
| Fechamento/emissão de apólice                                  | Capacidade          | `POLICY_ISSUANCE`         |
| Loop de esclarecimento após N=2 tentativas por dado essencial | Capacidade          | `CLARIFY_LOOP_EXHAUSTED`  |
| Dados contraditórios / suspeita de fraude                     | Autoridade          | `CONTRADICTORY_DATA`      |
| Fora de escopo (sinistro, boleto, cancelamento, resid.)       | Escopo              | `OUT_OF_SCOPE`            |
| Lead pede humano explicitamente                                | Respeito ao lead    | `EXPLICIT_REQUEST`        |
| Reclamação / conflito / ameaça                                 | Sensibilidade       | `COMPLAINT_CONFLICT`      |

Os gatilhos determinísticos (`for_*`) cobrem os casos objetivos (esgotamento
de retries, mídia, pedido explícito, loop de esclarecimento, fechamento). Os
"fuzzy" (fora de escopo, dados contraditórios, conflito) vêm de um
classificador **injetável/mockável** (`FuzzyClassifier`) via `classify_fuzzy` —
mas a decisão final sempre passa por aqui e sempre grava o `reason`, então
todo transbordo é auditável.

**NÃO transbordam** (o agente resolve sozinho — ver `resolve_*` abaixo,
mantidos só para documentar/testar explicitamente a fronteira):
- Recusa 422 (regra dura, ex.: idade > 75 / veículo > 20a): explica e encerra.
- Plano fora do catálogo: informa os 3 planos e resolve.
- Objeção de preço/desconto: não há mecanismo de desconto no sistema — o
  agente oferece um plano mais barato (alavanca real), nunca fabrica desconto.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:  # evita import circular / custo em runtime — só para type hints
    from .extraction import QualificationSession
    from .quote_client import QuoteUnavailable


class HandoffReason(str, Enum):
    """Código de motivo auditável — gravado em todo evento de handoff."""

    QUOTE_UNAVAILABLE = "quote_unavailable"
    AGENT_ERROR = "agent_error"
    MEDIA_UNREADABLE = "media_unreadable"
    POLICY_ISSUANCE = "policy_issuance"
    CLARIFY_LOOP_EXHAUSTED = "clarify_loop_exhausted"
    CONTRADICTORY_DATA = "contradictory_data"
    OUT_OF_SCOPE = "out_of_scope"
    EXPLICIT_REQUEST = "explicit_request"
    COMPLAINT_CONFLICT = "complaint_conflict"


@dataclass(frozen=True)
class HandoffDecision:
    """Um transbordo decidido: motivo auditável + mensagem ao lead + contexto.

    `context` carrega o suficiente para o humano continuar sem re-perguntar
    tudo (idade, veículo, CEP, plano de interesse, cotação se houve, etc.) —
    quem grava em `trace.jsonl`/logs (Group F) é responsável por mascarar PII
    (Q3) antes de persistir.
    """

    reason: HandoffReason
    message: str
    context: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Gatilhos determinísticos — capacidade / autoridade / respeito
# ---------------------------------------------------------------------------


def for_quote_unavailable(exc: "QuoteUnavailable") -> HandoffDecision:
    """`/quote` esgotou retries/deadline/breaker (Capacidade — infra).

    Nunca inventa preço: a mensagem ao lead não menciona valor nenhum, só que
    o sistema está instável e um humano vai continuar.
    """
    return HandoffDecision(
        reason=HandoffReason.QUOTE_UNAVAILABLE,
        message=(
            "No momento não estou conseguindo confirmar sua cotação — nosso "
            "sistema de cálculo está instável. Vou te encaminhar para um "
            "consultor humano continuar o atendimento."
        ),
        context={
            "quote_reason": getattr(exc, "reason", None),
            "attempts": getattr(exc, "attempts", None),
            **getattr(exc, "context", {}),
        },
    )


def for_agent_error(exc: Exception) -> HandoffDecision:
    """Erro inesperado no agente (Capacidade — fail-safe).

    Rede de segurança: qualquer exceção não mapeada nunca deve travar a
    conversa sem handoff — melhor transbordar com contexto do que quebrar.
    """
    return HandoffDecision(
        reason=HandoffReason.AGENT_ERROR,
        message=(
            "Encontrei um problema inesperado por aqui. Vou te encaminhar "
            "para um consultor humano continuar o atendimento."
        ),
        context={"error": repr(exc)},
    )


_MEDIA_TYPES_UNREADABLE = frozenset({"image", "audio", "document"})


def for_media_unreadable(media_type: str) -> HandoffDecision:
    """Mídia essencial (áudio/imagem/documento) sem transcrição (Capacidade).

    O dataset do desafio confirma: mensagens de mídia só trazem marcador, sem
    transcrição — o agente estruturalmente não consegue ler o conteúdo.
    """
    return HandoffDecision(
        reason=HandoffReason.MEDIA_UNREADABLE,
        message=(
            f"Recebi um(a) {media_type}, mas não consigo processar esse tipo "
            "de conteúdo por aqui. Vou te encaminhar para um consultor "
            "humano dar continuidade."
        ),
        context={"media_type": media_type},
    )


def is_media_essential(media_type: str | None) -> bool:
    """True quando `media_type` é uma mídia sem transcrição (ver `for_media_unreadable`)."""
    return media_type in _MEDIA_TYPES_UNREADABLE


_POLICY_ISSUANCE_RE = re.compile(
    r"emitir\s+a?\s*ap[oó]lice"
    r"|fechar\s+(o\s+)?(contrato|seguro|neg[oó]cio)"
    r"|quero\s+contratar"
    r"|gerar\s+(o\s+)?boleto"
    r"|forma\s+de\s+pagamento"
    r"|como\s+(eu\s+)?pago",
    re.IGNORECASE,
)


def for_policy_issuance(text: str) -> HandoffDecision | None:
    """Fechamento/emissão de apólice (Capacidade — sem tool de emissão).

    O agente cota, mas não tem ferramenta pra fechar/emitir/cobrar — fabricar
    isso seria pior do que transbordar. Retorna `None` quando o texto não
    indica intenção de fechamento (não é gatilho — é uma conversa normal).
    """
    if _POLICY_ISSUANCE_RE.search(text):
        return HandoffDecision(
            reason=HandoffReason.POLICY_ISSUANCE,
            message=(
                "Consigo te dar a cotação certinha, mas fechar e emitir a "
                "apólice é com um consultor humano. Vou te encaminhar para "
                "continuar por lá."
            ),
            context={"trigger_text": text},
        )
    return None


#  "atendente"/"humano"/"pessoa de verdade"/"falar com alguém" já são
# inequívocos por si só. "gerente"/"supervisor" são ambíguos (o lead pode só
# estar descrevendo o próprio cargo, ex.: "sou gerente de vendas, quero
# cotar") -- endurecido (P1-2/stopgap) pra só disparar quando há um verbo de
# pedido logo antes (falar com/quero/chamar/chama/passa pra/passa para).
_EXPLICIT_REQUEST_RE = re.compile(
    r"\b(atendente|humano)\b"
    r"|pessoa\s+de\s+verdade"
    r"|falar\s+com\s+(algu[ée]m|uma\s+pessoa)"
    r"|(?:falar\s+com|quero|chamar|chama|passa\s+pra|passa\s+para)\s+(?:o\s+|a\s+)?(gerente|supervisor)",
    re.IGNORECASE,
)


def _build_explicit_request_decision(text: str) -> HandoffDecision:
    return HandoffDecision(
        reason=HandoffReason.EXPLICIT_REQUEST,
        message="Sem problemas! Vou te encaminhar para um atendente humano agora.",
        context={"trigger_text": text},
    )


def for_explicit_request(text: str) -> HandoffDecision | None:
    """Lead pede humano explicitamente (Respeito ao lead — imediato).

    Retorna `None` quando não há pedido explícito no texto. Stopgap
    regex-based (ver `classify_scope` pro caminho primário via `intent`).
    """
    if _EXPLICIT_REQUEST_RE.search(text):
        return _build_explicit_request_decision(text)
    return None


def for_clarify_loop_exhausted(session: "QualificationSession") -> HandoffDecision | None:
    """Loop de esclarecimento esgotado após N=2 tentativas (Capacidade).

    Delega a decisão para `QualificationSession.needs_handoff()` (Group D);
    retorna `None` enquanto ainda há tentativas disponíveis ou os dados
    essenciais já foram completados.
    """
    if not session.needs_handoff():
        return None
    return HandoffDecision(
        reason=HandoffReason.CLARIFY_LOOP_EXHAUSTED,
        message=(
            "Não consegui reunir todos os dados necessários para cotar. Vou "
            "te encaminhar para um consultor humano continuar o atendimento."
        ),
        context={
            "attempts": session.attempts,
            "missing": session.missing_essential(),
            "data": vars(session.data),
        },
    )


# ---------------------------------------------------------------------------
# Gatilhos "fuzzy" — classificador LLM injetável/mockável
# ---------------------------------------------------------------------------

_FUZZY_REASONS = frozenset(
    {
        HandoffReason.OUT_OF_SCOPE,
        HandoffReason.CONTRADICTORY_DATA,
        HandoffReason.COMPLAINT_CONFLICT,
    }
)


class FuzzyClassifier(Protocol):
    """Interface mínima do classificador fuzzy de handoff.

    Implementações reais podem envolver uma chamada LLM; nos testes, um
    dublê simples que devolve um `HandoffReason | None` já basta. Só pode
    devolver um dos motivos "fuzzy" (`OUT_OF_SCOPE`, `CONTRADICTORY_DATA`,
    `COMPLAINT_CONFLICT`) — os determinísticos (ex.: `EXPLICIT_REQUEST`) são
    sempre decididos pelas funções `for_*` acima, nunca pelo classificador.
    """

    def classify(self, text: str) -> HandoffReason | None:
        ...


def for_contradictory_data(text: str) -> HandoffDecision:
    """Dados contraditórios / suspeita de inconsistência insanável (Autoridade)."""
    return HandoffDecision(
        reason=HandoffReason.CONTRADICTORY_DATA,
        message=(
            "Percebi algumas informações contraditórias na nossa conversa. "
            "Para evitar erro na sua cotação, vou te encaminhar para um "
            "consultor humano revisar com calma."
        ),
        context={"trigger_text": text},
    )


def for_out_of_scope(text: str) -> HandoffDecision:
    """Fora de escopo — sinistro, boleto, cancelamento, seguro residencial etc. (Escopo)."""
    return HandoffDecision(
        reason=HandoffReason.OUT_OF_SCOPE,
        message=(
            "Esse assunto foge do que eu resolvo por aqui (cotação de seguro "
            "de veículo). Vou te encaminhar para um consultor humano."
        ),
        context={"trigger_text": text},
    )


def for_complaint_conflict(text: str) -> HandoffDecision:
    """Reclamação / conflito / ameaça (Sensibilidade)."""
    return HandoffDecision(
        reason=HandoffReason.COMPLAINT_CONFLICT,
        message=(
            "Sinto muito pela situação. Vou te encaminhar agora para um "
            "consultor humano cuidar disso com a atenção que merece."
        ),
        context={"trigger_text": text},
    )


_FUZZY_BUILDERS = {
    HandoffReason.OUT_OF_SCOPE: for_out_of_scope,
    HandoffReason.CONTRADICTORY_DATA: for_contradictory_data,
    HandoffReason.COMPLAINT_CONFLICT: for_complaint_conflict,
}


def classify_fuzzy(text: str, classifier: FuzzyClassifier | None) -> HandoffDecision | None:
    """Consulta o classificador fuzzy (se houver) e monta a `HandoffDecision`.

    `classifier is None` é o default seguro (nenhum classificador fuzzy
    plugado ainda) — retorna `None` sem chamar nada. Se o classificador
    devolver um motivo fora dos "fuzzy" permitidos, levanta `ValueError`
    (o determinístico nunca deveria vir por essa porta — todo handoff segue
    auditável mesmo em caso de mau uso do classificador).
    """
    if classifier is None:
        return None
    reason = classifier.classify(text)
    if reason is None:
        return None
    if reason not in _FUZZY_REASONS:
        raise ValueError(
            f"classificador fuzzy retornou motivo fora do permitido: {reason!r}"
        )
    return _FUZZY_BUILDERS[reason](text)


# Mapeia intenção (2.1, fundida no EXTRACT_TOOL) direto pro builder do motivo
# fuzzy correspondente -- dispensa o classificador/regex quando o sinal já
# veio da própria extração.
_INTENT_TO_FUZZY_BUILDER = {
    "out_of_scope": for_out_of_scope,
    "complaint": for_complaint_conflict,
}


def classify_scope(
    intent: str | None,
    text: str,
    fuzzy_classifier: "FuzzyClassifier | None" = None,
) -> HandoffDecision | None:
    """Decide escopo/handoff a partir do sinal de intenção (2.1, P1-2).

    Ordem: primeiro o sinal de `intent` (vem fundido na mesma chamada de
    extração — sem custo extra de LLM); só cai no stopgap
    (`for_explicit_request` + `classify_fuzzy`/`KeywordFuzzyClassifier`)
    quando `intent` não ajuda (`None`/`"other"`/`"provide_data"`/etc. -- ex.:
    extractor não plugado, ou o LLM não viu sinal de escopo no turno).

    Nunca é o LLM quem *decide* transbordar — só classifica; a decisão e a
    `HandoffDecision` auditável continuam sendo montadas aqui, de forma
    determinística.
    """
    if intent == "explicit_human":
        return _build_explicit_request_decision(text)
    builder = _INTENT_TO_FUZZY_BUILDER.get(intent or "")
    if builder is not None:
        return builder(text)

    # Stopgap: sem sinal de intenção útil -- regex/keyword endurecidos como
    # rede de segurança.
    return for_explicit_request(text) or classify_fuzzy(text, fuzzy_classifier)


_COMPLAINT_ABRIR_PROCESSO_RE = re.compile(r"abrir\s+(um\s+)?processo", re.IGNORECASE)
_OUT_OF_SCOPE_COBRANCA_INDEVIDA_RE = re.compile(r"cobran[çc]a\s+indevida", re.IGNORECASE)


class KeywordFuzzyClassifier:
    """`FuzzyClassifier` determinístico por palavras-chave (sem LLM).

    Cobre os casos claros de **fora de escopo** (sinistro/boleto/cancelamento/
    outros produtos) e **reclamação/conflito** de forma auditável e sem custo de
    token. Conservador de propósito: em caso de dúvida retorna `None` e a mensagem
    segue no fluxo normal de qualificação. `CONTRADICTORY_DATA` não é inferido por
    keyword (depende de inconsistência de dados, não de vocabulário).

    Substrings largas demais foram endurecidas para frases específicas (P1-2 —
    falso-positivo do red-team): "processo" (bare) casava "qual o processo pra
    contratar?"; "cobrança"/"cobranca" (bare) casava "como funciona a cobrança
    mensal?". Agora exigem a frase completa (`abrir processo`, `cobrança
    indevida`).
    """

    _OUT_OF_SCOPE = (
        "sinistro", "boleto", "segunda via", "2a via", "2ª via", "cancelar seguro",
        "cancelar apólice", "cancelar apolice", "cancelamento", "seguro residencial",
        "seguro de vida", "seguro viagem", "reembolso",
    )
    _COMPLAINT = (
        "reclamação", "reclamacao", "reclamar", "procon", "processar", "advogado",
        "jurídico", "juridico", "absurdo", "ridículo", "ridiculo",
        "péssimo", "pessimo", "vergonha", "descaso",
    )

    def classify(self, text: str) -> "HandoffReason | None":
        low = text.lower()
        # Reclamação tem prioridade: "quero cancelar, isso é um absurdo" → conflito.
        if any(k in low for k in self._COMPLAINT) or _COMPLAINT_ABRIR_PROCESSO_RE.search(low):
            return HandoffReason.COMPLAINT_CONFLICT
        if any(k in low for k in self._OUT_OF_SCOPE) or _OUT_OF_SCOPE_COBRANCA_INDEVIDA_RE.search(low):
            return HandoffReason.OUT_OF_SCOPE
        # Cancelamento de apólice/seguro existente (fora do escopo de vendas),
        # tolerante a palavras no meio ("cancelar MEU seguro").
        if "cancel" in low and any(w in low for w in ("seguro", "apólice", "apolice", "plano")):
            return HandoffReason.OUT_OF_SCOPE
        return None


# ---------------------------------------------------------------------------
# "Não transbordam" — o agente resolve sozinho (Q6). Mantidas como funções
# explícitas só para documentar/testar a fronteira: sempre retornam `None`.
# ---------------------------------------------------------------------------


def resolve_quote_refusal(motivo: str) -> None:
    """422 (`CotacaoRecusada`) é regra dura (ex.: idade > 75, veículo > 20a).

    Um humano não reverteria a regra — o agente explica o motivo e encerra.
    NUNCA gera handoff.
    """
    return None


def resolve_plan_not_in_catalog(plano_mencionado: str) -> None:
    """Plano fora do catálogo (`essencial`/`completo`/`premium`).

    O agente informa os 3 planos disponíveis e segue a conversa. NUNCA gera
    handoff.
    """
    return None


def resolve_price_objection(text: str) -> None:
    """Objeção de preço / pedido de desconto.

    Não há mecanismo de desconto no sistema (a `/quote` não tem esse campo) —
    o agente é honesto sobre isso e oferece um plano mais barato (alavanca
    real) em vez de fabricar um desconto. NUNCA gera handoff por si só; só
    transborda se o lead **pedir humano** (aí é `EXPLICIT_REQUEST`, não este
    caso).
    """
    return None
