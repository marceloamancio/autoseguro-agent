"""Core do agente de vendas AutoSeguro (Group E, DEC-1/6/7/8 + refinamento #3).

Fluxo: **conversa → qualifica → cota → decide** (ver `DECISOES.md`, Q1/Q6 e
`WISH.md`, Group E).

Design (DEC-1, "agente enxuto"): a lógica de qualificação/confirmação/cotação
e o critério de handoff são **controle determinístico em Python**, não uma
decisão livre do LLM — é o que Q1 pede quando justifica o SDK direto da
Anthropic ("a lógica de resiliência e o critério de handoff ficam explícitos
e legíveis"). O LLM (`AsyncAnthropic`-like, injetado e mockável) entra só onde
é seguro e não-crítico: respostas de conversa livre depois que a cotação já
foi entregue/recusada (nunca na formatação do preço, que é sempre construída
a partir da resposta real da `/quote`).

- `call_quote` é o método que corresponde à tool única exposta ao domínio
  (`CALL_QUOTE_TOOL_SCHEMA` documenta o schema que seria registrado como tool
  da Anthropic numa integração completa de tool-use); aqui é invocado pelo
  fluxo determinístico assim que a qualificação é confirmada.
- Extração/qualificação delega a `QualificationSession` (Group D) — o agente
  só pluga um `extractor` (mesmo Protocol `LlmExtractorClient` de
  `extraction.py`) e trata os sinais de completude/handoff que ela expõe.
- Explicação da cotação (`format_quote_explanation`) é 100% derivada dos
  campos do `QuoteResult` — nunca inventa preço fora da resposta da `/quote`.
- Minimização (Q3): o agente nunca pergunta CPF, e-mail, telefone ou placa —
  só os campos que a `/quote` usa (`idade`, `veiculo_ano`, `cep`) mais
  `data_inicio` (opcional, com fallback avisado — refinamento #3).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Protocol

from . import handoff
from .extraction import ExtractedData, LlmExtractorClient, QualificationSession
from .handoff import FuzzyClassifier, HandoffDecision
from .quote_client import (
    CotacaoRecusada,
    PayloadInvalido,
    QuoteClient,
    QuoteResult,
    QuoteUnavailable,
)

_CONFIRM_YES_RE = re.compile(
    r"\b(sim|isso\s+mesmo|confirmo|correto|exato|perfeito|confere|"
    r"pode\s+ser|ok|beleza)\b",
    re.IGNORECASE,
)
_CONFIRM_NO_RE = re.compile(r"\b(n[aã]o|errado|incorreto)\b", re.IGNORECASE)

# Perguntas que o agente pode fazer para qualificar — de propósito, restritas
# aos únicos 3 campos essenciais que a `/quote` usa (Q3, minimização: nunca
# CPF/e-mail/telefone/placa).
FIELD_QUESTIONS: dict[str, str] = {
    "idade": "sua idade",
    "veiculo_ano": "o ano do seu veículo",
    "cep": "o CEP onde o carro fica",
}

# Schema documentado da tool única exposta ao domínio (DEC-1). Numa
# integração completa de tool-use da Anthropic, seria passado em
# `messages.create(tools=[CALL_QUOTE_TOOL_SCHEMA])`. Aqui o fluxo
# determinístico chama `Agent.call_quote` diretamente (ver docstring do
# módulo).
CALL_QUOTE_TOOL_SCHEMA: dict[str, Any] = {
    "name": "call_quote",
    "description": (
        "Cota um seguro de veículo a partir do plano, idade, ano do "
        "veículo, CEP e data de início da vigência."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "plano_id": {
                "type": "string",
                "enum": ["essencial", "completo", "premium"],
            },
            "idade": {"type": "integer", "minimum": 0, "maximum": 200},
            "veiculo_ano": {"type": "integer", "minimum": 1950, "maximum": 2100},
            "cep": {"type": "string"},
            "data_inicio": {"type": "string", "description": "YYYY-MM-DD"},
        },
        "required": ["plano_id", "idade", "veiculo_ano"],
    },
}

SYSTEM_PROMPT = """\
Você é um vendedor da AutoSeguro, uma seguradora de veículos, conversando \
por WhatsApp com um lead.

Seu trabalho: conversar, qualificar os dados necessários, confirmar o que \
entendeu antes de cotar, cotar e explicar o resultado — ou encaminhar para \
um humano quando o caso sair da sua capacidade ou alçada.

Dados que você PODE e DEVE coletar: idade do lead, o ano do veículo (marca/ \
modelo servem só pra rapport, não entram na cotação), o CEP onde o carro \
fica e, se o lead souber, a data em que quer que o seguro comece a valer \
(data_inicio).

Minimização de dados — NUNCA peça CPF, e-mail, telefone ou placa do \
veículo: a cotação não usa nenhum desses dados, então não há motivo para \
coletá-los. Se o lead informar algum deles espontaneamente, não repita nem \
confirme o valor de volta — apenas siga com o que é necessário para cotar.

Nunca invente um preço: o valor da cotação só existe depois de uma chamada \
bem-sucedida à ferramenta de cotação. Explique coberturas, franquia, \
carência (30 dias para roubo/furto) e pró-rata do primeiro pagamento \
sempre a partir da resposta real da cotação, nunca de memória.

Encaminhe para um humano quando: a cotação falhar por instabilidade do \
sistema, o lead pedir para falar com uma pessoa, o assunto sair do escopo \
de venda de seguro de veículo, houver reclamação/conflito, ou for necessário \
fechar/emitir a apólice (você não tem essa ferramenta).\
"""


class _MessagesApi(Protocol):
    async def create(self, **kwargs: Any) -> Any:
        ...


class AsyncLlmClient(Protocol):
    """Interface mínima esperada do cliente LLM (`AsyncAnthropic`-like).

    Só precisa expor `.messages.create(**kwargs)` (assíncrono) — o suficiente
    pra um dublê de teste simples e compatível com o SDK real da Anthropic.
    """

    messages: _MessagesApi


@dataclass
class AgentState:
    """Estado de uma conversa (um `Agent` cuida de uma conversa por vez).

    `session` é a `QualificationSession` injetada no construtor — acumula os
    dados extraídos turno a turno (Group D). Os demais campos são o controle
    de fluxo específico do agente (confirmação, plano de interesse, se já
    fechou ou transbordou).
    """

    session: QualificationSession
    confirmed: bool = False
    awaiting_confirmation: bool = False
    plano_id: str | None = None
    closed: bool = False
    quote_delivered: bool = False
    handoff: HandoffDecision | None = None
    asked_data_inicio: bool = False


@dataclass
class AgentTurn:
    """Resultado de processar um turno — o que responder ao lead + sinais."""

    reply: str
    handoff: HandoffDecision | None = None
    quote: QuoteResult | None = None
    closed: bool = False


def normalize_plano_mention(text: str) -> str | None:
    """Detecta menção a um dos 3 planos do catálogo no texto do lead."""
    match = re.search(r"\b(essencial|completo|premium)\b", text, re.IGNORECASE)
    return match.group(1).lower() if match else None


def build_missing_fields_question(missing: list[str], *, ask_data_inicio: bool) -> str:
    """Monta a pergunta pelos dados essenciais que ainda faltam.

    Restrita aos campos de `FIELD_QUESTIONS` — nunca menciona CPF, e-mail,
    telefone ou placa (Q3, minimização).
    """
    partes = [FIELD_QUESTIONS[campo] for campo in missing if campo in FIELD_QUESTIONS]
    pergunta = "Pra eu conseguir cotar, me conta " + ", ".join(partes) + "?"
    if ask_data_inicio:
        pergunta += (
            " Se já souber, me diga também quando você quer que o seguro "
            "comece a valer (data de início)."
        )
    return pergunta


def build_confirmation_message(data: ExtractedData) -> str:
    """Resume o que foi entendido e pede confirmação antes de cotar (DEC-5)."""
    veiculo_partes = [p for p in (data.marca, data.modelo) if p]
    veiculo_desc = " ".join([*veiculo_partes, str(data.veiculo_ano)]).strip()
    return (
        f"Só confirmando antes de cotar: você tem {data.idade} anos, "
        f"veículo {veiculo_desc}, CEP {data.cep}. Está correto?"
    )


def format_quote_explanation(
    quote: QuoteResult, *, fallback_note: str | None = None
) -> str:
    """Explica a cotação ao lead a partir da resposta real da `/quote`.

    Cobre coberturas, franquia, carência (30d roubo/furto) e pró-rata do
    primeiro pagamento quando presentes na resposta (refinamento #3). Nunca
    inclui um valor que não tenha vindo de `quote`.
    """
    linhas = [
        f"Cotação do plano {quote.plano_nome}: R$ {quote.premio_mensal:.2f}/mês.",
        f"Coberturas: {', '.join(quote.coberturas)}.",
        f"Franquia: R$ {quote.franquia:.2f}.",
    ]

    carencia = quote.carencia or {}
    carencia_coberturas = carencia.get("coberturas") or []
    carencia_dias = carencia.get("dias")
    if carencia_coberturas and carencia_dias:
        linhas.append(
            f"Carência: {carencia_dias} dias para acionar "
            f"{', '.join(carencia_coberturas)}."
        )

    if quote.primeiro_pagamento_pro_rata:
        pro = quote.primeiro_pagamento_pro_rata
        linhas.append(
            "Como a vigência não começa no dia 1, o primeiro pagamento é "
            f"proporcional (pró-rata): R$ {pro['valor_primeiro_pagamento']:.2f} "
            f"({pro['dias_cobrados']} de {pro['dias_no_mes']} dias do mês)."
        )

    if fallback_note:
        linhas.append(fallback_note)

    return "\n".join(linhas)


def format_refusal(exc: CotacaoRecusada) -> str:
    """Explica uma recusa 422 (regra dura) e encerra — nunca gera handoff."""
    handoff.resolve_quote_refusal(exc.motivo)
    return (
        f"Não consigo fechar essa cotação: {exc.motivo} Essa é uma regra "
        "fixa do nosso sistema de cotação, então não tem como eu contornar "
        "por aqui."
    )


def format_payload_invalido(exc: PayloadInvalido) -> str:
    """Pede o dado faltante/errado apontado pela API (400) — sem handoff."""
    return (
        f"Preciso ajustar um dado antes de cotar: {exc.detalhe} Pode "
        "confirmar de novo sua idade, o ano do veículo e o CEP?"
    )


def extract_reply_text(response: Any) -> str:
    """Extrai o texto de uma resposta do Messages API (ou de um dublê simples).

    Aceita uma `Message` real (content = lista de blocos com `.text`), um
    dublê com atributo `.text`, ou uma `str` direta — duck-typing generoso
    pra não amarrar os testes a um shape rígido do SDK.
    """
    if isinstance(response, str):
        return response
    content = getattr(response, "content", None)
    if content:
        for block in content:
            text = getattr(block, "text", None)
            if text:
                return text
    text = getattr(response, "text", None)
    return text or ""


class Agent:
    """Agente de vendas AutoSeguro — uma instância cuida de uma conversa.

    Dependências injetáveis (nenhuma chamada real de rede a partir daqui):
    - `llm`: cliente `AsyncAnthropic`-like (mockável) — usado só para
      conversa livre depois que a cotação já foi entregue/recusada.
    - `quote_client`: `QuoteClient` (Group B) — cota via `call_quote`.
    - `session`: `QualificationSession` (Group D) — acumula os dados
      essenciais extraídos turno a turno.
    - `extractor` (opcional): `LlmExtractorClient` (mesmo Protocol de
      `extraction.py`) repassado ao `session.process_turn`; sem ele, a
      extração cai no backstop regex.
    - `fuzzy_classifier` (opcional): `FuzzyClassifier` (`handoff.py`) para os
      motivos "fuzzy" de handoff (fora de escopo, dados contraditórios,
      conflito).
    """

    def __init__(
        self,
        llm: AsyncLlmClient,
        quote_client: QuoteClient,
        session: QualificationSession,
        *,
        extractor: LlmExtractorClient | None = None,
        fuzzy_classifier: FuzzyClassifier | None = None,
        model: str = "claude-sonnet-5",
    ) -> None:
        self._llm = llm
        self._quote_client = quote_client
        self._extractor = extractor
        self._fuzzy_classifier = fuzzy_classifier
        self._model = model
        self.state = AgentState(session=session)

    async def handle_turn(
        self, user_msg: str, *, media_type: str | None = None
    ) -> AgentTurn:
        """Processa mais um turno da conversa e decide a próxima resposta.

        Ordem de avaliação (gatilhos determinísticos primeiro, Q6): handoff
        já decidido > mídia essencial > pedido explícito de humano >
        fechamento/emissão > classificador fuzzy > fluxo de
        qualificação/confirmação/cotação.
        """
        state = self.state

        if state.handoff is not None:
            return AgentTurn(
                reply=(
                    "Você já está com um atendimento humano em andamento — "
                    "ele(a) vai continuar por aqui."
                ),
                handoff=state.handoff,
            )

        if handoff.is_media_essential(media_type):
            decision = handoff.for_media_unreadable(media_type)  # type: ignore[arg-type]
            state.handoff = decision
            return AgentTurn(reply=decision.message, handoff=decision)

        decision = (
            handoff.for_explicit_request(user_msg)
            or handoff.for_policy_issuance(user_msg)
            or handoff.classify_fuzzy(user_msg, self._fuzzy_classifier)
        )
        if decision is not None:
            state.handoff = decision
            return AgentTurn(reply=decision.message, handoff=decision)

        plano_mencionado = normalize_plano_mention(user_msg)
        if plano_mencionado is not None:
            state.plano_id = plano_mencionado

        if state.closed or state.quote_delivered:
            reply = await self._llm_reply(user_msg)
            return AgentTurn(
                reply=reply or "Fico à disposição se precisar de mais alguma coisa!"
            )

        if state.awaiting_confirmation:
            return await self._handle_confirmation_turn(user_msg)

        return await self._handle_qualification_turn(user_msg)

    async def _handle_qualification_turn(self, user_msg: str) -> AgentTurn:
        state = self.state
        state.session.process_turn(user_msg, llm_client=self._extractor)

        decision = handoff.for_clarify_loop_exhausted(state.session)
        if decision is not None:
            state.handoff = decision
            return AgentTurn(reply=decision.message, handoff=decision)

        if state.session.is_complete():
            state.awaiting_confirmation = True
            return AgentTurn(reply=build_confirmation_message(state.session.data))

        missing = state.session.missing_essential()
        question = build_missing_fields_question(
            missing, ask_data_inicio=not state.asked_data_inicio
        )
        state.asked_data_inicio = True
        return AgentTurn(reply=question)

    async def _handle_confirmation_turn(self, user_msg: str) -> AgentTurn:
        state = self.state

        if _CONFIRM_YES_RE.search(user_msg) and not _CONFIRM_NO_RE.search(user_msg):
            state.confirmed = True
            state.awaiting_confirmation = False
            return await self._quote_and_reply()

        # Não confirmou: trata como correção/dado novo, reprocessa e
        # re-pergunta a confirmação com os dados atualizados.
        state.session.process_turn(user_msg, llm_client=self._extractor)

        decision = handoff.for_clarify_loop_exhausted(state.session)
        if decision is not None:
            state.handoff = decision
            return AgentTurn(reply=decision.message, handoff=decision)

        if not state.session.is_complete():
            state.awaiting_confirmation = False
            missing = state.session.missing_essential()
            question = build_missing_fields_question(
                missing, ask_data_inicio=not state.asked_data_inicio
            )
            state.asked_data_inicio = True
            return AgentTurn(reply=question)

        return AgentTurn(reply=build_confirmation_message(state.session.data))

    async def _quote_and_reply(self) -> AgentTurn:
        state = self.state
        data = state.session.data
        plano_id = state.plano_id or "essencial"

        data_inicio = data.data_inicio
        fallback_note = None
        if not data_inicio:
            data_inicio = date.today().isoformat()
            fallback_note = (
                f"Não me disse a data de início, então considerei hoje "
                f"({data_inicio}) como data de início da vigência."
            )

        payload = {
            "plano_id": plano_id,
            "idade": data.idade,
            "veiculo_ano": data.veiculo_ano,
            "cep": data.cep,
            "data_inicio": data_inicio,
        }

        try:
            quote = await self.call_quote(payload)
        except CotacaoRecusada as exc:
            state.closed = True
            return AgentTurn(reply=format_refusal(exc), closed=True)
        except PayloadInvalido as exc:
            state.awaiting_confirmation = True
            return AgentTurn(reply=format_payload_invalido(exc))
        except QuoteUnavailable as exc:
            decision = handoff.for_quote_unavailable(exc)
            state.handoff = decision
            return AgentTurn(reply=decision.message, handoff=decision)
        except Exception as exc:  # fail-safe: nunca trava sem handoff (Q6)
            decision = handoff.for_agent_error(exc)
            state.handoff = decision
            return AgentTurn(reply=decision.message, handoff=decision)

        state.quote_delivered = True
        explanation = format_quote_explanation(quote, fallback_note=fallback_note)
        return AgentTurn(reply=explanation, quote=quote)

    async def call_quote(self, payload: dict[str, Any]) -> QuoteResult:
        """A tool única do domínio (`CALL_QUOTE_TOOL_SCHEMA`): cota via `QuoteClient`.

        Nunca inventa preço — propaga `CotacaoRecusada`/`PayloadInvalido`/
        `QuoteUnavailable` do `quote_client` (Group B) pro chamador decidir.
        """
        return await self._quote_client.cotar(payload)

    async def _llm_reply(self, user_msg: str) -> str:
        """Conversa livre (pós-cotação/pós-recusa) — única chamada real ao LLM.

        Nunca usado para decidir qualificação/confirmação/cotação/handoff
        (isso é sempre determinístico, ver docstring do módulo); serve só
        pra manter a conversa natural depois que o caminho já foi resolvido.
        """
        response = await self._llm.messages.create(
            model=self._model,
            max_tokens=400,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        return extract_reply_text(response)
