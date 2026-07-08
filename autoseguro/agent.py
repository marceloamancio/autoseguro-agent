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

- `call_quote` é o **único** ponto de entrada do domínio para a `/quote`,
  invocado pelo fluxo determinístico assim que a qualificação é confirmada.
  Deliberadamente NÃO é registrado como tool do LLM: o LLM não tem acesso ao
  caminho que produz preço — é essa fronteira, garantida por arquitetura e
  não por prompt, que sustenta o "nunca inventa preço".
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
from .extraction import ExtractedData, ExtractionResult, LlmExtractorClient, QualificationSession
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

# P1-5: `marca`/`modelo` são campos de rapport de baixo controle (vêm direto
# do texto do lead, nunca validados) e são ecoados de volta na mensagem de
# confirmação -- sanitiza antes de ecoar (remove caracteres de controle e
# marcadores óbvios de instrução/injeção, limita o tamanho). Nomes reais de
# marca/modelo são curtos; qualquer payload de injeção plausível excede o
# limite e vem cortado.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")
_INSTRUCTION_MARKER_RE = re.compile(
    r"</?system[^>]*>|ignore\s+(todas\s+)?as\s+instru\w*|novo_prompt|"
    r"instru\w*\s+anteriores",
    re.IGNORECASE,
)
_MAX_FREE_TEXT_FIELD_LEN = 30


def _sanitize_free_text_field(value: str | None) -> str | None:
    """Sanitiza um campo de rapport (marca/modelo) antes de ecoar ao lead."""
    if not value:
        return value
    text = _CONTROL_CHARS_RE.sub("", value)
    text = _INSTRUCTION_MARKER_RE.sub("", text)
    text = text.strip()[:_MAX_FREE_TEXT_FIELD_LEN].strip()
    return text or None


def _essential_snapshot(data: ExtractedData) -> tuple[Any, Any, Any]:
    """Retrato dos 3 campos essenciais -- usado pra detectar mudança real de
    dado entre turnos (1.1/1.3), independente do sinal de `intent`."""
    return (data.idade, data.veiculo_ano, data.cep)

# Perguntas que o agente pode fazer para qualificar — de propósito, restritas
# aos únicos 3 campos essenciais que a `/quote` usa (Q3, minimização: nunca
# CPF/e-mail/telefone/placa).
FIELD_QUESTIONS: dict[str, str] = {
    "idade": "sua idade",
    "veiculo_ano": "o ano do seu veículo",
    "cep": "o CEP onde o carro fica",
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
    # 1.2 (P0-1): último payload que a /quote recusou com 400 -- nunca
    # reenviado idêntico (ver `_quote_and_reply`).
    last_invalid_payload: dict[str, Any] | None = None
    last_invalid_detalhe: str | None = None


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
    """Resume o que foi entendido e pede confirmação antes de cotar (DEC-5).

    `marca`/`modelo` são sanitizados antes de ecoar (P1-5) -- nunca voltam
    crus, mesmo que carreguem texto de instrução/injeção.
    """
    marca = _sanitize_free_text_field(data.marca)
    modelo = _sanitize_free_text_field(data.modelo)
    veiculo_partes = [p for p in (marca, modelo) if p]
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
    """Pede o dado faltante/errado apontado pela API (400) — sem handoff.

    Menciona `data_inicio` como possível causa (1.2 / P0-1) -- a data é um
    campo tão sujeito a erro de formato quanto idade/ano/CEP, mas antes não
    era citada aqui, então o lead nunca sabia que precisava corrigi-la.
    """
    return (
        f"Preciso ajustar um dado antes de cotar: {exc.detalhe} Pode "
        "confirmar de novo sua idade, o ano do veículo, o CEP e a data de "
        "início (se informou uma, no formato dd/mm/aaaa)?"
    )


def format_payload_invalido_repetido(detalhe: str | None) -> str:
    """Payload já recusado (400) chegou idêntico de novo (1.2 / P0-1).

    Nunca reenvia o mesmo payload pra `/quote` -- pede especificamente o
    campo mais provável de estar malformado (`data_inicio`) em vez de repetir
    a mesma pergunta genérica, que travaria a conversa em loop.
    """
    return (
        "Ainda não consegui fechar a cotação com os dados que você "
        f"confirmou (motivo: {detalhe or 'dado inválido'}). Pra eu não ficar "
        "tentando de novo do mesmo jeito: pode me confirmar especificamente "
        "a data de início que você quer, no formato dd/mm/aaaa?"
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
        já decidido > mídia essencial > extração do turno (2.1: já vem com
        o sinal de `intent`, zero chamada extra de LLM) > escopo/pedido
        explícito de humano > fechamento/emissão > fluxo de
        qualificação/confirmação/cotação/re-cotação.
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

        # Extração única do turno (2.1): a MESMA chamada que extrai os dados
        # de cotação já classifica a intenção -- é o sinal que alimenta
        # tanto a decisão de escopo abaixo quanto a confirmação/re-cotação
        # mais adiante (1.1/1.3), sem nenhuma chamada extra de LLM.
        before_essential = _essential_snapshot(state.session.data)
        result = state.session.process_turn(user_msg, llm_client=self._extractor)
        intent = result.data.intent
        essential_changed = _essential_snapshot(state.session.data) != before_essential

        if result.contradiction:
            # 2.2 (P1-4): sobrescrita materialmente diferente de um
            # essencial (ex.: idade 35 -> 90) -- autoridade, não capacidade;
            # nunca resolvido sozinho (`QualificationSession.process_turn`).
            decision = handoff.for_contradictory_data(user_msg)
            state.handoff = decision
            return AgentTurn(reply=decision.message, handoff=decision)

        decision = handoff.for_policy_issuance(user_msg) or handoff.classify_scope(
            intent, user_msg, self._fuzzy_classifier
        )
        if decision is not None:
            state.handoff = decision
            return AgentTurn(reply=decision.message, handoff=decision)

        plano_mencionado = normalize_plano_mention(user_msg)
        if plano_mencionado is not None:
            state.plano_id = plano_mencionado

        if state.closed or state.quote_delivered:
            if state.quote_delivered and (intent == "requote" or essential_changed):
                # 1.3 (P0-3): re-cotar após entrega -- reabre a cotação com
                # o dado novo em vez de engolir o pedido no papo livre.
                state.quote_delivered = False
                state.confirmed = False
                return await self._quote_and_reply()
            reply = await self._llm_reply(user_msg)
            return AgentTurn(
                reply=reply or "Fico à disposição se precisar de mais alguma coisa!"
            )

        if state.awaiting_confirmation:
            return await self._handle_confirmation_turn(user_msg, result, essential_changed)

        return await self._handle_qualification_turn(result)

    async def _handle_qualification_turn(self, result: ExtractionResult) -> AgentTurn:
        state = self.state

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

    async def _handle_confirmation_turn(
        self, user_msg: str, result: ExtractionResult, essential_changed: bool
    ) -> AgentTurn:
        """Confirmação: extrai-então-diferencia (1.1/P0-2).

        A extração (com `intent`) já rodou em `handle_turn`, antes de checar
        "sim" -- nunca cota com o dado velho quando o "sim" vem com uma
        correção embutida (`intent == "correct"` ou um essencial mudou de
        verdade): re-confirma com o dado atualizado em vez de cotar direto.
        """
        state = self.state
        intent = result.data.intent

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

        # Correção embutida (sinal de intenção ou dado essencial mudou de
        # verdade): nunca cota com o valor velho -- re-confirma com o
        # atualizado em vez de seguir pro "sim" cru (P0-2).
        if intent == "correct" or essential_changed:
            return AgentTurn(reply=build_confirmation_message(state.session.data))

        confirmed_clean = intent == "confirm" or (
            intent not in ("reject", "correct")
            and _CONFIRM_YES_RE.search(user_msg)
            and not _CONFIRM_NO_RE.search(user_msg)
        )
        if confirmed_clean:
            state.confirmed = True
            state.awaiting_confirmation = False
            return await self._quote_and_reply()

        # Nem confirmação limpa nem correção reconhecida (ex.: "não" puro,
        # ou intent == "reject"): repete a confirmação com os dados atuais.
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

        # 1.2 (P0-1): este EXATO payload já levou 400 antes -- nunca reenvia
        # igual esperando um resultado diferente (é assim que o loop
        # infinito acontecia). Pede o campo ofensor em vez de martelar a API.
        if state.last_invalid_payload == payload:
            state.awaiting_confirmation = True
            return AgentTurn(
                reply=format_payload_invalido_repetido(state.last_invalid_detalhe)
            )

        try:
            quote = await self.call_quote(payload)
        except CotacaoRecusada as exc:
            state.closed = True
            return AgentTurn(reply=format_refusal(exc), closed=True)
        except PayloadInvalido as exc:
            state.awaiting_confirmation = True
            state.last_invalid_payload = payload
            state.last_invalid_detalhe = exc.detalhe
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
        state.last_invalid_payload = None
        state.last_invalid_detalhe = None
        explanation = format_quote_explanation(quote, fallback_note=fallback_note)
        return AgentTurn(reply=explanation, quote=quote)

    async def call_quote(self, payload: dict[str, Any]) -> QuoteResult:
        """Único ponto de entrada do domínio para a `/quote`: cota via `QuoteClient`.

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
