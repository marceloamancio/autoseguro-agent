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
from dataclasses import dataclass, field
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

# Teto de mensagens (lead+agente) mantidas na íntegra em `AgentState.history`
# -- só contexto pra conversa livre pós-cotação (`_llm_reply`). O que
# transborda esse teto NÃO é descartado: vira uma linha condensada em
# `AgentState.history_summary` (ver `_remember_turn`), então o contexto antigo
# continua chegando ao LLM sem deixar o payload crescer sem limite.
_MAX_HISTORY_MESSAGES = 20

# Resumo das mensagens que saíram da janela: cada uma vira uma linha
# `papel: texto truncado`. Condensação determinística de propósito -- sem
# chamada extra de LLM (que custaria token e poderia alucinar num prompt que
# fica ao lado dos fatos de preço); os fatos que realmente importam (a cotação
# real) chegam estruturados via `last_quote` e nunca dependem deste resumo.
#
# 20 verbatim + 40 resumidas = ~60 mensagens (~30 turnos) de contexto retido.
# Além disso as linhas mais antigas caem: perde-se texto de conversa, nunca o
# estado que carrega decisão (`last_quote`, `session.data`, `plano_id`).
_MAX_SUMMARY_LINES = 40
_SUMMARY_LINE_MAX_CHARS = 120


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

# Prompt da conversa livre (pós-cotação/pós-recusa). Deliberadamente
# diferente do `SYSTEM_PROMPT` de vendas: nesse ponto o LLM só redige texto —
# não cota (o preço vem sempre de `_quote_and_reply`) e não transborda (o
# handoff é decidido em `handoff.py`, antes daqui). Instruí-lo a prometer
# qualquer uma das duas coisas produzia mentira: ele dizia "vou te encaminhar
# pro consultor" sem que nenhum handoff fosse registrado.
FREE_CHAT_SYSTEM_PROMPT = """\
Você é um vendedor da AutoSeguro conversando por WhatsApp com um lead que já \
passou pela etapa de cotação.

O bloco "Fatos da cotação" abaixo é a SUA ÚNICA FONTE DE VERDADE. Ele vem \
direto da API de cotação. Você não tem nenhuma outra informação sobre este \
seguro, e seu conhecimento geral sobre seguros NÃO se aplica à AutoSeguro.

Você só pode afirmar estes cinco itens, e apenas copiando o valor exato dos \
fatos:
1. o plano cotado;
2. o prêmio mensal;
3. a franquia;
4. a lista de coberturas incluídas;
5. a carência (dias e quais coberturas) e o primeiro pagamento pró-rata.

É PROIBIDO afirmar qualquer coisa fora dessa lista. Em especial, nunca \
afirme, estime, calcule ou dê a entender: outro valor em reais (mesmo \
"aproximado", "a partir de" ou como exemplo); desconto, promoção ou margem \
de negociação; cobertura que não esteja na lista de coberturas dos fatos; \
carência, prazo ou data que não esteja nos fatos; forma de pagamento, \
parcelamento, cartão ou boleto; duração da vigência, reajuste, renovação ou \
cancelamento; regra de aceitação, exclusão ou franquia de outro plano.

Nunca recalcule nem estime um preço por conta própria — nem para outro \
plano, outra idade, outro veículo ou outro CEP. Você não faz contas: o preço \
existe só quando a API o devolve.

Se o lead perguntar qualquer coisa que não esteja nos fatos, responda \
literalmente que você não tem essa informação e que um consultor humano vai \
confirmar. Isso não é falha: é o comportamento correto. Preferir "não sei" a \
um palpite é a regra mais importante deste prompt.

Você NÃO consegue, por conta própria, recalcular a cotação, fechar negócio, \
emitir apólice ou transferir o atendimento — então nunca prometa nenhuma \
dessas ações. Se o lead pedir uma delas, apenas diga que é o próximo passo e \
deixe que o sistema cuide do encaminhamento.

Minimização de dados — nunca peça CPF, e-mail, telefone ou placa.\
"""

_NO_QUOTE_FACTS = "Nenhuma cotação foi calculada ainda nesta conversa."


def build_quote_facts(quote: QuoteResult | None) -> str:
    """Serializa a última cotação real como fatos pro prompt de papo livre.

    Sem isso o LLM não sabe que já cotou e alucina ("ainda não fizemos uma
    cotação") ou responde "não tenho essa informação" sobre uma cobertura que
    está na resposta real da `/quote`.
    """
    if quote is None:
        return (
            "Fatos da cotação: " + _NO_QUOTE_FACTS + " Portanto NENHUM valor em "
            "reais, cobertura, franquia ou prazo pode ser citado por você."
        )
    linhas = [
        f"- Plano cotado: {quote.plano_nome} ({quote.plano_id})",
        f"- Prêmio mensal: R$ {quote.premio_mensal:.2f}",
        f"- Franquia: R$ {quote.franquia:.2f}",
        f"- Coberturas incluídas: {', '.join(quote.coberturas)}",
    ]
    carencia = quote.carencia or {}
    if carencia.get("dias") is not None:
        coberturas = ", ".join(carencia.get("coberturas", []))
        linhas.append(f"- Carência: {carencia['dias']} dias para {coberturas}")
    pro_rata = quote.primeiro_pagamento_pro_rata or {}
    valor = pro_rata.get("valor_primeiro_pagamento")
    if valor is not None:
        linhas.append(
            f"- Primeiro pagamento (pró-rata): R$ {valor:.2f} "
            f"({pro_rata.get('dias_cobrados')} de {pro_rata.get('dias_no_mes')} dias)"
        )
    return "Fatos da cotação já entregue (a única fonte de valores):\n" + "\n".join(linhas)


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
    # Guarda anti-loop: o catálogo de planos é informado no máximo uma vez.
    informed_catalog: bool = False
    # 1.2 (P0-1): último payload que a /quote recusou com 400 -- nunca
    # reenviado idêntico (ver `_quote_and_reply`).
    last_invalid_payload: dict[str, Any] | None = None
    last_invalid_detalhe: str | None = None
    # Histórico turno a turno (lead/agente), usado só para dar contexto à
    # conversa livre pós-cotação (`_llm_reply`) -- nunca lido pelo caminho
    # determinístico de qualificação/cotação/handoff. Mantém as
    # `_MAX_HISTORY_MESSAGES` mais recentes na íntegra; o que transborda vai
    # condensado pra `history_summary` em vez de ser jogado fora.
    history: list[dict[str, str]] = field(default_factory=list)
    history_summary: list[str] = field(default_factory=list)
    # Última cotação REAL entregue (resposta 200 da /quote). É a única fonte
    # de valores monetários que a conversa livre pode citar -- ver
    # `guard_no_fabricated_price`.
    last_quote: QuoteResult | None = None


@dataclass
class AgentTurn:
    """Resultado de processar um turno — o que responder ao lead + sinais."""

    reply: str
    handoff: HandoffDecision | None = None
    quote: QuoteResult | None = None
    closed: bool = False
    # True quando o guard descartou a resposta do LLM por conter um valor
    # monetário que não veio da `/quote` (auditável no trace, ver `cli.py`).
    llm_reply_blocked: bool = False


PLANOS_CATALOGO: tuple[str, ...] = ("essencial", "completo", "premium")


def normalize_plano_mention(text: str) -> str | None:
    """Detecta menção a um dos 3 planos do catálogo no texto do lead."""
    match = re.search(r"\b(essencial|completo|premium)\b", text, re.IGNORECASE)
    return match.group(1).lower() if match else None


# "plano ouro master", "plano gold", "plano top" -- nome de plano que o lead
# inventa/lembra errado. Captura só as duas primeiras palavras depois de
# "plano" (nomes reais são curtos) e ignora conectivos que indicam que a
# palavra seguinte não é um nome ("plano de saúde" é outro assunto, tratado
# por `classify_scope`).
_PLANO_MENTION_RE = re.compile(
    r"\bplano\s+((?!de\b|do\b|da\b|que\b|mais\b)[\wçãáéíóúâêô]+(?:\s+[\wçãáéíóúâêô]+)?)",
    re.IGNORECASE,
)


def detect_unknown_plano(text: str) -> str | None:
    """Nome de plano citado pelo lead que não existe no catálogo.

    Devolve `None` quando o lead cita um plano válido (ou nenhum). Existe
    porque `normalize_plano_mention` devolvendo `None` era indistinguível de
    "não falou de plano": o `_quote_and_reply` caía no default `essencial` e
    cotava **o plano mais barato em silêncio** para quem pediu "plano ouro
    master" (achado `b24` da bateria adversarial).
    """
    if normalize_plano_mention(text) is not None:
        return None
    match = _PLANO_MENTION_RE.search(text)
    if not match:
        return None
    return " ".join(match.group(1).split()).strip(" ,.!?")


def format_plano_catalog(plano_mencionado: str) -> str:
    """Informa que o plano citado não existe e lista os 3 do catálogo.

    O agente resolve isso sozinho — nunca transborda (ver
    `handoff.resolve_plan_not_in_catalog`).
    """
    return (
        f"Não temos um plano chamado \"{plano_mencionado}\". Os planos "
        f"disponíveis são: Essencial, Completo e Premium. Qual deles você "
        f"quer que eu cote?"
    )


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


_ROLE_LABELS = {"user": "lead", "assistant": "agente"}


def _summarize_message(message: dict[str, str]) -> str:
    """Condensa uma mensagem que saiu da janela numa linha `papel: texto`."""
    role = _ROLE_LABELS.get(message.get("role", ""), message.get("role", "?"))
    content = " ".join((message.get("content") or "").split())
    if len(content) > _SUMMARY_LINE_MAX_CHARS:
        content = content[: _SUMMARY_LINE_MAX_CHARS - 1].rstrip() + "…"
    return f"{role}: {content}"


# ---------------------------------------------------------------------------
# Fronteira do preço: o LLM nunca emite um valor que não veio de um 200 da
# `/quote`. Garantido aqui, deterministicamente -- não por instrução de prompt.
# ---------------------------------------------------------------------------

# Valor monetário prefixado por `R$`: aceita `137.88`, `219,40`, `3.200,00`,
# `3000`. Só o que vem com `R$` é fiscalizado -- "30 dias" ou "17 de 31 dias"
# passam livres.
_MONEY_RE = re.compile(r"R\$\s*([\d.,]+)")


def _money_to_cents(raw: str) -> int | None:
    """Normaliza um literal monetário brasileiro/ASCII para centavos.

    Trata o separador decimal como os 2 últimos dígitos quando precedidos de
    `,` ou `.`; qualquer outro `.`/`,` é separador de milhar. Devolve `None`
    quando o literal não é um número reconhecível.
    """
    cleaned = raw.strip().rstrip(".,")
    if not cleaned or not any(ch.isdigit() for ch in cleaned):
        return None
    if len(cleaned) >= 3 and cleaned[-3] in ".," and cleaned[-2:].isdigit():
        inteiro, centavos = cleaned[:-3], cleaned[-2:]
    else:
        inteiro, centavos = cleaned, "00"
    digits = re.sub(r"\D", "", inteiro)
    if not digits:
        return None
    return int(digits) * 100 + int(centavos)


def extract_money_cents(text: str) -> set[int]:
    """Todos os valores `R$` do texto, normalizados em centavos."""
    found = (_money_to_cents(match) for match in _MONEY_RE.findall(text))
    return {cents for cents in found if cents is not None}


def allowed_money_cents(quote: QuoteResult | None) -> set[int]:
    """Valores que o agente PODE citar: os da última cotação real.

    Sem cotação entregue, o conjunto é vazio — nenhum valor monetário é
    legítimo, porque nenhum preço real existe ainda.
    """
    if quote is None:
        return set()
    valores = [quote.premio_mensal, quote.franquia]
    pro_rata = quote.primeiro_pagamento_pro_rata or {}
    valor_primeiro = pro_rata.get("valor_primeiro_pagamento")
    if valor_primeiro is not None:
        valores.append(valor_primeiro)
    return {round(float(v) * 100) for v in valores if v is not None}


NO_QUOTE_YET_REPLY = (
    "Ainda não tenho um valor calculado pra você — o preço só sai depois que "
    "eu rodo a cotação no sistema. Quer que eu faça isso agora?"
)


def guard_no_fabricated_price(
    reply: str, quote: QuoteResult | None
) -> tuple[str, bool]:
    """Descarta a resposta do LLM se ela citar um preço que a `/quote` não deu.

    É esta função — e não o prompt — que sustenta o "nunca inventa preço" na
    conversa livre. O LLM já não calcula o preço (`call_quote` nunca é
    exposto como tool); aqui fechamos a outra metade da fronteira: ele também
    não pode *escrever* um valor monetário que não venha de uma resposta 200
    real. Ao detectar um valor fora do allow-list, a resposta inteira é
    descartada em favor de um recap determinístico da cotação real (ou de um
    aviso honesto, quando ainda não há cotação).

    Devolve `(reply_segura, blocked)`.
    """
    citados = extract_money_cents(reply)
    if not citados:
        return reply, False
    if citados <= allowed_money_cents(quote):
        return reply, False
    if quote is None:
        return NO_QUOTE_YET_REPLY, True
    return format_quote_explanation(quote), True


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
        """Processa mais um turno da conversa e registra no histórico.

        Delega a `_handle_turn` para a lógica de decisão e só cuida de
        acumular o histórico (lead/agente) ao final -- é ele que dá contexto
        à conversa livre pós-cotação (`_llm_reply`); o caminho determinístico
        de qualificação/cotação/handoff nunca o lê.
        """
        turn = await self._handle_turn(user_msg, media_type=media_type)
        self._remember_turn(user_msg, turn.reply)
        return turn

    def _remember_turn(self, user_msg: str, reply: str) -> None:
        """Anexa o turno ao histórico, condensando o que sair da janela.

        As `_MAX_HISTORY_MESSAGES` mensagens mais recentes ficam na íntegra;
        as que transbordam viram uma linha em `history_summary` (truncada,
        determinística) -- assim uma conversa longa não perde o contexto
        antigo, só a literalidade dele.
        """
        state = self.state
        state.history.append({"role": "user", "content": user_msg})
        state.history.append({"role": "assistant", "content": reply})

        overflow = len(state.history) - _MAX_HISTORY_MESSAGES
        if overflow <= 0:
            return

        dropped = state.history[:overflow]
        del state.history[:overflow]
        state.history_summary.extend(_summarize_message(m) for m in dropped)
        del state.history_summary[: max(0, len(state.history_summary) - _MAX_SUMMARY_LINES)]

    async def _handle_turn(
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

        if result.extractor_failed:
            # LLM de extração fora do ar: infra nossa, não culpa do lead.
            # Escala IMEDIATAMENTE -- sem isso, a sessão acumulava turnos
            # estagnados e acabava em `clarify_loop_exhausted`, pedindo duas
            # vezes dados que o lead já tinha mandado.
            decision = handoff.for_extractor_unavailable(
                result.extractor_error or RuntimeError("extractor indisponível")
            )
            state.handoff = decision
            return AgentTurn(reply=decision.message, handoff=decision)

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

        before_plano = state.plano_id
        plano_mencionado = normalize_plano_mention(user_msg)
        if plano_mencionado is not None:
            state.plano_id = plano_mencionado
        plano_changed = state.plano_id != before_plano

        # Plano fora do catálogo: informa os 3 disponíveis em vez de cotar o
        # `essencial` (default) calado. Resolve sozinho, nunca transborda
        # (`handoff.resolve_plan_not_in_catalog`). Só quando o lead ainda não
        # escolheu um plano válido -- "quero o completo, não o ouro" não deve
        # parar o fluxo.
        if state.plano_id is None and not state.informed_catalog:
            desconhecido = detect_unknown_plano(user_msg)
            if desconhecido is not None:
                state.informed_catalog = True
                return AgentTurn(reply=format_plano_catalog(desconhecido))

        if state.closed or state.quote_delivered:
            if state.quote_delivered and (
                intent == "requote" or essential_changed or plano_changed
            ):
                # 1.3 (P0-3): re-cotar após entrega -- reabre a cotação com o
                # dado novo em vez de engolir o pedido no papo livre. Trocar de
                # plano conta: sem isso, "e se eu escolher o completo?" caía no
                # papo livre e o LLM inventava a cotação inteira (preço,
                # franquia, coberturas) sem nunca chamar a `/quote`.
                state.quote_delivered = False
                state.confirmed = False
                return await self._quote_and_reply()
            reply, blocked = await self._llm_reply(user_msg)
            return AgentTurn(
                reply=reply or "Fico à disposição se precisar de mais alguma coisa!",
                llm_reply_blocked=blocked,
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
        state.last_quote = quote
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

    def _build_free_chat_system(self) -> str:
        """Prompt de papo livre + fatos da cotação real + resumo do que saiu
        da janela de histórico."""
        blocos = [FREE_CHAT_SYSTEM_PROMPT, build_quote_facts(self.state.last_quote)]
        if self.state.history_summary:
            blocos.append(
                "Resumo do trecho mais antigo da conversa:\n"
                + "\n".join(self.state.history_summary)
            )
        return "\n\n".join(blocos)

    async def _llm_reply(self, user_msg: str) -> tuple[str, bool]:
        """Conversa livre (pós-cotação/pós-recusa) — única chamada real ao LLM.

        Nunca usado para decidir qualificação/confirmação/cotação/handoff
        (isso é sempre determinístico, ver docstring do módulo); serve só
        pra manter a conversa natural depois que o caminho já foi resolvido.

        Manda o histórico acumulado (`state.history`, mais o resumo do que
        transbordou) e os fatos da cotação real -- sem isso o LLM não sabe
        que já cotou e alucina diante de uma objeção de preço. A saída passa
        obrigatoriamente por `guard_no_fabricated_price`: o LLM pode redigir,
        nunca precificar.

        Devolve `(reply, blocked)`.
        """
        messages = [*self.state.history, {"role": "user", "content": user_msg}]
        response = await self._llm.messages.create(
            model=self._model,
            max_tokens=400,
            system=self._build_free_chat_system(),
            messages=messages,
        )
        return guard_no_fabricated_price(
            extract_reply_text(response), self.state.last_quote
        )
