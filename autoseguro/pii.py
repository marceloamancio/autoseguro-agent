"""Mascaramento de PII **at-rest** (logs/trace/histórico) — Group C, DEC-4 (Q3).

Threat-model (ver `DECISOES.md`, Q3): o agente principal é o próprio Sonnet e
precisa ler o texto do lead pra qualificar — PII inevitavelmente passa pelo
modelo no caminho quente. Este módulo **não** protege o trânsito ao vivo; ele
protege o que fica **em repouso** (histórico/dataset, `trace.jsonl`, logs, o
log de execução entregue).

Design em duas camadas, aplicadas em lote (nunca por mensagem no hot path):

1. **Regex simples e certeiro** (`redact_text`): substitui os padrões óbvios e
   bem-formados (CPF, e-mail, telefone, placa, CEP) por marcadores
   (`⟨CPF⟩`, `⟨EMAIL⟩`, `⟨TELEFONE⟩`, `⟨PLACA⟩`, `⟨CEP⟩`). Alta **precisão**;
   deliberadamente **não exaustivo** — não tenta cobrir toda variação de
   formato (isso fica pra camada 2). Formatos calibrados a partir do gerador
   de referência do desafio
   (`namastex-fde-challenge/scripts/generate_dataset.py`).
2. **Varredura LLM em lote** (`llm_sweep`): pega o que o regex simples deixou
   passar — nome de terceiro, formato exótico, e categorias **adicionais**
   (RG, endereço, data de nascimento...). Roda em lote, é **mockável** (o
   `client` é qualquer callable `(texts, categories) -> texts`) e
   **desligada por padrão**: sem `client`, é no-op — nunca faz chamada de
   rede, então roda sem `ANTHROPIC_API_KEY`.

Minimização de coleta (o resto do DEC-4) é comportamento do agente
(Group E): ele não pede CPF/e-mail/telefone/placa. Este módulo só cobre o que
sobrar caso o lead mande PII espontaneamente.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

# ---------------------------------------------------------------------------
# Camada 1 — regex simples e certeiro
# ---------------------------------------------------------------------------

# CPF: xxx.xxx.xxx-xx (formato padrão brasileiro, com pontuação).
CPF_RE = re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b")

# E-mail: local@domínio, formato usual (o gerador varia separador . _ ou nada,
# e sufixo numérico opcional no local-part — tudo coberto por \w).
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")

# Telefone celular BR com DDI: +55 DD 9dddd-dddd (formato do gerador; o "9"
# inicial do nono dígito faz parte do bloco de 5 dígitos antes do hífen).
TELEFONE_RE = re.compile(r"\+55\s*\d{2}\s*9\d{4}-\d{4}")

# Placa Mercosul: 3 letras + 1 dígito + 1 letra + 2 dígitos, sem separador.
PLACA_RE = re.compile(r"\b[A-Za-z]{3}\d[A-Za-z]\d{2}\b")

# CEP: 5 dígitos + hífen + 3 dígitos.
CEP_RE = re.compile(r"\b\d{5}-\d{3}\b")

# ---------------------------------------------------------------------------
# 2.4 (P1-1) — ampliação de recall: formatos plausíveis e comuns em chat real
# que o regex calibrado (acima) contra o formato exato do gerador não pega.
# Este módulo é **at-rest** (Q3) — over-masking é aceitável, então o viés
# aqui é deliberadamente pra **recall**, não precisão: preferimos mascarar um
# número comum demais a deixar PII real vazar em claro no log entregue.
# ---------------------------------------------------------------------------

# Telefone móvel cru (sem nenhuma formatação): DDD (2 dígitos) + "9" do nono
# dígito + 8 dígitos = 11 dígitos corridos, ex. "11912345678".
#
# Ambiguidade documentada: um CPF cru (11 dígitos, ver `CPF_CRU_RE` abaixo)
# cujo 3º dígito por acaso seja "9" também bate neste padrão e sai mascarado
# como TELEFONE em vez de CPF. Aceitável: at-rest, over-masking é o viés
# desejado — o que importa é que a PII (seja qual for a categoria) não
# vaza, não que o marcador acerte a categoria exata. Este regex roda antes
# de `CPF_CRU_RE` na ordem de aplicação, então qualquer sequência de 11
# dígitos com "9" na 3ª posição é sempre atribuída a TELEFONE primeiro.
TELEFONE_CRU_RE = re.compile(r"\b\d{2}9\d{8}\b")

# Telefone móvel com DDD entre parênteses (com/sem espaço) e hífen antes dos
# últimos 4 dígitos: "(11) 91234-5678".
TELEFONE_PARENTESES_RE = re.compile(r"\(\d{2}\)\s*9\d{4}-\d{4}")

# Telefone fixo: 4 dígitos + hífen + 4 dígitos, sem o "9" inicial do celular
# (com ou sem DDD entre parênteses antes — o parêntese em si não precisa
# fazer parte do match, só os 8 dígitos com hífen já bastam pra mascarar).
TELEFONE_FIXO_RE = re.compile(r"\b\d{4}-\d{4}\b")

# CPF cru (11 dígitos sem pontuação nenhuma), ex. "12345678901". Roda depois
# dos padrões de telefone acima (ver ambiguidade documentada em
# `TELEFONE_CRU_RE`) — só sobra pra CPF o que não tiver a marca do "9" na 3ª
# posição nem já foi consumido por um formato de telefone.
CPF_CRU_RE = re.compile(r"\b\d{11}\b")

# Placa antiga (pré-Mercosul): 3 letras + 4 dígitos, sem separador, ex.
# "ABC1234" — ainda circulando aos milhões. Não se sobrepõe ao padrão
# Mercosul (`PLACA_RE`, que exige uma letra na 5ª posição): aqui as 4
# posições finais são todas dígitos.
PLACA_ANTIGA_RE = re.compile(r"\b[A-Za-z]{3}\d{4}\b")

# CEP colado, sem hífen: 8 dígitos direto, ex. "01310100".
CEP_CRU_RE = re.compile(r"\b\d{8}\b")

MARKERS = {
    "cpf": "⟨CPF⟩",
    "email": "⟨EMAIL⟩",
    "telefone": "⟨TELEFONE⟩",
    "placa": "⟨PLACA⟩",
    "cep": "⟨CEP⟩",
}

# Ordem de aplicação: CPF/telefone/email antes de placa/cep — os padrões não
# se sobrepõem (dígitos com pontuação distinta vs. letras+dígitos), mas a
# ordem deixa explícito que os formatos "mais específicos" (com prefixo/
# pontuação characteristic, como +55 ou pontos de CPF) são checados primeiro.
#
# Os formatos "de recall" (2.4 / P1-1) vêm depois dos formatos exatos do
# gerador, e entre si na ordem: variantes de telefone antes de `CPF_CRU_RE`
# (mesmo comprimento, 11 dígitos — ver ambiguidade documentada acima do
# `TELEFONE_CRU_RE`), e placa/CEP crus por último (não competem com nada).
_REGEX_MARKER_ORDER: tuple[tuple[re.Pattern[str], str], ...] = (
    (CPF_RE, MARKERS["cpf"]),
    (TELEFONE_RE, MARKERS["telefone"]),
    (EMAIL_RE, MARKERS["email"]),
    (PLACA_RE, MARKERS["placa"]),
    (CEP_RE, MARKERS["cep"]),
    (TELEFONE_CRU_RE, MARKERS["telefone"]),
    (TELEFONE_PARENTESES_RE, MARKERS["telefone"]),
    (TELEFONE_FIXO_RE, MARKERS["telefone"]),
    (CPF_CRU_RE, MARKERS["cpf"]),
    (PLACA_ANTIGA_RE, MARKERS["placa"]),
    (CEP_CRU_RE, MARKERS["cep"]),
)


def redact_text(text: str) -> str:
    """Aplica o regex simples e certeiro, substituindo PII óbvia por marcadores.

    Não faz nenhuma chamada externa; puro e determinístico. Alta precisão —
    só mascara o que bate exatamente com os formatos conhecidos (CPF, e-mail,
    telefone, placa, CEP). Não exaustivo por design: o restante (nomes de
    terceiros, formatos exóticos, categorias adicionais) é responsabilidade
    da varredura LLM em lote (`llm_sweep`).
    """
    if not text:
        return text
    out = text
    for pattern, marker in _REGEX_MARKER_ORDER:
        out = pattern.sub(marker, out)
    return out


def redact_record(
    record: dict,
    text_fields: Sequence[str] = ("message_body",),
) -> dict:
    """Aplica `redact_text` aos campos de texto configurados de um registro.

    Uso típico: mascarar `message_body` de uma linha de `trace.jsonl`/log
    antes de gravar em disco. Não muta o `record` recebido — devolve uma
    cópia rasa com os campos de texto substituídos.
    """
    out = dict(record)
    for field_name in text_fields:
        value = out.get(field_name)
        if isinstance(value, str):
            out[field_name] = redact_text(value)
    return out


# ---------------------------------------------------------------------------
# Camada 2 — varredura LLM em lote (mockável, desligável)
# ---------------------------------------------------------------------------

# Categorias obrigatórias que todo prompt de varredura deve cobrir (as mesmas
# do regex simples, camada 1) — a varredura LLM as reforça e ainda "pede
# adicionais" (ver `build_sweep_prompt`).
MANDATORY_CATEGORIES: tuple[str, ...] = ("CPF", "EMAIL", "TELEFONE", "PLACA", "CEP")

_ADDITIONAL_CATEGORIES_HINT = (
    "Além dessas categorias obrigatórias, procure e marque também PII adicional "
    "que aparecer no texto (por exemplo: RG, endereço, data de nascimento, nome "
    "de terceiro), usando um marcador ⟨CATEGORIA⟩ apropriado para cada achado."
)

# Contrato do client de varredura: um callable síncrono que recebe o lote de
# textos e a lista de categorias, e devolve o lote já mascarado (mesma ordem/
# tamanho). Isso é o que torna `llm_sweep` mockável em teste (passe uma
# função simples) e plugável em produção (passe um adaptador fino sobre o
# client real da Anthropic).
LlmSweepClient = Callable[[Sequence[str], Sequence[str]], Sequence[str]]


def build_sweep_prompt(
    texts: Sequence[str],
    categories: Sequence[str] = MANDATORY_CATEGORIES,
) -> str:
    """Monta (sem chamar rede) o prompt da varredura LLM em lote.

    Mostra as categorias **obrigatórias** e pede pra LLM caçar PII
    **adicional** que o regex simples deixou passar. Só monta a string —
    quem for plugar um client real decide como enviá-la (ex.: como mensagem
    de usuário pro Sonnet com structured output). Não é chamado nos testes
    além de verificar seu conteúdo — não faz nenhuma requisição.
    """
    joined_categories = ", ".join(categories)
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    return (
        "Você é um redator de PII operando em lote, at-rest (nunca no caminho "
        f"quente da conversa). Mascare as categorias obrigatórias: {joined_categories}. "
        f"{_ADDITIONAL_CATEGORIES_HINT}\n\nTextos:\n{numbered}"
    )


def llm_sweep(
    texts: Sequence[str],
    client: Optional[LlmSweepClient] = None,
    categories: Sequence[str] = MANDATORY_CATEGORIES,
) -> list[str]:
    """Varredura LLM em lote — mockável e desligável.

    - **Desligada por padrão:** sem `client` (default `None`), é **no-op** —
      devolve `texts` inalterado e não faz nenhuma chamada de rede. Isso
      garante que roda sem `ANTHROPIC_API_KEY` e nunca no caminho quente.
    - **Mockável:** `client` é qualquer callable `(texts, categories) ->
      texts_mascarados`, chamado **uma única vez em lote** (não por
      mensagem) — em teste, um mock simples; em produção, um adaptador fino
      sobre o client real da Anthropic (fora de escopo aqui — Group C só
      entrega a interface).
    - Recebe as categorias obrigatórias (`MANDATORY_CATEGORIES` por padrão) e
      as repassa ao `client`, que também é instruído (via `build_sweep_prompt`,
      se usado por trás do adaptador real) a caçar categorias adicionais.
    """
    if not texts:
        return list(texts)
    if client is None:
        return list(texts)
    return list(client(texts, categories))


# ---------------------------------------------------------------------------
# Fachada — combina as duas camadas
# ---------------------------------------------------------------------------


@dataclass
class PiiRedactor:
    """Redator de PII at-rest: regex simples e certeiro + varredura LLM em lote.

    `llm_client` é opcional e desligado por padrão (`None`): sem ele, a
    varredura em lote é no-op e só o regex roda — determinístico, sem rede,
    sem chave. Passe um `llm_client` (mock em teste; adaptador real em
    produção) pra habilitar a camada 2.
    """

    llm_client: Optional[LlmSweepClient] = None
    categories: Sequence[str] = field(default_factory=lambda: MANDATORY_CATEGORIES)

    def redact_text(self, text: str) -> str:
        """Mascara um único texto usando só o regex simples (camada 1)."""
        return redact_text(text)

    def redact_record(
        self,
        record: dict,
        text_fields: Sequence[str] = ("message_body",),
    ) -> dict:
        """Mascara os campos de texto de um registro (linha de trace/log)."""
        return redact_record(record, text_fields=text_fields)

    def redact_batch(self, texts: Sequence[str]) -> list[str]:
        """Mascara um lote de textos: regex simples primeiro, depois a
        varredura LLM em lote (no-op se `llm_client` não estiver configurado).

        Uso pretendido: processamento em lote/at-rest do histórico, `trace.jsonl`
        ou dataset — nunca por mensagem no caminho quente da conversa.
        """
        pre_masked = [redact_text(t) for t in texts]
        return llm_sweep(pre_masked, client=self.llm_client, categories=self.categories)


# ---------------------------------------------------------------------------
# Adaptador real da varredura LLM (2.4 / P1-1) — SDK Anthropic, lazy import
# ---------------------------------------------------------------------------

# Uma linha por texto, numerada como `build_sweep_prompt` apresenta o lote de
# entrada (ex.: "1. texto mascarado") — é o formato que pedimos de volta pra
# conseguir realinhar a resposta com o lote original.
_NUMBERED_LINE_RE = re.compile(r"^\s*(\d+)\.\s?(.*)$")


def _parse_swept_response(raw_text: str, expected_count: int) -> Optional[list[str]]:
    """Recompõe a resposta numerada da varredura de volta numa lista alinhada.

    Devolve `None` (nunca levanta) quando a resposta não numerou exatamente
    `expected_count` linhas — resposta em formato inesperado é tratada como
    "a varredura não ajudou", não como erro fatal; quem chama decide o
    fallback (`AnthropicSweepClient.__call__` devolve os textos originais).
    """
    parsed: dict[int, str] = {}
    for line in raw_text.splitlines():
        match = _NUMBERED_LINE_RE.match(line)
        if match:
            parsed[int(match.group(1))] = match.group(2)
    if len(parsed) != expected_count:
        return None
    return [parsed[i] for i in range(1, expected_count + 1)]


class AnthropicSweepClient:
    """`LlmSweepClient` real: adaptador fino sobre `build_sweep_prompt` +
    o SDK da Anthropic.

    Implementa o contrato de `LlmSweepClient` (`__call__(texts, categories)
    -> texts_mascarados`) — plugável direto em `PiiRedactor(llm_client=...)`.
    Monta o prompt com `build_sweep_prompt` (já existente), chama a API
    **uma única vez em lote** e realinha a resposta numerada de volta à lista
    de entrada.

    Import do SDK é **lazy** (dentro do `__init__`, só quando `client` não é
    injetado) — mesmo padrão de `anthropic_extractor.AnthropicExtractor`.
    Nunca instanciado nos testes: eles injetam um `client` fake (dublê de
    `client.messages.create`) e nunca tocam a rede.
    """

    def __init__(self, api_key: str, model: str, *, client: Any | None = None) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key)
        self._client = client
        self._model = model

    def __call__(self, texts: Sequence[str], categories: Sequence[str]) -> list[str]:
        """Varre `texts` em uma única chamada; fallback pros textos originais
        se a resposta não vier no formato numerado esperado (nunca perde
        texto, nunca propaga erro de parsing pro caller)."""
        if not texts:
            return list(texts)

        prompt = build_sweep_prompt(texts, categories)
        response = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_text = "".join(
            getattr(block, "text", "") for block in getattr(response, "content", [])
        )
        parsed = _parse_swept_response(raw_text, len(texts))
        return parsed if parsed is not None else list(texts)
