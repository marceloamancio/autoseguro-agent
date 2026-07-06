"""Adapter real de extração via API da Anthropic (tool-use com schema estrito).

Implementa o Protocol `LlmExtractorClient` de `extraction.py`: recebe o texto
livre do lead e devolve um `dict` com os campos de cotação, forçando uma tool
com `strict: true` (schema garantido). Síncrono, como o Protocol exige.

O import do SDK é **lazy** (dentro do `__init__`, só quando `client` não é
injetado), então os testes injetam um client mockado e nunca tocam a rede.
"""

from __future__ import annotations

from typing import Any

# Tool de extração com schema estrito: o modelo é forçado a devolver exatamente
# estes campos (null para o que não estiver na mensagem). É aqui que o Q2
# ("structured outputs / strict:true") é de fato exercitado.
EXTRACT_TOOL: dict[str, Any] = {
    "name": "registrar_dados_cotacao",
    "description": (
        "Registra os dados de cotação de seguro auto extraídos da mensagem do "
        "lead. Use null para qualquer campo que não esteja presente na mensagem."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "veiculo_ano": {"type": ["integer", "null"], "description": "Ano do veículo, ex.: 2008"},
            "idade": {"type": ["integer", "null"], "description": "Idade do lead em anos"},
            "cep": {"type": ["string", "null"], "description": "CEP informado pelo lead"},
            "marca": {"type": ["string", "null"], "description": "Marca do veículo"},
            "modelo": {"type": ["string", "null"], "description": "Modelo do veículo"},
            "data_inicio": {
                "type": ["string", "null"],
                "description": "Início da vigência em YYYY-MM-DD, se o lead mencionar",
            },
        },
        "required": ["veiculo_ano", "idade", "cep", "marca", "modelo", "data_inicio"],
        "additionalProperties": False,
    },
}

_SYSTEM = (
    "Você extrai dados para cotação de seguro auto de mensagens em português "
    "informal e bagunçado. Chame a tool registrar_dados_cotacao com os campos "
    "que conseguir inferir da mensagem do lead; use null para o que não estiver "
    "presente. Nunca invente dados que o lead não informou."
)


class AnthropicExtractor:
    """`LlmExtractorClient` real, baseado na API da Anthropic."""

    def __init__(self, api_key: str, model: str, *, client: Any | None = None) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key)
        self._client = client
        self._model = model

    def extract(self, text: str) -> dict[str, Any]:
        """Extrai os campos de cotação de `text` via tool-use estrito.

        Devolve o `input` da tool como dict; `{}` se o modelo não chamou a tool.
        Exceções são deixadas propagar — `extract_once` já as trata como
        "LLM não ajudou" sem derrubar o fluxo.
        """
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=512,
            system=_SYSTEM,
            tools=[EXTRACT_TOOL],
            tool_choice={"type": "tool", "name": "registrar_dados_cotacao"},
            messages=[{"role": "user", "content": text}],
        )
        for block in resp.content:
            if getattr(block, "type", "") == "tool_use":
                return dict(block.input)
        return {}
