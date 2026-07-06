"""Testes do adapter real de extração (com client da Anthropic mockado).

Nunca toca a rede: injeta um client fake que espelha a forma de
`client.messages.create(...)` e a resposta (blocos com `.type`/`.input`).
"""

from types import SimpleNamespace

from autoseguro.anthropic_extractor import AnthropicExtractor


class _FakeMessages:
    def __init__(self, response):
        self._response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


class _FakeClient:
    def __init__(self, response):
        self.messages = _FakeMessages(response)


def _tool_use_response(input_dict):
    block = SimpleNamespace(type="tool_use", name="registrar_dados_cotacao", input=input_dict)
    return SimpleNamespace(content=[block])


def test_extract_devolve_o_input_da_tool():
    payload = {
        "veiculo_ano": 2008,
        "idade": 35,
        "cep": "01310-100",
        "marca": "Toyota",
        "modelo": "Corolla",
        "data_inicio": None,
    }
    client = _FakeClient(_tool_use_response(payload))
    extractor = AnthropicExtractor("sk-ant-fake", "claude-sonnet-5", client=client)

    out = extractor.extract("Corolla 2008, tenho 35 anos, CEP 01310-100")

    assert out["veiculo_ano"] == 2008
    assert out["idade"] == 35
    assert out["cep"] == "01310-100"


def test_extract_forca_tool_estrita():
    client = _FakeClient(_tool_use_response({"veiculo_ano": None, "idade": None, "cep": None,
                                             "marca": None, "modelo": None, "data_inicio": None}))
    extractor = AnthropicExtractor("sk-ant-fake", "claude-sonnet-5", client=client)

    extractor.extract("oi")

    call = client.messages.calls[0]
    assert call["tool_choice"] == {"type": "tool", "name": "registrar_dados_cotacao"}
    assert call["tools"][0]["strict"] is True
    assert call["tools"][0]["input_schema"]["additionalProperties"] is False


def test_extract_sem_tool_use_devolve_vazio():
    resp = SimpleNamespace(content=[SimpleNamespace(type="text", text="não chamei tool")])
    extractor = AnthropicExtractor("sk-ant-fake", "m", client=_FakeClient(resp))

    assert extractor.extract("qualquer coisa") == {}
