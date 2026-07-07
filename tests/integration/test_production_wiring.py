"""Teste de wiring de produção (Stage 0.1 do fix-wish).

Pega a classe de bug "Protocol não plugado": monta o `Agent` de produção via
`cli.build_agent_from_config` (o mesmo caminho que `cli.main()` usa) e
assevera que as dependências opcionais-mas-essenciais (`extractor`,
`fuzzy_classifier`) foram de fato instanciadas e plugadas — nunca ficaram
`None` por omissão silenciosa. Isso teria pegado, em teste, o bug do
extractor que só apareceu num run real (`build_agent_from_config` nunca
passava `extractor=` explicitamente, mas a classe de bug é justamente essa:
um `Protocol` que existe no código mas nunca chega a ser plugado no caminho
de produção).

Não sobe rede real: `anthropic.AsyncAnthropic(api_key=...)` e
`anthropic.Anthropic(api_key=...)` apenas constroem o client em memória, sem
tocar a rede — só uma chamada de API real dispararia tráfego.
"""

from __future__ import annotations

from autoseguro import cli
from doubles import make_config


def test_build_agent_from_config_wires_deps(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-test-key")
    config = make_config()

    agent = cli.build_agent_from_config(config)

    assert agent._extractor is not None
    assert agent._fuzzy_classifier is not None
