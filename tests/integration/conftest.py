"""Fixtures dos testes de integração de produção (Stage 0.1 do fix-wish).

Sobe um `FakeQuoteServer` real (uvicorn, socket TCP de verdade) — necessário
porque o furo mais importante que este pacote de testes prova
(`quote_client.py` — `httpx.AsyncClient` sem `timeout=`) só se manifesta
atrás de uma conexão HTTP real (`httpx.MockTransport` não passa pelo
enforcement de timeout do httpcore).
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def fake_quote_server():
    """Servidor `/quote` real (uvicorn), controlável, escopo por módulo de
    teste — cada módulo sobe sua própria instância numa porta livre e a
    derruba ao final.
    """
    from fake_quote import FakeQuoteServer

    server = FakeQuoteServer()
    server.start()
    yield server
    server.stop()


@pytest.fixture()
def fake_quote(fake_quote_server):
    """Igual a `fake_quote_server`, mas reseta o estado (modo/contadores)
    antes de cada teste individual, pra os testes não vazarem estado entre si
    mesmo compartilhando o processo do servidor (module-scoped, por custo de
    subir/derrubar uvicorn a cada teste).
    """
    fake_quote_server.reset()
    yield fake_quote_server
    fake_quote_server.reset()
