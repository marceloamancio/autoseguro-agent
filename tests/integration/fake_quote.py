"""Servidor `/quote` REAL e controlável para os testes de integração de produção.

Vendorizado de `adversarial-lab/red_team/fake_quote.py` (Stage 0.1 do plano de
conserto) para que os testes de integração não dependam do lab externo.

Por que um servidor de verdade (uvicorn) e não `httpx.MockTransport`: o bug
mais importante que este arquivo prova (`quote_client.py` — `httpx.AsyncClient`
criado sem `timeout=`) só se manifesta quando o transporte é uma conexão HTTP
real — o `httpx.MockTransport` chama o handler diretamente, sem passar pelo
enforcement de timeout do httpcore, então nunca dispara o timeout default de
5s da lib. Precisamos de latência de parede real (`time.sleep`) atrás de um
socket TCP de verdade.

Reaproveita a lógica de negócio real do desafio (`quote_logic.cotar`,
`CotacaoRecusada`) por importação (leitura), nunca modifica
`namastex-fde-challenge`. Adiciona modos de falha controláveis via um
endpoint `/_control` (fora do contrato real da `/quote` — só para teste):

- `normal`: delega para `cotar()` real (200/422/400 como o serviço real).
- `fail_status`: sempre devolve `fail_status` (ex.: 429, 500, 502, 503).
- `slow_then_normal`: dorme `slow_seconds` (bloqueante de verdade) e então
  delega para `cotar()` — para provar que uma chamada lenta-mas-válida é
  jogada fora pelo timeout default do httpx.
- `flap`: alterna sucesso/falha segundo `flap_pattern` (lista de bool, True =
  falha) — para o cenário de "circuit breaker sonda /health (sempre up)
  enquanto só /quote cai".

`/health` sempre responde 200 a menos que `health_down=True` seja setado via
`/_control` (usado só no cenário de sonda de health também indisponível).
"""

from __future__ import annotations

import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

_ROOT = Path(__file__).resolve().parent
_CHALLENGE_QUOTE_SERVICE = _ROOT.parent.parent.parent / "namastex-fde-challenge" / "quote-service"
if str(_CHALLENGE_QUOTE_SERVICE) not in sys.path:
    sys.path.insert(0, str(_CHALLENGE_QUOTE_SERVICE))

from app.quote_logic import CotacaoRecusada, cotar  # type: ignore  # noqa: E402


class QuoteRequest(BaseModel):
    plano_id: str = Field("essencial")
    idade: int = Field(..., ge=0, le=200)
    veiculo_ano: int = Field(..., ge=1950, le=2100)
    cep: str | None = None
    data_inicio: str | None = None


class ControlBody(BaseModel):
    mode: str | None = None
    fail_status: int | None = None
    slow_seconds: float | None = None
    flap_pattern: list[bool] | None = None
    health_down: bool | None = None
    reset: bool = False


class _State:
    def __init__(self) -> None:
        # RLock (não Lock): `control()` mantém o lock e chama `state.reset()`,
        # que também adquire o mesmo lock -- com um Lock comum isso seria um
        # autodeadlock na mesma thread.
        self.lock = threading.RLock()
        self.mode = "normal"
        self.fail_status = 500
        self.slow_seconds = 8.0
        self.flap_pattern: list[bool] = []
        self.flap_index = 0
        self.health_down = False
        self.quote_calls: list[float] = []
        self.health_calls: list[float] = []

    def reset(self) -> None:
        with self.lock:
            self.mode = "normal"
            self.fail_status = 500
            self.slow_seconds = 8.0
            self.flap_pattern = []
            self.flap_index = 0
            self.health_down = False
            self.quote_calls = []
            self.health_calls = []


def build_app() -> FastAPI:
    app = FastAPI(title="fake_quote (integration tests)")
    state = _State()
    app.state.redteam = state

    @app.get("/health")
    def health() -> Any:
        with state.lock:
            state.health_calls.append(time.monotonic())
            down = state.health_down
        if down:
            return JSONResponse(status_code=503, content={"status": "down"})
        return {"status": "ok"}

    @app.post("/_control")
    def control(body: ControlBody) -> Any:
        with state.lock:
            if body.reset:
                state.reset()
            if body.mode is not None:
                state.mode = body.mode
            if body.fail_status is not None:
                state.fail_status = body.fail_status
            if body.slow_seconds is not None:
                state.slow_seconds = body.slow_seconds
            if body.flap_pattern is not None:
                state.flap_pattern = body.flap_pattern
                state.flap_index = 0
            if body.health_down is not None:
                state.health_down = body.health_down
            return {
                "mode": state.mode,
                "fail_status": state.fail_status,
                "slow_seconds": state.slow_seconds,
                "flap_pattern": state.flap_pattern,
                "health_down": state.health_down,
                "quote_calls": len(state.quote_calls),
                "health_calls": len(state.health_calls),
            }

    @app.get("/_control")
    def control_get() -> Any:
        with state.lock:
            return {
                "mode": state.mode,
                "quote_calls": len(state.quote_calls),
                "health_calls": len(state.health_calls),
            }

    def _cotar_response(payload: dict) -> Any:
        try:
            return cotar(payload)
        except CotacaoRecusada as e:
            return JSONResponse(status_code=422, content={"error": "cotacao_recusada", "motivo": e.motivo})
        except (KeyError, ValueError, TypeError) as e:
            return JSONResponse(status_code=400, content={"error": "payload_invalido", "detalhe": str(e)})

    @app.post("/quote")
    def quote(req: QuoteRequest) -> Any:
        # NOTA: handler sync (`def`, não `async def`) de propósito — Starlette
        # despacha pra threadpool, então `time.sleep` bloqueante aqui não trava
        # o event loop nem outras requisições concorrentes (mesmo shape do
        # serviço real do desafio, `quote-service/app/main.py:quote`).
        with state.lock:
            state.quote_calls.append(time.monotonic())
            mode = state.mode
            fail_status = state.fail_status
            slow_seconds = state.slow_seconds
            flap_pattern = list(state.flap_pattern)
            idx = state.flap_index
            state.flap_index += 1

        payload = req.model_dump()

        if mode == "fail_status":
            return JSONResponse(status_code=fail_status, content={"error": "forced_failure"})

        if mode == "slow_then_normal":
            time.sleep(slow_seconds)
            return _cotar_response(payload)

        if mode == "flap":
            if flap_pattern:
                should_fail = flap_pattern[idx % len(flap_pattern)]
            else:
                should_fail = False
            if should_fail:
                return JSONResponse(status_code=fail_status, content={"error": "forced_failure_flap"})
            return _cotar_response(payload)

        return _cotar_response(payload)

    return app


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class FakeQuoteServer:
    """Roda `build_app()` num uvicorn de verdade, numa thread, porta livre."""

    def __init__(self) -> None:
        import uvicorn

        self.port = _free_port()
        self.app = build_app()
        config = uvicorn.Config(
            self.app, host="127.0.0.1", port=self.port, log_level="warning", loop="asyncio"
        )
        self.server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self.server.run, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self, timeout: float = 5.0) -> None:
        self._thread.start()
        deadline = time.monotonic() + timeout
        while not self.server.started:
            if time.monotonic() > deadline:
                raise RuntimeError("fake_quote server nao subiu a tempo")
            time.sleep(0.02)
        # `server.started` vira True um instante antes do socket realmente
        # aceitar conexões de forma confiável (race benigna do próprio
        # uvicorn) -- confirma com um GET /health real, com retry, antes de
        # considerar o servidor pronto pra uso pelos testes.
        self._wait_until_reachable(timeout=timeout)

    def _wait_until_reachable(self, timeout: float = 5.0) -> None:
        import httpx

        deadline = time.monotonic() + timeout
        last_exc: Exception | None = None
        while time.monotonic() < deadline:
            try:
                with httpx.Client(base_url=self.base_url, timeout=1.0) as c:
                    r = c.get("/health")
                    if r.status_code == 200:
                        return
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
            time.sleep(0.05)
        raise RuntimeError(f"fake_quote server nao ficou alcancavel a tempo: {last_exc!r}")

    def stop(self, timeout: float = 5.0) -> None:
        self.server.should_exit = True
        self._thread.join(timeout=timeout)

    @staticmethod
    def _post_control(base_url: str, body: dict) -> dict:
        """POST /_control com retry curto -- absorve qualquer soletrança
        residual de conexão logo após o startup do servidor."""
        import httpx

        last_exc: Exception | None = None
        for _ in range(5):
            try:
                with httpx.Client(base_url=base_url, timeout=10.0) as c:
                    resp = c.post("/_control", json=body)
                    resp.raise_for_status()
                    return resp.json()
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                time.sleep(0.1)
        raise RuntimeError(f"POST /_control falhou apos retries: {last_exc!r}")

    # -- controle (via HTTP, mesma superficie que um cliente real usaria) ---

    def set_mode(self, mode: str, **kwargs: Any) -> None:
        self._post_control(self.base_url, {"mode": mode, **kwargs})

    def reset(self) -> None:
        self._post_control(self.base_url, {"reset": True})

    def stats(self) -> dict:
        import httpx

        with httpx.Client(base_url=self.base_url, timeout=10.0) as c:
            return c.get("/_control").json()
