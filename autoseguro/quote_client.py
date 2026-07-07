"""Cliente HTTP resiliente para a API `/quote` (Group B, DEC-7 / Q5).

Política de resiliência (ver `.genie/wishes/agente-autoseguro/WISH.md` e
`DECISOES.md`, Q5):

- **Timeout por tentativa = `quote_timeout_s`** (default 9s = `SLOW_SECONDS + 1`):
  suficiente pra capturar a chamada lenta de 8s do mock, tratando-a como cotação
  válida em vez de descartá-la.
- **Retry só em infra** (5xx, 429/408, ou timeout/erro de transporte), com
  backoff exponencial (`quote_backoff_base_s * 2**tentativa`) + jitter — que
  respeita o cabeçalho `Retry-After` da resposta quando presente —, até
  `quote_max_retries` tentativas extras (nunca em 422/400 — seria desperdício e
  "burrice" do agente insistir numa recusa de regra ou payload malformado).
- **Deadline total** (`quote_deadline_s`) sobre o conjunto de tentativas de uma
  chamada — estourou, vira sinal de handoff. Os defaults de produção mantêm
  `(quote_max_retries + 1) * quote_timeout_s <= quote_deadline_s` (ver
  `config.py`), senão o deadline cortaria antes do orçamento de retries.
- **Circuit breaker leve:** abre após `quote_cb_failure_threshold` falhas de
  infra seguidas → fast-fail sem martelar o serviço; após `quote_cb_reset_s`,
  deixa passar **uma chamada real ao `/quote`** (canary) para decidir se
  fecha — nunca sonda `/health`, que fica estável mesmo com `/quote` caído.
- **Nunca inventa preço:** 422 vira `CotacaoRecusada`, 400 vira
  `PayloadInvalido`, e o esgotamento de retries/deadline/breaker vira
  `QuoteUnavailable` — sinal para o motor de handoff (Q6), carregando contexto.
  O preço só existe se vier de uma resposta 200 real da API.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Any

import httpx

from .config import Config


class QuoteClientError(Exception):
    """Base para os erros do cliente de cotação."""


class CotacaoRecusada(QuoteClientError):
    """A API recusou a cotação por regra de negócio (HTTP 422).

    Erro de negócio, não de infra: nunca dispara retry nem afeta o circuit
    breaker (o serviço respondeu corretamente).
    """

    def __init__(self, motivo: str):
        self.motivo = motivo
        super().__init__(motivo)


class PayloadInvalido(QuoteClientError):
    """O payload enviado é inválido/incompleto (HTTP 400).

    Erro de negócio (dado faltante do lado do chamador), não de infra: nunca
    dispara retry nem afeta o circuit breaker.
    """

    def __init__(self, detalhe: str):
        self.detalhe = detalhe
        super().__init__(detalhe)


class QuoteUnavailable(QuoteClientError):
    """Sinal de handoff: a cotação não pôde ser obtida por motivo de infra.

    Cobre esgotamento de retries, deadline total excedido e circuit breaker
    aberto. Carrega `context` para o motor de handoff (Q6) — nunca contém um
    preço, porque nenhum preço confiável foi obtido.
    """

    def __init__(self, reason: str, *, attempts: int = 0, context: dict[str, Any] | None = None):
        self.reason = reason
        self.attempts = attempts
        self.context = context or {}
        super().__init__(f"{reason} (attempts={attempts})")


@dataclass(frozen=True)
class QuoteResult:
    """Parse tipado da resposta 200 de `/quote` (ver `quote_logic.cotar`)."""

    plano_id: str
    plano_nome: str
    premio_mensal: float
    franquia: float
    coberturas: list[str]
    carencia: dict[str, Any]
    moeda: str
    multiplicadores: dict[str, float]
    primeiro_pagamento_pro_rata: dict[str, Any] | None = None

    @classmethod
    def from_response(cls, data: dict[str, Any]) -> "QuoteResult":
        return cls(
            plano_id=data["plano_id"],
            plano_nome=data["plano_nome"],
            premio_mensal=data["premio_mensal"],
            franquia=data["franquia"],
            coberturas=list(data["coberturas"]),
            carencia=data["carencia"],
            moeda=data["moeda"],
            multiplicadores=data["multiplicadores"],
            primeiro_pagamento_pro_rata=data.get("primeiro_pagamento_pro_rata"),
        )


class _CircuitBreaker:
    """Circuit breaker leve, contagem de falhas de infra consecutivas.

    Fechado -> passa tudo. Após `failure_threshold` falhas seguidas, abre
    (fast-fail). Depois de `reset_s` de aberto, a próxima chamada vira um
    *canary*: uma única tentativa real ao `/quote` decide o destino do
    breaker — sucesso fecha (`record_success`), falha reabre
    (`record_failure` reinicia o cronômetro de reset automaticamente, já que
    `_consecutive_failures` permanece >= `failure_threshold`). Nunca sonda
    `/health` (P0-7): `/health` fica estável mesmo com `/quote` caído, então
    fechava o breaker incorretamente.
    """

    def __init__(self, failure_threshold: int, reset_s: float):
        self.failure_threshold = failure_threshold
        self.reset_s = reset_s
        self._consecutive_failures = 0
        self._opened_at: float | None = None

    @property
    def is_open(self) -> bool:
        return self._opened_at is not None

    def seconds_since_open(self) -> float | None:
        if self._opened_at is None:
            return None
        return time.monotonic() - self._opened_at

    def record_success(self) -> None:
        self._consecutive_failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.failure_threshold:
            self._opened_at = time.monotonic()


class QuoteClient:
    """Cliente async resiliente para `POST /quote`."""

    def __init__(self, config: Config, *, transport: httpx.AsyncBaseTransport | None = None):
        self._config = config
        self._breaker = _CircuitBreaker(
            config.quote_cb_failure_threshold, config.quote_cb_reset_s
        )
        self._client = httpx.AsyncClient(
            base_url=config.quote_api_url,
            transport=transport,
            timeout=httpx.Timeout(config.quote_timeout_s),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "QuoteClient":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def cotar(self, payload: dict[str, Any]) -> QuoteResult:
        """Chama `POST /quote` com toda a política de resiliência.

        Levanta `CotacaoRecusada` (422), `PayloadInvalido` (400) ou
        `QuoteUnavailable` (esgotou retries/deadline/breaker aberto). Nunca
        retorna um preço que não tenha vindo de uma resposta 200 real.
        """
        deadline = time.monotonic() + self._config.quote_deadline_s

        # Se o breaker acabou de sair da janela de reset, esta chamada vira um
        # canary: uma única tentativa real ao /quote decide se ele fecha ou
        # reabre (P0-7) -- nunca consome o orçamento normal de retries.
        is_canary = await self._ensure_breaker_allows(payload)

        max_attempts = 1 if is_canary else self._config.quote_max_retries + 1
        attempts = 0
        last_reason = "erro_desconhecido"

        while attempts < max_attempts:
            if time.monotonic() >= deadline:
                raise QuoteUnavailable(
                    "deadline_excedido", attempts=attempts, context={"payload": payload}
                )

            attempts += 1
            try:
                response = await self._send(payload)
            except (asyncio.TimeoutError, httpx.TimeoutException):
                last_reason = "timeout"
                self._breaker.record_failure()
                if attempts >= max_attempts:
                    break
                await self._sleep_backoff(attempts, deadline)
                continue
            except httpx.TransportError as exc:
                last_reason = f"erro_transporte:{exc}"
                self._breaker.record_failure()
                if attempts >= max_attempts:
                    break
                await self._sleep_backoff(attempts, deadline)
                continue

            if response.status_code == 200:
                self._breaker.record_success()
                return QuoteResult.from_response(response.json())

            if response.status_code == 422:
                self._breaker.record_success()
                motivo = self._safe_json(response).get("motivo", "cotação recusada")
                raise CotacaoRecusada(motivo)

            if response.status_code == 400:
                self._breaker.record_success()
                detalhe = self._safe_json(response).get("detalhe", "payload inválido")
                raise PayloadInvalido(detalhe)

            # 5xx (infra caída), 429 (rate limit) e 408 (timeout do servidor)
            # são todos infra retryável -- transitório, nunca "burrice" do
            # agente insistir (P0-5). Respeita `Retry-After` se presente.
            if response.status_code >= 500 or response.status_code in (429, 408):
                last_reason = f"http_{response.status_code}"
                self._breaker.record_failure()
                if attempts >= max_attempts:
                    break
                await self._sleep_backoff(
                    attempts, deadline, retry_after=self._parse_retry_after(response)
                )
                continue

            # Status inesperado (nem sucesso, nem erro de negócio conhecido,
            # nem infra retryável): não inventamos preço nem insistimos às cegas.
            last_reason = f"http_inesperado_{response.status_code}"
            self._breaker.record_failure()
            break

        raise QuoteUnavailable(
            f"esgotou_tentativas:{last_reason}",
            attempts=attempts,
            context={"payload": payload},
        )

    async def _send(self, payload: dict[str, Any]) -> httpx.Response:
        """Executa a tentativa com timeout duro de `quote_timeout_s`.

        `asyncio.wait_for` garante o corte mesmo quando o transporte (ex.:
        `httpx.MockTransport` em teste) não aplica timeout de rede real —
        único jeito de capturar deterministicamente a chamada lenta simulada
        pelo mock (`QUOTE_SLOW_SECONDS`).
        """
        return await asyncio.wait_for(
            self._client.post("/quote", json=payload),
            timeout=self._config.quote_timeout_s,
        )

    async def _ensure_breaker_allows(self, payload: dict[str, Any]) -> bool:
        """Retorna `True` se esta chamada deve ser tratada como canary
        (sonda de meia-abertura via `/quote` real, P0-7), `False` se o
        breaker está fechado (fluxo normal). Levanta `QuoteUnavailable` se o
        breaker está aberto e a janela de reset ainda não passou.
        """
        if not self._breaker.is_open:
            return False

        elapsed = self._breaker.seconds_since_open()
        if elapsed is None or elapsed < self._breaker.reset_s:
            raise QuoteUnavailable(
                "circuit_breaker_aberto", attempts=0, context={"payload": payload}
            )

        # Janela de reset passada: deixa UMA chamada real ao /quote decidir
        # (nunca /health -- fica estável mesmo com /quote caído, ver P0-7).
        return True

    async def _sleep_backoff(
        self, attempt: int, deadline: float, *, retry_after: float | None = None
    ) -> None:
        base = self._config.quote_backoff_base_s
        delay = base * (2 ** (attempt - 1)) + random.uniform(0, base)
        if retry_after is not None:
            delay = max(delay, retry_after)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        await asyncio.sleep(min(delay, remaining))

    @staticmethod
    def _parse_retry_after(response: httpx.Response) -> float | None:
        """Lê o cabeçalho `Retry-After` (segundos) se presente e numérico.

        Ignora silenciosamente o formato HTTP-date (raro em `429`/`408`) e
        valores inválidos -- nesse caso o backoff exponencial padrão decide.
        """
        raw = response.headers.get("Retry-After")
        if raw is None:
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    @staticmethod
    def _safe_json(response: httpx.Response) -> dict[str, Any]:
        try:
            return response.json()
        except ValueError:
            return {}
