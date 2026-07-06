# Wish: Agente de cotação AutoSeguro (WhatsApp)

**Status:** DRAFT
**Slug:** `agente-autoseguro`
**Created:** 2026-07-06

---

## Summary

Construir um agente de vendas por WhatsApp para a seguradora fictícia AutoSeguro (desafio
FDE Namastex) que atende um lead de ponta a ponta — conversa → qualifica → cota (via a API
`/quote`) → decide (resolve ou transborda pro humano). O foco de nota é lidar com a `/quote`
instável de forma elegante, ter critério de handoff explícito/defensável, rastreabilidade
(id+status) e cuidado com PII. Decisões cristalizadas em `../../../DECISOES.md` (Q1–Q7 + Q3b
+ refinamentos).

---

## Scope

### IN
- Core de agente assíncrono no **SDK da Anthropic** (`AsyncAnthropic`), com a `/quote`
  exposta como **tool única** (`claude-sonnet-5`).
- Cliente HTTP resiliente da `/quote`: distinguir infra (5xx/timeout → retry) de negócio
  (422/400 → tratar na hora), retry com backoff+jitter, **timeout 9s**, deadline total,
  **circuit breaker leve**, nunca inventar preço.
- **Extração/qualificação** de `idade`, `veiculo_ano`, `cep` (+ `data_inicio` opcional) de
  texto livre via structured outputs, com confirmação antes de cotar e backstop regex.
- **Mascaramento de PII at-rest** (regex simples certeiro → varredura LLM em lote) +
  **minimização de coleta** (não pedir CPF/e-mail/telefone/placa).
- **Handoff** por fronteira de capacidade/autoridade, com `reason_code` auditável.
- **CLI de chat** turn-based (ack imediato + nudge), chave via env com fail-fast.
- **Rastreabilidade** `trace.jsonl` (ids + status por mensagem/cotação), stdlib logging + JSON.
- **Explicar a cotação** ao lead: coberturas, franquia, carência 30d (roubo/furto), pró-rata.
- **README** (como rodar + decisões) + **2 logs de execução** mascarados (happy + falha/handoff).

### OUT
- **Não** implementar o webhook FastAPI (apenas documentar como forma de produção).
- **Não** reimplementar as regras de precificação da `/quote` (consumir a API; o preço só
  vem dela).
- **Não** construir o agente adversarial externo (Q8, adiado para pós-implementação).
- **Não** integrar com WhatsApp real (a CLI simula a conversa).
- **Não** emitir apólice/boleto (fora de capacidade → handoff).
- **Não** rodar LLM-mascarador no caminho quente da conversa (só em lote/at-rest).
- **Não** construir UI além da CLI.

---

## Decisions

- **DEC-1 (Q1):** Agente enxuto no SDK Anthropic, `/quote` como tool — resiliência e handoff
  explícitos e legíveis; sem framework pesado.
- **DEC-2 (Q1b):** Chave nunca no repo — `ANTHROPIC_API_KEY` do avaliador; `.env` gitignored;
  fail-fast sem imprimir o valor.
- **DEC-3 (Q2):** `claude-sonnet-5` — custo/qualidade com function-calling (`strict:true`).
- **DEC-4 (Q3):** PII mascarada **at-rest** (camada Silver/logs) via regex simples → LLM em
  lote; **minimização** de coleta. Sem LLM-mascarador no hot path.
- **DEC-5 (Q3b):** Extração via LLM (structured outputs) + confirmação obrigatória + backstop
  regex; do veículo só o **ano** importa pra cotar.
- **DEC-6 (Q4):** CLI turn-based sobre core async; replay opcional; webhook documentado.
- **DEC-7 (Q5):** Resiliência — infra×negócio, ack imediato, timeout 9s (`SLOW_SECONDS+1`),
  3 retries backoff+jitter, deadline ~25s, circuit breaker leve, nunca inventar preço.
- **DEC-8 (Q6):** Handoff por capacidade/autoridade/respeito/sensibilidade, com `reason_code`.
- **DEC-9 (Q7):** `trace.jsonl` canônico (ids+status), stdlib logging + JSON; OTel documentado.

---

## Success Criteria

- [ ] Caminho feliz roda ponta a ponta: conversa → qualifica → cota → entrega cotação (com
      coberturas, franquia, carência e pró-rata explicados a partir da resposta da API).
- [ ] Sob falha da `/quote` (5xx/timeout) o agente faz retry com backoff, captura a lenta
      (timeout 9s) e, ao esgotar, **transborda com contexto** — nunca inventa preço.
- [ ] 422 (recusa por regra) e 400 (payload inválido) são tratados **sem retry**.
- [ ] Critério de handoff é explícito, com `reason_code` auditável em cada transbordo.
- [ ] `trace.jsonl` registra cada mensagem e cada cotação com **id e status**.
- [ ] PII conhecida do dataset é mascarada nos logs/trace (recall verificado em teste).
- [ ] Agente **não pede** CPF/e-mail/telefone/placa.
- [ ] `ANTHROPIC_API_KEY` ausente → fail-fast claro, sem vazar o valor.
- [ ] README explica como rodar e as decisões; 2 logs de execução mascarados versionados.
- [ ] `uv run pytest` verde; todos os testes rodam **sem chave** (LLM e `/quote` mockados).

---

## Assumptions

- **ASM-1:** A `/quote` do desafio roda em `http://localhost:8000` (Docker) durante a demo;
  os testes usam mocks/fake server, não a API real.
- **ASM-2:** O avaliador fornece a própria `ANTHROPIC_API_KEY` (ou perfil `ant`).
- **ASM-3:** `claude-sonnet-5` e `strict:true` são válidos na API atual da Anthropic
  (verificado contra a doc; confirmar no primeiro request real).

## Risks

- **RISK-1:** LLM não-determinístico em extração/PII — Mitigação: testes checam **recall**
  e schema, não texto exato; backstop regex; validação de faixas da `/quote`.
- **RISK-2:** Custo de token na varredura LLM de PII — Mitigação: roda **em lote/at-rest**,
  com cache e desligável; nunca por mensagem no hot path.
- **RISK-3:** Over-engineering (CB/async) contra brief "enxuto" — Mitigação: núcleo simples;
  CB escopado ao modo replay/lote e async justificado pela resiliência, documentado no README.
- **RISK-4:** Falhas correlacionadas com `QUOTE_SEED` fixo podem estourar os 3 retries —
  Mitigação: é o mecanismo de demonstrar o handoff; não vender ~99% como probabilidade real.

---

## Execution Groups

### Group A: Setup do projeto
**Goal:** Base do projeto Python com deps, config e scaffolding de testes.

**Deliverables:**
- `pyproject.toml` (uv) com `anthropic`, `httpx`, `pytest`, `pytest-asyncio`, `pandas`,
  `pyarrow`, `python-dotenv`.
- Loader de config (env: `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`, `QUOTE_API_URL`,
  thresholds de resiliência) com **fail-fast** sem imprimir segredo.
- Estrutura de pacote `autoseguro/` + `tests/`.

**Acceptance Criteria:**
- [ ] `uv run pytest` roda (mesmo que vazio) e passa.
- [ ] Config sem `ANTHROPIC_API_KEY` levanta erro claro; teste cobre o fail-fast.

**Validation:** `uv run pytest tests/test_config.py`

---

### Group B: Cliente resiliente da /quote (Q5)
**Goal:** Cliente async da `/quote` com toda a política de resiliência.

**Deliverables:**
- Cliente `httpx` async: timeout 9s (`SLOW_SECONDS+1`), retry (3) backoff+jitter só em
  5xx/timeout, deadline total, **circuit breaker leve** (abre após N falhas → fast-fail,
  sonda `/health`), parse tipado da resposta.
- Distinção infra×negócio: 422 → `CotacaoRecusada`; 400 → `PayloadInvalido`; esgotou → sinal
  de handoff. Nunca inventa preço.

**Acceptance Criteria:**
- [ ] 5xx → faz retry; 422/400 → **não** faz retry (testado com fake server).
- [ ] Chamada lenta (sleep > 8s) é capturada pelo timeout 9s.
- [ ] Circuit breaker abre após N falhas e fecha ao sondar `/health`.
- [ ] Ao esgotar retries, emite sinal de handoff com contexto.

**Validation:** `uv run pytest tests/test_quote_client.py`

---

### Group C: Mascaramento de PII + minimização (Q3)
**Goal:** Redator de PII at-rest (regex + LLM em lote) e política de minimização.

**Deliverables:**
- `PiiRedactor`: regex simples/certeiro (CPF, e-mail, telefone, placa, CEP → marcadores) +
  interface de varredura LLM **em lote** (mockável, desligável).
- Aplicação nos logs/trace e no log entregue.

**Acceptance Criteria:**
- [ ] Recall sobre amostra do dataset: CPF/e-mail/telefone/placa/CEP mascarados (teste).
- [ ] Regex não mascara não-PII (precisão) em casos de controle.
- [ ] Varredura LLM é mockada nos testes (roda sem chave).

**Validation:** `uv run pytest tests/test_pii.py`

---

### Group D: Extração e qualificação (Q3b)
**Goal:** Extrair `idade`, `veiculo_ano`, `cep` de texto livre, com confirmação e backstop.

**Deliverables:**
- Extração via structured outputs (LLM mockável) → schema tipado; normalização (ano 2/4
  dígitos, CEP com/sem hífen, "nasci em X"→idade); validação de faixas da `/quote`.
- Definição de "dado essencial" e sinal de handoff após N=2 tentativas.
- Backstop regex leve (ano, CEP).

**Acceptance Criteria:**
- [ ] Extrai corretamente de fixtures ("e um Sandero 2022", "Toyota Corolla, ano 2008").
- [ ] Faltando dado essencial após N=2 → sinaliza handoff.
- [ ] Normalização coberta por testes.

**Validation:** `uv run pytest tests/test_extraction.py`

---

### Group E: Core do agente + handoff (Q1, Q6, refinamento #3)
**Goal:** Loop do agente, decisão de handoff e explicação da cotação.

**Deliverables:**
- Agente async (`AsyncAnthropic`), system prompt, tool `call_quote`, máquina de conversa
  (conversa→qualifica→cota→decide).
- Motor de handoff (tabela capacidade/autoridade/respeito/sensibilidade) com `reason_code`.
- Explicação da cotação (coberturas, franquia, carência 30d, pró-rata) da resposta da API.
- Coleta `data_inicio` (fallback com aviso).

**Acceptance Criteria:**
- [ ] Happy path (LLM+quote mockados) produz cotação e explica carência/pró-rata.
- [ ] Cada gatilho de handoff dispara com o `reason_code` correto (testes por caso).
- [ ] Agente nunca emite preço fora da resposta da `/quote`.

**Validation:** `uv run pytest tests/test_agent.py tests/test_handoff.py`

---

### Group F: CLI + rastreabilidade (Q4, Q7)
**Goal:** CLI de chat turn-based e trace estruturado.

**Deliverables:**
- CLI REPL (ack imediato "calculando...", nudge se >5s), fail-fast de chave.
- `trace.jsonl`: `conversation_id`/`event_id`/`quote_request_id`/`run_id`, tipos e status;
  stdlib logging + formatter JSON; mascarado (Q3).

**Acceptance Criteria:**
- [ ] Execução ponta a ponta (mocks) gera `trace.jsonl` com id+status por mensagem/cotação.
- [ ] Chave ausente → fail-fast; PII mascarada no trace.

**Validation:** `uv run pytest tests/test_cli.py tests/test_trace.py`

---

### Group G: Entrega — README, logs e replay opcional
**Goal:** Fechar os entregáveis do desafio.

**Deliverables:**
- README completo (como rodar + decisões, com link/resumo do `DECISOES.md`; nota sobre
  webhook/CB/async e mundo real).
- **2 logs de execução** mascarados versionados: happy-path (cotação saindo) e falha→handoff,
  com `QUOTE_SEED` fixo.
- **Replay harness opcional** sobre o dataset (batch, concorrência, breaker) — marcado opcional.

**Acceptance Criteria:**
- [ ] README explica como rodar e as decisões; sem "em aberto" pendente.
- [ ] Os 2 logs existem, mascarados e reproduzíveis.
- [ ] `uv run pytest` verde no conjunto todo.

**Validation:** `uv run pytest && ls docs/execucao-*.md`

---

## Review Results

_Populated by `/review` after execution completes._

---

## Files to Create/Modify

```
autoseguro-agent/
  pyproject.toml
  autoseguro/
    __init__.py
    config.py              # env + fail-fast
    quote_client.py        # Group B (resiliência)
    pii.py                 # Group C (regex + LLM lote)
    extraction.py          # Group D (structured outputs + backstop)
    agent.py               # Group E (loop + tool)
    handoff.py             # Group E (reason_code, tabela)
    tracing.py             # Group F (trace.jsonl)
    cli.py                 # Group F (REPL)
    replay.py              # Group G (opcional)
  tests/
    test_config.py test_quote_client.py test_pii.py test_extraction.py
    test_agent.py test_handoff.py test_cli.py test_trace.py
  docs/
    execucao-happy.md  execucao-falha-handoff.md
  README.md              # atualizar (Group G)
```
