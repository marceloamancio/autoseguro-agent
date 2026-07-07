# AutoSeguro Agent

Agente de vendas por WhatsApp da seguradora fictícia **AutoSeguro** (desafio FDE
Namastex). Atende leads de seguro de veículo de ponta a ponta:

> **conversa → qualifica → cota (via API `/quote`) → decide** (resolve sozinho ou passa
> pro humano)

O agente conversa com o lead, extrai idade/ano do veículo/CEP de texto livre e bagunçado,
confirma o que entendeu, chama a API de cotação do desafio (`quote-service`, propositalmente
instável) com uma política de resiliência explícita e, quando não consegue resolver por
conta própria — infra caiu, faltam dados, o assunto saiu do escopo, o lead pediu um humano —
transborda o atendimento com um motivo auditável em vez de travar ou inventar preço.

---

## Como rodar

Pré-requisitos: Python 3.11+, [`uv`](https://docs.astral.sh/uv/), Docker (pra subir a API de
cotação) e uma chave da Anthropic.

### 1. Suba a API de cotação do desafio

Num checkout do repo `namastex-fde-challenge` (irmão deste, não faz parte deste repo):

```bash
docker compose up --build      # API em http://localhost:8000
```

### 2. Configure o ambiente

```bash
cp .env.example .env
# edite .env e cole a SUA ANTHROPIC_API_KEY — o arquivo .env real fica no .gitignore,
# nunca vai pro git. Quem avalia usa a própria chave.
```

Variáveis (ver `.env.example` e `autoseguro/config.py`):

| Variável | Default | Papel |
|---|---|---|
| `ANTHROPIC_API_KEY` | _(obrigatória)_ | Chave da Anthropic. Ausente → `ConfigError` no boot, sem nunca imprimir o valor (fail-fast). |
| `ANTHROPIC_MODEL` | `claude-sonnet-5` | Modelo usado nas chamadas de conversa livre. |
| `QUOTE_API_URL` | `http://localhost:8000` | Base URL do `quote-service`. |
| `QUOTE_TIMEOUT_S` | `9.0` | Timeout por tentativa de `/quote` (`SLOW_SECONDS + 1`). |
| `QUOTE_MAX_RETRIES` | `3` | Retries extras em falha de infra (5xx/timeout). |
| `QUOTE_BACKOFF_BASE_S` | `0.5` | Base do backoff exponencial + jitter. |
| `QUOTE_DEADLINE_S` | `25.0` | Deadline total sobre o conjunto de tentativas de uma chamada. |
| `QUOTE_CB_FAILURE_THRESHOLD` | `5` | Falhas de infra seguidas até o circuit breaker abrir. |
| `QUOTE_CB_RESET_S` | `30.0` | Tempo aberto antes de sondar `/health` e tentar fechar. |

### 3. Rode a CLI de chat

```bash
uv run python -m autoseguro.cli
```

Digite as mensagens como se fosse o lead; o agente responde no terminal. `sair` / `exit` /
`quit` encerra. Cada execução grava `logs/trace.jsonl` (rastreabilidade — ver seção Q7 abaixo).

### 4. Rode os testes

```bash
uv run pytest
```

Todos os 136 testes rodam **sem chave e sem rede** — LLM e `QuoteClient` são sempre
dublês/mocks nos testes (ver `Protocol`s injetáveis em `agent.py`/`extraction.py`/
`quote_client.py`).

---

## Decisões de engenharia

Este projeto nasceu de um documento de estudo mais longo (`DECISOES.md`, fora deste repo,
no workspace de desenvolvimento). O resumo abaixo é a versão **curada e completa**, versionada
dentro do repo — nada fica "em aberto" fora daqui.

### Runtime do agente: SDK da Anthropic direto, sem framework (Q1)

Agente enxuto, direto no **SDK da Anthropic** (`AsyncAnthropic`), com a `/quote` exposta como
**tool única** do domínio (`agent.CALL_QUOTE_TOOL_SCHEMA`). Alternativas descartadas:
`automagik-hive` (esconde a resiliência que a avaliação quer ver, adiciona dependência),
Pydantic AI / LangGraph (dependência de terceiro, overkill pra um fluxo linear de
qualificação→cotação). Menos dependências = menos risco de setup na hora de rodar, e a
lógica de resiliência/handoff fica **explícita e legível no código Python**, não escondida
dentro de um framework — é o que os critérios "outro engenheiro entende" e "o que faz quando
a `/quote` falha" pedem.

### Nota de arquitetura: orquestração determinística, não loop agêntico

**O fluxo `conversa → qualifica → cota → decide` é controle determinístico em Python — não
um loop agêntico autônomo.** O LLM entra só em dois lugares seguros e não-críticos:

1. **Extração** (`extraction.py`): structured output de um texto livre pra um schema tipado.
2. **Conversa livre** (`agent.Agent._llm_reply`): só depois que a cotação já foi
   entregue/recusada, pra manter o papo natural — nunca antes disso.

O LLM **nunca decide sozinho** quais tools chamar, se cota, ou se transborda. Por quê:

- A tarefa é **linear e bem definida** (qualificação → cotação → decisão), não exploração
  aberta — é o tier mais simples que atende, seguindo a recomendação da própria Anthropic
  de "começar simples" antes de subir pra um agente autônomo.
- **"O que faz quando a `/quote` falha" é o critério que mais separa** nesta avaliação:
  retry, timeout, circuit breaker e handoff ficam **explícitos e testáveis de ponta a ponta
  sem rede**, em vez de depender do modelo "decidir" retry na hora certa.
- **"Nunca inventar preço" é garantido por arquitetura**, não por instrução de prompt: o LLM
  literalmente não tem acesso ao caminho que produz o preço nem à decisão de handoff — só
  formata texto a partir de dados que o Python já validou.
- Existe sim um **loop de conversa** turno a turno (`Agent.handle_turn`), mas ele é dirigido
  pelas mensagens do lead, não por um LLM decidindo o próximo passo.

### Modelo: `claude-sonnet-5` (Q2)

- **Custo-benefício:** qualidade quase-Opus em fluxos agentic/tool-use a ~metade do preço
  do Opus ($3/$15 por 1M tokens; $2/$10 em preço de lançamento). Haiku é mais fraco em
  objeção/negociação nuançada; Opus é overkill pra um fluxo de vendas linear.
- **Confiabilidade de structured outputs é o ponto central:** a extração de dados do lead
  usa schema estrito (`strict:true`), garantindo `veiculo_ano`/`idade`/`cep` bem-formados. A
  chamada da `/quote` em si **não é dirigida pelo LLM** (ver nota de arquitetura acima) —
  então o preço nunca é fabricado e o tratamento de 422/400 é sempre explícito em Python.
- **Coerência de stack:** já escolhido o SDK da Anthropic (Q1); trocar de provedor perderia
  as garantias de `strict:true`/structured outputs sem ganho.
- **Reprodutibilidade:** modelo configurável via `ANTHROPIC_MODEL` — o avaliador pode trocar
  sem tocar no código.

### PII: mascaramento at-rest + minimização de coleta (Q3, Q3b)

**Threat-model:** o agente principal é o próprio Sonnet e precisa ler o texto do lead pra
qualificar — PII inevitavelmente passa pelo modelo no caminho quente (trânsito ao vivo).
Este projeto **não** tenta proteger esse trânsito; ele protege o que fica **em repouso**
(histórico, `logs/trace.jsonl`, o log de execução entregue) — que é onde o desafio pede o
mascaramento ("mascarar a PII na camada Silver/dados").

Design em duas camadas, aplicadas em lote — nunca por mensagem no caminho quente
(`autoseguro/pii.py`):

1. **Regex simples e certeiro** (`redact_text`): substitui CPF, e-mail, telefone, placa e
   CEP — nos formatos óbvios e bem-formados do gerador do desafio — por marcadores
   (`⟨CPF⟩`, `⟨EMAIL⟩`, `⟨TELEFONE⟩`, `⟨PLACA⟩`, `⟨CEP⟩`). Alta precisão, deliberadamente não
   exaustivo.
2. **Varredura LLM em lote** (`llm_sweep`): pega nome de terceiro, formato exótico e
   categorias adicionais (RG, endereço, data de nascimento...) que o regex deixou passar.
   Roda em lote (não por mensagem), é **desligada por padrão** (sem `client`, é no-op — nunca
   chama rede) e **mockável** em teste.

**Minimização de coleta** (o argumento que reduz risco na origem): o agente **nunca pede**
CPF, e-mail, telefone ou placa — a `/quote` não usa nenhum desses campos. Se o lead mandar
espontaneamente, o dado é mascarado at-rest e simplesmente não entra no fluxo de cotação.

**Extração e qualificação** (Q3b, `extraction.py`): do veículo, só o **ano** importa pra
cotar (marca/modelo servem só pra rapport). Extração via LLM com structured outputs
(`{veiculo_ano, idade, cep, marca?, modelo?, data_inicio?}`), com **confirmação obrigatória**
do lead antes de cotar ("Corolla 2008, você 35 anos, CEP 01310-100 — confere?"), normalização
(ano 2 vs 4 dígitos, CEP com/sem hífen, "nasci em X" → idade) e validação contra as faixas
reais da `/quote` (`idade` 0–200, `veiculo_ano` 1950–2100). Um **backstop regex leve** cobre
ano/CEP quando o LLM não está disponível ou falha. Faltando dado essencial após **N=2**
tentativas → sinal de handoff (`CLARIFY_LOOP_EXHAUSTED`).

### Resiliência da `/quote`: infra × negócio, timeout 9s, retry, breaker, nunca inventar preço (Q5)

O modelo de falha do `quote-service` (`FAILURE_RATE=0.20`, `SLOW_RATE=0.10` @
`SLOW_SECONDS=8`, mais 422/400 de negócio) é o ponto que a avaliação mais valoriza. Política
em `autoseguro/quote_client.py`:

- **Distinguir infra de negócio:** 5xx e timeout → **retry**; **422 (`CotacaoRecusada`) e
  400 (`PayloadInvalido`) nunca fazem retry** — são erros de negócio, o serviço respondeu
  corretamente, insistir seria desperdício e "burrice" do agente.
- **Timeout por tentativa = `SLOW_SECONDS + 1` (9s):** o mínimo que **captura a chamada
  lenta** (8s) como cotação válida, em vez de descartá-la. Menos que 8s jogaria fora ~10% de
  cotações boas; mais que 9s não ganha nada.
- **3 tentativas extras**, backoff exponencial `0.5 → 1 → 2s` + jitter (evita thundering
  herd em lote/replay); `/quote` é idempotente, então retry é seguro.
- **Deadline total ~25s** sobre o conjunto de tentativas — estourou, vira sinal de handoff.
- **Circuit breaker leve:** abre após N falhas de infra seguidas → fast-fail sem martelar o
  serviço; sonda `GET /health` (sempre estável no desafio) pra tentar fechar depois do
  `reset_s`. Ganho real no modo replay/lote; no caminho de uma conversa isolada, o timeout +
  retry já bastam na prática.
- **Nunca inventa preço:** ao esgotar tentativas/deadline/breaker, o cliente levanta
  `QuoteUnavailable` (nunca um preço) e o agente transborda com contexto (`QUOTE_UNAVAILABLE`)
  — o preço só existe quando vem de uma resposta 200 real.
- Tudo configurável por env; `QUOTE_SEED` no `quote-service` permite runs reproduzíveis
  (inclusive forçar 100% de falha, pra demonstrar o handoff de propósito).

**Nota de mundo real:** em produção, a latência aceitável seria capada por SLA (provavelmente
um timeout bem menor que 9s, com fila/callback assíncrono em vez de segurar a conexão) — o
timeout de 9s aqui é a escolha certa **especificamente** porque o desafio injeta uma lentidão
fixa de 8s e o objetivo é capturá-la como sucesso, não simular uma latência real de produção.

### Critério de handoff: fronteira de capacidade/autoridade, nunca conveniência (Q6)

Transborda **apenas** quando o humano pode fazer algo que o agente **estruturalmente não
pode** (`autoseguro/handoff.py`, `HandoffReason`):

| Motivo (`reason_code`) | Perna | Gatilho |
|---|---|---|
| `quote_unavailable` | Capacidade (infra) | `/quote` esgotou retries/deadline/breaker |
| `agent_error` | Capacidade (fail-safe) | Erro inesperado no agente |
| `media_unreadable` | Capacidade | Mídia essencial (áudio/imagem/doc) sem transcrição |
| `policy_issuance` | Capacidade | Fechamento/emissão de apólice (sem tool pra isso) |
| `clarify_loop_exhausted` | Capacidade | N=2 tentativas sem completar dado essencial |
| `contradictory_data` | Autoridade | Dados contraditórios / suspeita de fraude |
| `out_of_scope` | Escopo | Sinistro, boleto, cancelamento, seguro residencial |
| `explicit_request` | Respeito ao lead | Lead pede humano diretamente |
| `complaint_conflict` | Sensibilidade | Reclamação / conflito / ameaça |

**O agente resolve sozinho** (não transborda): recusa 422 por regra dura (idade > 75, veículo
> 20 anos) — explica e encerra, um humano não reverteria a regra; plano fora do catálogo —
informa os 3 planos disponíveis; objeção de preço/desconto — não há mecanismo de desconto no
sistema, então o agente é honesto sobre isso e oferece um plano mais barato como alavanca
real, só transbordando se o lead pedir humano explicitamente.

Cada `HandoffDecision` carrega um `reason_code` (enum) + contexto (idade, veículo, CEP, plano
de interesse, cotação se houve) — **mascarado antes de persistir** (Q3) — o que torna todo
transbordo auditável. Gatilhos determinísticos (`for_*`) são checados primeiro; um
classificador fuzzy **injetável/mockável** cobre só os casos ambíguos (fora de escopo, dados
contraditórios, conflito), nunca os determinísticos.

### Rastreabilidade: `trace.jsonl` (Q7)

`autoseguro/tracing.py` grava um evento JSON por linha, append-only, em `logs/trace.jsonl`:

- **Ids:** `run_id` (execução), `conversation_id` (conversa), `event_id` (cada evento,
  sempre único), `quote_request_id` (cada cotação).
- **Tipos:** `message.in`/`message.out` (status recebida/enviada), `quote.result` (status
  `success`/`recusado`/`unavailable`, com `attempts` quando aplicável), `decision` (status
  `resolved`/`handoff`), `handoff` (com `reason_code`).
- **Mascaramento:** todo evento passa por `PiiRedactor.redact_record` antes de gravar — nunca
  PII em claro em disco.

O JSONL sozinho já satisfaz literalmente "cada mensagem/cotação com id e status" e "log de
execução completa". Um transcrito legível (markdown) é só polimento derivado do mesmo JSONL,
não bloqueante para a entrega. `logging` stdlib + um formatter JSON cobrem tudo isso sem
dependência extra; OpenTelemetry/spans ficam documentados aqui como evolução natural (o
JSONL já é correlacionável por `run_id`/`conversation_id`/`quote_request_id`), não implementados
neste desafio.

### Interface de entrega: CLI de chat (Q4)

**CLI de terminal obrigatória**, turn-based, sobre um **core assíncrono** (`AsyncAnthropic` +
`httpx` async): sem porta exposta (superfície mínima de segurança), chave via env com
fail-fast, tool única (`/quote`) → blast radius mínimo. É a menor superfície que entrega
exatamente o artefato pedido (log de execução completa), com a maior segurança e o menor
atrito pro avaliador rodar.

- **Ack imediato + nudge:** ao iniciar uma chamada de cotação, a CLI avisa
  ("Deixa eu calcular sua cotação, só um instante...") e reforça com um nudge se passar de 5s
  — comportamento nativo de WhatsApp, onde a latência percebida importa.
- **Core assíncrono:** mantém timeouts/retries limpos e já deixa concorrência pronta pro
  replay opcional, sem reescrever a camada de resiliência.
- **Webhook (FastAPI):** **apenas documentado** aqui como forma de produção — não
  implementado, pra não adicionar superfície de segurança/custo sem necessidade dentro do
  escopo do desafio. Em produção, a mesma `Agent`/`QuoteClient`/`Tracer` seriam reaproveitados
  por trás de um endpoint webhook do WhatsApp Business API, mantendo o mesmo `trace.jsonl`.
- **Replay sobre o dataset:** engenharia **opcional** de robustez/testes (ver
  `autoseguro/replay.py` abaixo) — não é entregável do desafio.

### Refinamentos: caminho feliz completo e sem over-engineering

- Ao entregar a cotação, o agente explica coberturas, franquia, **carência de 30 dias**
  (roubo/furto) e **pró-rata** do primeiro pagamento — sempre a partir da resposta real da
  `/quote` (`agent.format_quote_explanation`), nunca de memória. `data_inicio` tem fallback
  (assume hoje e avisa) quando o lead não informa.
- O núcleo de resiliência simples e legível é o que mais importa; o circuit breaker (leve) e
  o core assíncrono foram mantidos, mas o CB rende mais no modo replay/lote do que numa
  conversa isolada, e o async é sobretudo o que mantém retry/timeout limpos — documentado
  aqui em vez de vendido como sofisticação desnecessária.

---

## Estrutura do código

```
autoseguro/
  config.py        # env + fail-fast (Group A)
  quote_client.py   # cliente resiliente da /quote (Group B, Q5)
  pii.py            # mascaramento at-rest + minimização (Group C, Q3)
  extraction.py     # extração/qualificação via LLM + backstop (Group D, Q3b)
  agent.py          # loop do agente, tool call_quote, explicação da cotação (Group E, Q1)
  handoff.py        # motor de handoff com reason_code (Group E, Q6)
  tracing.py        # trace.jsonl (Group F, Q7)
  cli.py            # CLI REPL turn-based (Group F, Q4)
  replay.py         # harness opcional sobre o dataset (Group G — ver abaixo)
tests/              # 136 testes, todos sem chave/rede (LLM e /quote mockados)
```

---

## Replay harness (opcional)

`autoseguro/replay.py` **não é um entregável do desafio** — é uma ferramenta de robustez
para rodar o `Agent` sobre conversas do `dataset/conversations.parquet` do
`namastex-fde-challenge` (fora deste repo) e detectar buracos de extração/handoff em lote,
antes de gravar os logs de execução reais.

- `load_conversations(path)`: carrega o parquet, ordenado por `conversation_id`/
  `message_index`.
- `lead_messages(df, conversation_id)`: extrai só as mensagens do `lead`, mapeando
  `message_type` de mídia (`image`/`audio`/`document`) pro parâmetro `media_type` que
  `Agent.handle_turn` espera.
- `replay_conversation(...)` / `replay_dataset(...)`: rodam o `Agent` turno a turno sobre uma
  ou várias conversas, com **LLM e `QuoteClient` sempre injetáveis** — em produção usaria os
  clients reais (próximo de `cli.build_agent_from_config`); nos testes (`tests/test_replay.py`)
  só entram dublês, sem chave nem rede. Concorrência é opcional (`concurrency=N`, default
  sequencial).

`uv run pytest tests/test_replay.py` cobre esse harness com conversas sintéticas que seguem o
schema de `DICIONARIO.md` — não depende do dataset real estar presente no ambiente de quem
avalia.

---

## Log de execução

Os logs de execução (happy-path e falha→handoff, mascarados, com `QUOTE_SEED` fixo pra
reprodutibilidade) ficam em `docs/execucao-happy.md` e `docs/execucao-falha-handoff.md`,
gerados a partir de rodadas reais da CLI contra o `quote-service` do desafio.
