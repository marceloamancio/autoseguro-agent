# AutoSeguro Agent

Agente de vendas por WhatsApp da seguradora fictícia **AutoSeguro** (desafio FDE
Namastex). Atende leads de seguro de veículo de ponta a ponta:

> **conversa → qualifica → cota (via API) → decide** (resolve sozinho ou passa pro humano)

> ⚠️ **Status:** em construção. Este README é o entregável de decisões do desafio e
> vai sendo preenchido conforme a solução avança.

## Como rodar

Pré-requisitos: a API de cotação do desafio no ar e a sua chave Anthropic.

```bash
# 1. Suba o quote-service (no repo do desafio)
docker compose up --build            # API em http://localhost:8000

# 2. Configure o ambiente (a chave é SUA — nunca vai pro git)
cp .env.example .env                 # edite e cole sua ANTHROPIC_API_KEY

# 3. Rode o agente
# (comando definido na implementação)
```

Variáveis (ver `.env.example`): `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` (default
`claude-sonnet-5`), `QUOTE_API_URL`.

## Decisões de engenharia

Registro completo em [`../DECISOES.md`](../DECISOES.md) no workspace. Resumo:

- **Runtime:** agente enxuto direto no **SDK da Anthropic**, com a `/quote` como *tool* —
  mantém a lógica de resiliência e de handoff explícita e legível.
- **Modelo:** `claude-sonnet-5` — melhor custo-benefício para agente com function-calling
  (tool-use nativo + `strict:true` garantem payload válido pra `/quote`).
- **Segredos:** a chave nunca entra no repo; o avaliador usa a dele via `ANTHROPIC_API_KEY`.
  `.env` fica no `.gitignore`; fail-fast claro se faltar a chave.
- _(Em aberto: mascaramento de PII, interface de entrega, política de resiliência da
  `/quote`, critério de handoff, formato de rastreabilidade — ver `DECISOES.md`.)_

## Log de execução

Uma execução completa (conversa → cotação) mascarada será versionada aqui na entrega.
