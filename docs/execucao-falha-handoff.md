# Log de execução — `/quote` falhando → handoff (o ponto que mais separa)

Execução **real** da CLI: Sonnet real + `quote-service` real do desafio configurado para
falhar **sempre** (`QUOTE_FAILURE_RATE=1.0`), para demonstrar o comportamento sob falha da
`/quote`. O agente aplica a política de resiliência, **não inventa preço** e transborda o
atendimento com um motivo auditável. PII mascarada at-rest (`⟨CEP⟩`).

Trace estruturado bruto (mascarado): [`traces/execucao-falha-handoff.trace.jsonl`](traces/execucao-falha-handoff.trace.jsonl).

## Conversa (mascarada)

> **Lead:** Oi, preciso de uma cotação de seguro
>
> **Agente:** Pra eu conseguir cotar, me conta sua idade, o ano do seu veículo, o CEP onde
> o carro fica? Se já souber, me diga também quando você quer que o seguro comece a valer
> (data de início).
>
> **Lead:** Tenho 35 anos, um Corolla 2018, CEP `⟨CEP⟩`, plano essencial, início 15/08/2026
>
> **Agente:** Só confirmando antes de cotar: você tem 35 anos, veículo Toyota Corolla 2018,
> CEP `⟨CEP⟩`. Está correto?
>
> **Lead:** sim, pode confirmar
>
> **Agente:** No momento não estou conseguindo confirmar sua cotação — nosso sistema de
> cálculo está instável. Vou te encaminhar para um consultor humano continuar o atendimento.

## O que este log demonstra

- **Resiliência:** o cliente da `/quote` fez **3 tentativas** (1 + 2 retries com backoff),
  todas falharam — visível no trace:

  ```json
  {"type": "quote.result", "status": "unavailable", "attempts": 3,
   "reason": "esgotou_tentativas:http_500",
   "quote_request_id": "0b9f2e5a45284b9c9502d3cec03b6a0b"}
  {"type": "handoff", "status": "handoff", "reason_code": "quote_unavailable"}
  ```

- **Nunca inventa preço:** ao esgotar, o agente **não fabrica** uma cotação. Note que o evento
  `quote.result` de falha não carrega **nenhum** valor monetário — não há preço a registrar
  porque nenhum existiu. (Num `success`, o mesmo evento traz `premio_mensal`, `franquia` e
  `valor_primeiro_pagamento`.)
- **Handoff explícito e defensável:** transborda com `reason_code: quote_unavailable`
  (fronteira de **capacidade** — infra fora do controle do agente), registrado no trace.
- **Cada infra caída tem seu próprio motivo:** `/quote` fora → `quote_unavailable`; LLM de
  extração fora → `agent_error` com `component: "extractor"`. Nunca
  `clarify_loop_exhausted`, que significa "o lead não informou" e culparia o lead por uma
  falha nossa.
- **Mensagem honesta ao lead**, sem travar.
- **Rastreabilidade:** `quote_request_id`, `attempts`, `reason`, `status` e `reason_code`
  do handoff, todos no trace com ids.
