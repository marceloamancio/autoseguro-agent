# Log de execução — `/quote` falhando → handoff (o ponto que mais separa)

Execução **real** da CLI: Sonnet real + `quote-service` real do desafio configurado para
falhar **sempre** (`QUOTE_FAILURE_RATE=1.0`), para demonstrar o comportamento sob falha da
`/quote`. O agente aplica a política de resiliência, **não inventa preço** e transborda o
atendimento com um motivo auditável. PII mascarada at-rest (`⟨CEP⟩`).

Trace estruturado bruto (mascarado): [`traces/execucao-falha-handoff.trace.jsonl`](traces/execucao-falha-handoff.trace.jsonl).

## Conversa (mascarada)

> **Lead:** Oi, quero um seguro pro meu carro.
>
> **Agente:** Pra eu conseguir cotar, me conta sua idade, o ano do seu veículo, o CEP onde
> o carro fica? Se já souber, me diga também quando você quer que o seguro comece a valer
> (data de início).
>
> **Lead:** Toyota Corolla 2008, tenho 35 anos, CEP `⟨CEP⟩`, começar em 15/08/2026.
>
> **Agente:** Só confirmando antes de cotar: você tem 35 anos, veículo Toyota Corolla 2008,
> CEP `⟨CEP⟩`. Está correto?
>
> **Lead:** sim, está correto
>
> **Agente:** No momento não estou conseguindo confirmar sua cotação — nosso sistema de
> cálculo está instável. Vou te encaminhar para um consultor humano continuar o atendimento.

## O que este log demonstra

- **Resiliência:** o cliente da `/quote` fez **4 tentativas** (1 + 3 retries com backoff),
  todas falharam (`http_502`) — visível no trace: `"attempts": 4`,
  `"reason": "esgotou_tentativas:http_502"`, `"status": "unavailable"`.
- **Nunca inventa preço:** ao esgotar, o agente **não fabrica** uma cotação.
- **Handoff explícito e defensável:** transborda com `reason_code: quote_unavailable`
  (fronteira de **capacidade** — infra fora do controle do agente), registrado no trace.
- **Mensagem honesta ao lead**, sem travar.
- **Rastreabilidade:** `quote_request_id`, `attempts`, `reason`, `status` e `reason_code`
  do handoff, todos no trace com ids.
