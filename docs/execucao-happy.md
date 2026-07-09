# Log de execução — caminho feliz (cotação saindo)

Execução **real** da CLI: Sonnet real (extração + intenção + confirmação) + `quote-service`
real do desafio. `/quote` estável neste run (`QUOTE_FAILURE_RATE=0`) para demonstrar o caminho
feliz de ponta a ponta. PII mascarada **at-rest** (o CEP aparece como `⟨CEP⟩` no trace e neste
log — o mascaramento é aplicado ao que é gravado/entregue, não ao vivo).

Trace estruturado bruto (mascarado): [`traces/execucao-happy.trace.jsonl`](traces/execucao-happy.trace.jsonl).

## Conversa (mascarada)

> **Lead:** Oi, quero fazer um seguro pro meu carro
>
> **Agente:** Pra eu conseguir cotar, me conta sua idade, o ano do seu veículo, o CEP onde
> o carro fica? Se já souber, me diga também quando você quer que o seguro comece a valer
> (data de início).
>
> **Lead:** Tenho 35 anos, dirijo um Corolla 2018, meu CEP é `⟨CEP⟩`, quero o plano essencial
> e começar dia 15/08/2026
>
> **Agente:** Só confirmando antes de cotar: você tem 35 anos, veículo Toyota Corolla 2018,
> CEP `⟨CEP⟩`. Está correto?
>
> **Lead:** sim, pode confirmar
>
> **Agente:** Cotação do plano **Essencial: R$ 137.88/mês**.
> Coberturas: colisao, roubo, furto.
> Franquia: R$ 4500.00.
> **Carência:** 30 dias para acionar roubo, furto.
> Como a vigência não começa no dia 1, o primeiro pagamento é proporcional (**pró-rata**):
> R$ 75.61 (17 de 31 dias do mês).
>
> **Lead:** valeu, obrigado!
>
> **Agente:** Por nada! Se precisar confirmar mais alguma coisa sobre essa cotação, é só
> chamar. O próximo passo (fechamento e emissão da apólice) fica com um consultor humano, que
> vai te dar sequência. 🙂

## O que este log demonstra

- **Fluxo completo:** conversa → qualifica (extração de idade/ano/CEP de texto livre) →
  confirma → **cota** → resolve → conversa livre pós-cotação.
- **Cotação real saindo** com `premio_mensal=137.88`, plano `essencial` (Corolla 2018 = 7
  anos → multiplicador de idade do veículo 1.15; `119.90 × 1.15 = 137.88`).
- **Regras específicas explicadas** a partir da resposta da `/quote`: **carência** de 30d
  (roubo/furto) e **pró-rata** do 1º pagamento — R$ 75.61 = `137.88 × 17/31` (a vigência
  começa em 15/08). Nada é inventado — o valor é repassado **fiel** à API.
- **Rastreabilidade:** cada mensagem tem `event_id`; a cotação tem `quote_request_id`,
  `status: success` e **os três valores monetários** (`premio_mensal`, `franquia`,
  `valor_primeiro_pagamento`) — é isso que torna verificável por máquina a invariante
  "todo `R$` mostrado ao lead veio de uma cotação real":

  ```json
  {"type": "quote.result", "status": "success", "plano_id": "essencial",
   "premio_mensal": 137.88, "franquia": 4500, "valor_primeiro_pagamento": 75.61,
   "quote_request_id": "309e32b287f34eb0b385f650caf766bd"}
  ```

- **PII mascarada** no registro (`⟨CEP⟩`).
- **A conversa livre não promete o que não pode cumprir:** no último turno o agente diz que o
  fechamento "fica com um consultor humano" em vez de afirmar que já transferiu — o handoff é
  decisão determinística do Python (`handoff.py`), nunca do LLM.
