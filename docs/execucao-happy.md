# Log de execução — caminho feliz (cotação saindo)

Execução **real** da CLI: Sonnet real (extração + confirmação) + `quote-service` real
do desafio. `/quote` estável neste run (`QUOTE_FAILURE_RATE=0`) para demonstrar o caminho
feliz de ponta a ponta. PII mascarada **at-rest** (o CEP aparece como `⟨CEP⟩` no trace e
neste log — o mascaramento é aplicado ao que é gravado/entregue, não ao vivo).

Trace estruturado bruto (mascarado): [`traces/execucao-happy.trace.jsonl`](traces/execucao-happy.trace.jsonl).

## Conversa (mascarada)

> **Lead:** Oi, boa tarde! Quero fazer um seguro pro meu carro.
>
> **Agente:** Pra eu conseguir cotar, me conta sua idade, o ano do seu veículo, o CEP onde
> o carro fica? Se já souber, me diga também quando você quer que o seguro comece a valer
> (data de início).
>
> **Lead:** É um Toyota Corolla 2008, tenho 35 anos, meu CEP é `⟨CEP⟩` e quero que a
> vigência comece em 15/08/2026.
>
> **Agente:** Só confirmando antes de cotar: você tem 35 anos, veículo Toyota Corolla 2008,
> CEP `⟨CEP⟩`. Está correto?
>
> **Lead:** sim, está correto
>
> **Agente:** Cotação do plano **Essencial: R$ 173.85/mês**.
> Coberturas: colisao, roubo, furto.
> Franquia: R$ 4500.00.
> **Carência:** 30 dias para acionar roubo, furto.
> Como a vigência não começa no dia 1, o primeiro pagamento é proporcional (**pró-rata**):
> R$ 95.34 (17 de 31 dias do mês).

## O que este log demonstra

- **Fluxo completo:** conversa → qualifica (extração de idade/ano/CEP de texto livre) →
  confirma → **cota** → resolve.
- **Cotação real saindo** com `premio_mensal=173.85`, plano `essencial`.
- **Regras específicas explicadas** a partir da resposta da `/quote`: **carência** de 30d
  (roubo/furto) e **pró-rata** do 1º pagamento (17 de 31 dias, pois a vigência começa em
  15/08). Nada é inventado — tudo vem da API.
- **Rastreabilidade:** cada mensagem tem `event_id`; a cotação tem `quote_request_id` e
  `status: success` no trace.
- **PII mascarada** no registro (`⟨CEP⟩`).
