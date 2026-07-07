# Wish: Conserto dos furos adversariais (red-team)

**Status:** SHIPPED
**Slug:** `fix-adversarial`
**Created:** 2026-07-07

## Summary
Consertar os ~28 furos que o teste de fogo adversarial achou (detalhe em
`../../../adversarial-lab/REPORT.md`), **regressão-primeiro (TDD)**: cada furo vira um teste
que falha em `tests/`, depois implementa até verde. Conserto cirúrgico — mantém o LLM fora de
preço/resiliência/execução de handoff (validado pelo red-team); emenda onde intenção era regex
e resiliência era config não-testada.

## Scope
### IN
- P0→P1→P2, estrutural primeiro; teste de regressão por furo; vendorizar `fake_quote.py`/
  `doubles.py` do red-team; ledger `MELHORIAS.md` atualizado por stage.
### OUT
- Não reverter determinismo no que importa; handoff segue terminal; sem deps novas; não tocar
  `format_quote_explanation`/pró-rata.

## Execution Groups (= Stages do plano aprovado)
- **Stage 0** — Fundação: vendorizar `fake_quote`/`doubles`; teste de integração de produção
  (`build_agent_from_config` + QuoteClient real); **httpx `timeout=`** (P0-4).
- **Stage 1** — P0: confirmação extrair-então-diferenciar (P0-2); `normalize_data_inicio` +
  fim do loop 400 (P0-1); re-cotar pós-entrega (P0-3); resiliência coerente — 429/408 retry,
  deadline×retries coerente, breaker half-open via `/quote` (P0-5/6/7).
- **Stage 2** — P1: intenção/escopo via LLM fundido no `EXTRACT_TOOL` (P1-2/1-5);
  `CONTRADICTORY_DATA` (P1-4); `media_type` na CLI (P1-3); PII recall + sweep no log (P1-1).
- **Stage 3** — P2: CEP int zero à esquerda (P2-1); "nasci em AAAA" boundary (P2-2).

## Success Criteria
- [ ] Cada furo tem teste de regressão em `tests/` (vermelho antes, verde depois).
- [ ] `uv run pytest` verde (unit + integração de produção com `fake_quote` real).
- [ ] Re-rodar `adversarial-lab/red_team` → ataques "furo confirmado" agora falham.
- [ ] Robustez preservada: nunca inventar preço; handoff auditável; minimização de PII.
- [ ] `MELHORIAS.md` com P0/P1 marcados corrigidos + teste vinculado.

## Review Results
_Populado por /review._
