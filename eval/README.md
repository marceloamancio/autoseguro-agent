# Harness de avaliação

Replay das conversas do dataset do desafio pelo pipeline real de extração
(100% LLM) e agregação de métricas contra o ground-truth.

## Métricas

- **Extração** (o que a `/quote` usa), medida contra o ground-truth do dataset
  (`lead_idade_informada`; ano parseado de `veiculo_texto`):
  - acurácia de **idade** e **ano do veículo**;
  - taxa de captura de **CEP**;
  - **cobertura**: % de conversas com os 3 essenciais completos.
- **Custo**: tokens reais (`usage`) × preço do modelo (Batch aplica −50%).

## Resultados

Modelo de produção: **Claude Sonnet 5** (100% nos campos que mudam o preço).

| Amostra | Modelo | Via | idade | ano | Custo |
|---|---|---|---|---|---|
| 80 (aleatória) | Sonnet 5 | API Batch | 100% | 100% | $1,10 |
| **150 (6%, estratificada)** | **Sonnet 5** | **Claude Code** | **100%** | **100%** | $0 (plano) |
| 300 (aleatória) | Opus 4.8 | Claude Code | 100% | 100% | $0 (plano) |
| 300 (aleatória) | Haiku 4.5 | Claude Code | 100% | 99% | $0 (plano) |

Full 2.500 estimado: **~$34,53 via Batch** / ~$67,15 live (Sonnet 5).

**Duas formas de rodar a extração do LLM:**

- **API** (`run_eval.py`): número oficial que reflete produção (tool estrita,
  Sonnet 5). Use `--mode batch` (−50%) para runs grandes.
- **Claude Code** (subagents): extração via plano, custo de API = $0. Proxy
  ótimo para iterar; ~11× mais eficiente em tokens (lê ~30 conversas por
  contexto em vez de reenviar o schema por mensagem). Não usa a tool estrita,
  então é proxy, não o pipeline exato.

## Dataset (fora do repo — contém PII)

O `.parquet` do desafio não é versionado (nomes, CPF, telefone, e-mail dos
leads). Aponte via `--dataset` ou `$NAMA_DATASET`. Sem redundância: 0 duplicatas
exatas, 2.492 de 2.500 esqueletos distintos — não dá pra encolher por dedup.

## Amostras fixas (reproduzíveis)

- `sample_30.txt` — 30 conversas (seed 42), inclui `conv_00018` (caso
  adversarial: telefone terminando em `-2080`, que o antigo regex confundia com
  ano de veículo).
- `sample_80.txt` — 80 conversas (seed 7), run "oficial" via Batch.
- `sample_300.txt` — 300 conversas (seed 99).
- `sample_repr_150.txt` — **150 conversas (6%) estratificadas** por
  `outcome × telefone × documento`, gerada por `make_subset.py`. Espelha a
  distribuição do full (ver abaixo).

### Amostra estratificada (`make_subset.py`)

Em vez de sortear ao acaso, divide o dataset em estratos pelas dimensões que
variam e importam (outcome, telefone na mensagem, documento anexo) e sorteia
proporcionalmente de cada um — o subset espelha o full. Paridade da amostra de
150:

| Dimensão | FULL | SUBSET (150) |
|---|---|---|
| em_negociacao | 30% | 30% |
| ganho | 28% | 29% |
| perdido | 22% | 21% |
| sem_resposta | 20% | 20% |
| telefone (adversarial) | 60% | 60% |
| documento | 31% | 30% |

```bash
uv run python -m eval.make_subset --n 150 --out eval/sample_repr_150.txt
```

## Uso

```bash
# amostra estratificada de 6% (representativa), live
uv run python -m eval.run_eval --sample eval/sample_repr_150.txt

# run oficial de 80 via Batch API (-50%, assíncrono)
uv run python -m eval.run_eval --sample eval/sample_80.txt --mode batch

# comparar modelo mais barato
uv run python -m eval.run_eval --sample eval/sample_30.txt --model claude-haiku-4-5

# amostra aleatória nova
uv run python -m eval.run_eval --n 50 --seed 123
```

A chave da Anthropic vem de `.env`/ambiente (via `autoseguro.config`) e nunca é
impressa.
