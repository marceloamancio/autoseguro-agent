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

## Dataset (fora do repo — contém PII)

O `.parquet` do desafio não é versionado (nomes, CPF, telefone, e-mail dos
leads). Aponte via `--dataset` ou `$NAMA_DATASET`.

## Amostras fixas (reproduzíveis)

- `sample_30.txt` — 30 conversas (seed 42), inclui `conv_00018` (caso
  adversarial: telefone terminando em `-2080`, que o antigo regex confundia com
  ano de veículo).
- `sample_80.txt` — 80 conversas (seed 7), run "oficial".

## Uso

```bash
# amostra fixa de 30, live (paralelo)
uv run python -m eval.run_eval --sample eval/sample_30.txt

# run oficial de 80 via Batch API (-50%, assíncrono)
uv run python -m eval.run_eval --sample eval/sample_80.txt --mode batch

# comparar modelo mais barato
uv run python -m eval.run_eval --sample eval/sample_30.txt --model claude-haiku-4-5

# amostra aleatória nova
uv run python -m eval.run_eval --n 50 --seed 123
```

A chave da Anthropic vem de `.env`/ambiente (via `autoseguro.config`) e nunca é
impressa.
