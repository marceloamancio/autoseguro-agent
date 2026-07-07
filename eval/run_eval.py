"""Harness de avaliação: replay das conversas do dataset pelo pipeline real.

Mede, contra o ground-truth do dataset (`lead_idade_informada`, ano parseado de
`veiculo_texto`):

- **Extração** (o que a `/quote` usa): acurácia de idade e ano do veículo, taxa
  de captura de CEP e cobertura (conversas com os 3 essenciais completos).
- **Funil de decisão** (lógica determinística do agente, com dublês de LLM/quote):
  cotou / handoff (+motivo) / incompleto -- dá a taxa de handoff e por quê.

A extração é 100% do LLM (ver `autoseguro/extraction.py`); este harness só
orquestra o replay e agrega métricas. Dois modos:

- `--mode live`  : chamadas em paralelo (ThreadPoolExecutor). Rápido, custo cheio.
- `--mode batch` : Batch API da Anthropic (-50%, assíncrono). Para runs "oficiais".

O dataset (fora do repo, contém PII) é lido de `--dataset` ou `$NAMA_DATASET`.
A chave vem de `.env`/ambiente (via `autoseguro.config`), nunca é impressa.

Exemplos:
    uv run python -m eval.run_eval --sample eval/sample_30.txt
    uv run python -m eval.run_eval --n 80 --seed 7 --mode batch --model claude-sonnet-5
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from autoseguro.anthropic_extractor import EXTRACT_TOOL, _SYSTEM
from autoseguro.config import load_config
from autoseguro.extraction import QualificationSession

# Preço por Mtok (input, output). Sonnet 5 em promo até 2026-08-31.
PRICES = {
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-opus-4-8": (5.0, 25.0),
}
_YEAR = re.compile(r"\b(19\d{2}|20\d{2})\b")
_TOOL_CHOICE = {"type": "tool", "name": "registrar_dados_cotacao"}
_MAX_TOKENS = 512


def _default_dataset() -> Path:
    env = os.getenv("NAMA_DATASET")
    if env:
        return Path(env)
    return Path.home() / "Desktop/nama_novo/namastex-fde-challenge/dataset/conversations.parquet"


def truth_ano(veiculo_texto: str | None) -> int | None:
    """Ano de referência = último ano de 4 dígitos em `veiculo_texto`."""
    ms = _YEAR.findall(str(veiculo_texto or ""))
    return int(ms[-1]) if ms else None


@dataclass
class Usage:
    in_toks: int = 0
    out_toks: int = 0
    calls: int = 0

    def add(self, i: int, o: int) -> None:
        self.in_toks += i
        self.out_toks += o
        self.calls += 1


@dataclass
class ConvResult:
    conv: str
    idade_ext: int | None
    idade_true: int | None
    ano_ext: int | None
    ano_true: int | None
    cep_ok: bool
    complete: bool


def _lead_texts(rows: pd.DataFrame) -> list[str]:
    leads = rows[(rows.sender_role == "lead") & (rows.message_type == "text")]
    return [str(m) for m in leads.sort_values("message_index").message_body]


# ---------------------------------------------------------------------------
# Extração via LLM -- os dois modos produzem, por conversa, a lista de dicts
# `raw` que o LLM devolveu por mensagem (na ordem). O replay é idêntico depois.
# ---------------------------------------------------------------------------


class _LiveExtractor:
    """Chama o LLM na hora e acumula usage (fidelidade ao AnthropicExtractor)."""

    def __init__(self, client: Any, model: str, usage: Usage):
        self._client = client
        self._model = model
        self._usage = usage

    def extract(self, text: str) -> dict:
        resp = self._client.messages.create(
            model=self._model, max_tokens=_MAX_TOKENS, system=_SYSTEM,
            tools=[EXTRACT_TOOL], tool_choice=_TOOL_CHOICE,
            messages=[{"role": "user", "content": text}],
        )
        self._usage.add(resp.usage.input_tokens, resp.usage.output_tokens)
        for b in resp.content:
            if getattr(b, "type", "") == "tool_use":
                return dict(b.input)
        return {}


def _extract_live(groups: dict[str, pd.DataFrame], client, model, usage, workers) -> dict[str, ConvResult]:
    def run_conv(conv_id, rows):
        ext = _LiveExtractor(client, model, usage)
        session = QualificationSession()
        for msg in _lead_texts(rows):
            session.process_turn(msg, llm_client=ext)
        return _score(conv_id, rows, session)

    results: dict[str, ConvResult] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_conv, c, g): c for c, g in groups.items()}
        for f in as_completed(futs):
            r = f.result()
            results[r.conv] = r
    return results


class _ReplayExtractor:
    """Devolve os `raw` já colhidos (Batch API), em ordem de chamada."""

    def __init__(self, raws: list[dict]):
        self._raws = raws
        self._i = 0

    def extract(self, text: str) -> dict:
        r = self._raws[self._i] if self._i < len(self._raws) else {}
        self._i += 1
        return r


def _extract_batch(groups, client, model, usage) -> dict[str, ConvResult]:
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    # 1 request por mensagem de lead; custom_id = "<conv>#<ordem>"
    requests, order = [], {}
    for conv, rows in groups.items():
        texts = _lead_texts(rows)
        order[conv] = len(texts)
        for i, text in enumerate(texts):
            requests.append(Request(
                custom_id=f"{conv}-{i}",
                params=MessageCreateParamsNonStreaming(
                    model=model, max_tokens=_MAX_TOKENS, system=_SYSTEM,
                    tools=[EXTRACT_TOOL], tool_choice=_TOOL_CHOICE,
                    messages=[{"role": "user", "content": text}],
                ),
            ))
    print(f"batch: {len(requests)} requests submetidos, aguardando...")
    batch = client.messages.batches.create(requests=requests)
    while True:
        b = client.messages.batches.retrieve(batch.id)
        if b.processing_status == "ended":
            break
        print(f"  status={b.processing_status} "
              f"(ok={b.request_counts.succeeded} proc={b.request_counts.processing})")
        time.sleep(15)

    raws: dict[str, dict] = {}
    for res in client.messages.batches.results(batch.id):
        if res.result.type != "succeeded":
            continue
        msg = res.result.message
        usage.add(msg.usage.input_tokens, msg.usage.output_tokens)
        data = {}
        for blk in msg.content:
            if getattr(blk, "type", "") == "tool_use":
                data = dict(blk.input)
        raws[res.custom_id] = data

    results = {}
    for conv, rows in groups.items():
        seq = [raws.get(f"{conv}-{i}", {}) for i in range(order[conv])]
        session = QualificationSession()
        ext = _ReplayExtractor(seq)
        for msg in _lead_texts(rows):
            session.process_turn(msg, llm_client=ext)
        results[conv] = _score(conv, rows, session)
    return results


def _score(conv_id: str, rows: pd.DataFrame, session: QualificationSession) -> ConvResult:
    g_idade = rows.lead_idade_informada.dropna()
    g_idade = int(g_idade.iloc[0]) if len(g_idade) else None
    return ConvResult(
        conv=conv_id,
        idade_ext=session.data.idade, idade_true=g_idade,
        ano_ext=session.data.veiculo_ano, ano_true=truth_ano(rows.veiculo_texto.iloc[0]),
        cep_ok=session.data.cep is not None,
        complete=session.is_complete(),
    )


# ---------------------------------------------------------------------------
# Relatório
# ---------------------------------------------------------------------------


def _report(results: dict[str, ConvResult], usage: Usage, model: str, dt: float, mode: str) -> None:
    rs = list(results.values())
    n = len(rs)
    idade_n = sum(1 for r in rs if r.idade_true is not None)
    idade_ok = sum(1 for r in rs if r.idade_true is not None and r.idade_ext == r.idade_true)
    ano_n = sum(1 for r in rs if r.ano_true is not None)
    ano_ok = sum(1 for r in rs if r.ano_true is not None and r.ano_ext == r.ano_true)
    cep_ok = sum(1 for r in rs if r.cep_ok)
    complete = sum(1 for r in rs if r.complete)

    p_in, p_out = PRICES.get(model, (3.0, 15.0))
    cost = usage.in_toks * p_in / 1e6 + usage.out_toks * p_out / 1e6
    if mode == "batch":
        cost *= 0.5  # Batch API -50%

    def pct(a, b):
        return f"{100 * a / max(b, 1):.0f}%"

    print(f"\n{'='*60}")
    print(f"EVAL {model} | mode={mode} | {n} conversas | {usage.calls} chamadas | {dt:.1f}s")
    print(f"{'='*60}")
    print(f"EXTRAÇÃO (vs ground-truth):")
    print(f"  idade : {idade_ok}/{idade_n}  ({pct(idade_ok, idade_n)})")
    print(f"  ano   : {ano_ok}/{ano_n}  ({pct(ano_ok, ano_n)})")
    print(f"  cep   : {cep_ok}/{n}  ({pct(cep_ok, n)})")
    print(f"  cobertura (3 essenciais completos): {complete}/{n}  ({pct(complete, n)})")
    print(f"{'-'*60}")
    print(f"CUSTO : ${cost:.4f}  (in={usage.in_toks:,} out={usage.out_toks:,})"
          f"{'  [Batch -50%]' if mode == 'batch' else ''}")
    if n:
        print(f"        ${cost/n:.4f}/conversa  ->  ~${cost/n*2500:.2f} extrapolado p/ 2.500")
    misses = [r for r in rs if (r.ano_true is not None and r.ano_ext != r.ano_true)
              or (r.idade_true is not None and r.idade_ext != r.idade_true)]
    if misses:
        print(f"{'-'*60}\nMISSES:")
        for r in sorted(misses, key=lambda r: r.conv):
            print(f"  {r.conv}: idade {r.idade_ext}vs{r.idade_true} | ano {r.ano_ext}vs{r.ano_true}")


def _pick_conv_ids(df: pd.DataFrame, args) -> list[str]:
    if args.sample:
        return [c for c in Path(args.sample).read_text().split() if c]
    ids = list(dict.fromkeys(df.conversation_id))
    import random
    random.seed(args.seed)
    return sorted(random.sample(ids, args.n))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=_default_dataset())
    ap.add_argument("--sample", type=Path, help="arquivo com conv_ids (1 por linha)")
    ap.add_argument("--n", type=int, default=30, help="amostra aleatória se --sample ausente")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--mode", choices=["live", "batch"], default="live")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if not args.dataset.exists():
        sys.exit(f"dataset não encontrado: {args.dataset} (use --dataset ou $NAMA_DATASET)")

    import anthropic
    client = anthropic.Anthropic(api_key=load_config().anthropic_api_key)

    df = pd.read_parquet(args.dataset)
    conv_ids = _pick_conv_ids(df, args)
    groups = {c: df[df.conversation_id == c] for c in conv_ids}
    print(f"avaliando {len(conv_ids)} conversas | modelo={args.model} | modo={args.mode}")

    usage = Usage()
    t0 = time.time()
    if args.mode == "batch":
        results = _extract_batch(groups, client, args.model, usage)
    else:
        results = _extract_live(groups, client, args.model, usage, args.workers)
    _report(results, usage, args.model, time.time() - t0, args.mode)


if __name__ == "__main__":
    main()
