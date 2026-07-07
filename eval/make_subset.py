"""Gera um subset ESTRATIFICADO representativo do dataset.

O dataset do desafio não tem redundância (0 duplicatas exatas; 2.492 de 2.500
esqueletos distintos), então não dá pra encolher por deduplicação. Mas dá pra
amostrar de forma estratificada pelas dimensões que de fato variam e importam
pra extração/handoff:

- `outcome`   : ganho / em_negociacao / perdido / sem_resposta
- `telefone`  : mensagem contém telefone (adversarial pro ano -- o caso "2080")
- `documento` : conversa tem anexo (relevante pra mídia/handoff)

Alocação proporcional por célula (com seed fixa), preservando a distribuição do
full. Uso:

    uv run python -m eval.make_subset --n 150 --out eval/sample_repr_150.txt
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

_PHONE = re.compile(r"\+?55|\d{4,5}-?\d{4}")


def _features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cid, g in df.groupby("conversation_id"):
        leads = g[(g.sender_role == "lead") & (g.message_type == "text")]
        txt = " ".join(str(m).lower() for m in leads.message_body)
        rows.append({
            "conv": cid,
            "outcome": g.conversation_outcome.iloc[0],
            "telefone": bool(_PHONE.search(txt)),
            "documento": bool((g.message_type == "document").any()),
        })
    return pd.DataFrame(rows)


def stratified_sample(df: pd.DataFrame, n: int, seed: int = 42) -> list[str]:
    feats = _features(df)
    feats["strato"] = (
        feats.outcome + "|tel=" + feats.telefone.astype(str) + "|doc=" + feats.documento.astype(str)
    )
    total = len(feats)
    picked: list[str] = []
    # alocação proporcional por estrato, arredondando pra baixo; sobra é preenchida
    for strato, grp in feats.groupby("strato"):
        k = round(n * len(grp) / total)
        k = min(k, len(grp))
        if k > 0:
            picked += list(grp.sample(k, random_state=seed).conv)
    # ajuste fino pro tamanho exato
    if len(picked) > n:
        picked = list(pd.Series(picked).sample(n, random_state=seed))
    elif len(picked) < n:
        resto = feats[~feats.conv.isin(picked)].sample(n - len(picked), random_state=seed)
        picked += list(resto.conv)
    return sorted(set(picked))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path,
                    default=Path.home() / "Desktop/nama_novo/namastex-fde-challenge/dataset/conversations.parquet")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.dataset)
    sample = stratified_sample(df, args.n, args.seed)
    args.out.write_text("\n".join(sample) + "\n")

    # relatório de paridade full vs subset
    feats = _features(df)
    sub = feats[feats.conv.isin(sample)]
    print(f"subset: {len(sample)} conversas -> {args.out}\n")
    print(f"{'dimensão':<22}{'FULL':>10}{'SUBSET':>10}")
    print("-" * 42)
    for col in ["outcome"]:
        for val, p in feats[col].value_counts(normalize=True).items():
            ps = (sub[col] == val).mean()
            print(f"{val:<22}{100*p:>9.0f}%{100*ps:>9.0f}%")
    for col in ["telefone", "documento"]:
        print(f"{col+' (True)':<22}{100*feats[col].mean():>9.0f}%{100*sub[col].mean():>9.0f}%")


if __name__ == "__main__":
    main()
