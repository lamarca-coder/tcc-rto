"""
Cache de sentimento por `id`.

A inferência com transformer (BERT) é a etapa mais cara do pipeline (~minutos a
horas). Como o corpus cresce de forma incremental (novas coletas), este cache
guarda o `sentiment_score` já calculado por publicação, de modo que cada novo
processamento só rode o modelo nos itens **novos**.

O cache é por método (lexicon/transformer), em data/processed/sentiment_cache_*.parquet.
"""
from __future__ import annotations

import gc

import pandas as pd

from config import PROCESSED_DIR
from src.analysis.sentiment import add_sentiment, _label_from_score, SentimentAnalyzer


def _cache_path(method: str):
    return PROCESSED_DIR / f"sentiment_cache_{method}.parquet"


def load_cache(method: str) -> pd.DataFrame:
    p = _cache_path(method)
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame(columns=["id", "sentiment_score"])


def seed_cache_from(df: pd.DataFrame, method: str = "transformer") -> int:
    """Popula o cache a partir de um DataFrame que já tem `sentiment_score`."""
    if "sentiment_score" not in df.columns or "id" not in df.columns:
        return 0
    cache = load_cache(method)
    add = df[["id", "sentiment_score"]].dropna()
    cache = pd.concat([cache, add], ignore_index=True).drop_duplicates("id")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cache.to_parquet(_cache_path(method), index=False)
    return len(cache)


def add_sentiment_cached(df: pd.DataFrame, method: str = "transformer",
                         chunk: int = 400) -> pd.DataFrame:
    """
    Como `add_sentiment`, mas reaproveita scores já calculados (por `id`) e
    grava o cache a cada `chunk` itens (checkpoint incremental) — assim um
    crash/OOM não perde o progresso e o run é retomável.

    Para o método 'lexicon' (rápido) não usa cache.
    """
    if method == "lexicon":
        return add_sentiment(df, method="lexicon")

    cache = load_cache(method)
    known = set(cache["id"])
    new = df[~df["id"].isin(known)].reset_index(drop=True)

    if len(new):
        print(f"[sentimento] cache: {len(df) - len(new)} reaproveitados, "
              f"{len(new)} novos via {method} (checkpoint a cada {chunk}).")
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        # UM único analisador reutilizado (evita recarregar o modelo a cada lote)
        analyzer = SentimentAnalyzer(method=method)
        for i in range(0, len(new), chunk):
            part = new.iloc[i:i + chunk]
            scored = analyzer.analyze_dataframe(part)
            cache = (pd.concat([cache, scored[["id", "sentiment_score"]]],
                               ignore_index=True).drop_duplicates("id"))
            cache.to_parquet(_cache_path(method), index=False)  # checkpoint
            print(f"   checkpoint: {min(i + chunk, len(new))}/{len(new)} pontuados")
            del scored, part
            gc.collect()  # libera memória entre lotes (evita OOM)
    else:
        print(f"[sentimento] cache: todos os {len(df)} reaproveitados.")

    out = df.merge(cache, on="id", how="left")
    out["sentiment_score"] = out["sentiment_score"].fillna(0.0)
    out["sentiment_label"] = out["sentiment_score"].map(_label_from_score)
    return out
