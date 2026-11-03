"""
Análise temporal das percepções.

Observa a evolução das opiniões ao longo do tempo, permitindo identificar
possíveis mudanças de percepção associadas a eventos externos (políticas
organizacionais, tendências de mercado etc.).
"""
from __future__ import annotations

import pandas as pd

from config import TIME_FREQ


def _norm_freq(freq: str) -> str:
    """Normaliza aliases de frequência depreciados do pandas ('M' -> 'ME')."""
    return {"M": "ME", "Y": "YE", "Q": "QE"}.get(freq, freq)


def _ensure_datetime(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    df = df.dropna(subset=[date_col])
    return df


def volume_over_time(df: pd.DataFrame, date_col: str = "created_utc",
                     freq: str = TIME_FREQ) -> pd.DataFrame:
    """Contagem de publicações por período."""
    freq = _norm_freq(freq)
    df = _ensure_datetime(df, date_col)
    series = (
        df.set_index(date_col)
        .resample(freq)
        .size()
        .rename("volume")
        .reset_index()
    )
    return series


def sentiment_over_time(df: pd.DataFrame, date_col: str = "created_utc",
                        label_col: str = "sentiment_label",
                        score_col: str = "sentiment_score",
                        freq: str = TIME_FREQ) -> pd.DataFrame:
    """
    Evolução do sentimento por período:
      - escore médio;
      - proporção de positivos/negativos/neutros.
    """
    freq = _norm_freq(freq)
    df = _ensure_datetime(df, date_col).set_index(date_col)

    mean_score = df[score_col].resample(freq).mean().rename("score_medio")

    props = (
        df.groupby([pd.Grouper(freq=freq), label_col])
        .size()
        .unstack(fill_value=0)
    )
    proportions = props.div(props.sum(axis=1), axis=0)
    proportions.columns = [f"prop_{c}" for c in proportions.columns]

    out = pd.concat([mean_score, proportions], axis=1).reset_index()
    return out


def sentiment_summary(df: pd.DataFrame,
                      label_col: str = "sentiment_label") -> pd.DataFrame:
    """Distribuição geral de sentimento (contagem e percentual)."""
    counts = df[label_col].value_counts()
    perc = (counts / counts.sum() * 100).round(1)
    return pd.DataFrame({"contagem": counts, "percentual": perc})


def sentiment_by_group(df: pd.DataFrame, group_col: str,
                       label_col: str = "sentiment_label") -> pd.DataFrame:
    """Distribuição de sentimento por grupo (ex.: plataforma, idioma, tópico)."""
    tab = (
        df.groupby([group_col, label_col]).size().unstack(fill_value=0)
    )
    return tab.div(tab.sum(axis=1), axis=0).round(3)
