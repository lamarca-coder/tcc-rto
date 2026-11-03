"""
Detecção de alvo (aspecto) e posicionamento (stance) sobre o RTO.

Motivação: a análise de sentimento mede a *polaridade* do comentário, mas não
o *alvo* nem a *posição*. Dois comentários com sentimentos opostos podem
expressar a MESMA opinião sobre o retorno presencial:
    "Odiei voltar pro presencial"        -> negativo, mas ANTI-RTO
    "Home office melhorou minha vida"    -> positivo, mas ANTI-RTO

Este módulo identifica sobre qual alvo o texto fala (presencial / remoto /
híbrido) e, combinando com o sentimento, deriva o posicionamento em relação ao
retorno ao trabalho presencial: pró-RTO, anti-RTO ou neutro.

É uma heurística (não lida perfeitamente com ironia ou múltiplos alvos), mas
alinha a análise à pergunta de pesquisa. Pode evoluir para um classificador de
stance treinado quando houver dados rotulados.
"""
from __future__ import annotations

import pandas as pd

from config import SENTIMENT_THRESHOLD

# Marcadores de cada alvo (substring em texto minúsculo)
TARGET_MARKERS = {
    "presencial": [
        "presencial", "escritóri", "escritori", "voltar pro escrit",
        "volta ao escrit", "retorno ao escrit", "retorno ao presenc",
        "volta ao presenc", "bater ponto", "ir pro trabalho", "no escritório",
        " rto", "return to office",
    ],
    "remoto": [
        "home office", "homeoffice", "home-office", "remoto", "remota",
        "trabalhar de casa", "trabalho de casa", "trampar de casa",
        "teletrabalho", "trabalho a distância", "trabalho a distancia",
    ],
    "hibrido": [
        "híbrido", "hibrido", "modelo hibr", "dois dias no escrit",
        "três dias no escrit", "tres dias no escrit",
    ],
}

STANCE_PRO = "pró-RTO"       # a favor do retorno presencial
STANCE_ANTI = "anti-RTO"     # contra o retorno presencial (pró-remoto)
STANCE_NEUTRO = "neutro"
STANCE_INDEF = "indefinido"  # sem alvo claro


def infer_target(text: str) -> str:
    """Retorna o alvo dominante: presencial | remoto | hibrido | geral."""
    if not isinstance(text, str):
        return "geral"
    t = text.lower()
    counts = {
        target: sum(t.count(m) for m in markers)
        for target, markers in TARGET_MARKERS.items()
    }
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else "geral"


def infer_stance(target: str, score: float,
                 threshold: float = SENTIMENT_THRESHOLD) -> str:
    """Deriva o posicionamento sobre o RTO a partir de (alvo, sentimento)."""
    pos = score >= threshold
    neg = score <= -threshold

    if target == "presencial":
        if pos:
            return STANCE_PRO
        if neg:
            return STANCE_ANTI
        return STANCE_NEUTRO
    if target == "remoto":
        if pos:
            return STANCE_ANTI    # gostar do remoto = ser contra o RTO
        if neg:
            return STANCE_PRO
        return STANCE_NEUTRO
    if target == "hibrido":
        return STANCE_NEUTRO      # híbrido = posição intermediária/flexível
    return STANCE_INDEF           # alvo "geral": não dá para inferir a posição


def add_stance(df: pd.DataFrame, text_col: str = "text",
               score_col: str = "sentiment_score",
               threshold: float = SENTIMENT_THRESHOLD) -> pd.DataFrame:
    """Adiciona as colunas `target` e `stance` ao DataFrame."""
    df = df.copy()
    df["target"] = df[text_col].map(infer_target)
    df["stance"] = [
        infer_stance(tg, sc, threshold)
        for tg, sc in zip(df["target"], df[score_col])
    ]
    return df


def stance_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Distribuição geral de posicionamento (contagem e %)."""
    counts = df["stance"].value_counts()
    perc = (counts / counts.sum() * 100).round(1)
    return pd.DataFrame({"contagem": counts, "percentual": perc})
