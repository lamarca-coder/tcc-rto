"""
Análise de enquadramento (framing): conteúdo original × comentários.

Compara o "enquadramento" dado pelo conteúdo-pai (título do vídeo, post
original) com a reação do público (comentários). Funciona para qualquer fonte
que preencha `parent_id`/`context_text`; para o YouTube há fallback que deriva
o vídeo a partir da `url` e usa `source` (título) como contexto.

Exemplos de perguntas que habilita:
  - Um vídeo com título anti-RTO atrai comentários majoritariamente anti-RTO?
  - Há conteúdos "neutros/informativos" onde a audiência se divide?
"""
from __future__ import annotations

import re

import pandas as pd

from src.analysis.stance import infer_target, infer_stance, STANCE_NEUTRO, STANCE_INDEF
from src.analysis.sentiment import SentimentAnalyzer

_YT_ID_RE = re.compile(r"[?&]v=([\w-]+)")


def _derive_frame(row) -> tuple[str | None, str | None]:
    """Retorna (frame_id, frame_text) usando parent_id/context_text ou fallback."""
    pid = row.get("parent_id")
    ctx = row.get("context_text")
    if pid and isinstance(pid, str) and pid.strip():
        return pid, (ctx if isinstance(ctx, str) else "")
    # fallback YouTube: extrai o id do vídeo da URL e usa `source` como título
    url = str(row.get("url") or "")
    m = _YT_ID_RE.search(url)
    if m:
        return f"yt_video_{m.group(1)}", str(row.get("source") or "")
    # fallback genérico: agrupa pela própria origem
    src = row.get("source")
    return (str(src), str(src)) if src else (None, None)


def build_frames(df: pd.DataFrame, min_comments: int = 5) -> pd.DataFrame:
    """
    Agrega comentários por conteúdo-pai e calcula o stance do enquadramento.

    Retorna um DataFrame por "frame" com: nº de comentários, sentimento médio,
    stance dominante dos comentários, stance do título e se há concordância.
    """
    df = df.copy()
    frames = df.apply(_derive_frame, axis=1)
    df["frame_id"] = [f[0] for f in frames]
    df["frame_text"] = [f[1] for f in frames]
    df = df[df["frame_id"].notna()]

    # stance do enquadramento (título), calculado uma vez por frame
    titles = (df[["frame_id", "frame_text"]]
              .drop_duplicates("frame_id").reset_index(drop=True))
    analyzer = SentimentAnalyzer(method="lexicon")  # leve, suficiente p/ títulos
    title_scores, title_stances = [], []
    for txt in titles["frame_text"]:
        sc = analyzer.score_one(str(txt), "pt")
        title_scores.append(sc)
        title_stances.append(infer_stance(infer_target(str(txt)), sc))
    titles["frame_score"] = title_scores
    titles["frame_stance"] = title_stances

    # agregação dos comentários por frame
    def _dominant(s: pd.Series) -> str:
        vc = s[~s.isin([STANCE_INDEF])].value_counts()
        return vc.index[0] if len(vc) else STANCE_INDEF

    agg = df.groupby("frame_id").agg(
        n_comentarios=("text", "size"),
        sentimento_medio=("sentiment_score", "mean"),
        stance_comentarios=("stance", _dominant),
        url=("url", "first"),
    ).reset_index()

    # proporção de cada stance por frame (para o gráfico empilhado)
    props = (df.groupby(["frame_id", "stance"]).size().unstack(fill_value=0))
    props = props.div(props.sum(axis=1), axis=0)
    props.columns = [f"prop_{c}" for c in props.columns]
    agg = agg.merge(props.reset_index(), on="frame_id", how="left")

    out = agg.merge(titles, on="frame_id", how="left")
    out = out[out["n_comentarios"] >= min_comments]
    out["concorda"] = out["frame_stance"] == out["stance_comentarios"]
    out = out.sort_values("n_comentarios", ascending=False).reset_index(drop=True)
    return out


def framing_summary(frames: pd.DataFrame) -> dict:
    """Resumo: concordância título×comentários entre frames com stance definido."""
    definidos = frames[~frames["frame_stance"].isin([STANCE_NEUTRO, STANCE_INDEF])]
    if definidos.empty:
        return {"frames": len(frames), "com_stance_definido": 0, "concordancia_pct": None}
    pct = round(definidos["concorda"].mean() * 100, 1)
    return {
        "frames": len(frames),
        "com_stance_definido": len(definidos),
        "concordancia_pct": pct,
    }
