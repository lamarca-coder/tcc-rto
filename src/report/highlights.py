"""
Destaques (highlights) para verificação manual e inclusão no TCC.

Gera tabelas em Markdown — com **links clicáveis** para os vídeos — que
permitem ao pesquisador inspecionar exemplos concretos:
  - vídeos mais comentados;
  - vídeos mais positivos e mais negativos (sentimento médio dos comentários);
  - vídeos mais divergentes (comentários mais divididos entre pró e anti-RTO);
  - comentários de exemplo (mais curtidos, mais positivos, mais negativos).

Observação ética: os autores são anonimizados; mostramos apenas o texto público
e o link do conteúdo. Para o trabalho final, prefira citar trechos de forma
agregada/anônima (coerente com a metodologia do projeto).
"""
from __future__ import annotations

import pandas as pd

from config import OUTPUTS_DIR

EXAMPLES_DIR = OUTPUTS_DIR / "examples"


def export_examples_csv(frames: pd.DataFrame, df: pd.DataFrame) -> list[str]:
    """
    Exporta as tabelas de destaques para CSV (abríveis no Excel, utf-8-sig):
      - outputs/examples/videos_destaques.csv
      - outputs/examples/comentarios_exemplo.csv
    """
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    paths = []

    if frames is not None and not frames.empty:
        base_cols = ["frame_text", "url", "n_comentarios", "sentimento_medio",
                     "stance_comentarios", "frame_stance", "concorda"]
        prop_cols = [c for c in frames.columns if c.startswith("prop_")]
        cols = [c for c in base_cols if c in frames.columns] + prop_cols
        vid = (frames.sort_values("n_comentarios", ascending=False)[cols]
               .rename(columns={"frame_text": "titulo"}))
        p = EXAMPLES_DIR / "videos_destaques.csv"
        vid.to_csv(p, index=False, encoding="utf-8-sig")
        paths.append(str(p))

    title_col = "context_text" if "context_text" in df.columns else "source"
    base = df[["text", "sentiment_score", "stance", "score", "url"]].copy()
    base["titulo"] = df[title_col] if title_col in df.columns else ""
    parts = [
        base.sort_values("score", ascending=False).head(25).assign(categoria="mais_curtidos"),
        base.sort_values("sentiment_score").head(25).assign(categoria="mais_negativos"),
        base.sort_values("sentiment_score", ascending=False).head(25).assign(categoria="mais_positivos"),
    ]
    p = EXAMPLES_DIR / "comentarios_exemplo.csv"
    pd.concat(parts).to_csv(p, index=False, encoding="utf-8-sig")
    paths.append(str(p))
    return paths


def _cell(s, maxlen: int = 90) -> str:
    """Sanitiza uma célula de tabela Markdown (remove | e quebras de linha)."""
    s = str(s).replace("|", "/").replace("\n", " ").replace("\r", " ").strip()
    return (s[:maxlen] + "…") if len(s) > maxlen else s


def _link(title, url) -> str:
    t = _cell(title, 70) or "(sem título)"
    u = str(url or "").strip()
    return f"[{t}]({u})" if u.startswith("http") else t


def _video_table(frames: pd.DataFrame, cols_header: str) -> list[str]:
    lines = [
        f"| Vídeo (link) | {cols_header} | Coment. | Sent. médio | Stance |",
        "|---|---|---:|---:|---|",
    ]
    for _, r in frames.iterrows():
        extra = r.get("_extra", "")
        lines.append(
            f"| {_link(r['frame_text'], r['url'])} | {extra} | "
            f"{int(r['n_comentarios'])} | {r['sentimento_medio']:+.2f} | "
            f"{r['stance_comentarios']} |"
        )
    return lines


def render_highlights_md(frames: pd.DataFrame, df: pd.DataFrame,
                         top_n: int = 8) -> list[str]:
    """Monta a seção 'Exemplos para verificação' do relatório (lista de linhas)."""
    lines: list[str] = []
    lines.append("## 6. Exemplos para verificação (vídeos e comentários)\n")
    lines.append("Tabelas com links para inspeção manual e uso no trabalho. "
                 "Versões em CSV (Excel) ficam em `outputs/examples/`.\n")

    # exporta também como CSV para abrir no Excel
    try:
        export_examples_csv(frames, df)
    except Exception as exc:
        lines.append(f"_(falha ao exportar CSV: {exc})_\n")

    if frames is None or frames.empty:
        lines.append("_Sem conteúdos agregados suficientes._\n")
        return lines

    f = frames.copy()

    # 6.1 mais comentados
    lines.append("### 6.1 Vídeos mais comentados\n")
    top = f.sort_values("n_comentarios", ascending=False).head(top_n).copy()
    top["_extra"] = ""
    lines += _video_table(top, "—")
    lines.append("")

    # 6.2 mais positivos / negativos (com volume mínimo)
    fv = f[f["n_comentarios"] >= 10].copy()
    if not fv.empty:
        lines.append("### 6.2 Vídeos com comentários mais positivos\n")
        pos = fv.sort_values("sentimento_medio", ascending=False).head(top_n).copy()
        pos["_extra"] = ""
        lines += _video_table(pos, "—")
        lines.append("")

        lines.append("### 6.3 Vídeos com comentários mais negativos\n")
        neg = fv.sort_values("sentimento_medio", ascending=True).head(top_n).copy()
        neg["_extra"] = ""
        lines += _video_table(neg, "—")
        lines.append("")

        # 6.4 mais divergentes (pró e anti presentes em equilíbrio)
        pa = fv.get("prop_anti-RTO", pd.Series(0, index=fv.index)).fillna(0)
        pp = fv.get("prop_pró-RTO", pd.Series(0, index=fv.index)).fillna(0)
        fv["divergencia"] = 2 * (pa * pp) ** 0.5  # alta quando ambos presentes
        lines.append("### 6.4 Vídeos mais divergentes (comentários divididos)\n")
        div = fv.sort_values("divergencia", ascending=False).head(top_n).copy()
        div["_extra"] = [f"{d:.2f}" for d in div["divergencia"]]
        lines += _video_table(div, "Divergência")
        lines.append("")

    # 6.5 comentários de exemplo
    lines += _comment_examples_md(df)
    return lines


def _comment_table(rows: pd.DataFrame) -> list[str]:
    lines = [
        "| Comentário | Sent. | Stance | Likes | Vídeo |",
        "|---|---:|---|---:|---|",
    ]
    title_col = "context_text" if "context_text" in rows.columns else "source"
    for _, r in rows.iterrows():
        title = r.get(title_col) or r.get("source") or ""
        lines.append(
            f"| {_cell(r['text'], 110)} | {r['sentiment_score']:+.2f} | "
            f"{r.get('stance', '')} | {int(r.get('score', 0) or 0)} | "
            f"{_link(title, r.get('url'))} |"
        )
    return lines


def _comment_examples_md(df: pd.DataFrame, n: int = 6) -> list[str]:
    lines = ["### 6.5 Comentários de exemplo\n"]

    lines.append("**Mais curtidos:**\n")
    lines += _comment_table(df.sort_values("score", ascending=False).head(n))
    lines.append("")

    lines.append("**Mais negativos (pró-remoto / críticas ao presencial):**\n")
    lines += _comment_table(df.sort_values("sentiment_score").head(n))
    lines.append("")

    lines.append("**Mais positivos:**\n")
    lines += _comment_table(df.sort_values("sentiment_score", ascending=False).head(n))
    lines.append("")
    return lines
