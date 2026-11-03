"""utilitário (v0): Importa datasets externos (Kaggle, Hugging Face, Pushshift, LinkedIn manual) para o esquema interno e o banco SQLite.

Etapa: utilitário (v0)
Objetivo: Importa datasets externos (Kaggle, Hugging Face, Pushshift, LinkedIn manual) para o esquema interno e o banco SQLite.
Entradas: CSV ou parquet informado na linha de comando
Saídas: data/rto.db
Como rodar: python -m scripts.utilitarios.import_dataset
Tempo aproximado: segundos
"""

from __future__ import annotations

import argparse
import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.storage import database as db
from src.storage.database import SCHEMA_COLUMNS, anonymize_author


def _read_any(path: str, sep: str = ",") -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    if p.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(p)
    if p.suffix.lower() in (".csv", ".txt", ".tsv"):
        sep = "\t" if p.suffix.lower() == ".tsv" else sep
        return pd.read_csv(p, sep=sep)
    if p.suffix.lower() in (".json", ".jsonl"):
        return pd.read_json(p, lines=p.suffix.lower() == ".jsonl")
    raise ValueError(f"Extensão não suportada: {p.suffix}")


def _make_id(platform: str, text: str, date_val) -> str:
    h = hashlib.sha1(f"{text}|{date_val}".encode()).hexdigest()[:16]
    return f"{platform}_{h}"


def import_dataset(
    path: str,
    text_col: str,
    date_col: str | None = None,
    platform: str = "external",
    source: str | None = None,
    kind: str = "post",
    author_col: str | None = None,
    score_col: str | None = None,
    url_col: str | None = None,
    id_col: str | None = None,
    parent_id_col: str | None = None,
    context_col: str | None = None,
    lang: str | None = None,
    sep: str = ",",
) -> pd.DataFrame:
    """Lê um dataset externo, mapeia para o esquema e grava no SQLite."""
    raw = _read_any(path, sep=sep)
    if text_col not in raw.columns:
        raise KeyError(
            f"Coluna de texto '{text_col}' não existe. Colunas: {list(raw.columns)}"
        )

    now = datetime.now(UTC).isoformat()
    out = pd.DataFrame()
    out["text"] = raw[text_col].astype(str)

    # data
    if date_col and date_col in raw.columns:
        out["created_utc"] = pd.to_datetime(
            raw[date_col], errors="coerce", utc=True
        ).dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    else:
        out["created_utc"] = None

    # id
    if id_col and id_col in raw.columns:
        out["id"] = platform + "_" + raw[id_col].astype(str)
    else:
        out["id"] = [
            _make_id(platform, t, d) for t, d in zip(out["text"], out["created_utc"])
        ]

    out["platform"] = platform
    out["kind"] = kind
    out["source"] = source or Path(path).stem
    out["author_hash"] = (
        raw[author_col].map(anonymize_author)
        if author_col and author_col in raw.columns
        else "anon"
    )
    out["score"] = (
        pd.to_numeric(raw[score_col], errors="coerce").fillna(0).astype(int)
        if score_col and score_col in raw.columns
        else 0
    )
    out["url"] = raw[url_col].astype(str) if url_col and url_col in raw.columns else ""
    out["lang_hint"] = lang  # se informado, fixa o idioma (ex.: 'pt')
    # vínculo comentário -> conteúdo-pai (para a análise de framing)
    out["parent_id"] = (
        platform + "_" + raw[parent_id_col].astype(str)
        if parent_id_col and parent_id_col in raw.columns
        else None
    )
    out["context_text"] = (
        raw[context_col].astype(str)
        if context_col and context_col in raw.columns
        else None
    )
    out["collected_at"] = now

    # normaliza, remove textos vazios e duplicatas
    out = out[[c for c in SCHEMA_COLUMNS]]
    out = out[out["text"].str.strip().str.len() > 0]
    out = out.drop_duplicates(subset="id")

    inserted = db.upsert_posts(out)
    db.save_parquet(out, processed=False)
    print(f"Importados {len(out)} registros de '{path}' ({inserted} novos no SQLite).")
    print("Agora rode:  python run_pipeline.py --source db")
    return out


def main():
    ap = argparse.ArgumentParser(description="Importa dataset externo para o pipeline.")
    ap.add_argument("path", help="Caminho do CSV/parquet/json.")
    ap.add_argument("--text-col", required=True, help="Coluna com o texto.")
    ap.add_argument("--date-col", help="Coluna com a data de publicação.")
    ap.add_argument(
        "--platform",
        default="external",
        help="Origem lógica (reddit, linkedin, twitter, kaggle...).",
    )
    ap.add_argument("--source", help="Identificador do dataset (procedência).")
    ap.add_argument("--kind", default="post", choices=["post", "comment"])
    ap.add_argument("--author-col", help="Coluna de autor (será anonimizada).")
    ap.add_argument("--score-col", help="Coluna de engajamento (likes/upvotes).")
    ap.add_argument("--url-col", help="Coluna de URL.")
    ap.add_argument("--id-col", help="Coluna de id único (se houver).")
    ap.add_argument(
        "--parent-id-col", help="Coluna do id do conteúdo-pai (post original)."
    )
    ap.add_argument("--context-col", help="Coluna do texto-pai (título/post original).")
    ap.add_argument("--lang", help="Fixa o idioma dos registros (ex.: pt).")
    ap.add_argument("--sep", default=",", help="Separador do CSV (padrão ',').")
    args = ap.parse_args()

    import_dataset(
        path=args.path,
        text_col=args.text_col,
        date_col=args.date_col,
        platform=args.platform,
        source=args.source,
        kind=args.kind,
        author_col=args.author_col,
        score_col=args.score_col,
        url_col=args.url_col,
        id_col=args.id_col,
        parent_id_col=args.parent_id_col,
        context_col=args.context_col,
        lang=args.lang,
        sep=args.sep,
    )


if __name__ == "__main__":
    main()
