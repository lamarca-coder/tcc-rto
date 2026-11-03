"""
Armazenamento estruturado dos dados coletados.

Conforme a metodologia do projeto, os dados são guardados em:
  - Banco de dados SQLite (consultas e persistência);
  - Arquivos parquet (formato colunar, eficiente para análise).

Os autores são anonimizados (hash) antes da persistência, respeitando os
princípios éticos de pesquisa e privacidade (Boyd e Crawford, 2012).
"""
from __future__ import annotations

import hashlib
import sqlite3
from typing import Iterable

import pandas as pd

from config import DB_PATH, RAW_PARQUET, PROCESSED_PARQUET, ensure_dirs

# Esquema canônico de um registro coletado. Todos os coletores devem
# produzir dicionários/DataFrames com (ao menos) estas colunas.
SCHEMA_COLUMNS = [
    "id",            # identificador único e estável (string)
    "platform",      # reddit | twitter | sample | linkedin
    "kind",          # post | comment
    "source",        # subreddit ou termo de busca de origem
    "author_hash",   # autor anonimizado
    "text",          # conteúdo textual
    "created_utc",   # data/hora de publicação (UTC)
    "score",         # métrica de engajamento (upvotes/likes), se houver
    "url",           # link da publicação, se houver
    "lang_hint",     # idioma sugerido pela coleta (pode ser refinado depois)
    "parent_id",     # id do conteúdo "pai" (vídeo/post) ao qual este responde
    "context_text",  # texto do "pai" (título do vídeo / post original) — framing
    "collected_at",  # quando o registro foi coletado
]


def anonymize_author(author: str | None) -> str:
    """Gera um hash curto e irreversível do autor (anonimização)."""
    if not author:
        return "anon"
    return hashlib.sha256(str(author).encode("utf-8")).hexdigest()[:16]


def get_connection() -> sqlite3.Connection:
    ensure_dirs()
    return sqlite3.connect(DB_PATH)


def init_db() -> None:
    """Cria a tabela `posts` (se não existir) e migra colunas novas."""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS posts (
                id           TEXT PRIMARY KEY,
                platform     TEXT,
                kind         TEXT,
                source       TEXT,
                author_hash  TEXT,
                text         TEXT,
                created_utc  TEXT,
                score        INTEGER,
                url          TEXT,
                lang_hint    TEXT,
                parent_id    TEXT,
                context_text TEXT,
                collected_at TEXT
            )
            """
        )
        # Migração: adiciona colunas novas a bancos criados antes do esquema atual
        existing = {row[1] for row in conn.execute("PRAGMA table_info(posts)")}
        for col in SCHEMA_COLUMNS:
            if col not in existing:
                conn.execute(f"ALTER TABLE posts ADD COLUMN {col} TEXT")
        conn.commit()


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Garante a presença de todas as colunas do esquema, na ordem certa."""
    df = df.copy()
    for col in SCHEMA_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[SCHEMA_COLUMNS]


def upsert_posts(records: Iterable[dict] | pd.DataFrame) -> int:
    """
    Insere registros no SQLite ignorando duplicatas (mesma `id`).
    Retorna o número de novos registros efetivamente inseridos.
    """
    df = records if isinstance(records, pd.DataFrame) else pd.DataFrame(list(records))
    if df.empty:
        return 0
    df = _normalize(df)

    init_db()
    with get_connection() as conn:
        before = conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
        conn.executemany(
            f"""
            INSERT OR IGNORE INTO posts ({", ".join(SCHEMA_COLUMNS)})
            VALUES ({", ".join(["?"] * len(SCHEMA_COLUMNS))})
            """,
            df.astype(object).where(pd.notnull(df), None).values.tolist(),
        )
        conn.commit()
        after = conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
    return after - before


def read_posts() -> pd.DataFrame:
    """Lê todos os registros do SQLite como DataFrame."""
    init_db()
    with get_connection() as conn:
        df = pd.read_sql_query("SELECT * FROM posts", conn)
    if not df.empty:
        df["created_utc"] = pd.to_datetime(df["created_utc"], errors="coerce", utc=True)
    return df


# --------------------------------------------------------------------------- #
# Parquet                                                                      #
# --------------------------------------------------------------------------- #
def save_parquet(df: pd.DataFrame, processed: bool = False) -> str:
    """Salva o DataFrame em parquet (bruto ou processado)."""
    ensure_dirs()
    path = PROCESSED_PARQUET if processed else RAW_PARQUET
    df.to_parquet(path, index=False)
    return str(path)


def load_parquet(processed: bool = False) -> pd.DataFrame:
    path = PROCESSED_PARQUET if processed else RAW_PARQUET
    return pd.read_parquet(path)
