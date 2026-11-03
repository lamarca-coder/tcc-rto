"""
Filtro de relevância SEMÂNTICO por embeddings (sentence-transformers).

Complementa o filtro por palavra-chave: em vez de casar termos, mede a
**proximidade de significado** de cada comentário com o tema do estudo (debate
sobre o retorno ao trabalho presencial). Assim, descarta conteúdo que apenas
cita "home office" fora de contexto (ex.: vídeo de decoração de home office) e
mantém opiniões genuínas sobre o tema.

Modelo multilíngue leve (roda em CPU). Os escores são cacheados por `id`
(data/processed/relevance_embed_cache.parquet), com checkpoint incremental.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import PROCESSED_DIR

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_CACHE = PROCESSED_DIR / "relevance_embed_cache.parquet"

# Frases-âncora que descrevem o tema (centroide semântico do "assunto RTO")
ANCHORS = [
    "Discussão sobre o retorno ao trabalho presencial e o fim do home office.",
    "Opinião sobre trabalho remoto versus trabalho presencial no escritório.",
    "Empresas obrigando funcionários a voltar ao escritório após a pandemia.",
    "Preferência entre home office, modelo híbrido e trabalho 100% presencial.",
    "Produtividade e qualidade de vida trabalhando de casa ou no escritório.",
]

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _topic_vector() -> np.ndarray:
    m = _get_model()
    embs = m.encode(ANCHORS, normalize_embeddings=True)
    v = np.asarray(embs).mean(axis=0)
    return v / np.linalg.norm(v)


def compute_similarity(texts: list[str], batch_size: int = 256) -> np.ndarray:
    """Similaridade de cosseno de cada texto com o centroide do tema."""
    m = _get_model()
    tv = _topic_vector()
    embs = m.encode(list(texts), normalize_embeddings=True,
                    batch_size=batch_size, show_progress_bar=True)
    return np.asarray(embs) @ tv


def load_cache() -> pd.DataFrame:
    if _CACHE.exists():
        return pd.read_parquet(_CACHE)
    return pd.DataFrame(columns=["id", "rto_similarity"])


def add_relevance_embed(df: pd.DataFrame, text_col: str = "text",
                        chunk: int = 3000) -> pd.DataFrame:
    """Adiciona a coluna `rto_similarity` (cacheada por id, com checkpoint)."""
    cache = load_cache()
    known = set(cache["id"])
    new = df[~df["id"].isin(known)].reset_index(drop=True)

    if len(new):
        print(f"[embeddings] {len(df) - len(new)} em cache, {len(new)} novos.")
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        for i in range(0, len(new), chunk):
            part = new.iloc[i:i + chunk]
            sims = compute_similarity(part[text_col].astype(str).tolist())
            add = pd.DataFrame({"id": part["id"].values,
                                "rto_similarity": sims.astype(float)})
            cache = (pd.concat([cache, add], ignore_index=True)
                     .drop_duplicates("id"))
            cache.to_parquet(_CACHE, index=False)  # checkpoint
            print(f"   checkpoint embeddings: {min(i + chunk, len(new))}/{len(new)}")

    out = df.merge(cache, on="id", how="left")
    out["rto_similarity"] = out["rto_similarity"].fillna(0.0)
    return out


def filter_by_similarity(df: pd.DataFrame, threshold: float = 0.35):
    """Retorna (df_filtrado, n_removidos) mantendo similaridade >= threshold."""
    mask = df["rto_similarity"] >= threshold
    return df[mask].reset_index(drop=True), int((~mask).sum())
