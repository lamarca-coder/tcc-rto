"""B10b (passo 1): Embeddings MiniLM (paraphrase-multilingual-MiniLM-L12-v2, o do filtro semântico) de todos os textos do corpus v2.

Etapa: B10b (passo 1)
Objetivo: Embeddings MiniLM (paraphrase-multilingual-MiniLM-L12-v2, o do filtro semântico) de todos os textos do corpus v2. Retomável (blocos de 5.000).
Entradas: dados/v2/corpus_v2.parquet
Saídas: data/processed/embeddings_minilm.npy, embeddings_minilm_index.parquet, embeddings_minilm_info.json
Como rodar: python -m scripts.b_validacao.b10b_embeddings
Tempo aproximado: cerca de 1 hora em CPU
"""

import json
import time

import numpy as np
import pandas as pd

from scripts.utilitarios import util_v2 as u
from src.preprocess import relevance_embed as re_

PROC = u.ROOT / "data/processed"
PARTS = PROC / "embeddings_minilm_parts"
PARTS.mkdir(parents=True, exist_ok=True)
CHUNK, BATCH = 5000, 32
c = pd.read_parquet(u.CORPUS, columns=["id", "text", "rto_similarity"])
n = len(c)
print("textos:", n, flush=True)
model = re_._get_model()
t0 = time.time()
for i in range(0, n, CHUNK):
    f = PARTS / f"part_{i // CHUNK:03d}.npy"
    if f.exists():
        continue
    E = model.encode(
        c.text.iloc[i : i + CHUNK].astype(str).tolist(),
        normalize_embeddings=True,
        batch_size=BATCH,
        show_progress_bar=False,
    )
    np.save(f, np.asarray(E, dtype=np.float32))
    print(
        f"bloco {i // CHUNK + 1}/{(n + CHUNK - 1) // CHUNK} salvo | {time.time() - t0:.0f}s",
        flush=True,
    )
E = np.concatenate(
    [np.load(PARTS / f"part_{k:03d}.npy") for k in range((n + CHUNK - 1) // CHUNK)]
)
assert E.shape == (n, 384), E.shape
np.save(PROC / "embeddings_minilm.npy", E)
pd.DataFrame({"id": c.id.values, "pos": np.arange(n)}).to_parquet(
    PROC / "embeddings_minilm_index.parquet", index=False
)
# conferência: a similaridade com o centroide do tema recalculada deve bater com a do cache
sim = E @ re_._topic_vector()
dif = np.abs(sim - c.rto_similarity.values)
json.dump(
    {
        "n": n,
        "dim": int(E.shape[1]),
        "max_abs_dif_sim": float(dif.max()),
        "media_abs_dif_sim": float(dif.mean()),
        "tempo_s": round(time.time() - t0),
    },
    open(PROC / "embeddings_minilm_info.json", "w"),
    indent=1,
)
print(
    "EMBEDDINGS CONCLUIDOS",
    E.shape,
    "dif máx. da similaridade:",
    float(dif.max()),
    flush=True,
)
