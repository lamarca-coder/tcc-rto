"""D5 (passos 1 a 4): LDA com k de 4 a 12, 3 sementes, coerência c_v (gensim); LDA e vetorização iguais às do pipeline.

Etapa: D5 (passos 1 a 4)
Objetivo: LDA com k de 4 a 12, 3 sementes, coerência c_v (gensim); LDA e vetorização iguais às do pipeline. Retomável.
Entradas: dados/v2/corpus_v2.parquet
Saídas: dados/v2/d5_coerencia.csv
Como rodar: python -m scripts.d_analise.d5_lda_coerencia
Tempo aproximado: horas
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

KS = range(4, 13)
SEEDS = (42, 43, 44)
NWORDS = 10
NSALVAR = 15
OUT = Path("dados/v2")
CSV = OUT / "d5_coerencia.csv"

df = pd.read_parquet(OUT / "corpus_v2.parquet", columns=["id", "tokens_str"])
docs = [d for d in df.tokens_str.fillna("") if d.strip()]
print(f"documentos com tokens: {len(docs)} de {len(df)}", flush=True)
vec = CountVectorizer(
    min_df=2, max_df=0.9, max_features=5000, token_pattern=r"(?u)\b\w\w+\b"
)
dtm = vec.fit_transform(docs)
vocab = np.array(vec.get_feature_names_out())
print("matriz documento-termo:", dtm.shape, flush=True)
texts = [d.split() for d in docs]
dic = Dictionary(texts)

feitos = (
    pd.read_csv(CSV)
    if CSV.exists()
    else pd.DataFrame(
        columns=["k", "seed", "coerencia_cv", "t_treino_s", "t_coerencia_s"]
    )
)
for k in KS:
    for seed in SEEDS:
        if ((feitos.k == k) & (feitos.seed == seed)).any():
            continue
        t0 = time.time()
        lda = LatentDirichletAllocation(
            n_components=k, learning_method="batch", max_iter=25, random_state=seed
        )
        lda.fit(dtm)
        t1 = time.time()
        top15 = [vocab[c.argsort()[::-1][:NSALVAR]].tolist() for c in lda.components_]
        topics = [t[:NWORDS] for t in top15]  # c_v sobre as 10 palavras principais
        json.dump(
            top15,
            open(OUT / f"d5_topicos_k{k}_s{seed}.json", "w", encoding="utf-8"),
            ensure_ascii=False,
        )
        cm = CoherenceModel(
            topics=topics, texts=texts, dictionary=dic, coherence="c_v", processes=1
        )
        coh = float(cm.get_coherence())
        t2 = time.time()
        feitos.loc[len(feitos)] = [k, seed, coh, round(t1 - t0, 1), round(t2 - t1, 1)]
        feitos.to_csv(CSV, index=False)
        print(
            f"k={k} seed={seed} c_v={coh:.4f} treino={t1 - t0:.0f}s coerencia={t2 - t1:.0f}s",
            flush=True,
        )
print("D5 PASSOS 1-3 CONCLUIDOS", flush=True)
