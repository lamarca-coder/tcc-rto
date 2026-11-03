"""C1 (passo 6): Aplica ao Bluesky os mesmos filtros do pipeline (léxico, semântico 0,30, dedup, idioma).

Etapa: C1 (passo 6)
Objetivo: Aplica ao Bluesky os mesmos filtros do pipeline (léxico, semântico 0,30, dedup, idioma).
Entradas: dados/bluesky/bluesky_bruto.parquet
Saídas: dados/bluesky/bluesky_filtrado.parquet, funil.json
Como rodar: python -m scripts.c_coleta.filtrar_bluesky
Tempo aproximado: alguns minutos
"""

import json

import pandas as pd

from config import LANGUAGES
from src.preprocess import relevance_embed as re_
from src.preprocess.cleaning import filter_relevant, preprocess_dataframe
from src.preprocess.relevance_embed import add_relevance_embed, filter_by_similarity

# lotes menores só neste script (evita falta de memória com coletas rodando em paralelo); resultado idêntico
_orig = re_.compute_similarity
re_.compute_similarity = lambda texts, batch_size=32: _orig(texts, 32)

df = pd.read_parquet("dados/bluesky/bluesky_bruto.parquet")
f = {"bruto": len(df)}
df, _ = filter_relevant(df)
f["apos_filtro_lexico_e_exclusoes"] = len(df)
df = add_relevance_embed(df)
df, _ = filter_by_similarity(df, 0.30)
f["apos_filtro_semantico_0.30"] = len(df)
df = preprocess_dataframe(
    df, do_stem=False
)  # remove duplicatas de texto e textos vazios
f["apos_dedup_e_vazios"] = len(df)
df = df[df["lang"].isin(LANGUAGES)].reset_index(drop=True)
f["apos_idioma_pt"] = len(df)
df.drop(columns=["tokens"], errors="ignore").to_parquet(
    "dados/bluesky/bluesky_filtrado.parquet", index=False
)
d = pd.to_datetime(df.created_utc, utc=True, errors="coerce")
f["final"] = len(df)
f["data_min"] = str(d.min())
f["data_max"] = str(d.max())
f["por_ano"] = d.dt.year.value_counts().sort_index().to_dict()
json.dump(f, open("dados/bluesky/funil.json", "w"), indent=1)
print(f)
