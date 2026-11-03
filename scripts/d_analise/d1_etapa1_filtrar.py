"""D1 (etapa 1): Filtra YouTube C3 e Bluesky e reconstrói o funil do v0.

Etapa: D1 (etapa 1)
Objetivo: Filtra YouTube C3 e Bluesky e reconstrói o funil do v0.
Entradas: dados/youtube_c3/youtube_viewcount.parquet, dados/bluesky/bluesky_filtrado.parquet, data/processed/posts_raw.parquet
Saídas: arquivos parciais e etapa1_funil.json em dados/v2/
Como rodar: python -m scripts.d_analise.d1_etapa1_filtrar
Tempo aproximado: dezenas de minutos
"""

import html
import json
import time
from pathlib import Path

import pandas as pd

from config import LANGUAGES
from src.preprocess import relevance_embed as re_
from src.preprocess.cleaning import filter_relevant, preprocess_dataframe

_orig = re_.compute_similarity
re_.compute_similarity = lambda texts, batch_size=32: _orig(
    texts, 32
)  # lotes menores; resultado idêntico
LIM = 0.30
OUT = Path("dados/v2")
OUT.mkdir(parents=True, exist_ok=True)
funil, tempos = {}, {}


def passa_filtros(df, nome, lexico=True):
    f = {"bruto": len(df)}
    t = time.time()
    if lexico:
        df, _ = filter_relevant(df)
    f["apos_lexico_e_exclusoes"] = len(df)
    tempos[f"{nome}_lexico_s"] = round(time.time() - t, 1)
    t = time.time()
    df = re_.add_relevance_embed(df)
    df, _ = re_.filter_by_similarity(df, LIM)
    f["apos_semantico_0.30"] = len(df)
    tempos[f"{nome}_semantico_s"] = round(time.time() - t, 1)
    t = time.time()
    df = preprocess_dataframe(df, do_stem=False)
    f["apos_duplicatas_e_vazios"] = len(df)
    df = df[df["lang"].isin(LANGUAGES)].reset_index(drop=True)
    f["apos_idioma_pt"] = len(df)
    tempos[f"{nome}_preproc_s"] = round(time.time() - t, 1)
    f["final"] = len(df)
    return df, f


# --- YouTube C3 (não filtrado): unescape -> mesmos filtros do v0 ---
yt = pd.read_parquet("dados/youtube_c3/youtube_viewcount.parquet")
yt["text"] = yt["text"].map(html.unescape)
yt_f, funil["youtube_c3"] = passa_filtros(yt, "youtube_c3")
yt_f.drop(columns=["tokens"], errors="ignore").to_parquet(
    OUT / "ck_youtube_c3.parquet", index=False
)
print("youtube_c3", funil["youtube_c3"], flush=True)

# --- Bluesky (já filtrado em C1): unescape + pré-processamento de novo; sem refazer léxico/semântico ---
bs = pd.read_parquet("dados/bluesky/bluesky_filtrado.parquet")
bs = bs[
    [c for c in bs.columns if c not in ("lang", "clean_text", "tokens_str", "tokens")]
]
bs["text"] = bs["text"].map(html.unescape)
fb = {
    "bruto": 43409,
    "apos_lexico_e_exclusoes": 41866,
    "apos_semantico_0.30": 36906,
    "apos_duplicatas_e_vazios": 36335,
    "apos_idioma_pt": 36335,
    "nota": "etapas até aqui feitas em C1 (resultados/C1.md)",
}
bs_f = preprocess_dataframe(bs, do_stem=False)
fb["reprocessado_com_unescape_dup_e_vazios"] = len(bs_f)
bs_f = bs_f[bs_f["lang"].isin(LANGUAGES)].reset_index(drop=True)
fb["final"] = len(bs_f)
funil["bluesky"] = fb
bs_f.drop(columns=["tokens"], errors="ignore").to_parquet(
    OUT / "ck_bluesky.parquet", index=False
)
print("bluesky", fb, flush=True)

# --- v0: funil por plataforma a partir do bruto original (121.480), só para o relatório ---
raw = pd.read_parquet("data/processed/posts_raw.parquet")
fv = {}
for p in ("youtube", "reddit"):
    d, f = passa_filtros(raw[raw.platform == p].copy(), f"v0_{p}")
    fv[p] = f
funil["v0_por_plataforma"] = fv
json.dump(
    {"funil": funil, "tempos_s": tempos},
    open(OUT / "etapa1_funil.json", "w"),
    indent=1,
    default=str,
)
print(json.dumps(funil["v0_por_plataforma"]), tempos, flush=True)
