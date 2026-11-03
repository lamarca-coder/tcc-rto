"""D1 (passo 1): Estimativa de tempo da construção do corpus v2 (300 textos, semente 42).

Etapa: D1 (passo 1)
Objetivo: Estimativa de tempo da construção do corpus v2 (300 textos, semente 42).
Entradas: dados/youtube_c3, dados/bluesky, dados/v0, caches em data/processed
Saídas: dados/v2/v2_estimativa.json
Como rodar: python -m scripts.d_analise.d1_estimativa
Tempo aproximado: poucos minutos
"""

import html
import json
import time

import pandas as pd

from src.analysis.sentiment import SentimentAnalyzer
from src.preprocess import relevance_embed as re_
from src.preprocess.cleaning import filter_relevant

yt = pd.read_parquet("dados/youtube_c3/youtube_viewcount.parquet")
bs = pd.read_parquet("dados/bluesky/bluesky_filtrado.parquet")
v0 = pd.read_parquet("dados/v0/corpus_v0.parquet")
out = {"n_yt_c3": len(yt), "n_bs": len(bs), "n_v0": len(v0)}

# contagens exatas das etapas baratas (léxico) e de cache
t = time.time()
yt["text"] = yt.text.map(html.unescape)
yt_lex, _ = filter_relevant(yt)
out["t_lexico_s_total_yt"] = round(time.time() - t, 1)
out["yt_apos_lexico"] = len(yt_lex)
emb_cache = set(pd.read_parquet("data/processed/relevance_embed_cache.parquet").id)
sent_cache = set(
    pd.read_parquet("data/processed/sentiment_cache_transformer.parquet").id
)
out["yt_lexico_sem_cache_embedding"] = int((~yt_lex.id.isin(emb_cache)).sum())
out["yt_lexico_ja_com_sentimento_cache"] = int(yt_lex.id.isin(sent_cache).sum())
out["bs_sem_cache_embedding"] = int((~bs.id.isin(emb_cache)).sum())
out["bs_sem_cache_sentimento"] = int((~bs.id.isin(sent_cache)).sum())

# amostra de 300 (semente 42)
amostra = pd.concat(
    [
        yt.sample(100, random_state=42),
        bs.sample(100, random_state=42),
        v0.sample(100, random_state=42),
    ]
)
textos = amostra.text.astype(str).tolist()

t = time.time()
m = re_._get_model()
out["t_carga_modelo_embedding_s"] = round(time.time() - t, 1)
t = time.time()
_ = re_.compute_similarity(textos[:300], batch_size=32)
out["t_embedding_300_s"] = round(time.time() - t, 1)
t = time.time()
an = SentimentAnalyzer(method="transformer")
out["t_carga_modelo_sentimento_s"] = round(time.time() - t, 1)
df = pd.DataFrame({"text": textos, "lang": "pt"})
t = time.time()
an.analyze_dataframe(df)
out["t_sentimento_300_s"] = round(time.time() - t, 1)
json.dump(out, open("dados/v2/v2_estimativa.json", "w"), indent=1)
print(out)
