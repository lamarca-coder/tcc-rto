"""D1 (etapa 2): Junta v0, YouTube C3 e Bluesky, remove duplicatas entre fontes (mantém v0) e calcula sentimento (transformer, checkpoint a cada 400) só nas linhas novas.

Etapa: D1 (etapa 2)
Objetivo: Junta v0, YouTube C3 e Bluesky, remove duplicatas entre fontes (mantém v0) e calcula sentimento (transformer, checkpoint a cada 400) só nas linhas novas.
Entradas: dados/v0/corpus_v0.parquet, saídas da etapa 1
Saídas: união com sentimento e etapa2_resumo.json em dados/v2/
Como rodar: python -m scripts.d_analise.d1_etapa2_sentimento
Tempo aproximado: horas (BERT em CPU)
"""

import json
import re
import time
import unicodedata
from pathlib import Path

import pandas as pd

from src.analysis.sentiment_cache import add_sentiment_cached

OUT = Path("dados/v2")


def norm(s):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(s))).strip().lower()


v0 = pd.read_parquet("dados/v0/corpus_v0.parquet")
v0["fonte_coleta"] = "v0"
yt = pd.read_parquet(OUT / "ck_youtube_c3.parquet")
yt["fonte_coleta"] = "youtube_c3"
bs = pd.read_parquet(OUT / "ck_bluesky.parquet")
bs["fonte_coleta"] = "bluesky"
todos = pd.concat([v0, yt, bs], ignore_index=True)
todos["_chave"] = todos["text"].map(norm)
antes = todos.fonte_coleta.value_counts().to_dict()
# duplicatas ENTRE fontes (texto normalizado); o v0 não perde nenhuma linha, mesmo com repetições internas
chaves_v0 = set(todos.loc[todos.fonte_coleta == "v0", "_chave"])
ytm = (todos.fonte_coleta == "youtube_c3") & todos["_chave"].isin(chaves_v0)
chaves_yt = set(todos.loc[(todos.fonte_coleta == "youtube_c3") & ~ytm, "_chave"])
bsm = (todos.fonte_coleta == "bluesky") & todos["_chave"].isin(chaves_v0 | chaves_yt)
dup = {"v0": 0, "youtube_c3": int(ytm.sum()), "bluesky": int(bsm.sum())}
dup["youtube_c3_igual_ao_v0"] = int(ytm.sum())
dup["bluesky_igual_ao_v0_ou_youtube_c3"] = int(bsm.sum())
todos = todos[~(ytm | bsm)]
depois = todos.fonte_coleta.value_counts().to_dict()
print("antes", antes, "depois", depois, "duplicatas removidas", dup, flush=True)
todos = todos.drop(columns=["_chave"])
todos.to_parquet(OUT / "ck_unido_sem_sentimento.parquet", index=False)

novas = todos[todos.fonte_coleta != "v0"].reset_index(drop=True)
print("linhas novas para sentimento:", len(novas), flush=True)
t = time.time()
novas = novas.drop(
    columns=[c for c in ("sentiment_score", "sentiment_label") if c in novas.columns]
)  # vêm vazias da concatenação com o v0
novas = add_sentiment_cached(novas, method="transformer", chunk=400)
tempo = round(time.time() - t, 1)
novas.to_parquet(OUT / "ck_novas_com_sentimento.parquet", index=False)
json.dump(
    {
        "antes": antes,
        "depois": depois,
        "duplicatas_removidas": dup,
        "linhas_novas": len(novas),
        "tempo_sentimento_s": tempo,
    },
    open(OUT / "etapa2_resumo.json", "w"),
    indent=1,
)
print("ETAPA2 CONCLUIDA", tempo, "s", flush=True)
