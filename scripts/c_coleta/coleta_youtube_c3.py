"""C3: Ampliar o YouTube com order=viewCount; conta a cota (search=100, commentThreads=1) e para em 9.800 unidades.

Etapa: C3
Objetivo: Ampliar o YouTube com order=viewCount; conta a cota (search=100, commentThreads=1) e para em 9.800 unidades.
Entradas: config_coleta.yaml, .env (YOUTUBE_API_KEY)
Saídas: dados/youtube_c3/youtube_viewcount.parquet, coleta_resumo.json
Como rodar: python -m scripts.c_coleta.coleta_youtube_c3
Tempo aproximado: dezenas de minutos
"""

import json
import re
import sqlite3
import sys
import time
from datetime import UTC, datetime

import pandas as pd
import requests
import yaml

from config import KEYWORDS_PT
from src.collect import youtube_collector as yc
from src.storage.database import upsert_posts

CFG = yaml.safe_load(open("config_coleta.yaml", encoding="utf-8"))["youtube"]
LIMITE = 9800
quota = {"search": 0, "comments": 0}
_get = requests.get


def contado(url, **kw):
    if url == yc.SEARCH_URL:
        quota["search"] += 100
    elif url == yc.COMMENTS_URL:
        quota["comments"] += 1
    return _get(url, **kw)


yc.requests.get = contado


def limpa(e):
    return re.sub(r"key=[^&\s]+", "key=***", str(e))


def total():
    return quota["search"] + quota["comments"]


def busca(kw, key, after, before):
    p = {
        "key": key,
        "q": kw,
        "part": "snippet",
        "type": "video",
        "maxResults": CFG["videos_por_busca"],
        "relevanceLanguage": "pt",
        "regionCode": "BR",
        "order": "viewCount",
    }
    if after:
        p["publishedAfter"] = after
    if before:
        p["publishedBefore"] = before
    for t in range(5):
        r = yc.requests.get(yc.SEARCH_URL, params=p, timeout=30)
        if r.status_code != 429:
            break
        time.sleep(15 * (t + 1))
    r.raise_for_status()
    return [
        (i["id"]["videoId"], i["snippet"]["title"])
        for i in r.json().get("items", [])
        if i.get("id", {}).get("videoId")
    ]


def main():
    key = yc._api_key()
    con = sqlite3.connect("data/rto.db")
    ja = {
        r[0]
        for r in con.execute(
            "SELECT DISTINCT parent_id FROM posts WHERE platform='youtube' AND parent_id IS NOT NULL"
        )
    }
    ja = {x.replace("yt_video_", "") for x in ja}
    print("vídeos já coletados:", len(ja), flush=True)
    videos, buscas, erros = {}, 0, []
    refazer = None
    if "--falhas" in sys.argv:  # refaz apenas as buscas com erro da execução anterior
        ant = json.load(open("dados/youtube_c3/coleta_resumo.json", encoding="utf-8"))
        refazer = {
            (m.group(1), m.group(2))
            for e in ant["erros"]
            for m in [re.match(r"busca '(.+)' \[(.+)\]:", e)]
            if m
        }
        print("buscas a refazer:", len(refazer), flush=True)
    for kw in KEYWORDS_PT:
        for after, before in CFG["janelas"]:
            if refazer is not None and (kw, str(after)) not in refazer:
                continue
            if total() + 100 > LIMITE:
                erros.append("limite de cota atingido nas buscas")
                break
            try:
                for vid, t in busca(kw, key, after, before):
                    videos.setdefault(vid, t)
            except Exception as exc:
                erros.append(f"busca '{kw}' [{after}]: {limpa(exc)}")
                if "quota" in str(exc).lower() or "403" in str(exc):
                    break
            buscas += 1
    novos = {v: t for v, t in videos.items() if v not in ja}
    print(
        f"buscas: {buscas} | vídeos distintos: {len(videos)} | novos: {len(novos)} | cota após buscas: {total()}",
        flush=True,
    )
    recs, feitos, vazios = [], 0, 0
    for vid, t in novos.items():
        if total() + 3 > LIMITE:
            erros.append("limite de cota atingido nos comentários")
            break
        try:
            recs += yc._video_comments(vid, t, key, CFG["comentarios_por_video"])
        except Exception as exc:
            erros.append(f"comentários {vid}: {limpa(exc)}")
        feitos += 1
        vazios = (
            0 if recs and recs[-1]["parent_id"] == f"yt_video_{vid}" else vazios + 1
        )
        if vazios >= 30:
            erros.append(
                "30 vídeos seguidos sem comentários: parando (possível cota esgotada)"
            )
            break
        if feitos % 100 == 0:
            print(f"  {feitos}/{len(novos)} vídeos | cota {total()}", flush=True)
    df = pd.DataFrame(recs)
    if len(df):
        df = df.drop_duplicates("id")
        df = df[df.text.str.len() > 0]
    df.to_parquet(
        "dados/youtube_c3/youtube_viewcount_falhas.parquet"
        if refazer is not None
        else "dados/youtube_c3/youtube_viewcount.parquet",
        index=False,
    )
    ins = upsert_posts(df) if len(df) else 0
    res = {
        "coletado_em": datetime.now(UTC).isoformat(),
        "buscas": buscas,
        "videos_distintos": len(videos),
        "videos_ja_coletados_descartados": len(videos) - len(novos),
        "videos_novos": len(novos),
        "videos_com_comentarios_coletados": feitos,
        "comentarios_unicos": len(df),
        "novos_no_sqlite": ins,
        "cota": {**quota, "total": total()},
        "erros": erros[:30],
        "n_erros": len(erros),
    }
    json.dump(
        res,
        open(
            "dados/youtube_c3/coleta_resumo_falhas.json"
            if refazer is not None
            else "dados/youtube_c3/coleta_resumo.json",
            "w",
            encoding="utf-8",
        ),
        ensure_ascii=False,
        indent=1,
    )
    print(res, flush=True)


if __name__ == "__main__":
    main()
