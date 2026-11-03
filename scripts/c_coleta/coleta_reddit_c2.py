"""C2: Ampliar o Reddit via pullpush.io (descobrir subreddits e coletar).

Etapa: C2
Objetivo: Ampliar o Reddit via pullpush.io (descobrir subreddits e coletar). Parâmetros em config_coleta.yaml. Retomável. O serviço passou a bloquear agentes.
Entradas: config_coleta.yaml, config.py
Saídas: dados/reddit_c2/
Como rodar: python -m scripts.c_coleta.coleta_reddit_c2
Tempo aproximado: variável
"""

import collections
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml
from langdetect import DetectorFactory, detect

from config import KEYWORDS_PT, SUBREDDITS
from src.collect import pullpush_collector as pp
from src.storage.database import upsert_posts

DetectorFactory.seed = 0
import requests

CFG = yaml.safe_load(open("config_coleta.yaml", encoding="utf-8"))["reddit"]


def _get_paciente(endpoint, params, tentativas=8):
    """Como pp._get, mas com espera longa no 429 e ERRO (não lista vazia) se esgotar as tentativas."""
    for i in range(tentativas):
        try:
            r = requests.get(
                f"{pp.BASE}/{endpoint}/", params=params, headers=pp.HEADERS, timeout=60
            )
            if r.status_code == 200:
                return r.json().get("data", [])
            time.sleep(min(30 * (i + 1), 180) if r.status_code == 429 else 10)
        except Exception:
            time.sleep(10)
    raise RuntimeError(f"pullpush {endpoint} falhou após {tentativas} tentativas")


pp._get = _get_paciente  # _paged (inalterado) passa a usar a versão paciente

OUT = Path("dados/reddit_c2")
OUT.mkdir(parents=True, exist_ok=True)


def descobrir():
    cont, pt = collections.Counter(), collections.defaultdict(list)
    for kw in CFG["expressoes_descoberta"]:
        for ep in ("submission", "comment"):
            items = pp._paged(
                ep,
                {"q": kw, "size": CFG["size"]},
                CFG["paginas_descoberta"],
                CFG["pausa_s"],
            )
            for it in items:
                sub = it.get("subreddit")
                txt = (
                    (it.get("title") or "")
                    + " "
                    + (it.get("selftext") or it.get("body") or "")
                )
                if not sub or len(txt.strip()) < 20:
                    continue
                cont[sub] += 1
                try:
                    pt[sub].append(detect(txt) == "pt")
                except Exception:
                    pass
            print(f"descoberta '{kw}' {ep}: {len(items)}", flush=True)
    rows = [
        {
            "subreddit": s,
            "resultados": n,
            "pct_pt": round(sum(pt[s]) / max(len(pt[s]), 1), 2),
        }
        for s, n in cont.items()
    ]
    df = pd.DataFrame(rows).sort_values("resultados", ascending=False)
    df.to_csv(OUT / "descoberta_subreddits.csv", index=False)
    atuais = {s.lower() for s in SUBREDDITS}
    novos = df[(df.pct_pt >= 0.6) & ~df.subreddit.str.lower().isin(atuais)].head(8)
    novos.to_csv(OUT / "novos_subreddits.csv", index=False)
    print(novos.to_string(), flush=True)


def coletar():
    novos = pd.read_csv(OUT / "novos_subreddits.csv").subreddit.tolist()
    subs = list(SUBREDDITS) + novos
    for sub in subs:
        f = OUT / f"sub_{sub}.parquet"
        if f.exists():
            continue
        recs = []
        for kw in KEYWORDS_PT:
            for ep, fn in (
                ("submission", pp._submission_record),
                ("comment", pp._comment_record),
            ):
                items = pp._paged(
                    ep,
                    {"q": kw, "subreddit": sub, "size": CFG["size"]},
                    CFG["paginas"],
                    CFG["pausa_s"],
                )
                recs += [fn(it, kw) for it in items]
        df = pd.DataFrame(recs)
        if len(df):
            df = df.drop_duplicates("id")
            df = df[df.text.str.strip().str.len() > 0]
        df.to_parquet(f, index=False)
        print(f"r/{sub}: {len(df)} registros", flush=True)
    df = pd.concat(
        [
            pd.read_parquet(OUT / f"sub_{s}.parquet")
            for s in subs
            if (OUT / f"sub_{s}.parquet").exists()
        ]
    ).drop_duplicates("id")
    ins = upsert_posts(df)
    d = pd.to_datetime(df.created_utc, utc=True, errors="coerce")
    res = {
        "coletado_em": datetime.now(UTC).isoformat(),
        "subreddits": subs,
        "registros_unicos": len(df),
        "novos_no_sqlite": ins,
        "data_min": str(d.min()),
        "data_max": str(d.max()),
        "por_subreddit": df.source.value_counts().to_dict(),
        "por_tipo": df.kind.value_counts().to_dict(),
    }
    json.dump(
        res,
        open(OUT / "coleta_resumo.json", "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=1,
    )
    print(res, flush=True)


if __name__ == "__main__":
    {"descobrir": descobrir, "coletar": coletar}[sys.argv[1]]()
