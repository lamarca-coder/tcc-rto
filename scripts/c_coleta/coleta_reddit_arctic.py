"""C2 (alternativa): Coleta Reddit pela API pública do Arctic Shift, 1 requisição por segundo, retomável por subreddit; janela de 2019-01-01 em diante.

Etapa: C2 (alternativa)
Objetivo: Coleta Reddit pela API pública do Arctic Shift, 1 requisição por segundo, retomável por subreddit; janela de 2019-01-01 em diante.
Entradas: config.py (KEYWORDS_PT, SUBREDDITS)
Saídas: dados/reddit_c2/
Como rodar: python -m scripts.c_coleta.coleta_reddit_arctic
Tempo aproximado: variável (timeouts frequentes)
"""

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

from config import KEYWORDS_PT, SUBREDDITS
from src.collect import pullpush_collector as pp
from src.storage.database import upsert_posts

B = "https://arctic-shift.photon-reddit.com/api"
H = {
    "User-Agent": "tcc-rto-research/0.1 (academic thesis, USP MBA; contact via repository owner)"
}
OUT = Path("dados/reddit_c2")
PAGES = 10
PAUSA = 1.0
req = {"n": 0, "429": 0, "timeouts": 0, "erros": 0}


def get(path, params):
    for t in range(6):
        time.sleep(PAUSA)
        req["n"] += 1
        try:
            r = requests.get(f"{B}/{path}", params=params, headers=H, timeout=90)
        except Exception:
            time.sleep(10)
            continue
        if r.status_code == 200:
            return r.json().get("data") or []
        if r.status_code == 429:
            req["429"] += 1
            time.sleep(min(int(r.headers.get("x-ratelimit-reset", 30)) + 2, 120))
            continue
        if "imeout" in r.text:
            req["timeouts"] += 1
            return None  # sinaliza: dividir janela
        time.sleep(10)
    req["erros"] += 1
    raise RuntimeError(f"Arctic Shift {path} falhou: {r.status_code}")


def paginar(path, base, ini, fim):
    """Pagina para trás no tempo (sort desc). Retorna lista ou None se houver timeout."""
    items, before = [], fim
    for _ in range(PAGES):
        p = dict(base, after=ini, limit=100)
        if before:
            p["before"] = before
        d = get(path, p)
        if d is None:
            return None
        if not d:
            break
        items += d
        if len(d) < 100:
            break
        before = min(int(x["created_utc"]) for x in d)
    return items


def consulta(path, base):
    r = paginar(path, base, "2019-01-01", None)
    if r is not None:
        return r
    out = []
    for y in range(2019, 2027):
        r = paginar(path, base, f"{y}-01-01", f"{y + 1}-01-01")
        out += r or []
    return out


def main():
    novos = pd.read_csv(OUT / "novos_subreddits.csv").subreddit.tolist()
    subs = list(SUBREDDITS) + novos
    for sub in subs:
        f = OUT / f"arctic_{sub}.parquet"
        if f.exists():
            continue
        recs = []
        for kw in KEYWORDS_PT:
            for it in consulta("posts/search", {"subreddit": sub, "title": kw}):
                recs.append(pp._submission_record(it, kw))
            for it in consulta("posts/search", {"subreddit": sub, "selftext": kw}):
                recs.append(pp._submission_record(it, kw))
            for it in consulta(
                "comments/search", {"subreddit": sub, "body": f'"{kw}"'}
            ):
                recs.append(pp._comment_record(it, kw))
        df = pd.DataFrame(recs)
        if len(df):
            df = df.drop_duplicates("id")
            df = df[df.text.str.strip().str.len() > 0]
        df.to_parquet(f, index=False)
        print(f"r/{sub}: {len(df)} registros | requisições {req}", flush=True)
    df = pd.concat(
        [pd.read_parquet(OUT / f"arctic_{s}.parquet") for s in subs]
    ).drop_duplicates("id")
    ins = upsert_posts(df)
    d = pd.to_datetime(df.created_utc, utc=True, errors="coerce")
    res = {
        "fonte": "arctic-shift API",
        "coletado_em": datetime.now(UTC).isoformat(),
        "subreddits": subs,
        "registros_unicos": len(df),
        "novos_no_sqlite": ins,
        "data_min": str(d.min()),
        "data_max": str(d.max()),
        "por_subreddit": df.source.value_counts().to_dict(),
        "por_tipo": df.kind.value_counts().to_dict(),
        "requisicoes": req,
    }
    json.dump(
        res,
        open(OUT / "coleta_arctic_resumo.json", "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=1,
    )
    print(res, flush=True)


if __name__ == "__main__":
    main()
