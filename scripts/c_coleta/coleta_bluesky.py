"""C1: Coleta Bluesky (AT Protocol, searchPosts).

Etapa: C1
Objetivo: Coleta Bluesky (AT Protocol, searchPosts). Credenciais BSKY_HANDLE/BSKY_APP_PASSWORD no .env. Retomável.
Entradas: config.py (KEYWORDS_PT), .env
Saídas: dados/bluesky/bluesky_bruto.parquet, coleta_resumo.json
Como rodar: python -m scripts.c_coleta.coleta_bluesky
Tempo aproximado: dezenas de minutos (depende da API)
"""

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests
from dotenv import dotenv_values

from config import KEYWORDS_PT
from src.storage.database import anonymize_author, upsert_posts

PDS = "https://bsky.social/xrpc"
SINCE = "2023-01-01T00:00:00Z"
env = dotenv_values(".env")
HANDLE = env.get("BSKY_HANDLE") or env.get("BLUESKY_HANDLE")
PWD = env.get("BSKY_APP_PASSWORD") or env.get("BLUESKY_APP_PASSWORD")
H = {}


def login():
    r = requests.post(
        f"{PDS}/com.atproto.server.createSession",
        json={"identifier": HANDLE, "password": PWD},
        timeout=30,
    )
    r.raise_for_status()
    H["Authorization"] = "Bearer " + r.json()["accessJwt"]


def get(params):
    """4 tentativas, espera de 5 s em caso de erro; refaz login se o token expirar."""
    last = None
    for _ in range(4):
        try:
            r = requests.get(
                f"{PDS}/app.bsky.feed.searchPosts", params=params, headers=H, timeout=30
            )
            if r.status_code == 200:
                return r.json()
            last = f"{r.status_code} {r.text[:120]}"
            if r.status_code in (400, 401) and "xpired" in r.text:
                login()
                continue
        except Exception as exc:
            last = repr(exc)
        time.sleep(5)
    raise RuntimeError(last)


def record(p, now):
    rec = p.get("record", {}) or {}
    text = (rec.get("text") or "").strip()
    if not text:
        return None
    uri = p["uri"]
    return {
        "id": "bluesky_" + hashlib.sha1(uri.encode()).hexdigest()[:16],
        "platform": "bluesky",
        "kind": "post",
        "source": "bluesky",
        "author_hash": anonymize_author(p.get("author", {}).get("did")),
        "text": text,
        "created_utc": rec.get("createdAt"),
        "score": int(p.get("likeCount", 0) or 0),
        "url": f"https://bsky.app/profile/{p.get('author', {}).get('handle', '')}/post/{uri.split('/')[-1]}",
        "lang_hint": "pt",
        "parent_id": None,
        "context_text": None,
        "collected_at": now,
    }


def main():
    login()
    now = datetime.now(UTC).isoformat()
    ck = Path("dados/bluesky/ck")
    ck.mkdir(parents=True, exist_ok=True)
    stats = {}
    for i, kw in enumerate(KEYWORDS_PT):
        f = ck / f"kw_{i:02d}.parquet"
        if f.exists():  # retomável: expressão já concluída
            stats[kw] = {"posts": len(pd.read_parquet(f)), "retomado": True}
            print(f"'{kw}': já coletada ({stats[kw]['posts']})", flush=True)
            continue
        rows, cursor, n, pages, ok = [], None, 0, 0, True
        while True:
            params = {"q": kw, "lang": "pt", "since": SINCE, "limit": 100}
            if cursor:
                params["cursor"] = cursor
            try:
                d = get(params)
            except Exception as exc:
                print(f"[erro] '{kw}' pág {pages}: {exc}", flush=True)
                stats[kw + " (erro)"] = str(exc)
                ok = False
                break
            posts = d.get("posts", [])
            rows += [r for r in (record(p, now) for p in posts) if r]
            n += len(posts)
            pages += 1
            if pages % 20 == 0:
                print(
                    f"   '{kw}': {n} posts, {pages} págs, até {rows[-1]['created_utc'] if rows else ''}",
                    flush=True,
                )
            cursor = d.get("cursor")
            if not posts or not cursor:
                break
            time.sleep(1)
        if ok:  # só grava o checkpoint se a expressão terminou sem erro
            pd.DataFrame(rows).to_parquet(f, index=False)
        stats[kw] = {"posts": n, "paginas": pages, "completa": ok}
        print(f"'{kw}': {n} posts em {pages} páginas (completa={ok})", flush=True)
    df = pd.concat(
        [pd.read_parquet(p) for p in sorted(ck.glob("kw_*.parquet"))]
    ).drop_duplicates("id")
    df.to_parquet("dados/bluesky/bluesky_bruto.parquet", index=False)
    ins = upsert_posts(df)
    print(f"bruto único: {len(df)} | novos no SQLite: {ins}", flush=True)
    json.dump(
        {
            "coletado_em": now,
            "since": SINCE,
            "stats": stats,
            "bruto_unico": len(df),
            "novos_sqlite": ins,
            "data_min": str(df.created_utc.min()),
            "data_max": str(df.created_utc.max()),
        },
        open("dados/bluesky/coleta_resumo.json", "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=1,
    )


if __name__ == "__main__":
    main()
