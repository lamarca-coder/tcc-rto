"""
Coletor do Bluesky (AT Protocol / app.bsky.feed.searchPosts).

Rede social aberta e gratuita. Bom para conteúdo em PT **recente** (a rede é
nova; sem histórico pré-2023). Complementa o presente do corpus.

Pré-requisito (.env):
  BLUESKY_HANDLE        seu handle (ex.: fulano.bsky.social)
  BLUESKY_APP_PASSWORD  uma "App Password" criada em
                        Bluesky -> Settings -> App Passwords (NÃO a senha real)
"""
from __future__ import annotations

import hashlib
import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests
from dotenv import load_dotenv

from config import KEYWORDS_PT
from src.storage.database import anonymize_author

load_dotenv()

PDS = "https://bsky.social"
SEARCH = "https://bsky.social/xrpc/app.bsky.feed.searchPosts"


def _session() -> str:
    handle = os.getenv("BLUESKY_HANDLE")
    pwd = os.getenv("BLUESKY_APP_PASSWORD")
    if not handle or not pwd:
        raise RuntimeError(
            "Credenciais do Bluesky ausentes. Preencha BLUESKY_HANDLE e "
            "BLUESKY_APP_PASSWORD no .env (Settings -> App Passwords)."
        )
    r = requests.post(f"{PDS}/xrpc/com.atproto.server.createSession",
                      json={"identifier": handle, "password": pwd}, timeout=30)
    r.raise_for_status()
    return r.json()["accessJwt"]


def _record(post: dict, now: str) -> dict | None:
    rec = post.get("record", {}) or {}
    text = (rec.get("text") or "").strip()
    if not text:
        return None
    uri = post.get("uri", "")
    rkey = uri.split("/")[-1]
    handle = post.get("author", {}).get("handle", "")
    return {
        "id": "bluesky_" + hashlib.sha1(uri.encode("utf-8")).hexdigest()[:16],
        "platform": "bluesky", "kind": "post",
        "source": "bluesky",
        "author_hash": anonymize_author(post.get("author", {}).get("did")),
        "text": text,
        "created_utc": rec.get("createdAt") or now,
        "score": int(post.get("likeCount", 0) or 0),
        "url": f"https://bsky.app/profile/{handle}/post/{rkey}",
        "lang_hint": "pt",
        "parent_id": None,
        "context_text": None,
        "collected_at": now,
    }


def collect_bluesky(keywords: list[str] | None = None,
                    per_keyword: int = 200, pause: float = 0.8) -> pd.DataFrame:
    keywords = keywords or KEYWORDS_PT
    token = _session()
    headers = {"Authorization": f"Bearer {token}"}
    now = datetime.now(timezone.utc).isoformat()
    records: list[dict] = []

    for kw in keywords:
        cursor, got = None, 0
        while got < per_keyword:
            params = {"q": kw, "limit": min(100, per_keyword - got), "lang": "pt"}
            if cursor:
                params["cursor"] = cursor
            try:
                r = requests.get(SEARCH, params=params, headers=headers, timeout=30)
                if r.status_code != 200:
                    print(f"  [aviso] Bluesky '{kw}': status {r.status_code}")
                    break
                data = r.json()
            except Exception as exc:
                print(f"  [aviso] Bluesky '{kw}' falhou: {exc}")
                break
            posts = data.get("posts", [])
            if not posts:
                break
            for p in posts:
                rec = _record(p, now)
                if rec:
                    records.append(rec)
            got += len(posts)
            cursor = data.get("cursor")
            if not cursor:
                break
            time.sleep(pause)
        print(f"  Bluesky '{kw}': ~{got} posts")

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.drop_duplicates(subset="id")
    print(f"Bluesky: {len(df)} registros.")
    return df


if __name__ == "__main__":
    out = collect_bluesky(keywords=["home office", "trabalho presencial"], per_keyword=50)
    if not out.empty:
        print(out[["created_utc", "score", "text"]].head().to_string())
