"""
Coletor de base aberta: pullpush.io (sucessor comunitário do Pushshift).

Permite puxar **histórico** de submissões e comentários do Reddit (2019+) sem
autenticação — útil porque a API oficial do Reddit só retorna conteúdo recente.

É uma API comunitária e gratuita: pode ser lenta, instável ou aplicar limites
de taxa. O coletor trata erros de forma tolerante (ignora a consulta que falhar).

Doc: https://pullpush.io/  (endpoints /reddit/search/comment e /submission)
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import pandas as pd
import requests

from config import KEYWORDS, SUBREDDITS
from src.storage.database import anonymize_author

BASE = "https://api.pullpush.io/reddit/search"
HEADERS = {"User-Agent": "tcc-rto-perceptions/0.1 (research)"}


def _epoch(ts) -> int:
    """created_utc vem ora como int, ora como string ('123' ou '123.0')."""
    try:
        return int(float(ts))
    except (TypeError, ValueError):
        return 0


def _get(endpoint: str, params: dict, retries: int = 4) -> list[dict]:
    """GET com retry e backoff — pullpush.io aplica rate-limit (429) com frequência."""
    for attempt in range(retries):
        try:
            r = requests.get(f"{BASE}/{endpoint}/", params=params,
                             headers=HEADERS, timeout=45)
            if r.status_code == 200:
                return r.json().get("data", [])
            if r.status_code == 429:  # rate limit: espera progressiva
                wait = 5 * (attempt + 1)
                print(f"  [rate-limit] aguardando {wait}s...")
                time.sleep(wait)
                continue
            print(f"  [aviso] pullpush {endpoint} status {r.status_code}")
            return []
        except Exception as exc:
            print(f"  [aviso] pullpush {endpoint} falhou ({exc}); tentando de novo")
            time.sleep(3)
    return []


def _submission_record(it: dict, kw: str) -> dict:
    created = datetime.fromtimestamp(_epoch(it.get("created_utc")), tz=timezone.utc)
    title = it.get("title") or ""
    body = (title + "\n" + (it.get("selftext") or "")).strip()
    return {
        "id": f"reddit_{it.get('id')}",
        "platform": "reddit", "kind": "post",
        "source": it.get("subreddit") or kw,
        "author_hash": anonymize_author(it.get("author")),
        "text": body,
        "created_utc": created.isoformat(),
        "score": int(it.get("score", 0) or 0),
        "url": it.get("full_link") or f"https://reddit.com{it.get('permalink','')}",
        "lang_hint": None,
        "parent_id": None,
        "context_text": title,
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }


def _comment_record(it: dict, kw: str) -> dict:
    created = datetime.fromtimestamp(_epoch(it.get("created_utc")), tz=timezone.utc)
    return {
        "id": f"reddit_{it.get('id')}",
        "platform": "reddit", "kind": "comment",
        "source": it.get("subreddit") or kw,
        "author_hash": anonymize_author(it.get("author")),
        "text": (it.get("body") or "").strip(),
        "created_utc": created.isoformat(),
        "score": int(it.get("score", 0) or 0),
        "url": f"https://reddit.com{it.get('permalink','')}",
        "lang_hint": None,
        "parent_id": f"reddit_{it.get('link_id','').replace('t3_','')}"
        if it.get("link_id") else None,
        "context_text": None,  # título do post não vem no comentário
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }


def _paged(endpoint: str, base_params: dict, pages: int, pause: float) -> list[dict]:
    """Pagina no tempo (via `before`) para puxar mais de 100 itens por consulta."""
    items: list[dict] = []
    before = None
    for _ in range(pages):
        params = dict(base_params)
        if before:
            params["before"] = before
        batch = _get(endpoint, params)
        if not batch:
            break
        items += batch
        # próxima página: anterior ao item mais antigo desta leva
        epochs = [_epoch(it.get("created_utc")) for it in batch]
        epochs = [e for e in epochs if e > 0]
        if not epochs:
            break
        before = min(epochs)
        time.sleep(pause)
    return items


def collect_pullpush(
    keywords: list[str] | None = None,
    subreddits: list[str] | None = None,
    size: int = 100,
    pages: int = 1,
    include_comments: bool = True,
    pause: float = 1.2,
) -> pd.DataFrame:
    """
    Coleta submissões (e comentários) históricos do Reddit via pullpush.io.

    Faz `pages` consultas por (subreddit, palavra-chave), paginando no tempo —
    então o total por consulta é até `pages*size`. Como o conteúdo em PT é
    concentrado em subreddits brasileiros, o padrão usa `SUBREDDITS` do config.
    """
    keywords = keywords or KEYWORDS
    subreddits = subreddits or SUBREDDITS
    records: list[dict] = []

    for sub in subreddits:
        for kw in keywords:
            subs = _paged("submission", {"q": kw, "subreddit": sub, "size": size},
                          pages, pause)
            records += [_submission_record(it, kw) for it in subs]
            n_c = 0
            if include_comments:
                cmts = _paged("comment", {"q": kw, "subreddit": sub, "size": size},
                              pages, pause)
                records += [_comment_record(it, kw) for it in cmts]
                n_c = len(cmts)
            print(f"  r/{sub} '{kw}': +{len(subs)} posts, +{n_c} coment.")

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.drop_duplicates(subset="id")
        df = df[df["text"].str.strip().str.len() > 0]
    print(f"pullpush (Reddit histórico): {len(df)} registros.")
    return df


if __name__ == "__main__":
    out = collect_pullpush(keywords=["home office", "trabalho remoto"],
                           subreddits=["brasil"], size=50)
    if not out.empty:
        print(out[["created_utc", "kind", "score", "text"]].head().to_string())
