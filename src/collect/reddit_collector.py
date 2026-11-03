"""
Coletor de dados do Reddit (API oficial via PRAW).

Coleta submissões e comentários em subreddits relevantes (trabalho, carreira,
tecnologia) que contenham as palavras-chave definidas no projeto.

Pré-requisitos:
  1. Criar um app "script" em https://www.reddit.com/prefs/apps
  2. Preencher REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET e REDDIT_USER_AGENT no .env
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv

from config import KEYWORDS, SUBREDDITS
from src.storage.database import anonymize_author

load_dotenv()


def _get_client():
    """Cria o cliente PRAW a partir das credenciais do .env."""
    try:
        import praw
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Pacote 'praw' não instalado. Rode: pip install praw"
        ) from exc

    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "tcc-rto-perceptions/0.1")

    if not client_id or not client_secret:
        raise RuntimeError(
            "Credenciais do Reddit ausentes. Preencha REDDIT_CLIENT_ID e "
            "REDDIT_CLIENT_SECRET no arquivo .env (veja .env.example)."
        )

    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        check_for_async=False,
    )


def _to_record(obj, kind: str, subreddit: str, text: str,
               parent_id: str | None = None,
               context_text: str | None = None) -> dict:
    created = datetime.fromtimestamp(
        getattr(obj, "created_utc", 0), tz=timezone.utc
    )
    author = getattr(obj, "author", None)
    author = str(author) if author else None
    return {
        "id": f"reddit_{obj.id}",
        "platform": "reddit",
        "kind": kind,
        "source": subreddit,
        "author_hash": anonymize_author(author),
        "text": text,
        "created_utc": created.isoformat(),
        "score": int(getattr(obj, "score", 0) or 0),
        "url": f"https://reddit.com{getattr(obj, 'permalink', '')}",
        "lang_hint": None,  # idioma é detectado depois, no pré-processamento
        "parent_id": parent_id,        # post ao qual o comentário responde
        "context_text": context_text,  # título do post original (enquadramento)
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }


def collect_reddit(
    subreddits: list[str] | None = None,
    keywords: list[str] | None = None,
    limit_per_query: int = 50,
    comments_per_post: int = 10,
) -> pd.DataFrame:
    """
    Coleta posts e comentários do Reddit.

    Para cada (subreddit, palavra-chave) faz uma busca e coleta as submissões
    e os comentários de topo de cada uma.
    """
    subreddits = subreddits or SUBREDDITS
    keywords = keywords or KEYWORDS
    reddit = _get_client()

    records: list[dict] = []
    for sub_name in subreddits:
        subreddit = reddit.subreddit(sub_name)
        for kw in keywords:
            try:
                results = subreddit.search(
                    kw, sort="relevance", time_filter="year", limit=limit_per_query
                )
            except Exception as exc:  # rede, subreddit inexistente, rate limit...
                print(f"  [aviso] busca falhou em r/{sub_name} '{kw}': {exc}")
                continue

            for submission in results:
                title = submission.title or ""
                body = title + "\n" + (submission.selftext or "")
                sub_pid = f"reddit_{submission.id}"
                records.append(_to_record(
                    submission, "post", sub_name, body.strip(),
                    parent_id=None, context_text=title))

                # Comentários de topo (sem expandir 'MoreComments')
                try:
                    submission.comments.replace_more(limit=0)
                    for comment in submission.comments[:comments_per_post]:
                        text = (getattr(comment, "body", "") or "").strip()
                        if text and text not in ("[deleted]", "[removed]"):
                            records.append(_to_record(
                                comment, "comment", sub_name, text,
                                parent_id=sub_pid, context_text=title))
                except Exception as exc:
                    print(f"  [aviso] comentários falharam em {submission.id}: {exc}")

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.drop_duplicates(subset="id")
    print(f"Reddit: {len(df)} registros coletados.")
    return df


if __name__ == "__main__":
    out = collect_reddit(limit_per_query=10, comments_per_post=5)
    print(out.head().to_string())
