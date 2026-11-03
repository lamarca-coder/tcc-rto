"""
Coletor de dados do X/Twitter (API oficial v2 via Tweepy).

ATENÇÃO — limitações da plataforma:
  O acesso de busca da API do X foi fortemente restringido e, na prática, exige
  um plano pago. O nível gratuito atual praticamente não permite busca de tweets.
  Este coletor está pronto para uso caso você possua um Bearer Token com acesso
  de leitura/busca; caso contrário, recomenda-se priorizar o Reddit e/ou a
  coleta complementar manual (como previsto para o LinkedIn no projeto).

Pré-requisitos:
  Preencher TWITTER_BEARER_TOKEN no .env (veja .env.example).
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv

from config import KEYWORDS_PT, KEYWORDS_EN
from src.storage.database import anonymize_author

load_dotenv()


def _get_client():
    try:
        import tweepy
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Pacote 'tweepy' não instalado. Rode: pip install tweepy") from exc

    bearer = os.getenv("TWITTER_BEARER_TOKEN")
    if not bearer:
        raise RuntimeError(
            "TWITTER_BEARER_TOKEN ausente no .env. Sem ele não é possível usar a API do X."
        )
    return tweepy.Client(bearer_token=bearer, wait_on_rate_limit=True)


def _build_query(keywords: list[str], lang: str) -> str:
    """Monta uma query do tipo: ("a" OR "b") lang:pt -is:retweet."""
    terms = " OR ".join(f'"{k}"' for k in keywords)
    return f"({terms}) lang:{lang} -is:retweet"


def collect_twitter(max_results_per_lang: int = 100) -> pd.DataFrame:
    """
    Coleta tweets recentes (últimos ~7 dias na busca padrão) em PT e EN.

    `max_results_per_lang` é limitado a 100 por requisição na API v2 (recent search).
    """
    client = _get_client()
    records: list[dict] = []

    queries = {
        "pt": _build_query(KEYWORDS_PT, "pt"),
        "en": _build_query(KEYWORDS_EN, "en"),
    }

    for lang, query in queries.items():
        try:
            resp = client.search_recent_tweets(
                query=query,
                max_results=min(max_results_per_lang, 100),
                tweet_fields=["created_at", "lang", "public_metrics", "author_id"],
            )
        except Exception as exc:
            print(f"  [aviso] busca no X falhou ({lang}): {exc}")
            continue

        for tw in (resp.data or []):
            metrics = getattr(tw, "public_metrics", {}) or {}
            created = getattr(tw, "created_at", None) or datetime.now(timezone.utc)
            records.append(
                {
                    "id": f"twitter_{tw.id}",
                    "platform": "twitter",
                    "kind": "post",
                    "source": query,
                    "author_hash": anonymize_author(getattr(tw, "author_id", None)),
                    "text": (tw.text or "").strip(),
                    "created_utc": created.isoformat()
                    if hasattr(created, "isoformat") else str(created),
                    "score": int(metrics.get("like_count", 0)),
                    "url": f"https://x.com/i/web/status/{tw.id}",
                    "lang_hint": getattr(tw, "lang", lang),
                    "collected_at": datetime.now(timezone.utc).isoformat(),
                }
            )

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.drop_duplicates(subset="id")
    print(f"X/Twitter: {len(df)} registros coletados.")
    return df


if __name__ == "__main__":
    out = collect_twitter(max_results_per_lang=10)
    print(out.head().to_string())
