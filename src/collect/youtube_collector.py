"""
Coletor de comentários do YouTube (API oficial — YouTube Data API v3).

Por que YouTube? É a forma mais viável e gratuita de obter conteúdo em
**português** com **profundidade histórica**: os comentários de um vídeo
publicado em 2019 vão de 2019 até hoje, permitindo reconstruir a evolução das
percepções (2019-2026) — algo que as APIs do Reddit/X não entregam de graça.

Fluxo:
  1) search.list  -> encontra vídeos por palavra-chave (em PT, região BR);
  2) commentThreads.list -> coleta os comentários de cada vídeo (com data).

Cota: a API tem cota diária (padrão 10.000 unidades/dia). search.list custa
100 unidades por chamada; commentThreads.list custa ~1. Por isso fazemos poucas
buscas (uma por palavra-chave) e muitos comentários por vídeo.

Pré-requisito:
  YOUTUBE_API_KEY no .env (Google Cloud -> YouTube Data API v3 -> Chave de API).
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import pandas as pd
import requests
from dotenv import load_dotenv

from config import KEYWORDS
from src.storage.database import anonymize_author

load_dotenv()

SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
COMMENTS_URL = "https://www.googleapis.com/youtube/v3/commentThreads"


def _api_key() -> str:
    key = os.getenv("YOUTUBE_API_KEY")
    if not key:
        raise RuntimeError(
            "YOUTUBE_API_KEY ausente no .env. Crie uma chave em "
            "https://console.cloud.google.com (ative 'YouTube Data API v3')."
        )
    return key


def _search_videos(keyword: str, key: str, max_results: int = 8,
                   published_after: str | None = None,
                   published_before: str | None = None) -> list[tuple[str, str]]:
    """Retorna lista de (video_id, título) para uma palavra-chave."""
    params = {
        "key": key,
        "q": keyword,
        "part": "snippet",
        "type": "video",
        "maxResults": min(max_results, 50),
        "relevanceLanguage": "pt",
        "regionCode": "BR",
        "order": "relevance",
    }
    if published_after:
        params["publishedAfter"] = published_after   # ex.: 2019-01-01T00:00:00Z
    if published_before:
        params["publishedBefore"] = published_before

    resp = requests.get(SEARCH_URL, params=params, timeout=30)
    resp.raise_for_status()
    items = resp.json().get("items", [])
    out = []
    for it in items:
        vid = it.get("id", {}).get("videoId")
        title = it.get("snippet", {}).get("title", "")
        if vid:
            out.append((vid, title))
    return out


def _video_comments(video_id: str, title: str, key: str,
                    max_comments: int = 100) -> list[dict]:
    """Coleta comentários de topo de um vídeo (paginando)."""
    records: list[dict] = []
    page_token = None
    now = datetime.now(timezone.utc).isoformat()

    while len(records) < max_comments:
        params = {
            "key": key,
            "videoId": video_id,
            "part": "snippet",
            "maxResults": 100,
            "order": "relevance",
            "textFormat": "plainText",
        }
        if page_token:
            params["pageToken"] = page_token

        resp = requests.get(COMMENTS_URL, params=params, timeout=30)
        if resp.status_code != 200:
            # comentários desativados, vídeo removido etc. — ignora e segue
            break
        data = resp.json()

        for it in data.get("items", []):
            top = it["snippet"]["topLevelComment"]["snippet"]
            records.append({
                "id": f"youtube_{it['id']}",
                "platform": "youtube",
                "kind": "comment",
                "source": title[:120],
                "author_hash": anonymize_author(top.get("authorChannelId", {}).get("value")
                                                or top.get("authorDisplayName")),
                "text": (top.get("textOriginal") or "").strip(),
                "created_utc": top.get("publishedAt"),
                "score": int(top.get("likeCount", 0) or 0),
                "url": f"https://youtube.com/watch?v={video_id}",
                "lang_hint": "pt",
                "parent_id": f"yt_video_{video_id}",   # comentário responde ao vídeo
                "context_text": title,                 # título do vídeo (enquadramento)
                "collected_at": now,
            })
            if len(records) >= max_comments:
                break

        page_token = data.get("nextPageToken")
        if not page_token:
            break

    return records


# Janelas temporais para cobrir o histórico (vídeos antigos -> comentários antigos).
# Cada janela é (publishedAfter, publishedBefore) em ISO 8601 (ou None).
# Mais janelas => mais vídeos distintos por palavra-chave (cada busca custa cota).
HISTORICAL_WINDOWS = [
    ("2019-01-01T00:00:00Z", "2021-01-01T00:00:00Z"),  # pré/início da pandemia
    ("2021-01-01T00:00:00Z", "2023-06-01T00:00:00Z"),  # transição híbrida
    ("2023-06-01T00:00:00Z", None),                     # onda de RTO -> hoje
]


def collect_youtube(
    keywords: list[str] | None = None,
    max_videos_per_keyword: int = 50,   # 1-50 custam a MESMA cota de busca
    comments_per_video: int = 300,
    published_after: str | None = None,
    published_before: str | None = None,
    published_windows: list[tuple] | None = None,
) -> pd.DataFrame:
    """
    Coleta comentários do YouTube para as palavras-chave do projeto.

    Para cobrir a janela histórica (2019+), passe `published_windows` com uma
    lista de (após, antes) — o coletor faz uma busca por janela e por
    palavra-chave, capturando vídeos de diferentes épocas. Alternativamente,
    use `published_after`/`published_before` para uma única janela.
    """
    keywords = keywords or KEYWORDS
    key = _api_key()
    records: list[dict] = []

    # Define as janelas de busca
    if published_windows:
        windows = published_windows
    else:
        windows = [(published_after, published_before)]

    for kw in keywords:
        videos: list[tuple[str, str]] = []
        for after, before in windows:
            try:
                videos += _search_videos(kw, key, max_videos_per_keyword, after, before)
            except Exception as exc:
                print(f"  [aviso] busca no YouTube falhou para '{kw}' "
                      f"[{after}..{before}]: {exc}")
        # remove vídeos repetidos entre janelas
        videos = list({vid: title for vid, title in videos}.items())
        print(f"  '{kw}': {len(videos)} vídeos")
        for vid, title in videos:
            try:
                records.extend(_video_comments(vid, title, key, comments_per_video))
            except Exception as exc:
                print(f"  [aviso] comentários falharam ({vid}): {exc}")

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.drop_duplicates(subset="id")
        df = df[df["text"].str.len() > 0]
    print(f"YouTube: {len(df)} comentários coletados.")
    return df


if __name__ == "__main__":
    out = collect_youtube(max_videos_per_keyword=2, comments_per_video=20)
    if not out.empty:
        print(out[["created_utc", "score", "text"]].head(10).to_string())
