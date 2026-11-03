"""
Coletor de notícias via GDELT (GDELT DOC 2.0 API).

GDELT monitora a mídia mundial em tempo real. Aqui usamos para capturar
**manchetes/notícias** sobre o retorno ao trabalho presencial em português,
abrindo o ângulo de **enquadramento da mídia** — que pode ser contrastado com
a percepção do público (redes sociais).

Gratuito e sem autenticação. Doc: https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/
Observação: a unidade aqui é o *título da notícia* (não o texto completo).
"""
from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone

import pandas as pd
import requests

from config import KEYWORDS_PT

API = "https://api.gdeltproject.org/api/v2/doc/doc"
HEADERS = {"User-Agent": "tcc-rto-perceptions/0.1 (research)"}


def _build_query(keywords: list[str]) -> str:
    phrases = " OR ".join(f'"{k}"' for k in keywords)
    # foco em fontes em português e do Brasil
    return f"({phrases}) sourcelang:portuguese sourcecountry:BR"


def _fetch(params: dict, retries: int = 5) -> list[dict]:
    """GET com retry/backoff — GDELT exige no máx. 1 requisição a cada 5 s."""
    for attempt in range(retries):
        try:
            r = requests.get(API, params=params, headers=HEADERS, timeout=45)
            txt = r.text.strip()
            if r.status_code == 200 and txt.startswith("{"):
                return r.json().get("articles", [])
            if r.status_code == 429 or "limit requests" in txt.lower():
                wait = 7 * (attempt + 1)
                print(f"  [rate-limit GDELT] aguardando {wait}s...")
                time.sleep(wait)
                continue
            # 200 sem JSON costuma ser 'sem resultados' para a janela
            return []
        except Exception as exc:
            print(f"  [aviso] GDELT falhou ({exc}); retry")
            time.sleep(5)
    return []


def _parse_date(s: str) -> str:
    for fmt in ("%Y%m%dT%H%M%SZ", "%Y%m%d%H%M%S"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc).isoformat()
        except (ValueError, TypeError):
            continue
    return datetime.now(timezone.utc).isoformat()


def collect_gdelt(keywords: list[str] | None = None,
                  start_year: int = 2019, end_year: int = 2026,
                  max_records: int = 250, pause: float = 5.5) -> pd.DataFrame:
    """
    Coleta notícias por ano (para cobrir o histórico), em PT/Brasil.
    """
    keywords = keywords or KEYWORDS_PT
    query = _build_query(keywords)
    now = datetime.now(timezone.utc).isoformat()
    records: list[dict] = []

    for year in range(start_year, end_year + 1):
        params = {
            "query": query,
            "mode": "ArtList",
            "format": "json",
            "maxrecords": max_records,
            "sort": "DateAsc",
            "startdatetime": f"{year}0101000000",
            "enddatetime": f"{year}1231235959",
        }
        arts = _fetch(params)
        for a in arts:
            url = a.get("url", "")
            title = (a.get("title") or "").strip()
            if not title or not url:
                continue
            records.append({
                "id": "news_" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:16],
                "platform": "news",
                "kind": "post",
                "source": a.get("domain", "gdelt"),
                "author_hash": "veiculo",
                "text": title,
                "created_utc": _parse_date(a.get("seendate", "")),
                "score": 0,
                "url": url,
                "lang_hint": "pt",
                "parent_id": None,
                "context_text": None,
                "collected_at": now,
            })
        print(f"  GDELT {year}: +{len(arts)} notícias")
        time.sleep(pause)

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.drop_duplicates(subset="id")
    print(f"GDELT (notícias PT/BR): {len(df)} registros.")
    return df


if __name__ == "__main__":
    out = collect_gdelt(start_year=2023, end_year=2024, max_records=50)
    if not out.empty:
        print(out[["created_utc", "source", "text"]].head(10).to_string())
