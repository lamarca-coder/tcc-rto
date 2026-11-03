"""
Análise de sentimento.

Classifica cada texto em positivo / negativo / neutro, permitindo identificar
a polaridade das percepções sobre o retorno ao trabalho presencial (Liu, 2012).

Backends disponíveis:
  - "lexicon" (padrão): VADER para inglês + léxico próprio para português.
    Leve, sem downloads pesados, roda em qualquer máquina.
  - "transformer" (opcional): modelos BERT multilíngues via pysentimiento,
    mais precisos. Requer `pip install -r requirements-optional.txt`.

A análise é feita sobre o texto original (não sobre os tokens lematizados),
pois os métodos de sentimento dependem da estrutura e da negação do texto.
"""
from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from config import SENTIMENT_THRESHOLD
from src.analysis.sentiment_lexicon_pt import score_pt


def _label_from_score(score: float, threshold: float = SENTIMENT_THRESHOLD) -> str:
    if score >= threshold:
        return "positivo"
    if score <= -threshold:
        return "negativo"
    return "neutro"


class SentimentAnalyzer:
    """Fachada para os diferentes backends de análise de sentimento."""

    def __init__(self, method: str = "lexicon",
                 threshold: float = SENTIMENT_THRESHOLD):
        self.method = method
        self.threshold = threshold
        self._vader = None
        self._tf = {}  # cache de pipelines transformer por idioma

        if method == "lexicon":
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._vader = SentimentIntensityAnalyzer()
        elif method == "transformer":
            self._init_transformer()
        else:
            raise ValueError(f"Método desconhecido: {method!r} (use 'lexicon' ou 'transformer').")

    # ----------------------------------------------------------------- #
    def _init_transformer(self):
        try:
            from pysentimiento import create_analyzer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Backend 'transformer' requer pysentimiento. "
                "Rode: pip install -r requirements-optional.txt"
            ) from exc
        # carrega apenas os modelos dos idiomas usados (economia de memória)
        from config import LANGUAGES
        for lang in LANGUAGES:
            if lang in ("pt", "en"):
                self._tf[lang] = create_analyzer(task="sentiment", lang=lang)
        # fallback mínimo para idioma não previsto
        if "en" not in self._tf and "pt" not in self._tf:
            self._tf["pt"] = create_analyzer(task="sentiment", lang="pt")

    # ----------------------------------------------------------------- #
    def _analyzer_for(self, lang: str):
        """Retorna o analisador do idioma, com fallback para qualquer carregado."""
        return self._tf.get(lang) or next(iter(self._tf.values()))

    # ----------------------------------------------------------------- #
    def score_one(self, text: str, lang: str) -> float:
        """Escore composto em [-1, 1] para um texto."""
        if not text:
            return 0.0
        if self.method == "lexicon":
            if lang == "pt":
                return score_pt(text)
            return self._vader.polarity_scores(text)["compound"]
        # transformer
        analyzer = self._analyzer_for(lang)
        out = analyzer.predict(text)
        probas = out.probas  # {'POS':..., 'NEU':..., 'NEG':...}
        return float(probas.get("POS", 0) - probas.get("NEG", 0))

    # ----------------------------------------------------------------- #
    def _scores_transformer_batch(self, texts: list[str], lang: str,
                                  batch_size: int = 64) -> list[float]:
        """Predição em lote (menor batch = menor pico de memória na CPU)."""
        import gc
        analyzer = self._analyzer_for(lang)
        out_scores: list[float] = []
        for i in tqdm(range(0, len(texts), batch_size),
                      desc=f"Sentimento (transformer/{lang})"):
            chunk = texts[i:i + batch_size]
            results = analyzer.predict(chunk)  # pysentimiento aceita lista
            for r in results:
                p = r.probas
                out_scores.append(float(p.get("POS", 0) - p.get("NEG", 0)))
            del results
            if i % (batch_size * 10) == 0:
                gc.collect()
        return out_scores

    # ----------------------------------------------------------------- #
    def analyze_dataframe(self, df: pd.DataFrame,
                          text_col: str = "text",
                          lang_col: str = "lang") -> pd.DataFrame:
        """
        Adiciona as colunas `sentiment_score` (float) e `sentiment_label`
        (positivo/negativo/neutro) ao DataFrame.
        """
        df = df.copy().reset_index(drop=True)
        langs = (df[lang_col] if lang_col in df.columns
                 else pd.Series(["en"] * len(df)))
        scores = [0.0] * len(df)

        if self.method == "transformer":
            # processa em lote, agrupando por idioma (preservando a ordem)
            for lang in ("pt", "en"):
                idx = [i for i, lg in enumerate(langs) if lg == lang]
                if not idx:
                    continue
                texts = [str(df[text_col].iloc[i]) for i in idx]
                lang_scores = self._scores_transformer_batch(texts, lang)
                for i, s in zip(idx, lang_scores):
                    scores[i] = s
        else:
            for i, (text, lang) in enumerate(tqdm(
                    zip(df[text_col], langs), total=len(df),
                    desc=f"Sentimento ({self.method})")):
                lang = lang if lang in ("pt", "en") else "en"
                scores[i] = self.score_one(str(text), lang)

        df["sentiment_score"] = scores
        df["sentiment_label"] = [
            _label_from_score(s, self.threshold) for s in scores
        ]
        return df


def add_sentiment(df: pd.DataFrame, method: str = "lexicon") -> pd.DataFrame:
    """Função de conveniência usada pelo pipeline."""
    return SentimentAnalyzer(method=method).analyze_dataframe(df)
