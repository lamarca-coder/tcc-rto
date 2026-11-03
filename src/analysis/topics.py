"""
Modelagem de tópicos com LDA (Latent Dirichlet Allocation; Blei et al., 2003).

Identifica temas recorrentes nas discussões agrupando documentos por padrões de
coocorrência de palavras. Usa scikit-learn (CountVectorizer + LDA), que tem
instalação simples no Windows e bom desempenho para análises preliminares.

Trabalha sobre a coluna `tokens_str` (texto já limpo, sem stopwords e
opcionalmente radicalizado/lematizado).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from config import N_TOPICS, RANDOM_STATE


@dataclass
class TopicModelResult:
    model: LatentDirichletAllocation
    vectorizer: CountVectorizer
    doc_topic: np.ndarray          # matriz documento x tópico (proporções)
    dominant_topic: np.ndarray     # tópico dominante por documento
    topic_words: dict[int, list[str]]  # palavras mais relevantes por tópico

    def topics_as_dataframe(self, n_words: int = 10) -> pd.DataFrame:
        rows = []
        for tid, words in self.topic_words.items():
            rows.append({"topico": tid, "palavras": ", ".join(words[:n_words])})
        return pd.DataFrame(rows)


def get_topic_words(model, vectorizer, n_words: int = 12) -> dict[int, list[str]]:
    feature_names = np.array(vectorizer.get_feature_names_out())
    topic_words = {}
    for tid, comp in enumerate(model.components_):
        top_idx = comp.argsort()[::-1][:n_words]
        topic_words[tid] = feature_names[top_idx].tolist()
    return topic_words


def fit_topics(
    docs: list[str],
    n_topics: int = N_TOPICS,
    min_df: int = 2,
    max_df: float = 0.9,
    max_features: int = 5000,
    n_words: int = 12,
) -> TopicModelResult:
    """
    Ajusta um modelo LDA aos documentos fornecidos.

    `docs` é a lista de textos pré-processados (tokens_str).
    """
    docs = [d for d in docs if isinstance(d, str)]
    vectorizer = CountVectorizer(
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        token_pattern=r"(?u)\b\w\w+\b",
    )
    dtm = vectorizer.fit_transform(docs)

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        learning_method="batch",
        max_iter=25,
        random_state=RANDOM_STATE,
    )
    doc_topic = lda.fit_transform(dtm)
    dominant = doc_topic.argmax(axis=1)
    topic_words = get_topic_words(lda, vectorizer, n_words=n_words)

    return TopicModelResult(
        model=lda,
        vectorizer=vectorizer,
        doc_topic=doc_topic,
        dominant_topic=dominant,
        topic_words=topic_words,
    )


def add_topics(df: pd.DataFrame, text_col: str = "tokens_str",
               n_topics: int = N_TOPICS) -> tuple[pd.DataFrame, TopicModelResult]:
    """
    Ajusta o LDA e adiciona a coluna `topic` (tópico dominante) ao DataFrame.
    Retorna (df_atualizado, resultado_do_modelo).
    """
    df = df.copy().reset_index(drop=True)
    result = fit_topics(df[text_col].fillna("").tolist(), n_topics=n_topics)
    df["topic"] = result.dominant_topic
    return df, result
