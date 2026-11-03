"""
Geração das figuras dos resultados preliminares.

Salva os gráficos em outputs/figuras_v0/ (PNG, 150 dpi) prontos para inclusão no
TCC. Usa backend não interativo (Agg), funcionando em execução headless.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

from config import FIGURES_DIR, TIME_FREQ
from src.analysis.temporal import (
    sentiment_over_time,
    sentiment_summary,
    volume_over_time,
)

sns.set_theme(style="whitegrid")

# Paleta fixa por rótulo de sentimento
SENT_COLORS = {"positivo": "#2e8b57", "neutro": "#9aa0a6", "negativo": "#c0392b"}
SENT_ORDER = ["positivo", "neutro", "negativo"]


def _save(fig, name: str) -> str:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


# --------------------------------------------------------------------------- #
def plot_sentiment_distribution(df: pd.DataFrame) -> str:
    summ = sentiment_summary(df).reindex(SENT_ORDER).dropna()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(
        summ.index,
        summ["contagem"],
        color=[SENT_COLORS.get(i, "#888") for i in summ.index],
    )
    for bar, perc in zip(bars, summ["percentual"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{perc:.1f}%", ha="center", va="bottom", fontsize=10)
    ax.set_title("Distribuição de sentimento sobre o retorno ao presencial")
    ax.set_xlabel("Polaridade")
    ax.set_ylabel("Nº de publicações")
    return _save(fig, "01_distribuicao_sentimento.png")


def plot_sentiment_by_group(df: pd.DataFrame, group_col: str,
                            titulo: str, fname: str) -> str:
    tab = (df.groupby([group_col, "sentiment_label"]).size()
           .unstack(fill_value=0))
    for c in SENT_ORDER:
        if c not in tab.columns:
            tab[c] = 0
    tab = tab[SENT_ORDER]
    prop = tab.div(tab.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    prop.plot(kind="bar", stacked=True, ax=ax,
              color=[SENT_COLORS[c] for c in SENT_ORDER])
    ax.set_title(titulo)
    ax.set_xlabel(group_col)
    ax.set_ylabel("Proporção")
    ax.legend(title="Sentimento", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=0)
    return _save(fig, fname)


# --------------------------------------------------------------------------- #
def plot_topic_words(topic_words: dict[int, list[str]],
                     model, n_words: int = 10) -> str:
    n_topics = len(topic_words)
    ncols = 2
    nrows = (n_topics + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.2 * nrows))
    axes = axes.flatten()

    for tid, words in topic_words.items():
        comp = model.components_[tid]
        # pesos das palavras mais relevantes (na mesma ordem de `words`)
        top_idx = comp.argsort()[::-1][:n_words]
        weights = comp[top_idx]
        labels = words[:n_words][::-1]
        vals = weights[::-1]
        ax = axes[tid]
        ax.barh(labels, vals, color=sns.color_palette("viridis", n_topics)[tid])
        ax.set_title(f"Tópico {tid}")
        ax.tick_params(labelsize=9)

    for j in range(n_topics, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Palavras mais relevantes por tópico (LDA)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return _save(fig, "04_topicos_palavras.png")


def plot_wordcloud(df: pd.DataFrame, text_col: str = "tokens_str") -> str:
    text = " ".join(df[text_col].dropna().tolist())
    if not text.strip():
        text = "sem dados"
    wc = WordCloud(width=1000, height=500, background_color="white",
                   colormap="viridis", collocations=False).generate(text)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Nuvem de palavras — termos mais frequentes")
    return _save(fig, "03_nuvem_palavras.png")


# --------------------------------------------------------------------------- #
def plot_volume_over_time(df: pd.DataFrame, freq: str = TIME_FREQ) -> str:
    vol = volume_over_time(df, freq=freq)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(vol["created_utc"], vol["volume"], marker="o", color="#34495e")
    ax.fill_between(vol["created_utc"], vol["volume"], alpha=0.15, color="#34495e")
    ax.set_title("Volume de publicações ao longo do tempo")
    ax.set_xlabel("Período")
    ax.set_ylabel("Nº de publicações")
    fig.autofmt_xdate()
    return _save(fig, "05_volume_temporal.png")


def plot_sentiment_over_time(df: pd.DataFrame, freq: str = TIME_FREQ) -> str:
    sot = sentiment_over_time(df, freq=freq)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    # (a) escore médio
    ax1.plot(sot["created_utc"], sot["score_medio"], marker="o", color="#2c3e50")
    ax1.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax1.set_title("Escore médio de sentimento ao longo do tempo")
    ax1.set_ylabel("Escore médio")

    # (b) proporções por classe
    for label in SENT_ORDER:
        col = f"prop_{label}"
        if col in sot.columns:
            ax2.plot(sot["created_utc"], sot[col], marker=".",
                     label=label, color=SENT_COLORS[label])
    ax2.set_title("Proporção de cada sentimento ao longo do tempo")
    ax2.set_ylabel("Proporção")
    ax2.set_xlabel("Período")
    ax2.legend(title="Sentimento")
    fig.autofmt_xdate()
    fig.tight_layout()
    return _save(fig, "06_sentimento_temporal.png")


# --------------------------------------------------------------------------- #
# Posicionamento (stance) e enquadramento (framing)                            #
# --------------------------------------------------------------------------- #
STANCE_COLORS = {
    "anti-RTO": "#2e86c1",    # azul (pró-remoto / contra o retorno)
    "pró-RTO": "#cb4335",     # vermelho (a favor do retorno)
    "neutro": "#9aa0a6",
    "indefinido": "#d5d8dc",
}
STANCE_ORDER = ["anti-RTO", "neutro", "pró-RTO", "indefinido"]


def plot_stance_distribution(df: pd.DataFrame) -> str:
    counts = df["stance"].value_counts().reindex(STANCE_ORDER).dropna()
    perc = counts / counts.sum() * 100
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(counts.index, counts.values,
                  color=[STANCE_COLORS.get(i, "#888") for i in counts.index])
    for bar, p in zip(bars, perc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{p:.1f}%", ha="center", va="bottom", fontsize=10)
    ax.set_title("Posicionamento sobre o retorno ao trabalho presencial")
    ax.set_ylabel("Nº de publicações")
    return _save(fig, "07_distribuicao_stance.png")


def plot_stance_over_time(df: pd.DataFrame, freq: str = TIME_FREQ) -> str:
    from src.analysis.temporal import _norm_freq, _ensure_datetime
    d = _ensure_datetime(df, "created_utc").set_index("created_utc")
    props = (d.groupby([pd.Grouper(freq=_norm_freq(freq)), "stance"]).size()
             .unstack(fill_value=0))
    props = props.div(props.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for st in STANCE_ORDER:
        if st in props.columns:
            ax.plot(props.index, props[st], marker=".", label=st,
                    color=STANCE_COLORS[st])
    ax.set_title("Evolução do posicionamento sobre o RTO ao longo do tempo")
    ax.set_ylabel("Proporção")
    ax.set_xlabel("Período")
    ax.legend(title="Posicionamento", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.autofmt_xdate()
    return _save(fig, "08_stance_temporal.png")


def plot_framing(frames: pd.DataFrame, top_n: int = 12) -> str:
    """Para os vídeos/posts com mais comentários: stance dos comentários × título."""
    top = frames.head(top_n).iloc[::-1]
    labels = [str(t)[:42] + ("…" if len(str(t)) > 42 else "")
              for t in top["frame_text"].fillna("(sem título)")]
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(top) + 1.5))
    left = [0] * len(top)
    # barra empilhada da proporção de stance por frame (recalculada)
    # usa apenas as colunas de stance agregadas no DataFrame de frames
    for st in ["anti-RTO", "neutro", "pró-RTO", "indefinido"]:
        col = f"prop_{st}"
        if col not in top.columns:
            continue
        ax.barh(labels, top[col], left=left, color=STANCE_COLORS[st], label=st)
        left = [a + b for a, b in zip(left, top[col])]
    ax.set_title("Enquadramento × reação: posicionamento dos comentários por conteúdo")
    ax.set_xlabel("Proporção de comentários")
    ax.legend(title="Stance dos comentários", bbox_to_anchor=(1.02, 1), loc="upper left")
    return _save(fig, "09_framing_titulo_vs_comentarios.png")


# --------------------------------------------------------------------------- #
def generate_all(df: pd.DataFrame, topic_result, freq: str = TIME_FREQ,
                 frames: pd.DataFrame | None = None) -> list[str]:
    """Gera todas as figuras e retorna a lista de caminhos."""
    paths = [
        plot_sentiment_distribution(df),
        plot_sentiment_by_group(df, "platform",
                                "Sentimento por plataforma",
                                "02a_sentimento_por_plataforma.png"),
    ]
    # Figura por idioma só faz sentido quando há mais de um idioma no corpus
    if "lang" in df.columns and df["lang"].nunique() > 1:
        paths.append(plot_sentiment_by_group(
            df, "lang", "Sentimento por idioma",
            "02b_sentimento_por_idioma.png"))
    paths += [
        plot_wordcloud(df),
        plot_topic_words(topic_result.topic_words, topic_result.model),
        plot_volume_over_time(df, freq=freq),
        plot_sentiment_over_time(df, freq=freq),
    ]
    # Posicionamento (stance)
    if "stance" in df.columns:
        paths.append(plot_stance_distribution(df))
        paths.append(plot_stance_over_time(df, freq=freq))
    # Enquadramento (framing)
    if frames is not None and not frames.empty:
        paths.append(plot_framing(frames))
    return paths
