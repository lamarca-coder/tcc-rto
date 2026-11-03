"""utilitário (D2 a D7, B): Funções comuns: intervalo de Wilson, qui-quadrado e V de Cramér, carga do corpus, tabelas markdown e padrão de figuras.

Etapa: utilitário (D2 a D7, B)
Objetivo: Funções comuns: intervalo de Wilson, qui-quadrado e V de Cramér, carga do corpus, tabelas markdown e padrão de figuras.
Entradas: dados/v2/corpus_v2.parquet
Saídas: (biblioteca; não gera arquivos)
Como rodar: python -m scripts.utilitarios.util_v2
Tempo aproximado: não se aplica
"""

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from scripts.caminhos import ROOT

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CORPUS = ROOT / "dados" / "v2" / "corpus_v2.parquet"
FIG_DIR = ROOT / "outputs" / "figuras_v2"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES = ROOT / "resultados"
PROVISORIO = "PROVISÓRIO (heurística, aguardando B5)"
LARGURA_POL = 15.9 / 2.54  # 15,9 cm em polegadas
COR = {
    "bluesky": "#0072B2",
    "reddit": "#D55E00",
    "youtube": "#009E73",
    "total": "#444444",
}
NOME = {
    "bluesky": "Bluesky",
    "reddit": "Reddit",
    "youtube": "YouTube",
    "total": "Total",
}
FONTE_MIN = 8

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "font.family": "DejaVu Sans",
    }
)


def wilson(k, n, z=1.959964):
    """Intervalo de Wilson de 95% para uma proporção k/n (devolve proporção, inferior, superior em [0,1])."""
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    den = 1 + z**2 / n
    centro = (p + z**2 / (2 * n)) / den
    meia = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return (p, max(0.0, centro - meia), min(1.0, centro + meia))


def pct_ic(k, n):
    """Texto 'xx,x% [li-ls]' com Wilson."""
    p, lo, hi = wilson(k, n)
    if n == 0:
        return "-"
    return f"{p * 100:.1f}% [{lo * 100:.1f}-{hi * 100:.1f}]".replace(".", ",")


def qui2(tabela):
    """Qui-quadrado de independência em tabela de contingência; devolve (chi2, gl, p, V de Cramér, minimo esperado)."""
    t = np.asarray(tabela, dtype=float)
    t = t[t.sum(axis=1) > 0][:, t.sum(axis=0) > 0]
    if t.shape[0] < 2 or t.shape[1] < 2:
        return (np.nan, 0, np.nan, np.nan, np.nan)
    chi2, p, gl, esp = chi2_contingency(t, correction=False)
    v = np.sqrt(chi2 / (t.sum() * (min(t.shape) - 1)))
    return (chi2, gl, p, v, esp.min())


def fmt_p(p):
    if p is None or np.isnan(p):
        return "-"
    return "< 0,001" if p < 0.001 else f"{p:.3f}".replace(".", ",")


def num(n):
    return f"{int(n):,}".replace(",", ".")


def carrega(cols=None):
    base = [
        "id",
        "platform",
        "kind",
        "fonte_coleta",
        "created_utc",
        "sentiment_label",
        "stance_final",
        "rto_similarity",
    ]
    df = pd.read_parquet(CORPUS, columns=cols or base)
    if "created_utc" in df.columns:
        df["data"] = pd.to_datetime(
            df.created_utc, utc=True, format="mixed", errors="coerce"
        )
        assert df.data.notna().all()
    return df


def salva(fig, nome):
    caminho = FIG_DIR / nome
    fig.savefig(caminho, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return caminho


def marca_provisoria(fig, y=0.005):
    """Nota de rodapé dentro da figura para posicionamento (a legenda do TCC fica fora)."""
    fig.text(
        0.5,
        y,
        PROVISORIO,
        ha="center",
        va="bottom",
        fontsize=8,
        color="#B00020",
        style="italic",
    )


def md_tabela(cab, linhas):
    return [
        "| " + " | ".join(cab) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(cab) - 1)) + "|",
    ] + ["| " + " | ".join(str(c) for c in l) + " |" for l in linhas]
