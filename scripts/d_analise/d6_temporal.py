"""D6: Sentimento por fase e por mês, Spearman com Google Trends, posicionamento por fase (provisório) e eventos (dados/eventos.csv).

Etapa: D6
Objetivo: Sentimento por fase e por mês, Spearman com Google Trends, posicionamento por fase (provisório) e eventos (dados/eventos.csv).
Entradas: dados/v2/corpus_v2.parquet, dados/trends/, dados/eventos.csv
Saídas: resultados/D6.md, figuras em outputs/figuras_v2/
Como rodar: python -m scripts.d_analise.d6_temporal
Tempo aproximado: segundos
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scripts.utilitarios import util_v2 as u

PL = ["bluesky", "reddit", "youtube"]
FASES = ["até 2019", "2020-2021", "jan/2022 a mai/2023", "jun/2023 em diante"]
SENT = ["negativo", "neutro", "positivo"]
POS = ["anti-RTO", "pró-RTO", "neutro", "indefinido"]


def fase(d):
    if d.year <= 2019:
        return FASES[0]
    if d.year <= 2021:
        return FASES[1]
    if (d.year, d.month) <= (2023, 5):
        return FASES[2]
    return FASES[3]


df = u.carrega(
    ["id", "platform", "kind", "sentiment_label", "stance_final", "created_utc"]
)
df["fase"] = df.data.map(fase)
df["mes"] = df.data.dt.tz_localize(None).dt.to_period("M")
ULTIMO_MES_COMPLETO = pd.Period(
    "2026-09", "M"
)  # 2026-10 está incompleto (coleta até 24/10)


vp = lambda x: f"{x:.1f}".replace(".", ",")


def n30(n):
    return " (amostra pequena)" if n < 30 else ""


# ------------------------- 2. sentimento por fase e plataforma -------------------------
linhas, chi_fase = [], []
grupos = PL + ["total"]
for g in grupos:
    d = df if g == "total" else df[df.platform == g]
    for f in FASES:
        dd = d[d.fase == f]
        n = len(dd)
        k = int((dd.sentiment_label == "negativo").sum())
        linhas.append(
            [
                u.NOME[g],
                f,
                u.num(n),
                (u.pct_ic(k, n) if n else "-") + n30(n) if n else "-",
            ]
        )
    c = pd.crosstab(d.fase, d.sentiment_label).reindex(
        index=FASES, columns=SENT, fill_value=0
    )
    ch, gl, p, v, emin = u.qui2(c.values)
    ativas = int((c.sum(axis=1) > 0).sum())
    chi_fase.append(
        [
            u.NOME[g],
            str(ativas),
            "-" if gl == 0 else f"{ch:,.1f}".replace(",", " ").replace(".", ","),
            str(gl) if gl else "-",
            u.fmt_p(p),
            "-" if gl == 0 else f"{v:.3f}".replace(".", ","),
        ]
    )

# ------------------------- 6. posicionamento por fase (PROVISÓRIO) -------------------------
lpos = []
for g in grupos:
    d = df if g == "total" else df[df.platform == g]
    for f in FASES:
        dd = d[d.fase == f]
        n = len(dd)
        if n == 0:
            lpos.append([u.NOME[g], f, "0", "-", "-"])
            continue
        lpos.append(
            [
                u.NOME[g],
                f,
                u.num(n),
                u.pct_ic(int((dd.stance_final == "anti-RTO").sum()), n) + n30(n),
                u.pct_ic(int((dd.stance_final == "pró-RTO").sum()), n) + n30(n),
            ]
        )

# ------------------------- 3. séries mensais -------------------------
meses = pd.period_range("2019-01", ULTIMO_MES_COMPLETO, freq="M")
vol = {
    g: df[df.platform == g].groupby("mes").size().reindex(meses, fill_value=0)
    for g in PL
}
vol["total"] = df.groupby("mes").size().reindex(meses, fill_value=0)
neg = {
    g: df[(df.platform == g) & (df.sentiment_label == "negativo")]
    .groupby("mes")
    .size()
    .reindex(meses, fill_value=0)
    for g in PL
}
neg["total"] = (
    df[df.sentiment_label == "negativo"]
    .groupby("mes")
    .size()
    .reindex(meses, fill_value=0)
)
ult_mes_g = {g: df[df.platform == g].mes.max() for g in PL}
prim_mes_g = {g: df[df.platform == g].mes.min() for g in PL}


def media_movel_neg(g):
    """% negativo em janela móvel de 3 meses (soma de negativos / soma de textos na janela); janelas com n < 30 ficam vazias."""
    n3 = vol[g].rolling(3, min_periods=3).sum()
    k3 = neg[g].rolling(3, min_periods=3).sum()
    r = k3 / n3 * 100
    return r.where(n3 >= 30)


def figura_temporal(eventos=None, nome="fig_temporal_v2.png"):
    """Dois painéis (volume em cima, % negativo embaixo), uma linha por plataforma.
    eventos: lista de (data, rótulo) desenhados como linhas verticais."""
    if eventos is None:
        p = u.ROOT / "dados/eventos.csv"
        eventos = (
            [(pd.Timestamp(r.data), r.rotulo) for r in pd.read_csv(p).itertuples()]
            if p.exists()
            else []
        )
    fig, (a1, a2) = plt.subplots(
        2,
        1,
        figsize=(u.LARGURA_POL, 6.4),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1]},
    )
    x = meses.to_timestamp()
    for g in PL:
        v = vol[g].where((meses >= prim_mes_g[g]) & (meses <= ult_mes_g[g]))
        a1.plot(x, v.replace(0, np.nan), color=u.COR[g], lw=1.4, label=u.NOME[g])
        a2.plot(x, media_movel_neg(g), color=u.COR[g], lw=1.4, label=u.NOME[g])
    a1.set_yscale("log")
    a1.set_ylabel("Textos por mês (escala log)")
    a2.set_ylabel("% negativo (média móvel de 3 meses)")
    a2.set_ylim(10, 70)
    a1.legend(frameon=False, loc="upper left")
    for a in (a1, a2):
        a.spines[["top", "right"]].set_visible(False)
        for dt, rot in eventos:
            a.axvline(dt, color="#888", lw=0.8, ls="--")
    for i, (dt, rot) in enumerate(eventos):
        a2.text(
            dt,
            11 + (i % 4) * 4,
            rot,
            rotation=90,
            fontsize=8,
            va="bottom",
            ha="right",
            color="#444",
        )
    a2.xaxis.set_major_locator(mdates.YearLocator())
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    a2.set_xlabel("Mês (janelas com menos de 30 textos omitidas)")
    fig.tight_layout()
    u.salva(fig, nome)


figura_temporal()

# ------------------------- 4. Spearman com Google Trends -------------------------
t1 = pd.read_csv(u.ROOT / "dados/trends/trends_1.csv", skiprows=2)
t2 = pd.read_csv(u.ROOT / "dados/trends/trends_2.csv", skiprows=2)
tr = t1.merge(t2, on="Mês")
tr.columns = ["mes"] + [c.split(":")[0] for c in tr.columns[1:]]
for c in tr.columns[1:]:
    tr[c] = pd.to_numeric(tr[c].astype(str).str.replace("<1", "0.5"), errors="coerce")
tr["mes"] = pd.PeriodIndex(tr.mes, freq="M")
tr.to_csv(
    u.ROOT / "dados/trends/trends_mensal.csv", index=False
)  # gerado aqui a partir de trends_1/2 (não existia)
tr = tr.set_index("mes")
sp = []
for g in grupos:
    ini = pd.Period("2023-06", "M") if g == "bluesky" else pd.Period("2019-01", "M")
    fim = (
        min(ult_mes_g[g], ULTIMO_MES_COMPLETO) if g != "total" else ULTIMO_MES_COMPLETO
    )
    idx = pd.period_range(ini, fim, freq="M")
    for termo in ["home office", "trabalho presencial"]:
        rho, p = spearmanr(vol[g].reindex(idx).values, tr[termo].reindex(idx).values)
        sp.append(
            [
                u.NOME[g],
                termo,
                f"{ini}–{fim}",
                str(len(idx)),
                f"{rho:.2f}".replace(".", ","),
                u.fmt_p(p),
            ]
        )

# ------------------------- 5. participação do Bluesky -------------------------
pos24 = df[df.data >= "2024-01-01"]
sh = {
    a: (df[(df.data.dt.year == a)].platform == "bluesky").mean() * 100
    for a in (2024, 2025, 2026)
}
sh24 = (pos24.platform == "bluesky").mean() * 100

# ------------------------- relatório -------------------------
L = [
    "# D6: análise temporal (parte que não depende do B5)",
    "",
    "## O que foi feito",
    "Sentimento por fase e por mês no corpus v2 (49.269 linhas), com intervalo de Wilson de 95%; Spearman entre o volume mensal de textos e o Google Trends; posicionamento por fase (PROVISÓRIO). Script `d6_temporal.py`; figura `outputs/figuras_v2/fig_temporal_v2.png` (sem título, 15,9 cm, 300 dpi). A função `figura_temporal(eventos=[(data, rótulo), ...])` aceita uma lista de eventos (linhas verticais) ou o arquivo `dados/eventos.csv` (colunas `data`, `rotulo`).",
    "",
    "Fases: até 2019; 2020-2021; jan/2022 a mai/2023; jun/2023 em diante.",
    "",
    "## Aviso principal: troca de plataforma",
    f"A partir de 2024 o Bluesky responde por {vp(sh24)}% dos textos do corpus (por ano: 2024 = {vp(sh[2024])}%; 2025 = {vp(sh[2025])}%; 2026 = {vp(sh[2026])}%), e só começa em jun/2023. O valor de cerca de 85% citado no pedido corresponde ao ano de 2024; para o conjunto de 2024 em diante o valor calculado é {vp(sh24)}%. **A comparação entre fases mistura mudança de opinião com troca de plataforma**: a última fase é quase inteira Bluesky (posts), enquanto as anteriores são YouTube e Reddit. Por isso a comparação de fases deve ser lida por plataforma, e só o YouTube e o Reddit têm mais de uma fase.",
    "",
    "## 1. % de sentimento negativo por fase e plataforma (Wilson 95%)",
    "",
]
L += u.md_tabela(["Plataforma", "Fase", "n", "% negativo [IC 95%]"], linhas)
L += [
    "",
    "Qui-quadrado entre fases (fase x sentimento negativo/neutro/positivo), dentro de cada plataforma; fases com n = 0 saem do teste e com n < 30 estão marcadas:",
    "",
]
L += u.md_tabela(
    ["Plataforma", "Fases com texto", "χ²", "gl", "p", "V de Cramér"], chi_fase
)
L += [
    "",
    'Bluesky tem uma só fase (jun/2023 em diante), então não há teste entre fases para ele. O "Total" mistura plataformas e é afetado pela troca de plataforma.',
    "",
    "## 2. Séries mensais",
    "",
    "Figura `fig_temporal_v2.png`: painel superior com o volume mensal por plataforma (escala log) e painel inferior com % negativo em janela móvel de 3 meses (soma de negativos / soma de textos nos 3 meses; janelas com menos de 30 textos omitidas). O mês 2026-10 foi excluído por estar incompleto (coleta até 24/10).",
    "",
    "## 3. Spearman: volume mensal x Google Trends",
    "",
    '`dados/trends/trends_mensal.csv` **não existia**: foi gerado a partir de `trends_1.csv` e `trends_2.csv` (7 termos, mensal, Brasil, 2019-01 a 2026-10; valores "<1" tratados como 0,5). Cada arquivo tem sua própria escala (0 a 100 dentro do arquivo), o que não afeta o Spearman, que usa postos.',
    "",
]
L += u.md_tabela(
    ["Série de volume", "Termo (Trends)", "Meses", "n de meses", "rho", "p"], sp
)
L += [
    "",
    "Bluesky usa só meses a partir de 2023-06. Reddit termina onde acaba a coleta (último mês com texto: "
    + str(ult_mes_g["reddit"])
    + "). O volume mensal de YouTube e Reddit depende de como os vídeos e subreddits foram coletados (comentários se acumulam em torno da data de publicação dos vídeos), então o rho mede coincidência com a atenção de busca, não representatividade.",
    "",
    "## 4. Posicionamento por fase (PROVISÓRIO: heurística, aguardando B5)",
    "",
    f"**{u.PROVISORIO}.** `stance_final` hoje é igual à heurística; células com n < 30 estão marcadas.",
    "",
]
L += u.md_tabela(
    ["Plataforma", "Fase", "n", "% anti-RTO [IC 95%]", "% pró-RTO [IC 95%]"], lpos
)
L += [
    "",
    "## Problemas e decisões",
    '- Fases com n < 30 estão marcadas como "amostra pequena" (principalmente "até 2019" no Reddit).',
    "- Média móvel calculada sobre as contagens (não sobre a média das proporções mensais), para que meses com poucos textos não distorçam.",
    "- Reddit: o corpus tem 0 textos em 2026 (a coleta histórica termina em 2025); no Spearman o Reddit usa só os meses até o último com dado.",
    "- Spearman sem correção para autocorrelação temporal; os p-valores devem ser lidos com cautela (séries mensais são autocorrelacionadas).",
    "- Nenhuma lista de eventos foi desenhada (aguarda a lista do Chat); a função está pronta.",
    "- Nenhum modelo foi carregado.",
]
open(u.RES / "D6.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("D6 OK", round(sh24, 1), sp[:2])
