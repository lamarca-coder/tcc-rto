"""D3: Comparação entre plataformas (sentimento e posicionamento), com Wilson e qui-quadrado.

Etapa: D3
Objetivo: Comparação entre plataformas (sentimento e posicionamento), com Wilson e qui-quadrado.
Entradas: dados/v2/corpus_v2.parquet
Saídas: resultados/D3.md, figuras em outputs/figuras_v2/
Como rodar: python -m scripts.d_analise.d3_plataformas
Tempo aproximado: segundos
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.utilitarios import util_v2 as u

df = u.carrega(["platform", "kind", "sentiment_label", "stance_final"])
PLATS = ["bluesky", "reddit", "youtube"]
SENT = ["negativo", "neutro", "positivo"]
POS = [
    c
    for c in ["anti-RTO", "pró-RTO", "neutro", "indefinido"]
    if c in set(df.stance_final)
]
ROT_SENT = {"negativo": "Negativo", "neutro": "Neutro", "positivo": "Positivo"}


def bloco(d, col, grupo, ordem_grupo, ordem_cat, rotulo_grupo):
    c = pd.crosstab(d[grupo], d[col]).reindex(
        index=ordem_grupo, columns=ordem_cat, fill_value=0
    )
    chi2, gl, p, v, emin = u.qui2(c.values)
    cab = [rotulo_grupo, "n"] + [f"{k}: n e % [IC 95%]" for k in ordem_cat]
    linhas = []
    for g in ordem_grupo:
        n = int(c.loc[g].sum())
        linhas.append(
            [g, u.num(n)]
            + [f"{u.num(c.loc[g, k])} ({u.pct_ic(c.loc[g, k], n)})" for k in ordem_cat]
        )
    n = int(c.values.sum())
    linhas.append(
        ["Total", u.num(n)]
        + [f"{u.num(c[k].sum())} ({u.pct_ic(c[k].sum(), n)})" for k in ordem_cat]
    )
    teste = (
        f"Qui-quadrado de independência: χ² = {chi2:,.1f}".replace(",", " ").replace(
            ".", ","
        )
        + f", gl = {gl}, p {u.fmt_p(p)}, V de Cramér = {v:.3f}".replace(".", ",")
        + f" (menor frequência esperada = {emin:,.0f})."
    )
    return c, linhas, cab, teste, v


def figura(c, ordem_cat, rot_cat, nome, ylabel, provisorio=False):
    fig, ax = plt.subplots(figsize=(u.LARGURA_POL, 4.0))
    w = 0.26
    x = np.arange(len(ordem_cat))
    for i, g in enumerate(c.index):
        n = c.loc[g].sum()
        vals, lo, hi = zip(*[u.wilson(c.loc[g, k], n) for k in ordem_cat])
        vals, lo, hi = np.array(vals) * 100, np.array(lo) * 100, np.array(hi) * 100
        ax.bar(
            x + (i - (len(c.index) - 1) / 2) * w,
            vals,
            w,
            color=u.COR.get(g, "#888"),
            label=f"{u.NOME.get(g, g)} (n = {u.num(n)})",
            yerr=[vals - lo, hi - vals],
            capsize=3,
            error_kw={"lw": 1},
        )
    ax.set_xticks(x)
    ax.set_xticklabels([rot_cat[k] for k in ordem_cat])
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.05)
    ax.legend(frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if provisorio:
        fig.subplots_adjust(bottom=0.17)
        u.marca_provisoria(fig)
    u.salva(fig, nome)


# 1. sentimento x plataforma
c1, l1, cab1, t1, v1 = bloco(
    df, "sentiment_label", "platform", PLATS, SENT, "Plataforma"
)
figura(
    c1,
    SENT,
    ROT_SENT,
    "fig_sentimento_plataforma.png",
    "Proporção de textos (%), com IC de Wilson 95%",
)
# 2. Reddit: post x comentário
rd = df[df.platform == "reddit"].copy()
rd["tipo"] = rd.kind.map({"post": "post", "comment": "comentário"})
c2, l2, cab2, t2, v2 = bloco(
    rd, "sentiment_label", "tipo", ["post", "comentário"], SENT, "Tipo (Reddit)"
)
# 3. posicionamento x plataforma (provisório)
c3, l3, cab3, t3, v3 = bloco(df, "stance_final", "platform", PLATS, POS, "Plataforma")
figura(
    c3,
    POS,
    {k: k for k in POS},
    "fig_posicionamento_plataforma.png",
    "Proporção de textos (%), com IC de Wilson 95%",
    provisorio=True,
)
# apoio: pares de plataformas (V de Cramér) para sentimento
pares = []
for a, b in [("bluesky", "reddit"), ("bluesky", "youtube"), ("reddit", "youtube")]:
    ch, gl, p, v, _ = u.qui2(c1.loc[[a, b]].values)
    pares.append(
        [
            f"{u.NOME[a]} x {u.NOME[b]}",
            f"{ch:,.1f}".replace(",", " ").replace(".", ","),
            u.fmt_p(p),
            f"{v:.3f}".replace(".", ","),
        ]
    )

L = (
    [
        "# D3: comparação entre plataformas",
        "",
        "## O que foi feito",
        "Sentimento (`sentiment_label`) e posicionamento (`stance_final`) por plataforma no corpus v2 (49.269 linhas), com proporções e intervalo de Wilson de 95%, qui-quadrado de independência (sem correção de continuidade), V de Cramér e p-valor. Script `d3_plataformas.py`; figuras em `outputs/figuras_v2/` (sem título, 15,9 cm, 300 dpi).",
        "",
        "## 1. Sentimento por plataforma",
        "",
    ]
    + u.md_tabela(cab1, l1)
    + ["", t1, "", "Comparações par a par (sentimento):", ""]
    + u.md_tabela(["Par", "χ²", "p", "V de Cramér"], pares)
    + [
        "",
        "Figura: `fig_sentimento_plataforma.png`.",
        "",
        "## 2. Sentimento no Reddit, post x comentário",
        "",
    ]
    + u.md_tabela(cab2, l2)
    + [
        "",
        t2,
        "",
        "## 3. Posicionamento por plataforma (PROVISÓRIO: heurística, aguardando B5)",
        "",
        f"**{u.PROVISORIO}.** A coluna lida é `stance_final`, hoje igual à heurística (sem validação humana).",
        "",
    ]
    + u.md_tabela(cab3, l3)
    + [
        "",
        t3,
        "",
        "Figura: `fig_posicionamento_plataforma.png` (marcada como provisória na própria imagem).",
        "",
        "## Números principais",
        "- % negativo: "
        + ", ".join(
            f"{u.NOME[g]} {c1.loc[g, 'negativo'] / c1.loc[g].sum() * 100:.1f}%".replace(
                ".", ","
            )
            for g in PLATS
        )
        + f"; V de Cramér do sentimento entre as 3 plataformas = {v1:.3f}.".replace(
            ".", ","
        ),
        f"- Reddit: posts {c2.loc['post', 'negativo'] / c2.loc['post'].sum() * 100:.1f}% negativos, comentários {c2.loc['comentário', 'negativo'] / c2.loc['comentário'].sum() * 100:.1f}%; V = {v2:.3f}.".replace(
            ".", ","
        ),
        "",
        "## Problemas e decisões",
        "- Com n da ordem de dezenas de milhares, o p-valor fica abaixo de 0,001 mesmo com diferenças pequenas; o tamanho de efeito (V de Cramér) é o que deve ser lido. Regra prática de Cohen para V com 2 graus de liberdade: 0,07 pequeno, 0,21 médio, 0,35 grande.",
        "- Plataforma e tipo de texto se confundem: o Bluesky só tem posts, o YouTube só tem comentários e o Reddit tem os dois; por isso o item 2 separa o Reddit. Diferenças entre plataformas misturam público, formato do texto e época (o Bluesky só cobre 2023 em diante).",
        "- O Bluesky representa 73,7% do corpus, então o total do corpus é dominado por ele.",
        "- Posicionamento: a heurística combina alvo e polaridade e subestima o anti-RTO quando alguém defende o home office em tom negativo; refazer depois do B5.",
        "- Nenhum modelo foi carregado; os scripts só leem `corpus_v2.parquet`.",
    ]
)
open(u.RES / "D3.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print(
    "D3 OK",
    [f"{g}:{c1.loc[g, 'negativo'] / c1.loc[g].sum():.3f}" for g in PLATS],
    round(v1, 3),
)
