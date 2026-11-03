"""B11: Resultados por amostra: rótulos do LLM no B9 com pesos de sorteio e bootstrap estratificado (1.000, semente 42); sentimento, alvo e posicionamento por plataforma e fase; comparação com a heurística e o survey.

Etapa: B11
Objetivo: Resultados por amostra: rótulos do LLM no B9 com pesos de sorteio e bootstrap estratificado (1.000, semente 42); sentimento, alvo e posicionamento por plataforma e fase; comparação com a heurística e o survey.
Entradas: rotulagem/b9/ (rótulos e pesos), dados/eventos.csv, data/kaggle/
Saídas: resultados/B11.md, B11_numeros.csv, D6_legenda_eventos.md; outputs/figuras_v2/fig_b11_*.png
Como rodar: python -m scripts.d_analise.b11_resultados
Tempo aproximado: cerca de 1 minuto
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from scripts.utilitarios import util_v2 as u

R = u.ROOT / "rotulagem"
SEED = 42
B = 1000
IRR = "irrelevante"
PO = ["anti_RTO", "pro_RTO", "neutro_misto"]
SE = ["negativo", "positivo", "neutro"]
AL = ["presencial", "remoto", "hibrido", "geral"]
FASES = ["até 2019", "2020-2021", "jan/2022 a mai/2023", "jun/2023 em diante"]
PLS = ["bluesky", "reddit", "youtube"]
NOMES = (
    ["irrel"]
    + [f"pos_{c}" for c in PO]
    + [f"sent_{c}" for c in SE]
    + [f"alvo_{c}" for c in AL]
    + [f"todas_{c}" for c in PO]
)
IDX_N = {n: i for i, n in enumerate(NOMES)}
pcn = lambda x: "-" if x is None or np.isnan(x) else f"{x * 100:.1f}%".replace(".", ",")

# ---------------- dados ----------------
c = pd.read_parquet(
    u.CORPUS, columns=["id", "platform", "created_utc", "stance_heuristica"]
)
c["data"] = pd.to_datetime(c.created_utc, utc=True, format="mixed", errors="coerce")
assert c.data.notna().all()
c["ano"] = c.data.dt.year
pesos = pd.read_parquet(R / "b9/b9_mapa_pesos.parquet")
l9 = pd.concat(
    [pd.read_excel(R / f"b9/rotulos_llm_b9_parte{k}.xlsx") for k in (1, 2, 3, 4)]
)[["id_rotulagem", "alvo", "sentimento", "posicionamento"]]
est = (
    l9.merge(pesos, on="id_rotulagem")
    .merge(c[["id", "ano"]], on="id")
    .reset_index(drop=True)
)
assert (
    len(est) == 2000 and ((est.sentimento == IRR) == (est.posicionamento == IRR)).all()
)
N = len(est)
w = est.peso.values.astype(float)
irr = (est.sentimento == IRR).values.astype(float)
rel = 1 - irr
onehot = lambda col, cats: np.stack(
    [(est[col].values == k).astype(float) for k in cats], axis=1
)
P, S, A = onehot("posicionamento", PO), onehot("sentimento", SE), onehot("alvo", AL)
estr = (est.plataforma + "|" + est.fase).values
pos_estrato = [np.where(estr == s)[0] for s in sorted(set(estr))]


def stats(idx, m):
    ww = w[idx] * m[idx]
    wr = ww * rel[idx]
    tot, tr = ww.sum(), wr.sum()
    if tot == 0:
        return np.full(len(NOMES), np.nan)
    out = [(ww * irr[idx]).sum() / tot]
    for M in (P, S, A):
        out += (
            list((wr[:, None] * M[idx]).sum(0) / tr)
            if tr > 0
            else [np.nan] * M.shape[1]
        )
    out += list((ww[:, None] * P[idx]).sum(0) / tot)
    return np.array(out)


# grupos
grupos = {"Total": np.ones(N, bool)}
for p in PLS:
    grupos[f"{u.NOME[p]}, todas as fases"] = (est.plataforma == p).values
for f in FASES:
    grupos[f"Todas as plataformas, {f}"] = (est.fase == f).values
for p in PLS:
    for f in FASES:
        m = ((est.plataforma == p) & (est.fase == f)).values
        if m.any():
            grupos[f"{u.NOME[p]}, {f}"] = m
ANOS = sorted(est.ano.unique())
for a in ANOS:
    grupos[f"ano {a}"] = (est.ano == a).values

rng = np.random.default_rng(SEED)
pt = {g: stats(np.arange(N), m) for g, m in grupos.items()}
boot = {g: np.empty((B, len(NOMES))) for g in grupos}
for b in range(B):
    idx = np.concatenate(
        [ph[rng.integers(0, len(ph), len(ph))] for ph in pos_estrato]
    )  # reamostra dentro de cada estrato
    for g, m in grupos.items():
        boot[g][b] = stats(idx, m)
lo = {g: np.nanpercentile(boot[g], 2.5, axis=0) for g in grupos}
hi = {g: np.nanpercentile(boot[g], 97.5, axis=0) for g in grupos}
info = {
    g: (int(m.sum()), int(round(w[m].sum())), int(rel[m].sum()))
    for g, m in grupos.items()
}
cel = lambda g, nome: (
    "-"
    if np.isnan(pt[g][IDX_N[nome]])
    else f"{pcn(pt[g][IDX_N[nome]])} [{pcn(lo[g][IDX_N[nome]])}-{pcn(hi[g][IDX_N[nome]])}]"
)

# arquivo com todos os números (para conferir depois da reorganização)
linhas = []
for g in grupos:
    for nome in NOMES:
        linhas.append(
            {
                "grupo": g,
                "estatistica": nome,
                "estimativa": round(float(pt[g][IDX_N[nome]]), 6),
                "ic_inf": round(float(lo[g][IDX_N[nome]]), 6),
                "ic_sup": round(float(hi[g][IDX_N[nome]]), 6),
                "n_sorteado": info[g][0],
                "n_relevantes_amostra": info[g][2],
                "N_representado": info[g][1],
            }
        )
pd.DataFrame(linhas).to_csv(u.RES / "B11_numeros.csv", index=False)

GR_TAB = (
    ["Total"]
    + [f"{u.NOME[p]}, todas as fases" for p in PLS]
    + [f"Todas as plataformas, {f}" for f in FASES]
    + [
        g
        for g in grupos
        if g.split(", ")[0] in [u.NOME[p] for p in PLS]
        and not g.endswith("todas as fases")
    ]
)


def tabela(cats, prefixo, rot):
    cab = ["Grupo", "n sorteado", "n relevantes (amostra)", "N representado"] + [
        f"% {c} [IC 95%]" for c in cats
    ]
    return u.md_tabela(
        cab,
        [
            [g, u.num(info[g][0]), u.num(info[g][2]), u.num(info[g][1])]
            + [cel(g, f"{prefixo}_{c}") for c in cats]
            for g in GR_TAB
        ],
    )


# ---------------- figura: posicionamento por plataforma ----------------
COR_PO = {"anti_RTO": "#0072B2", "pro_RTO": "#D55E00", "neutro_misto": "#999999"}
ROT_PO = {"anti_RTO": "anti-RTO", "pro_RTO": "pró-RTO", "neutro_misto": "neutro/misto"}
fig, ax = plt.subplots(figsize=(u.LARGURA_POL, 3.9))
gs_ = [(f"{u.NOME[p]}, todas as fases", u.NOME[p]) for p in PLS] + [("Total", "Total")]
x = np.arange(len(gs_))
wd = 0.26
for j, cl in enumerate(PO):
    v = np.array([pt[g][IDX_N[f"pos_{cl}"]] for g, _ in gs_]) * 100
    l_ = np.array([lo[g][IDX_N[f"pos_{cl}"]] for g, _ in gs_]) * 100
    h_ = np.array([hi[g][IDX_N[f"pos_{cl}"]] for g, _ in gs_]) * 100
    ax.bar(
        x + (j - 1) * wd,
        v,
        wd,
        color=COR_PO[cl],
        label=ROT_PO[cl],
        yerr=[v - l_, h_ - v],
        capsize=3,
        error_kw={"lw": 1},
    )
ax.set_xticks(x)
ax.set_xticklabels([f"{n}\n(n relev. = {info[g][2]})" for g, n in gs_])
ax.set_ylabel("% entre os textos relevantes (IC 95%)")
ax.set_ylim(0, 80)
ax.legend(frameon=False, ncol=3, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
u.salva(fig, "fig_b11_posicionamento.png")

# ---------------- eventos ----------------
ev = pd.read_csv(u.ROOT / "dados/eventos.csv", encoding="utf-8-sig")
ev = ev[ev.incluir_na_figura.astype(str).str.strip().str.lower() == "s"].copy()
ev["n"] = ev.numero_figura.astype(float).astype(int)
ev["d"] = pd.to_datetime(ev.data)
ev = ev.sort_values("d")
assert len(ev) == 15, len(ev)
GRUPOS_EV = sorted(ev.grupo.unique())
assert len(GRUPOS_EV) == 4
COR_EV = {  # seguras para daltônicos e distintas das cores das plataformas
    "empresas brasileiras": "#6A3D9A",  # roxo
    "empresas globais": "#9A6B00",  # ocre
    "legal e governo": "#8B0000",  # vermelho escuro
    "pandemia e reabertura": "#4D4D4D",  # cinza escuro
}
assert set(COR_EV) == set(GRUPOS_EV)
leg = (
    [
        "# D6: legenda dos eventos da figura temporal",
        "",
        "Eventos de `dados/eventos.csv` com `incluir_na_figura = s` (15), na ordem do número na figura (`fig_b11_fases.png`). A cor do número e da linha tracejada indica o grupo.",
        "",
        "| Grupo | Cor |",
        "|---|---|",
    ]
    + [f"| {g} | `{COR_EV[g]}` |" for g in GRUPOS_EV]
    + ["", "| Nº | Data | Rótulo | Fonte |", "|---:|---|---|---|"]
)
for r in ev.itertuples():
    fonte = (
        f"[{r.fonte}]({r.url})"
        if isinstance(r.url, str) and r.url.startswith("http")
        else str(r.fonte)
    )
    leg.append(f"| {r.n} | {r.d:%d/%m/%Y} | {r.rotulo_figura} | {fonte} |")
leg += [
    "",
    'Observação: datas com confiança "média" no arquivo de origem (`confianca_da_data`) vêm da data da matéria e não do ato: '
    + ", ".join(
        f"nº {r.n}"
        for r in ev.itertuples()
        if str(r.confianca_da_data).lower().startswith("m")
    )
    + ".",
]
(u.RES / "D6_legenda_eventos.md").write_text("\n".join(leg) + "\n", encoding="utf-8")

# ---------------- figura: fases (anti e pró) com eventos ----------------
SPAN = {
    "até 2019": ("2019-01-01", "2020-01-01"),
    "2020-2021": ("2020-01-01", "2022-01-01"),
    "jan/2022 a mai/2023": ("2022-01-01", "2023-06-01"),
    "jun/2023 em diante": ("2023-06-01", "2026-10-25"),
}
fig, axs = plt.subplots(2, 1, figsize=(u.LARGURA_POL, 6.6), sharex=True)
off = {"bluesky": -18, "reddit": 0, "youtube": 18}
for ax, cl, rot in zip(
    axs,
    ["anti_RTO", "pro_RTO"],
    ["% anti-RTO entre os relevantes", "% pró-RTO entre os relevantes"],
):
    for p in PLS:
        xs, ys, ls, hs = [], [], [], []
        for f in FASES:
            g = f"{u.NOME[p]}, {f}"
            if g not in grupos or info[g][2] < 10:
                continue  # estrato com menos de 10 relevantes na amostra: não desenhado
            a, b_ = pd.Timestamp(SPAN[f][0]), pd.Timestamp(SPAN[f][1])
            mid = a + (b_ - a) / 2 + pd.Timedelta(days=off[p])
            v, l_, h_ = (
                pt[g][IDX_N[f"pos_{cl}"]] * 100,
                lo[g][IDX_N[f"pos_{cl}"]] * 100,
                hi[g][IDX_N[f"pos_{cl}"]] * 100,
            )
            ax.hlines(v, a, b_, color=u.COR[p], lw=2.2, alpha=0.45)
            ax.errorbar(
                [mid],
                [v],
                yerr=[[v - l_], [h_ - v]],
                fmt="o",
                color=u.COR[p],
                capsize=3,
                ms=4,
                lw=1.2,
            )
            xs.append(mid)
            ys.append(v)
        if len(xs) > 1:
            ax.plot(xs, ys, color=u.COR[p], lw=0.8, alpha=0.8)
    ax.set_ylabel(rot)
    ax.set_ylim(0, 85 if cl == "anti_RTO" else 25)
    ax.spines[["top", "right"]].set_visible(False)
    for r in ev.itertuples():
        ax.axvline(r.d, color=COR_EV[r.grupo], lw=0.8, ls="--", alpha=0.7)
for k, r in enumerate(ev.itertuples()):
    axs[0].annotate(
        str(r.n),
        xy=(r.d, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(0, 3 + (k % 2) * 9),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color=COR_EV[r.grupo],
        annotation_clip=False,
    )
axs[1].xaxis.set_major_locator(mdates.YearLocator())
axs[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axs[1].set_xlim(pd.Timestamp("2018-10-01"), pd.Timestamp("2026-12-01"))
h1 = [
    Line2D([0], [0], color=u.COR[p], marker="o", lw=1.5, ms=4, label=u.NOME[p])
    for p in PLS
]
h2 = [Line2D([0], [0], color=COR_EV[g], lw=1.2, ls="--", label=g) for g in GRUPOS_EV]
axs[1].legend(
    handles=h1, frameon=False, loc="upper left", title="Plataforma", title_fontsize=8
)
fig.legend(
    handles=h2,
    frameon=False,
    loc="lower center",
    ncol=2,
    title="Eventos (números no topo)",
    title_fontsize=8,
    bbox_to_anchor=(0.5, -0.01),
)
fig.tight_layout(rect=(0, 0.07, 1, 0.97))
u.salva(fig, "fig_b11_fases.png")

# ---------------- heurística x LLM (total e por plataforma) ----------------
c["h"] = c.stance_heuristica.map(
    lambda s: (
        "anti_RTO"
        if str(s).startswith("anti")
        else "pro_RTO"
        if str(s).startswith("pr")
        else "neutro_misto"
    )
)
comp = []
for g, sel in [("Total", np.ones(len(c), bool))] + [
    (u.NOME[p], (c.platform == p).values) for p in PLS
]:
    h = c[sel].h.value_counts(normalize=True)
    key = "Total" if g == "Total" else f"{g}, todas as fases"
    comp.append(
        [g, u.num(sel.sum())]
        + [pcn(h.get(k, 0)) for k in PO]
        + [cel(key, "irrel")]
        + [cel(key, f"pos_{k}") for k in PO]
        + [cel(key, f"todas_{k}") for k in PO]
    )

# ---------------- survey x redes ----------------
from scripts.utilitarios import survey_triangulation as st

sv = {}
for ano, caminho in st.SURVEYS.items():
    d = pd.read_csv(u.ROOT / caminho, low_memory=False)
    col = st._find_col(d, st.Q_IDEAL)
    v = d[col].map(st._norm_model).dropna()
    sv[ano] = (int((v == "100% presencial").sum()), len(v))
surv_linhas = []
for a in ANOS:
    g = f"ano {a}"
    if info[g][2] == 0:
        continue  # anos sem texto relevante na amostra (2015 a 2018) ficam fora da tabela
    k, n = sv.get(a, (None, None))
    surv_linhas.append(
        [
            a,
            u.num(info[g][0]),
            u.num(info[g][2]),
            cel(g, "pos_pro_RTO"),
            cel(g, "pos_anti_RTO"),
            (f"{u.num(n)}" if n else "-"),
            (u.pct_ic(k, n) if n else "-"),
        ]
    )
# ---------------- relatório ----------------
L = (
    [
        "# B11: resultados principais por amostra (rótulos do LLM no B9)",
        "",
        "## O que foi feito",
        'Estimativas ponderadas sobre os **2.000 textos do B9 rotulados pelo LLM** (`rotulagem/b9/rotulos_llm_b9_parte1..4.xlsx`), com o peso de cada texto (N do estrato / n sorteado; `b9_mapa_pesos.parquet`). Intervalo de 95% por **bootstrap estratificado** (1.000 reamostragens dentro de cada estrato plataforma x fase, semente 42). "Entre os relevantes" é a razão entre o peso da classe e o peso dos relevantes. Todos os números estão em `resultados/B11_numeros.csv` (conferência depois da reorganização). Script `b11_resultados.py`.',
        "",
        "**Os rótulos são do LLM, não de humanos.** O LLM r1 teve kappa 0,74 no posicionamento, 0,67 no sentimento e 0,72 no alvo contra o gabarito humano de 150 linhas (tratamento b do B5); no Bluesky não houve validação humana (B8). A heurística do pipeline não é usada nas estimativas (só na seção 4, para comparação).",
        "",
        "O universo é o corpus v2 sem os 650 textos já rotulados (48.619 textos). Fases: até 2019; 2020-2021; jan/2022 a mai/2023; jun/2023 em diante. O Bluesky só existe na última fase (80% dos textos de 2024 em diante).",
        "",
        "## 1. Posicionamento entre os relevantes (a tabela repete a da seção 5 do B10)",
        "",
    ]
    + tabela(PO, "pos", "")
    + [
        "",
        "Figura: `outputs/figuras_v2/fig_b11_posicionamento.png` (% por plataforma, com barras de erro).",
        "",
        "## 2. Sentimento entre os relevantes",
        "",
    ]
    + tabela(SE, "sent", "")
    + ["", "## 3. Alvo entre os relevantes", ""]
    + tabela(AL, "alvo", "")
    + [
        "",
        "## 4. Heurística no corpus inteiro x estimativa por amostra do LLM",
        "",
        'Heurística: `stance_heuristica` dos 49.269 textos (neutro e indefinido contados como neutro/misto; a heurística não tem classe irrelevante). LLM: estimativa ponderada; a coluna "entre relevantes" tem como base só os textos relevantes e a coluna "em todas as linhas" tem como base o universo inteiro (a soma com os irrelevantes dá 100%).',
        "",
    ]
    + u.md_tabela(
        [
            "Grupo",
            "n (heurística)",
            "Heur. % anti",
            "Heur. % pró",
            "Heur. % neutro/misto",
            "LLM % irrelevante",
            "LLM % anti (relev.)",
            "LLM % pró (relev.)",
            "LLM % neutro/misto (relev.)",
            "LLM % anti (todas)",
            "LLM % pró (todas)",
            "LLM % neutro/misto (todas)",
        ],
        comp,
    )
    + [
        "",
        "A heurística infla o pró-RTO e deixa o anti-RTO menor porque trata sentimento negativo sobre o remoto como pró-RTO (B5, seção 6); além disso, não separa os irrelevantes.",
        "",
        "## 5. Redes sociais x survey (State of Data Brazil)",
        "",
        'Redes: % pró-RTO e % anti-RTO entre os relevantes em cada ano (estimativa de domínio com os pesos do B9; poucos relevantes por ano, intervalos largos). Survey: % que prefere o modelo 100% presencial como ideal (todos os respondentes de cada edição, Wilson 95%; resultados/D7.md e triangulacao_survey.md). A tabela é uma justaposição, não uma comparação direta: as populações e as perguntas são diferentes (comentários públicos em português x profissionais de dados que responderam a uma pesquisa), e "pró-RTO" nas redes inclui defender a volta ao escritório, não só preferir o modelo 100% presencial.',
        "",
    ]
    + u.md_tabela(
        [
            "Ano",
            "n sorteado",
            "n relevantes (amostra)",
            "Redes: % pró-RTO [IC]",
            "Redes: % anti-RTO [IC]",
            "Survey: n",
            "Survey: % prefere 100% presencial [IC]",
        ],
        surv_linhas,
    )
    + [
        "",
        "Leitura: nos anos com survey (2021 a 2024) só 1% a 2% preferem o modelo 100% presencial; nas redes, o pró-RTO entre os relevantes fica numa faixa de poucos por cento a cerca de 15% (dependendo da plataforma e da fase, seção 1). Os dois indicadores apontam que a posição pró-presencial é minoritária, mas o do survey mede preferência pessoal e o das redes mede posição expressa sobre a volta ao escritório.",
        "",
        "## 6. Figuras",
        "",
        "- `fig_b11_posicionamento.png`: % anti_RTO, pro_RTO e neutro_misto entre os relevantes, por plataforma (e total), com intervalo de 95%.",
        '- `fig_b11_fases.png`: % anti_RTO (painel superior) e % pro_RTO (inferior) entre os relevantes por fase, uma linha por plataforma (YouTube e Reddit nas 4 fases; Bluesky só na última), com barras de erro; as 15 linhas verticais tracejadas são os eventos de `dados/eventos.csv` com `incluir_na_figura = s`, numerados no topo e coloridos por grupo. Legenda em `resultados/D6_legenda_eventos.md`. Cada fase é desenhada como um traço horizontal sobre o período que cobre, com o ponto e a barra de erro no meio dele (deslocados alguns dias entre plataformas para não se sobrepor); a fase "até 2019" é desenhada só sobre 2019 embora inclua textos de anos anteriores; estratos com menos de 10 textos relevantes na amostra não são desenhados (Reddit até 2019 tem 1 relevante; a estimativa de 100% anti-RTO seria degenerada).',
        "- Figuras sem título, 15,9 cm de largura, 300 dpi, fonte de pelo menos 8 pt.",
        "",
        "## Problemas e decisões",
        "- Estimativas por fase combinam mudança de opinião com troca de plataforma: a última fase é quase toda Bluesky. Por isso a figura por fase mostra uma linha por plataforma.",
        "- Estratos pequenos (Reddit até 2019, N = 14; YouTube até 2019, N = 132) têm intervalos pouco informativos; no Reddit até 2019 há um só texto relevante, e a estimativa do posicionamento é degenerada (100% ou 0%).",
        "- Os eventos não entram em nenhuma estimativa; são só referência visual para as fases.",
        '- Os rótulos do LLM foram dados com o livro de códigos v2/v3 e confiança variável (31% "baixa" no B8); o erro do LLM não está incluído nos intervalos, que medem só o erro de amostragem.',
        "- Nenhum modelo foi carregado.",
    ]
)
(u.RES / "B11.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("B11 OK")
for g in ["Total"] + [f"{u.NOME[p]}, todas as fases" for p in PLS]:
    print(
        g,
        {
            n: round(float(pt[g][IDX_N[n]]), 3)
            for n in [
                "irrel",
                "pos_anti_RTO",
                "pos_pro_RTO",
                "pos_neutro_misto",
                "sent_negativo",
            ]
        },
    )
