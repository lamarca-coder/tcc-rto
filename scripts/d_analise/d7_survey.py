"""D7: Survey State of Data (2021-2024) por senioridade e porte: % que procuraria outro emprego com 100% presencial e % que prefere o modelo presencial (Wilson 95%).

Etapa: D7
Objetivo: Survey State of Data (2021-2024) por senioridade e porte: % que procuraria outro emprego com 100% presencial e % que prefere o modelo presencial (Wilson 95%).
Entradas: data/kaggle/<ano>/*.csv
Saídas: resultados/D7.md, figuras em outputs/figuras_v2/
Como rodar: python -m scripts.d_analise.d7_survey
Tempo aproximado: segundos
"""

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.utilitarios import util_v2 as u

ARQ = {
    2021: "data/kaggle/2021/State of Data 2021 - Dataset - Pgina1.csv",
    2022: "data/kaggle/2022/State_of_data_2022.csv",
    2023: "data/kaggle/2023/State_of_data_BR_2023_Kaggle - df_survey_2023.csv",
    2024: "data/kaggle/2024/Final Dataset - State of Data 2024 - Kaggle - df_survey_2024.csv",
}
# (trecho exato do nome da coluna). Em 2021-2023 o nome é uma tupla "('código ', 'texto')"; em 2024 é um texto único.
MAPA = {
    2021: {
        "senioridade": "'P2_g '",
        "porte": "'P2_c '",
        "ideal": "'P2_r '",
        "atitude": "'P2_s '",
    },
    2022: {
        "senioridade": "'P2_g '",
        "porte": "'P2_c '",
        "ideal": "'P2_q '",
        "atitude": "'P2_r '",
    },
    2023: {
        "senioridade": "'P2_g '",
        "porte": "'P2_c '",
        "ideal": "'P2_s '",
        "atitude": "'P2_t '",
    },
    2024: {
        "senioridade": "2.g_nivel",
        "porte": "2.c_numero_de_funcionarios",
        "ideal": "2.s_modelo_de_trabalho_ideal",
        "atitude": "2.t_atitude_em_caso_de_retorno_presencial",
    },
}
RESTO = {
    "senioridade": "Nivel",
    "porte": "Numero de Funcionarios",
    "ideal": "forma de trabalho ideal",
    "atitude": "100% presencial qual",
}
SENI = ["Júnior", "Pleno", "Sênior"]
EDS = [2021, 2022, 2023, 2024]
COR_ED = {2021: "#56B4E9", 2022: "#009E73", 2023: "#E69F00", 2024: "#D55E00"}


def acha(df, ano, campo):
    chave = MAPA[ano][campo]
    c = [
        x
        for x in df.columns
        if chave in str(x) and (ano == 2024 or RESTO[campo].lower() in str(x).lower())
    ]
    return c[0] if len(c) == 1 else None


def norm_cat(s):
    return re.sub(r"(?<=\d)\.(?=\d{3})", "", str(s)).strip()


def chave_porte(c):
    m = re.search(r"\d+", c)
    n = int(m.group()) if m else 10**9
    return (1 if c.lower().startswith("acima") else 0, n)


dados, mapa_md, ausentes = {}, [], []
for ano in EDS:
    d = pd.read_csv(u.ROOT / ARQ[ano], low_memory=False)
    cols = {c: acha(d, ano, c) for c in MAPA[ano]}
    dados[ano] = (d, cols)
    mapa_md.append(f"## {ano} ({len(d)} respostas)\n")
    for campo, col in cols.items():
        if col is None:
            ausentes.append((ano, campo))
            mapa_md.append(f"- **{campo}**: coluna NÃO encontrada.\n")
            continue
        vc = d[col].dropna().astype(str).value_counts()
        mapa_md.append(
            f"- **{campo}**: `{col}` ({int(vc.sum())} respostas válidas). Categorias originais: "
            + "; ".join(f"«{k}» ({v})" for k, v in vc.items())
            + "\n"
        )


def pega(ano, campo):
    d, cols = dados[ano]
    return d[cols[campo]] if cols[campo] is not None else pd.Series(dtype=object)


def tabela(grupo):
    """Linhas: edição x categoria do grupo (senioridade ou porte); calcula % procuraria e % prefere 100% presencial."""
    linhas = []
    for ano in EDS:
        g, a, i = pega(ano, grupo), pega(ano, "atitude"), pega(ano, "ideal")
        if len(g) == 0 or len(a) == 0 or len(i) == 0:
            continue
        df = pd.DataFrame(
            {
                "g": g.map(lambda x: norm_cat(x) if pd.notna(x) else np.nan),
                "a": a,
                "i": i,
            }
        )
        cats = (
            SENI
            if grupo == "senioridade"
            else sorted([c for c in df.g.dropna().unique()], key=chave_porte)
        )
        for cat in cats + ["Total"]:
            dd = df if cat == "Total" else df[df.g == cat]
            da, di = dd.dropna(subset=["a"]), dd.dropna(subset=["i"])
            ka = int(da.a.astype(str).str.startswith("Vou procurar").sum())
            na = len(da)
            ki = int((di.i.astype(str).str.strip() == "Modelo 100% presencial").sum())
            ni = len(di)
            linhas.append(dict(ano=ano, cat=cat, na=na, ka=ka, ni=ni, ki=ki))
    return pd.DataFrame(linhas)


def cel(k, n):
    return "-" if n == 0 else u.pct_ic(k, n) + (" (amostra pequena)" if n < 30 else "")


def figura(t, cats, nome, largura_rot=False):
    fig, axs = plt.subplots(2, 1, figsize=(u.LARGURA_POL, 6.4), sharex=True)
    w = 0.2
    x = np.arange(len(cats))
    for ax, (k, n, rot) in zip(
        axs,
        [
            ("ka", "na", "Procuraria outro emprego (%)"),
            ("ki", "ni", "Prefere o modelo 100% presencial (%)"),
        ],
    ):
        for j, ano in enumerate(EDS):
            vals, lo, hi = [], [], []
            for c in cats:
                r = t[(t.ano == ano) & (t.cat == c)]
                if len(r) == 0 or r.iloc[0][n] == 0:
                    vals.append(np.nan)
                    lo.append(np.nan)
                    hi.append(np.nan)
                    continue
                p, l, h = u.wilson(r.iloc[0][k], r.iloc[0][n])
                vals.append(p * 100)
                lo.append(l * 100)
                hi.append(h * 100)
            vals, lo, hi = np.array(vals), np.array(lo), np.array(hi)
            ax.bar(
                x + (j - 1.5) * w,
                vals,
                w,
                color=COR_ED[ano],
                label=str(ano),
                yerr=[vals - lo, hi - vals],
                capsize=2,
                error_kw={"lw": 0.8},
            )
        ax.set_ylabel(rot)
        ax.spines[["top", "right"]].set_visible(False)
        if k == "ka":
            ax.set_ylim(0, 105)
    axs[0].legend(
        frameon=False, ncol=4, loc="upper left", title="Edição", title_fontsize=8
    )
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(
        cats,
        rotation=30 if len(cats) > 4 else 0,
        ha="right" if len(cats) > 4 else "center",
    )
    fig.tight_layout()
    u.salva(fig, nome)


ts, tp = tabela("senioridade"), tabela("porte")
figura(ts, SENI + ["Total"], "fig_survey_senioridade.png")
catsp = sorted(
    [
        c
        for c in tp.cat.unique()
        if c != "Total" and tp[(tp.cat == c)][["na", "ni"]].max().max() >= 30
    ],
    key=chave_porte,
)  # figura: só categorias com n >= 30 em alguma edição
cats_pequenas = sorted(
    [c for c in tp.cat.unique() if c != "Total" and c not in catsp], key=chave_porte
)
figura(tp, catsp + ["Total"], "fig_survey_porte.png")

# ---- arquivos ----
M = [
    "# D7: mapeamento de colunas do State of Data (a validar)",
    "",
    "**Mapeamento a validar.** Os nomes abaixo são os das colunas lidas nos CSVs de `data/kaggle/` (DataHackers, Kaggle). Em 2021-2023 o nome da coluna é uma tupla `('código', 'texto')`; em 2024 é um texto único. Os códigos mudam a cada edição. Categorias originais com o n de respostas válidas.",
    "",
    "Definições usadas nos cálculos:",
    '- **Senioridade**: coluna "Nivel" (Júnior, Pleno, Sênior). Só quem trabalha na área e respondeu esse campo entra.',
    '- **Porte da empresa**: "Numero de Funcionarios", categorias originais (normalizadas só na formatação dos números: `3.000` vira `3000`).',
    '- **Prefere 100% presencial**: resposta "Qual a forma de trabalho ideal para você?" igual a "Modelo 100% presencial".',
    '- **Procuraria outro emprego**: resposta a "Caso sua empresa decida pelo modelo 100% presencial qual será sua atitude?" que começa com "Vou procurar outra oportunidade" (as duas opções: no modelo híbrido ou remoto, ou 100% remoto); a outra opção é "Vou aceitar e retornar ao modelo 100% presencial".',
    "",
] + mapa_md
if ausentes:
    M += ["## Perguntas ausentes", ""] + [f"- {a}: {c}" for a, c in ausentes]
else:
    M += [
        "## Perguntas ausentes",
        "",
        "Nenhuma: as quatro colunas existem nas quatro edições (2021 a 2024).",
    ]
open(u.RES / "D7_mapa_colunas.md", "w", encoding="utf-8").write("\n".join(M) + "\n")


def md(t, rot):
    out = [
        "| Edição | "
        + rot
        + " | n (atitude) | Procuraria outro emprego [IC 95%] | n (ideal) | Prefere 100% presencial [IC 95%] |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in t.itertuples():
        out.append(
            f"| {r.ano} | {r.cat} | {u.num(r.na)} | {cel(r.ka, r.na)} | {u.num(r.ni)} | {cel(r.ki, r.ni)} |"
        )
    return out


tot = ts[ts.cat == "Total"]
L = (
    [
        "# D7: survey por senioridade e porte da empresa",
        "",
        "## O que foi feito",
        'Cálculo, por edição do State of Data (2021 a 2024), da % que procuraria outro emprego se a empresa exigisse o modelo 100% presencial e da % que prefere o modelo 100% presencial, por senioridade e por porte da empresa, com intervalo de Wilson de 95%. **O mapeamento de colunas está marcado como "a validar"** (ver `resultados/D7_mapa_colunas.md`). Script `d7_survey.py`; figuras `outputs/figuras_v2/fig_survey_senioridade.png` e `fig_survey_porte.png` (extra; sem título, 15,9 cm, 300 dpi).',
        "",
        "## 1. Por senioridade",
        "",
    ]
    + md(ts, "Senioridade")
    + [
        "",
        "Figura: `fig_survey_senioridade.png` (painel superior: procuraria outro emprego; inferior: prefere 100% presencial).",
        "",
        "## 2. Por porte da empresa (número de funcionários)",
        "",
    ]
    + md(tp, "Porte")
    + [
        "",
        "Figura extra: `fig_survey_porte.png`.",
        "",
        "## Números principais",
        "- Procuraria outro emprego (total, `Nivel` respondido): "
        + ", ".join(f"{r.ano} {u.pct_ic(r.ka, r.na)}" for r in tot.itertuples())
        + ".",
        "- Prefere 100% presencial (total): "
        + ", ".join(f"{r.ano} {u.pct_ic(r.ki, r.ni)}" for r in tot.itertuples())
        + ".",
        "",
        "## Problemas e decisões",
        "- Os totais por edição (procuraria outro emprego: 63,8%, 74,1%, 74,8% e 75,0%) coincidem com os de `survey_triangulation.py` (resultados/triangulacao_survey.md), o que confirma o mapeamento das perguntas de atitude e de modelo ideal nas quatro edições.",
        '- Os dois grupos de interesse são perguntas diferentes: a de atitude só tem 3 respostas (duas "procurar outra oportunidade" e "aceitar").',
        '- Categorias de porte com n < 30 estão marcadas como "amostra pequena" na tabela. A figura de porte omite categorias que nunca chegam a n = 30 em nenhuma edição: '
        + (", ".join(cats_pequenas) if cats_pequenas else "nenhuma")
        + '. Em 2023 aparece uma categoria "de 501 a 100" (n muito pequeno), provável erro de digitação na própria pesquisa; foi mantida na tabela e não foi reclassificada.',
        "- Não fiz teste de hipótese entre grupos (não foi pedido); os intervalos de Wilson permitem leitura visual.",
        "- A amostra do State of Data é de profissionais de dados que responderam voluntariamente à pesquisa; não representa a força de trabalho brasileira.",
    ]
)
open(u.RES / "D7.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("D7 OK", len(ts), len(tp), "ausentes:", ausentes)
