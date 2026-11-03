"""D4: Sensibilidade do limiar semântico (0,25 / 0,30 / 0,35) usando a similaridade em cache.

Etapa: D4
Objetivo: Sensibilidade do limiar semântico (0,25 / 0,30 / 0,35) usando a similaridade em cache.
Entradas: data/processed/relevance_embed_cache.parquet, posts_raw.parquet, dados/youtube_c3, dados/bluesky
Saídas: resultados/D4.md, qualitativo/d4_bordas.xlsx
Como rodar: python -m scripts.d_analise.d4_limiar
Tempo aproximado: poucos minutos
"""

import html
import re
import unicodedata

import numpy as np
import pandas as pd

from config import LANGUAGES
from scripts.utilitarios import util_v2 as u
from src.preprocess.cleaning import filter_relevant, preprocess_dataframe

LIMS = [0.25, 0.30, 0.35]
sim = (
    pd.read_parquet(u.ROOT / "data/processed/relevance_embed_cache.parquet")
    .set_index("id")
    .rto_similarity
)
norm = lambda s: (
    re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(s))).strip().lower()
)

# --- textos que passam do filtro léxico em cada fonte (feito uma vez) ---
v0raw = pd.read_parquet(u.ROOT / "data/processed/posts_raw.parquet")
yt = pd.read_parquet(u.ROOT / "dados/youtube_c3/youtube_viewcount.parquet")
yt["text"] = yt.text.map(html.unescape)
bs = pd.read_parquet(u.ROOT / "dados/bluesky/bluesky_bruto.parquet")
bs["text"] = bs.text.map(html.unescape)
lex = {}
for nome, d in (("v0", v0raw), ("youtube_c3", yt), ("bluesky", bs)):
    l, _ = filter_relevant(d)
    assert l.id.isin(sim.index).all(), f"{nome}: há textos sem similaridade em cache"
    l = l[
        [
            "id",
            "platform",
            "kind",
            "text",
            "lang_hint",
            "source",
            "context_text",
            "parent_id",
        ]
    ].copy()
    l["sim"] = l.id.map(sim).values
    l["fonte_coleta"] = nome
    lex[nome] = l


def monta(th):
    partes = []
    for nome, l in lex.items():
        d = l[l.sim >= th - 1e-12].reset_index(drop=True)
        d = preprocess_dataframe(
            d, do_stem=False
        )  # duplicatas de texto e vazios (como no pipeline)
        d = d[d["lang"].isin(LANGUAGES)]
        partes.append(
            d.drop(columns=["tokens", "tokens_str", "clean_text"], errors="ignore")
        )
    v0, y, b = partes
    for d in partes:
        d["_k"] = d.text.map(norm)
    kv0 = set(v0._k)
    y = y[~y._k.isin(kv0)]
    b = b[~b._k.isin(kv0 | set(y._k))]
    return pd.concat([v0, y, b], ignore_index=True).drop(columns="_k")


corpus = u.carrega(
    ["id", "platform", "fonte_coleta", "sentiment_label", "rto_similarity"]
)
res = {}
for th in LIMS:
    res[th] = monta(th)
    print(th, len(res[th]), res[th].platform.value_counts().to_dict(), flush=True)

# --- validação: o 0,30 reconstruído deve igualar o corpus v2 ---
r30 = set(res[0.30].id)
c = set(corpus.id)
validacao = (len(r30), len(c), len(r30 - c), len(c - r30))

# --- tabela de n por limiar e plataforma ---
PL = ["bluesky", "reddit", "youtube"]
tabn = pd.DataFrame(
    {th: res[th].platform.value_counts().reindex(PL, fill_value=0) for th in LIMS}
)
tabn.loc["Total"] = tabn.sum()
extra_25 = res[0.25][~res[0.25].id.isin(res[0.30].id)]
extra_35 = res[0.30][~res[0.30].id.isin(res[0.35].id)]

# --- sentimento: 0,30 (corpus v2) x 0,35 (subconjunto do v2 com similaridade >= 0,35) ---
s35 = corpus[corpus.rto_similarity >= 0.35 - 1e-12]


def pcts(d):
    out = {}
    for g, dd in list(d.groupby("platform")) + [("total", d)]:
        n = len(dd)
        out[g] = (
            n,
            {
                k: int((dd.sentiment_label == k).sum())
                for k in ["negativo", "neutro", "positivo"]
            },
        )
    return out


p30, p35 = pcts(corpus), pcts(s35)
linhas_s = []
for g in PL + ["total"]:
    for rot, p in (("0,30", p30), ("0,35", p35)):
        n, cts = p[g]
        linhas_s.append(
            [u.NOME[g], rot, u.num(n)]
            + [u.pct_ic(cts[k], n) for k in ["negativo", "neutro", "positivo"]]
        )
t_chi = {}
for g in PL + ["total"]:
    t = np.array(
        [
            [p30[g][1][k] - p35[g][1][k] for k in ["negativo", "neutro", "positivo"]],
            [p35[g][1][k] for k in ["negativo", "neutro", "positivo"]],
        ]
    )
    t_chi[g] = u.qui2(t)
neg = lambda p, g: p[g][1]["negativo"] / p[g][0] * 100
maior_neg = {
    th: all(
        p[g][1]["negativo"] >= max(p[g][1]["neutro"], p[g][1]["positivo"])
        for g in PL + ["total"]
    )
    for th, p in (("0,30", p30), ("0,35", p35))
}

# --- leitura qualitativa: 20 textos entre 0,25 e 0,30 e 20 entre 0,30 e 0,35 (semente 42) ---
borda_inf = extra_25.sample(20, random_state=42).assign(
    faixa="0,25 a 0,30 (cortado no limiar 0,30)"
)
borda_sup = extra_35.sample(20, random_state=42).assign(
    faixa="0,30 a 0,35 (mantido em 0,30, cortado em 0,35)"
)
bordas = pd.concat([borda_inf, borda_sup])[["faixa", "sim", "platform", "text"]].rename(
    columns={"sim": "similaridade", "platform": "plataforma", "text": "texto"}
)
bordas["similaridade"] = bordas.similaridade.round(3)
(u.ROOT / "qualitativo").mkdir(exist_ok=True)
bordas.to_excel(u.ROOT / "qualitativo/d4_bordas.xlsx", index=False)

# --- relatório ---
L = [
    "# D4: sensibilidade do limiar semântico",
    "",
    "## O que foi feito",
    "Reconstruí o corpus com o filtro semântico em 0,25, 0,30 e 0,35 usando a similaridade já calculada (cache `data/processed/relevance_embed_cache.parquet`), sem carregar nenhum modelo. Para cada fonte: filtro léxico e exclusões, corte de similaridade, duplicatas de texto e vazios, idioma `pt` e remoção de duplicatas entre fontes (mantendo o v0), na mesma ordem do D1. Script `d4_limiar.py`.",
    "",
    f"**Validação:** o limiar 0,30 reconstruído tem {u.num(validacao[0])} linhas contra {u.num(validacao[1])} do corpus v2; {validacao[2]} linhas só na reconstrução e {validacao[3]} só no corpus v2.",
    "",
    "## 1. n final por limiar e plataforma",
    "",
]
L += u.md_tabela(
    ["Plataforma"] + [f"limiar {t:.2f}".replace(".", ",") for t in LIMS],
    [[u.NOME.get(i, i)] + [u.num(tabn.loc[i, t]) for t in LIMS] for i in tabn.index],
)
L += [
    "",
    f"- Linhas extras do limiar 0,25 em relação ao 0,30: **{u.num(len(extra_25))}** ({', '.join(f'{u.NOME[k]} {u.num(v)}' for k, v in extra_25.platform.value_counts().reindex(PL, fill_value=0).items())}). O sentimento delas **não foi calculado** (conforme o pedido).",
    f"- Linhas que o limiar 0,35 tira em relação ao 0,30: {u.num(len(extra_35))} ({', '.join(f'{u.NOME[k]} {u.num(v)}' for k, v in extra_35.platform.value_counts().reindex(PL, fill_value=0).items())}).",
    "",
    "## 2. Sentimento: 0,30 contra 0,35 (o 0,35 é subconjunto do 0,30)",
    "",
]
L += u.md_tabela(
    [
        "Plataforma",
        "Limiar",
        "n",
        "negativo [IC 95%]",
        "neutro [IC 95%]",
        "positivo [IC 95%]",
    ],
    linhas_s,
)
L += [
    "",
    "Diferença entre as linhas mantidas e as retiradas pelo 0,35 (qui-quadrado entre o grupo retirado e o mantido; as amostras são complementares, não independentes do corte):",
    "",
]
L += u.md_tabela(
    ["Plataforma", "χ²", "p", "V de Cramér"],
    [
        [
            u.NOME[g],
            f"{t_chi[g][0]:.1f}".replace(".", ","),
            u.fmt_p(t_chi[g][2]),
            f"{t_chi[g][3]:.3f}".replace(".", ","),
        ]
        for g in PL + ["total"]
    ],
)
L += [
    "",
    "## 3. A conclusão muda?",
    "",
    f"- % negativo no total: 0,30 = {neg(p30, 'total'):.1f}%; 0,35 = {neg(p35, 'total'):.1f}%".replace(
        ".", ","
    )
    + ".",
    "- Em todas as plataformas e no total, o negativo continua sendo a classe mais frequente nos dois limiares: "
    + (
        "**sim**"
        if all(maior_neg.values())
        else "**não, em pelo menos um caso (ver tabela)**"
    )
    + ".",
    '- A conclusão "predominância de sentimento negativo" '
    + (
        "**não muda** entre 0,30 e 0,35."
        if all(maior_neg.values())
        else "**muda**; ver tabela."
    )
    + " Para o 0,25 não há sentimento calculado, então a conclusão sobre ele fica não testada.",
    "",
    "## 4. Leitura qualitativa",
    "",
    "`qualitativo/d4_bordas.xlsx`: 20 textos sorteados (semente 42) com similaridade entre 0,25 e 0,30 e 20 entre 0,30 e 0,35; colunas `faixa`, `similaridade`, `plataforma`, `texto`; sem `author_hash`. A leitura de relevância (os textos da faixa baixa são off-topic?) é sua; não fiz julgamento automático.",
    "",
    "## Problemas e decisões",
    "- Não foi preciso recalcular embeddings: o cache cobre 100% dos textos que passam do filtro léxico (v0, YouTube C3 e Bluesky).",
    "- Só o filtro semântico varia entre os limiares; o filtro léxico e as exclusões são fixos.",
    "- Sentimento do limiar 0,25 não foi calculado; só o n das linhas extras.",
    "- Nenhum modelo transformer ou de embeddings foi carregado (o D5 roda em paralelo).",
]
open(u.RES / "D4.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("D4 OK", validacao, {t: len(res[t]) for t in LIMS}, maior_neg)
