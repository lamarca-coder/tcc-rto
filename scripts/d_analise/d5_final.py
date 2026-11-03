"""D5 (passos 5 a 8): Modelo final de tópicos (k = 9, semente 42), distribuição por plataforma e fase, excertos para a análise qualitativa.

Etapa: D5 (passos 5 a 8)
Objetivo: Modelo final de tópicos (k = 9, semente 42), distribuição por plataforma e fase, excertos para a análise qualitativa.
Entradas: dados/v2/corpus_v2.parquet, dados/v2/d5_topicos_k9_s42.json
Saídas: dados/v2/topicos_v2.parquet, qualitativo/excertos.xlsx, resultados/D5.md
Como rodar: python -m scripts.d_analise.d5_final
Tempo aproximado: poucos minutos
"""

import json
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from scripts.utilitarios import util_v2 as u

K, SEED, NW = 9, 42, 15
FASES = ["até 2019", "2020-2021", "jan/2022 a mai/2023", "jun/2023 em diante"]


def fase(d):
    if d.year <= 2019:
        return FASES[0]
    if d.year <= 2021:
        return FASES[1]
    if (d.year, d.month) <= (2023, 5):
        return FASES[2]
    return FASES[3]


c = pd.read_parquet(
    u.CORPUS, columns=["id", "platform", "kind", "text", "created_utc", "tokens_str"]
)
c["data"] = pd.to_datetime(c.created_utc, utc=True, format="mixed", errors="coerce")
assert c.data.notna().all()
c["fase"] = c.data.map(fase)
assert (c.tokens_str.fillna("").str.strip() != "").all()

# ---- passo 5: modelo final ----
t0 = time.time()
vec = CountVectorizer(
    min_df=2, max_df=0.9, max_features=5000, token_pattern=r"(?u)\b\w\w+\b"
)
dtm = vec.fit_transform(c.tokens_str)
lda = LatentDirichletAllocation(
    n_components=K, learning_method="batch", max_iter=25, random_state=SEED
).fit(dtm)
P = lda.transform(dtm)  # probabilidade de cada tópico por texto (linhas somam 1)
tempo = time.time() - t0
vocab = np.array(vec.get_feature_names_out())
palavras = [vocab[comp.argsort()[::-1][:NW]].tolist() for comp in lda.components_]
# confere com o modelo da semente 42 já usado no D5_topicos.md
ref = json.load(
    open(u.ROOT / f"dados/v2/d5_topicos_k{K}_s{SEED}.json", encoding="utf-8")
)
mesmos = [set(a[:10]) == set(b[:10]) for a, b in zip(palavras, ref)]
topicos = pd.DataFrame(
    {"id": c.id, "topico_dominante": P.argmax(axis=1), "prob_dominante": P.max(axis=1)}
)
for t in range(K):
    topicos[f"topico_{t}"] = P[:, t]
topicos.to_parquet(u.ROOT / "dados/v2/topicos_v2.parquet", index=False)

# ---- passo 6: distribuição por plataforma e por fase ----
c["topico"] = topicos.topico_dominante.values


def dist(col, ordem):
    n = pd.crosstab(c.topico, c[col]).reindex(
        index=range(K), columns=ordem, fill_value=0
    )
    p = n / n.sum() * 100
    return n, p


n_pl, p_pl = dist("platform", ["bluesky", "reddit", "youtube"])
n_fa, p_fa = dist("fase", FASES)
tot = c.topico.value_counts().reindex(range(K), fill_value=0)

# ---- passo 7: excertos (C6a) ----
linhas, aviso = [], []
for t in range(K):
    pt = pd.Series(P[:, t], index=c.index)
    top = pt.sort_values(ascending=False).head(4)
    pool = pt[(pt > 0.5) & ~pt.index.isin(top.index)]
    if len(pool) >= 4:
        sort_ = pool.sample(n=4, random_state=SEED)
        crit_s = "sorteio"
    else:
        sort_ = pool
        crit_s = "sorteio"
        resto = (
            pt[~pt.index.isin(top.index) & ~pt.index.isin(pool.index)]
            .sort_values(ascending=False)
            .head(4 - len(pool))
        )
        aviso.append((t, int((pt > 0.5).sum()), len(pool), len(resto)))
    sel = (
        [(i, "top") for i in top.index]
        + [(i, crit_s) for i in sort_.index]
        + ([(i, "seguinte_por_prob") for i in resto.index] if len(pool) < 4 else [])
    )
    for i, crit in sel:
        linhas.append(
            {
                "topico": t,
                "criterio": crit,
                "prob": round(float(pt[i]), 4),
                "plataforma": c.at[i, "platform"],
                "data": c.at[i, "data"].strftime("%Y-%m-%d"),
                "texto": c.at[i, "text"],
            }
        )
ex = pd.DataFrame(linhas)
assert len(ex) == K * 8
(u.ROOT / "qualitativo").mkdir(exist_ok=True)
ex.to_excel(u.ROOT / "qualitativo/excertos.xlsx", index=False)
n_acima = {t: int((P[:, t] > 0.5).sum()) for t in range(K)}

# ---- passo 8: relatório ----
coh = pd.read_csv(u.RES / "D5_coerencia.csv")
pc = lambda x: f"{x:.1f}%".replace(".", ",")
L = [
    "# D5: escolha do k, tópicos finais e excertos",
    "",
    "## O que foi feito",
    'LDA (scikit-learn) sobre `tokens_str` do corpus v2 (49.269 textos), com o mesmo pré-processamento e os mesmos parâmetros do v0 (`CountVectorizer(min_df=2, max_df=0.9, max_features=5000)`, `LatentDirichletAllocation(learning_method="batch", max_iter=25)`); só muda o número de tópicos. Scripts `d5_lda_coerencia.py`, `d5_passo4.py` e `d5_final.py`.',
    "",
    "## 1. Escolha do k (passos 1 a 4)",
    "Coerência c_v (gensim, 10 palavras por tópico) para k = 4 a 12, com 3 sementes (42, 43, 44): média e desvio padrão em `resultados/D5_coerencia.csv`; figura `outputs/d5_coerencia.png`; palavras dos 3 melhores k em `resultados/D5_topicos.md`.",
    "",
]
L += u.md_tabela(
    ["k", "média c_v", "desvio padrão", "42", "43", "44"],
    [
        [
            int(r.k),
            f"{r.media_cv:.4f}".replace(".", ","),
            f"{r.dp_cv:.4f}".replace(".", ","),
            f"{r.cv_semente_42:.4f}".replace(".", ","),
            f"{r.cv_semente_43:.4f}".replace(".", ","),
            f"{r.cv_semente_44:.4f}".replace(".", ","),
        ]
        for r in coh.itertuples()
    ],
)
L += [
    "",
    "**Decisão (Marcos, 28/10/2026): k = 9.** Os três maiores c_v médios são k = 11 (0,5110), k = 9 (0,5106) e k = 10 (0,4987). A diferença entre 11 e 9 é de 0,0004, muito menor que o desvio entre sementes (0,015 a 0,025): empate técnico. Pela regra do plano (em caso de empate, o menor k), fica **k = 9**.",
    "",
    "## 2. Modelo final (k = 9, semente 42)",
    f"Treinado com os {u.num(len(c))} textos (vocabulário de {u.num(dtm.shape[1])} termos, tempo {tempo:.0f} s). As 10 primeiras palavras de cada tópico coincidem com as do modelo da semente 42 do passo 4 em {sum(mesmos)} dos {K} tópicos. Probabilidades por texto em `dados/v2/topicos_v2.parquet` (`id`, `topico_dominante`, `prob_dominante`, `topico_0` a `topico_8`; numeração a partir de 0, como em `D5_topicos.md`; contém o `id` do corpus e é proibido no repositório público).",
    "",
    "As 15 palavras de maior peso de cada tópico (para nomear os temas):",
    "",
]
L += [
    f"- **Tópico {t}** ({u.num(tot[t])} textos como dominante, {pc(tot[t] / len(c) * 100)}): "
    + ", ".join(palavras[t])
    for t in range(K)
]
L += [
    "",
    "## 3. Distribuição de tópicos (tópico dominante)",
    "",
    "### Por plataforma (n e % da plataforma)",
    "",
]
L += u.md_tabela(
    ["Tópico", "Bluesky", "Reddit", "YouTube"],
    [
        [t]
        + [
            f"{u.num(n_pl.loc[t, p])} ({pc(p_pl.loc[t, p])})"
            for p in ["bluesky", "reddit", "youtube"]
        ]
        for t in range(K)
    ],
)
L += ["", "### Por fase (n e % da fase)", ""]
L += u.md_tabela(
    ["Tópico"] + FASES,
    [
        [t] + [f"{u.num(n_fa.loc[t, f])} ({pc(p_fa.loc[t, f])})" for f in FASES]
        for t in range(K)
    ],
)
L += [
    "",
    "Total por fase: "
    + ", ".join(f"{f} = {u.num(n_fa[f].sum())}" for f in FASES)
    + ". A última fase é quase inteira Bluesky (80% dos textos), então a distribuição por fase mistura mudança de assunto com troca de plataforma.",
    "",
    "## 4. Excertos para a análise qualitativa (C6a)",
    "",
    f"`qualitativo/excertos.xlsx`: {len(ex)} excertos ({K} tópicos x 8): os 4 de maior probabilidade do tópico e 4 sorteados (semente 42) entre os de probabilidade acima de 0,5, sem repetir os 4 primeiros. Colunas `topico`, `criterio`, `prob`, `plataforma`, `data`, `texto`; sem `author_hash` e sem id do corpus. Contém textos de usuários: não versionar.",
    "",
    "Textos com probabilidade acima de 0,5 em cada tópico: "
    + ", ".join(f"{t}: {u.num(n_acima[t])}" for t in range(K))
    + ".",
]
L += (
    [
        "",
        "**Tópicos com menos de 4 textos disponíveis acima de 0,5 além dos 4 primeiros** (completados com os seguintes por probabilidade, criterio `seguinte_por_prob`): "
        + "; ".join(
            f"tópico {t} ({a} acima de 0,5; {b} no sorteio; {r} completados)"
            for t, a, b, r in aviso
        )
        + ".",
    ]
    if aviso
    else [
        "",
        "Todos os tópicos tinham pelo menos 4 textos acima de 0,5 além dos 4 primeiros; nenhum precisou ser completado.",
    ]
)
L += [
    "",
    "## Problemas e decisões",
    "- O universo dos excertos é o corpus v2 inteiro: o corpus ainda contém textos irrelevantes (cerca de 60% pela estimativa do B10), então alguns excertos podem ser irrelevantes para a pergunta de pesquisa. O tópico só mede coocorrência de palavras.",
    "- Os tópicos usam texto sem stemming (`--no-stem`), como no v0. A coerência c_v é sensível à semente (desvio de até 0,03), por isso a escolha do k por empate técnico.",
    "- O D5 original rodou em 2 execuções: o processo morreu quando a sessão anterior terminou (k = 6, semente 43) e foi retomado a partir do arquivo de coerência, sem perder modelos (`outputs/d5.log` e `d5_retomada.log`).",
    "- Nomear os 9 temas e a codificação temática dos excertos ficam com o Chat e com você.",
]
open(u.RES / "D5.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("D5 FINAL OK", f"{tempo:.0f}s", "mesmos:", sum(mesmos), "avisos:", aviso)
print(tot.to_dict())
