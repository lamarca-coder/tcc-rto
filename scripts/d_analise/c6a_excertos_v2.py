"""C6a v2: excertos para a análise qualitativa, só de textos validados (relevantes segundo o LLM no B9).

Etapa: C6a (v2)
Objetivo: Por tópico dominante (k = 9), até 8 excertos entre os textos do B9 que o LLM marcou como relevantes: os 4 de maior probabilidade do tópico e até 4 sorteados (semente 42). Exclui idioma diferente de pt (inclui falhas do detector), quase duplicatas (mantém um por grupo), spam e conteúdo sexual (marcas do D1b).
Entradas: dados/v2/corpus_v2.parquet, dados/v2/topicos_v2.parquet, dados/v2/d1b_flags.parquet, rotulagem/b9/ (rótulos e pesos)
Saídas: qualitativo/excertos_v2.xlsx (não versionado), resultados/C6a_v2.md (só contagens)
Como rodar: python -m scripts.d_analise.c6a_excertos_v2
Tempo aproximado: segundos
"""

import html

import numpy as np
import pandas as pd

from scripts.utilitarios import util_v2 as u

R = u.ROOT / "rotulagem"
SEED = 42
IRR = "irrelevante"
K = 9

pesos = pd.read_parquet(R / "b9/b9_mapa_pesos.parquet")
l9 = pd.concat(
    [pd.read_excel(R / f"b9/rotulos_llm_b9_parte{k}.xlsx") for k in (1, 2, 3, 4)]
)[["id_rotulagem", "sentimento", "posicionamento"]]
corp = pd.read_parquet(u.CORPUS, columns=["id", "text", "created_utc"])
top = pd.read_parquet(
    u.ROOT / "dados/v2/topicos_v2.parquet",
    columns=["id", "topico_dominante", "prob_dominante"],
)
fl = pd.read_parquet(u.ROOT / "dados/v2/d1b_flags.parquet")[
    ["id", "lang_det", "grupo_dup", "dup_qualquer", "spam_qualquer"]
]
d = (
    l9.merge(pesos[["id_rotulagem", "id", "plataforma"]], on="id_rotulagem")
    .merge(corp, on="id")
    .merge(top, on="id")
    .merge(fl, on="id")
)
assert len(d) == 2000
d["data"] = pd.to_datetime(
    d.created_utc, utc=True, format="mixed", errors="coerce"
).dt.strftime("%Y-%m-%d")
d["texto"] = d.text.map(html.unescape)

rel = d[d.sentimento != IRR].copy()
n_rel = rel.groupby("topico_dominante").size()
motivos = {}
rel["excl_idioma"] = rel.lang_det != "pt"
rel["excl_spam"] = rel.spam_qualquer  # inclui conteúdo adulto/sexual
ok = rel[~rel.excl_idioma & ~rel.excl_spam].copy()
# quase duplicatas: um por grupo (o de maior probabilidade do tópico); fora de grupo, todos ficam
ok = ok.sort_values("prob_dominante", ascending=False)
sem_dup = ok[~ok.dup_qualquer | ~ok.duplicated("grupo_dup", keep="first")]

rng = np.random.default_rng(SEED)
saida = []
linhas = []
for t in range(K):
    c = sem_dup[sem_dup.topico_dominante == t].sort_values(
        "prob_dominante", ascending=False
    )
    a = c.head(4).assign(criterio="maior_probabilidade")
    resto = c.iloc[4:]
    n_sort = min(4, len(resto))
    b = (
        resto.iloc[rng.choice(len(resto), n_sort, replace=False)].assign(
            criterio="sorteio"
        )
        if n_sort
        else resto.iloc[0:0]
    )
    saida += [a, b]
    r = rel[rel.topico_dominante == t]
    linhas.append(
        (
            t,
            int(n_rel.get(t, 0)),
            int(r.excl_idioma.sum()),
            int(r.excl_spam.sum()),
            len(ok[ok.topico_dominante == t]),
            len(c),
            len(a) + len(b),
        )
    )
ex = pd.concat(saida)
ex = ex.rename(
    columns={
        "topico_dominante": "topico",
        "prob_dominante": "prob",
        "posicionamento": "posicionamento_llm",
        "sentimento": "sentimento_llm",
    }
)[
    [
        "topico",
        "criterio",
        "prob",
        "plataforma",
        "data",
        "posicionamento_llm",
        "sentimento_llm",
        "texto",
    ]
].sort_values(["topico", "criterio", "prob"], ascending=[True, True, False])
ex["prob"] = ex.prob.round(4)
(u.ROOT / "qualitativo").mkdir(exist_ok=True)
ex.to_excel(u.ROOT / "qualitativo/excertos_v2.xlsx", index=False)

L = [
    "# C6a v2: excertos só de textos validados",
    "",
    'Universo: os textos do B9 (2.000, rotulados pelo LLM) que o LLM marcou como relevantes, agrupados pelo tópico dominante do D5 (k = 9). Por tópico, até 8 excertos: os 4 de maior probabilidade do tópico (`maior_probabilidade`) e até 4 sorteados entre os demais (`sorteio`, semente 42). Exclusões (marcas do D1b): idioma diferente de pt (inclui falhas do detector), spam e conteúdo adulto ou sexual (lista de termos, que também pega usos legítimos de "sexo"), e quase duplicatas (mantém o de maior probabilidade por grupo). Script `c6a_excertos_v2.py`; planilha `qualitativo/excertos_v2.xlsx` (não versionada, contém textos).',
    "",
    "| Tópico | Relevantes no B9 | Excluídos por idioma | Excluídos por spam/adulto | Após idioma e spam | Após quase duplicatas (candidatos) | Excertos |",
    "|---|---|---|---|---|---|---|",
]
for t, a1, a2, a3, a4, a5, a6 in linhas:
    L.append(f"| {t} | {a1} | {a2} | {a3} | {a4} | {a5} | {a6} |")
L += [
    f"| **Total** | {sum(x[1] for x in linhas)} | {sum(x[2] for x in linhas)} | {sum(x[3] for x in linhas)} | {sum(x[4] for x in linhas)} | {sum(x[5] for x in linhas)} | {sum(x[6] for x in linhas)} |",
    "",
    "As colunas de exclusão por idioma e por spam podem se sobrepor (um texto pode ter os dois problemas). Se algum tópico ficar com menos de 8 excertos, é porque teve poucos candidatos relevantes e válidos no B9. O tópico 5 tem só 9 candidatos, então o sorteio dele sai de um pool de 5.",
]
(u.RES / "C6a_v2.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("C6a OK", [(x[0], x[6]) for x in linhas])
