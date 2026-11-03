"""B10b (passo 2): Relevância com variantes de embeddings (tfidf, tfidf+sim, embeddings, embeddings+tfidf); regressão logística balanceada, C por validação cruzada, teste no gabarito de 150, bootstrap (semente 42).

Etapa: B10b (passo 2)
Objetivo: Relevância com variantes de embeddings (tfidf, tfidf+sim, embeddings, embeddings+tfidf); regressão logística balanceada, C por validação cruzada, teste no gabarito de 150, bootstrap (semente 42). Aplica ao corpus só se o kappa for >= 0,60.
Entradas: rotulagem/, data/processed/embeddings_minilm*, dados/v2/corpus_v2.parquet
Saídas: anexo a resultados/B10.md; dados/v2/corpus_v2_classificado.parquet (condicional)
Como rodar: python -m scripts.b_validacao.b10b_relevancia
Tempo aproximado: poucos minutos
"""

import warnings

from scripts.utilitarios import util_v2 as u

warnings.filterwarnings("ignore")
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

R = u.ROOT / "rotulagem"
PROC = u.ROOT / "data/processed"
SEED = 42
IRR = "irrelevante"
C_GRID = [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]
vc = lambda x, d=2: (
    "-"
    if x is None or (isinstance(x, float) and np.isnan(x))
    else f"{x:.{d}f}".replace(".", ",")
)
pcn = lambda x: "-" if np.isnan(x) else f"{x * 100:.1f}%".replace(".", ",")

# ---------------- embeddings ----------------
E = np.load(PROC / "embeddings_minilm.npy")
idx = pd.read_parquet(PROC / "embeddings_minilm_index.parquet")
corp = pd.read_parquet(u.CORPUS)
assert E.shape[0] == len(corp) and (idx.id.values == corp.id.values).all(), (
    "índice de embeddings não bate com o corpus"
)
pos = dict(zip(idx.id, idx.pos))
ECOLS = [f"e{i}" for i in range(E.shape[1])]

# ---------------- dados (mesma montagem do B10) ----------------
gab = pd.read_csv(R / "gabarito_humano_150.csv")
mapa = pd.read_parquet(R / "mapa_ids.parquet")[["id_rotulagem", "id"]]
mar = pd.read_excel(R / "planilha_rotulada_marcos.xlsx")[
    ["id_rotulagem", "sentimento_humano"]
].rename(columns={"sentimento_humano": "sentimento"})
hum = (
    mar[~mar.id_rotulagem.isin(gab.id_rotulagem)]
    .merge(mapa, on="id_rotulagem")
    .assign(origem="humano")
)
pesos = pd.read_parquet(R / "b9/b9_mapa_pesos.parquet")
l9 = (
    pd.concat(
        [pd.read_excel(R / f"b9/rotulos_llm_b9_parte{k}.xlsx") for k in (1, 2, 3, 4)]
    )[["id_rotulagem", "sentimento"]]
    .merge(pesos[["id_rotulagem", "id"]], on="id_rotulagem")
    .assign(origem="llm")
)
l8 = (
    pd.read_excel(R / "rotulos_llm_b8.xlsx")[["id_rotulagem", "sentimento"]]
    .merge(
        pd.read_parquet(R / "b8_mapa_ids.parquet")[["id_rotulagem", "id"]],
        on="id_rotulagem",
    )
    .assign(origem="llm")
)
treino = pd.concat([l9, l8, hum], ignore_index=True)
teste = gab[["id_rotulagem", "sentimento"]].merge(mapa, on="id_rotulagem")
assert (
    len(set(treino.id) & set(teste.id)) == 0
    and treino.id.is_unique
    and len(treino) == 2500
    and len(teste) == 150
)
base = corp.set_index("id")


def matriz(df):
    X = pd.DataFrame(E[[pos[i] for i in df.id]], columns=ECOLS)
    X.insert(0, "rto_similarity", df.id.map(base.rto_similarity).values)
    X.insert(0, "text", df.id.map(base.text).values)
    return X


for d in (treino, teste):
    d["relevante"] = (d.sentimento != IRR).astype(int)
Xtr, Xte = matriz(treino), matriz(teste)
ytr, yte = treino.relevante.values, teste.relevante.values


def pipe(variante, C):
    tf = (
        "t",
        TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            strip_accents="unicode",
            max_features=100000,
        ),
        "text",
    )
    em = ("e", "passthrough", ECOLS)
    sim = ("s", StandardScaler(), ["rto_similarity"])
    partes = {
        "tfidf": [tf],
        "tfidf+sim": [tf, sim],
        "embeddings": [em],
        "embeddings+tfidf": [em, tf],
    }[variante]
    return Pipeline(
        [
            ("f", ColumnTransformer(partes)),
            ("m", LogisticRegression(class_weight="balanced", C=C, max_iter=5000)),
        ]
    )


cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
VARS = ["tfidf", "tfidf+sim", "embeddings", "embeddings+tfidf"]
ajustes, tempos = {}, {}
for v in VARS:
    t0 = time.time()
    gs = GridSearchCV(
        pipe(v, 1.0), {"m__C": C_GRID}, scoring="f1_macro", cv=cv, n_jobs=1, refit=True
    ).fit(Xtr, ytr)
    i = gs.best_index_
    ajustes[v] = (
        gs.best_estimator_,
        gs.best_params_["m__C"],
        gs.cv_results_["mean_test_score"][i],
        gs.cv_results_["std_test_score"][i],
    )
    tempos[v] = time.time() - t0
    print(
        v,
        "C",
        ajustes[v][1],
        "CV",
        round(ajustes[v][2], 3),
        f"{tempos[v]:.0f}s",
        flush=True,
    )
melhor = max(
    VARS, key=lambda v: ajustes[v][2]
)  # melhor na validação cruzada, não no teste

# ---------------- teste ----------------
IDX = np.random.default_rng(SEED).integers(0, len(teste), size=(1000, len(teste)))


def um(y, p):
    pr, rc, f, _ = precision_recall_fscore_support(
        y, p, labels=[1], average=None, zero_division=0
    )
    k = cohen_kappa_score(y, p) if len(set(y) | set(p)) > 1 else np.nan
    return (y == p).mean(), float(pr[0]), float(rc[0]), float(f[0]), float(k)


def metricas(y, p):
    pt = um(y, p)
    bs = np.array([um(y[i], p[i]) for i in IDX])
    ci = lambda j: (np.nanpercentile(bs[:, j], 2.5), np.nanpercentile(bs[:, j], 97.5))
    return dict(
        acc=pt[0],
        prec=pt[1],
        rec=pt[2],
        f1=pt[3],
        kappa=pt[4],
        a_ci=ci(0),
        f_ci=ci(3),
        k_ci=ci(4),
    )


res = {v: metricas(yte, ajustes[v][0].predict(Xte)) for v in VARS}
r1 = pd.read_excel(R / "rotulos_llm_chat_r1.xlsx")[
    ["id_rotulagem", "sentimento"]
].merge(teste[["id_rotulagem"]], on="id_rotulagem")
res_r1 = metricas(
    yte,
    (teste[["id_rotulagem"]].merge(r1, on="id_rotulagem", how="left").sentimento != IRR)
    .astype(int)
    .values,
)
kappa_melhor = res[melhor]["kappa"]
aplica = bool(kappa_melhor >= 0.60)

# ---------------- aplicação ao corpus ----------------
saida = None
if aplica:
    Xc = matriz(corp[["id"]])
    prob = ajustes[melhor][0].predict_proba(Xc)[:, 1]
    saida = corp.copy()
    saida["prob_relevante"] = prob
    saida["relevante_clf"] = prob >= 0.5
    saida["variante_relevancia"] = melhor
    saida["no_treino"] = saida.id.isin(set(treino.id))
    saida["no_teste"] = saida.id.isin(set(teste.id))
    saida.to_parquet(u.ROOT / "dados/v2/corpus_v2_classificado.parquet", index=False)

# ---------------- relatório (acrescentado ao B10.md) ----------------
info = pd.read_json(PROC / "embeddings_minilm_info.json", typ="series")
L = [
    "",
    "## B10b: relevância com embeddings (acrescentado em 29/10/2026)",
    "",
    "Os vetores MiniLM (`paraphrase-multilingual-MiniLM-L12-v2`, o mesmo modelo do filtro semântico) foram recalculados para os 49.269 textos do corpus v2 e salvos em `data/processed/embeddings_minilm.npy` (float32, 384 dimensões, normalizados) com índice em `data/processed/embeddings_minilm_index.parquet` (`id`, `pos`). "
    f"Conferência: a similaridade com o centroide do tema recalculada a partir dos vetores difere da do cache em no máximo {info['max_abs_dif_sim']:.1e} (média {info['media_abs_dif_sim']:.1e}), ou seja, são os mesmos vetores. Tempo do cálculo: {info['tempo_s'] / 60:.0f} min. Script `b10b_embeddings.py`; classificação em `b10b_relevancia.py`.",
    "",
    "Mesmo procedimento do B10: regressão logística balanceada, C escolhido por validação cruzada estratificada de 5 dobras só no treino (F1 macro; grade de C ampliada até 1000, necessária para os embeddings), teste só no gabarito de 150, bootstrap de 1.000 reamostragens (semente 42). Só a relevância foi refeita; o posicionamento não.",
    "",
    "### Validação cruzada (só no treino, n = 2.500)",
    "",
]
L += u.md_tabela(
    ["Variante", "C escolhido", "F1 macro (CV 5 dobras): média ± desvio", "Tempo (s)"],
    [
        [
            v,
            ajustes[v][1],
            f"{vc(ajustes[v][2], 3)} ± {vc(ajustes[v][3], 3)}",
            f"{tempos[v]:.0f}",
        ]
        for v in VARS
    ],
)
L += [
    "",
    f"Melhor na validação cruzada: **{melhor}**.",
    "",
    "### Teste (gabarito humano de 150; classe positiva = relevante)",
    "",
]
fl = lambda n, m: [
    n,
    f"{pcn(m['acc'])} [{pcn(m['a_ci'][0])}-{pcn(m['a_ci'][1])}]",
    pcn(m["prec"]),
    pcn(m["rec"]),
    f"{vc(m['f1'])} [{vc(m['f_ci'][0])}-{vc(m['f_ci'][1])}]",
    f"{vc(m['kappa'])} [{vc(m['k_ci'][0])}-{vc(m['k_ci'][1])}]",
]
L += u.md_tabela(
    ["Variante", "Acurácia [IC]", "Precisão", "Revocação", "F1 [IC]", "Kappa [IC]"],
    [fl(v + (" (escolhida)" if v == melhor else ""), res[v]) for v in VARS]
    + [fl("LLM r1 (referência)", res_r1)],
)
L += [
    "",
    f"- Kappa da variante escolhida no teste: **{vc(kappa_melhor)}** (limiar 0,60) → "
    + (
        "**aplicada** aos 49.269 textos; `dados/v2/corpus_v2_classificado.parquet` tem as colunas `relevante_clf`, `prob_relevante`, `variante_relevancia`, `no_treino` e `no_teste` além das do corpus. O arquivo tem ids de plataforma: não vai para o repositório público."
        if aplica
        else "**não aplicada**: abaixo de 0,60, então `corpus_v2_classificado.parquet` não foi gerado."
    ),
    "- A escolha da variante foi feita só pela validação cruzada; o teste não entrou na escolha.",
    "- O posicionamento não foi refeito (decisão do Marcos); os resultados de posicionamento do TCC vêm da estimativa por amostra da seção 5 deste relatório.",
]
if saida is not None:
    L += [
        f"- Relevantes pelo classificador: {u.num(int(saida.relevante_clf.sum()))} de {u.num(len(saida))} ({pcn(saida.relevante_clf.mean())}); por plataforma: "
        + "; ".join(
            f"{u.NOME[p]} {pcn(saida[saida.platform == p].relevante_clf.mean())}"
            for p in ["bluesky", "reddit", "youtube"]
        )
        + ". Compare com a estimativa por amostra do LLM (seção 5): 39,8% de relevantes no total (60,2% irrelevantes).",
        f"- As {u.num(int(saida.no_treino.sum()))} linhas que estavam no treino têm previsão dentro da amostra; as 150 do teste não.",
    ]
p = u.RES / "B10.md"
atual = p.read_text(encoding="utf-8")
marca = "## B10b: relevância com embeddings"
if marca in atual:
    atual = atual[: atual.index(marca)].rstrip() + "\n"
p.write_text(atual + "\n".join(L) + "\n", encoding="utf-8")
print(
    "B10b OK | melhor:",
    melhor,
    "| kappa teste:",
    round(kappa_melhor, 3),
    "| aplica:",
    aplica,
)
for v in VARS:
    print(v, round(res[v]["acc"], 3), round(res[v]["kappa"], 3))
