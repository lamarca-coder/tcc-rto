"""B10: Classificador local (TF-IDF + regressão logística) treinado com rótulos do LLM e do Marcos; teste só no gabarito de 150.

Etapa: B10
Objetivo: Classificador local (TF-IDF + regressão logística) treinado com rótulos do LLM e do Marcos; teste só no gabarito de 150.
Entradas: rotulagem/, data/processed/relevance_embed_cache.parquet, dados/v2/corpus_v2.parquet
Saídas: resultados/B10.md; dados/v2/corpus_v2_classificado.parquet só se o kappa da relevância for >= 0,60
Como rodar: python -m scripts.b_validacao.b10_classificador
Tempo aproximado: poucos minutos
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.utilitarios import util_v2 as u

R = u.ROOT / "rotulagem"
SEED = 42
IRR = "irrelevante"
EMBEDDINGS_DISPONIVEIS = (
    False  # o cache só tem a similaridade (escalar), não os vetores
)
C_GRID = [0.1, 0.3, 1, 3, 10, 30, 100]
TAREFAS = {
    "posicionamento": ["pro_RTO", "anti_RTO", "neutro_misto"],
    "alvo": ["presencial", "remoto", "hibrido", "geral"],
    "sentimento": ["positivo", "negativo", "neutro"],
}
ROT = {
    "relevancia": "Relevância",
    "sentimento": "Sentimento",
    "alvo": "Alvo",
    "posicionamento": "Posicionamento",
}
vc = lambda x, d=2: (
    "-"
    if x is None or (isinstance(x, float) and np.isnan(x))
    else f"{x:.{d}f}".replace(".", ",")
)
pcn = lambda x: "-" if np.isnan(x) else f"{x * 100:.1f}%".replace(".", ",")

# ---------------- dados ----------------
corp = pd.read_parquet(u.CORPUS)
cache = pd.read_parquet(u.ROOT / "data/processed/relevance_embed_cache.parquet")
cobertura_corpus = float(corp.id.isin(set(cache.id)).mean())
base = corp[["id", "text", "rto_similarity"]].set_index("id")

gab = pd.read_csv(R / "gabarito_humano_150.csv")
mapa = pd.read_parquet(R / "mapa_ids.parquet")[["id_rotulagem", "id"]]
mar = pd.read_excel(R / "planilha_rotulada_marcos.xlsx")[
    ["id_rotulagem", "sentimento_humano", "alvo_humano", "posicionamento_humano"]
].rename(
    columns={
        "sentimento_humano": "sentimento",
        "alvo_humano": "alvo",
        "posicionamento_humano": "posicionamento",
    }
)
hum = (
    mar[~mar.id_rotulagem.isin(gab.id_rotulagem)]
    .merge(mapa, on="id_rotulagem")
    .assign(origem="humano")
)
l9 = pd.concat(
    [pd.read_excel(R / f"b9/rotulos_llm_b9_parte{k}.xlsx") for k in (1, 2, 3, 4)]
)[["id_rotulagem", "alvo", "sentimento", "posicionamento"]]
pesos = pd.read_parquet(R / "b9/b9_mapa_pesos.parquet")
l9 = l9.merge(pesos[["id_rotulagem", "id"]], on="id_rotulagem").assign(origem="llm")
l8 = (
    pd.read_excel(R / "rotulos_llm_b8.xlsx")[
        ["id_rotulagem", "alvo", "sentimento", "posicionamento"]
    ]
    .merge(
        pd.read_parquet(R / "b8_mapa_ids.parquet")[["id_rotulagem", "id"]],
        on="id_rotulagem",
    )
    .assign(origem="llm")
)
treino = pd.concat([l9, l8, hum], ignore_index=True)
teste = gab.merge(mapa, on="id_rotulagem")
for d in (treino, teste):
    assert d.id.isin(base.index).all()
assert len(set(treino.id) & set(teste.id)) == 0, "treino e teste se sobrepõem"
assert treino.id.is_unique
for d in (treino, teste):
    consist = (d.sentimento == IRR) == (d.alvo == IRR)
    consist &= (d.sentimento == IRR) == (d.posicionamento == IRR)
    assert consist.all(), "irrelevante inconsistente entre colunas"
    d["relevante"] = (d.sentimento != IRR).astype(int)
    d["text"] = d.id.map(base.text)
    d["rto_similarity"] = d.id.map(base.rto_similarity)
cob = {
    "treino": float(treino.id.isin(set(cache.id)).mean()),
    "teste": float(teste.id.isin(set(cache.id)).mean()),
    "corpus_v2": cobertura_corpus,
}
n_origem = treino.origem.value_counts().to_dict()


# ---------------- modelos ----------------
def pipe_var(variante, C):
    txt = (
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
    cols = [txt] + (
        [("s", StandardScaler(), ["rto_similarity"])] if variante == "tfidf+sim" else []
    )
    return Pipeline(
        [
            ("f", ColumnTransformer(cols)),
            ("m", LogisticRegression(class_weight="balanced", C=C, max_iter=3000)),
        ]
    )


def treina(df, alvo_col):
    """Para cada variante, escolhe C por validação cruzada estratificada de 5 dobras SÓ no treino (f1 macro); fica com a melhor variante."""
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    X = df[["text", "rto_similarity"]]
    y = df[alvo_col]
    linhas, melhor = [], None
    for var in ["tfidf", "tfidf+sim"]:
        gs = GridSearchCV(
            pipe_var(var, 1.0),
            {"m__C": C_GRID},
            scoring="f1_macro",
            cv=cv,
            n_jobs=1,
            refit=True,
        )
        gs.fit(X, y)
        i = gs.best_index_
        linhas.append(
            (
                var,
                gs.best_params_["m__C"],
                gs.cv_results_["mean_test_score"][i],
                gs.cv_results_["std_test_score"][i],
            )
        )
        if melhor is None or gs.best_score_ > melhor[1].best_score_:
            melhor = (var, gs)
    if EMBEDDINGS_DISPONIVEIS:
        pass
    return melhor[0], melhor[1].best_estimator_, linhas


rel_var, rel_clf, rel_cv = treina(treino, "relevante")
modelos, cv_tab = {}, {"relevancia": rel_cv}
treino_rel = treino[treino.relevante == 1]
for t in TAREFAS:
    v, m, l = treina(treino_rel, t)
    modelos[t] = (v, m)
    cv_tab[t] = l

# ---------------- teste ----------------
pipe = pd.read_parquet(R / "gabarito_modelo.parquet")


def stance(s):
    s = str(s)
    return (
        "anti_RTO"
        if s.startswith("anti")
        else "pro_RTO"
        if s.startswith("pr")
        else "neutro_misto"
    )


pipe = pd.DataFrame(
    {
        "id_rotulagem": pipe.id_rotulagem,
        "sentimento_pipe": pipe.sentiment_label.astype(str),
        "alvo_pipe": pipe.target.astype(str),
        "posicionamento_pipe": pipe.stance.map(stance),
    }
)
r1 = pd.read_excel(R / "rotulos_llm_chat_r1.xlsx")[
    ["id_rotulagem", "alvo", "sentimento", "posicionamento"]
].rename(columns=lambda c: c if c == "id_rotulagem" else f"{c}_r1")
T = teste.merge(pipe, on="id_rotulagem", how="left").merge(
    r1, on="id_rotulagem", how="left"
)
assert T.sentimento_pipe.notna().all() and T.sentimento_r1.notna().all()
Xt = T[["text", "rto_similarity"]]
T["relevante_clf"] = rel_clf.predict(Xt)
T["prob_rel"] = rel_clf.predict_proba(Xt)[:, 1]
for t, (v, m) in modelos.items():
    T[f"{t}_clf"] = m.predict(Xt)
    T[f"{t}_clf_b"] = np.where(
        T.relevante_clf == 1, T[f"{t}_clf"], IRR
    )  # pipeline completo: relevância e depois a classe
T["relevante_r1"] = (T.sentimento_r1 != IRR).astype(int)

IDX = np.random.default_rng(SEED).integers(0, len(T), size=(1000, len(T)))
IDX_REL = None


def f1m(y, p, labels):
    _, _, f, _ = precision_recall_fscore_support(
        y, p, labels=labels, average=None, zero_division=0
    )
    return float(np.mean(f))


def met_classes(y, p, labels, idx):
    y, p = np.asarray(y), np.asarray(p)
    accs = [(y[i] == p[i]).mean() for i in idx]
    fs = [f1m(y[i], p[i], labels) for i in idx]
    kap = cohen_kappa_score(y, p) if len(set(y) | set(p)) > 1 else np.nan
    return dict(
        n=len(y),
        acc=float((y == p).mean()),
        a_lo=np.percentile(accs, 2.5),
        a_hi=np.percentile(accs, 97.5),
        kappa=float(kap),
        f1=f1m(y, p, labels),
        f_lo=np.percentile(fs, 2.5),
        f_hi=np.percentile(fs, 97.5),
    )


def met_binaria(y, p, idx):
    y, p = np.asarray(y), np.asarray(p)

    def um(yy, pp):
        pr, rc, f, _ = precision_recall_fscore_support(
            yy, pp, labels=[1], average=None, zero_division=0
        )
        k = cohen_kappa_score(yy, pp) if len(set(yy) | set(pp)) > 1 else np.nan
        return (yy == pp).mean(), float(pr[0]), float(rc[0]), float(f[0]), float(k)

    ac, pr, rc, f, k = um(y, p)
    bs = np.array([um(y[i], p[i]) for i in idx])
    ci = lambda j: (np.nanpercentile(bs[:, j], 2.5), np.nanpercentile(bs[:, j], 97.5))
    return dict(
        n=len(y),
        acc=ac,
        prec=pr,
        rec=rc,
        f1=f,
        kappa=k,
        a_ci=ci(0),
        f_ci=ci(3),
        k_ci=ci(4),
    )


cel = lambda m: (
    f"{pcn(m['acc'])} [{pcn(m['a_lo'])}-{pcn(m['a_hi'])}] · κ {vc(m['kappa'])} · F1 {vc(m['f1'])} [{vc(m['f_lo'])}-{vc(m['f_hi'])}]"
)
res_rel = {
    "Classificador novo": met_binaria(T.relevante, T.relevante_clf, IDX),
    "LLM r1": met_binaria(T.relevante, T.relevante_r1, IDX),
    "Pipeline atual (não filtra: equivale a tudo relevante)": met_binaria(
        T.relevante, np.ones(len(T), int), IDX
    ),
}
kappa_rel = res_rel["Classificador novo"]["kappa"]

tab_a, tab_b, extra = {}, {}, {}
rel_mask = (T.relevante == 1).values
idx_a = np.random.default_rng(SEED).integers(
    0, int(rel_mask.sum()), size=(1000, int(rel_mask.sum()))
)
for t, labs in TAREFAS.items():
    Tr = T[rel_mask]
    quem = "sentimento_pipe" if t == "sentimento" else f"{t}_pipe"
    tab_a[t] = {
        "Pipeline atual ("
        + ("BERT" if t == "sentimento" else "heurística")
        + ")": met_classes(Tr[t], Tr[quem], labs, idx_a),
        "LLM r1": met_classes(Tr[t], Tr[f"{t}_r1"], labs, idx_a),
        "Classificador novo": met_classes(Tr[t], Tr[f"{t}_clf"], labs, idx_a),
    }
    tab_b[t] = {
        "Pipeline atual ("
        + ("BERT" if t == "sentimento" else "heurística")
        + ")": met_classes(T[t], T[quem], labs + [IRR], IDX),
        "LLM r1": met_classes(T[t], T[f"{t}_r1"], labs + [IRR], IDX),
        "Classificador novo": met_classes(T[t], T[f"{t}_clf_b"], labs + [IRR], IDX),
    }
kappa_pos = tab_a["posicionamento"]["Classificador novo"]["kappa"]
# BERT x classificador para sentimento no mesmo teste (tratamento a)
bert_sent = tab_a["sentimento"]["Pipeline atual (BERT)"]
clf_sent = tab_a["sentimento"]["Classificador novo"]

# ---------------- aplicação ao corpus ----------------
aplica_rel = bool(kappa_rel >= 0.60)
aplica_pos = bool(kappa_pos >= 0.55)
saida = None
if aplica_rel:
    Xc = corp[["text", "rto_similarity"]]
    out = corp.copy()
    out["prob_relevante"] = rel_clf.predict_proba(Xc)[:, 1]
    out["relevante_clf"] = out.prob_relevante >= 0.5
    for t, (v, m) in modelos.items():
        if t == "posicionamento" and not aplica_pos:
            out["posicionamento_clf"] = np.nan
            continue
        pr = m.predict_proba(Xc)
        cls = list(m.classes_)
        lab = np.array(cls)[pr.argmax(axis=1)].astype(object)
        out[f"{t}_clf"] = np.where(out.relevante_clf, lab, np.nan)
        for j, c in enumerate(cls):
            out[f"prob_{t}_{c}"] = pr[:, j]
    out["no_treino"] = out.id.isin(set(treino.id))
    out["no_teste"] = out.id.isin(set(teste.id))
    out.to_parquet(u.ROOT / "dados/v2/corpus_v2_classificado.parquet", index=False)
    saida = out

# ---------------- estimativa por amostra (rótulos do LLM nos 2.000 e pesos) ----------------
est = l9.merge(pesos, on=["id_rotulagem", "id"])
est["irrel"] = (est.sentimento == IRR).astype(int)
est["estrato"] = est.plataforma + "|" + est.fase
CLS = ["anti_RTO", "pro_RTO", "neutro_misto"]
FASES = ["até 2019", "2020-2021", "jan/2022 a mai/2023", "jun/2023 em diante"]


def estima(d, rng_idx=None):
    """% irrelevante e, entre os relevantes, % de cada posicionamento (estimador de razão com os pesos)."""
    w = d.peso.values
    irr = (w * d.irrel.values).sum() / w.sum()
    rel = d.irrel.values == 0
    wr = (w * rel).sum()
    out = [irr] + [
        ((w * rel * (d.posicionamento.values == c)).sum() / wr if wr > 0 else np.nan)
        for c in CLS
    ]
    return np.array(out)


def boot(d, B=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    grupos = {k: g.reset_index(drop=True) for k, g in d.groupby("estrato")}
    vals = []
    for _ in range(B):
        amostras = [
            g.iloc[rng.integers(0, len(g), len(g))] for g in grupos.values()
        ]  # reamostra dentro de cada estrato
        vals.append(estima(pd.concat(amostras)))
    return np.array(vals)


def linha_est(nome, d):
    pt = estima(d)
    b = boot(d)
    cels = [
        f"{pcn(pt[i])} [{pcn(np.nanpercentile(b[:, i], 2.5))}-{pcn(np.nanpercentile(b[:, i], 97.5))}]"
        for i in range(4)
    ]
    return [nome, u.num(len(d)), u.num(round(d.peso.sum()))] + cels


est_linhas = [linha_est("Total", est)]
for p in ["bluesky", "reddit", "youtube"]:
    est_linhas.append(
        linha_est(u.NOME[p] + ", todas as fases", est[est.plataforma == p])
    )
for f in FASES:
    est_linhas.append(linha_est("Todas as plataformas, " + f, est[est.fase == f]))
for p in ["bluesky", "reddit", "youtube"]:
    for f in FASES:
        d = est[(est.plataforma == p) & (est.fase == f)]
        if len(d):
            est_linhas.append(linha_est(f"{u.NOME[p]}, {f}", d))

# ---------------- relatório ----------------
L = [
    "# B10: classificador local treinado com rótulos do LLM",
    "",
    "## O que foi feito",
    f'Classificadores de regressão logística (`class_weight="balanced"`) para relevância, posicionamento, alvo e sentimento, treinados com **{u.num(len(treino))} linhas**: {u.num(n_origem.get("llm", 0))} rotuladas pelo LLM (B9: 2.000; B8: 150) e {n_origem.get("humano", 0)} pelo Marcos (as 350 linhas fora do gabarito de 150). Origem marcada em cada linha. Teste: **só** `rotulagem/gabarito_humano_150.csv` (150 linhas, consenso humano), nunca usado no treino nem na escolha de C ou de variante. Script `b10_classificador.py`.',
    "",
    "Escolha de parâmetros: validação cruzada estratificada de 5 dobras, só no treino, com F1 macro; C em "
    + str(C_GRID)
    + "; o TF-IDF é ajustado dentro de cada dobra. Posicionamento, alvo e sentimento são treinados só nas linhas relevantes.",
    "",
    "## 1. Atributos e cobertura",
    "",
    '- **O cache de embeddings guarda só a similaridade com o tema (`rto_similarity`, um número por texto), não os vetores.** Não há arquivo com os vetores no projeto. Por isso as variantes "embeddings" e "embeddings + TF-IDF" **não foram executadas**: calcular os vetores exigiria carregar de novo o modelo `paraphrase-multilingual-MiniLM-L12-v2` (cerca de 30 minutos para os 49 mil textos), o que o pedido proibia. Fica a decisão de rodar essa comparação.',
    f"- Variantes comparadas: **TF-IDF** (1 e 2 gramas) e **TF-IDF + `rto_similarity`**. A similaridade está em cache para {pcn(cob['treino'])} das linhas de treino, {pcn(cob['teste'])} do teste e {pcn(cob['corpus_v2'])} do corpus v2.",
    "- Nenhum modelo novo foi carregado.",
    "",
    "## 2. Validação cruzada (só no treino)",
    "",
]
for t, l in cv_tab.items():
    n_t = len(treino) if t == "relevancia" else len(treino_rel)
    L += [f"**{ROT[t]}** (n = {u.num(n_t)})", ""] + u.md_tabela(
        ["Variante", "C escolhido", "F1 macro (CV 5 dobras): média ± desvio"],
        [[v, c, f"{vc(m)} ± {vc(s)}"] for v, c, m, s in l],
    )
    escolhida = rel_var if t == "relevancia" else modelos[t][0]
    L += ["", f"Variante escolhida pela validação cruzada: **{escolhida}**.", ""]
dist_t = {t: treino_rel[t].value_counts().to_dict() for t in TAREFAS}
L += [
    "Distribuição no treino: relevância "
    + str(treino.relevante.value_counts().to_dict())
    + "; posicionamento "
    + str(dist_t["posicionamento"])
    + "; alvo "
    + str(dist_t["alvo"])
    + "; sentimento "
    + str(dist_t["sentimento"])
    + ".",
    "",
    "## 3. Avaliação no teste (gabarito humano de 150 linhas)",
    "",
    "Cada célula: acurácia [IC 95% bootstrap] · kappa · F1 macro [IC 95% bootstrap]; 1.000 reamostragens, semente 42.",
    "",
    "### Relevância (binária, classe positiva = relevante; 74 relevantes e 76 irrelevantes no gabarito)",
    "",
]
L += u.md_tabela(
    ["Método", "Acurácia [IC]", "Precisão", "Revocação", "F1 [IC]", "Kappa [IC]"],
    [
        [
            k,
            f"{pcn(m['acc'])} [{pcn(m['a_ci'][0])}-{pcn(m['a_ci'][1])}]",
            pcn(m["prec"]),
            pcn(m["rec"]),
            f"{vc(m['f1'])} [{vc(m['f_ci'][0])}-{vc(m['f_ci'][1])}]",
            f"{vc(m['kappa'])} [{vc(m['k_ci'][0])}-{vc(m['k_ci'][1])}]",
        ]
        for k, m in res_rel.items()
    ],
)
L += ["", "### (a) Só nas 74 linhas relevantes do gabarito", ""]
mets = ["Pipeline atual", "LLM r1", "Classificador novo"]
for tt, titulo in (("a", "a"), ("b", "b")):
    if tt == "b":
        L += [
            "",
            "### (b) Pipeline completo nas 150 linhas: relevância e depois a classe (irrelevante entra como classe; erro do pipeline atual por construção)",
            "",
        ]
    tabela = tab_a if tt == "a" else tab_b
    linhas = []
    for t in TAREFAS:
        d = tabela[t]
        ks = list(d.keys())
        linhas.append(
            [ROT[t], u.num(list(d.values())[0]["n"])] + [cel(d[k]) for k in ks]
        )
    L += u.md_tabela(
        [
            "Coluna",
            "n",
            "Pipeline atual (heurística; BERT no sentimento)",
            "LLM r1",
            "Classificador novo",
        ],
        linhas,
    )
L += [
    "",
    f"Sentimento, BERT contra o classificador no mesmo teste (74 relevantes): BERT {cel(bert_sent)}; classificador {cel(clf_sent)}.",
    "",
    "## 4. Aplicação ao corpus",
    "",
    f"- Kappa da relevância no teste: **{vc(kappa_rel)}** (limiar 0,60) → "
    + (
        "**aplicada** aos 49.269 textos."
        if aplica_rel
        else "**não aplicada**: o kappa ficou abaixo de 0,60, então `corpus_v2_classificado.parquet` não foi gerado."
    ),
    f"- Kappa do posicionamento nas 74 relevantes do teste: **{vc(kappa_pos)}** (limiar 0,55) → "
    + (
        "**aplicado** (`posicionamento_clf` preenchido nas linhas relevantes)."
        if aplica_pos
        else "**não aplicado**: `posicionamento_clf` ficou vazio."
    ),
    "- Alvo e sentimento classificados apenas nas linhas `relevante_clf = True` (sem limiar de kappa no pedido); as probabilidades de todas as classes estão nas colunas `prob_*` para todas as linhas.",
]
if saida is not None:
    L += [
        f"- Arquivo: `dados/v2/corpus_v2_classificado.parquet` ({u.num(len(saida))} linhas; colunas novas `relevante_clf`, `prob_relevante`, `posicionamento_clf`, `alvo_clf`, `sentimento_clf`, `prob_*`, `no_treino`, `no_teste`). `corpus_v2.parquet` não foi alterado. Contém ids de plataforma: não vai para o repositório público.",
        f"- Relevantes pelo classificador: {u.num(int(saida.relevante_clf.sum()))} de {u.num(len(saida))} ({pcn(saida.relevante_clf.mean())}). Linhas que estavam no treino ({u.num(int(saida.no_treino.sum()))}) têm previsão dentro da amostra; as 150 do teste não.",
        "- Distribuição prevista (só relevantes) por plataforma: "
        + "; ".join(
            f"{u.NOME[p]} {pcn(saida[saida.platform == p].relevante_clf.mean())} relevantes"
            for p in ["bluesky", "reddit", "youtube"]
        )
        + ".",
    ]
L += [
    "",
    "## 5. Estimativa por amostra (rótulos do LLM nos 2.000 do B9, com pesos)",
    "",
    'Proporções ponderadas pelo inverso da probabilidade de sorteio (`b9_mapa_pesos.parquet`). Intervalo de 95% por bootstrap estratificado (1.000 reamostragens dentro de cada estrato plataforma x fase, semente 42). "Entre os relevantes" é a razão entre o peso da classe e o peso dos relevantes. **São estimativas sobre os rótulos do LLM, não do humano** (kappa do LLM r1 contra o gabarito humano no posicionamento: 0,74).',
    "",
]
L += u.md_tabela(
    [
        "Grupo",
        "n sorteado",
        "N representado",
        "% irrelevante [IC]",
        "% anti_RTO entre relevantes [IC]",
        "% pro_RTO entre relevantes [IC]",
        "% neutro_misto entre relevantes [IC]",
    ],
    est_linhas,
)
L += [
    "",
    "## Problemas e decisões",
    "- **Embeddings:** ver a seção 1; a comparação com embeddings (e com embeddings + TF-IDF) está pendente da sua decisão de recalcular os vetores. A escolha de variante foi feita entre as duas que existem, pela validação cruzada.",
    "- Os rótulos de treino do LLM não são verdade de campo: o classificador aprende o julgamento do LLM e herda seus erros e a sua definição de relevância (livro de códigos v2/v3). O teste humano de 150 linhas tem n pequeno (74 relevantes) e os intervalos são largos.",
    "- O gabarito de 150 é um subconjunto da amostra A do B2 (balanceada pelo sentimento do BERT); as métricas descrevem essa amostra, não o corpus.",
    "- Posicionamento tem poucos `pro_RTO` no treino; a classe é instável. O LLM no B8 (Bluesky) não marcou nenhum `pro_RTO`.",
    "- A estimativa da seção 5 usa só o B9 (os 2.000 com pesos); os 150 do B8 e as 350 do Marcos não entram porque não têm esse desenho de pesos.",
    '- Reddit "até 2019" (N = 14) e YouTube "até 2019" (N = 132) entram inteiros na amostra: o intervalo desses estratos vem só da reamostragem dentro de n pequeno.',
]
open(u.RES / "B10.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print(
    "B10 OK | kappa relevância:",
    round(kappa_rel, 3),
    "| kappa posicionamento (74):",
    round(kappa_pos, 3),
    "| aplica:",
    aplica_rel,
    aplica_pos,
)
print(
    "cobertura:",
    cob,
    "| treino por origem:",
    n_origem,
    "| variante relevância:",
    rel_var,
    {t: v for t, (v, m) in modelos.items()},
)
