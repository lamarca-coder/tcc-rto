"""B5: Métricas do BERT (sentimento), da heurística (alvo, posicionamento) e dos LLM r1/r2 contra o gabarito humano (150 de consenso; 500 do Marcos).

Etapa: B5
Objetivo: Métricas do BERT (sentimento), da heurística (alvo, posicionamento) e dos LLM r1/r2 contra o gabarito humano (150 de consenso; 500 do Marcos).
Entradas: arquivos de rotulagem/, dados/v2/corpus_v2.parquet
Saídas: resultados/B5.md, B5_metricas.csv
Como rodar: python -m scripts.b_validacao.b5_metricas
Tempo aproximado: menos de 1 minuto
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from scripts.utilitarios import util_v2 as u

R = u.ROOT / "rotulagem"
COLS = {
    "sentimento": ["positivo", "negativo", "neutro"],
    "alvo": ["presencial", "remoto", "hibrido", "geral"],
    "posicionamento": ["pro_RTO", "anti_RTO", "neutro_misto"],
}
ROT = {"sentimento": "Sentimento", "alvo": "Alvo", "posicionamento": "Posicionamento"}
IRR = "irrelevante"
MET = {
    "modelo": "Pipeline (BERT / heurística)",
    "r1": "LLM r1 (Sonnet 5.5)",
    "r2": "LLM r2 (Opus 5.5)",
}

# ---------------- entradas ----------------
gab = pd.read_csv(R / "gabarito_humano_150.csv").rename(
    columns={
        "sentimento": "sentimento_h",
        "alvo": "alvo_h",
        "posicionamento": "posicionamento_h",
    }
)
mod = pd.read_parquet(R / "gabarito_modelo.parquet")
mar = pd.read_excel(R / "planilha_rotulada_marcos.xlsx").rename(
    columns={
        "sentimento_humano": "sentimento_h",
        "alvo_humano": "alvo_h",
        "posicionamento_humano": "posicionamento_h",
    }
)
llm = {r: pd.read_excel(R / f"rotulos_llm_chat_r{r}.xlsx") for r in (1, 2)}
mapa = pd.read_parquet(R / "mapa_ids.parquet")[["id_rotulagem", "amostra", "platform"]]


# ---------------- mapeamento do modelo para os códigos do livro ----------------
def mapeia_stance(s):
    s = str(s)
    if s.startswith("anti"):
        return "anti_RTO"
    if s.startswith("pr"):
        return "pro_RTO"  # "pró-RTO"
    return "neutro_misto"  # "neutro" e "indefinido"


MAP_ALVO = {
    "presencial": "presencial",
    "remoto": "remoto",
    "hibrido": "hibrido",
    "geral": "geral",
}
pipe = pd.DataFrame(
    {
        "id_rotulagem": mod.id_rotulagem,
        "sentimento_modelo": mod.sentiment_label.astype(str),
        "alvo_modelo": mod.target.map(MAP_ALVO),
        "posicionamento_modelo": mod.stance.map(mapeia_stance),
    }
)
assert pipe.alvo_modelo.notna().all() and set(pipe.sentimento_modelo) <= set(
    COLS["sentimento"]
)

for r, d in llm.items():
    pipe = pipe.merge(
        d[["id_rotulagem", "sentimento", "alvo", "posicionamento"]].rename(
            columns={
                "sentimento": f"sentimento_r{r}",
                "alvo": f"alvo_r{r}",
                "posicionamento": f"posicionamento_r{r}",
            }
        ),
        on="id_rotulagem",
        how="left",
    )


def junta(ref):
    d = ref.merge(pipe, on="id_rotulagem", how="left").merge(
        mapa, on="id_rotulagem", how="left"
    )
    assert (
        d.sentimento_modelo.notna().all()
        and d.sentimento_r1.notna().all()
        and d.sentimento_r2.notna().all()
    )
    return d


G, M = (
    junta(gab[["id_rotulagem", "sentimento_h", "alvo_h", "posicionamento_h"]]),
    junta(mar[["id_rotulagem", "sentimento_h", "alvo_h", "posicionamento_h"]]),
)


# ---------------- métricas ----------------
def bootstrap_idx(n, B=1000, seed=42):
    return np.random.default_rng(seed).integers(0, n, size=(B, n))


def f1_macro(y, p, labels):
    _, _, f, _ = precision_recall_fscore_support(
        y, p, labels=labels, average=None, zero_division=0
    )
    return float(np.mean(f))


def calcula(y, p, labels, idx):
    y, p = np.asarray(y), np.asarray(p)
    pr, rc, f1, sup = precision_recall_fscore_support(
        y, p, labels=labels, average=None, zero_division=0
    )
    acc = float((y == p).mean())
    accs = [(y[i] == p[i]).mean() for i in idx]
    f1s = [f1_macro(y[i], p[i], labels) for i in idx]
    kap = (
        float(cohen_kappa_score(y, p)) if len(set(y)) > 1 or len(set(p)) > 1 else np.nan
    )
    return {
        "n": len(y),
        "acuracia": acc,
        "ac_lo": float(np.percentile(accs, 2.5)),
        "ac_hi": float(np.percentile(accs, 97.5)),
        "kappa": kap,
        "f1_macro": float(np.mean(f1)),
        "f_lo": float(np.percentile(f1s, 2.5)),
        "f_hi": float(np.percentile(f1s, 97.5)),
        "classes": {
            l: (float(pr[k]), float(rc[k]), float(f1[k]), int(sup[k]))
            for k, l in enumerate(labels)
        },
    }


def roda(df, ref_nome, tratamento):
    """tratamento 'a': só relevantes; 'b': todas (irrelevante vira uma classe da referência que o modelo erra)."""
    out = []
    for col, labs in COLS.items():
        d = df[df[f"{col}_h"] != IRR] if tratamento == "a" else df
        labels = labs if tratamento == "a" else labs + [IRR]
        idx = bootstrap_idx(len(d))
        for m in ("modelo", "r1", "r2"):
            res = calcula(d[f"{col}_h"], d[f"{col}_{m}"], labels, idx)
            out.append(
                {
                    "referencia": ref_nome,
                    "tratamento": tratamento,
                    "coluna": col,
                    "metodo": m,
                    **{k: v for k, v in res.items() if k != "classes"},
                    "_classes": res["classes"],
                }
            )
    return out


resultados = (
    roda(G, "gabarito_150", "a")
    + roda(G, "gabarito_150", "b")
    + roda(M, "marcos_500", "a")
    + roda(M, "marcos_500", "b")
)
tab = pd.DataFrame(resultados)
tab.drop(columns="_classes").round(4).to_csv(u.RES / "B5_metricas.csv", index=False)

# ---------------- % de irrelevantes ----------------
nI_g, nI_m = int((G.sentimento_h == IRR).sum()), int((M.sentimento_h == IRR).sum())
irr = {"gab": (nI_g, len(G)), "mar": (nI_m, len(M))}
for nome, d in (("llm r1", "r1"), ("llm r2", "r2")):
    irr[nome] = (int((G[f"sentimento_{d}"] == IRR).sum()), len(G))
irrM = {d: (int((M[f"sentimento_{d}"] == IRR).sum()), len(M)) for d in ("r1", "r2")}
irr_amostra = {
    a: (
        int(((M.amostra == a) & (M.sentimento_h == IRR)).sum()),
        int((M.amostra == a).sum()),
    )
    for a in ("A", "B")
}


def cel(r):
    return (
        f"{r.acuracia * 100:.1f}% [{r.ac_lo * 100:.1f}-{r.ac_hi * 100:.1f}] · κ {r.kappa:.2f} · F1 {r.f1_macro:.2f} [{r.f_lo:.2f}-{r.f_hi:.2f}]"
    ).replace(".", ",")


def tabela_principal(ref, trat):
    t = tab[(tab.referencia == ref) & (tab.tratamento == trat)]
    linhas = []
    for col in COLS:
        tt = t[t.coluna == col]
        n = int(tt.n.iloc[0])
        linhas.append(
            [ROT[col], u.num(n)]
            + [cel(tt[tt.metodo == m].iloc[0]) for m in ("modelo", "r1", "r2")]
        )
    return u.md_tabela(["Coluna", "n", MET["modelo"], MET["r1"], MET["r2"]], linhas)


def por_classe(ref, trat, col):
    t = tab[(tab.referencia == ref) & (tab.tratamento == trat) & (tab.coluna == col)]
    labs = COLS[col] + ([IRR] if trat == "b" else [])
    linhas = []
    for c in labs:
        linha = [c, u.num(t.iloc[0]["_classes"][c][3])]
        for m in ("modelo", "r1", "r2"):
            p, rc, f, _ = t[t.metodo == m].iloc[0]["_classes"][c]
            linha.append(f"{p:.2f} / {rc:.2f} / {f:.2f}".replace(".", ","))
        linhas.append(linha)
    return u.md_tabela(
        [
            "Classe",
            "n (ref.)",
            "Pipeline P / R / F1",
            "LLM r1 P / R / F1",
            "LLM r2 P / R / F1",
        ],
        linhas,
    )


# ---------------- matrizes de confusão (gabarito 150, tratamento b) ----------------
ABREV = {
    "irrelevante": "irrel.",
    "presencial": "presenc.",
    "hibrido": "híbrido",
    "neutro_misto": "neutro/misto",
    "pro_RTO": "pró-RTO",
    "anti_RTO": "anti-RTO",
}
for col, labs in COLS.items():
    ls = labs + [IRR]
    fig, axs = plt.subplots(
        1, 3, figsize=(u.LARGURA_POL, 2.9 if len(ls) <= 4 else 3.2), sharey=True
    )
    for ax, m in zip(axs, ("modelo", "r1", "r2")):
        cm = confusion_matrix(G[f"{col}_h"], G[f"{col}_{m}"], labels=ls)
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(ls)):
            for j in range(len(ls)):
                ax.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if norm[i, j] > 0.55 else "black",
                )
        ax.set_xticks(range(len(ls)))
        ax.set_xticklabels(
            [ABREV.get(x, x) for x in ls], rotation=45, ha="right", fontsize=8
        )
        ax.set_yticks(range(len(ls)))
        ax.set_yticklabels([ABREV.get(x, x) for x in ls], fontsize=8)
        ax.set_xlabel(
            {"modelo": "Pipeline", "r1": "LLM r1", "r2": "LLM r2"}[m], fontsize=8
        )
    axs[0].set_ylabel("Gabarito humano", fontsize=8)
    fig.tight_layout()
    u.salva(fig, f"fig_b5_confusao_{col}.png")


# ---------------- análise de erros da heurística ----------------
def erros_heuristica(df):
    rel = df[df.posicionamento_h != IRR]
    anti = rel[rel.posicionamento_h == "anti_RTO"]
    h = anti.posicionamento_modelo.value_counts().reindex(
        ["pro_RTO", "anti_RTO", "neutro_misto"], fill_value=0
    )
    pro_err = anti[anti.posicionamento_modelo == "pro_RTO"]
    return {
        "n_rel": len(rel),
        "n_anti": len(anti),
        "dist": h.to_dict(),
        "n_pro": len(pro_err),
        "neg_bert": int((pro_err.sentimento_modelo == "negativo").sum()),
        "neg_humano": int((pro_err.sentimento_h == "negativo").sum()),
        "alvo_remoto_humano": int((pro_err.alvo_h == "remoto").sum()),
        "alvo_remoto_heur": int((pro_err.alvo_modelo == "remoto").sum()),
        "pred_pro_total": int((rel.posicionamento_modelo == "pro_RTO").sum()),
        "pred_pro_gold_anti": int(
            (
                (rel.posicionamento_modelo == "pro_RTO")
                & (rel.posicionamento_h == "anti_RTO")
            ).sum()
        ),
        "pred_pro_gold_pro": int(
            (
                (rel.posicionamento_modelo == "pro_RTO")
                & (rel.posicionamento_h == "pro_RTO")
            ).sum()
        ),
    }


eg, em = erros_heuristica(G), erros_heuristica(M)
em_A, em_B = (
    erros_heuristica(M[M.amostra == "A"]),
    erros_heuristica(M[M.amostra == "B"]),
)
cruz = pd.crosstab(
    G[G.posicionamento_h != IRR].posicionamento_h,
    G[G.posicionamento_h != IRR].posicionamento_modelo,
).reindex(index=COLS["posicionamento"], columns=COLS["posicionamento"], fill_value=0)


def pc(a, b):
    return f"{a} de {b} ({a / b * 100:.1f}%)".replace(".", ",") if b else "0 de 0"


def erro_md(nome, e):
    return [
        f"**{nome}**: entre as {e['n_rel']} linhas relevantes, {e['n_anti']} são anti_RTO no gabarito. A heurística marcou {e['dist']['pro_RTO']} como pro_RTO ({pc(e['dist']['pro_RTO'], e['n_anti'])}), {e['dist']['anti_RTO']} como anti_RTO e {e['dist']['neutro_misto']} como neutro/misto. "
        f"Nos {e['n_pro']} erros pró: sentimento do BERT negativo em {pc(e['neg_bert'], e['n_pro'])}; sentimento humano negativo em {pc(e['neg_humano'], e['n_pro'])}; alvo remoto pela heurística em {pc(e['alvo_remoto_heur'], e['n_pro'])} (humano: {pc(e['alvo_remoto_humano'], e['n_pro'])}). "
        f"Das {e['pred_pro_total']} linhas que a heurística chamou de pro_RTO, {e['pred_pro_gold_anti']} são anti_RTO e {e['pred_pro_gold_pro']} são pro_RTO no gabarito."
    ]


# ---------------- relatório ----------------
L = (
    [
        "# B5: métricas do modelo contra o gabarito humano",
        "",
        "## O que foi feito",
        "Comparação dos rótulos do pipeline (BERT para sentimento; heurística de alvo e posicionamento) e dos dois anotadores LLM (r1 e r2, `resultados/B6.md`) contra (1) o **gabarito de consenso das 150 linhas** (`rotulagem/gabarito_humano_150.csv`) e (2) as **500 linhas do Marcos** (`planilha_rotulada_marcos.xlsx`; **um só anotador**, sem verificação de concordância fora das 150). Script `b5_metricas.py`; tabela completa com todas as métricas em `resultados/B5_metricas.csv`; figuras `outputs/figuras_v2/fig_b5_confusao_*.png`.",
        "",
        "Em cada célula: acurácia [IC 95% bootstrap] · kappa de Cohen · F1 macro [IC 95% bootstrap]. Bootstrap com 1.000 reamostragens das linhas (semente 42, mesmos índices para todos os métodos).",
        "",
        "## 1. Mapeamento dos rótulos do modelo para os códigos do livro",
        "",
        "| Coluna | Rótulo no pipeline | Código do livro |",
        "|---|---|---|",
        "| Sentimento | positivo, negativo, neutro | positivo, negativo, neutro (BERT `pysentimiento`; igual) |",
        "| Posicionamento | anti-RTO | anti_RTO |",
        "| Posicionamento | pró-RTO | pro_RTO |",
        "| Posicionamento | neutro e indefinido | neutro_misto |",
        "| Alvo | presencial, remoto, hibrido, geral | presencial, remoto, hibrido, geral (a heurística não produziu `geral` nas 500 linhas) |",
        "| Todas | (não existe) | irrelevante: **o pipeline nunca prevê**; cada linha irrelevante é erro no tratamento (b) |",
        "",
        "## 2. Irrelevantes",
        f"- Gabarito de 150: **{pc(*irr['gab'])}** irrelevantes ({u.pct_ic(*irr['gab'])} Wilson). Os dois LLM marcam: r1 {pc(*irr['llm r1'])}, r2 {pc(*irr['llm r2'])}.",
        f"- Marcos, 500 linhas: **{pc(*irr['mar'])}** irrelevantes; r1 {pc(*irrM['r1'])}, r2 {pc(*irrM['r2'])}. Por amostra: A (400, balanceada por sentimento do modelo) {pc(*irr_amostra['A'])}; B (100, só alvo remoto e sentimento negativo pelo modelo) {pc(*irr_amostra['B'])}.",
        "- O pipeline só filtra por palavras-chave e similaridade; o gabarito mostra que metade do corpus filtrado ainda é irrelevante para a pergunta de pesquisa.",
        "",
        "## 3. Tabela principal, gabarito de 150 linhas (consenso)",
        "",
        "### (a) Só as linhas relevantes para o gabarito (n = 74)",
        "",
    ]
    + tabela_principal("gabarito_150", "a")
    + [
        "",
        "### (b) Todas as 150 linhas, irrelevante contado como erro do modelo (classe `irrelevante` entra no F1 macro)",
        "",
    ]
    + tabela_principal("gabarito_150", "b")
    + [
        "",
        "## 4. Detalhe por classe (precisão / revocação / F1), gabarito de 150, tratamento (a)",
        "",
    ]
)
for col in COLS:
    L += [f"**{ROT[col]}**", ""] + por_classe("gabarito_150", "a", col) + [""]
L += [
    "Tratamento (b), classe `irrelevante` (revocação do pipeline = 0 por construção):",
    "",
]
for col in COLS:
    t = tab[
        (tab.referencia == "gabarito_150")
        & (tab.tratamento == "b")
        & (tab.coluna == col)
    ]
    L.append(
        f"- {ROT[col]}: "
        + "; ".join(
            f"{MET[m]} P/R/F1 = "
            + " / ".join(
                f"{x:.2f}" for x in t[t.metodo == m].iloc[0]["_classes"][IRR][:3]
            ).replace(".", ",")
            for m in ("modelo", "r1", "r2")
        )
    )
L += (
    [
        "",
        "Matrizes de confusão (tratamento b, contagens; cor = proporção da linha): `fig_b5_confusao_sentimento.png`, `fig_b5_confusao_alvo.png`, `fig_b5_confusao_posicionamento.png`.",
        "",
        "## 5. Contra as 500 linhas do Marcos (um só anotador)",
        "",
        "**Aviso:** a referência é um único anotador, sem consenso. As 500 linhas não são uma amostra aleatória do corpus: 400 vêm de uma amostra balanceada pelo sentimento do próprio modelo (67/67/66 por plataforma) e 100 foram escolhidas por serem alvo remoto e sentimento negativo pelo modelo (por construção, todas pró-RTO na heurística). As métricas descrevem essa amostra, não o corpus.",
        "",
        f"### (a) Só as linhas relevantes para o Marcos (n = {500 - nI_m})",
        "",
    ]
    + tabela_principal("marcos_500", "a")
    + ["", "### (b) Todas as 500 linhas, irrelevante contado como erro do modelo", ""]
    + tabela_principal("marcos_500", "b")
    + [
        "",
        "## 6. Erros da heurística de posicionamento",
        "",
        "Matriz do gabarito de 150 (relevantes; linhas = gabarito, colunas = heurística):",
        "",
    ]
    + u.md_tabela(
        ["Gabarito \\ Heurística"] + COLS["posicionamento"],
        [
            [i] + [cruz.loc[i, c] for c in COLS["posicionamento"]]
            for i in COLS["posicionamento"]
        ],
    )
    + [""]
    + erro_md("Gabarito de 150 (consenso)", eg)
    + [""]
    + erro_md("Marcos, 500 linhas", em)
    + [""]
    + erro_md("Marcos, só amostra A (400, não enviesada para alvo remoto)", em_A)
    + [""]
    + erro_md("Marcos, só amostra B (100, remoto e negativo pelo modelo)", em_B)
    + [
        "",
        "Leitura: a heurística classifica sentimento negativo sobre o alvo remoto como pró-RTO. Os números acima dizem quantos dos anti_RTO do gabarito caem nessa regra e se o sentimento de fato era negativo.",
        "",
        "## Problemas e decisões",
        "- O gabarito de 150 tem n pequeno (74 relevantes): os intervalos de bootstrap são largos e diferenças de poucos pontos entre métodos não são distinguíveis. Em classes com poucos casos (pro_RTO, hibrido) precisão e revocação são instáveis.",
        '- Tratamento (a) só avalia as linhas relevantes; um LLM que marca "irrelevante" numa linha relevante conta como erro (a classe fica fora do conjunto de classes de referência, então baixa a revocação). No tratamento (b) o LLM pode acertar "irrelevante".',
        "- F1 macro: média simples sobre as classes da referência (3 ou 4 no tratamento a; 4 ou 5 no b, incluindo `irrelevante`, com F1 = 0 para o pipeline). Kappa calculado sobre todos os rótulos presentes.",
        "- O pipeline compara o rótulo BERT de sentimento (positivo/negativo/neutro) com a coluna `sentimento` do gabarito. O alvo e o posicionamento são da heurística, que não é um modelo treinado.",
        "- O LLM r1 e r2 foram rodados no chat do claude.ai sem temperatura fixa (`resultados/B6.md`); r1 e r2 diferem entre si.",
        "- Não usei a Carolina separadamente: o gabarito já é o consenso das duas anotadoras (`resultados/B4.md`).",
        "- Nenhum modelo foi carregado; só leitura de rótulos já calculados (o D5 continua rodando).",
    ]
)
open(u.RES / "B5.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("B5 OK")
for r in resultados:
    if r["referencia"] == "gabarito_150":
        print(
            r["tratamento"],
            r["coluna"],
            r["metodo"],
            r["n"],
            round(r["acuracia"], 3),
            round(r["kappa"], 2),
            round(r["f1_macro"], 3),
        )
print(eg, "\n", em)
