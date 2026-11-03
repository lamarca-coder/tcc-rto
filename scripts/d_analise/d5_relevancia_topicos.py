"""D5 (acréscimo): relevância e posicionamento por tópico dominante (k = 9), com os rótulos do LLM no B9.

Etapa: D5 (acréscimo)
Objetivo: Para cada tópico, % ponderada de textos relevantes e, entre os relevantes, % de anti_RTO, pro_RTO e neutro_misto. Pesos de sorteio do B9 e bootstrap estratificado (plataforma x fase, 1.000 reamostragens, semente 42).
Entradas: dados/v2/topicos_v2.parquet, dados/v2/d5_topicos_k9_s42.json, rotulagem/b9/ (rótulos e pesos)
Saídas: resultados/D5_relevancia_topicos.md, outputs/figuras_v2/fig_topicos_relevancia.png
Como rodar: python -m scripts.d_analise.d5_relevancia_topicos
Tempo aproximado: menos de 1 minuto
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.utilitarios import util_v2 as u

R = u.ROOT / "rotulagem"
SEED = 42
B = 1000
IRR = "irrelevante"
PO = ["anti_RTO", "pro_RTO", "neutro_misto"]
K = 9

pesos = pd.read_parquet(R / "b9/b9_mapa_pesos.parquet")
l9 = pd.concat(
    [pd.read_excel(R / f"b9/rotulos_llm_b9_parte{k}.xlsx") for k in (1, 2, 3, 4)]
)[["id_rotulagem", "sentimento", "posicionamento"]]
top = pd.read_parquet(
    u.ROOT / "dados/v2/topicos_v2.parquet", columns=["id", "topico_dominante"]
)
est = l9.merge(pesos, on="id_rotulagem").merge(top, on="id").reset_index(drop=True)
assert len(est) == 2000
palavras = json.load(open(u.ROOT / "dados/v2/d5_topicos_k9_s42.json", encoding="utf-8"))
rot = {i: f"T{i}: " + ", ".join(palavras[i][:4]) for i in range(K)}

N = len(est)
w = est.peso.values.astype(float)
rel = (est.sentimento != IRR).values.astype(float)
P = np.stack([(est.posicionamento.values == k).astype(float) for k in PO], axis=1)
tp = est.topico_dominante.values
estr = (est.plataforma + "|" + est.fase).values
pos_estrato = [np.where(estr == s)[0] for s in sorted(set(estr))]
NOMES = ["relevantes"] + PO


def stats(idx):
    out = np.full((K, 4), np.nan)
    for t in range(K):
        i = idx[tp[idx] == t]
        wt = w[i].sum()
        if wt == 0:
            continue
        wr = w[i] * rel[i]
        out[t, 0] = wr.sum() / wt
        if wr.sum() > 0:
            out[t, 1:] = (wr[:, None] * P[i]).sum(0) / wr.sum()
    return out


pt = stats(np.arange(N))
rng = np.random.default_rng(SEED)
boot = np.empty((B, K, 4))
for b in range(B):
    idx = np.concatenate([ph[rng.integers(0, len(ph), len(ph))] for ph in pos_estrato])
    boot[b] = stats(idx)
lo = np.nanpercentile(boot, 2.5, axis=0)
hi = np.nanpercentile(boot, 97.5, axis=0)

n_amostra = np.array([(tp == t).sum() for t in range(K)])
n_rel = np.array([int(rel[tp == t].sum()) for t in range(K)])
N_rep = np.array([w[tp == t].sum() for t in range(K)])


def pcn(x):
    return "-" if np.isnan(x) else f"{x * 100:.1f}%".replace(".", ",")


def cel(t, j):
    return f"{pcn(pt[t, j])} [{pcn(lo[t, j])}-{pcn(hi[t, j])}]"


ordem = sorted(range(K), key=lambda t: -pt[t, 0])
L = [
    "# D5: relevância e posicionamento por tópico (k = 9)",
    "",
    'Tópico = tópico dominante do texto no modelo final do D5 (k = 9, semente 42). Estimativas ponderadas sobre os 2.000 textos do B9 rotulados pelo LLM (peso = N do estrato / n sorteado), com intervalo de 95% por bootstrap estratificado (1.000 reamostragens dentro de cada estrato plataforma x fase, semente 42). "Entre os relevantes" = razão entre o peso da classe e o peso dos relevantes. Script `d5_relevancia_topicos.py`.',
    "",
    "Ordenado pela % de relevantes.",
    "",
    "| Tópico (4 palavras principais) | n no B9 | n relevantes | Textos que representa | % relevantes | anti_RTO entre relevantes | pro_RTO entre relevantes | neutro_misto entre relevantes |",
    "|---|---|---|---|---|---|---|---|",
]
for t in ordem:
    rep = f"{int(round(N_rep[t])):,}".replace(",", ".")
    L.append(
        f"| {rot[t]} | {n_amostra[t]} | {n_rel[t]} | {rep} | {cel(t, 0)} | {cel(t, 1)} | {cel(t, 2)} | {cel(t, 3)} |"
    )
L += [
    "",
    "## Leitura",
    "",
]
top_rel = ordem[0]
bot_rel = ordem[-1]
L += [
    f"- Tópico mais relevante: {rot[top_rel]} ({pcn(pt[top_rel, 0])}); menos relevante: {rot[bot_rel]} ({pcn(pt[bot_rel, 0])}).",
    "- Tópicos com poucos relevantes na amostra (coluna n relevantes) têm intervalos largos; as porcentagens de posicionamento nesses casos são pouco precisas.",
    "- Os tópicos são do LDA sobre os textos já filtrados; a tabela mostra quais deles concentram texto fora do tema, e serve para escolher onde buscar excertos (C6a).",
    "",
    "Figura: `outputs/figuras_v2/fig_topicos_relevancia.png`.",
]
(u.RES / "D5_relevancia_topicos.md").write_text("\n".join(L) + "\n", encoding="utf-8")

# ---------- figura ----------
fig, ax = plt.subplots(figsize=(u.LARGURA_POL, 4.2))
ys = np.arange(K)[::-1]
vals = np.array([pt[t, 0] * 100 for t in ordem])
err = np.array(
    [
        [pt[t, 0] * 100 - lo[t, 0] * 100 for t in ordem],
        [hi[t, 0] * 100 - pt[t, 0] * 100 for t in ordem],
    ]
)
ax.barh(ys, vals, xerr=err, color="#4C78A8", ecolor="#333333", capsize=2, height=0.65)
ax.set_yticks(ys)
ax.set_yticklabels([rot[t] for t in ordem])
for y, t in zip(ys, ordem):
    ax.text(101, y, f"n={n_rel[t]}/{n_amostra[t]}", va="center", fontsize=u.FONTE_MIN)
ax.set_xlim(0, 112)
ax.set_xticks(range(0, 101, 20))
ax.set_xlabel(
    "% de textos relevantes (ponderada, IC 95%); n = relevantes/amostra no B9"
)
ax.spines[["top", "right"]].set_visible(False)
u.salva(fig, "fig_topicos_relevancia.png")
print("D5 relevancia OK", [(t, pcn(pt[t, 0])) for t in ordem])
