"""D5 (passos 3 e 4): Média e desvio da coerência por k, figura e 15 palavras dos 3 k de maior coerência.

Etapa: D5 (passos 3 e 4)
Objetivo: Média e desvio da coerência por k, figura e 15 palavras dos 3 k de maior coerência. Não escolhe o k.
Entradas: dados/v2/d5_coerencia.csv
Saídas: outputs/d5_coerencia.png, resultados/D5_coerencia.csv e D5_topicos.md
Como rodar: python -m scripts.d_analise.d5_passo4
Tempo aproximado: segundos
"""

import json

import matplotlib.pyplot as plt
import pandas as pd

from scripts.utilitarios import util_v2 as u

raw = pd.read_csv(u.ROOT / "dados/v2/d5_coerencia.csv")
raw["k"] = raw["k"].astype(int)
raw["seed"] = raw["seed"].astype(int)
assert len(raw) == 27, f"faltam modelos: {len(raw)} de 27"
g = raw.groupby("k").coerencia_cv
res = pd.DataFrame({"media_cv": g.mean(), "dp_cv": g.std(ddof=1)})
for s in (42, 43, 44):
    res[f"cv_semente_{s}"] = raw[raw.seed == s].set_index("k").coerencia_cv
res = res.reset_index()
res["k"] = res["k"].astype(int)
res.round(4).to_csv(u.RES / "D5_coerencia.csv", index=False)

fig, ax = plt.subplots(figsize=(u.LARGURA_POL, 3.8))
ax.errorbar(
    res.k, res.media_cv, yerr=res.dp_cv, fmt="o-", color="#0072B2", capsize=4, lw=1.5
)
ax.set_xlabel("Número de tópicos (k)")
ax.set_ylabel("Coerência c_v (média de 3 sementes ± desvio padrão)")
ax.set_xticks(res.k)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
(u.ROOT / "outputs").mkdir(exist_ok=True)
fig.savefig(u.ROOT / "outputs/d5_coerencia.png", dpi=300, bbox_inches="tight")
plt.close(fig)

top3 = res.sort_values("media_cv", ascending=False).head(3).k.tolist()
L = [
    "# D5: palavras principais dos 3 valores de k com maior coerência média",
    "",
    "Modelos treinados com semente 42 (as palavras vêm do modelo da semente 42 de cada k; as outras sementes só entram na coerência). Cada tópico mostra as 15 palavras de maior peso. **Não escolhi o k.**",
    "",
    "Coerência c_v por k (média e desvio padrão de 3 sementes):",
    "",
]
L += u.md_tabela(
    ["k", "média c_v", "desvio padrão", "42", "43", "44"],
    [
        [
            int(r.k),
            f"{r.media_cv:.4f}",
            f"{r.dp_cv:.4f}",
            f"{r.cv_semente_42:.4f}",
            f"{r.cv_semente_43:.4f}",
            f"{r.cv_semente_44:.4f}",
        ]
        for r in res.itertuples()
    ],
)
L += ["", f"Os 3 k de maior coerência média: {', '.join(str(k) for k in top3)}.", ""]
for k in top3:
    top = json.load(
        open(u.ROOT / f"dados/v2/d5_topicos_k{k}_s42.json", encoding="utf-8")
    )
    L += [f"## k = {k}", ""]
    L += [f"- **Tópico {i}**: " + ", ".join(w) for i, w in enumerate(top)]
    L += [""]
open(u.RES / "D5_topicos.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print(res.round(4).to_string(), top3)
