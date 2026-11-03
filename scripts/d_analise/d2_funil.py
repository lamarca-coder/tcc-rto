"""D2: Funil de filtragem por fonte (tabela e figura).

Etapa: D2
Objetivo: Funil de filtragem por fonte (tabela e figura).
Entradas: dados/v2/etapa1_funil.json, etapa2_resumo.json
Saídas: resultados/D2.md, figura em outputs/figuras_v2/
Como rodar: python -m scripts.d_analise.d2_funil
Tempo aproximado: segundos
"""

import json

import matplotlib.pyplot as plt

from scripts.utilitarios import util_v2 as u

f1 = json.load(open(u.ROOT / "dados/v2/etapa1_funil.json"))["funil"]
e2 = json.load(open(u.ROOT / "dados/v2/etapa2_resumo.json"))
corpus = u.carrega(["platform", "fonte_coleta"])
final_v0 = corpus[corpus.fonte_coleta == "v0"].platform.value_counts()
etapas = [
    "Bruto",
    "Após filtro léxico\ne exclusões",
    "Após filtro\nsemântico (0,30)",
    "Após duplicatas\ne vazios",
    "Após idioma (pt)",
    "Após duplicatas\nentre fontes",
]
chaves = [
    "bruto",
    "apos_lexico_e_exclusoes",
    "apos_semantico_0.30",
    "apos_duplicatas_e_vazios",
    "apos_idioma_pt",
]
fontes = {
    "v0, YouTube": [f1["v0_por_plataforma"]["youtube"][k] for k in chaves]
    + [int(final_v0["youtube"])],
    "v0, Reddit": [f1["v0_por_plataforma"]["reddit"][k] for k in chaves]
    + [int(final_v0["reddit"])],
    "YouTube C3": [f1["youtube_c3"][k] for k in chaves]
    + [int((corpus.fonte_coleta == "youtube_c3").sum())],
    "Bluesky": [f1["bluesky"][k] for k in chaves]
    + [int((corpus.fonte_coleta == "bluesky").sum())],
}
# confere: finais somam o corpus v2
assert sum(v[-1] for v in fontes.values()) == len(corpus), (
    sum(v[-1] for v in fontes.values()),
    len(corpus),
)

# ---- figura: um painel por fonte, barras horizontais, escala log ----
cores = {
    "v0, YouTube": u.COR["youtube"],
    "v0, Reddit": u.COR["reddit"],
    "YouTube C3": u.COR["youtube"],
    "Bluesky": u.COR["bluesky"],
}
fig, axs = plt.subplots(2, 2, figsize=(u.LARGURA_POL, 7.2), sharex=False)
for ax, (nome, vals) in zip(axs.ravel(), fontes.items()):
    y = list(range(len(etapas)))[::-1]
    ax.barh(y, vals, color=cores[nome], height=0.62)
    ax.set_xscale("log")
    ax.set_xlim(1000, max(vals) * 12)
    for yi, v in zip(y, vals):
        ax.text(v * 1.08, yi, u.num(v), va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(
        etapas if ax in (axs[0, 0], axs[1, 0]) else [""] * len(etapas), fontsize=8
    )
    ax.set_xlabel("Textos (escala logarítmica)")
    ax.text(
        0.98,
        0.04,
        nome,
        transform=ax.transAxes,
        ha="right",
        fontsize=9,
        fontweight="bold",
    )
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
u.salva(fig, "fig_funil.png")

# ---- tabela ----
L = [
    "# D2: funil de filtragem",
    "",
    "## O que foi feito",
    "Tabela e figura do funil por fonte, com os números de `resultados/D1.md` (arquivos `dados/v2/etapa1_funil.json` e `etapa2_resumo.json`, que geraram o D1) e a remoção de duplicatas entre fontes. Script `d2_funil.py`; figura `outputs/figuras_v2/fig_funil.png` (sem título, 15,9 cm, 300 dpi, escala logarítmica; o n está escrito em cada barra).",
    "",
    "## Números",
    "",
]
cab = ["Fonte"] + [e.replace("\n", " ") for e in etapas]
L += u.md_tabela(cab, [[n] + [u.num(x) for x in v] for n, v in fontes.items()])
tot = [sum(v[i] for v in fontes.values()) for i in range(len(etapas))]
L += ["| **Total** | " + " | ".join("**" + u.num(x) + "**" for x in tot) + " |", ""]
d = e2["duplicatas_removidas"]
L += [
    "Retenção em relação ao bruto (final / bruto): "
    + ", ".join(
        f"{n} {v[-1] / v[0] * 100:.1f}%".replace(".", ",") for n, v in fontes.items()
    )
    + ".",
    "",
    f"Duplicatas entre fontes (texto normalizado, mantendo o v0): YouTube C3 perdeu {d['youtube_c3']} linhas (iguais ao v0); Bluesky perdeu {d['bluesky']} (iguais ao v0 ou ao YouTube C3); v0 não perdeu nenhuma.",
    "",
    "## Problemas e decisões",
    '- O funil do v0 foi reconstruído em D1 a partir do bruto original (121.480 itens); o Reddit do v0 dá 5.164 depois do idioma, mas o corpus tem 5.163 porque no run original a deduplicação por texto foi feita com YouTube e Reddit juntos. A coluna "Após duplicatas entre fontes" do v0 Reddit usa o valor real do corpus (5.163).',
    "- O bruto do Bluesky (43.409) é o número único coletado em C1; as etapas léxico e semântico foram feitas em C1 (resultados/C1.md) e não foram repetidas em D1.",
    "- Escala logarítmica porque os brutos vão de 8.010 a 113.470; as barras comparam etapas dentro de cada painel, não entre painéis.",
    "- Reddit ampliado (C2) e as 24 buscas pendentes do YouTube (C3) não fazem parte deste funil.",
]
open(u.RES / "D2.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("D2 OK", tot)
