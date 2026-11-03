"""B9: Amostra aleatória estratificada (plataforma x fase) de 2.000 textos para rotulagem pelo LLM; pesos N/n.

Etapa: B9
Objetivo: Amostra aleatória estratificada (plataforma x fase) de 2.000 textos para rotulagem pelo LLM; pesos N/n. Universo: corpus_v2 sem os 650 já rotulados. Semente 42.
Entradas: dados/v2/corpus_v2.parquet, rotulagem/mapa_ids.parquet, b8_mapa_ids.parquet, data/rto.db
Saídas: rotulagem/b9/ (planilhas, b9_mapa_pesos.parquet), resultados/B9_amostra.md
Como rodar: python -m scripts.b_validacao.b9_amostra
Tempo aproximado: segundos
"""

import sqlite3

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font

from scripts.utilitarios import util_v2 as u

OUT = u.ROOT / "rotulagem" / "b9"
OUT.mkdir(parents=True, exist_ok=True)
FASES = ["até 2019", "2020-2021", "jan/2022 a mai/2023", "jun/2023 em diante"]
META = {
    "bluesky": {"jun/2023 em diante": 800},
    "youtube": {f: 150 for f in FASES},
    "reddit": {f: 150 for f in FASES},
}


def fase(d):
    if d.year <= 2019:
        return FASES[0]
    if d.year <= 2021:
        return FASES[1]
    if (d.year, d.month) <= (2023, 5):
        return FASES[2]
    return FASES[3]


c = pd.read_parquet(
    u.CORPUS,
    columns=[
        "id",
        "platform",
        "kind",
        "text",
        "created_utc",
        "context_text",
        "parent_id",
        "source",
    ],
)
c["data"] = pd.to_datetime(c.created_utc, utc=True, format="mixed", errors="coerce")
assert c.data.notna().all()
c["fase"] = c.data.map(fase)
rotulados = set(pd.read_parquet(u.ROOT / "rotulagem/mapa_ids.parquet").id) | set(
    pd.read_parquet(u.ROOT / "rotulagem/b8_mapa_ids.parquet").id
)
assert len(rotulados) == 650 and c.id.isin(rotulados).sum() == 650
univ = c[~c.id.isin(rotulados)].reset_index(drop=True)
N = univ.groupby(["platform", "fase"]).size()


def aloca(plat):
    """Meta por estrato; estrato com N < meta entra inteiro e a sobra vai, proporcional ao N, para os estratos que ainda comportam mais."""
    alvo = {f: float(m) for f, m in META[plat].items()}
    n = {f: int(N.get((plat, f), 0)) for f in alvo}
    total = sum(alvo.values())
    aloc = {f: min(alvo[f], n[f]) for f in alvo}
    for _ in range(20):
        sobra = total - sum(aloc.values())
        if sobra < 1e-9:
            break
        livres = [f for f in alvo if aloc[f] < n[f]]
        if not livres:
            break
        pesos = np.array([n[f] for f in livres], float)
        pesos /= pesos.sum()
        extra = sobra * pesos  # proporcional ao N do estrato
        for f, e in zip(livres, extra):
            aloc[f] = min(n[f], aloc[f] + e)
    # arredondamento pelo maior resto, mantendo a soma
    base = {f: int(np.floor(v + 1e-9)) for f, v in aloc.items()}
    falta = int(round(total)) - sum(base.values())
    resto = sorted(aloc, key=lambda f: aloc[f] - base[f], reverse=True)
    for f in resto:
        if falta <= 0:
            break
        if base[f] < n[f]:
            base[f] += 1
            falta -= 1
    return base


aloc = {p: aloca(p) for p in META}
partes = []
rng_seed = 42
for p, fs in aloc.items():
    for f, k in fs.items():
        pool = univ[(univ.platform == p) & (univ.fase == f)]
        s = (
            pool.sample(n=k, random_state=rng_seed) if k < len(pool) else pool.copy()
        )  # amostra aleatória simples
        s = s.assign(N_estrato=len(pool), n_estrato=len(s), peso=len(pool) / len(s))
        partes.append(s)
amostra = pd.concat(partes, ignore_index=True)
assert len(amostra) == 2000 and amostra.id.is_unique

# embaralha (semente 42) e numera 2001..4000
amostra = amostra.sample(frac=1, random_state=42).reset_index(drop=True)
amostra["id_rotulagem"] = np.arange(2001, 4001)
amostra["parte"] = np.repeat([1, 2, 3, 4], 500)
amostra["tipo"] = amostra.kind.map({"post": "post", "comment": "comentário"})

# contexto: título do vídeo/post (mesma lógica da B2); Bluesky vazio
con = sqlite3.connect(u.ROOT / "data/rto.db")
titulos = dict(
    con.execute(
        "SELECT id, context_text FROM posts WHERE kind='post' AND context_text IS NOT NULL"
    )
)


def contexto(r):
    if r.platform == "bluesky":
        return ""
    if isinstance(r.context_text, str) and r.context_text.strip():
        return r.context_text
    if isinstance(r.parent_id, str) and r.parent_id in titulos:
        return titulos[r.parent_id]
    return r.source if r.platform == "youtube" and isinstance(r.source, str) else ""


amostra["contexto"] = amostra.apply(contexto, axis=1)

# mapa de pesos (contém o id do corpus: não vai para o repositório público)
amostra[["id_rotulagem", "id", "platform", "fase", "peso"]].rename(
    columns={"platform": "plataforma"}
).to_parquet(OUT / "b9_mapa_pesos.parquet", index=False)

# planilhas (sem rótulos do modelo, sem author_hash, sem id do corpus)
for k in (1, 2, 3, 4):
    d = amostra[amostra.parte == k]
    wb = Workbook()
    ws = wb.active
    ws.title = "rotulagem"
    ws.append(["id_rotulagem", "plataforma", "tipo", "texto", "contexto"])
    for r in d.itertuples():
        ws.append([r.id_rotulagem, r.platform, r.tipo, r.text, r.contexto])
    for cel in ws[1]:
        cel.font = Font(bold=True)
    for col, w in zip("ABCDE", (14, 11, 12, 90, 50)):
        ws.column_dimensions[col].width = w
    for row in ws.iter_rows(min_row=2, min_col=4, max_col=5):
        for cel in row:
            cel.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "A2"
    wb.save(OUT / f"b9_parte{k}.xlsx")

# ---------------- relatório ----------------
est = (
    amostra.groupby(["platform", "fase"])
    .agg(N=("N_estrato", "first"), n=("n_estrato", "first"), peso=("peso", "first"))
    .reset_index()
)
est["meta"] = [META[p].get(f, 0) for p, f in zip(est.platform, est.fase)]
est["ordem"] = est.fase.map({f: i for i, f in enumerate(FASES)})
est = est.sort_values(["platform", "ordem"])
vc = lambda x, d=2: f"{x:.{d}f}".replace(".", ",")
L = [
    "# B9: amostra estratificada para rotulagem pelo LLM no chat",
    "",
    "## O que foi feito",
    "Amostra aleatória simples dentro de cada estrato plataforma x fase (semente 42), 2.000 textos, do corpus v2 sem os 650 textos já rotulados (500 do B2 em `rotulagem/mapa_ids.parquet` e 150 do B8 em `rotulagem/b8_mapa_ids.parquet`). Universo: "
    + u.num(len(univ))
    + " textos. Meta: Bluesky 800 (só existe na última fase), YouTube 150 por fase, Reddit 150 por fase. `id_rotulagem` de 2001 a 4000, embaralhados (semente 42) e divididos em 4 planilhas de 500 linhas (`rotulagem/b9/b9_parte1.xlsx` a `b9_parte4.xlsx`; colunas `id_rotulagem`, `plataforma`, `tipo`, `texto`, `contexto`; sem rótulos do modelo, sem `author_hash`, sem id do corpus). Pesos em `rotulagem/b9/b9_mapa_pesos.parquet` (`id_rotulagem`, `id`, `plataforma`, `fase`, `peso`). Script `b9_amostra.py`.",
    "",
    "Peso = N do estrato no universo / n sorteado no estrato. A soma dos pesos de cada estrato é o N do estrato.",
    "",
    "## Estratos",
    "",
]
L += u.md_tabela(
    ["Plataforma", "Fase", "N no universo", "Meta", "n sorteado", "Peso"],
    [
        [u.NOME[r.platform], r.fase, u.num(r.N), r.meta, r.n, vc(r.peso)]
        for r in est.itertuples()
    ]
    + [
        ["**Total**", "", u.num(est.N.sum()), int(est.meta.sum()), int(est.n.sum()), ""]
    ],
)
red = [
    f"- {u.NOME[p]}: "
    + "; ".join(
        f"{f} (N = {u.num(int(N.get((p, f), 0)))}) entrou inteiro e a sobra de {m - int(N.get((p, f), 0))} foi redistribuída"
        for f, m in META[p].items()
        if int(N.get((p, f), 0)) < m
    )
    for p in META
    if any(int(N.get((p, f), 0)) < m for f, m in META[p].items())
]
L += (
    ["", "### Redistribuição da sobra", ""]
    + red
    + [
        "",
        "A sobra de cada plataforma foi repartida entre os estratos da mesma plataforma que ainda comportavam mais textos, proporcionalmente ao N do estrato (arredondamento pelo maior resto, mantendo a soma da plataforma).",
    ]
)
L += ["", "## Distribuição por plataforma em cada parte", ""]
dist = pd.crosstab(amostra.parte, amostra.platform).reindex(
    columns=["bluesky", "reddit", "youtube"], fill_value=0
)
L += u.md_tabela(
    ["Parte", "Bluesky", "Reddit", "YouTube", "Total", "Intervalo de id_rotulagem"],
    [
        [
            f"parte{k}",
            dist.loc[k, "bluesky"],
            dist.loc[k, "reddit"],
            dist.loc[k, "youtube"],
            int(dist.loc[k].sum()),
            f"{amostra[amostra.parte == k].id_rotulagem.min()}-{amostra[amostra.parte == k].id_rotulagem.max()}",
        ]
        for k in (1, 2, 3, 4)
    ]
    + [
        [
            "**Total**",
            int(dist.bluesky.sum()),
            int(dist.reddit.sum()),
            int(dist.youtube.sum()),
            int(dist.values.sum()),
            "2001-4000",
        ]
    ],
)
tp = amostra.tipo.value_counts().to_dict()
L += [
    "",
    "## Problemas e decisões",
    "- Os pesos devem ser usados em qualquer estimativa para o corpus: os estratos têm taxas de amostragem muito diferentes (por exemplo, Bluesky amostra cerca de 2% do estrato; Reddit e YouTube em fases iniciais amostram quase tudo).",
    "- As planilhas têm só as cinco colunas pedidas (sem listas suspensas, sem aba Instruções), porque a rotulagem é feita pelo LLM no chat.",
    f"- Tipos na amostra: {tp.get('comentário', 0)} comentários e {tp.get('post', 0)} posts.",
    "- `contexto` (título do vídeo ou do post) vem do corpus e, para comentários do Reddit, do post-pai no banco; ficou vazio quando o título não existe (Bluesky e comentários do Reddit sem post-pai coletado).",
    "- `b9_mapa_pesos.parquet` tem o `id` do corpus (id original de YouTube e Reddit): **proibido no repositório público**; as planilhas têm textos de usuários: **não versionar**. `rotulagem/b9/` foi acrescentada ao `.gitignore`.",
    "- Nenhum modelo foi carregado; o D5 continua rodando.",
]
open(u.RES / "B9_amostra.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("B9 OK", len(amostra))
print(est[["platform", "fase", "N", "meta", "n", "peso"]].to_string())
