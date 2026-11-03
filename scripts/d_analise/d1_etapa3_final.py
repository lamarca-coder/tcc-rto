"""D1 (etapa 3): Alvo e heurística de posicionamento; grava o corpus_v2, o hash SHA-256 e o relatório.

Etapa: D1 (etapa 3)
Objetivo: Alvo e heurística de posicionamento; grava o corpus_v2, o hash SHA-256 e o relatório.
Entradas: saídas da etapa 2
Saídas: dados/v2/corpus_v2.parquet, resultados/D1.md, D1_hashes.txt
Como rodar: python -m scripts.d_analise.d1_etapa3_final
Tempo aproximado: alguns minutos
"""

import hashlib
import json
import time
from pathlib import Path

import pandas as pd

from src.analysis.stance import add_stance

OUT = Path("dados/v2")
R = Path("resultados")
v0 = pd.read_parquet("dados/v0/corpus_v0.parquet")
v0["fonte_coleta"] = "v0"
uni = pd.read_parquet(OUT / "ck_unido_sem_sentimento.parquet")
novas = pd.read_parquet(OUT / "ck_novas_com_sentimento.parquet")
assert len(uni) == len(v0) + len(novas), (len(uni), len(v0), len(novas))
assert set(uni[uni.fonte_coleta == "v0"].id) == set(v0.id)
corpus = pd.concat(
    [v0, novas], ignore_index=True
)  # v0 mantém seus rótulos de sentimento
corpus = corpus.drop(
    columns=["topic", "stance"], errors="ignore"
)  # topic: LDA refeito em D5; stance antigo é recalculado
t = time.time()
corpus = add_stance(corpus)
tempo_stance = round(time.time() - t, 1)
# confere que a heurística reproduz o stance já gravado no v0
cmp = corpus[corpus.fonte_coleta == "v0"].merge(
    v0[["id", "stance", "target"]], on="id", suffixes=("", "_v0")
)
igual_stance = float((cmp.stance == cmp.stance_v0).mean())
igual_alvo = float((cmp.target == cmp.target_v0).mean())
corpus = corpus.rename(columns={"stance": "stance_heuristica"})
corpus["stance_final"] = corpus["stance_heuristica"]
corpus.to_parquet(OUT / "corpus_v2.parquet", index=False)
h = hashlib.sha256(open(OUT / "corpus_v2.parquet", "rb").read()).hexdigest()
open(R / "D1_hashes.txt", "w").write(f"{h} *dados/v2/corpus_v2.parquet\n")

# ------------------------- relatório -------------------------
f1 = json.load(open(OUT / "etapa1_funil.json"))
e2 = json.load(open(OUT / "etapa2_resumo.json"))
fun = f1["funil"]
tm = f1["tempos_s"]
corpus["ano"] = pd.to_datetime(
    corpus.created_utc, utc=True, format="mixed", errors="coerce"
).dt.year
assert corpus["ano"].notna().all(), f"datas sem ano: {int(corpus.ano.isna().sum())}"
corpus["tipo"] = corpus.kind.map({"post": "post", "comment": "comentário"})
pt = lambda n: f"{n:,}".replace(",", ".")


def tab_pct(df, idx, col, ordem):
    c = pd.crosstab(df[idx], df[col])
    tot = c.sum(axis=1)
    rows = []
    for i in c.index:
        rows.append(
            "| "
            + str(i)
            + " | "
            + " | ".join(
                f"{pt(c.loc[i, k])} ({c.loc[i, k] / tot[i] * 100:.1f}%)"
                if k in c.columns
                else "0"
                for k in ordem
            )
            + f" | {pt(tot[i])} |"
        )
    t_all = c.sum()
    rows.append(
        "| **Total** | "
        + " | ".join(
            f"{pt(t_all[k])} ({t_all[k] / t_all.sum() * 100:.1f}%)"
            if k in c.columns
            else "0"
            for k in ordem
        )
        + f" | {pt(t_all.sum())} |"
    )
    return rows


chaves = [
    "bruto",
    "apos_lexico_e_exclusoes",
    "apos_semantico_0.30",
    "apos_duplicatas_e_vazios",
    "apos_idioma_pt",
]
linha = lambda nome, f: (
    "| " + nome + " | " + " | ".join(pt(f[k]) for k in chaves) + " |"
)
d = e2["duplicatas_removidas"]
L = [
    "# D1: corpus v2",
    "",
    f"Arquivo: `dados/v2/corpus_v2.parquet` ({pt(len(corpus))} linhas, não versionado). SHA-256 em `resultados/D1_hashes.txt`.",
    "",
    "## 1. Funil por fonte",
    "",
    "| Fonte | Bruto | Após léxico e exclusões | Após semântico (0,30) | Após duplicatas e vazios | Após idioma pt |",
    "|---|---:|---:|---:|---:|---:|",
    linha("v0 YouTube (reconstruído do bruto)", fun["v0_por_plataforma"]["youtube"]),
    linha("v0 Reddit (reconstruído do bruto)", fun["v0_por_plataforma"]["reddit"]),
    linha("YouTube C3", fun["youtube_c3"]),
    linha("Bluesky (filtrado em C1)", fun["bluesky"]),
    "",
    f"Depois do funil de cada fonte, removi as duplicatas **entre fontes** pelo texto normalizado (NFKC, minúsculas, espaços colapsados), mantendo o v0: YouTube C3 perdeu {d['youtube_c3']} linhas (iguais ao v0) e Bluesky perdeu {d['bluesky']} (iguais ao v0 ou ao YouTube C3). O v0 não perdeu nenhuma linha.",
    "",
    "Linhas finais por fonte: "
    + ", ".join(f"{k} = {pt(v)}" for k, v in corpus.fonte_coleta.value_counts().items())
    + f". Total **{pt(len(corpus))}**.",
    "",
    "Observação: o v0 reconstruído dá 5.515 (YouTube) + 5.164 (Reddit) = 10.679; o corpus v0 tem 10.678 porque no run original a deduplicação por texto foi feita com as duas plataformas juntas (uma linha do Reddit repetia texto do YouTube).",
    "",
    "## 2. Tabela final por plataforma e tipo",
    "",
    "| Plataforma | Tipo | n | % |",
    "|---|---|---:|---:|",
]
g = corpus.groupby(["platform", "tipo"]).size().reset_index(name="n")
L += [
    f"| {r.platform} | {r.tipo} | {pt(r.n)} | {r.n / g.n.sum() * 100:.1f} |"
    for r in g.itertuples()
]
L += [
    f"| **Total** | | **{pt(len(corpus))}** | 100 |",
    "",
    "## 3. Distribuição por ano e plataforma (n)",
    "",
]
a = pd.crosstab(corpus.ano, corpus.platform, margins=True)
L += ["| Ano | " + " | ".join(a.columns) + " |", "|---|" + "---:|" * len(a.columns)]
L += [
    "| " + str(i) + " | " + " | ".join(pt(x) for x in a.loc[i]) + " |" for i in a.index
]
ordem_s = ["negativo", "neutro", "positivo"]
L += [
    "",
    "## 4. Sentimento por plataforma (n e %)",
    "",
    "| Plataforma | negativo | neutro | positivo | Total |",
    "|---|---:|---:|---:|---:|",
]
L += tab_pct(corpus, "platform", "sentiment_label", ordem_s)
ordem_p = [
    c
    for c in ["anti-RTO", "pró-RTO", "neutro", "indefinido"]
    if c in set(corpus.stance_heuristica)
]
L += [
    "",
    "Posicionamento pela heurística (`stance_heuristica`, só para referência; `stance_final` é igual por enquanto):",
    "",
    "| Plataforma | " + " | ".join(ordem_p) + " | Total |",
    "|---|" + "---:|" * (len(ordem_p) + 1),
]
L += tab_pct(corpus, "platform", "stance_heuristica", ordem_p)
L += [
    "",
    "## 5. Tempo real de cada etapa",
    "",
    "| Etapa | Tempo |",
    "|---|---:|",
    f"| Etapa 1, YouTube C3: léxico / semântico / pré-processamento | {tm['youtube_c3_lexico_s']} s / {tm['youtube_c3_semantico_s']} s / {tm['youtube_c3_preproc_s']} s |",
    f"| Etapa 2: sentimento (transformer) em {pt(e2['linhas_novas'])} linhas novas | ~{e2['tempo_sentimento_s'] / 3600:.1f} h (estimado pelos horários dos checkpoints) |",
    f"| Etapa 3: alvo e heurística de posicionamento | {tempo_stance} s |",
    "",
    "## 6. Problemas e decisões",
    "",
    f"- A heurística recalculada reproduz o `stance` e o `target` gravados no v0 em {igual_stance * 100:.2f}% e {igual_alvo * 100:.2f}% das linhas.",
    "- O v0 manteve seus rótulos de sentimento; só as linhas novas foram pontuadas (cache `data/processed/sentiment_cache_transformer.parquet`, checkpoint a cada 400).",
    "- A coluna `topic` do v0 foi removida (o LDA será refeito em D5). `stance` virou `stance_heuristica`; `stance_final` = `stance_heuristica` até o B5/B6.",
    "- `html.unescape` aplicado ao texto do YouTube C3 e do Bluesky antes dos filtros e do pré-processamento (no Bluesky, depois do filtro léxico e semântico feito em C1, que não foi repetido; o pré-processamento foi refeito).",
    "- Filtro semântico com lotes de 32 (não 256) para evitar falta de memória; resultado idêntico. Os scripts rodam com `python -I`, o que obrigou a inserir a raiz do projeto e o diretório de pacotes do usuário em `sys.path` dentro de cada script.",
    "- Idioma: o filtro usa `lang_hint = pt` (comportamento original do pipeline); não há detecção real de idioma nessas fontes.",
    "- A primeira versão da etapa 2 removia também duplicatas dentro do v0 (52 linhas); foi corrigida para comparar só entre fontes e a etapa foi reiniciada.",
    "- `created_utc` mantém o formato original de cada fonte; o ano foi extraído com `pd.to_datetime(utc=True)`.",
    "- O YouTube C3 contribui pouco porque só 6% dos comentários passam no filtro léxico; as 24 buscas pendentes de C3 e o Reddit ampliado (C2) não estão neste corpus.",
    "- Dados e logs (`dados/`, `data/`) não foram versionados.",
]
open(R / "D1.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("OK", len(corpus), h)
