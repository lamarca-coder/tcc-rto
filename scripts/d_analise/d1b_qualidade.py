"""D1b: diagnóstico de qualidade do corpus v2 (idioma, quase duplicatas, spam), sem alterar o corpus.

Etapa: D1b
Objetivo: Conta textos de idioma não português (langdetect, semente 0), grupos de textos idênticos após normalização, pares com cosseno >= 0,95 nos embeddings e textos com sinais de spam ou conteúdo adulto; cruza com os 2.000 rótulos do B9. Não remove nada.
Entradas: dados/v2/corpus_v2.parquet, data/processed/embeddings_minilm.npy e _index.parquet, rotulagem/b9/ (rótulos e pesos)
Saídas: resultados/D1b_qualidade.md, dados/v2/d1b_flags.parquet (marcas por texto, não versionado)
Como rodar: python -m scripts.d_analise.d1b_qualidade
Tempo aproximado: 5 a 10 minutos
"""

import html
import re
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd
from langdetect import DetectorFactory, detect
from langdetect.lang_detect_exception import LangDetectException
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from scripts.utilitarios import util_v2 as u

R = u.ROOT / "rotulagem"
SAIDA_FLAGS = u.ROOT / "dados/v2/d1b_flags.parquet"
LIMIAR_COS = 0.95
PLS = ["bluesky", "reddit", "youtube"]
IRR = "irrelevante"

# termos de conteúdo adulto (lista curta; "sexo" também aparece em discussões legítimas, ver nota no relatório)
ADULTO = re.compile(
    r"\b(porn[oô]?|pornografia|xvideos|onlyfans|nudes?|buceta|xoxota|xereca|punheta|siririca|"
    r"putaria|sexo|sexual|tes[aã]o|pau duro|safad[ao]|gostosa)\b",
    re.I,
)
MENCOES = re.compile(r"@\w+")
REPETIDA = re.compile(
    r"\b(\w+)(?:\W+\1\b){4,}", re.I
)  # mesma palavra 5 ou mais vezes seguidas


def pc(k, n):
    return f"{100 * k / n:.1f}%".replace(".", ",") if n else "-"


def fm(n):
    return f"{int(n):,}".replace(",", ".")


def norm(t):
    t = html.unescape(str(t))
    t = "\n".join(ln for ln in t.split("\n") if not ln.lstrip().startswith(">"))
    t = unicodedata.normalize("NFKC", t).lower()
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


c = pd.read_parquet(u.CORPUS, columns=["id", "platform", "text"])
N = len(c)
c["t"] = c.text.map(html.unescape)

# ---------- a) idioma ----------
DetectorFactory.seed = 0
lang = []
for t in c.t:
    try:
        lang.append(detect(t))
    except LangDetectException:
        lang.append("falha")
c["lang_det"] = lang

# ---------- b) quase duplicatas ----------
c["norm"] = c.t.map(norm)
vazio = c.norm == ""
g_norm = c[~vazio].groupby("norm").size()
g_norm = g_norm[g_norm > 1]
c["grupo_norm"] = c.norm.map({v: i for i, v in enumerate(g_norm.index)})
c["em_grupo_norm"] = c.grupo_norm.notna()

E = np.load(u.ROOT / "data/processed/embeddings_minilm.npy", mmap_mode="r")
ix = pd.read_parquet(u.ROOT / "data/processed/embeddings_minilm_index.parquet")
assert len(ix) == N and (ix.id.values == c.id.values).all() and E.shape[0] == N
E = np.ascontiguousarray(E, dtype=np.float32)
ii, jj = [], []
BL = 1500
for a in range(0, N, BL):
    S = E[a : a + BL] @ E.T
    r, col = np.nonzero(S >= LIMIAR_COS)
    r = r + a
    m = col > r
    ii.append(r[m].astype(np.int32))
    jj.append(col[m].astype(np.int32))
ii, jj = np.concatenate(ii), np.concatenate(jj)
plat = c.platform.values
par_same = {p: int(((plat[ii] == p) & (plat[jj] == p)).sum()) for p in PLS}
par_cross = int((plat[ii] != plat[jj]).sum())
tem_par = np.zeros(N, bool)
tem_par[ii] = True
tem_par[jj] = True
c["par_emb"] = tem_par

# grupos: união dos idênticos normalizados e dos pares com cosseno >= 0,95
gn = c[c.em_grupo_norm].reset_index().groupby("grupo_norm")["index"].apply(list)
ea, eb = [], []
for lst in gn:
    ea += lst[:-1]
    eb += lst[1:]
A = np.concatenate([ii, np.array(ea, dtype=np.int32)])
B_ = np.concatenate([jj, np.array(eb, dtype=np.int32)])
_, comp = connected_components(
    coo_matrix((np.ones(len(A)), (A, B_)), shape=(N, N)), directed=False
)
c["grupo_dup"] = comp
tam = pd.Series(comp).map(pd.Series(comp).value_counts()).values
c["dup_grupo_tam"] = tam
c["dup_qualquer"] = tam > 1
c["dup_copia"] = c.dup_qualquer & c.duplicated(
    "grupo_dup", keep="first"
)  # cópia excedente

# ---------- c) spam ----------
c["spam_mencoes"] = c.t.map(lambda t: len(MENCOES.findall(t)) >= 4)
c["spam_repete"] = c.t.map(lambda t: bool(REPETIDA.search(t)))
c["adulto"] = c.t.map(lambda t: bool(ADULTO.search(t)))
c["spam_qualquer"] = c.spam_mencoes | c.spam_repete | c.adulto
c["nao_pt"] = c.lang_det != "pt"
c["algum_problema"] = c.nao_pt | c.dup_qualquer | c.spam_qualquer
c["problema_sem_idioma"] = c.dup_qualquer | c.spam_qualquer

c[
    [
        "id",
        "platform",
        "lang_det",
        "grupo_dup",
        "dup_grupo_tam",
        "dup_qualquer",
        "dup_copia",
        "em_grupo_norm",
        "par_emb",
        "spam_mencoes",
        "spam_repete",
        "adulto",
        "spam_qualquer",
        "nao_pt",
        "algum_problema",
    ]
].to_parquet(SAIDA_FLAGS, index=False)

# ---------- d) cruzamento com o B9 ----------
pesos = pd.read_parquet(R / "b9/b9_mapa_pesos.parquet")
l9 = pd.concat(
    [pd.read_excel(R / f"b9/rotulos_llm_b9_parte{k}.xlsx") for k in (1, 2, 3, 4)]
)[["id_rotulagem", "sentimento"]]
est = l9.merge(pesos, on="id_rotulagem").merge(
    c.drop(columns=["text", "t", "norm"]), on="id"
)
assert len(est) == 2000
est["irr"] = est.sentimento == IRR

# ---------- relatório ----------
L = ["# D1b: diagnóstico de qualidade do corpus v2", ""]
L += [
    f"Corpus v2 de {fm(N)} textos, **sem nenhuma remoção** (`dados/v2/corpus_v2.parquet` não foi alterado). Script `d1b_qualidade.py`; marcas por texto em `dados/v2/d1b_flags.parquet` (não versionado).",
    "",
    "## a) Idioma (langdetect 1.0.9, `DetectorFactory.seed = 0`, texto após `html.unescape`)",
    "",
    "| Plataforma | Textos | pt | % pt | Outro idioma | % outro | Falhas | % falhas |",
    "|---|---|---|---|---|---|---|---|",
]
for p in PLS + ["total"]:
    d = c if p == "total" else c[c.platform == p]
    n = len(d)
    k_pt = int((d.lang_det == "pt").sum())
    k_f = int((d.lang_det == "falha").sum())
    k_o = n - k_pt - k_f
    L.append(
        f"| {u.NOME[p]} | {fm(n)} | {fm(k_pt)} | {pc(k_pt, n)} | {fm(k_o)} | {pc(k_o, n)} | {fm(k_f)} | {pc(k_f, n)} |"
    )
L += [
    "",
    "Os 5 idiomas não portugueses mais comuns (contagem; % dos textos da plataforma):",
    "",
]
for p in PLS + ["total"]:
    d = c if p == "total" else c[c.platform == p]
    top = Counter(d.lang_det[~d.lang_det.isin(["pt", "falha"])]).most_common(5)
    L.append(
        f"- **{u.NOME[p]}**: "
        + ", ".join(f"{k} {fm(v)} ({pc(v, len(d))})" for k, v in top)
    )
L = [x if x.startswith("- **") else x for x in L]

L += [
    "",
    "## b) Quase duplicatas",
    "",
    "Normalização: `html.unescape`, minúsculas, remoção de linhas iniciadas por `>`, remoção de pontuação e espaços colapsados.",
    "",
    "| Plataforma | Textos | Grupos de idênticos (normalizados) | Textos nesses grupos | Cópias excedentes | % do corpus em grupos | Pares com cosseno >= 0,95 (mesma plataforma) | Textos com algum par |",
    "|---|---|---|---|---|---|---|---|",
]
for p in PLS + ["total"]:
    d = c if p == "total" else c[c.platform == p]
    gg = (
        d[d.em_grupo_norm].groupby("grupo_norm").ngroups
        if p != "total"
        else len(g_norm)
    )
    kk = int(d.em_grupo_norm.sum())
    pr = par_same[p] if p != "total" else sum(par_same.values()) + par_cross
    L.append(
        f"| {u.NOME[p]} | {fm(len(d))} | {fm(gg)} | {fm(kk)} | {fm(kk - gg)} | {pc(kk, len(d))} | {fm(pr)} | {fm(int(d.par_emb.sum()))} |"
    )
L += [
    "",
    f"Na linha Total, os pares incluem {fm(par_cross)} pares entre plataformas diferentes. Textos vazios após a normalização: {fm(int(vazio.sum()))} (fora dos grupos).",
    f"Grupos de quase duplicatas pela união das duas regras (idênticos normalizados ou cosseno >= 0,95): {fm(int(c[c.dup_qualquer].grupo_dup.nunique()))} grupos, {fm(int(c.dup_qualquer.sum()))} textos ({pc(int(c.dup_qualquer.sum()), N)} do corpus), dos quais {fm(int(c.dup_copia.sum()))} são cópias excedentes ({pc(int(c.dup_copia.sum()), N)}).",
    "",
    "## c) Spam e conteúdo adulto",
    "",
    'Critérios: 4 ou mais menções `@`; mesma palavra 5 ou mais vezes seguidas; termo da lista de conteúdo adulto (`porn`, `xvideos`, `onlyfans`, `nudes`, `buceta`, `punheta`, `putaria`, `sexo`, `sexual`, `tesão` e semelhantes; "sexo" e "sexual" também aparecem em discussões legítimas, então essa coluna superestima).',
    "",
    "| Plataforma | Textos | 4+ menções | % | Palavra repetida 5+ | % | Conteúdo adulto | % | Qualquer um | % |",
    "|---|---|---|---|---|---|---|---|---|---|",
]
for p in PLS + ["total"]:
    d = c if p == "total" else c[c.platform == p]
    n = len(d)
    ks = [
        int(d[k].sum())
        for k in ("spam_mencoes", "spam_repete", "adulto", "spam_qualquer")
    ]
    L.append(
        f"| {u.NOME[p]} | {fm(n)} | "
        + " | ".join(f"{fm(k)} | {pc(k, n)}" for k in ks)
        + " |"
    )

L += [
    "",
    "## Resumo: % do corpus por problema",
    "",
    "| Problema | Textos | % do corpus |",
    "|---|---|---|",
]
resumo = [
    ("Idioma diferente de pt (inclui falhas)", c.nao_pt),
    ("  das quais falhas do detector", c.lang_det == "falha"),
    ("Em grupo de quase duplicatas (qualquer cópia)", c.dup_qualquer),
    ("Cópia excedente de quase duplicata", c.dup_copia),
    ("Spam ou conteúdo adulto (qualquer critério)", c.spam_qualquer),
    ("Pelo menos um dos três problemas", c.algum_problema),
]
for nome, m in resumo:
    L.append(f"| {nome} | {fm(int(m.sum()))} | {pc(int(m.sum()), N)} |")

L += [
    "",
    "## d) Cruzamento com os 2.000 rótulos do B9 (LLM)",
    "",
    f"Nos 2.000 textos do B9, o LLM marcou {fm(int(est.irr.sum()))} como irrelevantes ({pc(int(est.irr.sum()), 2000)}, sem pesos; a estimativa ponderada para o corpus é 60,2%, ver `resultados/B11.md`). A tabela mostra, para cada problema, quantos dos textos do B9 têm o problema e quantos deles o LLM marcou como irrelevantes. Contagens sem pesos (n pequeno nas linhas raras).",
    "",
    "| Problema | Textos no B9 | Irrelevantes (LLM) | % irrelevantes | % irrelevantes nos textos sem o problema |",
    "|---|---|---|---|---|",
]
cruz = [
    ("Idioma diferente de pt", est.nao_pt),
    ("Falha do detector de idioma", est.lang_det == "falha"),
    ("Em grupo de quase duplicatas", est.dup_qualquer),
    ("Cópia excedente de duplicata", est.dup_copia),
    ("4+ menções", est.spam_mencoes),
    ("Palavra repetida 5+", est.spam_repete),
    ("Conteúdo adulto", est.adulto),
    ("Spam ou adulto (qualquer)", est.spam_qualquer),
    ("Algum dos três problemas", est.algum_problema),
]
for nome, m in cruz:
    k = int(m.sum())
    ki = int(est.irr[m].sum())
    sem = est[~m]
    L.append(
        f"| {nome} | {k} | {ki} | {pc(ki, k)} | {pc(int(sem.irr.sum()), len(sem))} |"
    )
L += [
    "",
    "Por plataforma (B9), textos de idioma diferente de pt:",
    "",
    "| Plataforma | B9 não-pt | Irrelevantes | % |",
    "|---|---|---|---|",
]
for p in PLS:
    d = est[(est.plataforma == p) & est.nao_pt]
    L.append(
        f"| {u.NOME[p]} | {len(d)} | {int(d.irr.sum())} | {pc(int(d.irr.sum()), len(d))} |"
    )
L += [
    "",
    "## Observações",
    "",
    "- Nada foi removido. Estas contagens só dimensionam o problema; remover ou não é decisão posterior.",
    "- O langdetect é pouco confiável em textos curtos (comentários de poucas palavras, gírias, emojis); falhas e erros nesses casos são esperados.",
    "- Os embeddings são os do `paraphrase-multilingual-MiniLM-L12-v2` já salvos no B10b; o cosseno >= 0,95 também pega paráfrases muito próximas, não só cópias.",
]
(u.RES / "D1b_qualidade.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print(
    "D1b OK",
    int(c.nao_pt.sum()),
    int(c.dup_qualquer.sum()),
    int(c.spam_qualquer.sum()),
    len(ii),
)
