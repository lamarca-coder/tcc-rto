"""B11b: sensibilidade das estimativas do B11 à qualidade do corpus (marcas do D1b).

Etapa: B11b
Objetivo: Repete as estimativas principais do B11 (irrelevantes; posicionamento e sentimento entre os relevantes), total e por plataforma, excluindo da amostra os textos marcados no D1b (idioma diferente de pt, cópia excedente de quase duplicata, spam ou conteúdo adulto). Os pesos dos estratos são recalculados com o N do universo também sem esses textos. Bootstrap estratificado (1.000, semente 42), como no B11.
Entradas: dados/v2/corpus_v2.parquet, dados/v2/d1b_flags.parquet, rotulagem/b9/ (rótulos e pesos), rotulagem/mapa_ids.parquet, b8_mapa_ids.parquet
Saídas: resultados/B11b_sensibilidade.md
Como rodar: python -m scripts.d_analise.b11b_sensibilidade
Tempo aproximado: menos de 1 minuto
"""

import numpy as np
import pandas as pd

from scripts.utilitarios import util_v2 as u

R = u.ROOT / "rotulagem"
SEED = 42
B = 1000
IRR = "irrelevante"
PO = ["anti_RTO", "pro_RTO", "neutro_misto"]
SE = ["negativo", "positivo", "neutro"]
FASES = ["até 2019", "2020-2021", "jan/2022 a mai/2023", "jun/2023 em diante"]
PLS = ["bluesky", "reddit", "youtube"]
NOMES = ["irrel"] + [f"pos_{c}" for c in PO] + [f"sent_{c}" for c in SE]


def fase(d):
    if d.year <= 2019:
        return FASES[0]
    if d.year <= 2021:
        return FASES[1]
    if (d.year, d.month) <= (2023, 5):
        return FASES[2]
    return FASES[3]


def pcn(x):
    return "-" if np.isnan(x) else f"{x * 100:.1f}%".replace(".", ",")


# ---------- universo do B9 e marcas do D1b ----------
c = pd.read_parquet(u.CORPUS, columns=["id", "platform", "created_utc"])
c["fase"] = pd.to_datetime(
    c.created_utc, utc=True, format="mixed", errors="coerce"
).map(fase)
fl = pd.read_parquet(u.ROOT / "dados/v2/d1b_flags.parquet")
c = c.merge(fl[["id", "nao_pt", "dup_copia", "spam_qualquer"]], on="id")
c["problema"] = c.nao_pt | c.dup_copia | c.spam_qualquer
rotulados = set(pd.read_parquet(R / "mapa_ids.parquet").id) | set(
    pd.read_parquet(R / "b8_mapa_ids.parquet").id
)
univ = c[~c.id.isin(rotulados)]
N_orig = univ.groupby(["platform", "fase"]).size()
N_sem = univ[~univ.problema].groupby(["platform", "fase"]).size()

# ---------- amostra rotulada ----------
pesos = pd.read_parquet(R / "b9/b9_mapa_pesos.parquet")
l9 = pd.concat(
    [pd.read_excel(R / f"b9/rotulos_llm_b9_parte{k}.xlsx") for k in (1, 2, 3, 4)]
)[["id_rotulagem", "alvo", "sentimento", "posicionamento"]]
est = l9.merge(pesos, on="id_rotulagem").merge(c[["id", "problema"]], on="id")
assert len(est) == 2000
# confere que o peso original é N/n do universo
chk = est.groupby(["plataforma", "fase"]).peso.agg(["first", "count"])
for (p, f), r in chk.iterrows():
    assert abs(r["first"] - N_orig[(p, f)] / r["count"]) < 1e-6, (p, f)

est["peso_sem"] = np.nan
ok = est[~est.problema]
n_sem = ok.groupby(["plataforma", "fase"]).size()
for (p, f), n in n_sem.items():
    est.loc[(est.plataforma == p) & (est.fase == f) & ~est.problema, "peso_sem"] = (
        N_sem[(p, f)] / n
    )


def estima(df, wcol):
    df = df.reset_index(drop=True)
    w = df[wcol].values.astype(float)
    irr = (df.sentimento == IRR).values.astype(float)
    rel = 1 - irr
    P = np.stack([(df.posicionamento.values == k).astype(float) for k in PO], axis=1)
    S = np.stack([(df.sentimento.values == k).astype(float) for k in SE], axis=1)
    estr = (df.plataforma + "|" + df.fase).values
    pos = [np.where(estr == s)[0] for s in sorted(set(estr))]
    grupos = {"Total": np.ones(len(df), bool)}
    for p in PLS:
        grupos[p] = (df.plataforma == p).values

    def stats(idx, m):
        ww = w[idx] * m[idx]
        wr = ww * rel[idx]
        tot, tr = ww.sum(), wr.sum()
        if tot == 0:
            return np.full(len(NOMES), np.nan)
        out = [(ww * irr[idx]).sum() / tot]
        for M in (P, S):
            out += (
                list((wr[:, None] * M[idx]).sum(0) / tr)
                if tr > 0
                else [np.nan] * M.shape[1]
            )
        return np.array(out)

    rng = np.random.default_rng(SEED)
    pt = {g: stats(np.arange(len(df)), m) for g, m in grupos.items()}
    boot = {g: np.empty((B, len(NOMES))) for g in grupos}
    for b in range(B):
        idx = np.concatenate([ph[rng.integers(0, len(ph), len(ph))] for ph in pos])
        for g, m in grupos.items():
            boot[g][b] = stats(idx, m)
    lo = {g: np.nanpercentile(boot[g], 2.5, axis=0) for g in grupos}
    hi = {g: np.nanpercentile(boot[g], 97.5, axis=0) for g in grupos}
    info = {
        g: (int(m.sum()), int(round(w[m].sum())), int(rel[m].sum()))
        for g, m in grupos.items()
    }
    return pt, lo, hi, info


orig = estima(est, "peso")
sem = estima(est[~est.problema], "peso_sem")

# confere o original com o B11
ref = pd.read_csv(u.RES / "B11_numeros.csv")
for g, gb in (("Total", "Total"),) + tuple(
    (p, f"{u.NOME[p]}, todas as fases") for p in PLS
):
    for nome in NOMES:
        r = ref[(ref.grupo == gb) & (ref.estatistica == nome)].estimativa.iloc[0]
        assert abs(r - orig[0][g][NOMES.index(nome)]) < 1e-5, (g, nome)

ROT = {
    "irrel": "Irrelevantes (% de todos os textos)",
    "pos_anti_RTO": "anti_RTO (% entre os relevantes)",
    "pos_pro_RTO": "pro_RTO (% entre os relevantes)",
    "pos_neutro_misto": "neutro_misto (% entre os relevantes)",
    "sent_negativo": "Sentimento negativo (% entre os relevantes)",
    "sent_positivo": "Sentimento positivo (% entre os relevantes)",
    "sent_neutro": "Sentimento neutro (% entre os relevantes)",
}


def cel(res, g, nome):
    pt, lo, hi, _ = res
    i = NOMES.index(nome)
    return f"{pcn(pt[g][i])} [{pcn(lo[g][i])}-{pcn(hi[g][i])}]"


def fmt(n):
    return f"{int(n):,}".replace(",", ".")


nexc = int(est.problema.sum())
L = [
    "# B11b: sensibilidade das estimativas do B11 à qualidade do corpus",
    "",
    "Mesmas estimativas do B11 (rótulos do LLM no B9, pesos de sorteio, bootstrap estratificado de 1.000 reamostragens por plataforma x fase, semente 42), calculadas duas vezes: **original** (as 2.000 linhas, pesos do B9) e **sem problemas**, excluindo da amostra os textos marcados no D1b (idioma diferente de pt, cópia excedente de quase duplicata, spam ou conteúdo adulto). Na versão sem problemas, o peso de cada estrato é recalculado como N/n, com N = tamanho do universo do estrato também sem os textos marcados e n = linhas da amostra que restaram. Script `b11b_sensibilidade.py`. Nada mais foi alterado; as marcas vêm de `dados/v2/d1b_flags.parquet`.",
    "",
    "## O que saiu",
    "",
    f"- Da amostra: {nexc} de 2.000 linhas ({pcn(nexc / 2000)}).",
    f"- Do universo do B9 ({fmt(N_orig.sum())} textos): {fmt(N_orig.sum() - N_sem.sum())} ({pcn(1 - N_sem.sum() / N_orig.sum())}), restando {fmt(N_sem.sum())}.",
    "",
    "| Grupo | Amostra original | Amostra sem problemas | Relevantes (original) | Relevantes (sem problemas) | Universo original | Universo sem problemas |",
    "|---|---|---|---|---|---|---|",
]
for g in ["Total"] + PLS:
    nome = "Total" if g == "Total" else u.NOME[g]
    Nu0 = N_orig.sum() if g == "Total" else N_orig[g].sum()
    Nu1 = N_sem.sum() if g == "Total" else N_sem[g].sum()
    L.append(
        f"| {nome} | {orig[3][g][0]} | {sem[3][g][0]} | {orig[3][g][2]} | {sem[3][g][2]} | {fmt(Nu0)} | {fmt(Nu1)} |"
    )
for g in ["Total"] + PLS:
    nome = "Total" if g == "Total" else u.NOME[g]
    L += [
        "",
        f"## {nome}",
        "",
        "| Estatística | Original | Sem problemas | Diferença (p.p.) |",
        "|---|---|---|---|",
    ]
    for k in NOMES:
        i = NOMES.index(k)
        dif = (sem[0][g][i] - orig[0][g][i]) * 100
        L.append(
            f"| {ROT[k]} | {cel(orig, g, k)} | {cel(sem, g, k)} | {dif:+.1f}".replace(
                ".", ","
            )
            + " |"
        )
L += [
    "",
    "## Leitura",
    "",
    "- Os intervalos de 95% são do bootstrap estratificado; diferenças menores que a largura dos intervalos não são distinguíveis do acaso de amostragem. As duas colunas usam a mesma amostra, menos as linhas excluídas, então não são independentes.",
    '- A exclusão se baseia nas marcas heurísticas do D1b: langdetect (pouco confiável em textos curtos), cópias excedentes (o primeiro texto de cada grupo fica) e lista curta de termos adultos ("sexo" e "sexual" também aparecem em usos legítimos).',
]
(u.RES / "B11b_sensibilidade.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("B11b OK", nexc, cel(orig, "Total", "irrel"), cel(sem, "Total", "irrel"))
