"""B8 (acréscimo): Rótulos do LLM no B8 contra a heurística (alvo, posicionamento) e o BERT (sentimento).

Etapa: B8 (acréscimo)
Objetivo: Rótulos do LLM no B8 contra a heurística (alvo, posicionamento) e o BERT (sentimento).
Entradas: rotulagem/rotulos_llm_b8.xlsx, b8_gabarito_modelo.parquet
Saídas: anexo ao fim de resultados/B8.md
Como rodar: python -m scripts.b_validacao.b8_comparacao_llm
Tempo aproximado: segundos
"""

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from scripts.utilitarios import util_v2 as u

R = u.ROOT / "rotulagem"
llm = pd.read_excel(R / "rotulos_llm_b8.xlsx")
mod = pd.read_parquet(R / "b8_gabarito_modelo.parquet")
d = llm.merge(mod, on="id_rotulagem", how="inner")
assert len(d) == 150


def stance(s):
    s = str(s)
    return (
        "anti_RTO"
        if s.startswith("anti")
        else "pro_RTO"
        if s.startswith("pr")
        else "neutro_misto"
    )  # neutro e indefinido -> neutro_misto


d["pos_h"] = d.stance_heuristica.map(stance)
d["alvo_h"] = d.target.astype(str)
d["sent_b"] = d.sentiment_label.astype(str)
IRR = "irrelevante"
rel = d[d.sentimento != IRR]


def conc(a, b):
    return int((a == b).sum()), len(a)


linhas = []
for nome, col_llm, col_mod, quem in [
    ("Sentimento", "sentimento", "sent_b", "BERT"),
    ("Alvo", "alvo", "alvo_h", "heurística"),
    ("Posicionamento", "posicionamento", "pos_h", "heurística"),
]:
    k1, n1 = conc(d[col_llm], d[col_mod])
    k2, n2 = conc(rel[col_llm], rel[col_mod])
    kap = cohen_kappa_score(rel[col_llm], rel[col_mod])
    linhas.append(
        [
            nome,
            quem,
            f"{k1} de {n1} ({u.pct_ic(k1, n1)})",
            f"{k2} de {n2} ({u.pct_ic(k2, n2)})",
            f"{kap:.2f}".replace(".", ","),
        ]
    )

labs_l = ["pro_RTO", "anti_RTO", "neutro_misto", IRR]
labs_h = ["pro_RTO", "anti_RTO", "neutro_misto"]
cm = pd.crosstab(d.posicionamento, d.pos_h).reindex(
    index=labs_l, columns=labs_h, fill_value=0
)
anti = d[d.posicionamento == "anti_RTO"]
pro_h = d[d.pos_h == "pro_RTO"]
L = (
    [
        "",
        "## Comparação do LLM com o pipeline (acrescentado em 28/10/2026)",
        "",
        "Os 150 posts do B8 foram rotulados pelo LLM (`rotulos_llm_b8.xlsx`) e comparados com os rótulos do pipeline em `rotulagem/b8_gabarito_modelo.parquet`: sentimento do BERT e alvo e posicionamento da heurística (mapeamento do B5: anti-RTO > anti_RTO, pró-RTO > pro_RTO, neutro e indefinido > neutro_misto). **O LLM não é um gabarito humano**: a tabela mede concordância, não acerto. Script `b8_comparacao_llm.py`.",
        "",
        'Concordância (acordo simples, IC 95% de Wilson). Na primeira coluna o LLM marcar "irrelevante" conta como discordância, porque o pipeline nunca prevê essa classe; na segunda, só as linhas que o LLM considera relevantes:',
        "",
    ]
    + u.md_tabela(
        [
            "Coluna",
            "Pipeline",
            "Concordância nas 150",
            f"Concordância nas {len(rel)} relevantes pelo LLM",
            "Kappa (relevantes)",
        ],
        linhas,
    )
    + [
        "",
        "Matriz de confusão do posicionamento (linhas = LLM, colunas = heurística; 150 posts):",
        "",
    ]
    + u.md_tabela(
        ["LLM \\ Heurística"] + labs_h,
        [[i] + [int(cm.loc[i, c]) for c in labs_h] for i in labs_l],
    )
    + [
        "",
        f"- A heurística marcou {len(pro_h)} dos 150 posts como pro_RTO ({u.pct_ic(len(pro_h), 150)}); o LLM marcou 0 como pro_RTO.",
        f"- Dos {len(pro_h)} posts que a heurística chamou de pro_RTO, o LLM considerou {int((pro_h.posicionamento == IRR).sum())} irrelevantes, {int((pro_h.posicionamento == 'anti_RTO').sum())} anti_RTO e {int((pro_h.posicionamento == 'neutro_misto').sum())} neutro_misto.",
        f"- Dos {len(anti)} posts anti_RTO pelo LLM, a heurística marcou {int((anti.pos_h == 'pro_RTO').sum())} como pro_RTO ({u.pct_ic(int((anti.pos_h == 'pro_RTO').sum()), len(anti))}); nesses, o BERT deu sentimento negativo em {int(((anti.pos_h == 'pro_RTO') & (anti.sent_b == 'negativo')).sum())}.",
        "- Padrão idêntico ao do B5 (gabarito humano): a heurística chama de pro_RTO o que o julgamento (humano ou LLM) chama de anti_RTO ou neutro, quando há sentimento negativo sobre o remoto.",
    ]
)
texto = "\n".join(L) + "\n"
p = u.RES / "B8.md"
atual = p.read_text(encoding="utf-8")
marca = "## Comparação do LLM com o pipeline"
if marca in atual:  # reexecução: substitui o bloco anterior
    atual = atual[: atual.index(marca)].rstrip() + "\n"
p.write_text(atual + texto, encoding="utf-8")
print("OK")
print(linhas)
print(cm)
