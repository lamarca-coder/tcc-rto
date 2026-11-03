"""B2: Amostra de 500 textos do v0 para rotulagem humana (semente 42) e dupla anotação de 150 (semente 43).

Etapa: B2
Objetivo: Amostra de 500 textos do v0 para rotulagem humana (semente 42) e dupla anotação de 150 (semente 43).
Entradas: dados/v0/corpus_v0.parquet, data/rto.db
Saídas: rotulagem/planilha_rotulagem.xlsx, gabarito_modelo.parquet, mapa_ids.parquet, ids_dupla_anotacao.csv
Como rodar: python -m scripts.b_validacao.b2_amostra_rotulagem
Tempo aproximado: segundos
"""

import sqlite3

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font
from openpyxl.worksheet.datavalidation import DataValidation

SEED = 42
corpus = pd.read_parquet("dados/v0/corpus_v0.parquet")
plats = ["youtube", "reddit"]
log = {}

# ---- Amostra A: 200 por plataforma, 67 pos / 67 neg / 66 neu ----
parts = []
for p in plats:
    for lab, n in (("positivo", 67), ("negativo", 67), ("neutro", 66)):
        pool = corpus[(corpus.platform == p) & (corpus.sentiment_label == lab)]
        parts.append(pool.sample(n=n, random_state=SEED))
A = pd.concat(parts).assign(amostra="A")
log["A"] = A.groupby(["platform", "sentiment_label"]).size().to_dict()

# ---- Amostra B: target=remoto & negativo, 50 por plataforma, fora de A ----
parts = []
for p in plats:
    pool = corpus[
        (corpus.platform == p)
        & (corpus.target == "remoto")
        & (corpus.sentiment_label == "negativo")
        & ~corpus.id.isin(A.id)
    ]
    n = min(50, len(pool))
    log[f"B_{p}_disponiveis"], log[f"B_{p}_usados"] = len(pool), n
    parts.append(pool.sample(n=n, random_state=SEED))
B = pd.concat(parts).assign(amostra="B")

S = pd.concat([A, B])
assert S.id.is_unique
S = S.sample(frac=1, random_state=SEED).reset_index(drop=True)
S["id_rotulagem"] = np.arange(1, len(S) + 1)

# ---- contexto: título do vídeo/post ----
con = sqlite3.connect("data/rto.db")
titles = dict(
    con.execute(
        "SELECT id, context_text FROM posts WHERE kind='post' AND context_text IS NOT NULL"
    )
)


def ctx(r):
    if isinstance(r.context_text, str) and r.context_text.strip():
        return r.context_text
    if isinstance(r.parent_id, str) and r.parent_id in titles:
        return titles[r.parent_id]
    return r.source if r.platform == "youtube" else ""


S["contexto"] = S.apply(ctx, axis=1)
log["contexto_vazio"] = int((S.contexto == "").sum())

# ---- planilha (sem rótulos do modelo, sem author_hash) ----
wb = Workbook()
ws = wb.active
ws.title = "rotulagem"
cols = [
    "id_rotulagem",
    "plataforma",
    "texto",
    "contexto",
    "sentimento_humano",
    "alvo_humano",
    "posicionamento_humano",
    "observacao",
]
ws.append(cols)
for r in S.itertuples():
    ws.append([r.id_rotulagem, r.platform, r.text, r.contexto, None, None, None, None])
for c in ws[1]:
    c.font = Font(bold=True)
n = len(S) + 1
opts = {
    "E": ["positivo", "negativo", "neutro", "irrelevante"],
    "F": ["presencial", "remoto", "híbrido", "geral"],
    "G": ["pró-RTO", "anti-RTO", "neutro ou misto", "irrelevante"],
}
for col, o in opts.items():
    dv = DataValidation(type="list", formula1='"' + ",".join(o) + '"', allow_blank=True)
    dv.error = "Escolha uma opção da lista."
    dv.showErrorMessage = True
    ws.add_data_validation(dv)
    dv.add(f"{col}2:{col}{n}")
for col, w in zip("ABCDEFGH", (14, 11, 80, 45, 18, 14, 20, 30)):
    ws.column_dimensions[col].width = w
for row in ws.iter_rows(min_row=2, min_col=3, max_col=4):
    for c in row:
        c.alignment = Alignment(wrap_text=True, vertical="top")
ws.freeze_panes = "A2"
wb.save("rotulagem/planilha_rotulagem.xlsx")

# ---- gabarito do modelo ----
S[["id_rotulagem", "sentiment_label", "target", "stance"]].to_parquet(
    "rotulagem/gabarito_modelo.parquet", index=False
)
# mapa auxiliar (rastreabilidade; NÃO entregar ao anotador)
S[["id_rotulagem", "id", "amostra", "platform"]].to_parquet(
    "rotulagem/mapa_ids.parquet", index=False
)

# ---- dupla anotação: 150 entre as 400 da amostra A (semente 43) ----
dup = S[S.amostra == "A"].sample(n=150, random_state=43).id_rotulagem.sort_values()
dup.to_csv("rotulagem/ids_dupla_anotacao.csv", index=False)
log["dupla_por_plataforma"] = (
    S[S.id_rotulagem.isin(dup)].platform.value_counts().to_dict()
)
log["total"] = len(S)
log["por_plataforma"] = S.platform.value_counts().to_dict()
log["por_amostra"] = S.amostra.value_counts().to_dict()
log["modelo_stance_amostra"] = S.groupby("amostra").stance.value_counts().to_dict()
print(log)
