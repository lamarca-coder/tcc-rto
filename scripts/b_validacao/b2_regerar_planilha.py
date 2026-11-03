"""B2: Regera a planilha de rotulagem (mesma amostra) com novas listas suspensas, coluna tipo e aba Instruções.

Etapa: B2
Objetivo: Regera a planilha de rotulagem (mesma amostra) com novas listas suspensas, coluna tipo e aba Instruções.
Entradas: rotulagem/planilha_rotulagem.xlsx, mapa_ids.parquet, _livro.txt, dados/v0/corpus_v0.parquet
Saídas: rotulagem/planilha_rotulagem.xlsx
Como rodar: python -m scripts.b_validacao.b2_regerar_planilha
Tempo aproximado: segundos
"""

import re

import openpyxl
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font
from openpyxl.worksheet.datavalidation import DataValidation

old = openpyxl.load_workbook("rotulagem/planilha_rotulagem.xlsx").active
hdr = [c.value for c in old[1]]
rows = [dict(zip(hdr, [c.value for c in r])) for r in old.iter_rows(min_row=2)]
ids = [r["id_rotulagem"] for r in rows]
assert ids == list(range(1, 501))

mapa = pd.read_parquet("rotulagem/mapa_ids.parquet").set_index("id_rotulagem")
corpus = pd.read_parquet("dados/v0/corpus_v0.parquet").set_index("id")
kind = {i: corpus.loc[mapa.loc[i, "id"], "kind"] for i in ids}
tipo = {i: ("post" if kind[i] == "post" else "comentário") for i in ids}

cols = [
    "id_rotulagem",
    "plataforma",
    "tipo",
    "texto",
    "contexto",
    "sentimento_humano",
    "alvo_humano",
    "posicionamento_humano",
    "observacao",
]
wb = Workbook()
ws = wb.active
ws.title = "rotulagem"
ws.append(cols)
for r in rows:
    ws.append(
        [
            r["id_rotulagem"],
            r["plataforma"],
            tipo[r["id_rotulagem"]],
            r["texto"],
            r["contexto"],
            None,
            None,
            None,
            None,
        ]
    )
for c in ws[1]:
    c.font = Font(bold=True)
n = len(rows) + 1
OPTS = {
    "F": ["positivo", "negativo", "neutro", "irrelevante"],
    "G": ["presencial", "remoto", "hibrido", "geral", "irrelevante"],
    "H": ["pro_RTO", "anti_RTO", "neutro_misto", "irrelevante"],
}
for col, o in OPTS.items():
    dv = DataValidation(type="list", formula1='"' + ",".join(o) + '"', allow_blank=True)
    dv.error = "Escolha uma opção da lista."
    dv.errorTitle = "Valor inválido"
    dv.showErrorMessage = True
    ws.add_data_validation(dv)
    dv.add(f"{col}2:{col}{n}")
for col, w in zip("ABCDEFGHI", (14, 11, 12, 80, 45, 18, 14, 20, 30)):
    ws.column_dimensions[col].width = w
for row in ws.iter_rows(min_row=2, min_col=4, max_col=5):
    for c in row:
        c.alignment = Alignment(wrap_text=True, vertical="top")
ws.freeze_panes = "A2"

# ---- aba Instruções: texto do livro de códigos v1 ----
txt = open("rotulagem/_livro.txt", encoding="utf-8").read()
lines = []
for ln in txt.splitlines():
    s = ln.strip().replace("\u200b", "")
    if (
        re.match(r"^Page \d+ of \d+$", s)
        or s.startswith("Livro de códigos v1: rotulagem do corpus")
        and lines
    ):
        continue
    lines.append(s)
wi = wb.create_sheet("Instruções")
wi.column_dimensions["A"].width = 130
for s in lines:
    wi.append([s])
    wi.cell(row=wi.max_row, column=1).alignment = Alignment(
        wrap_text=True, vertical="top"
    )
wi["A1"].font = Font(bold=True, size=13)
wb.save("rotulagem/planilha_rotulagem.xlsx")
print(len(lines), "linhas de instruções")
