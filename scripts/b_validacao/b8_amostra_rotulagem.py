"""B8: Amostra do Bluesky para rotulagem (150 linhas, semente 42) e planilha da segunda anotadora (50, semente 43).

Etapa: B8
Objetivo: Amostra do Bluesky para rotulagem (150 linhas, semente 42) e planilha da segunda anotadora (50, semente 43). Instruções: livro de códigos v3.
Entradas: dados/v2/corpus_v2.parquet, rotulagem/mapa_ids.parquet
Saídas: planilhas do B8 e b8_mapa_ids.parquet, b8_gabarito_modelo.parquet em rotulagem/, resultados/B8_amostra.md
Como rodar: python -m scripts.b_validacao.b8_amostra_rotulagem
Tempo aproximado: segundos
"""

from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font
from openpyxl.worksheet.datavalidation import DataValidation

from scripts.utilitarios import util_v2 as u

R = u.ROOT / "rotulagem"
LIVRO = Path(
    r"D:\DADOS DISCO C\DELL\Downloads\Livro de códigos v1 rotulagem do corpus sobre retorno ao trabalho presencial.docx"
)
OPTS = {
    "F": ["positivo", "negativo", "neutro", "irrelevante"],
    "G": ["presencial", "remoto", "hibrido", "geral", "irrelevante"],
    "H": ["pro_RTO", "anti_RTO", "neutro_misto", "irrelevante"],
}
REGRAS = [
    "5. Relevância estrita: só é relevante quando o autor avalia, defende, critica ou compara um modelo de trabalho ou a obrigação de voltar. Perguntas práticas, moradia, mudança de cidade, ergonomia, busca de vaga ou salário e remoto citado de passagem são irrelevantes.",
    "6. anti_RTO precisa de juízo geral a favor do remoto ou contra o presencial. Relato de conveniência pessoal sem generalizar é neutro_misto. O mesmo vale, invertido, para pro_RTO.",
]

# ---------------- amostra ----------------
c = pd.read_parquet(
    u.CORPUS,
    columns=[
        "id",
        "platform",
        "kind",
        "text",
        "created_utc",
        "sentiment_label",
        "target",
        "stance_heuristica",
    ],
)
bs = c[c.platform == "bluesky"].reset_index(drop=True)
amostra = bs.sample(n=150, random_state=42).reset_index(
    drop=True
)  # amostragem aleatória simples
amostra["id_rotulagem"] = np.arange(1001, 1151)
segunda_ids = set(
    amostra.sample(n=50, random_state=43).id_rotulagem
)  # 50 das 150, semente 43
amostra["segunda"] = amostra.id_rotulagem.isin(segunda_ids)
amostra["tipo"] = amostra.kind.map({"post": "post", "comment": "comentário"})


# ---------------- texto da aba Instruções (livro v2 -> v3) ----------------
def texto_livro():
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    d = docx.Document(str(LIVRO))
    linhas = []
    for el in d.element.body.iterchildren():
        if el.tag.endswith("}p"):
            t = Paragraph(el, d).text.strip()
            linhas.append(t)
        elif el.tag.endswith("}tbl"):
            for row in Table(el, d).rows:
                linhas.append(
                    " | ".join(cel.text.strip().replace("\n", " ") for cel in row.cells)
                )
    linhas = [x for x in linhas if x != "" or True]
    # título v3
    assert linhas[0].startswith("Livro de códigos v2")
    linhas[0] = linhas[0].replace("v2", "v3", 1)
    # data/origem das regras novas: depois do texto atual (que termina com a regra 4 e seu exemplo)
    while linhas and linhas[-1] == "":
        linhas.pop()
    linhas += [
        "",
        "Regras acrescentadas na v3 (28/10/2026)",
        "Estas regras valem para todos os anotadores, incluindo o anotador LLM. Numeração continua a das regras da v2.",
    ] + REGRAS
    return linhas


def planilha(df, caminho, instrucoes):
    wb = Workbook()
    ws = wb.active
    ws.title = "rotulagem"
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
    ws.append(cols)
    for r in df.itertuples():
        ws.append(
            [r.id_rotulagem, r.platform, r.tipo, r.text, None, None, None, None, None]
        )  # contexto vazio
    for cel in ws[1]:
        cel.font = Font(bold=True)
    n = len(df) + 1
    for col, o in OPTS.items():
        dv = DataValidation(
            type="list", formula1='"' + ",".join(o) + '"', allow_blank=True
        )
        dv.error = "Escolha uma opção da lista."
        dv.errorTitle = "Valor inválido"
        dv.showErrorMessage = True
        ws.add_data_validation(dv)
        dv.add(f"{col}2:{col}{n}")
    for col, w in zip("ABCDEFGHI", (14, 11, 12, 80, 45, 18, 14, 20, 30)):
        ws.column_dimensions[col].width = w
    for row in ws.iter_rows(min_row=2, min_col=4, max_col=5):
        for cel in row:
            cel.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "A2"
    wi = wb.create_sheet("Instruções")
    wi.column_dimensions["A"].width = 130
    for s in instrucoes:
        wi.append([s])
        wi.cell(row=wi.max_row, column=1).alignment = Alignment(
            wrap_text=True, vertical="top"
        )
    wi["A1"].font = Font(bold=True, size=13)
    wb.save(caminho)


instr = texto_livro()
planilha(amostra, R / "b8_planilha_rotulagem.xlsx", instr)
planilha(
    amostra[amostra.segunda].sort_values("id_rotulagem"),
    R / "b8_planilha_segunda.xlsx",
    instr,
)
amostra[
    ["id_rotulagem", "id", "sentiment_label", "target", "stance_heuristica"]
].to_parquet(R / "b8_gabarito_modelo.parquet", index=False)
amostra[["id_rotulagem", "id", "platform", "segunda"]].assign(amostra="B8").rename(
    columns={"segunda": "segunda_anotadora"}
).to_parquet(R / "b8_mapa_ids.parquet", index=False)

# ---------------- relatório ----------------
SENT = ["negativo", "neutro", "positivo"]
ano = lambda s: pd.to_datetime(s, utc=True, format="mixed", errors="coerce").dt.year
bs["ano"] = ano(bs.created_utc)
amostra["ano"] = ano(amostra.created_utc)
assert bs.ano.notna().all() and amostra.ano.notna().all()


vc = lambda x, d: f"{x:.{d}f}".replace(".", ",")


def gof(obs, pop, B=10000, seed=42):
    """Aderência da amostra à população: χ² e p por simulação multinomial (n = 150; frequências esperadas pequenas em alguns anos)."""
    p = pop / pop.sum()
    n = obs.sum()
    esp = p * n
    est = float(((obs - esp) ** 2 / esp).sum())
    sims = np.random.default_rng(seed).multinomial(n, p, size=B)
    chi_s = ((sims - esp) ** 2 / esp).sum(axis=1)
    return est, float((chi_s >= est).mean()), float(esp.min())


ps = bs.sentiment_label.value_counts().reindex(SENT, fill_value=0).values.astype(float)
os_ = (
    amostra.sentiment_label.value_counts()
    .reindex(SENT, fill_value=0)
    .values.astype(float)
)
est_s, p_s, emin_s = gof(os_, ps)
anos = sorted(bs.ano.unique())
pa = bs.ano.value_counts().reindex(anos, fill_value=0).values.astype(float)
oa = amostra.ano.value_counts().reindex(anos, fill_value=0).values.astype(float)
est_a, p_a, emin_a = gof(oa, pa)
pst = bs.stance_heuristica.value_counts()
os2 = amostra.stance_heuristica.value_counts()

L = [
    "# B8: amostra do Bluesky para rotulagem",
    "",
    "## O que foi feito",
    "Amostra aleatória simples de **150** linhas de `platform = bluesky` do corpus v2 (36.291 posts), semente 42, `id_rotulagem` de 1001 a 1150. Planilha `rotulagem/b8_planilha_rotulagem.xlsx` com as colunas e as listas suspensas da planilha principal (sentimento: positivo, negativo, neutro, irrelevante; alvo: presencial, remoto, hibrido, geral, irrelevante; posicionamento: pro_RTO, anti_RTO, neutro_misto, irrelevante), `contexto` vazio, sem rótulos do modelo e sem `author_hash`. Sorteio de **50** das 150 (semente 43) para a segunda anotadora: `rotulagem/b8_planilha_segunda.xlsx` (só essas 50 linhas, mesmos `id_rotulagem`). Script `b8_amostra_rotulagem.py`.",
    "",
    "## Arquivos",
    "- `rotulagem/b8_planilha_rotulagem.xlsx` (150 linhas) e `rotulagem/b8_planilha_segunda.xlsx` (50 linhas); contêm textos de usuários, **não versionar**.",
    "- `rotulagem/b8_gabarito_modelo.parquet`: `id_rotulagem`, `id` (id do corpus), `sentiment_label`, `target`, `stance_heuristica`. **Não abrir antes de rotular.**",
    "- `rotulagem/b8_mapa_ids.parquet`: `id_rotulagem`, `id`, `platform`, `segunda_anotadora`, `amostra`.",
    "- Os dois parquet têm o `id` do corpus, então entram na lista de **proibidos no repositório público** (resultados/REORG_status.md).",
    "",
    "## Aba Instruções (livro de códigos v3)",
    'O texto atual usado é o do livro **v2** (`Livro de códigos v1 rotulagem do corpus sobre retorno ao trabalho presencial.docx` em Downloads, modificado em 27/10/2026, cujo título interno é "Livro de códigos v2", com as regras 1 a 4 acrescentadas em 24/10/2026). O título foi trocado para "Livro de códigos v3" e, abaixo do texto, foram acrescentadas as regras 5 e 6 (Relevância estrita; anti_RTO precisa de juízo geral), com o texto que você enviou. **As planilhas anteriores (B2, B3, B4) tinham na aba Instruções o texto do PDF v1**, sem as regras 1 a 4 da v2.',
    "",
    "## Sentimento do modelo: amostra x Bluesky inteiro",
    "",
]
linhas = [
    [
        s,
        u.num(ps[i]),
        f"{ps[i] / ps.sum() * 100:.1f}%".replace(".", ","),
        u.num(os_[i]),
        u.pct_ic(int(os_[i]), 150),
    ]
    for i, s in enumerate(SENT)
]
L += u.md_tabela(
    [
        "Sentimento (BERT)",
        "Bluesky inteiro (n)",
        "Bluesky inteiro (%)",
        "Amostra (n)",
        "Amostra (% [IC 95% Wilson])",
    ],
    linhas,
)
L += [
    "",
    f"Aderência da amostra à distribuição do Bluesky inteiro: χ² = {vc(est_s, 2)}, p simulado = {vc(p_s, 3)} (n = 150; menor frequência esperada = {vc(emin_s, 1)}). "
    + (
        "As distribuições são parecidas (não há evidência de diferença)."
        if p_s > 0.05
        else "**Atenção: há diferença entre as distribuições.**"
    ),
    "",
    "## Distribuição por ano",
    "",
]
linhas = [
    [
        int(a),
        u.num(pa[i]),
        f"{pa[i] / pa.sum() * 100:.1f}%".replace(".", ","),
        u.num(oa[i]),
        f"{oa[i] / oa.sum() * 100:.1f}%".replace(".", ","),
    ]
    for i, a in enumerate(anos)
]
L += u.md_tabela(
    ["Ano", "Bluesky inteiro (n)", "Bluesky inteiro (%)", "Amostra (n)", "Amostra (%)"],
    linhas,
)
L += [
    "",
    f"Aderência por ano: χ² = {vc(est_a, 2)}, p simulado = {vc(p_a, 3)}; a menor frequência esperada é {vc(emin_a, 1)} (anos com poucos posts, como 2023), então o χ² por si só é aproximado e o p foi obtido por simulação multinomial (10.000 sorteios, semente 42). "
    + (
        "Sem evidência de diferença entre a amostra e o Bluesky inteiro."
        if p_a > 0.05
        else "**Atenção: a distribuição por ano da amostra difere do Bluesky inteiro.**"
    ),
    "",
    "## Heurística de posicionamento do modelo na amostra (referência)",
    "",
]
L += u.md_tabela(
    [
        "Posicionamento (heurística)",
        "Bluesky inteiro (%)",
        "Amostra (n)",
        "Amostra (%)",
    ],
    [
        [
            k,
            f"{pst.get(k, 0) / pst.sum() * 100:.1f}%".replace(".", ","),
            int(os2.get(k, 0)),
            f"{os2.get(k, 0) / 150 * 100:.1f}%".replace(".", ","),
        ]
        for k in ["anti-RTO", "pró-RTO", "neutro", "indefinido"]
    ],
)
L += [
    "",
    "## Problemas e decisões",
    "- Todos os 150 textos são `post` (o Bluesky só tem posts); `tipo` = post. `contexto` ficou vazio, como pedido (o corpus não guarda contexto para o Bluesky).",
    "- Amostragem aleatória simples: não é estratificada. Com n = 150, o IC de uma proporção é largo (cerca de ±8 pontos); a validação do Bluesky terá pouca precisão em classes pequenas.",
    "- Segunda anotadora: 50 das 150, sorteadas com semente 43 entre as 150 (não entre as anotadas pelo primeiro anotador), conforme pedido.",
    "- O `id` do corpus nos dois parquet é o hash derivado da URI da publicação (não é o id original da plataforma), mas permite conferir a publicação a quem tiver a URI.",
    "- Nenhum modelo foi carregado; o D5 continua rodando.",
]
open(u.RES / "B8_amostra.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print(
    "B8 OK",
    len(amostra),
    int(amostra.segunda.sum()),
    round(p_s, 3),
    round(p_a, 3),
    len(instr),
)
