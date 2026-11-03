"""utilitário (v0): Triangulação com o survey State of Data Brasil (2021-2024) para reforçar ou contrastar a inclinação anti-RTO.

Etapa: utilitário (v0)
Objetivo: Triangulação com o survey State of Data Brasil (2021-2024) para reforçar ou contrastar a inclinação anti-RTO.
Entradas: data/kaggle/<ano>/*.csv
Saídas: outputs/triangulacao_survey.md, figura em outputs/figuras_v0/
Como rodar: python -m scripts.utilitarios.survey_triangulation
Tempo aproximado: segundos
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import FIGURES_DIR, OUTPUTS_DIR

SURVEYS = {
    2021: "data/kaggle/2021/State of Data 2021 - Dataset - Pgina1.csv",
    2022: "data/kaggle/2022/State_of_data_2022.csv",
    2023: "data/kaggle/2023/State_of_data_BR_2023_Kaggle - df_survey_2023.csv",
    2024: "data/kaggle/2024/Final Dataset - State of Data 2024 - Kaggle - df_survey_2024.csv",
}
# Obs.: a edição 2019 (pré-pandemia) não traz a pergunta de modelo de trabalho.

# As perguntas mudam de código/nome a cada ano — buscamos por possíveis trechos.
Q_ATUAL = ["atualmente qual a sua forma de trabalho", "modelo_de_trabalho_atual"]
Q_IDEAL = ["forma de trabalho ideal", "modelo_de_trabalho_ideal"]
Q_ATITUDE = [
    "caso sua empresa decida pelo modelo 100% presencial",
    "atitude_em_caso_de_retorno_presencial",
]

MODEL_ORDER = ["100% remoto", "Híbrido flexível", "Híbrido fixo", "100% presencial"]
MODEL_COLORS = {
    "100% remoto": "#2e86c1",
    "Híbrido flexível": "#48c9b0",
    "Híbrido fixo": "#f5b041",
    "100% presencial": "#c0392b",
}
ATT_ORDER = ["Procuraria outro emprego", "Aceitaria o presencial"]
ATT_COLORS = {
    "Procuraria outro emprego": "#2e86c1",
    "Aceitaria o presencial": "#c0392b",
}


def _find_col(df: pd.DataFrame, needles) -> str | None:
    if isinstance(needles, str):
        needles = [needles]
    for n in needles:
        for c in df.columns:
            if n in str(c).lower():
                return c
    return None


def _norm_model(s) -> str | None:
    s = str(s).lower()
    if "100% remoto" in s:
        return "100% remoto"
    if "flex" in s and ("híbrido" in s or "hibrido" in s):
        return "Híbrido flexível"
    if "dias fixos" in s or ("híbrido" in s and "fixo" in s):
        return "Híbrido fixo"
    if "100% presencial" in s:
        return "100% presencial"
    return None


def _norm_attitude(s) -> str | None:
    s = str(s).lower()
    if "aceitar" in s and "retornar" in s:
        return "Aceitaria o presencial"
    if "procurar" in s:
        return "Procuraria outro emprego"
    return None


def _dist(df, needle, normfn, order) -> dict:
    col = _find_col(df, needle)
    if col is None:
        return {}
    vals = df[col].map(normfn).dropna()
    pct = vals.value_counts(normalize=True) * 100
    return {k: round(pct.get(k, 0.0), 1) for k in order}


def build_tables():
    atual, ideal, atitude = {}, {}, {}
    n_resp = {}
    for yr, path in SURVEYS.items():
        if not Path(path).exists():
            print(f"[aviso] survey {yr} não encontrado: {path}")
            continue
        df = pd.read_csv(path, low_memory=False)
        n_resp[yr] = len(df)
        atual[yr] = _dist(df, Q_ATUAL, _norm_model, MODEL_ORDER)
        ideal[yr] = _dist(df, Q_IDEAL, _norm_model, MODEL_ORDER)
        atitude[yr] = _dist(df, Q_ATITUDE, _norm_attitude, ATT_ORDER)
    return (
        pd.DataFrame(atual).T,
        pd.DataFrame(ideal).T,
        pd.DataFrame(atitude).T,
        n_resp,
    )


def _stacked(ax, tab, order, colors, title):
    left_bottom = [0] * len(tab)
    x = [str(i) for i in tab.index]
    for cat in order:
        if cat not in tab.columns:
            continue
        ax.bar(x, tab[cat], bottom=left_bottom, label=cat, color=colors[cat])
        left_bottom = [a + b for a, b in zip(left_bottom, tab[cat])]
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("% das respostas")
    ax.set_ylim(0, 100)


def make_figure(atual, ideal, atitude) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    _stacked(axes[0], atual, MODEL_ORDER, MODEL_COLORS, "Forma de trabalho ATUAL")
    _stacked(axes[1], ideal, MODEL_ORDER, MODEL_COLORS, "Forma de trabalho IDEAL")
    _stacked(
        axes[2], atitude, ATT_ORDER, ATT_COLORS, "Se a empresa exigir 100% presencial"
    )
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4, fontsize=8)
    axes[2].legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=8)
    fig.suptitle(
        "Triangulação — State of Data Brazil (profissionais de dados)", fontsize=13
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    path = FIGURES_DIR / "10_triangulacao_survey.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def write_report(atual, ideal, atitude, n_resp, fig_path) -> str:
    lines = ["# Triangulação quantitativa — State of Data Brazil\n"]
    lines.append(
        "Survey anual da comunidade DataHackers com profissionais de dados "
        "no Brasil. Usado para **triangular** o achado de inclinação "
        "anti-RTO observado nas redes sociais.\n"
    )
    lines.append(
        "Respostas por edição: "
        + ", ".join(f"{y}={n}" for y, n in n_resp.items())
        + "\n"
    )

    def _tbl(title, tab):
        out = [f"## {title}\n", tab.to_markdown(), ""]
        return out

    lines += _tbl("Forma de trabalho ATUAL (% por ano)", atual)
    lines += _tbl("Forma de trabalho IDEAL (% por ano)", ideal)
    lines += _tbl("Atitude se a empresa exigir 100% presencial (% por ano)", atitude)

    # destaque
    if not atitude.empty:
        yr = atitude.index.max()
        proc = atitude.loc[yr].get("Procuraria outro emprego", None)
        if proc is not None:
            lines.append(
                f"\n> **Destaque ({yr}):** {proc:.1f}% dos profissionais "
                "**procurariam outro emprego** caso fossem obrigados ao "
                "modelo 100% presencial — convergente com a inclinação "
                "anti-RTO observada nas redes sociais.\n"
            )
    if not ideal.empty:
        yr = ideal.index.max()
        pres = ideal.loc[yr].get("100% presencial", None)
        if pres is not None:
            lines.append(
                f"> Apenas **{pres:.1f}%** consideram o 100% presencial o "
                "modelo **ideal** ({yr}).\n".replace("{yr}", str(yr))
            )

    lines.append(f"\n![triangulacao](figures/{Path(fig_path).name})\n")

    path = OUTPUTS_DIR / "triangulacao_survey.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    # CSVs
    (OUTPUTS_DIR / "examples").mkdir(parents=True, exist_ok=True)
    atual.to_csv(OUTPUTS_DIR / "examples/survey_forma_atual.csv", encoding="utf-8-sig")
    ideal.to_csv(OUTPUTS_DIR / "examples/survey_forma_ideal.csv", encoding="utf-8-sig")
    atitude.to_csv(
        OUTPUTS_DIR / "examples/survey_atitude_presencial.csv", encoding="utf-8-sig"
    )
    return str(path)


def main():
    atual, ideal, atitude, n_resp = build_tables()
    print("ATUAL:\n", atual, "\n\nIDEAL:\n", ideal, "\n\nATITUDE:\n", atitude)
    fig = make_figure(atual, ideal, atitude)
    report = write_report(atual, ideal, atitude, n_resp, fig)
    print(f"\nFigura: {fig}\nRelatório: {report}")


if __name__ == "__main__":
    main()
