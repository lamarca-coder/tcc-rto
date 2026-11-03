"""
Gerador de dados de EXEMPLO (sintéticos).

ATENÇÃO: estes dados NÃO são reais. Servem apenas para:
  - demonstrar o pipeline de ponta a ponta sem depender de chaves de API;
  - validar as etapas de pré-processamento, sentimento, tópicos e análise
    temporal antes da coleta real.

Os textos são montados a partir de "temas" + "enquadramentos" (positivo,
negativo, neutro) sobre o retorno ao trabalho presencial, em PT e EN, com
datas distribuídas ao longo do tempo — de modo que as análises exploratórias
produzam estrutura visível (distribuição de sentimento, tópicos e tendências
temporais). Nenhum resultado obtido a partir deles deve ser reportado como
achado empírico real.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.storage.database import anonymize_author

# --------------------------------------------------------------------------- #
# Blocos de construção de frases por tema e polaridade                        #
# --------------------------------------------------------------------------- #
TEMPLATES_PT = {
    "produtividade": {
        "pos": [
            "Trabalhando de casa eu rendo muito mais, sem interrupções o dia todo.",
            "Minha produtividade no home office disparou, foco total nas entregas.",
            "Em casa consigo entregar mais e com mais qualidade, sem reuniões inúteis.",
        ],
        "neg": [
            "No escritório eu produzo mais, em casa me distraio o tempo todo.",
            "O retorno ao presencial melhorou minha concentração e o ritmo da equipe.",
            "Acho o trabalho presencial mais produtivo, a colaboração flui melhor.",
        ],
        "neu": [
            "A produtividade depende muito do tipo de tarefa, presencial ou remoto.",
            "Tem dias que rendo mais em casa e outros no escritório.",
        ],
    },
    "deslocamento": {
        "pos": [
            "Sem deslocamento eu ganhei duas horas de vida por dia, isso é ótimo.",
            "Acabou o trânsito infernal, o home office me devolveu tempo e dinheiro.",
        ],
        "neg": [
            "Voltar a encarar duas horas de trânsito por dia para ir ao escritório é desumano.",
            "O retorno ao escritório significa gastar fortuna com transporte de novo.",
            "Detesto a volta ao presencial só pelo deslocamento cansativo.",
        ],
        "neu": [
            "Moro perto do escritório, então o deslocamento não muda muito pra mim.",
        ],
    },
    "qualidade_de_vida": {
        "pos": [
            "Consigo almoçar com a família e levar meu filho na escola trabalhando remoto.",
            "Minha qualidade de vida melhorou muito com o trabalho remoto.",
            "Home office me deu equilíbrio entre vida pessoal e profissional.",
        ],
        "neg": [
            "Trabalhar de casa misturou tudo, nunca desligo, minha saúde mental piorou.",
            "O home office me deixou isolado e ansioso, sinto falta do convívio.",
        ],
        "neu": [
            "O impacto na qualidade de vida varia de pessoa para pessoa.",
        ],
    },
    "cultura_organizacional": {
        "pos": [
            "A empresa confia na gente e o modelo híbrido fortaleceu a cultura.",
            "O presencial ajuda a integrar os novatos e fortalecer o time.",
        ],
        "neg": [
            "Forçar o retorno ao escritório só mostra falta de confiança da liderança.",
            "A política de RTO da empresa ignorou totalmente a opinião dos funcionários.",
            "Mandaram todo mundo voltar sem critério, vou procurar outro emprego remoto.",
        ],
        "neu": [
            "A cultura da empresa está em transição entre remoto e presencial.",
        ],
    },
}

TEMPLATES_EN = {
    "productivity": {
        "pos": [
            "Working from home I get so much more done, no pointless meetings all day.",
            "My productivity skyrocketed with remote work, deep focus on real tasks.",
        ],
        "neg": [
            "Honestly I focus better at the office, at home I get distracted constantly.",
            "Return to office actually improved our team's collaboration and pace.",
        ],
        "neu": [
            "Productivity really depends on the task, remote or in person.",
        ],
    },
    "commute": {
        "pos": [
            "No commute means two extra hours of life every single day, love it.",
            "Remote work saved me money and a brutal daily commute.",
        ],
        "neg": [
            "Going back to a two hour commute for RTO is absolutely soul crushing.",
            "Return to office means wasting money and time on transport again.",
        ],
        "neu": [
            "I live close to the office so the commute is not a big deal for me.",
        ],
    },
    "wellbeing": {
        "pos": [
            "I can have lunch with my family and pick up my kid thanks to remote work.",
            "Work from home gave me real work life balance, my wellbeing improved.",
        ],
        "neg": [
            "Working from home blurred everything, I never log off, burnout is real.",
            "Remote work left me isolated and anxious, I miss being around people.",
        ],
        "neu": [
            "The wellbeing impact varies a lot from person to person.",
        ],
    },
    "company_culture": {
        "pos": [
            "Our hybrid model strengthened the culture, leadership actually trusts us.",
            "Being in person helps onboard new hires and build the team.",
        ],
        "neg": [
            "Forcing return to office just shows leadership does not trust employees.",
            "The RTO mandate ignored every single piece of employee feedback.",
            "They ordered everyone back with no reason, I'm finding a remote job.",
        ],
        "neu": [
            "The company culture is in transition between remote and in office.",
        ],
    },
}

POLARITIES = ["pos", "neg", "neu"]


def polarity_weights_for(dt) -> list[float]:
    """
    Pesos [pos, neg, neu] que variam ao longo do tempo, simulando a evolução
    do debate (apenas para os dados de exemplo):
      - pré-pandemia (<=2019): tema pouco saliente, mais neutro;
      - pandemia (2020-2021): home office bem avaliado (mais positivo);
      - transição híbrida (2022): equilíbrio;
      - onda de RTO (2023-2024): tensão, mais negativo;
      - 2025-2026: ainda tenso, levemente negativo.
    """
    y = dt.year
    if y <= 2019:
        return [0.25, 0.30, 0.45]
    if y in (2020, 2021):
        return [0.50, 0.25, 0.25]
    if y == 2022:
        return [0.40, 0.30, 0.30]
    if y in (2023, 2024):
        return [0.25, 0.55, 0.20]
    return [0.30, 0.45, 0.25]

# Prefixos e sufixos NEUTROS usados para compor frases variadas (evita que a
# deduplicação colapse o corpus e aproxima o texto da linguagem de redes sociais).
PREFIXES_PT = [
    "", "Sinceramente,", "Na minha experiência,", "Pra ser honesto,",
    "Depois de meses nesse modelo,", "Falando como funcionário,", "Olha,",
    "No fim das contas,", "Resumindo:", "Cá entre nós,", "Hoje em dia,",
]
SUFFIXES_PT = [
    "", "e isso muda tudo no dia a dia.", "mas cada um sente de um jeito.",
    "é o que tenho percebido por aqui.", "pelo menos no meu caso.",
    "e a empresa precisa ouvir isso.", "ninguém comenta, mas é real.",
    "depois dessa pandemia toda.", "e ponto final.", "#trabalho #rto",
]
PREFIXES_EN = [
    "", "Honestly,", "In my experience,", "To be fair,",
    "After months of this,", "Speaking as an employee,", "Look,",
    "At the end of the day,", "Long story short:", "Between us,", "These days,",
]
SUFFIXES_EN = [
    "", "and it changes everything day to day.", "but everyone feels it differently.",
    "at least that's my case.", "and companies need to hear this.",
    "nobody talks about it but it's real.", "after this whole pandemic.",
    "and that's that.", "just my two cents.", "#work #rto",
]


def _compose(core: str, lang: str, rng: random.Random) -> str:
    """Combina prefixo + frase-núcleo + sufixo para gerar texto variado."""
    if lang == "pt":
        prefix = rng.choice(PREFIXES_PT)
        suffix = rng.choice(SUFFIXES_PT)
    else:
        prefix = rng.choice(PREFIXES_EN)
        suffix = rng.choice(SUFFIXES_EN)
    parts = [p for p in (prefix, core, suffix) if p]
    return " ".join(parts)


def _pick_text(templates: dict, lang: str, rng: random.Random,
               weights: list[float]) -> tuple[str, str, str]:
    theme = rng.choice(list(templates.keys()))
    polarity = rng.choices(POLARITIES, weights=weights, k=1)[0]
    core = rng.choice(templates[theme][polarity])
    text = _compose(core, lang, rng)
    return theme, polarity, text


def generate_sample(n: int = 1500, seed: int = 42, lang: str = "pt",
                    start: str = "2019-06-01", end: str = "2026-06-21") -> pd.DataFrame:
    """
    Gera `n` registros sintéticos distribuídos entre `start` e `end`.

    Por padrão gera conteúdo em português (recorte Brasil) ao longo da janela
    2019-2026, com a polaridade variando no tempo (ver `polarity_weights_for`).

    Retorna um DataFrame no esquema canônico de `storage.database`.
    """
    rng = random.Random(seed)
    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
    span_seconds = int((end_dt - start_dt).total_seconds())

    templates = TEMPLATES_PT if lang == "pt" else TEMPLATES_EN
    platforms = ["reddit", "twitter"]
    now = datetime.now(timezone.utc).isoformat()
    rows = []

    for i in range(n):
        created = start_dt + timedelta(seconds=rng.randint(0, span_seconds))
        weights = polarity_weights_for(created)
        theme, polarity, text = _pick_text(templates, lang, rng, weights)

        platform = rng.choices(platforms, weights=[0.6, 0.4], k=1)[0]
        author = f"user_{rng.randint(1, 250)}"

        rows.append(
            {
                "id": f"sample_{i:05d}",
                "platform": platform,
                "kind": rng.choice(["post", "comment"]),
                "source": theme,           # tema-semente (apenas referência interna)
                "author_hash": anonymize_author(author),
                "text": text,
                "created_utc": created.isoformat(),
                "score": rng.randint(0, 500),
                "url": "",
                "lang_hint": lang,
                "collected_at": now,
            }
        )

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    sample = generate_sample()
    print(f"Gerados {len(sample)} registros de exemplo.")
    print(sample[["platform", "lang_hint", "source", "text"]].head(10).to_string())
