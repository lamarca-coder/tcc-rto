"""
Analisador de sentimento léxico para português (PT-BR).

Abordagem leve, sem dependências pesadas, usada por padrão para textos em
português (o VADER é treinado para inglês). Faz contagem de termos positivos e
negativos com tratamento simples de negação ("não gosto" -> negativo).

Para análises de maior precisão, o módulo `sentiment.py` permite trocar para um
backend baseado em transformers (pysentimiento) — veja requirements-optional.txt.
"""
from __future__ import annotations

import math
import re

# Léxicos compactos orientados ao domínio (trabalho remoto/presencial) + termos
# gerais de polaridade. Não pretendem ser exaustivos; servem à análise preliminar.
POSITIVE = {
    "bom", "boa", "ótimo", "ótima", "otimo", "otima", "excelente", "melhor",
    "melhorou", "melhora", "gosto", "gostei", "adoro", "adorei", "amo", "amei",
    "feliz", "felicidade", "ganho", "ganhei", "ganhou", "vantagem", "vantagens",
    "produtivo", "produtiva", "produtividade", "eficiente", "eficiência",
    "tranquilo", "tranquila", "equilíbrio", "equilibrio", "qualidade",
    "confiança", "confianca", "confia", "flexível", "flexivel", "flexibilidade",
    "liberdade", "autonomia", "maravilha", "perfeito", "satisfeito", "satisfação",
    "satisfacao", "positivo", "recomendo", "sucesso", "conforto", "confortável",
    "confortavel", "prazer", "animado", "agradável", "agradavel", "fortaleceu",
    "apoio", "benefício", "beneficio", "ótimos", "felizes", "vale", "incrível",
    "incrivel", "amor", "valeu", "devolveu", "respeito", "respeita",
}

NEGATIVE = {
    "ruim", "péssimo", "pessimo", "péssima", "pessima", "horrível", "horrivel",
    "terrível", "terrivel", "pior", "piorou", "piora", "odeio", "odiei",
    "detesto", "detestei", "desumano", "cansativo", "cansaço", "cansaco",
    "cansada", "cansado", "estresse", "estressado", "estressante", "ansiedade",
    "ansioso", "ansiosa", "isolado", "isolada", "isolamento", "burnout",
    "exaustão", "exaustao", "exausto", "infernal", "inferno", "problema",
    "problemas", "difícil", "dificil", "dificuldade", "prejuízo", "prejuizo",
    "perda", "perdi", "desconfiança", "desconfianca", "forçar", "forcar",
    "forçado", "forcado", "imposição", "imposicao", "descaso", "frustrado",
    "frustração", "frustracao", "insatisfeito", "insatisfação", "insatisfacao",
    "negativo", "reclamar", "reclamação", "reclamacao", "sofrimento", "fortuna",
    "falta", "gasto", "gastar", "demissão", "demissao", "demitir", "triste",
    "tristeza", "raiva", "péssimos", "cansativa", "ignorou", "ignora", "sem",
}

NEGATIONS = {"não", "nao", "nunca", "nem", "jamais", "nenhum", "nenhuma"}

_TOKEN_RE = re.compile(r"\b[\wà-úãõâêôçáéíóúü]+\b", re.IGNORECASE)

# Quantos tokens à frente uma negação afeta
_NEGATION_WINDOW = 3
_ALPHA = 4.0  # constante de normalização (estilo VADER)


def score_pt(text: str) -> float:
    """
    Retorna um escore composto em [-1, 1] para um texto em português.
    Valores > 0 indicam polaridade positiva; < 0, negativa.
    """
    if not text:
        return 0.0
    tokens = _TOKEN_RE.findall(text.lower())
    pos = neg = 0
    negate_until = -1

    for i, tok in enumerate(tokens):
        if tok in NEGATIONS:
            negate_until = i + _NEGATION_WINDOW
            continue
        polarity = 0
        if tok in POSITIVE:
            polarity = 1
        elif tok in NEGATIVE:
            polarity = -1
        if polarity == 0:
            continue
        # inverte a polaridade se estiver sob efeito de uma negação recente
        if i <= negate_until:
            polarity = -polarity
        if polarity > 0:
            pos += 1
        else:
            neg += 1

    net = pos - neg
    if net == 0:
        return 0.0
    return net / math.sqrt(net * net + _ALPHA)
