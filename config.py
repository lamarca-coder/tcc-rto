"""
Configurações centrais do projeto.

Análise de percepções públicas sobre o retorno ao trabalho presencial (RTO)
a partir de dados de redes sociais.

Aluno: Marcos Eduardo Lamarca Nunes
Orientador: Christian Eric Barrantes Briceño
MBA em Data Science & Analytics para Operações
"""
from pathlib import Path

# --------------------------------------------------------------------------- #
# Caminhos                                                                     #
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figuras_v0"

# Banco de dados SQLite (conforme metodologia do projeto)
DB_PATH = DATA_DIR / "rto.db"

# Arquivos parquet (armazenamento estruturado, conforme metodologia)
RAW_PARQUET = PROCESSED_DIR / "posts_raw.parquet"
PROCESSED_PARQUET = PROCESSED_DIR / "posts_processed.parquet"


def ensure_dirs() -> None:
    """Garante que todas as pastas de saída existam."""
    for d in (RAW_DIR, PROCESSED_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Palavras-chave e fontes (conforme metodologia)                              #
# --------------------------------------------------------------------------- #
KEYWORDS_PT = [
    "home office",
    "trabalho remoto",
    "retorno ao escritório",
    "retorno ao trabalho presencial",
    "trabalho presencial",
    "trabalho híbrido",            # mantido: ancorado em "trabalho"
    "volta ao escritório",
    "volta ao trabalho presencial",
    "fim do home office",
    "obrigado a voltar ao escritório",
    # termos mais orientados a opinião/debate (ampliam a coleta com menos ruído)
    "fim do trabalho remoto",
    "acabou o home office",
    "empresas acabando com home office",
    "trabalho remoto vs presencial",
    "produtividade home office",
    "obrigando funcionários a voltar",
]
# Removido "modelo híbrido": é ambíguo e puxava conteúdo sobre carros híbridos.

KEYWORDS_EN = [
    "return to office",
    "RTO",
    "work from home",
    "remote work",
    "hybrid work",
    "back to office",
]

# Foco inicial: Brasil / português. (KEYWORDS_EN mantido apenas para referência.)
KEYWORDS = KEYWORDS_PT

# Subreddits brasileiros (trabalho, carreira, desabafos, tecnologia, regionais)
SUBREDDITS = [
    "brasil",
    "brdev",
    "desabafos",
    "conversas",
    "investimentos",
    "empreendedorismo",
    "Tudosobreinternet",
    "brazil",
    # comunidades regionais (permitem um recorte de "região por comunidade")
    "saopaulo",
    "riodejaneiro",
    "financasbr",
    "brasilivre",
]

# Idiomas processados pelo pipeline (foco em PT no recorte Brasil)
LANGUAGES = ("pt",)

# --------------------------------------------------------------------------- #
# Parâmetros de análise (valores-padrão; ajustáveis via CLI)                  #
# --------------------------------------------------------------------------- #
N_TOPICS = 6                 # nº de tópicos do LDA
SENTIMENT_THRESHOLD = 0.05   # limiar para classificar neutro vs. pos/neg
RANDOM_STATE = 42            # reprodutibilidade

# Granularidade da análise temporal: "W" (semana) ou "M" (mês).
# Para a janela longa (2019-2026) usamos mês.
TIME_FREQ = "M"

# Janela temporal de interesse — do pré-pandemia até hoje.
# Permite observar a evolução das percepções (remoto forçado -> RTO).
WINDOW_START = "2019-06-01"
WINDOW_END = "2026-06-21"
