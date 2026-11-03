"""
Pré-processamento de texto (etapa essencial em projetos de NLP).

Implementa, conforme a metodologia do projeto:
  - remoção de duplicatas;
  - limpeza de caracteres especiais, URLs e elementos irrelevantes;
  - normalização textual (minúsculas);
  - tokenização;
  - remoção de stopwords (PT e EN);
  - stemming (RSLP, PT) / lematização (WordNet, EN).

O código degrada graciosamente: se os recursos do NLTK não estiverem
disponíveis, usa listas de stopwords embutidas e pula o stemming, de modo que
o pipeline rode mesmo sem downloads.
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# Detecção de idioma                                                          #
# --------------------------------------------------------------------------- #
try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0  # torna a detecção determinística
    _HAS_LANGDETECT = True
except Exception:  # pragma: no cover
    _HAS_LANGDETECT = False


def detect_language(text: str, default: str = "pt") -> str:
    """Retorna 'pt' ou 'en' (idiomas suportados); demais caem no default."""
    if not text or len(text.strip()) < 3:
        return default
    if not _HAS_LANGDETECT:
        # heurística simples: presença de acentos/palavras comuns em PT
        pt_markers = (" não ", " que ", " com ", " para ", " você ", "ção", "ã", "õ")
        low = f" {text.lower()} "
        return "pt" if any(m in low for m in pt_markers) else default
    try:
        lang = detect(text)
    except Exception:
        return default
    return lang if lang in ("pt", "en") else default


# --------------------------------------------------------------------------- #
# Stopwords (NLTK + fallback embutido)                                        #
# --------------------------------------------------------------------------- #
_FALLBACK_STOPWORDS = {
    "pt": {
        "a", "o", "e", "é", "de", "do", "da", "dos", "das", "em", "no", "na",
        "nos", "nas", "um", "uma", "uns", "umas", "que", "com", "por", "para",
        "se", "os", "as", "ao", "aos", "à", "às", "ou", "mas", "como", "mais",
        "muito", "já", "também", "só", "pelo", "pela", "até", "isso", "este",
        "esta", "esse", "essa", "eu", "tu", "ele", "ela", "nós", "vós", "eles",
        "elas", "meu", "minha", "seu", "sua", "ser", "estar", "ter", "foi",
        "são", "sem", "sobre", "entre", "depois", "antes", "quando", "porque",
        "todo", "toda", "todos", "todas", "vai", "tem", "não", "sim", "lá", "cá",
    },
    "en": {
        "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on",
        "at", "by", "for", "with", "about", "as", "is", "are", "was", "were",
        "be", "been", "being", "i", "you", "he", "she", "it", "we", "they",
        "my", "your", "his", "her", "its", "our", "their", "this", "that",
        "these", "those", "do", "does", "did", "have", "has", "had", "not",
        "no", "so", "than", "too", "very", "just", "more", "most", "some",
        "any", "can", "will", "would", "should", "could", "from", "up", "out",
        "all", "over", "again", "me", "us", "them", "what", "which", "who",
        "when", "where", "why", "how", "there", "here", "im", "ive", "dont",
    },
}

# Termos genéricos do domínio que poluem os tópicos (opcional, conservador)
_EXTRA_STOPWORDS = {"https", "http", "amp", "rt", "via", "gt", "lt"}

# Fillers coloquiais e ruído típico de comentários do YouTube (só PT).
# Não inclui palavras de conteúdo nem de sentimento (estas só afetariam os
# tópicos; o sentimento é calculado sobre o texto original).
_PT_COLLOQUIAL_STOPWORDS = {
    "pra", "pro", "pras", "pros", "tá", "ta", "né", "ne", "aí", "ai", "vc",
    "vcs", "tipo", "cara", "gente", "pessoal", "tô", "to", "tava", "então",
    "entao", "assim", "coisa", "aqui", "agora", "hoje", "ano", "anos", "dia",
    "dias", "vídeo", "video", "canal", "inscrever", "inscrevam", "inscrito",
    "like", "likes", "comentário", "comentario", "kkk", "kkkk", "rs", "poxa",
    "nossa", "faz", "fazer", "ser", "ter", "vai", "vou", "fica", "ficar",
    "quer", "querer", "deu", "dar", "fala", "falar", "gente", "todo", "toda",
}


def _load_nltk_stopwords() -> dict[str, set]:
    out = {}
    try:
        import nltk
        from nltk.corpus import stopwords
        for lang, name in (("pt", "portuguese"), ("en", "english")):
            try:
                out[lang] = set(stopwords.words(name))
            except LookupError:
                nltk.download("stopwords", quiet=True)
                out[lang] = set(stopwords.words(name))
    except Exception:
        return {}
    return out


_NLTK_SW = _load_nltk_stopwords()
STOPWORDS = {
    lang: (_NLTK_SW.get(lang, set()) | _FALLBACK_STOPWORDS[lang] | _EXTRA_STOPWORDS)
    for lang in ("pt", "en")
}
STOPWORDS["pt"] |= _PT_COLLOQUIAL_STOPWORDS


# --------------------------------------------------------------------------- #
# Filtro de relevância temática                                               #
# --------------------------------------------------------------------------- #
# Mantém apenas comentários ligados a trabalho/escritório e descarta conteúdo
# fora de tema (ex.: carros híbridos/elétricos puxados pela busca por "híbrido")
# e meta-comentários sobre o vídeo. Aplicado sobre o texto original.
WORK_ANCHORS = (
    "trabalho", "trabalhar", "trabalhando", "emprego", "empreg", "escritóri",
    "escritori", "home office", "homeoffice", "remoto", "remota", "presencial",
    "empresa", "chefe", "patrão", "patrao", "clt", "expediente", "funcionári",
    "funcionario", "colaborador", "reunião", "reuniao", "salári", "salario",
    "demiti", "demissão", "demissao", "contrat", "carreira", "profission",
    "equipe", "gestor", "liderança", "lideranca", "recursos humanos",
    "produtividade", "deslocamento", "crachá", "cracha", "carteira assinada",
)
CAR_TERMS = (
    "carro", "carros", "elétrico", "eletrico", "combustão", "combustao",
    "gasolina", "etanol", "veículo", "veiculo", "automóvel", "automovel",
    "quilômetro", "quilometro", "aspirado", "cavalos de potência",
)

# Exclusão por CONTEXTO (título do vídeo/post): se o conteúdo-pai for sobre
# recrutamento, música, INSS/perícia ou EAD/faculdade, descartamos seus
# comentários — são fontes de ruído fora do tema "percepção sobre o RTO".
EXCLUDE_CONTEXT_PATTERNS = (
    # recrutamento / vagas / renda
    "vaga", "vagas", "contratando", "aceita iniciante", "trabalhe em casa",
    "ganhe dinheiro", "renda extra", "sites para trabalhar", "dinheiro online",
    "empresas para trabalhar", "atendente de chat", "como conseguir",
    # música / entretenimento
    "clipe", "música", "musica", "lyric", "cover", "playlist", "lofi",
    "cool music", "beat", " ao vivo", "sua melhor",
    # INSS / perícia / benefícios
    "inss", "perícia", "pericia", "auxílio", "auxilio", "benefício", "beneficio",
    "afastamento", "atestado", "aposentad",
    # EAD / faculdade
    " ead", "faculdade", "vestibular", "enem", "graduação", "graduacao",
    # esportes (ruído pontual)
    "grêmio", "gremio", "escalação", "escalacao",
)


def context_is_excluded(title: str | None) -> bool:
    """True se o título do conteúdo-pai indica tema fora de escopo."""
    if not title:
        return False
    t = str(title).lower()
    return any(p in t for p in EXCLUDE_CONTEXT_PATTERNS)


# Âncoras ESPECÍFICAS do debate RTO (mais restritivas que WORK_ANCHORS).
# Exigir uma destas evita "desabafos"/carreira genéricos que só citam "trabalho".
RTO_ANCHORS = (
    "presencial", "remoto", "remota", "home office", "homeoffice", "home-office",
    "escritóri", "escritori", "híbrido", "hibrido", "teletrabalho",
    "trabalhar de casa", "trabalho de casa", "trabalho a distância",
    "trabalho a distancia", "volta ao escrit", "volta ao trabalho",
    "retorno ao", "no escritório", " rto",
)


def is_relevant(text: str, title: str | None = None) -> bool:
    """
    True se o comentário (ou o título do conteúdo-pai) trata especificamente do
    debate RTO, não é dominado por contexto de carro e o título não cai em tema
    excluído. Exigir uma âncora RTO (e não só palavra de trabalho genérica)
    reduz muito o ruído de desabafos/carreira.
    """
    if context_is_excluded(title):
        return False
    if not isinstance(text, str):
        return False
    t = text.lower()
    # exige a âncora RTO no PRÓPRIO comentário (mais restritivo; o refino
    # semântico por embeddings faz a limpeza fina depois)
    has_rto = any(a in t for a in RTO_ANCHORS)
    car_hits = sum(1 for c in CAR_TERMS if c in t)
    return has_rto and car_hits < 2


def _best_title(row) -> str | None:
    """Escolhe o título do conteúdo-pai: context_text (se houver) ou source."""
    ctx = row.get("context_text")
    if isinstance(ctx, str) and ctx.strip():
        return ctx
    return row.get("source")


def filter_relevant(df: pd.DataFrame, text_col: str = "text"):
    """Retorna (df_filtrado, n_removidos) mantendo só registros relevantes."""
    titles = df.apply(_best_title, axis=1) if len(df) else []
    mask = [is_relevant(t, ti) for t, ti in zip(df[text_col].astype(str), titles)]
    mask = pd.Series(mask, index=df.index)
    return df[mask].reset_index(drop=True), int((~mask).sum())


# --------------------------------------------------------------------------- #
# Stemming / Lematização (opcionais, com fallback)                            #
# --------------------------------------------------------------------------- #
def _make_pt_stemmer():
    try:
        import nltk
        from nltk.stem import RSLPStemmer
        try:
            return RSLPStemmer()
        except LookupError:
            nltk.download("rslp", quiet=True)
            return RSLPStemmer()
    except Exception:
        return None


def _make_en_lemmatizer():
    try:
        import nltk
        from nltk.stem import WordNetLemmatizer
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
        lem = WordNetLemmatizer()
        lem.lemmatize("running")  # força carga; lança se faltar recurso
        return lem
    except Exception:
        return None


_PT_STEMMER = _make_pt_stemmer()
_EN_LEMMATIZER = _make_en_lemmatizer()


# --------------------------------------------------------------------------- #
# Limpeza e tokenização                                                       #
# --------------------------------------------------------------------------- #
_URL_RE = re.compile(r"http\S+|www\.\S+")
_MENTION_RE = re.compile(r"[@#]\w+")
_NON_WORD_RE = re.compile(r"[^a-zà-úãõâêôçáéíóúü\s]", re.IGNORECASE)
_TOKEN_RE = re.compile(r"\b[a-zà-úãõâêôçáéíóúü]{2,}\b", re.IGNORECASE)


def clean_text(text: str) -> str:
    """Aplica normalização e limpeza (sem tokenizar)."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)       # remove @menções e #hashtags
    # normaliza compatibilidade unicode (mantém acentos do PT)
    text = unicodedata.normalize("NFKC", text)
    text = _NON_WORD_RE.sub(" ", text)      # remove emojis, dígitos, pontuação
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """Tokenização por regex (evita dependência de modelos do NLTK)."""
    return _TOKEN_RE.findall(text)


def remove_stopwords(tokens: list[str], lang: str) -> list[str]:
    sw = STOPWORDS.get(lang, set())
    return [t for t in tokens if t not in sw]


def stem_or_lemmatize(tokens: list[str], lang: str) -> list[str]:
    """Stemming (PT/RSLP) ou lematização (EN/WordNet). Fallback: identidade."""
    if lang == "pt" and _PT_STEMMER is not None:
        return [_PT_STEMMER.stem(t) for t in tokens]
    if lang == "en" and _EN_LEMMATIZER is not None:
        return [_EN_LEMMATIZER.lemmatize(t) for t in tokens]
    return tokens


def preprocess_text(text: str, lang: str | None = None,
                    do_stem: bool = True) -> dict:
    """
    Executa o pipeline completo de pré-processamento em um único texto.
    Retorna dict com: lang, clean_text, tokens, tokens_str.
    """
    lang = lang or detect_language(text)
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    tokens = remove_stopwords(tokens, lang)
    if do_stem:
        tokens = stem_or_lemmatize(tokens, lang)
    # remove tokens muito curtos remanescentes
    tokens = [t for t in tokens if len(t) > 2]
    return {
        "lang": lang,
        "clean_text": cleaned,
        "tokens": tokens,
        "tokens_str": " ".join(tokens),
    }


def preprocess_dataframe(df: pd.DataFrame, text_col: str = "text",
                         do_stem: bool = True,
                         drop_empty: bool = True) -> pd.DataFrame:
    """
    Aplica o pré-processamento a um DataFrame inteiro.

    - Remove duplicatas exatas de texto.
    - Adiciona as colunas: lang, clean_text, tokens, tokens_str.
    - Opcionalmente descarta linhas sem tokens úteis.
    """
    df = df.copy()
    before = len(df)
    df = df.drop_duplicates(subset=[text_col]).reset_index(drop=True)
    print(f"Pré-processamento: {before} -> {len(df)} após remover duplicatas.")

    langs, cleans, toks, toks_str = [], [], [], []
    # usa a dica de idioma da coleta quando disponível
    hints = df["lang_hint"] if "lang_hint" in df.columns else [None] * len(df)

    for text, hint in tqdm(zip(df[text_col], hints), total=len(df),
                           desc="Pré-processando"):
        hint = hint if hint in ("pt", "en") else None
        res = preprocess_text(text, lang=hint, do_stem=do_stem)
        langs.append(res["lang"])
        cleans.append(res["clean_text"])
        toks.append(res["tokens"])
        toks_str.append(res["tokens_str"])

    df["lang"] = langs
    df["clean_text"] = cleans
    df["tokens"] = toks
    df["tokens_str"] = toks_str

    if drop_empty:
        mask = df["tokens_str"].str.len() > 0
        print(f"Pré-processamento: removendo {int((~mask).sum())} textos vazios.")
        df = df[mask].reset_index(drop=True)

    return df
