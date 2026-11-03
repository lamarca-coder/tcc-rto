"""
Pipeline de ponta a ponta — Resultados Preliminares (Entrega 4).

Executa: coleta -> armazenamento -> pré-processamento -> análise de sentimento
-> modelagem de tópicos -> análise temporal -> figuras -> relatório.

Exemplos de uso:
    # Roda com dados de exemplo (não precisa de chaves de API):
    python run_pipeline.py --source sample --sample-n 800

    # Coleta real do Reddit (requer .env configurado):
    python run_pipeline.py --source reddit --limit 50

    # Tudo (exemplo + reddit + twitter), sentimento por transformers:
    python run_pipeline.py --source all --sentiment-method transformer
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

import pandas as pd

from config import (
    N_TOPICS, TIME_FREQ, PROCESSED_PARQUET, FIGURES_DIR, OUTPUTS_DIR, LANGUAGES,
)
from src.storage import database as db
from src.preprocess.cleaning import preprocess_dataframe, filter_relevant
from src.preprocess.relevance_embed import add_relevance_embed, filter_by_similarity
from src.analysis.sentiment import add_sentiment
from src.analysis.sentiment_cache import add_sentiment_cached
from src.analysis.topics import add_topics
from src.analysis.temporal import sentiment_summary, sentiment_by_group
from src.analysis.stance import add_stance, stance_summary
from src.analysis.framing import build_frames, framing_summary
from src.report import figures
from src.report.highlights import render_highlights_md


# --------------------------------------------------------------------------- #
# 1. Coleta                                                                    #
# --------------------------------------------------------------------------- #
def collect_data(source: str, sample_n: int, limit: int) -> pd.DataFrame:
    # Fonte "db": analisa tudo que já está armazenado (coletado + importado)
    if source == "db":
        df = db.read_posts()
        if df.empty:
            raise SystemExit("Banco vazio. Colete ou importe dados antes de usar --source db.")
        # read_posts devolve created_utc como datetime; volta para ISO string
        df["created_utc"] = pd.to_datetime(
            df["created_utc"], utc=True, errors="coerce"
        ).dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        print(f"[coleta] {len(df)} registros lidos do banco (SQLite).")
        return df

    frames = []

    if source in ("sample", "all"):
        from src.collect.sample_data import generate_sample
        print(f"[coleta] gerando {sample_n} registros de exemplo...")
        frames.append(generate_sample(n=sample_n))

    if source in ("reddit", "all"):
        try:
            from src.collect.reddit_collector import collect_reddit
            print("[coleta] coletando do Reddit (API)...")
            frames.append(collect_reddit(limit_per_query=limit))
        except Exception as exc:
            print(f"[coleta] Reddit indisponível: {exc}")

    if source in ("pullpush", "all"):
        try:
            from src.collect.pullpush_collector import collect_pullpush
            print("[coleta] coletando histórico do Reddit (pullpush.io)...")
            frames.append(collect_pullpush(size=100, pages=2))
        except Exception as exc:
            print(f"[coleta] pullpush indisponível: {exc}")

    if source in ("gdelt", "all"):
        try:
            from src.collect.gdelt_collector import collect_gdelt
            print("[coleta] coletando notícias (GDELT)...")
            frames.append(collect_gdelt())
        except Exception as exc:
            print(f"[coleta] GDELT indisponível: {exc}")

    if source in ("bluesky", "all"):
        try:
            from src.collect.bluesky_collector import collect_bluesky
            print("[coleta] coletando do Bluesky...")
            frames.append(collect_bluesky())
        except Exception as exc:
            print(f"[coleta] Bluesky indisponível: {exc}")

    if source in ("youtube", "all"):
        try:
            from src.collect.youtube_collector import (
                collect_youtube, HISTORICAL_WINDOWS,
            )
            print("[coleta] coletando comentários do YouTube (API, 2019->hoje)...")
            # janelas históricas + mais comentários/vídeo => cobre 2019-2026 em PT
            frames.append(collect_youtube(
                comments_per_video=max(limit * 5, 300),
                published_windows=HISTORICAL_WINDOWS,
            ))
        except Exception as exc:
            print(f"[coleta] YouTube indisponível: {exc}")

    if source in ("twitter", "all"):
        try:
            from src.collect.twitter_collector import collect_twitter
            print("[coleta] coletando do X/Twitter (API)...")
            frames.append(collect_twitter())
        except Exception as exc:
            print(f"[coleta] X/Twitter indisponível: {exc}")

    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        raise SystemExit("Nenhum dado coletado. Verifique a fonte/credenciais.")

    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset="id")
    print(f"[coleta] total bruto: {len(df)} registros.")
    return df


# --------------------------------------------------------------------------- #
# Relatório em Markdown                                                        #
# --------------------------------------------------------------------------- #
def write_report(df: pd.DataFrame, topic_result, args, fig_paths, frames=None) -> str:
    summ = sentiment_summary(df)
    by_platform = sentiment_by_group(df, "platform")
    topics_df = topic_result.topics_as_dataframe(n_words=10)

    has_sample = (df["platform"] == "sample").any() or \
        df["id"].astype(str).str.startswith("sample_").any()

    lines = []
    lines.append("# Resultados Preliminares — Percepções sobre o Retorno ao Trabalho Presencial\n")
    lines.append(f"_Gerado em {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}_\n")
    lines.append("**Projeto:** Análise de percepções públicas sobre o retorno ao "
                 "trabalho presencial a partir de dados de redes sociais.  ")
    lines.append("**Aluno:** Marcos Eduardo Lamarca Nunes — MBA em Data Science & "
                 "Analytics para Operações.\n")

    if has_sample:
        lines.append("> ⚠️ **Aviso:** esta execução inclui dados de EXEMPLO "
                     "(sintéticos), usados para validar o pipeline. Os números "
                     "abaixo **não** constituem achados empíricos reais — servem "
                     "para demonstrar as etapas de análise antes da coleta real.\n")

    # 1. Visão geral do corpus
    lines.append("## 1. Visão geral do corpus\n")
    lines.append(f"- **Total de publicações analisadas:** {len(df)}")
    plat = df["platform"].value_counts().to_dict()
    lines.append(f"- **Por plataforma:** " +
                 ", ".join(f"{k}={v}" for k, v in plat.items()))
    langs = df["lang"].value_counts().to_dict()
    lines.append(f"- **Por idioma:** " +
                 ", ".join(f"{k}={v}" for k, v in langs.items()))
    dmin = pd.to_datetime(df["created_utc"], utc=True, errors="coerce").min()
    dmax = pd.to_datetime(df["created_utc"], utc=True, errors="coerce").max()
    lines.append(f"- **Período coberto:** {dmin:%Y-%m-%d} a {dmax:%Y-%m-%d}\n")

    # 2. Análise de sentimento
    lines.append("## 2. Análise de sentimento\n")
    lines.append(f"Método: `{args.sentiment_method}`.\n")
    lines.append("| Sentimento | Contagem | % |")
    lines.append("|------------|---------:|---:|")
    for idx, row in summ.iterrows():
        lines.append(f"| {idx} | {int(row['contagem'])} | {row['percentual']:.1f}% |")
    lines.append("")
    lines.append("**Proporção de sentimento por plataforma:**\n")
    lines.append(by_platform.to_markdown())
    lines.append("")

    # 2b. Posicionamento (stance) — a métrica central do estudo
    if "stance" in df.columns:
        st = stance_summary(df)
        lines.append("## 3. Posicionamento sobre o retorno presencial (stance)\n")
        lines.append("Combina o *alvo* (presencial/remoto/híbrido) com o "
                     "sentimento para inferir a posição sobre o RTO — distinguindo, "
                     "p.ex., \"gostar do home office\" (anti-RTO) de \"gostar de "
                     "voltar ao escritório\" (pró-RTO).\n")
        lines.append("| Posicionamento | Contagem | % |")
        lines.append("|----------------|---------:|---:|")
        for idx, row in st.iterrows():
            lines.append(f"| {idx} | {int(row['contagem'])} | {row['percentual']:.1f}% |")
        lines.append("")
        # distribuição de alvos
        tg = df["target"].value_counts()
        lines.append("**Alvos mencionados:** " +
                     ", ".join(f"{k}={v}" for k, v in tg.items()) + "\n")

    # 4. Tópicos
    lines.append("## 4. Modelagem de tópicos (LDA)\n")
    lines.append(f"Número de tópicos: {args.n_topics}.\n")
    lines.append("| Tópico | Palavras mais relevantes |")
    lines.append("|-------:|--------------------------|")
    for _, row in topics_df.iterrows():
        lines.append(f"| {row['topico']} | {row['palavras']} |")
    lines.append("")

    # 4b. Enquadramento (framing)
    if frames is not None and not frames.empty:
        fsum = framing_summary(frames)
        lines.append("## 5. Enquadramento: conteúdo original × comentários\n")
        lines.append(f"- Conteúdos (vídeos/posts) agregados: **{fsum['frames']}** "
                     f"(com ≥5 comentários).")
        if fsum["concordancia_pct"] is not None:
            lines.append(f"- Entre os de posicionamento definido "
                         f"({fsum['com_stance_definido']}), os comentários "
                         f"**concordam com o enquadramento do título em "
                         f"{fsum['concordancia_pct']}%** dos casos.")
        lines.append("")

    # 6. Exemplos para verificação (tabelas com links)
    if frames is not None and not frames.empty:
        lines += render_highlights_md(frames, df)

    # 7. Figuras
    lines.append("## 7. Figuras geradas\n")
    for p in fig_paths:
        name = p.replace("\\", "/").split("/")[-1]
        rel = f"figures/{name}"
        lines.append(f"- ![{name}]({rel})")
    lines.append("")

    # 8. Observações
    lines.append("## 8. Próximos passos\n")
    if has_sample:
        lines.append("- Substituir/complementar os dados de exemplo por coleta real "
                     "(YouTube/Reddit).")
    else:
        lines.append("- Ampliar o corpus com novas fontes (Reddit, datasets do "
                     "Kaggle/Pushshift) via `import_dataset.py` e `--source db`.")
    lines.append("- Refinar o nº de tópicos do LDA (coerência) e a lista de "
                 "stopwords/relevância conforme novos ruídos aparecerem.")
    lines.append("- Aprofundar a análise temporal cruzando com eventos "
                 "(políticas de RTO, notícias).")
    lines.append("- Integrar a análise qualitativa complementar (LinkedIn).\n")

    report_path = OUTPUTS_DIR / "resultados_preliminares.md"
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return str(report_path)


# --------------------------------------------------------------------------- #
# Orquestração                                                                 #
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Pipeline de resultados preliminares (RTO).")
    parser.add_argument("--source",
                        choices=["sample", "reddit", "youtube", "pullpush",
                                 "gdelt", "bluesky", "twitter", "all", "db"],
                        default="sample", help="Fonte dos dados ('db' = tudo que já está armazenado).")
    parser.add_argument("--sample-n", type=int, default=1500,
                        help="Quantidade de registros de exemplo a gerar.")
    parser.add_argument("--limit", type=int, default=50,
                        help="Limite por consulta nas APIs reais.")
    parser.add_argument("--n-topics", type=int, default=N_TOPICS,
                        help="Número de tópicos do LDA.")
    parser.add_argument("--sentiment-method", choices=["lexicon", "transformer"],
                        default="lexicon", help="Backend de análise de sentimento.")
    parser.add_argument("--no-stem", action="store_true",
                        help="Desativa stemming/lematização (tópicos mais legíveis).")
    parser.add_argument("--no-relevance", action="store_true",
                        help="Desativa o filtro de relevância temática.")
    parser.add_argument("--embed-relevance", action="store_true",
                        help="Aplica o filtro de relevância semântico (embeddings).")
    parser.add_argument("--embed-threshold", type=float, default=0.25,
                        help="Limiar de similaridade do filtro semântico.")
    parser.add_argument("--collect-only", action="store_true",
                        help="Apenas coleta e armazena no banco (não processa).")
    parser.add_argument("--freq", choices=["W", "M"], default=TIME_FREQ,
                        help="Granularidade temporal: W=semana, M=mês.")
    args = parser.parse_args()

    print("=" * 70)
    print(" PIPELINE — Resultados Preliminares (RTO nas redes sociais)")
    print("=" * 70)

    # 1. Coleta
    raw = collect_data(args.source, args.sample_n, args.limit)

    # 2. Armazenamento (SQLite + parquet)
    inserted = db.upsert_posts(raw)
    db.save_parquet(raw, processed=False)
    print(f"[armazenamento] {inserted} novos registros no SQLite; "
          f"parquet bruto salvo.")

    if args.collect_only:
        total = len(db.read_posts())
        print(f"[collect-only] coleta concluída. Banco agora tem {total} registros.")
        print("Para analisar tudo: python run_pipeline.py --source db --no-stem --sentiment-method transformer")
        return

    # 2b. Filtro de relevância temática por palavra-chave (rápido)
    if not args.no_relevance:
        n0 = len(raw)
        raw, removed = filter_relevant(raw)
        print(f"[relevância] removidos {removed} fora de tema: {n0} -> {len(raw)}.")

    # 2c. Filtro de relevância SEMÂNTICO por embeddings (refino fino)
    if args.embed_relevance:
        n0 = len(raw)
        raw = add_relevance_embed(raw)
        raw, removed = filter_by_similarity(raw, args.embed_threshold)
        print(f"[embeddings] limiar {args.embed_threshold}: removidos {removed}; "
              f"{n0} -> {len(raw)}.")

    # 3. Pré-processamento
    proc = preprocess_dataframe(raw, do_stem=not args.no_stem)

    # 3b. Filtro de idioma (recorte do estudo — ver LANGUAGES no config)
    antes = len(proc)
    proc = proc[proc["lang"].isin(LANGUAGES)].reset_index(drop=True)
    print(f"[filtro] idioma {LANGUAGES}: {antes} -> {len(proc)} registros.")

    # 4. Sentimento (com cache: transformer só roda em itens novos)
    proc = add_sentiment_cached(proc, method=args.sentiment_method)

    # 4b. Posicionamento (alvo + stance sobre o RTO)
    proc = add_stance(proc)
    print("[stance] alvo e posicionamento (pró/anti-RTO) calculados.")

    # 5. Tópicos
    proc, topic_result = add_topics(proc, n_topics=args.n_topics)

    # 5b. Enquadramento (conteúdo-pai × comentários)
    frames = build_frames(proc)
    fsum = framing_summary(frames)
    print(f"[framing] {fsum['frames']} conteúdos agregados; "
          f"concordância título×comentários: {fsum['concordancia_pct']}%.")

    # 6. Persistir processado (sem a coluna 'tokens' que é lista)
    to_save = proc.drop(columns=["tokens"], errors="ignore")
    to_save.to_parquet(PROCESSED_PARQUET, index=False)
    print(f"[armazenamento] parquet processado salvo em {PROCESSED_PARQUET}")

    # 7. Figuras
    print("[figuras] gerando gráficos...")
    fig_paths = figures.generate_all(proc, topic_result, freq=args.freq, frames=frames)
    for p in fig_paths:
        print(f"   - {p}")

    # 8. Relatório
    report = write_report(proc, topic_result, args, fig_paths, frames)
    print("\n" + "=" * 70)
    print(f" Concluído. Relatório: {report}")
    print(f" Figuras em: {FIGURES_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
