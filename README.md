# Manifestações sobre o retorno ao trabalho presencial no YouTube, Reddit e Bluesky

Código do TCC do MBA em Data Science & Analytics para Operações (POLI USP PRO), de Marcos Eduardo Lamarca Nunes.

## O que o projeto faz

O trabalho mede como o retorno ao trabalho presencial (RTO) aparece em textos públicos em português do Brasil, de 2019 a 2026. O código cobre:

1. **Coleta** de comentários e posts do YouTube (API de dados v3), Reddit (arquivos públicos) e Bluesky (AT Protocol).
2. **Filtragem** por palavras-chave e por similaridade semântica (embeddings MiniLM), detecção de idioma e remoção de duplicatas.
3. **Análise**: sentimento (BERT em português), alvo (presencial, remoto, híbrido), posicionamento pró ou anti RTO, tópicos (LDA) e série temporal.
4. **Validação** com amostras rotuladas por humanos e por LLM, e estimativas ponderadas por amostragem estratificada com bootstrap.
5. **Triangulação** com o survey State of Data Brazil e com o Google Trends.

Há duas versões do pipeline: a **v0** (`run_pipeline.py`, comando citado no Método do TCC) e a **v2** (scripts em `scripts/`, uma etapa por script).

## Estrutura de pastas

```
.
├── run_pipeline.py        pipeline v0
├── config.py              caminhos, palavras-chave, parâmetros
├── config_coleta.yaml     parâmetros dos coletores
├── requirements.txt
├── .env.example           modelo das chaves de API
├── src/                   código do pipeline v0 (coleta, banco, pré-processamento, análise, figuras)
├── scripts/               etapas da v2, rodadas como módulos
│   ├── caminhos.py        define a raiz do projeto
│   ├── b_validacao/       amostras, métricas e classificadores
│   ├── c_coleta/          coletores e filtro do Bluesky
│   ├── d_analise/         construção do corpus v2, análises e resultados
│   └── utilitarios/       funções comuns, importação de datasets, Kaggle
├── dados/                 só dois CSVs públicos (eventos e coerência dos tópicos)
├── outputs/figuras_v2/    figuras geradas na v2
└── docs/                  prompt usado com o anotador LLM
```

## Instalação

Requer Python 3.10 ou superior (testado com 3.12).

```bash
pip install -r requirements.txt
```

## Chaves de API

Copie `.env.example` para `.env` e preencha só o que for usar. O `.env` nunca deve ser versionado.

| Variável | Para quê | Onde obter |
|---|---|---|
| `YOUTUBE_API_KEY` | coleta do YouTube | Google Cloud, YouTube Data API v3, chave de API |
| `BLUESKY_HANDLE`, `BLUESKY_APP_PASSWORD` | coleta do Bluesky | Bluesky, Settings, App Passwords (use uma senha de app, nunca a senha da conta) |
| `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT` | coletor v0 via API do Reddit (opcional) | reddit.com/prefs/apps |
| `TWITTER_BEARER_TOKEN` | coletor do X (opcional, acesso pago) | developer.x.com |
| `KAGGLE_USERNAME`, `KAGGLE_KEY` | download do State of Data | Kaggle, Settings, API |

## Como rodar

Rode sempre da raiz do projeto, como módulo: `python -m scripts.<pasta>.<script>`. O cabeçalho de cada script informa entradas, saídas e tempo aproximado.

**v0** (comando citado no TCC):

```bash
python run_pipeline.py --source db --no-stem --sentiment-method transformer --embed-relevance --embed-threshold 0.30
```

**v2**, na ordem:

| Etapa | Comando |
|---|---|
| Coleta Bluesky | `python -m scripts.c_coleta.coleta_bluesky` e `python -m scripts.c_coleta.filtrar_bluesky` |
| Coleta YouTube | `python -m scripts.c_coleta.coleta_youtube_c3` |
| Coleta Reddit (opcional) | `python -m scripts.c_coleta.coleta_reddit_arctic` |
| Corpus v2 | `python -m scripts.d_analise.d1_etapa1_filtrar`, `d1_etapa2_sentimento`, `d1_etapa3_final` |
| Funil, plataformas, limiar | `d2_funil`, `d3_plataformas`, `d4_limiar` |
| Tópicos | `d5_lda_coerencia`, `d5_passo4`, `d5_final` |
| Temporal e survey | `d6_temporal`, `d7_survey` |
| Validação | `python -m scripts.b_validacao.b5_metricas`, `b9_amostra`, `b10_classificador`, `b10b_embeddings`, `b10b_relevancia` |
| Resultados ponderados | `python -m scripts.d_analise.b11_resultados`, `b11b_sensibilidade` |
| Qualidade do corpus | `python -m scripts.d_analise.d1b_qualidade` |

Os scripts de rotulagem (B2, B8, B9) geram planilhas para anotação manual ou por LLM. O prompt usado com o anotador LLM está em `docs/prompt_anotador_llm.md`. As etapas D1 (sentimento) e D5 levam horas em CPU e são retomáveis.

## Por que os dados não estão incluídos

Os textos coletados são de terceiros (usuários do YouTube, Reddit e Bluesky). Os termos de uso das plataformas e a LGPD não permitem redistribuí-los, e os identificadores de publicações permitiriam localizar os autores. Por isso o repositório não traz textos, bancos, planilhas de rotulagem nem tabelas com identificadores. Só estão incluídos o código, as figuras agregadas e dois CSVs sem dados pessoais: `dados/eventos.csv` (eventos públicos da linha do tempo) e `dados/v2/d5_coerencia.csv` (coerência dos tópicos).

Para reproduzir, colete de novo com os scripts de `scripts/c_coleta/` usando suas próprias chaves. Os resultados podem diferir um pouco, porque as plataformas mudam ao longo do tempo. Os autores são anonimizados antes de qualquer análise (SHA-256 do identificador, 16 primeiros caracteres).

## State of Data Brazil

O survey vem dos datasets `datahackers/state-of-data-*` (edições 2021 a 2024) do Kaggle, publicados pela comunidade Data Hackers. Baixe com:

```bash
python -m scripts.utilitarios.kaggle_fetch download <dono>/<nome-do-dataset>
```

e coloque os CSVs em `data/kaggle/<ano>/`. Os arquivos não são redistribuídos aqui. Consulte a licença de cada dataset no Kaggle.

## Como citar

```
NUNES, Marcos Eduardo Lamarca. Manifestações sobre o retorno ao trabalho presencial
no YouTube, Reddit e Bluesky. Trabalho de Conclusão de Curso (MBA em Data Science &
Analytics para Operações) - Escola Politécnica, Universidade de São Paulo (POLI USP
PRO), 2026. Orientador: Christian Eric Barrantes Briceño.
```

## Licença

Código sob licença MIT (ver `LICENSE`). A licença cobre o código, não os dados de terceiros.
