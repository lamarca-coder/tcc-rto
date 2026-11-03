"""utilitário (v0): Busca e baixa datasets abertos do Kaggle (KAGGLE_USERNAME e KAGGLE_KEY no .env).

Etapa: utilitário (v0)
Objetivo: Busca e baixa datasets abertos do Kaggle (KAGGLE_USERNAME e KAGGLE_KEY no .env).
Entradas: API do Kaggle
Saídas: data/kaggle/
Como rodar: python -m scripts.utilitarios.kaggle_fetch
Tempo aproximado: depende do download
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from config import DATA_DIR

load_dotenv()  # carrega KAGGLE_USERNAME / KAGGLE_KEY do .env para o ambiente

KAGGLE_DIR = DATA_DIR / "kaggle"


def _api():
    user = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")
    if not user or not key:
        raise SystemExit(
            "Credenciais do Kaggle ausentes. Preencha KAGGLE_USERNAME e KAGGLE_KEY "
            "no .env (kaggle.com -> Settings -> API -> Create New API Token)."
        )
    # garante que a lib enxergue as credenciais via ambiente
    os.environ["KAGGLE_USERNAME"] = user
    os.environ["KAGGLE_KEY"] = key
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def search(query: str, n: int = 25) -> None:
    api = _api()
    results = api.dataset_list(search=query, sort_by="hottest")
    print(f"Resultados para '{query}':\n")
    print(f"{'REF (owner/slug)':50} {'TAM.':>8}  TÍTULO")
    print("-" * 100)
    for d in results[:n]:
        size = getattr(d, "totalBytes", None) or getattr(d, "size", "")
        if isinstance(size, int):
            size = f"{size / 1e6:.1f}MB"
        print(f"{d.ref!s:50} {size!s:>8}  {d.title}")
    print("\nBaixe com:  python kaggle_fetch.py download owner/slug")


def files(ref: str) -> None:
    api = _api()
    fs = api.dataset_list_files(ref).files
    print(f"Arquivos de {ref}:")
    for f in fs:
        print(f"  - {f.name}  ({getattr(f, 'size', '?')})")


def download(ref: str, out: str | None = None) -> None:
    api = _api()
    out_dir = Path(out or KAGGLE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Baixando {ref} ...")
    api.dataset_download_files(ref, path=str(out_dir), unzip=True)
    print(f"Concluído em {out_dir}:")
    for f in sorted(out_dir.glob("*")):
        print(f"  - {f.name}")
    print(
        "\nImporte com:  python import_dataset.py <arquivo>.csv "
        "--text-col <col> --date-col <col> --platform kaggle --lang pt"
    )


def main():
    ap = argparse.ArgumentParser(description="Busca/baixa datasets do Kaggle.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search", help="Busca datasets por termo.")
    s.add_argument("query")
    s.add_argument("-n", type=int, default=25)

    f = sub.add_parser("files", help="Lista arquivos de um dataset.")
    f.add_argument("ref", help="owner/slug")

    d = sub.add_parser("download", help="Baixa e descompacta um dataset.")
    d.add_argument("ref", help="owner/slug")
    d.add_argument("--out", help="Pasta de saída (padrão data/kaggle).")

    args = ap.parse_args()
    if args.cmd == "search":
        search(args.query, args.n)
    elif args.cmd == "files":
        files(args.ref)
    elif args.cmd == "download":
        download(args.ref, args.out)


if __name__ == "__main__":
    main()
