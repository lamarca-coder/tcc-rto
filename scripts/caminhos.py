"""Caminhos do projeto: ROOT é definido aqui, num só lugar. Rode sempre a partir da raiz: python -m scripts.<pasta>.<script>."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
