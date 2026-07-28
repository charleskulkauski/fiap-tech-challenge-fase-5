"""CLI: executa o pipeline LangGraph (visão → analista → relatório)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.graph import SAMPLE_IMAGE, run_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Modelagem de ameaças: diagrama → STRIDE → relatório."
    )
    parser.add_argument(
        "imagem",
        nargs="?",
        default=str(SAMPLE_IMAGE),
        help=f"Caminho do diagrama (padrão: {SAMPLE_IMAGE.name})",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Log detalhado dos nós",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    caminho = Path(args.imagem)
    if not caminho.is_file():
        print(f"Imagem não encontrada: {caminho}", file=sys.stderr)
        return 1

    resultado = run_pipeline(str(caminho.resolve()))

    componentes = resultado.get("componentes_detectados") or []
    vulns = resultado.get("vulnerabilidades") or []
    etapa = resultado.get("etapa_atual")
    relatorio = resultado.get("relatorio_final") or ""

    print(f"etapa_atual: {etapa}")
    print(f"componentes ({len(componentes)}): {componentes}")
    print(f"vulnerabilidades: {len(vulns)}")
    if resultado.get("imagem_anotada"):
        print(f"imagem_anotada: {resultado['imagem_anotada']}")
    print("--- relatório (trecho) ---")
    print("\n".join(relatorio.splitlines()[:25]))
    if relatorio.count("\n") > 25:
        print("...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
