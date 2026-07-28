"""Grafo LangGraph: visão → analista → relatório."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from langgraph.graph import END, START, StateGraph

from src.nodes.agente_analista import agente_analista
from src.nodes.agente_relatorio import agente_relatorio
from src.nodes.agente_visao import agente_visao
from src.state.estado_ameaca import EstadoAmeacas

_ROOT = Path(__file__).resolve().parents[1]
_MATERIAL = _ROOT / "data" / "material-fiap"

# Figuras de avaliação do PDF do hackathon (IADT Fase 5).
SAMPLE_IMAGES: dict[str, Path] = {
    "Figura 2 — Azure (PDF)": _MATERIAL / "figura-2-azure.jpg",
    "Figura 1 — AWS / SEI (PDF)": _MATERIAL / "figura-1-aws.jpg",
    "Diagrama Azure (material FIAP)": _MATERIAL / "diagrama-arquitetura.jpeg",
}
# Default: Figura 2 (Azure), a mesma arquitetura do enunciado / demo.
SAMPLE_IMAGE = SAMPLE_IMAGES["Figura 2 — Azure (PDF)"]
if not SAMPLE_IMAGE.is_file():
    SAMPLE_IMAGE = _MATERIAL / "diagrama-arquitetura.jpeg"


def build_graph():
    graph = StateGraph(EstadoAmeacas)
    graph.add_node("visao", agente_visao)
    graph.add_node("analista", agente_analista)
    graph.add_node("relatorio", agente_relatorio)

    graph.add_edge(START, "visao")
    graph.add_edge("visao", "analista")
    graph.add_edge("analista", "relatorio")
    graph.add_edge("relatorio", END)

    return graph.compile()


app = build_graph()


def estado_inicial(caminho_imagem: Optional[str] = None) -> EstadoAmeacas:
    return {
        "caminho_imagem": caminho_imagem or str(SAMPLE_IMAGE),
        "componentes_detectados": [],
        "deteccoes": [],
        "imagem_anotada": "",
        "analise_stride": {},
        "vulnerabilidades": [],
        "relatorio_final": "",
        "avisos": [],
        "etapa_atual": "inicio",
    }


def run_pipeline(caminho_imagem: Optional[str] = None) -> dict[str, Any]:
    return app.invoke(estado_inicial(caminho_imagem))
