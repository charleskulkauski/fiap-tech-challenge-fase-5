"""Nó LangGraph: detecção de componentes de arquitetura via YOLO local."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.vision.yolo_detector import (
    YoloDetectionError,
    detect_architecture_components,
    draw_detections,
)
from src.state.estado_ameaca import EstadoAmeacas

logger = logging.getLogger(__name__)


def _limiar_confianca() -> float:
    """Lê YOLO_CONF do .env (default 0.25); mesmo limiar da inferência."""
    env = os.getenv("YOLO_CONF", "").strip()
    try:
        return float(env) if env else 0.25
    except ValueError:
        return 0.25


CONFIDENCE_UNCERTAIN = 0.55

SAMPLE_IMAGE = _ROOT / "data" / "material-fiap" / "figura-2-azure.jpg"
if not SAMPLE_IMAGE.is_file():
    SAMPLE_IMAGE = _ROOT / "data" / "material-fiap" / "diagrama-arquitetura.jpeg"
OUTPUT_IMAGE = _ROOT / "data" / "outputs" / "diagrama-arquitetura-annotated.jpg"


def _filtrar_deteccoes(
    detections: list[dict[str, Any]], limiar: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    aceitas: list[dict[str, Any]] = []
    descartadas: list[dict[str, Any]] = []
    for det in detections:
        conf = det.get("confidence")
        if isinstance(conf, (int, float)) and conf < limiar:
            descartadas.append(det)
            continue
        aceitas.append(det)
    return aceitas, descartadas


def _fmt_det(det: dict[str, Any]) -> str:
    classe = det.get("class", "?")
    conf = det.get("confidence")
    conf_txt = f"{conf:.2f}" if isinstance(conf, (int, float)) else "n/a"
    return f"{classe} ({conf_txt})"


def agente_visao(state: EstadoAmeacas) -> dict[str, Any]:
    """Detecta componentes na imagem e atualiza o estado do grafo."""
    caminho = state.get("caminho_imagem") or str(SAMPLE_IMAGE)
    avisos: list[str] = []

    try:
        result = detect_architecture_components(caminho)
    except YoloDetectionError as exc:
        logger.error("Falha na detecção YOLO: %s", exc)
        avisos.append(
            "Visão: falha na detecção YOLO — o pipeline continua sem "
            f"componentes. Detalhe: {exc}"
        )
        return {
            "caminho_imagem": str(caminho),
            "componentes_detectados": [],
            "deteccoes": [],
            "imagem_anotada": "",
            "avisos": avisos,
            "etapa_atual": "visao",
        }

    brutas = list(result.detections or [])
    limiar = _limiar_confianca()
    deteccoes, descartadas = _filtrar_deteccoes(brutas, limiar)
    componentes = sorted(
        {str(det["class"]).strip() for det in deteccoes if det.get("class")}
    )

    if descartadas:
        amostra = ", ".join(_fmt_det(d) for d in descartadas[:5])
        extra = f" (ex.: {amostra})" if amostra else ""
        avisos.append(
            f"Visão: {len(descartadas)} detecção(ões) descartada(s) por confiança "
            f"< {limiar:.2f}{extra}."
        )

    incertas = [
        det
        for det in deteccoes
        if isinstance(det.get("confidence"), (int, float))
        and det["confidence"] < CONFIDENCE_UNCERTAIN
    ]
    if incertas:
        amostra = ", ".join(_fmt_det(d) for d in incertas[:5])
        avisos.append(
            f"Visão: {len(incertas)} detecção(ões) com confiança moderada "
            f"(<{CONFIDENCE_UNCERTAIN:.2f}) — a classe pode estar errada: {amostra}."
        )

    if not brutas:
        avisos.append(
            "Visão: o modelo não retornou nenhuma detecção. O diagrama pode "
            "estar ambíguo, incompleto ou fora das classes do dataset treinado."
        )
    elif not deteccoes:
        avisos.append(
            "Visão: nenhuma detecção passou no limiar de confiança. "
            "Análise STRIDE ficará vazia ou parcial."
        )

    imagem_anotada = ""
    if deteccoes:
        try:
            imagem_anotada = str(draw_detections(caminho, deteccoes, OUTPUT_IMAGE))
        except YoloDetectionError as exc:
            logger.warning("Não foi possível gerar a imagem anotada: %s", exc)
            avisos.append(
                f"Visão: componentes detectados, mas a imagem anotada não foi "
                f"gerada ({exc})."
            )

    logger.info(
        "Visão: %d detecções, %d componentes únicos, %d avisos (pesos=%s).",
        len(deteccoes),
        len(componentes),
        len(avisos),
        result.weights_path or "n/a",
    )
    return {
        "caminho_imagem": str(caminho),
        "componentes_detectados": componentes,
        "deteccoes": deteccoes,
        "imagem_anotada": imagem_anotada,
        "avisos": avisos,
        "etapa_atual": "visao",
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    resultado = agente_visao({"caminho_imagem": str(SAMPLE_IMAGE)})
    print("componentes_detectados:", resultado["componentes_detectados"])
    print("detections:", len(resultado["deteccoes"]))
    for det in resultado["deteccoes"]:
        conf = det.get("confidence")
        conf_txt = f"{conf:.3f}" if isinstance(conf, (int, float)) else "n/a"
        print(f"  - {det.get('class')} conf={conf_txt} xy=({det.get('x')}, {det.get('y')})")
    print("imagem_anotada:", resultado["imagem_anotada"])
    print("avisos:")
    for aviso in resultado.get("avisos") or []:
        print(f"  - {aviso}")
