"""Inferência local com Ultralytics YOLOv8 (pesos do treino Colab)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")

DEFAULT_WEIGHTS = (
    _PROJECT_ROOT
    / "models"
    / "software_architecture_model"
    / "weights"
    / "best.pt"
)
TRAINING_RUN_WEIGHTS = (
    _PROJECT_ROOT
    / "scripts"
    / "runs"
    / "detect"
    / "runs"
    / "software_architecture_model"
    / "weights"
    / "best.pt"
)

ImageInput = Union[str, Path, bytes]

_model_cache: dict[str, Any] = {}


class YoloDetectionError(Exception):
    """Falha ao carregar pesos ou rodar inferência YOLO."""


@dataclass
class DetectionResult:
    detections: list[dict[str, Any]] = field(default_factory=list)
    weights_path: str = ""
    names: dict[int, str] = field(default_factory=dict)

    @property
    def class_names(self) -> list[str]:
        return [
            str(det["class"])
            for det in self.detections
            if isinstance(det, Mapping) and "class" in det
        ]


def resolve_weights_path(weights: Optional[Union[str, Path]] = None) -> Path:
    """Prioridade: argumento → YOLO_WEIGHTS_PATH → defaults do projeto."""
    if weights is not None:
        path = Path(weights)
    else:
        env = os.getenv("YOLO_WEIGHTS_PATH", "").strip()
        if env:
            path = Path(env)
        elif DEFAULT_WEIGHTS.is_file():
            path = DEFAULT_WEIGHTS
        else:
            path = TRAINING_RUN_WEIGHTS

    if not path.is_absolute():
        path = (_PROJECT_ROOT / path).resolve()
    else:
        path = path.resolve()
    return path


def _load_model(weights_path: Path):
    key = str(weights_path)
    cached = _model_cache.get(key)
    if cached is not None:
        return cached

    if not weights_path.is_file():
        raise YoloDetectionError(
            f"Pesos YOLO não encontrados em {weights_path}. "
            "Copie o arquivo best.pt do treino Colab "
            "(Drive: yolov8_training_results/software_architecture_model/weights/best.pt) "
            "para models/software_architecture_model/weights/best.pt "
            "ou scripts/runs/detect/runs/software_architecture_model/weights/best.pt "
            "ou defina YOLO_WEIGHTS_PATH no .env."
        )

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise YoloDetectionError(
            "Pacote ultralytics não instalado. Rode: pip install ultralytics"
        ) from exc

    logger.info("Carregando modelo YOLO: %s", weights_path)
    model = YOLO(str(weights_path))
    _model_cache[key] = model
    return model


def _boxes_to_detections(result: Any) -> list[dict[str, Any]]:
    """Converte boxes Ultralytics (xyxy) para center xywh em pixels."""
    detections: list[dict[str, Any]] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return detections

    names = getattr(result, "names", None) or {}
    xyxy = boxes.xyxy.cpu().tolist()
    confs = boxes.conf.cpu().tolist() if boxes.conf is not None else [None] * len(xyxy)
    clss = boxes.cls.cpu().tolist() if boxes.cls is not None else [None] * len(xyxy)

    for coords, conf, cls_id in zip(xyxy, confs, clss):
        x1, y1, x2, y2 = coords
        width = float(x2 - x1)
        height = float(y2 - y1)
        cx = float(x1 + width / 2.0)
        cy = float(y1 + height / 2.0)

        label = "?"
        if cls_id is not None:
            idx = int(cls_id)
            label = str(names.get(idx, idx)).strip()

        det: dict[str, Any] = {
            "class": label,
            "x": cx,
            "y": cy,
            "width": width,
            "height": height,
        }
        if conf is not None:
            det["confidence"] = float(conf)
        if cls_id is not None:
            det["class_id"] = int(cls_id)
        detections.append(det)

    return detections


def detect_architecture_components(
    image: ImageInput,
    *,
    weights: Optional[Union[str, Path]] = None,
    conf: Optional[float] = None,
    imgsz: int = 640,
) -> DetectionResult:
    """Roda YOLOv8 em uma imagem de diagrama."""
    weights_path = resolve_weights_path(weights)
    model = _load_model(weights_path)

    if conf is None:
        conf_env = os.getenv("YOLO_CONF", "").strip()
        conf = float(conf_env) if conf_env else 0.25

    if isinstance(image, bytes):
        import numpy as np

        source: Any = np.frombuffer(image, dtype=np.uint8)
        import cv2

        decoded = cv2.imdecode(source, cv2.IMREAD_COLOR)
        if decoded is None:
            raise YoloDetectionError("Não foi possível decodificar a imagem em bytes.")
        source = decoded
    elif isinstance(image, Path):
        source = str(image.resolve())
        if not Path(source).is_file():
            raise YoloDetectionError(f"Imagem não encontrada: {source}")
    else:
        source = str(image)
        if not source.startswith(("http://", "https://")) and not Path(source).is_file():
            raise YoloDetectionError(f"Imagem não encontrada: {source}")

    try:
        results = model.predict(
            source=source,
            conf=conf,
            imgsz=imgsz,
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001
        raise YoloDetectionError(f"Falha na inferência YOLO: {exc}") from exc

    if not results:
        return DetectionResult(detections=[], weights_path=str(weights_path))

    first = results[0]
    names = dict(getattr(first, "names", None) or {})
    detections = _boxes_to_detections(first)

    return DetectionResult(
        detections=detections,
        weights_path=str(weights_path),
        names={int(k): str(v) for k, v in names.items()},
    )


def draw_detections(
    image: ImageInput,
    detections: Sequence[Mapping[str, Any]],
    output_path: Union[str, Path],
    *,
    box_color: tuple[int, int, int] = (0, 180, 80),
    text_color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
) -> Path:
    """Desenha detecções (center xywh) e salva a imagem anotada."""
    import cv2

    if isinstance(image, bytes):
        import numpy as np

        arr = np.frombuffer(image, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise YoloDetectionError("Não foi possível decodificar a imagem em bytes.")
    else:
        path = Path(image)
        frame = cv2.imread(str(path))
        if frame is None:
            raise YoloDetectionError(f"Não foi possível ler a imagem: {path}")

    for det in detections:
        try:
            cx = float(det["x"])
            cy = float(det["y"])
            w = float(det["width"])
            h = float(det["height"])
        except (KeyError, TypeError, ValueError):
            continue

        x1 = int(round(cx - w / 2))
        y1 = int(round(cy - h / 2))
        x2 = int(round(cx + w / 2))
        y2 = int(round(cy + h / 2))
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)

        label = str(det.get("class", "?"))
        conf_val = det.get("confidence")
        if isinstance(conf_val, (int, float)):
            label = f"{label} {conf_val:.2f}"

        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(0, y1 - 4)
        cv2.rectangle(
            frame,
            (x1, ty - th - baseline - 2),
            (x1 + tw + 4, ty + 2),
            box_color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 2, ty - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            text_color,
            1,
            cv2.LINE_AA,
        )

    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(dest), frame):
        raise YoloDetectionError(f"Falha ao salvar imagem anotada em {dest}")
    return dest.resolve()
