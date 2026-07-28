"""
Smoke test do detector YOLO local.

Pré-requisito: best.pt em models/software_architecture_model/weights/
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vision.yolo_detector import (  # noqa: E402
    YoloDetectionError,
    detect_architecture_components,
    resolve_weights_path,
)

SAMPLE_IMAGE = ROOT / "data" / "material-fiap" / "diagrama-arquitetura.jpeg"


def main() -> None:
    assert SAMPLE_IMAGE.is_file(), f"Sample image missing: {SAMPLE_IMAGE}"

    weights = resolve_weights_path()
    if not weights.is_file():
        raise SystemExit(
            f"Pesos não encontrados em {weights}.\n"
            "Copie best.pt do Drive "
            "(yolov8_training_results/software_architecture_model/weights/best.pt) "
            "para models/software_architecture_model/weights/best.pt"
        )

    try:
        result = detect_architecture_components(SAMPLE_IMAGE)
    except YoloDetectionError as exc:
        raise SystemExit(f"YOLO detection failed: {exc}") from exc

    assert isinstance(result.detections, list)
    assert result.weights_path

    print("smoke ok")
    print("weights:", result.weights_path)
    print("detections:", len(result.detections))
    print("classes:", result.class_names)
    for det in result.detections[:10]:
        conf = det.get("confidence")
        conf_txt = f"{conf:.3f}" if isinstance(conf, (int, float)) else "n/a"
        print(f"  - {det.get('class')} conf={conf_txt}")


if __name__ == "__main__":
    main()
