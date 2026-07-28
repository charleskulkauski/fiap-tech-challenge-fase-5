"""Detecção local de componentes de arquitetura via Ultralytics YOLO."""

from src.vision.yolo_detector import (
    DetectionResult,
    YoloDetectionError,
    detect_architecture_components,
    draw_detections,
)

__all__ = [
    "DetectionResult",
    "YoloDetectionError",
    "detect_architecture_components",
    "draw_detections",
]
