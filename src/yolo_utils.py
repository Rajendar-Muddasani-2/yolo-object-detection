"""
Wafer Defect Detection Utilities — YOLOv8 / Ultralytics

Inference helpers, ONNX export, and evaluation metrics for
semiconductor wafer defect detection using YOLOv8.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image


DEFECT_CLASSES = [
    "scratch",
    "particle",
    "edge_chip",
    "void",
    "pattern_shift",
    "bridge",
    "missing_bond",
    "crack",
    "contamination",
    "delamination",
]


def load_model(weights: str | Path = "models/best.pt"):
    """Load a YOLOv8 model from weights file (.pt or .onnx)."""
    from ultralytics import YOLO

    return YOLO(str(weights))


def detect(
    model,
    image: str | Path | np.ndarray | Image.Image,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
) -> list[dict]:
    """Run detection on a single image.

    Returns a list of dicts with keys: class_id, class_name, confidence, bbox (xyxy).
    """
    results = model(image, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            detections.append(
                {
                    "class_id": cls_id,
                    "class_name": DEFECT_CLASSES[cls_id] if cls_id < len(DEFECT_CLASSES) else f"class_{cls_id}",
                    "confidence": round(float(box.conf[0]), 4),
                    "bbox": [round(float(c), 2) for c in box.xyxy[0].cpu().tolist()],
                }
            )
    return detections


def export_onnx(
    weights: str | Path = "models/best.pt",
    output_dir: str | Path = "models",
    imgsz: int = 640,
    dynamic: bool = True,
    simplify: bool = True,
) -> Path:
    """Export YOLOv8 model to ONNX format."""
    model = load_model(weights)
    model.export(format="onnx", imgsz=imgsz, dynamic=dynamic, simplify=simplify, opset=17)
    onnx_path = Path(weights).with_suffix(".onnx")
    dest = Path(output_dir) / onnx_path.name
    if onnx_path != dest:
        import shutil
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(onnx_path, dest)
    return dest


def benchmark(
    model,
    imgsz: int = 640,
    n_warmup: int = 10,
    n_runs: int = 50,
) -> dict:
    """Benchmark inference latency on random input."""
    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(n_warmup):
        model(dummy, verbose=False)
    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model(dummy, verbose=False)
        latencies.append((time.perf_counter() - t0) * 1000)
    arr = np.array(latencies)
    return {
        "mean_ms": round(float(arr.mean()), 2),
        "p50_ms": round(float(np.median(arr)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "fps": round(float(1000 / arr.mean()), 1),
    }


def compute_iou(box1: tuple, box2: tuple) -> float:
    """Compute IoU between two boxes in (x1, y1, x2, y2) format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def draw_detections(
    image: Image.Image,
    detections: list[dict],
    font_size: int = 14,
) -> Image.Image:
    """Draw bounding boxes on a PIL image. Returns a new image."""
    from PIL import ImageDraw

    COLORS = [
        "#ef4444", "#f97316", "#eab308", "#22c55e", "#06b6d4",
        "#3b82f6", "#8b5cf6", "#ec4899", "#f43f5e", "#14b8a6",
    ]
    img = image.copy()
    draw = ImageDraw.Draw(img)
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = COLORS[det["class_id"] % len(COLORS)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{det['class_name']} {det['confidence']:.2f}"
        draw.text((x1 + 2, y1 - 14), label, fill=color)
    return img
