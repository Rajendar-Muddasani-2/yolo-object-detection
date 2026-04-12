"""
FastAPI gateway for YOLO wafer defect detection.

Routes requests to Triton Inference Server (or falls back to local Ultralytics model).
Provides REST API for single/batch detection, reports, and health checks.
"""

import io
import json
import logging
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Wafer Defect Detection API",
    description="YOLOv8-Large wafer defect detection with Triton Inference Server backend",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASSES = [
    "scratch", "particle", "edge_chip", "void", "pattern_shift",
    "bridge", "missing_bond", "crack", "contamination", "delamination",
]

# Global model reference (loaded on startup)
_model = None
_triton_client = None


class DetectionResult(BaseModel):
    class_name: str
    class_id: int
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]


class DetectionResponse(BaseModel):
    image_name: str
    detections: List[DetectionResult]
    inference_time_ms: float
    total_defects: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    backend: str  # "triton" or "ultralytics"


def _try_triton_connect():
    """Attempt to connect to Triton Inference Server."""
    global _triton_client
    try:
        import tritonclient.http as httpclient
        client = httpclient.InferenceServerClient(url="localhost:8000")
        if client.is_server_live():
            _triton_client = client
            logger.info("Connected to Triton Inference Server")
            return True
    except Exception:
        pass
    return False


def _load_ultralytics_model():
    """Load Ultralytics YOLOv8 model as fallback."""
    global _model
    try:
        from ultralytics import YOLO
        model_path = Path("models/best.pt")
        if model_path.exists():
            _model = YOLO(str(model_path))
        else:
            # Try ONNX
            onnx_path = Path("models/best.onnx")
            if onnx_path.exists():
                _model = YOLO(str(onnx_path))
            else:
                logger.warning("No trained model found. Place best.pt or best.onnx in models/")
                return False
        logger.info("Loaded Ultralytics model: %s", model_path)
        return True
    except ImportError:
        logger.error("ultralytics not installed")
        return False


@app.on_event("startup")
async def startup():
    """Try Triton first, fall back to Ultralytics."""
    if not _try_triton_connect():
        logger.info("Triton not available, using Ultralytics backend")
        _load_ultralytics_model()


@app.get("/health", response_model=HealthResponse)
async def health():
    backend = "triton" if _triton_client else ("ultralytics" if _model else "none")
    return HealthResponse(
        status="healthy" if (_triton_client or _model) else "no_model",
        model_loaded=bool(_triton_client or _model),
        backend=backend,
    )


def _predict_ultralytics(img: Image.Image, conf: float = 0.25) -> List[DetectionResult]:
    """Run inference with Ultralytics model."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = _model(img, conf=conf, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            detections.append(DetectionResult(
                class_name=CLASSES[cls_id] if cls_id < len(CLASSES) else f"class_{cls_id}",
                class_id=cls_id,
                confidence=float(box.conf[0]),
                bbox=[float(x) for x in box.xyxy[0].tolist()],
            ))
    return detections


def _predict_triton(img: Image.Image, conf: float = 0.25) -> List[DetectionResult]:
    """Run inference via Triton Inference Server."""
    import tritonclient.http as httpclient

    # Preprocess
    img_resized = img.resize((640, 640))
    img_arr = np.array(img_resized).astype(np.float32) / 255.0
    img_arr = np.transpose(img_arr, (2, 0, 1))[np.newaxis, ...]

    # Create Triton input
    inputs = [httpclient.InferInput("images", img_arr.shape, "FP32")]
    inputs[0].set_data_from_numpy(img_arr)

    outputs = [httpclient.InferRequestedOutput("output0")]
    result = _triton_client.infer("yolo_wafer_defect", inputs, outputs=outputs)
    output = result.as_numpy("output0")

    # Parse YOLO output (post-processing)
    detections = []
    if output is not None and len(output.shape) == 3:
        preds = output[0].T  # (num_preds, 4+num_classes)
        for pred in preds:
            x, y, w, h = pred[:4]
            scores = pred[4:]
            max_score = float(scores.max())
            if max_score >= conf:
                cls_id = int(scores.argmax())
                x1 = float(x - w / 2)
                y1 = float(y - h / 2)
                x2 = float(x + w / 2)
                y2 = float(y + h / 2)
                detections.append(DetectionResult(
                    class_name=CLASSES[cls_id] if cls_id < len(CLASSES) else f"class_{cls_id}",
                    class_id=cls_id,
                    confidence=max_score,
                    bbox=[x1, y1, x2, y2],
                ))
    return detections


@app.post("/detect", response_model=DetectionResponse)
async def detect(
    file: UploadFile = File(...),
    confidence: float = 0.25,
):
    """Detect wafer defects in a single image."""
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    t0 = time.perf_counter()
    if _triton_client:
        detections = _predict_triton(img, conf=confidence)
    else:
        detections = _predict_ultralytics(img, conf=confidence)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return DetectionResponse(
        image_name=file.filename or "unknown",
        detections=detections,
        inference_time_ms=round(elapsed_ms, 2),
        total_defects=len(detections),
    )


@app.post("/detect/batch")
async def detect_batch(
    files: List[UploadFile] = File(...),
    confidence: float = 0.25,
):
    """Detect defects in multiple images."""
    results = []
    for f in files:
        contents = await f.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        t0 = time.perf_counter()
        if _triton_client:
            detections = _predict_triton(img, conf=confidence)
        else:
            detections = _predict_ultralytics(img, conf=confidence)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        results.append(DetectionResponse(
            image_name=f.filename or "unknown",
            detections=detections,
            inference_time_ms=round(elapsed_ms, 2),
            total_defects=len(detections),
        ))
    return results


@app.get("/classes")
async def get_classes():
    """Return the defect class taxonomy."""
    return {"classes": CLASSES, "num_classes": len(CLASSES)}


@app.get("/metrics")
async def get_metrics():
    """Prometheus-compatible metrics endpoint."""
    # Basic metrics — Prometheus client library adds more in production
    return {
        "model_loaded": bool(_triton_client or _model),
        "backend": "triton" if _triton_client else "ultralytics",
    }
