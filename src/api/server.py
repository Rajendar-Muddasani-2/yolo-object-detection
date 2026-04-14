"""
FastAPI gateway for YOLO wafer defect detection.

Routes requests to Triton Inference Server (or falls back to local Ultralytics model).
Provides REST API for single/batch detection, reports, and health checks.
Includes JWT authentication, API key support, rate limiting, and structured logging.
"""

import io
import json
import logging
import os
import secrets
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("wafer_api")

# ---------------------------------------------------------------------------
# Auth config (env-driven, safe defaults for local dev)
# ---------------------------------------------------------------------------
API_KEY = os.getenv("API_KEY", "dev-api-key-change-in-prod")
JWT_SECRET = os.getenv("JWT_SECRET", "dev-jwt-secret-change-in-prod")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Rate limiting config
# ---------------------------------------------------------------------------
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))  # per window
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW", "60"))

# In-memory rate limiter (use Redis in production cluster)
_rate_limit_store: dict[str, list[float]] = defaultdict(list)


def _check_rate_limit(client_ip: str) -> None:
    """Sliding window rate limiter."""
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS
    # Prune old entries
    _rate_limit_store[client_ip] = [
        ts for ts in _rate_limit_store[client_ip] if ts > window_start
    ]
    if len(_rate_limit_store[client_ip]) >= RATE_LIMIT_REQUESTS:
        logger.warning("Rate limit exceeded for %s", client_ip)
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW_SECONDS}s.",
        )
    _rate_limit_store[client_ip].append(now)


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------
def _create_jwt(subject: str) -> str:
    """Create a signed JWT token."""
    import hashlib
    import hmac
    import base64

    header = base64.urlsafe_b64encode(
        json.dumps({"alg": "HS256", "typ": "JWT"}).encode()
    ).rstrip(b"=").decode()

    now = datetime.now(timezone.utc)
    payload = base64.urlsafe_b64encode(json.dumps({
        "sub": subject,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=JWT_EXPIRE_MINUTES)).timestamp()),
    }).encode()).rstrip(b"=").decode()

    signing_input = f"{header}.{payload}"
    sig = base64.urlsafe_b64encode(
        hmac.new(JWT_SECRET.encode(), signing_input.encode(), hashlib.sha256).digest()
    ).rstrip(b"=").decode()

    return f"{header}.{payload}.{sig}"


def _verify_jwt(token: str) -> dict:
    """Verify a JWT token. Raises HTTPException on failure."""
    import hashlib
    import hmac
    import base64

    parts = token.split(".")
    if len(parts) != 3:
        raise HTTPException(status_code=401, detail="Invalid token format")

    header_b64, payload_b64, sig_b64 = parts
    signing_input = f"{header_b64}.{payload_b64}"

    # Verify signature
    expected_sig = base64.urlsafe_b64encode(
        hmac.new(JWT_SECRET.encode(), signing_input.encode(), hashlib.sha256).digest()
    ).rstrip(b"=").decode()

    if not hmac.compare_digest(sig_b64, expected_sig):
        raise HTTPException(status_code=401, detail="Invalid token signature")

    # Decode payload
    padding = 4 - len(payload_b64) % 4
    payload_b64_padded = payload_b64 + "=" * padding
    payload = json.loads(base64.urlsafe_b64decode(payload_b64_padded))

    # Check expiry
    if payload.get("exp", 0) < time.time():
        raise HTTPException(status_code=401, detail="Token expired")

    return payload


async def _authenticate(
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
) -> Optional[str]:
    """Authenticate via Bearer JWT or X-API-Key header.

    When AUTH_ENABLED=false (default for local dev), all requests pass through.
    """
    if not AUTH_ENABLED:
        return "anonymous"

    # API key auth
    if x_api_key:
        if not secrets.compare_digest(x_api_key, API_KEY):
            logger.warning("Invalid API key from %s", request.client.host if request.client else "unknown")
            raise HTTPException(status_code=401, detail="Invalid API key")
        return "api_key_user"

    # JWT auth
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:]
        payload = _verify_jwt(token)
        return payload.get("sub", "jwt_user")

    raise HTTPException(
        status_code=401,
        detail="Authentication required. Provide X-API-Key header or Bearer token.",
    )

app = FastAPI(
    title="Wafer Defect Detection API",
    description="YOLOv8-Large wafer defect detection with Triton Inference Server backend",
    version="2.0.0",
)

# CORS — restrict in production via ALLOWED_ORIGINS env var
_allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request logging & rate-limit middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def request_middleware(request: Request, call_next):
    """Log every request and enforce rate limits."""
    client_ip = request.client.host if request.client else "unknown"
    start = time.perf_counter()

    # Rate limiting (skip health checks)
    if request.url.path not in ("/health", "/docs", "/openapi.json"):
        _check_rate_limit(client_ip)

    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000

    # Track metrics
    _metrics["total_requests"] += 1
    if response.status_code >= 400:
        _metrics["error_count"] += 1
    if request.url.path == "/detect":
        _metrics["inference_count"] += 1
        _metrics["latency_sum_ms"] += elapsed_ms

    logger.info(
        "%s %s %d %.1fms client=%s",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
        client_ip,
    )
    return response

CLASSES = [
    "scratch", "particle", "edge_chip", "void", "pattern_shift",
    "bridge", "missing_bond", "crack", "contamination", "delamination",
]

# Global model reference (loaded on startup)
_model = None
_triton_client = None

# Prometheus-compatible metrics counters
_metrics = {
    "total_requests": 0,
    "inference_count": 0,
    "error_count": 0,
    "latency_sum_ms": 0.0,
    "startup_time": None,
}


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
    _metrics["startup_time"] = datetime.now(timezone.utc).isoformat()
    if not _try_triton_connect():
        logger.info("Triton not available, loading Ultralytics backend")
        _load_ultralytics_model()
    backend = "triton" if _triton_client else ("ultralytics" if _model else "none")
    logger.info("API ready | backend=%s | auth=%s | rate_limit=%d/%ds",
                backend, AUTH_ENABLED, RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SECONDS)


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
    request: Request,
    file: UploadFile = File(...),
    confidence: float = 0.25,
    user: str = Depends(_authenticate),
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

    logger.info(
        "Detection complete | image=%s | defects=%d | time=%.1fms | user=%s",
        file.filename, len(detections), elapsed_ms, user,
    )

    return DetectionResponse(
        image_name=file.filename or "unknown",
        detections=detections,
        inference_time_ms=round(elapsed_ms, 2),
        total_defects=len(detections),
    )


@app.post("/detect/batch")
async def detect_batch(
    request: Request,
    files: List[UploadFile] = File(...),
    confidence: float = 0.25,
    user: str = Depends(_authenticate),
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
    avg_latency = (
        _metrics["latency_sum_ms"] / _metrics["inference_count"]
        if _metrics["inference_count"] > 0
        else 0
    )
    return {
        "model_loaded": bool(_triton_client or _model),
        "backend": "triton" if _triton_client else "ultralytics",
        "total_requests": _metrics["total_requests"],
        "inference_count": _metrics["inference_count"],
        "error_count": _metrics["error_count"],
        "avg_inference_ms": round(avg_latency, 2),
        "startup_time": _metrics["startup_time"],
        "auth_enabled": AUTH_ENABLED,
        "rate_limit": f"{RATE_LIMIT_REQUESTS}/{RATE_LIMIT_WINDOW_SECONDS}s",
    }


# ---------------------------------------------------------------------------
# Auth token endpoint (for JWT-based auth)
# ---------------------------------------------------------------------------
class TokenRequest(BaseModel):
    api_key: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


@app.post("/token", response_model=TokenResponse)
async def get_token(req: TokenRequest):
    """Exchange API key for a JWT token."""
    if not secrets.compare_digest(req.api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid API key")
    token = _create_jwt(subject="api_user")
    logger.info("JWT token issued for api_user")
    return TokenResponse(
        access_token=token,
        expires_in=JWT_EXPIRE_MINUTES * 60,
    )
