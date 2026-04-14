"""
Integration tests for the wafer defect detection API.

Tests the FastAPI server end-to-end with real model inference.
Run with: pytest tests/test_integration.py -v

For Docker stack integration:
    docker compose up -d
    pytest tests/test_integration.py -v --docker
"""

import io
import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _create_wafer_image(size: int = 640) -> bytes:
    """Create a synthetic wafer image for testing."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    center = size // 2
    Y, X = np.ogrid[:size, :size]
    mask = (X - center) ** 2 + (Y - center) ** 2 <= (center - 20) ** 2
    img[mask] = [180, 180, 190]
    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.getvalue()


def _create_invalid_file() -> bytes:
    """Create an invalid (non-image) file."""
    return b"This is not an image file"


_startup_done = False


async def _ensure_startup():
    """Ensure the FastAPI startup event has run (loads the model)."""
    global _startup_done
    if not _startup_done:
        from src.api.server import startup
        await startup()
        _startup_done = True


@pytest.fixture
def wafer_image():
    return _create_wafer_image()


@pytest.fixture
def invalid_file():
    return _create_invalid_file()


# ---------------------------------------------------------------------------
# API Server Tests (in-process, no Docker needed)
# ---------------------------------------------------------------------------
class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_200(self):
        from httpx import ASGITransport, AsyncClient
        from src.api.server import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] in ("healthy", "no_model")
            assert "backend" in data
            assert "model_loaded" in data

    @pytest.mark.asyncio
    async def test_health_response_schema(self):
        from httpx import ASGITransport, AsyncClient
        from src.api.server import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
            data = resp.json()
            assert isinstance(data["model_loaded"], bool)
            assert data["backend"] in ("triton", "ultralytics", "none")


class TestClassesEndpoint:
    @pytest.mark.asyncio
    async def test_returns_10_classes(self):
        from httpx import ASGITransport, AsyncClient
        from src.api.server import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/classes")
            assert resp.status_code == 200
            data = resp.json()
            assert data["num_classes"] == 10
            assert "scratch" in data["classes"]
            assert "delamination" in data["classes"]


class TestMetricsEndpoint:
    @pytest.mark.asyncio
    async def test_metrics_returns_counters(self):
        from httpx import ASGITransport, AsyncClient
        from src.api.server import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/metrics")
            assert resp.status_code == 200
            data = resp.json()
            assert "total_requests" in data
            assert "inference_count" in data
            assert "backend" in data


class TestDetectEndpoint:
    @pytest.fixture(autouse=True)
    def _check_model(self):
        from pathlib import Path

        if not Path("models/best.pt").exists():
            pytest.skip("Model file not available (models/best.pt)")

    @pytest.mark.asyncio
    async def test_detect_returns_valid_response(self, wafer_image):
        from httpx import ASGITransport, AsyncClient
        from src.api.server import app

        await _ensure_startup()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/detect",
                files={"file": ("wafer.jpg", wafer_image, "image/jpeg")},
                params={"confidence": 0.25},
            )
            # Should return 200 even with no detections (synthetic image has no defects)
            assert resp.status_code == 200
            data = resp.json()
            assert "detections" in data
            assert "inference_time_ms" in data
            assert "total_defects" in data
            assert isinstance(data["detections"], list)
            assert data["inference_time_ms"] > 0

    @pytest.mark.asyncio
    async def test_detect_with_low_confidence(self, wafer_image):
        from httpx import ASGITransport, AsyncClient
        from src.api.server import app

        await _ensure_startup()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/detect",
                files={"file": ("wafer.jpg", wafer_image, "image/jpeg")},
                params={"confidence": 0.01},
            )
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_detect_response_schema(self, wafer_image):
        from httpx import ASGITransport, AsyncClient
        from src.api.server import app

        await _ensure_startup()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/detect",
                files={"file": ("wafer.jpg", wafer_image, "image/jpeg")},
            )
            data = resp.json()
            assert "image_name" in data
            assert data["image_name"] == "wafer.jpg"
            for det in data["detections"]:
                assert "class_name" in det
                assert "class_id" in det
                assert "confidence" in det
                assert "bbox" in det
                assert len(det["bbox"]) == 4
                assert 0 <= det["confidence"] <= 1


class TestBatchDetectEndpoint:
    @pytest.fixture(autouse=True)
    def _check_model(self):
        from pathlib import Path

        if not Path("models/best.pt").exists():
            pytest.skip("Model file not available (models/best.pt)")

    @pytest.mark.asyncio
    async def test_batch_detect_multiple_images(self, wafer_image):
        from httpx import ASGITransport, AsyncClient
        from src.api.server import app

        await _ensure_startup()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            files = [
                ("files", ("wafer1.jpg", wafer_image, "image/jpeg")),
                ("files", ("wafer2.jpg", wafer_image, "image/jpeg")),
            ]
            resp = await client.post("/detect/batch", files=files)
            assert resp.status_code == 200
            data = resp.json()
            assert isinstance(data, list)
            assert len(data) == 2


class TestAuthEndpoints:
    @pytest.mark.asyncio
    async def test_token_endpoint_valid_key(self):
        from httpx import ASGITransport, AsyncClient
        from src.api.server import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/token",
                json={"api_key": "dev-api-key-change-in-prod"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "access_token" in data
            assert data["token_type"] == "bearer"
            assert data["expires_in"] > 0

    @pytest.mark.asyncio
    async def test_token_endpoint_invalid_key(self):
        from httpx import ASGITransport, AsyncClient
        from src.api.server import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/token",
                json={"api_key": "wrong-key"},
            )
            assert resp.status_code == 401


class TestRealWaferInference:
    """Tests with actual trained model on realistic wafer images (requires model files)."""

    @pytest.fixture(autouse=True)
    def _check_model(self):
        from pathlib import Path

        if not Path("models/best.pt").exists():
            pytest.skip("Model file not available (models/best.pt)")

    @pytest.mark.asyncio
    async def test_realistic_wafer_detection(self):
        """Run inference on a realistic wafer image and verify detections."""
        from pathlib import Path

        from httpx import ASGITransport, AsyncClient
        from src.api.server import app

        await _ensure_startup()

        # Use one of the realistic wafer images if available
        realistic_dir = Path("outputs/realistic_unseen")
        images = list(realistic_dir.glob("realistic_*.jpg")) if realistic_dir.exists() else []
        if not images:
            pytest.skip("No realistic wafer images found")

        with open(images[0], "rb") as f:
            img_bytes = f.read()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/detect",
                files={"file": (images[0].name, img_bytes, "image/jpeg")},
                params={"confidence": 0.25},
            )
            assert resp.status_code == 200
            data = resp.json()
            # Should detect at least 1 defect on realistic wafer images
            assert data["total_defects"] >= 1
            # All class names should be valid
            valid_classes = {
                "scratch", "particle", "edge_chip", "void", "pattern_shift",
                "bridge", "missing_bond", "crack", "contamination", "delamination",
            }
            for det in data["detections"]:
                assert det["class_name"] in valid_classes
