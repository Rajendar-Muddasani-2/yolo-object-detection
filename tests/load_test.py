"""
Locust load testing for Wafer Defect Detection API.

Usage:
    # Install: pip install locust
    # Run:     locust -f tests/load_test.py --host http://localhost:8080
    # Web UI:  http://localhost:8089

    # Headless (CI/CD):
    # locust -f tests/load_test.py --host http://localhost:8080 \
    #        --headless -u 50 -r 10 --run-time 60s --csv results/load_test
"""

import io
import os

import numpy as np
from locust import HttpUser, between, task
from PIL import Image


def _generate_test_image(size: int = 640) -> bytes:
    """Generate a random wafer-like test image."""
    # Gray circle on dark background (simulates wafer)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    center = size // 2
    Y, X = np.ogrid[:size, :size]
    mask = (X - center) ** 2 + (Y - center) ** 2 <= (center - 20) ** 2
    img[mask] = [180, 180, 190]  # silicon gray
    # Add some noise
    noise = np.random.randint(-10, 10, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return buf.getvalue()


# Pre-generate test images to avoid overhead during load test
_TEST_IMAGES = [_generate_test_image() for _ in range(5)]
_API_KEY = os.getenv("API_KEY", "dev-api-key-change-in-prod")


class WaferDetectionUser(HttpUser):
    """Simulates a wafer inspection station sending images for defect detection."""

    wait_time = between(0.5, 2.0)  # 0.5-2s between requests

    def on_start(self):
        """Get JWT token on session start."""
        self.headers = {"X-API-Key": _API_KEY}

    @task(10)
    def detect_single(self):
        """POST /detect — single image inference (most common operation)."""
        img_data = _TEST_IMAGES[np.random.randint(len(_TEST_IMAGES))]
        self.client.post(
            "/detect",
            files={"file": ("wafer.jpg", img_data, "image/jpeg")},
            params={"confidence": 0.25},
            headers=self.headers,
        )

    @task(2)
    def detect_batch(self):
        """POST /detect/batch — batch inference (less common)."""
        files = [
            ("files", (f"wafer_{i}.jpg", _TEST_IMAGES[i % len(_TEST_IMAGES)], "image/jpeg"))
            for i in range(3)
        ]
        self.client.post(
            "/detect/batch",
            files=files,
            params={"confidence": 0.25},
            headers=self.headers,
        )

    @task(5)
    def health_check(self):
        """GET /health — lightweight health probe."""
        self.client.get("/health")

    @task(1)
    def get_metrics(self):
        """GET /metrics — monitoring endpoint."""
        self.client.get("/metrics")

    @task(1)
    def get_classes(self):
        """GET /classes — class taxonomy."""
        self.client.get("/classes")


class HeavyLoadUser(HttpUser):
    """Simulates burst traffic — rapid-fire detection requests."""

    wait_time = between(0.1, 0.5)

    def on_start(self):
        self.headers = {"X-API-Key": _API_KEY}

    @task
    def rapid_detect(self):
        img_data = _TEST_IMAGES[0]
        self.client.post(
            "/detect",
            files={"file": ("wafer.jpg", img_data, "image/jpeg")},
            headers=self.headers,
        )
