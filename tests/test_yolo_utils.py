"""Tests for wafer defect detection utilities."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from PIL import Image

from src.yolo_utils import (
    DEFECT_CLASSES,
    compute_iou,
    draw_detections,
)


class TestDefectClasses:
    def test_has_10_classes(self):
        assert len(DEFECT_CLASSES) == 10

    def test_expected_classes(self):
        assert "scratch" in DEFECT_CLASSES
        assert "particle" in DEFECT_CLASSES
        assert "crack" in DEFECT_CLASSES
        assert "delamination" in DEFECT_CLASSES

    def test_no_duplicates(self):
        assert len(DEFECT_CLASSES) == len(set(DEFECT_CLASSES))


class TestComputeIoU:
    def test_identical_boxes(self):
        box = (10, 10, 50, 50)
        assert compute_iou(box, box) == 1.0

    def test_no_overlap(self):
        box1 = (0, 0, 10, 10)
        box2 = (20, 20, 30, 30)
        assert compute_iou(box1, box2) == 0.0

    def test_partial_overlap(self):
        box1 = (0, 0, 20, 20)
        box2 = (10, 10, 30, 30)
        iou_val = compute_iou(box1, box2)
        assert 0 < iou_val < 1
        expected = 100 / (400 + 400 - 100)
        assert abs(iou_val - expected) < 1e-6

    def test_contained_box(self):
        outer = (0, 0, 100, 100)
        inner = (25, 25, 75, 75)
        iou_val = compute_iou(outer, inner)
        expected = 2500 / (10000 + 2500 - 2500)
        assert abs(iou_val - expected) < 1e-6

    def test_touching_edges(self):
        box1 = (0, 0, 10, 10)
        box2 = (10, 0, 20, 10)
        assert compute_iou(box1, box2) == 0.0

    def test_symmetry(self):
        box1 = (5, 5, 25, 25)
        box2 = (15, 15, 35, 35)
        assert compute_iou(box1, box2) == compute_iou(box2, box1)


class TestDrawDetections:
    def test_draw_on_image(self):
        img = Image.new("RGB", (640, 640), color=(128, 128, 128))
        detections = [
            {"class_id": 0, "class_name": "scratch", "confidence": 0.95, "bbox": [10, 10, 100, 100]},
            {"class_id": 3, "class_name": "void", "confidence": 0.87, "bbox": [200, 200, 350, 350]},
        ]
        result = draw_detections(img, detections)
        assert isinstance(result, Image.Image)
        assert result.size == (640, 640)
        # Original should not be modified
        assert img is not result

    def test_empty_detections(self):
        img = Image.new("RGB", (640, 640))
        result = draw_detections(img, [])
        assert isinstance(result, Image.Image)


class TestDataGenerator:
    def test_import(self):
        from src.data_generator import generate_dataset
        assert callable(generate_dataset)

    def test_generate_small_dataset(self, tmp_path):
        from src.data_generator import generate_dataset

        stats = generate_dataset(
            output_dir=str(tmp_path / "data"),
            n_images=10,
            max_defects_per_image=3,
            seed=42,
            n_workers=1,
        )
        # Check splits exist
        for split in ["train", "val", "test"]:
            assert (tmp_path / "data" / split / "images").exists()
            assert (tmp_path / "data" / split / "labels").exists()
        # Check data.yaml
        assert (tmp_path / "data" / "data.yaml").exists()


class TestAPIServer:
    def test_import(self):
        from src.api.server import app
        assert app is not None

    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        from httpx import ASGITransport, AsyncClient
        from src.api.server import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert "status" in data

    @pytest.mark.asyncio
    async def test_classes_endpoint(self):
        from httpx import ASGITransport, AsyncClient
        from src.api.server import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/classes")
            assert resp.status_code == 200
            data = resp.json()
            assert data["num_classes"] == 10
            assert "scratch" in data["classes"]
