# Product Requirements Document (PRD)
# YOLOv8 Wafer Defect Detection

## Project Overview

**Project Name:** YOLOv8 Wafer Defect Detection  
**Status:** Active Development  
**Version:** 2.0.0

## Executive Summary

Production-grade semiconductor wafer defect detection system using YOLOv8-Large (44M parameters). Trained on 20K+ synthetic wafer defect images across 10 defect classes with optional MVTec AD real-world industrial defect integration. Full serving stack: Triton Inference Server, FastAPI gateway, React frontend, Prometheus/Grafana monitoring.

## Key Features

### 1. Multi-Class Defect Detection
- 10 semiconductor-specific defect classes (scratch, particle, edge_chip, void, pattern_shift, bridge, missing_bond, crack, contamination, delamination)
- YOLOv8-Large backbone (43.7M parameters)
- GPU-trained on Colab A100 with 100 epochs

### 2. Synthetic + Real Data Pipeline
- 20K synthetic wafer defect images with multiprocessing generation
- MVTec AD integration for real-world industrial defects
- Automated YOLO-format label conversion and dataset merging

### 3. Production Serving
- NVIDIA Triton Inference Server with ONNX backend
- FastAPI REST gateway with batch inference support
- React TypeScript frontend with drag-drop UI and canvas-based bounding box overlay
- Redis response caching

### 4. Monitoring & Observability
- Prometheus metrics (latency, throughput, error rate)
- Grafana dashboards
- MLflow experiment tracking with model comparison (S vs M vs L)

### 5. Model Export & Benchmarking
- ONNX export with dynamic batching
- TensorRT FP16 quantization
- Speed benchmarks: PyTorch vs ONNX vs TensorRT

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Model | Ultralytics YOLOv8-Large (PyTorch) |
| Training | Google Colab A100, MLflow tracking |
| Export | ONNX 17, TensorRT FP16 |
| Serving | NVIDIA Triton Inference Server |
| API | FastAPI + Uvicorn |
| Frontend | React 18 + TypeScript + Vite |
| Monitoring | Prometheus + Grafana |
| Cache | Redis |
| CI/CD | GitHub Actions |
| Container | Docker + Docker Compose |

## Data Pipeline

1. **Synthetic Generation** — 20K images, 10 defect classes, 640x640, YOLO format labels
2. **MVTec AD Integration** — Real industrial defects converted to YOLO bounding boxes
3. **Dataset Merge** — Weighted combination with unified class mapping
4. **Augmentation** — Mosaic, MixUp, CopyPaste, geometric + color transforms

## Metrics

| Metric | Target |
|--------|--------|
| mAP@50 | > 0.85 |
| mAP@50:95 | > 0.65 |
| Inference (ONNX) | < 15ms on A100 |
| Inference (TensorRT) | < 8ms on A100 |
