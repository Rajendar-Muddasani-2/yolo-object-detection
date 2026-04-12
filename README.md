# YOLOv8 Wafer Defect Detection

Production-grade semiconductor wafer defect detection using **YOLOv8-Large** (44M parameters), trained on 20K+ synthetic defect images across 10 defect classes with optional MVTec AD real-world industrial defect integration.

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌───────────────┐     ┌──────────────┐
│  React UI    │────▶│  FastAPI      │────▶│  Triton /     │────▶│  YOLOv8-L    │
│  (TypeScript)│◀────│  Gateway      │◀────│  Ultralytics  │◀────│  ONNX Model  │
└──────────────┘     └──────────────┘     └───────────────┘     └──────────────┘
                            │                                          │
                     ┌──────┴──────┐                           ┌───────┴───────┐
                     │ Prometheus  │                           │   MLflow      │
                     │ + Grafana   │                           │   Tracking    │
                     └─────────────┘                           └───────────────┘
```

## Defect Classes

| ID | Class | Description |
|----|-------|-------------|
| 0 | scratch | Linear surface damage |
| 1 | particle | Foreign material contamination |
| 2 | edge_chip | Chipping at wafer edges |
| 3 | void | Missing material / holes |
| 4 | pattern_shift | Lithography misalignment |
| 5 | bridge | Unintended metal connections |
| 6 | missing_bond | Failed wire bonds |
| 7 | crack | Structural fractures |
| 8 | contamination | Chemical residue |
| 9 | delamination | Layer separation |

## Quick Start

### 1. Generate Synthetic Dataset

```bash
python -c "from src.data_generator import generate_dataset; generate_dataset('data/wafer_defects', n_images=1000)"
```

### 2. Train on GPU (Colab/Kaggle)

Upload `notebooks/train_yolov8_colab.ipynb` to Google Colab with A100/T4 runtime. The notebook handles:
- 20K image generation + MVTec AD integration
- YOLOv8-Large training (100 epochs)
- Multi-scale model comparison (S vs M vs L)
- ONNX + TensorRT export with speed benchmarks

### 3. Run Inference API

```bash
# With Docker
docker compose up api

# Without Docker
uvicorn src.api.server:app --host 0.0.0.0 --port 8080
```

### 4. Full Stack (Triton + API + React + Monitoring)

```bash
docker compose up -d
```

Services:
| Service | Port | Description |
|---------|------|-------------|
| React Frontend | 3000 | Drag-drop detection UI |
| FastAPI Gateway | 8080 | REST API with /detect endpoint |
| Triton Server | 8000-8002 | GPU inference server |
| MLflow | 5000 | Experiment tracking |
| Prometheus | 9090 | Metrics collection |
| Grafana | 3001 | Dashboards |
| Redis | 6379 | Response caching |

## API Endpoints

```
POST /detect           - Detect defects in uploaded image
POST /detect/batch     - Batch detection (up to 16 images)
GET  /health           - Service health check
GET  /classes          - List defect classes
GET  /metrics          - Prometheus metrics
```

## Project Structure

```
├── notebooks/
│   └── train_yolov8_colab.ipynb    # GPU training notebook (Colab A100)
├── src/
│   ├── data_generator.py           # 10-class synthetic wafer defect generator
│   ├── mvtec_integration.py        # MVTec AD converter + dataset merger
│   ├── yolo_utils.py               # Detection utilities (Ultralytics-native)
│   └── api/
│       └── server.py               # FastAPI gateway (Triton + fallback)
├── frontend/                       # React + TypeScript + Vite
│   ├── src/App.tsx                 # Detection UI with canvas overlay
│   └── Dockerfile                  # Multi-stage Node → Nginx
├── triton_model_repo/              # NVIDIA Triton model config
├── monitoring/                     # Prometheus + Grafana configs
├── docker-compose.yml              # Full 7-service stack
├── Dockerfile                      # FastAPI service container
└── tests/                          # pytest suite
```

## Training Results

*Results populated after GPU training — see `outputs/` directory after running notebook.*

| Model | mAP@50 | mAP@50:95 | Params | Training Time |
|-------|--------|-----------|--------|---------------|
| YOLOv8-S | — | — | 11.2M | — |
| YOLOv8-M | — | — | 25.9M | — |
| **YOLOv8-L** | — | — | **43.7M** | — |

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT

⭐ **Star this repository if you find it helpful!**
