# Model Card: YOLOv8-Large Wafer Defect Detector

## Model Details

| Field | Value |
|-------|-------|
| Model | YOLOv8-Large (Ultralytics) |
| Parameters | 43.7M |
| Input Size | 640 x 640 |
| Framework | PyTorch 2.x → ONNX 17 |
| Training Hardware | Google Colab A100 (40 GB) |
| Training Time | ~3 hours (100 epochs, early stopped at 82) |
| License | AGPL-3.0 (Ultralytics) |

## Intended Use

**Primary use**: Automated visual inspection of semiconductor wafer surfaces for manufacturing defect detection in ATE (Automated Test Equipment) pipelines.

**Users**: Semiconductor process engineers, quality assurance teams, fab operations.

**Out of scope**: Medical imaging, autonomous vehicles, general object detection, security surveillance.

## Training Data

- **Source**: Synthetically generated wafer defect images
- **Volume**: 20,000+ images at 640x640 resolution
- **Split**: 70% train / 15% validation / 15% test
- **Classes**: 10 semiconductor defect types
- **Augmentation**: Mosaic, MixUp, CopyPaste, HSV shift, flip, rotate, scale

### Defect Classes

| ID | Class | Description |
|----|-------|-------------|
| 0 | scratch | Linear surface damage |
| 1 | particle | Foreign material contamination |
| 2 | edge_chip | Wafer edge damage |
| 3 | void | Missing material / air pockets |
| 4 | pattern_shift | Lithography misalignment |
| 5 | bridge | Unintended material connections |
| 6 | missing_bond | Failed wire bond connections |
| 7 | crack | Structural fractures |
| 8 | contamination | Chemical or organic residue |
| 9 | delamination | Layer separation |

## Performance

### Overall Metrics

| Metric | Value |
|--------|-------|
| mAP@50 | 99.22% |
| mAP@50:95 | 87.3% |
| Precision | 96.4% |
| Recall | 97.7% |

### Model Size Comparison

| Variant | Parameters | mAP@50 | Inference Time |
|---------|-----------|--------|---------------|
| YOLOv8-S | 11.2M | 78.4% | Fastest |
| YOLOv8-M | 25.9M | 92.1% | Medium |
| YOLOv8-L | 43.7M | 99.2% | 2.1ms (ONNX) |

### Export Formats

| Format | Size | Notes |
|--------|------|-------|
| PyTorch (.pt) | 84 MB | Training checkpoint |
| ONNX (.onnx) | 167 MB | Production serving via Triton |

## Limitations

- **Synthetic data only**: Trained on programmatically generated defects, not real fab images. Performance on real wafers may vary.
- **Single wafer type**: Training data uses uniform circular wafer backgrounds. Non-standard wafer geometries not tested.
- **Lighting conditions**: Synthetic images use controlled lighting. Performance under variable fab lighting not validated.
- **Defect overlap**: When multiple defects overlap spatially, detection accuracy may decrease.

## Ethical Considerations

- No personally identifiable information in training data
- No demographic or bias concerns (industrial application)
- False negatives could allow defective chips to pass QA — should be used alongside existing inspection processes, not as sole gate

## Deployment

Served via NVIDIA Triton Inference Server with ONNX backend. FastAPI gateway handles preprocessing and postprocessing. Docker Compose orchestrates the full 7-service stack.

```
Client → React Frontend → FastAPI Gateway → Triton (ONNX) → Response
                                ↓
                    Prometheus → Grafana (monitoring)
```

## Citation

```
@software{yolov8_wafer_defect,
  title = {YOLOv8 Wafer Defect Detection},
  year = {2025},
  url = {https://github.com/Rajendar-Muddasani-2/yolo-object-detection}
}
```
