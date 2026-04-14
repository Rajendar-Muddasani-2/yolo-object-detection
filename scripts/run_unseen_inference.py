"""
Unseen data inference script for YOLOv8 Wafer Defect Detection.
Runs best.pt on outputs/unseen_test_images/ and saves annotated results.
"""

import json
import time
from pathlib import Path

from ultralytics import YOLO

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "best.pt"
UNSEEN_DIR = ROOT / "outputs" / "unseen_test_images"
RESULTS_DIR = ROOT / "outputs" / "unseen_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

CLASS_NAMES = [
    "scratch", "particle", "edge_chip", "void", "pattern_shift",
    "bridge", "missing_bond", "crack", "contamination", "delamination",
]


def run_inference():
    print(f"Model  : {MODEL_PATH}")
    print(f"Input  : {UNSEEN_DIR}")
    print(f"Output : {RESULTS_DIR}")
    print(f"Conf   : {CONF_THRESHOLD}  IoU: {IOU_THRESHOLD}")
    print()

    model = YOLO(str(MODEL_PATH))

    images = sorted(UNSEEN_DIR.glob("*.jpg")) + sorted(UNSEEN_DIR.glob("*.png"))
    if not images:
        raise FileNotFoundError(f"No images found in {UNSEEN_DIR}")
    print(f"Found {len(images)} unseen images\n")

    all_results = []
    class_counts: dict[str, int] = {c: 0 for c in CLASS_NAMES}
    total_detections = 0
    latencies = []

    for img_path in images:
        t0 = time.perf_counter()
        results = model.predict(
            source=str(img_path),
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False,
            save=True,
            project=str(RESULTS_DIR),
            name="annotated",
            exist_ok=True,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

        result = results[0]
        boxes = result.boxes
        detections = []
        if boxes is not None and len(boxes):
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].tolist()
                cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
                detections.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": round(conf, 4),
                    "bbox_xyxy": [round(v, 1) for v in xyxy],
                })
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                total_detections += 1

        img_result = {
            "image": img_path.name,
            "num_detections": len(detections),
            "inference_ms": round(elapsed_ms, 1),
            "detections": detections,
        }
        all_results.append(img_result)

        status = f"{len(detections)} det" if detections else "CLEAN"
        classes_found = list({d["class_name"] for d in detections})
        print(f"  {img_path.name} → {status} | {elapsed_ms:.0f}ms"
              + (f" | {', '.join(classes_found)}" if classes_found else ""))

    # ── Summary ────────────────────────────────────────────────────────────────
    mean_ms = sum(latencies) / len(latencies)
    fps = 1000 / mean_ms
    images_with_defects = sum(1 for r in all_results if r["num_detections"] > 0)
    defect_rate = images_with_defects / len(all_results)

    summary = {
        "model": "YOLOv8-Large (best.pt)",
        "conf_threshold": CONF_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
        "total_images": len(all_results),
        "images_with_defects": images_with_defects,
        "clean_images": len(all_results) - images_with_defects,
        "defect_detection_rate": round(defect_rate, 4),
        "total_detections": total_detections,
        "mean_detections_per_image": round(total_detections / len(all_results), 2),
        "inference_speed": {
            "mean_ms": round(mean_ms, 1),
            "fps": round(fps, 1),
        },
        "class_distribution": {k: v for k, v in class_counts.items() if v > 0},
        "per_image": all_results,
    }

    out_json = RESULTS_DIR / "unseen_inference_results.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 60)
    print("UNSEEN INFERENCE SUMMARY")
    print("=" * 60)
    print(f"  Images processed : {summary['total_images']}")
    print(f"  With defects     : {summary['images_with_defects']} ({defect_rate:.1%})")
    print(f"  Clean            : {summary['clean_images']}")
    print(f"  Total detections : {summary['total_detections']}")
    print(f"  Mean det/image   : {summary['mean_detections_per_image']}")
    print(f"  Inference speed  : {mean_ms:.1f}ms ({fps:.1f} FPS) on CPU")
    print()
    print("  Class distribution:")
    for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
        if cnt > 0:
            print(f"    {cls:<20} {cnt}")
    print()
    print(f"  Results saved to : {RESULTS_DIR}")
    print(f"  JSON summary     : {out_json}")
    return summary


if __name__ == "__main__":
    run_inference()
