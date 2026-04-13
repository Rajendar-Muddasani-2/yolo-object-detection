"""Write train_yolov8_colab_part2.ipynb as valid JSON."""
import json
from pathlib import Path

def code_cell(src):
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src}

def md_cell(src):
    return {"cell_type":"markdown","metadata":{},"source":src}

cells = []

# ── Cell 1: Title / results table ──────────────────────────────────────────
cells.append(md_cell(
    "# YOLOv8-Large Wafer Defect Detection — Part 2 (Steps 7–12)\n\n"
    "Continues from the completed A100 training run in Part 1.\n\n"
    "## Results Already Captured (Part 1 — 100 epochs on A100)\n\n"
    "| Metric | Value |\n"
    "|--------|-------|\n"
    "| GPU | NVIDIA A100-SXM4-40GB |\n"
    "| Model | YOLOv8-Large (43.7M params) |\n"
    "| Epochs | 100 |\n"
    "| **mAP@50** | **0.9922** |\n"
    "| **mAP@50:95** | **0.9576** |\n"
    "| **Precision** | **0.9907** |\n"
    "| **Recall** | **0.9861** |\n"
    "| Training time | 341.9 minutes |\n\n"
    "> **Run Cell 2 FIRST** — it checks that `best.pt` still exists in `/content/`.\n"
    "> If the Colab runtime was disconnected, weights are gone and you must re-run Part 1.\n"
    "> **Run Cell 3 immediately after** — it backs up `best.pt` to Google Drive."
))

# ── Cell 2: Runtime check ──────────────────────────────────────────────────
cells.append(md_cell("## Step 0A — Verify Runtime Is Still Alive\n\nFail fast if `best.pt` is missing."))

cells.append(code_cell(
    "import subprocess, os, sys\n"
    "from pathlib import Path\n\n"
    "result = subprocess.run(\n"
    "    ['find', '/content', '-name', 'best.pt', '-type', 'f'],\n"
    "    capture_output=True, text=True\n"
    ")\n"
    "best_pts = [p for p in result.stdout.strip().split('\\n') if p]\n\n"
    "print('=== RUNTIME STATUS ===')\n"
    "print(f'best.pt files found: {best_pts}')\n"
    "print(f'Repo exists: {Path(\"/content/yolo-object-detection\").exists()}')\n"
    "print(f'Dataset yaml: {Path(\"/content/yolo-object-detection/data/wafer_defects/data.yaml\").exists()}')\n\n"
    "if not best_pts:\n"
    "    raise RuntimeError(\n"
    "        'Runtime was disconnected — weights lost.\\n'\n"
    "        'Re-run Part 1 notebook to re-train. Then come back here.'\n"
    "    )\n\n"
    "# Prefer the yolov8l_wafer run directory if multiple matches\n"
    "BEST_PT = next((p for p in best_pts if 'yolov8l_wafer' in p), best_pts[0])\n"
    "print(f'\\nUsing: {BEST_PT}  ({Path(BEST_PT).stat().st_size / 1e6:.1f} MB)')\n"
    "print('Runtime alive — proceed to backup!')\n"
))

# ── Cell 3: Google Drive backup ────────────────────────────────────────────
cells.append(md_cell("## Step 0B — Backup best.pt to Google Drive (do this FIRST)"))

cells.append(code_cell(
    "from google.colab import drive\n"
    "import shutil\n\n"
    "drive.mount('/drive')\n\n"
    "drive_dir = Path('/drive/MyDrive/yolo_wafer_defect_a100')\n"
    "drive_dir.mkdir(parents=True, exist_ok=True)\n\n"
    "dest = drive_dir / 'yolov8l_wafer_best.pt'\n"
    "shutil.copy(BEST_PT, dest)\n"
    "print(f'Model backed up: {dest}  ({dest.stat().st_size / 1e6:.1f} MB)')\n\n"
    "# Backup any training plots that exist\n"
    "run_dir = Path(BEST_PT).parent.parent\n"
    "for f in list(run_dir.glob('*.png')) + list(run_dir.glob('*.csv')):\n"
    "    shutil.copy(f, drive_dir / f.name)\n"
    "    print(f'  Backed up: {f.name}')\n\n"
    "print('\\nBackup complete. Safe to continue with remaining steps.')\n"
))

# ── Cell 4: Restore env ────────────────────────────────────────────────────
cells.append(md_cell(
    "## Step 1 — Reinstall Packages + Restore Variables\n\n"
    "The Colab kernel was reset after the 5.7-hour session. Re-install `ultralytics` "
    "and restore all constants from the completed A100 run."
))

cells.append(code_cell(
    "import subprocess\n"
    "subprocess.run(['pip', 'install', '-q', 'ultralytics', 'mlflow', 'onnx', 'onnxruntime-gpu'], check=True)\n\n"
    "import os, sys, time, json, shutil\n"
    "import numpy as np\n"
    "import matplotlib.pyplot as plt\n"
    "from pathlib import Path\n"
    "from ultralytics import YOLO\n"
    "import torch\n"
    "import mlflow\n\n"
    "os.chdir('/content/yolo-object-detection')\n"
    "sys.path.insert(0, '.')\n\n"
    "print(f'PyTorch  : {torch.__version__}')\n"
    "print(f'CUDA     : {torch.cuda.is_available()}')\n"
    "if torch.cuda.is_available():\n"
    "    print(f'GPU      : {torch.cuda.get_device_name(0)}')\n"
    "    print(f'VRAM     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')\n\n"
    "# ── constants ──────────────────────────────────────────────────────────\n"
    "DATA_YAML  = 'data/wafer_defects/data.yaml'\n"
    "BATCH_SIZE = 16\n"
    "WORKERS    = 4\n\n"
    "CLASSES = [\n"
    "    'scratch', 'particle', 'edge_chip', 'void', 'pattern_shift',\n"
    "    'bridge', 'missing_bond', 'crack', 'contamination', 'delamination',\n"
    "]\n\n"
    "# Hardcoded from completed Cell 13 output\n"
    "L_RESULTS = {\n"
    "    'mAP50': 0.9922, 'mAP50_95': 0.9576,\n"
    "    'precision': 0.9907, 'recall': 0.9861,\n"
    "    'training_time_min': 341.9,\n"
    "}\n\n"
    "PER_CLASS_RESULTS = {\n"
    "    'scratch':       {'mAP50': 0.979,  'mAP50_95': 0.878},\n"
    "    'particle':      {'mAP50': 0.995,  'mAP50_95': 0.938},\n"
    "    'edge_chip':     {'mAP50': 0.995,  'mAP50_95': 0.984},\n"
    "    'void':          {'mAP50': 0.995,  'mAP50_95': 0.995},\n"
    "    'pattern_shift': {'mAP50': 0.995,  'mAP50_95': 0.995},\n"
    "    'bridge':        {'mAP50': 0.984,  'mAP50_95': 0.845},\n"
    "    'missing_bond':  {'mAP50': 0.995,  'mAP50_95': 0.994},\n"
    "    'crack':         {'mAP50': 0.994,  'mAP50_95': 0.967},\n"
    "    'contamination': {'mAP50': 0.995,  'mAP50_95': 0.987},\n"
    "    'delamination':  {'mAP50': 0.995,  'mAP50_95': 0.993},\n"
    "}\n\n"
    "Path('outputs').mkdir(exist_ok=True)\n"
    "Path('models').mkdir(exist_ok=True)\n\n"
    "print('\\nVariables restored from completed A100 training:')\n"
    "print(f'  mAP@50={L_RESULTS[\"mAP50\"]:.4f}  mAP@50:95={L_RESULTS[\"mAP50_95\"]:.4f}')\n"
    "print(f'  Training time: {L_RESULTS[\"training_time_min\"]:.1f} min  |  Model: {BEST_PT}')\n"
))

# ── Cell 5: Model size comparison ─────────────────────────────────────────
cells.append(md_cell(
    "## Step 7 — Model Size Comparison (S vs M vs L)\n\n"
    "Train YOLOv8-S and YOLOv8-M for 20 epochs as baselines.\n"
    "YOLOv8-L was already trained for 100 epochs — results are hardcoded above."
))

cells.append(code_cell(
    "mlflow.set_tracking_uri('file:///content/yolo-object-detection/mlruns')\n"
    "mlflow.set_experiment('wafer-defect-comparison')\n\n"
    "comparison_results = {}\n\n"
    "for model_file, label, batch in [('yolov8s.pt', 'YOLOv8-S', 32),\n"
    "                                  ('yolov8m.pt', 'YOLOv8-M', 16)]:\n"
    "    print(f'\\n{\"=\"*50}')\n"
    "    print(f'Training {label} — 20 epochs')\n"
    "    print('='*50)\n"
    "    comp_model = YOLO(model_file)\n"
    "    if mlflow.active_run():\n"
    "        mlflow.end_run()\n"
    "    with mlflow.start_run(run_name=f'{label.lower().replace(\"-\",\"\")}-comparison'):\n"
    "        t0 = time.time()\n"
    "        r = comp_model.train(\n"
    "            data=DATA_YAML, epochs=20, imgsz=640,\n"
    "            batch=batch, patience=10, device=0,\n"
    "            workers=WORKERS,\n"
    "            project='runs/detect',\n"
    "            name=f'{label.lower().replace(\"-\",\"\")}_wafer',\n"
    "            exist_ok=True, mosaic=1.0, cos_lr=True,\n"
    "        )\n"
    "        elapsed = time.time() - t0\n"
    "        comparison_results[label] = {\n"
    "            'mAP50':            r.results_dict.get('metrics/mAP50(B)', 0),\n"
    "            'mAP50_95':         r.results_dict.get('metrics/mAP50-95(B)', 0),\n"
    "            'precision':        r.results_dict.get('metrics/precision(B)', 0),\n"
    "            'recall':           r.results_dict.get('metrics/recall(B)', 0),\n"
    "            'training_time_min': round(elapsed / 60, 1),\n"
    "        }\n"
    "        mlflow.log_metrics(comparison_results[label])\n\n"
    "if mlflow.active_run():\n"
    "    mlflow.end_run()\n\n"
    "# Add L results (100 epochs)\n"
    "comparison_results['YOLOv8-L'] = L_RESULTS.copy()\n\n"
    "print(f'\\n{\"=\"*72}')\n"
    "print(f'{\"Model\":<12} {\"mAP@50\":>10} {\"mAP@50:95\":>12} {\"Precision\":>10} {\"Recall\":>10} {\"Time\":>8}')\n"
    "print('-'*72)\n"
    "for name, res in comparison_results.items():\n"
    "    ep = '(100ep)' if name == 'YOLOv8-L' else '(20ep) '\n"
    "    print(f'{name:<12} {res[\"mAP50\"]:>10.4f} {res[\"mAP50_95\"]:>12.4f} '\n"
    "          f'{res.get(\"precision\",0):>10.4f} {res.get(\"recall\",0):>10.4f} '\n"
    "          f'{res[\"training_time_min\"]:>6.1f}m {ep}')\n"
    "print('='*72)\n"
))

# ── Cell 6: Test evaluation ────────────────────────────────────────────────
cells.append(md_cell("## Step 8 — Evaluate on Held-Out Test Set\n\n3 000 unseen test images. Recall is critical for semiconductor inspection."))

cells.append(code_cell(
    "best_model = YOLO(BEST_PT)\n\n"
    "test_results = best_model.val(\n"
    "    data=DATA_YAML, split='test',\n"
    "    batch=16, device=0,\n"
    "    plots=True, save_json=True,\n"
    ")\n\n"
    "print(f'\\n{\"=\"*50}')\n"
    "print('TEST SET RESULTS  (3 000 unseen images)')\n"
    "print('='*50)\n"
    "rd = test_results.results_dict\n"
    "print(f\"  mAP@50:     {rd.get('metrics/mAP50(B)',    0):.4f}\")\n"
    "print(f\"  mAP@50:95:  {rd.get('metrics/mAP50-95(B)', 0):.4f}\")\n"
    "print(f\"  Precision:  {rd.get('metrics/precision(B)',0):.4f}\")\n"
    "print(f\"  Recall:     {rd.get('metrics/recall(B)',   0):.4f}\")\n\n"
    "if hasattr(test_results, 'maps'):\n"
    "    print('\\nPer-class mAP@50:')\n"
    "    for i, cls_name in enumerate(CLASSES):\n"
    "        if i < len(test_results.maps):\n"
    "            print(f'  {cls_name:20s}: {test_results.maps[i]:.4f}')\n"
))

# ── Cell 7: ONNX export + speed ────────────────────────────────────────────
cells.append(md_cell("## Step 9 — ONNX Export + Speed Benchmark"))

cells.append(code_cell(
    "# Export ONNX\n"
    "print('Exporting to ONNX...')\n"
    "best_model.export(format='onnx', dynamic=True, simplify=True, opset=17)\n\n"
    "r2 = subprocess.run(\n"
    "    ['find', '/content', '-name', 'best.onnx', '-type', 'f'],\n"
    "    capture_output=True, text=True\n"
    ")\n"
    "BEST_ONNX = [p for p in r2.stdout.strip().split('\\n') if p][0]\n\n"
    "shutil.copy(BEST_PT,   'models/best.pt')\n"
    "shutil.copy(BEST_ONNX, 'models/best.onnx')\n"
    "print(f'PT  : {Path(\"models/best.pt\").stat().st_size / 1e6:.1f} MB')\n"
    "print(f'ONNX: {Path(\"models/best.onnx\").stat().st_size / 1e6:.1f} MB')\n\n"
    "# Save ONNX to Drive\n"
    "drive_dir = Path('/drive/MyDrive/yolo_wafer_defect_a100')\n"
    "if drive_dir.exists():\n"
    "    shutil.copy('models/best.onnx', drive_dir / 'yolov8l_wafer_best.onnx')\n"
    "    print('ONNX copied to Google Drive')\n\n"
    "# Speed benchmark\n"
    "dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)\n"
    "N_WARMUP, N_RUNS = 10, 50\n"
    "results_bench = {}\n\n"
    "for fmt, path in [('PyTorch', 'models/best.pt'), ('ONNX', 'models/best.onnx')]:\n"
    "    m = YOLO(path)\n"
    "    for _ in range(N_WARMUP):\n"
    "        m(dummy_img, verbose=False)\n"
    "    latencies = []\n"
    "    for _ in range(N_RUNS):\n"
    "        t0 = time.perf_counter()\n"
    "        m(dummy_img, verbose=False)\n"
    "        latencies.append((time.perf_counter() - t0) * 1000)\n"
    "    results_bench[fmt] = {\n"
    "        'mean_ms': round(float(np.mean(latencies)), 1),\n"
    "        'p50_ms':  round(float(np.median(latencies)), 1),\n"
    "        'p95_ms':  round(float(np.percentile(latencies, 95)), 1),\n"
    "        'fps':     round(1000 / float(np.mean(latencies)), 1),\n"
    "    }\n"
    "    print(f\"{fmt:12s}: {results_bench[fmt]['mean_ms']}ms avg | {results_bench[fmt]['fps']} FPS\")\n\n"
    "# TensorRT FP16 (may not be available on all Colab GPUs)\n"
    "try:\n"
    "    print('\\nExporting TensorRT FP16...')\n"
    "    best_model.export(format='engine', device=0, half=True)\n"
    "    r3 = subprocess.run(\n"
    "        ['find', '/content', '-name', 'best.engine', '-type', 'f'],\n"
    "        capture_output=True, text=True\n"
    "    )\n"
    "    trt_path = [p for p in r3.stdout.strip().split('\\n') if p][0]\n"
    "    trt_model = YOLO(trt_path)\n"
    "    for _ in range(N_WARMUP):\n"
    "        trt_model(dummy_img, verbose=False)\n"
    "    latencies = []\n"
    "    for _ in range(N_RUNS):\n"
    "        t0 = time.perf_counter()\n"
    "        trt_model(dummy_img, verbose=False)\n"
    "        latencies.append((time.perf_counter() - t0) * 1000)\n"
    "    results_bench['TensorRT-FP16'] = {\n"
    "        'mean_ms': round(float(np.mean(latencies)), 1),\n"
    "        'p50_ms':  round(float(np.median(latencies)), 1),\n"
    "        'p95_ms':  round(float(np.percentile(latencies, 95)), 1),\n"
    "        'fps':     round(1000 / float(np.mean(latencies)), 1),\n"
    "    }\n"
    "    print(f\"TensorRT-FP16: {results_bench['TensorRT-FP16']['mean_ms']}ms | \"\n"
    "          f\"{results_bench['TensorRT-FP16']['fps']} FPS\")\n"
    "except Exception as e:\n"
    "    print(f'TensorRT skipped (not available): {e}')\n\n"
    "with open('outputs/speed_benchmark.json', 'w') as f:\n"
    "    json.dump(results_bench, f, indent=2)\n"
    "print('\\nSaved: outputs/speed_benchmark.json')\n"
))

# ── Cell 8: Visualizations ─────────────────────────────────────────────────
cells.append(md_cell("## Step 10 — Publication-Quality Visualizations (3 figures)"))

cells.append(code_cell(
    "import matplotlib.patches as mpatches\n\n"
    "plt.style.use('dark_background')\n"
    "plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13})\n"
    "C3 = ['#3b82f6', '#22c55e', '#ef4444']\n\n"
    "# ── Figure 1: Model comparison ─────────────────────────────────────────\n"
    "models = ['YOLOv8-S', 'YOLOv8-M', 'YOLOv8-L']\n"
    "m50_vals  = [comparison_results.get(m, {}).get('mAP50', 0)             for m in models]\n"
    "m95_vals  = [comparison_results.get(m, {}).get('mAP50_95', 0)          for m in models]\n"
    "time_vals = [comparison_results.get(m, {}).get('training_time_min', 0) for m in models]\n\n"
    "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n"
    "for ax, vals, ylabel, title in [\n"
    "    (axes[0], m50_vals,  'mAP@50',    'mAP@50 by Model Size'),\n"
    "    (axes[1], m95_vals,  'mAP@50:95', 'mAP@50:95 by Model Size'),\n"
    "    (axes[2], time_vals, 'Minutes',   'Training Time'),\n"
    "]:\n"
    "    bars = ax.bar(models, vals, color=C3)\n"
    "    ax.set_title(title, fontweight='bold'); ax.set_ylabel(ylabel)\n"
    "    for bar, v in zip(bars, vals):\n"
    "        label = f'{v:.3f}' if v < 10 else f'{v:.0f} min'\n"
    "        ax.text(\n"
    "            bar.get_x() + bar.get_width() / 2,\n"
    "            bar.get_height() * 1.01,\n"
    "            label, ha='center', fontsize=10, fontweight='bold'\n"
    "        )\n"
    "plt.suptitle('YOLOv8 Model Comparison — Wafer Defect Detection', fontsize=15, fontweight='bold')\n"
    "plt.tight_layout()\n"
    "plt.savefig('outputs/model_comparison.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()\n\n"
    "# ── Figure 2: Speed benchmark ──────────────────────────────────────────\n"
    "fmts = list(results_bench.keys())\n"
    "fps_vals = [results_bench[f]['fps'] for f in fmts]\n"
    "ms_vals  = [results_bench[f]['mean_ms'] for f in fmts]\n\n"
    "fig, ax = plt.subplots(figsize=(10, 5))\n"
    "bars = ax.bar(fmts, fps_vals, color=C3[:len(fmts)])\n"
    "ax.set_title('Inference Speed Comparison (A100)', fontsize=14, fontweight='bold')\n"
    "ax.set_ylabel('FPS (higher is better)')\n"
    "for bar, fps, ms in zip(bars, fps_vals, ms_vals):\n"
    "    ax.text(\n"
    "        bar.get_x() + bar.get_width() / 2,\n"
    "        bar.get_height() + 0.5,\n"
    "        f'{fps:.1f} FPS\\n({ms:.1f}ms)',\n"
    "        ha='center', fontsize=10, fontweight='bold'\n"
    "    )\n"
    "plt.tight_layout()\n"
    "plt.savefig('outputs/speed_benchmark.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()\n\n"
    "# ── Figure 3: Per-class performance ───────────────────────────────────\n"
    "cls_names = list(PER_CLASS_RESULTS.keys())\n"
    "m50_cls  = [PER_CLASS_RESULTS[c]['mAP50']    for c in cls_names]\n"
    "m95_cls  = [PER_CLASS_RESULTS[c]['mAP50_95'] for c in cls_names]\n"
    "y = np.arange(len(cls_names))\n\n"
    "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n"
    "colors_50 = ['#22c55e' if v >= 0.99 else '#f97316' if v >= 0.97 else '#ef4444' for v in m50_cls]\n"
    "axes[0].barh(y, m50_cls, color=colors_50)\n"
    "axes[0].set_yticks(y); axes[0].set_yticklabels(cls_names)\n"
    "axes[0].set_xlabel('mAP@50'); axes[0].set_title('Per-Class mAP@50', fontweight='bold')\n"
    "axes[0].set_xlim(0.80, 1.02)\n"
    "for i, v in enumerate(m50_cls):\n"
    "    axes[0].text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=9)\n\n"
    "axes[1].barh(y, m95_cls, color='#3b82f6')\n"
    "axes[1].set_yticks(y); axes[1].set_yticklabels(cls_names)\n"
    "axes[1].set_xlabel('mAP@50:95'); axes[1].set_title('Per-Class mAP@50:95', fontweight='bold')\n"
    "axes[1].set_xlim(0.70, 1.05)\n"
    "for i, v in enumerate(m95_cls):\n"
    "    axes[1].text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=9)\n\n"
    "plt.suptitle('YOLOv8-L Per-Class Performance (A100, 100 epochs)', fontsize=13, fontweight='bold')\n"
    "plt.tight_layout()\n"
    "plt.savefig('outputs/per_class_performance.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()\n\n"
    "print('3 figures saved to outputs/')\n"
))

# ── Cell 9: Inference ─────────────────────────────────────────────────────
cells.append(md_cell("## Step 11 — Inference on Sample Test Images"))

cells.append(code_cell(
    "import matplotlib.patches as patches\n"
    "from PIL import Image\n\n"
    "PALETTE = [\n"
    "    '#ef4444','#f97316','#eab308','#22c55e','#06b6d4',\n"
    "    '#3b82f6','#8b5cf6','#ec4899','#f43f5e','#14b8a6',\n"
    "]\n\n"
    "test_imgs = sorted(Path('data/wafer_defects/test/images').glob('*.jpg'))[:8]\n"
    "if not test_imgs:\n"
    "    test_imgs = sorted(Path('data/wafer_defects/valid/images').glob('*.jpg'))[:8]\n\n"
    "fig, axes = plt.subplots(2, 4, figsize=(20, 10))\n"
    "for ax, img_path in zip(axes.flat, test_imgs):\n"
    "    preds = best_model(str(img_path), conf=0.25, verbose=False)\n"
    "    img = Image.open(img_path)\n"
    "    ax.imshow(img)\n"
    "    n_det = 0\n"
    "    for r in preds:\n"
    "        for box in r.boxes:\n"
    "            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()\n"
    "            cls_id = int(box.cls[0])\n"
    "            conf   = float(box.conf[0])\n"
    "            color  = PALETTE[cls_id % len(PALETTE)]\n"
    "            ax.add_patch(\n"
    "                patches.Rectangle(\n"
    "                    (x1, y1), x2 - x1, y2 - y1,\n"
    "                    linewidth=2, edgecolor=color, facecolor='none'\n"
    "                )\n"
    "            )\n"
    "            ax.text(\n"
    "                x1, y1 - 4,\n"
    "                f'{CLASSES[cls_id]} {conf:.2f}',\n"
    "                fontsize=7, color='white', backgroundcolor=color\n"
    "            )\n"
    "            n_det += 1\n"
    "    ax.set_title(f'{img_path.name[:20]}  ({n_det} det)', fontsize=9)\n"
    "    ax.axis('off')\n\n"
    "plt.suptitle('YOLOv8-L Predictions on Test Images (conf >= 0.25)', fontsize=14, fontweight='bold')\n"
    "plt.tight_layout()\n"
    "plt.savefig('outputs/sample_predictions.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()\n"
    "print('Saved: outputs/sample_predictions.png')\n"
))

# ── Cell 10: Collect + ZIP + download ─────────────────────────────────────
cells.append(md_cell("## Step 12 — Collect Artifacts + Download ZIP"))

cells.append(code_cell(
    "import zipfile\n\n"
    "# Copy training plots from runs/ into outputs/\n"
    "r4 = subprocess.run(\n"
    "    ['find', '/content/yolo-object-detection/runs', '-name', '*.png', '-type', 'f'],\n"
    "    capture_output=True, text=True\n"
    ")\n"
    "for pf in [p for p in r4.stdout.strip().split('\\n') if p]:\n"
    "    try:\n"
    "        shutil.copy2(pf, f'outputs/{Path(pf).name}')\n"
    "    except Exception:\n"
    "        pass\n\n"
    "# Build results summary\n"
    "rd = test_results.results_dict\n"
    "summary = {\n"
    "    'model': 'YOLOv8-Large',\n"
    "    'params_M': 43.7,\n"
    "    'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'A100-SXM4-40GB',\n"
    "    'dataset': '20K synthetic wafer defects (10 classes)',\n"
    "    'epochs': 100,\n"
    "    'training_time_min': 341.9,\n"
    "    'test_metrics': {\n"
    "        'mAP50':     rd.get('metrics/mAP50(B)',     L_RESULTS['mAP50']),\n"
    "        'mAP50_95':  rd.get('metrics/mAP50-95(B)',  L_RESULTS['mAP50_95']),\n"
    "        'precision': rd.get('metrics/precision(B)', L_RESULTS['precision']),\n"
    "        'recall':    rd.get('metrics/recall(B)',    L_RESULTS['recall']),\n"
    "    },\n"
    "    'per_class':        PER_CLASS_RESULTS,\n"
    "    'model_comparison': comparison_results,\n"
    "    'speed_benchmark':  results_bench,\n"
    "}\n"
    "with open('outputs/results_summary.json', 'w') as f:\n"
    "    json.dump(summary, f, indent=2)\n\n"
    "print('=== FINAL METRICS ===')\n"
    "print(f'  mAP@50:    {summary[\"test_metrics\"][\"mAP50\"]:.4f}')\n"
    "print(f'  mAP@50:95: {summary[\"test_metrics\"][\"mAP50_95\"]:.4f}')\n"
    "print(f'  Precision: {summary[\"test_metrics\"][\"precision\"]:.4f}')\n"
    "print(f'  Recall:    {summary[\"test_metrics\"][\"recall\"]:.4f}')\n\n"
    "print('\\n=== OUTPUTS ===')\n"
    "for ff in sorted(Path('outputs').glob('*')):\n"
    "    print(f'  {ff.name:45s} {ff.stat().st_size / 1e3:7.0f} KB')\n\n"
    "# ZIP\n"
    "zip_path = '/content/yolov8_wafer_results.zip'\n"
    "with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:\n"
    "    for ff in Path('outputs').glob('*'):\n"
    "        zf.write(ff, f'outputs/{ff.name}')\n"
    "    for ff in Path('models').glob('*'):\n"
    "        zf.write(ff, f'models/{ff.name}')\n"
    "print(f'\\nZIP: {zip_path}  ({Path(zip_path).stat().st_size / 1e6:.1f} MB)')\n\n"
    "# Save to Drive\n"
    "drive_dir = Path('/drive/MyDrive/yolo_wafer_defect_a100')\n"
    "if drive_dir.exists():\n"
    "    shutil.copy(zip_path, drive_dir / 'yolov8_wafer_results.zip')\n"
    "    shutil.copy('outputs/results_summary.json', drive_dir / 'results_summary.json')\n"
    "    print('All artifacts saved to Google Drive')\n"
))

cells.append(code_cell(
    "from google.colab import files\n"
    "files.download('/content/yolov8_wafer_results.zip')\n"
    "print('Download started. Check your browser downloads folder.')\n"
))

# ── Assemble notebook ──────────────────────────────────────────────────────
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path('/Users/rajendarmuddasani/AIML/58_End2End/projects/yolo-object-detection/notebooks/train_yolov8_colab_part2.ipynb')
with open(out, 'w') as f:
    json.dump(nb, f, indent=1)

# ── Validate ───────────────────────────────────────────────────────────────
with open(out) as f:
    loaded = json.load(f)

print(f"Written  : {out}")
print(f"Size     : {out.stat().st_size:,} bytes")
print(f"Cells    : {len(loaded['cells'])}")
print(f"Valid JSON: yes")
