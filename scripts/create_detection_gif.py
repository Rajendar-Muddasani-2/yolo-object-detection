"""
Generate an accurate YOLO algorithm visualization GIF for the wafer defect project.

YOLO = You Only Look Once. The algorithm processes the ENTIRE image in ONE forward pass:

Phase 1: INPUT        - Raw wafer image enters the CNN
Phase 2: GRID         - CNN divides image into S×S grid (feature map)
Phase 3: PREDICT      - ALL grid cells simultaneously predict boxes + confidence + class
Phase 4: CANDIDATES   - Many candidate bounding boxes with varying confidence
Phase 5: NMS          - Non-Max Suppression removes overlapping/low-confidence boxes
Phase 6: RESULT       - Clean final detections with class labels

Layout:
  ┌──────────────────────────────────────────────────────────┐
  │  Title bar with model info                               │
  ├──────────────┬───────────────────────────────────────────┤
  │              │  Phase label + description                 │
  │   Wafer      │                                           │
  │   Image      │  Algorithm visualization                  │
  │              │  (grid overlay / candidate boxes / NMS)    │
  │              │                                           │
  ├──────────────┴───────────────────────────────────────────┤
  │  Stats / metrics bar                                     │
  └──────────────────────────────────────────────────────────┘
"""

import math
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ── Configuration ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
REALISTIC_DIR = ROOT / "outputs" / "realistic_unseen"
OUTPUT_GIF = ROOT / "outputs" / "yolo_wafer_detection.gif"

CANVAS_W = 800
CANVAS_H = 500
WAFER_DISPLAY = 340  # wafer panel size
BG_COLOR = (12, 15, 24)  # dark navy
GRID_COLOR_DIM = (40, 55, 80)
GRID_COLOR_ACTIVE = (0, 200, 255)
ACCENT_GREEN = (0, 255, 128)
ACCENT_CYAN = (0, 200, 255)
ACCENT_RED = (255, 60, 80)
ACCENT_YELLOW = (255, 220, 50)
ACCENT_ORANGE = (255, 140, 30)
TITLE_COLOR = (235, 240, 250)
SUBTITLE_COLOR = (140, 160, 190)
PHASE_BG = (25, 30, 45)

CLASS_COLORS = {
    "scratch": (255, 100, 100),
    "particle": (100, 200, 255),
    "edge_chip": (255, 255, 80),
    "void": (200, 100, 255),
    "pattern_shift": (255, 160, 50),
    "bridge": (100, 255, 180),
    "missing_bond": (255, 100, 200),
    "crack": (255, 80, 80),
    "contamination": (255, 200, 100),
    "delamination": (150, 220, 255),
}

FPS = 10
FRAME_MS = int(1000 / FPS)
GRID_CELLS = 20  # YOLOv8 uses ~20x20 grid at one scale


def _font(size: int):
    for fp in ["/System/Library/Fonts/Helvetica.ttc",
               "/System/Library/Fonts/SFNSMono.ttf",
               "/Library/Fonts/Arial.ttf",
               "/System/Library/Fonts/Supplemental/Arial.ttf"]:
        try:
            return ImageFont.truetype(fp, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def _get_detections(model_path: str, img_path: str) -> list:
    from ultralytics import YOLO
    CLASSES = ["scratch", "particle", "edge_chip", "void", "pattern_shift",
               "bridge", "missing_bond", "crack", "contamination", "delamination"]
    model = YOLO(model_path)
    results = model.predict(img_path, conf=0.15, iou=0.45, verbose=False)  # lower conf to get more candidates
    r = results[0]
    dets = []
    if r.boxes is not None:
        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = [int(v) for v in box.xyxy[0].tolist()]
            cls_name = CLASSES[cls_id] if cls_id < len(CLASSES) else f"cls_{cls_id}"
            dets.append({"class": cls_name, "conf": conf, "bbox": xyxy})
    dets.sort(key=lambda d: -d["conf"])
    return dets


def _generate_fake_candidates(real_dets: list, rng: np.random.Generator, n_extra: int = 12) -> list:
    """Generate plausible low-confidence candidate boxes that NMS would eliminate.
    These simulate what YOLO produces before NMS filtering."""
    CLASSES = ["scratch", "particle", "edge_chip", "void", "pattern_shift",
               "bridge", "missing_bond", "crack", "contamination", "delamination"]
    candidates = []
    # Duplicates of real detections (slightly shifted, lower confidence) - these are what NMS removes
    for det in real_dets:
        for _ in range(rng.integers(1, 4)):
            bbox = det["bbox"]
            shift_x = rng.integers(-25, 26)
            shift_y = rng.integers(-25, 26)
            scale = rng.uniform(0.8, 1.25)
            cx = (bbox[0] + bbox[2]) / 2 + shift_x
            cy = (bbox[1] + bbox[3]) / 2 + shift_y
            w = (bbox[2] - bbox[0]) * scale
            h = (bbox[3] - bbox[1]) * scale
            candidates.append({
                "class": det["class"],
                "conf": det["conf"] * rng.uniform(0.3, 0.85),
                "bbox": [int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)],
                "is_suppressed": True,
            })
    # Random false positives (very low confidence)
    for _ in range(n_extra):
        cx = rng.integers(80, 560)
        cy = rng.integers(80, 560)
        w = rng.integers(20, 80)
        h = rng.integers(20, 80)
        candidates.append({
            "class": rng.choice(CLASSES),
            "conf": rng.uniform(0.05, 0.22),
            "bbox": [cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2],
            "is_suppressed": True,
        })
    return candidates


def _draw_title(frame: Image.Image, phase_name: str, wafer_idx: int, total: int):
    draw = ImageDraw.Draw(frame)
    f_title = _font(22)
    f_sub = _font(13)
    f_badge = _font(12)

    draw.text((24, 12), "YOLOv8-L  Wafer Defect Detection", fill=TITLE_COLOR, font=f_title)
    draw.text((24, 40), "44M params | mAP@50 99.22% | 10 defect classes",
              fill=SUBTITLE_COLOR, font=f_sub)

    # Phase badge
    bx = CANVAS_W - 260
    draw.rounded_rectangle([bx, 12, bx + 240, 38], radius=10,
                           fill=(30, 40, 60), outline=ACCENT_CYAN, width=1)
    draw.text((bx + 12, 16), phase_name, fill=ACCENT_CYAN, font=f_badge)

    # Progress dots
    for i in range(total):
        x = CANVAS_W - 240 + i * 18
        c = ACCENT_GREEN if i < wafer_idx else (ACCENT_CYAN if i == wafer_idx else (50, 60, 80))
        draw.ellipse([x, 44, x + 10, 54], fill=c)

    draw.line([(16, 62), (CANVAS_W - 16, 62)], fill=(35, 45, 65), width=1)


def _draw_phase_label(draw: ImageDraw.Draw, x: int, y: int, phase: str, desc: str):
    f_phase = _font(16)
    f_desc = _font(11)
    draw.text((x, y), phase, fill=ACCENT_CYAN, font=f_phase)
    draw.text((x, y + 22), desc, fill=SUBTITLE_COLOR, font=f_desc)


def _draw_grid_overlay(draw: ImageDraw.Draw, px: int, py: int, size: int,
                       n_cells: int, progress: float, rng: np.random.Generator):
    """Draw the YOLO feature grid. progress: 0->1 controls how many cells are visible."""
    cell_size = size / n_cells
    total_cells = n_cells * n_cells
    cells_to_show = int(total_cells * progress)

    for i in range(cells_to_show):
        row = i // n_cells
        col = i % n_cells
        x1 = px + col * cell_size
        y1 = py + row * cell_size
        x2 = x1 + cell_size
        y2 = y1 + cell_size

        # Varying activation intensity
        intensity = rng.uniform(0.2, 1.0)
        r = int(GRID_COLOR_ACTIVE[0] * intensity * 0.3)
        g = int(GRID_COLOR_ACTIVE[1] * intensity * 0.3)
        b = int(GRID_COLOR_ACTIVE[2] * intensity * 0.3)
        draw.rectangle([x1, y1, x2, y2], outline=(r, g, b), width=1)

    # Full grid lines (dim)
    for i in range(n_cells + 1):
        pos = py + i * cell_size
        draw.line([(px, pos), (px + size, pos)], fill=GRID_COLOR_DIM, width=1)
        pos_x = px + i * cell_size
        draw.line([(pos_x, py), (pos_x, py + size)], fill=GRID_COLOR_DIM, width=1)


def _draw_grid_activations(draw: ImageDraw.Draw, px: int, py: int, size: int,
                           n_cells: int, detections: list, scale: float, progress: float):
    """Highlight grid cells that contain object centers - these cells 'fire'."""
    cell_size = size / n_cells

    # Find which cells should activate (contain detection centers)
    active_cells = set()
    for det in detections:
        bbox = det["bbox"]
        cx = (bbox[0] + bbox[2]) / 2 * scale
        cy = (bbox[1] + bbox[3]) / 2 * scale
        col = int(cx / cell_size)
        row = int(cy / cell_size)
        col = min(max(col, 0), n_cells - 1)
        row = min(max(row, 0), n_cells - 1)
        active_cells.add((row, col))
        # Also activate neighbors
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = row + dr, col + dc
                if 0 <= nr < n_cells and 0 <= nc < n_cells:
                    active_cells.add((nr, nc))

    n_to_show = int(len(active_cells) * progress)
    active_list = sorted(active_cells)[:n_to_show]

    for row, col in active_list:
        x1 = px + col * cell_size
        y1 = py + row * cell_size
        x2 = x1 + cell_size
        y2 = y1 + cell_size
        # Bright activation
        draw.rectangle([x1 + 1, y1 + 1, x2 - 1, y2 - 1], fill=(0, 50, 40), outline=ACCENT_GREEN, width=1)


def _draw_candidate_box(draw: ImageDraw.Draw, det: dict, px: int, py: int,
                        scale: float, opacity: float = 1.0, dashed: bool = False):
    """Draw a candidate bounding box (thin, possibly dashed for suppressed ones)."""
    bbox = det["bbox"]
    x1 = int(px + bbox[0] * scale)
    y1 = int(py + bbox[1] * scale)
    x2 = int(px + bbox[2] * scale)
    y2 = int(py + bbox[3] * scale)
    color = CLASS_COLORS.get(det["class"], (200, 200, 200))

    if det.get("is_suppressed"):
        # Dim color for suppressed boxes
        color = tuple(int(c * 0.4) for c in color)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=1)
    else:
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)


def _draw_final_box(draw: ImageDraw.Draw, det: dict, px: int, py: int,
                    scale: float, progress: float):
    """Draw final detection box with label. progress: 0->1 for animation."""
    if progress <= 0:
        return
    bbox = det["bbox"]
    color = CLASS_COLORS.get(det["class"], (255, 255, 255))

    x1 = int(px + bbox[0] * scale)
    y1 = int(py + bbox[1] * scale)
    x2 = int(px + bbox[2] * scale)
    y2 = int(py + bbox[3] * scale)

    # Pop-in from center
    if progress < 1.0:
        cx_b = (x1 + x2) / 2
        cy_b = (y1 + y2) / 2
        ease = progress ** 0.5
        hw = (x2 - x1) / 2 * ease
        hh = (y2 - y1) / 2 * ease
        x1, y1 = int(cx_b - hw), int(cy_b - hh)
        x2, y2 = int(cx_b + hw), int(cy_b + hh)

    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

    # Corner accents
    cl = min(12, (x2 - x1) // 4, (y2 - y1) // 4)
    if progress >= 0.6 and cl > 2:
        for cx, cy, dx, dy in [(x1, y1, 1, 1), (x2, y1, -1, 1),
                                (x1, y2, 1, -1), (x2, y2, -1, -1)]:
            draw.line([(cx, cy), (cx + cl * dx, cy)], fill=color, width=3)
            draw.line([(cx, cy), (cx, cy + cl * dy)], fill=color, width=3)

    # Label
    if progress >= 0.75:
        f = _font(12)
        label = f"{det['class']} {det['conf']:.0%}"
        tb = draw.textbbox((0, 0), label, font=f)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ly = y1 - th - 5
        if ly < py:
            ly = y2 + 2
        draw.rectangle([x1, ly - 1, x1 + tw + 6, ly + th + 3], fill=color)
        draw.text((x1 + 3, ly), label, fill=(0, 0, 0), font=f)


def _draw_nms_cross(draw: ImageDraw.Draw, det: dict, px: int, py: int,
                    scale: float, progress: float):
    """Draw a red X through a suppressed box during NMS phase."""
    if progress <= 0:
        return
    bbox = det["bbox"]
    x1 = int(px + bbox[0] * scale)
    y1 = int(py + bbox[1] * scale)
    x2 = int(px + bbox[2] * scale)
    y2 = int(py + bbox[3] * scale)

    # Dim box
    color = tuple(int(c * 0.3) for c in CLASS_COLORS.get(det["class"], (200, 200, 200)))
    draw.rectangle([x1, y1, x2, y2], outline=color, width=1)

    # Red X
    if progress > 0.3:
        cross_alpha = min(1.0, (progress - 0.3) / 0.4)
        r = int(255 * cross_alpha)
        draw.line([(x1, y1), (x2, y2)], fill=(r, 30, 30), width=2)
        draw.line([(x2, y1), (x1, y2)], fill=(r, 30, 30), width=2)


def _draw_stats(draw: ImageDraw.Draw, dets: list, phase: str):
    f = _font(12)
    y = CANVAS_H - 32
    x = 24
    n = len([d for d in dets if not d.get("is_suppressed")])
    classes = list({d["class"] for d in dets if not d.get("is_suppressed")})
    avg = sum(d["conf"] for d in dets if not d.get("is_suppressed")) / max(n, 1)

    for text, color in [
        (f"Detections: {n}", ACCENT_GREEN),
        (f"Classes: {', '.join(classes)}", ACCENT_CYAN),
        (f"Avg conf: {avg:.0%}", ACCENT_YELLOW),
    ]:
        draw.text((x, y), text, fill=color, font=f)
        tb = draw.textbbox((0, 0), text, font=f)
        x += (tb[2] - tb[0]) + 35


def generate_gif():
    model_path = str(ROOT / "models" / "best.pt")
    rng = np.random.default_rng(42)

    images_to_use = [
        "realistic_05.jpg",  # 4 detections (blue wafer)
        "realistic_01.jpg",  # 2 detections (bronze wafer)
        "realistic_04.jpg",  # 4 detections (gold wafer)
    ]

    print("Loading model and running inference...")
    wafer_data = []
    for fname in images_to_use:
        img_path = str(REALISTIC_DIR / fname)
        orig = Image.open(img_path).convert("RGB")
        dets = _get_detections(model_path, img_path)
        real_dets = [d for d in dets if d["conf"] >= 0.25]
        for d in real_dets:
            d["is_suppressed"] = False
        candidates = _generate_fake_candidates(real_dets, rng)
        wafer_data.append({"name": fname, "image": orig, "real": real_dets, "candidates": candidates})
        print(f"  {fname}: {len(real_dets)} real + {len(candidates)} candidate boxes")

    # Panel layout
    wafer_x = 30
    wafer_y = 75
    detect_x = 420
    detect_y = 75

    all_frames: list[Image.Image] = []
    total_wafers = len(wafer_data)

    for w_idx, wdata in enumerate(wafer_data):
        orig = wdata["image"]
        real_dets = wdata["real"]
        candidates = wdata["candidates"]
        all_dets = real_dets + candidates

        # Scale to panel
        scale = WAFER_DISPLAY / max(orig.size)
        dw = int(orig.size[0] * scale)
        dh = int(orig.size[1] * scale)
        orig_small = orig.resize((dw, dh), Image.LANCZOS)

        # ── PHASE 1: INPUT (5 frames) ──
        for f in range(5):
            frame = Image.new("RGB", (CANVAS_W, CANVAS_H), BG_COLOR)
            draw = ImageDraw.Draw(frame)
            _draw_title(frame, "PHASE 1: CNN Forward Pass", w_idx, total_wafers)

            _draw_phase_label(draw, detect_x, wafer_y - 5,
                              "1. Single Forward Pass",
                              "Entire image → CNN backbone → feature maps")

            # Original wafer
            frame.paste(orig_small, (wafer_x, wafer_y))
            draw.rectangle([wafer_x - 1, wafer_y - 1, wafer_x + dw + 1, wafer_y + dh + 1],
                           outline=(60, 70, 90), width=2)

            # Arrow from original to detection area
            arr_y = wafer_y + dh // 2
            arrow_progress = f / 4
            ax = int(wafer_x + dw + 10 + (detect_x - wafer_x - dw - 30) * arrow_progress)
            draw.line([(wafer_x + dw + 8, arr_y), (ax, arr_y)], fill=ACCENT_CYAN, width=2)
            if arrow_progress > 0.3:
                draw.polygon([(ax, arr_y - 5), (ax + 8, arr_y), (ax, arr_y + 5)], fill=ACCENT_CYAN)

            # "Input 640×640" label
            f_sm = _font(11)
            draw.text((wafer_x, wafer_y + dh + 5), "Input: 640×640", fill=SUBTITLE_COLOR, font=f_sm)

            all_frames.append(frame)

        # ── PHASE 2: GRID OVERLAY (10 frames) ──
        for f in range(10):
            frame = Image.new("RGB", (CANVAS_W, CANVAS_H), BG_COLOR)
            draw = ImageDraw.Draw(frame)
            progress = f / 9
            _draw_title(frame, "PHASE 2: Feature Grid (20×20)", w_idx, total_wafers)

            _draw_phase_label(draw, detect_x, wafer_y - 5,
                              "2. Grid Division",
                              f"Image divided into {GRID_CELLS}×{GRID_CELLS} = {GRID_CELLS**2} cells")

            frame.paste(orig_small, (wafer_x, wafer_y))
            draw.rectangle([wafer_x - 1, wafer_y - 1, wafer_x + dw + 1, wafer_y + dh + 1],
                           outline=(60, 70, 90), width=2)

            # Detection panel with grid appearing
            frame.paste(orig_small, (detect_x, detect_y))
            _draw_grid_overlay(draw, detect_x, detect_y, dw, GRID_CELLS, progress, rng)
            draw.rectangle([detect_x - 1, detect_y - 1, detect_x + dw + 1, detect_y + dh + 1],
                           outline=ACCENT_CYAN, width=2)

            f_sm = _font(11)
            draw.text((wafer_x, wafer_y + dh + 5), "Input: 640×640", fill=SUBTITLE_COLOR, font=f_sm)
            draw.text((detect_x, detect_y + dh + 5),
                      f"Grid: {GRID_CELLS}×{GRID_CELLS} = {GRID_CELLS**2} cells",
                      fill=ACCENT_CYAN, font=f_sm)

            all_frames.append(frame)

        # ── PHASE 3: GRID ACTIVATION (8 frames) ──
        for f in range(8):
            frame = Image.new("RGB", (CANVAS_W, CANVAS_H), BG_COLOR)
            draw = ImageDraw.Draw(frame)
            progress = f / 7
            _draw_title(frame, "PHASE 3: Cell Activation", w_idx, total_wafers)

            _draw_phase_label(draw, detect_x, wafer_y - 5,
                              "3. Each Cell Predicts Simultaneously",
                              "Cells containing objects activate (green)")

            frame.paste(orig_small, (wafer_x, wafer_y))
            draw.rectangle([wafer_x - 1, wafer_y - 1, wafer_x + dw + 1, wafer_y + dh + 1],
                           outline=(60, 70, 90), width=2)

            # Detection panel with grid + activations
            frame.paste(orig_small, (detect_x, detect_y))
            _draw_grid_overlay(draw, detect_x, detect_y, dw, GRID_CELLS, 1.0, rng)
            _draw_grid_activations(draw, detect_x, detect_y, dw, GRID_CELLS,
                                   real_dets, scale, progress)
            draw.rectangle([detect_x - 1, detect_y - 1, detect_x + dw + 1, detect_y + dh + 1],
                           outline=ACCENT_CYAN, width=2)

            f_sm = _font(11)
            draw.text((detect_x, detect_y + dh + 5),
                      f"Active cells: {int(len(real_dets) * 3 * progress)}/{GRID_CELLS**2}",
                      fill=ACCENT_GREEN, font=f_sm)

            all_frames.append(frame)

        # ── PHASE 4: CANDIDATE BOXES (8 frames) ──
        for f in range(8):
            frame = Image.new("RGB", (CANVAS_W, CANVAS_H), BG_COLOR)
            draw = ImageDraw.Draw(frame)
            progress = f / 7
            total_candidates = len(all_dets)
            n_show = int(total_candidates * progress)
            _draw_title(frame, f"PHASE 4: Candidate Boxes ({n_show}/{total_candidates})", w_idx, total_wafers)

            _draw_phase_label(draw, detect_x, wafer_y - 5,
                              "4. Raw Predictions",
                              f"{total_candidates} candidate boxes (many overlapping)")

            frame.paste(orig_small, (wafer_x, wafer_y))
            draw.rectangle([wafer_x - 1, wafer_y - 1, wafer_x + dw + 1, wafer_y + dh + 1],
                           outline=(60, 70, 90), width=2)

            # Detection panel with candidate boxes appearing
            frame.paste(orig_small, (detect_x, detect_y))
            shuffled = all_dets[:n_show]
            for det in shuffled:
                _draw_candidate_box(draw, det, detect_x, detect_y, scale)
            draw.rectangle([detect_x - 1, detect_y - 1, detect_x + dw + 1, detect_y + dh + 1],
                           outline=ACCENT_ORANGE, width=2)

            f_sm = _font(11)
            draw.text((detect_x, detect_y + dh + 5),
                      f"Candidates: {n_show} (conf 0.05-0.95)",
                      fill=ACCENT_ORANGE, font=f_sm)

            all_frames.append(frame)

        # ── PHASE 5: NMS SUPPRESSION (10 frames) ──
        suppressed = [d for d in all_dets if d.get("is_suppressed")]
        n_suppressed = len(suppressed)
        for f in range(10):
            frame = Image.new("RGB", (CANVAS_W, CANVAS_H), BG_COLOR)
            draw = ImageDraw.Draw(frame)
            progress = f / 9
            n_removed = int(n_suppressed * progress)
            _draw_title(frame, f"PHASE 5: Non-Max Suppression", w_idx, total_wafers)

            _draw_phase_label(draw, detect_x, wafer_y - 5,
                              "5. NMS: Remove Overlapping Boxes",
                              f"Suppressed {n_removed}/{n_suppressed} | IoU threshold: 0.45")

            frame.paste(orig_small, (wafer_x, wafer_y))
            draw.rectangle([wafer_x - 1, wafer_y - 1, wafer_x + dw + 1, wafer_y + dh + 1],
                           outline=(60, 70, 90), width=2)

            # Show remaining real boxes + suppressed ones getting crossed out
            frame.paste(orig_small, (detect_x, detect_y))

            # Real detections stay
            for det in real_dets:
                _draw_candidate_box(draw, det, detect_x, detect_y, scale)

            # Suppressed ones get red X
            for i, det in enumerate(suppressed):
                if i < n_removed:
                    _draw_nms_cross(draw, det, detect_x, detect_y, scale, 1.0)
                else:
                    _draw_candidate_box(draw, det, detect_x, detect_y, scale)

            draw.rectangle([detect_x - 1, detect_y - 1, detect_x + dw + 1, detect_y + dh + 1],
                           outline=ACCENT_RED, width=2)

            f_sm = _font(11)
            remaining = len(all_dets) - n_removed
            draw.text((detect_x, detect_y + dh + 5),
                      f"Remaining: {remaining} → {len(real_dets)} final",
                      fill=ACCENT_RED, font=f_sm)

            all_frames.append(frame)

        # ── PHASE 6: FINAL RESULT (per-box pop + hold) ──
        for det_idx in range(len(real_dets)):
            for f in range(5):
                frame = Image.new("RGB", (CANVAS_W, CANVAS_H), BG_COLOR)
                draw = ImageDraw.Draw(frame)
                box_progress = f / 4
                _draw_title(frame, f"PHASE 6: Final Detection ({det_idx + 1}/{len(real_dets)})",
                            w_idx, total_wafers)

                _draw_phase_label(draw, detect_x, wafer_y - 5,
                                  "6. Final Detections",
                                  f"conf ≥ 0.25 | {len(real_dets)} defects confirmed")

                frame.paste(orig_small, (wafer_x, wafer_y))
                draw.rectangle([wafer_x - 1, wafer_y - 1, wafer_x + dw + 1, wafer_y + dh + 1],
                               outline=(60, 70, 90), width=2)

                frame.paste(orig_small, (detect_x, detect_y))

                # Previously completed boxes
                for prev in range(det_idx):
                    _draw_final_box(draw, real_dets[prev], detect_x, detect_y, scale, 1.0)
                # Current box animating
                _draw_final_box(draw, real_dets[det_idx], detect_x, detect_y, scale, box_progress)

                draw.rectangle([detect_x - 1, detect_y - 1, detect_x + dw + 1, detect_y + dh + 1],
                               outline=ACCENT_GREEN, width=2)

                _draw_stats(draw, real_dets[:det_idx + 1] if box_progress > 0.5 else real_dets[:det_idx],
                            "final")
                all_frames.append(frame)

        # HOLD on final (8 frames)
        for f in range(8):
            frame = Image.new("RGB", (CANVAS_W, CANVAS_H), BG_COLOR)
            draw = ImageDraw.Draw(frame)
            _draw_title(frame, f"COMPLETE: {len(real_dets)} defects detected", w_idx, total_wafers)
            _draw_phase_label(draw, detect_x, wafer_y - 5,
                              "Detection Complete",
                              "You Only Look Once — single pass, real-time inference")

            frame.paste(orig_small, (wafer_x, wafer_y))
            draw.rectangle([wafer_x - 1, wafer_y - 1, wafer_x + dw + 1, wafer_y + dh + 1],
                           outline=(60, 70, 90), width=2)

            frame.paste(orig_small, (detect_x, detect_y))
            for det in real_dets:
                _draw_final_box(draw, det, detect_x, detect_y, scale, 1.0)
            draw.rectangle([detect_x - 1, detect_y - 1, detect_x + dw + 1, detect_y + dh + 1],
                           outline=ACCENT_GREEN, width=2)
            _draw_stats(draw, real_dets, "final")
            all_frames.append(frame)

        # TRANSITION (4 frames)
        if w_idx < total_wafers - 1:
            last = all_frames[-1].copy()
            dark = Image.new("RGB", (CANVAS_W, CANVAS_H), BG_COLOR)
            for f in range(4):
                all_frames.append(Image.blend(last, dark, f / 4))

    # ── Save GIF ──
    print(f"\nEncoding GIF: {len(all_frames)} frames...")
    frames_p = []
    for fr in all_frames:
        frames_p.append(fr.quantize(colors=128, method=Image.Quantize.MEDIANCUT,
                                     dither=Image.Dither.FLOYDSTEINBERG))

    frames_p[0].save(
        str(OUTPUT_GIF), save_all=True, append_images=frames_p[1:],
        duration=FRAME_MS, loop=0, optimize=True,
    )

    sz = OUTPUT_GIF.stat().st_size / 1e6
    dur = len(all_frames) / FPS
    print(f"\nSaved: {OUTPUT_GIF}")
    print(f"  Size    : {sz:.1f} MB")
    print(f"  Frames  : {len(all_frames)}")
    print(f"  Duration: {dur:.1f}s @ {FPS} FPS")


if __name__ == "__main__":
    generate_gif()
