"""
Generate an eye-catching LinkedIn showcase GIF for YOLOv8 wafer defect detection.

Layout per frame:
  ┌─────────────────────┬─────────────────────┐
  │                     │                     │
  │   Original Wafer    │   YOLO Detection    │
  │   (clean)           │   (progressive)     │
  │                     │                     │
  └─────────────────────┴─────────────────────┘

Animation phases:
  1. Scanning line sweeps across the wafer (green laser effect)
  2. Bounding boxes appear one by one with pop animation
  3. Labels fade in with confidence scores
  4. Brief hold on final result
  5. Transition to next wafer image

Uses the best-looking realistic wafer images.
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
ANNOTATED_DIR = REALISTIC_DIR / "annotated"
OUTPUT_GIF = ROOT / "outputs" / "yolo_wafer_detection_showcase.gif"

CANVAS_W = 800  # optimized for LinkedIn
CANVAS_H = 480
WAFER_SIZE = 340  # display size for each wafer panel
PANEL_Y_OFFSET = 100  # top area for title
BG_COLOR = (15, 18, 28)  # dark navy
ACCENT_GREEN = (0, 255, 128)
ACCENT_CYAN = (0, 200, 255)
ACCENT_RED = (255, 60, 80)
ACCENT_YELLOW = (255, 220, 50)
ACCENT_ORANGE = (255, 140, 30)
TITLE_COLOR = (240, 245, 255)
SUBTITLE_COLOR = (160, 175, 200)

# Color per defect class
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
FRAME_DURATION_MS = int(1000 / FPS)

# Frames per phase
FRAMES_SCAN = 12       # scanning animation
FRAMES_PER_BOX = 5     # per bounding box pop-in
FRAMES_HOLD = 10       # hold on final result
FRAMES_TRANSITION = 4  # fade to next


def _load_font(size: int):
    """Try to load a nice font, fallback to default."""
    font_paths = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSMono.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for fp in font_paths:
        try:
            return ImageFont.truetype(fp, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def _get_detections(model_path: str, img_path: str) -> list:
    """Run YOLO inference and return detections with boxes."""
    from ultralytics import YOLO
    model = YOLO(model_path)
    results = model.predict(img_path, conf=0.25, iou=0.45, verbose=False)
    r = results[0]
    dets = []
    CLASSES = ["scratch", "particle", "edge_chip", "void", "pattern_shift",
               "bridge", "missing_bond", "crack", "contamination", "delamination"]
    if r.boxes is not None:
        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = [int(v) for v in box.xyxy[0].tolist()]
            cls_name = CLASSES[cls_id] if cls_id < len(CLASSES) else f"cls_{cls_id}"
            dets.append({"class": cls_name, "conf": conf, "bbox": xyxy})
    # Sort by confidence descending
    dets.sort(key=lambda d: -d["conf"])
    return dets


def _draw_title_bar(frame: Image.Image, wafer_idx: int, total: int,
                    phase_text: str):
    """Draw the top title bar."""
    draw = ImageDraw.Draw(frame)
    font_title = _load_font(26)
    font_sub = _load_font(16)
    font_badge = _load_font(14)

    # Title
    draw.text((30, 18), "YOLOv8-L Wafer Defect Detection", fill=TITLE_COLOR, font=font_title)

    # Subtitle
    draw.text((30, 52), "mAP@50: 99.22% | A100 GPU | 10 Defect Classes",
              fill=SUBTITLE_COLOR, font=font_sub)

    # Phase badge
    badge_x = CANVAS_W - 280
    draw.rounded_rectangle([badge_x, 18, badge_x + 250, 48], radius=12,
                           fill=(40, 50, 70), outline=ACCENT_CYAN, width=1)
    draw.text((badge_x + 16, 22), phase_text, fill=ACCENT_CYAN, font=font_badge)

    # Progress dots
    dot_y = 60
    dot_x_start = CANVAS_W - 260
    for i in range(total):
        x = dot_x_start + i * 22
        color = ACCENT_GREEN if i < wafer_idx else (60, 70, 90)
        if i == wafer_idx:
            color = ACCENT_CYAN
        draw.ellipse([x, dot_y, x + 12, dot_y + 12], fill=color)

    # Divider line
    draw.line([(20, 82), (CANVAS_W - 20, 82)], fill=(40, 50, 70), width=1)


def _draw_panel_label(draw: ImageDraw.Draw, x: int, text: str, color: tuple):
    """Label above each panel."""
    font = _load_font(15)
    draw.text((x, PANEL_Y_OFFSET - 22), text, fill=color, font=font)


def _draw_scan_line(draw: ImageDraw.Draw, panel_x: int, panel_y: int,
                    progress: float, wafer_size: int):
    """Draw a green scanning line across the detection panel."""
    line_y = int(panel_y + progress * wafer_size)
    line_y = min(line_y, panel_y + wafer_size)
    # Glow effect (multiple lines with decreasing opacity)
    for offset, alpha_frac in [(-3, 0.15), (-1, 0.4), (0, 1.0), (1, 0.4), (3, 0.15)]:
        y = line_y + offset
        if panel_y <= y <= panel_y + wafer_size:
            g = int(255 * alpha_frac)
            color = (0, g, int(128 * alpha_frac))
            draw.line([(panel_x, y), (panel_x + wafer_size, y)], fill=color, width=1)


def _draw_bbox_animated(draw: ImageDraw.Draw, det: dict, panel_x: int, panel_y: int,
                        scale: float, progress: float, wafer_size: int):
    """Draw a bounding box with pop-in animation. progress: 0->1."""
    if progress <= 0:
        return

    bbox = det["bbox"]
    cls_name = det["class"]
    conf = det["conf"]
    color = CLASS_COLORS.get(cls_name, (255, 255, 255))

    # Scale bbox to panel coordinates
    x1 = int(panel_x + bbox[0] * scale)
    y1 = int(panel_y + bbox[1] * scale)
    x2 = int(panel_x + bbox[2] * scale)
    y2 = int(panel_y + bbox[3] * scale)

    # Pop animation: start from center, expand to full size
    if progress < 1.0:
        cx_box = (x1 + x2) / 2
        cy_box = (y1 + y2) / 2
        ease = progress ** 0.5  # ease-out
        half_w = (x2 - x1) / 2 * ease
        half_h = (y2 - y1) / 2 * ease
        x1 = int(cx_box - half_w)
        y1 = int(cy_box - half_h)
        x2 = int(cx_box + half_w)
        y2 = int(cy_box + half_h)

    # Draw box
    line_width = 3 if progress >= 0.5 else 2
    draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

    # Corner accents (thick L-shapes at corners)
    corner_len = min(15, (x2 - x1) // 3, (y2 - y1) // 3)
    if progress >= 0.7 and corner_len > 3:
        for cx, cy, dx, dy in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
            draw.line([(cx, cy), (cx + corner_len * dx, cy)], fill=color, width=3)
            draw.line([(cx, cy), (cx, cy + corner_len * dy)], fill=color, width=3)

    # Label with background
    if progress >= 0.8:
        font = _load_font(13)
        label = f"{cls_name} {conf:.0%}"
        bbox_text = draw.textbbox((0, 0), label, font=font)
        tw = bbox_text[2] - bbox_text[0]
        th = bbox_text[3] - bbox_text[1]

        lx = x1
        ly = y1 - th - 6
        if ly < panel_y:
            ly = y2 + 3

        # Label background
        draw.rectangle([lx - 2, ly - 1, lx + tw + 6, ly + th + 3],
                       fill=color)
        # Label text (black on colored background)
        draw.text((lx + 2, ly), label, fill=(0, 0, 0), font=font)


def _draw_stats_overlay(draw: ImageDraw.Draw, detections: list, progress: float):
    """Draw detection stats at bottom."""
    if progress < 0.5:
        return
    font = _load_font(13)
    y_base = CANVAS_H - 45
    x = 30

    n_det = len(detections)
    classes = list({d["class"] for d in detections})
    avg_conf = sum(d["conf"] for d in detections) / max(n_det, 1)

    stats = [
        (f"Detections: {n_det}", ACCENT_GREEN),
        (f"Classes: {', '.join(classes)}", ACCENT_CYAN),
        (f"Avg Confidence: {avg_conf:.0%}", ACCENT_YELLOW),
    ]

    for text, color in stats:
        draw.text((x, y_base), text, fill=color, font=font)
        bbox_text = draw.textbbox((0, 0), text, font=font)
        x += (bbox_text[2] - bbox_text[0]) + 40


def generate_showcase_gif():
    """Generate the full animated GIF."""
    model_path = str(ROOT / "models" / "best.pt")

    # Select best images (most defects, varied)
    images_to_use = [
        "realistic_05.jpg",  # scratch + crack + edge chip (blue wafer, 4 det)
        "realistic_01.jpg",  # crack + edge chip (bronze wafer, 2 det)
        "realistic_04.jpg",  # crack + contamination + particle (gold, 4 det)
    ]

    # Pre-load images and run detection
    print("Loading model and running inference...")
    wafer_data = []
    for fname in images_to_use:
        img_path = str(REALISTIC_DIR / fname)
        orig_img = Image.open(img_path).convert("RGB")
        dets = _get_detections(model_path, img_path)
        wafer_data.append({"name": fname, "image": orig_img, "detections": dets})
        print(f"  {fname}: {len(dets)} detections")

    # Generate all frames
    all_frames: list[Image.Image] = []
    total_wafers = len(wafer_data)

    for w_idx, wdata in enumerate(wafer_data):
        orig = wdata["image"]
        dets = wdata["detections"]
        n_dets = len(dets)

        # Scale factor to fit panel
        orig_w, orig_h = orig.size
        scale = WAFER_SIZE / max(orig_w, orig_h)
        display_w = int(orig_w * scale)
        display_h = int(orig_h * scale)

        orig_resized = orig.resize((display_w, display_h), Image.LANCZOS)

        # Panel positions
        left_x = 60
        right_x = CANVAS_W // 2 + 40
        panel_y = PANEL_Y_OFFSET

        # Center vertically
        y_offset = (CANVAS_H - PANEL_Y_OFFSET - 55 - display_h) // 2
        panel_y += max(y_offset, 0)

        # Phase 1: Scanning animation
        for f in range(FRAMES_SCAN):
            frame = Image.new("RGB", (CANVAS_W, CANVAS_H), BG_COLOR)
            draw = ImageDraw.Draw(frame)
            progress = f / FRAMES_SCAN

            _draw_title_bar(frame, w_idx, total_wafers,
                            f"SCANNING wafer {w_idx + 1}/{total_wafers}")
            _draw_panel_label(draw, left_x, "ORIGINAL WAFER", SUBTITLE_COLOR)
            _draw_panel_label(draw, right_x, "YOLOv8 DETECTION", ACCENT_GREEN)

            # Paste original on left
            frame.paste(orig_resized, (left_x, panel_y))

            # Paste original on right (detection panel) with scan overlay
            frame.paste(orig_resized, (right_x, panel_y))
            _draw_scan_line(draw, right_x, panel_y, progress, display_h)

            # Dim area above scan line slightly (already scanned effect)
            scan_y = int(panel_y + progress * display_h)
            for sy in range(panel_y, min(scan_y, panel_y + display_h)):
                for sx in range(right_x, right_x + display_w):
                    if 0 <= sx < CANVAS_W and 0 <= sy < CANVAS_H:
                        pass  # Skip pixel manipulation for speed

            # Panel borders
            draw.rectangle([left_x - 2, panel_y - 2, left_x + display_w + 2, panel_y + display_h + 2],
                           outline=(50, 60, 80), width=2)
            draw.rectangle([right_x - 2, panel_y - 2, right_x + display_w + 2, panel_y + display_h + 2],
                           outline=ACCENT_GREEN, width=2)

            all_frames.append(frame)

        # Phase 2: Bounding boxes pop in one by one
        for det_idx in range(n_dets):
            for f in range(FRAMES_PER_BOX):
                frame = Image.new("RGB", (CANVAS_W, CANVAS_H), BG_COLOR)
                draw = ImageDraw.Draw(frame)
                box_progress = f / FRAMES_PER_BOX

                _draw_title_bar(frame, w_idx, total_wafers,
                                f"DETECTING defect {det_idx + 1}/{n_dets}")
                _draw_panel_label(draw, left_x, "ORIGINAL WAFER", SUBTITLE_COLOR)
                _draw_panel_label(draw, right_x, "YOLOv8 DETECTION", ACCENT_GREEN)

                frame.paste(orig_resized, (left_x, panel_y))
                frame.paste(orig_resized, (right_x, panel_y))

                # Draw all previous boxes (fully)
                for prev_idx in range(det_idx):
                    _draw_bbox_animated(draw, dets[prev_idx], right_x, panel_y,
                                        scale, 1.0, display_h)

                # Draw current box animating
                _draw_bbox_animated(draw, dets[det_idx], right_x, panel_y,
                                    scale, box_progress, display_h)

                # Draw also on the original side (faint highlight)
                if box_progress > 0.5:
                    det = dets[det_idx]
                    bbox = det["bbox"]
                    ox1 = int(left_x + bbox[0] * scale)
                    oy1 = int(panel_y + bbox[1] * scale)
                    ox2 = int(left_x + bbox[2] * scale)
                    oy2 = int(panel_y + bbox[3] * scale)
                    draw.rectangle([ox1, oy1, ox2, oy2],
                                   outline=(255, 255, 255, 100), width=1)

                # Panel borders
                draw.rectangle([left_x - 2, panel_y - 2, left_x + display_w + 2, panel_y + display_h + 2],
                               outline=(50, 60, 80), width=2)
                draw.rectangle([right_x - 2, panel_y - 2, right_x + display_w + 2, panel_y + display_h + 2],
                               outline=ACCENT_GREEN, width=2)

                # Stats
                detected_so_far = dets[:det_idx + 1] if box_progress > 0.5 else dets[:det_idx]
                if detected_so_far:
                    _draw_stats_overlay(draw, detected_so_far, 1.0)

                all_frames.append(frame)

        # Phase 3: Hold on final result
        for f in range(FRAMES_HOLD):
            frame = Image.new("RGB", (CANVAS_W, CANVAS_H), BG_COLOR)
            draw = ImageDraw.Draw(frame)

            _draw_title_bar(frame, w_idx, total_wafers,
                            f"COMPLETE  {n_dets} defects found")
            _draw_panel_label(draw, left_x, "ORIGINAL WAFER", SUBTITLE_COLOR)
            _draw_panel_label(draw, right_x, "YOLOv8 DETECTION", ACCENT_GREEN)

            frame.paste(orig_resized, (left_x, panel_y))
            frame.paste(orig_resized, (right_x, panel_y))

            for det in dets:
                _draw_bbox_animated(draw, det, right_x, panel_y, scale, 1.0, display_h)

            # Panel borders
            draw.rectangle([left_x - 2, panel_y - 2, left_x + display_w + 2, panel_y + display_h + 2],
                           outline=(50, 60, 80), width=2)
            draw.rectangle([right_x - 2, panel_y - 2, right_x + display_w + 2, panel_y + display_h + 2],
                           outline=ACCENT_GREEN, width=2)

            _draw_stats_overlay(draw, dets, 1.0)
            all_frames.append(frame)

        # Phase 4: Transition (fade to dark)
        if w_idx < total_wafers - 1:
            last_frame = all_frames[-1].copy()
            for f in range(FRAMES_TRANSITION):
                fade = f / FRAMES_TRANSITION
                dark = Image.new("RGB", (CANVAS_W, CANVAS_H), BG_COLOR)
                blended = Image.blend(last_frame, dark, fade)
                all_frames.append(blended)

    # Save GIF
    print(f"\nSaving GIF: {len(all_frames)} frames at {FPS} FPS...")
    OUTPUT_GIF.parent.mkdir(parents=True, exist_ok=True)

    # Convert to P mode with dithering for better GIF quality
    frames_p = []
    for fr in all_frames:
        # Resize down further if needed for file size
        fr_small = fr.resize((CANVAS_W, CANVAS_H), Image.LANCZOS)
        frames_p.append(fr_small.quantize(colors=128, method=Image.Quantize.MEDIANCUT,
                                     dither=Image.Dither.FLOYDSTEINBERG))

    frames_p[0].save(
        str(OUTPUT_GIF),
        save_all=True,
        append_images=frames_p[1:],
        duration=FRAME_DURATION_MS,
        loop=0,
        optimize=True,
    )

    gif_size_mb = OUTPUT_GIF.stat().st_size / 1e6
    duration_sec = len(all_frames) / FPS
    print(f"\nGIF saved: {OUTPUT_GIF}")
    print(f"  Size    : {gif_size_mb:.1f} MB")
    print(f"  Frames  : {len(all_frames)}")
    print(f"  Duration: {duration_sec:.1f}s")
    print(f"  FPS     : {FPS}")
    return str(OUTPUT_GIF)


if __name__ == "__main__":
    generate_showcase_gif()
