#!/usr/bin/env python3
"""
YOLO Wafer Defect Detection — LinkedIn GIF Animation

Post-silicon wafer defect detection walkthrough using YOLOv8-L.
- Starts with realistic wafer image (not text slide)
- Ends with annotated wafer + real benchmark numbers
- Realistic silicon gray/silver wafer color
- 13×13 grid with thick, visible lines (width=3)
- 2× larger text for numbers and right-side content
- Real benchmark numbers: T4 FP16 + A100 TensorRT

Target: ~45s @ 3 FPS, <8MB, LinkedIn-optimized (900×620)
"""

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
REALISTIC_DIR = ROOT / "outputs" / "realistic_unseen"
OUTPUT_GIF = ROOT / "outputs" / "yolo_wafer_detection.gif"

# ── Canvas ──
W, H = 900, 620
BG = (10, 14, 22)
PANEL_BG = (18, 22, 34)
ACCENT = (0, 200, 255)
GREEN = (0, 230, 120)
RED = (255, 65, 80)
YELLOW = (255, 215, 50)
ORANGE = (255, 140, 40)
WHITE = (235, 240, 250)
DIM = (120, 140, 170)
GRID_COL = (90, 130, 200)

CLASS_COLORS = {
    "scratch": (255, 100, 100), "particle": (100, 200, 255),
    "edge_chip": (255, 255, 80), "void": (200, 100, 255),
    "crack": (255, 80, 80), "contamination": (255, 200, 100),
    "delamination": (150, 220, 255), "pattern_shift": (255, 160, 50),
    "bridge": (100, 255, 180), "missing_bond": (255, 100, 200),
}

FPS = 3
FRAME_MS = 333
GRID_N = 13


def _font(sz):
    for p in ["/System/Library/Fonts/Helvetica.ttc",
              "/System/Library/Fonts/SFNSMono.ttf",
              "/Library/Fonts/Arial.ttf",
              "/System/Library/Fonts/Supplemental/Arial.ttf"]:
        try:
            return ImageFont.truetype(p, sz)
        except (IOError, OSError):
            pass
    return ImageFont.load_default()


# ── Fonts (2× larger for readability) ──
F_HERO = _font(34)
F_TITLE = _font(26)
F_PHASE = _font(24)
F_STAT = _font(28)
F_BODY = _font(20)
F_LABEL = _font(20)
F_SMALL = _font(17)
F_BOX = _font(15)

# ── Layout ──
INFO_X = 470


def recolor_wafer_silver(img):
    """Convert synthetic blue/lavender wafer to realistic silicon gray/silver."""
    arr = np.array(img).astype(np.float32)
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    p2, p98 = np.percentile(gray, [2, 98])
    if p98 > p2:
        gray = (gray - p2) / (p98 - p2) * 210 + 25
    gray = np.clip(gray, 0, 255)
    r = np.clip(gray * 0.93, 0, 255)
    g = np.clip(gray * 0.95, 0, 255)
    b = np.clip(gray * 1.00, 0, 255)
    rgb = np.stack([r, g, b], axis=2).astype(np.uint8)
    return Image.fromarray(rgb)


def get_detections(img_path):
    from ultralytics import YOLO
    CLS = ["scratch", "particle", "edge_chip", "void", "pattern_shift",
           "bridge", "missing_bond", "crack", "contamination", "delamination"]
    model = YOLO(str(ROOT / "models" / "best.pt"))
    results = model.predict(img_path, conf=0.15, iou=0.45, verbose=False)
    r = results[0]
    dets = []
    if r.boxes is not None:
        for box in r.boxes:
            cid = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = [int(v) for v in box.xyxy[0].tolist()]
            dets.append({"cls": CLS[cid] if cid < len(CLS) else f"c{cid}",
                         "conf": conf, "bbox": xyxy, "suppressed": False})
    dets.sort(key=lambda d: -d["conf"])
    return dets


def make_fake_candidates(real, rng, n_extra=10):
    CLS = ["scratch", "particle", "edge_chip", "void", "crack",
           "contamination", "delamination", "pattern_shift"]
    cands = []
    for d in real:
        for _ in range(rng.integers(2, 5)):
            b = d["bbox"]
            sx, sy = rng.integers(-30, 31), rng.integers(-30, 31)
            s = rng.uniform(0.75, 1.3)
            cx = (b[0] + b[2]) / 2 + sx
            cy = (b[1] + b[3]) / 2 + sy
            hw, hh = (b[2] - b[0]) / 2 * s, (b[3] - b[1]) / 2 * s
            cands.append({"cls": d["cls"],
                          "conf": d["conf"] * rng.uniform(0.2, 0.75),
                          "bbox": [int(cx - hw), int(cy - hh), int(cx + hw), int(cy + hh)],
                          "suppressed": True})
    for _ in range(n_extra):
        cx, cy = rng.integers(60, 580), rng.integers(60, 580)
        hw, hh = rng.integers(15, 50), rng.integers(15, 50)
        cands.append({"cls": rng.choice(CLS), "conf": rng.uniform(0.04, 0.18),
                      "bbox": [cx - hw, cy - hh, cx + hw, cy + hh], "suppressed": True})
    return cands


# ── Drawing helpers ──

def draw_header(draw, step_num=0, step_total=6):
    draw.rectangle([0, 0, W, 52], fill=(15, 20, 32))
    draw.text((20, 8), "YOLO", fill=ACCENT, font=F_TITLE)
    draw.text((90, 8), "Post-Silicon Defect Detection", fill=WHITE, font=F_BODY)
    if step_num > 0:
        txt = f"Step {step_num}/{step_total}"
        draw.text((W - 140, 14), txt, fill=DIM, font=F_BODY)
    draw.line([(0, 52), (W, 52)], fill=(40, 55, 75), width=1)


def draw_explanation_box(draw, lines):
    box_y = H - 80
    draw.rectangle([0, box_y, W, H], fill=(15, 20, 32))
    draw.line([(0, box_y), (W, box_y)], fill=(40, 55, 75), width=1)
    y = box_y + 8
    for text, color, font in lines:
        draw.text((30, y), text, fill=color, font=font)
        y += font.size + 6


def draw_step_indicators(draw, current, total=6):
    sx = W - total * 40 - 10
    for i in range(1, total + 1):
        x = sx + (i - 1) * 40
        if i < current:
            draw.ellipse([x, 58, x + 30, 88], fill=(0, 60, 40), outline=GREEN, width=2)
            draw.text((x + 8, 64), "v", fill=GREEN, font=F_BODY)
        elif i == current:
            draw.ellipse([x, 58, x + 30, 88], fill=(30, 40, 55), outline=ACCENT, width=2)
            draw.text((x + 9, 62), str(i), fill=ACCENT, font=F_BODY)
        else:
            draw.ellipse([x, 58, x + 30, 88], fill=(20, 25, 35), outline=(45, 55, 70), width=2)
            draw.text((x + 9, 62), str(i), fill=(60, 70, 85), font=F_BODY)


def draw_wafer_panel(frame, img, x, y, w, h, label=None, border=(60, 70, 90)):
    draw = ImageDraw.Draw(frame)
    draw.rectangle([x - 4, y - 4, x + w + 4, y + h + 4], fill=PANEL_BG)
    frame.paste(img, (x, y))
    draw.rectangle([x - 1, y - 1, x + w + 1, y + h + 1], outline=border, width=2)
    if label:
        tw = draw.textlength(label, font=F_LABEL)
        draw.text((x + (w - tw) // 2, y + h + 6), label, fill=DIM, font=F_LABEL)


def draw_grid(draw, px, py, size, n, progress=1.0):
    """Draw 13×13 grid with THICK visible lines (width=3)."""
    cs = size / n
    alpha = min(1.0, progress * 1.5) if progress < 1.0 else 1.0
    col = tuple(int(c * alpha) for c in GRID_COL)
    for i in range(n + 1):
        yy = int(py + i * cs)
        draw.line([(px, yy), (px + size, yy)], fill=col, width=3)
        xx = int(px + i * cs)
        draw.line([(xx, py), (xx, py + size)], fill=col, width=3)


def draw_cell_activations(draw, px, py, size, n, dets, img_scale, progress=1.0):
    cs = size / n
    active = set()
    for d in dets:
        b = d["bbox"]
        cx = (b[0] + b[2]) / 2 * img_scale
        cy = (b[1] + b[3]) / 2 * img_scale
        r, c = int(cy / cs), int(cx / cs)
        r, c = min(max(r, 0), n - 1), min(max(c, 0), n - 1)
        active.add((r, c))
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n:
                    active.add((nr, nc))
    show = sorted(active)[:int(len(active) * progress)]
    for r, c in show:
        x1, y1 = px + c * cs, py + r * cs
        draw.rectangle([x1 + 1, y1 + 1, x1 + cs - 1, y1 + cs - 1],
                       fill=(0, 45, 35), outline=GREEN, width=1)


def draw_candidate_box(draw, d, px, py, sc, ghost=False):
    b = d["bbox"]
    x1, y1 = int(px + b[0] * sc), int(py + b[1] * sc)
    x2, y2 = int(px + b[2] * sc), int(py + b[3] * sc)
    col = CLASS_COLORS.get(d["cls"], (200, 200, 200))
    if ghost or d.get("suppressed"):
        col = tuple(int(c * 0.35) for c in col)
        draw.rectangle([x1, y1, x2, y2], outline=col, width=1)
    else:
        draw.rectangle([x1, y1, x2, y2], outline=col, width=2)


def draw_final_box(draw, d, px, py, sc, progress=1.0):
    if progress <= 0:
        return
    b = d["bbox"]
    col = CLASS_COLORS.get(d["cls"], (255, 255, 255))
    x1, y1 = int(px + b[0] * sc), int(py + b[1] * sc)
    x2, y2 = int(px + b[2] * sc), int(py + b[3] * sc)
    if progress < 1.0:
        cx_b, cy_b = (x1 + x2) / 2, (y1 + y2) / 2
        e = progress ** 0.5
        hw, hh = (x2 - x1) / 2 * e, (y2 - y1) / 2 * e
        x1, y1, x2, y2 = int(cx_b - hw), int(cy_b - hh), int(cx_b + hw), int(cy_b + hh)
    draw.rectangle([x1, y1, x2, y2], outline=col, width=3)
    cl = min(10, (x2 - x1) // 4, (y2 - y1) // 4)
    if progress > 0.5 and cl > 2:
        for cx, cy, dx, dy in [(x1, y1, 1, 1), (x2, y1, -1, 1),
                                (x1, y2, 1, -1), (x2, y2, -1, -1)]:
            draw.line([(cx, cy), (cx + cl * dx, cy)], fill=col, width=3)
            draw.line([(cx, cy), (cx, cy + cl * dy)], fill=col, width=3)
    if progress > 0.7:
        label = f"{d['cls']} {d['conf']:.0%}"
        tb = draw.textbbox((0, 0), label, font=F_BOX)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ly = y1 - th - 5
        if ly < py:
            ly = y2 + 2
        draw.rectangle([x1, ly - 1, x1 + tw + 6, ly + th + 3], fill=col)
        draw.text((x1 + 3, ly), label, fill=(0, 0, 0), font=F_BOX)


def draw_nms_cross(draw, d, px, py, sc):
    b = d["bbox"]
    x1, y1 = int(px + b[0] * sc), int(py + b[1] * sc)
    x2, y2 = int(px + b[2] * sc), int(py + b[3] * sc)
    col = tuple(int(c * 0.25) for c in CLASS_COLORS.get(d["cls"], (200, 200, 200)))
    draw.rectangle([x1, y1, x2, y2], outline=col, width=1)
    draw.line([(x1, y1), (x2, y2)], fill=RED, width=2)
    draw.line([(x2, y1), (x1, y2)], fill=RED, width=2)


def draw_defect_hints(draw, dets, px, py, sc):
    """Draw subtle circles around defect locations to show they exist."""
    for d in dets:
        b = d["bbox"]
        cx = int(px + (b[0] + b[2]) / 2 * sc)
        cy = int(py + (b[1] + b[3]) / 2 * sc)
        r = int(max(b[2] - b[0], b[3] - b[1]) * sc * 0.4) + 8
        col = CLASS_COLORS.get(d["cls"], (255, 255, 255))
        dim_col = tuple(int(c * 0.5) for c in col)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=dim_col, width=2)


# ── Layout ──
LX, LY = 25, 95
RX, RY = 465, 95
PS = 395


def generate():
    rng = np.random.default_rng(42)
    img_path = str(REALISTIC_DIR / "realistic_05.jpg")
    orig = Image.open(img_path).convert("RGB")

    # Recolor to realistic silicon gray/silver
    silver_orig = recolor_wafer_silver(orig)

    real_dets = [d for d in get_detections(img_path) if d["conf"] >= 0.25]
    for d in real_dets:
        d["suppressed"] = False
    candidates = make_fake_candidates(real_dets, rng)
    all_dets = real_dets + candidates
    n_real, n_cands = len(real_dets), len(all_dets)
    suppressed = [d for d in all_dets if d.get("suppressed")]
    n_supp = len(suppressed)

    sc = PS / max(silver_orig.size)
    dw, dh = int(silver_orig.size[0] * sc), int(silver_orig.size[1] * sc)
    wafer_sm = silver_orig.resize((dw, dh), Image.LANCZOS)

    # Larger wafer for opening/closing (fills left side)
    OPS = 440
    osc = OPS / max(silver_orig.size)
    odw, odh = int(silver_orig.size[0] * osc), int(silver_orig.size[1] * osc)
    wafer_lg = silver_orig.resize((odw, odh), Image.LANCZOS)

    print(f"  Wafer: realistic_05.jpg (silver) — {n_real} real + {len(candidates)} candidates")

    frames = []

    def base(step):
        f = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(f)
        draw_header(d, step, 6)
        if step > 0:
            draw_step_indicators(d, step)
        return f, d

    # ═══════════════════════════════════════════
    # OPENING — Wafer image + title (4s = 12 frames)
    # ═══════════════════════════════════════════
    for i in range(12):
        f = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(f)
        draw_header(d)

        # Large wafer on left
        alpha = min(1.0, i / 4)
        if alpha < 1.0:
            dark = Image.new("RGB", (odw, odh), BG)
            panel = Image.blend(dark, wafer_lg, alpha)
        else:
            panel = wafer_lg
        draw_wafer_panel(f, panel, LX, 90, odw, odh, border=ACCENT)

        # Defect hints on wafer (after fade-in)
        if i >= 6:
            d2 = ImageDraw.Draw(f)
            draw_defect_hints(d2, real_dets, LX, 90, osc)

        # Right-side title text
        tx = INFO_X
        ty = 110
        d.text((tx, ty), "Post-Silicon Wafer", fill=ACCENT, font=F_HERO)
        d.text((tx, ty + 42), "Defect Detection", fill=WHITE, font=F_HERO)
        ty += 100
        d.text((tx, ty), "YOLOv8-L  |  44M Parameters", fill=DIM, font=F_BODY)
        ty += 35
        d.text((tx, ty), "mAP@50: 99.22%", fill=YELLOW, font=F_STAT)
        ty += 40
        d.text((tx, ty), "10 Semiconductor Defect Classes", fill=DIM, font=F_BODY)
        ty += 50
        d.line([(tx, ty), (tx + 350, ty)], fill=(40, 55, 75), width=1)
        ty += 15
        d.text((tx, ty), '"You Only Look Once"', fill=ACCENT, font=F_BODY)
        ty += 30
        d.text((tx, ty), "Full image >> Single forward pass", fill=DIM, font=F_SMALL)
        ty += 22
        d.text((tx, ty), ">> All defects detected simultaneously", fill=DIM, font=F_SMALL)

        frames.append(f)

    # ═══════════════════════════════════════════
    # STEP 1: INPUT IMAGE (5s = 15 frames)
    # ═══════════════════════════════════════════
    for i in range(15):
        f, d = base(1)
        d.text((LX, LY - 30), "Step 1: Input Image", fill=ACCENT, font=F_PHASE)

        # Wafer fading in
        alpha = min(1.0, i / 5)
        if alpha < 1.0:
            dark = Image.new("RGB", (dw, dh), BG)
            draw_wafer_panel(f, Image.blend(dark, wafer_sm, alpha),
                             LX, LY, dw, dh, "Input: 640×640 wafer image", ACCENT)
        else:
            draw_wafer_panel(f, wafer_sm, LX, LY, dw, dh,
                             "Input: 640×640 wafer image", ACCENT)

        # Arrow to CNN
        if i >= 5:
            ap = min(1.0, (i - 5) / 5)
            ax1 = LX + dw + 12
            ax2 = int(ax1 + (RX - ax1 - 10) * ap)
            ay = LY + dh // 2
            d.line([(ax1, ay), (ax2, ay)], fill=ACCENT, width=3)
            if ap > 0.5:
                d.polygon([(ax2, ay - 8), (ax2 + 14, ay), (ax2, ay + 8)], fill=ACCENT)
                d.text((ax1 + 8, ay - 25), "CNN backbone", fill=WHITE, font=F_LABEL)

        # Right placeholder
        if i >= 10:
            d.rectangle([RX - 4, RY - 4, RX + dw + 4, RY + dh + 4],
                        fill=PANEL_BG, outline=(40, 50, 65), width=2)
            d.text((RX + dw // 2 - 60, RY + dh // 2 - 12),
                   "Processing...", fill=DIM, font=F_BODY)

        # Defect hints after full reveal
        if i >= 8:
            d2 = ImageDraw.Draw(f)
            draw_defect_hints(d2, real_dets, LX, LY, sc)

        draw_explanation_box(d, [
            ("Feed the ENTIRE wafer image into the neural network", WHITE, F_BODY),
            ("No scanning, no sliding window — one forward pass", YELLOW, F_SMALL),
        ])
        frames.append(f)

    # ═══════════════════════════════════════════
    # STEP 2: FEATURE GRID (5s = 15 frames)
    # ═══════════════════════════════════════════
    for i in range(15):
        f, d = base(2)
        d.text((LX, LY - 30), "Step 2: Feature Grid (13×13)", fill=ACCENT, font=F_PHASE)
        draw_wafer_panel(f, wafer_sm, LX, LY, dw, dh, "Original", (60, 70, 90))

        # Right panel: wafer + thick grid
        f.paste(wafer_sm, (RX, RY))
        d2 = ImageDraw.Draw(f)
        gp = min(1.0, i / 8)
        draw_grid(d2, RX, RY, dw, GRID_N, gp)
        d2.rectangle([RX - 1, RY - 1, RX + dw + 1, RY + dh + 1], outline=ACCENT, width=2)
        d2.text((RX + dw // 2 - 100, RY + dh + 8),
                f"Feature map: {GRID_N}×{GRID_N} = {GRID_N**2} cells", fill=ACCENT, font=F_LABEL)

        draw_explanation_box(d, [
            (f"CNN outputs a {GRID_N}×{GRID_N} feature grid over the image", WHITE, F_BODY),
            ("Each cell detects objects whose CENTER falls within it", ACCENT, F_SMALL),
        ])
        frames.append(f)

    # ═══════════════════════════════════════════
    # STEP 3: SIMULTANEOUS PREDICTION (5s = 15 frames)
    # ═══════════════════════════════════════════
    for i in range(15):
        f, d = base(3)
        d.text((LX, LY - 30), "Step 3: Simultaneous Prediction", fill=GREEN, font=F_PHASE)
        draw_wafer_panel(f, wafer_sm, LX, LY, dw, dh, "Original", (60, 70, 90))

        f.paste(wafer_sm, (RX, RY))
        d2 = ImageDraw.Draw(f)
        draw_grid(d2, RX, RY, dw, GRID_N)
        ap = min(1.0, i / 8)
        draw_cell_activations(d2, RX, RY, dw, GRID_N, real_dets, sc, ap)
        d2.rectangle([RX - 1, RY - 1, RX + dw + 1, RY + dh + 1], outline=GREEN, width=2)
        na = int(len(real_dets) * 9 * ap)
        d2.text((RX + dw // 2 - 80, RY + dh + 8),
                f"Active cells: {na} (defect regions)", fill=GREEN, font=F_LABEL)

        draw_explanation_box(d, [
            ("All grid cells predict SIMULTANEOUSLY — not one by one!", WHITE, F_BODY),
            ("Green cells: defect center detected → box + class + confidence", GREEN, F_SMALL),
        ])
        frames.append(f)

    # ═══════════════════════════════════════════
    # STEP 4: RAW CANDIDATE BOXES (5s = 15 frames)
    # ═══════════════════════════════════════════
    for i in range(15):
        f, d = base(4)
        d.text((LX, LY - 30), "Step 4: Raw Candidate Boxes", fill=ORANGE, font=F_PHASE)
        draw_wafer_panel(f, wafer_sm, LX, LY, dw, dh, "Original", (60, 70, 90))

        f.paste(wafer_sm, (RX, RY))
        d2 = ImageDraw.Draw(f)
        ns = int(n_cands * min(1.0, i / 8))
        for det in all_dets[:ns]:
            draw_candidate_box(d2, det, RX, RY, sc, det.get("suppressed", False))
        d2.rectangle([RX - 1, RY - 1, RX + dw + 1, RY + dh + 1], outline=ORANGE, width=2)
        d2.text((RX + dw // 2 - 90, RY + dh + 8),
                f"Candidates: {ns} boxes (overlapping)", fill=ORANGE, font=F_LABEL)

        max_conf = max((det["conf"] for det in real_dets), default=0.9)
        draw_explanation_box(d, [
            (f"{n_cands} raw predictions — mostly overlapping duplicates", WHITE, F_BODY),
            (f"Confidence: 4% to {max_conf:.0%} — duplicates need filtering next", ORANGE, F_SMALL),
        ])
        frames.append(f)

    # ═══════════════════════════════════════════
    # STEP 5: NON-MAX SUPPRESSION (6s = 18 frames)
    # ═══════════════════════════════════════════
    for i in range(18):
        f, d = base(5)
        d.text((LX, LY - 30), "Step 5: Non-Max Suppression", fill=RED, font=F_PHASE)
        draw_wafer_panel(f, wafer_sm, LX, LY, dw, dh, "Original", (60, 70, 90))

        f.paste(wafer_sm, (RX, RY))
        d2 = ImageDraw.Draw(f)
        nr = int(n_supp * min(1.0, i / 10))
        for det in real_dets:
            draw_candidate_box(d2, det, RX, RY, sc)
        for j, det in enumerate(suppressed):
            if j < nr:
                draw_nms_cross(d2, det, RX, RY, sc)
            else:
                draw_candidate_box(d2, det, RX, RY, sc, ghost=True)
        d2.rectangle([RX - 1, RY - 1, RX + dw + 1, RY + dh + 1], outline=RED, width=2)
        d2.text((RX + dw // 2 - 80, RY + dh + 8),
                f"Removed: {nr}/{n_supp} duplicates", fill=RED, font=F_LABEL)

        draw_explanation_box(d, [
            ("NMS removes duplicate and low-confidence predictions", WHITE, F_BODY),
            (f"Overlap > 45% IoU → weaker box discarded. {n_supp} → {n_real} final", RED, F_SMALL),
        ])
        frames.append(f)

    # ═══════════════════════════════════════════
    # STEP 6: FINAL DETECTIONS (4 frames/box + 15 hold)
    # ═══════════════════════════════════════════
    for di in range(n_real):
        for i in range(4):
            f, d = base(6)
            d.text((LX, LY - 30),
                   f"Step 6: Detection {di + 1}/{n_real}", fill=GREEN, font=F_PHASE)
            draw_wafer_panel(f, wafer_sm, LX, LY, dw, dh, "Original", (60, 70, 90))
            f.paste(wafer_sm, (RX, RY))
            d2 = ImageDraw.Draw(f)
            for prev in range(di):
                draw_final_box(d2, real_dets[prev], RX, RY, sc, 1.0)
            draw_final_box(d2, real_dets[di], RX, RY, sc, (i + 1) / 4)
            d2.rectangle([RX - 1, RY - 1, RX + dw + 1, RY + dh + 1], outline=GREEN, width=2)
            det = real_dets[di]
            d2.text((RX + dw // 2 - 80, RY + dh + 8),
                    f"Found: {det['cls']} ({det['conf']:.0%})", fill=GREEN, font=F_LABEL)

            draw_explanation_box(d, [
                (f"Confirmed: {det['cls']} at {det['conf']:.0%} confidence", WHITE, F_BODY),
                (f"Detection {di + 1} of {n_real} defects on this wafer", GREEN, F_SMALL),
            ])
            frames.append(f)

    # Hold final detections
    for _ in range(15):
        f, d = base(6)
        d.text((LX, LY - 30), "All Defects Detected", fill=GREEN, font=F_PHASE)
        draw_wafer_panel(f, wafer_sm, LX, LY, dw, dh, "Original", (60, 70, 90))
        f.paste(wafer_sm, (RX, RY))
        d2 = ImageDraw.Draw(f)
        for det in real_dets:
            draw_final_box(d2, det, RX, RY, sc, 1.0)
        d2.rectangle([RX - 1, RY - 1, RX + dw + 1, RY + dh + 1], outline=GREEN, width=2)
        classes = sorted({d["cls"] for d in real_dets})
        d2.text((RX + 10, RY + dh + 8),
                f"{n_real} defects: {', '.join(classes)}", fill=GREEN, font=F_LABEL)

        draw_explanation_box(d, [
            (f"Complete: {n_real} defects detected in ONE forward pass", GREEN, F_BODY),
            ("T4 FP16: 16.1ms / 62 FPS  |  A100 TRT: 4.5ms / 221 FPS", YELLOW, F_SMALL),
        ])
        frames.append(f)

    # ═══════════════════════════════════════════
    # CLOSING — Annotated wafer + benchmark numbers (6s = 18 frames)
    # ═══════════════════════════════════════════
    for _ in range(18):
        f = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(f)
        draw_header(d)

        # Left: large annotated wafer with all detections
        draw_wafer_panel(f, wafer_lg, LX, 90, odw, odh, border=GREEN)
        d2 = ImageDraw.Draw(f)
        for det in real_dets:
            draw_final_box(d2, det, LX, 90, osc, 1.0)

        # Right: BIG benchmark numbers
        tx = INFO_X
        ty = 100
        d.text((tx, ty), f"{n_real} Defects Detected", fill=GREEN, font=F_HERO)
        ty += 48
        classes = sorted({det["cls"] for det in real_dets})
        d.text((tx, ty), f"Classes: {', '.join(classes)}", fill=WHITE, font=F_BODY)

        ty += 50
        d.line([(tx, ty), (tx + 350, ty)], fill=(40, 55, 75), width=1)
        ty += 20

        # GPU Performance (large, prominent)
        d.text((tx, ty), "GPU Performance", fill=ACCENT, font=F_PHASE)
        ty += 38
        d.text((tx, ty), "T4 FP16", fill=DIM, font=F_BODY)
        d.text((tx + 140, ty), "16.1ms  |  62 FPS", fill=ACCENT, font=F_STAT)
        ty += 40
        d.text((tx, ty), "A100 TRT", fill=DIM, font=F_BODY)
        d.text((tx + 140, ty), "4.5ms  |  221 FPS", fill=YELLOW, font=F_STAT)

        ty += 55
        d.line([(tx, ty), (tx + 350, ty)], fill=(40, 55, 75), width=1)
        ty += 15
        d.text((tx, ty), "mAP@50: 99.22%  |  10 classes", fill=WHITE, font=F_BODY)
        ty += 30
        d.text((tx, ty), "YOLOv8-L  |  44M parameters", fill=DIM, font=F_SMALL)
        ty += 25
        d.text((tx, ty), "github.com/AIML-Engineering-Lab/", fill=DIM, font=F_SMALL)
        ty += 22
        d.text((tx, ty), "054_wafer_defect_yolo_detection_mlops", fill=ACCENT, font=F_SMALL)

        frames.append(f)

    # ── Encode ──
    print(f"\nEncoding: {len(frames)} frames @ {FPS} FPS = {len(frames)/FPS:.1f}s")
    q = [fr.quantize(colors=128, method=Image.Quantize.MEDIANCUT,
                      dither=Image.Dither.FLOYDSTEINBERG) for fr in frames]
    q[0].save(str(OUTPUT_GIF), save_all=True, append_images=q[1:],
              duration=FRAME_MS, loop=0, optimize=True)
    sz = OUTPUT_GIF.stat().st_size / 1e6
    print(f"  Saved: {OUTPUT_GIF}")
    print(f"  Size : {sz:.1f} MB  |  {len(frames)} frames  |  {len(frames)/FPS:.1f}s @ {FPS} FPS")


if __name__ == "__main__":
    generate()
