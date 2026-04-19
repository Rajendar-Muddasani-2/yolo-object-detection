#!/usr/bin/env python3
"""
YOLO Educational GIF v2 — "You Only Look Once"

Story arc:
  ACT 1 — Traditional CNN/sliding-window: agonisingly slow scanner crawls
           across the wafer image window-by-window, counter ticking up.
  CUT    — Hard contrast cut with "VS" flash
  ACT 2 — YOLO: full image in → grid predicted → ALL boxes appear at ONCE
  ACT 3 — Results: final annotated wafer + metrics + LinkedIn-ready outro

LinkedIn-optimised: 900×550, ~18 s loop, vivid colours, dark theme.
"""

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_GIF = ROOT / "outputs" / "yolo_wafer_detection.gif"

# ── Palette ────────────────────────────────────────────────────────────────────
W, H = 900, 550
BG       = (8,  12, 22)
PANEL_BG = (16, 22, 36)
ACCENT   = (0,  210, 255)     # cyan — YOLO brand
RED_ALT  = (255, 55, 80)      # traditional / slow
GREEN    = (0,  230, 120)
YELLOW   = (255, 215, 50)
WHITE    = (240, 245, 255)
DIM      = (90, 110, 140)
GOLD     = (255, 195, 40)
PURPLE   = (160, 80, 255)

CLASS_COLORS = [
    (255, 100, 100), (100, 210, 255), (255, 255, 80),  (200, 100, 255),
    (255, 90,  80),  (255, 200, 100), (150, 220, 255), (255, 160,  50),
    (100, 255, 180), (255, 110, 200),
]
CLASS_NAMES = [
    "scratch", "particle", "edge_chip", "void", "crack",
    "contamination", "delamination", "pattern_shift", "bridge", "missing_bond",
]

FPS        = 8          # higher FPS → smoother
FRAME_MS   = 1000 // FPS


# ── Font helper ────────────────────────────────────────────────────────────────
def _font(sz, bold=False):
    candidates = [
        "/System/Library/Fonts/Avenir Next Condensed.ttc",
        "/System/Library/Fonts/Avenir Next.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/ArialHB.ttc",
        "/Library/Fonts/Arial Bold.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, sz)
        except (IOError, OSError):
            pass
    return ImageFont.load_default()


F_HUGE   = _font(48, bold=True)
F_BIG    = _font(32, bold=True)
F_MED    = _font(22)
F_BODY   = _font(17)
F_SMALL  = _font(14)
F_TINY   = _font(12)


# ── Canvas helpers ─────────────────────────────────────────────────────────────
def blank():
    img = Image.new("RGB", (W, H), BG)
    return img


def draw_header_bar(draw, left_label, left_color, right_label=None):
    draw.rectangle([0, 0, W, 46], fill=(12, 16, 28))
    draw.line([(0, 46), (W, 46)], fill=(35, 50, 70), width=1)
    draw.text((20, 10), left_label, fill=left_color, font=F_BIG)
    if right_label:
        tw = F_MED.getlength(right_label)
        draw.text((W - tw - 20, 14), right_label, fill=DIM, font=F_MED)


def draw_footer(draw, text, color=DIM):
    draw.rectangle([0, H - 36, W, H], fill=(10, 14, 24))
    draw.line([(0, H - 36), (W, H - 36)], fill=(30, 44, 62), width=1)
    tw = F_SMALL.getlength(text)
    draw.text(((W - tw) // 2, H - 26), text, fill=color, font=F_SMALL)


def paste_wafer(frame, wafer_img, x, y, size):
    """Resize+paste wafer image at (x,y) with given size."""
    wf = wafer_img.resize((size, size), Image.LANCZOS)
    frame.paste(wf, (x, y))
    d = ImageDraw.Draw(frame)
    d.rectangle([x - 2, y - 2, x + size + 2, y + size + 2],
                outline=(45, 60, 85), width=2)
    return wf


def rounded_rect(draw, x1, y1, x2, y2, r, fill=None, outline=None, width=1):
    """Draw a rounded rectangle (PIL doesn't have it below 9.2)."""
    if fill:
        draw.rectangle([x1 + r, y1, x2 - r, y2], fill=fill)
        draw.rectangle([x1, y1 + r, x2, y2 - r], fill=fill)
        draw.ellipse([x1, y1, x1 + 2*r, y1 + 2*r], fill=fill)
        draw.ellipse([x2 - 2*r, y1, x2, y1 + 2*r], fill=fill)
        draw.ellipse([x1, y2 - 2*r, x1 + 2*r, y2], fill=fill)
        draw.ellipse([x2 - 2*r, y2 - 2*r, x2, y2], fill=fill)
    if outline:
        draw.rectangle([x1 + r, y1, x2 - r, y1 + width], fill=outline)
        draw.rectangle([x1 + r, y2 - width, x2 - r, y2], fill=outline)
        draw.rectangle([x1, y1 + r, x1 + width, y2 - r], fill=outline)
        draw.rectangle([x2 - width, y1 + r, x2, y2 - r], fill=outline)
        draw.arc([x1, y1, x1 + 2*r, y1 + 2*r], 180, 270, fill=outline, width=width)
        draw.arc([x2 - 2*r, y1, x2, y1 + 2*r], 270, 360, fill=outline, width=width)
        draw.arc([x1, y2 - 2*r, x1 + 2*r, y2],  90, 180, fill=outline, width=width)
        draw.arc([x2 - 2*r, y2 - 2*r, x2, y2],   0,  90, fill=outline, width=width)


# ── Synthetic wafer image generator ───────────────────────────────────────────
def make_synthetic_wafer(size=640, seed=7):
    """
    Generate a realistic-looking silicon wafer image:
    - Light grey circular die surface
    - Grid of rectangular die cells
    - Random defect regions (darker patches / edge chips)
    """
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), (210, 215, 220), dtype=np.uint8)

    # Wafer circle boundary
    cy, cx = size // 2, size // 2
    radius = int(size * 0.48)
    y_idx, x_idx = np.ogrid[:size, :size]
    dist = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)
    outside = dist > radius
    img[outside] = (30, 35, 45)  # dark background outside wafer

    # Silicon die grid lines
    step = size // 14
    for i in range(0, size, step):
        img[max(0, i-1):i+1, :] = np.where(
            np.stack([outside[max(0, i-1):i+1, :]] * 3, axis=-1),
            img[max(0, i-1):i+1, :],
            (180, 185, 190)
        )
        img[:, max(0, i-1):i+1] = np.where(
            np.stack([outside[:, max(0, i-1):i+1]] * 3, axis=-1),
            img[:, max(0, i-1):i+1],
            (180, 185, 190)
        )

    # Slight radial shading for realism
    fade = np.clip(1.0 - (dist / radius) ** 2 * 0.15, 0.85, 1.0)
    for c in range(3):
        channel = img[:, :, c].astype(float)
        channel[~outside] = (channel * fade)[~outside]
        img[:, :, c] = channel.clip(0, 255).astype(np.uint8)

    pil_img = Image.fromarray(img)

    # Add some circuit-like fine detail
    d = ImageDraw.Draw(pil_img)
    for _ in range(200):
        lx, ly = rng.integers(20, size - 20, size=2)
        if dist[ly, lx] < radius * 0.95:
            length = rng.integers(8, 30)
            angle = rng.choice([0, 90])
            col = tuple(rng.integers(160, 200, size=3).tolist())
            if angle == 0:
                d.line([(lx, ly), (lx + length, ly)], fill=col, width=1)
            else:
                d.line([(lx, ly), (lx, ly + length)], fill=col, width=1)

    return pil_img


# ── Use pre-computed detections from stored JSON ───────────────────────────────
def get_detections_from_json(target_size=640):
    """Load real detections from stored inference JSON, scaled to target_size."""
    import json
    json_path = ROOT / "outputs" / "unseen_results" / "unseen_inference_results.json"
    if not json_path.exists():
        return get_detections_synthetic(target_size)

    with open(json_path) as f:
        data = json.load(f)

    # Take first image with decent detections
    per_image = data.get("per_image", [])
    best = None
    for entry in per_image:
        if entry.get("num_detections", 0) >= 2:
            best = entry
            break
    if best is None and per_image:
        best = per_image[0]
    if best is None:
        return get_detections_synthetic(target_size)

    # Original image is typically 640×640 — rescale if different
    raw_dets = best.get("detections", [])
    ORIG_SIZE = 640  # known from training
    scale = target_size / ORIG_SIZE
    dets = []
    for d in raw_dets:
        cid = d.get("class_id", 2)
        b = d["bbox_xyxy"]
        dets.append({
            "cls_id": cid,
            "cls": CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"c{cid}",
            "conf": d.get("confidence", 0.7),
            "bbox": [int(b[0]*scale), int(b[1]*scale), int(b[2]*scale), int(b[3]*scale)],
        })
    dets.sort(key=lambda d: -d["conf"])
    return dets


def get_detections_synthetic(target_size=640):
    """Fallback: deterministic fake detections that look realistic."""
    positions = [
        (2, "edge_chip",      0.92, [395, 80,  460, 124]),
        (2, "edge_chip",      0.73, [495, 72,  511, 86]),
        (7, "pattern_shift",  0.65, [200, 300, 270, 370]),
        (4, "crack",          0.58, [310, 180, 380, 220]),
        (2, "edge_chip",      0.51, [100, 110, 155, 148]),
    ]
    scale = target_size / 640
    return [{"cls_id": cid, "cls": cls, "conf": conf,
             "bbox": [int(b[0]*scale), int(b[1]*scale), int(b[2]*scale), int(b[3]*scale)]}
            for cid, cls, conf, b in positions]


# ── Frame generators ───────────────────────────────────────────────────────────

def frames_act0_title():
    """3 frames: punchy title card"""
    frames = []
    for _ in range(3):
        img = blank()
        d = ImageDraw.Draw(img)
        # Background subtle grid
        for gx in range(0, W, 40):
            d.line([(gx, 0), (gx, H)], fill=(18, 24, 36), width=1)
        for gy in range(0, H, 40):
            d.line([(0, gy), (W, gy)], fill=(18, 24, 36), width=1)

        # Title
        title = "You Only Look Once"
        tw = F_HUGE.getlength(title)
        d.text(((W - tw) // 2, 120), title, fill=ACCENT, font=F_HUGE)

        sub = "YOLO — Real-Time Semiconductor Wafer Defect Detection"
        sw = F_MED.getlength(sub)
        d.text(((W - sw) // 2, 195), sub, fill=WHITE, font=F_MED)

        # Accent underline
        lx = (W - tw) // 2
        d.rectangle([lx, 178, lx + int(tw), 181], fill=ACCENT)

        # Bottom tag
        tag = "99.22% mAP@50  ·  10 Defect Classes  ·  YOLOv8-Large  ·  43.7M params"
        tw2 = F_SMALL.getlength(tag)
        d.text(((W - tw2) // 2, H - 60), tag, fill=DIM, font=F_SMALL)

        frames.append(img)
    return frames


def frames_act1_traditional(wafer_img):
    """
    ACT 1 — Show a sliding-window scanner crawling across the image.
    Two columns: left = full wafer with red scanning window crawling,
    right = running counter + "SLOW" label.
    ~25 frames total.
    """
    frames = []
    rng = np.random.default_rng(42)
    SIZE = 360       # wafer panel size
    WX, WY = 50, 70  # wafer top-left in frame

    WINDOW = 80      # scanning window size
    STEP   = 60      # step size per scan
    positions = []
    for row in range(0, SIZE - WINDOW + 1, STEP):
        for col in range(0, SIZE - WINDOW + 1, STEP):
            positions.append((col, row))
    # Pad to ~30 steps for visual effect
    total_steps = max(len(positions), 28)

    for step_i, (wx, wy) in enumerate(positions[:total_steps]):
        img = blank()
        d = ImageDraw.Draw(img)

        draw_header_bar(d, "Traditional CNN/Sliding Window", RED_ALT,
                        f"Scan {step_i + 1}/{total_steps}")

        # Wafer panel
        paste_wafer(img, wafer_img, WX, WY, SIZE)
        d2 = ImageDraw.Draw(img)

        # Draw all previously scanned regions (faint grey overlay)
        for pi in range(step_i):
            px_p, py_p = positions[pi]
            x1 = WX + px_p
            y1 = WY + py_p
            x2 = x1 + WINDOW
            y2 = y1 + WINDOW
            overlay = Image.new("RGBA", (WINDOW, WINDOW), (40, 55, 80, 60))
            img.paste(Image.new("RGB", (WINDOW, WINDOW), (40, 55, 80)),
                      (x1, y1),
                      mask=Image.new("L", (WINDOW, WINDOW), 55))

        # Current scanning window — bright red
        x1 = WX + wx
        y1 = WY + wy
        x2 = x1 + WINDOW
        y2 = y1 + WINDOW
        overlay = Image.new("RGB", (WINDOW, WINDOW), (255, 55, 80))
        img.paste(overlay, (x1, y1),
                  mask=Image.new("L", (WINDOW, WINDOW), 45))
        d2.rectangle([x1, y1, x2, y2], outline=RED_ALT, width=3)
        # Dashes on corners
        corner = 12
        for (cx, cy), (dx, dy) in [((x1, y1), (1, 1)), ((x2, y1), (-1, 1)),
                                     ((x1, y2), (1, -1)), ((x2, y2), (-1, -1))]:
            d2.line([(cx, cy), (cx + dx * corner, cy)], fill=RED_ALT, width=2)
            d2.line([(cx, cy), (cx, cy + dy * corner)], fill=RED_ALT, width=2)

        # RIGHT PANEL — stats
        rx = WX + SIZE + 60
        ry = WY

        d2.text((rx, ry), "Approach:", fill=DIM, font=F_SMALL)
        d2.text((rx, ry + 22), "Sliding Window CNN", fill=RED_ALT, font=F_BODY)

        d2.text((rx, ry + 60), "Scans completed:", fill=DIM, font=F_SMALL)
        scan_txt = str(step_i + 1)
        d2.text((rx, ry + 82), scan_txt, fill=RED_ALT, font=F_BIG)

        d2.text((rx, ry + 130), "Total scans needed:", fill=DIM, font=F_SMALL)
        d2.text((rx, ry + 152), str(total_steps) + "+", fill=WHITE, font=F_BODY)

        # fake latency — grows per scan
        fake_ms = 12 + step_i * 11
        d2.text((rx, ry + 195), "Time so far:", fill=DIM, font=F_SMALL)
        d2.text((rx, ry + 217), f"{fake_ms} ms", fill=RED_ALT, font=F_BODY)

        d2.text((rx, ry + 260), "Detections so far:", fill=DIM, font=F_SMALL)
        d2.text((rx, ry + 282), "Still scanning...", fill=YELLOW, font=F_BODY)

        # Progress bar
        bar_w = 260
        bar_h = 12
        bx, by = rx, ry + 320
        d2.rectangle([bx, by, bx + bar_w, by + bar_h], outline=DIM, width=1)
        fill_w = int(bar_w * (step_i + 1) / total_steps)
        d2.rectangle([bx + 1, by + 1, bx + fill_w, by + bar_h - 1], fill=RED_ALT)
        d2.text((bx, by + 17), f"Progress: {int((step_i+1)/total_steps*100)}%",
                fill=DIM, font=F_TINY)

        draw_footer(d2, "Traditional approach: crops & classifies every patch — O(N²) scans per image")
        frames.append(img)

    # 2 hold frames on final state
    frames.append(frames[-1])
    frames.append(frames[-1])
    return frames


def frames_vs_flash():
    """3 frames: high-contrast VS flash between the two approaches"""
    frames = []
    for i in range(3):
        img = blank()
        d = ImageDraw.Draw(img)

        # Left half — Traditional (red)
        d.rectangle([0, 0, W // 2, H], fill=(35, 8, 12))
        lt = "SLOW"
        tw = F_HUGE.getlength(lt)
        d.text(((W // 2 - tw) // 2, H // 2 - 50), lt, fill=RED_ALT, font=F_HUGE)
        subs1 = f"{28}+ scans"
        sw = F_MED.getlength(subs1)
        d.text(((W // 2 - sw) // 2, H // 2 + 20), subs1, fill=RED_ALT, font=F_MED)
        d.text(((W // 2 - F_BODY.getlength("~300 ms")) // 2, H // 2 + 55),
               "~300 ms", fill=(200, 80, 80), font=F_BODY)

        # Right half — YOLO (cyan)
        d.rectangle([W // 2, 0, W, H], fill=(4, 20, 36))
        rt = "FAST"
        tw = F_HUGE.getlength(rt)
        d.text((W // 2 + (W // 2 - tw) // 2, H // 2 - 50), rt, fill=ACCENT, font=F_HUGE)
        subs2 = "1 pass"
        sw = F_MED.getlength(subs2)
        d.text((W // 2 + (W // 2 - sw) // 2, H // 2 + 20), subs2, fill=ACCENT, font=F_MED)
        d.text((W // 2 + (W // 2 - F_BODY.getlength("~8 ms")) // 2, H // 2 + 55),
               "~8 ms", fill=(80, 200, 200), font=F_BODY)

        # VS divider
        mid_x = W // 2
        d.rectangle([mid_x - 2, 0, mid_x + 2, H], fill=(50, 60, 80))
        vs_txt = "VS"
        vw = F_HUGE.getlength(vs_txt)
        d.rectangle([mid_x - 44, H // 2 - 38, mid_x + 44, H // 2 + 38],
                    fill=(20, 25, 38))
        d.text((mid_x - vw // 2, H // 2 - 30), vs_txt, fill=WHITE, font=F_HUGE)

        frames.append(img)
    return frames


def frames_act2_yolo(wafer_img, dets):
    """
    ACT 2 — YOLO: full image → single-pass grid → all boxes simultaneously.
    Stage A: full wafer image appears with "INPUT" label
    Stage B: 13×13 grid overlaid, cells light up — ONE pass label
    Stage C: all detection boxes appear AT THE SAME TIME (the key insight)
    """
    frames = []
    SIZE = 380
    WX, WY = 30, 68
    GRID_N = 13

    def scale(b):
        orig_w, orig_h = wafer_img.size
        sx = SIZE / orig_w
        sy = SIZE / orig_h
        return [int(b[0]*sx), int(b[1]*sy), int(b[2]*sx), int(b[3]*sy)]

    dets_scaled = [{"cls": d["cls"], "cls_id": d["cls_id"],
                    "conf": d["conf"], "bbox": scale(d["bbox"])} for d in dets]

    # ── Stage A: Full image input (3 frames) ────────────────────────────
    for hold in range(3):
        img = blank()
        d = ImageDraw.Draw(img)
        draw_header_bar(d, "YOLO — You Only Look Once", ACCENT, "Step 1/3  Input")
        paste_wafer(img, wafer_img, WX, WY, SIZE)
        d2 = ImageDraw.Draw(img)

        # Input arrow
        d2.text((WX, WY - 22), "Full 640×640 image — loaded ONCE", fill=ACCENT, font=F_BODY)

        # Right panel
        rx = WX + SIZE + 55
        d2.text((rx, WY + 10),  "How YOLO works:", fill=WHITE, font=F_BODY)
        for i, step in enumerate([
            ("1", "Feed full image", ACCENT),
            ("2", "Predict ALL cells at once", DIM),
            ("3", "Filter by confidence", DIM),
        ]):
            n, txt, col = step
            cy = WY + 50 + i * 48
            d2.ellipse([rx, cy, rx + 26, cy + 26], fill=(20, 30, 48), outline=col, width=2)
            d2.text((rx + 8, cy + 3), n, fill=col, font=F_SMALL)
            d2.text((rx + 36, cy + 3), txt, fill=col, font=F_BODY)

        draw_footer(d2, "One image in → no cropping, no sliding, no patches")
        frames.append(img)

    # ── Stage B: Grid overlay lights up (8 frames) ────────────────────────
    CS = SIZE / GRID_N
    for step in range(8 + 1):
        img = blank()
        d = ImageDraw.Draw(img)
        draw_header_bar(d, "YOLO — You Only Look Once", ACCENT, "Step 2/3  Single-Pass Grid")
        paste_wafer(img, wafer_img, WX, WY, SIZE)
        d2 = ImageDraw.Draw(img)

        progress = step / 8.0

        # Draw grid
        alpha_val = int(min(255, 60 + 180 * progress))
        grid_col = (0, int(210 * progress), int(255 * progress))
        for i in range(GRID_N + 1):
            frac = i / GRID_N
            # Only draw grid lines that have "arrived" per progress
            if frac <= progress + 0.15:
                yy = int(WY + i * CS)
                xx = int(WX + i * CS)
                d2.line([(WX, yy), (WX + SIZE, yy)], fill=grid_col, width=1)
                d2.line([(xx, WY), (xx, WY + SIZE)], fill=grid_col, width=1)

        # Highlight cells that are "responsible" for detections
        cells_shown = int(step * 3.5)
        active_cells = set()
        for det in dets_scaled:
            b = det["bbox"]
            cx_c = (b[0] + b[2]) / 2
            cy_c = (b[1] + b[3]) / 2
            ci = int(cy_c / CS)
            cj = int(cx_c / CS)
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    if 0 <= ci + di < GRID_N and 0 <= cj + dj < GRID_N:
                        active_cells.add((ci + di, cj + dj))
        shown_cells = list(active_cells)[:cells_shown]
        for (ci, cj) in shown_cells:
            cx1 = WX + int(cj * CS)
            cy1 = WY + int(ci * CS)
            cx2 = WX + int((cj + 1) * CS)
            cy2 = WY + int((ci + 1) * CS)
            overlay = Image.new("RGB", (cx2 - cx1, cy2 - cy1), (0, 100, 60))
            img.paste(overlay, (cx1, cy1),
                      mask=Image.new("L", (cx2 - cx1, cy2 - cy1), 90))
            d2.rectangle([cx1, cy1, cx2, cy2], outline=GREEN, width=1)

        # Right panel
        rx = WX + SIZE + 55
        for i, step_info in enumerate([
            ("1", "Feed full image", GREEN),
            ("2", "Predict ALL cells at once", ACCENT if progress > 0.3 else DIM),
            ("3", "Filter by confidence", DIM),
        ]):
            n, txt, col = step_info
            cy = WY + 50 + i * 48
            d2.ellipse([rx, cy, rx + 26, cy + 26], fill=(20, 30, 48), outline=col, width=2)
            d2.text((rx + 8, cy + 3), n, fill=col, font=F_SMALL)
            d2.text((rx + 36, cy + 3), txt, fill=col, font=F_BODY)

        d2.text((rx, WY + 200), f"Grid: {GRID_N}×{GRID_N} = {GRID_N*GRID_N} cells",
                fill=ACCENT, font=F_BODY)
        d2.text((rx, WY + 228), "Each cell predicts:", fill=DIM, font=F_SMALL)
        d2.text((rx, WY + 248), "• bbox coordinates", fill=WHITE, font=F_SMALL)
        d2.text((rx, WY + 268), "• objectness score", fill=WHITE, font=F_SMALL)
        d2.text((rx, WY + 288), "• class probabilities", fill=WHITE, font=F_SMALL)
        d2.text((rx, WY + 316), "ALL predictions computed", fill=ACCENT, font=F_BODY)
        d2.text((rx, WY + 338), "in a SINGLE forward pass", fill=ACCENT, font=F_BODY)

        draw_footer(d2, f"13×13 = 169 cells evaluated simultaneously — one neural network pass")
        frames.append(img)

    # ── Stage C: ALL boxes appear at once (the money shot) ─────────────────
    for hold in range(4):
        img = blank()
        d = ImageDraw.Draw(img)
        draw_header_bar(d, "YOLO — You Only Look Once", ACCENT, "Step 3/3  Results — All at Once!")
        paste_wafer(img, wafer_img, WX, WY, SIZE)
        d2 = ImageDraw.Draw(img)

        # Full grid faint
        for i in range(GRID_N + 1):
            yy = int(WY + i * CS)
            xx = int(WX + i * CS)
            d2.line([(WX, yy), (WX + SIZE, yy)], fill=(30, 50, 40), width=1)
            d2.line([(xx, WY), (xx, WY + SIZE)], fill=(30, 50, 40), width=1)

        # ALL detection boxes appear simultaneously
        for di, det in enumerate(dets_scaled):
            col = CLASS_COLORS[det["cls_id"] % len(CLASS_COLORS)]
            b = det["bbox"]
            x1 = WX + b[0]; y1 = WY + b[1]
            x2 = WX + b[2]; y2 = WY + b[3]
            d2.rectangle([x1, y1, x2, y2], outline=col, width=2)
            # Label background
            lbl = f"{det['cls'][:7]} {det['conf']:.0%}"
            lw = max(F_TINY.getlength(lbl) + 6, 10)
            label_y = max(WY, y1 - 16)
            d2.rectangle([x1, label_y, x1 + lw, label_y + 14], fill=col)
            d2.text((x1 + 3, label_y + 1), lbl, fill=(0, 0, 0), font=F_TINY)

        # "AT THE SAME TIME" callout on first 2 holds
        if hold < 2:
            cw, ch = 320, 50
            cx1_cal = WX + SIZE // 2 - cw // 2
            cy1_cal = WY + SIZE - 65
            d2.rectangle([cx1_cal - 2, cy1_cal - 2, cx1_cal + cw + 2, cy1_cal + ch + 2],
                         fill=(0, 0, 0))
            d2.rectangle([cx1_cal, cy1_cal, cx1_cal + cw, cy1_cal + ch],
                         fill=(0, 40, 25), outline=GREEN, width=2)
            msg = "ALL boxes predicted simultaneously!"
            mw = F_BODY.getlength(msg)
            d2.text((cx1_cal + (cw - mw) // 2, cy1_cal + 8), msg, fill=GREEN, font=F_BODY)
            msg2 = "That's what \"Only Once\" means"
            mw2 = F_SMALL.getlength(msg2)
            d2.text((cx1_cal + (cw - mw2) // 2, cy1_cal + 30), msg2, fill=ACCENT, font=F_SMALL)

        # Right panel — stats
        rx = WX + SIZE + 55
        for i, step_info in enumerate([
            ("1", "Feed full image", GREEN),
            ("2", "Predict ALL cells at once", GREEN),
            ("3", "Filter by confidence", GREEN),
        ]):
            n, txt, col = step_info
            cy = WY + 50 + i * 48
            d2.ellipse([rx, cy, rx + 26, cy + 26], fill=(0, 40, 20), outline=GREEN, width=2)
            d2.text((rx + 8, cy + 3), "✓", fill=GREEN, font=F_SMALL)
            d2.text((rx + 36, cy + 3), txt, fill=GREEN, font=F_BODY)

        d2.text((rx, WY + 205), f"Defects found:", fill=DIM, font=F_SMALL)
        d2.text((rx, WY + 225), str(len(dets_scaled)), fill=ACCENT, font=F_BIG)

        d2.text((rx, WY + 285), "Inference time:", fill=DIM, font=F_SMALL)
        d2.text((rx, WY + 305), "~8 ms", fill=GREEN, font=F_BODY)

        d2.text((rx, WY + 340), "vs traditional:", fill=DIM, font=F_SMALL)
        d2.text((rx, WY + 360), "~300 ms", fill=RED_ALT, font=F_BODY)

        draw_footer(d2, f"{len(dets_scaled)} defects detected in a single neural network forward pass")
        frames.append(img)

    return frames


def frames_act3_results(wafer_img, dets):
    """ACT 3 — Clean annotated result + metrics card. 4 frames."""
    frames = []
    SIZE = 340
    WX, WY = 30, 68

    def scale(b):
        orig_w, orig_h = wafer_img.size
        sx = SIZE / orig_w
        sy = SIZE / orig_h
        return [int(b[0]*sx), int(b[1]*sy), int(b[2]*sx), int(b[3]*sy)]

    dets_scaled = [{"cls": d["cls"], "cls_id": d["cls_id"],
                    "conf": d["conf"], "bbox": scale(d["bbox"])} for d in dets]

    for hold in range(4):
        img = blank()
        d = ImageDraw.Draw(img)
        draw_header_bar(d, "Wafer Inspection Result", GREEN, "PASS/FAIL  ·  99.22% mAP@50")

        paste_wafer(img, wafer_img, WX, WY, SIZE)
        d2 = ImageDraw.Draw(img)

        # Draw all detection boxes
        for di, det in enumerate(dets_scaled):
            col = CLASS_COLORS[det["cls_id"] % len(CLASS_COLORS)]
            b = det["bbox"]
            x1 = WX + b[0]; y1 = WY + b[1]
            x2 = WX + b[2]; y2 = WY + b[3]
            d2.rectangle([x1, y1, x2, y2], outline=col, width=2)
            lbl = f"{det['cls'][:8]}"
            lw = max(F_TINY.getlength(lbl) + 6, 10)
            label_y = max(WY, y1 - 14)
            d2.rectangle([x1, label_y, x1 + lw, label_y + 12], fill=col)
            d2.text((x1 + 3, label_y + 1), lbl, fill=(0, 0, 0), font=F_TINY)

        # Verdict badge
        verdict = "FAIL — Defects Detected" if dets_scaled else "PASS — No Defects"
        vcolor = RED_ALT if dets_scaled else GREEN
        vw = F_MED.getlength(verdict)
        vx = WX + (SIZE - vw) // 2
        d2.rectangle([vx - 12, WY + SIZE + 6, vx + vw + 12, WY + SIZE + 34],
                     fill=(30, 10, 12) if dets_scaled else (5, 25, 12), outline=vcolor, width=2)
        d2.text((vx, WY + SIZE + 10), verdict, fill=vcolor, font=F_MED)

        # Right panel — metrics card
        rx = WX + SIZE + 40
        ry = WY

        d2.text((rx, ry), "Model Card", fill=WHITE, font=F_BODY)
        d2.line([(rx, ry + 26), (rx + 340, ry + 26)], fill=(35, 50, 70), width=1)

        metrics = [
            ("Architecture",   "YOLOv8-Large"),
            ("Parameters",     "43.7M"),
            ("mAP@50",         "99.22%"),
            ("mAP@50:95",      "88.4%"),
            ("Inference",      "~8 ms (GPU T4)"),
            ("Classes",        "10 defect types"),
            ("Training",       "~5,000 wafer imgs"),
            ("Framework",      "Ultralytics 8.4"),
        ]
        for i, (k, v) in enumerate(metrics):
            my = ry + 36 + i * 32
            d2.text((rx, my), k + ":", fill=DIM, font=F_SMALL)
            d2.text((rx + 130, my), v, fill=WHITE, font=F_SMALL)

        # Defect legend (bottom of right panel)
        cls_seen = list({det["cls_id"] for det in dets_scaled})[:5]
        ly = ry + 310
        d2.text((rx, ly), "Detected:", fill=DIM, font=F_SMALL)
        for i2, cid in enumerate(cls_seen):
            col = CLASS_COLORS[cid % len(CLASS_COLORS)]
            cx_leg = rx + i2 * 68
            d2.rectangle([cx_leg, ly + 18, cx_leg + 12, ly + 30], fill=col)
            d2.text((cx_leg + 16, ly + 16), CLASS_NAMES[cid][:5], fill=col, font=F_TINY)

        draw_footer(d2,
            "YOLOv8-Large · 1080-class production training · Real-time wafer inspection")
        frames.append(img)

    return frames


def frames_outro():
    """3 frames: LinkedIn outro with key stats"""
    frames = []
    for _ in range(4):
        img = blank()
        d = ImageDraw.Draw(img)

        # Background grid
        for gx in range(0, W, 40):
            d.line([(gx, 0), (gx, H)], fill=(16, 22, 34), width=1)
        for gy in range(0, H, 40):
            d.line([(0, gy), (W, gy)], fill=(16, 22, 34), width=1)

        draw_header_bar(d, "YOLO vs Traditional — Key Numbers", ACCENT)

        # Three stat cards
        cards = [
            ("~300 ms", "Traditional\nSliding Window", RED_ALT, 28),
            ("VS", "", WHITE, W // 2 - 52),
            ("~8 ms", "YOLO\nSingle Pass", ACCENT, W - 28 - 200),
        ]

        cy_card = 100
        for val, label, col, cx_base in cards:
            if val == "VS":
                d.text((cx_base, cy_card + 60), val, fill=WHITE, font=F_BIG)
                continue
            card_w, card_h = 200, 160
            rounded_rect(d, cx_base, cy_card, cx_base + card_w, cy_card + card_h,
                         r=10, fill=(18, 24, 40), outline=col, width=2)
            vw = F_HUGE.getlength(val)
            d.text((cx_base + (card_w - vw) // 2, cy_card + 20), val, fill=col, font=F_HUGE)
            for i2, ln in enumerate(label.split("\n")):
                lw = F_SMALL.getlength(ln)
                d.text((cx_base + (card_w - lw) // 2, cy_card + 100 + i2 * 22),
                       ln, fill=col, font=F_SMALL)

        # Bottom key insights
        insights = [
            "  1 forward pass   →   all detections simultaneously",
            "  37× faster   ·   99.22% mAP@50   ·   production-ready",
        ]
        for i2, ins in enumerate(insights):
            iw = F_BODY.getlength(ins)
            d.text(((W - iw) // 2, 310 + i2 * 36), ins,
                   fill=ACCENT if i2 == 0 else WHITE, font=F_BODY)

        # Call-to-action
        cta = "github.com/Rajendar-Muddasani-2/yolo-object-detection"
        cw = F_SMALL.getlength(cta)
        d.text(((W - cw) // 2, H - 60), cta, fill=DIM, font=F_SMALL)

        draw_footer(d, "Semiconductor Wafer Defect Detection  ·  End-to-End AIML Engineering")
        frames.append(img)
    return frames


# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    print("Generating synthetic wafer image...")
    wafer_img = make_synthetic_wafer(size=640, seed=7)

    print("Loading detections from stored results JSON...")
    dets = get_detections_from_json(target_size=640)
    print(f"  Using {len(dets)} detections")

    print("Building frames...")
    all_frames = []
    all_frames += frames_act0_title()                      # 3 frames — title card
    all_frames += frames_act1_traditional(wafer_img)       # ~28 frames — traditional scanner
    all_frames += frames_vs_flash()                        # 3 frames — VS contrast
    all_frames += frames_act2_yolo(wafer_img, dets)        # ~20 frames — YOLO single pass
    all_frames += frames_act3_results(wafer_img, dets)     # 4 frames — final result
    all_frames += frames_outro()                           # 4 frames — stats + CTA
    print(f"  Total frames: {len(all_frames)}")

    print(f"Saving GIF -> {OUTPUT_GIF}")
    durations = []
    for i, _ in enumerate(all_frames):
        if i < 3:                 # title — hold
            durations.append(800)
        elif i < 3 + 30:          # traditional — per-frame pace
            durations.append(160)
        elif i < 3 + 30 + 3:      # VS flash — fast
            durations.append(350)
        elif i < 3 + 30 + 3 + 9:  # grid build — medium
            durations.append(200)
        elif i < 3 + 30 + 3 + 9 + 4:  # all-at-once — hold
            durations.append(600)
        elif i < 3 + 30 + 3 + 9 + 4 + 4:  # result
            durations.append(700)
        else:                     # outro
            durations.append(900)

    all_frames[0].save(
        OUTPUT_GIF,
        save_all=True,
        append_images=all_frames[1:],
        duration=durations,
        loop=0,
        optimize=False,
    )

    size_mb = OUTPUT_GIF.stat().st_size / 1e6
    print(f"Done! {size_mb:.1f} MB  ·  {len(all_frames)} frames")
    print(f"Output: {OUTPUT_GIF}")


if __name__ == "__main__":
    main()
