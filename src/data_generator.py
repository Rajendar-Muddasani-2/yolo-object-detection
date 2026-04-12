"""
Enhanced synthetic wafer defect image generator for YOLOv8 training.

Generates 20K+ high-quality wafer defect images with YOLO-format bounding box annotations.
10 defect classes with multi-scale defects, realistic wafer backgrounds, and augmentation variety.

Defect classes (10):
  0: scratch        - line-shaped scratches across wafer surface
  1: particle       - small circular contamination spots
  2: edge_chip      - chips/breaks at the wafer edge
  3: void           - dark voids in the die pattern
  4: pattern_shift  - misaligned lithography pattern area
  5: bridge         - conductive bridges between features
  6: missing_bond   - missing wire bond pads
  7: crack          - fracture lines radiating from stress points
  8: contamination  - large irregular contamination zones
  9: delamination   - film peeling/delamination regions

Output format: YOLO (class_id x_center y_center width height) normalized 0-1
"""

import json
import logging
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance

logger = logging.getLogger(__name__)

CLASSES = [
    "scratch", "particle", "edge_chip", "void", "pattern_shift",
    "bridge", "missing_bond", "crack", "contamination", "delamination",
]
NUM_CLASSES = len(CLASSES)
IMG_SIZE = 640


def _draw_wafer_background(
    img: Image.Image, draw: ImageDraw.Draw, rng: np.random.Generator
) -> None:
    """Draw a realistic circular wafer with die grid, flat zone, and texture."""
    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
    radius = IMG_SIZE // 2 - rng.integers(15, 30)
    base_gray = rng.integers(170, 200)
    wafer_color = (base_gray, base_gray, base_gray + rng.integers(0, 15))

    draw.ellipse(
        [cx - radius, cy - radius, cx + radius, cy + radius],
        fill=wafer_color, outline=(100, 100, 100), width=2,
    )

    # Flat zone
    flat_y = cy + radius - 10
    draw.rectangle([cx - radius // 3, flat_y, cx + radius // 3, flat_y + 12],
                   fill=(120, 120, 130))

    # Die grid
    die_w = rng.integers(16, 28)
    die_h = rng.integers(16, 28)
    gc_base = base_gray - rng.integers(10, 25)
    for x in range(cx - radius, cx + radius, die_w):
        for y in range(cy - radius, cy + radius, die_h):
            if (x - cx) ** 2 + (y - cy) ** 2 < (radius - 5) ** 2:
                gc = max(0, min(255, gc_base + rng.integers(-5, 6)))
                draw.rectangle([x, y, x + die_w - 2, y + die_h - 2],
                               outline=(gc, gc, gc + 3))


def _add_noise(img: Image.Image, rng: np.random.Generator, intensity: float = 8.0) -> Image.Image:
    arr = np.array(img, dtype=np.float32)
    noise = rng.normal(0, intensity, arr.shape).astype(np.float32)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


# --- Defect drawing functions ---

def _add_scratch(draw: ImageDraw.Draw, rng: np.random.Generator) -> Tuple[int, float, float, float, float]:
    x1 = rng.integers(40, IMG_SIZE - 40)
    y1 = rng.integers(40, IMG_SIZE - 40)
    angle = rng.uniform(-0.8, 0.8)
    length = rng.integers(60, 300)
    width = rng.integers(1, 6)
    x2 = int(x1 + length * math.cos(angle))
    y2 = int(y1 + length * math.sin(angle))
    segments = rng.integers(2, 5)
    points = [(x1, y1)]
    for s in range(1, segments):
        frac = s / segments
        points.append((int(x1 + (x2 - x1) * frac + rng.integers(-8, 9)),
                       int(y1 + (y2 - y1) * frac + rng.integers(-8, 9))))
    points.append((x2, y2))
    cv = rng.integers(30, 70)
    draw.line(points, fill=(cv, cv, cv + 10), width=width)
    bx1, bx2 = min(p[0] for p in points), max(p[0] for p in points)
    by1, by2 = min(p[1] for p in points), max(p[1] for p in points)
    w = max(bx2 - bx1, 8) + width * 2
    h = max(by2 - by1, 8) + width * 2
    return 0, np.clip((bx1 + bx2) / 2 / IMG_SIZE, 0, 1), \
        np.clip((by1 + by2) / 2 / IMG_SIZE, 0, 1), min(w / IMG_SIZE, 1.0), min(h / IMG_SIZE, 1.0)


def _add_particle(draw: ImageDraw.Draw, rng: np.random.Generator) -> Tuple[int, float, float, float, float]:
    cx, cy = rng.integers(80, IMG_SIZE - 80), rng.integers(80, IMG_SIZE - 80)
    r = rng.integers(4, 25)
    glow_r = r + rng.integers(2, 6)
    gv = rng.integers(100, 140)
    draw.ellipse([cx - glow_r, cy - glow_r, cx + glow_r, cy + glow_r], fill=(gv, gv, gv))
    cv = rng.integers(20, 60)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(cv, cv, cv + 5))
    bs = max(glow_r * 2, r * 2 + 6)
    return 1, cx / IMG_SIZE, cy / IMG_SIZE, bs / IMG_SIZE, bs / IMG_SIZE


def _add_edge_chip(draw: ImageDraw.Draw, rng: np.random.Generator) -> Tuple[int, float, float, float, float]:
    angle = rng.uniform(0, 2 * math.pi)
    radius = IMG_SIZE // 2 - rng.integers(20, 40)
    cx = int(IMG_SIZE // 2 + radius * math.cos(angle))
    cy = int(IMG_SIZE // 2 + radius * math.sin(angle))
    size = rng.integers(12, 45)
    n_pts = rng.integers(4, 8)
    points = [(int(cx + size * rng.uniform(0.5, 1.0) * math.cos(2 * math.pi * i / n_pts + rng.uniform(-0.3, 0.3))),
               int(cy + size * rng.uniform(0.5, 1.0) * math.sin(2 * math.pi * i / n_pts + rng.uniform(-0.3, 0.3))))
              for i in range(n_pts)]
    draw.polygon(points, fill=(rng.integers(50, 80),) * 3)
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return 2, np.clip((min(xs) + max(xs)) / 2 / IMG_SIZE, 0, 1), \
        np.clip((min(ys) + max(ys)) / 2 / IMG_SIZE, 0, 1), \
        min((max(xs) - min(xs)) / IMG_SIZE, 1.0), min((max(ys) - min(ys)) / IMG_SIZE, 1.0)


def _add_void(draw: ImageDraw.Draw, rng: np.random.Generator) -> Tuple[int, float, float, float, float]:
    cx, cy = rng.integers(120, IMG_SIZE - 120), rng.integers(120, IMG_SIZE - 120)
    w, h = rng.integers(15, 55), rng.integers(15, 55)
    vv = rng.integers(15, 40)
    draw.ellipse([cx - w, cy - h, cx + w, cy + h], fill=(vv, vv, vv))
    draw.ellipse([cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2], fill=(vv - 10, vv - 10, vv - 5))
    return 3, cx / IMG_SIZE, cy / IMG_SIZE, 2 * w / IMG_SIZE, 2 * h / IMG_SIZE


def _add_pattern_shift(draw: ImageDraw.Draw, rng: np.random.Generator) -> Tuple[int, float, float, float, float]:
    cx, cy = rng.integers(120, IMG_SIZE - 120), rng.integers(120, IMG_SIZE - 120)
    w, h = rng.integers(35, 90), rng.integers(35, 90)
    sx, sy = rng.integers(3, 8), rng.integers(3, 8)
    oc = (rng.integers(180, 220), rng.integers(60, 100), rng.integers(60, 100))
    draw.rectangle([cx - w, cy - h, cx + w, cy + h], outline=oc, width=2)
    for dx in range(-w, w, 16):
        for dy in range(-h, h, 16):
            draw.rectangle([cx + dx + sx, cy + dy + sy, cx + dx + 14 + sx, cy + dy + 14 + sy], outline=oc)
    return 4, cx / IMG_SIZE, cy / IMG_SIZE, 2 * w / IMG_SIZE, 2 * h / IMG_SIZE


def _add_bridge(draw: ImageDraw.Draw, rng: np.random.Generator) -> Tuple[int, float, float, float, float]:
    cx, cy = rng.integers(100, IMG_SIZE - 100), rng.integers(100, IMG_SIZE - 100)
    horiz = rng.random() > 0.5
    color = (rng.integers(200, 240), rng.integers(180, 220), rng.integers(100, 140))
    if horiz:
        l, h = rng.integers(20, 60), rng.integers(3, 8)
        draw.rectangle([cx - l // 2, cy - h // 2, cx + l // 2, cy + h // 2], fill=color)
        return 5, cx / IMG_SIZE, cy / IMG_SIZE, l / IMG_SIZE, max(h, 6) / IMG_SIZE
    else:
        h2, w2 = rng.integers(20, 60), rng.integers(3, 8)
        draw.rectangle([cx - w2 // 2, cy - h2 // 2, cx + w2 // 2, cy + h2 // 2], fill=color)
        return 5, cx / IMG_SIZE, cy / IMG_SIZE, max(w2, 6) / IMG_SIZE, h2 / IMG_SIZE


def _add_missing_bond(draw: ImageDraw.Draw, rng: np.random.Generator) -> Tuple[int, float, float, float, float]:
    cx, cy = rng.integers(80, IMG_SIZE - 80), rng.integers(80, IMG_SIZE - 80)
    r = rng.integers(8, 20)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(220, 200, 100), width=2)
    draw.line([(cx - r + 2, cy - r + 2), (cx + r - 2, cy + r - 2)], fill=(220, 80, 80), width=2)
    draw.line([(cx + r - 2, cy - r + 2), (cx - r + 2, cy + r - 2)], fill=(220, 80, 80), width=2)
    bs = r * 2 + 4
    return 6, cx / IMG_SIZE, cy / IMG_SIZE, bs / IMG_SIZE, bs / IMG_SIZE


def _add_crack(draw: ImageDraw.Draw, rng: np.random.Generator) -> Tuple[int, float, float, float, float]:
    sx, sy = rng.integers(100, IMG_SIZE - 100), rng.integers(100, IMG_SIZE - 100)
    points = [(sx, sy)]
    branches: List[List[Tuple[int, int]]] = [[(sx, sy)]]
    for _ in range(rng.integers(5, 15)):
        last = branches[0][-1]
        new_pt = (int(last[0] + rng.integers(-20, 21)), int(last[1] + rng.integers(-20, 21)))
        branches[0].append(new_pt)
        points.append(new_pt)
        if rng.random() < 0.3 and len(branches) < 4:
            branch = [new_pt]
            for _ in range(rng.integers(2, 5)):
                bpt = (int(branch[-1][0] + rng.integers(-15, 16)), int(branch[-1][1] + rng.integers(-15, 16)))
                branch.append(bpt)
                points.append(bpt)
            branches.append(branch)
    cv = rng.integers(30, 60)
    for br in branches:
        if len(br) >= 2:
            draw.line(br, fill=(cv, cv, cv), width=rng.integers(1, 3))
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    w, h = max(max(xs) - min(xs), 10), max(max(ys) - min(ys), 10)
    return 7, np.clip((min(xs) + max(xs)) / 2 / IMG_SIZE, 0, 1), \
        np.clip((min(ys) + max(ys)) / 2 / IMG_SIZE, 0, 1), min(w / IMG_SIZE, 1.0), min(h / IMG_SIZE, 1.0)


def _add_contamination(draw: ImageDraw.Draw, rng: np.random.Generator) -> Tuple[int, float, float, float, float]:
    cx, cy = rng.integers(120, IMG_SIZE - 120), rng.integers(120, IMG_SIZE - 120)
    ax, ay = [], []
    for _ in range(rng.integers(3, 8)):
        ox, oy = cx + rng.integers(-30, 31), cy + rng.integers(-30, 31)
        rx, ry = rng.integers(10, 35), rng.integers(10, 35)
        cv = rng.integers(80, 120)
        draw.ellipse([ox - rx, oy - ry, ox + rx, oy + ry], fill=(cv, cv - 10, cv + 20))
        ax.extend([ox - rx, ox + rx])
        ay.extend([oy - ry, oy + ry])
    w, h = max(ax) - min(ax), max(ay) - min(ay)
    return 8, np.clip((min(ax) + max(ax)) / 2 / IMG_SIZE, 0, 1), \
        np.clip((min(ay) + max(ay)) / 2 / IMG_SIZE, 0, 1), min(w / IMG_SIZE, 1.0), min(h / IMG_SIZE, 1.0)


def _add_delamination(draw: ImageDraw.Draw, rng: np.random.Generator) -> Tuple[int, float, float, float, float]:
    cx, cy = rng.integers(100, IMG_SIZE - 100), rng.integers(100, IMG_SIZE - 100)
    w, h = rng.integers(25, 70), rng.integers(25, 70)
    n_pts = rng.integers(6, 12)
    points = [(int(cx + w * rng.uniform(0.7, 1.0) * math.cos(2 * math.pi * i / n_pts)),
               int(cy + h * rng.uniform(0.7, 1.0) * math.sin(2 * math.pi * i / n_pts)))
              for i in range(n_pts)]
    draw.polygon(points, fill=(rng.integers(200, 230), rng.integers(190, 220), rng.integers(180, 210)))
    draw.polygon(points, outline=(rng.integers(80, 120), rng.integers(70, 110), rng.integers(60, 100)), width=2)
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return 9, np.clip((min(xs) + max(xs)) / 2 / IMG_SIZE, 0, 1), \
        np.clip((min(ys) + max(ys)) / 2 / IMG_SIZE, 0, 1), \
        min((max(xs) - min(xs)) / IMG_SIZE, 1.0), min((max(ys) - min(ys)) / IMG_SIZE, 1.0)


DEFECT_FUNCS = [
    _add_scratch, _add_particle, _add_edge_chip, _add_void, _add_pattern_shift,
    _add_bridge, _add_missing_bond, _add_crack, _add_contamination, _add_delamination,
]


def _generate_single_image(
    idx: int, seed: int, max_defects: int,
) -> Tuple[Image.Image, List[str]]:
    """Generate a single wafer defect image with YOLO labels."""
    rng = np.random.default_rng(seed + idx)
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (200, 200, 210))
    draw = ImageDraw.Draw(img)
    _draw_wafer_background(img, draw, rng)

    n_defects = rng.integers(1, max_defects + 1)
    weights = np.ones(NUM_CLASSES) / NUM_CLASSES
    labels: List[str] = []
    for _ in range(n_defects):
        cls_idx = rng.choice(NUM_CLASSES, p=weights)
        cls_id, cx, cy, w, h = DEFECT_FUNCS[cls_idx](draw, rng)
        w, h = max(w, 0.005), max(h, 0.005)
        cx = np.clip(cx, w / 2, 1 - w / 2)
        cy = np.clip(cy, h / 2, 1 - h / 2)
        labels.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    img = _add_noise(img, rng, intensity=rng.uniform(3.0, 12.0))
    if rng.random() > 0.5:
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.85, 1.15))
    if rng.random() > 0.5:
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.85, 1.15))
    return img, labels


def _generate_batch(args: Tuple) -> List[Tuple[int, Image.Image, List[str]]]:
    start_idx, count, seed, max_defects = args
    return [(start_idx + i, *_generate_single_image(start_idx + i, seed, max_defects))
            for i in range(count)]


def generate_dataset(
    output_dir: str = "data/wafer_defects",
    n_images: int = 20000,
    max_defects_per_image: int = 6,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    n_workers: int = 4,
) -> Dict[str, int]:
    """Generate a YOLO-format dataset of synthetic wafer defect images."""
    out = Path(output_dir)
    n_train = int(n_images * train_ratio)
    n_val = int(n_images * val_ratio)

    for split in ("train", "val", "test"):
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)

    class_counts = {cls: 0 for cls in CLASSES}
    split_counts = {"train": 0, "val": 0, "test": 0}

    logger.info("Generating %d images (%d train, %d val, %d test) with %d workers...",
                n_images, n_train, n_val, n_images - n_train - n_val, n_workers)

    batch_size = max(1, n_images // (n_workers * 4))
    batches = [(s, min(batch_size, n_images - s), seed, max_defects_per_image)
               for s in range(0, n_images, batch_size)]

    generated = 0

    def _save_results(results):
        nonlocal generated
        for idx, img, labels in results:
            split = "train" if idx < n_train else ("val" if idx < n_train + n_val else "test")
            fname = f"wafer_{idx:06d}"
            img.save(out / split / "images" / f"{fname}.jpg", quality=95)
            (out / split / "labels" / f"{fname}.txt").write_text("\n".join(labels) + "\n")
            split_counts[split] += 1
            for lbl in labels:
                class_counts[CLASSES[int(lbl.split()[0])]] += 1
            generated += 1
            if generated % 2000 == 0:
                logger.info("  Generated %d/%d images...", generated, n_images)

    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_generate_batch, b): b for b in batches}
            for future in as_completed(futures):
                _save_results(future.result())
    else:
        for batch_args in batches:
            _save_results(_generate_batch(batch_args))

    # Dataset YAML (Ultralytics format)
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: train/images\nval: val/images\ntest: test/images\n"
        f"nc: {NUM_CLASSES}\nnames: {CLASSES}\n"
    )

    metadata = {
        "total_images": n_images, "splits": split_counts,
        "class_distribution": class_counts, "classes": CLASSES,
        "num_classes": NUM_CLASSES, "image_size": IMG_SIZE,
        "max_defects_per_image": max_defects_per_image, "seed": seed,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    logger.info("Dataset generated: %s", split_counts)
    logger.info("Class distribution: %s", class_counts)
    return {"splits": split_counts, "class_distribution": class_counts}


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Generate synthetic wafer defect dataset")
    parser.add_argument("--output", default="data/wafer_defects", help="Output directory")
    parser.add_argument("--n-images", type=int, default=20000, help="Number of images")
    parser.add_argument("--max-defects", type=int, default=6, help="Max defects per image")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    args = parser.parse_args()
    stats = generate_dataset(
        output_dir=args.output, n_images=args.n_images,
        max_defects_per_image=args.max_defects, seed=args.seed, n_workers=args.workers,
    )
    print(f"\nDataset stats: {json.dumps(stats, indent=2)}")
