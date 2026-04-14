"""
Generate photorealistic-looking wafer defect images for showcase/demo.

These are synthetic but look much more realistic than the training data:
- Silicon wafer with subtle blue/purple tint and die grid shimmer
- Concentric zone rings like real wafers
- Natural-looking defects: scratches with depth, particles with scatter,
  cracks with branching, contamination with gradients
- Higher resolution appearance effects (shadows, specular highlights)
"""

import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from pathlib import Path


IMG_SIZE = 640
CLASSES = [
    "scratch", "particle", "edge_chip", "void", "pattern_shift",
    "bridge", "missing_bond", "crack", "contamination", "delamination",
]

# Color palette for the demo
WAFER_PALETTES = [
    {"base": (190, 195, 210), "tint": (0, -5, 15), "die": (175, 180, 200)},      # blue-silver
    {"base": (200, 195, 185), "tint": (10, -5, -15), "die": (185, 180, 170)},     # warm bronze
    {"base": (185, 200, 195), "tint": (-5, 10, 5), "die": (170, 190, 182)},       # cool green
    {"base": (195, 185, 205), "tint": (5, -10, 15), "die": (180, 170, 195)},      # purple hint
    {"base": (210, 200, 190), "tint": (15, 5, -10), "die": (195, 185, 178)},      # gold
    {"base": (185, 190, 210), "tint": (-5, 0, 20), "die": (172, 178, 200)},       # deep blue
    {"base": (200, 200, 200), "tint": (0, 0, 0), "die": (185, 185, 190)},         # classic silver
]


def _make_wafer_base(rng: np.random.Generator, palette: dict) -> Image.Image:
    """Create a realistic silicon wafer base with zone rings and die grid."""
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (25, 25, 30))
    arr = np.array(img, dtype=np.float32)

    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
    radius = IMG_SIZE // 2 - 18

    # Create wafer mask
    Y, X = np.mgrid[:IMG_SIZE, :IMG_SIZE]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    wafer_mask = dist < radius

    # Base color with radial gradient (center brighter)
    base = np.array(palette["base"], dtype=np.float32)
    tint = np.array(palette["tint"], dtype=np.float32)
    radial_factor = 1.0 - 0.12 * (dist / radius)
    radial_factor = np.clip(radial_factor, 0, 1)

    for c in range(3):
        arr[:, :, c] = np.where(wafer_mask, base[c] * radial_factor + tint[c], arr[:, :, c])

    # Concentric zone rings (like real wafer processing zones)
    for ring_r in range(50, radius, rng.integers(60, 100)):
        ring_mask = (np.abs(dist - ring_r) < 2) & wafer_mask
        ring_brightness = rng.uniform(-6, 6)
        for c in range(3):
            arr[:, :, c] = np.where(ring_mask, arr[:, :, c] + ring_brightness, arr[:, :, c])

    # Angular shimmer (anisotropic reflection from die grid / crystal orientation)
    angle = np.arctan2(Y - cy, X - cx)
    shimmer = 4 * np.sin(angle * 8 + rng.uniform(0, 2 * math.pi))
    for c in range(3):
        arr[:, :, c] = np.where(wafer_mask, arr[:, :, c] + shimmer, arr[:, :, c])

    # Die grid pattern
    die_size = rng.integers(18, 28)
    die_c = np.array(palette["die"], dtype=np.float32)
    for x in range(cx - radius, cx + radius, die_size):
        for y in range(cy - radius, cy + radius, die_size):
            if (x - cx) ** 2 + (y - cy) ** 2 < (radius - 12) ** 2:
                x1, y1 = max(x, 0), max(y, 0)
                x2, y2 = min(x + die_size - 1, IMG_SIZE - 1), min(y + die_size - 1, IMG_SIZE - 1)
                # Just the grid lines
                arr[y1, x1:x2, :] = np.clip(die_c + rng.normal(0, 2, 3), 0, 255)
                arr[y1:y2, x1, :] = np.clip(die_c + rng.normal(0, 2, 3), 0, 255)

    # Flat zone at bottom
    flat_y_start = cy + radius - 8
    flat_mask = (Y >= flat_y_start) & (Y < flat_y_start + 6) & (np.abs(X - cx) < radius // 3) & wafer_mask
    for c in range(3):
        arr[:, :, c] = np.where(flat_mask, arr[:, :, c] - 30, arr[:, :, c])

    # Slight Gaussian noise for realism
    noise = rng.normal(0, 3.5, arr.shape).astype(np.float32)
    arr = np.where(wafer_mask[:, :, None], arr + noise, arr)

    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    # Subtle wafer edge bevel / shadow
    draw = ImageDraw.Draw(img)
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius],
                 outline=(90, 90, 95), width=2)
    draw.ellipse([cx - radius + 2, cy - radius + 2, cx + radius - 2, cy + radius - 2],
                 outline=(140, 140, 148), width=1)

    return img, radius


def _add_realistic_scratch(img: Image.Image, rng: np.random.Generator) -> list:
    """Natural scratch: deep line with halo, slight wobble."""
    draw = ImageDraw.Draw(img)
    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
    radius = IMG_SIZE // 2 - 30

    angle = rng.uniform(-0.6, 0.6)
    length = rng.integers(100, 350)
    # Start within wafer
    sx = rng.integers(cx - radius // 2, cx + radius // 2)
    sy = rng.integers(cy - radius // 2, cy + radius // 2)
    ex = int(sx + length * math.cos(angle))
    ey = int(sy + length * math.sin(angle))

    # Wobble points
    n_pts = rng.integers(6, 14)
    points = []
    for i in range(n_pts + 1):
        frac = i / n_pts
        px = int(sx + (ex - sx) * frac + rng.integers(-4, 5))
        py = int(sy + (ey - sy) * frac + rng.integers(-4, 5))
        points.append((px, py))

    # Halo (wider, lighter)
    halo_w = rng.integers(5, 10)
    halo_c = rng.integers(140, 160)
    draw.line(points, fill=(halo_c, halo_c, halo_c + 5), width=halo_w)
    # Core scratch (dark, thin)
    core_w = rng.integers(1, 4)
    core_c = rng.integers(60, 90)
    draw.line(points, fill=(core_c, core_c - 5, core_c), width=core_w)

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    pad = halo_w
    return [{"class": "scratch", "bbox": [min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad]}]


def _add_realistic_particle(img: Image.Image, rng: np.random.Generator) -> list:
    """Bright/dark irregular particle with scatter ring."""
    arr = np.array(img)
    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
    radius_wafer = IMG_SIZE // 2 - 40

    px = rng.integers(cx - radius_wafer // 2, cx + radius_wafer // 2)
    py = rng.integers(cy - radius_wafer // 2, cy + radius_wafer // 2)
    r = rng.integers(6, 22)

    Y, X = np.mgrid[:IMG_SIZE, :IMG_SIZE]
    pdist = np.sqrt((X - px) ** 2 + (Y - py) ** 2)

    # Scatter halo
    halo_mask = (pdist < r * 2.5) & (pdist > r * 0.8)
    halo_val = rng.integers(-15, -5)
    arr[halo_mask] = np.clip(arr[halo_mask].astype(np.int16) + halo_val, 0, 255).astype(np.uint8)

    # Core particle
    is_bright = rng.random() > 0.5
    core_mask = pdist < r
    if is_bright:
        arr[core_mask] = np.clip(arr[core_mask].astype(np.int16) + rng.integers(60, 100), 0, 255).astype(np.uint8)
    else:
        arr[core_mask] = np.clip(arr[core_mask].astype(np.int16) - rng.integers(80, 130), 0, 255).astype(np.uint8)

    img_out = Image.fromarray(arr)
    img.paste(img_out)
    pad = int(r * 2.5) + 2
    return [{"class": "particle", "bbox": [px - pad, py - pad, px + pad, py + pad]}]


def _add_realistic_crack(img: Image.Image, rng: np.random.Generator) -> list:
    """Branching crack with depth shadow."""
    draw = ImageDraw.Draw(img)
    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
    radius = IMG_SIZE // 2 - 50

    sx = rng.integers(cx - radius // 2, cx + radius // 2)
    sy = rng.integers(cy - radius // 2, cy + radius // 2)

    all_points = [(sx, sy)]
    branches = [[(sx, sy)]]

    # Main crack
    main_len = rng.integers(8, 18)
    for _ in range(main_len):
        last = branches[0][-1]
        dx = rng.integers(-18, 19)
        dy = rng.integers(-18, 19)
        nxt = (int(np.clip(last[0] + dx, 20, IMG_SIZE - 20)),
               int(np.clip(last[1] + dy, 20, IMG_SIZE - 20)))
        branches[0].append(nxt)
        all_points.append(nxt)

        # Branch
        if rng.random() < 0.35 and len(branches) < 5:
            branch = [nxt]
            for _ in range(rng.integers(3, 7)):
                bp = branch[-1]
                bnxt = (int(np.clip(bp[0] + rng.integers(-14, 15), 20, IMG_SIZE - 20)),
                        int(np.clip(bp[1] + rng.integers(-14, 15), 20, IMG_SIZE - 20)))
                branch.append(bnxt)
                all_points.append(bnxt)
            branches.append(branch)

    # Draw shadow first (offset), then crack
    for br in branches:
        if len(br) >= 2:
            shadow_pts = [(p[0] + 1, p[1] + 2) for p in br]
            draw.line(shadow_pts, fill=(50, 48, 45), width=rng.integers(2, 4))
    for br in branches:
        if len(br) >= 2:
            draw.line(br, fill=(35, 30, 28), width=rng.integers(1, 3))

    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    return [{"class": "crack", "bbox": [min(xs) - 5, min(ys) - 5, max(xs) + 5, max(ys) + 5]}]


def _add_realistic_contamination(img: Image.Image, rng: np.random.Generator) -> list:
    """Large irregular contamination zone with gradient edges."""
    arr = np.array(img, dtype=np.float32)
    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2

    # Center of contamination
    pcx = rng.integers(cx - 150, cx + 150)
    pcy = rng.integers(cy - 150, cy + 150)
    w = rng.integers(40, 100)
    h = rng.integers(40, 100)

    Y, X = np.mgrid[:IMG_SIZE, :IMG_SIZE]
    # Elliptical distance
    edist = np.sqrt(((X - pcx) / w) ** 2 + ((Y - pcy) / h) ** 2)

    # Soft falloff contamination
    contam_mask = edist < 1.5
    falloff = np.clip(1.5 - edist, 0, 1) ** 2

    # Choose contamination color (yellowish, brownish, or dark)
    contam_type = rng.integers(0, 3)
    if contam_type == 0:  # dark
        delta = np.stack([-60 * falloff, -55 * falloff, -50 * falloff], axis=-1)
    elif contam_type == 1:  # yellowish
        delta = np.stack([15 * falloff, 10 * falloff, -40 * falloff], axis=-1)
    else:  # brownish
        delta = np.stack([-20 * falloff, -35 * falloff, -45 * falloff], axis=-1)

    arr = np.where(contam_mask[:, :, None], arr + delta, arr)
    img_out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    img.paste(img_out)

    pad_w = int(w * 1.3)
    pad_h = int(h * 1.3)
    return [{"class": "contamination", "bbox": [pcx - pad_w, pcy - pad_h, pcx + pad_w, pcy + pad_h]}]


def _add_realistic_edge_chip(img: Image.Image, rng: np.random.Generator) -> list:
    """Irregular chip at wafer edge - dark, with jagged outline."""
    draw = ImageDraw.Draw(img)
    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
    radius = IMG_SIZE // 2 - 20

    angle = rng.uniform(0, 2 * math.pi)
    ecx = int(cx + (radius - 8) * math.cos(angle))
    ecy = int(cy + (radius - 8) * math.sin(angle))

    n_pts = rng.integers(5, 9)
    size = rng.integers(15, 40)
    points = []
    for i in range(n_pts):
        a = angle + rng.uniform(-0.8, 0.8)
        r = size * rng.uniform(0.4, 1.0)
        points.append((int(ecx + r * math.cos(a + 2 * math.pi * i / n_pts)),
                        int(ecy + r * math.sin(a + 2 * math.pi * i / n_pts))))

    # Dark fill with slight color
    fill_c = (rng.integers(30, 55), rng.integers(28, 50), rng.integers(32, 58))
    draw.polygon(points, fill=fill_c)
    draw.polygon(points, outline=(rng.integers(70, 100), rng.integers(65, 95), rng.integers(75, 105)), width=1)

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [{"class": "edge_chip", "bbox": [min(xs) - 3, min(ys) - 3, max(xs) + 3, max(ys) + 3]}]


def _add_realistic_delamination(img: Image.Image, rng: np.random.Generator) -> list:
    """Film peeling region - lighter area with sharp boundary on one side."""
    arr = np.array(img, dtype=np.float32)
    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2

    pcx = rng.integers(cx - 120, cx + 120)
    pcy = rng.integers(cy - 120, cy + 120)
    w = rng.integers(30, 80)
    h = rng.integers(30, 80)

    Y, X = np.mgrid[:IMG_SIZE, :IMG_SIZE]
    edist = np.sqrt(((X - pcx) / w) ** 2 + ((Y - pcy) / h) ** 2)

    delam_mask = edist < 1.0
    # Lighter region
    lift = rng.uniform(20, 45)
    arr[:, :, 0] = np.where(delam_mask, arr[:, :, 0] + lift, arr[:, :, 0])
    arr[:, :, 1] = np.where(delam_mask, arr[:, :, 1] + lift * 0.9, arr[:, :, 1])
    arr[:, :, 2] = np.where(delam_mask, arr[:, :, 2] + lift * 0.7, arr[:, :, 2])

    img_out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    # Sharp edge on one side
    draw = ImageDraw.Draw(img_out)
    edge_angle = rng.uniform(0, 2 * math.pi)
    ex1 = int(pcx + w * math.cos(edge_angle))
    ey1 = int(pcy + h * math.sin(edge_angle))
    ex2 = int(pcx + w * math.cos(edge_angle + 1.5))
    ey2 = int(pcy + h * math.sin(edge_angle + 1.5))
    draw.line([(ex1, ey1), (pcx, pcy), (ex2, ey2)], fill=(95, 90, 85), width=2)
    img.paste(img_out)

    return [{"class": "delamination", "bbox": [pcx - w - 3, pcy - h - 3, pcx + w + 3, pcy + h + 3]}]


# Defect scenarios for each image
SCENARIOS = [
    # (defect_funcs, description)
    ([_add_realistic_scratch, _add_realistic_particle], "scratch + particle"),
    ([_add_realistic_crack, _add_realistic_edge_chip], "crack + edge chip"),
    ([_add_realistic_contamination, _add_realistic_scratch], "contamination + scratch"),
    ([_add_realistic_edge_chip, _add_realistic_delamination], "edge chip + delamination"),
    ([_add_realistic_crack, _add_realistic_contamination, _add_realistic_particle], "crack + contamination + particle"),
    ([_add_realistic_scratch, _add_realistic_crack, _add_realistic_edge_chip], "scratch + crack + edge chip"),
    ([_add_realistic_contamination, _add_realistic_delamination, _add_realistic_particle], "contamination + delamination + particle"),
]


def generate_realistic_images(output_dir: Path, n_images: int = 7, seed: int = 2026):
    """Generate photorealistic wafer images with defects and YOLO labels."""
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(seed)
    all_info = []

    for i in range(n_images):
        palette = WAFER_PALETTES[i % len(WAFER_PALETTES)]
        scenario = SCENARIOS[i % len(SCENARIOS)]

        img, wafer_radius = _make_wafer_base(rng, palette)
        defects = []
        for defect_fn in scenario[0]:
            dets = defect_fn(img, rng)
            defects.extend(dets)

        # Slight blur for realism
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

        # Save image
        fname = f"realistic_{i:02d}.jpg"
        img.save(output_dir / fname, quality=95)

        # Save YOLO labels
        label_lines = []
        for det in defects:
            cls_id = CLASSES.index(det["class"])
            x1, y1, x2, y2 = det["bbox"]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(IMG_SIZE, x2), min(IMG_SIZE, y2)
            xc = (x1 + x2) / 2 / IMG_SIZE
            yc = (y1 + y2) / 2 / IMG_SIZE
            w = (x2 - x1) / IMG_SIZE
            h = (y2 - y1) / IMG_SIZE
            label_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        with open(labels_dir / f"realistic_{i:02d}.txt", "w") as f:
            f.write("\n".join(label_lines))

        info = {"image": fname, "scenario": scenario[1], "defects": defects}
        all_info.append(info)
        print(f"  [{i+1}/{n_images}] {fname}: {scenario[1]} ({len(defects)} defects)")

    print(f"\nGenerated {n_images} realistic wafer images in {output_dir}")
    return all_info


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "outputs" / "realistic_unseen"
    generate_realistic_images(out, n_images=7)
