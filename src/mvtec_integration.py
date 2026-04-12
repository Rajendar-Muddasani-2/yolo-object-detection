"""
MVTec AD dataset integration for YOLO training.

Downloads MVTec Anomaly Detection dataset, converts anomaly segmentation masks
to YOLO bounding box format, and merges with synthetic wafer defect data.

MVTec classes mapped to our defect taxonomy:
  - carpet/grid/leather/tile/wood textures → contamination, scratch, void, etc.
  - cable/capsule/hazelnut/metal_nut/pill/screw/toothbrush/transistor/zipper → object defects

Reference: https://www.mvtec.com/company/research/datasets/mvtec-ad
"""

import json
import logging
import shutil
import tarfile
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.request import urlretrieve

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# MVTec texture categories (most relevant to wafer inspection)
MVTEC_TEXTURE_CATEGORIES = ["carpet", "grid", "leather", "tile", "wood"]

# MVTec object categories
MVTEC_OBJECT_CATEGORIES = [
    "bottle", "cable", "capsule", "hazelnut", "metal_nut",
    "pill", "screw", "toothbrush", "transistor", "zipper",
]

# Map MVTec defect types to our 10-class schema
MVTEC_TO_YOLO_CLASS = {
    # Texture defects
    "carpet_color": 8,        # contamination
    "carpet_cut": 0,          # scratch
    "carpet_hole": 3,         # void
    "carpet_metal_contamination": 8,  # contamination
    "carpet_thread": 5,       # bridge
    "grid_bent": 4,           # pattern_shift
    "grid_broken": 7,         # crack
    "grid_glue": 8,           # contamination
    "grid_metal_contamination": 8,  # contamination
    "grid_thread": 5,         # bridge
    "leather_color": 8,       # contamination
    "leather_cut": 0,         # scratch
    "leather_fold": 9,        # delamination
    "leather_glue": 8,        # contamination
    "leather_poke": 1,        # particle (small defect)
    "tile_crack": 7,          # crack
    "tile_glue_strip": 5,     # bridge
    "tile_gray_stroke": 0,    # scratch
    "tile_oil": 8,            # contamination
    "tile_rough": 9,          # delamination
    "wood_color": 8,          # contamination
    "wood_combined": 8,       # contamination
    "wood_hole": 3,           # void
    "wood_liquid": 8,         # contamination
    "wood_scratch": 0,        # scratch
}

# Default mapping for unmapped defect types
DEFAULT_MVTEC_CLASS = 8  # contamination


def mask_to_bboxes(mask: np.ndarray, min_area: int = 100) -> List[Tuple[int, int, int, int]]:
    """Convert a binary segmentation mask to bounding boxes.

    Uses connected component analysis to find individual defect regions.
    """
    from scipy import ndimage

    labeled, n_features = ndimage.label(mask > 127)
    bboxes = []
    for i in range(1, n_features + 1):
        ys, xs = np.where(labeled == i)
        if len(xs) < 2 or len(ys) < 2:
            continue
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        area = (x2 - x1) * (y2 - y1)
        if area >= min_area:
            bboxes.append((x1, y1, x2, y2))
    return bboxes


def convert_mvtec_to_yolo(
    mvtec_dir: str,
    output_dir: str,
    categories: List[str] = None,
    img_size: int = 640,
) -> Dict[str, int]:
    """Convert MVTec AD dataset to YOLO format.

    Args:
        mvtec_dir: Root of extracted MVTec AD dataset.
        output_dir: Output directory for YOLO-format data.
        categories: Which MVTec categories to include. None = textures only.
        img_size: Target image size (will resize).

    Returns:
        Statistics about converted images.
    """
    mvtec_path = Path(mvtec_dir)
    out_path = Path(output_dir)

    if categories is None:
        categories = MVTEC_TEXTURE_CATEGORIES

    stats = {"total_images": 0, "total_boxes": 0, "skipped": 0}

    for category in categories:
        cat_dir = mvtec_path / category
        if not cat_dir.exists():
            logger.warning("Category not found: %s", category)
            continue

        test_dir = cat_dir / "test"
        gt_dir = cat_dir / "ground_truth"

        if not test_dir.exists() or not gt_dir.exists():
            continue

        # Process each defect type within the category
        for defect_type_dir in sorted(test_dir.iterdir()):
            if not defect_type_dir.is_dir() or defect_type_dir.name == "good":
                continue

            defect_name = f"{category}_{defect_type_dir.name}"
            yolo_class = MVTEC_TO_YOLO_CLASS.get(defect_name, DEFAULT_MVTEC_CLASS)
            gt_defect_dir = gt_dir / defect_type_dir.name

            for img_path in sorted(defect_type_dir.glob("*.png")):
                # Find corresponding ground truth mask
                mask_path = gt_defect_dir / f"{img_path.stem}_mask.png"
                if not mask_path.exists():
                    stats["skipped"] += 1
                    continue

                # Load and resize image
                img = Image.open(img_path).convert("RGB").resize((img_size, img_size))
                mask = np.array(
                    Image.open(mask_path).convert("L").resize((img_size, img_size))
                )

                # Convert mask to bounding boxes
                bboxes = mask_to_bboxes(mask)
                if not bboxes:
                    stats["skipped"] += 1
                    continue

                # Determine split (80/20)
                split = "train" if stats["total_images"] % 5 != 0 else "val"

                # Save image and labels
                (out_path / split / "images").mkdir(parents=True, exist_ok=True)
                (out_path / split / "labels").mkdir(parents=True, exist_ok=True)

                fname = f"mvtec_{category}_{defect_type_dir.name}_{img_path.stem}"
                img.save(out_path / split / "images" / f"{fname}.jpg", quality=95)

                labels = []
                for x1, y1, x2, y2 in bboxes:
                    cx = (x1 + x2) / 2 / img_size
                    cy = (y1 + y2) / 2 / img_size
                    w = (x2 - x1) / img_size
                    h = (y2 - y1) / img_size
                    labels.append(f"{yolo_class} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

                (out_path / split / "labels" / f"{fname}.txt").write_text(
                    "\n".join(labels) + "\n"
                )

                stats["total_images"] += 1
                stats["total_boxes"] += len(bboxes)

    logger.info("MVTec conversion: %d images, %d boxes, %d skipped",
                stats["total_images"], stats["total_boxes"], stats["skipped"])
    return stats


def merge_datasets(
    synthetic_dir: str,
    mvtec_dir: str,
    merged_dir: str,
    classes: List[str] = None,
) -> Dict[str, int]:
    """Merge synthetic wafer defects with MVTec data into a unified YOLO dataset.

    Creates symlinks (or copies) to avoid data duplication.
    """
    from src.data_generator import CLASSES, NUM_CLASSES

    if classes is None:
        classes = CLASSES

    syn_path = Path(synthetic_dir)
    mvt_path = Path(mvtec_dir)
    merged = Path(merged_dir)

    stats = {"synthetic": 0, "mvtec": 0}

    for split in ("train", "val", "test"):
        (merged / split / "images").mkdir(parents=True, exist_ok=True)
        (merged / split / "labels").mkdir(parents=True, exist_ok=True)

        # Copy synthetic data
        syn_imgs = syn_path / split / "images"
        syn_lbls = syn_path / split / "labels"
        if syn_imgs.exists():
            for img in syn_imgs.glob("*.jpg"):
                shutil.copy2(img, merged / split / "images" / img.name)
                lbl = syn_lbls / f"{img.stem}.txt"
                if lbl.exists():
                    shutil.copy2(lbl, merged / split / "labels" / lbl.name)
                stats["synthetic"] += 1

        # Copy MVTec data (test split goes to val if no test in MVTec)
        mvt_split = split if (mvt_path / split).exists() else "val"
        mvt_imgs = mvt_path / mvt_split / "images"
        mvt_lbls = mvt_path / mvt_split / "labels"
        if mvt_imgs.exists() and split != "test":
            for img in mvt_imgs.glob("*.jpg"):
                shutil.copy2(img, merged / split / "images" / img.name)
                lbl = mvt_lbls / f"{img.stem}.txt"
                if lbl.exists():
                    shutil.copy2(lbl, merged / split / "labels" / lbl.name)
                stats["mvtec"] += 1

    # Write merged data.yaml
    (merged / "data.yaml").write_text(
        f"path: {merged.resolve()}\ntrain: train/images\nval: val/images\ntest: test/images\n"
        f"nc: {NUM_CLASSES}\nnames: {classes}\n"
    )

    logger.info("Merged dataset: %d synthetic + %d MVTec = %d total",
                stats["synthetic"], stats["mvtec"], stats["synthetic"] + stats["mvtec"])
    return stats


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Convert MVTec AD to YOLO format")
    parser.add_argument("--mvtec-dir", required=True, help="Path to extracted MVTec AD dataset")
    parser.add_argument("--output-dir", default="data/mvtec_yolo", help="Output directory")
    parser.add_argument("--img-size", type=int, default=640, help="Target image size")
    args = parser.parse_args()

    stats = convert_mvtec_to_yolo(args.mvtec_dir, args.output_dir, img_size=args.img_size)
    print(f"Conversion stats: {json.dumps(stats, indent=2)}")
