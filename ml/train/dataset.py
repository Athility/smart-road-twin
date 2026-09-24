"""
RDD2022 YOLO Dataset Preparation Pipeline
=========================================

Converts authentic RDD2022 Pascal VOC XML annotations into normalized YOLO format
with strict train/validation/test split isolation.

Strict Scientific Principle:
Training strictly learns visual damage from authentic camera imagery.
Synthetic sensor values (LiDAR, sonar, accelerometer, traffic, GIS) are NEVER
included in the CV training pipeline.
"""

import os
import sys
import shutil
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_rdd2022 import parse_voc_xml

CLASS_TO_YOLO_INDEX = {
    "D00": 0,  # Longitudinal Crack
    "D10": 1,  # Transverse Crack
    "D20": 2,  # Alligator Crack
    "D40": 3   # Pothole
}


def prepare_yolo_dataset(
    raw_images_dir: Path,
    raw_xmls_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.20,
    test_ratio: float = 0.10,
    seed: int = 2026
) -> Dict[str, Any]:
    """
    Transforms Pascal VOC XML annotations into standard YOLO dataset directory hierarchy:
        output_dir/
            images/train, images/val, images/test
            labels/train, labels/val, labels/test
    
    Guarantees:
    - Deterministic partitioning (seed=2026)
    - Zero data leakage: train_ids ∩ test_ids = ∅
    - Purely visual ground truth (no synthetic fields)
    """
    output_dir = Path(output_dir)
    raw_images_dir = Path(raw_images_dir)
    raw_xmls_dir = Path(raw_xmls_dir)

    # Setup directories
    splits = ["train", "val", "test"]
    for s in splits:
        (output_dir / "images" / s).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / s).mkdir(parents=True, exist_ok=True)

    # Collect XML annotation files
    xml_files = sorted(list(raw_xmls_dir.glob("*.xml")))
    if not xml_files:
        sample_xmls = PROJECT_ROOT / "data" / "samples" / "annotations" / "xmls"
        if sample_xmls.exists() and list(sample_xmls.glob("*.xml")):
            xml_files = sorted(list(sample_xmls.glob("*.xml")))
            raw_images_dir = PROJECT_ROOT / "data" / "samples" / "images"
        else:
            raise FileNotFoundError(f"No XML files found in {raw_xmls_dir} or {sample_xmls}")

    # Parse and index annotations
    valid_samples = []
    for xf in xml_files:
        parsed = parse_voc_xml(xf)
        if parsed and parsed.get("objects"):
            img_name = parsed.get("filename", f"{xf.stem}.jpg")
            img_path = raw_images_dir / img_name
            valid_samples.append({
                "image_id": xf.stem,
                "image_name": img_name,
                "xml_path": xf,
                "image_path": img_path if img_path.exists() else None,
                "parsed": parsed
            })

    # Deterministic partition
    rng = random.Random(seed)
    rng.shuffle(valid_samples)

    n_total = len(valid_samples)
    n_train = int(round(n_total * train_ratio))
    n_val = int(round(n_total * val_ratio))
    n_test = n_total - n_train - n_val

    # Guarantee non-empty splits when possible
    if n_total >= 3 and n_test == 0:
        n_test = 1
        n_train -= 1

    train_samples = valid_samples[:n_train]
    val_samples = valid_samples[n_train:n_train + n_val]
    test_samples = valid_samples[n_train + n_val:]

    # Assert strict disjointness
    train_ids = {s["image_id"] for s in train_samples}
    val_ids = {s["image_id"] for s in val_samples}
    test_ids = {s["image_id"] for s in test_samples}

    assert len(train_ids & test_ids) == 0, "Data leakage detected: train and test sets overlap!"
    assert len(train_ids & val_ids) == 0, "Data leakage detected: train and val sets overlap!"

    split_groups = [
        ("train", train_samples),
        ("val", val_samples),
        ("test", test_samples)
    ]

    class_stats = {c: {"train": 0, "val": 0, "test": 0} for c in CLASS_TO_YOLO_INDEX}

    for split_name, samples in split_groups:
        for s in samples:
            parsed = s["parsed"]
            w = float(parsed.get("width", 720))
            h = float(parsed.get("height", 720))
            label_lines = []

            for obj in parsed.get("objects", []):
                cid = obj["class_id"]
                if cid not in CLASS_TO_YOLO_INDEX:
                    continue

                yolo_idx = CLASS_TO_YOLO_INDEX[cid]
                class_stats[cid][split_name] += 1

                # Normalize bounding box to (x_center, y_center, width, height) in [0, 1]
                xmin = float(obj["xmin"])
                ymin = float(obj["ymin"])
                xmax = float(obj["xmax"])
                ymax = float(obj["ymax"])

                xc = ((xmin + xmax) / 2.0) / w
                yc = ((ymin + ymax) / 2.0) / h
                bw = (xmax - xmin) / w
                bh = (ymax - ymin) / h

                label_lines.append(f"{yolo_idx} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

            # Write label file
            label_file = output_dir / "labels" / split_name / f"{s['image_id']}.txt"
            with open(label_file, "w", encoding="utf-8") as lf:
                lf.write("\n".join(label_lines) + "\n")

            # Copy image if available
            if s["image_path"] and s["image_path"].exists():
                dst_img = output_dir / "images" / split_name / s["image_name"]
                shutil.copy2(s["image_path"], dst_img)

    manifest = {
        "dataset_name": "RDD2022_India_YOLO",
        "seed": seed,
        "total_images": n_total,
        "splits": {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples)
        },
        "class_distribution": class_stats,
        "leakage_verification": {
            "train_test_overlap": len(train_ids & test_ids),
            "train_val_overlap": len(train_ids & val_ids)
        }
    }

    return manifest
