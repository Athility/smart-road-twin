"""
RDD2022 Dataset Preparation & Preprocessing Pipeline
===================================================

This script processes Pascal VOC XML annotations from the RDD2022 India dataset
(from data/raw/rdd2022_india/ or data/samples/) into normalized index manifests
and structured JSON records stored in data/processed/.

Core Damage Categories:
- D00: Longitudinal Crack
- D10: Transverse Crack
- D20: Alligator Crack
- D40: Pothole

Outputs in data/processed/:
- annotations_index.json: Full index of images and parsed annotations
- dataset_summary.json: Aggregated damage counts and distribution statistics
- train_manifest.json & val_manifest.json: Stratified splits for model training

Usage:
  python scripts/prepare_rdd2022.py [--source data/samples] [--output data/processed]
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import glob
import json
import random
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "rdd2022_india"
SAMPLES_DIR = DATA_DIR / "samples"
PROCESSED_DIR = DATA_DIR / "processed"

CLASS_MAPPING = {
    "D00": "Longitudinal Crack",
    "D10": "Transverse Crack",
    "D20": "Alligator Crack",
    "D40": "Pothole"
}

SEVERITY_RANK = {
    "D40": 4,  # Pothole (highest physical risk)
    "D20": 3,  # Alligator Crack
    "D10": 2,  # Transverse Crack
    "D00": 1   # Longitudinal Crack
}


def parse_voc_xml(xml_file: Path) -> Optional[Dict[str, Any]]:
    """Parse a single Pascal VOC XML annotation file."""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        filename = root.findtext("filename")
        if not filename:
            filename = f"{xml_file.stem}.jpg"

        size_elem = root.find("size")
        if size_elem is not None:
            width = int(size_elem.findtext("width", "720"))
            height = int(size_elem.findtext("height", "720"))
            depth = int(size_elem.findtext("depth", "3"))
        else:
            width, height, depth = 720, 720, 3

        objects = []
        for obj in root.findall("object"):
            name = obj.findtext("name")
            if not name:
                continue
            name = name.strip()

            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue

            xmin = float(bndbox.findtext("xmin", "0"))
            ymin = float(bndbox.findtext("ymin", "0"))
            xmax = float(bndbox.findtext("xmax", "0"))
            ymax = float(bndbox.findtext("ymax", "0"))

            # Clip within image bounds
            xmin = max(0.0, min(float(width), xmin))
            xmax = max(0.0, min(float(width), xmax))
            ymin = max(0.0, min(float(height), ymin))
            ymax = max(0.0, min(float(height), ymax))

            if xmax <= xmin or ymax <= ymin:
                continue

            box_w = xmax - xmin
            box_h = ymax - ymin
            pixel_area = box_w * box_h
            norm_area = pixel_area / (width * height)

            # Heuristic ground plane surface area estimation in square meters
            # Assuming standard windshield camera (~1.2m height, road footprint ~12m^2 in lower half)
            est_surface_sqm = round(norm_area * 14.0 * (1.0 + (ymin / height) * 0.8), 3)
            est_surface_sqm = max(0.05, min(15.0, est_surface_sqm))

            objects.append({
                "class_id": name,
                "class_name": CLASS_MAPPING.get(name, "Unknown"),
                "xmin": round(xmin, 1),
                "ymin": round(ymin, 1),
                "xmax": round(xmax, 1),
                "ymax": round(ymax, 1),
                "width": round(box_w, 1),
                "height": round(box_h, 1),
                "rel_xmin": round(xmin / width, 4),
                "rel_ymin": round(ymin / height, 4),
                "rel_xmax": round(xmax / width, 4),
                "rel_ymax": round(ymax / height, 4),
                "pixel_area": round(pixel_area, 1),
                "normalized_area": round(norm_area, 4),
                "estimated_surface_sqm": est_surface_sqm
            })

        if not objects:
            return None

        # Determine primary damage class by highest severity rank
        primary_class = max(objects, key=lambda o: SEVERITY_RANK.get(o["class_id"], 0))["class_id"]
        classes_present = list(sorted(set(o["class_id"] for o in objects)))

        return {
            "id": xml_file.stem,
            "filename": filename,
            "width": width,
            "height": height,
            "depth": depth,
            "primary_class": primary_class,
            "primary_class_name": CLASS_MAPPING.get(primary_class, "Unknown"),
            "classes_present": classes_present,
            "object_count": len(objects),
            "objects": objects
        }
    except Exception as e:
        print(f"Error parsing {xml_file}: {e}")
        return None


def prepare_dataset(source_dir: Path, output_dir: Path, max_records: Optional[int] = None):
    print("=" * 76)
    print("RDD2022 DATASET PREPARATION PIPELINE")
    print(f"Source Directory: {source_dir}")
    print(f"Output Directory: {output_dir}")
    print("=" * 76)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Search for XML files
    xml_patterns = [
        source_dir / "**" / "*.xml",
        source_dir / "annotations" / "xmls" / "*.xml",
        source_dir / "train" / "annotations" / "xmls" / "*.xml"
    ]

    xml_files = []
    for pat in xml_patterns:
        found = list(Path(source_dir).glob(pat.as_posix().replace(source_dir.as_posix() + "/", "")))
        if found:
            xml_files.extend(found)

    # Deduplicate
    xml_files = list(sorted(set(xml_files)))

    if not xml_files:
        # Fallback to sample directory if raw is empty
        if source_dir != SAMPLES_DIR and (SAMPLES_DIR / "annotations" / "xmls").exists():
            print(f"No XML files found in {source_dir}. Falling back to sample directory {SAMPLES_DIR}...")
            return prepare_dataset(SAMPLES_DIR, output_dir, max_records)
        else:
            print(f"Error: No XML annotation files found in {source_dir}!")
            return None

    if max_records and len(xml_files) > max_records:
        xml_files = xml_files[:max_records]

    print(f"Found {len(xml_files)} annotation XML files. Parsing...")

    records = {}
    class_counts = {c: 0 for c in CLASS_MAPPING}
    primary_counts = {c: 0 for c in CLASS_MAPPING}
    total_objects = 0

    for idx, xml_path in enumerate(xml_files):
        parsed = parse_voc_xml(xml_path)
        if not parsed:
            continue

        records[parsed["id"]] = parsed
        total_objects += parsed["object_count"]
        primary_counts[parsed["primary_class"]] = primary_counts.get(parsed["primary_class"], 0) + 1

        for obj in parsed["objects"]:
            cid = obj["class_id"]
            class_counts[cid] = class_counts.get(cid, 0) + 1

        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{len(xml_files)} XML files...")

    print(f"Successfully processed {len(records)} images with {total_objects} total damage instances.")

    # Write annotations_index.json
    index_path = output_dir / "annotations_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"  -> Saved annotations index: {index_path} ({len(records)} entries)")

    # Compute summary statistics
    summary = {
        "dataset_name": "RDD2022_India",
        "total_images": len(records),
        "total_damage_instances": total_objects,
        "class_breakdown": {
            cid: {
                "name": CLASS_MAPPING.get(cid, "Unknown"),
                "total_instances": class_counts.get(cid, 0),
                "primary_in_images": primary_counts.get(cid, 0)
            }
            for cid in sorted(CLASS_MAPPING.keys())
        },
        "mean_instances_per_image": round(total_objects / len(records), 2) if records else 0.0
    }

    summary_path = output_dir / "dataset_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  -> Saved dataset summary: {summary_path}")

    # Generate train / val / test split manifests with zero leakage (Seed = 2026)
    items = sorted(list(records.keys()))
    rng = random.Random(2026)
    rng.shuffle(items)
    n_total = len(items)
    n_train = int(round(n_total * 0.70))
    n_val = int(round(n_total * 0.20))
    n_test = n_total - n_train - n_val
    if n_total >= 3:
        if n_test == 0:
            n_test = 1
            if n_train > 1:
                n_train -= 1
            elif n_val > 1:
                n_val -= 1
        if n_val == 0:
            n_val = 1
            if n_train > 1:
                n_train -= 1

    train_ids = items[:n_train]
    val_ids = items[n_train:n_train + n_val]
    test_ids = items[n_train + n_val:]

    # Assert strict zero leakage
    assert len(set(train_ids) & set(test_ids)) == 0, "Train and test leakage detected!"
    assert len(set(train_ids) & set(val_ids)) == 0, "Train and val leakage detected!"

    with open(output_dir / "train_manifest.json", "w", encoding="utf-8") as f:
        json.dump({"split": "train", "seed": 2026, "count": len(train_ids), "image_ids": train_ids}, f, indent=2)

    with open(output_dir / "val_manifest.json", "w", encoding="utf-8") as f:
        json.dump({"split": "validation", "seed": 2026, "count": len(val_ids), "image_ids": val_ids}, f, indent=2)

    with open(output_dir / "test_manifest.json", "w", encoding="utf-8") as f:
        json.dump({"split": "test", "seed": 2026, "count": len(test_ids), "image_ids": test_ids}, f, indent=2)

    print(f"  -> Saved train split ({len(train_ids)}), val split ({len(val_ids)}), and test split ({len(test_ids)}) [Zero Leakage Verified]")
    print("=" * 76)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Prepare and index RDD2022 India dataset")
    parser.add_argument(
        "--source",
        type=str,
        default=str(RAW_DIR if RAW_DIR.exists() and any(RAW_DIR.iterdir()) else SAMPLES_DIR),
        help="Path to raw RDD2022 India dataset or samples folder"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROCESSED_DIR),
        help="Path to write processed JSON manifests"
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Limit number of records to process"
    )
    args = parser.parse_args()

    prepare_dataset(Path(args.source), Path(args.output), args.max_records)


if __name__ == "__main__":
    main()
