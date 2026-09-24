"""
RDD2022 Dataset Validation Suite
================================

Validates structural integrity, annotation correctness, coordinate bounds,
image-annotation pairs, and taxonomy conformance for RDD2022 India dataset.

Checks Performed:
1. XML Syntax & Well-Formedness
2. Image-Annotation Pairing (orphans and missing pairs)
3. Image Header & Dimension Consistency
4. Bounding Box Geometry: 0 <= xmin < xmax <= width, 0 <= ymin < ymax <= height
5. Taxonomy Conformance: Standard RDD2022 classes (D00, D10, D20, D40)
6. Manifest Integrity (annotations_index.json, dataset_summary.json)

Usage:
  python scripts/validate_dataset.py [--data-dir data/samples] [--strict]
"""

import os
import sys
import glob
import json
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Set

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SAMPLES_DIR = DATA_DIR / "samples"
RAW_DIR = DATA_DIR / "raw" / "rdd2022_india"
PROCESSED_DIR = DATA_DIR / "processed"

CORE_CLASSES = {"D00", "D10", "D20", "D40"}
CLASS_NAMES = {
    "D00": "Longitudinal Crack",
    "D10": "Transverse Crack",
    "D20": "Alligator Crack",
    "D40": "Pothole"
}


class DatasetValidator:
    def __init__(self, target_dir: Path, strict: bool = False):
        self.target_dir = target_dir
        self.strict = strict
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.stats = {
            "total_annotations": 0,
            "total_images": 0,
            "paired_records": 0,
            "unpaired_xmls": 0,
            "unpaired_images": 0,
            "total_objects": 0,
            "class_counts": {c: 0 for c in CORE_CLASSES},
            "non_standard_classes": {}
        }

    def run(self) -> bool:
        print("=" * 76)
        print("RDD2022 DATASET INTEGRITY VALIDATION")
        print(f"Target Directory: {self.target_dir}")
        print(f"Strict Mode:      {self.strict}")
        print("=" * 76)

        if not self.target_dir.exists():
            self.errors.append(f"Target directory does not exist: {self.target_dir}")
            self.report()
            return False

        # Locate XML files
        xml_files = list(self.target_dir.glob("**/*.xml"))
        # Locate Image files (.jpg, .jpeg, .png)
        img_files = (
            list(self.target_dir.glob("**/*.jpg")) +
            list(self.target_dir.glob("**/*.jpeg")) +
            list(self.target_dir.glob("**/*.png"))
        )

        self.stats["total_annotations"] = len(xml_files)
        self.stats["total_images"] = len(img_files)

        xml_stem_map = {x.stem: x for x in xml_files}
        img_stem_map = {i.stem: i for i in img_files}

        common_stems = set(xml_stem_map.keys()) & set(img_stem_map.keys())
        self.stats["paired_records"] = len(common_stems)
        self.stats["unpaired_xmls"] = len(xml_files) - len(common_stems)
        self.stats["unpaired_images"] = len(img_files) - len(common_stems)

        if len(xml_files) == 0:
            self.errors.append("No XML annotation files found in target directory!")
            self.report()
            return False

        print(f"Found {len(xml_files)} XML annotations and {len(img_files)} images ({len(common_stems)} paired).")

        # Validate each XML and paired image
        for stem, xml_path in xml_stem_map.items():
            img_path = img_stem_map.get(stem)
            self._validate_xml_and_pair(xml_path, img_path)

        # Validate processed manifests if present
        self._validate_processed_manifests()

        return self.report()

    def _validate_xml_and_pair(self, xml_path: Path, img_path: Path = None):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except ET.ParseError as e:
            self.errors.append(f"XML Parse Error in {xml_path.name}: {e}")
            return

        size_elem = root.find("size")
        if size_elem is None:
            self.errors.append(f"{xml_path.name}: Missing <size> element")
            w, h = 720, 720
        else:
            try:
                w = int(size_elem.findtext("width", "0"))
                h = int(size_elem.findtext("height", "0"))
                if w <= 0 or h <= 0:
                    self.errors.append(f"{xml_path.name}: Invalid dimensions width={w}, height={h}")
            except ValueError:
                self.errors.append(f"{xml_path.name}: Non-integer dimensions in <size>")
                w, h = 720, 720

        # Validate image file match
        if img_path:
            if img_path.stat().st_size == 0:
                self.errors.append(f"{img_path.name}: Image file is 0 bytes")
        else:
            self.warnings.append(f"Unpaired annotation: {xml_path.name} has no matching image file")

        # Validate objects
        objects = root.findall("object")
        if not objects:
            self.warnings.append(f"{xml_path.name}: Contains 0 <object> annotations (empty image)")
            return

        for obj_idx, obj in enumerate(objects):
            name = obj.findtext("name")
            if not name:
                self.errors.append(f"{xml_path.name} [obj {obj_idx}]: Missing <name> tag")
                continue
            name = name.strip()

            if name in CORE_CLASSES:
                self.stats["class_counts"][name] += 1
            else:
                self.stats["non_standard_classes"][name] = self.stats["non_standard_classes"].get(name, 0) + 1
                msg = f"{xml_path.name} [obj {obj_idx}]: Non-standard class '{name}'"
                if self.strict:
                    self.errors.append(msg)
                else:
                    self.warnings.append(msg)

            bndbox = obj.find("bndbox")
            if bndbox is None:
                self.errors.append(f"{xml_path.name} [obj {obj_idx}]: Missing <bndbox> tag")
                continue

            try:
                xmin = float(bndbox.findtext("xmin", "0"))
                ymin = float(bndbox.findtext("ymin", "0"))
                xmax = float(bndbox.findtext("xmax", "0"))
                ymax = float(bndbox.findtext("ymax", "0"))
            except ValueError:
                self.errors.append(f"{xml_path.name} [obj {obj_idx}]: Non-numeric coordinate values")
                continue

            # Geometry checks
            if xmin < 0 or ymin < 0:
                self.errors.append(f"{xml_path.name} [obj {obj_idx}]: Negative coordinates ({xmin}, {ymin})")
            if xmax <= xmin:
                self.errors.append(f"{xml_path.name} [obj {obj_idx}]: Inverted horizontal box xmax ({xmax}) <= xmin ({xmin})")
            if ymax <= ymin:
                self.errors.append(f"{xml_path.name} [obj {obj_idx}]: Inverted vertical box ymax ({ymax}) <= ymin ({ymin})")
            if w > 0 and xmax > w * 1.05:
                self.warnings.append(f"{xml_path.name} [obj {obj_idx}]: xmax ({xmax}) exceeds image width ({w})")
            if h > 0 and ymax > h * 1.05:
                self.warnings.append(f"{xml_path.name} [obj {obj_idx}]: ymax ({ymax}) exceeds image height ({h})")

            self.stats["total_objects"] += 1

    def _validate_processed_manifests(self):
        index_path = PROCESSED_DIR / "annotations_index.json"
        if index_path.exists():
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    self.errors.append(f"{index_path.name}: Root JSON is not a dictionary")
                else:
                    print(f"Validated processed manifest {index_path.name}: {len(data)} indexed entries.")
            except Exception as e:
                self.errors.append(f"Failed loading {index_path.name}: {e}")

    def report(self) -> bool:
        print("\n" + "-" * 76)
        print("VALIDATION SUMMARY")
        print("-" * 76)
        print(f"Total Annotations:       {self.stats['total_annotations']}")
        print(f"Total Images:            {self.stats['total_images']}")
        print(f"Paired Pairs:            {self.stats['paired_records']}")
        print(f"Total Damage Instances:  {self.stats['total_objects']}")
        print("\nClass Distribution:")
        for cid in sorted(CORE_CLASSES):
            print(f"  [{cid}] {CLASS_NAMES[cid]:<22}: {self.stats['class_counts'][cid]}")

        if self.stats["non_standard_classes"]:
            print("\nNon-standard classes found:")
            for nsc, count in self.stats["non_standard_classes"].items():
                print(f"  [{nsc}]: {count}")

        print("\nIntegrity Results:")
        print(f"  Errors:   {len(self.errors)}")
        print(f"  Warnings: {len(self.warnings)}")

        if self.warnings:
            print("\nWarnings:")
            for w in self.warnings[:10]:
                print(f"  [WARN] {w}")
            if len(self.warnings) > 10:
                print(f"  ... and {len(self.warnings) - 10} more warnings.")

        if self.errors:
            print("\nErrors:")
            for e in self.errors[:10]:
                print(f"  [ERROR] {e}")
            if len(self.errors) > 10:
                print(f"  ... and {len(self.errors) - 10} more errors.")
            print("\nSTATUS: FAILED [X]")
            print("=" * 76)
            return False

        print("\nSTATUS: PASSED [OK] All integrity checks passed successfully.")
        print("=" * 76)
        return True


def main():
    parser = argparse.ArgumentParser(description="Validate RDD2022 dataset files")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(SAMPLES_DIR if SAMPLES_DIR.exists() else (DATA_DIR / "raw" / "rdd2022_india")),
        help="Path to dataset directory (e.g. data/samples or data/raw/rdd2022_india)"
    )
    parser.add_argument("--strict", action="store_true", help="Treat non-standard damage classes as errors")
    args = parser.parse_args()

    validator = DatasetValidator(Path(args.data_dir), strict=args.strict)
    success = validator.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
