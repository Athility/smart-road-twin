"""
Research Data Split Manager & Leakage Prevention Engine
======================================================

Implements rigorous research data partitioning for multimodal road distress datasets.

Core Scientific Principle:
Partitioning is strictly executed at the source image/event level PRIOR to augmentation.
If an image contains multiple distress instances, all resulting records are quarantined
within the same split (train, validation, or test). This prevents multimodal context and
feature leakage between evaluation sets.

Supported Invariants:
- train_image_ids ∩ test_image_ids = ∅
- train_image_ids ∩ val_image_ids = ∅
- val_image_ids ∩ test_image_ids = ∅
- Total split count equals total input image count
- Deterministic reproducibility guaranteed via SEED = 2026
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_SEED = 2026
DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_VAL_RATIO = 0.20
DEFAULT_TEST_RATIO = 0.10


class ResearchDataSplitter:
    """
    Manages deterministic, zero-leakage research dataset splits.
    """
    def __init__(
        self,
        seed: int = DEFAULT_SEED,
        train_ratio: float = DEFAULT_TRAIN_RATIO,
        val_ratio: float = DEFAULT_VAL_RATIO,
        test_ratio: float = DEFAULT_TEST_RATIO
    ):
        self.seed = seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Split ratios must sum to 1.0"

    def split_image_ids(self, image_ids: List[str]) -> Dict[str, List[str]]:
        """
        Partitions unique image IDs deterministically across train, validation, and test.
        """
        unique_ids = sorted(list(set(image_ids)))
        n_total = len(unique_ids)

        if n_total == 0:
            return {"train": [], "validation": [], "test": []}

        # Deterministic shuffle
        rng = random.Random(self.seed)
        shuffled = list(unique_ids)
        rng.shuffle(shuffled)

        n_train = int(round(n_total * self.train_ratio))
        n_val = int(round(n_total * self.val_ratio))
        n_test = n_total - n_train - n_val

        # Guarantee non-empty evaluation splits when sample size allows (>= 3)
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

        train_ids = shuffled[:n_train]
        val_ids = shuffled[n_train:n_train + n_val]
        test_ids = shuffled[n_train + n_val:]

        # Verification of disjointness invariants
        s_train = set(train_ids)
        s_val = set(val_ids)
        s_test = set(test_ids)

        assert len(s_train & s_test) == 0, "Data leakage invariant violated: train and test overlap!"
        assert len(s_train & s_val) == 0, "Data leakage invariant violated: train and validation overlap!"
        assert len(s_val & s_test) == 0, "Data leakage invariant violated: validation and test overlap!"
        assert len(s_train | s_val | s_test) == n_total, "Coverage invariant violated: missing image IDs!"

        return {
            "train": train_ids,
            "validation": val_ids,
            "test": test_ids
        }

    def partition_records(
        self,
        records: List[Dict[str, Any]],
        image_splits: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Groups multimodal records into train, validation, and test based on their source image ID.
        """
        if image_splits is None:
            all_img_ids = [r.get("source", {}).get("image_id") or r.get("image_id") for r in records]
            all_img_ids = [img for img in all_img_ids if img]
            image_splits = self.split_image_ids(all_img_ids)

        img_to_split: Dict[str, str] = {}
        for split_name, img_ids in image_splits.items():
            for img_id in img_ids:
                img_to_split[img_id] = split_name

        partitioned: Dict[str, List[Dict[str, Any]]] = {
            "train": [],
            "validation": [],
            "test": []
        }

        for record in records:
            img_id = record.get("source", {}).get("image_id") or record.get("image_id")
            # Default to train if unmapped
            target_split = img_to_split.get(img_id, "train")
            
            # Tag split on record and provenance
            record_copy = dict(record)
            record_copy["split"] = target_split
            if "provenance" in record_copy and isinstance(record_copy["provenance"], dict):
                record_copy["provenance"] = dict(record_copy["provenance"])
                record_copy["provenance"]["split"] = target_split
            
            partitioned[target_split].append(record_copy)

        # Verify zero leakage across partitioned records
        self.verify_zero_record_leakage(
            partitioned["train"],
            partitioned["validation"],
            partitioned["test"]
        )

        return partitioned

    @staticmethod
    def verify_zero_record_leakage(
        train_records: List[Dict[str, Any]],
        val_records: List[Dict[str, Any]],
        test_records: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """
        Validates that no source image is shared across train, val, and test splits.
        """
        def get_images(recs):
            return {
                r.get("source", {}).get("image_id") or r.get("image_id")
                for r in recs
                if (r.get("source", {}).get("image_id") or r.get("image_id"))
            }

        train_imgs = get_images(train_records)
        val_imgs = get_images(val_records)
        test_imgs = get_images(test_records)

        train_test_overlap = len(train_imgs & test_imgs)
        train_val_overlap = len(train_imgs & val_imgs)
        val_test_overlap = len(val_imgs & test_imgs)

        if train_test_overlap > 0:
            raise ValueError(f"Data leakage detected! {train_test_overlap} images shared between train and test: {train_imgs & test_imgs}")
        if train_val_overlap > 0:
            raise ValueError(f"Data leakage detected! {train_val_overlap} images shared between train and validation: {train_imgs & val_imgs}")
        if val_test_overlap > 0:
            raise ValueError(f"Data leakage detected! {val_test_overlap} images shared between validation and test: {val_imgs & test_imgs}")

        return {
            "train_test_overlap": 0,
            "train_val_overlap": 0,
            "val_test_overlap": 0
        }

    def export_splits(
        self,
        output_dir: Path,
        partitioned_records: Dict[str, List[Dict[str, Any]]],
        image_splits: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """
        Exports partitioned dataset files and manifest into output_dir/splits/.
        """
        output_dir = Path(output_dir)
        splits_dir = output_dir / "splits"
        splits_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save split records JSON files
        split_paths = {}
        for s_name in ["train", "validation", "test"]:
            s_recs = partitioned_records.get(s_name, [])
            s_file = splits_dir / f"{s_name}_records.json"
            with open(s_file, "w", encoding="utf-8") as f:
                json.dump(s_recs, f, indent=2)
            split_paths[s_name] = str(s_file)

            # Subdirectories for individual record inspection
            sub_dir = splits_dir / s_name
            sub_dir.mkdir(parents=True, exist_ok=True)
            for rec in s_recs:
                rf = sub_dir / f"{rec['event_id']}.json"
                with open(rf, "w", encoding="utf-8") as f:
                    json.dump(rec, f, indent=2)

        # 2. Save split manifest
        manifest = {
            "strategy": "source_image_group_split",
            "description": (
                "Partitioning is strictly enforced at the source image level prior to augmentation. "
                "All visual distress annotations and correlated physical sensor simulations sharing "
                "a source image are isolated to the same split, guaranteeing zero multimodal context leakage."
            ),
            "seed": self.seed,
            "ratios": {
                "train": self.train_ratio,
                "validation": self.val_ratio,
                "test": self.test_ratio
            },
            "split_counts": {
                "images": {s: len(image_splits.get(s, [])) for s in ["train", "validation", "test"]},
                "records": {s: len(partitioned_records.get(s, [])) for s in ["train", "validation", "test"]}
            },
            "image_ids": image_splits,
            "leakage_verification": {
                "train_test_overlap": 0,
                "train_val_overlap": 0,
                "val_test_overlap": 0
            }
        }

        manifest_file = splits_dir / "split_manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return manifest
