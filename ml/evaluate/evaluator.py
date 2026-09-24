"""
Damage Model Evaluator
======================

Evaluates RoadDamageDetector implementations against RDD2022 ground truth data.
Reports:
- Precision & Recall
- mAP50 & mAP50-95
- Class-wise performance (D00, D10, D20, D40)
- Full Confusion Matrix (including background / FP / FN)

Strict Scientific Separation:
Evaluation strictly uses visual ground truth annotations. Synthetic sensor
values are not evaluated here.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_rdd2022 import parse_voc_xml
from services.cv.detector import RoadDamageDetector
from services.cv.rdd2022_detector import RDD2022Detector
from ml.evaluate.metrics import evaluate_dataset_metrics, CLASS_LIST


class DamageModelEvaluator:
    """
    Evaluator executing test/validation split inference and computing
    comprehensive CV benchmark metrics.
    """

    def __init__(
        self,
        detector: Optional[RoadDamageDetector] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.50,
        project_root: Optional[Path] = None
    ):
        self.project_root = Path(project_root) if project_root else PROJECT_ROOT
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.detector = detector if detector is not None else RDD2022Detector(default_conf_threshold=conf_threshold)

    def evaluate_split(
        self,
        split_name: str = "test",
        yolo_dataset_dir: Optional[Path] = None,
        raw_annotations_dir: Optional[Path] = None,
        raw_images_dir: Optional[Path] = None,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Runs evaluation over a given dataset split.
        Prefers YOLO split directory if available; falls back to raw Pascal VOC dataset.
        """
        ground_truths: Dict[str, List[Dict[str, Any]]] = {}
        predictions: Dict[str, List[Dict[str, Any]]] = {}

        # Check for YOLO split directory
        yolo_dir = Path(yolo_dataset_dir) if yolo_dataset_dir else self.project_root / "data" / "processed" / "yolo"
        split_images_dir = yolo_dir / "images" / split_name
        split_labels_dir = yolo_dir / "labels" / split_name

        if split_images_dir.exists() and any(split_images_dir.glob("*.jpg")):
            # Load from YOLO directory
            image_files = sorted(list(split_images_dir.glob("*.jpg")))
            if limit:
                image_files = image_files[:limit]

            for img_p in image_files:
                img_id = img_p.stem
                label_p = split_labels_dir / f"{img_id}.txt"

                # Parse YOLO label file
                gt_boxes = []
                if label_p.exists():
                    # Need image dimensions for absolute bounding box
                    from PIL import Image
                    with Image.open(img_p) as im:
                        w, h = im.size

                    with open(label_p, "r", encoding="utf-8") as lf:
                        for line in lf:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                cls_idx = int(parts[0])
                                if 0 <= cls_idx < len(CLASS_LIST):
                                    cls_name = CLASS_LIST[cls_idx]
                                    xc, yc, bw, bh = map(float, parts[1:5])
                                    xmin = (xc - bw / 2.0) * w
                                    ymin = (yc - bh / 2.0) * h
                                    xmax = (xc + bw / 2.0) * w
                                    ymax = (yc + bh / 2.0) * h
                                    gt_boxes.append({
                                        "class_id": cls_name,
                                        "bbox": [xmin, ymin, xmax, ymax]
                                    })
                ground_truths[img_id] = gt_boxes

                # Run detector
                dets = self.detector.detect(img_p, conf_threshold=self.conf_threshold)
                pred_list = []
                for d in dets:
                    pred_list.append({
                        "class_id": d.class_id,
                        "confidence": d.confidence,
                        "bbox": [d.bbox.xmin, d.bbox.ymin, d.bbox.xmax, d.bbox.ymax]
                    })
                predictions[img_id] = pred_list

        else:
            # Fall back to raw Pascal VOC directory
            xmls_dir = Path(raw_annotations_dir) if raw_annotations_dir else self.project_root / "data" / "raw" / "rdd2022_india" / "annotations"
            imgs_dir = Path(raw_images_dir) if raw_images_dir else self.project_root / "data" / "raw" / "rdd2022_india" / "images"

            xml_files = sorted(list(xmls_dir.glob("*.xml")))
            if limit:
                xml_files = xml_files[:limit]

            for xf in xml_files:
                parsed = parse_voc_xml(xf)
                if not parsed:
                    continue
                img_id = xf.stem
                img_name = parsed.get("filename", f"{img_id}.jpg")
                img_p = imgs_dir / img_name

                gt_boxes = []
                for obj in parsed.get("objects", []):
                    cid = obj["class_id"]
                    if cid in CLASS_LIST:
                        gt_boxes.append({
                            "class_id": cid,
                            "bbox": [float(obj["xmin"]), float(obj["ymin"]), float(obj["xmax"]), float(obj["ymax"])]
                        })
                ground_truths[img_id] = gt_boxes

                # Run detector
                if img_p.exists():
                    dets = self.detector.detect(img_p, conf_threshold=self.conf_threshold)
                else:
                    dets = []

                pred_list = []
                for d in dets:
                    pred_list.append({
                        "class_id": d.class_id,
                        "confidence": d.confidence,
                        "bbox": [d.bbox.xmin, d.bbox.ymin, d.bbox.xmax, d.bbox.ymax]
                    })
                predictions[img_id] = pred_list

        # Compute benchmark metrics
        metrics = evaluate_dataset_metrics(
            ground_truths=ground_truths,
            predictions=predictions,
            iou_threshold=self.iou_threshold,
            conf_threshold=self.conf_threshold
        )

        metrics["split"] = split_name
        metrics["total_images_evaluated"] = len(ground_truths)
        metrics["total_ground_truth_boxes"] = sum(len(b) for b in ground_truths.values())
        metrics["total_predicted_boxes"] = sum(len(b) for b in predictions.values())

        return metrics

    def format_confusion_matrix_ascii(self, cm: List[List[int]], labels: List[str]) -> str:
        prefix = "GT \\ Pred"
        header = f"{prefix:<14}" + "".join([f"{lbl:>12}" for lbl in labels])
        lines = [header, "-" * len(header)]
        for i, row_lbl in enumerate(labels):
            row_str = f"{row_lbl:<14}" + "".join([f"{cm[i][j]:>12}" for j in range(len(labels))])
            lines.append(row_str)
        return "\n".join(lines)

    def save_report(self, metrics: Dict[str, Any], output_path: Union[str, Path]) -> None:
        """Saves evaluation report to JSON."""
        output_p = Path(output_path)
        output_p.parent.mkdir(parents=True, exist_ok=True)
        with open(output_p, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
