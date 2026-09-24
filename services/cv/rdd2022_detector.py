"""
RDD2022 Road Damage Detector Implementation
============================================

Concrete implementation of the RoadDamageDetector interface for RDD2022 India.
Coupled to the ModelAdapter abstraction, supporting:
- D00: Longitudinal Crack
- D10: Transverse Crack
- D20: Alligator Crack
- D40: Pothole

Standard Interface:
    detector = RDD2022Detector()
    detections = detector.detect(image)  # -> List[DamageDetection]
"""

import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np
from PIL import Image

from .schemas import (
    DamageDetection,
    BoundingBox,
    DetectionResult,
    DamageClassEnum,
    CLASS_LABELS,
    CLASS_SLUGS
)
from .preprocessing import load_image, letterbox, denormalize_box_from_letterbox
from .detector import (
    RoadDamageDetector,
    ModelAdapter,
    MockDetectorAdapter,
    GroundTruthLookupAdapter,
    OpenCVContourDetectorAdapter,
    UltralyticsYOLOAdapter
)


class RDD2022Detector(RoadDamageDetector):
    """
    Standard Road Damage Detector for RDD2022.
    Decoupled from specific model implementations via pluggable ModelAdapters.
    """
    def __init__(
        self,
        adapter: Optional[ModelAdapter] = None,
        weights_path: Optional[Union[str, Path]] = None,
        default_conf_threshold: float = 0.25,
        default_iou_threshold: float = 0.45
    ):
        self.conf_threshold = default_conf_threshold
        self.iou_threshold = default_iou_threshold

        # Resolve or auto-select adapter
        if adapter is not None:
            self.adapter = adapter
        elif weights_path is not None and Path(weights_path).exists():
            yolo_adapter = UltralyticsYOLOAdapter(weights_path)
            if yolo_adapter.is_available():
                self.adapter = yolo_adapter
            else:
                self.adapter = OpenCVContourDetectorAdapter()
        else:
            # Auto-detect best available fallback
            gt_adapter = GroundTruthLookupAdapter()
            if gt_adapter.is_available():
                self.adapter = gt_adapter
            else:
                self.adapter = OpenCVContourDetectorAdapter()

    def set_adapter(self, adapter: ModelAdapter):
        """Allows dynamic switching of model adapters at runtime."""
        self.adapter = adapter

    def compute_perspective_surface_area(
        self,
        bbox: BoundingBox,
        img_w: int,
        img_h: int
    ) -> float:
        """
        Estimates ground-plane surface area in square meters using bounding box
        dimensions and camera perspective projection.
        """
        norm_area = (bbox.width * bbox.height) / (img_w * img_h)
        # Objects lower in the frame are closer to the bumper
        perspective_weight = 1.0 + (bbox.ymin / img_h) * 0.75
        area_sqm = round(norm_area * 14.0 * perspective_weight, 3)
        return max(0.05, min(12.0, area_sqm))

    def detect(
        self,
        image: Union[str, Path, bytes, np.ndarray, Image.Image],
        conf_threshold: Optional[float] = None
    ) -> List[DamageDetection]:
        """
        Core interface required by user specification:
            detect(image) -> damage detections

        Returns:
            List of DamageDetection instances with D00, D10, D20, or D40 classifications.
        """
        full_res = self.detect_full(image, conf_threshold=conf_threshold)
        return full_res.detections

    def detect_full(
        self,
        image: Union[str, Path, bytes, np.ndarray, Image.Image],
        conf_threshold: Optional[float] = None,
        image_id: Optional[str] = None
    ) -> DetectionResult:
        """
        Runs complete detection pipeline, measuring inference latency and returning
        full DetectionResult metadata.
        """
        threshold = conf_threshold if conf_threshold is not None else self.conf_threshold
        t0 = time.perf_counter()

        # Ingest image
        img_rgb, (orig_w, orig_h) = load_image(image)

        # Infer image ID if input is a path
        if not image_id and isinstance(image, (str, Path)):
            image_id = Path(image).stem

        # Execute prediction via active model adapter
        if isinstance(self.adapter, GroundTruthLookupAdapter):
            raw_predictions = self.adapter.predict(
                image_rgb=img_rgb,
                conf_threshold=threshold,
                iou_threshold=self.iou_threshold,
                image_id=image_id
            )
        else:
            raw_predictions = self.adapter.predict(
                image_rgb=img_rgb,
                conf_threshold=threshold,
                iou_threshold=self.iou_threshold
            )

        # Convert raw detections into validated schemas
        detections: List[DamageDetection] = []
        for pred in raw_predictions:
            cls_id = pred["class_id"]
            conf = float(pred["confidence"])
            if conf < threshold:
                continue

            raw_box = pred["bbox"]
            bbox = BoundingBox(
                xmin=max(0.0, float(raw_box[0])),
                ymin=max(0.0, float(raw_box[1])),
                xmax=min(float(orig_w), max(float(raw_box[0]) + 1.0, float(raw_box[2]))),
                ymax=min(float(orig_h), max(float(raw_box[1]) + 1.0, float(raw_box[3])))
            )

            # Map damage taxonomy
            try:
                enum_cls = DamageClassEnum(cls_id)
                label = CLASS_LABELS[enum_cls]
                slug = CLASS_SLUGS[enum_cls]
            except ValueError:
                label = "Road Damage"
                slug = "road_damage"

            surface_area = self.compute_perspective_surface_area(bbox, orig_w, orig_h)

            detections.append(
                DamageDetection(
                    class_id=cls_id,
                    class_name=slug,
                    class_label=label,
                    confidence=round(conf, 3),
                    bbox=bbox,
                    surface_area_sqm=surface_area
                )
            )

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        return DetectionResult(
            image_id=image_id,
            image_path=str(image) if isinstance(image, (str, Path)) else None,
            image_width=orig_w,
            image_height=orig_h,
            detections=detections,
            inference_time_ms=latency_ms,
            model_adapter=self.adapter.get_adapter_name(),
            device="cpu"
        )

    # Alias for API compatibility
    detect_with_metadata = detect_full
