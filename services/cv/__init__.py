"""
Computer Vision Package for Road Damage Detection
=================================================

Provides model abstractions, adapters, and preprocessing for RDD2022 distress detection.
"""

from .schemas import (
    DamageClassEnum,
    CLASS_LABELS,
    CLASS_SLUGS,
    BoundingBox,
    DamageDetection,
    DetectionResult
)
from .preprocessing import (
    load_image,
    letterbox,
    denormalize_box_from_letterbox
)
from .detector import (
    RoadDamageDetector,
    ModelAdapter,
    MockDetectorAdapter,
    GroundTruthLookupAdapter,
    OpenCVContourDetectorAdapter,
    UltralyticsYOLOAdapter
)
from .rdd2022_detector import RDD2022Detector

__all__ = [
    "DamageClassEnum",
    "CLASS_LABELS",
    "CLASS_SLUGS",
    "BoundingBox",
    "DamageDetection",
    "DetectionResult",
    "load_image",
    "letterbox",
    "denormalize_box_from_letterbox",
    "RoadDamageDetector",
    "ModelAdapter",
    "MockDetectorAdapter",
    "GroundTruthLookupAdapter",
    "OpenCVContourDetectorAdapter",
    "UltralyticsYOLOAdapter",
    "RDD2022Detector"
]
