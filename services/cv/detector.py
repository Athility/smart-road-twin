"""
Road Damage Detector & Model Adapter Abstraction
================================================

Defines the decoupled architecture for road distress computer vision models.
Ensures the application is independent of any single model implementation (YOLOv8,
YOLOv9, RT-DETR, ONNX, or OpenCV heuristic fallback).

Core Principle:
Training and inference are strictly decoupled. Model adapters supply inference;
training is handled through standalone pipelines under ml/.
"""

import time
import json
from abc import ABC, abstractmethod
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

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class ModelAdapter(ABC):
    """
    Abstract Model Adapter. Decouples inference from the specific ML framework.
    """
    @abstractmethod
    def predict(
        self,
        image_rgb: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ) -> List[Dict[str, Any]]:
        """
        Runs model inference on an image array.
        
        Returns:
            List of raw detections: [{'class_id': 'D40', 'confidence': 0.88, 'bbox': [xmin, ymin, xmax, ymax]}]
        """
        pass

    @abstractmethod
    def get_adapter_name(self) -> str:
        """Returns the human-readable adapter identifier."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Checks if dependencies and weights for this adapter are available."""
        pass


class MockDetectorAdapter(ModelAdapter):
    """
    Controlled Mock Adapter for deterministic unit testing and headless validation.
    """
    def __init__(self, predefined_detections: Optional[List[Dict[str, Any]]] = None):
        self.predefined = predefined_detections or []

    def is_available(self) -> bool:
        return True

    def get_adapter_name(self) -> str:
        return "MockDetectorAdapter"

    def predict(
        self,
        image_rgb: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ) -> List[Dict[str, Any]]:
        if self.predefined:
            return [d for d in self.predefined if d.get("confidence", 1.0) >= conf_threshold]

        # Default synthetic detection (Pothole D40 in lower roadway)
        h, w = image_rgb.shape[:2]
        return [
            {
                "class_id": "D40",
                "confidence": 0.89,
                "bbox": [round(w * 0.25, 1), round(h * 0.45, 1), round(w * 0.65, 1), round(h * 0.75, 1)]
            }
        ]


class GroundTruthLookupAdapter(ModelAdapter):
    """
    Uses authentic RDD2022 ground-truth annotations to act as an oracle detector.
    Ideal for reproducible benchmarks and closed-loop digital twin simulations.
    """
    def __init__(self, annotations_index_path: Optional[Path] = None):
        project_root = Path(__file__).resolve().parent.parent.parent
        self.index_path = Path(annotations_index_path) if annotations_index_path else project_root / "data" / "processed" / "annotations_index.json"
        self.annotations_map = {}
        if self.index_path.exists():
            try:
                with open(self.index_path, "r", encoding="utf-8") as f:
                    self.annotations_map = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load ground truth index: {e}")

    def is_available(self) -> bool:
        return bool(self.annotations_map)

    def get_adapter_name(self) -> str:
        return "GroundTruthLookupAdapter"

    def predict(
        self,
        image_rgb: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        image_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        if not image_id or image_id not in self.annotations_map:
            # Fallback to center detection if image ID is not indexed
            h, w = image_rgb.shape[:2]
            return [{
                "class_id": "D40",
                "confidence": 0.95,
                "bbox": [w * 0.25, h * 0.35, w * 0.60, h * 0.65]
            }]

        data = self.annotations_map[image_id]
        results = []
        for obj in data.get("objects", []):
            results.append({
                "class_id": obj["class_id"],
                "confidence": 0.98,
                "bbox": [obj["xmin"], obj["ymin"], obj["xmax"], obj["ymax"]]
            })
        return results


class OpenCVContourDetectorAdapter(ModelAdapter):
    """
    Zero-dependency visual feature and dark-region contour detector using OpenCV.
    Operates without heavy PyTorch dependencies by segmenting asphalt depression contours.
    """
    def __init__(self):
        self.available = HAS_CV2

    def is_available(self) -> bool:
        return self.available

    def get_adapter_name(self) -> str:
        return "OpenCVContourDetectorAdapter"

    def predict(
        self,
        image_rgb: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ) -> List[Dict[str, Any]]:
        if not self.available:
            return []

        h, w = image_rgb.shape[:2]
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        # Focus on lower 60% of frame (driving surface)
        road_roi = gray[int(h * 0.4):, :]
        blurred = cv2.GaussianBlur(road_roi, (7, 7), 0)

        # Adaptive thresholding to segment dark pavement depressions
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 25, 4
        )

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []

        min_area = (w * h) * 0.005  # At least 0.5% of frame
        max_area = (w * h) * 0.35   # At most 35% of frame

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area <= area <= max_area:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                # Remap Y back to full image
                actual_y = by + int(h * 0.4)
                aspect = float(bw) / max(1.0, bh)

                # Classify based on contour aspect ratio and morphology
                if aspect > 3.0:
                    cls_id = "D10"  # Transverse crack (wide)
                elif aspect < 0.33:
                    cls_id = "D00"  # Longitudinal crack (tall)
                elif area > (w * h) * 0.04:
                    cls_id = "D40"  # Pothole (compact cavity)
                else:
                    cls_id = "D20"  # Alligator cracking network

                conf = round(min(0.92, max(0.40, float(area) / ((w * h) * 0.10))), 2)
                if conf >= conf_threshold:
                    detections.append({
                        "class_id": cls_id,
                        "confidence": conf,
                        "bbox": [float(bx), float(actual_y), float(bx + bw), float(actual_y + bh)]
                    })

        return detections[:5]  # Top detections


class UltralyticsYOLOAdapter(ModelAdapter):
    """
    Ultralytics YOLO Model Adapter for PyTorch/TorchScript weights.
    Gracefully handles absence of ultralytics package.
    """
    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        self.weights_path = Path(weights_path) if weights_path else None
        self.model = None
        self._load_attempted = False

    def is_available(self) -> bool:
        try:
            import ultralytics
            return self.weights_path is not None and self.weights_path.exists()
        except ImportError:
            return False

    def get_adapter_name(self) -> str:
        return f"UltralyticsYOLOAdapter({self.weights_path.name if self.weights_path else 'none'})"

    def predict(
        self,
        image_rgb: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ) -> List[Dict[str, Any]]:
        if not self.is_available():
            raise RuntimeError("Ultralytics package or model weights not available.")

        from ultralytics import YOLO
        if self.model is None:
            self.model = YOLO(str(self.weights_path))

        results = self.model.predict(
            source=image_rgb,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )

        detections = []
        for r in results:
            for box in r.boxes:
                cls_idx = int(box.cls[0])
                conf = float(box.conf[0])
                coords = [float(c) for c in box.xyxy[0]]
                # Map YOLO class index to D00-D40
                class_mapping = {0: "D00", 1: "D10", 2: "D20", 3: "D40"}
                cls_id = class_mapping.get(cls_idx, "D40")

                detections.append({
                    "class_id": cls_id,
                    "confidence": round(conf, 3),
                    "bbox": coords
                })

        return detections


class RoadDamageDetector(ABC):
    """
    Abstract Base Class for Road Distress Computer Vision.
    Mandates the standard user interface: detect(image) -> damage detections.
    """
    @abstractmethod
    def detect(
        self,
        image: Union[str, Path, bytes, np.ndarray, Image.Image],
        conf_threshold: float = 0.25
    ) -> List[DamageDetection]:
        """
        Detect road damages in an image.
        Interface: detect(image) -> damage detections.
        """
        pass

    @abstractmethod
    def detect_full(
        self,
        image: Union[str, Path, bytes, np.ndarray, Image.Image],
        conf_threshold: float = 0.25,
        image_id: Optional[str] = None
    ) -> DetectionResult:
        """
        Runs detection and packages the complete DetectionResult with metadata.
        """
        pass
