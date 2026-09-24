"""
Computer Vision Schemas & Data Structures
=========================================

Defines standard detection formats, bounding box models, and damage classifications
for the RDD2022 road damage detection pipeline.
"""

from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field, field_validator


class DamageClassEnum(str, Enum):
    """Official RDD2022 Challenge Damage Classes."""
    D00 = "D00"  # Longitudinal Crack
    D10 = "D10"  # Transverse Crack
    D20 = "D20"  # Alligator (Fatigue) Crack
    D40 = "D40"  # Pothole


CLASS_LABELS: Dict[DamageClassEnum, str] = {
    DamageClassEnum.D00: "Longitudinal Crack",
    DamageClassEnum.D10: "Transverse Crack",
    DamageClassEnum.D20: "Alligator Crack",
    DamageClassEnum.D40: "Pothole"
}

CLASS_SLUGS: Dict[DamageClassEnum, str] = {
    DamageClassEnum.D00: "longitudinal_crack",
    DamageClassEnum.D10: "transverse_crack",
    DamageClassEnum.D20: "alligator_crack",
    DamageClassEnum.D40: "pothole"
}


class BoundingBox(BaseModel):
    """
    Axis-aligned 2D bounding box in pixel coordinates.
    [xmin, ymin, xmax, ymax]
    """
    xmin: float = Field(..., description="Top-left X coordinate in pixels")
    ymin: float = Field(..., description="Top-left Y coordinate in pixels")
    xmax: float = Field(..., description="Bottom-right X coordinate in pixels")
    ymax: float = Field(..., description="Bottom-right Y coordinate in pixels")

    @field_validator("xmax")
    @classmethod
    def check_xmax_greater(cls, v: float, info) -> float:
        if "xmin" in info.data and v <= info.data["xmin"]:
            raise ValueError(f"xmax ({v}) must be strictly greater than xmin ({info.data['xmin']})")
        return v

    @field_validator("ymax")
    @classmethod
    def check_ymax_greater(cls, v: float, info) -> float:
        if "ymin" in info.data and v <= info.data["ymin"]:
            raise ValueError(f"ymax ({v}) must be strictly greater than ymin ({info.data['ymin']})")
        return v

    @property
    def width(self) -> float:
        return max(0.0, self.xmax - self.xmin)

    @property
    def height(self) -> float:
        return max(0.0, self.ymax - self.ymin)

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_list(self) -> List[float]:
        return [round(self.xmin, 1), round(self.ymin, 1), round(self.xmax, 1), round(self.ymax, 1)]

    def to_yolo(self, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
        """Convert to normalized YOLO format (x_center, y_center, width, height) in [0, 1]."""
        xc = ((self.xmin + self.xmax) / 2.0) / img_w
        yc = ((self.ymin + self.ymax) / 2.0) / img_h
        w = self.width / img_w
        h = self.height / img_h
        return (
            max(0.0, min(1.0, xc)),
            max(0.0, min(1.0, yc)),
            max(0.0, min(1.0, w)),
            max(0.0, min(1.0, h))
        )


class DamageDetection(BaseModel):
    """
    A single detected road damage instance with bounding box and taxonomy classification.
    """
    class_id: str = Field(..., description="RDD2022 class code: D00, D10, D20, or D40")
    class_name: str = Field(..., description="Slug identifier (e.g. pothole)")
    class_label: str = Field(..., description="Human-readable label (e.g. Pothole)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model prediction confidence score")
    bbox: BoundingBox = Field(..., description="Bounding box coordinates in pixels")
    surface_area_sqm: Optional[float] = Field(None, ge=0.0, description="Estimated ground surface area in square meters")


class DetectionResult(BaseModel):
    """
    Complete computer vision inference response for an image.
    """
    image_id: Optional[str] = None
    image_path: Optional[str] = None
    image_width: int = Field(..., gt=0)
    image_height: int = Field(..., gt=0)
    detections: List[DamageDetection] = Field(default_factory=list)
    inference_time_ms: float = Field(0.0, ge=0.0)
    model_adapter: str = Field("unknown", description="Name of the active model adapter")
    device: str = Field("cpu", description="Execution device (cpu or cuda)")
