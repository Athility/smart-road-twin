"""
Multimodal Data Fusion Schemas
==============================

Unified data contracts for fusing visual defect detections with multimodal
sensors, environmental conditions, infrastructure context, and TOPSIS MCDM.
"""

from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timezone
from pydantic import BaseModel, Field

from services.cv.schemas import DamageClassEnum, CLASS_LABELS


class VisualEvidence(BaseModel):
    """Visual pavement distress evidence from RDD2022 image."""
    image_id: str = Field(..., description="RDD2022 source image identifier")
    damage_class: str = Field(..., description="Damage classification code (D00, D10, D20, D40)")
    damage_label: str = Field(..., description="Human-readable damage category")
    bbox: List[float] = Field(..., description="Bounding box [xmin, ymin, xmax, ymax]")
    confidence: Optional[float] = Field(None, description="Detection confidence score")
    surface_area_sqm: float = Field(..., description="Estimated ground surface area in square meters")
    image_width: int = Field(720, description="Image pixel width")
    image_height: int = Field(720, description="Image pixel height")
    source: str = Field("RDD2022", description="Dataset provenance for visual evidence")


class EnvironmentalEvidence(BaseModel):
    """Ambient environmental conditions dictating sensor modality selection."""
    rain_detected: bool = Field(False, description="Presence of precipitation or standing water")
    mean_luminance: float = Field(65.0, description="Average scene illuminance in lux / IRE")
    weather_condition: str = Field("clear", description="Categorical weather status")
    ambient_lux: float = Field(350.0, description="Ambient light level")
    source: str = Field("synthetic_or_sensor", description="Environmental evidence provenance")


class SensorEvidence(BaseModel):
    """Multimodal depth, acoustic, and dynamic vertical acceleration telemetry."""
    lidar_depth_cm: float = Field(..., description="Raw optical distance from chassis to defect floor (cm)")
    sonar_depth_cm: float = Field(..., description="Raw acoustic distance from chassis to defect floor (cm)")
    active_modality: str = Field(..., description="Active sensor selected by switching logic ('optical_lidar' or 'sonar_submerged_acoustic')")
    effective_depth_cm: float = Field(..., description="Effective cavity depth below pavement grade (cm)")
    calculated_volume_liters: float = Field(..., description="Calculated geometric defect volume in liters")
    z_accel_g: float = Field(..., description="Peak vertical acceleration impact shock in g")
    dynamic_impact_confirmed: bool = Field(..., description="True if vertical acceleration exceeds 1.50g threshold")
    impact_regime: str = Field("no_impact", description="Physical impact regime ('confirmed_impact', 'borderline', 'no_impact')")
    sensor_source: str = Field("synthetic", description="Provenance marker for sensor telemetry")
    generation_method: str = Field("controlled_distribution", description="Sensor synthesis methodology")


class InfrastructureContext(BaseModel):
    """GIS, road hierarchy, traffic density, and municipal criticality context."""
    latitude: float = Field(..., description="Geographic latitude coordinate")
    longitude: float = Field(..., description="Geographic longitude coordinate")
    road_segment_id: Optional[str] = Field(None, description="Identifier of the road segment")
    road_class: str = Field("ARTERIAL", description="Functional classification (LOCAL, COLLECTOR, ARTERIAL, HIGH_TRAFFIC_CORRIDOR)")
    traffic_pcu: int = Field(..., description="Traffic volume in Passenger Car Units")
    dist_hospital_km: float = Field(..., description="Geodesic distance to nearest hospital (km)")
    nearest_hospital: str = Field("General Hospital", description="Name of closest medical facility")
    speed_kmh: float = Field(40.0, description="Operating vehicle speed (km/h)")
    location_mode: str = Field("synthetic_location", description="Location coordinate mode")
    location_source: str = Field("synthetic", description="Provenance marker for coordinates")
    traffic_source: str = Field("synthetic", description="Provenance marker for traffic data")
    hospital_source: str = Field("synthetic", description="Provenance marker for hospital GIS data")


class TOPSISAssessment(BaseModel):
    """Multi-Criteria Decision Making priority ranking."""
    criteria_vector: List[float] = Field(..., description="[volume_liters, traffic_pcu, dist_hospital_km, speed_kmh]")
    topsis_score: float = Field(..., description="Relative closeness to ideal solution ∈ [0.0, 1.0]")
    priority_level: str = Field(..., description="Prioritization category ('Critical', 'High', 'Medium', 'Low')")
    rank: Optional[int] = Field(None, description="Relative priority rank among evaluated alternatives")


class RoadDefectEvent(BaseModel):
    """
    Unified Road Defect Event
    -------------------------
    Synthesizes authentic visual damage evidence, physical sensor telemetry,
    environmental state, and municipal infrastructure context into a single
    coherent digital twin event ready for TOPSIS prioritization.
    """
    event_id: str = Field(..., description="Unique event identifier")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    visual: VisualEvidence
    environmental: EnvironmentalEvidence
    sensor: SensorEvidence
    infrastructure: InfrastructureContext
    physical_consistency: Dict[str, Any] = Field(default_factory=dict, description="Latent severity and consistency invariants")
    evidence_summary: Optional[Dict[str, Any]] = Field(None, description="Detailed multi-sensor evidence breakdown and disagreement diagnosis")
    topsis: Optional[TOPSISAssessment] = Field(None, description="TOPSIS MCDM prioritization assessment")
    provenance: Dict[str, Any] = Field(default_factory=dict, description="Complete provenance audit trail")

    def to_telemetry_dict(self) -> Dict[str, Any]:
        """
        Converts the unified event into the flat dictionary format consumed
        by the main.py FastAPI telemetry store and Next.js digital twin map.
        """
        return {
            "id": self.event_id,
            "timestamp": self.timestamp,
            "vehicle_id": "INSPECTION_FLEET_01",
            "latitude": self.infrastructure.latitude,
            "longitude": self.infrastructure.longitude,
            "damage_class": self.visual.damage_class,
            "damage_label": self.visual.damage_label,
            "rain_detected": self.environmental.rain_detected,
            "mean_luminance": self.environmental.mean_luminance,
            "lidar_depth_cm": self.sensor.lidar_depth_cm,
            "sonar_depth_cm": self.sensor.sonar_depth_cm,
            "z_accel_g": self.sensor.z_accel_g,
            "surface_area_sqm": self.visual.surface_area_sqm,
            "traffic_pcu": self.infrastructure.traffic_pcu,
            "dist_hospital_km": self.infrastructure.dist_hospital_km,
            "speed_kmh": self.infrastructure.speed_kmh,
            "processed_telemetry": {
                "detection_modality": self.sensor.active_modality,
                "is_submerged": self.environmental.rain_detected,
                "estimated_depth_cm": self.sensor.effective_depth_cm,
                "calculated_volume_liters": self.sensor.calculated_volume_liters,
                "dynamic_impact_confirmed": self.sensor.dynamic_impact_confirmed
            },
            "topsis_score": self.topsis.topsis_score if self.topsis else 0.0,
            "priority_level": self.topsis.priority_level if self.topsis else "Low",
            "provenance": self.provenance
        }
