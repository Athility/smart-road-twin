"""
Sensor Augmentation Engine Orchestrator
=======================================

Master coordination pipeline for the Smart Road Digital Twin multimodal augmentation layer.
Combines:
1. LiDAR Optical Profilometry (lidar_model.py)
2. Sonar Acoustic Profilometry (sonar_model.py)
3. Quarter-Car Accelerometer Dynamic Impact (accelerometer_model.py)
4. Highway Capacity & Urban PCU Traffic Flow (traffic_model.py)
5. Municipal Critical Infrastructure & Hospital Proximity (infrastructure_model.py)

Strict Provenance Policy:
Visual ground truth is strictly preserved from RDD2022 India.
All sensor, environmental, dynamic, and urban context dimensions are generated through
physically correlated models with auditable provenance metadata.
"""

import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from .lidar_model import LiDARModel
from .sonar_model import SonarModel
from .accelerometer_model import AccelerometerModel
from .traffic_model import TrafficModel
from .infrastructure_model import InfrastructureModel
from .location_model import LocationModel, MODE_SYNTHETIC_LOCATION
from .consistency_model import PhysicalConsistencyModel

CLASS_TAXONOMY = {
    "D00": {"class_name": "longitudinal_crack", "label": "Longitudinal Crack"},
    "D10": {"class_name": "transverse_crack", "label": "Transverse Crack"},
    "D20": {"class_name": "alligator_crack", "label": "Alligator Crack"},
    "D40": {"class_name": "pothole", "label": "Pothole"}
}


class SensorAugmentationEngine:
    def __init__(self, seed: int = 2026, baseline_chassis_cm: float = 30.0):
        self.seed = seed
        self.baseline_chassis_cm = baseline_chassis_cm
        self.rng = random.Random(seed)

        # Initialize submodels with deterministically derived seeds
        self.lidar = LiDARModel(baseline_chassis_cm=baseline_chassis_cm, seed=seed + 1)
        self.sonar = SonarModel(baseline_chassis_cm=baseline_chassis_cm, seed=seed + 2)
        self.accel = AccelerometerModel(seed=seed + 3)
        self.traffic = TrafficModel(seed=seed + 4)
        self.infra = InfrastructureModel(seed=seed + 5)
        self.location = LocationModel(mode=MODE_SYNTHETIC_LOCATION, seed=seed)
        self.consistency = PhysicalConsistencyModel(seed=seed, baseline_chassis_cm=baseline_chassis_cm)

    def reseed(self, seed: int):
        self.seed = seed
        self.rng = random.Random(seed)
        self.lidar.reseed(seed + 1)
        self.sonar.reseed(seed + 2)
        self.accel.reseed(seed + 3)
        self.traffic.reseed(seed + 4)
        self.infra.reseed(seed + 5)
        self.location.reseed(seed)
        self.consistency.reseed(seed)

    def compute_surface_area_sqm(
        self,
        bbox: List[float],
        image_dims: Dict[str, int]
    ) -> float:
        """
        Estimate ground-plane surface area in square meters using bounding box dimensions
        and perspective projection camera geometry.
        """
        xmin, ymin, xmax, ymax = bbox
        width = max(1, image_dims.get("width", 720))
        height = max(1, image_dims.get("height", 720))

        box_w = max(1.0, xmax - xmin)
        box_h = max(1.0, ymax - ymin)
        norm_area = (box_w * box_h) / (width * height)

        # Lower bounding box coordinates indicate roadway closer to vehicle bumper
        perspective_factor = 1.0 + (ymin / height) * 0.75
        area_sqm = round(norm_area * 14.0 * perspective_factor, 3)
        return max(0.05, min(12.0, area_sqm))

    def augment_defect(
        self,
        image_id: str,
        filename: str,
        damage_class: str,
        bbox: List[float],
        image_dims: Dict[str, int],
        annotation_idx: int = 0,
        gps_coords: Optional[Tuple[float, float]] = None,
        road_segment_id: Optional[str] = None,
        road_name: Optional[str] = None,
        weather_mode: str = "auto"
    ) -> Dict[str, Any]:
        """
        Produce a full, physically plausible multimodal SRMD record for a single defect.
        """
        tax = CLASS_TAXONOMY.get(damage_class, {"class_name": "pothole", "label": "Pothole"})
        surface_area_sqm = self.consistency.compute_perspective_surface_area(bbox, image_dims)

        # 1. Latent Defect Severity
        defect_severity = self.consistency.derive_latent_severity(damage_class, surface_area_sqm)

        # 2. Physics-grounded Depth and Volume
        defect_depth_cm, volume_liters, severity_tier = self.consistency.derive_depth_and_volume(
            defect_severity=defect_severity,
            damage_class=damage_class,
            surface_area_sqm=surface_area_sqm
        )

        # 3. Environmental conditions
        if weather_mode == "rain":
            is_rain = True
            lux = round(self.rng.uniform(14.0, 32.0), 1)
            weather_desc = "Monsoon Rain (Submerged Road)"
        elif weather_mode == "dry_day":
            is_rain = False
            lux = round(self.rng.uniform(60.0, 95.0), 1)
            weather_desc = "Clear / Dry Daylight"
        elif weather_mode == "low_light":
            is_rain = False
            lux = round(self.rng.uniform(16.0, 36.0), 1)
            weather_desc = "Low Luminance / Night"
        else:
            # Stochastic Mumbai climate profile (30% rain/wet, 70% dry)
            is_rain = self.rng.random() < 0.30
            if is_rain:
                lux = round(self.rng.uniform(14.0, 32.0), 1)
                weather_desc = "Monsoon Rain (Submerged Road)"
            else:
                is_day = self.rng.random() < 0.85
                lux = round(self.rng.uniform(55.0, 95.0) if is_day else self.rng.uniform(18.0, 38.0), 1)
                weather_desc = "Clear / Dry Daylight" if is_day else "Low Luminance / Night"

        # 4. Sensor measurements & Modality selection
        sensor_eval = self.consistency.derive_sensor_measurements(
            defect_depth_cm=defect_depth_cm,
            rain_detected=is_rain,
            luminance_lux=lux
        )

        # 5. Urban Location & Road Context
        if gps_coords:
            loc_res = {
                "latitude": gps_coords[0],
                "longitude": gps_coords[1],
                "road_segment_id": road_segment_id or "EXT-SEGMENT-01",
                "road_name": road_name or "Custom GPS Roadway",
                "road_class": "ARTERIAL",
                "location_mode": "external_gis_location",
                "location_source": "custom_coords",
                "chainage_km": 0.0,
                "demonstration_area": "Custom Coordinates"
            }
        else:
            loc_res = self.location.generate_location_for_defect(route_index=annotation_idx)

        lat = loc_res["latitude"]
        lon = loc_res["longitude"]
        road_class = loc_res.get("road_class", "HIGH_TRAFFIC_CORRIDOR")
        seg_id = road_segment_id or loc_res.get("road_segment_id", "WEH-MUM-SEC-01")
        r_name = road_name or loc_res.get("road_name", "Western Express Highway")

        # 6. Municipal Infrastructure & Hospital Proximity
        infra_res = self.infra.simulate(latitude=lat, longitude=lon)

        # 7. Traffic Flow & Vehicle Speed (correlated with roadway location)
        traffic_res = self.traffic.simulate(
            road_class=road_class,
            latitude=lat,
            longitude=lon,
            road_segment_id=seg_id,
            road_name=r_name,
            peak_hour=True
        )
        speed_kmh = traffic_res["operating_speed_kmh"]

        # 8. Quarter-Car Accelerometer Dynamic Impact
        accel_res = self.consistency.derive_accelerometer_impact(
            defect_severity=defect_severity,
            defect_depth_cm=defect_depth_cm,
            surface_area_sqm=surface_area_sqm,
            vehicle_speed_kmh=speed_kmh,
            damage_class=damage_class
        )

        # 9. Infrastructure Risk / TOPSIS Composite
        risk_res = self.consistency.derive_infrastructure_risk(
            defect_severity=defect_severity,
            defect_depth_cm=defect_depth_cm,
            surface_area_sqm=surface_area_sqm,
            traffic_pcu=traffic_res["traffic_pcu"],
            vehicle_speed_kmh=speed_kmh,
            dist_hospital_km=infra_res["dist_hospital_km"]
        )

        event_id = f"SRMD-IND-{image_id.replace('India_', '')}-{annotation_idx:02d}"

        # 10. Complete SRMD Record Assembly
        return {
            "event_id": event_id,
            "source": {
                "dataset": "RDD2022",
                "country": "India",
                "image_id": image_id,
                "annotation_id": f"{image_id}_obj{annotation_idx}",
                "annotation_class": damage_class
            },
            "visual": {
                "damage_class": tax["class_name"],
                "damage_label": tax["label"],
                "confidence": 1.0,
                "bbox": [round(c, 1) for c in bbox],
                "image_width": image_dims.get("width", 720),
                "image_height": image_dims.get("height", 720),
                "image_filename": filename,
                "image_url": f"/static/rdd2022/images/{filename}",
                "surface_area_sqm": surface_area_sqm,
                "defect_severity": defect_severity,
                "calculated_volume_liters": volume_liters
            },
            "location": {
                "latitude": lat,
                "longitude": lon,
                "road_segment_id": seg_id,
                "road_name": r_name,
                "road_class": road_class,
                "chainage_km": loc_res.get("chainage_km", 0.0),
                "location_mode": loc_res.get("location_mode", "synthetic_location"),
                "location_source": loc_res.get("location_source", "synthetic"),
                "demonstration_area": loc_res.get("demonstration_area", "Mumbai Metropolitan Urban Corridor"),
                "synthesis_seed": loc_res.get("synthesis_seed", self.seed)
            },
            "lidar": {
                "depth_cm": sensor_eval["raw_lidar_cm"],
                "effective_depth_cm": defect_depth_cm,
                "severity_tier": severity_tier,
                "confidence": 0.92 if not is_rain else 0.45,
                "sensor_source": "synthetic",
                "generation_method": "controlled_distribution",
                "source": "synthetic"
            },
            "sonar": {
                "depth_cm": sensor_eval["raw_sonar_cm"],
                "effective_depth_cm": sensor_eval["sonar_effective_depth_cm"],
                "confidence": 0.94 if is_rain else 0.88,
                "is_submerged_echo": sensor_eval["is_submerged"],
                "sensor_error_cm": sensor_eval["sonar_error_cm"],
                "sensor_source": "synthetic",
                "generation_method": "controlled_distribution",
                "source": "synthetic"
            },
            "accelerometer": {
                "z_accel_g": accel_res["z_accel_g"],
                "dynamic_impact_triggered": accel_res["dynamic_impact_triggered"],
                "impact_regime": accel_res["impact_regime"],
                "impact_category": accel_res["impact_category"],
                "cross_modal_assessment": accel_res["cross_modal_assessment"],
                "threshold_g": accel_res["threshold_g"],
                "borderline_min_g": accel_res["borderline_min_g"],
                "rms_vibration_g": accel_res["rms_vibration_g"],
                "suspension_deflection_cm": accel_res["suspension_deflection_cm"],
                "vehicle_speed_kmh": accel_res["vehicle_speed_kmh"],
                "sensor_source": "synthetic",
                "generation_method": "controlled_distribution",
                "source": "synthetic"
            },
            "traffic": {
                "road_class": traffic_res["road_class"],
                "road_class_name": traffic_res.get("road_class_name", traffic_res["road_class"]),
                "traffic_pcu": traffic_res["traffic_pcu"],
                "pcu": traffic_res["pcu"],
                "traffic_source": "synthetic",
                "level_of_service": traffic_res["level_of_service"],
                "volume_capacity_ratio": traffic_res["volume_capacity_ratio"],
                "roadway_capacity_pcu": traffic_res["roadway_capacity_pcu"],
                "sensor_source": "synthetic",
                "generation_method": "controlled_distribution",
                "source": "synthetic"
            },
            "infrastructure": {
                "dist_hospital_km": infra_res["dist_hospital_km"],
                "hospital_distance_km": infra_res["hospital_distance_km"],
                "geodesic_distance_km": infra_res["geodesic_distance_km"],
                "nearest_hospital_name": infra_res["nearest_hospital_name"],
                "nearest_hospital_lat": infra_res["nearest_hospital_lat"],
                "nearest_hospital_lon": infra_res["nearest_hospital_lon"],
                "hospital_tier": infra_res["hospital_tier"],
                "district": infra_res["district"],
                "is_emergency_corridor": infra_res["is_emergency_corridor"],
                "infrastructure_risk_score": risk_res["infrastructure_risk_score"],
                "priority_level": risk_res["priority_level"],
                "hospital_source": "synthetic",
                "location_source": "synthetic",
                "generation_method": "gis_geodesic_nearest_neighbor",
                "source": "synthetic"
            },
            "environment": {
                "rain_detected": is_rain,
                "luminance_lux": lux,
                "weather_condition": weather_desc,
                "preferred_modality": sensor_eval["preferred_modality"],
                "modality_reason": sensor_eval["modality_reason"],
                "selected_depth_cm": sensor_eval["selected_depth_cm"],
                "source": "synthetic"
            },
            "vehicle": {
                "vehicle_id": "INSPECTION_FLEET_01",
                "speed_kmh": speed_kmh,
                "chassis_height_baseline_cm": self.baseline_chassis_cm,
                "source": "synthetic"
            },
            "provenance": {
                "visual_source": "RDD2022",
                "sensor_source": "synthetic",
                "context_source": "synthetic",
                "synthesis_version": "1.0.0",
                "synthesis_seed": self.seed,
                "generation_timestamp": datetime.now(timezone.utc).isoformat(),
                "citation_doi": "10.6084/m9.figshare.21431547"
            }
        }
