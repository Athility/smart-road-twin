"""
Smart Road Multimodal Dataset (SRMD) Builder & Sensor Synthesizer
=================================================================

Constructs the derived multimodal dataset (SRMD) combining authentic RDD2022 India
visual evidence with controlled, physics-grounded synthetic sensors and urban context.

Provenance Architecture:
- Visual Ground Truth: RDD2022 India (Sekilab / CRDDC'2022, DOI: 10.6084/m9.figshare.21431547)
- Sensor Augmentation: LiDAR depth, Sonar depth, Z-acceleration (physics modeled)
- Context Augmentation: Traffic PCU, Hospital proximity, GPS corridor location
"""

import os
import sys
import json
import random
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SRMD_DIR = PROCESSED_DIR / "srmd"

# Official RDD2022 class taxonomy
CLASS_TAXONOMY = {
    "D00": {
        "damage_class": "longitudinal_crack",
        "damage_label": "Longitudinal Crack",
        "description": "Linear crack along the longitudinal wheel-path direction",
        "depth_range_cm": (0.6, 2.0),
        "impact_alpha": 0.15,
        "base_severity": "Low"
    },
    "D10": {
        "damage_class": "transverse_crack",
        "damage_label": "Transverse Crack",
        "description": "Linear crack across the road surface perpendicular to traffic",
        "depth_range_cm": (0.5, 2.2),
        "impact_alpha": 0.18,
        "base_severity": "Low"
    },
    "D20": {
        "damage_class": "alligator_crack",
        "damage_label": "Alligator Crack",
        "description": "Interconnected fatigue cracking network",
        "depth_range_cm": (2.0, 5.0),
        "impact_alpha": 0.35,
        "base_severity": "Medium"
    },
    "D40": {
        "damage_class": "pothole",
        "damage_label": "Pothole",
        "description": "Structural pavement cavity causing physical wheel displacement",
        "depth_range_cm": (6.0, 16.0),
        "impact_alpha": 0.85,
        "base_severity": "Critical"
    }
}

# Real-world Indian municipal road corridors (Mumbai Western Express & NH corridors)
CORRIDOR_WAYPOINTS = [
    {
        "segment_id": "WEH-MUM-SEC-01",
        "road_name": "Western Express Highway, Bandra-Kalanagar Junction",
        "road_class": "HIGH_TRAFFIC_CORRIDOR",
        "lat": 19.0596, "lon": 72.8488,
        "hospital_name": "Lilavati Hospital & Research Centre",
        "hospital_dist_km": 0.45
    },
    {
        "segment_id": "SVR-MUM-SEC-02",
        "road_name": "Swami Vivekananda (SV) Road, Santacruz West",
        "road_class": "ARTERIAL",
        "lat": 19.0780, "lon": 72.8410,
        "hospital_name": "Nanavati Max Super Speciality Hospital",
        "hospital_dist_km": 0.65
    },
    {
        "segment_id": "LNK-MUM-SEC-03",
        "road_name": "Linking Road Commercial Sub-Arterial, Bandra West",
        "road_class": "COLLECTOR",
        "lat": 19.0620, "lon": 72.8330,
        "hospital_name": "Bhabha Municipal General Hospital",
        "hospital_dist_km": 0.90
    },
    {
        "segment_id": "RES-MUM-SEC-04",
        "road_name": "Pali Hill Residential Access Corridor, Bandra West",
        "road_class": "LOCAL",
        "lat": 19.0650, "lon": 72.8250,
        "hospital_name": "Holy Family Multi-Speciality Hospital",
        "hospital_dist_km": 1.10
    },
    {
        "segment_id": "WEH-MUM-SEC-05",
        "road_name": "Western Express Highway, Jogeshwari Link Flyover",
        "road_class": "HIGH_TRAFFIC_CORRIDOR",
        "lat": 19.1360, "lon": 72.8600,
        "hospital_name": "Bal Thackeray Trauma Care Municipal Hospital",
        "hospital_dist_km": 0.85
    },
    {
        "segment_id": "SVR-MUM-SEC-06",
        "road_name": "Swami Vivekananda (SV) Road, Andheri West",
        "road_class": "ARTERIAL",
        "lat": 19.1200, "lon": 72.8460,
        "hospital_name": "Dr. R.N. Cooper Municipal General Hospital",
        "hospital_dist_km": 1.25
    }
]


class SRMDRecordBuilder:
    """
    Deterministically synthesizes SRMD records combining RDD2022 visual annotations
    with physics-based sensor augmentation.
    """
    def __init__(self, seed: int = 2026, baseline_chassis_cm: float = 30.0):
        from services.data_generation.accelerometer_model import AccelerometerModel
        from services.data_generation.traffic_model import TrafficModel
        from services.data_generation.infrastructure_model import InfrastructureModel
        from services.data_generation.location_model import LocationModel, MODE_SYNTHETIC_LOCATION
        from services.data_generation.consistency_model import PhysicalConsistencyModel
        self.seed = seed
        self.baseline_chassis_cm = baseline_chassis_cm
        self.rng = random.Random(seed)
        self.location_model = LocationModel(mode=MODE_SYNTHETIC_LOCATION, seed=seed)
        self.consistency = PhysicalConsistencyModel(seed=seed, baseline_chassis_cm=baseline_chassis_cm)
        self.accel_model = AccelerometerModel(seed=seed)
        self.traffic_model = TrafficModel(seed=seed)
        self.infra_model = InfrastructureModel(seed=seed)

    def reseed(self, seed: int):
        self.seed = seed
        self.rng = random.Random(seed)
        self.location_model.reseed(seed)
        self.consistency.reseed(seed)
        self.accel_model.reseed(seed)
        self.traffic_model.reseed(seed)
        self.infra_model.reseed(seed)

    def synthesize_record(
        self,
        image_id: str,
        filename: str,
        damage_class: str,
        bbox: List[float],
        image_dims: Dict[str, int],
        annotation_idx: int = 0,
        waypoint_idx: Optional[int] = None,
        weather_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Builds a single SRMD record with full provenance.
        
        Args:
            image_id: Source image ID (e.g. 'India_000045')
            filename: Source image filename (e.g. 'India_000045.jpg')
            damage_class: RDD2022 class ('D00', 'D10', 'D20', 'D40')
            bbox: [xmin, ymin, xmax, ymax]
            image_dims: {'width': int, 'height': int, 'depth': int}
            annotation_idx: Index of annotation within image
            waypoint_idx: Corridor waypoint index
            weather_override: 'rain', 'dry_day', 'low_light', or None for stochastic
        """
        tax = CLASS_TAXONOMY.get(damage_class, CLASS_TAXONOMY["D40"])
        xmin, ymin, xmax, ymax = bbox
        width = image_dims.get("width", 720)
        height = image_dims.get("height", 720)

        # 1. Surface area estimation from bounding box & perspective projection
        surface_area_sqm = self.consistency.compute_perspective_surface_area(bbox, image_dims)

        # 2. Derive latent defect_severity ∈ [0.0, 1.0]
        defect_severity = self.consistency.derive_latent_severity(damage_class, surface_area_sqm)

        # 3. Derive physical depth & cavity volume strictly coupled to severity & area
        defect_depth_cm, volume_liters, severity_tier = self.consistency.derive_depth_and_volume(
            defect_severity=defect_severity,
            damage_class=damage_class,
            surface_area_sqm=surface_area_sqm
        )

        # 4. Urban Location & Context via LocationModel
        route_idx = waypoint_idx if waypoint_idx is not None else annotation_idx
        loc_res = self.location_model.generate_location_for_defect(route_index=route_idx)
        lat = loc_res["latitude"]
        lon = loc_res["longitude"]
        road_class = loc_res["road_class"]
        road_segment_id = loc_res["road_segment_id"]
        road_name = loc_res["road_name"]

        traffic_res = self.traffic_model.simulate(
            road_class=road_class,
            latitude=lat,
            longitude=lon,
            road_segment_id=road_segment_id,
            road_name=road_name,
            peak_hour=True
        )
        traffic_pcu = traffic_res["traffic_pcu"]
        speed_kmh = traffic_res["operating_speed_kmh"]
        infra_res = self.infra_model.simulate(latitude=lat, longitude=lon)
        hosp_dist = infra_res["dist_hospital_km"]

        # 5. Environmental conditions
        if weather_override == "rain":
            is_rain = True
            lux = round(self.rng.uniform(12.0, 32.0), 1)
            weather_desc = "Rain / Wet Road Surface"
        elif weather_override == "dry_day":
            is_rain = False
            lux = round(self.rng.uniform(60.0, 95.0), 1)
            weather_desc = "Clear / Dry Daylight"
        elif weather_override == "low_light":
            is_rain = False
            lux = round(self.rng.uniform(18.0, 38.0), 1)
            weather_desc = "Low Luminance Dusk / Overcast"
        else:
            # Stochastic: 35% chance of rain (typical Mumbai monsoon profile)
            is_rain = self.rng.random() < 0.35
            if is_rain:
                lux = round(self.rng.uniform(12.0, 32.0), 1)
                weather_desc = "Rain / Wet Road Surface"
            else:
                # 85% daylight, 15% dusk/night
                is_day = self.rng.random() < 0.85
                lux = round(self.rng.uniform(55.0, 95.0) if is_day else self.rng.uniform(18.0, 38.0), 1)
                weather_desc = "Clear / Dry Daylight" if is_day else "Low Luminance / Night"

        # 6. Sensor measurements (LiDAR & Sonar) + Modality switching
        sensor_eval = self.consistency.derive_sensor_measurements(
            defect_depth_cm=defect_depth_cm,
            rain_detected=is_rain,
            luminance_lux=lux
        )

        # 7. Accelerometer Z-axis Dynamic Impact Model
        accel_res = self.consistency.derive_accelerometer_impact(
            defect_severity=defect_severity,
            defect_depth_cm=defect_depth_cm,
            surface_area_sqm=surface_area_sqm,
            vehicle_speed_kmh=speed_kmh,
            damage_class=damage_class
        )

        # 8. Infrastructure Risk / TOPSIS Composite Index
        risk_res = self.consistency.derive_infrastructure_risk(
            defect_severity=defect_severity,
            defect_depth_cm=defect_depth_cm,
            surface_area_sqm=surface_area_sqm,
            traffic_pcu=traffic_pcu,
            vehicle_speed_kmh=speed_kmh,
            dist_hospital_km=hosp_dist
        )

        event_id = f"SRMD-IND-{image_id.replace('India_', '')}-{annotation_idx:02d}"

        # 9. Assemble complete SRMD Record matching user schema
        record = {
            "event_id": event_id,
            "source": {
                "dataset": "RDD2022",
                "country": "India",
                "image_id": image_id,
                "annotation_id": f"{image_id}_obj{annotation_idx}",
                "annotation_class": damage_class
            },
            "visual": {
                "damage_class": tax["damage_class"],
                "damage_label": tax["damage_label"],
                "confidence": 1.0,
                "bbox": [round(c, 1) for c in bbox],
                "image_width": width,
                "image_height": height,
                "image_filename": filename,
                "image_url": f"/static/rdd2022/images/{filename}",
                "surface_area_sqm": surface_area_sqm,
                "defect_severity": defect_severity,
                "calculated_volume_liters": volume_liters
            },
            "location": {
                "latitude": lat,
                "longitude": lon,
                "road_segment_id": road_segment_id,
                "road_name": road_name,
                "road_class": road_class,
                "chainage_km": loc_res.get("chainage_km", 0.0),
                "location_mode": loc_res["location_mode"],
                "location_source": loc_res["location_source"],
                "demonstration_area": loc_res.get("demonstration_area", "Mumbai Metropolitan Urban Corridor"),
                "synthesis_seed": loc_res.get("synthesis_seed", self.seed)
            },
            "lidar": {
                "depth_cm": sensor_eval["raw_lidar_cm"],
                "effective_depth_cm": defect_depth_cm,
                "severity_tier": severity_tier,
                "sensor_source": "synthetic",
                "generation_method": "controlled_distribution",
                "source": "synthetic"
            },
            "sonar": {
                "depth_cm": sensor_eval["raw_sonar_cm"],
                "effective_depth_cm": sensor_eval["sonar_effective_depth_cm"],
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
                "road_class": road_class,
                "road_class_name": traffic_res.get("road_class_name", road_class),
                "traffic_pcu": traffic_pcu,
                "pcu": traffic_pcu,
                "traffic_source": "synthetic",
                "roadway_capacity_pcu": traffic_res["roadway_capacity_pcu"],
                "volume_capacity_ratio": traffic_res["volume_capacity_ratio"],
                "level_of_service": traffic_res["level_of_service"],
                "sensor_source": "synthetic",
                "generation_method": "controlled_distribution",
                "source": "synthetic"
            },
            "infrastructure": {
                "dist_hospital_km": hosp_dist,
                "hospital_distance_km": hosp_dist,
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
        return record


def srmd_to_telemetry_payload(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts an SRMD multimodal record into the flat TelemetryPayload format
    expected by the FastAPI /api/v1/telemetry/ingest endpoint.
    """
    return {
        "vehicle_id": record["vehicle"]["vehicle_id"],
        "latitude": record["location"]["latitude"],
        "longitude": record["location"]["longitude"],
        "rain_detected": record["environment"]["rain_detected"],
        "mean_luminance": record["environment"]["luminance_lux"],
        "lidar_depth_cm": record["lidar"]["depth_cm"],
        "sonar_depth_cm": record["sonar"]["depth_cm"],
        "z_accel_g": record["accelerometer"]["z_accel_g"],
        "surface_area_sqm": record["visual"]["surface_area_sqm"],
        "traffic_pcu": record["traffic"]["pcu"],
        "dist_hospital_km": record["infrastructure"]["hospital_distance_km"],
        "speed_kmh": record["vehicle"]["speed_kmh"],
        # Extended multimodal fields
        "event_id": record["event_id"],
        "image_url": record["visual"]["image_url"],
        "damage_class": record["source"]["annotation_class"],
        "damage_label": record["visual"]["damage_label"],
        "bounding_boxes": [
            {
                "class_id": record["source"]["annotation_class"],
                "name": record["visual"]["damage_label"],
                "xmin": record["visual"]["bbox"][0],
                "ymin": record["visual"]["bbox"][1],
                "xmax": record["visual"]["bbox"][2],
                "ymax": record["visual"]["bbox"][3],
                "confidence": record["visual"]["confidence"]
            }
        ],
        "provenance": record["provenance"]
    }


def build_srmd_dataset(
    index_file: Optional[Path] = None,
    output_dir: Path = SRMD_DIR,
    seed: int = 2026
) -> Dict[str, Any]:
    """
    Builds the full Smart Road Multimodal Dataset (SRMD) from the processed annotations index.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    records_dir = output_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)

    if index_file is None:
        index_file = PROCESSED_DIR / "annotations_index.json"

    if not index_file.exists():
        raise FileNotFoundError(f"Annotations index not found: {index_file}. Run scripts/prepare_rdd2022.py first.")

    with open(index_file, "r", encoding="utf-8") as f:
        annotations_index = json.load(f)

    builder = SRMDRecordBuilder(seed=seed)
    srmd_records = []
    class_distribution = {c: 0 for c in CLASS_TAXONOMY}

    print(f"Building SRMD from {len(annotations_index)} indexed images (seed={seed})...")

    wp_idx = 0
    for img_id, data in annotations_index.items():
        filename = data.get("filename", f"{img_id}.jpg")
        dims = {"width": data.get("width", 720), "height": data.get("height", 720), "depth": data.get("depth", 3)}

        for obj_idx, obj in enumerate(data.get("objects", [])):
            cls_id = obj["class_id"]
            bbox = [obj["xmin"], obj["ymin"], obj["xmax"], obj["ymax"]]

            record = builder.synthesize_record(
                image_id=img_id,
                filename=filename,
                damage_class=cls_id,
                bbox=bbox,
                image_dims=dims,
                annotation_idx=obj_idx,
                waypoint_idx=wp_idx
            )
            srmd_records.append(record)
            class_distribution[cls_id] = class_distribution.get(cls_id, 0) + 1

            # Save individual record JSON
            record_path = records_dir / f"{record['event_id']}.json"
            with open(record_path, "w", encoding="utf-8") as rf:
                json.dump(record, rf, indent=2)

            wp_idx += 1

    # Save combined records
    combined_path = output_dir / "srmd_records.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(srmd_records, f, indent=2)

    # Save dataset manifest
    manifest = {
        "dataset_name": "Smart Road Multimodal Dataset (SRMD)",
        "dataset_version": "1.0.0",
        "description": "Reproducible multimodal road-infrastructure dataset combining authentic RDD2022 India imagery with physics-calibrated synthetic sensor augmentation.",
        "creation_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_records": len(srmd_records),
        "source_dataset": {
            "name": "RDD2022",
            "country": "India",
            "challenge": "CRDDC'2022",
            "doi": "10.6084/m9.figshare.21431547"
        },
        "class_distribution": {
            cid: {
                "name": CLASS_TAXONOMY[cid]["damage_label"],
                "count": class_distribution.get(cid, 0)
            }
            for cid in sorted(CLASS_TAXONOMY.keys())
        },
        "modalities": [
            "visual_rgb_imagery",
            "optical_lidar_distance",
            "acoustic_sonar_distance",
            "chassis_z_axis_accelerometry",
            "ambient_luminance_photometer",
            "precipitation_detector",
            "traffic_pcu_gis",
            "hospital_trauma_proximity_gis",
            "gps_geospatial_telemetry"
        ],
        "provenance_policy": "Strict distinction maintained between source visual evidence (RDD2022) and controlled synthetic sensor augmentation.",
        "random_seed": seed
    }

    manifest_path = output_dir / "srmd_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"SRMD build complete: {len(srmd_records)} records saved to {output_dir}")
    return manifest
