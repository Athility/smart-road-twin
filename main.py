from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import numpy as np
import sys, os, threading, json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.abspath("."))
from services.ai_engine.pipeline.depth_estimator import EnvironmentalSensorSwitching
from services.backend.app.services.mcdm_engine import (
    MCDMPrioritizationEngine,
    get_fuzzy_bwm_profile,
    encode_road_class,
    NAMED_BWM_PROFILES,
    THRESHOLD_CALIBRATION_NOTE,
    build_mcdm_row_from_store_item,
)
from services.data_generation.infrastructure_model import InfrastructureModel
from services.cv.preprocessing import load_image, apply_clahe, encode_image_base64, HAS_CV2

app = FastAPI(
    title="Smart Digital Twin API",
    description="Municipal road infrastructure monitoring, AI depth estimation, and Fuzzy BWM / TOPSIS / VIKOR hazard prioritization API",
    version="1.0.0"
)

# Standardized CORS for frontend dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:8000"],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:[0-9]+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static images directory if present
sample_images_dir = os.path.join(os.path.dirname(__file__), "data", "samples", "images")
if os.path.exists(sample_images_dir):
    app.mount("/static/rdd2022/images", StaticFiles(directory=sample_images_dir), name="rdd2022_images")

_infra_lookup = InfrastructureModel(seed=2026)

class TelemetryPayload(BaseModel):
    vehicle_id: str = Field(..., min_length=1, max_length=64, description="Vehicle fleet ID")
    latitude: float = Field(..., ge=-90.0, le=90.0, description="GPS Latitude (-90 to 90)")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="GPS Longitude (-180 to 180)")
    rain_detected: bool = Field(..., description="Precipitation/rain sensor state")
    mean_luminance: float = Field(..., ge=0.0, le=150000.0, description="Ambient light lux/luminance")
    lidar_depth_cm: float = Field(..., ge=0.0, le=500.0, description="Raw LiDAR distance measurement (cm)")
    sonar_depth_cm: float = Field(..., ge=0.0, le=500.0, description="Raw Sonar distance measurement (cm)")
    z_accel_g: float = Field(..., ge=-20.0, le=20.0, description="Z-axis accelerometer reading (g)")
    surface_area_sqm: float = Field(..., ge=0.001, le=100.0, description="Defect surface area (m^2)")
    traffic_pcu: int = Field(..., ge=0, le=200000, description="Traffic volume in Passenger Car Units (PCU/day)")
    dist_hospital_km: float = Field(..., ge=0.01, le=200.0, description="Distance to nearest hospital (km)")
    school_proximity_km: Optional[float] = Field(None, ge=0.01, le=200.0, description="Distance to nearest school (km)")
    speed_kmh: float = Field(..., ge=0.0, le=250.0, description="Vehicle travel speed (km/h)")
    # Provenance and visual detection attributes
    data_source: Optional[str] = Field("MULTIMODAL", description="Data source: RDD2022, SYNTHETIC SENSOR, SYNTHETIC GIS, or MULTIMODAL")
    damage_class: Optional[str] = Field(None, description="Damage class code (D00, D10, D20, D40)")
    damage_label: Optional[str] = Field(None, description="Damage class label (e.g. Pothole)")
    confidence: Optional[float] = Field(None, description="Detection confidence score (0-1)")
    image_id: Optional[str] = Field(None, description="RDD2022 source image ID")
    image_url: Optional[str] = Field(None, description="Image inspection URL")
    bbox: Optional[List[float]] = Field(None, description="Bounding box [ymin, xmin, ymax, xmax]")
    road_name: Optional[str] = Field(None, description="Roadway name")
    road_class: Optional[str] = Field(None, description="Roadway hierarchy classification")


class CustomPrioritizationRequest(BaseModel):
    method: str = Field("both", description="Prioritization method: 'topsis', 'vikor', or 'both'")
    profile: str = Field("custom", description="Weight profile name ('safety-first', 'traffic-first', 'equity-first', or 'custom')")
    weights: Optional[List[float]] = Field(None, description="Custom crisp weights (length 4 or 6)")
    best_to_others: Optional[List[Any]] = Field(None, description="Fuzzy BWM Best-to-Others TFN vector")
    others_to_worst: Optional[List[Any]] = Field(None, description="Fuzzy BWM Others-to-Worst TFN vector")
    best_idx: Optional[int] = Field(None, description="Index of Best criterion (0-5)")
    worst_idx: Optional[int] = Field(None, description="Index of Worst criterion (0-5)")
    v: float = Field(0.5, ge=0.0, le=1.0, description="VIKOR group utility weight v in [0, 1]")


# In-memory storage and thread-safe lock for telemetry events
telemetry_store: List[Dict[str, Any]] = []
_store_lock = threading.Lock()
_pothole_counter = 0

def get_active_detector_label() -> str:
    try:
        from services.cv.rdd2022_detector import RDD2022Detector
        return RDD2022Detector().get_detector_slug()
    except Exception:
        return "ground-truth lookup"

ACTIVE_DETECTOR = get_active_detector_label()

def _get_item_severity_magnitude(item: Dict[str, Any]) -> float:
    pt = item.get("processed_telemetry", {})
    mag = pt.get("defect_severity_magnitude")
    if mag is not None:
        return float(mag)
    vol = pt.get("calculated_volume_liters")
    if vol is not None:
        return float(vol)
    return 0.0


def _build_record_6criteria_row(item: Dict[str, Any]) -> List[float]:
    """Delegates to the shared MCDM row builder (see mcdm_engine.build_mcdm_row_from_store_item)."""
    return build_mcdm_row_from_store_item(item)



def recompute_topsis_for_all():
    if not telemetry_store:
        return
    engine = MCDMPrioritizationEngine(profile="safety-first", use_calibrated_thresholds=True)
    if len(telemetry_store) == 1:
        # Reference against standard road thresholds if only 1 item
        ref_worst = [0.1, 1000.0, 10.0, 10.0, 5.0, 1.0]
        ref_best = [80.0, 30000.0, 0.1, 70.0, 0.1, 4.0]
        item = telemetry_store[0]
        mat = np.array([
            _build_record_6criteria_row(item),
            ref_worst,
            ref_best
        ])
        all_res = engine.compute_all(mat, calibrated=True)
        score = float(all_res["topsis_scores"][0])
        item["topsis_score"] = score
        item["topsis_rank"] = 1
        item["vikor_S"] = float(all_res["vikor_S"][0])
        item["vikor_R"] = float(all_res["vikor_R"][0])
        item["vikor_Q"] = float(all_res["vikor_Q"][0])
        item["vikor_rank"] = 1
        item["priority_level"] = engine.classify_priority(score, calibrated=True)
        return

    matrix = np.array([_build_record_6criteria_row(item) for item in telemetry_store])
    all_res = engine.compute_all(matrix, calibrated=True)
    for idx, item in enumerate(telemetry_store):
        score = float(all_res["topsis_scores"][idx])
        item["topsis_score"] = score
        item["topsis_rank"] = int(all_res["topsis_ranks"][idx])
        item["vikor_S"] = float(all_res["vikor_S"][idx])
        item["vikor_R"] = float(all_res["vikor_R"][idx])
        item["vikor_Q"] = float(all_res["vikor_Q"][idx])
        item["vikor_rank"] = int(all_res["vikor_ranks"][idx])
        item["priority_level"] = engine.classify_priority(score, calibrated=True)

def process_and_store_telemetry(payload_dict: dict) -> dict:
    global _pothole_counter
    dmg_cls = payload_dict.get("damage_class") or "D40"
    bbox = payload_dict.get("bbox", [180.0, 260.0, 420.0, 480.0])
    switcher = EnvironmentalSensorSwitching(baseline_chassis_height_cm=30.0)
    depth_res = switcher.evaluate_environment_and_measure(
        rain_detected=payload_dict["rain_detected"],
        mean_luminance=payload_dict["mean_luminance"],
        lidar_depth_cm=payload_dict["lidar_depth_cm"],
        sonar_depth_cm=payload_dict["sonar_depth_cm"],
        z_accel_g=payload_dict["z_accel_g"],
        surface_area_sqm=payload_dict["surface_area_sqm"],
        damage_class=dmg_cls,
        bbox=bbox
    )

    lat = float(payload_dict["latitude"])
    lon = float(payload_dict["longitude"])
    school_info, school_geodesic = _infra_lookup.find_nearest_school(lat, lon)
    computed_school_km = round(max(0.05, school_geodesic * _infra_lookup.tortuosity_factor), 2)
    school_km = float(payload_dict.get("school_proximity_km") or computed_school_km)
    road_cls = payload_dict.get("road_class") or "HIGH_TRAFFIC_CORRIDOR"

    with _store_lock:
        _pothole_counter += 1
        img_id = payload_dict.get("image_id", f"India_{_pothole_counter:06d}")
        img_url = payload_dict.get("image_url", f"/api/v1/images/{img_id}.jpg")
        record = {
            "id": payload_dict.get("id") or f"POTHOLE-{_pothole_counter:04d}",
            "timestamp": payload_dict.get("timestamp") or datetime.now(timezone.utc).isoformat(),
            "vehicle_id": payload_dict["vehicle_id"],
            "latitude": lat,
            "longitude": lon,
            "rain_detected": bool(payload_dict["rain_detected"]),
            "mean_luminance": float(payload_dict["mean_luminance"]),
            "lidar_depth_cm": float(payload_dict["lidar_depth_cm"]),
            "sonar_depth_cm": float(payload_dict["sonar_depth_cm"]),
            "z_accel_g": float(payload_dict["z_accel_g"]),
            "surface_area_sqm": float(payload_dict["surface_area_sqm"]),
            "traffic_pcu": int(payload_dict["traffic_pcu"]),
            "dist_hospital_km": float(payload_dict["dist_hospital_km"]),
            "school_proximity_km": school_km,
            "nearest_school_name": payload_dict.get("nearest_school_name", school_info["name"]),
            "speed_kmh": float(payload_dict["speed_kmh"]),
            "data_source": payload_dict.get("data_source", "MULTIMODAL"),
            "damage_class": dmg_cls,
            "damage_label": payload_dict.get("damage_label", "Pothole"),
            "confidence": float(payload_dict.get("confidence", 0.94)),
            "image_id": img_id,
            "image_url": img_url,
            "bbox": bbox,
            "road_name": payload_dict.get("road_name", "Western Express Highway Corridor"),
            "road_class": road_cls,
            "road_class_weight": encode_road_class(road_cls),
            "detector": ACTIVE_DETECTOR,
            "processed_telemetry": depth_res,
            "topsis_score": 0.0,
            "priority_level": "Low"
        }
        telemetry_store.append(record)
        # Prevent unbounded memory growth in long-running simulations (retain 500 events)
        if len(telemetry_store) > 500:
            telemetry_store.pop(0)
        recompute_topsis_for_all()
    return record

# Load from SRMD dataset if available, else seed with default 6-record SRMD telemetry
srmd_dataset_file = os.path.join(os.path.dirname(__file__), "data", "processed", "srmd", "srmd_records.json")
srmd_loaded = False


def _load_srmd_records_into_store() -> bool:
    global srmd_loaded
    if not os.path.exists(srmd_dataset_file):
        return False
    try:
        with open(srmd_dataset_file, "r", encoding="utf-8") as f:
            srmd_items = json.load(f)
        if not srmd_items:
            return False
        with _store_lock:
            telemetry_store.clear()
            for r in srmd_items:
                img_file = r["visual"].get("image_filename", f"{r['source']['image_id']}.jpg")
                dmg_cls = r["source"].get("annotation_class", "D40")
                bbox = r["visual"].get("bbox", [])
                switcher = EnvironmentalSensorSwitching(baseline_chassis_height_cm=30.0)
                depth_eval = switcher.evaluate_environment_and_measure(
                    rain_detected=r["environment"]["rain_detected"],
                    mean_luminance=r["environment"]["luminance_lux"],
                    lidar_depth_cm=r["lidar"]["depth_cm"],
                    sonar_depth_cm=r["sonar"]["depth_cm"],
                    z_accel_g=r["accelerometer"]["z_accel_g"],
                    surface_area_sqm=r["visual"]["surface_area_sqm"],
                    damage_class=dmg_cls,
                    bbox=bbox
                )
                depth_eval["estimated_depth_cm"] = r["environment"]["selected_depth_cm"]
                depth_eval["detection_modality"] = r["environment"]["preferred_modality"]
                # Always restore the pre-computed visual fields from the SRMD JSON so
                # the decision-matrix row is identical whether built from the JSON or the
                # telemetry_store item (fixes SRMD-path vs API-path rho discrepancy).
                depth_eval["calculated_volume_liters"] = r["visual"].get("calculated_volume_liters")
                depth_eval["crack_severity_index"] = r["visual"].get("crack_severity_index")
                depth_eval["defect_severity_magnitude"] = float(
                    r["visual"].get("defect_severity_magnitude", 0.0)
                )
                depth_eval["dynamic_impact_confirmed"] = r["accelerometer"]["dynamic_impact_triggered"]

                lat = float(r["location"]["latitude"])
                lon = float(r["location"]["longitude"])
                road_cls = r["location"].get("road_class", "HIGH_TRAFFIC_CORRIDOR")
                infra_block = r.get("infrastructure", {})
                if "school_proximity_km" in infra_block:
                    school_km = float(infra_block["school_proximity_km"])
                    school_name = infra_block.get("nearest_school_name", "")
                else:
                    s_info, s_geo = _infra_lookup.find_nearest_school(lat, lon)
                    school_km = round(max(0.05, s_geo * _infra_lookup.tortuosity_factor), 2)
                    school_name = s_info["name"]

                rec = {
                    "id": r["event_id"],
                    "timestamp": r.get("provenance", {}).get("generation_timestamp", datetime.now(timezone.utc).isoformat()),
                    "vehicle_id": r.get("vehicle", {}).get("vehicle_id", "INSPECTION_FLEET_01"),
                    "latitude": lat,
                    "longitude": lon,
                    "rain_detected": bool(r["environment"]["rain_detected"]),
                    "mean_luminance": float(r["environment"]["luminance_lux"]),
                    "lidar_depth_cm": float(r["lidar"]["depth_cm"]),
                    "sonar_depth_cm": float(r["sonar"]["depth_cm"]),
                    "z_accel_g": float(r["accelerometer"]["z_accel_g"]),
                    "surface_area_sqm": float(r["visual"]["surface_area_sqm"]),
                    "traffic_pcu": int(r["traffic"]["traffic_pcu"]),
                    "dist_hospital_km": float(infra_block["dist_hospital_km"]),
                    "school_proximity_km": school_km,
                    "nearest_school_name": school_name,
                    "speed_kmh": float(r.get("vehicle", {}).get("speed_kmh", 40.0)),
                    "data_source": "MULTIMODAL",
                    "damage_class": dmg_cls,
                    "damage_label": r["visual"].get("damage_label", "Pothole"),
                    "confidence": float(r["visual"].get("confidence", 0.94)),
                    "image_id": r["source"].get("image_id", "India_000045"),
                    "image_url": f"/api/v1/images/{img_file}",
                    "bbox": bbox,
                    "road_name": r["location"].get("road_name", "Western Express Highway"),
                    "road_class": road_cls,
                    "road_class_weight": encode_road_class(road_cls),
                    "detector": ACTIVE_DETECTOR,
                    "processed_telemetry": depth_eval,
                    "topsis_score": 0.0,
                    "priority_level": "Low"
                }
                telemetry_store.append(rec)
            recompute_topsis_for_all()
            srmd_loaded = True
        return True
    except Exception as e:
        print(f"Warning: Failed to load SRMD dataset ({e}), using default seeds.")
        return False


if not _load_srmd_records_into_store():
    initial_seeds = [
        {
            "id": "SRMD-IND-000045-00",
            "timestamp": "2026-09-24T12:30:01.800067+00:00",
            "vehicle_id": "INSPECTION_FLEET_01",
            "latitude": 19.05939, "longitude": 72.8488,
            "rain_detected": True, "mean_luminance": 22.1,
            "lidar_depth_cm": 45.94, "sonar_depth_cm": 45.95,
            "z_accel_g": 2.53, "surface_area_sqm": 1.812,
            "traffic_pcu": 27151, "dist_hospital_km": 2.4, "speed_kmh": 41.7,
            "data_source": "MULTIMODAL",
            "damage_class": "D40", "damage_label": "Pothole", "confidence": 1.0,
            "image_id": "India_000045", "image_url": "/api/v1/images/India_000045.jpg",
            "bbox": [180.0, 260.0, 420.0, 480.0],
            "road_name": "Western Express Highway, Bandra-Kalanagar Junction",
            "road_class": "HIGH_TRAFFIC_CORRIDOR"
        },
        {
            "id": "SRMD-IND-000102-00",
            "timestamp": "2026-09-24T12:30:01.800067+00:00",
            "vehicle_id": "INSPECTION_FLEET_01",
            "latitude": 19.07801, "longitude": 72.84119,
            "rain_detected": False, "mean_luminance": 20.1,
            "lidar_depth_cm": 34.13, "sonar_depth_cm": 34.27,
            "z_accel_g": 1.16, "surface_area_sqm": 3.587,
            "traffic_pcu": 16381, "dist_hospital_km": 1.31, "speed_kmh": 36.5,
            "data_source": "MULTIMODAL",
            "damage_class": "D20", "damage_label": "Alligator Crack", "confidence": 1.0,
            "image_id": "India_000102", "image_url": "/api/v1/images/India_000102.jpg",
            "bbox": [120.0, 300.0, 580.0, 520.0],
            "road_name": "Swami Vivekananda (SV) Road, Santacruz West",
            "road_class": "ARTERIAL"
        },
        {
            "id": "SRMD-IND-000155-00",
            "timestamp": "2026-09-24T12:30:01.801067+00:00",
            "vehicle_id": "INSPECTION_FLEET_01",
            "latitude": 19.06179, "longitude": 72.83285,
            "rain_detected": True, "mean_luminance": 24.0,
            "lidar_depth_cm": 30.79, "sonar_depth_cm": 30.92,
            "z_accel_g": 1.03, "surface_area_sqm": 0.918,
            "traffic_pcu": 11391, "dist_hospital_km": 0.88, "speed_kmh": 25.0,
            "data_source": "MULTIMODAL",
            "damage_class": "D00", "damage_label": "Longitudinal Crack", "confidence": 1.0,
            "image_id": "India_000155", "image_url": "/api/v1/images/India_000155.jpg",
            "bbox": [310.0, 150.0, 370.0, 640.0],
            "road_name": "Linking Road Commercial Sub-Arterial, Bandra West",
            "road_class": "COLLECTOR"
        },
        {
            "id": "SRMD-IND-000210-00",
            "timestamp": "2026-09-24T12:30:01.801067+00:00",
            "vehicle_id": "INSPECTION_FLEET_01",
            "latitude": 19.06505, "longitude": 72.82503,
            "rain_detected": False, "mean_luminance": 76.9,
            "lidar_depth_cm": 30.99, "sonar_depth_cm": 30.80,
            "z_accel_g": 1.02, "surface_area_sqm": 1.221,
            "traffic_pcu": 5596, "dist_hospital_km": 0.96, "speed_kmh": 20.5,
            "data_source": "MULTIMODAL",
            "damage_class": "D10", "damage_label": "Transverse Crack", "confidence": 1.0,
            "image_id": "India_000210", "image_url": "/api/v1/images/India_000210.jpg",
            "bbox": [80.0, 380.0, 620.0, 440.0],
            "road_name": "Pali Hill Residential Access Corridor, Bandra West",
            "road_class": "LOCAL"
        },
        {
            "id": "SRMD-IND-000278-00",
            "timestamp": "2026-09-24T12:30:01.802066+00:00",
            "vehicle_id": "INSPECTION_FLEET_01",
            "latitude": 19.13615, "longitude": 72.86003,
            "rain_detected": False, "mean_luminance": 85.0,
            "lidar_depth_cm": 36.09, "sonar_depth_cm": 35.96,
            "z_accel_g": 1.41, "surface_area_sqm": 0.85,
            "traffic_pcu": 29131, "dist_hospital_km": 0.27, "speed_kmh": 38.3,
            "data_source": "MULTIMODAL",
            "damage_class": "D40", "damage_label": "Pothole", "confidence": 1.0,
            "image_id": "India_000278", "image_url": "/api/v1/images/India_000278.jpg",
            "bbox": [220.0, 310.0, 390.0, 450.0],
            "road_name": "Western Express Highway, Jogeshwari Link Flyover",
            "road_class": "HIGH_TRAFFIC_CORRIDOR"
        },
        {
            "id": "SRMD-IND-000278-01",
            "timestamp": "2026-09-24T12:30:01.802066+00:00",
            "vehicle_id": "INSPECTION_FLEET_01",
            "latitude": 19.12012, "longitude": 72.84614,
            "rain_detected": False, "mean_luminance": 79.6,
            "lidar_depth_cm": 34.14, "sonar_depth_cm": 33.92,
            "z_accel_g": 1.11, "surface_area_sqm": 2.897,
            "traffic_pcu": 19406, "dist_hospital_km": 2.03, "speed_kmh": 33.0,
            "data_source": "MULTIMODAL",
            "damage_class": "D20", "damage_label": "Alligator Crack", "confidence": 1.0,
            "image_id": "India_000278", "image_url": "/api/v1/images/India_000278.jpg",
            "bbox": [150.0, 250.0, 520.0, 480.0],
            "road_name": "Swami Vivekananda (SV) Road, Andheri West",
            "road_class": "ARTERIAL"
        }
    ]
    for seed in initial_seeds:
        process_and_store_telemetry(seed)

def calculate_summary():
    with _store_lock:
        pothole_vols = [
            float(item["processed_telemetry"]["calculated_volume_liters"])
            for item in telemetry_store
            if item.get("processed_telemetry", {}).get("calculated_volume_liters") is not None
        ]
        total_volume = round(sum(pothole_vols), 2)
        avg_volume = round(total_volume / len(pothole_vols), 2) if pothole_vols else 0.0
        
        priority_counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
        rain_count = 0
        submerged_count = 0
        total_luminance = 0.0

        for item in telemetry_store:
            p = item.get("priority_level", "Low")
            if p in priority_counts:
                priority_counts[p] += 1
            else:
                priority_counts[p] = 1
            
            if item.get("rain_detected"):
                rain_count += 1
            if item.get("processed_telemetry", {}).get("is_submerged"):
                submerged_count += 1
            total_luminance += item.get("mean_luminance", 50.0)
        
        latest_item = telemetry_store[-1] if telemetry_store else None
        latest_rain = latest_item["rain_detected"] if latest_item else False
        latest_submerged = latest_item["processed_telemetry"].get("is_submerged", False) if latest_item else False

        # Current weather state reflects the latest real-time sensor reading
        if latest_rain or latest_submerged:
            weather_state = "Rain Detected (Wet / Submerged Road)"
        else:
            weather_state = "Clear (Dry Road)"

        avg_luminance = round(total_luminance / len(telemetry_store), 1) if telemetry_store else 50.0

        return {
            "total_defects": len(telemetry_store),
            "total_volume_liters": total_volume,
            "average_volume_liters": avg_volume,
            "weather_state": weather_state,
            "is_raining": latest_rain,
            "submerged_defects": submerged_count,
            "average_luminance": avg_luminance,
            "priority_counts": priority_counts,
            "detector": ACTIVE_DETECTOR,
            "latest_update": latest_item["timestamp"] if latest_item else None
        }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Smart Digital Twin API",
        "detector": ACTIVE_DETECTOR,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.post("/api/v1/telemetry/ingest")
async def ingest_telemetry(payload: TelemetryPayload):
    record = process_and_store_telemetry(payload.model_dump())
    return {
        "status": "success",
        "detector": ACTIVE_DETECTOR,
        "record": record,
        "processed_telemetry": record["processed_telemetry"],
        "topsis_score": record["topsis_score"],
        "priority_level": record["priority_level"]
    }

@app.get("/api/v1/telemetry/ingest")
@app.get("/api/v1/telemetry")
async def get_telemetry():
    if not srmd_loaded and os.path.exists(srmd_dataset_file):
        _load_srmd_records_into_store()
    return {
        "status": "success",
        "detector": ACTIVE_DETECTOR,
        "total": len(telemetry_store),
        "summary": calculate_summary(),
        "data": telemetry_store
    }

@app.get("/api/v1/images/{image_name}")
async def get_inspection_image(image_name: str):
    """
    Serves authentic RDD2022 inspection images from data/samples/images/
    with graceful fallback to an SVG inspection placeholder.
    """
    safe_name = os.path.basename(image_name)
    sample_dir = os.path.join(os.path.dirname(__file__), "data", "samples", "images")
    img_path = os.path.join(sample_dir, safe_name)
    if os.path.exists(img_path) and os.path.isfile(img_path):
        return FileResponse(img_path, media_type="image/jpeg")

    # If the user passed e.g. "India_000045" without extension:
    if not safe_name.endswith(".jpg"):
        alt_path = os.path.join(sample_dir, f"{safe_name}.jpg")
        if os.path.exists(alt_path) and os.path.isfile(alt_path):
            return FileResponse(alt_path, media_type="image/jpeg")

    # Clean high-tech SVG inspection placeholder
    svg_fallback = f'''<svg xmlns="http://www.w3.org/2000/svg" width="480" height="280" viewBox="0 0 480 280">
        <defs>
            <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stop-color="#0F1420"/>
                <stop offset="100%" stop-color="#161B27"/>
            </linearGradient>
            <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
                <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#1E2435" stroke-width="0.8"/>
            </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#bg)"/>
        <rect width="100%" height="100%" fill="url(#grid)" opacity="0.6"/>
        <rect x="24" y="24" width="432" height="232" rx="6" fill="#111520" stroke="#2563EB" stroke-width="1.5" stroke-dasharray="4 4"/>
        <circle cx="240" cy="110" r="30" fill="#1E293B" stroke="#3B82F6" stroke-width="1.5"/>
        <path d="M 230 110 L 250 110 M 240 100 L 240 120" stroke="#60A5FA" stroke-width="2" stroke-linecap="round"/>
        <text x="240" y="162" dominant-baseline="middle" text-anchor="middle" fill="#EEF2FF" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif" font-size="13" font-weight="600">RDD2022 Ground-Truth Image</text>
        <text x="240" y="184" dominant-baseline="middle" text-anchor="middle" fill="#60A5FA" font-family="ui-monospace, SFMono-Regular, monospace" font-size="11">{safe_name}</text>
        <text x="240" y="206" dominant-baseline="middle" text-anchor="middle" fill="#64748B" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif" font-size="10">Visual distress reference frame</text>
    </svg>'''
    return Response(content=svg_fallback, media_type="image/svg+xml")

@app.get("/api/v1/dataset/explorer")
async def get_dataset_explorer():
    """
    Returns research-grade dataset explorer summary and individual multimodal records
    computed from the canonical telemetry_store (shared with /api/v1/telemetry):
    - Total records
    - D00, D10, D20, D40 counts
    - Averages: LiDAR depth, sonar depth, acceleration, traffic PCU, hospital distance
    - Percentages: Rain %, LiDAR-selected %, Sonar-selected %
    - TOPSIS Priority Distribution (Critical, High, Medium, Low)
    """
    if not srmd_loaded and os.path.exists(srmd_dataset_file):
        _load_srmd_records_into_store()

    with _store_lock:
        N = max(1, len(telemetry_store))
        p_dist = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
        for item in telemetry_store:
            p = item.get("priority_level", "Low")
            p_dist[p] = p_dist.get(p, 0) + 1

        return {
            "status": "success",
            "detector": ACTIVE_DETECTOR,
            "total_records": len(telemetry_store),
            "d00_count": sum(1 for item in telemetry_store if item.get("damage_class") == "D00"),
            "d10_count": sum(1 for item in telemetry_store if item.get("damage_class") == "D10"),
            "d20_count": sum(1 for item in telemetry_store if item.get("damage_class") == "D20"),
            "d40_count": sum(1 for item in telemetry_store if item.get("damage_class") == "D40" or not item.get("damage_class")),
            "avg_lidar_depth": round(sum(item.get("lidar_depth_cm", 35.0) for item in telemetry_store) / N, 2),
            "avg_sonar_depth": round(sum(item.get("sonar_depth_cm", 35.0) for item in telemetry_store) / N, 2),
            "avg_acceleration": round(sum(item.get("z_accel_g", 1.2) for item in telemetry_store) / N, 2),
            "avg_traffic_pcu": round(sum(item.get("traffic_pcu", 15000) for item in telemetry_store) / N, 1),
            "avg_hospital_distance": round(sum(item.get("dist_hospital_km", 1.5) for item in telemetry_store) / N, 2),
            "rain_percentage": round((sum(1 for item in telemetry_store if item.get("rain_detected")) / N) * 100, 1),
            "lidar_selected_percentage": round((sum(1 for item in telemetry_store if "lidar" in item.get("processed_telemetry", {}).get("detection_modality", "lidar")) / N) * 100, 1),
            "sonar_selected_percentage": round((sum(1 for item in telemetry_store if "sonar" in item.get("processed_telemetry", {}).get("detection_modality", "")) / N) * 100, 1),
            "priority_distribution": p_dist,
            "records": list(telemetry_store)
        }


@app.get("/api/v1/preprocess/preview/{image_name}")
async def get_preprocess_preview(
    image_name: str,
    clip_limit: float = Query(2.0, ge=0.1, le=20.0),
    tile_grid_size: int = Query(8, ge=1, le=64)
):
    """
    Returns original and CLAHE-enhanced RGB images (as base64 data URIs and metadata)
    so the UI can display an interactive before/after contrast slider.
    """
    safe_name = os.path.basename(image_name)
    sample_dir = os.path.join(os.path.dirname(__file__), "data", "samples", "images")
    candidates = [
        os.path.join(sample_dir, safe_name),
        os.path.join(sample_dir, f"{safe_name}.jpg") if not safe_name.lower().endswith(".jpg") else os.path.join(sample_dir, safe_name)
    ]
    resolved_path = None
    for c in candidates:
        if os.path.exists(c) and os.path.isfile(c):
            resolved_path = c
            break

    if resolved_path is None:
        raise HTTPException(status_code=404, detail=f"Inspection image '{image_name}' not found.")

    orig_rgb, (w, h) = load_image(resolved_path)
    tile_grid = (int(tile_grid_size), int(tile_grid_size))
    clahe_rgb = apply_clahe(orig_rgb, clip_limit=float(clip_limit), tile_grid=tile_grid)

    orig_b64 = encode_image_base64(orig_rgb, fmt="JPEG", quality=90)
    clahe_b64 = encode_image_base64(clahe_rgb, fmt="JPEG", quality=90)
    resolved_filename = os.path.basename(resolved_path)

    return {
        "status": "success",
        "image_name": resolved_filename,
        "width": w,
        "height": h,
        "clip_limit": float(clip_limit),
        "tile_grid": list(tile_grid),
        "backend": "opencv_lab_clahe" if HAS_CV2 else "numpy_pil_fallback",
        "original_url": f"/api/v1/images/{resolved_filename}",
        "original_image": orig_b64,
        "clahe_image": clahe_b64,
        "original_image_base64": orig_b64,
        "clahe_image_base64": clahe_b64,
        "metrics": {
            "original_mean_luminance": round(float(np.mean(orig_rgb)), 2),
            "clahe_mean_luminance": round(float(np.mean(clahe_rgb)), 2),
            "original_contrast_std": round(float(np.std(orig_rgb)), 2),
            "clahe_contrast_std": round(float(np.std(clahe_rgb)), 2)
        }
    }


def _execute_prioritization_query(
    method: str,
    profile_info: Dict[str, Any],
    v: float = 0.5
) -> Dict[str, Any]:
    method_norm = (method or "both").strip().lower()
    if method_norm not in ("topsis", "vikor", "both"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid method '{method}'. Must be 'topsis', 'vikor', or 'both'."
        )

    if not srmd_loaded and os.path.exists(srmd_dataset_file):
        _load_srmd_records_into_store()

    with _store_lock:
        items = list(telemetry_store)

    w = np.asarray(profile_info["weights"], dtype=float)
    engine = MCDMPrioritizationEngine(weights=w, use_calibrated_thresholds=True)

    if not items:
        return {
            "status": "success",
            "method": method_norm,
            "profile": profile_info["profile"],
            "weights": w.tolist(),
            "fuzzy_bwm": profile_info,
            "priority_thresholds": engine.calibrated_thresholds,
            "calibration_note": THRESHOLD_CALIBRATION_NOTE,
            "spearman_correlation": 1.0,
            "total": 0,
            "results": []
        }

    if w.shape[0] == 4:
        matrix = np.array([
            [
                _get_item_severity_magnitude(it),
                float(it["traffic_pcu"]),
                float(it["dist_hospital_km"]),
                float(it["speed_kmh"])
            ]
            for it in items
        ])
    else:
        matrix = np.array([_build_record_6criteria_row(it) for it in items])

    all_res = engine.compute_all(matrix, v=v, weights=w, calibrated=True)

    ranked_rows = []
    for idx, it in enumerate(items):
        t_score = float(all_res["topsis_scores"][idx])
        t_rank = int(all_res["topsis_ranks"][idx])
        v_s = float(all_res["vikor_S"][idx])
        v_r = float(all_res["vikor_R"][idx])
        v_q = float(all_res["vikor_Q"][idx])
        v_rank = int(all_res["vikor_ranks"][idx])
        primary_rank = v_rank if method_norm == "vikor" else t_rank
        primary_score = round(1.0 - v_q, 4) if method_norm == "vikor" else t_score
        raw_vol = it["processed_telemetry"].get("calculated_volume_liters")
        raw_csi = it["processed_telemetry"].get("crack_severity_index")

        row_entry: Dict[str, Any] = {
            "id": it["id"],
            "rank": primary_rank,
            "priority_level": engine.classify_priority(primary_score, calibrated=True),
            "damage_class": it.get("damage_class", "D40"),
            "damage_label": it.get("damage_label", "Pothole"),
            "road_name": it.get("road_name", ""),
            "road_class": it.get("road_class", "ARTERIAL"),
            "road_class_weight": encode_road_class(it.get("road_class", "ARTERIAL")),
            "calculated_volume_liters": float(raw_vol) if raw_vol is not None else None,
            "crack_severity_index": float(raw_csi) if raw_csi is not None else None,
            "defect_severity_magnitude": _get_item_severity_magnitude(it),
            "traffic_pcu": int(it["traffic_pcu"]),
            "dist_hospital_km": float(it["dist_hospital_km"]),
            "speed_kmh": float(it["speed_kmh"]),
            "school_proximity_km": float(it.get("school_proximity_km", 0.5)),
            "nearest_school_name": it.get("nearest_school_name", "")
        }
        if method_norm in ("topsis", "both"):
            row_entry["topsis_score"] = t_score
            row_entry["topsis_rank"] = t_rank
        if method_norm in ("vikor", "both"):
            row_entry["vikor_S"] = v_s
            row_entry["vikor_R"] = v_r
            row_entry["vikor_Q"] = v_q
            row_entry["vikor_rank"] = v_rank
        ranked_rows.append(row_entry)

    ranked_rows.sort(key=lambda r: r["rank"])

    compromise_ids = [items[i]["id"] for i in all_res["vikor"]["compromise_solution"]]
    return {
        "status": "success",
        "method": method_norm,
        "profile": profile_info["profile"],
        "weights": np.round(w, 4).tolist(),
        "fuzzy_bwm": {
            "profile": profile_info["profile"],
            "label": profile_info["label"],
            "fuzzy_weights": [list(fw) for fw in profile_info["fuzzy_weights"]],
            "ksi_star": profile_info["ksi_star"],
            "consistency_index": profile_info["consistency_index"],
            "consistency_ratio": profile_info["consistency_ratio"],
            "best_idx": profile_info["best_idx"],
            "worst_idx": profile_info["worst_idx"]
        },
        "criteria": MCDMPrioritizationEngine.get_criteria_specs(extended=(w.shape[0] == 6)),
        "priority_thresholds": engine.calibrated_thresholds,
        "calibration_note": THRESHOLD_CALIBRATION_NOTE,
        "spearman_correlation": all_res["spearman_correlation"],
        "vikor_checks": {
            "v": all_res["vikor"]["v"],
            "DQ": all_res["vikor"]["DQ"],
            "q_gap": all_res["vikor"]["q_gap"],
            "acceptable_advantage": all_res["vikor"]["acceptable_advantage"],
            "acceptable_stability": all_res["vikor"]["acceptable_stability"],
            "compromise_solution": compromise_ids
        },
        "total": len(ranked_rows),
        "results": ranked_rows
    }


@app.get("/api/v1/prioritization")
async def get_prioritization(
    method: str = Query("both", description="Ranking method: 'topsis', 'vikor', or 'both'"),
    profile: str = Query("safety-first", description="Fuzzy BWM weight profile: 'safety-first', 'traffic-first', or 'equity-first'"),
    v: float = Query(0.5, ge=0.0, le=1.0, description="VIKOR group utility weight v")
):
    """
    Computes multi-criteria prioritization rankings over active road defects using
    TOPSIS, VIKOR, or both with Fuzzy BWM derived criteria weights.
    """
    try:
        prof = get_fuzzy_bwm_profile(profile=profile)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _execute_prioritization_query(method=method, profile_info=prof, v=v)


@app.post("/api/v1/prioritization")
@app.post("/api/v1/prioritization/custom")
async def post_custom_prioritization(req: CustomPrioritizationRequest):
    """
    Computes multi-criteria prioritization rankings using custom crisp weights or
    custom Fuzzy BWM Best-to-Others / Others-to-Worst comparison vectors.
    """
    try:
        prof = get_fuzzy_bwm_profile(
            profile=req.profile,
            custom_weights=req.weights,
            best_to_others=req.best_to_others,
            others_to_worst=req.others_to_worst,
            best_idx=req.best_idx,
            worst_idx=req.worst_idx
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _execute_prioritization_query(method=req.method, profile_info=prof, v=req.v)

