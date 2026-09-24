from fastapi import FastAPI
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
from services.backend.app.services.mcdm_engine import MCDMPrioritizationEngine

app = FastAPI(
    title="Smart Digital Twin API",
    description="Municipal road infrastructure monitoring, AI depth estimation, and TOPSIS hazard prioritization API",
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
    traffic_pcu: int = Field(..., ge=0, le=200000, description="Traffic volume in Passenger Car Units")
    dist_hospital_km: float = Field(..., ge=0.01, le=200.0, description="Distance to nearest hospital (km)")
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

# In-memory storage and thread-safe lock for telemetry events
telemetry_store: List[Dict[str, Any]] = []
_store_lock = threading.Lock()
_pothole_counter = 0

def recompute_topsis_for_all():
    if not telemetry_store:
        return
    engine = MCDMPrioritizationEngine()
    if len(telemetry_store) == 1:
        # Reference against standard road thresholds if only 1 item
        ref_worst = [0.1, 1000, 10.0, 10.0]
        ref_best = [80.0, 30000, 0.1, 70.0]
        item = telemetry_store[0]
        mat = np.array([
            [
                item["processed_telemetry"]["calculated_volume_liters"],
                item["traffic_pcu"],
                item["dist_hospital_km"],
                item["speed_kmh"]
            ],
            ref_worst,
            ref_best
        ])
        scores = engine.compute_topsis(mat)
        score = float(scores[0])
        item["topsis_score"] = score
        item["priority_level"] = engine.classify_priority(score)
        return

    matrix_rows = []
    for item in telemetry_store:
        matrix_rows.append([
            item["processed_telemetry"]["calculated_volume_liters"],
            item["traffic_pcu"],
            item["dist_hospital_km"],
            item["speed_kmh"]
        ])
    matrix = np.array(matrix_rows)
    scores = engine.compute_topsis(matrix)
    for idx, item in enumerate(telemetry_store):
        item["topsis_score"] = float(scores[idx])
        item["priority_level"] = engine.classify_priority(float(scores[idx]))

def process_and_store_telemetry(payload_dict: dict) -> dict:
    global _pothole_counter
    switcher = EnvironmentalSensorSwitching(baseline_chassis_height_cm=30.0)
    depth_res = switcher.evaluate_environment_and_measure(
        rain_detected=payload_dict["rain_detected"],
        mean_luminance=payload_dict["mean_luminance"],
        lidar_depth_cm=payload_dict["lidar_depth_cm"],
        sonar_depth_cm=payload_dict["sonar_depth_cm"],
        z_accel_g=payload_dict["z_accel_g"],
        surface_area_sqm=payload_dict["surface_area_sqm"]
    )

    with _store_lock:
        _pothole_counter += 1
        img_id = payload_dict.get("image_id", f"India_{_pothole_counter:06d}")
        img_url = payload_dict.get("image_url", f"/api/v1/images/{img_id}.jpg")
        record = {
            "id": payload_dict.get("id") or f"POTHOLE-{_pothole_counter:04d}",
            "timestamp": payload_dict.get("timestamp") or datetime.now(timezone.utc).isoformat(),
            "vehicle_id": payload_dict["vehicle_id"],
            "latitude": float(payload_dict["latitude"]),
            "longitude": float(payload_dict["longitude"]),
            "rain_detected": bool(payload_dict["rain_detected"]),
            "mean_luminance": float(payload_dict["mean_luminance"]),
            "lidar_depth_cm": float(payload_dict["lidar_depth_cm"]),
            "sonar_depth_cm": float(payload_dict["sonar_depth_cm"]),
            "z_accel_g": float(payload_dict["z_accel_g"]),
            "surface_area_sqm": float(payload_dict["surface_area_sqm"]),
            "traffic_pcu": int(payload_dict["traffic_pcu"]),
            "dist_hospital_km": float(payload_dict["dist_hospital_km"]),
            "speed_kmh": float(payload_dict["speed_kmh"]),
            "data_source": payload_dict.get("data_source", "MULTIMODAL"),
            "damage_class": payload_dict.get("damage_class", "D40"),
            "damage_label": payload_dict.get("damage_label", "Pothole"),
            "confidence": float(payload_dict.get("confidence", 0.94)),
            "image_id": img_id,
            "image_url": img_url,
            "bbox": payload_dict.get("bbox", [180.0, 260.0, 420.0, 480.0]),
            "road_name": payload_dict.get("road_name", "Western Express Highway Corridor"),
            "road_class": payload_dict.get("road_class", "HIGH_TRAFFIC_CORRIDOR"),
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

# Load from SRMD dataset if available, else seed with default telemetry
srmd_dataset_file = os.path.join(os.path.dirname(__file__), "data", "processed", "srmd", "srmd_records.json")
srmd_loaded = False

if os.path.exists(srmd_dataset_file):
    try:
        with open(srmd_dataset_file, "r", encoding="utf-8") as f:
            srmd_items = json.load(f)
        for r in srmd_items:
            img_file = r["visual"].get("image_filename", f"{r['source']['image_id']}.jpg")
            switcher = EnvironmentalSensorSwitching(baseline_chassis_height_cm=30.0)
            depth_eval = switcher.evaluate_environment_and_measure(
                rain_detected=r["environment"]["rain_detected"],
                mean_luminance=r["environment"]["luminance_lux"],
                lidar_depth_cm=r["lidar"]["depth_cm"],
                sonar_depth_cm=r["sonar"]["depth_cm"],
                z_accel_g=r["accelerometer"]["z_accel_g"],
                surface_area_sqm=r["visual"]["surface_area_sqm"]
            )
            # Override calculated volume with exact SRMD modeled volume
            depth_eval["calculated_volume_liters"] = r["visual"]["calculated_volume_liters"]
            depth_eval["estimated_depth_cm"] = r["environment"]["selected_depth_cm"]
            depth_eval["detection_modality"] = r["environment"]["preferred_modality"]
            depth_eval["dynamic_impact_confirmed"] = r["accelerometer"]["dynamic_impact_triggered"]

            rec = {
                "id": r["event_id"],
                "timestamp": r.get("provenance", {}).get("generation_timestamp", datetime.now(timezone.utc).isoformat()),
                "vehicle_id": r.get("vehicle", {}).get("vehicle_id", "INSPECTION_FLEET_01"),
                "latitude": float(r["location"]["latitude"]),
                "longitude": float(r["location"]["longitude"]),
                "rain_detected": bool(r["environment"]["rain_detected"]),
                "mean_luminance": float(r["environment"]["luminance_lux"]),
                "lidar_depth_cm": float(r["lidar"]["depth_cm"]),
                "sonar_depth_cm": float(r["sonar"]["depth_cm"]),
                "z_accel_g": float(r["accelerometer"]["z_accel_g"]),
                "surface_area_sqm": float(r["visual"]["surface_area_sqm"]),
                "traffic_pcu": int(r["traffic"]["traffic_pcu"]),
                "dist_hospital_km": float(r["infrastructure"]["dist_hospital_km"]),
                "speed_kmh": float(r.get("vehicle", {}).get("speed_kmh", 40.0)),
                "data_source": "MULTIMODAL",
                "damage_class": r["source"].get("annotation_class", "D40"),
                "damage_label": r["visual"].get("damage_label", "Pothole"),
                "confidence": float(r["visual"].get("confidence", 0.94)),
                "image_id": r["source"].get("image_id", "India_000045"),
                "image_url": f"/api/v1/images/{img_file}",
                "bbox": r["visual"].get("bbox", []),
                "road_name": r["location"].get("road_name", "Western Express Highway"),
                "road_class": r["location"].get("road_class", "HIGH_TRAFFIC_CORRIDOR"),
                "processed_telemetry": depth_eval,
                "topsis_score": 0.0,
                "priority_level": "Low"
            }
            telemetry_store.append(rec)
        recompute_topsis_for_all()
        srmd_loaded = True
    except Exception as e:
        print(f"Warning: Failed to load SRMD dataset ({e}), using default seeds.")

if not srmd_loaded:
    initial_seeds = [
        {
            "vehicle_id": "INSPECTION_FLEET_01",
            "latitude": 19.05939, "longitude": 72.8488,
            "rain_detected": True, "mean_luminance": 22.1,
            "lidar_depth_cm": 45.94, "sonar_depth_cm": 45.95,
            "z_accel_g": 2.53, "surface_area_sqm": 1.81,
            "traffic_pcu": 27151, "dist_hospital_km": 2.4, "speed_kmh": 41.7,
            "data_source": "MULTIMODAL",
            "damage_class": "D40", "damage_label": "Pothole", "confidence": 0.96,
            "image_id": "India_000045", "image_url": "/api/v1/images/India_000045.jpg",
            "bbox": [180.0, 260.0, 420.0, 480.0],
            "road_name": "Western Express Highway, Bandra-Kalanagar Junction",
            "road_class": "HIGH_TRAFFIC_CORRIDOR"
        },
        {
            "vehicle_id": "INSPECTION_FLEET_01",
            "latitude": 19.07801, "longitude": 72.84119,
            "rain_detected": False, "mean_luminance": 20.1,
            "lidar_depth_cm": 34.13, "sonar_depth_cm": 34.27,
            "z_accel_g": 1.16, "surface_area_sqm": 3.59,
            "traffic_pcu": 16381, "dist_hospital_km": 1.31, "speed_kmh": 36.5,
            "data_source": "MULTIMODAL",
            "damage_class": "D20", "damage_label": "Alligator Crack", "confidence": 0.91,
            "image_id": "India_000102", "image_url": "/api/v1/images/India_000102.jpg",
            "bbox": [120.0, 300.0, 580.0, 520.0],
            "road_name": "Swami Vivekananda (SV) Road, Santacruz West",
            "road_class": "ARTERIAL"
        },
        {
            "vehicle_id": "INSPECTION_FLEET_01",
            "latitude": 19.0800, "longitude": 72.8810,
            "rain_detected": False, "mean_luminance": 75.0,
            "lidar_depth_cm": 30.79, "sonar_depth_cm": 30.92,
            "z_accel_g": 1.03, "surface_area_sqm": 0.32,
            "traffic_pcu": 11391, "dist_hospital_km": 0.88, "speed_kmh": 50.0,
            "data_source": "MULTIMODAL",
            "damage_class": "D00", "damage_label": "Longitudinal Crack", "confidence": 0.88,
            "image_id": "India_000155", "image_url": "/api/v1/images/India_000155.jpg",
            "bbox": [100.0, 200.0, 350.0, 400.0],
            "road_name": "Link Road Corridor",
            "road_class": "COLLECTOR"
        },
        {
            "vehicle_id": "INSPECTION_FLEET_01",
            "latitude": 19.0825, "longitude": 72.8835,
            "rain_detected": False, "mean_luminance": 82.0,
            "lidar_depth_cm": 30.99, "sonar_depth_cm": 30.80,
            "z_accel_g": 1.02, "surface_area_sqm": 0.18,
            "traffic_pcu": 5596, "dist_hospital_km": 0.96, "speed_kmh": 55.0,
            "data_source": "MULTIMODAL",
            "damage_class": "D10", "damage_label": "Transverse Crack", "confidence": 0.84,
            "image_id": "India_000210", "image_url": "/api/v1/images/India_000210.jpg",
            "bbox": [150.0, 180.0, 320.0, 380.0],
            "road_name": "Local Arterial Bypass",
            "road_class": "LOCAL"
        }
    ]
    for seed in initial_seeds:
        process_and_store_telemetry(seed)

def calculate_summary():
    with _store_lock:
        total_volume = round(sum(item["processed_telemetry"]["calculated_volume_liters"] for item in telemetry_store), 2)
        avg_volume = round(total_volume / len(telemetry_store), 2) if telemetry_store else 0.0
        
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
            "latest_update": latest_item["timestamp"] if latest_item else None
        }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Smart Digital Twin API",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.post("/api/v1/telemetry/ingest")
async def ingest_telemetry(payload: TelemetryPayload):
    record = process_and_store_telemetry(payload.model_dump())
    return {
        "status": "success",
        "record": record,
        "processed_telemetry": record["processed_telemetry"],
        "topsis_score": record["topsis_score"],
        "priority_level": record["priority_level"]
    }

@app.get("/api/v1/telemetry/ingest")
@app.get("/api/v1/telemetry")
async def get_telemetry():
    return {
        "status": "success",
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
    Returns research-grade dataset explorer summary and individual multimodal records:
    - Total records
    - D00, D10, D20, D40 counts
    - Averages: LiDAR depth, sonar depth, acceleration, traffic PCU, hospital distance
    - Percentages: Rain %, LiDAR-selected %, Sonar-selected %
    - TOPSIS Priority Distribution (Critical, High, Medium, Low)
    """
    srmd_path = os.path.join(os.path.dirname(__file__), "data", "processed", "srmd", "srmd_records.json")
    if os.path.exists(srmd_path):
        try:
            with open(srmd_path, "r", encoding="utf-8") as f:
                records = json.load(f)
            N = len(records)
            d00 = sum(1 for r in records if r["source"]["annotation_class"] == "D00")
            d10 = sum(1 for r in records if r["source"]["annotation_class"] == "D10")
            d20 = sum(1 for r in records if r["source"]["annotation_class"] == "D20")
            d40 = sum(1 for r in records if r["source"]["annotation_class"] == "D40")

            avg_lidar = round(sum(r["lidar"]["depth_cm"] for r in records) / N, 2)
            avg_sonar = round(sum(r["sonar"]["depth_cm"] for r in records) / N, 2)
            avg_accel = round(sum(r["accelerometer"]["z_accel_g"] for r in records) / N, 2)
            avg_pcu = round(sum(r["traffic"]["traffic_pcu"] for r in records) / N, 1)
            avg_hosp = round(sum(r["infrastructure"]["dist_hospital_km"] for r in records) / N, 2)

            rain_pct = round((sum(1 for r in records if r["environment"]["rain_detected"]) / N) * 100, 1)
            lidar_pct = round((sum(1 for r in records if "lidar" in r["environment"]["preferred_modality"]) / N) * 100, 1)
            sonar_pct = round((sum(1 for r in records if "sonar" in r["environment"]["preferred_modality"]) / N) * 100, 1)

            p_dist = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
            for item in telemetry_store:
                p = item.get("priority_level", "Low")
                p_dist[p] = p_dist.get(p, 0) + 1

            return {
                "status": "success",
                "total_records": N,
                "d00_count": d00,
                "d10_count": d10,
                "d20_count": d20,
                "d40_count": d40,
                "avg_lidar_depth": avg_lidar,
                "avg_sonar_depth": avg_sonar,
                "avg_acceleration": avg_accel,
                "avg_traffic_pcu": avg_pcu,
                "avg_hospital_distance": avg_hosp,
                "rain_percentage": rain_pct,
                "lidar_selected_percentage": lidar_pct,
                "sonar_selected_percentage": sonar_pct,
                "priority_distribution": p_dist,
                "records": telemetry_store
            }
        except Exception as e:
            print("Error computing dataset explorer stats from srmd_records:", e)

    # Fallback to computing from in-memory telemetry_store
    N = max(1, len(telemetry_store))
    p_dist = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
    for item in telemetry_store:
        p = item.get("priority_level", "Low")
        p_dist[p] = p_dist.get(p, 0) + 1

    return {
        "status": "success",
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
        "records": telemetry_store
    }


