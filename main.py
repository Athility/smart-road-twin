from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import sys, os, threading
from datetime import datetime, timezone
from typing import List, Dict, Any

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
        record = {
            "id": f"POTHOLE-{_pothole_counter:04d}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
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

# Seed with initial telemetry data along the inspection route
initial_seeds = [
    {
        "vehicle_id": "INSPECTION_FLEET_01",
        "latitude": 19.0760, "longitude": 72.8777,
        "rain_detected": True, "mean_luminance": 18.5,
        "lidar_depth_cm": 44.5, "sonar_depth_cm": 45.8,
        "z_accel_g": 2.4, "surface_area_sqm": 0.65,
        "traffic_pcu": 24500, "dist_hospital_km": 0.4, "speed_kmh": 42.0
    },
    {
        "vehicle_id": "INSPECTION_FLEET_01",
        "latitude": 19.0780, "longitude": 72.8790,
        "rain_detected": True, "mean_luminance": 22.0,
        "lidar_depth_cm": 38.5, "sonar_depth_cm": 39.2,
        "z_accel_g": 1.9, "surface_area_sqm": 0.45,
        "traffic_pcu": 21000, "dist_hospital_km": 0.6, "speed_kmh": 44.0
    },
    {
        "vehicle_id": "INSPECTION_FLEET_01",
        "latitude": 19.0800, "longitude": 72.8810,
        "rain_detected": False, "mean_luminance": 75.0,
        "lidar_depth_cm": 35.2, "sonar_depth_cm": 34.0,
        "z_accel_g": 1.4, "surface_area_sqm": 0.32,
        "traffic_pcu": 14200, "dist_hospital_km": 1.5, "speed_kmh": 50.0
    },
    {
        "vehicle_id": "INSPECTION_FLEET_01",
        "latitude": 19.0825, "longitude": 72.8835,
        "rain_detected": False, "mean_luminance": 82.0,
        "lidar_depth_cm": 32.1, "sonar_depth_cm": 31.5,
        "z_accel_g": 0.9, "surface_area_sqm": 0.18,
        "traffic_pcu": 8500, "dist_hospital_km": 3.2, "speed_kmh": 55.0
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

