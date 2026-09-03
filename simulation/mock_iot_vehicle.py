import argparse
import random
import sys
import time
import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

API_URL = "http://localhost:8000/api/v1/telemetry/ingest"

# Mumbai Western Corridor inspection coordinates
CORRIDOR_WAYPOINTS = [
    {"lat": 19.0760, "lon": 72.8777, "pcu": 24500, "hosp_dist": 0.4, "speed": 42.0},
    {"lat": 19.0780, "lon": 72.8790, "pcu": 21000, "hosp_dist": 0.6, "speed": 44.0},
    {"lat": 19.0800, "lon": 72.8810, "pcu": 14200, "hosp_dist": 1.5, "speed": 50.0},
    {"lat": 19.0825, "lon": 72.8835, "pcu": 8500,  "hosp_dist": 3.2, "speed": 55.0},
    {"lat": 19.0850, "lon": 72.8860, "pcu": 18000, "hosp_dist": 1.1, "speed": 48.0},
    {"lat": 19.0875, "lon": 72.8885, "pcu": 26000, "hosp_dist": 0.5, "speed": 40.0},
]

def run_simulation(loop: bool = False, count: int = 3, interval: float = 2.0):
    print(f"🚗 Starting Vehicle Telemetry Simulation (loop={loop}, count={count}, interval={interval}s)...")
    iteration = 0

    try:
        while True:
            for wp in CORRIDOR_WAYPOINTS:
                iteration += 1
                is_raining = random.random() < 0.35
                luminance = random.uniform(12.0, 32.0) if is_raining else random.uniform(55.0, 95.0)

                # Realistic depth variation above 30cm chassis clearance
                depth_offset = random.uniform(2.0, 18.0)
                sensor_reading = round(30.0 + depth_offset, 1)

                # Heavy impact randomly occurring when hitting deeper holes
                is_heavy_impact = depth_offset > 8.0 and random.random() < 0.6
                z_accel = round(random.uniform(1.6, 2.8) if is_heavy_impact else random.uniform(0.6, 1.4), 2)

                payload = {
                    "vehicle_id": "INSPECTION_FLEET_01",
                    "latitude": round(wp["lat"] + random.uniform(-0.0003, 0.0003), 4),
                    "longitude": round(wp["lon"] + random.uniform(-0.0003, 0.0003), 4),
                    "rain_detected": is_raining,
                    "mean_luminance": round(luminance, 1),
                    "lidar_depth_cm": sensor_reading,
                    "sonar_depth_cm": round(sensor_reading + random.uniform(-0.5, 0.5), 1),
                    "z_accel_g": z_accel,
                    "surface_area_sqm": round(random.uniform(0.15, 0.75), 2),
                    "traffic_pcu": wp["pcu"],
                    "dist_hospital_km": wp["hosp_dist"],
                    "speed_kmh": wp["speed"]
                }

                try:
                    res = requests.post(API_URL, json=payload, timeout=5)
                    if res.status_code == 200:
                        data = res.json()
                        pothole_id = data.get("record", {}).get("id", "UNKNOWN")
                        p_level = data.get("priority_level", "N/A")
                        t_score = data.get("topsis_score", 0.0)
                        modality = data.get("processed_telemetry", {}).get("detection_modality", "N/A")
                        print(f"[{iteration}] 📍 {pothole_id} ({payload['latitude']}, {payload['longitude']}) | {modality} | Priority: {p_level} ({t_score})")
                    else:
                        print(f"[{iteration}] ⚠️ Ingest HTTP {res.status_code}: {res.text}")
                except Exception as e:
                    print(f"[{iteration}] ❌ Connection error to {API_URL}: {e}")

                time.sleep(interval)
                if not loop and iteration >= count:
                    print(f"✅ Finished sending {iteration} simulated telemetry events.")
                    return
    except KeyboardInterrupt:
        print("\n🛑 Simulation stopped by user.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate IoT inspection vehicle telemetry")
    parser.add_argument("--loop", action="store_true", help="Run simulation in an endless loop")
    parser.add_argument("--count", type=int, default=3, help="Number of points to send if not looping (default: 3)")
    parser.add_argument("--interval", type=float, default=2.0, help="Interval between requests in seconds (default: 2.0)")
    args = parser.parse_args()

    run_simulation(loop=args.loop, count=args.count, interval=args.interval)
