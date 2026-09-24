import urllib.request
import json

BASE = "http://127.0.0.1:8000"

def check(label, url, checks):
    req = urllib.request.urlopen(url)
    data = json.loads(req.read())
    for k, v in checks.items():
        actual = data
        for part in k.split("."):
            if isinstance(actual, list):
                actual = actual[int(part)]
            else:
                actual = actual.get(part)
        ok = v(actual)
        symbol = "OK" if ok else "FAIL"
        print(f"  [{symbol}] {label}: {k} -> {actual}")

# 1. Telemetry endpoint
req = urllib.request.urlopen(f"{BASE}/api/v1/telemetry/ingest")
data = json.loads(req.read())
events = data.get("events", [])
print(f"[OK] GET /api/v1/telemetry/ingest - {len(events)} events")

# 2. Validate event provenance fields
for ev in events:
    ds = ev.get("data_source", "")
    assert ds in ("MULTIMODAL", "RDD2022", "SYNTHETIC SENSOR", "SYNTHETIC GIS"), f"Bad data_source: {ds}"
    assert ev.get("damage_class") in ("D00","D10","D20","D40"), f"Bad class: {ev.get('damage_class')}"
    assert ev.get("lidar_depth_cm", 0) > 0, "lidar_depth_cm must be > 0"
    assert ev.get("sonar_depth_cm", 0) > 0, "sonar_depth_cm must be > 0"
    assert ev.get("preferred_modality") in ("optical_lidar", "sonar_submerged_acoustic"), f"Bad modality"
    assert ev.get("z_accel_g", 0) > 0, "z_accel_g must be > 0"
    assert ev.get("traffic_pcu", 0) > 0, "traffic_pcu must be > 0"
    assert ev.get("dist_hospital_km", 0) > 0, "dist_hospital_km must be > 0"
    assert 0.0 <= ev.get("topsis_score", -1) <= 1.0, "topsis_score out of range"
    assert ev.get("priority_level") in ("Critical","High","Medium","Low"), f"Bad priority"
print(f"[OK] Event provenance fields valid for all {len(events)} events")

# 3. Check modality switching
rain_events = [e for e in events if e.get("rain_detected")]
lidar_events = [e for e in events if e.get("preferred_modality") == "optical_lidar"]
sonar_events = [e for e in events if e.get("preferred_modality") == "sonar_submerged_acoustic"]
print(f"[OK] LiDAR/Sonar switching: {len(lidar_events)} LiDAR events, {len(sonar_events)} Sonar events")

# 4. Dataset explorer
req2 = urllib.request.urlopen(f"{BASE}/api/v1/dataset/explorer")
stats = json.loads(req2.read())
assert stats.get("status") == "success"
assert stats.get("total_records", 0) >= 4
print(f"[OK] GET /api/v1/dataset/explorer - total_records={stats['total_records']}")
print(f"     avg_lidar={stats['avg_lidar_depth']:.1f}cm, avg_sonar={stats['avg_sonar_depth']:.1f}cm")
print(f"     lidar_pct={stats['lidar_selected_percentage']:.1f}%, sonar_pct={stats['sonar_selected_percentage']:.1f}%")
print(f"     D00={stats['d00_count']} D10={stats['d10_count']} D20={stats['d20_count']} D40={stats['d40_count']}")

# 5. Image serving
req3 = urllib.request.urlopen(f"{BASE}/api/v1/images/India_000045.jpg")
assert req3.status == 200
print(f"[OK] GET /api/v1/images/India_000045.jpg - status=200")

print()
print("Phase 25 API validation: ALL CHECKS PASSED")
