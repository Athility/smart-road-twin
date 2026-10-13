#!/usr/bin/env python3
"""
scripts/sync_frontend_fallback.py

Generates authoritative frontend fallback datasets directly from
data/processed/srmd/srmd_records.json so that when the FastAPI backend is offline,
the frontend displays identical data, scores, ranks, and tiers.

Outputs:
  - frontend/src/data/fallback_records.json
  - frontend/src/data/fallback_profile_map.json
  - frontend/public/fallback_records.json
"""

import json
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import main as app_module
from services.backend.app.services.mcdm_engine import (
    MCDMPrioritizationEngine,
    build_mcdm_row_from_store_item,
)

def sync_fallback_data():
    srmd_json_path = PROJECT_ROOT / "data" / "processed" / "srmd" / "srmd_records.json"
    if not srmd_json_path.exists():
        raise FileNotFoundError(f"SRMD records not found at {srmd_json_path}")

    # Load canonical records into store using the backend's shared loader
    loaded = app_module._load_srmd_records_into_store()
    if not loaded or not app_module.telemetry_store:
        raise RuntimeError("Failed to load SRMD records into telemetry_store")

    records = list(app_module.telemetry_store)
    print(f"[Sync] Loaded {len(records)} authoritative records from {srmd_json_path}")

    # Build decision matrix using the shared MCDM row builder
    matrix = np.array([build_mcdm_row_from_store_item(item) for item in records])

    # Compute MCDM rankings across all three weight profiles
    profiles = ["safety-first", "traffic-first", "equity-first"]
    profile_map = {}

    for profile in profiles:
        engine = MCDMPrioritizationEngine(profile=profile, use_calibrated_thresholds=True)
        res = engine.compute_all(matrix, v=0.5, calibrated=True)
        profile_map[profile] = {}
        for idx, item in enumerate(records):
            profile_map[profile][item["id"]] = {
                "topsis_rank": int(res["topsis_ranks"][idx]),
                "topsis_score": round(float(res["topsis_scores"][idx]), 4),
                "vikor_rank": int(res["vikor_ranks"][idx]),
                "priority_level": str(res["priority_levels"][idx]),
                "vikor_Q": round(float(res["vikor_Q"][idx]), 4),
            }

    # Ensure output directories exist
    frontend_data_dir = PROJECT_ROOT / "frontend" / "src" / "data"
    frontend_public_dir = PROJECT_ROOT / "frontend" / "public"
    frontend_data_dir.mkdir(parents=True, exist_ok=True)
    frontend_public_dir.mkdir(parents=True, exist_ok=True)

    # Export fallback records
    fallback_records_path = frontend_data_dir / "fallback_records.json"
    fallback_records_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"[Sync] Wrote {len(records)} fallback records to {fallback_records_path}")

    # Also write to public for static assets
    public_fallback_path = frontend_public_dir / "fallback_records.json"
    public_fallback_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"[Sync] Wrote public fallback records to {public_fallback_path}")

    # Export fallback profile map
    fallback_profile_map_path = frontend_data_dir / "fallback_profile_map.json"
    fallback_profile_map_path.write_text(json.dumps(profile_map, indent=2), encoding="utf-8")
    print(f"[Sync] Wrote fallback profile map to {fallback_profile_map_path}")

    return records, profile_map

if __name__ == "__main__":
    sync_fallback_data()
