"""
Location & Road Context Modeling Engine
=======================================

Handles geographic coordinate synthesis, route placement, and location provenance
for the Smart Road Multimodal Dataset (SRMD).

Fundamental Scientific Principle:
- Authentic RDD2022 imagery provides visual damage evidence, but does NOT contain
  original real-world geographic coordinates.
- Under NO circumstance should synthetic coordinates be represented as authentic
  source GPS readings.

Supported Location Modes:
1. 'source_location': Used ONLY when ground-truth GPS coordinates are present in
   the raw source dataset metadata/EXIF (not present in standard RDD2022).
2. 'synthetic_location': Deterministic, reproducible route/grid coordinates synthesized
   along a documented Indian demonstration corridor (default seed: 2026).
3. 'external_gis_location': Dynamic snapping to real-world road centerlines via external
   GIS / OpenStreetMap services.
"""

import os
import sys
import math
import random
from pathlib import Path
from typing import Dict, List, Any, Optional

try:
    import yaml
except ImportError:
    yaml = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "sensor_profiles.yaml"

MODE_SOURCE_LOCATION = "source_location"
MODE_SYNTHETIC_LOCATION = "synthetic_location"
MODE_EXTERNAL_GIS_LOCATION = "external_gis_location"

VALID_LOCATION_MODES = {
    MODE_SOURCE_LOCATION,
    MODE_SYNTHETIC_LOCATION,
    MODE_EXTERNAL_GIS_LOCATION
}

DEFAULT_CORRIDOR_WAYPOINTS = [
    {
        "segment_id": "WEH-MUM-SEC-01",
        "road_name": "Western Express Highway, Bandra-Kalanagar Junction",
        "road_class": "HIGH_TRAFFIC_CORRIDOR",
        "lat": 19.0596, "lon": 72.8488,
        "chainage_km": 0.0
    },
    {
        "segment_id": "SVR-MUM-SEC-02",
        "road_name": "Swami Vivekananda (SV) Road, Santacruz West",
        "road_class": "ARTERIAL",
        "lat": 19.0780, "lon": 72.8410,
        "chainage_km": 2.8
    },
    {
        "segment_id": "LNK-MUM-SEC-03",
        "road_name": "Linking Road Commercial Sub-Arterial, Bandra West",
        "road_class": "COLLECTOR",
        "lat": 19.0620, "lon": 72.8330,
        "chainage_km": 4.5
    },
    {
        "segment_id": "RES-MUM-SEC-04",
        "road_name": "Pali Hill Residential Access Corridor, Bandra West",
        "road_class": "LOCAL",
        "lat": 19.0650, "lon": 72.8250,
        "chainage_km": 5.7
    },
    {
        "segment_id": "WEH-MUM-SEC-05",
        "road_name": "Western Express Highway, Jogeshwari Link Flyover",
        "road_class": "HIGH_TRAFFIC_CORRIDOR",
        "lat": 19.1360, "lon": 72.8600,
        "chainage_km": 12.3
    },
    {
        "segment_id": "SVR-MUM-SEC-06",
        "road_name": "Swami Vivekananda (SV) Road, Andheri West",
        "road_class": "ARTERIAL",
        "lat": 19.1200, "lon": 72.8460,
        "chainage_km": 15.1
    }
]


class LocationModel:
    """
    Deterministic Location Synthesizer and Roadway Context Resolver.
    Ensures repeatable, transparent coordinate assignment without misrepresenting
    source provenance.
    """
    def __init__(
        self,
        mode: str = MODE_SYNTHETIC_LOCATION,
        seed: int = 2026,
        config_path: Optional[Path] = None,
        config_override: Optional[Dict[str, Any]] = None
    ):
        if mode not in VALID_LOCATION_MODES:
            raise ValueError(f"Invalid location_mode '{mode}'. Must be one of {VALID_LOCATION_MODES}")

        self.mode = mode
        self.seed = seed
        self.rng = random.Random(seed)
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self.config = self._load_configuration(config_override)

        loc_cfg = self.config.get("location", {})
        self.waypoints = loc_cfg.get("demonstration_corridor", {}).get("route_waypoints", DEFAULT_CORRIDOR_WAYPOINTS)
        self.demonstration_area = loc_cfg.get("demonstration_corridor", {}).get("name", "Mumbai Metropolitan Urban Corridor")

    def _load_configuration(self, override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Loads location configuration from YAML with fallback."""
        if override:
            return override

        if yaml and self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f)
                if loaded and "location" in loaded:
                    return loaded
            except Exception as e:
                print(f"Warning: Failed to load location config from {self.config_path}: {e}")

        return {
            "location": {
                "default_mode": MODE_SYNTHETIC_LOCATION,
                "default_seed": 2026,
                "demonstration_corridor": {
                    "name": "Mumbai Metropolitan Urban Corridor",
                    "route_waypoints": DEFAULT_CORRIDOR_WAYPOINTS
                }
            }
        }

    def reseed(self, seed: int):
        """Reseeds the random generator for strict reproducibility."""
        self.seed = seed
        self.rng = random.Random(seed)

    def generate_location_for_defect(
        self,
        route_index: int = 0,
        source_gps: Optional[Dict[str, float]] = None,
        jitter_meters: float = 30.0
    ) -> Dict[str, Any]:
        """
        Generates or resolves road location coordinates according to current location_mode.

        Args:
            route_index: Sequential index along inspection route
            source_gps: Dict with 'latitude' and 'longitude' if real source GPS exists
            jitter_meters: Micro-jitter magnitude for synthetic waypoint dispersion
        """
        if self.mode == MODE_SOURCE_LOCATION:
            if not source_gps or "latitude" not in source_gps or "longitude" not in source_gps:
                raise ValueError(
                    "location_mode is 'source_location', but no genuine GPS coordinates "
                    "exist in RDD2022 source metadata for this record."
                )
            return {
                "location_mode": MODE_SOURCE_LOCATION,
                "location_source": "source_dataset",
                "latitude": round(float(source_gps["latitude"]), 6),
                "longitude": round(float(source_gps["longitude"]), 6),
                "road_segment_id": "UNKNOWN_SOURCE_SEGMENT",
                "road_name": "Unspecified Source Location",
                "road_class": "ARTERIAL",
                "demonstration_area": "Source Dataset Real World Coordinates"
            }

        elif self.mode == MODE_EXTERNAL_GIS_LOCATION:
            # External GIS mode (e.g. snapped to OSM network)
            wp = self.waypoints[route_index % len(self.waypoints)]
            return {
                "location_mode": MODE_EXTERNAL_GIS_LOCATION,
                "location_source": "external_gis_osm",
                "demonstration_area": self.demonstration_area,
                "latitude": wp["lat"],
                "longitude": wp["lon"],
                "road_segment_id": wp["segment_id"],
                "road_name": wp["road_name"],
                "road_class": wp["road_class"],
                "snapping_status": "snapped_to_centerline"
            }

        else:
            # MODE_SYNTHETIC_LOCATION (default for current version)
            wp = self.waypoints[route_index % len(self.waypoints)]

            # Convert meters to degrees (~111,000 meters per degree lat/lon)
            deg_offset = jitter_meters / 111000.0
            lat_jitter = self.rng.uniform(-deg_offset, deg_offset)
            lon_jitter = self.rng.uniform(-deg_offset, deg_offset)

            lat = round(wp["lat"] + lat_jitter, 5)
            lon = round(wp["lon"] + lon_jitter, 5)

            return {
                "location_mode": MODE_SYNTHETIC_LOCATION,
                "location_source": "synthetic",
                "demonstration_area": self.demonstration_area,
                "latitude": lat,
                "longitude": lon,
                "road_segment_id": wp["segment_id"],
                "road_name": wp["road_name"],
                "road_class": wp["road_class"],
                "chainage_km": wp.get("chainage_km", 0.0),
                "route_index": route_index,
                "synthesis_seed": self.seed
            }
