"""
Urban Traffic Density & PCU Model
=================================

Simulates realistic roadway traffic volume in Passenger Car Units (PCU)
and macroscopic speed-density flow characteristics based on Indian Roads Congress
(IRC) urban arterial guidelines.

Physical Dynamics:
- Road classification profiles:
    * LOCAL: Lower traffic volume (residential / neighborhood streets, 1,500 - 6,000 PCU)
    * COLLECTOR: Medium traffic volume (commercial feeder streets, 6,000 - 14,000 PCU)
    * ARTERIAL: High traffic volume (primary city thoroughfares, 14,000 - 24,000 PCU)
    * HIGH_TRAFFIC_CORRIDOR: Very high traffic volume (expressways & national corridors, 24,000 - 38,000 PCU)
- Traffic is intrinsically correlated with roadway location and segment hierarchy.
- Output includes road_class, traffic_pcu, and traffic_source = "synthetic",
  enabling seamless future replacement with real ITS sensor feeds without altering
  the TOPSIS prioritization interface.
"""

import os
import sys
import math
import random
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

try:
    import yaml
except ImportError:
    yaml = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "sensor_profiles.yaml"

DEFAULT_TRAFFIC_CONFIG = {
    "traffic": {
        "traffic_source": "synthetic",
        "generation_method": "controlled_distribution",
        "road_classes": {
            "LOCAL": {
                "name": "Local Municipal Street",
                "description": "Residential and neighborhood access roads",
                "capacity_pcu": 8000,
                "pcu_range": {"min_pcu": 1500, "max_pcu": 6000},
                "free_flow_speed_kmh": 30.0,
                "congestion_speed_kmh": 15.0,
                "commercial_vehicle_percentage_range": [2.0, 8.0]
            },
            "COLLECTOR": {
                "name": "Urban Collector Street",
                "description": "Sub-arterial feeder streets channeling neighborhood traffic",
                "capacity_pcu": 15000,
                "pcu_range": {"min_pcu": 6000, "max_pcu": 14000},
                "free_flow_speed_kmh": 40.0,
                "congestion_speed_kmh": 20.0,
                "commercial_vehicle_percentage_range": [6.0, 14.0]
            },
            "ARTERIAL": {
                "name": "Primary Urban Arterial",
                "description": "Major multi-lane city thoroughfares linking commercial nodes",
                "capacity_pcu": 26000,
                "pcu_range": {"min_pcu": 14000, "max_pcu": 24000},
                "free_flow_speed_kmh": 52.0,
                "congestion_speed_kmh": 25.0,
                "commercial_vehicle_percentage_range": [10.0, 18.0]
            },
            "HIGH_TRAFFIC_CORRIDOR": {
                "name": "High Traffic Express Corridor",
                "description": "Regional expressways and national highway transit links",
                "capacity_pcu": 38000,
                "pcu_range": {"min_pcu": 24000, "max_pcu": 38000},
                "free_flow_speed_kmh": 68.0,
                "congestion_speed_kmh": 32.0,
                "commercial_vehicle_percentage_range": [14.0, 24.0]
            }
        }
    }
}

CLASS_ALIASES = {
    "local": "LOCAL",
    "local_collector": "LOCAL",
    "residential": "LOCAL",
    "collector": "COLLECTOR",
    "sub_arterial": "COLLECTOR",
    "arterial": "ARTERIAL",
    "primary_arterial": "ARTERIAL",
    "secondary_arterial": "COLLECTOR",
    "expressway": "HIGH_TRAFFIC_CORRIDOR",
    "expressway_arterial": "HIGH_TRAFFIC_CORRIDOR",
    "highway": "HIGH_TRAFFIC_CORRIDOR",
    "high_traffic_corridor": "HIGH_TRAFFIC_CORRIDOR"
}


class TrafficModel:
    """
    Configurable Urban Traffic Model with location-correlated PCU ranges
    and standard road-context profiles.
    """
    def __init__(
        self,
        config_path: Optional[Path] = None,
        config_override: Optional[Dict[str, Any]] = None,
        seed: int = 42
    ):
        self.rng = random.Random(seed)
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self.config = self._load_configuration(config_override)

    def _load_configuration(self, override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Loads traffic configuration from YAML with fallback."""
        if override:
            return override

        if yaml and self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f)
                if loaded and "traffic" in loaded:
                    return loaded
            except Exception as e:
                print(f"Warning: Failed to load traffic config from {self.config_path}: {e}")

        return DEFAULT_TRAFFIC_CONFIG

    def reseed(self, seed: int):
        self.rng = random.Random(seed)

    def resolve_road_class(self, raw_class: Optional[str]) -> str:
        """Normalizes road class name or alias to canonical form."""
        if not raw_class:
            return "HIGH_TRAFFIC_CORRIDOR"
        clean = raw_class.strip().lower()
        if clean in CLASS_ALIASES:
            return CLASS_ALIASES[clean]
        upper = raw_class.strip().upper()
        road_classes = self.config.get("traffic", {}).get("road_classes", DEFAULT_TRAFFIC_CONFIG["traffic"]["road_classes"])
        if upper in road_classes:
            return upper
        return "HIGH_TRAFFIC_CORRIDOR"

    def infer_road_class_from_location(
        self,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        road_segment_id: Optional[str] = None,
        road_name: Optional[str] = None
    ) -> str:
        """
        Infers the road class based on spatial attributes and segment identifiers.
        Ensures traffic volume is never generated in isolation from geography.
        """
        text = f"{road_segment_id or ''} {road_name or ''}".lower()

        if any(k in text for k in ["express", "highway", "weh-", "eeh-", "freeway", "nh-"]):
            return "HIGH_TRAFFIC_CORRIDOR"
        if any(k in text for k in ["arterial", "marg", "sv road", "svr-", "link road", "broadway"]):
            return "ARTERIAL"
        if any(k in text for k in ["collector", "feeder", "lnk-", "cross road", "junction"]):
            return "COLLECTOR"
        if any(k in text for k in ["local", "lane", "gali", "res-", "colony", "residential", "internal"]):
            return "LOCAL"

        # Spatial heuristics for Mumbai metropolitan region if lat/lon available
        if latitude and longitude:
            # Western Express Highway spine corridor (~19.04 to 19.25, ~72.84 to 72.87)
            if 19.04 <= latitude <= 19.25 and 72.84 <= longitude <= 72.87:
                return "HIGH_TRAFFIC_CORRIDOR"

        return "ARTERIAL"

    def simulate(
        self,
        road_class: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        road_segment_id: Optional[str] = None,
        road_name: Optional[str] = None,
        peak_hour: bool = True,
        pcu_override: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Simulate traffic volume and operating speed correlated with road context.

        Args:
            road_class: 'LOCAL', 'COLLECTOR', 'ARTERIAL', 'HIGH_TRAFFIC_CORRIDOR'
            latitude: Geographic latitude
            longitude: Geographic longitude
            road_segment_id: Segment identifier
            road_name: Municipal roadway name
            peak_hour: True for peak congestion profile
            pcu_override: Optional explicit PCU value
        """
        # Determine road class with location correlation if not specified
        if road_class is None:
            resolved_class = self.infer_road_class_from_location(
                latitude=latitude,
                longitude=longitude,
                road_segment_id=road_segment_id,
                road_name=road_name
            )
        else:
            resolved_class = self.resolve_road_class(road_class)

        traffic_cfg = self.config.get("traffic", DEFAULT_TRAFFIC_CONFIG["traffic"])
        road_classes_cfg = traffic_cfg.get("road_classes", DEFAULT_TRAFFIC_CONFIG["traffic"]["road_classes"])
        profile = road_classes_cfg.get(resolved_class, road_classes_cfg.get("HIGH_TRAFFIC_CORRIDOR"))

        capacity = profile.get("capacity_pcu", 38000)
        pcu_range = profile.get("pcu_range", {"min_pcu": 24000, "max_pcu": 38000})

        if isinstance(pcu_range, dict):
            min_pcu = pcu_range.get("min_pcu", 24000)
            max_pcu = pcu_range.get("max_pcu", 38000)
        elif isinstance(pcu_range, (list, tuple)):
            min_pcu, max_pcu = pcu_range[0], pcu_range[1]
        else:
            min_pcu, max_pcu = 24000, 38000

        if pcu_override is not None:
            traffic_pcu = int(pcu_override)
        else:
            if peak_hour:
                traffic_pcu = int(self.rng.randint(int(min_pcu * 1.05), max_pcu))
            else:
                traffic_pcu = int(self.rng.randint(min_pcu, int(max_pcu * 0.85)))

        v_over_c = round(traffic_pcu / capacity, 2)

        # Level of service determination (IRC / HCM standards)
        if v_over_c < 0.40:
            los = "A"
        elif v_over_c < 0.60:
            los = "B"
        elif v_over_c < 0.75:
            los = "C"
        elif v_over_c < 0.90:
            los = "D"
        elif v_over_c < 1.00:
            los = "E"
        else:
            los = "F"

        # Macroscopic speed-density Greenshields relation
        v_free = profile.get("free_flow_speed_kmh", 60.0)
        v_cong = profile.get("congestion_speed_kmh", 25.0)
        speed_kmh = round(v_free - (v_free - v_cong) * min(1.0, v_over_c) + self.rng.uniform(-2.0, 2.0), 1)
        speed_kmh = max(12.0, min(85.0, speed_kmh))

        comm_range = profile.get("commercial_vehicle_percentage_range", [10.0, 20.0])
        comm_pct = round(self.rng.uniform(comm_range[0], comm_range[1]), 1)

        return {
            "road_class": resolved_class,
            "road_class_name": profile.get("name", resolved_class),
            "traffic_pcu": traffic_pcu,
            "pcu": traffic_pcu,
            "traffic_source": "synthetic",
            "roadway_capacity_pcu": capacity,
            "volume_capacity_ratio": v_over_c,
            "level_of_service": los,
            "operating_speed_kmh": speed_kmh,
            "commercial_vehicle_percentage": comm_pct,
            "sensor_source": "synthetic",
            "generation_method": "controlled_distribution",
            "source": "synthetic"
        }

