"""
Critical Municipal Infrastructure & Hospital Proximity Model
============================================================

Models spatial proximity to nearest emergency trauma center and critical municipal
infrastructure along Indian urban corridors using a pluggable GIS layer.

Architecture:
    road defect coordinate
    ↓
    nearest hospital (spatial search)
    ↓
    geodesic distance (Haversine WGS84)
    ↓
    dist_hospital_km

Documented Demonstration Context:
- Uses a documented synthetic geographic coordinate system anchored in the
  Mumbai Metropolitan Urban Corridor (Western Express & Coastal Corridor, Maharashtra, India).
- Coordinates and hospital associations are strictly marked:
    location_source = "synthetic"
    hospital_source = "synthetic"
  RDD2022 visual evidence does not supply these coordinates.
- Pluggable design: The GIS provider can be replaced with an OpenStreetMap Overpass
  provider or live municipal GeoJSON feed without altering the TOPSIS interface.
"""

import os
import sys
import math
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

try:
    import yaml
except ImportError:
    yaml = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "sensor_profiles.yaml"

DEFAULT_HOSPITALS = [
    {
        "name": "Lilavati Hospital & Research Centre",
        "lat": 19.0510, "lon": 72.8290,
        "tier": "Level-1 Super Speciality & Emergency Trauma Center",
        "district": "Bandra West"
    },
    {
        "name": "Nanavati Max Super Speciality Hospital",
        "lat": 19.0880, "lon": 72.8420,
        "tier": "Level-1 Critical Care & Trauma Unit",
        "district": "Santacruz West"
    },
    {
        "name": "Dr. R.N. Cooper Municipal General Hospital",
        "lat": 19.1080, "lon": 72.8360,
        "tier": "Level-1 Municipal Trauma General Hospital",
        "district": "Juhu / Vile Parle"
    },
    {
        "name": "Bal Thackeray Trauma Care Municipal Hospital",
        "lat": 19.1380, "lon": 72.8590,
        "tier": "Apex Municipal Emergency Trauma Center",
        "district": "Jogeshwari East"
    },
    {
        "name": "Kokilaben Dhirubhai Ambani Hospital",
        "lat": 19.1310, "lon": 72.8250,
        "tier": "Level-1 Tertiary Care & Multi-organ Trauma Center",
        "district": "Andheri West"
    },
    {
        "name": "SevenHills Hospital",
        "lat": 19.1220, "lon": 72.8830,
        "tier": "Level-1 Emergency & Critical Care Hospital",
        "district": "Andheri East"
    },
    {
        "name": "Bhabha Municipal General Hospital",
        "lat": 19.0550, "lon": 72.8300,
        "tier": "Level-2 Municipal Secondary Trauma Center",
        "district": "Bandra West"
    },
    {
        "name": "Holy Family Multi-Speciality Hospital",
        "lat": 19.0580, "lon": 72.8270,
        "tier": "Level-2 Acute Medical Emergency Hospital",
        "district": "Bandra West"
    }
]


DEFAULT_SCHOOLS = [
    {
        "name": "St. Stanislaus High School",
        "lat": 19.0548, "lon": 72.8286,
        "category": "Primary & Secondary School",
        "district": "Bandra West"
    },
    {
        "name": "Apostolic Carmel High School",
        "lat": 19.0562, "lon": 72.8295,
        "category": "Primary & Secondary School",
        "district": "Bandra West"
    },
    {
        "name": "St. Andrew's High School",
        "lat": 19.0583, "lon": 72.8280,
        "category": "Secondary School",
        "district": "Bandra West"
    },
    {
        "name": "Arya Vidya Mandir",
        "lat": 19.0645, "lon": 72.8322,
        "category": "ICSE K-10 School",
        "district": "Bandra West"
    },
    {
        "name": "Dhirubhai Ambani International School",
        "lat": 19.0668, "lon": 72.8712,
        "category": "K-12 International School",
        "district": "Bandra Kurla Complex"
    },
    {
        "name": "Maneckji Cooper Education Trust School",
        "lat": 19.0865, "lon": 72.8285,
        "category": "ICSE K-10 School",
        "district": "Juhu / Santacruz West"
    },
    {
        "name": "Podar International School",
        "lat": 19.0825, "lon": 72.8388,
        "category": "K-12 Educational Complex",
        "district": "Santacruz West"
    },
    {
        "name": "Jamnabai Narsee School",
        "lat": 19.1074, "lon": 72.8368,
        "category": "K-12 School",
        "district": "JVPD Scheme, Vile Parle West"
    },
    {
        "name": "Utpal Shanghvi Global School",
        "lat": 19.1105, "lon": 72.8342,
        "category": "K-12 School",
        "district": "Juhu / Vile Parle West"
    },
    {
        "name": "Parle Tilak Vidyalaya",
        "lat": 19.0992, "lon": 72.8515,
        "category": "Primary & Secondary School",
        "district": "Vile Parle East"
    },
    {
        "name": "Bhavan's A.H. Wadia High School",
        "lat": 19.1268, "lon": 72.8362,
        "category": "Secondary School & Campus",
        "district": "Andheri West"
    },
    {
        "name": "St. Blaise High School",
        "lat": 19.1245, "lon": 72.8458,
        "category": "Primary & Secondary School",
        "district": "Amboli, Andheri West"
    },
    {
        "name": "Holy Family High School",
        "lat": 19.1142, "lon": 72.8618,
        "category": "Primary & Secondary School",
        "district": "Chakala, Andheri East"
    },
    {
        "name": "Ismail Yusuf College & High School Campus",
        "lat": 19.1392, "lon": 72.8565,
        "category": "Higher Secondary & Junior College",
        "district": "Jogeshwari East"
    },
    {
        "name": "St. Xavier's High School",
        "lat": 19.1645, "lon": 72.8450,
        "category": "Primary & Secondary School",
        "district": "Goregaon West"
    },
    {
        "name": "Vibgyor High School",
        "lat": 19.1610, "lon": 72.8385,
        "category": "K-12 School",
        "district": "Goregaon West"
    },
    {
        "name": "Ryan International School",
        "lat": 19.1865, "lon": 72.8360,
        "category": "K-12 School",
        "district": "Malad West"
    },
    {
        "name": "St. Francis D'Assisi High School",
        "lat": 19.2360, "lon": 72.8520,
        "category": "Primary & Secondary School",
        "district": "Borivali West"
    }
]

ROAD_CLASS_ORDINAL_WEIGHTS: Dict[str, int] = {
    "LOCAL": 1,
    "COLLECTOR": 2,
    "ARTERIAL": 3,
    "HIGH_TRAFFIC_CORRIDOR": 4
}


def encode_road_class(road_class: Any) -> int:
    """
    Maps a categorical road class string (or numeric weight) to its ordinal MCDM weight:
        LOCAL = 1, COLLECTOR = 2, ARTERIAL = 3, HIGH_TRAFFIC_CORRIDOR = 4.
    """
    if isinstance(road_class, (int, float)):
        return max(1, min(4, int(round(float(road_class)))))
    if isinstance(road_class, str):
        key = road_class.strip().upper()
        if key in ROAD_CLASS_ORDINAL_WEIGHTS:
            return ROAD_CLASS_ORDINAL_WEIGHTS[key]
    return 3  # Default to ARTERIAL (3) if unspecified


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates great-circle geodesic distance between two GPS coordinates in kilometers."""
    R = 6371.0  # Earth's mean radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return round(R * c, 3)


class HospitalGISProvider:
    """Base interface for hospital and school registry data sources (synthetic or OpenStreetMap)."""
    def get_hospitals(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def get_schools(self) -> List[Dict[str, Any]]:
        return DEFAULT_SCHOOLS

    def find_nearest(self, lat: float, lon: float) -> Tuple[Dict[str, Any], float]:
        """Finds nearest hospital to coordinates and returns (hospital, geodesic_dist_km)."""
        hospitals = self.get_hospitals()
        if not hospitals:
            raise ValueError("No hospitals available in GIS provider registry.")
        best_hospital = min(
            hospitals,
            key=lambda h: haversine_distance_km(lat, lon, h["lat"], h["lon"])
        )
        geodesic_dist = haversine_distance_km(lat, lon, best_hospital["lat"], best_hospital["lon"])
        return best_hospital, geodesic_dist

    def find_nearest_school(self, lat: float, lon: float) -> Tuple[Dict[str, Any], float]:
        """Finds nearest school to coordinates and returns (school, geodesic_dist_km)."""
        schools = self.get_schools()
        if not schools:
            raise ValueError("No schools available in GIS provider registry.")
        best_school = min(
            schools,
            key=lambda s: haversine_distance_km(lat, lon, s["lat"], s["lon"])
        )
        geodesic_dist = haversine_distance_km(lat, lon, best_school["lat"], best_school["lon"])
        return best_school, geodesic_dist


class SyntheticGISLayerProvider(HospitalGISProvider):
    """
    In-memory GIS provider populated from YAML configuration representing
    geocoded emergency trauma centers and schools in the Mumbai demonstration corridor.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config and "infrastructure" in config:
            infra_cfg = config["infrastructure"]
            self.hospitals = infra_cfg.get("hospitals", DEFAULT_HOSPITALS)
            self.schools = infra_cfg.get("schools", DEFAULT_SCHOOLS)
            self.area_name = infra_cfg.get("demonstration_area", {}).get("name", "Mumbai Urban Corridor")
            self.tortuosity = infra_cfg.get("network_tortuosity_factor", 1.18)
        else:
            self.hospitals = DEFAULT_HOSPITALS
            self.schools = DEFAULT_SCHOOLS
            self.area_name = "Mumbai Urban Corridor"
            self.tortuosity = 1.18

    def get_hospitals(self) -> List[Dict[str, Any]]:
        return self.hospitals

    def get_schools(self) -> List[Dict[str, Any]]:
        return self.schools


class InfrastructureModel:
    """
    Location-correlated Infrastructure Proximity Model.
    Computes geodesic distance to nearest trauma center and nearest school from road defect coordinates.
    """
    def __init__(
        self,
        config_path: Optional[Path] = None,
        config_override: Optional[Dict[str, Any]] = None,
        gis_provider: Optional[HospitalGISProvider] = None,
        seed: int = 42
    ):
        self.rng = random.Random(seed)
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self.config = self._load_configuration(config_override)

        infra_cfg = self.config.get("infrastructure", {})
        self.emergency_threshold_km = infra_cfg.get("emergency_corridor_threshold_km", 1.0)
        self.school_zone_threshold_km = infra_cfg.get("school_zone_threshold_km", 0.5)
        self.tortuosity_factor = infra_cfg.get("network_tortuosity_factor", 1.18)

        # Allow injection of OpenStreetMap or alternative GIS provider
        if gis_provider is not None:
            self.gis_provider = gis_provider
        else:
            self.gis_provider = SyntheticGISLayerProvider(self.config)

    def _load_configuration(self, override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Loads infrastructure configuration from YAML with fallback."""
        if override:
            return override

        if yaml and self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f)
                if loaded and "infrastructure" in loaded:
                    return loaded
            except Exception as e:
                print(f"Warning: Failed to load infrastructure config from {self.config_path}: {e}")

        return {
            "infrastructure": {
                "hospitals": DEFAULT_HOSPITALS,
                "schools": DEFAULT_SCHOOLS,
                "emergency_corridor_threshold_km": 1.0,
                "school_zone_threshold_km": 0.5
            }
        }

    def reseed(self, seed: int):
        self.rng = random.Random(seed)

    def find_nearest_hospital(self, latitude: float, longitude: float) -> Tuple[Dict[str, Any], float]:
        """Queries the GIS layer for the nearest hospital and geodesic distance."""
        return self.gis_provider.find_nearest(latitude, longitude)

    def find_nearest_school(self, latitude: float, longitude: float) -> Tuple[Dict[str, Any], float]:
        """Queries the GIS layer for the nearest school and geodesic distance."""
        return self.gis_provider.find_nearest_school(latitude, longitude)

    def simulate(
        self,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        prescribed_dist_km: Optional[float] = None,
        prescribed_school_dist_km: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Computes geodesic distance to nearest emergency medical facility and school:
            road defect coordinate -> nearest hospital -> geodesic distance -> dist_hospital_km
            road defect coordinate -> nearest school   -> geodesic distance -> school_proximity_km

        Args:
            latitude: Road defect latitude
            longitude: Road defect longitude
            prescribed_dist_km: Explicit hospital distance override for testing
            prescribed_school_dist_km: Explicit school distance override for testing
        """
        if prescribed_dist_km is not None:
            effective_dist_km = round(max(0.05, float(prescribed_dist_km)), 2)
            geodesic_km = round(effective_dist_km / self.tortuosity_factor, 3)
            hospitals = self.gis_provider.get_hospitals()
            hospital = self.rng.choice(hospitals)
            ref_lat = latitude if latitude is not None else hospital["lat"]
            ref_lon = longitude if longitude is not None else hospital["lon"]
        elif latitude is not None and longitude is not None:
            # Full geodesic spatial pipeline
            hospital, geodesic_km = self.find_nearest_hospital(latitude, longitude)
            # Route tortuosity accounts for street network turns
            effective_dist_km = round(max(0.10, geodesic_km * self.tortuosity_factor), 2)
            ref_lat, ref_lon = latitude, longitude
        else:
            # Fallback for uncoordinated tests
            hospitals = self.gis_provider.get_hospitals()
            hospital = self.rng.choice(hospitals)
            effective_dist_km = round(self.rng.uniform(0.35, 3.5), 2)
            geodesic_km = round(effective_dist_km / self.tortuosity_factor, 3)
            ref_lat, ref_lon = hospital["lat"], hospital["lon"]

        # Nearest school proximity via Haversine
        school, school_geodesic_km = self.find_nearest_school(ref_lat, ref_lon)
        if prescribed_school_dist_km is not None:
            school_effective_km = round(max(0.05, float(prescribed_school_dist_km)), 2)
            school_geodesic_km = round(school_effective_km / self.tortuosity_factor, 3)
        else:
            school_effective_km = round(max(0.05, school_geodesic_km * self.tortuosity_factor), 2)

        is_emergency = bool(effective_dist_km <= self.emergency_threshold_km)
        is_school_zone = bool(school_effective_km <= self.school_zone_threshold_km)

        return {
            "dist_hospital_km": effective_dist_km,
            "hospital_distance_km": effective_dist_km,
            "geodesic_distance_km": geodesic_km,
            "nearest_hospital_name": hospital["name"],
            "nearest_hospital_lat": hospital["lat"],
            "nearest_hospital_lon": hospital["lon"],
            "hospital_tier": hospital.get("tier", "Emergency Trauma Center"),
            "district": hospital.get("district", "Mumbai Metropolitan Area"),
            "is_emergency_corridor": is_emergency,
            "emergency_threshold_km": self.emergency_threshold_km,
            "school_proximity_km": school_effective_km,
            "dist_school_km": school_effective_km,
            "school_geodesic_distance_km": school_geodesic_km,
            "nearest_school_name": school["name"],
            "nearest_school_lat": school["lat"],
            "nearest_school_lon": school["lon"],
            "school_category": school.get("category", "School"),
            "is_school_zone": is_school_zone,
            "school_zone_threshold_km": self.school_zone_threshold_km,
            "location_source": "synthetic",
            "hospital_source": "synthetic",
            "school_source": "synthetic",
            "generation_method": "gis_geodesic_nearest_neighbor",
            "source": "synthetic"
        }

