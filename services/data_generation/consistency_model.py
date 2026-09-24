"""
Physical Consistency Model & Latent Severity Engine
===================================================

Enforces physical laws and cross-modal correlations across the Smart Road Multimodal Dataset:

Core Conceptual Flow:
    visual defect
         ↓
  latent severity (defect_severity ∈ [0.0, 1.0])
         ↓
  ┌──────┼──────────────┬──────────────┬────────────────┐
depth   area          impact         sensors        traffic & context
  ↓      ↓              ↓              ↓                ↓
LiDAR  volume     accelerometer      Sonar        TOPSIS / Infrastructure Risk

Physical Laws & Invariants:
1. Monotonicity: Deeper pothole → larger volume & larger dynamic impact shock.
2. Volume Integrity: volume > 0 ⟺ (depth > 0 AND area > 0).
3. Dynamic Impact Confirmation: dynamic_impact_triggered ⟺ z_accel_g > 1.50g.
4. Modality Selection:
   - rain_detected OR luminance < 40 lux → acoustic sonar selected / preferred.
   - dry daylight (luminance >= 40 lux, no rain) → optical LiDAR selected / preferred.
5. Infrastructure Risk:
   Higher traffic + higher speed + larger defect severity + hospital proximity → higher infrastructure risk.
"""

import math
import random
from typing import Dict, List, Any, Optional, Tuple


# Baseline chassis clearance in centimeters
DEFAULT_CHASSIS_BASELINE_CM = 30.0

# Canonical class severity prior distributions [min_severity, max_severity]
CLASS_SEVERITY_PRIORS = {
    "D00": {"min_s": 0.05, "max_s": 0.28, "name": "longitudinal_crack", "label": "Longitudinal Crack"},
    "D10": {"min_s": 0.08, "max_s": 0.32, "name": "transverse_crack", "label": "Transverse Crack"},
    "D20": {"min_s": 0.28, "max_s": 0.62, "name": "alligator_crack", "label": "Alligator Crack"},
    "D40": {"min_s": 0.52, "max_s": 0.98, "name": "pothole", "label": "Pothole"}
}


class PhysicalConsistencyModel:
    """
    Unified physical consistency engine that resolves latent defect severity
    and derives correlated physical, sensor, dynamic, and municipal risk properties.
    """
    def __init__(self, seed: int = 2026, baseline_chassis_cm: float = DEFAULT_CHASSIS_BASELINE_CM):
        self.seed = seed
        self.baseline_chassis_cm = baseline_chassis_cm
        self.rng = random.Random(seed)

    def reseed(self, seed: int):
        self.seed = seed
        self.rng = random.Random(seed)

    def compute_perspective_surface_area(
        self,
        bbox: List[float],
        image_dims: Dict[str, int]
    ) -> float:
        """
        Calculates ground-plane surface area in square meters using bounding box
        dimensions and camera perspective geometry.
        """
        xmin, ymin, xmax, ymax = bbox
        width = max(1, image_dims.get("width", 720))
        height = max(1, image_dims.get("height", 720))

        box_w = max(1.0, xmax - xmin)
        box_h = max(1.0, ymax - ymin)
        norm_area = (box_w * box_h) / (width * height)

        # Objects located lower in the frame (higher ymin) are closer to the front bumper
        perspective_weight = 1.0 + (ymin / height) * 0.75
        area_sqm = round(norm_area * 14.0 * perspective_weight, 3)
        return max(0.05, min(12.0, area_sqm))

    def derive_latent_severity(
        self,
        damage_class: str,
        surface_area_sqm: float
    ) -> float:
        """
        Derives the latent scalar variable `defect_severity` ∈ [0.01, 1.00].
        
        The latent variable couples visual bounding box size and damage taxonomy.
        """
        prior = CLASS_SEVERITY_PRIORS.get(damage_class, CLASS_SEVERITY_PRIORS["D40"])
        s_min = prior["min_s"]
        s_max = prior["max_s"]

        # Scale severity monotonically with observed geometric area
        if damage_class == "D40":
            # Pothole area normalization: 0.1 m² (small) to 2.5 m² (huge)
            area_factor = min(1.0, max(0.0, (surface_area_sqm - 0.1) / 2.4))
        elif damage_class == "D20":
            # Alligator area normalization: 0.2 m² to 4.0 m²
            area_factor = min(1.0, max(0.0, (surface_area_sqm - 0.2) / 3.8))
        else:
            # Linear cracks: 0.05 m² to 1.5 m²
            area_factor = min(1.0, max(0.0, (surface_area_sqm - 0.05) / 1.45))

        # Base severity interpolated between min and max prior
        base_s = s_min + (s_max - s_min) * (area_factor ** 0.8)

        # Micro-jitter with controlled standard deviation (±0.02)
        jitter = self.rng.uniform(-0.02, 0.02)
        severity = round(max(0.01, min(1.00, base_s + jitter)), 4)
        return severity

    def derive_depth_and_volume(
        self,
        defect_severity: float,
        damage_class: str,
        surface_area_sqm: float
    ) -> Tuple[float, float, str]:
        """
        Derives physical defect depth (cm) and cavity volume (liters).
        
        Strict physical invariant:
        volume > 0 ⟺ (depth > 0 AND surface_area_sqm > 0)
        """
        s = defect_severity

        if damage_class == "D40":
            # Potholes: 4.8 cm to 18.5 cm
            # Moderate potholes (< 1.0 m² area) produce moderate void (4.8 - 6.2 cm, borderline shock)
            # Large severe potholes (>= 1.0 m² area) produce deep void (10.0 - 16.0 cm, confirmed impact)
            if surface_area_sqm < 1.0:
                depth_cm = round(4.8 + 1.8 * (s ** 1.1), 2)
            else:
                depth_cm = round(9.0 + 8.5 * (s ** 1.3), 2)
        elif damage_class == "D20":
            # Alligator fatigue: 2.0 cm to 5.5 cm
            depth_cm = round(1.8 + 4.2 * s, 2)
        elif damage_class == "D10":
            # Transverse crack: 0.6 cm to 2.2 cm
            depth_cm = round(0.5 + 2.0 * (s ** 1.2), 2)
        else:
            # Longitudinal crack (D00): 0.5 cm to 2.0 cm
            depth_cm = round(0.5 + 1.8 * (s ** 1.2), 2)

        depth_cm = max(0.2, min(25.0, depth_cm))

        # Severity tier categorization
        if depth_cm >= 9.0:
            tier = "deep"
        elif depth_cm >= 4.5:
            tier = "moderate"
        else:
            tier = "shallow"

        # Volume calculation (parabolic / conical cavity approximation):
        # V = shape_factor * area_cm² * depth_cm / 1000 cm³/L
        # With area in m²: area_cm² = surface_area_sqm * 10,000
        # shape_factor = 0.50 (consistent with depth_estimator.py EnvironmentalSensorSwitching)
        shape_factor = 0.50
        if surface_area_sqm > 0 and depth_cm > 0:
            volume_liters = round(shape_factor * (surface_area_sqm * 10000.0) * depth_cm / 1000.0, 2)
        else:
            volume_liters = 0.0

        return depth_cm, volume_liters, tier

    def derive_accelerometer_impact(
        self,
        defect_severity: float,
        defect_depth_cm: float,
        surface_area_sqm: float,
        vehicle_speed_kmh: float,
        damage_class: str
    ) -> Dict[str, Any]:
        """
        Derives vertical acceleration z_accel_g from quarter-car shock mechanics.
        
        Strict physical invariant:
        dynamic_impact_triggered ⟺ z_accel_g > 1.50g
        """
        speed = max(10.0, min(80.0, vehicle_speed_kmh))
        alpha = {
            "D40": 0.88,
            "D20": 0.38,
            "D10": 0.18,
            "D00": 0.14
        }.get(damage_class, 0.40)

        # Baseline road roughness vibration
        baseline_g = 1.00
        speed_factor = (speed / 45.0) ** 0.65
        depth_factor = (defect_depth_cm / 10.0) ** 1.15
        area_factor = (surface_area_sqm / 1.5) ** 0.25

        dynamic_shock = alpha * speed_factor * depth_factor * area_factor
        noise = self.rng.uniform(-0.04, 0.04)
        z_accel_g = round(max(0.70, baseline_g + dynamic_shock + noise), 2)

        threshold_g = 1.50
        borderline_min_g = 1.35
        dynamic_impact_triggered = bool(z_accel_g > threshold_g)

        if z_accel_g > threshold_g:
            regime = "confirmed_impact"
            category = "Confirmed Dynamic Impact"
            assessment = "Visual defect confirmed by acute dynamic impact (> 1.5g). Pavement cavity produces severe suspension shock, providing incontrovertible multi-modal evidence of hazard."
        elif z_accel_g > borderline_min_g:
            regime = "borderline"
            category = "Borderline Impact"
            assessment = "Visual defect accompanied by moderate vertical chassis disturbance (1.35g - 1.50g). Physical wheel engagement detected, warranting high monitoring priority."
        else:
            regime = "no_impact"
            category = "No Significant Impact"
            assessment = "Superficial visual distress with nominal vehicle chassis response (<= 1.35g). Visual distress alone does not confirm an immediate vehicular hazard."

        rms_vibration_g = round(0.12 + 0.09 * (z_accel_g - 1.0), 3)
        suspension_deflection_cm = round(max(0.1, defect_depth_cm * 0.35 * (speed / 45.0) ** 0.5), 2)

        return {
            "z_accel_g": z_accel_g,
            "dynamic_impact_triggered": dynamic_impact_triggered,
            "impact_regime": regime,
            "impact_category": category,
            "cross_modal_assessment": assessment,
            "threshold_g": threshold_g,
            "borderline_min_g": borderline_min_g,
            "rms_vibration_g": rms_vibration_g,
            "suspension_deflection_cm": suspension_deflection_cm,
            "vehicle_speed_kmh": round(speed, 1),
            "sensor_source": "synthetic",
            "generation_method": "controlled_distribution"
        }

    def derive_sensor_measurements(
        self,
        defect_depth_cm: float,
        rain_detected: bool,
        luminance_lux: float
    ) -> Dict[str, Any]:
        """
        Derives LiDAR and Sonar measurements along with the active modality.
        
        Strict physical invariant:
        - rain_detected OR luminance_lux < 40.0 → acoustic sonar selected / preferred
        - dry daylight → optical LiDAR selected / preferred
        """
        # Sensor noise models
        lidar_noise = round(self.rng.uniform(-0.15, 0.15), 2)
        sonar_noise = round(self.rng.uniform(-0.30, 0.30), 2)

        # Raw distance from chassis down to road defect base
        raw_lidar_cm = round(self.baseline_chassis_cm + defect_depth_cm + lidar_noise, 2)
        raw_sonar_cm = round(self.baseline_chassis_cm + defect_depth_cm + sonar_noise, 2)

        # Environmental modality selection
        use_sonar = bool(rain_detected or (luminance_lux < 40.0))
        if use_sonar:
            preferred_modality = "sonar_submerged_acoustic"
            modality_reason = "Rain/submerged water or low luminance (< 40 lux) triggers acoustic sonar"
            selected_depth_cm = round(max(0.0, raw_sonar_cm - self.baseline_chassis_cm), 2)
        else:
            preferred_modality = "optical_lidar"
            modality_reason = "Dry daylight provides optimal optical LiDAR reflectivity"
            selected_depth_cm = round(max(0.0, raw_lidar_cm - self.baseline_chassis_cm), 2)

        return {
            "raw_lidar_cm": raw_lidar_cm,
            "lidar_depth_cm": raw_lidar_cm,
            "lidar_effective_depth_cm": defect_depth_cm,
            "raw_sonar_cm": raw_sonar_cm,
            "sonar_depth_cm": raw_sonar_cm,
            "sonar_effective_depth_cm": round(max(0.1, defect_depth_cm + sonar_noise), 2),
            "sonar_error_cm": sonar_noise,
            "preferred_modality": preferred_modality,
            "modality_reason": modality_reason,
            "selected_depth_cm": selected_depth_cm,
            "is_submerged": bool(rain_detected)
        }

    def derive_infrastructure_risk(
        self,
        defect_severity: float,
        defect_depth_cm: float,
        surface_area_sqm: float,
        traffic_pcu: int,
        vehicle_speed_kmh: float,
        dist_hospital_km: float
    ) -> Dict[str, Any]:
        """
        Synthesizes composite infrastructure risk index coupled with traffic,
        speed, hospital proximity, and defect geometry.
        
        Strict physical invariant:
        Higher traffic + higher speed + larger defect → higher infrastructure risk.
        """
        # Normalized dimensions [0.0, 1.0]
        norm_depth = min(1.0, defect_depth_cm / 18.0)
        norm_area = min(1.0, surface_area_sqm / 6.0)
        norm_traffic = min(1.0, traffic_pcu / 38000.0)
        norm_speed = min(1.0, vehicle_speed_kmh / 70.0)
        # Hospital proximity factor (closer to hospital = higher emergency vulnerability)
        hosp_proximity = min(1.0, 1.0 / max(0.3, dist_hospital_km))

        # Composite multi-criteria risk formula aligned with TOPSIS
        # Weights: depth (0.32), area (0.18), traffic (0.24), speed (0.12), hospital (0.14)
        risk_score = (
            0.32 * norm_depth +
            0.18 * norm_area +
            0.24 * norm_traffic +
            0.12 * norm_speed +
            0.14 * hosp_proximity
        )
        risk_score = round(max(0.05, min(0.99, risk_score)), 4)

        if risk_score >= 0.75:
            priority = "Critical"
        elif risk_score >= 0.50:
            priority = "High"
        elif risk_score >= 0.25:
            priority = "Medium"
        else:
            priority = "Low"

        return {
            "infrastructure_risk_score": risk_score,
            "priority_level": priority,
            "norm_depth": round(norm_depth, 3),
            "norm_area": round(norm_area, 3),
            "norm_traffic": round(norm_traffic, 3),
            "hosp_proximity": round(hosp_proximity, 3)
        }

    def derive_consistent_telemetry(
        self,
        damage_class: str,
        surface_area_sqm: float,
        rain_detected: bool = False,
        luminance_lux: float = 65.0,
        vehicle_speed_kmh: float = 40.0,
        road_class: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Derives an integrated package of physically consistent telemetry:
        latent severity, depth, volume, impact acceleration, sensors, and context.
        """
        # 1. Latent Defect Severity
        defect_severity = self.derive_latent_severity(damage_class, surface_area_sqm)

        # 2. Physics-grounded Depth and Volume
        defect_depth_cm, volume_liters, severity_tier = self.derive_depth_and_volume(
            defect_severity=defect_severity,
            damage_class=damage_class,
            surface_area_sqm=surface_area_sqm
        )

        # 3. Dynamic Impact Acceleration
        impact_data = self.derive_accelerometer_impact(
            defect_severity=defect_severity,
            defect_depth_cm=defect_depth_cm,
            surface_area_sqm=surface_area_sqm,
            vehicle_speed_kmh=vehicle_speed_kmh,
            damage_class=damage_class
        )

        # 4. Multimodal Depth Sensors
        sensors = self.derive_sensor_measurements(
            defect_depth_cm=defect_depth_cm,
            rain_detected=rain_detected,
            luminance_lux=luminance_lux
        )

        # 5. Infrastructure and Road Context
        rc = road_class if road_class else "ARTERIAL"
        traffic_ranges = {
            "LOCAL": (1500, 5000),
            "COLLECTOR": (5000, 15000),
            "ARTERIAL": (15000, 28000),
            "HIGH_TRAFFIC_CORRIDOR": (28000, 42000)
        }
        t_min, t_max = traffic_ranges.get(rc, (12000, 25000))
        pcu = self.rng.randint(t_min, t_max)
        hosp_dist = round(self.rng.uniform(0.4, 7.5), 2)

        return {
            "defect_severity": defect_severity,
            "defect_depth_cm": defect_depth_cm,
            "volume_liters": volume_liters,
            "severity_tier": severity_tier,
            "accelerometer": impact_data,
            "sensors": sensors,
            "traffic": {
                "pcu": pcu,
                "road_class": rc
            },
            "infrastructure": {
                "dist_hospital_km": hosp_dist,
                "nearest_hospital": "General Hospital"
            },
            "location": {
                "latitude": round(19.0760 + self.rng.uniform(-0.02, 0.02), 5),
                "longitude": round(72.8777 + self.rng.uniform(-0.02, 0.02), 5),
                "road_segment_id": f"SEG-{rc[:3]}-{self.rng.randint(10, 99)}"
            }
        }
