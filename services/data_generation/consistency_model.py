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

import logging
import math
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

try:
    import yaml
except ImportError:
    yaml = None

logger = logging.getLogger(__name__)

# Baseline chassis clearance in centimeters
DEFAULT_CHASSIS_BASELINE_CM = 30.0

# Scaling factor mapping bounded crack_severity_index ∈ [0, 1] to equivalent
# structural repair magnitude (L-equivalent) for unified MCDM comparison with potholes
CRACK_SEVERITY_TO_MAGNITUDE_SCALE = 15.0

DEFAULT_CAMERA_CONFIG: Dict[str, Any] = {
    "model": "pinhole_ground_plane",
    "camera_height_m": 1.35,
    "pitch_deg": 22.0,
    "hfov_deg": 62.0,
    "vfov_deg": 48.0,
    "focal_length_px": 650.0,
    "reference_image_width_px": 720,
    "reference_image_height_px": 720,
    "min_depression_deg": 5.0,
    "class_area_bounds_sqm": {
        "D40": [0.01, 1.00],
        "D20": [0.05, 4.00],
        "D10": [0.01, 1.50],
        "D00": [0.01, 1.50],
    },
}


def load_camera_profile_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Loads pinhole camera ground-plane calibration parameters from config/sensor_profiles.yaml."""
    cfg = {
        **DEFAULT_CAMERA_CONFIG,
        "class_area_bounds_sqm": dict(DEFAULT_CAMERA_CONFIG["class_area_bounds_sqm"]),
    }
    target = config_path or (Path(__file__).resolve().parents[2] / "config" / "sensor_profiles.yaml")
    if yaml is not None and target.exists():
        try:
            with open(target, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            cam = raw.get("camera", {})
            if isinstance(cam, dict):
                for k in (
                    "model",
                    "camera_height_m",
                    "pitch_deg",
                    "hfov_deg",
                    "vfov_deg",
                    "focal_length_px",
                    "reference_image_width_px",
                    "reference_image_height_px",
                    "min_depression_deg",
                ):
                    if k in cam and cam[k] is not None:
                        cfg[k] = cam[k]
                bounds = cam.get("class_area_bounds_sqm")
                if isinstance(bounds, dict):
                    for cls_k, pair in bounds.items():
                        if isinstance(pair, (list, tuple)) and len(pair) == 2:
                            cfg["class_area_bounds_sqm"][str(cls_k).upper()] = [float(pair[0]), float(pair[1])]
        except Exception as exc:
            logger.debug("Fallback to default camera profile config (%s)", exc)
    return cfg


# Canonical class severity prior distributions [min_severity, max_severity]
CLASS_SEVERITY_PRIORS = {
    "D00": {"min_s": 0.05, "max_s": 0.28, "name": "longitudinal_crack", "label": "Longitudinal Crack"},
    "D10": {"min_s": 0.08, "max_s": 0.32, "name": "transverse_crack", "label": "Transverse Crack"},
    "D20": {"min_s": 0.28, "max_s": 0.62, "name": "alligator_crack", "label": "Alligator Crack"},
    "D40": {"min_s": 0.52, "max_s": 0.98, "name": "pothole", "label": "Pothole"}
}

_CLASS_NAME_TO_CODE = {
    "D00": "D00", "LONGITUDINAL_CRACK": "D00",
    "D10": "D10", "TRANSVERSE_CRACK": "D10",
    "D20": "D20", "ALLIGATOR_CRACK": "D20",
    "D40": "D40", "POTHOLE": "D40",
}


def normalize_damage_class_code(damage_class: Optional[str]) -> str:
    """Normalizes a damage class code or slug into canonical RDD2022 code (D00, D10, D20, D40)."""
    if not damage_class:
        return "D40"
    key = str(damage_class).strip().upper()
    return _CLASS_NAME_TO_CODE.get(key, "D40")


def estimate_ground_plane_area_pinhole(
    bbox: List[float],
    image_dims: Dict[str, int],
    damage_class: Optional[str] = "D40",
    camera_cfg: Optional[Dict[str, Any]] = None
) -> float:
    """
    Estimates real-world ground-plane surface area (m^2) using a pinhole camera model:
    - Converts bbox bottom-edge vertical coordinate (ymax) into depression angle below horizon
      and ground-plane longitudinal distance Z_bottom (m) using configurable camera height,
      downward pitch, and focal length / FOV from config/sensor_profiles.yaml.
    - Projects bbox width (xmax - xmin) and height (ymax - ymin) onto the road plane at Z_bottom.
    - Clamps to physically plausible per-class bounds (e.g. pothole D40: 0.01 - 1.0 m^2)
      and logs whenever clamping occurs.
    """
    cfg = camera_cfg or load_camera_profile_config()
    xmin, ymin, xmax, ymax = [float(v) for v in bbox[:4]]
    width = float(max(1, image_dims.get("width", cfg.get("reference_image_width_px", 720))))
    height = float(max(1, image_dims.get("height", cfg.get("reference_image_height_px", 720))))

    box_w_px = max(1.0, xmax - xmin)
    box_h_px = max(1.0, ymax - ymin)

    cam_h_m = float(cfg.get("camera_height_m", 1.35))
    pitch_rad = math.radians(float(cfg.get("pitch_deg", 22.0)))
    hfov_rad = math.radians(float(cfg.get("hfov_deg", 62.0)))
    vfov_rad = math.radians(float(cfg.get("vfov_deg", 48.0)))
    min_dep_rad = math.radians(float(cfg.get("min_depression_deg", 5.0)))

    # Derive effective horizontal and vertical focal lengths in pixels from FOV / focal_length_px
    fx = (width / 2.0) / math.tan(max(1e-3, hfov_rad / 2.0))
    fy = (height / 2.0) / math.tan(max(1e-3, vfov_rad / 2.0))

    # Convert bbox bottom-edge position (ymax) to vertical ray angle relative to optical axis
    y_center = height / 2.0
    y_bottom_clamped = min(height, max(0.0, ymax))
    alpha_bottom = math.atan2(y_bottom_clamped - y_center, fy)
    depression_rad = max(min_dep_rad, min(math.radians(85.0), pitch_rad + alpha_bottom))

    # Ground-plane longitudinal distance Z (m) to bbox bottom edge
    z_bottom_m = cam_h_m / math.tan(depression_rad)

    # Real-world lateral width (m) and ground-foreshortened longitudinal length (m)
    real_width_m = (box_w_px * z_bottom_m) / fx
    real_length_m = (box_h_px * z_bottom_m) / (fy * math.sin(depression_rad))
    raw_area_sqm = real_width_m * real_length_m

    cls_code = normalize_damage_class_code(damage_class)
    bounds_map = cfg.get("class_area_bounds_sqm", DEFAULT_CAMERA_CONFIG["class_area_bounds_sqm"])
    min_area, max_area = bounds_map.get(cls_code, [0.01, 1.00])
    min_area = float(min_area)
    max_area = float(max_area)

    if raw_area_sqm < min_area or raw_area_sqm > max_area:
        clamped_area = max(min_area, min(max_area, raw_area_sqm))
        logger.warning(
            "Clamped ground-plane area for class %s from %.4f m^2 to %.4f m^2 (bounds=[%.2f, %.2f], Z_bottom=%.2fm)",
            cls_code,
            raw_area_sqm,
            clamped_area,
            min_area,
            max_area,
            z_bottom_m,
        )
        return round(clamped_area, 3)

    return round(raw_area_sqm, 3)


class PhysicalConsistencyModel:
    """
    Unified physical consistency engine that resolves latent defect severity
    and derives correlated physical, sensor, dynamic, and municipal risk properties.
    """
    def __init__(
        self,
        seed: int = 2026,
        baseline_chassis_cm: float = DEFAULT_CHASSIS_BASELINE_CM,
        camera_config: Optional[Dict[str, Any]] = None
    ):
        self.seed = seed
        self.baseline_chassis_cm = baseline_chassis_cm
        self.rng = random.Random(seed)
        self.camera_config = camera_config or load_camera_profile_config()

    def reseed(self, seed: int):
        self.seed = seed
        self.rng = random.Random(seed)

    def compute_perspective_surface_area(
        self,
        bbox: List[float],
        image_dims: Dict[str, int],
        damage_class: Optional[str] = "D40"
    ) -> float:
        """
        Calculates ground-plane surface area in square meters using a pinhole camera
        ground-plane model (camera height, downward pitch, FOV/focal length from
        config/sensor_profiles.yaml), converting bbox bottom-edge position (ymax) to
        real-world meters and clamping to per-class physical bounds.
        """
        return estimate_ground_plane_area_pinhole(
            bbox=bbox,
            image_dims=image_dims,
            damage_class=damage_class,
            camera_cfg=self.camera_config
        )

    def derive_latent_severity(
        self,
        damage_class: str,
        surface_area_sqm: float
    ) -> float:
        """
        Derives the latent scalar variable `defect_severity` ∈ [0.01, 1.00].
        
        The latent variable couples visual bounding box size and damage taxonomy.
        """
        cls_code = normalize_damage_class_code(damage_class)
        prior = CLASS_SEVERITY_PRIORS.get(cls_code, CLASS_SEVERITY_PRIORS["D40"])
        s_min = prior["min_s"]
        s_max = prior["max_s"]

        # Scale severity monotonically with observed geometric area within per-class bounds
        if cls_code == "D40":
            # Pothole area normalization: 0.01 m² (small) to 1.00 m² (clamped max)
            area_factor = min(1.0, max(0.0, (surface_area_sqm - 0.01) / 0.99))
        elif cls_code == "D20":
            # Alligator area normalization: 0.05 m² to 4.00 m²
            area_factor = min(1.0, max(0.0, (surface_area_sqm - 0.05) / 3.95))
        else:
            # Linear cracks: 0.01 m² to 1.50 m²
            area_factor = min(1.0, max(0.0, (surface_area_sqm - 0.01) / 1.49))

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
    ) -> Tuple[float, Optional[float], str]:
        """
        Derives physical defect depth (cm) and cavity volume (liters).
        
        Class-aware physical rules:
        - Cavity volume (liters) is computed ONLY for D40 (pothole):
          volume > 0 ⟺ (depth > 0 AND surface_area_sqm > 0).
        - For crack classes (D00, D10, D20), calculated_volume_liters is None
          and crack_severity_index ∈ [0, 1] is computed instead.
        """
        cls_code = normalize_damage_class_code(damage_class)
        s = defect_severity

        if cls_code == "D40":
            # Potholes: 4.8 cm to 18.5 cm
            # Moderate potholes (< 1.0 m² area) produce moderate void (4.8 - 6.2 cm, borderline shock)
            # Large severe potholes (>= 1.0 m² area) produce deep void (10.0 - 16.0 cm, confirmed impact)
            if surface_area_sqm < 1.0:
                depth_cm = round(4.8 + 1.8 * (s ** 1.1), 2)
            else:
                depth_cm = round(9.0 + 8.5 * (s ** 1.3), 2)
        elif cls_code == "D20":
            # Alligator fatigue: 2.0 cm to 5.5 cm
            depth_cm = round(1.8 + 4.2 * s, 2)
        elif cls_code == "D10":
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

        if cls_code == "D40":
            # Volume calculation (parabolic / conical cavity approximation) ONLY for D40 potholes:
            # V = shape_factor * area_cm² * depth_cm / 1000 cm³/L
            shape_factor = 0.50
            if surface_area_sqm > 0 and depth_cm > 0:
                volume_liters: Optional[float] = round(
                    shape_factor * (surface_area_sqm * 10000.0) * depth_cm / 1000.0, 2
                )
            else:
                volume_liters = 0.0
        else:
            volume_liters = None

        return depth_cm, volume_liters, tier

    @staticmethod
    def compute_crack_severity_index(
        damage_class: str,
        surface_area_sqm: float,
        depth_cm: float,
        bbox: Optional[List[float]] = None
    ) -> Optional[float]:
        """
        Computes the bounded [0.0, 1.0] `crack_severity_index` for crack classes (D00, D10, D20),
        returning None for D40 potholes:
        - D00 / D10 (Linear Cracks): bbox_length_m * depth_cm normalized by 6.0 m·cm and clamped to [0, 1].
        - D20 (Alligator Fatigue Crack): affected surface_area_sqm * (depth_cm / 5.0) normalized by 2.5 m² and clamped to [0, 1].
        """
        cls_code = normalize_damage_class_code(damage_class)
        if cls_code == "D40":
            return None
        valid_area = max(0.0, float(surface_area_sqm))
        valid_depth = max(0.0, float(depth_cm))
        if valid_area <= 0.0 or valid_depth <= 0.0:
            return 0.0

        if cls_code == "D20":
            depth_weight = min(1.5, max(0.1, valid_depth / 5.0))
            raw_fatigue = valid_area * depth_weight
            return round(min(1.0, max(0.0, raw_fatigue / 2.5)), 4)

        # D00 / D10: bbox length (m) x depth (cm)
        if bbox is not None and len(bbox) == 4:
            bw = max(1.0, float(bbox[2]) - float(bbox[0]))
            bh = max(1.0, float(bbox[3]) - float(bbox[1]))
            aspect = max(1.0, min(12.0, max(bw, bh) / min(bw, bh)))
        else:
            aspect = 6.0
        bbox_length_m = math.sqrt(valid_area * aspect)
        raw_linear = bbox_length_m * valid_depth
        return round(min(1.0, max(0.0, raw_linear / 6.0)), 4)

    @staticmethod
    def compute_defect_severity_magnitude(
        damage_class: str,
        volume_liters: Optional[float],
        crack_severity_index: Optional[float]
    ) -> float:
        """
        Computes the unified `defect_severity_magnitude` column for MCDM (TOPSIS/VIKOR Criterion 0):
        - For D40 (pothole): calculated_volume_liters (L).
        - For D00/D10/D20 (cracks): crack_severity_index * CRACK_SEVERITY_TO_MAGNITUDE_SCALE (15.0 L-equivalent).
        """
        cls_code = normalize_damage_class_code(damage_class)
        if cls_code == "D40":
            return round(max(0.0, float(volume_liters if volume_liters is not None else 0.0)), 2)
        csi = max(0.0, float(crack_severity_index if crack_severity_index is not None else 0.0))
        return round(csi * CRACK_SEVERITY_TO_MAGNITUDE_SCALE, 2)

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
        - When rain_detected=True:
            * Sonar gets water-surface systematic bias (-0.65 cm), larger surface-ripple noise,
              and JSN-SR04T min-range dead zone (20.0 cm).
            * LiDAR gets wet-surface specular degradation (-1.85 cm) that is NOT applied to sonar.
        """
        if rain_detected:
            lidar_noise = round(self.rng.uniform(-0.42, 0.42), 2)
            wet_lidar_degradation_cm = -1.85
            sonar_ripple_noise = round(self.rng.uniform(-0.85, 0.85), 2)
            sonar_bias_cm = -0.65
            sonar_noise = round(sonar_bias_cm + sonar_ripple_noise, 2)
        else:
            lidar_noise = round(self.rng.uniform(-0.15, 0.15), 2)
            wet_lidar_degradation_cm = 0.0
            sonar_bias_cm = 0.0
            sonar_noise = round(self.rng.uniform(-0.30, 0.30), 2)

        # Raw distance from chassis down to road defect base
        lidar_effective_cm = round(max(0.1, defect_depth_cm + wet_lidar_degradation_cm + lidar_noise), 2)
        raw_lidar_cm = round(self.baseline_chassis_cm + lidar_effective_cm, 2)

        sonar_effective_cm = round(max(0.1, defect_depth_cm + sonar_noise), 2)
        raw_sonar_cm = round(max(20.0, self.baseline_chassis_cm + sonar_effective_cm), 2)

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
            "lidar_effective_depth_cm": lidar_effective_cm,
            "wet_lidar_degradation_cm": wet_lidar_degradation_cm,
            "raw_sonar_cm": raw_sonar_cm,
            "sonar_depth_cm": raw_sonar_cm,
            "sonar_effective_depth_cm": sonar_effective_cm,
            "sonar_error_cm": sonar_noise,
            "sonar_systematic_bias_cm": sonar_bias_cm,
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
        road_class: Optional[str] = None,
        bbox: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Derives an integrated package of physically consistent telemetry:
        latent severity, depth, volume/crack_severity_index, impact acceleration, sensors, and context.
        """
        # 1. Latent Defect Severity
        defect_severity = self.derive_latent_severity(damage_class, surface_area_sqm)

        # 2. Physics-grounded Depth and Class-Aware Volume / Crack Severity Index
        defect_depth_cm, volume_liters, severity_tier = self.derive_depth_and_volume(
            defect_severity=defect_severity,
            damage_class=damage_class,
            surface_area_sqm=surface_area_sqm
        )
        crack_severity_index = self.compute_crack_severity_index(
            damage_class=damage_class,
            surface_area_sqm=surface_area_sqm,
            depth_cm=defect_depth_cm,
            bbox=bbox
        )
        defect_severity_magnitude = self.compute_defect_severity_magnitude(
            damage_class=damage_class,
            volume_liters=volume_liters,
            crack_severity_index=crack_severity_index
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
            "crack_severity_index": crack_severity_index,
            "defect_severity_magnitude": defect_severity_magnitude,
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

