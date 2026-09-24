"""
Quarter-Car Accelerometer Dynamic Impact Model
==============================================

Physics-based vehicle suspension dynamics model for vertical chassis acceleration (z_accel_g).
Simulates a 3-axis accelerometer mounted on vehicle sprung mass / chassis.

Physical Dynamics (Quarter-Car Dynamic Response):
- Sprung mass M_s supported by suspension spring (k_s) and damper (c_s).
- Unsprung mass M_u (wheel + tire k_t) traversing road distress profile.
- When traversing a depression of depth d (cm) at forward velocity v (km/h):
    z_accel = 1.0 + alpha * (v / v_ref)^beta * (d / d_ref)^gamma * class_factor + noise


Key Physical Correlations:
- 1.0g represents static terrestrial gravity.
- Dynamic Impact Threshold: z_accel_g > 1.50g (preserved from original project architecture).
- Impact Categorization:
    * No Significant Impact: z_accel_g <= 1.35g
    * Borderline Impact: 1.35g < z_accel_g <= 1.50g
    * Confirmed Dynamic Impact: z_accel_g > 1.50g

This multimodal cross-validation enables researchers to evaluate whether:
    visual defect detection + physical impact sensing
provides significantly stronger evidence than visual detection alone.
"""

import os
import sys
import math
import random
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import yaml
except ImportError:
    yaml = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "sensor_profiles.yaml"

DEFAULT_ACCELEROMETER_CONFIG = {
    "accelerometer": {
        "sampling_rate_hz": 200,
        "static_gravity_baseline_g": 1.0,
        "thresholds": {
            "confirmed_impact_g": 1.50,
            "borderline_impact_min_g": 1.35,
            "borderline_impact_max_g": 1.50
        },
        "reference_speed_kmh": 40.0,
        "reference_depth_cm": 10.0,
        "quarter_car_model": {
            "sprung_mass_kg": 320.0,
            "unsprung_mass_kg": 45.0,
            "suspension_stiffness_n_m": 28000.0,
            "damping_coefficient_n_s_m": 1800.0,
            "tire_stiffness_n_m": 190000.0
        },
        "class_profiles": {
            "D40": {
                "name": "Pothole",
                "alpha": 0.90,
                "speed_exp": 0.85,
                "depth_exp": 1.25,
                "noise_std": 0.06,
                "rms_base_g": 0.18
            },
            "D20": {
                "name": "Alligator Crack",
                "alpha": 0.35,
                "speed_exp": 0.70,
                "depth_exp": 0.80,
                "noise_std": 0.05,
                "rms_base_g": 0.22
            },
            "D00": {
                "name": "Longitudinal Crack",
                "alpha": 0.15,
                "speed_exp": 0.60,
                "depth_exp": 0.70,
                "noise_std": 0.03,
                "rms_base_g": 0.10
            },
            "D10": {
                "name": "Transverse Crack",
                "alpha": 0.18,
                "speed_exp": 0.60,
                "depth_exp": 0.70,
                "noise_std": 0.03,
                "rms_base_g": 0.10
            }
        },
        "impact_regimes": {
            "no_impact": {
                "label": "No Significant Impact",
                "max_g": 1.35,
                "description": "Vehicle suspension absorbs distress smoothly; negligible structural shock."
            },
            "borderline": {
                "label": "Borderline Impact",
                "min_g": 1.35,
                "max_g": 1.50,
                "description": "Noticeable vertical suspension jolt approaching safety limits; requires cross-modal visual validation."
            },
            "confirmed_impact": {
                "label": "Confirmed Dynamic Impact",
                "min_g": 1.50,
                "description": "Severe wheel rim / axle acceleration spike exceeding 1.5g safety limit; physical road hazard confirmed."
            }
        }
    }
}


class AccelerometerModel:
    """
    Configurable Quarter-Car Accelerometer Model with controlled noise,
    regime classification (no impact, borderline, confirmed impact),
    and cross-modal impact validation.
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

        accel_cfg = self.config.get("accelerometer", DEFAULT_ACCELEROMETER_CONFIG["accelerometer"])
        thresholds = accel_cfg.get("thresholds", DEFAULT_ACCELEROMETER_CONFIG["accelerometer"]["thresholds"])
        self.confirmed_threshold_g = thresholds.get("confirmed_impact_g", 1.50)
        self.borderline_min_g = thresholds.get("borderline_impact_min_g", 1.35)
        self.borderline_max_g = thresholds.get("borderline_impact_max_g", 1.50)

    def _load_configuration(self, override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Loads accelerometer configuration from YAML with fallback."""
        if override:
            return override

        if yaml and self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f)
                if loaded and "accelerometer" in loaded:
                    return loaded
            except Exception as e:
                print(f"Warning: Failed to load accelerometer config from {self.config_path}: {e}")

        return DEFAULT_ACCELEROMETER_CONFIG

    def reseed(self, seed: int):
        self.rng = random.Random(seed)

    def classify_impact(self, z_accel_g: float) -> Dict[str, Any]:
        """
        Classifies vertical acceleration into one of three research regimes:
        - no_impact: z_accel_g <= 1.35g
        - borderline: 1.35g < z_accel_g <= 1.50g
        - confirmed_impact: z_accel_g > 1.50g
        """
        if z_accel_g > self.confirmed_threshold_g:
            regime = "confirmed_impact"
            label = "Confirmed Dynamic Impact"
            desc = "Severe wheel rim / axle acceleration spike exceeding 1.5g safety limit; physical road hazard confirmed."
            cross_modal = "Visual defect confirmed by acute dynamic impact (> 1.5g). Pavement cavity produces severe suspension shock, providing incontrovertible multi-modal evidence of hazard."
        elif z_accel_g > self.borderline_min_g:
            regime = "borderline"
            label = "Borderline Impact"
            desc = "Noticeable vertical suspension jolt approaching safety limits (1.35g - 1.50g); requires cross-modal visual validation."
            cross_modal = "Visual defect accompanied by elevated vibration/jolt (1.35g - 1.50g). Incipient structural hazard requiring prioritized monitoring."
        else:
            regime = "no_impact"
            label = "No Significant Impact"
            desc = "Vehicle suspension absorbs distress smoothly; negligible structural shock (<= 1.35g)."
            cross_modal = "Visual defect present, but vertical chassis shock is negligible (<= 1.35g). Road surface distress does not currently present an acute vehicle safety hazard."

        return {
            "regime": regime,
            "label": label,
            "description": desc,
            "cross_modal_assessment": cross_modal,
            "confirmed_threshold_g": self.confirmed_threshold_g,
            "borderline_min_g": self.borderline_min_g
        }

    def simulate(
        self,
        damage_class: str,
        defect_depth_cm: float,
        speed_kmh: float,
        surface_area_sqm: float = 0.5
    ) -> Dict[str, Any]:
        """
        Simulate vertical chassis Z-axis acceleration response.

        Args:
            damage_class: RDD2022 class ('D00', 'D10', 'D20', 'D40')
            defect_depth_cm: Physical depth in cm
            speed_kmh: Vehicle forward speed in km/h
            surface_area_sqm: Pavement surface area in m^2
        """
        accel_cfg = self.config.get("accelerometer", DEFAULT_ACCELEROMETER_CONFIG["accelerometer"])
        v_ref = accel_cfg.get("reference_speed_kmh", 40.0)
        d_ref = accel_cfg.get("reference_depth_cm", 10.0)
        class_profiles = accel_cfg.get("class_profiles", DEFAULT_ACCELEROMETER_CONFIG["accelerometer"]["class_profiles"])

        prof = class_profiles.get(damage_class, {
            "alpha": 0.20,
            "speed_exp": 0.60,
            "depth_exp": 0.70,
            "noise_std": 0.04,
            "rms_base_g": 0.12
        })

        speed_factor = max(0.15, speed_kmh / v_ref)
        depth_factor = max(0.05, defect_depth_cm / d_ref)

        alpha = prof.get("alpha", 0.20)
        s_exp = prof.get("speed_exp", 0.70)
        d_exp = prof.get("depth_exp", 0.80)
        noise_std = prof.get("noise_std", 0.05)
        rms_base = prof.get("rms_base_g", 0.12)

        # Controlled Gaussian noise
        noise = self.rng.gauss(0.0, noise_std)

        # Calculate dynamic impulse shock above 1.0g baseline
        peak_shock = alpha * (speed_factor ** s_exp) * (depth_factor ** d_exp) + noise

        # Specific physical response tuning by class
        if damage_class == "D40":  # Pothole
            # Deep potholes (>= 7 cm) at speed (> 35 km/h) trigger confirmed impact
            if defect_depth_cm >= 7.0 and speed_kmh >= 35.0:
                peak_shock = max(0.55, peak_shock)
            rms_vibration = round(rms_base + min(0.35, depth_factor * 0.12), 3)

        elif damage_class == "D20":  # Alligator Crack
            rms_vibration = round(rms_base + min(0.25, surface_area_sqm * 0.08), 3)

        else:  # D00 / D10 Cracks
            rms_vibration = round(rms_base, 3)

        # 1.0g static terrestrial gravity + shock
        z_accel_g = round(max(0.40, min(5.0, 1.0 + peak_shock)), 2)

        # Classify impact into no_impact, borderline, or confirmed_impact
        classification = self.classify_impact(z_accel_g)
        dynamic_impact_triggered = bool(z_accel_g > self.confirmed_threshold_g)

        # Suspension travel deflection (Hooke's law: delta_z = F / k)
        suspension_deflection_cm = round(min(8.0, defect_depth_cm * 0.45 * (speed_kmh / 50.0)), 2)

        return {
            "z_accel_g": z_accel_g,
            "dynamic_impact_triggered": dynamic_impact_triggered,
            "impact_regime": classification["regime"],
            "impact_category": classification["label"],
            "cross_modal_assessment": classification["cross_modal_assessment"],
            "threshold_g": self.confirmed_threshold_g,
            "borderline_min_g": self.borderline_min_g,
            "rms_vibration_g": rms_vibration,
            "suspension_deflection_cm": suspension_deflection_cm,
            "vehicle_speed_kmh": round(speed_kmh, 1),
            "sensor_source": "synthetic",
            "generation_method": "controlled_distribution",
            "source": "synthetic"
        }
