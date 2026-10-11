"""
Sonar Acoustic Echo-Ranging Model with Configurable Noise
=========================================================

Physics-based ultrasonic acoustic echo-ranging model for road surface profiling.
Simulates a 40 kHz downward-facing acoustic transceiver array mounted alongside LiDAR.

Physical Dynamics:
- Measurement formula: sonar_depth = true_depth + sensor_error
- Configurable noise profiles defined in config/sensor_profiles.yaml:
    * Dry Day: Beam divergence (~15 deg) causes spatial edge averaging; higher noise than optical LiDAR.
    * Rain / Wet / Submerged: Optical LiDAR suffers specular loss, while ultrasonic acoustic waves
      penetrate standing water puddles cleanly, reflecting off the void bottom with high confidence.
    * Low Light (< 40 lux): Acoustic echoes are immune to darkness, outperforming optical sensors.
- Integrates with the environmental sensor-switching rule:
    rain_detected OR (mean_luminance < 40.0 lux) -> sonar_submerged_acoustic
    otherwise -> optical_lidar

Every generated measurement records:
    sensor_source = "synthetic"
    generation_method = "controlled_distribution"
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

DEFAULT_SONAR_CONFIG = {
    "sonar": {
        "baseline_chassis_cm": 30.0,
        "transducer_frequency_khz": 40,
        "beam_divergence_deg": 15,
        "water_surface_model": {
            "transducer_model": "JSN-SR04T",
            "min_range_dead_zone_cm": 20.0,
            "systematic_bias_cm": -0.65,
            "speed_of_sound_humidity_factor": 1.006,
            "surface_ripple_std_cm": 0.85,
        },
        "noise": {
            "dry_day": {
                "noise_std_cm": 0.35,
                "systematic_bias_cm": 0.0,
                "beam_averaging_factor": 0.88,
                "confidence_min": 0.78,
                "confidence_max": 0.88
            },
            "rain_wet": {
                "noise_std_cm": 0.85,
                "systematic_bias_cm": -0.65,
                "min_range_dead_zone_cm": 20.0,
                "submerged_penetration_factor": 1.00,
                "confidence_min": 0.82,
                "confidence_max": 0.93
            },
            "low_light": {
                "noise_std_cm": 0.38,
                "systematic_bias_cm": 0.0,
                "confidence_min": 0.82,
                "confidence_max": 0.90
            }
        },
        "switching_thresholds": {
            "luminance_threshold_lux": 40.0,
            "rain_detected_triggers_sonar": True
        }
    }
}


class SonarModel:
    """
    Configurable Sonar Acoustic model with JSN-SR04T water-surface physics
    (rain systematic bias, surface ripple noise, minimum-range dead zone)
    and environmental switching integration.
    """
    def __init__(
        self,
        baseline_chassis_cm: float = 30.0,
        config_path: Optional[Path] = None,
        config_override: Optional[Dict[str, Any]] = None,
        seed: int = 42
    ):
        self.rng = random.Random(seed)
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self.config = self._load_configuration(config_override)

        cfg_baseline = self.config.get("sonar", {}).get("baseline_chassis_cm")
        self.baseline_chassis_cm = cfg_baseline if cfg_baseline is not None else baseline_chassis_cm

    def _load_configuration(self, override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Loads sonar configuration from YAML with fallback."""
        if override:
            return override

        if yaml and self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f)
                if loaded and "sonar" in loaded:
                    return loaded
            except Exception as e:
                print(f"Warning: Failed to load sonar config from {self.config_path}: {e}")

        return DEFAULT_SONAR_CONFIG

    def reseed(self, seed: int):
        self.rng = random.Random(seed)

    def evaluate_switching_recommendation(
        self,
        rain_detected: bool,
        mean_luminance: float
    ) -> Dict[str, Any]:
        """
        Evaluates the existing environmental sensor-switching logic:
        rain_detected OR (mean_luminance < 40.0) -> sonar_submerged_acoustic
        otherwise -> optical_lidar
        """
        sonar_cfg = self.config.get("sonar", DEFAULT_SONAR_CONFIG["sonar"])
        thresholds = sonar_cfg.get("switching_thresholds", DEFAULT_SONAR_CONFIG["sonar"]["switching_thresholds"])
        lux_thresh = thresholds.get("luminance_threshold_lux", 40.0)

        use_sonar = bool(rain_detected or (mean_luminance < lux_thresh))
        is_submerged = bool(rain_detected)
        active_modality = "sonar_submerged_acoustic" if use_sonar else "optical_lidar"

        if is_submerged:
            reason = "Precipitation / Submerged Road Condition"
        elif mean_luminance < lux_thresh:
            reason = f"Low Luminance ({mean_luminance:.1f} lux < {lux_thresh:.1f} lux)"
        else:
            reason = f"Clear Dry Daylight ({mean_luminance:.1f} lux >= {lux_thresh:.1f} lux)"

        return {
            "use_sonar": use_sonar,
            "is_submerged": is_submerged,
            "active_modality": active_modality,
            "switching_reason": reason
        }

    def simulate(
        self,
        damage_class: str,
        true_defect_depth_cm: float,
        rain_detected: bool = False,
        mean_luminance: float = 70.0
    ) -> Dict[str, Any]:
        """
        Simulate acoustic sonar measurement:
        sonar_depth = true_depth + sensor_error

        When rain_detected=True, applies the JSN-SR04T water-surface model:
        - systematic_bias_cm (speed-of-sound shift + water-surface ripple reflection offset)
        - larger surface_ripple_std_cm Gaussian noise
        - min_range_dead_zone_cm transducer blanking clamp
        Note: Wet-surface optical LiDAR specular dropout degradation is NEVER applied to sonar.
        """
        sonar_cfg = self.config.get("sonar", DEFAULT_SONAR_CONFIG["sonar"])
        noise_cfg = sonar_cfg.get("noise", DEFAULT_SONAR_CONFIG["sonar"]["noise"])
        wsm_cfg = sonar_cfg.get(
            "water_surface_model",
            DEFAULT_SONAR_CONFIG["sonar"]["water_surface_model"]
        )
        thresholds = sonar_cfg.get("switching_thresholds", DEFAULT_SONAR_CONFIG["sonar"]["switching_thresholds"])
        lux_thresh = thresholds.get("luminance_threshold_lux", 40.0)

        switching = self.evaluate_switching_recommendation(rain_detected, mean_luminance)
        min_dead_zone_cm = float(wsm_cfg.get("min_range_dead_zone_cm", 20.0))

        # Select environmental noise profile
        if rain_detected:
            env_prof = noise_cfg.get(
                "rain_wet",
                {"noise_std_cm": 0.85, "systematic_bias_cm": -0.65, "confidence_min": 0.82, "confidence_max": 0.93}
            )
            is_submerged_echo = True
            penetration_factor = float(env_prof.get("submerged_penetration_factor", 1.00))
            apparent_base = true_defect_depth_cm * penetration_factor
            systematic_bias_cm = float(
                env_prof.get("systematic_bias_cm", wsm_cfg.get("systematic_bias_cm", -0.65))
            )
            noise_std = float(
                env_prof.get("noise_std_cm", wsm_cfg.get("surface_ripple_std_cm", 0.85))
            )
            min_dead_zone_cm = float(env_prof.get("min_range_dead_zone_cm", min_dead_zone_cm))
        elif mean_luminance < lux_thresh:
            env_prof = noise_cfg.get(
                "low_light",
                {"noise_std_cm": 0.38, "systematic_bias_cm": 0.0, "confidence_min": 0.82, "confidence_max": 0.90}
            )
            is_submerged_echo = False
            apparent_base = true_defect_depth_cm
            systematic_bias_cm = 0.0
            noise_std = float(env_prof.get("noise_std_cm", 0.38))
        else:
            env_prof = noise_cfg.get(
                "dry_day",
                {"noise_std_cm": 0.35, "systematic_bias_cm": 0.0, "beam_averaging_factor": 0.88, "confidence_min": 0.78, "confidence_max": 0.88}
            )
            is_submerged_echo = False
            systematic_bias_cm = 0.0
            noise_std = float(env_prof.get("noise_std_cm", 0.35))
            # Wide acoustic beam cone causes slight boundary averaging on narrow cracks
            if damage_class in ("D00", "D10"):
                averaging = float(env_prof.get("beam_averaging_factor", 0.88))
                apparent_base = true_defect_depth_cm * averaging
            else:
                apparent_base = true_defect_depth_cm

        stochastic_noise = self.rng.gauss(0.0, noise_std)
        sensor_error = round(systematic_bias_cm + stochastic_noise, 2)

        # sonar_depth = true_depth + sensor_error, enforcing JSN-SR04T min-range dead zone
        effective_depth_cm = round(max(0.1, apparent_base + sensor_error), 2)
        unclamped_raw_cm = round(self.baseline_chassis_cm + effective_depth_cm, 2)
        dead_zone_clamped = bool(unclamped_raw_cm < min_dead_zone_cm)
        raw_depth_cm = round(max(min_dead_zone_cm, unclamped_raw_cm), 2)
        if dead_zone_clamped:
            effective_depth_cm = round(max(0.0, raw_depth_cm - self.baseline_chassis_cm), 2)

        conf_min = float(env_prof.get("confidence_min", 0.80))
        conf_max = float(env_prof.get("confidence_max", 0.95))
        confidence = round(self.rng.uniform(conf_min, conf_max), 3)

        return {
            "depth_cm": raw_depth_cm,
            "effective_depth_cm": effective_depth_cm,
            "baseline_chassis_cm": self.baseline_chassis_cm,
            "sensor_error_cm": sensor_error,
            "systematic_bias_cm": round(systematic_bias_cm, 2),
            "noise_std_cm": round(noise_std, 2),
            "min_range_dead_zone_cm": min_dead_zone_cm,
            "dead_zone_clamped": dead_zone_clamped,
            "wet_lidar_degradation_applied": False,
            "true_depth_reference_cm": round(true_defect_depth_cm, 2),
            "sensor_source": "synthetic",
            "generation_method": "controlled_distribution",
            "confidence": confidence,
            "transducer_frequency_khz": sonar_cfg.get("transducer_frequency_khz", 40),
            "beam_divergence_deg": sonar_cfg.get("beam_divergence_deg", 15),
            "is_submerged_echo": is_submerged_echo,
            "environmental_switching": switching
        }
