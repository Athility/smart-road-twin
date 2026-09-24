"""
LiDAR Profilometry Model with Controlled Depth Distributions
=============================================================

Physics-based optical time-of-flight LiDAR distance model for road surface profiling.
Implements controlled, configurable depth distributions based on:
1. Distress Class (D00, D10, D20, D40)
2. Defect Surface Area & Severity Tiering (shallow, moderate, deep)
3. Configuration from config/sensor_profiles.yaml

Core Physical Principle:
- Larger/more severe potholes exhibit higher probability of deeper voids,
  while maintaining realistic stochastic variation via controlled probability distributions.
- Every generated measurement records:
    sensor_source = "synthetic"
    generation_method = "controlled_distribution"
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

# Default fallback configuration if YAML file is unavailable
DEFAULT_CONFIG = {
    "lidar": {
        "baseline_chassis_cm": 30.0,
        "wavelength_nm": 905,
        "depth_tiers": {
            "shallow": {
                "min_depth_cm": 1.0,
                "max_depth_cm": 4.5,
                "distribution": "truncated_normal",
                "mean_cm": 2.5,
                "std_cm": 0.8
            },
            "moderate": {
                "min_depth_cm": 4.5,
                "max_depth_cm": 9.0,
                "distribution": "truncated_normal",
                "mean_cm": 6.8,
                "std_cm": 1.2
            },
            "deep": {
                "min_depth_cm": 9.0,
                "max_depth_cm": 18.5,
                "distribution": "truncated_normal",
                "mean_cm": 13.5,
                "std_cm": 2.2
            }
        },
        "class_profiles": {
            "D40": {
                "name": "Pothole",
                "area_thresholds_sqm": {"small": 0.25, "medium": 0.65, "large": 0.65},
                "tier_probabilities": {
                    "small":  {"shallow": 0.55, "moderate": 0.35, "deep": 0.10},
                    "medium": {"shallow": 0.15, "moderate": 0.60, "deep": 0.25},
                    "large":  {"shallow": 0.05, "moderate": 0.30, "deep": 0.65}
                }
            },
            "D20": {
                "name": "Alligator Crack",
                "area_thresholds_sqm": {"small": 0.30, "medium": 0.70, "large": 0.70},
                "tier_probabilities": {
                    "small":  {"shallow": 0.75, "moderate": 0.23, "deep": 0.02},
                    "medium": {"shallow": 0.50, "moderate": 0.45, "deep": 0.05},
                    "large":  {"shallow": 0.30, "moderate": 0.60, "deep": 0.10}
                }
            },
            "D00": {
                "name": "Longitudinal Crack",
                "tier_probabilities": {"default": {"shallow": 0.95, "moderate": 0.05, "deep": 0.00}},
                "depth_override": {"min_depth_cm": 0.4, "max_depth_cm": 2.2, "mean_cm": 1.2, "std_cm": 0.4}
            },
            "D10": {
                "name": "Transverse Crack",
                "tier_probabilities": {"default": {"shallow": 0.92, "moderate": 0.08, "deep": 0.00}},
                "depth_override": {"min_depth_cm": 0.5, "max_depth_cm": 2.5, "mean_cm": 1.3, "std_cm": 0.45}
            }
        },
        "noise": {
            "dry_day": {"noise_std_cm": 0.10, "confidence_min": 0.90, "confidence_max": 0.99},
            "rain_wet": {"noise_std_cm": 0.42, "confidence_min": 0.45, "confidence_max": 0.78, "specular_dropout_prob": 0.08}
        }
    }
}


class LiDARModel:
    """
    Configurable LiDAR model utilizing controlled probability distributions.
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
        
        # Override baseline chassis height if explicitly passed or defined in config
        cfg_baseline = self.config.get("lidar", {}).get("baseline_chassis_cm")
        self.baseline_chassis_cm = cfg_baseline if cfg_baseline is not None else baseline_chassis_cm

    def _load_configuration(self, override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Loads sensor profiles from YAML configuration with fallback."""
        if override:
            return override

        if yaml and self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f)
                if loaded and "lidar" in loaded:
                    return loaded
            except Exception as e:
                print(f"Warning: Failed to load {self.config_path}: {e}. Using default config.")

        return DEFAULT_CONFIG

    def reseed(self, seed: int):
        self.rng = random.Random(seed)

    def _determine_severity_tier(self, damage_class: str, surface_area_sqm: float) -> str:
        """
        Probabilistically selects a severity tier (shallow, moderate, deep)
        conditioned on defect distress class and surface area.
        """
        lidar_cfg = self.config.get("lidar", DEFAULT_CONFIG["lidar"])
        class_profiles = lidar_cfg.get("class_profiles", DEFAULT_CONFIG["lidar"]["class_profiles"])
        prof = class_profiles.get(damage_class, class_profiles.get("D40"))

        tier_probs = prof.get("tier_probabilities", {})

        if "default" in tier_probs:
            probs = tier_probs["default"]
        else:
            thresholds = prof.get("area_thresholds_sqm", {"small": 0.25, "medium": 0.65})
            if surface_area_sqm < thresholds.get("small", 0.25):
                size_cat = "small"
            elif surface_area_sqm <= thresholds.get("medium", 0.65):
                size_cat = "medium"
            else:
                size_cat = "large"
            probs = tier_probs.get(size_cat, {"shallow": 0.33, "moderate": 0.34, "deep": 0.33})

        # Categorical sampling via cumulative distribution
        r = self.rng.random()
        cum = 0.0
        for tier in ["shallow", "moderate", "deep"]:
            cum += probs.get(tier, 0.0)
            if r <= cum:
                return tier
        return "moderate"

    def _sample_truncated_normal(self, mean: float, std: float, low: float, high: float) -> float:
        """Samples from a Gaussian distribution bounded within [low, high]."""
        for _ in range(12):
            val = self.rng.gauss(mean, std)
            if low <= val <= high:
                return val
        return max(low, min(high, mean))

    def simulate(
        self,
        damage_class: str,
        surface_area_sqm: float = 0.5,
        rain_detected: bool = False,
        mean_luminance: float = 70.0
    ) -> Dict[str, Any]:
        """
        Simulate downward-facing LiDAR distance measurement using controlled distributions.

        Args:
            damage_class: RDD2022 class ('D00', 'D10', 'D20', 'D40')
            surface_area_sqm: Estimated surface area of the defect in m^2
            rain_detected: Whether rain/standing water is present
            mean_luminance: Ambient light lux
        """
        lidar_cfg = self.config.get("lidar", DEFAULT_CONFIG["lidar"])
        depth_tiers = lidar_cfg.get("depth_tiers", DEFAULT_CONFIG["lidar"]["depth_tiers"])
        class_profiles = lidar_cfg.get("class_profiles", DEFAULT_CONFIG["lidar"]["class_profiles"])
        prof = class_profiles.get(damage_class, class_profiles.get("D40"))

        # 1. Select severity tier (shallow, moderate, deep)
        tier_name = self._determine_severity_tier(damage_class, surface_area_sqm)

        # 2. Sample depth from configured distribution
        if "depth_override" in prof:
            override_params = prof["depth_override"]
            min_d = override_params["min_depth_cm"]
            max_d = override_params["max_depth_cm"]
            mean_d = override_params.get("mean_cm", (min_d + max_d) / 2.0)
            std_d = override_params.get("std_cm", (max_d - min_d) / 4.0)
            dist_type = override_params.get("distribution", "truncated_normal")
            sampled_depth = self._sample_truncated_normal(mean_d, std_d, min_d, max_d)
            tier_params = override_params
        else:
            tier_params = depth_tiers.get(tier_name, depth_tiers["moderate"])
            min_d = tier_params["min_depth_cm"]
            max_d = tier_params["max_depth_cm"]
            mean_d = tier_params.get("mean_cm", (min_d + max_d) / 2.0)
            std_d = tier_params.get("std_cm", (max_d - min_d) / 4.0)
            dist_type = tier_params.get("distribution", "truncated_normal")
            sampled_depth = self._sample_truncated_normal(mean_d, std_d, min_d, max_d)

        # 3. Apply noise & environmental attenuation
        noise_cfg = lidar_cfg.get("noise", DEFAULT_CONFIG["lidar"]["noise"])
        if rain_detected:
            env_cfg = noise_cfg.get("rain_wet", {"noise_std_cm": 0.42, "confidence_min": 0.45, "confidence_max": 0.78})
            noise = self.rng.gauss(0.0, env_cfg.get("noise_std_cm", 0.42))
            conf = self.rng.uniform(env_cfg.get("confidence_min", 0.45), env_cfg.get("confidence_max", 0.78))
            specular_scatter = True
        else:
            env_cfg = noise_cfg.get("dry_day", {"noise_std_cm": 0.10, "confidence_min": 0.90, "confidence_max": 0.99})
            noise = self.rng.gauss(0.0, env_cfg.get("noise_std_cm", 0.10))
            conf = self.rng.uniform(env_cfg.get("confidence_min", 0.90), env_cfg.get("confidence_max", 0.99))
            specular_scatter = False

        raw_depth_cm = round(max(0.0, self.baseline_chassis_cm + sampled_depth + noise), 2)
        effective_depth_cm = round(max(0.0, raw_depth_cm - self.baseline_chassis_cm), 2)

        return {
            "depth_cm": raw_depth_cm,
            "effective_depth_cm": effective_depth_cm,
            "baseline_chassis_cm": self.baseline_chassis_cm,
            "severity_tier": tier_name,
            "sensor_source": "synthetic",
            "generation_method": "controlled_distribution",
            "distribution_type": dist_type,
            "distribution_range_cm": [min_d, max_d],
            "confidence": round(conf, 3),
            "beam_wavelength_nm": lidar_cfg.get("wavelength_nm", 905),
            "specular_scatter_flag": specular_scatter,
            "surface_area_sqm": surface_area_sqm,
            "damage_class": damage_class
        }
