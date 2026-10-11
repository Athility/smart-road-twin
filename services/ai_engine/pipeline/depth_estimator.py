import math
from typing import Optional, List
from services.data_generation.consistency_model import (
    PhysicalConsistencyModel,
    normalize_damage_class_code
)


class EnvironmentalSensorSwitching:
    def __init__(self, baseline_chassis_height_cm: float = 30.0):
        self.baseline_height = baseline_chassis_height_cm

    def evaluate_environment_and_measure(
        self,
        rain_detected: bool,
        mean_luminance: float,
        lidar_depth_cm: float,
        sonar_depth_cm: float,
        z_accel_g: float,
        surface_area_sqm: float,
        damage_class: str = "D40",
        bbox: Optional[List[float]] = None
    ) -> dict:
        # Modality switching: rain or low luminance (< 40.0 lux) triggers acoustic sonar
        use_sonar = rain_detected or (mean_luminance < 40.0)
        is_submerged = bool(rain_detected)

        if use_sonar:
            detection_modality = "sonar_submerged_acoustic"
            raw_depth = max(0.0, float(sonar_depth_cm))
            depth_cm = max(0.0, raw_depth - self.baseline_height)
        else:
            detection_modality = "optical_lidar"
            raw_depth = max(0.0, float(lidar_depth_cm))
            depth_cm = max(0.0, raw_depth - self.baseline_height)

        # Dynamic physical impact is confirmed whenever z_accel_g > 1.5g
        dynamic_impact = bool(z_accel_g > 1.5)

        valid_area = max(0.0, float(surface_area_sqm))
        cls_code = normalize_damage_class_code(damage_class)

        if cls_code == "D40":
            # Parabolic/wedge cavity volume approximation in liters (cm^3 / 1000) ONLY for D40 potholes
            calculated_volume_liters: Optional[float] = round(
                0.5 * (valid_area * 10000.0) * depth_cm / 1000.0, 2
            )
            crack_severity_index: Optional[float] = None
        else:
            calculated_volume_liters = None
            crack_severity_index = PhysicalConsistencyModel.compute_crack_severity_index(
                damage_class=cls_code,
                surface_area_sqm=valid_area,
                depth_cm=depth_cm,
                bbox=bbox
            )

        defect_severity_magnitude = PhysicalConsistencyModel.compute_defect_severity_magnitude(
            damage_class=cls_code,
            volume_liters=calculated_volume_liters,
            crack_severity_index=crack_severity_index
        )

        return {
            "detection_modality": detection_modality,
            "is_submerged": is_submerged,
            "estimated_depth_cm": round(depth_cm, 2),
            "calculated_volume_liters": calculated_volume_liters,
            "crack_severity_index": crack_severity_index,
            "defect_severity_magnitude": defect_severity_magnitude,
            "dynamic_impact_confirmed": dynamic_impact
        }

