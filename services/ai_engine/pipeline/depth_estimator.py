class EnvironmentalSensorSwitching:
    def __init__(self, baseline_chassis_height_cm: float = 30.0):
        self.baseline_height = baseline_chassis_height_cm

    def evaluate_environment_and_measure(
        self, rain_detected: bool, mean_luminance: float, 
        lidar_depth_cm: float, sonar_depth_cm: float, 
        z_accel_g: float, surface_area_sqm: float
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

        # Parabolic/wedge volume approximation in liters (cm^3 / 1000)
        valid_area = max(0.0, float(surface_area_sqm))
        calculated_volume_liters = 0.5 * (valid_area * 10000.0) * depth_cm / 1000.0

        return {
            "detection_modality": detection_modality,
            "is_submerged": is_submerged,
            "estimated_depth_cm": round(depth_cm, 2),
            "calculated_volume_liters": round(calculated_volume_liters, 2),
            "dynamic_impact_confirmed": dynamic_impact
        }
