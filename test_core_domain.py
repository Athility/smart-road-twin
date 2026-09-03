import unittest
import numpy as np
from services.ai_engine.pipeline.depth_estimator import EnvironmentalSensorSwitching
from services.backend.app.services.mcdm_engine import MCDMPrioritizationEngine
from pydantic import ValidationError
import main

class TestEnvironmentalSensorSwitching(unittest.TestCase):
    def setUp(self):
        self.switcher = EnvironmentalSensorSwitching(baseline_chassis_height_cm=30.0)

    def test_rain_switches_to_acoustic_sonar(self):
        # Rain detected should switch modality to acoustic sonar and mark is_submerged=True
        res = self.switcher.evaluate_environment_and_measure(
            rain_detected=True,
            mean_luminance=85.0,
            lidar_depth_cm=40.0,
            sonar_depth_cm=44.0,
            z_accel_g=1.0,
            surface_area_sqm=0.5
        )
        self.assertEqual(res["detection_modality"], "sonar_submerged_acoustic")
        self.assertTrue(res["is_submerged"])
        self.assertAlmostEqual(res["estimated_depth_cm"], 14.0)

    def test_low_luminance_switches_to_acoustic_sonar(self):
        # mean_luminance < 40.0 should switch to acoustic sonar even without rain
        res = self.switcher.evaluate_environment_and_measure(
            rain_detected=False,
            mean_luminance=25.0,
            lidar_depth_cm=40.0,
            sonar_depth_cm=42.5,
            z_accel_g=1.1,
            surface_area_sqm=0.4
        )
        self.assertEqual(res["detection_modality"], "sonar_submerged_acoustic")
        self.assertFalse(res["is_submerged"], "Dry night should not be falsely marked as water-submerged")
        self.assertAlmostEqual(res["estimated_depth_cm"], 12.5)

    def test_dry_daylight_switches_to_optical_lidar(self):
        # Clear dry daylight should use optical LiDAR
        res = self.switcher.evaluate_environment_and_measure(
            rain_detected=False,
            mean_luminance=80.0,
            lidar_depth_cm=36.0,
            sonar_depth_cm=36.0,
            z_accel_g=1.2,
            surface_area_sqm=0.3
        )
        self.assertEqual(res["detection_modality"], "optical_lidar")
        self.assertFalse(res["is_submerged"])
        self.assertAlmostEqual(res["estimated_depth_cm"], 6.0)

    def test_depth_chassis_baseline_subtraction(self):
        # Baseline is 30.0 cm; reading of 45.8 cm means actual depth is 15.8 cm
        res = self.switcher.evaluate_environment_and_measure(
            rain_detected=False,
            mean_luminance=80.0,
            lidar_depth_cm=45.8,
            sonar_depth_cm=45.8,
            z_accel_g=1.0,
            surface_area_sqm=0.5
        )
        self.assertAlmostEqual(res["estimated_depth_cm"], 15.8)

    def test_depth_clamped_at_zero_when_below_baseline(self):
        # Sensor reading below baseline (e.g. speed bump / raised object at 25.0 cm)
        res = self.switcher.evaluate_environment_and_measure(
            rain_detected=False,
            mean_luminance=80.0,
            lidar_depth_cm=25.0,
            sonar_depth_cm=25.0,
            z_accel_g=1.0,
            surface_area_sqm=0.5
        )
        self.assertEqual(res["estimated_depth_cm"], 0.0)
        self.assertEqual(res["calculated_volume_liters"], 0.0)

    def test_volume_calculation_units(self):
        # Area = 0.5 m^2 (5000 cm^2), depth = 10 cm. Volume = 0.5 * 5000 * 10 / 1000 = 25.0 Liters
        res = self.switcher.evaluate_environment_and_measure(
            rain_detected=False,
            mean_luminance=80.0,
            lidar_depth_cm=40.0,
            sonar_depth_cm=40.0,
            z_accel_g=1.0,
            surface_area_sqm=0.5
        )
        self.assertAlmostEqual(res["calculated_volume_liters"], 25.0)

    def test_dynamic_impact_confirmed_across_modalities(self):
        # Optical LiDAR with z_accel_g = 2.4g (> 1.5g) MUST confirm dynamic physical impact
        res_dry = self.switcher.evaluate_environment_and_measure(
            rain_detected=False,
            mean_luminance=80.0,
            lidar_depth_cm=40.0,
            sonar_depth_cm=40.0,
            z_accel_g=2.4,
            surface_area_sqm=0.5
        )
        self.assertTrue(res_dry["dynamic_impact_confirmed"], "Dry impact > 1.5g must be confirmed")

        # Sonar acoustic with z_accel_g = 1.8g (> 1.5g) MUST confirm dynamic physical impact
        res_wet = self.switcher.evaluate_environment_and_measure(
            rain_detected=True,
            mean_luminance=20.0,
            lidar_depth_cm=40.0,
            sonar_depth_cm=40.0,
            z_accel_g=1.8,
            surface_area_sqm=0.5
        )
        self.assertTrue(res_wet["dynamic_impact_confirmed"], "Wet impact > 1.5g must be confirmed")

        # Sub-threshold z_accel_g = 1.3g (<= 1.5g) must NOT confirm impact
        res_sub = self.switcher.evaluate_environment_and_measure(
            rain_detected=False,
            mean_luminance=80.0,
            lidar_depth_cm=40.0,
            sonar_depth_cm=40.0,
            z_accel_g=1.3,
            surface_area_sqm=0.5
        )
        self.assertFalse(res_sub["dynamic_impact_confirmed"], "Sub-threshold impact must be False")


class TestMCDMPrioritizationEngine(unittest.TestCase):
    def setUp(self):
        self.engine = MCDMPrioritizationEngine()

    def test_criteria_weights_and_benefit_cost_flags(self):
        # Weights: Volume=0.35 (benefit), Traffic=0.30 (benefit), Distance=0.20 (cost), Speed=0.15 (benefit)
        np.testing.assert_allclose(self.engine.weights, [0.35, 0.30, 0.20, 0.15])
        self.assertAlmostEqual(float(self.engine.weights.sum()), 1.0)
        np.testing.assert_array_equal(self.engine.is_benefit, [True, True, False, True])

    def test_priority_thresholds(self):
        self.assertEqual(self.engine.classify_priority(0.85), "Critical")
        self.assertEqual(self.engine.classify_priority(0.75), "Critical")
        self.assertEqual(self.engine.classify_priority(0.749), "High")
        self.assertEqual(self.engine.classify_priority(0.50), "High")
        self.assertEqual(self.engine.classify_priority(0.499), "Medium")
        self.assertEqual(self.engine.classify_priority(0.25), "Medium")
        self.assertEqual(self.engine.classify_priority(0.249), "Low")
        self.assertEqual(self.engine.classify_priority(0.0), "Low")

    def test_zero_column_does_not_produce_nan(self):
        # If all volumes are 0.0 (zero-column sum), normalization must not produce NaN
        matrix = np.array([
            [0.0, 10000, 1.0, 40.0],
            [0.0, 25000, 0.5, 60.0]
        ])
        scores = self.engine.compute_topsis(matrix)
        self.assertEqual(len(scores), 2)
        self.assertFalse(np.isnan(scores).any(), "Scores must not contain NaN")
        self.assertTrue((scores >= 0.0).all() and (scores <= 1.0).all())

    def test_identical_alternatives_return_neutral_score(self):
        # Identical candidates should receive a neutral relative tie (0.5), not collapse to 0.0 Low priority
        matrix = np.array([
            [50.0, 20000, 0.5, 60.0],
            [50.0, 20000, 0.5, 60.0]
        ])
        scores = self.engine.compute_topsis(matrix)
        self.assertEqual(len(scores), 2)
        np.testing.assert_allclose(scores, [0.5, 0.5])

    def test_topsis_ranks_severe_higher_than_minor(self):
        # Severe: large volume, high traffic, close to hospital (low dist), high speed
        # Minor: small volume, low traffic, far from hospital (high dist), low speed
        matrix = np.array([
            [60.0, 30000, 0.2, 70.0], # Severe
            [2.0, 2000, 8.0, 25.0]     # Minor
        ])
        scores = self.engine.compute_topsis(matrix)
        self.assertGreater(scores[0], scores[1])
        self.assertEqual(self.engine.classify_priority(scores[0]), "Critical")
        self.assertEqual(self.engine.classify_priority(scores[1]), "Low")


class TestAPIAndStore(unittest.TestCase):
    def test_pydantic_validation_guards(self):
        # Valid payload should pass
        valid = main.TelemetryPayload(
            vehicle_id="FLEET_01",
            latitude=19.0760,
            longitude=72.8777,
            rain_detected=False,
            mean_luminance=50.0,
            lidar_depth_cm=35.0,
            sonar_depth_cm=35.0,
            z_accel_g=1.0,
            surface_area_sqm=0.3,
            traffic_pcu=10000,
            dist_hospital_km=1.2,
            speed_kmh=45.0
        )
        self.assertEqual(valid.vehicle_id, "FLEET_01")

        # Latitude > 90 must fail
        with self.assertRaises(ValidationError):
            main.TelemetryPayload(
                vehicle_id="FLEET_01", latitude=95.0, longitude=72.0,
                rain_detected=False, mean_luminance=50.0, lidar_depth_cm=35.0,
                sonar_depth_cm=35.0, z_accel_g=1.0, surface_area_sqm=0.3,
                traffic_pcu=10000, dist_hospital_km=1.2, speed_kmh=45.0
            )

        # Negative area must fail
        with self.assertRaises(ValidationError):
            main.TelemetryPayload(
                vehicle_id="FLEET_01", latitude=19.0, longitude=72.0,
                rain_detected=False, mean_luminance=50.0, lidar_depth_cm=35.0,
                sonar_depth_cm=35.0, z_accel_g=1.0, surface_area_sqm=-0.5,
                traffic_pcu=10000, dist_hospital_km=1.2, speed_kmh=45.0
            )

    def test_summary_and_weather_state_updates(self):
        summary = main.calculate_summary()
        self.assertIn("total_defects", summary)
        self.assertIn("weather_state", summary)
        self.assertIn("priority_counts", summary)
        self.assertGreaterEqual(summary["total_defects"], 4)

if __name__ == "__main__":
    unittest.main()
