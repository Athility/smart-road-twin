import unittest
import numpy as np
from services.ai_engine.pipeline.depth_estimator import EnvironmentalSensorSwitching
from services.backend.app.services.mcdm_engine import MCDMPrioritizationEngine
from pydantic import ValidationError
from pathlib import Path
import json
import main

PROJECT_ROOT = Path(__file__).resolve().parent

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
        # Current project weights: Volume=0.35 (benefit), Traffic=0.25 (benefit), Hospital Distance=0.25 (cost), Speed=0.15 (benefit)
        np.testing.assert_allclose(self.engine.weights, [0.35, 0.25, 0.25, 0.15])
        self.assertAlmostEqual(float(self.engine.weights.sum()), 1.0)
        np.testing.assert_array_equal(self.engine.is_benefit, [True, True, False, True])

    def test_hospital_distance_cost_direction_increases_urgency(self):
        """
        Verify that smaller hospital distance increases urgency when all other
        variables (volume, traffic PCU, vehicle speed) are held strictly constant.
        Criterion 2 is a COST criterion (is_benefit=False): min(dist) -> ideal_best.
        """
        # Two identical defects (Volume=25L, Traffic=15000 PCU, Speed=40 km/h)
        # Defect 1: Near hospital (dist = 0.5 km) -> should have higher urgency/TOPSIS score
        # Defect 2: Far from hospital (dist = 8.0 km) -> should have lower urgency/TOPSIS score
        matrix = np.array([
            [25.0, 15000.0, 0.5, 40.0],  # Close to hospital
            [25.0, 15000.0, 8.0, 40.0]   # Distant from hospital
        ])
        scores = self.engine.compute_topsis(matrix)
        self.assertEqual(len(scores), 2)
        self.assertGreater(
            scores[0], scores[1],
            f"Defect at 0.5 km ({scores[0]:.4f}) must have higher urgency than defect at 8.0 km ({scores[1]:.4f})"
        )

    def test_all_criteria_mathematical_directions(self):
        """
        Systematic verification of all 4 criteria directions:
        - Volume: benefit (larger volume -> higher score)
        - Traffic: benefit (higher PCU -> higher score)
        - Hospital dist: cost (smaller dist -> higher score)
        - Speed: benefit (higher speed -> higher score)
        """
        # Volume test: 50L vs 10L (others: 10000, 2.0, 40.0)
        mat_vol = np.array([[50.0, 10000.0, 2.0, 40.0], [10.0, 10000.0, 2.0, 40.0]])
        sc_vol = self.engine.compute_topsis(mat_vol)
        self.assertGreater(sc_vol[0], sc_vol[1], "Larger volume must produce higher TOPSIS score")

        # Traffic test: 30000 PCU vs 5000 PCU (others: 20L, 2.0, 40.0)
        mat_traf = np.array([[20.0, 30000.0, 2.0, 40.0], [20.0, 5000.0, 2.0, 40.0]])
        sc_traf = self.engine.compute_topsis(mat_traf)
        self.assertGreater(sc_traf[0], sc_traf[1], "Higher traffic PCU must produce higher TOPSIS score")

        # Hospital dist test: 0.3 km vs 7.0 km (others: 20L, 10000, 40.0)
        mat_hosp = np.array([[20.0, 10000.0, 0.3, 40.0], [20.0, 10000.0, 7.0, 40.0]])
        sc_hosp = self.engine.compute_topsis(mat_hosp)
        self.assertGreater(sc_hosp[0], sc_hosp[1], "Smaller hospital distance must produce higher TOPSIS score")

        # Speed test: 60 km/h vs 20 km/h (others: 20L, 10000, 2.0)
        mat_spd = np.array([[20.0, 10000.0, 2.0, 60.0], [20.0, 10000.0, 2.0, 20.0]])
        sc_spd = self.engine.compute_topsis(mat_spd)
        self.assertGreater(sc_spd[0], sc_spd[1], "Higher speed must produce higher TOPSIS score")

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


class TestSmartRoadMultimodalDataset(unittest.TestCase):
    def setUp(self):
        from services.dataset.srmd_builder import SRMDRecordBuilder, srmd_to_telemetry_payload
        self.builder = SRMDRecordBuilder(seed=123)
        self.srmd_to_payload = srmd_to_telemetry_payload

    def test_srmd_record_schema_conformance(self):
        record = self.builder.synthesize_record(
            image_id="India_000045",
            filename="India_000045.jpg",
            damage_class="D40",
            bbox=[100.0, 200.0, 350.0, 450.0],
            image_dims={"width": 720, "height": 720, "depth": 3},
            annotation_idx=0
        )
        # Verify required root keys
        required_keys = [
            "event_id", "source", "visual", "location", "lidar", "sonar",
            "accelerometer", "traffic", "infrastructure", "environment",
            "vehicle", "provenance"
        ]
        for k in required_keys:
            self.assertIn(k, record, f"Missing required root key: {k}")

        # Verify source metadata
        self.assertEqual(record["source"]["dataset"], "RDD2022")
        self.assertEqual(record["source"]["country"], "India")
        self.assertEqual(record["source"]["annotation_class"], "D40")

        # Verify strict provenance declaration
        self.assertEqual(record["provenance"]["visual_source"], "RDD2022")
        self.assertEqual(record["provenance"]["sensor_source"], "synthetic")
        self.assertEqual(record["provenance"]["context_source"], "synthetic")
        self.assertEqual(record["provenance"]["synthesis_seed"], 123)

        # Verify sensor sources marked synthetic
        self.assertEqual(record["lidar"]["source"], "synthetic")
        self.assertEqual(record["sonar"]["source"], "synthetic")
        self.assertEqual(record["accelerometer"]["source"], "synthetic")
        self.assertEqual(record["traffic"]["source"], "synthetic")
        self.assertEqual(record["infrastructure"]["source"], "synthetic")

    def test_deterministic_sensor_synthesis(self):
        from services.dataset.srmd_builder import SRMDRecordBuilder
        b1 = SRMDRecordBuilder(seed=999)
        b2 = SRMDRecordBuilder(seed=999)

        r1 = b1.synthesize_record("India_000045", "India_000045.jpg", "D40", [100, 200, 300, 400], {"width": 720, "height": 720})
        r2 = b2.synthesize_record("India_000045", "India_000045.jpg", "D40", [100, 200, 300, 400], {"width": 720, "height": 720})

        self.assertEqual(r1["lidar"]["depth_cm"], r2["lidar"]["depth_cm"])
        self.assertEqual(r1["accelerometer"]["z_accel_g"], r2["accelerometer"]["z_accel_g"])
        self.assertEqual(r1["environment"]["luminance_lux"], r2["environment"]["luminance_lux"])
        self.assertEqual(r1["location"]["latitude"], r2["location"]["latitude"])

    def test_physical_class_correlation_and_impact(self):
        # D40 Pothole should have deep void and dynamic physical impact > 1.5g
        pothole = self.builder.synthesize_record(
            "India_000045", "India_000045.jpg", "D40", [100, 200, 400, 500], {"width": 720, "height": 720}
        )
        self.assertGreater(pothole["lidar"]["effective_depth_cm"], 5.0)
        self.assertGreater(pothole["accelerometer"]["z_accel_g"], 1.5)
        self.assertTrue(pothole["accelerometer"]["dynamic_impact_triggered"])

        # D00 Longitudinal crack should have shallow depth and sub-threshold impact <= 1.5g
        crack = self.builder.synthesize_record(
            "India_000155", "India_000155.jpg", "D00", [250, 100, 290, 600], {"width": 720, "height": 720}
        )
        self.assertLess(crack["lidar"]["effective_depth_cm"], 4.0)
        self.assertLessEqual(crack["accelerometer"]["z_accel_g"], 1.5)
        self.assertFalse(crack["accelerometer"]["dynamic_impact_triggered"])

    def test_srmd_to_telemetry_payload_conversion(self):
        record = self.builder.synthesize_record(
            "India_000045", "India_000045.jpg", "D40", [120, 240, 380, 480], {"width": 720, "height": 720}
        )
        payload_dict = self.srmd_to_payload(record)

        # Validate that the converted payload satisfies Pydantic TelemetryPayload validation guards
        validated = main.TelemetryPayload(**payload_dict)
        self.assertEqual(validated.vehicle_id, record["vehicle"]["vehicle_id"])
        self.assertAlmostEqual(validated.latitude, record["location"]["latitude"])
        self.assertAlmostEqual(validated.lidar_depth_cm, record["lidar"]["depth_cm"])
        self.assertEqual(validated.rain_detected, record["environment"]["rain_detected"])


class TestSensorAugmentationEngine(unittest.TestCase):
    def setUp(self):
        from services.data_generation import (
            LiDARModel, SonarModel, AccelerometerModel,
            TrafficModel, InfrastructureModel, SensorAugmentationEngine,
            LocationModel
        )
        self.lidar = LiDARModel(seed=42)
        self.sonar = SonarModel(seed=42)
        self.accel = AccelerometerModel(seed=42)
        self.traffic = TrafficModel(seed=42)
        self.infra = InfrastructureModel(seed=42)
        self.location = LocationModel(seed=2026)
        self.engine = SensorAugmentationEngine(seed=42)

    def test_lidar_physics_and_optical_degradation(self):
        # Dry daylight gives high confidence (> 0.88)
        dry = self.lidar.simulate("D40", surface_area_sqm=0.8, rain_detected=False)
        self.assertGreaterEqual(dry["confidence"], 0.85)
        self.assertGreater(dry["effective_depth_cm"], 3.0)
        self.assertEqual(dry["sensor_source"], "synthetic")
        self.assertEqual(dry["generation_method"], "controlled_distribution")
        self.assertIn(dry["severity_tier"], ["shallow", "moderate", "deep"])

        # Wet/rain degrades optical confidence
        wet = self.lidar.simulate("D40", surface_area_sqm=0.8, rain_detected=True)
        self.assertTrue(wet["specular_scatter_flag"])
        self.assertLess(wet["confidence"], dry["confidence"])

    def test_lidar_controlled_distribution_area_correlation(self):
        # Large potholes must exhibit higher probability of deep voids than small potholes
        small_depths = []
        small_deep_count = 0
        for i in range(50):
            res = self.lidar.simulate("D40", surface_area_sqm=0.10)
            small_depths.append(res["effective_depth_cm"])
            if res["severity_tier"] == "deep":
                small_deep_count += 1

        large_depths = []
        large_deep_count = 0
        for i in range(50):
            res = self.lidar.simulate("D40", surface_area_sqm=1.80)
            large_depths.append(res["effective_depth_cm"])
            if res["severity_tier"] == "deep":
                large_deep_count += 1

        mean_small = sum(small_depths) / len(small_depths)
        mean_large = sum(large_depths) / len(large_depths)
        self.assertGreater(mean_large, mean_small, "Larger potholes must have higher mean depth")
        self.assertGreater(large_deep_count, small_deep_count, "Larger potholes must have higher deep tier frequency")

    def test_lidar_configurable_profile_override(self):
        from services.data_generation import LiDARModel
        custom_cfg = {
            "lidar": {
                "baseline_chassis_cm": 35.0,
                "wavelength_nm": 1550,
                "depth_tiers": {
                    "shallow": {"min_depth_cm": 2.0, "max_depth_cm": 5.0, "mean_cm": 3.5, "std_cm": 0.5},
                    "moderate": {"min_depth_cm": 5.0, "max_depth_cm": 10.0, "mean_cm": 7.5, "std_cm": 1.0},
                    "deep": {"min_depth_cm": 10.0, "max_depth_cm": 25.0, "mean_cm": 18.0, "std_cm": 2.0}
                },
                "class_profiles": {
                    "D40": {
                        "area_thresholds_sqm": {"small": 0.2, "medium": 0.5, "large": 0.5},
                        "tier_probabilities": {
                            "large": {"shallow": 0.0, "moderate": 0.0, "deep": 1.0}
                        }
                    }
                }
            }
        }
        custom_lidar = LiDARModel(config_override=custom_cfg, seed=101)
        res = custom_lidar.simulate("D40", surface_area_sqm=1.0)
        self.assertEqual(res["baseline_chassis_cm"], 35.0)
        self.assertEqual(res["severity_tier"], "deep")
        self.assertGreaterEqual(res["effective_depth_cm"], 10.0)
        self.assertEqual(res["sensor_source"], "synthetic")
        self.assertEqual(res["generation_method"], "controlled_distribution")

    def test_sonar_acoustic_submerged_echo(self):
        # Wet road activates acoustic submerged echo
        sonar_wet = self.sonar.simulate("D40", true_defect_depth_cm=12.0, rain_detected=True)
        self.assertTrue(sonar_wet["is_submerged_echo"])
        self.assertGreater(sonar_wet["effective_depth_cm"], 8.0)
        self.assertAlmostEqual(sonar_wet["baseline_chassis_cm"], 30.0)
        self.assertEqual(sonar_wet["sensor_source"], "synthetic")
        self.assertEqual(sonar_wet["generation_method"], "controlled_distribution")

    def test_sonar_measurement_difference_and_formula(self):
        # Sonar depth = true_depth + sensor_error
        # Sonar should NOT simply equal LiDAR
        true_depth = 10.0
        sonar_res = self.sonar.simulate("D40", true_defect_depth_cm=true_depth, rain_detected=False, mean_luminance=85.0)
        lidar_res = self.lidar.simulate("D40", surface_area_sqm=0.5, rain_detected=False)

        # Confirm formula: effective_depth_cm == true_depth + sensor_error
        expected_depth = round(true_depth + sonar_res["sensor_error_cm"], 2)
        self.assertAlmostEqual(sonar_res["effective_depth_cm"], expected_depth, places=2)
        self.assertEqual(sonar_res["true_depth_reference_cm"], true_depth)
        self.assertIn("sensor_error_cm", sonar_res)

        # Confirm sonar depth does NOT simply equal LiDAR depth
        self.assertNotEqual(sonar_res["effective_depth_cm"], lidar_res["effective_depth_cm"])
        self.assertNotEqual(sonar_res["depth_cm"], lidar_res["depth_cm"])

    def test_sonar_environmental_switching_demonstration(self):
        # DRY + DAYLIGHT -> LiDAR
        dry_day = self.sonar.evaluate_switching_recommendation(rain_detected=False, mean_luminance=85.0)
        self.assertFalse(dry_day["use_sonar"])
        self.assertEqual(dry_day["active_modality"], "optical_lidar")
        self.assertIn("Clear Dry Daylight", dry_day["switching_reason"])

        # RAIN / SUBMERGED -> Sonar
        rainy = self.sonar.evaluate_switching_recommendation(rain_detected=True, mean_luminance=70.0)
        self.assertTrue(rainy["use_sonar"])
        self.assertTrue(rainy["is_submerged"])
        self.assertEqual(rainy["active_modality"], "sonar_submerged_acoustic")
        self.assertIn("Precipitation", rainy["switching_reason"])

        # LOW LIGHT (< 40 lux) -> Sonar
        dark = self.sonar.evaluate_switching_recommendation(rain_detected=False, mean_luminance=20.0)
        self.assertTrue(dark["use_sonar"])
        self.assertFalse(dark["is_submerged"])
        self.assertEqual(dark["active_modality"], "sonar_submerged_acoustic")
        self.assertIn("Low Luminance", dark["switching_reason"])

    def test_sonar_configurable_override(self):
        from services.data_generation import SonarModel
        custom_cfg = {
            "sonar": {
                "baseline_chassis_cm": 28.0,
                "transducer_frequency_khz": 50,
                "noise": {
                    "dry_day": {
                        "noise_std_cm": 0.10,
                        "confidence_min": 0.90,
                        "confidence_max": 0.95
                    }
                },
                "switching_thresholds": {
                    "luminance_threshold_lux": 50.0,
                    "rain_detected_triggers_sonar": True
                }
            }
        }
        custom_sonar = SonarModel(config_override=custom_cfg, seed=77)
        res = custom_sonar.simulate("D40", true_defect_depth_cm=8.0, rain_detected=False, mean_luminance=60.0)
        self.assertEqual(res["baseline_chassis_cm"], 28.0)
        self.assertEqual(res["transducer_frequency_khz"], 50)
        self.assertEqual(res["sensor_source"], "synthetic")
        self.assertEqual(res["generation_method"], "controlled_distribution")
        # With threshold 50.0, 45 lux should trigger sonar
        sw = custom_sonar.evaluate_switching_recommendation(rain_detected=False, mean_luminance=45.0)
        self.assertTrue(sw["use_sonar"])
        self.assertEqual(sw["active_modality"], "sonar_submerged_acoustic")

    def test_accelerometer_quarter_car_shock(self):
        # Deep pothole at standard speed triggers > 1.5g dynamic impact
        pothole_shock = self.accel.simulate("D40", defect_depth_cm=10.0, speed_kmh=45.0)
        self.assertGreater(pothole_shock["z_accel_g"], 1.50)
        self.assertTrue(pothole_shock["dynamic_impact_triggered"])
        self.assertEqual(pothole_shock["impact_regime"], "confirmed_impact")
        self.assertEqual(pothole_shock["sensor_source"], "synthetic")
        self.assertEqual(pothole_shock["generation_method"], "controlled_distribution")

        # Shallow crack yields sub-threshold shock <= 1.35g (no significant impact)
        crack_shock = self.accel.simulate("D00", defect_depth_cm=1.2, speed_kmh=45.0)
        self.assertLessEqual(crack_shock["z_accel_g"], 1.35)
        self.assertFalse(crack_shock["dynamic_impact_triggered"])
        self.assertEqual(crack_shock["impact_regime"], "no_impact")

    def test_accelerometer_three_impact_regimes(self):
        # 1. No significant impact: minor crack / superficial depression (z <= 1.35g)
        no_imp = self.accel.simulate("D00", defect_depth_cm=1.0, speed_kmh=35.0)
        self.assertLessEqual(no_imp["z_accel_g"], 1.35)
        self.assertEqual(no_imp["impact_regime"], "no_impact")
        self.assertEqual(no_imp["impact_category"], "No Significant Impact")
        self.assertFalse(no_imp["dynamic_impact_triggered"])

        # 2. Borderline impact: moderate pothole or severe alligator fatigue (1.35g < z <= 1.50g)
        self.accel.reseed(42)
        borderline_imp = self.accel.simulate("D40", defect_depth_cm=5.2, speed_kmh=45.0)
        self.assertGreater(borderline_imp["z_accel_g"], 1.35)
        self.assertLessEqual(borderline_imp["z_accel_g"], 1.50)
        self.assertEqual(borderline_imp["impact_regime"], "borderline")
        self.assertEqual(borderline_imp["impact_category"], "Borderline Impact")
        self.assertFalse(borderline_imp["dynamic_impact_triggered"])

        # 3. Confirmed dynamic impact: severe deep void (z > 1.50g)
        self.accel.reseed(42)
        confirmed_imp = self.accel.simulate("D40", defect_depth_cm=13.0, speed_kmh=45.0)
        self.assertGreater(confirmed_imp["z_accel_g"], 1.50)
        self.assertEqual(confirmed_imp["impact_regime"], "confirmed_impact")
        self.assertEqual(confirmed_imp["impact_category"], "Confirmed Dynamic Impact")
        self.assertTrue(confirmed_imp["dynamic_impact_triggered"])

    def test_accelerometer_1_5g_threshold_preservation(self):
        # Preserves existing project threshold z_accel_g > 1.50
        classification_under = self.accel.classify_impact(1.49)
        self.assertEqual(classification_under["regime"], "borderline")

        classification_at = self.accel.classify_impact(1.50)
        self.assertEqual(classification_at["regime"], "borderline")

        classification_over = self.accel.classify_impact(1.51)
        self.assertEqual(classification_over["regime"], "confirmed_impact")

    def test_accelerometer_cross_modal_assessment_rationale(self):
        # Evaluates visual detection vs physical impact sensing cross-validation
        no_imp_class = self.accel.classify_impact(1.10)
        self.assertIn("negligible", no_imp_class["cross_modal_assessment"])

        borderline_class = self.accel.classify_impact(1.42)
        self.assertIn("Incipient structural hazard", borderline_class["cross_modal_assessment"])

        confirmed_class = self.accel.classify_impact(2.10)
        self.assertIn("incontrovertible multi-modal evidence", confirmed_class["cross_modal_assessment"])

    def test_accelerometer_dataset_contains_all_three_categories(self):
        # The generated SRMD dataset must contain examples of all three regimes
        from pathlib import Path
        import json
        records_dir = Path("data/processed/srmd/records")
        if records_dir.exists():
            records = [json.loads(p.read_text(encoding="utf-8")) for p in records_dir.glob("*.json")]
            regimes = {r["accelerometer"]["impact_regime"] for r in records if "accelerometer" in r}
            self.assertIn("no_impact", regimes, "Dataset must include no_impact examples")
            self.assertIn("borderline", regimes, "Dataset must include borderline examples")
            self.assertIn("confirmed_impact", regimes, "Dataset must include confirmed_impact examples")

    def test_accelerometer_configurable_override(self):
        from services.data_generation import AccelerometerModel
        custom_cfg = {
            "accelerometer": {
                "thresholds": {
                    "confirmed_impact_g": 1.60,
                    "borderline_impact_min_g": 1.40,
                    "borderline_impact_max_g": 1.60
                },
                "class_profiles": {
                    "D40": {
                        "alpha": 1.10,
                        "speed_exp": 0.85,
                        "depth_exp": 1.25,
                        "noise_std": 0.02,
                        "rms_base_g": 0.20
                    }
                }
            }
        }
        custom_accel = AccelerometerModel(config_override=custom_cfg, seed=55)
        self.assertEqual(custom_accel.confirmed_threshold_g, 1.60)
        self.assertEqual(custom_accel.borderline_min_g, 1.40)
        # 1.55g should now be classified as borderline under custom 1.60 threshold
        cl = custom_accel.classify_impact(1.55)
        self.assertEqual(cl["regime"], "borderline")

    def test_traffic_flow_greenshields_relation(self):
        # Expressways should generate realistic urban PCU and speed
        res = self.traffic.simulate("HIGH_TRAFFIC_CORRIDOR")
        self.assertGreaterEqual(res["traffic_pcu"], 20000)
        self.assertIn(res["level_of_service"], ["A", "B", "C", "D", "E", "F"])
        self.assertGreater(res["operating_speed_kmh"], 15.0)
        self.assertEqual(res["traffic_source"], "synthetic")
        self.assertEqual(res["road_class"], "HIGH_TRAFFIC_CORRIDOR")

    def test_traffic_four_road_classes_pcu_ranges(self):
        # Test all 4 required profiles: LOCAL, COLLECTOR, ARTERIAL, HIGH_TRAFFIC_CORRIDOR
        local = self.traffic.simulate(road_class="LOCAL")
        collector = self.traffic.simulate(road_class="COLLECTOR")
        arterial = self.traffic.simulate(road_class="ARTERIAL")
        corridor = self.traffic.simulate(road_class="HIGH_TRAFFIC_CORRIDOR")

        # Range assertions based on sensor_profiles.yaml
        self.assertTrue(1500 <= local["traffic_pcu"] <= 6000, f"LOCAL PCU out of range: {local['traffic_pcu']}")
        self.assertTrue(6000 <= collector["traffic_pcu"] <= 14000, f"COLLECTOR PCU out of range: {collector['traffic_pcu']}")
        self.assertTrue(14000 <= arterial["traffic_pcu"] <= 24000, f"ARTERIAL PCU out of range: {arterial['traffic_pcu']}")
        self.assertTrue(24000 <= corridor["traffic_pcu"] <= 38000, f"HIGH_TRAFFIC_CORRIDOR PCU out of range: {corridor['traffic_pcu']}")

        # Relative hierarchy
        self.assertLess(local["traffic_pcu"], collector["traffic_pcu"])
        self.assertLess(collector["traffic_pcu"], arterial["traffic_pcu"])
        self.assertLess(arterial["traffic_pcu"], corridor["traffic_pcu"])

        # Check provenance
        for r in [local, collector, arterial, corridor]:
            self.assertEqual(r["traffic_source"], "synthetic")
            self.assertEqual(r["sensor_source"], "synthetic")
            self.assertEqual(r["generation_method"], "controlled_distribution")

    def test_traffic_location_correlation_and_inference(self):
        # Traffic is NOT generated completely independently of location
        hwy = self.traffic.simulate(road_segment_id="WEH-MUM-SEC-01", road_name="Western Express Highway")
        self.assertEqual(hwy["road_class"], "HIGH_TRAFFIC_CORRIDOR")
        self.assertGreaterEqual(hwy["traffic_pcu"], 24000)

        art = self.traffic.simulate(road_segment_id="SVR-MUM-02", road_name="Swami Vivekananda Road Arterial")
        self.assertEqual(art["road_class"], "ARTERIAL")

        col = self.traffic.simulate(road_segment_id="LNK-MUM-03", road_name="Linking Road Commercial Collector")
        self.assertEqual(col["road_class"], "COLLECTOR")

        loc = self.traffic.simulate(road_segment_id="RES-MUM-04", road_name="Pali Hill Residential Access Lane")
        self.assertEqual(loc["road_class"], "LOCAL")
        self.assertLessEqual(loc["traffic_pcu"], 6000)

    def test_traffic_dataset_coverage_and_topsis_compatibility(self):
        # Verify that all 4 road classes are present in the generated SRMD records
        from pathlib import Path
        import json
        import numpy as np
        from services.backend.app.services.mcdm_engine import MCDMPrioritizationEngine

        records_dir = Path("data/processed/srmd/records")
        if records_dir.exists():
            records = [json.loads(p.read_text(encoding="utf-8")) for p in records_dir.glob("*.json")]
            classes = {r["traffic"]["road_class"] for r in records if "traffic" in r}
            for expected in ["LOCAL", "COLLECTOR", "ARTERIAL", "HIGH_TRAFFIC_CORRIDOR"]:
                self.assertIn(expected, classes, f"Dataset must contain {expected} road_class")

            # Verify TOPSIS prioritizes high-traffic corridor over local street for identical defects
            engine = MCDMPrioritizationEngine()
            # Matrix: [volume_liters, traffic_pcu, dist_hospital_km, speed_kmh]
            # Defect 1: on LOCAL street (PCU: 3000)
            # Defect 2: on HIGH_TRAFFIC_CORRIDOR (PCU: 35000)
            # All other attributes identical (volume=15L, dist=1.0km, speed=40km/h)
            mat = np.array([
                [15.0, 3000.0, 1.0, 40.0],
                [15.0, 35000.0, 1.0, 40.0]
            ])
            scores = engine.compute_topsis(mat)
            self.assertGreater(scores[1], scores[0], "Corridor with 35000 PCU must rank higher in TOPSIS than 3000 PCU")

    def test_traffic_configurable_override(self):
        from services.data_generation import TrafficModel
        custom_cfg = {
            "traffic": {
                "traffic_source": "synthetic",
                "road_classes": {
                    "RURAL_HIGHWAY": {
                        "capacity_pcu": 12000,
                        "pcu_range": {"min_pcu": 2000, "max_pcu": 5000},
                        "free_flow_speed_kmh": 80.0,
                        "congestion_speed_kmh": 40.0
                    }
                }
            }
        }
        custom_tm = TrafficModel(config_override=custom_cfg, seed=88)
        res = custom_tm.simulate(road_class="RURAL_HIGHWAY")
        self.assertEqual(res["road_class"], "RURAL_HIGHWAY")
        self.assertTrue(2000 <= res["traffic_pcu"] <= 5000)
        self.assertEqual(res["traffic_source"], "synthetic")

    def test_infrastructure_haversine_and_emergency_corridor(self):
        # 1. Pipeline: Defect coordinate -> nearest hospital -> geodesic distance -> dist_hospital_km
        # Coordinate near Lilavati Hospital (lat: 19.0510, lon: 72.8290)
        res_lilavati = self.infra.simulate(latitude=19.0512, longitude=72.8291)
        self.assertEqual(res_lilavati["nearest_hospital_name"], "Lilavati Hospital & Research Centre")
        self.assertLess(res_lilavati["dist_hospital_km"], 0.20)
        self.assertIn("geodesic_distance_km", res_lilavati)
        self.assertTrue(res_lilavati["is_emergency_corridor"])

        # Coordinate near Bal Thackeray Trauma Center (lat: 19.1380, lon: 72.8590)
        res_thackeray = self.infra.simulate(latitude=19.1385, longitude=72.8595)
        self.assertEqual(res_thackeray["nearest_hospital_name"], "Bal Thackeray Trauma Care Municipal Hospital")
        self.assertLess(res_thackeray["dist_hospital_km"], 0.30)
        self.assertTrue(res_thackeray["is_emergency_corridor"])

        # 2. Strict Provenance verification
        self.assertEqual(res_lilavati["location_source"], "synthetic")
        self.assertEqual(res_lilavati["hospital_source"], "synthetic")
        self.assertEqual(res_lilavati["generation_method"], "gis_geodesic_nearest_neighbor")

        # 3. Non-emergency location (> 1.0 km threshold)
        res_remote = self.infra.simulate(latitude=19.2000, longitude=72.8300)
        self.assertGreater(res_remote["dist_hospital_km"], 1.0)
        self.assertFalse(res_remote["is_emergency_corridor"])

    def test_infrastructure_pluggable_gis_provider(self):
        from services.data_generation.infrastructure_model import InfrastructureModel, HospitalGISProvider
        # Mock OpenStreetMap / external GIS registry provider
        class MockOSMHospitalProvider(HospitalGISProvider):
            def get_hospitals(self):
                return [
                    {"name": "OSM AIIMS Trauma Annex", "lat": 19.0600, "lon": 72.8300, "tier": "Level-1", "district": "Central"},
                    {"name": "OSM District General", "lat": 19.1500, "lon": 72.8500, "tier": "Level-2", "district": "North"}
                ]

        osm_provider = MockOSMHospitalProvider()
        osm_infra = InfrastructureModel(gis_provider=osm_provider, seed=99)
        res = osm_infra.simulate(latitude=19.0605, longitude=72.8302)
        self.assertEqual(res["nearest_hospital_name"], "OSM AIIMS Trauma Annex")
        self.assertLess(res["dist_hospital_km"], 0.20)
        self.assertEqual(res["hospital_source"], "synthetic")

    def test_infrastructure_dataset_records_provenance(self):
        from pathlib import Path
        import json
        records_dir = Path("data/processed/srmd/records")
        if records_dir.exists():
            records = [json.loads(p.read_text(encoding="utf-8")) for p in records_dir.glob("*.json")]
            for r in records:
                self.assertIn("dist_hospital_km", r["infrastructure"])
                self.assertIn("geodesic_distance_km", r["infrastructure"])
                self.assertEqual(r["infrastructure"]["hospital_source"], "synthetic")
                self.assertEqual(r["location"]["location_source"], "synthetic")

    def test_master_sensor_augmentation_orchestrator(self):
        record = self.engine.augment_defect(
            image_id="India_000045",
            filename="India_000045.jpg",
            damage_class="D40",
            bbox=[150.0, 220.0, 410.0, 460.0],
            image_dims={"width": 720, "height": 720},
            annotation_idx=0
        )
        self.assertEqual(record["source"]["annotation_class"], "D40")
        self.assertEqual(record["visual"]["damage_class"], "pothole")
        self.assertEqual(record["provenance"]["sensor_source"], "synthetic")
        self.assertGreater(record["lidar"]["effective_depth_cm"], 0.5)
        self.assertIn(record["lidar"]["severity_tier"], ["shallow", "moderate", "deep"])
        self.assertEqual(record["lidar"]["sensor_source"], "synthetic")
        self.assertEqual(record["lidar"]["generation_method"], "controlled_distribution")

        # Severe large pothole exhibits deeper void and physical dynamic impact
        large_record = self.engine.augment_defect(
            image_id="India_000045",
            filename="India_000045.jpg",
            damage_class="D40",
            bbox=[50.0, 100.0, 680.0, 680.0],
            image_dims={"width": 720, "height": 720},
            annotation_idx=1
        )
        self.assertGreater(large_record["lidar"]["effective_depth_cm"], 4.5)
        self.assertGreater(large_record["accelerometer"]["z_accel_g"], 1.5)
        self.assertTrue(large_record["accelerometer"]["dynamic_impact_triggered"])

    def test_location_three_modes(self):
        from services.data_generation.location_model import (
            LocationModel, MODE_SOURCE_LOCATION, MODE_SYNTHETIC_LOCATION, MODE_EXTERNAL_GIS_LOCATION
        )
        # Mode 1: synthetic_location (default)
        synth_model = LocationModel(mode=MODE_SYNTHETIC_LOCATION, seed=2026)
        loc_synth = synth_model.generate_location_for_defect(route_index=0)
        self.assertEqual(loc_synth["location_mode"], "synthetic_location")
        self.assertEqual(loc_synth["location_source"], "synthetic")
        self.assertEqual(loc_synth["road_segment_id"], "WEH-MUM-SEC-01")
        self.assertEqual(loc_synth["road_name"], "Western Express Highway, Bandra-Kalanagar Junction")
        self.assertEqual(loc_synth["road_class"], "HIGH_TRAFFIC_CORRIDOR")
        self.assertEqual(loc_synth["synthesis_seed"], 2026)
        self.assertIn("Mumbai", loc_synth["demonstration_area"])
        self.assertAlmostEqual(loc_synth["latitude"], 19.0596, delta=0.01)
        self.assertAlmostEqual(loc_synth["longitude"], 72.8488, delta=0.01)

        # Mode 2: source_location with authentic GPS
        src_model = LocationModel(mode=MODE_SOURCE_LOCATION)
        loc_src = src_model.generate_location_for_defect(source_gps={"latitude": 19.1234, "longitude": 72.5678})
        self.assertEqual(loc_src["location_mode"], "source_location")
        self.assertEqual(loc_src["location_source"], "source_dataset")
        self.assertAlmostEqual(loc_src["latitude"], 19.1234)
        self.assertAlmostEqual(loc_src["longitude"], 72.5678)

        # Provenance guard: RDD2022 has no source GPS. Reject false attribution!
        with self.assertRaises(ValueError) as ctx:
            src_model.generate_location_for_defect(source_gps=None)
        self.assertIn("no genuine GPS coordinates exist", str(ctx.exception))

        # Mode 3: external_gis_location
        gis_model = LocationModel(mode=MODE_EXTERNAL_GIS_LOCATION)
        loc_gis = gis_model.generate_location_for_defect(route_index=1)
        self.assertEqual(loc_gis["location_mode"], "external_gis_location")
        self.assertEqual(loc_gis["location_source"], "external_gis_osm")
        self.assertEqual(loc_gis["snapping_status"], "snapped_to_centerline")
        self.assertEqual(loc_gis["road_segment_id"], "SVR-MUM-SEC-02")

    def test_location_deterministic_reproducibility(self):
        from services.data_generation.location_model import LocationModel, MODE_SYNTHETIC_LOCATION
        loc1 = LocationModel(mode=MODE_SYNTHETIC_LOCATION, seed=2026)
        loc2 = LocationModel(mode=MODE_SYNTHETIC_LOCATION, seed=2026)

        # Sequences must match identically across all waypoints
        for idx in range(10):
            res1 = loc1.generate_location_for_defect(route_index=idx)
            res2 = loc2.generate_location_for_defect(route_index=idx)
            self.assertEqual(res1["latitude"], res2["latitude"])
            self.assertEqual(res1["longitude"], res2["longitude"])
            self.assertEqual(res1["road_segment_id"], res2["road_segment_id"])
            self.assertEqual(res1["road_class"], res2["road_class"])

        # Reseeding reproduces the exact same trajectory
        loc1.reseed(2026)
        res_reseeded = loc1.generate_location_for_defect(route_index=0)
        loc2.reseed(2026)
        res_expected = loc2.generate_location_for_defect(route_index=0)
        self.assertEqual(res_reseeded["latitude"], res_expected["latitude"])
        self.assertEqual(res_reseeded["longitude"], res_expected["longitude"])

        # Different seed produces distinct micro-jittered coordinates
        loc_diff = LocationModel(mode=MODE_SYNTHETIC_LOCATION, seed=9999)
        res_diff = loc_diff.generate_location_for_defect(route_index=0)
        self.assertNotEqual(res_diff["latitude"], res1["latitude"])

    def test_location_srmd_dataset_records_provenance(self):
        from pathlib import Path
        import json
        records_dir = Path("data/processed/srmd/records")
        if records_dir.exists():
            records = [json.loads(p.read_text(encoding="utf-8")) for p in records_dir.glob("*.json")]
            self.assertGreater(len(records), 0)
            for r in records:
                loc = r["location"]
                self.assertEqual(loc["location_mode"], "synthetic_location")
                self.assertEqual(loc["location_source"], "synthetic")
                self.assertEqual(loc["synthesis_seed"], 2026)
                self.assertIn("Mumbai", loc["demonstration_area"])
                self.assertTrue(18.9 <= loc["latitude"] <= 19.3)
                self.assertTrue(72.7 <= loc["longitude"] <= 73.1)
                self.assertIn(loc["road_class"], ["LOCAL", "COLLECTOR", "ARTERIAL", "HIGH_TRAFFIC_CORRIDOR"])


class TestPhysicalConsistencyAndQualityValidation(unittest.TestCase):
    def setUp(self):
        from services.data_generation.consistency_model import PhysicalConsistencyModel
        self.consistency = PhysicalConsistencyModel(seed=2026)

    def test_latent_severity_monotonicity_and_derivation(self):
        # 1. Severity ordering by class: Longitudinal Crack < Transverse Crack < Alligator < Pothole
        s_d00 = self.consistency.derive_latent_severity("D00", surface_area_sqm=0.5)
        s_d10 = self.consistency.derive_latent_severity("D10", surface_area_sqm=0.5)
        s_d20 = self.consistency.derive_latent_severity("D20", surface_area_sqm=0.5)
        s_d40 = self.consistency.derive_latent_severity("D40", surface_area_sqm=0.5)

        self.assertLess(s_d00, s_d20)
        self.assertLess(s_d10, s_d40)
        self.assertLess(s_d20, s_d40)

        # 2. Larger surface area yields higher latent severity within same class
        s_small_pothole = self.consistency.derive_latent_severity("D40", surface_area_sqm=0.2)
        s_large_pothole = self.consistency.derive_latent_severity("D40", surface_area_sqm=2.2)
        self.assertGreater(s_large_pothole, s_small_pothole)

    def test_physical_consistency_depth_area_volume_relationship(self):
        # Volume > 0 <=> (Depth > 0 AND Area > 0)
        depth, vol, tier = self.consistency.derive_depth_and_volume(
            defect_severity=0.85,
            damage_class="D40",
            surface_area_sqm=1.5
        )
        self.assertGreater(depth, 0.0)
        self.assertGreater(vol, 0.0)
        self.assertEqual(tier, "deep")
        # Volume = 0.5 * (1.5 * 10000) * depth / 1000
        expected_vol = round(0.5 * (1.5 * 10000.0) * depth / 1000.0, 2)
        self.assertAlmostEqual(vol, expected_vol, places=2)

        # Invariant: If area is 0, volume must be 0
        depth_zero_area, vol_zero_area, _ = self.consistency.derive_depth_and_volume(
            defect_severity=0.85,
            damage_class="D40",
            surface_area_sqm=0.0
        )
        self.assertEqual(vol_zero_area, 0.0)

        # Monotonicity: Deeper pothole -> larger volume
        depth_small, vol_small, _ = self.consistency.derive_depth_and_volume(0.55, "D40", 1.0)
        depth_large, vol_large, _ = self.consistency.derive_depth_and_volume(0.95, "D40", 1.0)
        self.assertGreater(depth_large, depth_small)
        self.assertGreater(vol_large, vol_small)

    def test_physical_consistency_dynamic_impact_relationship(self):
        # Deeper pothole produces larger dynamic impact acceleration
        shallow_impact = self.consistency.derive_accelerometer_impact(
            defect_severity=0.20,
            defect_depth_cm=1.2,
            surface_area_sqm=0.3,
            vehicle_speed_kmh=45.0,
            damage_class="D00"
        )
        deep_impact = self.consistency.derive_accelerometer_impact(
            defect_severity=0.90,
            defect_depth_cm=14.0,
            surface_area_sqm=1.8,
            vehicle_speed_kmh=45.0,
            damage_class="D40"
        )

        self.assertGreater(deep_impact["z_accel_g"], shallow_impact["z_accel_g"])
        # Preserves 1.50g threshold
        self.assertTrue(deep_impact["dynamic_impact_triggered"])
        self.assertGreater(deep_impact["z_accel_g"], 1.50)
        self.assertFalse(shallow_impact["dynamic_impact_triggered"])
        self.assertLessEqual(shallow_impact["z_accel_g"], 1.50)

    def test_physical_consistency_modality_switching(self):
        # Rain -> Sonar preferred
        rain_eval = self.consistency.derive_sensor_measurements(
            defect_depth_cm=10.0,
            rain_detected=True,
            luminance_lux=80.0
        )
        self.assertEqual(rain_eval["preferred_modality"], "sonar_submerged_acoustic")
        self.assertTrue(rain_eval["is_submerged"])

        # Low light (< 40 lux) -> Sonar preferred even without rain
        night_eval = self.consistency.derive_sensor_measurements(
            defect_depth_cm=10.0,
            rain_detected=False,
            luminance_lux=25.0
        )
        self.assertEqual(night_eval["preferred_modality"], "sonar_submerged_acoustic")

        # Dry daylight -> LiDAR preferred
        dry_eval = self.consistency.derive_sensor_measurements(
            defect_depth_cm=10.0,
            rain_detected=False,
            luminance_lux=75.0
        )
        self.assertEqual(dry_eval["preferred_modality"], "optical_lidar")
        self.assertFalse(dry_eval["is_submerged"])

    def test_physical_consistency_infrastructure_risk(self):
        # Higher traffic + higher speed + larger defect -> higher infrastructure risk
        low_risk = self.consistency.derive_infrastructure_risk(
            defect_severity=0.20,
            defect_depth_cm=1.5,
            surface_area_sqm=0.2,
            traffic_pcu=3000,
            vehicle_speed_kmh=30.0,
            dist_hospital_km=3.5
        )
        high_risk = self.consistency.derive_infrastructure_risk(
            defect_severity=0.92,
            defect_depth_cm=15.0,
            surface_area_sqm=2.5,
            traffic_pcu=32000,
            vehicle_speed_kmh=60.0,
            dist_hospital_km=0.3
        )
        self.assertGreater(high_risk["infrastructure_risk_score"], low_risk["infrastructure_risk_score"])
        self.assertEqual(high_risk["priority_level"], "Critical")

    def test_generate_srmd_pipeline_and_multi_format_exports(self):
        from pathlib import Path
        from scripts.generate_srmd import generate_srmd
        test_out = Path("data/processed/srmd_test_export")
        manifest = generate_srmd(
            input_path=Path("data/processed/annotations_index.json"),
            output_dir=test_out,
            seed=2026,
            limit=4,
            class_filter=None
        )
        self.assertEqual(manifest["total_records"], 4)
        self.assertTrue((test_out / "srmd_records.json").exists())
        self.assertTrue((test_out / "srmd_dataset.jsonl").exists())
        self.assertTrue((test_out / "srmd_dataset.csv").exists())
        self.assertTrue((test_out / "srmd_dataset.parquet").exists())
        self.assertTrue((test_out / "srmd_manifest.json").exists())

        # Cleanup test export
        import shutil
        shutil.rmtree(test_out, ignore_errors=True)

    def test_validation_suite_passes_compliant_records(self):
        from pathlib import Path
        import json
        from scripts.validate_srmd import validate_dataset
        records_file = Path("data/processed/srmd/srmd_records.json")
        self.assertTrue(records_file.exists())
        with open(records_file, "r", encoding="utf-8") as f:
            records = json.load(f)
        report = validate_dataset(records)
        self.assertTrue(report.is_valid)
        self.assertEqual(report.failed_records, 0)
        self.assertEqual(len(report.errors), 0)

    def test_validation_suite_detects_all_rule_violations(self):
        from scripts.validate_srmd import validate_single_record
        from pathlib import Path
        import json
        records_file = Path("data/processed/srmd/srmd_records.json")
        with open(records_file, "r", encoding="utf-8") as f:
            base_record = json.load(f)[0]

        # 1. Invalid GPS
        bad_gps = json.loads(json.dumps(base_record))
        bad_gps["location"]["latitude"] = 195.0
        errs = validate_single_record(bad_gps)
        self.assertTrue(any("Rule 1" in e for e in errs))

        # 2. Negative Depth
        bad_depth = json.loads(json.dumps(base_record))
        bad_depth["lidar"]["effective_depth_cm"] = -5.0
        errs = validate_single_record(bad_depth)
        self.assertTrue(any("Rule 2" in e for e in errs))

        # 3. Inverted Bounding Box
        bad_bbox = json.loads(json.dumps(base_record))
        bad_bbox["visual"]["bbox"] = [400, 500, 100, 200]
        errs = validate_single_record(bad_bbox)
        self.assertTrue(any("Rule 9" in e for e in errs))

        # 4. Modality Mismatch: Rain detected but optical_lidar preferred
        bad_modality = json.loads(json.dumps(base_record))
        bad_modality["environment"]["rain_detected"] = True
        bad_modality["environment"]["preferred_modality"] = "optical_lidar"
        errs = validate_single_record(bad_modality)
        self.assertTrue(any("Rule 13" in e for e in errs))

        # 5. Dynamic impact threshold violation: z_accel_g=2.2g but triggered=False
        bad_impact = json.loads(json.dumps(base_record))
        bad_impact["accelerometer"]["z_accel_g"] = 2.2
        bad_impact["accelerometer"]["dynamic_impact_triggered"] = False
        errs = validate_single_record(bad_impact)
        self.assertTrue(any("Rule 15" in e for e in errs))

        # 6. Volume inconsistency: volume > 0 but effective_depth_cm = 0
        bad_vol = json.loads(json.dumps(base_record))
        bad_vol["visual"]["calculated_volume_liters"] = 15.0
        bad_vol["lidar"]["effective_depth_cm"] = 0.0
        errs = validate_single_record(bad_vol)
        self.assertTrue(any("Rule 16" in e for e in errs))


class TestComputerVisionModel(unittest.TestCase):
    """Unit tests for Phase 13: Computer Vision Model Abstraction & RDD2022 Detector."""

    def test_bounding_box_validation_and_yolo(self):
        from services.cv.schemas import BoundingBox
        box = BoundingBox(xmin=100.0, ymin=150.0, xmax=300.0, ymax=450.0)
        self.assertEqual(box.width, 200.0)
        self.assertEqual(box.height, 300.0)
        self.assertEqual(box.area, 60000.0)
        self.assertEqual(box.to_list(), [100.0, 150.0, 300.0, 450.0])

        yolo_box = box.to_yolo(img_w=1000, img_h=1000)
        self.assertAlmostEqual(yolo_box[0], 0.20, places=2)  # xc = 200/1000
        self.assertAlmostEqual(yolo_box[1], 0.30, places=2)  # yc = 300/1000
        self.assertAlmostEqual(yolo_box[2], 0.20, places=2)  # w = 200/1000
        self.assertAlmostEqual(yolo_box[3], 0.30, places=2)  # h = 300/1000

        # Inverted box raises validation error
        with self.assertRaises(ValueError):
            BoundingBox(xmin=300.0, ymin=150.0, xmax=100.0, ymax=450.0)

    def test_detector_adapter_switching(self):
        from services.cv.rdd2022_detector import RDD2022Detector
        from services.cv.detector import OpenCVContourDetectorAdapter, GroundTruthLookupAdapter

        detector = RDD2022Detector()
        self.assertIsNotNone(detector.adapter)

        # Switch to OpenCV contour adapter
        detector.set_adapter(OpenCVContourDetectorAdapter())
        self.assertEqual(detector.adapter.get_adapter_name(), "OpenCVContourDetectorAdapter")

        # Switch back to GT lookup
        detector.set_adapter(GroundTruthLookupAdapter())
        self.assertEqual(detector.adapter.get_adapter_name(), "GroundTruthLookupAdapter")

    def test_rdd2022_detector_inference(self):
        from services.cv.rdd2022_detector import RDD2022Detector
        from services.cv.schemas import DamageDetection

        detector = RDD2022Detector()
        sample_img = Path("data/samples/images/India_000045.jpg")
        if sample_img.exists():
            detections = detector.detect(sample_img)
            self.assertIsInstance(detections, list)
            self.assertGreater(len(detections), 0)
            det = detections[0]
            self.assertIsInstance(det, DamageDetection)
            self.assertEqual(det.class_id, "D40")
            self.assertEqual(det.class_label, "Pothole")
            self.assertGreaterEqual(det.confidence, 0.25)


class TestTrainingAndEvaluationPipeline(unittest.TestCase):
    """Unit tests for Phase 14: Training Pipeline, Metrics, and Model Evaluator."""

    def test_dataset_preparation_and_leakage_invariants(self):
        from ml.train.dataset import prepare_yolo_dataset
        import tempfile
        import shutil

        tmp_dir = Path(tempfile.mkdtemp())
        try:
            raw_xmls = Path("data/samples/annotations/xmls")
            raw_imgs = Path("data/samples/images")
            manifest = prepare_yolo_dataset(
                raw_images_dir=raw_imgs,
                raw_xmls_dir=raw_xmls,
                output_dir=tmp_dir,
                train_ratio=0.60,
                val_ratio=0.20,
                test_ratio=0.20,
                seed=2026
            )
            self.assertIn("splits", manifest)
            self.assertEqual(manifest["leakage_verification"]["train_test_overlap"], 0)
            self.assertEqual(manifest["leakage_verification"]["train_val_overlap"], 0)
            self.assertTrue((tmp_dir / "images" / "train").exists())
            self.assertTrue((tmp_dir / "labels" / "train").exists())
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_evaluation_metrics_and_evaluator(self):
        from ml.evaluate.metrics import compute_iou, evaluate_dataset_metrics
        from ml.evaluate.evaluator import DamageModelEvaluator

        # IoU tests
        boxA = [0.0, 0.0, 10.0, 10.0]
        boxB = [5.0, 0.0, 15.0, 10.0]
        self.assertAlmostEqual(compute_iou(boxA, boxB), 50.0 / 150.0, places=4)

        # Disjoint boxes
        boxC = [20.0, 20.0, 30.0, 30.0]
        self.assertEqual(compute_iou(boxA, boxC), 0.0)

        # Dataset metrics computation
        gts = {"img1": [{"class_id": "D40", "bbox": [0, 0, 10, 10]}]}
        preds = {"img1": [{"class_id": "D40", "confidence": 0.90, "bbox": [1, 1, 10, 10]}]}
        res = evaluate_dataset_metrics(gts, preds, iou_threshold=0.50)
        self.assertEqual(res["class_metrics"]["D40"]["precision"], 1.0)
        self.assertEqual(res["class_metrics"]["D40"]["recall"], 1.0)
        self.assertEqual(res["precision"], 0.25)  # Macro-average across 4 classes
        self.assertGreater(res["mAP50"], 0.0)

        # Evaluator over test split
        evaluator = DamageModelEvaluator()
        eval_res = evaluator.evaluate_split("test", limit=2)
        self.assertIn("precision", eval_res)
        self.assertIn("recall", eval_res)
        self.assertIn("mAP50", eval_res)
        self.assertIn("confusion_matrix", eval_res)


class TestMultimodalDataFusion(unittest.TestCase):
    """Unit tests for Phase 15: Multimodal Data Fusion & TOPSIS Prioritization."""

    def setUp(self):
        from services.fusion.fusion_engine import MultimodalDataFusionEngine
        self.engine = MultimodalDataFusionEngine(seed=2026)

    def test_end_to_end_fusion_single_image(self):
        from services.fusion.schemas import RoadDefectEvent
        sample_img = Path("data/samples/images/India_000045.jpg")
        if sample_img.exists():
            events = self.engine.fuse_image(sample_img)
            self.assertEqual(len(events), 1)
            ev = events[0]
            self.assertIsInstance(ev, RoadDefectEvent)
            self.assertEqual(ev.visual.damage_class, "D40")
            self.assertEqual(ev.visual.damage_label, "Pothole")
            self.assertGreater(ev.visual.surface_area_sqm, 0.0)
            self.assertGreater(ev.sensor.calculated_volume_liters, 0.0)
            self.assertIsNotNone(ev.topsis)
            self.assertGreater(ev.topsis.topsis_score, 0.0)
            self.assertIn(ev.topsis.priority_level, ["Critical", "High", "Medium", "Low"])

    def test_environmental_sensor_switching_in_fusion(self):
        sample_img = Path("data/samples/images/India_000045.jpg")
        if sample_img.exists():
            # Rain condition triggers acoustic sonar
            events_rain = self.engine.fuse_image(sample_img, rain_detected=True, mean_luminance=50.0)
            self.assertEqual(events_rain[0].sensor.active_modality, "sonar_submerged_acoustic")

            # Low illuminance (<40 lux) triggers acoustic sonar
            events_dark = self.engine.fuse_image(sample_img, rain_detected=False, mean_luminance=25.0)
            self.assertEqual(events_dark[0].sensor.active_modality, "sonar_submerged_acoustic")

            # Dry daylight (>=40 lux and no rain) triggers optical LiDAR
            events_dry = self.engine.fuse_image(sample_img, rain_detected=False, mean_luminance=80.0)
            self.assertEqual(events_dry[0].sensor.active_modality, "optical_lidar")

    def test_batch_fusion_and_topsis_relative_ranking(self):
        images_dir = Path("data/samples/images")
        if images_dir.exists():
            imgs = sorted(list(images_dir.glob("*.jpg")))[:3]
            events = self.engine.fuse_batch(imgs, rain_detected=False, mean_luminance=70.0)
            self.assertGreater(len(events), 1)

            # Ranks must be 1, 2, ... N
            ranks = [e.topsis.rank for e in events if e.topsis]
            self.assertEqual(ranks, list(range(1, len(events) + 1)))

            # Scores must be in monotonically descending order
            scores = [e.topsis.topsis_score for e in events if e.topsis]
            for i in range(len(scores) - 1):
                self.assertGreaterEqual(scores[i], scores[i + 1])

    def test_telemetry_dict_ingestion(self):
        from main import process_and_store_telemetry
        sample_img = Path("data/samples/images/India_000045.jpg")
        if sample_img.exists():
            events = self.engine.fuse_image(sample_img)
            self.assertGreater(len(events), 0)
            ev = events[0]

            telemetry_payload = ev.to_telemetry_dict()
            stored = process_and_store_telemetry(telemetry_payload)
            self.assertIn("id", stored)
            self.assertIn("topsis_score", stored)
            self.assertIn("priority_level", stored)
            self.assertIn("processed_telemetry", stored)
            self.assertTrue(stored["processed_telemetry"]["dynamic_impact_confirmed"])


class TestSensorDisagreementAnalysis(unittest.TestCase):
    """Unit tests for Phase 18: Multi-Sensor Disagreement Analysis & Evidence Fusion."""

    def setUp(self):
        from services.fusion.disagreement_analyzer import SensorDisagreementAnalyzer, DisagreementCategory
        self.analyzer = SensorDisagreementAnalyzer(impact_threshold_g=1.50)
        self.categories = DisagreementCategory

    def test_concordant_confirmed_evidence(self):
        # Scenario: Vision=0.94, LiDAR=24cm, Accelerometer=2.1g -> Concordant Confirmed
        summary = self.analyzer.analyze_evidence(
            damage_class="D40",
            visual_confidence=0.94,
            surface_area_sqm=1.8,
            effective_depth_cm=24.0,
            active_modality="optical_lidar",
            z_accel_g=2.10,
            rain_detected=False,
            mean_luminance=70.0
        )
        self.assertEqual(summary.disagreement_category, self.categories.CONCORDANT_CONFIRMED)
        self.assertTrue(summary.depth_evidence["void_confirmed"])
        self.assertTrue(summary.impact_evidence["impact_confirmed"])
        self.assertGreater(summary.multimodal_confidence_score, 0.85)

    def test_visually_detected_physically_weak(self):
        # Scenario: Vision=0.92, LiDAR=8cm, Accelerometer=1.1g -> Visually detected but physically weak
        summary = self.analyzer.analyze_evidence(
            damage_class="D40",
            visual_confidence=0.92,
            surface_area_sqm=0.6,
            effective_depth_cm=8.0,
            active_modality="optical_lidar",
            z_accel_g=1.10,
            rain_detected=False,
            mean_luminance=65.0
        )
        self.assertEqual(summary.disagreement_category, self.categories.VISUALLY_DETECTED_PHYSICALLY_WEAK)
        self.assertTrue(summary.depth_evidence["void_confirmed"])
        self.assertFalse(summary.impact_evidence["impact_confirmed"])
        self.assertGreaterEqual(summary.multimodal_confidence_score, 0.50)

    def test_superficial_visual_only(self):
        # Scenario: Vision=0.90, Depth=0.3cm, Accelerometer=1.0g -> Superficial visual only
        summary = self.analyzer.analyze_evidence(
            damage_class="D40",
            visual_confidence=0.90,
            surface_area_sqm=0.4,
            effective_depth_cm=0.3,
            active_modality="optical_lidar",
            z_accel_g=1.00,
            rain_detected=False,
            mean_luminance=60.0
        )
        self.assertEqual(summary.disagreement_category, self.categories.SUPERFICIAL_VISUAL_ONLY)
        self.assertFalse(summary.depth_evidence["void_confirmed"])
        self.assertFalse(summary.impact_evidence["impact_confirmed"])
        self.assertLess(summary.multimodal_confidence_score, 0.50)

    def test_acoustic_salvage_in_rain(self):
        # Scenario: Rain=True, Sonar=18cm, Accelerometer=2.4g -> Acoustic Salvage in Rain
        summary = self.analyzer.analyze_evidence(
            damage_class="D40",
            visual_confidence=0.85,
            surface_area_sqm=1.2,
            effective_depth_cm=18.0,
            active_modality="sonar_submerged_acoustic",
            z_accel_g=2.40,
            rain_detected=True,
            mean_luminance=25.0
        )
        self.assertEqual(summary.disagreement_category, self.categories.ACOUSTIC_SALVAGE_IN_RAIN)
        self.assertTrue(summary.depth_evidence["void_confirmed"])
        self.assertTrue(summary.impact_evidence["impact_confirmed"])
        self.assertGreater(summary.multimodal_confidence_score, 0.80)


class TestSensorFusionExperiments(unittest.TestCase):
    """Unit tests for Phase 17: Multi-Modal Experiment Suite & Comparative Benchmarking."""

    def test_experiment_runner_all_configurations(self):
        from services.experiments.experiment_runner import FusionExperimentRunner
        runner = FusionExperimentRunner()
        summary = runner.run_all_experiments()

        self.assertIn("dataset_size", summary)
        self.assertIn("experiments", summary)
        exps = summary["experiments"]

        for code in ["A", "B", "C", "D", "E"]:
            self.assertIn(code, exps)
            exp = exps[code]
            self.assertIn("detection_confirmation_rate", exp)
            self.assertIn("unconfirmed_superficial_rate", exp)
            self.assertIn("impact_confirmation_rate", exp)
            self.assertIn("topsis_priority_stability", exp)
            self.assertIn("sensor_disagreement_rate", exp)
            self.assertTrue(0.0 <= exp["detection_confirmation_rate"] <= 1.0)
            self.assertTrue(0.0 <= exp["unconfirmed_superficial_rate"] <= 1.0)

        # Monotonicity checks: Exp D/E physical confirmation rate must exceed Exp A (0.0%)
        self.assertGreater(exps["E"]["detection_confirmation_rate"], exps["A"]["detection_confirmation_rate"])
        # Exp E stability is reference (1.0000)
        self.assertEqual(exps["E"]["topsis_priority_stability"], 1.0000)

    def test_spearman_rank_correlation(self):
        from services.experiments.experiment_runner import compute_spearman_rank_correlation
        # Perfect correlation
        self.assertAlmostEqual(compute_spearman_rank_correlation([1, 2, 3, 4], [1, 2, 3, 4]), 1.0)
        # Inverted correlation
        self.assertAlmostEqual(compute_spearman_rank_correlation([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        # Moderate correlation
        rho = compute_spearman_rank_correlation([1, 2, 3, 4, 5], [1, 3, 2, 5, 4])
        self.assertGreater(rho, 0.7)


class TestPhase19Phase20DashboardAndExplorer(unittest.TestCase):
    """
    Verification tests for Phase 19 (Digital Twin Dashboard Multimodal Display)
    and Phase 20 (Research Dataset Explorer).
    """

    def setUp(self):
        import asyncio
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    def test_dataset_explorer_metrics_contract(self):
        import main
        stats = self.loop.run_until_complete(main.get_dataset_explorer())

        self.assertEqual(stats["status"], "success")
        self.assertGreaterEqual(stats["total_records"], 4)
        # Class distribution checks
        self.assertIn("d00_count", stats)
        self.assertIn("d10_count", stats)
        self.assertIn("d20_count", stats)
        self.assertIn("d40_count", stats)
        self.assertEqual(
            stats["d00_count"] + stats["d10_count"] + stats["d20_count"] + stats["d40_count"],
            stats["total_records"]
        )

        # Average sensor & GIS metrics
        self.assertGreater(stats["avg_lidar_depth"], 20.0)
        self.assertGreater(stats["avg_sonar_depth"], 20.0)
        self.assertGreater(stats["avg_acceleration"], 0.5)
        self.assertGreater(stats["avg_traffic_pcu"], 1000)
        self.assertGreater(stats["avg_hospital_distance"], 0.1)

        # Environmental switching percentages
        self.assertTrue(0.0 <= stats["rain_percentage"] <= 100.0)
        self.assertTrue(0.0 <= stats["lidar_selected_percentage"] <= 100.0)
        self.assertTrue(0.0 <= stats["sonar_selected_percentage"] <= 100.0)

        # Priority distribution
        p_dist = stats["priority_distribution"]
        self.assertIn("Critical", p_dist)
        self.assertIn("High", p_dist)
        self.assertIn("Medium", p_dist)
        self.assertIn("Low", p_dist)
        self.assertEqual(sum(p_dist.values()), len(stats["records"]))

    def test_image_serving_endpoint(self):
        import main
        from fastapi.responses import FileResponse, Response
        # Existing sample image returns FileResponse
        resp = self.loop.run_until_complete(main.get_inspection_image("India_000045.jpg"))
        self.assertIsInstance(resp, FileResponse)
        self.assertEqual(resp.media_type, "image/jpeg")

        # Unknown image returns SVG placeholder Response
        missing_resp = self.loop.run_until_complete(main.get_inspection_image("India_999999.jpg"))
        self.assertIsInstance(missing_resp, Response)
        self.assertEqual(missing_resp.media_type, "image/svg+xml")
        self.assertIn(b"RDD2022 Ground-Truth Image", missing_resp.body)

    def test_telemetry_store_multimodal_provenance(self):
        import main
        self.assertGreaterEqual(len(main.telemetry_store), 4)
        for record in main.telemetry_store:
            # Data Source provenance tag
            self.assertIn(record.get("data_source"), ["MULTIMODAL", "RDD2022", "SYNTHETIC SENSOR", "SYNTHETIC GIS"])
            # Damage classification
            self.assertIn(record.get("damage_class"), ["D00", "D10", "D20", "D40"])
            self.assertIsInstance(record.get("damage_label"), str)
            # Confidence
            self.assertTrue(0.0 <= record.get("confidence", 0.0) <= 1.0)
            # Depth sensors
            self.assertGreater(record["lidar_depth_cm"], 0.0)
            self.assertGreater(record["sonar_depth_cm"], 0.0)
            # Accelerometer & context
            self.assertGreater(record["z_accel_g"], 0.0)
            self.assertGreater(record["traffic_pcu"], 0)
            self.assertGreater(record["dist_hospital_km"], 0.0)
            # TOPSIS score and priority
            self.assertTrue(0.0 <= record["topsis_score"] <= 1.0)
            self.assertIn(record["priority_level"], ["Critical", "High", "Medium", "Low"])


class TestPhase23VerificationComprehensive(unittest.TestCase):
    """
    Phase 23 Comprehensive Verification Test Suite:
    Explicitly covers all 12 system domains required by the research specification:
    1. RDD parsing
    2. Dataset generation
    3. Sensor distributions
    4. Sensor correlations
    5. Sensor switching
    6. Accelerometer threshold
    7. TOPSIS directionality
    8. Data provenance
    9. GPS generation
    10. Hospital distance
    11. Fusion
    12. Reproducibility
    """

    def test_01_rdd_parsing(self):
        from scripts.prepare_rdd2022 import parse_voc_xml
        sample_xml = PROJECT_ROOT / "data" / "samples" / "annotations" / "xmls" / "India_000045.xml"
        self.assertTrue(sample_xml.is_file(), f"Sample XML missing at {sample_xml}")
        parsed = parse_voc_xml(sample_xml)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["filename"], "India_000045.jpg")
        self.assertEqual(parsed["width"], 720)
        self.assertEqual(parsed["height"], 720)
        self.assertEqual(parsed["depth"], 3)
        self.assertGreaterEqual(len(parsed["objects"]), 1)
        for obj in parsed["objects"]:
            self.assertIn("class_id", obj)
            self.assertIn(obj["class_id"], ["D00", "D10", "D20", "D40"])
            self.assertTrue(0 <= obj["xmin"] < obj["xmax"] <= 720)
            self.assertTrue(0 <= obj["ymin"] < obj["ymax"] <= 720)

        # Handling non-existent XML gracefully
        non_existent = parse_voc_xml(PROJECT_ROOT / "data" / "samples" / "missing.xml")
        self.assertIsNone(non_existent)

    def test_02_dataset_generation(self):
        import tempfile
        import shutil
        from scripts.generate_srmd import generate_srmd
        
        temp_dir = Path(tempfile.mkdtemp(prefix="srmd_test_phase23_"))
        try:
            manifest = generate_srmd(
                input_path=PROJECT_ROOT / "data" / "samples",
                output_dir=temp_dir,
                seed=2026,
                limit=4
            )
            self.assertEqual(manifest["total_records"], 4)
            # Verify all required export files are produced
            self.assertTrue((temp_dir / "records").is_dir())
            self.assertTrue((temp_dir / "srmd_records.json").is_file())
            self.assertTrue((temp_dir / "srmd_dataset.jsonl").is_file())
            self.assertTrue((temp_dir / "srmd_dataset.csv").is_file())
            self.assertTrue((temp_dir / "srmd_manifest.json").is_file())
            self.assertTrue((temp_dir / "metadata.json").is_file())
            self.assertTrue((temp_dir / "splits" / "split_manifest.json").is_file())
            self.assertTrue((temp_dir / "splits" / "train_records.json").is_file())
            self.assertTrue((temp_dir / "splits" / "validation_records.json").is_file())
            self.assertTrue((temp_dir / "splits" / "test_records.json").is_file())

            # Verify class filter functionality
            temp_filter_dir = Path(tempfile.mkdtemp(prefix="srmd_filter_phase23_"))
            try:
                filter_manifest = generate_srmd(
                    input_path=PROJECT_ROOT / "data" / "samples",
                    output_dir=temp_filter_dir,
                    seed=2026,
                    class_filter="D40"
                )
                with open(temp_filter_dir / "srmd_records.json", "r", encoding="utf-8") as f:
                    recs = json.load(f)
                self.assertGreaterEqual(len(recs), 1)
                for r in recs:
                    self.assertEqual(r["source"]["annotation_class"], "D40")
                    self.assertEqual(r["visual"]["damage_class"], "pothole")
            finally:
                shutil.rmtree(temp_filter_dir, ignore_errors=True)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_03_sensor_distributions(self):
        from services.dataset.srmd_builder import SRMDRecordBuilder
        builder = SRMDRecordBuilder(seed=2026)
        classes = ["D00", "D10", "D20", "D40"]
        for idx, cid in enumerate(classes):
            rec = builder.synthesize_record(
                image_id=f"India_{idx:06d}",
                filename=f"India_{idx:06d}.jpg",
                damage_class=cid,
                bbox=[100, 150, 350, 400],
                image_dims={"width": 720, "height": 720, "depth": 3},
                annotation_idx=0,
                waypoint_idx=idx
            )
            # Physical bounds verification
            # Raw LiDAR & Sonar measurements include chassis baseline height (30cm)
            self.assertTrue(25.0 <= rec["lidar"]["depth_cm"] <= 60.0)
            self.assertTrue(0.1 <= rec["lidar"]["effective_depth_cm"] <= 25.0)
            self.assertTrue(25.0 <= rec["sonar"]["depth_cm"] <= 60.0)
            self.assertTrue(0.1 <= rec["sonar"]["effective_depth_cm"] <= 25.0)
            self.assertTrue(0.5 <= rec["accelerometer"]["z_accel_g"] <= 4.0)
            self.assertTrue(500 <= rec["traffic"]["traffic_pcu"] <= 40000)
            self.assertTrue(0.05 <= rec["infrastructure"]["dist_hospital_km"] <= 25.0)

    def test_04_sensor_correlations(self):
        from services.dataset.srmd_builder import SRMDRecordBuilder
        builder = SRMDRecordBuilder(seed=2026)
        # Compare shallow longitudinal crack vs severe pothole
        rec_d00 = builder.synthesize_record(
            image_id="India_D00", filename="India_D00.jpg", damage_class="D00",
            bbox=[100, 100, 150, 350], image_dims={"width": 720, "height": 720, "depth": 3},
            annotation_idx=0, waypoint_idx=0
        )
        rec_d40 = builder.synthesize_record(
            image_id="India_D40", filename="India_D40.jpg", damage_class="D40",
            bbox=[100, 100, 500, 500], image_dims={"width": 720, "height": 720, "depth": 3},
            annotation_idx=0, waypoint_idx=1
        )
        # Latent severity drives depth, area, volume, and vertical shock
        self.assertGreater(rec_d40["visual"]["defect_severity"], rec_d00["visual"]["defect_severity"])
        self.assertGreater(rec_d40["lidar"]["depth_cm"], rec_d00["lidar"]["depth_cm"])
        self.assertGreater(rec_d40["visual"]["calculated_volume_liters"], rec_d00["visual"]["calculated_volume_liters"])
        self.assertGreater(rec_d40["accelerometer"]["z_accel_g"], rec_d00["accelerometer"]["z_accel_g"])

    def test_05_sensor_switching(self):
        from services.ai_engine.pipeline.depth_estimator import EnvironmentalSensorSwitching
        switcher = EnvironmentalSensorSwitching(baseline_chassis_height_cm=30.0)

        # 1. Rain detected -> Acoustic Sonar
        rain_res = switcher.evaluate_environment_and_measure(
            rain_detected=True, mean_luminance=85.0,
            lidar_depth_cm=42.0, sonar_depth_cm=44.0, z_accel_g=1.1, surface_area_sqm=0.4
        )
        self.assertEqual(rain_res["detection_modality"], "sonar_submerged_acoustic")
        self.assertTrue(rain_res["is_submerged"])
        self.assertAlmostEqual(rain_res["estimated_depth_cm"], 14.0)

        # 2. Low luminance (< 40.0 lux) -> Acoustic Sonar (night)
        night_res = switcher.evaluate_environment_and_measure(
            rain_detected=False, mean_luminance=18.0,
            lidar_depth_cm=40.0, sonar_depth_cm=41.5, z_accel_g=1.0, surface_area_sqm=0.4
        )
        self.assertEqual(night_res["detection_modality"], "sonar_submerged_acoustic")
        self.assertFalse(night_res["is_submerged"])
        self.assertAlmostEqual(night_res["estimated_depth_cm"], 11.5)

        # 3. Daylight dry conditions -> Optical LiDAR
        day_res = switcher.evaluate_environment_and_measure(
            rain_detected=False, mean_luminance=70.0,
            lidar_depth_cm=38.0, sonar_depth_cm=38.0, z_accel_g=1.0, surface_area_sqm=0.4
        )
        self.assertEqual(day_res["detection_modality"], "optical_lidar")
        self.assertFalse(day_res["is_submerged"])
        self.assertAlmostEqual(day_res["estimated_depth_cm"], 8.0)

    def test_06_accelerometer_threshold(self):
        from services.data_generation import AccelerometerModel
        accel = AccelerometerModel(seed=2026)

        # Exact threshold boundary: 1.50g
        self.assertEqual(accel.classify_impact(1.49)["regime"], "borderline")
        self.assertEqual(accel.classify_impact(1.50)["regime"], "borderline")
        self.assertEqual(accel.classify_impact(1.51)["regime"], "confirmed_impact")
        self.assertEqual(accel.classify_impact(1.20)["regime"], "no_impact")

        # Simulation under deep vs shallow defects
        high_sim = accel.simulate("D40", defect_depth_cm=12.0, speed_kmh=50.0)
        self.assertGreater(high_sim["z_accel_g"], 1.50)
        self.assertTrue(high_sim["dynamic_impact_triggered"])
        self.assertEqual(high_sim["impact_regime"], "confirmed_impact")

        low_sim = accel.simulate("D00", defect_depth_cm=1.0, speed_kmh=30.0)
        self.assertLessEqual(low_sim["z_accel_g"], 1.50)
        self.assertFalse(low_sim["dynamic_impact_triggered"])

    def test_07_topsis_directionality(self):
        from services.backend.app.services.mcdm_engine import MCDMPrioritizationEngine
        engine = MCDMPrioritizationEngine()

        # Criteria weights check: Volume=0.35, Traffic=0.25, Hospital=0.25, Speed=0.15
        np.testing.assert_allclose(engine.weights, [0.35, 0.25, 0.25, 0.15])
        self.assertAlmostEqual(float(engine.weights.sum()), 1.0)

        # Directionality check: hospital distance is COST (False), all others BENEFIT (True)
        np.testing.assert_array_equal(engine.is_benefit, [True, True, False, True])

        # Strict test: smaller hospital distance must increase TOPSIS urgency score
        # Two identical defects (Volume=25L, Traffic=15000 PCU, Speed=40 km/h)
        # Defect 1: Near hospital (dist = 0.5 km) -> should have higher urgency/TOPSIS score
        # Defect 2: Far from hospital (dist = 8.0 km) -> should have lower urgency/TOPSIS score
        matrix = np.array([
            [25.0, 15000.0, 0.5, 40.0],  # Close to hospital
            [25.0, 15000.0, 8.0, 40.0]   # Distant from hospital
        ])
        scores = engine.compute_topsis(matrix)
        self.assertEqual(len(scores), 2)
        self.assertGreater(
            scores[0], scores[1],
            f"Defect closer to hospital ({scores[0]:.4f}) must receive higher urgency than distant defect ({scores[1]:.4f})"
        )

    def test_08_data_provenance(self):
        from services.dataset.srmd_builder import SRMDRecordBuilder
        builder = SRMDRecordBuilder(seed=2026)
        rec = builder.synthesize_record("India_000045", "India_000045.jpg", "D40", [100, 100, 300, 300], {"width": 720, "height": 720, "depth": 3}, 0, 0)
        self.assertEqual(rec["provenance"]["visual_source"], "RDD2022")
        self.assertEqual(rec["provenance"]["sensor_source"], "synthetic")
        self.assertEqual(rec["provenance"]["synthesis_seed"], 2026)
        self.assertEqual(rec["location"]["location_source"], "synthetic")
        self.assertEqual(rec["location"]["location_mode"], "synthetic_location")
        self.assertEqual(rec["traffic"]["traffic_source"], "synthetic")
        self.assertEqual(rec["infrastructure"]["hospital_source"], "synthetic")

    def test_09_gps_generation(self):
        from services.data_generation.location_model import (
            LocationModel, MODE_SOURCE_LOCATION, MODE_SYNTHETIC_LOCATION
        )
        # Deterministic reproducibility under seed 2026
        loc1 = LocationModel(mode=MODE_SYNTHETIC_LOCATION, seed=2026).generate_location_for_defect(route_index=3)
        loc2 = LocationModel(mode=MODE_SYNTHETIC_LOCATION, seed=2026).generate_location_for_defect(route_index=3)
        self.assertEqual(loc1["latitude"], loc2["latitude"])
        self.assertEqual(loc1["longitude"], loc2["longitude"])
        self.assertEqual(loc1["road_class"], loc2["road_class"])

        # Coordinates within documented Mumbai demonstration area
        self.assertTrue(18.9 <= loc1["latitude"] <= 19.3)
        self.assertTrue(72.7 <= loc1["longitude"] <= 73.1)

        # Source mode preservation with authentic coordinates
        source_loc = LocationModel(mode=MODE_SOURCE_LOCATION, seed=2026).generate_location_for_defect(
            route_index=0, source_gps={"latitude": 19.0760, "longitude": 72.8777}
        )
        self.assertEqual(source_loc["location_mode"], "source_location")
        self.assertEqual(source_loc["location_source"], "source_dataset")
        self.assertEqual(source_loc["latitude"], 19.0760)
        self.assertEqual(source_loc["longitude"], 72.8777)

        # Source mode without authentic coordinates raises ValueError (prevents false attribution)
        with self.assertRaises(ValueError):
            LocationModel(mode=MODE_SOURCE_LOCATION, seed=2026).generate_location_for_defect(route_index=0)

    def test_10_hospital_distance(self):
        from services.data_generation import InfrastructureModel
        infra = InfrastructureModel(seed=2026)
        # Defect near Lilavati Hospital (lat 19.0510, lon 72.8290)
        res = infra.simulate(latitude=19.0515, longitude=72.8295)
        self.assertEqual(res["nearest_hospital_name"], "Lilavati Hospital & Research Centre")
        self.assertLess(res["dist_hospital_km"], 0.25)
        self.assertTrue(res["is_emergency_corridor"])
        self.assertEqual(res["hospital_source"], "synthetic")
        self.assertEqual(res["location_source"], "synthetic")

    def test_11_fusion(self):
        from services.fusion.fusion_engine import MultimodalDataFusionEngine
        engine = MultimodalDataFusionEngine(seed=2026)
        sample_img = PROJECT_ROOT / "data" / "samples" / "images" / "India_000045.jpg"
        self.assertTrue(sample_img.is_file(), f"Sample image missing at {sample_img}")
        events = engine.fuse_image(sample_img, rain_detected=False, mean_luminance=80.0)
        self.assertGreater(len(events), 0)
        ev = events[0]
        self.assertIn(ev.visual.damage_class, ["D00", "D10", "D20", "D40"])
        self.assertGreater(ev.sensor.lidar_depth_cm, 0.0)
        self.assertGreater(ev.sensor.sonar_depth_cm, 0.0)
        self.assertGreater(ev.sensor.z_accel_g, 0.0)
        self.assertGreater(ev.topsis.topsis_score, 0.0)
        self.assertIn(ev.topsis.priority_level, ["Critical", "High", "Medium", "Low"])

    def test_12_reproducibility(self):
        from services.dataset.srmd_builder import SRMDRecordBuilder
        from services.dataset.split_manager import ResearchDataSplitter

        # 1. Deterministic sensor generation
        b1 = SRMDRecordBuilder(seed=2026)
        b2 = SRMDRecordBuilder(seed=2026)
        r1 = b1.synthesize_record("India_001", "India_001.jpg", "D40", [50, 50, 200, 200], {"width": 720, "height": 720, "depth": 3}, 0, 0)
        r2 = b2.synthesize_record("India_001", "India_001.jpg", "D40", [50, 50, 200, 200], {"width": 720, "height": 720, "depth": 3}, 0, 0)
        self.assertEqual(r1["lidar"]["depth_cm"], r2["lidar"]["depth_cm"])
        self.assertEqual(r1["sonar"]["depth_cm"], r2["sonar"]["depth_cm"])
        self.assertEqual(r1["accelerometer"]["z_accel_g"], r2["accelerometer"]["z_accel_g"])
        self.assertEqual(r1["traffic"]["traffic_pcu"], r2["traffic"]["traffic_pcu"])
        self.assertEqual(r1["infrastructure"]["dist_hospital_km"], r2["infrastructure"]["dist_hospital_km"])

        # 2. Deterministic split & zero leakage
        s1 = ResearchDataSplitter(seed=2026)
        s2 = ResearchDataSplitter(seed=2026)
        sample_ids = [f"India_{i:06d}" for i in range(15)]
        split1 = s1.split_image_ids(sample_ids)
        split2 = s2.split_image_ids(sample_ids)
        self.assertEqual(split1, split2)
        self.assertEqual(len(set(split1["train"]) & set(split1["test"])), 0)
        self.assertEqual(len(set(split1["train"]) & set(split1["validation"])), 0)
        self.assertEqual(len(set(split1["validation"]) & set(split1["test"])), 0)

        # 3. metadata.json schema conformance
        metadata_file = PROJECT_ROOT / "data" / "processed" / "srmd" / "metadata.json"
        self.assertTrue(metadata_file.is_file())
        with open(metadata_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.assertEqual(meta["dataset_name"], "Smart Road Multimodal Dataset")
        self.assertEqual(meta["base_dataset"], "RDD2022")
        self.assertEqual(meta["base_dataset_country"], "India")
        self.assertEqual(meta["generator_version"], "1.0.0")
        self.assertEqual(meta["seed"], 2026)
        self.assertEqual(meta["sensor_fields"], [
            "lidar_depth_cm",
            "sonar_depth_cm",
            "z_accel_g",
            "traffic_pcu",
            "dist_hospital_km"
        ])


if __name__ == "__main__":
    unittest.main()


