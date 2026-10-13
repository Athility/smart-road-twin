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

    def test_seed_initial_data_sonar_volume_not_overwritten(self):
        """
        Task 1: processed_telemetry["defect_severity_magnitude"] and
        processed_telemetry["calculated_volume_liters"] / ["crack_severity_index"]
        must always equal the pre-computed values stored in the SRMD JSON visual block,
        regardless of which sensor modality was active.
        This ensures the SRMD-path and API-path decision matrices are identical.
        """
        srmd_file = PROJECT_ROOT / "data" / "processed" / "srmd" / "srmd_records.json"
        self.assertTrue(srmd_file.exists())
        srmd_by_id = {
            r["event_id"]: r
            for r in json.loads(srmd_file.read_text(encoding="utf-8"))
        }
        self.assertEqual(len(main.telemetry_store), len(srmd_by_id),
                         "telemetry_store size must match srmd_records.json")
        for rec in main.telemetry_store:
            raw = srmd_by_id[rec["id"]]
            pt = rec["processed_telemetry"]
            self.assertAlmostEqual(
                pt["defect_severity_magnitude"],
                float(raw["visual"]["defect_severity_magnitude"]),
                places=4,
                msg=f"magnitude mismatch for {rec['id']}: "
                    f"store={pt['defect_severity_magnitude']}, "
                    f"srmd={raw['visual']['defect_severity_magnitude']}"
            )
            srmd_vol = raw["visual"].get("calculated_volume_liters")
            srmd_csi = raw["visual"].get("crack_severity_index")
            if srmd_vol is None:
                self.assertIsNone(pt.get("calculated_volume_liters"),
                                  f"Expected None volume for {rec['id']}")
            else:
                self.assertAlmostEqual(
                    pt["calculated_volume_liters"], float(srmd_vol), places=4,
                    msg=f"volume mismatch for {rec['id']}"
                )
            if srmd_csi is None:
                self.assertIsNone(pt.get("crack_severity_index"),
                                  f"Expected None CSI for {rec['id']}")
            else:
                self.assertAlmostEqual(
                    pt["crack_severity_index"], float(srmd_csi), places=4,
                    msg=f"CSI mismatch for {rec['id']}"
                )


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
        self.assertLessEqual(eval_res["mAP50_95"], eval_res["mAP50"])

    def test_map_perfect_detection_and_monotonicity(self):
        from ml.evaluate.metrics import evaluate_dataset_metrics

        # 1. Perfect detection across all 4 classes gives 1.0 for mAP50, mAP50_95, and per-class ap50
        perfect_gts = {
            "img1": [
                {"class_id": "D00", "bbox": [10, 10, 100, 100]},
                {"class_id": "D10", "bbox": [120, 10, 220, 100]},
            ],
            "img2": [
                {"class_id": "D20", "bbox": [10, 120, 150, 250]},
                {"class_id": "D40", "bbox": [200, 200, 350, 350]},
            ],
        }
        perfect_preds = {
            "img1": [
                {"class_id": "D00", "confidence": 0.99, "bbox": [10, 10, 100, 100]},
                {"class_id": "D10", "confidence": 0.95, "bbox": [120, 10, 220, 100]},
            ],
            "img2": [
                {"class_id": "D20", "confidence": 0.92, "bbox": [10, 120, 150, 250]},
                {"class_id": "D40", "confidence": 0.98, "bbox": [200, 200, 350, 350]},
            ],
        }
        res_perfect = evaluate_dataset_metrics(perfect_gts, perfect_preds)
        self.assertEqual(res_perfect["mAP50"], 1.0)
        self.assertEqual(res_perfect["mAP50_95"], 1.0)
        for cls_id in ["D00", "D10", "D20", "D40"]:
            self.assertEqual(res_perfect["class_metrics"][cls_id]["ap50"], 1.0)

        # 2. Imperfect box overlap (IoU = 0.81) passes at 0.50..0.80 (7 thresholds) and fails at 0.85..0.95 (3 thresholds)
        partial_gts = {"img1": [{"class_id": "D40", "bbox": [0, 0, 10, 10]}]}
        partial_preds = {"img1": [{"class_id": "D40", "confidence": 0.90, "bbox": [1, 1, 10, 10]}]}
        res_partial = evaluate_dataset_metrics(partial_gts, partial_preds)
        self.assertEqual(res_partial["mAP50"], 1.0)
        self.assertAlmostEqual(res_partial["mAP50_95"], 0.7, places=4)
        self.assertLessEqual(res_partial["mAP50_95"], res_partial["mAP50"])

        # 3. Verify per-class ap50 comes from 101-point PR curve, NOT prec * rec
        # 2 GT boxes: 1 TP (conf=0.95) + 1 FP (conf=0.40) -> prec=0.5, rec=0.5 (prec*rec = 0.25), PR-curve AP = 51/101 = 0.505
        pr_gts = {
            "img1": [
                {"class_id": "D40", "bbox": [0, 0, 10, 10]},
                {"class_id": "D40", "bbox": [50, 50, 60, 60]},
            ]
        }
        pr_preds = {
            "img1": [
                {"class_id": "D40", "confidence": 0.95, "bbox": [0, 0, 10, 10]},
                {"class_id": "D40", "confidence": 0.40, "bbox": [200, 200, 210, 210]},
            ]
        }
        res_pr = evaluate_dataset_metrics(pr_gts, pr_preds)
        self.assertEqual(res_pr["class_metrics"]["D40"]["precision"], 0.5)
        self.assertEqual(res_pr["class_metrics"]["D40"]["recall"], 0.5)
        self.assertAlmostEqual(res_pr["class_metrics"]["D40"]["ap50"], round(51.0 / 101.0, 4), places=4)
        self.assertNotEqual(res_pr["class_metrics"]["D40"]["ap50"], 0.25)
        self.assertLessEqual(res_pr["mAP50_95"], res_pr["mAP50"])



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

    def test_telemetry_and_explorer_share_identical_source(self):
        import main
        telemetry_res = self.loop.run_until_complete(main.get_telemetry())
        explorer_res = self.loop.run_until_complete(main.get_dataset_explorer())
        self.assertEqual(telemetry_res["total"], explorer_res["total_records"])
        self.assertGreaterEqual(telemetry_res["total"], 6)
        self.assertEqual(
            [r["id"] for r in telemetry_res["data"]],
            [r["id"] for r in explorer_res["records"]]
        )


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
        # Latent severity drives depth, area, volume/severity magnitude, and vertical shock
        self.assertGreater(rec_d40["visual"]["defect_severity"], rec_d00["visual"]["defect_severity"])
        self.assertGreater(rec_d40["lidar"]["depth_cm"], rec_d00["lidar"]["depth_cm"])
        self.assertIsNone(rec_d00["visual"]["calculated_volume_liters"])
        self.assertIsNotNone(rec_d00["visual"]["crack_severity_index"])
        self.assertIsNotNone(rec_d40["visual"]["calculated_volume_liters"])
        self.assertGreater(rec_d40["visual"]["defect_severity_magnitude"], rec_d00["visual"]["defect_severity_magnitude"])
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
        import tempfile
        import shutil
        from services.dataset.srmd_builder import SRMDRecordBuilder
        from services.dataset.split_manager import ResearchDataSplitter
        from scripts.generate_srmd import generate_srmd

        temp_dir = Path(tempfile.mkdtemp(prefix="srmd_repro_phase23_"))
        try:
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

            # 2. Deterministic split & zero leakage exported into temporary directory
            s1 = ResearchDataSplitter(seed=2026)
            s2 = ResearchDataSplitter(seed=2026)
            sample_ids = [f"India_{i:06d}" for i in range(15)]
            split1 = s1.split_image_ids(sample_ids)
            split2 = s2.split_image_ids(sample_ids)
            self.assertEqual(split1, split2)
            self.assertEqual(len(set(split1["train"]) & set(split1["test"])), 0)
            self.assertEqual(len(set(split1["train"]) & set(split1["validation"])), 0)
            self.assertEqual(len(set(split1["validation"]) & set(split1["test"])), 0)

            generate_srmd(
                input_path=PROJECT_ROOT / "data" / "samples",
                output_dir=temp_dir,
                seed=2026,
                limit=4
            )

            # 3. metadata.json schema conformance in temporary directory
            metadata_file = temp_dir / "metadata.json"
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
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestBackendCLAHEMCDMAndPrioritization(unittest.TestCase):
    """
    Unit tests for:
    1. CLAHE preprocessing (cv2 LAB L-channel + NumPy/PIL fallback), detector config flag, and preview endpoint.
    2. Fuzzy BWM weight derivation (TFN comparison vectors, 3 named profiles + custom).
    3. VIKOR compromise ranking (S_i, R_i, Q_i with v=0.5, acceptable advantage & stability checks).
    4. compute_all() returning TOPSIS rank, VIKOR rank, and Spearman correlation.
    5. School proximity (Haversine over >=15 Mumbai Western Corridor schools) & road_class ordinal weight.
    6. GET/POST /api/v1/prioritization endpoints and calibrated priority thresholds.
    """

    def setUp(self):
        import asyncio
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    def test_apply_clahe_cv2_and_numpy_fallback(self):
        from services.cv.preprocessing import apply_clahe, load_image
        # Synthetic low-contrast 2D road-surface texture image (128x128x3)
        xx, yy = np.meshgrid(np.linspace(0.0, 1.0, 128), np.linspace(0.0, 1.0, 128))
        pat = np.clip(100.0 + 25.0 * np.sin(2.0 * np.pi * 4.0 * xx) * np.cos(2.0 * np.pi * 4.0 * yy), 0, 255).astype(np.uint8)
        low_contrast_rgb = np.stack([pat, pat, pat], axis=-1)

        # 1. Default OpenCV LAB L-channel CLAHE path
        clahe_cv2 = apply_clahe(low_contrast_rgb, clip_limit=2.0, tile_grid=(8, 8), force_fallback=False)
        self.assertEqual(clahe_cv2.shape, low_contrast_rgb.shape)
        self.assertEqual(clahe_cv2.dtype, np.uint8)
        self.assertGreater(float(np.std(clahe_cv2)), float(np.std(low_contrast_rgb)))

        # 2. NumPy/PIL fallback path (when cv2 is missing or force_fallback=True)
        clahe_np = apply_clahe(low_contrast_rgb, clip_limit=2.0, tile_grid=(8, 8), force_fallback=True)
        self.assertEqual(clahe_np.shape, low_contrast_rgb.shape)
        self.assertEqual(clahe_np.dtype, np.uint8)
        self.assertGreater(float(np.std(clahe_np)), float(np.std(low_contrast_rgb)))

        # 3. Real sample image test
        sample_img = PROJECT_ROOT / "data" / "samples" / "images" / "India_000045.jpg"
        if sample_img.exists():
            orig_rgb, (w, h) = load_image(sample_img)
            enhanced = apply_clahe(orig_rgb, clip_limit=2.0, tile_grid=(8, 8))
            self.assertEqual(enhanced.shape, (h, w, 3))
            self.assertFalse(np.array_equal(orig_rgb, enhanced))

    def test_detector_clahe_config_flag(self):
        from services.cv.rdd2022_detector import RDD2022Detector
        sample_img = PROJECT_ROOT / "data" / "samples" / "images" / "India_000045.jpg"
        self.assertTrue(sample_img.exists())

        # Default detector has enable_clahe=True from config/sensor_profiles.yaml
        det_on = RDD2022Detector()
        self.assertTrue(det_on.enable_clahe)
        res_on = det_on.detect_full(sample_img)
        self.assertGreater(len(res_on.detections), 0)

        # Detector with enable_clahe=False
        det_off = RDD2022Detector(enable_clahe=False)
        self.assertFalse(det_off.enable_clahe)
        res_off = det_off.detect_full(sample_img)
        self.assertEqual(len(res_off.detections), len(res_on.detections))

    def test_preprocess_preview_endpoint(self):
        from fastapi import HTTPException
        resp = self.loop.run_until_complete(
            main.get_preprocess_preview("India_000045.jpg", clip_limit=2.0, tile_grid_size=8)
        )
        self.assertEqual(resp["status"], "success")
        self.assertEqual(resp["image_name"], "India_000045.jpg")
        self.assertEqual(resp["clip_limit"], 2.0)
        self.assertEqual(resp["tile_grid"], [8, 8])
        self.assertTrue(resp["original_image"].startswith("data:image/jpeg;base64,"))
        self.assertTrue(resp["clahe_image"].startswith("data:image/jpeg;base64,"))
        self.assertNotEqual(resp["original_image"], resp["clahe_image"])
        self.assertIn("original_contrast_std", resp["metrics"])
        self.assertIn("clahe_contrast_std", resp["metrics"])

        # Missing image raises 404
        with self.assertRaises(HTTPException) as ctx:
            self.loop.run_until_complete(
                main.get_preprocess_preview("NonExistent_999999.jpg", clip_limit=2.0, tile_grid_size=8)
            )
        self.assertEqual(ctx.exception.status_code, 404)

    def test_fuzzy_bwm_weight_derivation_and_profiles(self):
        from services.backend.app.services.mcdm_engine import (
            derive_fuzzy_bwm_weights, get_fuzzy_bwm_profile
        )
        # 1. Test all 3 named profiles ('safety-first', 'traffic-first', 'equity-first')
        for prof_name, expected_best, expected_worst in [
            ("safety-first", 0, 5),
            ("traffic-first", 1, 4),
            ("equity-first", 2, 3),
        ]:
            prof = get_fuzzy_bwm_profile(prof_name)
            w = np.asarray(prof["weights"], dtype=float)
            self.assertEqual(w.shape, (6,))
            self.assertAlmostEqual(float(np.sum(w)), 1.0, places=3)
            self.assertTrue(np.all(w > 0))
            self.assertEqual(prof["best_idx"], expected_best)
            self.assertEqual(prof["worst_idx"], expected_worst)
            # Best criterion must have max weight, worst criterion must have min weight
            self.assertGreaterEqual(w[expected_best], float(np.max(w)) - 1e-4)
            self.assertLessEqual(w[expected_worst], float(np.min(w)) + 1e-4)
            self.assertLess(prof["consistency_ratio"], 0.10)
            for l_v, m_v, u_v in prof["fuzzy_weights"]:
                self.assertTrue(0 < l_v <= m_v <= u_v)

        # 2. Custom TFN vectors via derive_fuzzy_bwm_weights
        custom_bwm = derive_fuzzy_bwm_weights(
            best_to_others=[(1.0, 1.0, 1.0), (1.5, 2.0, 2.5), (2.5, 3.0, 3.5), (3.5, 4.0, 4.5)],
            others_to_worst=[(3.5, 4.0, 4.5), (2.5, 3.0, 3.5), (1.5, 2.0, 2.5), (1.0, 1.0, 1.0)],
            best_idx=0,
            worst_idx=3
        )
        cw = custom_bwm["weights"]
        self.assertEqual(len(cw), 4)
        self.assertAlmostEqual(float(np.sum(cw)), 1.0, places=3)
        self.assertTrue(cw[0] > cw[1] > cw[2] > cw[3])
        self.assertLess(custom_bwm["consistency_ratio"], 0.10)

    def test_vikor_compromise_ranking_and_checks(self):
        engine = MCDMPrioritizationEngine(profile="safety-first", use_calibrated_thresholds=True)
        # 4 alternatives across 6 criteria:
        # Alt 0: Dominates all criteria -> Q=0.0, rank=1, acceptable advantage & stability True
        matrix = np.array([
            [140.0, 30000.0, 0.2, 55.0, 0.2, 4.0],  # Alt 0: Clear ideal best
            [40.0,  15000.0, 1.8, 35.0, 1.2, 3.0],  # Alt 1: Moderate
            [15.0,  8000.0,  3.2, 28.0, 2.0, 2.0],  # Alt 2: Minor
            [3.0,   3000.0,  5.0, 20.0, 3.5, 1.0],  # Alt 3: Least urgent
        ])
        vikor = engine.compute_vikor(matrix, v=0.5)
        self.assertIn("S", vikor)
        self.assertIn("R", vikor)
        self.assertIn("Q", vikor)
        self.assertEqual(vikor["v"], 0.5)
        self.assertAlmostEqual(vikor["DQ"], 1.0 / 3.0, places=4)
        self.assertEqual(vikor["ranks"], [1, 2, 3, 4])
        self.assertAlmostEqual(float(vikor["Q"][0]), 0.0, places=4)
        self.assertAlmostEqual(float(vikor["Q"][3]), 1.0, places=4)
        self.assertTrue(vikor["acceptable_advantage"])
        self.assertTrue(vikor["acceptable_stability"])
        self.assertEqual(vikor["compromise_solution"], [0])

        # Near-tied top 2 alternatives -> acceptable_advantage must be False
        near_tie_matrix = np.array([
            [100.0, 25000.0, 0.5, 45.0, 0.4, 4.0],
            [99.5,  24900.0, 0.51, 44.8, 0.41, 4.0],
            [10.0,  5000.0,  4.0, 20.0, 3.0, 1.0],
        ])
        vikor_tie = engine.compute_vikor(near_tie_matrix, v=0.5)
        self.assertFalse(vikor_tie["acceptable_advantage"])
        self.assertGreaterEqual(len(vikor_tie["compromise_solution"]), 2)

    def test_compute_all_topsis_vikor_and_spearman(self):
        """
        Verify rigorous mathematical invariants for TOPSIS, VIKOR, and Spearman correlation:
        1. Ranks form strict permutations of {1, ..., n} without duplicates or missing tiers.
        2. Scores are strictly bounded: TOPSIS C_i ∈ [0, 1], VIKOR Q_i ∈ [0, 1], S_i ≥ 0, R_i ≥ 0.
        3. Cross-path identity: SRMD-path and store-path produce 100% identical decision matrices and ranks.
        4. Defect severity dominance: Rank 1 is strictly occupied by a D40 pothole (cracks never attain Rank 1).
        5. Multi-profile concordance: traffic-first yields perfect rho=1.0000, equity-first yields rho >= 0.90.
        """
        from services.backend.app.services.mcdm_engine import (
            build_mcdm_row_from_srmd,
            build_mcdm_row_from_store_item,
        )
        import main as app_module

        engine = MCDMPrioritizationEngine(profile="safety-first", use_calibrated_thresholds=True)
        srmd_file = PROJECT_ROOT / "data" / "processed" / "srmd" / "srmd_records.json"
        records = json.loads(srmd_file.read_text(encoding="utf-8"))
        n = len(records)
        perm = list(range(1, n + 1))

        # 1. Compute on SRMD records
        matrix_srmd = np.array([build_mcdm_row_from_srmd(r) for r in records])
        res = engine.compute_all(matrix_srmd, v=0.5, calibrated=True)

        # Invariant 1: Rank vectors are exact permutations of {1, ..., n}
        self.assertEqual(len(res["topsis_ranks"]), n)
        self.assertEqual(len(res["vikor_ranks"]), n)
        self.assertEqual(sorted(res["topsis_ranks"]), perm)
        self.assertEqual(sorted(res["vikor_ranks"]), perm)

        # Invariant 2: Mathematical score boundaries
        self.assertTrue(-1.0 <= res["spearman_correlation"] <= 1.0)
        for s in res["topsis_scores"]:
            self.assertTrue(0.0 <= s <= 1.0, f"TOPSIS score out of [0, 1]: {s}")
        for q in res["vikor_Q"]:
            self.assertTrue(0.0 <= q <= 1.0, f"VIKOR Q out of [0, 1]: {q}")
        for s_val in res["vikor_S"]:
            self.assertGreaterEqual(s_val, 0.0)
        for r_val in res["vikor_R"]:
            self.assertGreaterEqual(r_val, 0.0)

        # Invariant 3: Cross-path identity between SRMD JSON and telemetry_store
        app_module._load_srmd_records_into_store()
        store_items = [item for item in app_module.telemetry_store if any(r["event_id"] == item["id"] for r in records)]
        self.assertEqual(len(store_items), n)
        matrix_store = np.array([build_mcdm_row_from_store_item(item) for item in store_items])
        np.testing.assert_allclose(
            matrix_srmd, matrix_store, atol=1e-5,
            err_msg="SRMD-path matrix and telemetry_store-path matrix must be 100% identical"
        )
        res_store = engine.compute_all(matrix_store, v=0.5, calibrated=True)
        self.assertEqual(res["topsis_ranks"], res_store["topsis_ranks"])
        self.assertEqual(res["vikor_ranks"], res_store["vikor_ranks"])
        self.assertAlmostEqual(res["spearman_correlation"], res_store["spearman_correlation"], places=6)

        # Invariant 4: Pothole severity dominance (cracks never attain Rank 1 on baseline demo set)
        pothole_indices = [i for i, r in enumerate(records) if r.get("source", {}).get("annotation_class") == "D40"]
        t_r1 = res["topsis_ranks"].index(1)
        v_r1 = res["vikor_ranks"].index(1)
        self.assertIn(t_r1, pothole_indices, "TOPSIS Rank 1 must be a D40 pothole on demo set")
        self.assertIn(v_r1, pothole_indices, "VIKOR Rank 1 must be a D40 pothole on demo set")

        # Invariant 5: Multi-profile concordance properties
        eng_traffic = MCDMPrioritizationEngine(profile="traffic-first", use_calibrated_thresholds=True)
        res_traffic = eng_traffic.compute_all(matrix_srmd, v=0.5, calibrated=True)
        self.assertAlmostEqual(res_traffic["spearman_correlation"], 1.0000, places=4,
                               msg="traffic-first profile must exhibit perfect TOPSIS-VIKOR concordance")

        eng_equity = MCDMPrioritizationEngine(profile="equity-first", use_calibrated_thresholds=True)
        res_equity = eng_equity.compute_all(matrix_srmd, v=0.5, calibrated=True)
        self.assertGreaterEqual(res_equity["spearman_correlation"], 0.90,
                                msg="equity-first profile must exhibit strong concordance (rho >= 0.90)")

    def test_clamping_metadata_and_realistic_pothole_geometry(self):
        """
        Task 2: verify area_clamped (bool) and area_unclamped_m2 (float) exist
        in SRMD records and API store items; verify pothole areas are realistic
        (0.05 - 0.40 m2) and depth is 2 - 12 cm with no clamping on sample data.
        """
        from services.data_generation.consistency_model import PhysicalConsistencyModel
        import main as app_module

        # 1. PhysicalConsistencyModel metadata output and clamping behavior
        pcm = PhysicalConsistencyModel()
        # Normal pothole bbox inside realistic municipal bounds
        norm_area, norm_clamped, norm_unclamped = pcm.compute_perspective_surface_area_with_meta(
            bbox=[240.0, 350.0, 380.0, 460.0],
            image_dims={"width": 720, "height": 720},
            damage_class="D40"
        )
        self.assertFalse(norm_clamped)
        self.assertAlmostEqual(norm_area, norm_unclamped, places=3)
        self.assertTrue(0.05 <= norm_area <= 0.40)

        # Huge bbox that exceeds 1.00 m2 upper bound for D40
        huge_area, huge_clamped, huge_unclamped = pcm.compute_perspective_surface_area_with_meta(
            bbox=[50.0, 100.0, 670.0, 700.0],
            image_dims={"width": 720, "height": 720},
            damage_class="D40"
        )
        self.assertTrue(huge_clamped)
        self.assertGreater(huge_unclamped, 1.00)
        self.assertEqual(huge_area, 1.00)

        # 2. SRMD records on disk
        srmd_file = PROJECT_ROOT / "data" / "processed" / "srmd" / "srmd_records.json"
        self.assertTrue(srmd_file.exists())
        records = json.loads(srmd_file.read_text(encoding="utf-8"))
        for r in records:
            vis = r["visual"]
            self.assertIn("area_clamped", vis)
            self.assertIn("area_unclamped_m2", vis)
            self.assertIsInstance(vis["area_clamped"], bool)
            self.assertIsInstance(vis["area_unclamped_m2"], (int, float))
            # On the realistic sample bboxes, clamping must be rare (0 out of 6)
            self.assertFalse(vis["area_clamped"])

            if r.get("source", {}).get("damage_class") == "D40":
                area = vis["surface_area_sqm"]
                self.assertTrue(0.05 <= area <= 0.40, f"Pothole area out of realistic range: {area}")
                # Effective depth must be between 2 and 12 cm
                lidar_depth = r.get("lidar", {}).get("depth_cm", 0.0)
                self.assertTrue(2.0 <= lidar_depth <= 12.0, f"Pothole depth out of bounds: {lidar_depth}")

        # 3. Telemetry store / API items
        store_items = list(app_module.telemetry_store)
        self.assertGreaterEqual(len(store_items), len(records))
        for item in store_items:
            self.assertIn("area_clamped", item)
            self.assertIn("area_unclamped_m2", item)
            self.assertIsInstance(item["area_clamped"], bool)
            self.assertIsInstance(item["area_unclamped_m2"], (int, float))
            pt = item.get("processed_telemetry", {})
            self.assertIn("area_clamped", pt)
            self.assertIn("area_unclamped_m2", pt)

    def test_srmd_and_api_matrix_are_identical(self):
        """
        Task 1 regression: SRMD-path matrix (built via build_mcdm_row_from_srmd) must be
        element-wise equal to the API-path matrix (built via build_mcdm_row_from_store_item).
        Guarantees a single Spearman rho value across both code paths.
        """
        from services.backend.app.services.mcdm_engine import (
            build_mcdm_row_from_srmd,
            build_mcdm_row_from_store_item,
        )
        import main as app_module
        srmd_file = PROJECT_ROOT / "data" / "processed" / "srmd" / "srmd_records.json"
        records = json.loads(srmd_file.read_text(encoding="utf-8"))
        mat_srmd = np.array([build_mcdm_row_from_srmd(r) for r in records])

        store_items = list(app_module.telemetry_store)
        self.assertEqual(len(store_items), len(records))
        id_to_item = {it["id"]: it for it in store_items}
        mat_api = np.array([
            build_mcdm_row_from_store_item(id_to_item[r["event_id"]])
            for r in records
        ])
        self.assertTrue(
            np.allclose(mat_srmd, mat_api, rtol=1e-5, atol=1e-6),
            f"Decision matrices differ:\nSRMD:\n{mat_srmd}\nAPI:\n{mat_api}"
        )
        engine = MCDMPrioritizationEngine(profile="safety-first", use_calibrated_thresholds=True)
        rho_srmd = engine.compute_all(mat_srmd, v=0.5, calibrated=True)["spearman_correlation"]
        rho_api  = engine.compute_all(mat_api,  v=0.5, calibrated=True)["spearman_correlation"]
        self.assertAlmostEqual(rho_srmd, rho_api, places=4,
                               msg=f"rho divergence: SRMD={rho_srmd}, API={rho_api}")

    def test_school_registry_and_road_class_criteria(self):
        from services.data_generation.infrastructure_model import (
            DEFAULT_SCHOOLS, InfrastructureModel, encode_road_class
        )
        # 1. Verify >= 15 real Mumbai Western Corridor schools
        self.assertGreaterEqual(len(DEFAULT_SCHOOLS), 15)
        for sch in DEFAULT_SCHOOLS:
            self.assertIn("name", sch)
            self.assertTrue(19.04 <= sch["lat"] <= 19.25, f"School lat out of corridor: {sch}")
            self.assertTrue(72.82 <= sch["lon"] <= 72.89, f"School lon out of corridor: {sch}")

        # 2. Haversine distance to nearest school (St. Stanislaus High School at 19.0548, 72.8286)
        infra = InfrastructureModel(seed=2026)
        sim_near_school = infra.simulate(latitude=19.0549, longitude=72.8287)
        self.assertEqual(sim_near_school["nearest_school_name"], "St. Stanislaus High School")
        self.assertLess(sim_near_school["school_proximity_km"], 0.15)
        self.assertTrue(sim_near_school["is_school_zone"])

        # 3. Ordinal road class encoding
        self.assertEqual(encode_road_class("LOCAL"), 1)
        self.assertEqual(encode_road_class("COLLECTOR"), 2)
        self.assertEqual(encode_road_class("ARTERIAL"), 3)
        self.assertEqual(encode_road_class("HIGH_TRAFFIC_CORRIDOR"), 4)

        # 4. Directionality in 6-criteria MCDM:
        # Closer school (smaller school_proximity_km) and higher road class must increase TOPSIS score
        engine = MCDMPrioritizationEngine(profile="safety-first", use_calibrated_thresholds=True)
        mat_school = np.array([
            [30.0, 15000.0, 1.5, 40.0, 0.15, 3.0],  # 0.15 km from school
            [30.0, 15000.0, 1.5, 40.0, 4.50, 3.0],  # 4.50 km from school
        ])
        sc_school = engine.compute_topsis(mat_school)
        self.assertGreater(sc_school[0], sc_school[1])

        mat_road = np.array([
            [30.0, 15000.0, 1.5, 40.0, 1.0, 4.0],  # HIGH_TRAFFIC_CORRIDOR (4)
            [30.0, 15000.0, 1.5, 40.0, 1.0, 1.0],  # LOCAL (1)
        ])
        sc_road = engine.compute_topsis(mat_road)
        self.assertGreater(sc_road[0], sc_road[1])

    def test_prioritization_get_and_post_endpoints_and_calibrated_thresholds(self):
        # 1. GET /api/v1/prioritization across methods and profiles
        for method in ("topsis", "vikor", "both"):
            res = self.loop.run_until_complete(
                main.get_prioritization(method=method, profile="safety-first", v=0.5)
            )
            self.assertEqual(res["status"], "success")
            self.assertEqual(res["method"], method)
            self.assertEqual(res["profile"], "safety-first")
            self.assertEqual(len(res["weights"]), 6)
            self.assertGreaterEqual(res["total"], 6)
            self.assertIn("spearman_correlation", res)
            self.assertIn("vikor_checks", res)
            # Calibrated thresholds must produce at least 1 Critical item on the demo set
            crit_items = [r for r in res["results"] if r["priority_level"] == "Critical"]
            self.assertGreaterEqual(
                len(crit_items), 1,
                f"Expected >= 1 Critical item for method={method}, got {res['results']}"
            )

        # 2. Test traffic-first and equity-first profiles
        for prof in ("traffic-first", "equity-first"):
            res_p = self.loop.run_until_complete(
                main.get_prioritization(method="both", profile=prof, v=0.5)
            )
            self.assertEqual(res_p["profile"], prof)
            crit_p = [r for r in res_p["results"] if r["priority_level"] == "Critical"]
            self.assertGreaterEqual(len(crit_p), 1)

        # 3. POST /api/v1/prioritization with custom crisp weights
        req_weights = main.CustomPrioritizationRequest(
            method="both",
            profile="custom",
            weights=[0.30, 0.20, 0.20, 0.10, 0.10, 0.10],
            v=0.5
        )
        post_res = self.loop.run_until_complete(main.post_custom_prioritization(req_weights))
        self.assertEqual(post_res["status"], "success")
        self.assertEqual(post_res["profile"], "custom")
        np.testing.assert_allclose(post_res["weights"], [0.30, 0.20, 0.20, 0.10, 0.10, 0.10], atol=1e-3)

        # 4. POST /api/v1/prioritization with custom Fuzzy BWM comparison vectors
        req_bwm = main.CustomPrioritizationRequest(
            method="both",
            profile="custom",
            best_to_others=["EI", "FI", "WI", "FI", "FI", "AI"],
            others_to_worst=["AI", "FI", "VI", "FI", "FI", "EI"],
            best_idx=0,
            worst_idx=5,
            v=0.5
        )
        post_bwm_res = self.loop.run_until_complete(main.post_custom_prioritization(req_bwm))
        self.assertEqual(post_bwm_res["status"], "success")
        self.assertLess(post_bwm_res["fuzzy_bwm"]["consistency_ratio"], 0.10)

        # 5. Verify ablation benchmark file in runs/experiments/fusion_benchmark.json
        bench_path = PROJECT_ROOT / "runs" / "experiments" / "fusion_benchmark.json"
        self.assertTrue(bench_path.exists())
        bench = json.loads(bench_path.read_text(encoding="utf-8"))
        self.assertGreaterEqual(bench["experiments"]["E"]["critical_priority_count"], 1)
        self.assertIn("vikor_priority_stability", bench["experiments"]["E"])


class TestDomainPhysicsAndThresholdHonesty(unittest.TestCase):
    """
    Unit tests for:
    1. Crack records never have a volume; potholes always do.
    2. Ground-plane pinhole area clamping and warning log.
    3. Sonar water-surface bias under rain and wet-surface LiDAR degradation isolation.
    4. Ranking sanity (large pothole outranks small crack with equal context) and threshold calibration_note.
    """

    def test_crack_records_never_have_volume_and_potholes_always_do(self):
        from services.ai_engine.pipeline.depth_estimator import EnvironmentalSensorSwitching

        switcher = EnvironmentalSensorSwitching(baseline_chassis_height_cm=30.0)

        # 1. Check across all 4 defect classes via EnvironmentalSensorSwitching
        for crack_cls in ("D00", "D10", "D20"):
            res = switcher.evaluate_environment_and_measure(
                rain_detected=False,
                mean_luminance=80.0,
                lidar_depth_cm=33.5,
                sonar_depth_cm=33.4,
                z_accel_g=1.15,
                surface_area_sqm=0.85,
                damage_class=crack_cls,
                bbox=[200.0, 200.0, 350.0, 500.0],
            )
            self.assertIsNone(
                res["calculated_volume_liters"],
                f"Crack class {crack_cls} must have calculated_volume_liters=None",
            )
            self.assertIsNotNone(res["crack_severity_index"])
            self.assertGreaterEqual(res["crack_severity_index"], 0.0)
            self.assertLessEqual(res["crack_severity_index"], 1.0)
            self.assertGreater(res["defect_severity_magnitude"], 0.0)

        pothole_res = switcher.evaluate_environment_and_measure(
            rain_detected=True,
            mean_luminance=25.0,
            lidar_depth_cm=44.0,
            sonar_depth_cm=45.0,
            z_accel_g=2.10,
            surface_area_sqm=0.65,
            damage_class="D40",
            bbox=[220.0, 260.0, 400.0, 460.0],
        )
        self.assertIsNotNone(pothole_res["calculated_volume_liters"])
        self.assertGreater(pothole_res["calculated_volume_liters"], 0.0)
        self.assertIsNone(pothole_res["crack_severity_index"])
        self.assertEqual(
            pothole_res["defect_severity_magnitude"],
            pothole_res["calculated_volume_liters"],
        )

        # 2. Check all generated SRMD records
        srmd_path = PROJECT_ROOT / "data" / "processed" / "srmd" / "srmd_records.json"
        srmd_records = json.loads(srmd_path.read_text(encoding="utf-8"))
        for r in srmd_records:
            cls_code = r["source"]["annotation_class"]
            vol = r["visual"]["calculated_volume_liters"]
            csi = r["visual"]["crack_severity_index"]
            if cls_code == "D40":
                self.assertIsNotNone(vol, f"Pothole {r['event_id']} must have volume")
                self.assertGreater(vol, 0.0)
                self.assertIsNone(csi, f"Pothole {r['event_id']} must have crack_severity_index=None")
            else:
                self.assertIsNone(vol, f"Crack {r['event_id']} ({cls_code}) must not have volume")
                self.assertIsNotNone(csi, f"Crack {r['event_id']} ({cls_code}) must have crack_severity_index")
                self.assertTrue(0.0 <= csi <= 1.0)

    def test_ground_plane_area_clamping_and_warning_log(self):
        from services.data_generation.consistency_model import estimate_ground_plane_area_pinhole

        # 1. Oversized pothole bbox [xmin, ymin, xmax, ymax] -> must clamp to 1.00 m^2 and emit WARNING log
        with self.assertLogs("services.data_generation.consistency_model", level="WARNING") as cm:
            area_upper = estimate_ground_plane_area_pinhole(
                bbox=[50.0, 100.0, 680.0, 420.0],
                image_dims={"width": 720, "height": 720},
                damage_class="D40",
            )
        self.assertAlmostEqual(area_upper, 1.00, places=4)
        self.assertTrue(any("Clamped ground-plane area for class D40" in msg for msg in cm.output))

        # 2. Tiny 1x1 pixel pothole bbox at bottom of frame -> must clamp to 0.01 m^2 and emit WARNING log
        with self.assertLogs("services.data_generation.consistency_model", level="WARNING") as cm_low:
            area_lower = estimate_ground_plane_area_pinhole(
                bbox=[350.0, 700.0, 351.0, 701.0],
                image_dims={"width": 720, "height": 720},
                damage_class="D40",
            )
        self.assertAlmostEqual(area_lower, 0.01, places=4)
        self.assertTrue(any("Clamped ground-plane area for class D40" in msg for msg in cm_low.output))

        # 3. Moderate pothole bbox in lower-middle foreground -> within [0.01, 1.00] m^2 without clamping
        area_mid = estimate_ground_plane_area_pinhole(
            bbox=[300.0, 520.0, 420.0, 620.0],
            image_dims={"width": 720, "height": 720},
            damage_class="D40",
        )
        self.assertGreater(area_mid, 0.01)
        self.assertLess(area_mid, 1.00)

    def test_sonar_bias_only_under_rain_and_lidar_wet_degradation(self):
        from services.data_generation.sonar_model import SonarModel
        from services.data_generation.lidar_model import LiDARModel

        # 1. Sonar systematic bias and increased noise appear ONLY when rain_detected=True
        dry_errors = []
        wet_errors = []
        for s in range(100):
            sonar_dry = SonarModel(seed=1000 + s).simulate(
                damage_class="D40", true_defect_depth_cm=12.0, rain_detected=False
            )
            sonar_wet = SonarModel(seed=1000 + s).simulate(
                damage_class="D40", true_defect_depth_cm=12.0, rain_detected=True
            )
            self.assertEqual(sonar_dry["systematic_bias_cm"], 0.0)
            self.assertLess(sonar_wet["systematic_bias_cm"], 0.0)
            self.assertGreater(sonar_wet["noise_std_cm"], sonar_dry["noise_std_cm"])
            self.assertFalse(sonar_wet["wet_lidar_degradation_applied"])
            dry_errors.append(sonar_dry["sensor_error_cm"])
            wet_errors.append(sonar_wet["sensor_error_cm"])

        self.assertAlmostEqual(float(np.mean(dry_errors)), 0.0, delta=0.12)
        self.assertLess(float(np.mean(wet_errors)), -0.40)
        self.assertGreater(float(np.std(wet_errors)), float(np.std(dry_errors)))

        # 2. JSN-SR04T min-range dead zone (20.0 cm) clamps sub-range transducer distance
        sm_dead = SonarModel(seed=42)
        sm_dead.baseline_chassis_cm = 10.0
        dead_zone_res = sm_dead.simulate(
            damage_class="D40", true_defect_depth_cm=2.0, rain_detected=True
        )
        self.assertTrue(dead_zone_res["dead_zone_clamped"])
        self.assertGreaterEqual(dead_zone_res["depth_cm"], 20.0)

        # 3. Wet-surface LiDAR degradation is applied to LiDAR under rain, NOT to Sonar
        lidar_dry = LiDARModel(seed=2026).simulate("D40", surface_area_sqm=0.8, rain_detected=False)
        lidar_wet = LiDARModel(seed=2026).simulate("D40", surface_area_sqm=0.8, rain_detected=True)
        self.assertEqual(lidar_dry["wet_surface_degradation_cm"], 0.0)
        self.assertLess(lidar_wet["wet_surface_degradation_cm"], -1.0)
        self.assertLess(lidar_wet["effective_depth_cm"], lidar_dry["effective_depth_cm"])

    def test_ranking_sanity_large_pothole_outranks_small_crack_and_calibration_note(self):
        import asyncio
        from services.backend.app.services.mcdm_engine import MCDMPrioritizationEngine
        from services.ai_engine.pipeline.depth_estimator import EnvironmentalSensorSwitching
        import main

        switcher = EnvironmentalSensorSwitching(baseline_chassis_height_cm=30.0)
        pothole_metrics = switcher.evaluate_environment_and_measure(
            rain_detected=False,
            mean_luminance=85.0,
            lidar_depth_cm=44.0,
            sonar_depth_cm=44.0,
            z_accel_g=2.1,
            surface_area_sqm=0.75,
            damage_class="D40",
            bbox=[220.0, 260.0, 420.0, 480.0],
        )
        crack_metrics = switcher.evaluate_environment_and_measure(
            rain_detected=False,
            mean_luminance=85.0,
            lidar_depth_cm=31.1,
            sonar_depth_cm=31.1,
            z_accel_g=1.05,
            surface_area_sqm=0.18,
            damage_class="D00",
            bbox=[300.0, 250.0, 320.0, 420.0],
        )
        self.assertIsNotNone(pothole_metrics["calculated_volume_liters"])
        self.assertIsNone(crack_metrics["calculated_volume_liters"])
        self.assertGreater(
            pothole_metrics["defect_severity_magnitude"],
            crack_metrics["defect_severity_magnitude"],
        )

        # Equal spatial & traffic context:
        # traffic=18000 PCU/day, hospital=1.2 km, speed=40 km/h, school=0.8 km, road_class=3 (ARTERIAL)
        equal_context = [18000.0, 1.2, 40.0, 0.8, 3.0]
        decision_matrix = np.array([
            [pothole_metrics["defect_severity_magnitude"], *equal_context],  # Row 0: Large pothole
            [crack_metrics["defect_severity_magnitude"], *equal_context],    # Row 1: Small crack
        ])

        for prof in ("safety-first", "traffic-first", "equity-first"):
            eng = MCDMPrioritizationEngine(profile=prof, use_calibrated_thresholds=True)
            res = eng.compute_all(decision_matrix, v=0.5, calibrated=True)
            self.assertGreater(
                res["topsis_scores"][0],
                res["topsis_scores"][1],
                f"Large pothole must have higher TOPSIS score than small crack under {prof}",
            )
            self.assertEqual(res["topsis_ranks"][0], 1)
            self.assertEqual(res["topsis_ranks"][1], 2)
            self.assertEqual(res["vikor_ranks"][0], 1)
            self.assertEqual(res["vikor_ranks"][1], 2)

        # Verify /api/v1/prioritization includes calibration_note and spans all 4 priority levels
        loop = asyncio.new_event_loop()
        try:
            prio_res = loop.run_until_complete(
                main.get_prioritization(method="both", profile="safety-first", v=0.5)
            )
        finally:
            loop.close()
        self.assertIn("calibration_note", prio_res)
        self.assertIn("not validated against real maintenance outcomes", prio_res["calibration_note"])
        tiers = {r["priority_level"] for r in prio_res["results"]}
        self.assertEqual(tiers, {"Critical", "High", "Medium", "Low"})

    def test_quantile_tiers_and_absolute_alarm_rule(self):
        """
        Task 3: verify (a) batch quantile tiers (top 10% Critical, next 20% High,
        next 30% Medium, rest Low) and (b) fixed absolute alarm rule (TOPSIS >= 0.60 => Critical).
        """
        from services.backend.app.services.mcdm_engine import MCDMPrioritizationEngine
        eng = MCDMPrioritizationEngine(profile="safety-first", use_calibrated_thresholds=True)

        # 1. Ten items with descending scores below 0.60 to test exact quantile partition
        # top 10% (1 item) -> Critical
        # next 20% (2 items) -> High
        # next 30% (3 items) -> Medium
        # rest (4 items) -> Low
        ten_scores = [0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10]
        tiers_10 = eng.classify_priority_batch(ten_scores, calibrated=True)
        self.assertEqual(tiers_10[0], "Critical")
        self.assertEqual(tiers_10[1:3], ["High", "High"])
        self.assertEqual(tiers_10[3:6], ["Medium", "Medium", "Medium"])
        self.assertEqual(tiers_10[6:], ["Low", "Low", "Low", "Low"])

        # 2. Fixed absolute alarm rule: any score >= 0.60 escalates to Critical regardless of quantile
        # Even if at the bottom rank of the batch:
        alarm_scores = [0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.62]
        tiers_alarm = eng.classify_priority_batch(alarm_scores, calibrated=True)
        self.assertEqual(tiers_alarm[-1], "Critical")  # 0.62 escalated to Critical

        # 3. Scalar classify_priority with absolute alarm rule
        self.assertEqual(eng.classify_priority(0.65, calibrated=True), "Critical")
        self.assertEqual(eng.classify_priority(0.60, calibrated=True), "Critical")

        # 4. Calibration note disclosure
        self.assertIn("quantile", eng.calibration_note.lower())
        self.assertIn("0.60", eng.calibration_note)

    def test_mcdm_sensitivity_analysis_script_and_output(self):
        """
        Task 4: verify scripts/run_sensitivity.py runs across 3 profiles x 3 v-values,
        writes runs/experiments/sensitivity.json, and contains per-setting ranks,
        pairwise Spearman rho, and rank-1 frequencies.
        """
        from scripts.run_sensitivity import run_sensitivity_analysis
        output_file = PROJECT_ROOT / "runs" / "experiments" / "sensitivity.json"
        res = run_sensitivity_analysis(output_path=output_file)

        self.assertTrue(output_file.exists())
        self.assertIn("metadata", res)
        self.assertIn("settings", res)
        self.assertEqual(len(res["settings"]), 9)

        # Check each setting structure
        for s in res["settings"]:
            self.assertIn("setting_id", s)
            self.assertIn("profile", s)
            self.assertIn(s["profile"], ("safety-first", "traffic-first", "equity-first"))
            self.assertIn(s["v"], (0.25, 0.50, 0.75))
            self.assertEqual(len(s["topsis_ranks"]), 6)
            self.assertEqual(len(s["vikor_ranks"]), 6)
            self.assertTrue(-1.0 <= s["spearman_rho"] <= 1.0)
            self.assertEqual(len(s["priority_levels"]), 6)

        # Check pairwise spearman matrices
        pw = res["pairwise_spearman"]
        self.assertIn("within_setting_topsis_vs_vikor", pw)
        self.assertIn("across_settings_topsis", pw)
        self.assertIn("across_settings_vikor", pw)
        self.assertEqual(len(pw["within_setting_topsis_vs_vikor"]), 9)

        # Check rank-1 frequencies
        r1 = res["rank_1_frequencies"]
        self.assertEqual(len(r1), 6)
        # On the 6-record demo set, only the two severe highway potholes ever win Rank 1
        pothole_wins = r1["SRMD-IND-000045-00"]["total_rank_1_count"] + r1["SRMD-IND-000278-00"]["total_rank_1_count"]
        self.assertEqual(pothole_wins, 18)  # 9 topsis + 9 vikor = 18 total wins

    def test_stray_and_duplicate_files_audit(self):
        """
        Task 5: verify docs/STRAY_FILES.md documents git ls-files audit,
        including Dashboard casing, main.py vs services/backend, ClientOnly.tsx,
        build_srmd vs generate_srmd, and vercel.json.
        """
        stray_doc = PROJECT_ROOT / "docs" / "STRAY_FILES.md"
        self.assertTrue(stray_doc.exists(), "docs/STRAY_FILES.md must exist")
        text = stray_doc.read_text(encoding="utf-8")
        self.assertIn("ClientOnly.tsx", text)
        self.assertIn("build_srmd.py", text)
        self.assertIn("generate_srmd.py", text)
        self.assertIn("vercel.json", text)
        self.assertIn("Dashboard", text)
        self.assertIn("services/backend", text)
        self.assertIn("main.py", text)

    def test_pothole_volume_recomputed_from_api_displayed_inputs(self):
        """
        Task 2: Verify that for every pothole record in the API/telemetry store,
        the displayed calculated_volume_liters can be exactly reproduced
        from the displayed inputs (surface_area_sqm and estimated_depth_cm)
        using the paraboloid cavity formula V = 0.5 * Area * Depth * 1000.
        """
        from main import telemetry_store, _load_srmd_records_into_store
        _load_srmd_records_into_store()
        for rec in telemetry_store:
            pt = rec.get("processed_telemetry", {})
            dmg_cls = rec.get("damage_class", "")
            if dmg_cls == "D40":
                area_m2 = rec["surface_area_sqm"]
                depth_cm = pt["estimated_depth_cm"]
                vol_liters = pt["calculated_volume_liters"]
                expected_vol = round(0.5 * area_m2 * (depth_cm / 100.0) * 1000.0, 2)
                self.assertIsNotNone(vol_liters)
                self.assertAlmostEqual(
                    vol_liters,
                    expected_vol,
                    places=2,
                    msg=f"Record {rec['id']} displayed volume {vol_liters} does not match 0.5 * {area_m2} * ({depth_cm}/100) * 1000 = {expected_vol}"
                )
            else:
                self.assertIsNone(pt.get("calculated_volume_liters"))

    def test_frontend_fallback_sync_with_srmd(self):
        """
        Freeze Task 1: Verify frontend FALLBACK data is 100% synchronized with
        data/processed/srmd/srmd_records.json.
        Asserts identical IDs, volume/severity magnitudes, TOPSIS scores, and priority tiers
        between backend SRMD store items and frontend fallback records.
        Also asserts concordance of fallback_profile_map across all 3 weight profiles.
        """
        import main as app_module
        from services.backend.app.services.mcdm_engine import (
            MCDMPrioritizationEngine,
            build_mcdm_row_from_store_item,
        )

        app_module._load_srmd_records_into_store()
        store_records = {item["id"]: item for item in app_module.telemetry_store}

        fallback_file = PROJECT_ROOT / "frontend" / "src" / "data" / "fallback_records.json"
        self.assertTrue(fallback_file.exists(), f"Missing fallback file {fallback_file}")
        fallback_records = json.loads(fallback_file.read_text(encoding="utf-8"))
        fallback_by_id = {item["id"]: item for item in fallback_records}

        self.assertEqual(len(fallback_records), len(store_records))
        self.assertEqual(set(fallback_by_id.keys()), set(store_records.keys()))

        for rec_id, store_item in store_records.items():
            fb_item = fallback_by_id[rec_id]

            # 1. ID check
            self.assertEqual(fb_item["id"], store_item["id"])

            # 2. Volume / Severity check
            fb_pt = fb_item["processed_telemetry"]
            st_pt = store_item["processed_telemetry"]

            if store_item.get("damage_class") == "D40":
                self.assertIsNotNone(fb_pt["calculated_volume_liters"])
                self.assertAlmostEqual(
                    fb_pt["calculated_volume_liters"],
                    st_pt["calculated_volume_liters"],
                    places=2,
                    msg=f"Volume mismatch on {rec_id}"
                )
            else:
                self.assertIsNone(fb_pt.get("calculated_volume_liters"))
                self.assertIsNone(st_pt.get("calculated_volume_liters"))

            self.assertAlmostEqual(
                fb_pt["defect_severity_magnitude"],
                st_pt["defect_severity_magnitude"],
                places=2,
                msg=f"Severity magnitude mismatch on {rec_id}"
            )

            # 3. TOPSIS score check
            self.assertAlmostEqual(
                fb_item["topsis_score"],
                store_item["topsis_score"],
                places=4,
                msg=f"TOPSIS score mismatch on {rec_id}"
            )

            # 4. Priority tier check
            self.assertEqual(
                fb_item["priority_level"],
                store_item["priority_level"],
                msg=f"Priority tier mismatch on {rec_id}"
            )

            # 5. Ranks check
            self.assertEqual(fb_item["topsis_rank"], store_item["topsis_rank"])
            self.assertEqual(fb_item["vikor_rank"], store_item["vikor_rank"])

        # 6. Fallback profile map check across all three weight profiles
        profile_map_file = PROJECT_ROOT / "frontend" / "src" / "data" / "fallback_profile_map.json"
        self.assertTrue(profile_map_file.exists(), f"Missing profile map file {profile_map_file}")
        profile_map = json.loads(profile_map_file.read_text(encoding="utf-8"))

        matrix = np.array([build_mcdm_row_from_store_item(app_module.telemetry_store[i]) for i in range(len(app_module.telemetry_store))])
        for profile in ("safety-first", "traffic-first", "equity-first"):
            self.assertIn(profile, profile_map)
            engine = MCDMPrioritizationEngine(profile=profile, use_calibrated_thresholds=True)
            res = engine.compute_all(matrix, v=0.5, calibrated=True)
            for i, st_item in enumerate(app_module.telemetry_store):
                p_entry = profile_map[profile][st_item["id"]]
                self.assertEqual(p_entry["topsis_rank"], int(res["topsis_ranks"][i]))
                self.assertAlmostEqual(p_entry["topsis_score"], float(res["topsis_scores"][i]), places=4)
                self.assertEqual(p_entry["vikor_rank"], int(res["vikor_ranks"][i]))
                self.assertEqual(p_entry["priority_level"], str(res["priority_levels"][i]))


if __name__ == "__main__":
    unittest.main()





