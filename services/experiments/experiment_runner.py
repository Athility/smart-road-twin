"""
Sensor Fusion Experiments & Comparative Benchmark Engine
=========================================================

Systematic comparative evaluation of 5 sensing configurations:
- EXPERIMENT A: Vision Only (Monocular camera)
- EXPERIMENT B: Vision + Accelerometer (Camera + vertical dynamic shock)
- EXPERIMENT C: Vision + LiDAR (Camera + optical depth profilometry)
- EXPERIMENT D: Vision + LiDAR + Sonar + Accelerometer (Physical sensor suite)
- EXPERIMENT E: Full Multimodal System (Physical suite + Environmental Switching + Municipal GIS Context)

Quantitative Evaluation Metrics:
1. Detection Confirmation Rate: Fraction of visual detections physically corroborated.
2. Unconfirmed / Superficial Rate: Fraction of visual distress lacking physical depth/shock.
3. Impact Confirmation Rate: Acute vertical chassis impact verification (z > 1.50g).
4. TOPSIS Priority Stability: Spearman rank correlation with full multimodal ground truth.
5. Sensor Disagreement Rate: Divergent signals between modalities.
"""

import math
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from services.backend.app.services.mcdm_engine import MCDMPrioritizationEngine
from services.fusion.disagreement_analyzer import SensorDisagreementAnalyzer, DisagreementCategory


def compute_spearman_rank_correlation(ranks_a: List[float], ranks_b: List[float]) -> float:
    """Computes Spearman's rank correlation coefficient between two rankings."""
    if len(ranks_a) < 2 or len(ranks_b) < 2:
        return 1.0
    n = len(ranks_a)
    d_sq = sum((ra - rb) ** 2 for ra, rb in zip(ranks_a, ranks_b))
    rho = 1.0 - (6.0 * d_sq) / (n * (n ** 2 - 1))
    return round(float(np.clip(rho, -1.0, 1.0)), 4)


class FusionExperimentRunner:
    """
    Executes controlled comparative experiments evaluating sensor fusion
    configurations against authentic/synthetic road defect records.
    """

    EXPERIMENT_CONFIGS = {
        "A": {
            "name": "EXPERIMENT A: Vision Only",
            "has_camera": True,
            "has_accelerometer": False,
            "has_lidar": False,
            "has_sonar": False,
            "has_switching": False,
            "has_gis_context": False,
            "description": "Monocular 2D bounding boxes only. Lacks depth and acceleration sensors."
        },
        "B": {
            "name": "EXPERIMENT B: Vision + Accelerometer",
            "has_camera": True,
            "has_accelerometer": True,
            "has_lidar": False,
            "has_sonar": False,
            "has_switching": False,
            "has_gis_context": False,
            "description": "Camera coupled with quarter-car vertical acceleration. Verifies dynamic wheel shock."
        },
        "C": {
            "name": "EXPERIMENT C: Vision + LiDAR",
            "has_camera": True,
            "has_accelerometer": False,
            "has_lidar": True,
            "has_sonar": False,
            "has_switching": False,
            "has_gis_context": False,
            "description": "Camera coupled with optical LiDAR depth. Vulnerable to rain / standing water scatter."
        },
        "D": {
            "name": "EXPERIMENT D: Vision + LiDAR + Sonar + Accelerometer",
            "has_camera": True,
            "has_accelerometer": True,
            "has_lidar": True,
            "has_sonar": True,
            "has_switching": True,
            "has_gis_context": False,
            "description": "Complete physical sensor suite with environmental switching. Lacks urban GIS context."
        },
        "E": {
            "name": "EXPERIMENT E: Full Multimodal System",
            "has_camera": True,
            "has_accelerometer": True,
            "has_lidar": True,
            "has_sonar": True,
            "has_switching": True,
            "has_gis_context": True,
            "description": "Physical sensor suite + environmental switching + municipal GIS & TOPSIS prioritization."
        }
    }

    def __init__(self, srmd_records_path: Optional[Path] = None):
        self.records_path = Path(srmd_records_path) if srmd_records_path else PROJECT_ROOT / "data" / "processed" / "srmd" / "srmd_records.json"
        self.topsis_engine = MCDMPrioritizationEngine()
        self.analyzer = SensorDisagreementAnalyzer(impact_threshold_g=1.50)

    def load_dataset_records(self) -> List[Dict[str, Any]]:
        """Loads records from SRMD records file or samples directory."""
        if self.records_path.exists():
            with open(self.records_path, "r", encoding="utf-8") as f:
                return json.load(f)

        records_dir = PROJECT_ROOT / "data" / "processed" / "srmd" / "records"
        if records_dir.exists():
            records = []
            for rf in sorted(list(records_dir.glob("*.json"))):
                with open(rf, "r", encoding="utf-8") as f:
                    records.append(json.load(f))
            if records:
                return records

        raise FileNotFoundError(f"SRMD records not found at {self.records_path} or records directory.")

    def run_all_experiments(self, records: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Executes Experiments A, B, C, D, and E across the dataset and compiles
        measurable comparative outcomes.
        """
        dataset = records if records is not None else self.load_dataset_records()
        if not dataset:
            raise ValueError("Dataset is empty. Cannot run experiments.")

        num_records = len(dataset)

        # 1. Compute reference ranking using Experiment E (Ground Truth Multimodal)
        ref_matrix = []
        for r in dataset:
            vol = float(r.get("visual", {}).get("calculated_volume_liters", 10.0))
            pcu = float(r.get("traffic", {}).get("pcu", 15000))
            dist = float(r.get("infrastructure", {}).get("dist_hospital_km", 3.0))
            spd = float(r.get("accelerometer", {}).get("vehicle_speed_kmh", 40.0))
            ref_matrix.append([vol, pcu, dist, spd])

        ref_scores = self.topsis_engine.compute_topsis(np.array(ref_matrix))
        # 1-based ranks
        ref_rank_pairs = sorted([(ref_scores[i], i) for i in range(num_records)], reverse=True)
        ref_ranks = [0] * num_records
        for rk, (_, idx) in enumerate(ref_rank_pairs):
            ref_ranks[idx] = rk + 1

        # 2. Evaluate each configuration
        results = {}

        for code in ["A", "B", "C", "D", "E"]:
            cfg = self.EXPERIMENT_CONFIGS[code]
            exp_metrics = self._evaluate_configuration(code, cfg, dataset, ref_ranks)
            results[code] = exp_metrics

        summary = {
            "dataset_size": num_records,
            "experiments": results,
            "benchmark_date": "2026-09-24",
            "topsis_weights": [0.35, 0.25, 0.25, 0.15],
            "impact_threshold_g": 1.50
        }
        return summary

    def _evaluate_configuration(
        self,
        exp_code: str,
        cfg: Dict[str, Any],
        dataset: List[Dict[str, Any]],
        ref_ranks: List[int]
    ) -> Dict[str, Any]:
        """Evaluates a single experiment configuration over the dataset."""
        n = len(dataset)
        confirmed_detections = 0
        unconfirmed_superficial = 0
        impact_confirmed_count = 0
        sensor_disagreements = 0
        critical_count = 0

        exp_matrix_rows = []

        for r in dataset:
            dmg_cls = r.get("source", {}).get("annotation_class", "D40")
            area = float(r.get("visual", {}).get("surface_area_sqm", 0.5))
            raw_vol = float(r.get("visual", {}).get("calculated_volume_liters", 10.0))
            z_g = float(r.get("accelerometer", {}).get("z_accel_g", 1.0))
            is_rain = bool(r.get("environment", {}).get("rain_detected", False))
            lidar_d = float(r.get("lidar", {}).get("effective_depth_cm", 5.0))
            sonar_d = float(r.get("sonar", {}).get("effective_depth_cm", 5.0))
            pcu = float(r.get("traffic", {}).get("pcu", 15000))
            dist = float(r.get("infrastructure", {}).get("dist_hospital_km", 3.0))
            spd = float(r.get("accelerometer", {}).get("vehicle_speed_kmh", 40.0))

            # Configuration-specific feature resolution
            if exp_code == "A":
                # Vision Only: Volume is estimated from 2D bounding box (heuristic)
                # No physical depth or acceleration confirmation possible
                est_vol = round(area * 15.0, 2)
                exp_pcu = 10000.0  # Default uncontextualized
                exp_dist = 5.0     # Default uncontextualized
                exp_spd = 40.0
                has_physical_confirmation = False
                is_superficial = True  # Cannot verify whether superficial or deep
                has_impact = False

            elif exp_code == "B":
                # Vision + Accelerometer
                est_vol = round(area * 15.0, 2)
                exp_pcu = 10000.0
                exp_dist = 5.0
                exp_spd = spd
                has_impact = bool(z_g > 1.50)
                has_physical_confirmation = has_impact
                is_superficial = bool(z_g <= 1.15)

            elif exp_code == "C":
                # Vision + Optical LiDAR (fails/degrades in rain)
                effective_d = 0.5 if is_rain else lidar_d  # Optical scatter in rain
                est_vol = 0.5 * (area * 10000.0) * effective_d / 1000.0
                exp_pcu = 10000.0
                exp_dist = 5.0
                exp_spd = 40.0
                has_physical_confirmation = bool(effective_d >= 2.0)
                is_superficial = bool(effective_d < 1.0)
                has_impact = False

            elif exp_code == "D":
                # Vision + LiDAR + Sonar + Accelerometer (No GIS Context)
                effective_d = sonar_d if is_rain else lidar_d
                est_vol = 0.5 * (area * 10000.0) * effective_d / 1000.0
                exp_pcu = 10000.0
                exp_dist = 5.0
                exp_spd = spd
                has_impact = bool(z_g > 1.50)
                has_physical_confirmation = bool(effective_d >= 2.0 or has_impact)
                is_superficial = bool(effective_d < 1.0 and z_g <= 1.15)

            else:  # "E"
                # Full Multimodal System
                effective_d = sonar_d if is_rain else lidar_d
                est_vol = raw_vol
                exp_pcu = pcu
                exp_dist = dist
                exp_spd = spd
                has_impact = bool(z_g > 1.50)
                has_physical_confirmation = bool(effective_d >= 2.0 or has_impact)
                is_superficial = bool(effective_d < 1.0 and z_g <= 1.15)

            # Analyze multi-sensor disagreement
            active_mod = "sonar_submerged_acoustic" if is_rain and cfg["has_sonar"] else "optical_lidar"
            ev_sum = self.analyzer.analyze_evidence(
                damage_class=dmg_cls,
                visual_confidence=0.92,
                surface_area_sqm=area,
                effective_depth_cm=effective_d if "effective_d" in locals() else 0.0,
                active_modality=active_mod,
                z_accel_g=z_g if cfg["has_accelerometer"] else 1.00,
                rain_detected=is_rain,
                mean_luminance=25.0 if is_rain else 65.0
            )

            if ev_sum.disagreement_category not in [DisagreementCategory.CONCORDANT_CONFIRMED, DisagreementCategory.CONCORDANT_NOMINAL]:
                sensor_disagreements += 1

            if has_physical_confirmation:
                confirmed_detections += 1
            if is_superficial:
                unconfirmed_superficial += 1
            if has_impact:
                impact_confirmed_count += 1

            exp_matrix_rows.append([est_vol, exp_pcu, exp_dist, exp_spd])

        # Compute TOPSIS for this experiment
        exp_matrix = np.array(exp_matrix_rows)
        exp_scores = self.topsis_engine.compute_topsis(exp_matrix)

        # 1-based ranks for this experiment
        exp_rank_pairs = sorted([(exp_scores[i], i) for i in range(n)], reverse=True)
        exp_ranks = [0] * n
        for rk, (_, idx) in enumerate(exp_rank_pairs):
            exp_ranks[idx] = rk + 1

        for sc in exp_scores:
            if self.topsis_engine.classify_priority(sc) == "Critical":
                critical_count += 1

        # Spearman rank correlation against Reference (Experiment E)
        stability = compute_spearman_rank_correlation(exp_ranks, ref_ranks)

        return {
            "name": cfg["name"],
            "description": cfg["description"],
            "total_defects_evaluated": n,
            "detection_confirmation_rate": round(float(confirmed_detections / n), 4),
            "unconfirmed_superficial_rate": round(float(unconfirmed_superficial / n), 4),
            "impact_confirmation_rate": round(float(impact_confirmed_count / n), 4),
            "sensor_disagreement_rate": round(float(sensor_disagreements / n), 4),
            "topsis_priority_stability": stability,
            "mean_topsis_score": round(float(np.mean(exp_scores)), 4),
            "critical_priority_count": critical_count
        }

    def format_results_markdown_table(self, summary: Dict[str, Any]) -> str:
        """Formats the comparative experiment results into a clean markdown table."""
        lines = [
            "| Configuration | Confirmation Rate | Unconfirmed Superficial | Impact Confirmation | TOPSIS Stability (rho) | Disagreement Rate | Critical Count |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
        ]
        for code in ["A", "B", "C", "D", "E"]:
            exp = summary["experiments"][code]
            name_short = exp["name"].replace("EXPERIMENT ", "Exp ")
            lines.append(
                f"| **{name_short}** | {exp['detection_confirmation_rate']*100:.1f}% | "
                f"{exp['unconfirmed_superficial_rate']*100:.1f}% | {exp['impact_confirmation_rate']*100:.1f}% | "
                f"**{exp['topsis_priority_stability']:.4f}** | {exp['sensor_disagreement_rate']*100:.1f}% | "
                f"{exp['critical_priority_count']} |"
            )
        return "\n".join(lines)
