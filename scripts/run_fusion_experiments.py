#!/usr/bin/env python
"""
Sensor Fusion Experiments & Comparative Benchmark CLI
======================================================

Executes controlled multi-modal experiments across the SRMD dataset:
- EXPERIMENT A: Vision Only
- EXPERIMENT B: Vision + Accelerometer
- EXPERIMENT C: Vision + LiDAR
- EXPERIMENT D: Vision + LiDAR + Sonar + Accelerometer
- EXPERIMENT E: Full Multimodal System

Outputs measurable research metrics:
- Detection confirmation rate
- Unconfirmed / superficial rate
- Impact confirmation rate
- TOPSIS priority stability (Spearman ρ)
- Sensor disagreement rate

Usage:
    python scripts/run_fusion_experiments.py
    python scripts/run_fusion_experiments.py --output runs/experiments/benchmark.json
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from services.experiments.experiment_runner import FusionExperimentRunner


def parse_args():
    parser = argparse.ArgumentParser(description="Run Sensor Fusion Research Experiments")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed" / "srmd" / "srmd_records.json"),
        help="Path to SRMD records JSON dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "runs" / "experiments" / "fusion_benchmark.json"),
        help="Path to save benchmark results JSON"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 80)
    print("SMART ROAD DIGITAL TWIN - SENSOR FUSION COMPARATIVE BENCHMARK")
    print("=" * 80)
    print(f"Dataset Path: {args.dataset}")
    print(f"Output File:  {args.output}")

    runner = FusionExperimentRunner(srmd_records_path=Path(args.dataset))

    print("\nExecuting Experiments A through E across dataset records...")
    results = runner.run_all_experiments()

    print(f"\nSuccessfully evaluated {results['dataset_size']} multimodal defect records.")

    print("\n" + "=" * 80)
    print("EXPERIMENTAL CONFIGURATIONS EVALUATED")
    print("=" * 80)
    for code in ["A", "B", "C", "D", "E"]:
        exp = results["experiments"][code]
        print(f"[{code}] {exp['name']}")
        print(f"    {exp['description']}")

    print("\n" + "=" * 80)
    print("QUANTITATIVE BENCHMARK COMPARISON TABLE")
    print("=" * 80)
    table_md = runner.format_results_markdown_table(results)
    print(table_md)

    print("\n" + "=" * 80)
    print("DETAILED CONFIGURATION BREAKDOWN")
    print("=" * 80)
    for code in ["A", "B", "C", "D", "E"]:
        exp = results["experiments"][code]
        print(f"\n--- {exp['name']} ---")
        print(f"  Confirmation Rate:        {exp['detection_confirmation_rate']*100:.1f}%")
        print(f"  Unconfirmed Superficial:  {exp['unconfirmed_superficial_rate']*100:.1f}%")
        print(f"  Dynamic Impact Confirmed: {exp['impact_confirmation_rate']*100:.1f}%")
        print(f"  Sensor Disagreements:     {exp['sensor_disagreement_rate']*100:.1f}%")
        print(f"  TOPSIS Stability (rho):   {exp['topsis_priority_stability']:.4f}")
        print(f"  Mean TOPSIS Score:        {exp['mean_topsis_score']:.4f}")
        print(f"  Critical Triage Defects:  {exp['critical_priority_count']}")

    # Save results
    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Benchmark results successfully exported to: {out_p}")
    print("=" * 80)


if __name__ == "__main__":
    main()
