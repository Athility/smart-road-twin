#!/usr/bin/env python3
"""
MCDM Sensitivity Analysis: Weight Profiles x VIKOR Compromise Parameter (v)
==========================================================================
Systematic sensitivity evaluation of TOPSIS and VIKOR ranking stability:
- 3 Fuzzy BWM weight profiles: 'safety-first', 'traffic-first', 'equity-first'
- 3 VIKOR consensus parameters: v in {0.25, 0.50, 0.75}

Outputs:
- Per-setting ranks (TOPSIS, VIKOR)
- Pairwise Spearman rho (between methods and across settings)
- Rank-1 frequency per record
- Output saved to runs/experiments/sensitivity.json
"""

import sys
import os
import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.backend.app.services.mcdm_engine import (
    MCDMPrioritizationEngine,
    build_mcdm_row_from_srmd,
    NAMED_BWM_PROFILES,
    _spearman_rank_correlation,
)


def run_sensitivity_analysis(
    srmd_file: Path = PROJECT_ROOT / "data" / "processed" / "srmd" / "srmd_records.json",
    output_path: Path = PROJECT_ROOT / "runs" / "experiments" / "sensitivity.json"
) -> Dict[str, Any]:
    """
    Executes the 3 weight profiles x 3 v-values MCDM sensitivity experiment.
    """
    if not srmd_file.exists():
        raise FileNotFoundError(f"SRMD records file not found at: {srmd_file}")

    records = json.loads(srmd_file.read_text(encoding="utf-8"))
    n_records = len(records)
    record_ids = [r["event_id"] for r in records]
    record_labels = [
        f"{r['event_id']} ({r.get('source', {}).get('annotation_class', 'D40')} @ {r.get('location', {}).get('road_name', 'WEH')[:25]})"
        for r in records
    ]

    matrix = np.array([build_mcdm_row_from_srmd(r) for r in records])

    profiles = ["safety-first", "traffic-first", "equity-first"]
    v_values = [0.25, 0.50, 0.75]

    settings: List[Dict[str, Any]] = []
    setting_keys: List[str] = []

    # 1. Evaluate all 9 settings
    for profile in profiles:
        for v in v_values:
            key = f"{profile}_v{v:.2f}"
            setting_keys.append(key)
            eng = MCDMPrioritizationEngine(profile=profile, use_calibrated_thresholds=True)
            res = eng.compute_all(matrix, v=v, calibrated=True)

            setting_entry = {
                "setting_id": key,
                "profile": profile,
                "v": float(v),
                "weights": [round(float(w), 4) for w in res["weights"]],
                "topsis_scores": [round(float(s), 4) for s in res["topsis_scores"]],
                "topsis_ranks": res["topsis_ranks"],
                "vikor_S": [round(float(s), 4) for s in res["vikor_S"]],
                "vikor_R": [round(float(r), 4) for r in res["vikor_R"]],
                "vikor_Q": [round(float(q), 4) for q in res["vikor_Q"]],
                "vikor_ranks": res["vikor_ranks"],
                "spearman_rho": float(res["spearman_correlation"]),
                "priority_levels": res["priority_levels"],
                "compromise_solution": [record_ids[i] for i in res["vikor"]["compromise_solution"]]
            }
            settings.append(setting_entry)

    # 2. Pairwise Spearman rho across settings
    # (a) TOPSIS pairwise across the 9 settings
    pairwise_spearman_topsis: Dict[str, Dict[str, float]] = {}
    # (b) VIKOR pairwise across the 9 settings
    pairwise_spearman_vikor: Dict[str, Dict[str, float]] = {}
    # (c) In-setting TOPSIS vs VIKOR
    in_setting_rho: Dict[str, float] = {}

    for s1 in settings:
        k1 = s1["setting_id"]
        pairwise_spearman_topsis[k1] = {}
        pairwise_spearman_vikor[k1] = {}
        in_setting_rho[k1] = s1["spearman_rho"]
        for s2 in settings:
            k2 = s2["setting_id"]
            rho_t = _spearman_rank_correlation(s1["topsis_ranks"], s2["topsis_ranks"])
            rho_v = _spearman_rank_correlation(s1["vikor_ranks"], s2["vikor_ranks"])
            pairwise_spearman_topsis[k1][k2] = float(rho_t)
            pairwise_spearman_vikor[k1][k2] = float(rho_v)

    # 3. Rank-1 Frequency per record
    rank_1_stats: Dict[str, Dict[str, Any]] = {}
    for idx, rec_id in enumerate(record_ids):
        t_count = sum(1 for s in settings if s["topsis_ranks"][idx] == 1)
        v_count = sum(1 for s in settings if s["vikor_ranks"][idx] == 1)
        total_evals = len(settings)
        cls_name = records[idx].get("source", {}).get("annotation_class", "D40")
        rank_1_stats[rec_id] = {
            "record_id": rec_id,
            "label": record_labels[idx],
            "damage_class": cls_name,
            "topsis_rank_1_count": t_count,
            "topsis_rank_1_frequency": round(t_count / float(total_evals), 4),
            "vikor_rank_1_count": v_count,
            "vikor_rank_1_frequency": round(v_count / float(total_evals), 4),
            "total_rank_1_count": t_count + v_count,
            "overall_rank_1_frequency": round((t_count + v_count) / float(2 * total_evals), 4)
        }

    # 4. Construct final payload
    payload: Dict[str, Any] = {
        "metadata": {
            "experiment": "MCDM Multi-Criteria Sensitivity & Ranking Stability",
            "dataset": "SRMD (Smart Road Multimodal Dataset) Western Express Corridor Demo Set",
            "num_records": n_records,
            "profiles_evaluated": profiles,
            "v_values": v_values,
            "total_configurations": len(settings),
            "honesty_disclosure": (
                "Evaluated on a 6-record demonstration set. While suitable for validating algorithm "
                "invariants, weight sensitivities, and method concordance, statistical rank correlation "
                "is sensitive to individual outliers in small sample sizes (n=6) and should be interpreted "
                "as a functional proof-of-concept rather than an operational municipal dispatch validation."
            )
        },
        "records": record_ids,
        "record_labels": record_labels,
        "settings": settings,
        "pairwise_spearman": {
            "within_setting_topsis_vs_vikor": in_setting_rho,
            "across_settings_topsis": pairwise_spearman_topsis,
            "across_settings_vikor": pairwise_spearman_vikor
        },
        "rank_1_frequencies": rank_1_stats
    }

    # 5. Write to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[Sensitivity] Analysis complete. Saved {len(settings)} configurations to {output_path}")

    return payload


def main():
    res = run_sensitivity_analysis()
    print("\n" + "=" * 80)
    print("MCDM SENSITIVITY BENCHMARK SUMMARY (3 Profiles x 3 v-values)")
    print("=" * 80)
    print(f"{'Setting':<22} {'v':<6} {'TOPSIS Ranks':<20} {'VIKOR Ranks':<20} {'Spearman rho':<12}")
    print("-" * 80)
    for s in res["settings"]:
        t_str = str(s["topsis_ranks"])
        v_str = str(s["vikor_ranks"])
        print(f"{s['setting_id']:<22} {s['v']:<6.2f} {t_str:<20} {v_str:<20} {s['spearman_rho']:<12.4f}")

    print("\n" + "=" * 80)
    print("RANK-1 FREQUENCY PER RECORD (out of 9 evaluations per method)")
    print("=" * 80)
    print(f"{'Record ID':<22} {'Damage':<8} {'TOPSIS R1':<12} {'VIKOR R1':<12} {'Overall Freq':<12}")
    print("-" * 80)
    for r_id, stats in res["rank_1_frequencies"].items():
        print(
            f"{r_id:<22} {stats['damage_class']:<8} "
            f"{stats['topsis_rank_1_count']}/9 ({stats['topsis_rank_1_frequency']*100:.0f}%)   "
            f"{stats['vikor_rank_1_count']}/9 ({stats['vikor_rank_1_frequency']*100:.0f}%)   "
            f"{stats['overall_rank_1_frequency']*100:.1f}%"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
