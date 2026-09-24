#!/usr/bin/env python
"""
RDD2022 Road Damage Model Evaluation CLI
========================================

Evaluates road damage detection models against RDD2022 visual ground truth.
Reports:
- Precision & Recall
- mAP50 & mAP50-95
- Class-wise performance (D00, D10, D20, D40)
- Confusion Matrix

Usage:
    python scripts/evaluate_damage_model.py --split test
    python scripts/evaluate_damage_model.py --split test --conf 0.25 --iou 0.50
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.evaluate.evaluator import DamageModelEvaluator
from services.cv.rdd2022_detector import RDD2022Detector


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RDD2022 Road Damage Model")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Optional path to custom model weights"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for predictions"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.50,
        help="IoU threshold for ground truth matching"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "runs" / "evaluate" / "eval_report.json"),
        help="Path to save evaluation report JSON"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of images to evaluate"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print("RDD2022 Road Damage Model Evaluation")
    print("=" * 70)
    print(f"Split:          {args.split}")
    print(f"Conf Threshold: {args.conf}")
    print(f"IoU Threshold:  {args.iou}")
    if args.weights:
        print(f"Weights:        {args.weights}")

    detector = RDD2022Detector(
        weights_path=args.weights,
        default_conf_threshold=args.conf,
        default_iou_threshold=args.iou
    )

    evaluator = DamageModelEvaluator(
        detector=detector,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        project_root=PROJECT_ROOT
    )

    print("\nRunning evaluation against ground truth...")
    results = evaluator.evaluate_split(split_name=args.split, limit=args.limit)

    print("\n" + "=" * 70)
    print("OVERALL DETECTION PERFORMANCE")
    print("=" * 70)
    print(f"Images Evaluated:     {results['total_images_evaluated']}")
    print(f"Ground Truth Boxes:   {results['total_ground_truth_boxes']}")
    print(f"Predicted Boxes:      {results['total_predicted_boxes']}")
    print(f"Precision:            {results['precision']:.4f}")
    print(f"Recall:               {results['recall']:.4f}")
    print(f"mAP50:                {results['mAP50']:.4f}")
    print(f"mAP50-95:             {results['mAP50_95']:.4f}")

    print("\n" + "=" * 70)
    print("CLASS-WISE PERFORMANCE BREAKDOWN")
    print("=" * 70)
    print(f"{'Class':<8} {'TP':>6} {'FP':>6} {'FN':>6} {'Precision':>12} {'Recall':>10} {'F1-Score':>10}")
    print("-" * 70)
    for cls_name, c_stats in results["class_metrics"].items():
        print(
            f"{cls_name:<8} {c_stats['true_positives']:>6} {c_stats['false_positives']:>6} "
            f"{c_stats['false_negatives']:>6} {c_stats['precision']:>12.4f} "
            f"{c_stats['recall']:>10.4f} {c_stats['f1_score']:>10.4f}"
        )

    print("\n" + "=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)
    cm_str = evaluator.format_confusion_matrix_ascii(
        results["confusion_matrix"],
        results["confusion_matrix_labels"]
    )
    print(cm_str)

    # Save report
    evaluator.save_report(results, args.output)
    print(f"\nDetailed report saved to: {args.output}")


if __name__ == "__main__":
    main()
