#!/usr/bin/env python
"""
RDD2022 Road Damage Model Training CLI
======================================

Executes visual road damage detection training on authentic RDD2022 India data.

Usage:
    python scripts/train_damage_model.py
    python scripts/train_damage_model.py --epochs 30 --seed 2026 --output runs/train/exp1
    python scripts/train_damage_model.py --prepare-data
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.train.trainer import DamageModelTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train RDD2022 Road Damage Detector")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "ml" / "configs" / "training_config.yaml"),
        help="Path to training configuration YAML"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "runs" / "train" / "exp"),
        help="Directory to save training run artifacts"
    )
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="Prepare YOLO dataset from raw Pascal VOC annotations before training"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print("RDD2022 Road Damage Model Training Pipeline")
    print("=" * 70)
    print(f"Config:      {args.config}")
    print(f"Seed:        {args.seed}")
    print(f"Output Dir:  {args.output}")

    trainer = DamageModelTrainer(config_path=args.config, project_root=PROJECT_ROOT)
    trainer.seed = args.seed
    if args.batch_size:
        trainer.batch_size = args.batch_size

    if args.prepare_data:
        print("\n[1/2] Preparing YOLO dataset split from RDD2022 Pascal VOC annotations...")
        manifest = trainer.prepare_data()
        print(f"      Total images: {manifest['total_images']}")
        print(f"      Splits: {manifest['splits']}")
        print(f"      Leakage verification: {manifest['leakage_verification']}")

    print("\n[2/2] Launching model training...")
    results = trainer.train(output_dir=Path(args.output), epochs_override=args.epochs)

    print("\n" + "=" * 70)
    print("TRAINING RUN SUMMARY")
    print("=" * 70)
    print(f"Status:          {results.get('status')}")
    print(f"Engine:          {results.get('engine')}")
    print(f"Epochs:          {results.get('epochs')}")
    print(f"Architecture:    {results.get('architecture')}")
    if "best_epoch" in results:
        print(f"Best Epoch:      {results.get('best_epoch')}")
        print(f"Best mAP50:      {results.get('best_mAP50'):.4f}")
        print(f"Final Precision: {results.get('final_precision'):.4f}")
        print(f"Final Recall:    {results.get('final_recall'):.4f}")
    print(f"Weights Dir:     {results.get('weights_dir')}")
    if results.get("best_weights"):
        print(f"Best Weights:    {results.get('best_weights')}")
    print("=" * 70)
    print("Training run completed successfully.")


if __name__ == "__main__":
    main()
