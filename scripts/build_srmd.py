"""
Build Smart Road Multimodal Dataset (SRMD) CLI
==============================================

Builds the derived multimodal dataset from prepared RDD2022 India annotations.

Usage:
  python scripts/build_srmd.py [--source data/processed/annotations_index.json] [--output data/processed/srmd] [--seed 42] [--inspect]
"""

import sys
import argparse
import json
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from services.dataset.srmd_builder import build_srmd_dataset, SRMD_DIR, PROCESSED_DIR


def main():
    parser = argparse.ArgumentParser(description="Generate Smart Road Multimodal Dataset (SRMD)")
    parser.add_argument(
        "--source",
        type=str,
        default=str(PROCESSED_DIR / "annotations_index.json"),
        help="Path to annotations_index.json"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(SRMD_DIR),
        help="Output directory for SRMD records"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Deterministic random seed for sensor synthesis"
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print a sample synthesized SRMD record"
    )
    args = parser.parse_args()

    source_path = Path(args.source)
    output_path = Path(args.output)

    manifest = build_srmd_dataset(index_file=source_path, output_dir=output_path, seed=args.seed)

    print("\n" + "=" * 76)
    print("SRMD DATASET MANIFEST SUMMARY")
    print("=" * 76)
    print(f"Dataset Name:     {manifest['dataset_name']} v{manifest['dataset_version']}")
    print(f"Total Records:    {manifest['total_records']}")
    print(f"Visual Source:    {manifest['source_dataset']['name']} ({manifest['source_dataset']['country']})")
    print(f"Synthesis Seed:   {manifest['random_seed']}")
    print("\nClass Breakdown:")
    for cid, info in manifest["class_distribution"].items():
        print(f"  [{cid}] {info['name']:<20}: {info['count']}")
    print("=" * 76)

    if args.inspect:
        records_file = output_path / "srmd_records.json"
        if records_file.exists():
            with open(records_file, "r", encoding="utf-8") as rf:
                records = json.load(rf)
            if records:
                print("\nSample SRMD Record (Record #1):")
                print(json.dumps(records[0], indent=2))


if __name__ == "__main__":
    main()
