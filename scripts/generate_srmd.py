"""
Smart Road Multimodal Dataset (SRMD) Generator
==============================================

Generates the derived multimodal road-infrastructure dataset (SRMD) combining
authentic RDD2022 India visual distress ground truth with physical consistency modeling.

Outputs generated:
- data/processed/srmd/records/*.json (Individual record files)
- data/processed/srmd/srmd_records.json (Combined JSON array)
- data/processed/srmd/srmd_dataset.jsonl (Line-delimited JSONL format)
- data/processed/srmd/srmd_dataset.csv (Flattened tabular CSV format)
- data/processed/srmd/srmd_dataset.parquet (High-performance columnar Parquet format)
- data/processed/srmd/srmd_manifest.json (Dataset metadata and provenance manifest)

Usage Examples:
    python scripts/generate_srmd.py --input data/raw/rdd2022_india --output data/processed/srmd --seed 2026
    python scripts/generate_srmd.py --limit 100 --class D40
    python scripts/generate_srmd.py --all-classes --seed 2026
"""

import os
import sys
import csv
import json
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from services.dataset.srmd_builder import SRMDRecordBuilder, CLASS_TAXONOMY
from services.dataset.split_manager import ResearchDataSplitter
from scripts.prepare_rdd2022 import parse_voc_xml

try:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False


def safe_rel_path(p: Path) -> str:
    """Returns a clean relative path to PROJECT_ROOT, or absolute if outside."""
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(p.resolve())


def find_annotations(input_path: Path) -> Dict[str, Any]:
    """
    Resolves annotations from either a JSON index file, raw XML directory, or fallback locations.
    """
    annotations_index = {}

    # Case 1: Input is a direct annotations JSON file
    if input_path.is_file() and input_path.suffix == ".json":
        with open(input_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Case 2: Input directory contains annotations_index.json
    idx_candidate = input_path / "annotations_index.json"
    if idx_candidate.is_file():
        with open(idx_candidate, "r", encoding="utf-8") as f:
            return json.load(f)

    # Case 3: Look in data/processed/annotations_index.json
    proc_idx = PROJECT_ROOT / "data" / "processed" / "annotations_index.json"

    # Case 4: Search for XML files in input_path
    xml_search_dirs = [
        input_path / "annotations" / "xmls",
        input_path / "India" / "annotations" / "xmls",
        input_path / "xmls",
        input_path
    ]
    xml_files = []
    for d in xml_search_dirs:
        if d.is_dir():
            xml_files = list(d.glob("*.xml"))
            if xml_files:
                break

    if xml_files:
        print(f"Parsing {len(xml_files)} Pascal VOC XML annotation files from {input_path}...")
        for xf in xml_files:
            parsed = parse_voc_xml(xf)
            if parsed and parsed.get("objects"):
                img_id = xf.stem
                annotations_index[img_id] = parsed
        return annotations_index

    # If no raw XMLs found in input_path, check data/processed/annotations_index.json
    if proc_idx.is_file():
        print(f"Loading pre-indexed annotations from {proc_idx}...")
        with open(proc_idx, "r", encoding="utf-8") as f:
            return json.load(f)

    # Case 5: Fallback to data/samples/annotations/xmls
    samples_xml = PROJECT_ROOT / "data" / "samples" / "annotations" / "xmls"
    if samples_xml.is_dir():
        fallback_xmls = list(samples_xml.glob("*.xml"))
        if fallback_xmls:
            print(f"Fallback: Parsing {len(fallback_xmls)} XML annotations from {samples_xml}...")
            for xf in fallback_xmls:
                parsed = parse_voc_xml(xf)
                if parsed and parsed.get("objects"):
                    annotations_index[xf.stem] = parsed
            return annotations_index

    raise FileNotFoundError(
        f"No annotations or XML files could be found under '{input_path}' or fallback locations. "
        "Run 'python scripts/download_rdd2022.py' or 'python scripts/prepare_rdd2022.py' first."
    )


def flatten_record_for_tabular(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flattens a hierarchical SRMD record into a tabular row suitable for CSV and Parquet.
    """
    vis = record.get("visual", {})
    src = record.get("source", {})
    loc = record.get("location", {})
    lidar = record.get("lidar", {})
    sonar = record.get("sonar", {})
    accel = record.get("accelerometer", {})
    traf = record.get("traffic", {})
    infra = record.get("infrastructure", {})
    env = record.get("environment", {})
    prov = record.get("provenance", {})
    bbox = vis.get("bbox", [0, 0, 0, 0])

    return {
        "event_id": record.get("event_id", ""),
        "split": record.get("split", "train"),
        "image_id": src.get("image_id", ""),
        "annotation_id": src.get("annotation_id", ""),
        "annotation_class": src.get("annotation_class", ""),
        "damage_class": vis.get("damage_class", ""),
        "damage_label": vis.get("damage_label", ""),
        "defect_severity": float(vis.get("defect_severity", 0.0)),
        "bbox_xmin": float(bbox[0]) if len(bbox) > 0 else 0.0,
        "bbox_ymin": float(bbox[1]) if len(bbox) > 1 else 0.0,
        "bbox_xmax": float(bbox[2]) if len(bbox) > 2 else 0.0,
        "bbox_ymax": float(bbox[3]) if len(bbox) > 3 else 0.0,
        "surface_area_sqm": float(vis.get("surface_area_sqm", 0.0)),
        "calculated_volume_liters": float(vis.get("calculated_volume_liters", 0.0)),
        "defect_depth_cm": float(lidar.get("effective_depth_cm", 0.0)),
        "lidar_depth_cm": float(lidar.get("depth_cm", 0.0)),
        "lidar_severity_tier": lidar.get("severity_tier", ""),
        "sonar_depth_cm": float(sonar.get("depth_cm", 0.0)),
        "sonar_effective_depth_cm": float(sonar.get("effective_depth_cm", 0.0)),
        "preferred_modality": env.get("preferred_modality", "optical_lidar"),
        "selected_depth_cm": float(env.get("selected_depth_cm", lidar.get("effective_depth_cm", 0.0))),
        "rain_detected": bool(env.get("rain_detected", False)),
        "luminance_lux": float(env.get("luminance_lux", 0.0)),
        "weather_condition": env.get("weather_condition", ""),
        "z_accel_g": float(accel.get("z_accel_g", 0.0)),
        "dynamic_impact_triggered": bool(accel.get("dynamic_impact_triggered", False)),
        "impact_regime": accel.get("impact_regime", "no_impact"),
        "vehicle_speed_kmh": float(accel.get("vehicle_speed_kmh", 0.0)),
        "latitude": float(loc.get("latitude", 0.0)),
        "longitude": float(loc.get("longitude", 0.0)),
        "road_segment_id": loc.get("road_segment_id", ""),
        "road_name": loc.get("road_name", ""),
        "road_class": loc.get("road_class", ""),
        "chainage_km": float(loc.get("chainage_km", 0.0)),
        "location_mode": loc.get("location_mode", "synthetic_location"),
        "traffic_pcu": int(traf.get("traffic_pcu", 0)),
        "dist_hospital_km": float(infra.get("dist_hospital_km", 0.0)),
        "nearest_hospital_name": infra.get("nearest_hospital_name", ""),
        "is_emergency_corridor": bool(infra.get("is_emergency_corridor", False)),
        "infrastructure_risk_score": float(infra.get("infrastructure_risk_score", 0.0)),
        "priority_level": infra.get("priority_level", "Low"),
        "visual_source": prov.get("visual_source", "RDD2022"),
        "sensor_source": prov.get("sensor_source", "synthetic"),
        "location_source": loc.get("location_source", "synthetic"),
        "hospital_source": infra.get("hospital_source", "synthetic"),
        "synthesis_seed": int(prov.get("synthesis_seed", 2026)),
        "generation_timestamp": prov.get("generation_timestamp", "")
    }


def generate_srmd(
    input_path: Path,
    output_dir: Path,
    seed: int = 2026,
    limit: Optional[int] = None,
    class_filter: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main generator execution pipeline.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    records_dir = output_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)

    annotations_index = find_annotations(input_path)
    builder = SRMDRecordBuilder(seed=seed)
    splitter = ResearchDataSplitter(seed=seed)

    # Deterministically split unique image IDs prior to augmentation (Zero-Leakage Invariant)
    all_image_ids = sorted(list(annotations_index.keys()))
    image_splits = splitter.split_image_ids(all_image_ids)
    img_to_split = {
        img_id: s_name
        for s_name, ids in image_splits.items()
        for img_id in ids
    }

    records = []
    tabular_rows = []
    class_counts = {c: 0 for c in CLASS_TAXONOMY}

    print(f"\n--- Generating SRMD Multimodal Dataset ---")
    print(f"Source:        {input_path}")
    print(f"Output:        {output_dir}")
    print(f"Seed:          {seed}")
    print(f"Class Filter:  {class_filter if class_filter else 'ALL CLASSES'}")
    print(f"Record Limit:  {limit if limit else 'UNLIMITED'}")
    print("-" * 42)

    wp_idx = 0
    total_generated = 0

    for img_id, data in annotations_index.items():
        if limit is not None and total_generated >= limit:
            break

        filename = data.get("filename", f"{img_id}.jpg")
        dims = {
            "width": data.get("width", 720),
            "height": data.get("height", 720),
            "depth": data.get("depth", 3)
        }

        for obj_idx, obj in enumerate(data.get("objects", [])):
            if limit is not None and total_generated >= limit:
                break

            cls_id = obj["class_id"]
            if class_filter and cls_id != class_filter:
                continue

            bbox = [obj["xmin"], obj["ymin"], obj["xmax"], obj["ymax"]]

            # Synthesize record with physical consistency
            record = builder.synthesize_record(
                image_id=img_id,
                filename=filename,
                damage_class=cls_id,
                bbox=bbox,
                image_dims=dims,
                annotation_idx=obj_idx,
                waypoint_idx=wp_idx
            )

            # Assign split based on source image quarantine
            assigned_split = img_to_split.get(img_id, "train")
            record["split"] = assigned_split
            if "provenance" in record and isinstance(record["provenance"], dict):
                record["provenance"]["split"] = assigned_split

            records.append(record)
            tabular_rows.append(flatten_record_for_tabular(record))
            class_counts[cls_id] = class_counts.get(cls_id, 0) + 1

            # Save individual record JSON file
            rec_file = records_dir / f"{record['event_id']}.json"
            with open(rec_file, "w", encoding="utf-8") as rf:
                json.dump(record, rf, indent=2)

            wp_idx += 1
            total_generated += 1

    # 1. Combined JSON
    combined_json_path = output_dir / "srmd_records.json"
    with open(combined_json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    # 2. JSONL Export
    jsonl_path = output_dir / "srmd_dataset.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 3. CSV Export
    csv_path = output_dir / "srmd_dataset.csv"
    if tabular_rows:
        fieldnames = list(tabular_rows[0].keys())
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(tabular_rows)

    # 4. Parquet Export (if pandas/pyarrow installed)
    parquet_path = output_dir / "srmd_dataset.parquet"
    parquet_status = "Skipped (dependencies not installed)"
    if HAS_PARQUET and tabular_rows:
        try:
            df = pd.DataFrame(tabular_rows)
            df.to_parquet(parquet_path, index=False, engine="pyarrow")
            parquet_status = f"Generated ({parquet_path.stat().st_size} bytes)"
        except Exception as e:
            parquet_status = f"Failed to write Parquet: {e}"

    # 5. Export Research Data Splits (train / validation / test)
    partitioned_records = splitter.partition_records(records, image_splits=image_splits)
    splits_manifest = splitter.export_splits(output_dir, partitioned_records, image_splits)

    # 6. Metadata JSON (conforming to Phase 21 specification)
    metadata = {
        "dataset_name": "Smart Road Multimodal Dataset",
        "base_dataset": "RDD2022",
        "base_dataset_country": "India",
        "generator_version": "1.0.0",
        "seed": seed,
        "sensor_fields": [
            "lidar_depth_cm",
            "sonar_depth_cm",
            "z_accel_g",
            "traffic_pcu",
            "dist_hospital_km"
        ],
        "dataset_version": "1.0.0",
        "source_dataset": "RDD2022",
        "preprocessing_version": "1.0.0",
        "sensor_generation_version": "1.0.0",
        "configuration_version": "1.0.0",
        "random_seed": seed,
        "model_version": "1.0.0",
        "splits": {
            "train_records": len(partitioned_records.get("train", [])),
            "validation_records": len(partitioned_records.get("validation", [])),
            "test_records": len(partitioned_records.get("test", [])),
            "train_images": len(image_splits.get("train", [])),
            "validation_images": len(image_splits.get("validation", [])),
            "test_images": len(image_splits.get("test", []))
        },
        "splitting_strategy": {
            "strategy": "source_image_group_split",
            "leakage_prevention": "Strict source-image quarantine. Zero image overlap across train, validation, and test splits.",
            "description": (
                "Partitioning is strictly enforced at the source image level prior to augmentation. "
                "All visual distress annotations and correlated physical sensor simulations sharing "
                "a source image are isolated to the same split, guaranteeing zero multimodal context leakage."
            )
        }
    }
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 7. Dataset Manifest
    manifest = {
        "dataset_name": "Smart Road Multimodal Dataset (SRMD)",
        "dataset_version": "1.0.0",
        "description": "Consistent multimodal road distress dataset coupling authentic RDD2022 visual annotations with physical consistency modeling.",
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        "synthesis_seed": seed,
        "total_records": len(records),
        "source_dataset": {
            "name": "RDD2022",
            "country": "India",
            "doi": "10.6084/m9.figshare.21431547"
        },
        "class_filter": class_filter,
        "class_distribution": {
            cid: {
                "label": CLASS_TAXONOMY[cid]["damage_label"],
                "count": class_counts.get(cid, 0)
            }
            for cid in sorted(class_counts.keys())
        },
        "exports": {
            "json_records_dir": safe_rel_path(records_dir),
            "combined_json": safe_rel_path(combined_json_path),
            "jsonl_dataset": safe_rel_path(jsonl_path),
            "csv_dataset": safe_rel_path(csv_path),
            "parquet_dataset": safe_rel_path(parquet_path) if HAS_PARQUET else None,
            "metadata_json": safe_rel_path(metadata_path),
            "splits_dir": safe_rel_path(output_dir / "splits")
        },
        "metadata_file": safe_rel_path(metadata_path),
        "splits": metadata["splits"],
        "splitting_strategy": metadata["splitting_strategy"],
        "provenance_policy": {
            "visual_source": "RDD2022 (India)",
            "sensor_source": "synthetic (physics-consistent)",
            "location_source": "synthetic (Mumbai demonstration corridor)",
            "hospital_source": "synthetic (geodesic nearest neighbor)",
            "citation": "Sekilab / CRDDC'2022"
        }
    }

    manifest_path = output_dir / "srmd_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 70)
    print("SRMD GENERATION COMPLETE")
    print("=" * 70)
    print(f"Total Records Generated: {len(records)}")
    print(f"Records Directory:       {records_dir}")
    print(f"Combined JSON:           {combined_json_path} ({combined_json_path.stat().st_size} bytes)")
    print(f"JSONL Export:            {jsonl_path} ({jsonl_path.stat().st_size} bytes)")
    print(f"CSV Export:              {csv_path} ({csv_path.stat().st_size} bytes)")
    print(f"Parquet Export:          {parquet_status}")
    print(f"Manifest File:           {manifest_path}")
    print("\nClass Distribution:")
    for cid, info in manifest["class_distribution"].items():
        print(f"  [{cid}] {info['label']:<22} : {info['count']}")
    print("=" * 70 + "\n")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Generate Smart Road Multimodal Dataset (SRMD)")
    parser.add_argument(
        "--input",
        type=str,
        default=str(PROJECT_ROOT / "data" / "raw" / "rdd2022_india"),
        help="Input path to raw RDD2022 directory or processed annotations index"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed" / "srmd"),
        help="Output directory for generated SRMD multimodal dataset"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of records to generate (e.g. 100, 1000)"
    )
    parser.add_argument(
        "--class",
        dest="class_filter",
        type=str,
        default=None,
        choices=["D00", "D10", "D20", "D40"],
        help="Filter generation to a specific damage class (e.g. D40)"
    )
    parser.add_argument(
        "--all-classes",
        action="store_true",
        default=True,
        help="Include all road damage classes (default)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Deterministic random seed for physical consistency modeling"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    generate_srmd(
        input_path=input_path,
        output_dir=output_dir,
        seed=args.seed,
        limit=args.limit,
        class_filter=args.class_filter
    )


if __name__ == "__main__":
    main()
