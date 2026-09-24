#!/usr/bin/env python
"""
Multimodal Data Fusion Demonstration CLI
========================================

Executes the unified end-to-end intelligence pipeline:
RDD2022 Image → CV Detector → Multimodal Context → Sensor Switching → RoadDefectEvent → TOPSIS MCDM

Usage:
    python scripts/run_data_fusion.py --image data/samples/images/India_000045.jpg
    python scripts/run_data_fusion.py --images-dir data/samples/images/ --output data/processed/fused_events.json
    python scripts/run_data_fusion.py --image data/samples/images/India_000045.jpg --rain --low-light
"""

import sys
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from services.fusion.fusion_engine import MultimodalDataFusionEngine
from services.fusion.schemas import RoadDefectEvent


def parse_args():
    parser = argparse.ArgumentParser(description="Run Multimodal Data Fusion Pipeline")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a single RDD2022 image to process"
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=None,
        help="Path to directory containing RDD2022 images"
    )
    parser.add_argument(
        "--rain",
        action="store_true",
        help="Simulate rain / wet road condition (triggers acoustic sonar)"
    )
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Simulate dry daylight condition (triggers optical LiDAR)"
    )
    parser.add_argument(
        "--low-light",
        action="store_true",
        help="Simulate low scene illuminance (< 40 lux, triggers acoustic sonar)"
    )
    parser.add_argument(
        "--road-class",
        type=str,
        default=None,
        choices=["LOCAL", "COLLECTOR", "ARTERIAL", "HIGH_TRAFFIC_CORRIDOR"],
        help="Functional road classification"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed" / "fused_events.json"),
        help="Path to save fused events JSON output"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for deterministic multimodal synthesis"
    )
    parser.add_argument(
        "--ingest-telemetry",
        action="store_true",
        help="Ingest generated events directly into the main.py telemetry store"
    )
    return parser.parse_args()


def print_event_summary(ev: RoadDefectEvent, index: int, total: int):
    print("\n" + "=" * 75)
    print(f"ROAD DEFECT EVENT [{index + 1}/{total}]: {ev.event_id}")
    print("=" * 75)
    print(f"Timestamp:       {ev.timestamp}")
    print(f"Visual Source:   {ev.visual.source} | Image: {ev.visual.image_id}")
    print(f"Damage Class:    {ev.visual.damage_class} ({ev.visual.damage_label})")
    print(f"Bounding Box:    {ev.visual.bbox} [w={ev.visual.image_width}, h={ev.visual.image_height}]")
    print(f"Confidence:      {ev.visual.confidence if ev.visual.confidence else 'Ground Truth / High'}")
    print(f"Surface Area:    {ev.visual.surface_area_sqm:.3f} m²")

    print("\n--- Environmental Evidence ---")
    print(f"Rain Detected:   {ev.environmental.rain_detected}")
    print(f"Mean Luminance:  {ev.environmental.mean_luminance:.1f} lux")
    print(f"Condition:       {ev.environmental.weather_condition}")

    print("\n--- Sensor Evidence & Environmental Switching ---")
    print(f"Active Modality: {ev.sensor.active_modality.upper()}")
    print(f"Effective Depth: {ev.sensor.effective_depth_cm:.1f} cm (LiDAR raw: {ev.sensor.lidar_depth_cm:.1f} cm, Sonar raw: {ev.sensor.sonar_depth_cm:.1f} cm)")
    print(f"Defect Volume:   {ev.sensor.calculated_volume_liters:.2f} Liters")
    print(f"Vertical Shock:  {ev.sensor.z_accel_g:.2f}g | Impact Confirmed (>1.50g): {ev.sensor.dynamic_impact_confirmed}")
    print(f"Impact Regime:   {ev.sensor.impact_regime}")

    print("\n--- Municipal Infrastructure Context ---")
    print(f"Location:        ({ev.infrastructure.latitude:.4f}, {ev.infrastructure.longitude:.4f}) [{ev.infrastructure.road_segment_id}]")
    print(f"Road Class:      {ev.infrastructure.road_class}")
    print(f"Traffic Volume:  {ev.infrastructure.traffic_pcu:,} PCU")
    print(f"Hospital Dist:   {ev.infrastructure.dist_hospital_km:.2f} km to {ev.infrastructure.nearest_hospital}")
    print(f"Operating Speed: {ev.infrastructure.speed_kmh:.1f} km/h")

    if ev.evidence_summary:
        print("\n--- Multi-Sensor Evidence & Disagreement Diagnosis ---")
        es = ev.evidence_summary
        print(f"Disagreement Category: {es.get('disagreement_category', '').upper()}")
        print(f"Multimodal Score:      {es.get('multimodal_confidence_score', 0.0):.4f} / 1.0000")
        print(f"Visual Confidence:     {es.get('visual_confidence', 0.0):.3f}")
        print(f"Depth Assessment:      {es.get('depth_evidence', {}).get('assessment', '')}")
        print(f"Impact Assessment:     {es.get('impact_evidence', {}).get('assessment', '')}")
        print(f"Fusion Rationale:      {es.get('fusion_rationale', '')}")

    if ev.topsis:
        print("\n--- TOPSIS MCDM Prioritization ---")
        print(f"Criteria Vector: [Volume: {ev.topsis.criteria_vector[0]:.2f} L, PCU: {int(ev.topsis.criteria_vector[1]):,}, Hosp: {ev.topsis.criteria_vector[2]:.2f} km, Speed: {ev.topsis.criteria_vector[3]:.1f} km/h]")
        print(f"TOPSIS Score:    {ev.topsis.topsis_score:.4f} / 1.0000")
        print(f"Priority Level:  {ev.topsis.priority_level.upper()}")
        print(f"Priority Rank:   #{ev.topsis.rank}")

    print("\n--- Provenance Audit Trail ---")
    for k, v in ev.provenance.items():
        print(f"  {k}: {v}")


def main():
    args = parse_args()
    print("=" * 75)
    print("Smart Road Digital Twin - Multimodal Data Fusion Engine")
    print("=" * 75)

    engine = MultimodalDataFusionEngine(seed=args.seed, project_root=PROJECT_ROOT)

    # Resolve environment overrides
    rain_flag = None
    if args.rain:
        rain_flag = True
    elif args.dry:
        rain_flag = False

    luma_val = None
    if args.low_light:
        luma_val = 25.0
    elif args.dry and not args.rain:
        luma_val = 75.0

    # Resolve image sources
    image_paths = []
    if args.image:
        image_paths.append(Path(args.image))
    elif args.images_dir:
        dir_p = Path(args.images_dir)
        image_paths = sorted(list(dir_p.glob("*.jpg")) + list(dir_p.glob("*.png")))
    else:
        # Default sample image
        sample_img = PROJECT_ROOT / "data" / "samples" / "images" / "India_000045.jpg"
        if sample_img.exists():
            image_paths.append(sample_img)
        else:
            sample_dir = PROJECT_ROOT / "data" / "samples" / "images"
            image_paths = sorted(list(sample_dir.glob("*.jpg")))[:3]

    if not image_paths:
        print("Error: No images found to process!")
        sys.exit(1)

    print(f"Processing {len(image_paths)} image(s)...")

    # Run data fusion
    if len(image_paths) == 1:
        events = engine.fuse_image(
            image_input=image_paths[0],
            rain_detected=rain_flag,
            mean_luminance=luma_val,
            road_class=args.road_class
        )
    else:
        events = engine.fuse_batch(
            image_paths=image_paths,
            rain_detected=rain_flag,
            mean_luminance=luma_val
        )

    print(f"Data Fusion Complete. Total RoadDefectEvents generated: {len(events)}")

    for i, ev in enumerate(events):
        print_event_summary(ev, i, len(events))

    # Ingestion into main.py telemetry store if requested
    if args.ingest_telemetry:
        print("\n" + "=" * 75)
        print("INGESTING EVENTS INTO DIGITAL TWIN TELEMETRY STORE")
        print("=" * 75)
        from main import process_and_store_telemetry
        for ev in events:
            payload = ev.to_telemetry_dict()
            stored = process_and_store_telemetry(payload)
            print(f"  Ingested {stored['id']} -> TOPSIS Score: {stored.get('topsis_score')}, Priority: {stored.get('priority_level')}")

    # Save output
    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    serializable_events = [ev.model_dump() for ev in events]
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(serializable_events, f, indent=2)

    print(f"\nAll fused events serialized to: {out_p}")


if __name__ == "__main__":
    main()
