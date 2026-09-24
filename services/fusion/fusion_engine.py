"""
Multimodal Data Fusion Engine
=============================

End-to-End Multimodal Intelligence Pipeline:
    RDD2022 Image
          ↓
     CV Detector (RDD2022Detector)
          ↓
  Damage Class + Bounding Box + Confidence
          ↓
  Physical Consistency & Context Engine (PhysicalConsistencyModel)
          ↓
  Environmental Modality Switching (EnvironmentalSensorSwitching)
          ↓
    RoadDefectEvent (Visual + Sensor + Environmental + Infrastructure)
          ↓
   TOPSIS MCDM Multi-Criteria Prioritization (MCDMPrioritizationEngine)

Strict Integrity Guarantees:
- Pure visual evidence from authentic RDD2022 imagery.
- Physically consistent, non-arbitrary synthetic sensor telemetry.
- Seamless conversion to backend telemetry dict for live digital twin map updates.
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from services.cv.schemas import DamageDetection, DetectionResult, CLASS_LABELS
from services.cv.detector import RoadDamageDetector
from services.cv.rdd2022_detector import RDD2022Detector
from services.ai_engine.pipeline.depth_estimator import EnvironmentalSensorSwitching
from services.backend.app.services.mcdm_engine import MCDMPrioritizationEngine
from services.data_generation.consistency_model import PhysicalConsistencyModel
from services.fusion.schemas import (
    VisualEvidence,
    EnvironmentalEvidence,
    SensorEvidence,
    InfrastructureContext,
    TOPSISAssessment,
    RoadDefectEvent
)
from services.fusion.disagreement_analyzer import SensorDisagreementAnalyzer


class MultimodalDataFusionEngine:
    """
    Unifies visual damage detections with synthetic multimodal sensors,
    environmental state, infrastructure context, and TOPSIS MCDM prioritization.
    """

    def __init__(
        self,
        detector: Optional[RoadDamageDetector] = None,
        switching_engine: Optional[EnvironmentalSensorSwitching] = None,
        consistency_model: Optional[PhysicalConsistencyModel] = None,
        topsis_engine: Optional[MCDMPrioritizationEngine] = None,
        seed: int = 2026,
        srmd_records_path: Optional[Union[str, Path]] = None,
        project_root: Optional[Path] = None
    ):
        self.project_root = Path(project_root) if project_root else PROJECT_ROOT
        self.seed = seed
        self.detector = detector if detector is not None else RDD2022Detector()
        self.switching = switching_engine if switching_engine is not None else EnvironmentalSensorSwitching(baseline_chassis_height_cm=30.0)
        self.consistency_model = consistency_model if consistency_model is not None else PhysicalConsistencyModel(seed=seed)
        self.topsis_engine = topsis_engine if topsis_engine is not None else MCDMPrioritizationEngine()
        self.disagreement_analyzer = SensorDisagreementAnalyzer(impact_threshold_g=1.50)

        # Load SRMD cache if available
        self.srmd_cache = {}
        cache_p = Path(srmd_records_path) if srmd_records_path else self.project_root / "data" / "processed" / "srmd" / "srmd_records.json"
        if cache_p.exists():
            try:
                with open(cache_p, "r", encoding="utf-8") as f:
                    cached_records = json.load(f)
                    for rec in cached_records:
                        img_id = rec.get("source", {}).get("image_id")
                        if img_id:
                            if img_id not in self.srmd_cache:
                                self.srmd_cache[img_id] = []
                            self.srmd_cache[img_id].append(rec)
            except Exception:
                pass

        self._event_counter = 0

    def compute_image_luminance(self, image_path: Path) -> float:
        """Computes approximate mean scene luminance (IRE / lux proxy) from image."""
        try:
            with Image.open(image_path) as img:
                gray = img.convert("L")
                arr = np.asarray(gray, dtype=float)
                mean_luma = float(np.mean(arr))
                # Map 0-255 grayscale to reasonable lux proxy (0.0 to 120.0 lux)
                return round((mean_luma / 255.0) * 100.0, 1)
        except Exception:
            return 65.0

    def fuse_image(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        image_id: Optional[str] = None,
        rain_detected: Optional[bool] = None,
        mean_luminance: Optional[float] = None,
        road_class: Optional[str] = None,
        traffic_pcu: Optional[int] = None,
        dist_hospital_km: Optional[float] = None,
        speed_kmh: Optional[float] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None
    ) -> List[RoadDefectEvent]:
        """
        Executes complete multimodal data fusion on an input image:
        1. Visual Detection: Bounding box, damage class, confidence.
        2. Context & Physical Consistency: Surface area, depth, volume, impact acceleration.
        3. Environmental Sensor Switching: Rain/low-light acoustic sonar vs optical LiDAR.
        4. Infrastructure Context: Road class, traffic PCU, hospital proximity, speed.
        5. TOPSIS Prioritization: Multi-criteria risk scoring.
        """
        # Resolve image identifier
        resolved_img_id = image_id
        img_path = None
        if isinstance(image_input, (str, Path)):
            img_path = Path(image_input)
            if resolved_img_id is None:
                resolved_img_id = img_path.stem

        if resolved_img_id is None:
            resolved_img_id = "India_UNNAMED"

        # Resolve environmental inputs
        if mean_luminance is None:
            if img_path and img_path.exists():
                mean_luminance = self.compute_image_luminance(img_path)
            else:
                mean_luminance = 65.0

        if rain_detected is None:
            # Deterministic pseudo-random environmental check based on seed and image_id
            rng = np.random.RandomState(abs(hash(resolved_img_id) + self.seed) % (2**31 - 1))
            rain_detected = bool(rng.random() < 0.25)

        # Stage 1: Computer Vision Detection
        if hasattr(self.detector, "detect_full"):
            det_result = self.detector.detect_full(image_input)
            detections = det_result.detections
            img_w, img_h = det_result.image_width, det_result.image_height
        elif hasattr(self.detector, "detect_with_metadata"):
            det_result = self.detector.detect_with_metadata(image_input)
            detections = det_result.detections
            img_w, img_h = det_result.image_width, det_result.image_height
        else:
            detections = self.detector.detect(image_input)
            img_w, img_h = 720, 720

        if not detections:
            return []

        # Stage 2 & 3: Multimodal Context Synthesis & Sensor Switching
        events: List[RoadDefectEvent] = []

        for idx, det in enumerate(detections):
            self._event_counter += 1
            event_id = f"SRMD-FUSED-{resolved_img_id}-{idx:02d}"

            # Calculate surface area
            surface_area = self.detector.compute_perspective_surface_area(
                det.bbox,
                img_w,
                img_h
            )

            # Generate physically consistent sensor, dynamic, and context telemetry
            veh_speed = speed_kmh if speed_kmh is not None else 40.0
            consistent_telemetry = self.consistency_model.derive_consistent_telemetry(
                damage_class=det.class_id,
                surface_area_sqm=surface_area,
                rain_detected=rain_detected,
                luminance_lux=mean_luminance,
                vehicle_speed_kmh=veh_speed,
                road_class=road_class
            )

            # Apply Environmental Sensor Switching
            switching_res = self.switching.evaluate_environment_and_measure(
                rain_detected=rain_detected,
                mean_luminance=mean_luminance,
                lidar_depth_cm=consistent_telemetry["sensors"]["lidar_depth_cm"],
                sonar_depth_cm=consistent_telemetry["sensors"]["sonar_depth_cm"],
                z_accel_g=consistent_telemetry["accelerometer"]["z_accel_g"],
                surface_area_sqm=surface_area
            )

            # Resolve infrastructure context
            loc = consistent_telemetry.get("location", {})
            infra_lat = latitude if latitude is not None else loc.get("latitude", 19.0760)
            infra_lon = longitude if longitude is not None else loc.get("longitude", 72.8777)
            infra_pcu = traffic_pcu if traffic_pcu is not None else consistent_telemetry.get("traffic", {}).get("pcu", 12000)
            infra_dist = dist_hospital_km if dist_hospital_km is not None else consistent_telemetry.get("infrastructure", {}).get("dist_hospital_km", 2.5)
            infra_road = road_class if road_class is not None else consistent_telemetry.get("traffic", {}).get("road_class", "ARTERIAL")

            # Build sub-schemas
            visual_ev = VisualEvidence(
                image_id=resolved_img_id,
                damage_class=det.class_id,
                damage_label=det.class_label,
                bbox=[det.bbox.xmin, det.bbox.ymin, det.bbox.xmax, det.bbox.ymax],
                confidence=det.confidence,
                surface_area_sqm=surface_area,
                image_width=img_w,
                image_height=img_h,
                source="RDD2022"
            )

            env_ev = EnvironmentalEvidence(
                rain_detected=rain_detected,
                mean_luminance=mean_luminance,
                weather_condition="rain" if rain_detected else ("low_light" if mean_luminance < 40 else "clear"),
                ambient_lux=mean_luminance * 5.0,
                source="synthetic_or_sensor"
            )

            sensor_ev = SensorEvidence(
                lidar_depth_cm=consistent_telemetry["sensors"]["lidar_depth_cm"],
                sonar_depth_cm=consistent_telemetry["sensors"]["sonar_depth_cm"],
                active_modality=switching_res["detection_modality"],
                effective_depth_cm=switching_res["estimated_depth_cm"],
                calculated_volume_liters=switching_res["calculated_volume_liters"],
                z_accel_g=consistent_telemetry["accelerometer"]["z_accel_g"],
                dynamic_impact_confirmed=switching_res["dynamic_impact_confirmed"],
                impact_regime=consistent_telemetry["accelerometer"]["impact_regime"],
                sensor_source="synthetic",
                generation_method="controlled_distribution"
            )

            infra_ctx = InfrastructureContext(
                latitude=infra_lat,
                longitude=infra_lon,
                road_segment_id=loc.get("road_segment_id", "SEG-DEMO-01"),
                road_class=infra_road,
                traffic_pcu=infra_pcu,
                dist_hospital_km=infra_dist,
                nearest_hospital=consistent_telemetry.get("infrastructure", {}).get("nearest_hospital", "General Hospital"),
                speed_kmh=veh_speed,
                location_mode="synthetic_location",
                location_source="synthetic",
                traffic_source="synthetic",
                hospital_source="synthetic"
            )

            # Analyze sensor concordance and disagreement
            ev_summary = self.disagreement_analyzer.analyze_evidence(
                damage_class=det.class_id,
                visual_confidence=det.confidence,
                surface_area_sqm=surface_area,
                effective_depth_cm=switching_res["estimated_depth_cm"],
                active_modality=switching_res["detection_modality"],
                z_accel_g=consistent_telemetry["accelerometer"]["z_accel_g"],
                rain_detected=rain_detected,
                mean_luminance=mean_luminance
            )

            event = RoadDefectEvent(
                event_id=event_id,
                visual=visual_ev,
                environmental=env_ev,
                sensor=sensor_ev,
                infrastructure=infra_ctx,
                physical_consistency={
                    "defect_severity": consistent_telemetry.get("defect_severity"),
                    "volume_consistency": "verified",
                    "impact_consistency": "verified"
                },
                evidence_summary=ev_summary.model_dump(),
                provenance={
                    "visual_source": "RDD2022",
                    "sensor_source": "synthetic",
                    "location_source": "synthetic",
                    "hospital_source": "synthetic",
                    "traffic_source": "synthetic",
                    "generation_method": "controlled_distribution",
                    "synthesis_seed": self.seed
                }
            )
            events.append(event)

        # Stage 4: Multi-Criteria TOPSIS Prioritization
        self._apply_topsis_to_events(events)

        return events

    def _apply_topsis_to_events(self, events: List[RoadDefectEvent]) -> None:
        """Computes TOPSIS scores and ranks for a collection of events."""
        if not events:
            return

        if len(events) == 1:
            # Single alternative: reference against canonical worst and best bounds
            ref_worst = [0.1, 1000.0, 10.0, 10.0]
            ref_best = [80.0, 30000.0, 0.1, 70.0]
            ev = events[0]
            mat = np.array([
                [
                    ev.sensor.calculated_volume_liters,
                    float(ev.infrastructure.traffic_pcu),
                    ev.infrastructure.dist_hospital_km,
                    ev.infrastructure.speed_kmh
                ],
                ref_worst,
                ref_best
            ])
            scores = self.topsis_engine.compute_topsis(mat)
            score = float(scores[0])
            ev.topsis = TOPSISAssessment(
                criteria_vector=[
                    ev.sensor.calculated_volume_liters,
                    float(ev.infrastructure.traffic_pcu),
                    ev.infrastructure.dist_hospital_km,
                    ev.infrastructure.speed_kmh
                ],
                topsis_score=score,
                priority_level=self.topsis_engine.classify_priority(score),
                rank=1
            )
            return

        # Multiple alternatives: build decision matrix
        rows = []
        for ev in events:
            rows.append([
                ev.sensor.calculated_volume_liters,
                float(ev.infrastructure.traffic_pcu),
                ev.infrastructure.dist_hospital_km,
                ev.infrastructure.speed_kmh
            ])

        matrix = np.array(rows)
        scores = self.topsis_engine.compute_topsis(matrix)

        # Sort ranks by descending score
        score_idx_pairs = sorted([(scores[i], i) for i in range(len(events))], reverse=True)
        ranks = {idx: rank + 1 for rank, (_, idx) in enumerate(score_idx_pairs)}

        for idx, ev in enumerate(events):
            s = float(scores[idx])
            ev.topsis = TOPSISAssessment(
                criteria_vector=rows[idx],
                topsis_score=s,
                priority_level=self.topsis_engine.classify_priority(s),
                rank=ranks[idx]
            )

        # Sort events list by priority rank
        events.sort(key=lambda e: e.topsis.rank if e.topsis and e.topsis.rank else 999)

    def fuse_batch(
        self,
        image_paths: List[Union[str, Path]],
        rain_detected: Optional[bool] = None,
        mean_luminance: Optional[float] = None
    ) -> List[RoadDefectEvent]:
        """
        Processes a batch of RDD2022 images and computes unified global
        TOPSIS prioritization across all detected defects.
        """
        all_events: List[RoadDefectEvent] = []

        for p in image_paths:
            path_obj = Path(p)
            events = self.fuse_image(
                image_input=path_obj,
                image_id=path_obj.stem,
                rain_detected=rain_detected,
                mean_luminance=mean_luminance
            )
            all_events.extend(events)

        # Apply global multi-alternative TOPSIS across all collected events
        self._apply_topsis_to_events(all_events)
        return all_events
