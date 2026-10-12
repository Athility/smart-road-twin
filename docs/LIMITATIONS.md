# System Limitations: Real vs. Simulated Components

This document provides a transparent, file-referenced breakdown of what is ** genuinely implemented**, **simulated/synthetic**, or **not yet implemented** in the **Smart Road Digital Twin** repository relative to the architecture described in *"Intelligent Road-Pothole Detection and Maintenance Prioritization System"*.

---

## 1. Summary Matrix

| Subsystem | Component | Implementation Status | Code Reference |
| :--- | :--- | :--- | :--- |
| **Sensors** | Road surface RGB images & bounding boxes | **Real (Observed)** — Authentic RDD2022 India JPEG images & Pascal VOC XML annotations | `data/samples/`, `scripts/prepare_rdd2022.py` |
| **Sensors** | LiDAR, Acoustic Sonar, IMU Z-accel, Rain, Luminance | **Simulated** — Physics-consistent stochastic models seeded with `SEED = 2026` | `services/data_generation/consistency_model.py`, `lidar_model.py`, `sonar_model.py`, `accelerometer_model.py` |
| **GIS & Context** | GPS coordinates, Traffic (`PCU/day`), Hospital distance | **Simulated** — Deterministic Mumbai corridor waypoints, Greenshields traffic model, Haversine distance to 8 hardcoded Mumbai hospitals | `services/data_generation/location_model.py`, `traffic_model.py`, `infrastructure_model.py` |
| **Preprocessing** | Resize, RGB normalization, letterbox utilities | **Real** — PIL/NumPy image loading and `[0, 1]` normalization | `services/cv/preprocessing.py` |
| **Preprocessing** | CLAHE, ESRGAN super-resolution, 3D VoxelGrid | **Not Implemented** — Not present in preprocessing pipeline | `services/cv/preprocessing.py` |
| **Detector** | Visual road damage detection (`D00`, `D10`, `D20`, `D40`) | **Ground-Truth Lookup by Default** — Uses `GroundTruthLookupAdapter` (`detector: "ground-truth lookup"`) reading XML annotations; falls back to `OpenCVContourDetectorAdapter` (`"opencv-contour"`) or `UltralyticsYOLOAdapter` (`"yolov8"`) only when `ultralytics` and real `.pt` weights are installed | `services/cv/detector.py`, `services/cv/rdd2022_detector.py` |
| **Segmentation & Area** | PoFormer Transformer / pixel mask segmentation | **Not Implemented (Pinhole BBox Ground-Plane Estimate)** — Pixel segmentation masks are not used; ground-plane footprint area ($\text{m}^2$) is estimated from the bounding-box bottom-edge (`ymax`) via a calibrated pinhole camera model (`camera_height_m=1.35`, `pitch_deg=22.0`, `hfov_deg=62.0`, `vfov_deg=48.0` in `config/sensor_profiles.yaml`) and clamped to per-class bounds (`D40`: `0.01–1.0 m²`) | `services/data_generation/consistency_model.py`, `services/cv/rdd2022_detector.py`, `config/sensor_profiles.yaml` |
| **Sensor Switching** | Weather/luminance modality switching (LiDAR $\leftrightarrow$ Sonar) | **Real Logic (on Simulated Inputs)** — Switches to acoustic Sonar when `rain_detected = True` or `mean_luminance < 40.0 lux`, and computes ellipsoid cavity volume (for `D40` potholes only) or `crack_severity_index` (for `D00`/`D10`/`D20`) | `services/ai_engine/pipeline/depth_estimator.py` |
| **Training** | YOLOv8 training pipeline | **Simulated by Default** — `_train_simulated()` generates deterministic exponential-decay loss curves in `training_summary.json` and writes JSON metadata to `best.pt.simulated.json` when `ultralytics` is not installed | `ml/train/trainer.py`, `runs/train/exp/training_summary.json` |
| **Metrics** | IoU, 101-point PR-curve `AP50`, `mAP50`, `mAP50_95`, Confusion Matrix, Spearman $\rho$ | **Real Math (Evaluated on Ground-Truth Lookup)** — `ml/evaluate/metrics.py` computes genuine 101-point interpolated AP across IoU thresholds `0.50:0.95`, but `runs/evaluate/eval_report.json` is generated with `detector: "ground-truth lookup"` and therefore does not represent neural network generalization | `ml/evaluate/metrics.py`, `ml/evaluate/evaluator.py`, `runs/evaluate/eval_report.json` |
| **Prioritization** | TOPSIS & VIKOR Multi-Criteria Decision Making (MCDM) | **Real Implementation** — Fuzzy BWM weight derivation (`safety-first`, `traffic-first`, `equity-first`, `custom`), TOPSIS closeness $C_i^*$, VIKOR ($S_i, R_i, Q_i$ with C1/C2 checks), Spearman $\rho$, and 6 criteria including `school_proximity_km` and `road_class` | `services/backend/app/services/mcdm_engine.py`, `services/data_generation/infrastructure_model.py` |
| **GIS & Dispatch** | Interactive Leaflet Digital Twin Map & Priority Queue | **Real UI** — Live/fallback markers, hazard radii, priority filtering, and 11-step provenance inspector | `frontend/src/components/Map/PotholeMap.tsx`, `frontend/src/components/Dashboard/` |
| **GIS & Dispatch** | Automated work orders, SMS/email alerts, persistent DB | **Not Implemented** — Storage is an in-memory Python list (`telemetry_store`), and maintenance actions are static UI text strings | `main.py`, `frontend/src/components/Dashboard/IncidentDetailPanel.tsx` |

---

## 2. Detailed Breakdown by Domain

### 2.1 Sensors, Ground-Plane Area Calibration & Urban Context (`services/data_generation/`)
* **What is Real:**
  * **Visual frames and bounding boxes (`OBSERVED`):** The 5 sample images in `data/samples/images/` (`India_000045.jpg`, `India_000102.jpg`, `India_000155.jpg`, `India_000210.jpg`, `India_000278.jpg`) and their 6 bounding-box annotations in `data/samples/annotations/xmls/` come directly from the **RDD2022 India** dataset without modification.
* **Pinhole Camera Ground-Plane Area Assumptions (`config/sensor_profiles.yaml`):**
  * Because RDD2022 provides 2D axis-aligned bounding boxes without per-image camera intrinsics/extrinsics or depth maps, `estimate_ground_plane_area_pinhole()` (`services/data_generation/consistency_model.py` and `services/cv/rdd2022_detector.py`) assumes a standardized forward-facing municipal inspection camera:
    * **Mounting height (`camera_height_m`)**: $1.35\text{ m}$ above the flat road plane (dashboard/windshield mount).
    * **Downward pitch (`pitch_deg`)**: $22.0^\circ$ below the horizontal horizon, with minimum depression guard $\gamma_{\min} = 5.0^\circ$.
    * **Field of view (`hfov_deg`, `vfov_deg`)**: $62.0^\circ$ horizontal $\times$ $48.0^\circ$ vertical on a $720 \times 720$ reference frame ($f_x \approx 599\text{ px}$, $f_y \approx 809\text{ px}$).
    * **Bottom-edge projection**: The bbox bottom edge ($y_{\max}$) is projected onto the road plane at longitudinal distance $Z_{\text{bottom}} = H / \tan(\theta_p + \arctan((y_{\max} - y_c)/f_y))$, converting pixel width and height into metric width $W_{\text{m}}$ and foreshortened ground length $L_{\text{m}}$.
    * **Per-class physical clamping**: Because axis-aligned bounding boxes of diagonal or distant defects overestimate true distress footprint, estimated areas are clamped to physically plausible per-class bounds (`D40` pothole: $0.01\text{–}1.00\,\text{m}^2$; `D20` alligator crack: $0.05\text{–}4.00\,\text{m}^2$; `D00`/`D10` linear cracks: $0.01\text{–}1.50\,\text{m}^2$), logging a warning whenever clamping occurs.
* **What is Simulated:**
  * **Latent severity & physical coupling:** `PhysicalConsistencyModel` (`services/data_generation/consistency_model.py`) computes a deterministic latent severity score $\in [0, 1]$ from the annotation class (`D00`, `D10`, `D20`, `D40`) and calibrated ground-plane area. Cavity volume (`calculated_volume_liters`) is computed only for `D40` potholes; cracks (`D00`, `D10`, `D20`) set `calculated_volume_liters = null` and compute `crack_severity_index` $\in [0, 1]$.
  * **Paraboloid Cavity Volume Shape Factor (`config/sensor_profiles.yaml`):**
    Pothole volume (`calculated_volume_liters`) is computed exclusively for `D40` defects using an idealized paraboloid of revolution cavity model:
    $$V = \text{shape\_factor} \times \text{Area} \times \text{Depth} \times 1000 = 0.50 \times \text{Area}\,(\text{m}^2) \times \frac{\text{Depth}\,(\text{cm})}{100} \times 1000$$
    The shape factor $k = 0.50$ is configured under `pothole_cavity.shape_factor` in `config/sensor_profiles.yaml`. The calculation uses the active sensor's observed cavity depth (`estimated_depth_cm`). In operational road environments, actual potholes exhibit irregular, fractal boundary geometry, stepped walls, and loose debris; the paraboloid model represents a smooth geometric lower-bound approximation.
  * **LiDAR & Acoustic Sonar depth (`SIMULATED`):** `lidar_model.py` and `sonar_model.py` add a `30.0 cm` vehicle chassis baseline plus Gaussian sensor noise to the derived cavity depth.
  * **Z-axis Accelerometer (`SIMULATED`):** `accelerometer_model.py` simulates vertical shock (`z_accel_g`) using a quarter-car response formula and flags `dynamic_impact_triggered = True` when `z_accel_g > 1.5g`.
  * **Rain & Luminance (`SIMULATED`):** Synthesized deterministically per record using `SEED = 2026`.
  * **GPS Coordinates (`SIMULATED`):** `location_model.py` assigns coordinates along 6 hardcoded road segments on the Mumbai Western Corridor (`latitude` `19.059`–`19.136`, `longitude` `72.825`–`72.860`), explicitly tagged `"location_mode": "synthetic_location"`.
  * **Traffic Volume (`SIMULATED`):** `traffic_model.py` simulates daily corridor traffic in **Passenger Car Units per day (`PCU/day`)** across four road classes (`LOCAL`, `COLLECTOR`, `ARTERIAL`, `HIGH_TRAFFIC_CORRIDOR`).
  * **Hospital & School Proximity (`SIMULATED`):** `infrastructure_model.py` computes real Haversine geodesic distance ($\times 1.18$ urban routing tortuosity factor) from the synthetic defect coordinates to static registries of 8 Mumbai trauma hospitals and 18 Mumbai Western Corridor schools.

### 2.2 Visual Detector (`services/cv/`)
* **Active Detector Mode (`ground-truth lookup`):**
  * By default, `RDD2022Detector` (`services/cv/rdd2022_detector.py`) selects `GroundTruthLookupAdapter` (`services/cv/detector.py`), which looks up human-annotated bounding boxes from `data/processed/annotations_index.json` or `data/samples/annotations/xmls/<image_id>.xml` and assigns a deterministic confidence score (`0.84`–`0.98` in the adapter, or `1.0` in pre-built SRMD records).
  * Every backend API response (`/api/v1/telemetry`, `/api/v1/telemetry/ingest`, `/api/v1/dataset/explorer`, `/health`) and the dashboard UI explicitly surface `"detector": "ground-truth lookup" | "opencv-contour" | "yolov8"`.
* **Fallback & Optional Modes:**
  * If an image has no XML ground-truth entry and `ultralytics` is not installed, `GroundTruthLookupAdapter` delegates to `OpenCVContourDetectorAdapter` (grayscale luminance thresholding and bounding-box heuristics).
  * `UltralyticsYOLOAdapter` is implemented in `services/cv/detector.py` and will run real YOLOv8 inference if `ultralytics` is installed and a valid PyTorch `.pt` checkpoint is supplied.
* **Implemented vs. Missing Paper Components:**
  * CIE LAB L-channel CLAHE contrast enhancement (`apply_clahe()` in `services/cv/preprocessing.py`) is implemented and enabled by default behind `preprocessing.enable_clahe`.
  * ESRGAN super-resolution, 3D VoxelGrid point-cloud filtering, and PoFormer segmentation are not implemented.

### 2.3 Training Pipeline (`ml/train/trainer.py`)
* **Simulated Training Mode:**
  * When `ultralytics` is not installed in the Python environment, `DamageModelTrainer.train()` executes `_train_simulated()`.
  * `_train_simulated()` generates a deterministic exponential-decay loss trajectory (`box_loss`, `cls_loss`, `dfl_loss`) and logistic-growth metric trajectory over the requested epochs.
  * Outputs are saved to `runs/train/exp/training_summary.json` (and `training_results.json`) with top-level flags:
    ```json
    {
      "simulated": true,
      "note": "Loss curve is simulated, not real training"
    }
    ```
  * Simulated weight metadata is written to `runs/train/exp/weights/best.pt.simulated.json` and `last.pt.simulated.json` (never `.pt`) so simulated JSON metadata is never mistaken for a real PyTorch checkpoint.

### 2.4 Evaluation & Prioritization Metrics (`ml/evaluate/`, `mcdm_engine.py`)
* **Detection Evaluation (`ml/evaluate/metrics.py` & `runs/evaluate/eval_report.json`):**
  * `ml/evaluate/metrics.py` implements genuine greedy IoU matching, 101-point interpolated Average Precision (`compute_ap_from_pr`), per-class `ap50`, macro `mAP50`, and 10-threshold `mAP50_95` (`0.50:0.05:0.95`).
  * However, when `scripts/evaluate_damage_model.py` runs without a trained YOLOv8 checkpoint, `DamageModelEvaluator` evaluates `RDD2022Detector` in `"ground-truth lookup"` mode against the ground-truth labels (`mAP50 = 1.0`, `mAP50_95 = 1.0` on present classes). Accordingly, `runs/evaluate/eval_report.json` includes:
    ```json
    {
      "detector": "ground-truth lookup",
      "note": "Evaluated with ground-truth lookup detector; metrics are NOT model performance"
    }
    ```
* **MCDM Prioritization (`services/backend/app/services/mcdm_engine.py`):**
  * Classical **TOPSIS** is genuinely implemented using vector normalization, fixed criteria weights (`Volume = 0.35`, `Traffic PCU/day = 0.25`, `Hospital Distance = 0.25`, `Speed = 0.15`), and explicit benefit/cost directionality (`[True, True, False, True]`, treating smaller hospital distance as higher urgency).
  * **Fuzzy BWM** weight optimization and **VIKOR** compromise ranking ($S_i, R_i, Q_i$) from the research paper are not implemented.
  * Segmentation **mIoU**, inference **latency (ms)**, and **dispatch time** metrics are not computed or displayed.

### 2.5 GIS Digital Twin & Maintenance Dispatch (`main.py`, `frontend/`)
* **What is Real:**
  * FastAPI backend (`main.py`) with Pydantic validation, environmental sensor-switching, live TOPSIS re-ranking across `telemetry_store`, and Next.js 15 / React-Leaflet dashboard (`frontend/`) sharing the same 6-record SRMD baseline between the Map Twin and Dataset Explorer views.
* **What is Simulated / Missing:**
  * **In-Memory Storage Only:** `telemetry_store` in `main.py` is an in-memory Python list (capped at 500 records) that resets on server restart; no PostgreSQL/PostGIS or SQLite persistence is wired up.
  * **No Automated Dispatch:** The "Recommended Maintenance Action" card in `IncidentDetailPanel.tsx` displays static guidance text based on `priority_level` (`Critical` / `High` / `Medium` / `Low`). Work-order creation, crew assignment, and SMS/email/webhook dispatch are not implemented.
