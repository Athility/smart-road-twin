# Codebase Technical Audit Report: Smart Road Digital Twin

**Repository:** `c:\smart-road-twin`  
**Target Reference Paper:** *"Intelligent Road-Pothole Detection and Maintenance Prioritization System"*  
**Audit Type:** Read-Only Factual Implementation Audit  

---

## 1. Overview

### App Purpose
The repository implements an interactive **Smart Road Digital Twin & Maintenance Prioritization Dashboard** paired with an offline/online Python data pipeline. It models a multi-modal road defect inspection and ranking workflow over the Mumbai Metropolitan Urban Corridor, combining 2D road-damage bounding boxes (modeled after the RDD2022 India dataset taxonomy: `D00`, `D10`, `D20`, `D40`) with deterministically synthesized 1D distance/inertial sensor readings (LiDAR, acoustic SONAR, Z-axis accelerometer, photometer, rain flag) and GIS context (traffic PCU, hospital distance) to rank repair urgency using TOPSIS multi-criteria decision analysis.

### Tech Stack, Frameworks, and Versions

| Layer | Technology | Version | Source Reference |
| :--- | :--- | :--- | :--- |
| **Frontend Framework** | Next.js (App Router) | `14.2.15` | `frontend/package.json:14` |
| **UI Library** | React / React DOM | `^18.3.1` | `frontend/package.json:15-16` |
| **Language (Frontend)** | TypeScript | `^5.6.2` | `frontend/package.json:27` |
| **Styling** | Tailwind CSS + PostCSS + Autoprefixer | `^3.4.13` / `^8.4.47` / `^10.4.20` | `frontend/package.json:22-30` |
| **GIS / Mapping** | Leaflet + React-Leaflet | `^1.9.4` / `^4.2.1` | `frontend/package.json:12,17` |
| **Icons** | `lucide-react` | `^0.446.0` | `frontend/package.json:13` |
| **Class Utilities** | `clsx` + `tailwind-merge` | `^2.1.1` / `^2.5.2` | `frontend/package.json:11,18` |
| **Backend Framework** | FastAPI + Uvicorn + Pydantic | Unpinned (Python `3.10+` target; `3.11` bytecode present) | `main.py:1-13`, `README.md:260-267` |
| **Scientific / CV Stack** | NumPy, PyYAML, Pillow (`PIL`), OpenCV (`cv2`, optional), PyArrow (optional), Ultralytics (optional) | Unpinned (no `requirements.txt` or `pyproject.toml` in repo) | `README.md:267`, `services/cv/detector.py:173,235`, `scripts/generate_srmd.py:210` |

### Package Manager
- **Frontend:** `npm` (`frontend/package-lock.json`, lockfileVersion 3).
- **Backend (Python):** Unmanaged in repository root — **no** `requirements.txt`, `pyproject.toml`, `Pipfile`, or `poetry.lock` exists. Dependencies are only documented as a manual `pip install` command in `README.md:267`.

### How to Run, Build, and Deploy
- **Backend Server (`main.py:471-475`):**
  - Run `python main.py` or `uvicorn main:app --host 0.0.0.0 --port 8000` from project root (`README.md:297-302`).
- **IoT Telemetry Simulator (`simulation/mock_iot_vehicle.py:56-84`):**
  - Run `python simulation/mock_iot_vehicle.py` to POST randomized telemetry payloads to `http://localhost:8000/api/v1/telemetry/ingest` every 3 seconds.
- **Data & ML Scripts (`scripts/`):**
  - `python scripts/download_rdd2022.py --sample-only` (`scripts/download_rdd2022.py:418-449`)
  - `python scripts/prepare_rdd2022.py` (`scripts/prepare_rdd2022.py:278-305`)
  - `python scripts/generate_srmd.py` (`scripts/generate_srmd.py:424-478`)
  - `python scripts/train_damage_model.py` (`scripts/train_damage_model.py:39-104`)
  - `python scripts/evaluate_damage_model.py` (`scripts/evaluate_damage_model.py:39-132`)
  - `python scripts/run_data_fusion.py` (`scripts/run_data_fusion.py:72-216`)
  - `python scripts/run_fusion_experiments.py` (`scripts/run_fusion_experiments.py:36-106`)
- **Frontend (`frontend/package.json:6-10`):**
  - Development: `cd frontend && npm install && npm run dev` (runs `next dev` on port `3000`).
  - Production build: `npm run build` (`next build`) and `npm run start` (`next start`).
- **Deployment (`vercel.json:1-28`, `frontend/vercel.json:1-7`):**
  - Root `vercel.json:3-8` configures only the static/Next.js build inside `frontend/package.json` (`framework: "nextjs"`, `outputDirectory: "frontend/.next"`) and sets cache headers (`vercel.json:11-26`). It **does not** configure Python serverless functions for `main.py`; when deployed to Vercel without a separate backend host, the frontend silently falls back to hardcoded client-side data (`frontend/src/lib/api.ts:261-269`).

### Folder Tree (3 Levels)

```text
c:\smart-road-twin\
├── main.py                              # FastAPI backend server, in-memory store, REST API
├── test_core_domain.py                  # 83-test Python unittest suite (1,798 lines)
├── api_test.py                          # Standalone HTTP verification script against :8000
├── vercel.json                          # Root Vercel deployment config (Next.js only)
├── README.md                            # System documentation & runbook
├── config/
│   └── sensor_profiles.yaml             # Physics/noise/threshold parameters for synthetic sensors
├── data/
│   ├── raw/                             # Empty placeholder (.gitkeep) for full RDD2022 download
│   ├── samples/
│   │   ├── images/                      # 5 PIL-drawn synthetic sample JPEGs (India_000045..278.jpg)
│   │   └── annotations/xmls/            # 5 Pascal VOC XML files matching the sample images
│   └── processed/
│       ├── annotations_index.json       # Indexed VOC annotations for the 5 sample images
│       ├── sample_manifest.json         # Manifest of sample dataset
│       ├── srmd/                        # Generated 6-record SRMD dataset (JSON, CSV, JSONL, Parquet, splits/)
│       └── yolo/                        # Generated YOLO format dataset (images/ & labels/ train/val/test)
├── frontend/
│   ├── package.json                     # Next.js 14.2.15 dependencies and scripts
│   ├── next.config.mjs                  # Rewrites (/api/backend/*) & image remotePatterns
│   ├── tailwind.config.js               # Custom priority color palette & pulse animations
│   ├── tsconfig.json                    # TypeScript configuration
│   └── src/
│       ├── app/                         # Next.js App Router (layout.tsx, page.tsx, error.js, globals.css)
│       ├── components/
│       │   ├── ClientOnly.tsx           # Hydration guard wrapper
│       │   ├── Dashboard/               # Header, SummaryPanel, TelemetryTable, IncidentDetailPanel, DatasetExplorerPanel
│       │   └── Map/                     # MapWrapper.tsx (dynamic import) & PotholeMap.tsx (Leaflet)
│       ├── lib/api.ts                   # Fetch client + hardcoded FALLBACK_DATA & FALLBACK_DATASET_STATS
│       └── types/telemetry.ts           # TypeScript interfaces for telemetry & dataset stats
├── ml/
│   ├── configs/                         # training_config.yaml & rdd2022_yolo.yaml
│   ├── train/                           # dataset.py (VOC-to-YOLO converter) & trainer.py (YOLO/simulated trainer)
│   └── evaluate/                        # evaluator.py & metrics.py (IoU, precision/recall, fabricated mAP)
├── runs/
│   ├── train/exp/                       # Simulated training outputs (args.yaml, training_results.json, JSON best.pt)
│   ├── evaluate/eval_report.json        # Saved evaluation report over 1 test image
│   └── experiments/fusion_benchmark.json# Saved 5-experiment ablation benchmark over 6 records
├── scripts/                             # 9 CLI utility scripts for dataset prep, training, evaluation, and fusion
├── services/
│   ├── ai_engine/pipeline/              # depth_estimator.py (EnvironmentalSensorSwitching)
│   ├── backend/app/services/            # mcdm_engine.py (MCDMPrioritizationEngine TOPSIS)
│   ├── cv/                              # detector.py, preprocessing.py, rdd2022_detector.py, schemas.py
│   ├── data_generation/                 # 8 synthetic sensor/GIS generator modules
│   ├── dataset/                         # srmd_builder.py & split_manager.py
│   ├── experiments/                     # experiment_runner.py (Ablation experiments A-E)
│   └── fusion/                          # fusion_engine.py, disagreement_analyzer.py, schemas.py
└── simulation/
    └── mock_iot_vehicle.py              # Random telemetry POST loop script
```

---

## 2. Feature Inventory

| Page / Route / Component / Feature | File Path | Status | One-Line Notes |
| :--- | :--- | :--- | :--- |
| **Root Layout** | `frontend/src/app/layout.tsx:14-22` | **Working** | Loads Google `Inter` font (`layout.tsx:2,7`), global CSS, and page metadata. |
| **Main Digital Twin Page (`/`)** | `frontend/src/app/page.tsx:16-288` | **Working** | Orchestrates 3s polling (`page.tsx:78`), view switching (`dashboard` vs `explorer`), filters, and split-pane layout. |
| **Route Error Boundary** | `frontend/src/app/error.js:1-27` | **Working** | Client error boundary displaying `error.message` and a retry button calling `reset()`. |
| **Global Error Boundary** | `frontend/src/app/global-error.js:1-14` | **Working** | Root `<html>/<body>` fallback error boundary with reset button. |
| **Hydration Wrapper (`ClientOnly`)** | `frontend/src/components/ClientOnly.tsx:10-23` | **Working** | Delays child rendering until client `useEffect` mounts (`hasMounted` state). |
| **Header & Navigation Bar** | `frontend/src/components/Dashboard/Header.tsx:29-242` | **Working** | Renders tabs ("Digital Twin Dashboard", "Research Dataset Explorer"), weather badge, Critical/High chips, and Sync button. |
| **Header Simulation Mode Toggle** | `frontend/src/components/Dashboard/Header.tsx:157-189` | **Partial** | Cycles client filter (`ALL`, `OPTICAL_LIDAR`, `ACOUSTIC_SONAR`, `IMPACT_VERIFIED`), overriding real weather switching only in the UI view. |
| **Header Telemetry Health Popover** | `frontend/src/components/Dashboard/Header.tsx:214-239` | **Partial** | Shows connection mode, `lastUpdated`, total packets, and hardcoded `"Polling Interval: 3,000 ms (Real-time)"` (`Header.tsx:231`). |
| **Summary KPI Cards (Row 1)** | `frontend/src/components/Dashboard/SummaryPanel.tsx:66-195` | **Working** | Computes Total Defects, Critical count, Submerged count, Dynamic Impact count, and Avg TOPSIS Score from active records. |
| **Analytics Strip (Row 2)** | `frontend/src/components/Dashboard/SummaryPanel.tsx:198-408` | **Working** | Pure-CSS bars for Priority Distribution, Sensor Modality Split, RDD2022 Class Breakdown, and Top-3 Repair Queue. |
| **Map Dynamic Loader (`MapWrapper`)** | `frontend/src/components/Map/MapWrapper.tsx:19-30` | **Working** | Uses `next/dynamic` (`ssr: false`) to load Leaflet map client-side with loading skeleton. |
| **Interactive GIS Map (`PotholeMap`)** | `frontend/src/components/Map/PotholeMap.tsx:101-453` | **Working** | Renders Leaflet map with Esri tiles, custom HTML priority pins, radar pulse on Critical pins, `<Circle>` hazard radius, and popups. |
| **Map Basemap Attribution Badge** | `frontend/src/app/page.tsx:152` | **Broken** | UI badge claims `"ESRI / OSM tiles"`, while `PotholeMap.tsx:222-225` loads Esri World Street Map tiles with OSM attribution text. |
| **Incident Detail Slide-Over Drawer** | `frontend/src/components/Dashboard/IncidentDetailPanel.tsx:156-614` | **Partial** | Shows bbox overlay, LiDAR vs Sonar depth bars, 1.5g shock gauge, TOPSIS breakdown, and static dispatch urgency label (no real dispatch action). |
| **Priority Filter & Sort Matrix (`TelemetryTable`)** | `frontend/src/components/Dashboard/TelemetryTable.tsx:58-403` | **Working** | Supports filtering by priority (`All`/`Critical`/`High`/`Medium`/`Low`), sorting by TOPSIS/volume/depth, and row selection. |
| **Research Dataset Explorer (`DatasetExplorerPanel`)** | `frontend/src/components/Dashboard/DatasetExplorerPanel.tsx:57-692` | **Working** | Displays SRMD dataset metrics, class/priority distributions, sensor switching breakdown, provenance legend, and filterable record table. |
| **Backend Health Check (`GET /`)** | `main.py:293-301` | **Working** | Returns JSON health status, active record count, and weather state. |
| **Telemetry Ingest (`POST /api/v1/telemetry/ingest`)** | `main.py:303-341` | **Working** | Validates `TelemetryPayload`, runs `EnvironmentalSensorSwitching`, recomputes TOPSIS across all stored items, and appends to memory. |
| **Telemetry List (`GET /api/v1/telemetry`)** | `main.py:343-351` | **Working** | Returns in-memory `telemetry_store` array and `calculate_summary()` object. |
| **Dataset Explorer API (`GET /api/v1/dataset/explorer`)** | `main.py:354-429` | **Working** | Aggregates SRMD/telemetry statistics, modality split percentages, and up to 50 records. |
| **Inspection Image API (`GET /api/v1/images/{image_name}`)** | `main.py:432-469` | **Working** | Serves JPEG from `data/samples/images/` or `data/raw/India/test/images/`, or generates an inline SVG fallback placeholder. |
| **Static Mount (`/static/rdd2022/images`)** | `main.py:287-291` | **Working** | Mounts `data/samples/images` (or raw dir) via FastAPI `StaticFiles`. |
| **IoT Vehicle Simulator** | `simulation/mock_iot_vehicle.py:1-85` | **Mocked** | Generates uniform random telemetry around `19.0760, 72.8777` and POSTs to backend every 3s. |
| **YOLOv8 Training Pipeline** | `ml/train/trainer.py:42-250` | **Partial** | Calls `ultralytics.YOLO` if installed (`trainer.py:127-157`), else runs `_train_simulated()` (`trainer.py:159-250`) writing fake JSON `.pt` files. |
| **Model Evaluation Pipeline** | `ml/evaluate/evaluator.py:39-190` | **Broken** | Computes real IoU/Precision/Recall (`metrics.py:17-40,74-169`), but fabricates `mAP50` and `mAP50_95` via hardcoded multipliers (`metrics.py:171-186`). |
| **Multimodal Ablation Runner** | `services/experiments/experiment_runner.py:86-322` | **Working** | Runs 5 ablation configs (Exp A–E) over the 6 SRMD records and computes Spearman rank stability (`experiment_runner.py:36-83`). |

---

## 3. Paper-to-App Mapping

| Paper Layer & Component | Implementation Status | Factual Details & Code References |
| :--- | :--- | :--- |
| **Layer 1: Camera (Visual RGB)** | **Simulated / Hardcoded (Sample Default) & Real Loader** | By default, uses **5 synthetic 720×720 JPEG images** drawn with PIL (`ImageDraw.polygon`/`line` + Gaussian noise) in `scripts/download_rdd2022.py:307-415` (`data/samples/images/India_000045.jpg` through `India_000278.jpg`). `scripts/download_rdd2022.py:134-171` includes a real Figshare downloader for the official RDD2022 India zip, and `services/cv/preprocessing.py:24-70` loads real RGB images into NumPy arrays. |
| **Layer 1: LiDAR Sensor** | **Simulated / Hardcoded** | No point cloud or hardware driver exists. Modeled as a 1D scalar distance (`lidar_depth_cm = 30.0 + defect_depth_cm + noise`) generated deterministically with `seed=2026` in `services/data_generation/lidar_model.py:122-251` and `services/data_generation/consistency_model.py:249-255`. |
| **Layer 1: Acoustic SONAR Sensor** | **Simulated / Hardcoded** | Modeled as a 1D scalar distance (`sonar_depth_cm = 30.0 + defect_depth_cm + sonar_error`) in `services/data_generation/sonar_model.py:141-211` and `services/data_generation/consistency_model.py:258-262`. |
| **Layer 1: IMU / Accelerometer** | **Simulated / Hardcoded** | Modeled as a scalar vertical peak shock (`z_accel_g`) and RMS vibration (`rms_vibration_g`) using a parametric power-law equation in `services/data_generation/accelerometer_model.py:157-271` and `services/data_generation/consistency_model.py:173-233`. No 6-axis raw waveform or time-series IMU signal processing exists. |
| **Layer 1: Rain Sensor & Photometer** | **Simulated / Hardcoded** | Modeled as a boolean `rain_detected` (35% stochastic probability or override) and float `mean_luminance` (lux) in `services/dataset/srmd_builder.py:213-237` and `services/data_generation/sensor_augmentation.py:155-179`. |
| **Layer 1: GPS Receiver** | **Simulated / Hardcoded** | RDD2022 images have no EXIF GPS (`services/data_generation/location_model.py:9-12`). Coordinates are deterministically sampled from 6 hardcoded Mumbai waypoints (`WEH-MUM-SEC-01`..`06`) with $\pm 0.0012^\circ$ Gaussian jitter in `services/data_generation/location_model.py:39-94,132-220`. |
| **Layer 2: CLAHE Preprocessing** | **Missing** | Contrast Limited Adaptive Histogram Equalization (`cv2.createCLAHE`) is **not implemented** anywhere in `services/cv/preprocessing.py` or the codebase. |
| **Layer 2: ESRGAN Super-Resolution** | **Missing** | Enhanced Super-Resolution Generative Adversarial Network is **not implemented** anywhere in the repository. |
| **Layer 2: VoxelGrid Downsampling** | **Missing** | No 3D point cloud or VoxelGrid filtering (`open3d` / `pcl`) exists; LiDAR is represented solely as a single scalar float (`lidar_depth_cm`). |
| **Layer 2: Actual Preprocessing in Code** | **Partial (Unused in Inference)** | `services/cv/preprocessing.py:73-133` implements `letterbox()` (640×640 aspect-ratio resize + gray pad) and `denormalize_box_from_letterbox()`. Both are imported in `services/cv/detector.py:30` and `services/cv/rdd2022_detector.py:31` but **never called** anywhere in the codebase. |
| **Layer 3: YOLOv8 Object Detection** | **Partial (Wrapper Real, Weights Simulated)** | `services/cv/detector.py:222-278` (`UltralyticsYOLOAdapter`) wraps `ultralytics.YOLO(weights_path)`. However, `ultralytics` is not installed by default, and `runs/train/exp/weights/best.pt` is a **280-byte JSON text file** created by `DamageModelTrainer._train_simulated()` (`ml/train/trainer.py:222-231`). At runtime, `RDD2022Detector` (`services/cv/rdd2022_detector.py:58-73`) falls back to `GroundTruthLookupAdapter` (`detector.py:102-148`, which reads `data/processed/annotations_index.json` and assigns `confidence=0.98`) or `OpenCVContourDetectorAdapter` (`detector.py:151-219`). |
| **Layer 3: PoFormer (Transformer Segmentation)** | **Missing** | No Pothole Transformer, Vision Transformer, or semantic/instance segmentation mask model exists. Surface area (`surface_area_sqm`) is heuristically estimated from 2D bounding box pixel area and vertical perspective factor (`services/cv/rdd2022_detector.py:78-92`, `services/data_generation/consistency_model.py:59-79`). |
| **Layer 4: Weather Mode Switching** | **Real Implementation (Rule-Based)** | Implemented in `services/ai_engine/pipeline/depth_estimator.py:8-37` (`EnvironmentalSensorSwitching.evaluate_environment_and_measure`), `services/data_generation/sonar_model.py:109-139`, and `services/data_generation/consistency_model.py:264-278`. Switches from `optical_lidar` to `sonar_submerged_acoustic` whenever `rain_detected == True` or `mean_luminance < 40.0` lux (`depth_estimator.py:11`). |
| **Layer 4: Depth & Volume Profiling** | **Real Implementation (Arithmetic)** | Subtracts `baseline_chassis_height_cm = 30.0` from active sensor distance: `depth_cm = max(0.0, active_reading - 30.0)` (`depth_estimator.py:14,19`). Computes cavity volume in liters via parabolic bowl approximation: `volume_liters = (0.5 * (surface_area_sqm * 10000.0) * depth_cm) / 1000.0` (`depth_estimator.py:28`). Confirms dynamic impact when `z_accel_g > 1.5` (`depth_estimator.py:24`). |
| **Layer 5: Fuzzy BWM (Best-Worst Method)** | **Missing** | No Fuzzy Best-Worst Method solver, pairwise comparison matrices, or fuzzy triangular numbers exist. Criteria weights are hardcoded static constants `[0.35, 0.25, 0.25, 0.15]` in `services/backend/app/services/mcdm_engine.py:42`. |
| **Layer 5: VIKOR Ranking** | **Missing** | No VIKOR ($S_i, R_i, Q_i$ compromise solution) implementation exists anywhere in the codebase. |
| **Layer 5: TOPSIS Ranking** | **Real Implementation** | Vector-normalized TOPSIS is implemented in `services/backend/app/services/mcdm_engine.py:84-151` (`MCDMPrioritizationEngine.compute_topsis`) using NumPy and recomputed on every telemetry ingestion in `main.py:114-147`. |
| **Layer 5: Traffic, Road Class, Hospital & School Proximity Inputs** | **Partial (Traffic/Road Class/Hospital Simulated; School Missing)** | `traffic_pcu` and 4 `road_class` tiers (`LOCAL`, `COLLECTOR`, `ARTERIAL`, `HIGH_TRAFFIC_CORRIDOR`) are modeled via Greenshields flow in `services/data_generation/traffic_model.py:173-270`. `dist_hospital_km` is computed via Haversine distance $\times 1.18$ tortuosity against 8 hardcoded Mumbai hospitals in `services/data_generation/infrastructure_model.py:43-92,165-245`. Note that `road_class` is used to generate `traffic_pcu` and `speed_kmh`, not passed as a separate 5th column into the TOPSIS matrix (`mcdm_engine.py:34-40`). **School proximity is completely missing.** |
| **GIS Map / Digital Twin** | **Real Implementation (Frontend)** | Implemented in `frontend/src/components/Map/PotholeMap.tsx:101-453` using `react-leaflet` (`MapContainer`, `TileLayer`, `Marker`, `Circle`, `Popup`), custom priority markers (`PotholeMap.tsx:31-75`), and synchronized selection between map pins, table rows, and detail drawer (`frontend/src/app/page.tsx:143-197`). |
| **Automated Dispatch (Email / SMS / Work Orders)** | **Missing** | No email (SMTP/SendGrid), SMS (Twilio), webhook, or work-order generation endpoint/UI exists. The UI only displays a static badge reading `"Dispatch Urgency: {priority_level} Priority"` inside `frontend/src/components/Dashboard/IncidentDetailPanel.tsx:588-595`. |

---

## 4. Data and Logic

### Where Data Comes From
1. **Synthetic Sample Images & VOC XMLs (`data/samples/`):**
   - `scripts/download_rdd2022.py:307-415` generates 5 synthetic 720×720 road-surface JPEGs (`India_000045.jpg`, `India_000102.jpg`, `India_000155.jpg`, `India_000210.jpg`, `India_000278.jpg`) and 5 Pascal VOC XML files containing 6 bounding boxes across classes `D40` (2), `D20` (1), `D00` (2), and `D10` (1).
2. **Pre-Generated Multimodal JSON (`data/processed/srmd/srmd_records.json`):**
   - Generated deterministically (`seed=2026`) by `scripts/generate_srmd.py` using `SRMDRecordBuilder` (`services/dataset/srmd_builder.py:120-388`) and `PhysicalConsistencyModel` (`services/data_generation/consistency_model.py:34-405`). Contains 6 records (`SRMD-IND-000045-00` through `SRMD-IND-000278-00`).
3. **Backend Startup Seeding (`main.py:149-280`):**
   - On FastAPI startup (`seed_initial_data()`), `main.py:153-209` loads `data/processed/srmd/srmd_records.json` if present and ingests those 6 records into the in-memory list `telemetry_store` (`main.py:21`).
   - If `srmd_records.json` is missing, `main.py:213-273` falls back to 4 hardcoded sample dicts (`MUM_WEH_01`, `MUM_SV_02`, `MUM_LNK_03`, `MUM_res_04`).
   - **No database (SQL, NoSQL, SQLite, Redis, or ORM) exists anywhere in the project.** All backend state resides in the Python global list `telemetry_store: List[dict] = []` (`main.py:21`).
4. **Frontend Offline Fallback (`frontend/src/lib/api.ts:27-247`):**
   - If `http://localhost:8000` is unreachable or times out (4,000 ms `AbortController` in `api.ts:253-254`), `fetchTelemetryData()` returns `FALLBACK_DATA` (4 hardcoded records in `api.ts:27-191`) and `fetchDatasetExplorer()` returns `FALLBACK_DATASET_STATS` (hardcoded 6-record aggregate stats + 4 sample records in `api.ts:196-247`).
5. **Random IoT Simulator (`simulation/mock_iot_vehicle.py:20-54`):**
   - Generates random floats via `random.uniform` (e.g., `lidar_depth_cm` in `32.0..52.0`, `z_accel_g` in `1.1..3.2`, `traffic_pcu` in `2000..35000`) and POSTs to `/api/v1/telemetry/ingest`.

### ML Models Loaded and How
- **No neural network weights exist in the repository.**
- `runs/train/exp/weights/best.pt` (280 bytes) and `last.pt` (303 bytes) are plain-text JSON files containing:
  ```json
  {
    "model_architecture": "yolov8s",
    "num_classes": 4,
    "classes": ["D00", "D10", "D20", "D40"],
    "best_epoch": 15,
    "val_loss": 0.453,
    "simulated_checkpoint": true,
    "timestamp": "2026-04-06T22:55:27.515062+00:00"
  }
  ```
- In `services/cv/rdd2022_detector.py:58-73` (`RDD2022Detector._resolve_default_adapter`):
  1. Checks if `weights_path` exists and tries `UltralyticsYOLOAdapter(weights_path)`. Because `ultralytics` is not installed (or if installed, loading a JSON file as PyTorch weights raises an exception), it catches `(ImportError, Exception)` (`rdd2022_detector.py:64`).
  2. Next, checks if `data/processed/annotations_index.json` exists (`rdd2022_detector.py:67-69`). Since this file exists in the repo, **`RDD2022Detector` always selects `GroundTruthLookupAdapter`** (`services/cv/detector.py:102-148`), which looks up the image filename in `annotations_index.json` and returns the XML ground-truth boxes with a hardcoded confidence of `0.98` (`detector.py:130`).
  3. If an image is not in `annotations_index.json`, `GroundTruthLookupAdapter` returns a hardcoded center bounding box `[w*0.25, h*0.35, w*0.75, h*0.75]` of class `D40` with confidence `0.95` (`detector.py:138-147`).

### Actual MCDM Formulas and Weights in Code
1. **TOPSIS Engine (`services/backend/app/services/mcdm_engine.py:31-151`):**
   - **Decision Matrix $X \in \mathbb{R}^{N \times 4}$ (`main.py:121-129`):**
     Each row $i$ represents a defect event with 4 criteria columns:
     - $C_0$: `calculated_volume_liters` (Benefit: `True`, Weight $w_0 = 0.35$)
     - $C_1$: `traffic_pcu` (Benefit: `True`, Weight $w_1 = 0.25$)
     - $C_2$: `dist_hospital_km` (Cost: `False`, Weight $w_2 = 0.25$)
     - $C_3$: `speed_kmh` (Benefit: `True`, Weight $w_3 = 0.15$)
   - **Single-Record Fallback (`mcdm_engine.py:92-93`):** If $N = 1$, returns `[0.5]`.
   - **Step 1 — Vector Normalization (`mcdm_engine.py:96-99`):**
     $$r_{ij} = \frac{x_{ij}}{\sqrt{\sum_{k=1}^{N} x_{kj}^2}}$$
     (Guards zero denominators by substituting `1e-9` where column norm is `0`).
   - **Step 2 — Weighted Normalized Matrix (`mcdm_engine.py:102`):**
     $$v_{ij} = w_j \cdot r_{ij}$$
   - **Step 3 — Ideal Best ($A^+$) and Ideal Worst ($A^-$) (`mcdm_engine.py:105-116`):**
     - For benefit criteria ($j \in \{0, 1, 3\}$): $A_j^+ = \max_i v_{ij}$, $A_j^- = \min_i v_{ij}$.
     - For cost criterion ($j = 2$, hospital distance): $A_2^+ = \min_i v_{i2}$, $A_2^- = \max_i v_{i2}$.
   - **Step 4 — Euclidean Separation Measures (`mcdm_engine.py:119-120`):**
     $$S_i^+ = \sqrt{\sum_{j=0}^{3} (v_{ij} - A_j^+)^2}, \quad S_i^- = \sqrt{\sum_{j=0}^{3} (v_{ij} - A_j^-)^2}$$
   - **Step 5 — Relative Closeness Coefficient (`mcdm_engine.py:122-135`):**
     - If $S_i^+ + S_i^- == 0$ (all alternatives identical), returns `0.5` for those rows.
     - Otherwise:
       $$C_i^* = \text{clip}\left(\frac{S_i^-}{S_i^+ + S_i^-}, 0.0, 1.0\right)$$
   - **Priority Classification Thresholds (`mcdm_engine.py:137-150`):**
     - `score >= 0.75` $\rightarrow$ `"Critical"`
     - `score >= 0.50` $\rightarrow$ `"High"`
     - `score >= 0.25` $\rightarrow$ `"Medium"`
     - `score < 0.25` $\rightarrow$ `"Low"`
2. **Offline Single-Record Approximation (`services/data_generation/consistency_model.py:284-328`):**
   - Used during static SRMD generation (`srmd_builder.py:255`) before batch TOPSIS:
     $$\text{risk\_score} = 0.35 \cdot S_{\text{vol}} + 0.25 \cdot S_{\text{traffic}} + 0.25 \cdot S_{\text{hosp}} + 0.15 \cdot S_{\text{speed}}$$
     where $S_{\text{vol}} = 0.6 \cdot \text{defect\_severity} + 0.4 \cdot \min(1, \text{volume}/60)$, $S_{\text{traffic}} = \min(1, \text{pcu}/35000)$, $S_{\text{hosp}} = \max(0, 1 - \text{dist}/3.0)$, $S_{\text{speed}} = \min(1, \text{speed}/80)$.

### State Management
- **Frontend (`frontend/src/app/page.tsx:17-29`):** Local React `useState` hooks in `DashboardPage` (`telemetryData`, `selectedEvent`, `isLoading`, `isSyncing`, `activeFilter`, `simulationMode`, `activeView`, `connectionMode`, `lastUpdated`). Props are drilled one level down to `Header`, `SummaryPanel`, `PotholeMap`, `TelemetryTable`, `IncidentDetailPanel`, and `DatasetExplorerPanel`. No Redux, Zustand, or React Context is used.
- **Backend (`main.py:21-24`):** Module-level Python global list `telemetry_store` mutated in-place by `process_and_store_telemetry()` (`main.py:66-112`) and `recalculate_all_priorities()` (`main.py:114-147`).

### Backend API Endpoints (`main.py`)

| Method | Path | Handler (`main.py`) | Description |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | `root()` (`lines 293-301`) | Health check returning active defect count and weather state. |
| `POST` | `/api/v1/telemetry/ingest` | `ingest_telemetry()` (`lines 303-341`) | Validates `TelemetryPayload` (`lines 31-63`), evaluates depth/modality, updates TOPSIS across all stored records, returns `{"status": "success", "data": record}`. |
| `GET` | `/api/v1/telemetry` | `get_all_telemetry()` (`lines 343-351`) | Returns `{"status": "success", "count": len(telemetry_store), "summary": calculate_summary(), "data": telemetry_store}`. |
| `GET` | `/api/v1/dataset/explorer` | `get_dataset_explorer()` (`lines 354-429`) | Computes summary stats over `data/processed/srmd/srmd_records.json` and `telemetry_store`. |
| `GET` | `/api/v1/images/{image_name}` | `get_inspection_image()` (`lines 432-469`) | Serves JPEG file from `data/samples/images/` or returns a dynamic SVG placeholder (`lines 445-469`). |
| `GET` | `/static/rdd2022/images/{filename}` | `StaticFiles` (`lines 287-291`) | Static file mount for `data/samples/images`. |

---

## 5. Metrics Honesty Check

### A. Numbers Shown in the Web UI

| Metric Shown in UI | Value(s) Displayed | Code Origin | Live Computed vs. Hardcoded |
| :--- | :--- | :--- | :--- |
| **Polling Interval** (Header health popover) | `"3,000 ms (Real-time)"` | `frontend/src/components/Dashboard/Header.tsx:231` | **Hardcoded string** in JSX (matches the `3000` ms `setInterval` in `page.tsx:78`). |
| **Total Defects Detected** (Summary KPI Card 1) | `6` (live backend) or `4` (offline fallback) | `frontend/src/components/Dashboard/SummaryPanel.tsx:41,92` | **Computed live** (`data.length` of filtered `telemetryData`). |
| **Critical Priority Count & %** (Summary KPI Card 2) | e.g., `2` (`33%` of total) | `frontend/src/components/Dashboard/SummaryPanel.tsx:42,113,120` | **Computed live** from `data.filter(d => d.priority_level === 'Critical')`. |
| **Submerged / Sonar Count** (Summary KPI Card 3) | e.g., `2` (live) or `1` (fallback) | `frontend/src/components/Dashboard/SummaryPanel.tsx:46-48,139` | **Computed live** from `is_submerged` or `sonar_submerged_acoustic`. |
| **Dynamic Impact (>1.5g) Count** (Summary KPI Card 4) | e.g., `2` (`33%` confirmed) | `frontend/src/components/Dashboard/SummaryPanel.tsx:49-51,163` | **Computed live** from `dynamic_impact_confirmed` or `z_accel_g > 1.5`. |
| **Mean TOPSIS Urgency Index** (Summary KPI Card 5) | e.g., `0.482` (live) or `0.505` (fallback) | `frontend/src/components/Dashboard/SummaryPanel.tsx:53-55,186` | **Computed live** as arithmetic mean of `topsis_score` across active records. |
| **Visual Detection Confidence** (Table & Incident Detail Drawer) | `100.0%` (`1.0`) for all 6 SRMD records on live backend; `96.0%`, `91.0%`, `88.0%`, `84.0%` in fallback mode | `services/dataset/srmd_builder.py:279` (`confidence: 1.0`), `main.py:224,238,252,266`, `frontend/src/lib/api.ts:65,98,133,167`, `IncidentDetailPanel.tsx:251` | **Hardcoded per generator/fallback.** Not produced by a neural network softmax/objectness score. |
| **Chassis Baseline Reference** (Incident Detail Drawer) | `"30.0 cm"` | `frontend/src/components/Dashboard/IncidentDetailPanel.tsx:333` | **Hardcoded constant** (`baselineChassisCm = 30.0` at `IncidentDetailPanel.tsx:167`). |
| **Shock Threshold Reference** (Incident Detail Drawer) | `"1.50g"` / `"Threshold: 1.50g"` | `frontend/src/components/Dashboard/IncidentDetailPanel.tsx:166,431,449` | **Hardcoded constant** (`impactThresholdG = 1.5` at `IncidentDetailPanel.tsx:166`). |
| **TOPSIS Criterion Weights** (Incident Detail Drawer & Explorer) | `35%` Volume, `25%` Traffic, `25%` Hospital, `15%` Speed | `frontend/src/components/Dashboard/IncidentDetailPanel.tsx:536,547,558,569` and `DatasetExplorerPanel.tsx:136` | **Hardcoded labels** in JSX matching the static weights in `mcdm_engine.py:42`. |
| **TOPSIS Progress Bars in Drawer** (Volume, Traffic, Hospital, Speed) | Percentage bar widths (`0%`–`100%`) | `frontend/src/components/Dashboard/IncidentDetailPanel.tsx:180-184` | **Computed live via UI heuristic** (`volume/50*100`, `traffic/35000*100`, `(1 - dist/5)*100`, `speed/80*100`), not the actual vector-normalized $v_{ij}$ from `mcdm_engine.py`. |
| **Dataset Explorer Aggregate Stats** (Avg LiDAR, Avg Sonar, Avg Accel, Avg PCU, Avg Hospital Dist, Modality %) | Live: computed from `srmd_records.json` (`38.23 cm`, `39.13 cm`, `1.59 g`, `16,801 PCU`, `0.73 km`); Fallback: hardcoded (`39.2 cm`, `40.4 cm`, `1.73 g`, `17,867 PCU`, `0.69 km`) | Live: `main.py:354-429`; Fallback: `frontend/src/lib/api.ts:196-247` | **Computed live** when backend is running; **hardcoded** in `FALLBACK_DATASET_STATS` when offline. |

### B. Numbers in Evaluation Artifacts (`runs/`) and `README.md`

| Metric | Value in File | Code Origin | Honesty Verdict |
| :--- | :--- | :--- | :--- |
| **`mAP50`** in `runs/evaluate/eval_report.json:4` | `0.1` (`10.0%`) | `ml/evaluate/metrics.py:184`: `mAP50 = float(np.clip(macro_prec * macro_rec + 0.15 * macro_rec, 0.0, 1.0))` | **Fabricated formula.** Does not integrate the area under the Precision-Recall curve (`compute_ap_from_pr` at `metrics.py:43-59` is dead code, never called). Instead, adds an arbitrary `+ 0.15 * macro_rec` bonus to `macro_prec * macro_rec`. |
| **`mAP50_95`** in `runs/evaluate/eval_report.json:5` | `0.85` (`85.0%`) | `ml/evaluate/metrics.py:174-186`: loops `for iou_th in np.linspace(0.50, 0.95, 10):` without using `iou_th` inside the loop body, then computes `np.mean(map_95_scores) * 0.85`. | **Broken / Fabricated.** Because `iou_th` is ignored in `metrics.py:174-181`, it averages `1.0` ten times and multiplies by `0.85`, yielding `mAP50_95 = 0.85` even though `mAP50 = 0.10` (mathematically impossible since $\text{mAP}_{50:95} \le \text{mAP}_{50}$). |
| **Per-Class `ap50`** in `runs/evaluate/eval_report.json:37,45,53,61` | `D40: 1.0`, others `0.0` | `ml/evaluate/metrics.py:165`: `ap50 = round(prec * rec, 4)` | **Inaccurate formula.** Computes $P \times R$ at a single confidence cutoff rather than 11-point or 101-point interpolated Average Precision. |
| **Training Loss Curve** in `runs/train/exp/training_results.json:8-84` | Epoch 1..15 `train_box_loss` (`1.8424` $\rightarrow$ `0.5535`), `val_loss` (`1.6822` $\rightarrow$ `0.453`) | `ml/train/trainer.py:194-205` (`_train_simulated`) | **Simulated exponential decay.** Computed from `1.85 * math.exp(-0.08 * epoch) + rng.uniform(-0.04, 0.04)`, not real gradient descent. |
| **mIoU, Processing Latency, Dispatch Time** | Not present | Nowhere in `frontend/` or `services/` | **Not implemented / Not shown** anywhere in the UI or backend. |

---

## 6. UI/UX Review

### Layout & Navigation
- **Two-View Single-Page Architecture (`frontend/src/app/page.tsx:114-275`):**
  - Top sticky navbar (`Header.tsx:38`) switches between two views via state (`activeView`):
    1. **Digital Twin Dashboard (`page.tsx:115-268`):** Top KPI cards + 4-column analytics strip (`SummaryPanel.tsx`), followed by a 12-column main grid with a 7-column interactive Leaflet map (`lg:col-span-7`, `height: 580px`) on the left and a 5-column Priority Repair Queue (`lg:col-span-5`, `height: 580px`) on the right.
    2. **Research Dataset Explorer (`page.tsx:269-274`):** Full-width analytical view (`DatasetExplorerPanel.tsx`) with 4 sub-tabs (`All Multimodal`, `RDD2022 Visual`, `Synthetic Sensor`, `GIS & TOPSIS`).
  - Clicking any map pin, table row, Top-3 Urgency Queue item, or Explorer "Inspect" button opens the right-side slide-over drawer (`IncidentDetailPanel.tsx:208`).

### Responsiveness (Mobile / Tablet)
- **Strengths:** Uses Tailwind responsive breakpoints (`grid-cols-1 sm:grid-cols-2 lg:grid-cols-5` in `SummaryPanel.tsx:66`, `grid-cols-1 lg:grid-cols-12` in `page.tsx:125`, `max-w-xl` slide-over drawer in `IncidentDetailPanel.tsx:214`).
- **Issues on Mobile (`< 640px`):**
  - `Header.tsx:38-211` places the brand logo, 2 view switcher tabs, simulation mode pill, critical/high badge, weather badge, and sync button in a single `h-16` horizontal flex bar. While secondary badges hide via `hidden md:flex` (`Header.tsx:107`) and `hidden xl:flex` (`Header.tsx:136`), the title + navigation tabs + sync button overflow horizontally on `320px–375px` viewports.
  - `page.tsx:128` and `page.tsx:252` fix both the map container and telemetry table at `h-[580px]`, requiring excessive vertical scrolling on mobile screens.

### Accessibility (a11y) Issues
1. **Slide-Over Modal Lacks Dialog Semantics & Focus Management (`IncidentDetailPanel.tsx:208-220`):**
   - No `role="dialog"`, `aria-modal="true"`, or `aria-labelledby` attributes.
   - No `Escape` key listener to close the drawer, and focus is not trapped inside the drawer when opened.
2. **Health Popover Lacks `aria-expanded` and Keyboard Dismissal (`Header.tsx:45-64, 214-239`):**
   - The brand button opens a floating status card without `aria-expanded` or `aria-haspopup`, and cannot be dismissed with `Escape`.
3. **Clickable `<tr onClick>` Elements Not Keyboard Accessible (`TelemetryTable.tsx:218`, `DatasetExplorerPanel.tsx:557`):**
   - Table rows rely on mouse `onClick` without `tabIndex={0}` or `onKeyDown` (`Enter`/`Space`) handlers.
4. **Low Contrast Micro-Typography:**
   - Extensive use of `text-[10px]` and `text-[11px]` in `text-slate-500` (`#64748b`) against `bg-slate-900` (`#0f172a`) (`SummaryPanel.tsx:96,120`, `TelemetryTable.tsx:86`), yielding a contrast ratio of ~`3.7:1` (below the WCAG AA `4.5:1` requirement for small text).

### Loading, Error, and Empty States
- **Loading State:** Implemented for initial telemetry fetch (`page.tsx:18, 161-165`), map dynamic import (`MapWrapper.tsx:21-27`), and Dataset Explorer initial load (`DatasetExplorerPanel.tsx:88-96`).
- **Error State:** Next.js route error boundaries (`error.js`, `global-error.js`) exist. API fetch errors do not crash the UI; instead, `api.ts:261-269` catches network failures, logs a `console.warn`, and seamlessly renders `FALLBACK_DATA` while switching the header badge to `"DEMO FALLBACK MODE"` (`Header.tsx:76-82`).
- **Empty State:** Handled in `TelemetryTable.tsx:194-213` (shows filter reset button when 0 defects match) and `DatasetExplorerPanel.tsx:547-553`.

### Theming, Charts, and Map Libraries
- **Theming:** Dark-only slate/indigo command-center theme (`#0B0F17` background, `#F8FAFC` text in `globals.css:4-8`). Custom priority colors (`critical: #EF4444`, `high: #F97316`, `medium: #EAB308`, `low: #22C55E`) defined in `tailwind.config.js:11-16`.
- **Charts:** **Zero third-party charting libraries** (no Recharts, Chart.js, or D3). All bar charts, progress gauges, and distribution strips in `SummaryPanel.tsx`, `IncidentDetailPanel.tsx`, and `DatasetExplorerPanel.tsx` are built with pure Tailwind CSS `div` width percentages.
- **Map Library:** `leaflet` (`1.9.4`) + `react-leaflet` (`4.2.1`) in `frontend/src/components/Map/PotholeMap.tsx`.

---

## 7. Code Quality

### Duplication
1. **Triple Duplication of Environmental Sensor Switching & Volume Math:**
   - Implemented independently in:
     1. `services/ai_engine/pipeline/depth_estimator.py:8-37` (`EnvironmentalSensorSwitching.evaluate_environment_and_measure`)
     2. `services/data_generation/sonar_model.py:109-139` (`SonarModel.evaluate_switching_recommendation`)
     3. `services/data_generation/consistency_model.py:264-278` (`PhysicalConsistencyModel.derive_sensor_measurements`)
     4. Plus re-implemented in TypeScript on the frontend in `IncidentDetailPanel.tsx:167-169` and `TelemetryTable.tsx:221`.
2. **Dual Duplication of SRMD Record Synthesis:**
   - `services/dataset/srmd_builder.py:120-388` (`SRMDRecordBuilder`) and `services/data_generation/sensor_augmentation.py:67-329` (`SensorAugmentationEngine`) both build the exact same 12-key SRMD record dictionary structure with slightly different internal model calls (`SRMDRecordBuilder` uses `PhysicalConsistencyModel`, whereas `SensorAugmentationEngine` calls `LiDARModel`, `SonarModel`, and `AccelerometerModel` directly).
3. **Duplicated Fallback Seed Data:**
   - The 4 Mumbai fallback telemetry records (`MUM_WEH_01`..`04`) are duplicated in both `main.py:213-273` and `frontend/src/lib/api.ts:27-191`.

### Dead Code & Unused Imports
1. `services/cv/preprocessing.py:73-133`: `letterbox()` and `denormalize_box_from_letterbox()` are imported in `services/cv/detector.py:30` and `services/cv/rdd2022_detector.py:31` but **never called**.
2. `ml/evaluate/metrics.py:43-59`: `compute_ap_from_pr()` is defined to compute area under the Precision-Recall curve using `np.trapz`, but is **never called** anywhere in `metrics.py` or the rest of the codebase.
3. `frontend/src/lib/api.ts:6`: `const API_BASE = BACKEND_URL.replace('/api/v1/telemetry', '')` is declared and **never used**.
4. `frontend/next.config.mjs:6-13`: Configures a rewrite `/api/backend/:path*` $\rightarrow$ `http://localhost:8000/api/v1/:path*`, which is **never used** by `frontend/src/lib/api.ts`.

### Large Files (> 450 lines)
- `test_core_domain.py`: **1,798 lines** (monolithic test file containing 10 test classes and 83 tests).
- `frontend/src/components/Dashboard/DatasetExplorerPanel.tsx`: **693 lines**.
- `frontend/src/components/Dashboard/IncidentDetailPanel.tsx`: **615 lines**.
- `services/dataset/srmd_builder.py`: **529 lines**.
- `scripts/generate_srmd.py`: **483 lines**.
- `main.py`: **476 lines**.
- `frontend/src/components/Map/PotholeMap.tsx`: **454 lines**.
- `scripts/download_rdd2022.py`: **454 lines**.

### Missing Types, Tests, and Lint Status
- **TypeScript (`npx tsc --noEmit`):** Passes with **0 errors**. However, `frontend/src/app/error.js` and `frontend/src/app/global-error.js` are untyped `.js` files, and `PotholeMap.tsx:16,80` uses `any` for Leaflet marker prototypes and `mapRef`.
- **ESLint (`npm run lint`):** **Unconfigured / Broken in non-interactive CI.** Running `npm run lint` triggers Next.js's interactive prompt (`? How would you like to configure ESLint?`) because no `.eslintrc.json` exists and `eslint` / `eslint-config-next` are missing from `frontend/package.json`.
- **Frontend Unit/Integration Tests:** **0 tests** (no Jest, Vitest, or Playwright setup in `frontend/`).
- **Backend Unit Tests (`python -m unittest test_core_domain.py`):** **83 tests pass** in `7.18s`. However, running `test_core_domain.py` has a workspace side effect: `test_12_reproducibility` (`test_core_domain.py:1765-1768`) calls `ResearchDataSplitter(seed=2026).split_image_ids(...)` without a temporary directory, writing `data/metadata/dataset_splits.json` into the repo tree.

### Hardcoded Secrets or API Keys
- **None found.** No private API keys, tokens, or credentials exist in `.env.local.example` or source files. Only `http://localhost:8000` URLs are hardcoded (`frontend/src/lib/api.ts:3`, `IncidentDetailPanel.tsx:200-201`).

### Unused Dependencies (`frontend/package.json`)
- **`clsx` (`^2.1.1`)** (`frontend/package.json:11`): Never imported in any file under `frontend/src/`.
- **`tailwind-merge` (`^2.5.2`)** (`frontend/package.json:18`): Never imported in any file under `frontend/src/`.

### Outdated or Vulnerable Packages (`npm audit`)
Running `npm audit` in `c:\smart-road-twin\frontend` exited with code `1` and reported **10 vulnerabilities (1 Critical, 7 High, 2 Moderate)**:

| Package | Installed / Vulnerable Range | Severity | Key Advisory / Issue |
| :--- | :--- | :--- | :--- |
| **`next`** | `14.2.15` (`0.9.9 - 16.3.0-preview.10`) | **Critical** | 32 advisories including Unauthenticated RCE on Windows servers (`GHSA-p293-qw3h-jr36`), RCE in Image Optimization API (`GHSA-2xp9-vwfh-vxw4`), Middleware Authorization Bypass (`GHSA-f82v-jwr5-mffw`), SSRF, Cache Poisoning, and Server Components DoS. |
| **`postcss`** (nested in `next`) | `<=8.5.22` | **High** | XSS via unescaped `</style>` (`GHSA-qx2v-qp2m-jg93`) & path traversal / arbitrary `.map` file read (`GHSA-6g55-p6wh-862q`, `GHSA-fxqj-rqcc-2cmp`, `GHSA-r28c-9q8g-f849`). |
| **`braces`** (via `chokidar` / `tailwindcss`) | `*` | **High** | Stack-exhaustion DoS via deeply nested patterns (`GHSA-vfj7-8cjw-p6xm`). |
| **`source-map-js`** | `1.0.0 - 1.2.1` | **High** | Event-loop DoS through indexed source-map section offsets (`GHSA-68fv-2mgg-jv7q`). |
| **`postcss-selector-parser`** | `<7.1.6` | **Moderate** | Quadratic complexity CPU exhaustion (`GHSA-rj75-hqrm-r3gf`). |

---

## 8. Performance

### Next.js Bundle Size (`frontend/.next/static/chunks/`)
- Total top client JS chunks inspected in `frontend/.next/static/chunks/`:
  - `fd9d1056-501b3fb451a730f0.js`: **172.8 KB** (uncompressed)
  - `d0deef33.bd75b11a952a2d3d.js`: **148.5 KB** (Leaflet bundle chunk)
  - `framework-f66176bb897dc684.js`: **140.0 KB**
  - `main-307be110f2f813c1.js`: **125.5 KB**
  - `117-ad07b5942fc8a6b6.js`: **123.5 KB**
  - `polyfills-42372ed130431b0a.js`: **112.6 KB**
  - `page-0f0ef6863a6b8216.js`: **61.3 KB** (App Router `/` page chunk containing all dashboard components)
- Because `DatasetExplorerPanel` (693 lines) and `IncidentDetailPanel` (615 lines) are statically imported in `frontend/src/app/page.tsx:8-9`, their code is bundled into the initial `/` page chunk (`61.3 KB`) even before the user opens the detail drawer or switches to the Dataset Explorer tab.

### Heavy Assets
- Sample images in `data/samples/images/` are lightweight (**39.1 KB to 44.1 KB** each; ~206 KB total for all 5 JPEGs).
- `IncidentDetailPanel.tsx:260` renders raw `<img>` tags rather than Next.js `<Image>`, bypassing image optimization, although sample JPEGs are already small.

### Unnecessary Re-Renders & Polling Overhead
1. **3-Second Unconditional State Replacement (`frontend/src/app/page.tsx:53-79`):**
   - Every 3,000 ms, `loadData(true)` fetches `/api/v1/telemetry` and calls `setTelemetryData(response.data)` with a brand-new array reference even when zero telemetry records have changed.
   - This triggers a full re-render every 3 seconds of `DashboardPage`, `Header`, `SummaryPanel`, `TelemetryTable`, `MapWrapper`, and `PotholeMap`.
2. **Leaflet `L.divIcon` Recreation on Every Render (`frontend/src/components/Map/PotholeMap.tsx:231-244`):**
   - Inside `data.map((defect) => ...)`, `createPriorityIcon(defect.priority_level, isSelected, ...)` is called on every 3-second poll without memoization, rebuilding DOM strings and Leaflet `DivIcon` instances for every marker on the map.
3. **Un-memoized Filtering & Sorting in `SummaryPanel.tsx:41-63`:**
   - Unlike `TelemetryTable.tsx:66-75` (which uses `useMemo`), `SummaryPanel` recomputes 11 `.filter()` / `.reduce()` / `.sort()` passes directly in the render body on every 3-second poll.

### Main-Thread / Event-Loop Blocking
- **Backend (`main.py:114-147` & `main.py:354-429`):**
  - `ingest_telemetry()` is declared `async def` (`main.py:304`), and on every single ingested packet it calls `recalculate_all_priorities()`, which rebuilds an $N \times 4$ NumPy matrix and runs TOPSIS over the entire `telemetry_store` synchronously on the event loop.
  - `get_dataset_explorer()` (`main.py:354`) is also `async def` and performs synchronous disk I/O (`open(srmd_path, "r")` + `json.load(f)` at `main.py:362-363`) on every request instead of caching or using a threadpool.

---

## 9. Known Bugs and Risks

1. **Mathematically Broken `mAP50` and `mAP50_95` Evaluation (`ml/evaluate/metrics.py:171-186`):**
   - `mAP50` (`line 184`) is calculated as `macro_prec * macro_rec + 0.15 * macro_rec` instead of integrating the PR curve (`compute_ap_from_pr` on `lines 43-59` is never called).
   - `mAP50_95` (`lines 174-186`) loops over `for iou_th in np.linspace(0.50, 0.95, 10):` (`line 174`), but **never uses `iou_th` inside the loop body**, evaluating pre-computed `iou_threshold=0.50` matches 10 times and multiplying by `0.85`. In `runs/evaluate/eval_report.json:4-5`, this produces `mAP50 = 0.1` and `mAP50_95 = 0.85` ($\text{mAP}_{50:95} > \text{mAP}_{50}$, which is impossible).
2. **Fake JSON Checkpoint Saved as `.pt` Breaks `UltralyticsYOLOAdapter` (`ml/train/trainer.py:227-231` & `services/cv/detector.py:253`):**
   - `_train_simulated()` writes a 280-byte JSON document to `runs/train/exp/weights/best.pt`. If a developer installs `ultralytics` and runs inference with `best.pt`, `YOLO(str(self.weights_path))` (`detector.py:253`) will fail with a PyTorch `UnpicklingError` / corrupt zip archive error.
3. **`api_test.py` Silently Tests Zero Events (`api_test.py:23-39`):**
   - `api_test.py:23` extracts `events = data.get("events", [])` from `GET /api/v1/telemetry`. However, `main.py:349` returns the list under the key `"data"`, not `"events"`. Consequently, `events` is always `[]`, the `for ev in events:` validation loop (`lines 27-38`) executes **0 iterations**, and `line 39` prints a false-positive `[OK] Event provenance fields valid for all 0 events`.
4. **Hardcoded `http://localhost:8000` Image URL in `IncidentDetailPanel.tsx:200-201`:**
   - When constructing `imgUrl`, `IncidentDetailPanel.tsx:201` hardcodes `` `http://localhost:8000${event.image_url}` `` instead of reading `process.env.NEXT_PUBLIC_API_URL`. In any non-localhost deployment, inspection images fail to load and fall back to the client SVG placeholder.
5. **Startup SRMD Volume Overwrite Uses LiDAR Depth Even When Switched to Sonar (`main.py:163-176`):**
   - In `seed_initial_data()`, `main.py:163` runs `sensor_switcher.evaluate_environment_and_measure(...)` which computes `calculated_volume_liters` using the active sensor (`sonar_depth_cm` when `rain_detected=True` or `mean_luminance < 40`). Immediately afterward, `main.py:175` overwrites `depth_eval["calculated_volume_liters"] = r["visual"].get("calculated_volume_liters", ...)`, which was pre-calculated in `srmd_builder.py:185` from `defect_depth_cm` (LiDAR effective depth) rather than the switched Sonar depth.
6. **YOLO Dataset Path Mismatch Between Trainer and Config (`ml/configs/rdd2022_yolo.yaml:4` vs. `ml/train/trainer.py:81`):**
   - `ml/configs/rdd2022_yolo.yaml:4` sets `path: data/processed/yolo_rdd2022`, whereas `ml/train/trainer.py:81` prepares and writes the YOLO dataset to `data/processed/yolo`.
7. **Inconsistent Traffic PCU Time Units Across UI and Docs (`DatasetExplorerPanel.tsx:386` vs. `README.md:56`):**
   - `DatasetExplorerPanel.tsx:386` labels `avg_traffic_pcu` (`16,801 PCU`) as `"Passenger Car Units / hour"`, whereas `README.md:56` documents `traffic_pcu` (`1,500–38,000`) as `"PCU/day"`, and `services/data_generation/traffic_model.py:11,36` mixes hourly Greenshields capacity with daily corridor PCU numbers (`24,000–38,000`).
8. **Tracked `__pycache__/*.pyc` Files in Git Repository:**
   - Although `.gitignore:11-12` lists `__pycache__/` and `*.py[cod]`, three compiled `.pyc` files (`__pycache__/main.cpython-311.pyc`, `__pycache__/test_core_domain.cpython-311.pyc`, `services/backend/app/services/__pycache__/mcdm_engine.cpython-311.pyc`) are tracked in the Git index and show as modified whenever Python runs on the repo.
9. **Unbounded In-Memory `telemetry_store` & Wildcard CORS (`main.py:15,21`):**
   - `telemetry_store` grows without bound on every `POST /api/v1/telemetry/ingest` request, and `allow_origins=["*"]` with `allow_credentials=True` (`main.py:15-16`) is insecure for production deployment.

---

## 10. Top 15 Upgrade Opportunities (Ranked by Impact vs. Effort)

### Quick Wins (High Impact, Low Effort)
1. **Fix `mAP50` and `mAP50_95` calculation in `ml/evaluate/metrics.py:171-186`:**
   - Wire up the existing `compute_ap_from_pr()` function (`metrics.py:43-59`) and actually pass `iou_th` into IoU matching inside the `np.linspace(0.50, 0.95, 10)` loop so evaluation metrics are mathematically genuine.
2. **Fix `api_test.py:23` key mismatch (`"events"` $\rightarrow$ `"data"`):**
   - Change `data.get("events", [])` to `data.get("data", [])` and assert `len(events) > 0` so the API test script actually validates the returned telemetry events.
3. **Remove hardcoded `http://localhost:8000` in `IncidentDetailPanel.tsx:200-201`:**
   - Derive the backend origin from `process.env.NEXT_PUBLIC_API_URL` (or a shared helper in `frontend/src/lib/api.ts`) so inspection images work outside `localhost`.
4. **Add a root `requirements.txt` / `pyproject.toml` and untrack `__pycache__/*.pyc`:**
   - Pin `fastapi`, `uvicorn`, `pydantic`, `numpy`, `pyyaml`, `pillow`, `requests` in `requirements.txt` and run `git rm --cached -r __pycache__ services/backend/app/services/__pycache__`.
5. **Upgrade `next` and run `npm audit fix` in `frontend/`:**
   - Resolve the 10 npm vulnerabilities (including the Critical Next.js advisories) and remove unused `clsx` and `tailwind-merge` packages from `frontend/package.json:11,18`.

### Medium Upgrades (High Impact, Moderate Effort)
6. **Implement Real CLAHE Preprocessing in `services/cv/preprocessing.py`:**
   - Add an actual `apply_clahe(img_rgb, clip_limit=2.0, tile_grid_size=(8,8))` step (via OpenCV LAB color space or PIL/NumPy histogram equalization) and invoke `letterbox()` + `denormalize_box_from_letterbox()` inside `detector.py`.
7. **Implement Fuzzy BWM Weight Derivation and VIKOR Comparison in `services/backend/app/services/mcdm_engine.py`:**
   - Add Fuzzy Best-Worst Method weight calculation (or configurable weight profiles) and a `compute_vikor()` method ($S_i, R_i, Q_i$) alongside `compute_topsis()` to match Layer 5 of the research paper.
8. **Consolidate Duplicated Sensor Switching & SRMD Synthesis Logic:**
   - Unify `EnvironmentalSensorSwitching` (`depth_estimator.py`), `SonarModel.evaluate_switching_recommendation` (`sonar_model.py`), and `PhysicalConsistencyModel.derive_sensor_measurements` (`consistency_model.py`), and merge `SRMDRecordBuilder` (`srmd_builder.py`) with `SensorAugmentationEngine` (`sensor_augmentation.py`).
9. **Prevent Unnecessary 3-Second Full-Tree React Re-Renders (`frontend/src/app/page.tsx:53-79` & `PotholeMap.tsx:231`):**
   - Compare incoming telemetry payload hashes/timestamps before calling `setTelemetryData`, memoize `SummaryPanel` calculations with `useMemo`, and memoize Leaflet `DivIcon` creation in `PotholeMap.tsx`.
10. **Fix Accessibility (`a11y`) in `IncidentDetailPanel.tsx` and `TelemetryTable.tsx`:**
    - Add `role="dialog"`, `aria-modal="true"`, `Escape` key handler, and focus management to `IncidentDetailPanel.tsx:208`, and add keyboard navigation (`tabIndex={0}`, `onKeyDown`) to table rows in `TelemetryTable.tsx:218`.

### Major Upgrades (Transformational Impact, Higher Effort)
11. **Train and Bundle a Real Lightweight YOLOv8n (`best.onnx` / `best.pt`) Checkpoint:**
    - Train a real `yolov8n` model on RDD2022 India images and export to ONNX Runtime (`onnxruntime`) or PyTorch so `RDD2022Detector` runs genuine neural network inference instead of `GroundTruthLookupAdapter` (`detector.py:102-148`) and fake JSON `.pt` files (`trainer.py:227-231`).
12. **Add Pothole Mask Segmentation (PoFormer / YOLOv8-seg) & Point-Cloud VoxelGrid Processing:**
    - Replace bounding-box pixel-area heuristics (`consistency_model.py:59-79`) with polygon mask segmentation (computing true mIoU and mask surface area) and 3D VoxelGrid point-cloud depth profiling.
13. **Implement Persistent Storage (SQLite / PostgreSQL + PostGIS) for Telemetry & Work Orders:**
    - Replace the volatile in-memory `telemetry_store` list (`main.py:21`) with a persistent database so ingested defects, historical status changes, and audit trails survive server restarts.
14. **Build a Real Maintenance Dispatch & Work-Order Workflow:**
    - Add backend endpoints (`POST /api/v1/dispatch/work-order`) and UI actions in `IncidentDetailPanel.tsx:583-596` to generate downloadable PDF/JSON work orders and trigger email/SMS/webhook notifications for Critical defects.
15. **Add School Proximity GIS Registry & Full Multi-Criteria Configurability:**
    - Extend `services/data_generation/infrastructure_model.py` and `MCDMPrioritizationEngine` (`mcdm_engine.py:34-42`) to include school zone proximity and explicit road-class weighting as required by the paper's Layer 5 specification.
