# Smart Road Digital Twin

> Multimodal Road Infrastructure Monitoring, AI Sensor-Switching & TOPSIS Repair Prioritisation Platform

[![Next.js 15](https://img.shields.io/badge/Next.js-15.5.27-black?style=flat-svg&logo=next.js)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-svg&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-svg&logo=python)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-105%2F105%20Passing-brightgreen?style=flat-svg)](./test_core_domain.py)
[![Seed](https://img.shields.io/badge/Seed-2026-blue?style=flat-svg)](./data/processed/srmd/metadata.json)

---

## 📌 Executive Summary

The **Smart Road Digital Twin** is a research platform and live infrastructure monitoring system that couples authentic road damage visual evidence from the **RDD2022 India dataset** with physically consistent simulated sensor telemetry. It demonstrates how multimodal sensor fusion — optical LiDAR, acoustic sonar, Z-axis accelerometry, road traffic context, and GIS hospital proximity — can improve both the *detection* and *prioritisation* of road surface defects compared to vision-only systems.

The platform produces the **Smart Road Multimodal Dataset (SRMD)**, provides a research sensor fusion experiment module, and renders results in a Next.js 14 digital twin command centre dashboard.

---

## 🗂️ Dataset Architecture & Provenance

### Source Dataset — RDD2022 India

| Field | Value |
|---|---|
| Name | Road Damage Dataset 2022 (RDD2022) |
| Country | India |
| Format | Pascal VOC XML annotations + JPEG images |
| Classes | D00 (Longitudinal Crack), D10 (Transverse Crack), D20 (Alligator Crack), D40 (Pothole) |
| DOI | [10.6084/m9.figshare.21431547](https://doi.org/10.6084/m9.figshare.21431547) |
| Citation | Sekilab / CRDDC'2022 |

**Data integrity guarantee:** RDD2022 images and annotations are never modified or re-labelled. They are used strictly read-only as ground-truth visual evidence.

---

### Derived Dataset — Smart Road Multimodal Dataset (SRMD)

| Field | Provenance Tag | Description |
|---|---|---|
| `image_id` | **OBSERVED** | Authentic RDD2022 source image filename |
| `damage_class` | **OBSERVED** | Damage class from RDD2022 VOC annotation (D00/D10/D20/D40) |
| `damage_label` | **OBSERVED** | Human label decoded from annotation class |
| `bbox` | **OBSERVED** | Bounding box pixel coordinates from VOC annotation |
| `confidence` | **DERIVED** | 1.0 for ground-truth annotations; model confidence when CV detector is used |
| `defect_severity` | **DERIVED** | Latent severity ∈ [0,1] derived from class type × normalised bounding box area |
| `surface_area_sqm` | **DERIVED** | Perspective-projected surface area derived from bbox dimensions and assumed camera height |
| `calculated_volume_liters` | **DERIVED** | Cavity volume derived from surface area × effective depth |
| `lidar_depth_cm` | **SIMULATED** | Raw LiDAR distance reading (chassis baseline + defect depth + sensor noise) |
| `sonar_depth_cm` | **SIMULATED** | Acoustic sonar reading with independent sensor error: `true_depth + N(0, σ²)` |
| `selected_depth_cm` | **SIMULATED** | Active depth measurement after environmental modality switching |
| `preferred_modality` | **SIMULATED** | Selected sensor modality: `optical_lidar` or `sonar_submerged_acoustic` |
| `z_accel_g` | **SIMULATED** | Z-axis accelerometer (g-force) derived from quarter-car shock model |
| `dynamic_impact_triggered` | **SIMULATED** | Boolean: `z_accel_g > 1.5g` physical impact confirmation |
| `traffic_pcu` | **SIMULATED** | Road-class-correlated traffic volume (PCU/day) from Greenshields model |
| `road_class` | **SIMULATED** | Road context profile: LOCAL / COLLECTOR / ARTERIAL / HIGH_TRAFFIC_CORRIDOR |
| `road_class_weight` | **DERIVED** | Ordinal road hierarchy weight (`LOCAL=1`, `COLLECTOR=2`, `ARTERIAL=3`, `HIGH_TRAFFIC_CORRIDOR=4`) |
| `dist_hospital_km` | **SIMULATED** | Geodesic distance to nearest hospital in documented Mumbai demonstration corridor |
| `is_emergency_corridor` | **SIMULATED** | True when `dist_hospital_km ≤ 1.0 km` |
| `school_proximity_km` | **SIMULATED** | Geodesic Haversine distance to nearest school across 18 Mumbai Western Corridor schools |
| `is_school_zone` | **SIMULATED** | True when `school_proximity_km ≤ 0.5 km` |
| `latitude` / `longitude` | **SIMULATED** | Deterministic synthetic coordinates along Mumbai Metropolitan Urban Corridor |
| `location_mode` | **SIMULATED** | GPS provenance flag: `synthetic_location` (default) |
| `topsis_score` | **DERIVED** | Multi-criteria urgency score from Fuzzy BWM TOPSIS/VIKOR MCDM engine |
| `priority_level` | **DERIVED** | Critical / High / Medium / Low classification from calibrated MCDM thresholds |

> **Important:** Simulated fields are generated using physics-consistent, reproducible models seeded with `SEED = 2026`. They do not represent real sensor readings from the RDD2022 source vehicles. This distinction is explicitly encoded in every record's `provenance` block and every `*_source` field.

---

### Field Provenance Key

| Tag | Meaning |
|---|---|
| `OBSERVED` | Field originates directly from the RDD2022 source dataset, unchanged |
| `DERIVED` | Field computed deterministically from one or more OBSERVED fields |
| `SIMULATED` | Field produced by a physics-consistent stochastic model seeded with `SEED = 2026` |
| `EXTERNAL` | Field from an external GIS or database service (not yet integrated; reserved for future OSM hospital data) |

---

### Every SRMD Record Carries Explicit Provenance

```json
{
  "provenance": {
    "visual_source": "RDD2022",
    "sensor_source": "synthetic",
    "location_source": "synthetic",
    "synthesis_seed": 2026,
    "split": "train"
  },
  "location": {
    "location_mode": "synthetic_location",
    "location_source": "synthetic",
    "demonstration_area": "Mumbai Metropolitan Urban Corridor"
  },
  "infrastructure": {
    "hospital_source": "synthetic"
  },
  "traffic": {
    "traffic_source": "synthetic"
  }
}
```

---

## 🏗️ System Architecture

```text
┌──────────────────────────────────────────────────────────────────────────┐
│                    RDD2022 India Source Dataset                           │
│            Pascal VOC XML Annotations + JPEG Images                      │
│                  ← OBSERVED: damage class, bbox →                        │
└────────────────────────────┬─────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────────┐
│               scripts/generate_srmd.py  (SEED = 2026)                    │
│                                                                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │
│  │  LiDAR Model    │  │  Sonar Model     │  │  Accelerometer Model    │  │
│  │  SIMULATED      │  │  SIMULATED       │  │  SIMULATED              │  │
│  │  Quarter-car    │  │  Acoustic echo   │  │  Quarter-car z-shock    │  │
│  │  chassis depth  │  │  + sensor noise  │  │  > 1.5g = impact        │  │
│  └────────┬────────┘  └────────┬─────────┘ └────────────┬────────────┘  │
│           │                   │                         │               │
│  ┌────────▼────────┐  ┌───────▼──────────┐  ┌──────────▼────────────┐  │
│  │  Traffic Model  │  │ Hospital/GIS Model│  │  Location Model        │  │
│  │  SIMULATED      │  │  SIMULATED        │  │  SIMULATED             │  │
│  │  Road-class PCU │  │  Geodesic nearest │  │  Mumbai corridor GPS   │  │
│  └────────┬────────┘  └────────┬──────────┘ └──────────┬────────────┘  │
│           └────────────────────┴────────────────────────┘               │
│                                      │                                   │
│              Physical Consistency Model (latent severity → all sensors)  │
└──────────────────────────────────────┬───────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                  SRMD: Smart Road Multimodal Dataset                      │
│  data/processed/srmd/                                                    │
│  ├── metadata.json          (Phase 21 provenance & version manifest)     │
│  ├── srmd_records.json      (all records, hierarchical JSON)             │
│  ├── srmd_dataset.jsonl     (line-delimited JSON for ML training)        │
│  ├── srmd_dataset.csv       (flat tabular CSV)                           │
│  ├── srmd_dataset.parquet   (columnar Parquet for analytics)             │
│  ├── srmd_manifest.json     (generation log & statistics)                │
│  └── splits/                (Phase 22: zero-leakage data splits)         │
│      ├── split_manifest.json                                             │
│      ├── train_records.json     (70% of source images)                   │
│      ├── validation_records.json (20% of source images)                  │
│      └── test_records.json      (10% of source images)                   │
└──────────────────────────────────────┬───────────────────────────────────┘
                                       │
                       ┌───────────────┴──────────────────┐
                       ▼                                  ▼
      ┌────────────────────────────┐     ┌─────────────────────────────────┐
      │   FastAPI Backend (main.py)│     │  CV Detector (services/cv/)      │
      │                            │     │  RDD2022Detector                 │
      │  /api/v1/telemetry/ingest  │     │  (default: ground-truth lookup) │
      │  /api/v1/dataset/explorer  │     └──────────────┬──────────────────┘
      │  /api/v1/images/{name}     │                    │
      │                            │     ┌──────────────▼──────────────────┐
      │  TOPSIS MCDM Engine        │     │  Multimodal Fusion Engine        │
      │  Sensor Switching          │◄────┤  services/fusion/fusion_engine.py│
      │  Environmental Context     │     │  CV + LiDAR + Sonar + Accel     │
      └──────────────┬─────────────┘     │  + Traffic + Hospital + GPS     │
                     │                   └─────────────────────────────────┘
                     ▼
      ┌──────────────────────────────────────────────────────────────────┐
      │         Next.js 14 Digital Twin Dashboard (frontend/)            │
      │                                                                  │
      │  ┌─────────────────────────┐   ┌───────────────────────────────┐ │
      │  │  Digital Twin Map View  │   │  Research Dataset Explorer    │ │
      │  │  Leaflet GIS markers    │   │  SRMD statistics & records    │ │
      │  │  Priority heatmaps      │   │  Sensor distribution metrics  │ │
      │  │  Incident detail panel  │   │  D00/D10/D20/D40 breakdown    │ │
      │  │  DATA SOURCE provenance │   │  Inspectable record table     │ │
      │  └─────────────────────────┘   └───────────────────────────────┘ │
      └──────────────────────────────────────────────────────────────────┘
```

> **Visual Detection Mode Note:** Visual road damage detection currently uses **ground-truth lookup** (`detector: "ground-truth lookup"`) against the authentic RDD2022 Pascal VOC annotations by default when `ultralytics` YOLOv8 weights are not loaded (falling back to `"opencv-contour"` for unindexed images). Furthermore, when `DamageModelTrainer` runs in simulated mode (without `ultralytics` installed), it writes JSON checkpoint metadata to `best.pt.simulated.json` and `last.pt.simulated.json` rather than `.pt` files so simulated artifacts are never mistaken for PyTorch weights. Every backend API response (`/api/v1/telemetry`, `/api/v1/telemetry/ingest`, `/api/v1/dataset/explorer`, `/health`) explicitly includes `"detector": "ground-truth lookup" | "opencv-contour" | "yolov8"`.

---

## 📐 Physical Consistency Model

All simulated sensor fields are **not independent**. They are derived from a shared latent variable `defect_severity ∈ [0,1]`:

```text
RDD2022 Visual Annotation
       ↓
  defect_severity (latent)
       ↓
  ┌────┼────────┬────────────┬──────────────┐
  ↓   ↓        ↓            ↓              ↓
depth  area  z_accel_g   volume   traffic/context
  ↓     ↓       ↓            ↓
LiDAR  Sonar   Impact     TOPSIS Priority
```

**Environmental sensor switching:**

| Condition | Active Sensor | Rationale |
|---|---|---|
| `rain_detected = False` AND `luminance ≥ 40 lux` | **Optical LiDAR** | Daylight, dry surface — LiDAR performs optimally |
| `rain_detected = True` OR `luminance < 40 lux` | **Acoustic Sonar** | Rain / night — LiDAR reflected by water film; sonar reads submerged void depth |

**Accelerometer impact confirmation:**

| z_accel_g | Regime | Interpretation |
|---|---|---|
| ≤ 1.35g | `no_impact` | Suspension absorbs defect smoothly |
| 1.35g < z ≤ 1.50g | `borderline` | Elevated jolt — requires cross-modal validation |
| > 1.50g | `confirmed_impact` | Physical road hazard confirmed by dynamic shock |

---

## 📊 Fuzzy BWM + TOPSIS / VIKOR MCDM Prioritisation

See [`docs/MCDM.md`](./docs/MCDM.md) for the complete mathematical specification (Triangular Fuzzy Numbers, GMIR defuzzification, Guo & Zhao minimax optimization, VIKOR $S_i, R_i, Q_i$ compromise solution checks, and Spearman rank concordance).

**Extended 6-Criteria Model (with Fuzzy BWM weight profiles):**

| Criterion | `safety-first` | `traffic-first` | `equity-first` | Legacy 4-Crit | Direction | Rationale |
|---|---:|---:|---:|---:|---|---|
| `calculated_volume_liters` | **0.2798** | 0.2263 | 0.1587 | 0.35 | Benefit (↑) | Larger void = greater vehicle damage & impact risk |
| `traffic_pcu` | **0.1346** | 0.2546 | 0.1021 | 0.25 | Benefit (↑) | Higher daily traffic (PCU/day) = more vehicles at risk |
| `dist_hospital_km` | **0.2418** | 0.0876 | 0.2950 | 0.25 | **Cost (↓)** | Smaller distance to trauma hospital = higher emergency access urgency |
| `speed_kmh` | **0.1346** | 0.1362 | 0.0799 | 0.15 | Benefit (↑) | Higher speed = greater dynamic impact severity |
| `school_proximity_km` | **0.1346** | 0.0690 | 0.2622 | — | **Cost (↓)** | Smaller Haversine distance across 18 Mumbai Western Corridor schools = higher pedestrian safety urgency |
| `road_class` | **0.0746** | 0.2263 | 0.1021 | — | Benefit (↑) | Ordinal hierarchy weight (`LOCAL=1`, `COLLECTOR=2`, `ARTERIAL=3`, `HIGH_TRAFFIC_CORRIDOR=4`) |

> **Critical direction note:** Both `dist_hospital_km` and `school_proximity_km` are **cost criteria**. Defects located near trauma hospitals or K-12 schools receive *higher* TOPSIS/VIKOR urgency than identical defects further away.

**Priority thresholds (calibrated in [`config/sensor_profiles.yaml`](./config/sensor_profiles.yaml)):**

| Calibrated Score (`use_calibrated_thresholds=True`) | Legacy Quartile Score | Priority Level |
|---|---|---|
| ≥ 0.57 | ≥ 0.75 | 🔴 Critical |
| ≥ 0.45 | ≥ 0.50 | 🟠 High |
| ≥ 0.35 | ≥ 0.25 | 🟡 Medium |
| < 0.35 | < 0.25 | 🟢 Low |

---

## 🔬 Sensor Fusion Experiments (Phase 17)

The platform includes a research experiment module comparing five fusion configurations:

| Experiment | Sensors Used |
|---|---|
| A | Vision only |
| B | Vision + Accelerometer |
| C | Vision + LiDAR |
| D | Vision + LiDAR + Sonar + Accelerometer |
| E | Full Multimodal (all sensors + GIS context) |

Run experiments:
```bash
python scripts/run_fusion_experiments.py
```

Results are saved to `runs/experiments/fusion_benchmark.json`.

---

## 🚀 Quick Start (Windows & Cross-Platform)

### Prerequisites
- Python 3.10+
- Node.js 18.x or 20.x
- pip packages: `fastapi uvicorn pydantic numpy pillow requests pyyaml`
- Optional: `pandas pyarrow` for Parquet export

---

### Exact Windows Quick-Start

Run these commands in separate PowerShell terminals from the repository root (`c:\smart-road-twin`):

#### Terminal 1: FastAPI Backend (Port 8000)
```powershell
# Launch FastAPI backend with uvicorn on 127.0.0.1:8000
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```
- Interactive Swagger API Documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health Check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

#### Terminal 2: Next.js Digital Twin Dashboard (Port 3000)
```powershell
cd frontend

# Optional: configure backend API endpoint (defaults to http://127.0.0.1:8000)
$env:NEXT_PUBLIC_API_URL = "http://127.0.0.1:8000"

# Install frontend dependencies (first run only)
npm install

# Start Next.js development server
npm run dev
```
- Digital Twin Command Centre: [http://localhost:3000](http://localhost:3000)
- **Offline Fallback Sync**: The dashboard is fully functional even if the backend is stopped, automatically falling back to verified local SRMD data in `frontend/src/data/fallback_records.json`.

#### Optional Terminal 3: Mock IoT Inspection Vehicle Simulator
```powershell
# Stream simulated multi-sensor telemetry packets to the backend ingestion endpoint
python simulation/mock_iot_vehicle.py --loop --interval 2.0
```
- Streams continuous telemetry (GPS waypoints along the Mumbai Western Express Corridor, LiDAR depth, acoustic sonar, chassis Z-acceleration impulses, ambient lux, and rain detection) to `POST /api/v1/telemetry/ingest`.

---

### Dataset Synthesis (Deterministic `SEED = 2026`)

The repository includes a 5-image demo sample in `data/samples/`. To regenerate or expand the Smart Road Multimodal Dataset (SRMD):

```powershell
# Regenerate canonical 6-record demonstration dataset (seed 2026)
python scripts/generate_srmd.py --seed 2026

# Export sensitivity benchmark grid across weights, VIKOR v, and crack scales
python scripts/run_sensitivity.py

# Re-synchronize frontend offline fallback records
python scripts/sync_frontend_fallback.py
```

---

## 🎮 5-Step Demo Walkthrough

Follow these 5 steps to verify all digital twin capabilities:

1. **Launch Command Centre & Basemap**:
   Open [http://localhost:3000](http://localhost:3000). The dashboard loads an Esri World Dark Gray basemap centered on the Mumbai Western Express Highway corridor, displaying color-coded defect markers with hazard radii, priority clustering, and the Top-3 priority queue summary in the right drawer.
2. **Inspect Priority Queue & Weight Profiles**:
   In the top-right toolbar, cycle through the three Fuzzy BWM weight profiles: **Safety-First**, **Traffic-First**, and **Equity-First**. Observe live multi-criteria re-ranking via `/api/v1/prioritization`, side-by-side TOPSIS vs. VIKOR ranks, and yellow "methods disagree" badges whenever $|\text{Rank}_{\text{TOPSIS}} - \text{Rank}_{\text{VIKOR}}| \ge 2$.
3. **Inspect Incident Drawer & Pipeline Flow (Layers 1–5)**:
   Click any table row or map marker (e.g., `SRMD-IND-000045-00` or `SRMD-IND-000278-00`) to open the incident drawer. Switch to the **Pipeline** tab to inspect the 5-layer audit trail:
   - **Layer 1 (Sensor Selection)**: Active sensor modality (`Optical LiDAR` vs `Sonar Submerged Acoustic`) with weather/lux switching rationale.
   - **Layer 2 (Preprocessing)**: Interactive CLAHE split-slider preview comparing raw vs contrast-enhanced frames with local contrast/luminance metrics.
   - **Layer 3 (Detection)**: Honest detector attribution badge (`Synthetic sample images, RDD2022 annotation schema` and `N/A (annotation, not a model prediction)`).
   - **Layer 4 (Geometry & Volume)**: Ground-plane pinhole footprint and exact paraboloid cavity volume formula ($V = 0.5 \times \text{Area} \times \text{Depth} \times 1000$) reproducing displayed liters.
   - **Layer 5 (MCDM Decision)**: TOPSIS closeness $C_i$ vs. VIKOR $Q_i$, evaluation of conditions C1 (Acceptable Advantage) and C2 (Acceptable Stability), and the compromise solution set.
4. **Interactive Environmental Scenario Simulation**:
   In the incident drawer header or top bar, toggle the **Simulation** scenarios: **Dry (LiDAR)**, **Rain (Sonar)**, or **Night (Sonar)**. Watch the active sensor switch dynamically, recalculating the effective depth, volume, and urgency score in real time.
5. **Explore Sensitivity Analysis & Large-Cohort Benchmarks**:
   Click the **Sensitivity** tab in the main view. Review the interactive 6-record $\times$ 9-setting rank heatmap and Recharts Rank-1 frequency distribution. Scroll down to inspect the large-cohort $N=300$ synthetic validation suite, displaying mean Spearman $\rho$, Kendall $\tau$, Top-5 priority overlap, and 1000-resample bootstrap 95% confidence intervals across crack multiplier scales $\{5, 15, 30, 60\}$.

---

## 📁 Repository Structure

```text
smart-road-twin/
├── README.md
├── main.py                         # FastAPI application entrypoint
├── test_core_domain.py             # 105-test domain validation suite
├── vercel.json                     # Vercel deployment configuration
│
├── config/
│   └── sensor_profiles.yaml        # Configurable sensor/road-class parameters
│
├── data/
│   ├── samples/                    # 5-image RDD2022 demo subset (tracked in git)
│   │   ├── images/                 # Sample JPEG images
│   │   └── annotations/xmls/       # Sample Pascal VOC XMLs
│   ├── raw/                        # Full RDD2022 download (git-ignored)
│   └── processed/                  # Generated SRMD outputs (git-ignored)
│
├── scripts/
│   ├── download_rdd2022.py         # RDD2022 dataset downloader
│   ├── prepare_rdd2022.py          # Annotation parser & index builder
│   ├── generate_srmd.py            # SRMD multimodal dataset generator
│   ├── run_data_fusion.py          # Batch CV + sensor fusion runner
│   ├── run_fusion_experiments.py   # Phase 17 experiment runner
│   ├── train_damage_model.py       # CV model training pipeline
│   └── evaluate_damage_model.py    # CV model evaluation pipeline
│
├── services/
│   ├── ai_engine/pipeline/
│   │   └── depth_estimator.py      # Environmental sensor-switching engine
│   ├── backend/app/services/
│   │   └── mcdm_engine.py          # TOPSIS MCDM prioritisation engine
│   ├── cv/
│   │   ├── detector.py             # Abstract RoadDamageDetector interface
│   │   ├── rdd2022_detector.py     # RDD2022 ground-truth & model detector
│   │   ├── schemas.py              # Detection result schemas
│   │   └── preprocessing.py        # Image preprocessing utilities
│   ├── data_generation/
│   │   ├── lidar_model.py          # LiDAR depth simulation
│   │   ├── sonar_model.py          # Acoustic sonar simulation
│   │   ├── accelerometer_model.py  # Quarter-car Z-axis shock model
│   │   ├── traffic_model.py        # Greenshields traffic PCU model
│   │   ├── infrastructure_model.py # Hospital GIS proximity model
│   │   ├── location_model.py       # GPS route synthesis (3 modes)
│   │   ├── consistency_model.py    # Physical consistency coordinator
│   │   └── sensor_augmentation.py  # Master augmentation orchestrator
│   ├── dataset/
│   │   ├── srmd_builder.py         # SRMD record synthesis engine
│   │   └── split_manager.py        # Zero-leakage research data splitter
│   ├── experiments/
│   │   └── experiment_runner.py    # Fusion experiment A–E runner
│   └── fusion/
│       ├── fusion_engine.py        # End-to-end multimodal fusion pipeline
│       ├── disagreement_analyzer.py # Sensor disagreement analysis
│       └── schemas.py              # Fused RoadDefectEvent schemas
│
├── ml/
│   ├── train/                      # Training pipeline modules
│   ├── evaluate/                   # Evaluation pipeline modules
│   └── configs/                    # Model training configurations
│
└── frontend/                       # Next.js 14 dashboard application
    ├── app/
    │   └── page.tsx                # Main dashboard page (Map + Explorer)
    └── components/
        ├── Header.tsx              # View switcher & navigation
        ├── IncidentDetailPanel.tsx # 11-step visual→decision inspector
        └── DatasetExplorerPanel.tsx # SRMD research metrics explorer
```

---

## 🔁 Reproducibility

**Seed:** `2026` — used consistently across all generators, splitters, and stochastic models.

Every generated dataset includes `metadata.json`:

```json
{
  "dataset_name": "Smart Road Multimodal Dataset",
  "base_dataset": "RDD2022",
  "base_dataset_country": "India",
  "generator_version": "1.0.0",
  "seed": 2026,
  "sensor_fields": [
    "lidar_depth_cm",
    "sonar_depth_cm",
    "z_accel_g",
    "traffic_pcu",
    "dist_hospital_km"
  ],
  "dataset_version": "1.0.0",
  "preprocessing_version": "1.0.0",
  "sensor_generation_version": "1.0.0",
  "configuration_version": "1.0.0",
  "random_seed": 2026,
  "model_version": "1.0.0"
}
```

Verify reproducibility manually:

```bash
python scripts/generate_srmd.py --seed 2026 --limit 100 --output /tmp/run1
python scripts/generate_srmd.py --seed 2026 --limit 100 --output /tmp/run2
python -c "
import json
r1 = json.load(open('/tmp/run1/srmd_records.json'))
r2 = json.load(open('/tmp/run2/srmd_records.json'))
assert [rec['lidar']['depth_cm'] for rec in r1] == [rec['lidar']['depth_cm'] for rec in r2]
print('✅ Deterministic reproducibility confirmed')
"
```

---

## 📦 Research Data Splits (Phase 22)

Splitting is performed **strictly at the source image level before record augmentation**.

An image containing multiple defect annotations (e.g. `India_000278` has both D40 and D20) has **all its records assigned to the same split**. This prevents visual context and physical feature leakage across training and evaluation sets.

**Invariants enforced at generation time:**

```text
train_image_ids ∩ test_image_ids  = ∅
train_image_ids ∩ val_image_ids   = ∅
val_image_ids   ∩ test_image_ids  = ∅
```

Default ratios: **Train 70% / Validation 20% / Test 10%**

Split files: `data/processed/srmd/splits/`

---

## 🧪 Testing (Phase 23)

```bash
# Full 105-test domain validation suite
python -m unittest test_core_domain.py -v

# Frontend type check and production build
cd frontend
npx tsc --noEmit
npm run build
```

**Test coverage includes:**

| Domain | Tests |
|---|---|
| Environmental sensor switching | LiDAR/sonar switching under rain, luminance, dry daylight |
| TOPSIS MCDM engine | Weight vector, benefit/cost directions, hospital distance urgency inversion |
| Physical consistency | Depth–area–volume monotonicity, severity correlation |
| LiDAR model | Optical degradation, depth tiers, area correlation |
| Sonar model | Acoustic error formula, submerged echo, environmental switching |
| Accelerometer model | 1.5g threshold, three-regime classification, quarter-car shock |
| Traffic model | Greenshields flow, 4 road-class PCU ranges |
| Infrastructure model | Haversine distance, hospital nearest-neighbour, emergency corridor |
| Location model | 3 GPS modes, Mumbai corridor bounds, seed 2026 determinism |
| Sensor fusion | End-to-end multimodal fusion pipeline |
| Sensor disagreement | Concordant/discordant visual-physical assessment |
| Fusion experiments | A–E experiment configurations, Spearman rank correlation |
| Dataset generation | SRMD schema conformance, multi-format export, split leakage invariants |
| Reproducibility | Identical records from identical seeds, metadata.json schema |
| Dashboard API | Telemetry provenance, dataset explorer metrics, image serving |
| Phase 23 comprehensive | 12-domain verification: RDD parsing → GPS → hospital → fusion → reproducibility |

---

## 📑 API Reference

### `GET /api/v1/telemetry/ingest`
Returns all active incident records with TOPSIS ranks and full multimodal provenance.

### `POST /api/v1/telemetry/ingest`
Ingests new inspection vehicle telemetry.

```json
{
  "vehicle_id": "INSPECTION_FLEET_01",
  "latitude": 19.0760,
  "longitude": 72.8777,
  "rain_detected": true,
  "mean_luminance": 25.0,
  "lidar_depth_cm": 45.8,
  "sonar_depth_cm": 45.8,
  "z_accel_g": 2.4,
  "surface_area_sqm": 0.65,
  "traffic_pcu": 24500,
  "dist_hospital_km": 0.4,
  "speed_kmh": 42.0
}
```

### `GET /api/v1/dataset/explorer`
Returns SRMD research statistics: class counts, average sensor values, modality percentages, priority distribution.

### `GET /api/v1/images/{image_name}`
Serves RDD2022 source images from `data/samples/images/`. Returns SVG placeholder for unknown images.

### `GET /api/v1/preprocess/preview/{image_name}`
Returns both the original RGB image and the CIE LAB L-channel CLAHE-enhanced image (`clip_limit=2.0`, `tile_grid=(8,8)`) as base64 data URIs along with luminance/contrast metrics for a before/after slider.

### `GET /api/v1/prioritization?method=topsis|vikor|both&profile=safety-first`
Computes multi-criteria rankings across active road defects using TOPSIS, VIKOR ($S_i, R_i, Q_i$ with $v=0.5$ and acceptable advantage/stability checks), or both (with Spearman rank correlation), using any of the 3 named Fuzzy BWM profiles (`safety-first`, `traffic-first`, `equity-first`).

### `POST /api/v1/prioritization`
Computes rankings using custom crisp weights (`weights: [w1, ..., w6]`) or custom Fuzzy BWM comparison vectors (`best_to_others`, `others_to_worst`).

---

## ☁️ Deployment (Vercel)

```bash
# Verify frontend build is clean
cd frontend && npm run build

# Deploy via Vercel CLI
vercel --prod
```

The root `vercel.json` routes `/api/v1/*` to FastAPI (`main.py`) and all other paths to Next.js (`frontend/`).

---

## ⚠️ Known Limitations & Real vs. Simulated Disclosures

For a rigorous, file-referenced accounting of what is genuinely implemented versus simulated or omitted, refer to [`docs/LIMITATIONS.md`](./docs/LIMITATIONS.md) and [`docs/PAPER_NUMBERS.md`](./docs/PAPER_NUMBERS.md). Key disclosures:

1. **Synthetic Demo Visual Data**: The 5 demo sample images in `data/samples/images/` are PIL-generated synthetic road defect textures conforming to RDD2022 image dimensions ($720 \times 720$). Annotation bounding boxes in `India_000045.xml` and `India_000278.xml` were adjusted to produce realistic physical footprints ($0.05\text{--}0.40\,\text{m}^2$) rather than arbitrary pixel areas.
2. **Detection Mode Attribution**: Visual detection defaults to `GroundTruthLookupAdapter` (`detector: "ground-truth lookup"`), reading VOC annotations with honest attribution (`"N/A (annotation, not a model prediction)"`). Real YOLOv8 inference requires installing `ultralytics` and supplying trained `.pt` model weights.
3. **Simulated Telemetry**: LiDAR ToF depth, acoustic sonar echoes, chassis Z-accelerometer impulses, ambient lux, and rain flags are physically consistent stochastic simulations seeded with `SEED = 2026`. GPS coordinates represent deterministic waypoints along the Mumbai Western Express Corridor.
4. **Paraboloid Volume Model**: Pothole cavity volume ($V = 0.5 \times \text{Area} \times \text{Depth} \times 1000$) models defects as smooth paraboloids of revolution (configured in `config/sensor_profiles.yaml`). Real-world potholes exhibit irregular, fractal geometries.
5. **N=6 Small Sample Behavior**: For the 6-record demonstration corridor, batch quantile tiering produces exactly one Critical defect, the absolute alarm rule ($C_i \ge 0.60$) does not trigger, and the top-2 potholes differ by only $0.0035$ TOPSIS score units (functionally tied). See [`docs/MCDM.md`](./docs/MCDM.md).
6. **In-Memory Storage & Dispatch**: Telemetry is ingested into an in-memory queue (`telemetry_store`, capped at 500 records) without persistent database storage. Maintenance recommendations are informational; automated municipal work-order dispatch is not connected.

---

## 🔒 Data Integrity Policy

- ✅ RDD2022 source images and annotations are **read-only** — never modified
- ✅ Simulated sensor fields are **explicitly labelled** `synthetic` in all provenance fields
- ✅ Synthetic GPS is **explicitly labelled** `synthetic_location` and references the documented Mumbai demonstration corridor
- ✅ Hospital locations are **explicitly labelled** `synthetic` pending real OSM integration
- ✅ Large generated datasets (`data/processed/`, `data/raw/`) are **git-ignored** and never committed
- ✅ All stochastic generation is **seeded** (`SEED = 2026`) and fully deterministic

---

## 🔐 Known Security Notes

Upgrading `next` and `eslint-config-next` to **`15.5.27`** resolved the **Critical** `next` runtime advisories (RCE, SSRF, middleware bypass, and cache-poisoning advisories in `14.x`) as well as the **High** `glob` (`10.2.0 - 10.4.5`, `GHSA-5j98-mcp5-4vw2`) CLI command-injection advisory in `@next/eslint-plugin-next`.

The remaining `npm audit` findings in `frontend/` are build-time / CSS-toolchain transitive dependencies that require major breaking upgrades (`next@16.4.0` / React 19 or `tailwindcss@4.3.3`) and are not exposed to untrusted user input at runtime:

| Package (Transitive Path) | Severity | Advisory | Fix Status |
|---|---|---|---|
| `postcss` (`<=8.5.22` bundled inside `node_modules/next/node_modules/postcss`) | **High** | `GHSA-qx2v-qp2m-jg93`, `GHSA-6g55-p6wh-862q`, `GHSA-fxqj-rqcc-2cmp`, `GHSA-r28c-9q8g-f849` (CSS stringify / sourceMap path traversal) | Requires `next@>=16.3.0` (`npm audit fix --force`, breaking React 19 upgrade incompatible with `react-leaflet@4.2.1`) |
| `braces` (`*` via `chokidar` / `micromatch` / `fast-glob` in `tailwindcss@3.4.x` & `@next/eslint-plugin-next`) | **High** | `GHSA-vfj7-8cjw-p6xm` (stack-exhaustion DoS on deeply nested glob patterns) | Requires `tailwindcss@4.x` (`npm audit fix --force`, breaking CSS config migration) |
| `postcss-selector-parser` (`<7.1.6` via `postcss-nested` in `tailwindcss@3.4.x`) | **Moderate** | `GHSA-rj75-hqrm-r3gf` (quadratic complexity in flat selector parsing) | Requires `tailwindcss@4.x` (`npm audit fix --force`, breaking CSS config migration) |

---

## 📜 Citation

If using the Smart Road Multimodal Dataset in research, please cite both:

**RDD2022 (source visual data):**
```
Arya, D., Maeda, H., Ghosh, S. K., Toshniwal, D., & Sekimoto, Y. (2022).
RDD2022: A multi-national image dataset for automatic Road Damage Detection.
figshare. https://doi.org/10.6084/m9.figshare.21431547
```

**Smart Road Digital Twin (this repository):**
```
Smart Road Multimodal Dataset (SRMD) v1.0.0
Derived from RDD2022 India with physics-consistent sensor augmentation.
Seed: 2026. Generator version: 1.0.0.
```

---

## 📜 License

Distributed under the MIT License for Smart Infrastructure & Municipal Intelligence Applications.
The RDD2022 source dataset is subject to its own licence terms (see DOI link above).
