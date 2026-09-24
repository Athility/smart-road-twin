# Smart Road Digital Twin

> Multimodal Road Infrastructure Monitoring, AI Sensor-Switching & TOPSIS Repair Prioritisation Platform

[![Next.js 14](https://img.shields.io/badge/Next.js-14.2.15-black?style=flat-svg&logo=next.js)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-svg&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-svg&logo=python)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-83%2F83%20Passing-brightgreen?style=flat-svg)](./test_core_domain.py)
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
| `dist_hospital_km` | **SIMULATED** | Geodesic distance to nearest hospital in documented Mumbai demonstration corridor |
| `is_emergency_corridor` | **SIMULATED** | True when `dist_hospital_km ≤ 1.0 km` |
| `latitude` / `longitude` | **SIMULATED** | Deterministic synthetic coordinates along Mumbai Metropolitan Urban Corridor |
| `location_mode` | **SIMULATED** | GPS provenance flag: `synthetic_location` (default) |
| `topsis_score` | **DERIVED** | Multi-criteria urgency score from TOPSIS MCDM engine |
| `priority_level` | **DERIVED** | Critical / High / Medium / Low classification from TOPSIS score |

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
      │                            │     │  RDD2022Detector (YOLOv8/mock)   │
      │  /api/v1/telemetry/ingest  │     │  Damage class + bbox + conf      │
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

## 📊 TOPSIS MCDM Prioritisation

**Criteria weights (preserved from original project specification):**

| Criterion | Weight | Direction | Rationale |
|---|---|---|---|
| `calculated_volume_liters` | **0.35** | Benefit (↑) | Larger void = greater vehicle damage risk |
| `traffic_pcu` | **0.25** | Benefit (↑) | Higher traffic = more vehicles at risk |
| `dist_hospital_km` | **0.25** | **Cost (↓)** | Smaller distance = higher emergency access urgency |
| `vehicle_speed_kmh` | **0.15** | Benefit (↑) | Higher speed = greater impact severity |

> **Critical direction note:** `dist_hospital_km` is a **cost criterion**. A defect located 0.3 km from a trauma hospital receives *higher* TOPSIS urgency than an identical defect 12 km away.

**Priority thresholds:**

| TOPSIS Score | Priority Level |
|---|---|
| ≥ 0.75 | 🔴 Critical |
| ≥ 0.50 | 🟠 High |
| ≥ 0.25 | 🟡 Medium |
| < 0.25 | 🟢 Low |

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

## 🚀 Quick Start (Local Development)

### Prerequisites
- Python 3.10+
- Node.js 18.x or 20.x
- pip packages: `fastapi uvicorn pydantic numpy pillow`
- Optional: `pandas pyarrow` for Parquet export

### 1. Clone and Install Backend

```bash
git clone <repo-url>
cd smart-road-twin

# Install backend dependencies
pip install fastapi uvicorn pydantic numpy pillow

# Optional: Parquet support
pip install pandas pyarrow
```

### 2. Download RDD2022 India (optional — sample data included)

```bash
# Full dataset (~1.2 GB)
python scripts/download_rdd2022.py

# Or use the 5-image sample already in data/samples/
```

### 3. Prepare and Index Annotations

```bash
python scripts/prepare_rdd2022.py \
    --input data/raw/rdd2022_india \
    --output data/processed
```

### 4. Generate the Smart Road Multimodal Dataset

```bash
# Full dataset (all annotations)
python scripts/generate_srmd.py \
    --input data/raw/rdd2022_india \
    --output data/processed/srmd \
    --seed 2026

# Limit to 100 records for testing
python scripts/generate_srmd.py \
    --seed 2026 \
    --limit 100
```

Every run with `--seed 2026` produces **identical records** — deterministically reproducible.

### 5. Start the FastAPI Backend

```bash
python main.py
# or: uvicorn main:app --reload --port 8000
```

API docs: `http://localhost:8000/docs`

### 6. Start the Next.js Dashboard

```bash
cd frontend
npm install
npm run dev
```

Dashboard: `http://localhost:3000`

---

## 📁 Repository Structure

```text
smart-road-twin/
├── README.md
├── main.py                         # FastAPI application entrypoint
├── test_core_domain.py             # 83-test domain validation suite
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
# Full 83-test domain validation suite
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

## 🔒 Data Integrity Policy

- ✅ RDD2022 source images and annotations are **read-only** — never modified
- ✅ Simulated sensor fields are **explicitly labelled** `synthetic` in all provenance fields
- ✅ Synthetic GPS is **explicitly labelled** `synthetic_location` and references the documented Mumbai demonstration corridor
- ✅ Hospital locations are **explicitly labelled** `synthetic` pending real OSM integration
- ✅ Large generated datasets (`data/processed/`, `data/raw/`) are **git-ignored** and never committed
- ✅ All stochastic generation is **seeded** (`SEED = 2026`) and fully deterministic

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
