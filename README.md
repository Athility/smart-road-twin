# Smart Road Digital Twin

> Autonomous Municipal Road Infrastructure Monitoring, AI Sensor-Switching & TOPSIS Repair Prioritization Platform

[![Next.js 14](https://img.shields.io/badge/Next.js-14.2.15-black?style=flat-svg&logo=next.js)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-svg&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-svg&logo=python)](https://www.python.org/)
[![Vercel Ready](https://img.shields.io/badge/Vercel-Deployable-000000?style=flat-svg&logo=vercel)](https://vercel.com/)
[![Tests](https://img.shields.io/badge/Tests-14%2F14%20Passing-brightgreen?style=flat-svg)](./test_core_domain.py)

---

## 📌 Executive Summary

The **Smart Road Digital Twin** is an end-to-end IoT and spatial digital twin platform designed for municipal infrastructure authorities. It ingests telemetry from mobile inspection vehicle fleets, estimates pothole/defect volumes using an environmental sensor-switching pipeline, and ranks road hazards in real time using a **TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)** Multi-Criteria Decision Making (MCDM) algorithm.

The system features a high-density, professional municipal command center dashboard built with Next.js 14, Leaflet GIS, and Tailwind CSS.

---

## 🏗️ Architecture

```text
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                  INSPECTION FLEET / IoT TELEMETRY SIMULATOR                 │
 │      (GPS, LiDAR depth, Sonar depth, Z-acceleration, Luminance, Rain)       │
 └──────────────────────────────────────┬──────────────────────────────────────┘
                                        │
                                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                       FASTAPI BACKEND SERVICE (Python)                      │
 │                                                                             │
 │   1. Environmental Sensor-Switching Pipeline                                │
 │      - Optical LiDAR (Daylight/Dry) vs. Acoustic Sonar (Rain/Night/Submerged)│
 │      - Baseline chassis distance subtraction & clamped non-negative depth   │
 │      - Dynamic impact detection threshold: z_accel_g > 1.5g                 │
 │                                                                             │
 │   2. TOPSIS MCDM Prioritization Engine                                       │
 │      - 4-Criteria Weight Vector: Volume (0.35), Traffic (0.25),             │
 │        Hospital Dist (0.25), Vehicle Speed (0.15)                          │
 │      - Normalized Decision Matrix & Ideal Best/Worst Vector Distance        │
 │                                                                             │
 │   3. Thread-Safe Telemetry Event Store & Validation Guards                  │
 └──────────────────────────────────────┬──────────────────────────────────────┘
                                        │
                                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                  NEXT.JS 14 MUNICIPAL COMMAND CENTER DASHBOARD               │
 │                                                                             │
 │   - Spatial GIS Digital Twin (Esri Street Map & Severity Heatmaps)          │
 │   - Horizontal KPI Metric Strip & Priority Filter Chips                     │
 │   - High-Density Incident Table with Column Sorting & Search                │
 │   - Interactive Anomaly Inspector Drawer (TOPSIS Score, Environment, GPS)   │
 └─────────────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

- **Spatial Digital Twin Map**: Interactive GIS map rendered with Esri World Street Map tiles, custom severity-coded hazard markers (`Critical`, `High`, `Medium`, `Low`), radar pulse animations on critical anomalies, and hazard radius heatmaps.
- **AI Environmental Sensor-Switching**: Automatically toggles between Optical LiDAR (high precision in daylight) and Acoustic Sonar (submerged voids, heavy rain, or luminance <40 lux).
- **Dynamic Impact Confirmation**: Validates physical road impact using chassis Z-axis acceleration ($>1.5g$) across both optical and acoustic modalities.
- **TOPSIS MCDM Prioritization**: Ranks repairs objectively based on defect volume, traffic load (PCU), hospital emergency route proximity, and vehicle speed.
- **Interactive Anomaly Inspector**: Click any map marker or incident row to open a full vertical detail drawer displaying TOPSIS score progress bars, environmental factors, and vehicle telemetry.
- **Resilient Offline/Fallback Mode**: Continues seamless simulation streaming if backend services are offline during demonstrations.

---

## 🛠️ Tech Stack

- **Frontend**: Next.js 14 (App Router), React 18, TypeScript, Tailwind CSS, Leaflet GIS (`react-leaflet`), Lucide Icons, Client-Side Hydration Guards.
- **Backend**: FastAPI, Pydantic v2 (Validation Guards), NumPy (Matrix Vectorization & Vector Distances), Python `unittest`.
- **Deployment**: Vercel Serverless & Static Build support.

---

## 🚀 Quick Start (Local Development)

### Prerequisites
- Node.js 18.x or 20.x
- Python 3.10+

### 1. Backend Setup
```bash
# Navigate to repository root
cd C:\smart-road-twin

# Create virtual environment (optional)
python -m venv venv
# Windows: venv\Scripts\activate | Linux/macOS: source venv/bin/activate

# Install backend dependencies
pip install fastapi uvicorn pydantic numpy

# Start FastAPI server on port 8000
python main.py
# or: uvicorn main:app --reload --port 8000
```
Backend API interactive documentation will be available at: `http://localhost:8000/docs`

### 2. Frontend Setup
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start Next.js development server
npm run dev
```
Dashboard will be live at: `http://localhost:3000`

---

## 🧪 Running Tests

Run the domain validation test suite (verifying sensor switching, TOPSIS mathematical correctness, Pydantic guards, and thread safety):

```bash
# Run from repository root
python -m unittest test_core_domain.py -v
```

Run frontend type check and production build:

```bash
cd frontend
npx tsc --noEmit
npm run build
```

---

## ☁️ Deployment Guide (Vercel)

The repository includes pre-configured `vercel.json` files for both Monorepo root and Standalone deployment modes.

### Option A: Monorepo Deployment (Root `vercel.json`)
Deploy both the Next.js frontend and FastAPI backend serverless functions together:

1. Import the repository into [Vercel](https://vercel.com/new).
2. Set Framework Preset to **Next.js**.
3. Deploy! Vercel will use the root `vercel.json` to route `/api/v1/*` to FastAPI (`main.py`) and all other routes to Next.js (`frontend`).

### Option B: Standalone Frontend Deployment
If hosting the FastAPI backend on Railway, Render, or AWS:

1. Import the `frontend/` directory in Vercel.
2. Add environment variable:
   ```env
   NEXT_PUBLIC_API_URL=https://your-backend-api.com/api/v1/telemetry/ingest
   ```
3. Deploy!

---

## 📑 API Specification

### `GET /api/v1/telemetry/ingest`
Returns telemetry event records and calculated TOPSIS priority ranks.

### `POST /api/v1/telemetry/ingest`
Ingests new IoT telemetry payload from inspection vehicles.

**Sample Request Body:**
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

---

## 📜 License

Distributed under the MIT License for Smart Infrastructure & Municipal Intelligence Applications.
