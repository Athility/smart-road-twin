# Smart Road Digital Twin - Frontend Dashboard

A Next.js 14 real-time telemetry dashboard using React Leaflet for spatial road surface defect monitoring, depth estimation, and TOPSIS priority hazard classification.

## Features

- **Interactive Leaflet Map**: Displays detected potholes with color-coded markers matching TOPSIS priority:
  - 🔴 **Critical** (`#EF4444`, TOPSIS &ge; 0.75) with animated radar pulse
  - 🟠 **High** (`#F97316`, TOPSIS &ge; 0.50)
  - 🟡 **Medium** (`#EAB308`, TOPSIS &ge; 0.25)
  - 🟢 **Low** (`#22C55E`, TOPSIS &lt; 0.25)
- **Live Telemetry Ingestion**: Seamlessly fetches from `http://localhost:8000/api/v1/telemetry/ingest` with configurable auto-sync (every 3 seconds).
- **Comprehensive Summary Panel**:
  - **Defect Volumetrics**: Total displaced volume (Liters), average volume per pothole, submerged defect count.
  - **Weather & Environmental State**: Rain detection status, ambient luminance (Lux), active sensor modality (Acoustic Sonar vs Optical LiDAR).
  - **TOPSIS Priority Tiers**: Proportional distribution bar and interactive filtering by priority.
- **Incident Telemetry Stream**: Interactive table with sorting, search, and click-to-focus on map.

## Quick Start

### 1. Start the FastAPI Backend
From the root repository directory:
```bash
python -m uvicorn main:app --reload --port 8000
```

### 2. Start the Next.js Frontend
From the `frontend` folder:
```bash
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser.

### 3. Run Simulated IoT Inspection Vehicle (Optional)
In another terminal:
```bash
python simulation/mock_iot_vehicle.py
```
As the simulated vehicle travels along the route, new potholes will be ingested, TOPSIS scores recalculated, and the dashboard will automatically reflect the live telemetry.
