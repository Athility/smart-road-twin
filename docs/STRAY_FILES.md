# Audit of Duplicate and Stray Files (`git ls-files`)

This document lists all duplicate, stray, unimported, and redundant files identified during the audit of the tracked repository index (`git ls-files`). Per project policy, **no files were deleted**; this document records their exact status, import relationships, and maintenance recommendations.

---

## 1. Summary of Repository Index

- **Total tracked files**: 122 files (including 18 tracked compiled `.pyc` binaries).
- **Core source code files**: 63 files (`services/`, `ml/`, `frontend/src/`, `main.py`).
- **Configuration & metadata files**: 15 files (`config/`, `package.json`, `tsconfig.json`, `vercel.json`, etc.).
- **CLI / Pipeline scripts**: 11 files (`scripts/`).
- **Data & test files**: 15 files (`data/samples/`, `test_core_domain.py`, `api_test.py`, benchmark JSONs).
- **Tracked `.pyc` bytecode files**: 18 files (stray compiled artifacts).

---

## 2. Duplicate and Stray File Inventory

| File Path | Classification | Imported? | Importing / Referencing Files | Description & Recommendation |
| :--- | :--- | :---: | :--- | :--- |
| `frontend/src/components/ClientOnly.tsx` | **Stray Component** | **No** | None | Client-side mounting guard (`mounted ? children : fallback`) intended to prevent SSR hydration mismatches. Not imported anywhere in `frontend/src/app/` or `frontend/src/components/`. Safe to deprecate or keep as utility. |
| `scripts/build_srmd.py` | **Duplicate CLI** | **No** | None | 81-line CLI entry point calling `services.dataset.srmd_builder.build_srmd_dataset()`. Overlaps with `scripts/generate_srmd.py`. Recommend keeping `scripts/generate_srmd.py` as primary and consolidating. |
| `scripts/generate_srmd.py` | **Active CLI** | **Yes** (docs) | `README.md`, `scripts/validate_srmd.py` | Primary 480-line SRMD multimodal synthesis CLI with CSV, Parquet, and JSONL export capabilities. Active; keep as primary. |
| `scripts/validate_dataset.py` | **Standalone Script** | **No** | None | CLI script validating raw RDD2022 Pascal VOC XML annotations and bounding box boundaries. Not imported by runtime code; used for dataset ingestion QA. |
| `scripts/validate_srmd.py` | **Standalone Script** | **No** | None | CLI script verifying multimodal schema compliance and cross-sensor physical bounds in `srmd_records.json`. Not imported by runtime code; used for SRMD generation QA. |
| `vercel.json` | **Root Config** | **No** (manifest) | `README.md` | Monorepo deployment manifest for Vercel; configures `@vercel/next` for `frontend/` and `@vercel/python` for root `main.py` serverless functions (`/api/v1/(.*)`). Active for monorepo. |
| `frontend/vercel.json` | **Duplicate Config** | **No** (manifest) | None | Standalone Next.js deployment manifest (`buildCommand: npm run build`, `outputDirectory: .next`). Redundant when deploying from repo root via root `vercel.json`. |
| `api_test.py` | **Standalone Test** | **No** | None | Standalone integration test script verifying `/api/v1/telemetry/history` over HTTP (`http://localhost:8000`). Not imported by `test_core_domain.py` (which runs in-memory domain tests). |
| `simulation/mock_iot_vehicle.py` | **Standalone Tool** | **No** | `frontend/README.md` | Background daemon simulating a physical inspection vehicle streaming GPS, LiDAR, and accelerometer packets to `/api/v1/telemetry`. Useful for live UI demos; not imported by backend. |
| `services/**/__pycache__/*.pyc` (18 files) | **Stray Bytecode** | **No** | None (loaded by Python runtime) | 18 compiled Python `.pyc` files across Python versions 3.11, 3.13, and 3.14 are tracked in git index despite `.gitignore` containing `__pycache__/`. Should be untracked (`git rm --cached`) in a future maintenance cycle. |

---

## 3. Structural & Architectural Analysis

### 3.1 `main.py` vs `services/backend/`
- **Root `main.py`**: The single, authoritative entry point for the FastAPI backend application. Contains all REST endpoints (`/api/v1/telemetry`, `/api/v1/prioritization`, `/api/v1/preprocess/preview/{image_name}`), in-memory telemetry store, sensor fusion integration, and background tasks.
- **`services/backend/`**: An incomplete nested package structure containing only:
  - `services/backend/__init__.py`
  - `services/backend/app/__init__.py`
  - `services/backend/app/services/__init__.py`
  - `services/backend/app/services/mcdm_engine.py`
- **Finding**: There is **no** `services/backend/main.py`. The root `main.py` is the actual backend. The `services/backend/app/services/` hierarchy houses the standalone `mcdm_engine.py`, which is imported by root `main.py` (`from services.backend.app.services.mcdm_engine import MCDMPrioritizationEngine`).

### 3.2 Directory Casing: `Dashboard/` vs `dashboard/`
- **Filesystem & Git Index**: All 5 dashboard panels reside in `frontend/src/components/Dashboard/` (uppercase `D`):
  - `DatasetExplorerPanel.tsx`
  - `Header.tsx`
  - `IncidentDetailPanel.tsx`
  - `SummaryPanel.tsx`
  - `TelemetryTable.tsx`
- **Imports in `page.tsx`**: All imports use the canonical uppercase path:
  ```typescript
  import { Header } from '@/components/Dashboard/Header';
  import { SummaryPanel } from '@/components/Dashboard/SummaryPanel';
  import { TelemetryTable } from '@/components/Dashboard/TelemetryTable';
  import { IncidentDetailPanel } from '@/components/Dashboard/IncidentDetailPanel';
  import { DatasetExplorerPanel } from '@/components/Dashboard/DatasetExplorerPanel';
  ```
- **Finding**: On Windows (case-preserving, case-insensitive), lowercase `@/components/dashboard` would resolve silently, but on case-sensitive Linux / Vercel build environments it would fail. All imports correctly use uppercase `Dashboard/`, matching the git index. No casing collisions or duplicate lowercase directories exist in the git index.

### 3.3 CLI Duplication: `scripts/build_srmd.py` vs `scripts/generate_srmd.py`
- `scripts/build_srmd.py`: A thin wrapper (81 lines) around `services/dataset/srmd_builder.py:build_srmd_dataset()`.
- `scripts/generate_srmd.py`: A comprehensive standalone script (480 lines) implementing raw annotation parsing, batch synthesis, tabular flattening, and Parquet/CSV/JSONL export.
- **Finding**: Both scripts generate SRMD records, but `generate_srmd.py` is more feature-complete. `build_srmd.py` is not imported anywhere and functions as an alternate CLI entry point.

### 3.4 Vercel Deployment Configurations
- `vercel.json` (repo root): Configures the full-stack monorepo (Next.js frontend + FastAPI serverless backend).
- `frontend/vercel.json`: Configures standalone frontend-only deployment.
- **Finding**: `frontend/vercel.json` is redundant when deploying the repository as a monorepo from the root directory.

### 3.5 Tracked Bytecode Artifacts (`.pyc`)
18 compiled Python `.pyc` files remain tracked in `git ls-files`:
```
services/__pycache__/__init__.cpython-311.pyc
services/__pycache__/__init__.cpython-313.pyc
services/__pycache__/__init__.cpython-314.pyc
services/ai_engine/__pycache__/__init__.cpython-311.pyc
services/ai_engine/__pycache__/__init__.cpython-313.pyc
services/ai_engine/__pycache__/__init__.cpython-314.pyc
services/ai_engine/pipeline/__pycache__/__init__.cpython-311.pyc
services/ai_engine/pipeline/__pycache__/__init__.cpython-313.pyc
services/ai_engine/pipeline/__pycache__/__init__.cpython-314.pyc
services/ai_engine/pipeline/__pycache__/depth_estimator.cpython-311.pyc
services/ai_engine/pipeline/__pycache__/depth_estimator.cpython-313.pyc
services/ai_engine/pipeline/__pycache__/depth_estimator.cpython-314.pyc
services/backend/__pycache__/__init__.cpython-311.pyc
services/backend/__pycache__/__init__.cpython-313.pyc
services/backend/__pycache__/__init__.cpython-314.pyc
services/backend/app/__pycache__/__init__.cpython-311.pyc
services/backend/app/__pycache__/__init__.cpython-313.pyc
services/backend/app/__pycache__/__init__.cpython-314.pyc
```
- **Finding**: These files were tracked in the git repository before `.gitignore` was configured to ignore `__pycache__/`. In git, files already tracked continue to be tracked even after matching `.gitignore` rules.
