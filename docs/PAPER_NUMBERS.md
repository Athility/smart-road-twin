# Authoritative Experimental Metrics & Benchmark Numbers

> **Source Disclosure**: All metrics in this document are automatically computed and exported
> by `scripts/export_paper_numbers.py` from verified experiment JSON outputs (`runs/experiments/sensitivity.json`)
> and the canonical `test_core_domain.py` test suite. No values are hand-typed.

---

## 1. 6-Record Demonstration Corridor Ranking

**Baseline Parameters**: `profile = safety-first`, VIKOR consensus `v = 0.50`, crack severity scale `= 15.0`, volume model `= 0.5 * Area * Depth * 1000` (paraboloid cavity).

| TOPSIS Rank | VIKOR Rank | Incident ID | Defect Class | Severity / Volume | Depth (cm) | TOPSIS Score ($C_i$) | VIKOR $Q_i$ | Priority Tier |
| :---: | :---: | :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **#1** | **#1** | `SRMD-IND-000278-00` | D40 (Pothole) | 8.4 L | 8.1 cm | **0.5782** | **0.0000** | **Critical** |
| **#2** | **#4** | `SRMD-IND-000045-00` | D40 (Pothole) | 22.0 L | 11.4 cm | **0.5747** | **0.4889** | **High** |
| **#3** | **#2** | `SRMD-IND-000102-00` | D20 (Alligator Crack) | 7.12 (mag) | 3.6 cm | **0.4283** | **0.3693** | **Medium** |
| **#4** | **#5** | `SRMD-IND-000210-00` | D10 (Transverse Crack) | 6.75 (mag) | 0.9 cm | **0.4112** | **0.6147** | **Medium** |
| **#5** | **#3** | `SRMD-IND-000278-01` | D20 (Alligator Crack) | 8.37 (mag) | 3.8 cm | **0.3975** | **0.4371** | **Low** |
| **#6** | **#6** | `SRMD-IND-000155-00` | D00 (Longitudinal Crack) | 1.24 (mag) | 0.4 cm | **0.3789** | **0.9317** | **Low** |

- **Top-2 TOPSIS Score Gap**: `| 0.5782 - 0.5747 | = 0.0035`.
  - *Interpretation*: The top-2 highway potholes (`SRMD-IND-000278-00` and `SRMD-IND-000045-00`) are within 0.0035 TOPSIS score units of each other (well within $\Delta < 0.01$) and should be treated as functionally tied in dispatch urgency.

---

## 2. VIKOR Compromise Decision Analysis (Conditions C1 & C2)

- **Rank 1 Alternative ($A^1$)**: `SRMD-IND-000278-00` ($Q = 0.0000$)
- **Rank 2 Alternative ($A^2$)**: `SRMD-IND-000102-00` ($Q = 0.3693$)
- **Threshold Advantage**: $DQ = \frac{1}{n - 1} = \frac{1}{6 - 1} = 0.2000$
- **Condition C1 (Acceptable Advantage)**: $Q(A^2) - Q(A^1) = 0.3693 - 0.0000 = 0.3693 \ge 0.2000$ $\implies$ **PASSED**
- **Condition C2 (Acceptable Stability)**: $A^1$ is best in group utility $S$ ($S = 0.2221$) and best in individual regret $R$ ($R = 0.1834$) $\implies$ **PASSED**
- **Compromise Solution Set**: `['SRMD-IND-000278-00']` (Single unique optimum).

---

## 3. Method Concordance: Spearman Rank Correlation ($\rho$)

Pairwise Spearman rank correlation between TOPSIS and VIKOR across 3 Fuzzy BWM weight profiles $\times$ 3 VIKOR consensus weights ($v$):

| Weight Profile | Consensus $v = 0.25$ | Consensus $v = 0.50$ | Consensus $v = 0.75$ | Profile Concordance Characterization |
| :--- | :---: | :---: | :---: | :--- |
| **`safety-first`** | `0.6000` | `0.7143` | `0.9429` | Moderate to strong concordance ($\rho = 0.60$ to $0.94$, regret-weighted) |
| **`traffic-first`** | `1.0000` | `1.0000` | `1.0000` | Perfect concordance ($\rho = 1.0000$ invariant to $v$) |
| **`equity-first`** | `0.7714` | `0.9429` | `0.9429` | Strong concordance ($\rho \ge 0.77$ to $0.94$) |

- **Overall Mean Spearman $\rho$ Across 9 Baseline Settings**: **`0.8794`**.

---

## 4. Crack Severity Multiplier Scale Sensitivity

Evaluation across crack-to-volume multipliers $\text{scale} \in \{5.0, 15.0, 30.0, 60.0\}$ (configured default: `15.0`):

| Crack Multiplier Scale | Mean Spearman $\rho$ | Pothole Rank-1 Count | Pothole Rank-1 % | Crack Rank-1 Count | Crack Rank-1 % | Crack Rank-1 Emergence |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **5.0** | `0.8222` | 18 / 18 | 100.0% | 0 / 18 | 0.0% | None (Potholes dominate 100%) |
| **15.0** | `0.8794` | 18 / 18 | 100.0% | 0 / 18 | 0.0% | None (Potholes dominate 100%) |
| **30.0** | `0.8540` | 16 / 18 | 88.9% | 2 / 18 | 11.1% | D20 Alligator Crack reaches Rank 1 (2/18 evals) |
| **60.0** | `0.7968` | 9 / 18 | 50.0% | 9 / 18 | 50.0% | D20 Alligator Crack reaches Rank 1 (9/18 evals) |

- **Scale Transition Finding**: Pothole Rank-1 dominance is strictly scale-dependent. At $\text{scale} \le 15.0$, potholes capture 100% of Rank-1 dispatches. At $\text{scale} \ge 30.0$, severe alligator cracking (`SRMD-IND-000102-00`) achieves Rank 1 under `safety-first` with individual regret weighting ($v \le 0.50$).

---

## 5. Large-Cohort $N = 300$ Validation Statistics

> **Cohort Specification & Provenance**: Explicitly labeled **`SYNTHETIC (templated bboxes, fixed seed)`**.
> Synthesized using `SRMDRecordBuilder(seed=2026)` with templated bounding boxes across 4 RDD2022 distress classes (75 D40, 75 D20, 75 D10, 75 D00).

| Multiplier Scale | Mean Cohort $\rho$ | Kendall $\tau$ | Top-5 Overlap | Safety-First $\rho$ | Traffic-First $\rho$ | Equity-First $\rho$ | Pothole Rank-1 Share |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **5.0** | `0.9479` | `0.8331` | `0.6667` | `0.8865` | `0.9685` | `0.9888` | 6 / 6 (100%) |
| **15.0** | `0.9466` | `0.8276` | `0.6667` | `0.9041` | `0.9464` | `0.9892` | 6 / 6 (100%) |
| **30.0** | `0.9359` | `0.8124` | `0.6667` | `0.9069` | `0.9127` | `0.9880` | 6 / 6 (100%) |
| **60.0** | `0.9774` | `0.8892` | `0.6667` | `0.9636` | `0.9780` | `0.9907` | 6 / 6 (100%) |

- **Overall Mean Cohort Concordance ($\bar{\rho}$)**: **`0.9520`**.
- **Overall Mean Kendall $\tau$**: **`0.8406`**.
- **Overall Mean Top-5 Overlap**: **`0.6667`**.
- **Baseline (Scale = 15.0) 1000-Resample Bootstrap 95% Confidence Intervals**:
  - Spearman Rank Correlation ($\rho$): `[0.9253, 0.9619]`
  - Kendall Rank Correlation ($\tau$): `[0.7970, 0.8550]`
  - Top-5 Priority Overlap: `[0.2000, 0.9333]`
- **Cohort Sample Size**: $N = 300$ multimodal inspection records.

---

## 6. Test Suite & Verification Rigor

- **Passing Test Cases**: `105` unit tests in `test_core_domain.py`.
- **Test Suite Status**: 100% passing (`Ran 105 tests, OK`).
- **Verification Scope**: Physics bounds, sensor switching, CLAHE contrast enhancement, pinhole projection geometry, paraboloid cavity volume, TOPSIS/VIKOR mathematical invariants, and offline fallback synchronization.

---

## 7. Real vs Simulated System Architecture Disclosure

| Subsystem / Layer | Real / Empirical Elements | Simulated / Synthetic Elements | Scientific Rationale & Constraints |
| :--- | :--- | :--- | :--- |
| **Sensors & Telemetry** | Physical camera resolution ($720 \times 720$), real-world GPS coordinates on Mumbai Western Express Highway, real hospital/school coordinates. | Optical LiDAR ToF depth, ultrasonic sonar acoustic echo, chassis z-acceleration impulses. | Physical vehicles require multi-sensor hardware rigs. Sensor switching simulates optical LiDAR absorption vs acoustic echo under rain conditions. |
| **Defect Detection** | Authentic RDD2022 dataset damage classes (`D00`, `D10`, `D20`, `D40`) and annotation bounding box schema. | Sample images are PIL-generated demo artifacts; ground-truth lookup detector bypasses live YOLOv8 inference during testing. | Eliminates GPU dependency and ensures deterministic evaluation of downstream sensor fusion and MCDM layers. |
| **Vision Enhancement** | OpenCV LAB color space CLAHE (`cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))`) applied to L-channel; real luminance standard deviation computation. | Pre-rendered visual split slider comparison for demonstration web UI. | Real CLAHE algorithm runs directly on backend via `/api/v1/preprocess/preview/{image_name}`. |
| **Depth & Volume** | Geometric pinhole projection equations ($Z = \frac{f \cdot H}{v - c_y}$); calibrated paraboloid cavity model ($V = 0.5 \cdot A \cdot D \cdot 1000$). | Fixed chassis height ($30\text{ cm}$); calibrated bounding box footprints in demo annotations (`India_000045`, `India_000278`). | Raw VOC bounding boxes in demo images were unconstrained; calibrated footprints ensure realistic physical volumes ($0.05\text{--}0.40\text{ m}^2$, $2\text{--}12\text{ cm}$ depth). |
| **Prioritization Metrics** | Rigorous Fuzzy BWM weight derivations, TOPSIS vector normalization & Euclidean separation, VIKOR $L_p$ compromise ranking. | Traffic volume (PCU) from regional corridor estimates; hospital route distance via geodesic approximation ($1.25\times$ tortuosity). | MCDM engine runs mathematically exact algorithms identically across live backend and offline frontend fallback. |
| **Dispatch & Fleet** | Quantile priority tiers (Top 10% Critical, 20% High, 30% Medium, 40% Low), absolute alarm rule ($C_i \ge 0.60$). | Mock IoT vehicle script (`simulation/mock_iot_vehicle.py`) for live telemetry streaming demonstrations. | Enables unattended demo streaming while maintaining identical mathematical prioritization on all received packets. |
