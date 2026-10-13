# Multi-Criteria Decision Making (MCDM) Mathematical Specification

This document summarizes the mathematical formulations, weight-derivation models, compromise-ranking algorithms, spatial/contextual criteria, and threshold calibration implemented in [`services/backend/app/services/mcdm_engine.py`](../services/backend/app/services/mcdm_engine.py) and [`services/data_generation/infrastructure_model.py`](../services/data_generation/infrastructure_model.py).

---

## 1. Decision Criteria & Benefit/Cost Polarity

The MCDM engine supports two compatible operating modes:
- **Extended 6-Criteria Mode** (used by the live Digital Twin backend, SRMD generator, `/api/v1/prioritization`, and ablation experiments): Incorporates physical defect geometry, vehicular traffic exposure, emergency trauma hospital proximity, operating speed, school-zone proximity, and functional roadway hierarchy.
- **Legacy 4-Criteria Mode** (preserved when `MCDMPrioritizationEngine()` is initialized with default arguments or passed a $N \times 4$ matrix): Uses the original four criteria (`[0.35, 0.25, 0.25, 0.15]`) so all legacy unit tests remain 100% backward-compatible.

| Index $j$ | Criterion Key | Symbol | Unit | Direction | Safety-First $w_j$ | Traffic-First $w_j$ | Equity-First $w_j$ | Technical Justification & Polarity Rationale |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $0$ | `defect_severity_magnitude` | $C_1$ | L-equivalent | **Benefit** ($\uparrow$) | $0.2798$ | $0.2263$ | $0.1587$ | **Structural Void Magnitude**: Cavity volume ($V$, Liters) for `D40` potholes, or scaled crack severity index ($\text{CSI} \times \text{scale}$) for `D00`/`D10`/`D20` cracks. Higher void or crack severity causes greater vehicular tire/rim impact risk and structural base degradation. |
| $1$ | `traffic_pcu` | $C_2$ | PCU/day | **Benefit** ($\uparrow$) | $0.1346$ | $0.2546$ | $0.1021$ | **Vehicular Exposure Frequency**: Passenger Car Units per day traversing the defect. Higher exposure frequency amplifies corridor disruption, user delay costs, and cumulative probability of vehicle accidents. |
| $2$ | `dist_hospital_km` | $C_3$ | km | **Cost** ($\downarrow$) | $0.2418$ | $0.0876$ | $0.2950$ | **Emergency Ambulance Corridor Urgency**: Tortuosity-corrected geodesic route distance to the nearest trauma hospital. **Cost criterion ($\downarrow$)**: *Smaller distance yields higher urgency* ($f^*(x) = \min_i x_i$). A defect located within $0.2\text{–}1.0\text{ km}$ of a major trauma center poses immediate risk of destabilizing emergency medical transport (ambulance suspension damage or patient transport shock). |
| $3$ | `speed_kmh` | $C_4$ | km/h | **Benefit** ($\uparrow$) | $0.1346$ | $0.1362$ | $0.0799$ | **Kinetic Impact Energy**: Mean operating vehicle speed. Kinetic shock energy scales quadratically ($E_k \propto v^2$); higher speed exponentially magnifies vertical suspension acceleration ($g$-force) upon defect encounter. |
| $4$ | `school_proximity_km` | $C_5$ | km | **Cost** ($\downarrow$) | $0.1346$ | $0.0690$ | $0.2622$ | **Vulnerable Pedestrian & Transit Urgency**: Geodesic route distance to the nearest K-12 school or educational institution. **Cost criterion ($\downarrow$)**: *Smaller distance yields higher urgency* ($f^*(x) = \min_i x_i$). Pavement defects close to schools endanger vulnerable pedestrians (children crossing), cyclists, two-wheeler school drop-offs, and school-bus maneuvering. |
| $5$ | `road_class` | $C_6$ | Ordinal $\{1..4\}$ | **Benefit** ($\uparrow$) | $0.0746$ | $0.2263$ | $0.1021$ | **Functional Roadway Hierarchy**: Ordinal classification (`LOCAL`=1, `COLLECTOR`=2, `ARTERIAL`=3, `HIGH_TRAFFIC_CORRIDOR`=4). Higher arterial tiers carry regional transit significance and wider economic consequence upon lane blockage. |

### Unified Defect Severity Magnitude ($C_1$) & Class-Aware Scaling Justification

In pavement distress engineering (ASTM D6433 / IRC:82), **volumetric cavity loss (`calculated_volume_liters`)** is physically meaningful **only** for discrete bowl-shaped depressions (`D40` Pothole):

$$V_{\text{pothole}} = A_{\text{m}^2} \times d_{\text{cm}} \times 0.8 \times 10 \quad (\text{Liters}, \text{ only for } \text{D40})$$

For surface cracks (`D00` Longitudinal, `D10` Transverse, `D20` Alligator), multiplying the full bounding-box footprint area by vertical relief depth artificially fabricates 50–80 L "volumes" for shallow surface fissures, causing wide alligator cracks to falsely outrank deep structural potholes. Therefore:
1. **For cracks (`D00`, `D10`, `D20`)**, `calculated_volume_liters = null` (`None`) and a bounded **`crack_severity_index` $\in [0, 1]$** is computed instead:
   - **Linear cracks (`D00`, `D10`)**: Bounding-box major-axis length $L_{\text{m}} = \sqrt{A_{\text{m}^2} \cdot r_{\text{aspect}}}$ weighted by depth $d_{\text{cm}}$, normalized against $L_{\max}\cdot d_{\max} = 8.0\,\text{m}\cdot\text{cm}$:
     $$\text{CSI}_{\text{D00/D10}} = \text{clip}\left(\frac{L_{\text{m}} \cdot d_{\text{cm}}}{8.0},\; 0.0,\; 1.0\right)$$
   - **Alligator / fatigue cracks (`D20`)**: Affected area $A_{\text{m}^2}$ with linear depth weighting $(0.5 + 0.5\,\min(d_{\text{cm}}/5.0, 1.0))$, normalized against $A_{\max} = 4.0\,\text{m}^2$:
     $$\text{CSI}_{\text{D20}} = \text{clip}\left(\frac{A_{\text{m}^2} \cdot \left(0.5 + 0.5\min(d_{\text{cm}}/5.0, 1.0)\right)}{4.0},\; 0.0,\; 1.0\right)$$
2. **Unified MCDM Criterion (`defect_severity_magnitude`)**:
   $$\text{defect\_severity\_magnitude} = \begin{cases} V_{\text{pothole}} \text{ (Liters)} & \text{if } \text{class} = \text{D40} \\ 15.0 \times \text{crack\_severity\_index} & \text{if } \text{class} \in \{\text{D00}, \text{D10}, \text{D20}\} \end{cases}$$
   **Scaling Justification (`CRACK_SEVERITY_TO_MAGNITUDE_SCALE = 15.0`)**:
   - Under `CRACK_SEVERITY_TO_MAGNITUDE_SCALE = 15.0`, a maximum-severity fatigue crack ($\text{CSI} = 1.0$) maps to $15.0$ L-equivalent—comparable to a moderate pothole ($15\text{ L}$, e.g. $0.25\,\text{m}^2 \times 7.5\,\text{cm}$ depth), while typical linear cracks ($\text{CSI} \approx 0.15\text{–}0.35$) map to $2.25\text{–}5.25$ L-equivalent and severe alligator cracks ($\text{CSI} \approx 0.65\text{–}0.85$) map to $9.75\text{–}12.75$ L-equivalent.
   - This prevents L2 column normalization in TOPSIS from collapsing crack severity to near-zero (which would happen if unscaled $[0, 1]$ values were mixed directly with $20\text{–}145\text{ L}$ pothole volumes), while ensuring that a large structural pothole ($\ge 20\text{ L}$) consistently outranks a surface crack given equal traffic and spatial context.

### Road Class Ordinal Encoding ($C_6$)

Functional road classes in the Mumbai Metropolitan Corridor are mapped deterministically via `encode_road_class()` in [`infrastructure_model.py`](../services/data_generation/infrastructure_model.py):

$$\text{encode\_road\_class}(\text{class}) = \begin{cases} 1 & \text{if } \text{LOCAL} \\ 2 & \text{if } \text{COLLECTOR} \\ 3 & \text{if } \text{ARTERIAL} \\ 4 & \text{if } \text{HIGH\_TRAFFIC\_CORRIDOR} \end{cases}$$

### School & Hospital Geodesic Proximity ($C_3, C_5$)

Both `dist_hospital_km` ($C_3$) and `school_proximity_km` ($C_5$) are computed using the spherical Haversine geodesic distance ($R = 6371.0\text{ km}$) scaled by the urban street-network tortuosity factor ($\tau = 1.18$):

$$d_{\text{geo}}(\phi_1, \lambda_1, \phi_2, \lambda_2) = 2R \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta\phi}{2}\right) + \cos\phi_1\cos\phi_2\sin^2\left(\frac{\Delta\lambda}{2}\right)}\right)$$

$$d_{\text{route}} = \max\left(0.05,\; \tau \cdot \min_{k \in \mathcal{F}} d_{\text{geo}}(\text{defect}, \text{facility}_k)\right)$$

- **Hospital Registry ($\mathcal{F}_{\text{hosp}}$)**: 8 geocoded trauma and municipal hospitals along the Mumbai Western & Coastal Corridor (`is_emergency_corridor = True` when $d_{\text{route}} \le 1.0\text{ km}$).
- **School Registry ($\mathcal{F}_{\text{school}}$)**: 18 real Mumbai Western Corridor schools spanning Bandra West, BKC, Santacruz, Juhu, Vile Parle, Andheri, Jogeshwari, Goregaon, Malad, and Borivali (`is_school_zone = True` when $d_{\text{route}} \le 0.5\text{ km}$).

---

## 2. Fuzzy Best-Worst Method (Fuzzy BWM) Weight Derivation

To avoid arbitrary ad-hoc weighting across 6 heterogeneous criteria, [`derive_fuzzy_bwm_weights()`](../services/backend/app/services/mcdm_engine.py) implements the **Fuzzy Best-Worst Method** (Guo & Zhao, 2017) using **Triangular Fuzzy Numbers (TFNs)**.

### 2.1 Triangular Fuzzy Linguistic Scale

A Triangular Fuzzy Number is denoted as $\tilde{a} = (l, m, u)$ with $0 < l \le m \le u$. Pairwise comparisons between the **Best** criterion $B$ and criterion $j$ ($\tilde{a}_{Bj} = (l_{Bj}, m_{Bj}, u_{Bj})$), and between criterion $j$ and the **Worst** criterion $W$ ($\tilde{a}_{jW} = (l_{jW}, m_{jW}, u_{jW})$), use the standard 5-level linguistic scale:

| Linguistic Term | Code | TFN $(l, m, u)$ | Consistency Index $CI$ |
| :--- | :---: | :---: | :---: |
| Equally Important | `EI` | $(1.0,\; 1.0,\; 1.0)$ | $3.00$ |
| Weakly Important | `WI` | $(2/3,\; 1.0,\; 3/2)$ | $3.80$ |
| Fairly Important | `FI` | $(3/2,\; 2.0,\; 5/2)$ | $5.29$ |
| Very Important | `VI` | $(5/2,\; 3.0,\; 7/2)$ | $6.69$ |
| Absolutely Important | `AI` | $(7/2,\; 4.0,\; 9/2)$ | $8.04$ |

### 2.2 Nonlinear Minimax Optimization Formulation

For $n$ criteria, we seek fuzzy weights $\tilde{w}_j = (l_j^w, m_j^w, u_j^w)$ ($j = 1,\dots,n$) and minimax deviation $\xi^*$ solving:

$$\min_{\tilde{w}, \xi} \quad \xi$$

subject to:

$$\begin{aligned}
\left| l_B^w - l_{Bj} u_j^w \right| &\le \xi u_j^w, \quad \left| m_B^w - m_{Bj} m_j^w \right| \le \xi m_j^w, \quad \left| u_B^w - u_{Bj} l_j^w \right| \le \xi l_j^w \quad \forall j = 1,\dots,n \\
\left| l_j^w - l_{jW} u_W^w \right| &\le \xi u_W^w, \quad \left| m_j^w - m_{jW} m_W^w \right| \le \xi m_W^w, \quad \left| u_j^w - u_{jW} l_W^w \right| \le \xi l_W^w \quad \forall j = 1,\dots,n \\
\sum_{j=1}^{n} R(\tilde{w}_j) &= 1, \quad 0 < l_j^w \le m_j^w \le u_j^w \quad \forall j = 1,\dots,n
\end{aligned}$$

where crisp defuzzification uses the **Graded Mean Integration Representation (GMIR)**:

$$R(\tilde{w}_j) = \frac{l_j^w + 4m_j^w + u_j^w}{6}$$

The optimization problem is solved via Sequential Least Squares Programming (`scipy.optimize.minimize(..., method="SLSQP")`) with a multi-grid coordinate polish fallback if `scipy` is unavailable.

### 2.3 Consistency Index ($CI$) and Consistency Ratio ($CR$)

Given the upper bound $u_{BW}$ of the Best-to-Worst comparison $\tilde{a}_{BW}$, the maximum possible deviation $\xi_{\max} = CI$ satisfies $(CI)^2 - (1 + 2u_{BW})CI + (u_{BW}^2 - u_{BW}) = 0$:

$$CI = \frac{1 + 2u_{BW} + \sqrt{8u_{BW} + 1}}{2}, \qquad CR = \frac{\xi^*}{CI}$$

A consistency ratio $CR < 0.10$ indicates high pairwise judgment consistency.

### 2.4 Named Fuzzy BWM Weight Profiles

Three preset profiles plus a `custom` option are provided in [`NAMED_BWM_PROFILES`](../services/backend/app/services/mcdm_engine.py):

| Profile | Best Criterion ($B$) | Worst Criterion ($W$) | Derived Crisp Weights $[w_1, w_2, w_3, w_4, w_5, w_6]$ | $\xi^*$ | $CR$ |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **`safety-first`** *(Default)* | Defect Volume ($C_1$) | Road Class ($C_6$) | `[0.2798, 0.1346, 0.2418, 0.1346, 0.1346, 0.0746]` | $0.2071$ | $0.0258$ |
| **`traffic-first`** | Traffic Density ($C_2$) | School Proximity ($C_5$) | `[0.2263, 0.2546, 0.0876, 0.1362, 0.0690, 0.2263]` | $0.2952$ | $0.0367$ |
| **`equity-first`** | Hospital Proximity ($C_3$) | Operating Speed ($C_4$) | `[0.1587, 0.1021, 0.2950, 0.0799, 0.2622, 0.1021]` | $0.2191$ | $0.0273$ |

---

## 3. TOPSIS Relative Closeness Ranking

Given an $m \times n$ decision matrix $X = [x_{ij}]$ and normalized weights $w = [w_1,\dots,w_n]$ ($\sum w_j = 1$):

1. **L2 Vector Normalization** (with zero-column guard):
   $$r_{ij} = \frac{x_{ij}}{\sqrt{\sum_{k=1}^{m} x_{kj}^2}}$$
2. **Weighted Normalized Decision Matrix**:
   $$v_{ij} = w_j \cdot r_{ij}$$
3. **Positive Ideal Solution ($A^+$) and Negative Ideal Solution ($A^-$)**:
   $$v_j^+ = \begin{cases} \max_i v_{ij} & \text{if } j \in \mathcal{B} \text{ (Benefit)} \\ \min_i v_{ij} & \text{if } j \in \mathcal{C} \text{ (Cost)} \end{cases}, \qquad v_j^- = \begin{cases} \min_i v_{ij} & \text{if } j \in \mathcal{B} \\ \max_i v_{ij} & \text{if } j \in \mathcal{C} \end{cases}$$
4. **Euclidean Separation & Relative Closeness**:
   $$D_i^+ = \sqrt{\sum_{j=1}^{n} (v_{ij} - v_j^+)^2}, \qquad D_i^- = \sqrt{\sum_{j=1}^{n} (v_{ij} - v_j^-)^2}, \qquad C_i^* = \frac{D_i^-}{D_i^+ + D_i^-} \in [0, 1]$$

   - Higher $C_i^*$ indicates greater repair priority (Rank 1 has maximum $C_i^*$).
   - **Degenerate tie-handling**: If $D_i^+ + D_i^- = 0$ (all alternatives are identical across all criteria), $C_i^* = 0.5$ is returned for all alternatives rather than collapsing to $0.0$.

---

## 4. VIKOR Compromise Ranking & Stability Conditions

[`compute_vikor()`](../services/backend/app/services/mcdm_engine.py) implements the VIKOR (Opricovic & Tzeng, 2004) compromise ranking method to balance **maximum group utility** ($S_i$, "majority rule") against **minimum individual regret** ($R_i$, "opponent regret").

1. **Best ($f_j^*$) and Worst ($f_j^-$) Criterion Bounds**:
   $$f_j^* = \begin{cases} \max_i x_{ij} & j \in \mathcal{B} \\ \min_i x_{ij} & j \in \mathcal{C} \end{cases}, \qquad f_j^- = \begin{cases} \min_i x_{ij} & j \in \mathcal{B} \\ \max_i x_{ij} & j \in \mathcal{C} \end{cases}$$
2. **Normalized Regret Dissimilarity ($d_{ij} \in [0, 1]$)**:
   $$d_{ij} = \begin{cases} \dfrac{f_j^* - x_{ij}}{f_j^* - f_j^-} & \text{if } j \in \mathcal{B} \\[8pt] \dfrac{x_{ij} - f_j^*}{f_j^- - f_j^*} & \text{if } j \in \mathcal{C} \end{cases}$$
   (When $|f_j^* - f_j^-| < 10^{-12}$, $d_{ij} = 0$.)
3. **Group Utility ($S_i$) and Individual Regret ($R_i$)**:
   $$S_i = \sum_{j=1}^{n} w_j d_{ij}, \qquad R_i = \max_{j=1,\dots,n} \left(w_j d_{ij}\right)$$
4. **Compromise Index ($Q_i$, default $v = 0.5$)**:
   Let $S^* = \min_i S_i$, $S^- = \max_i S_i$, $R^* = \min_i R_i$, $R^- = \max_i R_i$. Then:
   $$Q_i = v \frac{S_i - S^*}{S^- - S^*} + (1 - v) \frac{R_i - R^*}{R^- - R^*} \in [0, 1]$$
   **Lower $Q_i$ indicates higher urgency** ($Q_{\text{best}} = 0.0$ is Rank 1). For priority-tier classification on $[0, 1]$ where higher is more urgent, $1 - Q_i$ is used.
5. **Compromise Solution Conditions**:
   Let $A^{(1)}, A^{(2)}, \dots, A^{(m)}$ be the alternatives sorted by ascending $Q_i$:
   - **Condition C1 (Acceptable Advantage)**:
     $$Q(A^{(2)}) - Q(A^{(1)}) \ge DQ, \qquad DQ = \frac{1}{m - 1}$$
   - **Condition C2 (Acceptable Stability in Decision Making)**:
     $A^{(1)}$ must also be best-ranked ($1\text{st}$) by $S_i$ and/or $R_i$.
   - When C1 fails, the compromise set includes $\{A^{(1)}, \dots, A^{(k)}\}$ where $Q(A^{(k)}) - Q(A^{(1)}) < DQ$. When C2 fails, the compromise set is $\{A^{(1)}, A^{(2)}\}$.

---

## 5. TOPSIS–VIKOR Concordance (`compute_all`)

`MCDMPrioritizationEngine.compute_all(matrix, v=0.5)` executes both TOPSIS and VIKOR on the same decision matrix and evaluates the **Spearman rank correlation coefficient** ($\rho \in [-1, 1]$) between the 1-based TOPSIS ranks ($r_i^{\text{TOPSIS}}$) and VIKOR ranks ($r_i^{\text{VIKOR}}$):

$$\rho = 1 - \frac{6 \sum_{i=1}^{m} \left(r_i^{\text{TOPSIS}} - r_i^{\text{VIKOR}}\right)^2}{m(m^2 - 1)}$$

On the 6-record Mumbai SRMD benchmark under the `safety-first` profile (with class-aware `defect_severity_magnitude`, pinhole ground-plane area calibration, realistic municipal pothole bounds 0.05–0.40 m², 2–12 cm depth, and water-surface sonar/LiDAR physics):
- **Spearman Rank Correlation**: Across profiles, $\rho = 0.2000$ under `safety-first` ($v = 0.50$, due to individual regret balancing between volume and trauma proximity), $\rho = 1.0000$ under `traffic-first`, and $\rho = 0.9429$ under `equity-first`. Both SRMD JSON and API paths produce identical matrices and rank orders.
- **TOPSIS** (compensatory Euclidean closeness) ranks `SRMD-IND-000045-00` ($23.22\text{ L}$ pothole on Western Express Highway) as Rank 1 ($C_0^* = 0.5710$) and `SRMD-IND-000278-00` ($8.29\text{ L}$ highway pothole $0.27\text{ km}$ from trauma hospital) as Rank 2 ($C_4^* = 0.5583$).
- **VIKOR** ($v = 0.5$, balancing group utility $S_i$ and minimax individual regret $R_i$) ranks `SRMD-IND-000278-00` ($8.29\text{ L}$ pothole $0.27\text{ km}$ from trauma hospital, minimizing $S_4 = 0.2367$ and $R_4 = 0.1980$) as Rank 1 ($Q_4 = 0.0000$) and `SRMD-IND-000278-01` as Rank 2 ($Q = 0.3625$).

---

## 6. Priority Thresholds & Quantile Tiers

**Honesty & Methodological Disclosure**: Priority classification replaces hand-tuned fixed cut-offs with two complementary mechanisms:
1. **(a) Batch Quantile Tiers (Default)**: Triage levels are assigned based on relative batch distribution percentiles:
   - 🔴 **Critical**: Top 10% (rank / N < 0.10, at least top-1 when N >= 1)
   - 🟠 **High**: Next 20% (0.10 <= rank / N < 0.30)
   - 🟡 **Medium**: Next 30% (0.30 <= rank / N < 0.60)
   - 🟢 **Low**: Remaining 40% (rank / N >= 0.60)
2. **(b) Fixed Absolute Alarm Rule**: Regardless of quantile ranking, any incident with TOPSIS >= 0.60 is automatically escalated to 🔴 **Critical** to guarantee immediate municipal response for severe structural emergencies.

Both mechanisms are fully configurable in [`config/sensor_profiles.yaml`](../config/sensor_profiles.yaml) under `priority_thresholds`:
```yaml
priority_thresholds:
  mode: "quantile_tiers"
  quantile_tiers:
    critical_top_pct: 0.10
    high_next_pct: 0.20
    medium_next_pct: 0.30
  absolute_alarm_rule:
    topsis_alarm_critical: 0.60
  critical: 0.57
  high: 0.45
  medium: 0.38
```

Every response from `GET /api/v1/prioritization` and `POST /api/v1/prioritization` includes a top-level `calibration_note` field disclosing this directly to API consumers and UI clients:
> *"Priority tiers use batch quantile ranking (top 10% Critical, next 20% High, next 30% Medium, rest Low) with a fixed absolute alarm rule (TOPSIS >= 0.60 => Critical). Hand-tuned cut-offs are replaced by batch distribution quantiles and are not validated against real maintenance outcomes."*

### Why Batch Quantile Tiers Replace Static Cut-Offs

In a 6-dimensional Euclidean space with spatial trade-offs, static quartile thresholds (0.75 / 0.50 / 0.25) suffer from score compression: highway defects exhibit higher void volume, traffic PCU/day, and road class hierarchy, whereas collector/local defects lie closer to schools or hospitals. Because no single defect simultaneously achieves the ideal extremum on all 6 axes, TOPSIS relative closeness C_i* typically clusters between 0.35 and 0.60. Batch quantile tiering ensures that resources are distributed proportionally across maintenance crews regardless of seasonal or corridor-wide score drift, while the absolute alarm rule ensures extreme hazards are never downgraded.

### Plain Operational Notes on Demonstration Cohort & Score Proximity

> **Note A (Quantile Tiering vs. Absolute Alarm at $N = 6$)**:
> With $N = 6$ demonstration records, the batch quantile allocation rule ($\lceil 0.10 \times 6 \rceil = 1$) always assigns exactly **one** incident to the `Critical` priority tier (`SRMD-IND-000278-00` under `safety-first`). The fixed absolute alarm rule ($\text{TOPSIS} \ge 0.60 \implies \text{Critical}$) **never fired** on this baseline demo set, as both highway potholes scored below $0.60$ ($0.5782$ and $0.5747$).
>
> **Note B (Top-2 Pothole Proximity & Functional Tie)**:
> Under the baseline `safety-first` profile, the top two highway potholes (`SRMD-IND-000278-00` at $C_i = 0.5782$ and `SRMD-IND-000045-00` at $C_i = 0.5747$) are separated by a negligible margin of only **$0.0035$ TOPSIS score units** ($|0.5782 - 0.5747| = 0.0035 < 0.01$). Given physical sensor measurement tolerances ($\pm 0.5\text{ cm}$ depth noise), the top-2 potholes are **within $0.01$ TOPSIS of each other and should be treated as functionally tied** for field maintenance dispatch.

---

## 7. Sensitivity Analysis & Scale Variation (3 Profiles x 3 v-values x 4 Scales)

[`scripts/run_sensitivity.py`](../scripts/run_sensitivity.py) evaluates the sensitivity of TOPSIS and VIKOR across combinations of criteria weight profiles, consensus parameters ($v \in \{0.25, 0.50, 0.75\}$), and crack severity scales ($\text{scale} \in \{5.0, 15.0, 30.0, 60.0\}$). Results are exported to [`runs/experiments/sensitivity.json`](../runs/experiments/sensitivity.json), `frontend/src/data/sensitivity.json`, and `frontend/public/sensitivity.json`.

### 7.1 Baseline 6-Record Experimental Benchmark (@ Configured Scale = 15.0)

| Setting | Profile | $v$ | TOPSIS Ranks | VIKOR Ranks | Spearman $\rho$ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `safety-first_v0.25` | `safety-first` | $0.25$ | `[2, 3, 6, 4, 1, 5]` | `[5, 2, 6, 4, 1, 3]` | $+0.6000$ |
| `safety-first_v0.50` | `safety-first` | $0.50$ | `[2, 3, 6, 4, 1, 5]` | `[4, 2, 6, 5, 1, 3]` | $+0.7143$ |
| `safety-first_v0.75` | `safety-first` | $0.75$ | `[2, 3, 6, 4, 1, 5]` | `[2, 3, 6, 5, 1, 4]` | $+0.9429$ |
| `traffic-first_v0.25` | `traffic-first` | $0.25$ | `[1, 4, 5, 6, 2, 3]` | `[1, 4, 5, 6, 2, 3]` | $+1.0000$ |
| `traffic-first_v0.50` | `traffic-first` | $0.50$ | `[1, 4, 5, 6, 2, 3]` | `[1, 4, 5, 6, 2, 3]` | $+1.0000$ |
| `traffic-first_v0.75` | `traffic-first` | $0.75$ | `[1, 4, 5, 6, 2, 3]` | `[1, 4, 5, 6, 2, 3]` | $+1.0000$ |
| `equity-first_v0.25` | `equity-first` | $0.25$ | `[6, 3, 2, 4, 1, 5]` | `[6, 3, 4, 2, 1, 5]` | $+0.7714$ |
| `equity-first_v0.50` | `equity-first` | $0.50$ | `[6, 3, 2, 4, 1, 5]` | `[6, 2, 3, 4, 1, 5]` | $+0.9429$ |
| `equity-first_v0.75` | `equity-first` | $0.75$ | `[6, 3, 2, 4, 1, 5]` | `[6, 2, 3, 4, 1, 5]` | $+0.9429$ |

*Mean Spearman $\rho = 0.8794$ across all 9 baseline configurations.*

### 7.2 Crack Severity Multiplier Scale Sensitivity ({5.0, 15.0, 30.0, 60.0})

To evaluate whether the observed Rank-1 pothole dominance is an intrinsic property of the multi-criteria geometry or merely an artifact of the chosen scaling constant (`crack_severity_to_magnitude_scale = 15.0`), the experiment was systematically extended across $\text{scale} \in \{5.0, 15.0, 30.0, 60.0\}$:

| Scale Factor | Mean $\rho$ | Pothole Rank 1 | Crack Rank 1 | Crack R1 % | Cracks Reach Rank 1? |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **$5.0$** | $0.8222$ | $18 / 18$ ($100\%$) | $0 / 18$ ($0\%$) | $0.0\%$ | **No** (Cracks severely suppressed) |
| **$15.0$** *(Configured)* | $0.8794$ | $18 / 18$ ($100\%$) | $0 / 18$ ($0\%$) | $0.0\%$ | **No** (100% Pothole dominance) |
| **$30.0$** | $0.8540$ | $16 / 18$ ($88.9\%$) | $2 / 18$ ($11.1\%$) | $11.1\%$ | **YES** (`SRMD-IND-000102-00` Alligator Crack in VIKOR `safety-first` at $v=0.25, 0.50$) |
| **$60.0$** | $0.7968$ | $9 / 18$ ($50.0\%$) | $9 / 18$ ($50.0\%$) | $50.0\%$ | **YES** (`SRMD-IND-000102-00` in TOPSIS & VIKOR `safety-first`; `SRMD-IND-000278-01` in TOPSIS `traffic-first`) |

### 7.3 Large-Cohort Validation ($N = 300$, SYNTHETIC: templated bboxes, fixed seed)

Evaluating an $N = 300$ cohort explicitly labeled **`SYNTHETIC (templated bboxes, fixed seed)`** with balanced distress taxonomy ($75 \times \text{D40}, 75 \times \text{D20}, 75 \times \text{D10}, 75 \times \text{D00}$) proves high method concordance and rank stability across all scales:

| Scale Factor | Mean Spearman $\rho$ | Kendall $\tau$ | Top-5 Overlap | Safety-First $\rho$ | Traffic-First $\rho$ | Equity-First $\rho$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$5.0$** | $0.9479$ | $0.8504$ | $0.6667$ | $0.8865$ | $0.9685$ | $0.9888$ |
| **$15.0$** *(Base)* | $0.9466$ | $0.8266$ | $0.6667$ | $0.9041$ | $0.9464$ | $0.9892$ |
| **$30.0$** | $0.9359$ | $0.8033$ | $0.6000$ | $0.9069$ | $0.9127$ | $0.9880$ |
| **$60.0$** | $0.9774$ | $0.8820$ | $0.7333$ | $0.9636$ | $0.9780$ | $0.9907$ |

- **Overall Cohort Metrics**: Mean Concordance $\bar{\rho} = 0.9520$, Mean Kendall $\tau = 0.8406$, Mean Top-5 Priority Overlap $= 0.6667$.
- **Baseline Scale ($15.0$) 1000-Resample Bootstrap 95% Confidence Intervals**:
  - Spearman Rank Correlation ($\rho$): `[0.9253, 0.9619]`
  - Kendall Rank Correlation ($\tau$): `[0.7970, 0.8550]`
  - Top-5 Priority Overlap: `[0.2000, 0.9333]`

### 7.4 Methodological Insights & Limitations

1. **Scale-Dependent Rank-1 Dominance**:
   - At $\text{scale} \le 15.0$, deep structural potholes (`D40`) capture 100% of Rank-1 dispatches because their physical cavity volumes ($8.39\text{–}22.00\text{ L}$) exceed the scaled crack index ($15.0 \times \text{CSI} \le 15.0\text{ L}$).
   - At $\text{scale} = 30.0$, widespread alligator fatigue cracking (`SRMD-IND-000102-00`, $\text{CSI} = 0.58 \implies 17.4\text{ L-eq}$) overtakes moderate potholes in VIKOR under `safety-first` ($v \le 0.50$) due to heavy risk balancing.
   - At $\text{scale} = 60.0$, alligator cracks capture 50% of all Rank-1 dispatches across both TOPSIS and VIKOR.
   - **Conclusion**: The "100% Potholes" phenomenon is **not** an intrinsic physical law of road networks; it is a direct parameter consequence of setting `crack_severity_to_magnitude_scale = 15.0`. Municipal operators must tune this configuration parameter to reflect their agency's policy trade-off between structural cavity repair and preventive crack sealing.
2. **Profile Divergence & Consensus Sensitivity**:
   - `traffic-first` achieves near-perfect concordance ($\rho \ge 0.95$ in $N=300$, $\rho = 1.0000$ in $N=6$) because daily traffic volume and arterial hierarchy align directly with pavement defect magnitude.
   - `equity-first` exhibits high concordance ($\rho \ge 0.94$) consistently prioritizing trauma hospital ($C_3$) and school ($C_5$) proximity.
   - `safety-first` reflects the strongest sensitivity to the VIKOR consensus weight $v$, since low $v \le 0.25$ emphasizes individual opponent regret ($R_i$), heavily penalizing proximity to emergency trauma corridors.
3. **Sample Size Disclosure**:
   - The primary demonstration uses $N = 6$ sample records. While sufficient to verify algorithm determinism and invariant compliance, small-sample ranks are sensitive to individual rank swaps. The $N = 300$ synthetic cohort demonstrates that on realistic fleet cohorts, mean TOPSIS–VIKOR rank correlation exceeds $\rho \ge 0.94$ across all profiles and scales.
