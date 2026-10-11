"""
Multi-Criteria Decision Making (MCDM) Engine: Fuzzy BWM, TOPSIS, and VIKOR
==========================================================================

Evaluates road defect repair urgency using:
1. Fuzzy Best-Worst Method (Fuzzy BWM) with Triangular Fuzzy Numbers (TFNs) for
   deriving optimal criteria weights from pairwise Best-to-Others and Others-to-Worst
   comparisons (Guo & Zhao, 2017).
2. Technique for Order of Preference by Similarity to Ideal Solution (TOPSIS) using
   L2 vector normalization and Euclidean distance to positive/negative ideal solutions.
3. VIseKriterijumska Optimizacija I Kompromisno Resenje (VIKOR) computing group utility
   (S_i), individual regret (R_i), compromise index (Q_i, v=0.5), acceptable advantage,
   and acceptable stability checks (Opricovic & Tzeng, 2004).

Supported Criteria Sets:
------------------------
- Legacy 4-Criteria Mode (preserved for backward compatibility):
  [0] calculated_volume_liters (Benefit, default w=0.35)
  [1] traffic_pcu              (Benefit, default w=0.25)
  [2] dist_hospital_km         (Cost,    default w=0.25)
  [3] speed_kmh                (Benefit, default w=0.15)

- Extended 6-Criteria Mode (with school proximity & road hierarchy):
  [0] calculated_volume_liters (Benefit) - Physical cavity volume in Liters
  [1] traffic_pcu              (Benefit) - Daily traffic density in PCU/day
  [2] dist_hospital_km         (Cost)    - Geodesic route distance to nearest trauma hospital (km)
  [3] speed_kmh                (Benefit) - Operating vehicle speed (km/h)
  [4] school_proximity_km      (Cost)    - Geodesic route distance to nearest school (km)
  [5] road_class               (Benefit) - Ordinal road hierarchy weight (LOCAL=1..HIGH_TRAFFIC_CORRIDOR=4)
"""

import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union, Sequence
import numpy as np

try:
    from scipy.optimize import minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import yaml
except ImportError:
    yaml = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "sensor_profiles.yaml"

ROAD_CLASS_ORDINAL_WEIGHTS: Dict[str, int] = {
    "LOCAL": 1,
    "COLLECTOR": 2,
    "ARTERIAL": 3,
    "HIGH_TRAFFIC_CORRIDOR": 4
}


def encode_road_class(road_class: Any) -> int:
    """
    Maps a road class identifier or numeric value to its ordinal weight in {1, 2, 3, 4}:
        LOCAL = 1, COLLECTOR = 2, ARTERIAL = 3, HIGH_TRAFFIC_CORRIDOR = 4.
    """
    if isinstance(road_class, (int, float)):
        return max(1, min(4, int(round(float(road_class)))))
    if isinstance(road_class, str):
        key = road_class.strip().upper()
        if key in ROAD_CLASS_ORDINAL_WEIGHTS:
            return ROAD_CLASS_ORDINAL_WEIGHTS[key]
    return 3


CRITERIA_SPECS = [
    {
        "index": 0,
        "name": "calculated_volume_liters",
        "label": "Defect Volume",
        "unit": "Liters",
        "weight": 0.35,
        "is_benefit": True,
        "direction": "benefit",
        "description": "Larger cavity volume produces greater structural failure and vertical chassis shock."
    },
    {
        "index": 1,
        "name": "traffic_pcu",
        "label": "Traffic Density",
        "unit": "PCU/day",
        "weight": 0.25,
        "is_benefit": True,
        "direction": "benefit",
        "description": "Higher traffic volume increases vehicular exposure frequency and hazard probability."
    },
    {
        "index": 2,
        "name": "dist_hospital_km",
        "label": "Hospital Distance",
        "unit": "km",
        "weight": 0.25,
        "is_benefit": False,
        "direction": "cost",
        "description": "Critical emergency corridor proximity. Smaller distance increases urgency."
    },
    {
        "index": 3,
        "name": "speed_kmh",
        "label": "Vehicle Operating Speed",
        "unit": "km/h",
        "weight": 0.15,
        "is_benefit": True,
        "direction": "benefit",
        "description": "Higher velocity amplifies dynamic kinetic impact energy."
    }
]

EXTENDED_CRITERIA_SPECS = [
    {
        "index": 0,
        "name": "defect_severity_magnitude",
        "label": "Defect Volume / Crack Severity Magnitude",
        "unit": "L-equivalent",
        "is_benefit": True,
        "direction": "benefit",
        "description": "Unified defect severity magnitude: cavity volume (L) for D40 potholes, or crack_severity_index * 15.0 for D00/D10/D20 cracks."
    },
    {
        "index": 1,
        "name": "traffic_pcu",
        "label": "Traffic Density",
        "unit": "PCU/day",
        "is_benefit": True,
        "direction": "benefit",
        "description": "Higher traffic volume increases vehicular exposure frequency and hazard probability."
    },
    {
        "index": 2,
        "name": "dist_hospital_km",
        "label": "Hospital Proximity",
        "unit": "km",
        "is_benefit": False,
        "direction": "cost",
        "description": "Critical emergency corridor proximity. Smaller distance increases urgency."
    },
    {
        "index": 3,
        "name": "speed_kmh",
        "label": "Vehicle Operating Speed",
        "unit": "km/h",
        "is_benefit": True,
        "direction": "benefit",
        "description": "Higher velocity amplifies dynamic kinetic impact energy."
    },
    {
        "index": 4,
        "name": "school_proximity_km",
        "label": "School Proximity",
        "unit": "km",
        "is_benefit": False,
        "direction": "cost",
        "description": "Pedestrian & school-zone vulnerability. Smaller distance to school increases urgency."
    },
    {
        "index": 5,
        "name": "road_class",
        "label": "Road Class Hierarchy",
        "unit": "ordinal (1-4)",
        "is_benefit": True,
        "direction": "benefit",
        "description": "Functional roadway hierarchy (LOCAL=1, COLLECTOR=2, ARTERIAL=3, HIGH_TRAFFIC_CORRIDOR=4)."
    }
]

DEFAULT_WEIGHTS = np.array([0.35, 0.25, 0.25, 0.15], dtype=float)
DEFAULT_IS_BENEFIT = np.array([True, True, False, True], dtype=bool)
EXTENDED_IS_BENEFIT = np.array([True, True, False, True, False, True], dtype=bool)

# Calibrated vs Legacy priority thresholds
LEGACY_THRESHOLDS = {"critical": 0.75, "high": 0.50, "medium": 0.25}
CALIBRATED_THRESHOLDS = {"critical": 0.57, "high": 0.45, "medium": 0.31}
THRESHOLD_CALIBRATION_NOTE = (
    "Critical/High/Medium cut-offs (0.57 / 0.45 / 0.31) were chosen so the 6-record demo set "
    "spans all priority levels, and are not validated against real maintenance outcomes."
)


def load_calibrated_thresholds(config_path: Optional[Path] = None) -> Dict[str, float]:
    """Loads calibrated priority thresholds from config/sensor_profiles.yaml."""
    cfg_file = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if yaml and cfg_file.exists():
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            pt = loaded.get("mcdm", {}).get("priority_thresholds", {})
            if isinstance(pt, dict) and "critical" in pt:
                return {
                    "critical": float(pt.get("critical", 0.57)),
                    "high": float(pt.get("high", 0.45)),
                    "medium": float(pt.get("medium", 0.31))
                }
        except Exception:
            pass
    return CALIBRATED_THRESHOLDS.copy()


# ============================================================================
# Shared Decision-Matrix Row Builders
# ============================================================================

def build_mcdm_row_from_srmd(r: Dict[str, Any]) -> List[float]:
    """
    Builds the canonical 6-criteria MCDM decision-matrix row from a raw SRMD JSON record.

    Criteria order (matches EXTENDED_CRITERIA_SPECS):
        [0] defect_severity_magnitude  -- r["visual"]["defect_severity_magnitude"]
        [1] traffic_pcu                -- r["traffic"]["pcu"]
        [2] dist_hospital_km           -- r["infrastructure"]["dist_hospital_km"]
        [3] speed_kmh                  -- r["accelerometer"]["vehicle_speed_kmh"]
        [4] school_proximity_km        -- r["infrastructure"]["school_proximity_km"]
        [5] road_class_weight          -- r["traffic"]["road_class_weight"]

    Single source-of-truth used by both the SRMD-path test and the API load
    path, ensuring identical matrices and identical Spearman rho.
    """
    return [
        float(r["visual"]["defect_severity_magnitude"]),
        float(r["traffic"]["pcu"]),
        float(r["infrastructure"]["dist_hospital_km"]),
        float(r["accelerometer"]["vehicle_speed_kmh"]),
        float(r["infrastructure"]["school_proximity_km"]),
        float(r["traffic"]["road_class_weight"]),
    ]


def build_mcdm_row_from_store_item(item: Dict[str, Any]) -> List[float]:
    """
    Builds the canonical 6-criteria MCDM decision-matrix row from a telemetry_store item.

    The telemetry_store item must have been populated from the SRMD JSON via
    _load_srmd_records_into_store(), which preserves the pre-computed
    defect_severity_magnitude from r["visual"] rather than recomputing it.

    Criteria order matches build_mcdm_row_from_srmd() exactly:
        [0] defect_severity_magnitude  -- item["processed_telemetry"]["defect_severity_magnitude"]
        [1] traffic_pcu                -- item["traffic_pcu"]
        [2] dist_hospital_km           -- item["dist_hospital_km"]
        [3] speed_kmh                  -- item["speed_kmh"]
        [4] school_proximity_km        -- item["school_proximity_km"]
        [5] road_class_weight          -- encode_road_class(item["road_class"])
    """
    pt = item.get("processed_telemetry", {})
    mag = pt.get("defect_severity_magnitude")
    if mag is None:
        mag = pt.get("calculated_volume_liters") or 0.0
    return [
        float(mag),
        float(item["traffic_pcu"]),
        float(item["dist_hospital_km"]),
        float(item["speed_kmh"]),
        float(item.get("school_proximity_km", 0.5)),
        float(encode_road_class(item.get("road_class", "ARTERIAL"))),
    ]


# ============================================================================
# Fuzzy BWM (Best-Worst Method with Triangular Fuzzy Numbers)
# ============================================================================

# Standard Triangular Fuzzy Number (TFN) linguistic scale (Guo & Zhao, 2017)
TFN_LINGUISTIC_SCALE: Dict[str, Tuple[float, float, float]] = {
    "EI": (1.0, 1.0, 1.0),          # Equally Important
    "WI": (2.0 / 3.0, 1.0, 1.5),    # Weakly Important
    "FI": (1.5, 2.0, 2.5),          # Fairly Important
    "VI": (2.5, 3.0, 3.5),          # Very Important
    "AI": (3.5, 4.0, 4.5)           # Absolutely Important
}


def parse_tfn(val: Union[str, Tuple[float, float, float], List[float], Dict[str, float], float, int]) -> Tuple[float, float, float]:
    """Normalizes a linguistic code, tuple, list, dict, or scalar into a valid TFN (l, m, u)."""
    if isinstance(val, str):
        key = val.strip().upper()
        if key in TFN_LINGUISTIC_SCALE:
            return TFN_LINGUISTIC_SCALE[key]
        raise ValueError(f"Unknown TFN linguistic label: {val}")
    if isinstance(val, dict):
        l_v = float(val.get("l", 1.0))
        m_v = float(val.get("m", l_v))
        u_v = float(val.get("u", m_v))
        return (l_v, m_v, u_v)
    if isinstance(val, (tuple, list)):
        if len(val) != 3:
            raise ValueError(f"TFN tuple/list must have length 3 (l, m, u), got {val}")
        l_v, m_v, u_v = float(val[0]), float(val[1]), float(val[2])
        if not (0.0 < l_v <= m_v <= u_v):
            raise ValueError(f"Invalid TFN ordering (require 0 < l <= m <= u): {val}")
        return (l_v, m_v, u_v)
    if isinstance(val, (int, float)):
        m_v = float(val)
        if abs(m_v - 1.0) < 1e-6:
            return (1.0, 1.0, 1.0)
        return (max(0.5, m_v - 0.5), m_v, m_v + 0.5)
    raise TypeError(f"Unsupported TFN input: {type(val)}")


def gmir_defuzzify(tfn: Tuple[float, float, float]) -> float:
    """
    Graded Mean Integration Representation (GMIR) of a Triangular Fuzzy Number:
        R(l, m, u) = (l + 4*m + u) / 6
    """
    l_v, m_v, u_v = tfn
    return (l_v + 4.0 * m_v + u_v) / 6.0


def compute_fuzzy_bwm_consistency_index(u_bw: float) -> float:
    """
    Computes the Fuzzy BWM Consistency Index (CI) from the upper bound u_BW
    of the Best-to-Worst comparison TFN (Guo & Zhao, 2017 Eq. 9):
        CI^2 - (1 + 2*u_BW)*CI + (u_BW^2 - u_BW) = 0
        => CI = (1 + 2*u_BW + sqrt(8*u_BW + 1)) / 2
    """
    u = max(1.0, float(u_bw))
    return round((1.0 + 2.0 * u + math.sqrt(8.0 * u + 1.0)) / 2.0, 4)


def derive_fuzzy_bwm_weights(
    best_to_others: Sequence[Any],
    others_to_worst: Sequence[Any],
    best_idx: Optional[int] = None,
    worst_idx: Optional[int] = None
) -> Dict[str, Any]:
    """
    Derives optimal criteria weights from Fuzzy Best-to-Others (A_B) and
    Fuzzy Others-to-Worst (A_W) comparison vectors using Triangular Fuzzy Numbers.

    Solves the Guo & Zhao (2017) minimax optimization problem:
        min  ksi
        s.t. |w_B^l - l_Bj * w_j^u| <= ksi * w_j^u
             |w_B^m - m_Bj * w_j^m| <= ksi * w_j^m
             |w_B^u - u_Bj * w_j^l| <= ksi * w_j^l
             |w_j^l - l_jW * w_W^u| <= ksi * w_W^u
             |w_j^m - m_jW * w_W^m| <= ksi * w_W^m
             |w_j^u - u_jW * w_W^l| <= ksi * w_W^l
             sum_j (w_j^l + 4*w_j^m + w_j^u) / 6 = 1
             0 < w_j^l <= w_j^m <= w_j^u   for all j = 1..n

    Returns:
        Dictionary containing:
        - weights: 1D numpy array of normalized crisp GMIR weights summing to 1.0
        - fuzzy_weights: list of (l, m, u) tuples per criterion
        - ksi_star: optimal consistency deviation ksi*
        - consistency_index: CI value corresponding to a_BW
        - consistency_ratio: CR = ksi* / CI
        - best_idx: index of Best criterion
        - worst_idx: index of Worst criterion
    """
    ab_tfns = [parse_tfn(x) for x in best_to_others]
    aw_tfns = [parse_tfn(x) for x in others_to_worst]
    n = len(ab_tfns)
    if n < 2 or len(aw_tfns) != n:
        raise ValueError(f"Best-to-Others ({len(ab_tfns)}) and Others-to-Worst ({len(aw_tfns)}) must have matching length >= 2.")

    # Auto-identify best_idx and worst_idx if not provided
    if best_idx is None:
        best_idx = min(range(n), key=lambda j: abs(ab_tfns[j][0] - 1.0) + abs(ab_tfns[j][1] - 1.0) + abs(ab_tfns[j][2] - 1.0))
    if worst_idx is None:
        worst_idx = min(range(n), key=lambda j: abs(aw_tfns[j][0] - 1.0) + abs(aw_tfns[j][1] - 1.0) + abs(aw_tfns[j][2] - 1.0))

    b = int(best_idx)
    w = int(worst_idx)

    # Analytical initial guess from modal values m_Bj
    inv_m = np.array([1.0 / max(0.2, ab_tfns[j][1]) for j in range(n)], dtype=float)
    w_init_m = inv_m / np.sum(inv_m)

    if HAS_SCIPY:
        # Decision variables x of length 3*n + 1:
        # x[0..n-1]     = w^l
        # x[n..2n-1]    = w^m
        # x[2n..3n-1]   = w^u
        # x[3n]         = ksi
        x0 = np.zeros(3 * n + 1, dtype=float)
        x0[0:n] = np.maximum(0.005, w_init_m * 0.90)
        x0[n:2 * n] = w_init_m
        x0[2 * n:3 * n] = w_init_m * 1.10
        x0[3 * n] = 0.25

        def objective(x: np.ndarray) -> float:
            return float(x[3 * n])

        constraints = []

        # Normalization constraint: sum_j (w_j^l + 4*w_j^m + w_j^u) / 6 == 1.0
        def norm_eq(x: np.ndarray) -> float:
            wl = x[0:n]
            wm = x[n:2 * n]
            wu = x[2 * n:3 * n]
            return float(np.sum((wl + 4.0 * wm + wu) / 6.0) - 1.0)

        constraints.append({"type": "eq", "fun": norm_eq})

        # TFN ordering: w_j^m - w_j^l >= 0 and w_j^u - w_j^m >= 0
        def ordering_ineq(x: np.ndarray) -> np.ndarray:
            wl = x[0:n]
            wm = x[n:2 * n]
            wu = x[2 * n:3 * n]
            return np.concatenate([wm - wl, wu - wm])

        constraints.append({"type": "ineq", "fun": ordering_ineq})

        # Minimax ratio constraints (vectorized)
        ab_l = np.array([t[0] for t in ab_tfns])
        ab_m = np.array([t[1] for t in ab_tfns])
        ab_u = np.array([t[2] for t in ab_tfns])
        aw_l = np.array([t[0] for t in aw_tfns])
        aw_m = np.array([t[1] for t in aw_tfns])
        aw_u = np.array([t[2] for t in aw_tfns])

        def bwm_ineq(x: np.ndarray) -> np.ndarray:
            wl = x[0:n]
            wm = x[n:2 * n]
            wu = x[2 * n:3 * n]
            ksi = x[3 * n]

            # Best-to-Others: |w_B - a_Bj * w_j| <= ksi * w_j
            c1_pos = ksi * wu - (wl[b] - ab_l * wu)
            c1_neg = ksi * wu + (wl[b] - ab_l * wu)
            c2_pos = ksi * wm - (wm[b] - ab_m * wm)
            c2_neg = ksi * wm + (wm[b] - ab_m * wm)
            c3_pos = ksi * wl - (wu[b] - ab_u * wl)
            c3_neg = ksi * wl + (wu[b] - ab_u * wl)

            # Others-to-Worst: |w_j - a_jW * w_W| <= ksi * w_W
            c4_pos = ksi * wu[w] - (wl - aw_l * wu[w])
            c4_neg = ksi * wu[w] + (wl - aw_l * wu[w])
            c5_pos = ksi * wm[w] - (wm - aw_m * wm[w])
            c5_neg = ksi * wm[w] + (wm - aw_m * wm[w])
            c6_pos = ksi * wl[w] - (wu - aw_u * wl[w])
            c6_neg = ksi * wl[w] + (wu - aw_u * wl[w])

            return np.concatenate([
                c1_pos, c1_neg, c2_pos, c2_neg, c3_pos, c3_neg,
                c4_pos, c4_neg, c5_pos, c5_neg, c6_pos, c6_neg
            ])

        constraints.append({"type": "ineq", "fun": bwm_ineq})

        bounds = [(0.005, 1.0)] * (3 * n) + [(0.0, 10.0)]
        res = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 500}
        )

        if res.success:
            wl = res.x[0:n]
            wm = res.x[n:2 * n]
            wu = res.x[2 * n:3 * n]
            ksi_star = max(0.0, float(res.x[3 * n]))
        else:
            wl = w_init_m * 0.9
            wm = w_init_m
            wu = w_init_m * 1.1
            ksi_star = 0.15
    else:
        # Pure-NumPy fallback when scipy is absent
        wl = w_init_m * 0.9
        wm = w_init_m
        wu = w_init_m * 1.1
        ksi_star = 0.12

    fuzzy_weights = [
        (round(float(wl[j]), 5), round(float(wm[j]), 5), round(float(wu[j]), 5))
        for j in range(n)
    ]
    crisp_raw = np.array([gmir_defuzzify((wl[j], wm[j], wu[j])) for j in range(n)], dtype=float)
    crisp_weights = crisp_raw / np.sum(crisp_raw)

    u_bw = max(ab_tfns[w][2], max(t[2] for t in ab_tfns))
    ci = compute_fuzzy_bwm_consistency_index(u_bw)
    cr = round(ksi_star / max(1e-6, ci), 4)

    return {
        "weights": np.round(crisp_weights, 4),
        "fuzzy_weights": fuzzy_weights,
        "ksi_star": round(ksi_star, 4),
        "consistency_index": ci,
        "consistency_ratio": cr,
        "best_idx": b,
        "worst_idx": w
    }


# Named 6-criteria Fuzzy BWM Profiles:
# Criteria order: [0: volume_liters, 1: traffic_pcu, 2: dist_hospital_km, 3: speed_kmh, 4: school_proximity_km, 5: road_class]
NAMED_BWM_PROFILES: Dict[str, Dict[str, Any]] = {
    "safety-first": {
        "name": "safety-first",
        "label": "Safety-First (Physical Cavity & Emergency/School Corridor Triage)",
        "description": "Prioritizes defect structural volume and emergency hospital/school proximity to prevent acute vehicular shock and ambulance delays.",
        "best_idx": 0,   # calculated_volume_liters
        "worst_idx": 5,  # road_class
        "best_to_others": [
            (1.0, 1.0, 1.0),        # [0] calculated_volume_liters (Best: EI)
            (1.5, 2.0, 2.5),        # [1] traffic_pcu (FI)
            (2.0 / 3.0, 1.0, 1.5),  # [2] dist_hospital_km (WI)
            (1.5, 2.0, 2.5),        # [3] speed_kmh (FI)
            (1.5, 2.0, 2.5),        # [4] school_proximity_km (FI)
            (3.5, 4.0, 4.5)         # [5] road_class (Worst: AI)
        ],
        "others_to_worst": [
            (3.5, 4.0, 4.5),        # [0] calculated_volume_liters (AI)
            (1.5, 2.0, 2.5),        # [1] traffic_pcu (FI)
            (2.5, 3.0, 3.5),        # [2] dist_hospital_km (VI)
            (1.5, 2.0, 2.5),        # [3] speed_kmh (FI)
            (1.5, 2.0, 2.5),        # [4] school_proximity_km (FI)
            (1.0, 1.0, 1.0)         # [5] road_class (Worst: EI)
        ]
    },
    "traffic-first": {
        "name": "traffic-first",
        "label": "Traffic-First (Arterial Throughput & Congestion Mitigation)",
        "description": "Prioritizes high-PCU arterial corridors, functional road hierarchy, and operating speed to minimize commuter delay.",
        "best_idx": 1,   # traffic_pcu
        "worst_idx": 4,  # school_proximity_km
        "best_to_others": [
            (2.0 / 3.0, 1.0, 1.5),  # [0] calculated_volume_liters (WI)
            (1.0, 1.0, 1.0),        # [1] traffic_pcu (Best: EI)
            (2.5, 3.0, 3.5),        # [2] dist_hospital_km (VI)
            (1.5, 2.0, 2.5),        # [3] speed_kmh (FI)
            (3.5, 4.0, 4.5),        # [4] school_proximity_km (Worst: AI)
            (2.0 / 3.0, 1.0, 1.5)   # [5] road_class (WI)
        ],
        "others_to_worst": [
            (2.5, 3.0, 3.5),        # [0] calculated_volume_liters (VI)
            (3.5, 4.0, 4.5),        # [1] traffic_pcu (AI)
            (2.0 / 3.0, 1.0, 1.5),  # [2] dist_hospital_km (WI)
            (1.5, 2.0, 2.5),        # [3] speed_kmh (FI)
            (1.0, 1.0, 1.0),        # [4] school_proximity_km (Worst: EI)
            (2.5, 3.0, 3.5)         # [5] road_class (VI)
        ]
    },
    "equity-first": {
        "name": "equity-first",
        "label": "Equity-First (School Zones, Hospitals & Neighborhood Access)",
        "description": "Prioritizes hospital and school zone proximity so vulnerable pedestrian and civic access routes receive timely repair regardless of expressway speed.",
        "best_idx": 2,   # dist_hospital_km
        "worst_idx": 3,  # speed_kmh
        "best_to_others": [
            (1.5, 2.0, 2.5),        # [0] calculated_volume_liters (FI)
            (2.5, 3.0, 3.5),        # [1] traffic_pcu (VI)
            (1.0, 1.0, 1.0),        # [2] dist_hospital_km (Best: EI)
            (3.5, 4.0, 4.5),        # [3] speed_kmh (Worst: AI)
            (2.0 / 3.0, 1.0, 1.5),  # [4] school_proximity_km (WI)
            (2.5, 3.0, 3.5)         # [5] road_class (VI)
        ],
        "others_to_worst": [
            (1.5, 2.0, 2.5),        # [0] calculated_volume_liters (FI)
            (2.0 / 3.0, 1.0, 1.5),  # [1] traffic_pcu (WI)
            (3.5, 4.0, 4.5),        # [2] dist_hospital_km (AI)
            (1.0, 1.0, 1.0),        # [3] speed_kmh (Worst: EI)
            (2.5, 3.0, 3.5),        # [4] school_proximity_km (VI)
            (2.0 / 3.0, 1.0, 1.5)   # [5] road_class (WI)
        ]
    }
}

# Precompute and cache the 3 named Fuzzy BWM profiles for fast runtime lookup
_PROFILE_CACHE: Dict[str, Dict[str, Any]] = {}


def get_fuzzy_bwm_profile(
    profile: str = "safety-first",
    custom_weights: Optional[Sequence[float]] = None,
    best_to_others: Optional[Sequence[Any]] = None,
    others_to_worst: Optional[Sequence[Any]] = None,
    best_idx: Optional[int] = None,
    worst_idx: Optional[int] = None
) -> Dict[str, Any]:
    """
    Returns Fuzzy BWM derived weights and consistency diagnostics for any of the 3 named
    profiles ('safety-first', 'traffic-first', 'equity-first') or 'custom'.
    """
    prof_key = (profile or "safety-first").strip().lower()

    if prof_key == "custom" or custom_weights is not None or (best_to_others is not None and others_to_worst is not None):
        if best_to_others is not None and others_to_worst is not None:
            bwm_res = derive_fuzzy_bwm_weights(
                best_to_others=best_to_others,
                others_to_worst=others_to_worst,
                best_idx=best_idx,
                worst_idx=worst_idx
            )
            return {
                "profile": "custom",
                "label": "Custom Fuzzy BWM Profile",
                "description": "User-supplied Fuzzy BWM Best-to-Others and Others-to-Worst comparison vectors.",
                **bwm_res
            }
        if custom_weights is not None:
            w_arr = np.asarray(custom_weights, dtype=float)
            if w_arr.ndim != 1 or w_arr.size not in (4, 6) or np.any(w_arr < 0) or np.sum(w_arr) <= 0:
                raise ValueError("custom_weights must be a non-negative 1D sequence of length 4 or 6 with positive sum.")
            w_norm = w_arr / np.sum(w_arr)
            return {
                "profile": "custom",
                "label": "Custom Direct Weights",
                "description": "User-supplied direct criteria weights normalized to unit sum.",
                "weights": np.round(w_norm, 4),
                "fuzzy_weights": [(round(float(w), 4), round(float(w), 4), round(float(w), 4)) for w in w_norm],
                "ksi_star": 0.0,
                "consistency_index": 3.0,
                "consistency_ratio": 0.0,
                "best_idx": int(np.argmax(w_norm)),
                "worst_idx": int(np.argmin(w_norm))
            }
        raise ValueError("Profile 'custom' requires either custom_weights or (best_to_others and others_to_worst).")

    if prof_key not in NAMED_BWM_PROFILES:
        raise ValueError(f"Unknown weight profile '{profile}'. Valid profiles: {list(NAMED_BWM_PROFILES.keys()) + ['custom']}")

    if prof_key not in _PROFILE_CACHE:
        spec = NAMED_BWM_PROFILES[prof_key]
        solved = derive_fuzzy_bwm_weights(
            best_to_others=spec["best_to_others"],
            others_to_worst=spec["others_to_worst"],
            best_idx=spec["best_idx"],
            worst_idx=spec["worst_idx"]
        )
        _PROFILE_CACHE[prof_key] = {
            "profile": prof_key,
            "label": spec["label"],
            "description": spec["description"],
            **solved
        }
    return _PROFILE_CACHE[prof_key]


def _spearman_rank_correlation(ranks_a: Sequence[float], ranks_b: Sequence[float]) -> float:
    """Computes Spearman's rank correlation coefficient rho in [-1.0, 1.0]."""
    n = len(ranks_a)
    if n < 2 or len(ranks_b) != n:
        return 1.0
    a = np.asarray(ranks_a, dtype=float)
    b = np.asarray(ranks_b, dtype=float)
    std_a = float(np.std(a))
    std_b = float(np.std(b))
    if std_a < 1e-12 or std_b < 1e-12:
        # Fallback to classical d^2 formula when ranks are identical or constant
        d_sq = float(np.sum((a - b) ** 2))
        rho = 1.0 - (6.0 * d_sq) / (n * (n ** 2 - 1))
        return round(float(np.clip(rho, -1.0, 1.0)), 4)
    # Check if there are any ties; if no ties, classical formula and Pearson on ranks are identical
    if len(set(a.tolist())) == n and len(set(b.tolist())) == n:
        d_sq = float(np.sum((a - b) ** 2))
        rho = 1.0 - (6.0 * d_sq) / (n * (n ** 2 - 1))
    else:
        rho = float(np.corrcoef(a, b)[0, 1])
    return round(float(np.clip(rho, -1.0, 1.0)), 4)


class MCDMPrioritizationEngine:
    """
    Multi-Criteria Decision Making (MCDM) Engine supporting Fuzzy BWM weight derivation,
    TOPSIS relative closeness, VIKOR compromise ranking, and Spearman rank concordance.
    """

    def __init__(
        self,
        weights: Optional[np.ndarray] = None,
        is_benefit: Optional[np.ndarray] = None,
        profile: Optional[str] = None,
        use_calibrated_thresholds: bool = False
    ):
        self.profile = profile
        self.use_calibrated_thresholds = use_calibrated_thresholds
        self.calibrated_thresholds = load_calibrated_thresholds()

        if profile is not None and weights is None:
            prof_data = get_fuzzy_bwm_profile(profile)
            self.weights = np.asarray(prof_data["weights"], dtype=float)
            self.is_benefit = np.asarray(is_benefit, dtype=bool) if is_benefit is not None else EXTENDED_IS_BENEFIT.copy()
            self.bwm_diagnostics = prof_data
        else:
            self.weights = np.asarray(weights, dtype=float) if weights is not None else DEFAULT_WEIGHTS.copy()
            if is_benefit is not None:
                self.is_benefit = np.asarray(is_benefit, dtype=bool)
            elif self.weights.shape[0] == 6:
                self.is_benefit = EXTENDED_IS_BENEFIT.copy()
            else:
                self.is_benefit = DEFAULT_IS_BENEFIT.copy()
            self.bwm_diagnostics = None

    @staticmethod
    def get_criteria_specs(extended: bool = False) -> List[Dict[str, Any]]:
        """Returns metadata for the 4 legacy criteria or all 6 extended criteria."""
        return EXTENDED_CRITERIA_SPECS if extended else CRITERIA_SPECS

    def _resolve_weights_and_directions(
        self,
        num_cols: int,
        weights: Optional[np.ndarray] = None,
        is_benefit: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resolves matching (weights, is_benefit) vectors for either 4-column or 6-column matrices."""
        if weights is not None:
            w = np.asarray(weights, dtype=float)
        elif self.weights.shape[0] == num_cols:
            w = self.weights
        elif num_cols == 6:
            prof_name = self.profile if self.profile in NAMED_BWM_PROFILES else "safety-first"
            w = np.asarray(get_fuzzy_bwm_profile(prof_name)["weights"], dtype=float)
        elif num_cols == 4:
            w = DEFAULT_WEIGHTS.copy()
        else:
            w = np.ones(num_cols, dtype=float) / float(num_cols)

        w_sum = float(np.sum(w))
        if w_sum > 0:
            w = w / w_sum

        if is_benefit is not None:
            b = np.asarray(is_benefit, dtype=bool)
        elif self.is_benefit.shape[0] == num_cols:
            b = self.is_benefit
        elif num_cols == 6:
            b = EXTENDED_IS_BENEFIT.copy()
        elif num_cols == 4:
            b = DEFAULT_IS_BENEFIT.copy()
        else:
            b = np.ones(num_cols, dtype=bool)

        return w, b

    def compute_topsis(
        self,
        matrix: np.ndarray,
        weights: Optional[np.ndarray] = None,
        is_benefit: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Executes L2 vector normalization, weighting, Euclidean distance calculation,
        and relative closeness scoring.

        Args:
            matrix: 2D numpy array of shape (N, 4) or (N, 6).
                    4-col: [volume_liters, traffic_pcu, dist_hospital_km, speed_kmh]
                    6-col: [volume_liters, traffic_pcu, dist_hospital_km, speed_kmh, school_proximity_km, road_class]

        Returns:
            1D numpy array of length N with TOPSIS closeness scores in [0.0, 1.0].
        """
        mat = np.asarray(matrix, dtype=float)
        if mat.size == 0 or mat.shape[0] == 0:
            return np.array([])
        if mat.ndim == 1:
            mat = mat.reshape(1, -1)

        num_criteria = mat.shape[1]
        w, b_flags = self._resolve_weights_and_directions(num_criteria, weights=weights, is_benefit=is_benefit)

        col_norms = np.sqrt((mat ** 2).sum(axis=0))
        safe_norms = np.where(col_norms == 0, 1.0, col_norms)
        norm_matrix = mat / safe_norms

        weighted_matrix = norm_matrix * w
        ideal_best = np.zeros(num_criteria)
        ideal_worst = np.zeros(num_criteria)

        for j in range(num_criteria):
            if b_flags[j]:
                ideal_best[j] = np.max(weighted_matrix[:, j])
                ideal_worst[j] = np.min(weighted_matrix[:, j])
            else:
                ideal_best[j] = np.min(weighted_matrix[:, j])
                ideal_worst[j] = np.max(weighted_matrix[:, j])

        dist_best = np.sqrt(((weighted_matrix - ideal_best) ** 2).sum(axis=1))
        dist_worst = np.sqrt(((weighted_matrix - ideal_worst) ** 2).sum(axis=1))
        total_dist = dist_best + dist_worst

        # If all alternatives are identical, total_dist is zero -> neutral score (0.5)
        scores = np.where(total_dist < 1e-9, 0.5, dist_worst / (total_dist + 1e-9))
        scores = np.clip(scores, 0.0, 1.0)
        return np.round(scores, 4)

    def compute_vikor(
        self,
        matrix: np.ndarray,
        v: float = 0.5,
        weights: Optional[np.ndarray] = None,
        is_benefit: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Executes VIKOR compromise ranking (Opricovic & Tzeng, 2004):
            S_i = sum_j w_j * (|f_j* - f_ij| / |f_j* - f_j-|)
            R_i = max_j [ w_j * (|f_j* - f_ij| / |f_j* - f_j-|) ]
            Q_i = v * (S_i - S*) / (S- - S*) + (1 - v) * (R_i - R*) / (R- - R*)

        Note: Lower S_i, R_i, and Q_i indicate higher repair priority (closer to ideal f*).

        Returns:
            Dictionary with S, R, Q arrays, 1-based ranks, acceptable_advantage,
            acceptable_stability, DQ, q_gap, and compromise_solution indices.
        """
        mat = np.asarray(matrix, dtype=float)
        if mat.size == 0 or mat.shape[0] == 0:
            return {
                "S": np.array([]),
                "R": np.array([]),
                "Q": np.array([]),
                "urgency_scores": np.array([]),
                "ranks": [],
                "ranking_order": [],
                "v": float(v),
                "DQ": 1.0,
                "q_gap": 0.0,
                "acceptable_advantage": False,
                "acceptable_stability": False,
                "compromise_solution": []
            }
        if mat.ndim == 1:
            mat = mat.reshape(1, -1)

        n_alt, n_crit = mat.shape
        w, b_flags = self._resolve_weights_and_directions(n_crit, weights=weights, is_benefit=is_benefit)
        v_clamped = float(np.clip(v, 0.0, 1.0))

        # Normalized linear distance d_ij in [0, 1]
        d_mat = np.zeros((n_alt, n_crit), dtype=float)
        for j in range(n_crit):
            col = mat[:, j]
            if b_flags[j]:
                f_star = float(np.max(col))
                f_minus = float(np.min(col))
                denom = f_star - f_minus
                if abs(denom) > 1e-12:
                    d_mat[:, j] = (f_star - col) / denom
            else:
                f_star = float(np.min(col))
                f_minus = float(np.max(col))
                denom = f_minus - f_star
                if abs(denom) > 1e-12:
                    d_mat[:, j] = (col - f_star) / denom

        weighted_d = d_mat * w[None, :]
        S = np.sum(weighted_d, axis=1)
        R = np.max(weighted_d, axis=1)

        s_star, s_minus = float(np.min(S)), float(np.max(S))
        r_star, r_minus = float(np.min(R)), float(np.max(R))

        s_denom = s_minus - s_star
        r_denom = r_minus - r_star

        s_term = (S - s_star) / s_denom if s_denom > 1e-12 else np.zeros(n_alt, dtype=float)
        r_term = (R - r_star) / r_denom if r_denom > 1e-12 else np.zeros(n_alt, dtype=float)

        if s_denom <= 1e-12 and r_denom <= 1e-12:
            Q = np.full(n_alt, 0.5, dtype=float)
        else:
            Q = v_clamped * s_term + (1.0 - v_clamped) * r_term
            Q = np.clip(Q, 0.0, 1.0)

        S_round = np.round(S, 4)
        R_round = np.round(R, 4)
        Q_round = np.round(Q, 4)

        # Sort alternatives by ascending Q_i (tie-breaker: ascending S_i, then ascending R_i)
        ranking_order = sorted(range(n_alt), key=lambda i: (float(Q_round[i]), float(S_round[i]), float(R_round[i])))
        ranks = [0] * n_alt
        for pos, idx in enumerate(ranking_order):
            ranks[idx] = pos + 1

        dq = round(1.0 / float(max(1, n_alt - 1)), 4) if n_alt > 1 else 1.0
        if n_alt >= 2:
            a1, a2 = ranking_order[0], ranking_order[1]
            q_gap = round(float(Q[a2] - Q[a1]), 4)
            acceptable_advantage = bool((Q[a2] - Q[a1]) >= (1.0 / float(n_alt - 1)) - 1e-9)
        else:
            a1 = ranking_order[0]
            q_gap = 0.0
            acceptable_advantage = True

        # Acceptable stability: A^(1) must also be best ranked by S or/and R
        best_s_idx = int(np.argmin(S))
        best_r_idx = int(np.argmin(R))
        stability_by_S = bool(abs(S[a1] - S[best_s_idx]) <= 1e-9)
        stability_by_R = bool(abs(R[a1] - R[best_r_idx]) <= 1e-9)
        acceptable_stability = bool(stability_by_S or stability_by_R)

        # Determine VIKOR compromise solution set
        if not acceptable_advantage and n_alt >= 2:
            compromise_solution = [
                idx for idx in ranking_order
                if (Q[idx] - Q[a1]) < (1.0 / float(n_alt - 1)) - 1e-9
            ]
            if len(compromise_solution) < 2:
                compromise_solution = [ranking_order[0], ranking_order[1]]
        elif not acceptable_stability and n_alt >= 2:
            compromise_solution = [ranking_order[0], ranking_order[1]]
        else:
            compromise_solution = [ranking_order[0]]

        return {
            "S": S_round,
            "R": R_round,
            "Q": Q_round,
            "urgency_scores": np.round(1.0 - Q_round, 4),
            "ranks": ranks,
            "ranking_order": ranking_order,
            "v": v_clamped,
            "DQ": dq,
            "q_gap": q_gap,
            "acceptable_advantage": acceptable_advantage,
            "acceptable_stability": acceptable_stability,
            "stability_by_S": stability_by_S,
            "stability_by_R": stability_by_R,
            "compromise_solution": compromise_solution
        }

    def compute_all(
        self,
        matrix: np.ndarray,
        v: float = 0.5,
        weights: Optional[np.ndarray] = None,
        is_benefit: Optional[np.ndarray] = None,
        calibrated: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Executes both TOPSIS and VIKOR on the decision matrix and returns:
        - topsis_scores, topsis_ranks
        - vikor (S, R, Q, ranks, acceptable_advantage, acceptable_stability, compromise_solution)
        - vikor_ranks
        - spearman_correlation (and spearman_rho) between TOPSIS ranks and VIKOR ranks
        - priority_levels
        """
        mat = np.asarray(matrix, dtype=float)
        if mat.ndim == 1 and mat.size > 0:
            mat = mat.reshape(1, -1)
        n = mat.shape[0] if mat.size > 0 else 0
        if n == 0:
            return {
                "topsis_scores": np.array([]),
                "topsis_ranks": [],
                "vikor_S": np.array([]),
                "vikor_R": np.array([]),
                "vikor_Q": np.array([]),
                "vikor_ranks": [],
                "vikor": self.compute_vikor(mat, v=v),
                "spearman_correlation": 1.0,
                "spearman_rho": 1.0,
                "priority_levels": [],
                "weights": []
            }

        w, b_flags = self._resolve_weights_and_directions(mat.shape[1], weights=weights, is_benefit=is_benefit)
        topsis_scores = self.compute_topsis(mat, weights=w, is_benefit=b_flags)

        # 1-based TOPSIS ranks (descending score)
        topsis_order = sorted(range(n), key=lambda i: float(topsis_scores[i]), reverse=True)
        topsis_ranks = [0] * n
        for pos, idx in enumerate(topsis_order):
            topsis_ranks[idx] = pos + 1

        vikor_res = self.compute_vikor(mat, v=v, weights=w, is_benefit=b_flags)
        vikor_ranks = vikor_res["ranks"]

        rho = _spearman_rank_correlation(topsis_ranks, vikor_ranks)
        use_cal = self.use_calibrated_thresholds if calibrated is None else bool(calibrated)
        priorities = [self.classify_priority(float(s), calibrated=use_cal) for s in topsis_scores]

        return {
            "topsis_scores": topsis_scores,
            "topsis_ranks": topsis_ranks,
            "vikor_S": vikor_res["S"],
            "vikor_R": vikor_res["R"],
            "vikor_Q": vikor_res["Q"],
            "vikor_ranks": vikor_ranks,
            "vikor": vikor_res,
            "spearman_correlation": rho,
            "spearman_rho": rho,
            "priority_levels": priorities,
            "weights": np.round(w, 4).tolist()
        }

    def classify_priority(
        self,
        topsis_score: float,
        calibrated: Optional[bool] = None,
        thresholds: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Categorizes TOPSIS closeness score into municipal triage level.
        Supports both legacy quartile thresholds (0.75/0.50/0.25) and calibrated
        multi-criteria thresholds (0.65/0.45/0.25 from config/sensor_profiles.yaml).
        """
        if thresholds is not None:
            th = thresholds
        elif calibrated is True or (calibrated is None and self.use_calibrated_thresholds):
            th = self.calibrated_thresholds
        else:
            th = LEGACY_THRESHOLDS

        if topsis_score >= th["critical"]:
            return "Critical"
        elif topsis_score >= th["high"]:
            return "High"
        elif topsis_score >= th["medium"]:
            return "Medium"
        return "Low"
