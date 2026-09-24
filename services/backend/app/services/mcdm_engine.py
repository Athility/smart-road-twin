"""
Multi-Criteria Decision Making (MCDM) TOPSIS Engine
==================================================

Evaluates road defect repair urgency using the Technique for Order of
Preference by Similarity to Ideal Solution (TOPSIS).

Criteria Definition & Mathematical Direction:
---------------------------------------------
1. Defect Volume (Liters):
   - Weight: 0.35
   - Direction: BENEFIT (is_benefit=True)
   - Rationale: Larger cavity volume produces greater structural failure and vertical wheel shock.

2. Traffic Volume (PCU):
   - Weight: 0.25
   - Direction: BENEFIT (is_benefit=True)
   - Rationale: Higher traffic density multiplies vehicle encounter frequency and catastrophic risk.

3. Hospital Distance (km):
   - Weight: 0.25
   - Direction: COST (is_benefit=False)
   - Rationale: Critical emergency corridor proximity. Smaller distance to hospital increases urgency.

4. Operating Vehicle Speed (km/h):
   - Weight: 0.15
   - Direction: BENEFIT (is_benefit=True)
   - Rationale: Higher velocity exponentially amplifies dynamic impact energy (E = 0.5 * m * v^2).

Total Weight Normalization:
   0.35 + 0.25 + 0.25 + 0.15 = 1.00
"""

import numpy as np
from typing import List, Dict, Any, Optional

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
        "unit": "PCU",
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

DEFAULT_WEIGHTS = np.array([0.35, 0.25, 0.25, 0.15], dtype=float)
DEFAULT_IS_BENEFIT = np.array([True, True, False, True], dtype=bool)


class MCDMPrioritizationEngine:
    """
    Standard TOPSIS Multi-Criteria Prioritization Engine.
    Computes Euclidean distances to positive and negative ideal solutions.
    """

    def __init__(self, weights: Optional[np.ndarray] = None, is_benefit: Optional[np.ndarray] = None):
        self.weights = np.asarray(weights, dtype=float) if weights is not None else DEFAULT_WEIGHTS.copy()
        self.is_benefit = np.asarray(is_benefit, dtype=bool) if is_benefit is not None else DEFAULT_IS_BENEFIT.copy()

    @staticmethod
    def get_criteria_specs() -> List[Dict[str, Any]]:
        """Returns metadata for all 4 TOPSIS criteria."""
        return CRITERIA_SPECS

    def compute_topsis(self, matrix: np.ndarray) -> np.ndarray:
        """
        Executes vector normalization, weighting, distance calculation,
        and relative closeness scoring.
        
        Args:
            matrix: 2D numpy array of shape (N, 4), where columns are:
                    [volume_liters, traffic_pcu, dist_hospital_km, speed_kmh]
                    
        Returns:
            1D numpy array of length N with TOPSIS closeness scores in [0.0, 1.0].
        """
        if matrix.size == 0 or matrix.shape[0] == 0:
            return np.array([])
        mat = np.asarray(matrix, dtype=float)
        col_norms = np.sqrt((mat**2).sum(axis=0))
        # Prevent division by zero if all values in a column are zero
        safe_norms = np.where(col_norms == 0, 1.0, col_norms)
        norm_matrix = mat / safe_norms

        weighted_matrix = norm_matrix * self.weights
        num_criteria = self.weights.shape[0]
        ideal_best = np.zeros(num_criteria)
        ideal_worst = np.zeros(num_criteria)

        for j in range(num_criteria):
            if self.is_benefit[j]:
                ideal_best[j] = np.max(weighted_matrix[:, j])
                ideal_worst[j] = np.min(weighted_matrix[:, j])
            else:
                ideal_best[j] = np.min(weighted_matrix[:, j])
                ideal_worst[j] = np.max(weighted_matrix[:, j])

        dist_best = np.sqrt(((weighted_matrix - ideal_best)**2).sum(axis=1))
        dist_worst = np.sqrt(((weighted_matrix - ideal_worst)**2).sum(axis=1))
        total_dist = dist_best + dist_worst

        # If all alternatives are identical, total_dist is zero -> neutral score (0.5)
        scores = np.where(total_dist < 1e-9, 0.5, dist_worst / (total_dist + 1e-9))
        scores = np.clip(scores, 0.0, 1.0)
        return np.round(scores, 4)

    @staticmethod
    def classify_priority(topsis_score: float) -> str:
        """Categorizes raw TOPSIS score into municipal triage level."""
        if topsis_score >= 0.75:
            return "Critical"
        elif topsis_score >= 0.50:
            return "High"
        elif topsis_score >= 0.25:
            return "Medium"
        return "Low"
