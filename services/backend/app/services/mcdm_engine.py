import numpy as np

class MCDMPrioritizationEngine:
    def __init__(self, weights=None):
        self.weights = weights if weights is not None else np.array([0.35, 0.30, 0.20, 0.15])
        self.is_benefit = np.array([True, True, False, True])

    def compute_topsis(self, matrix: np.ndarray) -> np.ndarray:
        if matrix.size == 0 or matrix.shape[0] == 0:
            return np.array([])
        mat = np.asarray(matrix, dtype=float)
        col_norms = np.sqrt((mat**2).sum(axis=0))
        # Prevent division by zero if all values in a column are zero
        safe_norms = np.where(col_norms == 0, 1.0, col_norms)
        norm_matrix = mat / safe_norms

        weighted_matrix = norm_matrix * self.weights
        ideal_best = np.zeros(4)
        ideal_worst = np.zeros(4)
        for j in range(4):
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
        if topsis_score >= 0.75: return "Critical"
        elif topsis_score >= 0.50: return "High"
        elif topsis_score >= 0.25: return "Medium"
        return "Low"
