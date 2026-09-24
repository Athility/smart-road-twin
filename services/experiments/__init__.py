"""
Sensor Fusion Experiments Module
================================

Exports:
- FusionExperimentRunner
- compute_spearman_rank_correlation
"""

from .experiment_runner import FusionExperimentRunner, compute_spearman_rank_correlation

__all__ = [
    "FusionExperimentRunner",
    "compute_spearman_rank_correlation"
]
