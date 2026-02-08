"""Conformal Prediction for TrustPPI.

Provides statistical coverage guarantees for PPI predictions.
"""

from .fcs import FullConformalPredictor, coverage_vs_alpha
from .aps import AdaptivePredictionSets, compare_fcs_aps
from .nonconformity import (
    NonconformityScore,
    MarginScore,
    ProbabilityScore,
    compute_nonconformity
)

__all__ = [
    'FullConformalPredictor',
    'AdaptivePredictionSets',
    'NonconformityScore',
    'MarginScore',
    'ProbabilityScore',
    'compute_nonconformity',
    'coverage_vs_alpha',
    'compare_fcs_aps'
]
