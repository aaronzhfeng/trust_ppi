"""
Nonconformity Score Functions for Conformal Prediction.

Nonconformity measures how "strange" a data point is relative to training data.
Lower scores = more conforming = more likely to be in prediction set.
"""

from abc import ABC, abstractmethod
from typing import Union
import numpy as np


class NonconformityScore(ABC):
    """Abstract base class for nonconformity scores."""

    @abstractmethod
    def __call__(
        self,
        probability: float,
        true_label: int
    ) -> float:
        """
        Compute nonconformity score.

        Args:
            probability: Predicted probability for class 1
            true_label: True class label (0 or 1)

        Returns:
            Nonconformity score (lower = more conforming)
        """
        pass


class MarginScore(NonconformityScore):
    """
    Margin-based nonconformity score.

    score = 1 - P(true_class)

    For binary classification:
    - If true_label=1: score = 1 - probability
    - If true_label=0: score = probability
    """

    def __call__(self, probability: float, true_label: int) -> float:
        if true_label == 1:
            return 1.0 - probability
        else:
            return probability


class ProbabilityScore(NonconformityScore):
    """
    Probability-based nonconformity score.

    score = -P(true_class)

    Simpler than margin but can be less calibrated.
    """

    def __call__(self, probability: float, true_label: int) -> float:
        if true_label == 1:
            return -probability
        else:
            return -(1.0 - probability)


class HingeLossScore(NonconformityScore):
    """
    Hinge loss-based nonconformity score.

    score = max(0, 1 - margin)
    where margin = P(true) - P(other)

    For binary: margin = P(1) - P(0) if true=1, else P(0) - P(1)
    """

    def __call__(self, probability: float, true_label: int) -> float:
        if true_label == 1:
            margin = probability - (1.0 - probability)  # P(1) - P(0)
        else:
            margin = (1.0 - probability) - probability  # P(0) - P(1)
        return max(0.0, 1.0 - margin)


def compute_nonconformity(
    probabilities: np.ndarray,
    labels: np.ndarray,
    score_fn: Union[NonconformityScore, str] = 'margin'
) -> np.ndarray:
    """
    Compute nonconformity scores for multiple samples.

    Args:
        probabilities: [N] array of predicted probabilities for class 1
        labels: [N] array of true labels (0 or 1)
        score_fn: Nonconformity score function or name ('margin', 'probability', 'hinge')

    Returns:
        [N] array of nonconformity scores
    """
    if isinstance(score_fn, str):
        score_fn = {
            'margin': MarginScore(),
            'probability': ProbabilityScore(),
            'hinge': HingeLossScore()
        }[score_fn]

    scores = np.array([
        score_fn(p, int(y)) for p, y in zip(probabilities, labels)
    ])
    return scores


def compute_quantile_threshold(
    scores: np.ndarray,
    alpha: float
) -> float:
    """
    Compute (1-alpha) quantile of calibration scores.

    For valid coverage guarantee, use:
    threshold = quantile(scores, (n+1)*(1-alpha)/n)

    Args:
        scores: [N] calibration scores
        alpha: Miscoverage rate (e.g., 0.1 for 90% coverage)

    Returns:
        Threshold for prediction sets
    """
    n = len(scores)
    # Finite-sample correction for valid coverage
    adjusted_quantile = min((n + 1) * (1 - alpha) / n, 1.0)
    return float(np.quantile(scores, adjusted_quantile))
