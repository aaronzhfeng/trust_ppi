"""
Full Conformal Set (FCS) Prediction.

Provides prediction sets with guaranteed coverage:
P(Y ∈ C(X)) ≥ 1 - α

For binary classification, prediction sets can be:
- {0}: Predict negative with confidence
- {1}: Predict positive with confidence
- {0, 1}: Uncertain, includes both classes
- {}: Invalid (shouldn't happen with proper calibration)
"""

from typing import List, Set, Tuple, Optional, Dict
from dataclasses import dataclass
import numpy as np

from .nonconformity import (
    NonconformityScore,
    MarginScore,
    compute_nonconformity,
    compute_quantile_threshold
)


@dataclass
class ConformalPrediction:
    """Result of conformal prediction."""
    prediction_set: Set[int]      # Set of plausible labels
    point_prediction: int         # Single best prediction
    probability: float            # Predicted probability for class 1
    set_size: int                 # Size of prediction set
    threshold: float              # Calibration threshold used


@dataclass
class ConformalMetrics:
    """Evaluation metrics for conformal predictor."""
    empirical_coverage: float     # Fraction where true label in set
    average_set_size: float       # Average prediction set size
    singleton_rate: float         # Fraction of single-class predictions
    empty_rate: float             # Fraction of empty sets (should be ~0)
    n_samples: int


class FullConformalPredictor:
    """
    Full Conformal Prediction for binary classification.

    Provides finite-sample coverage guarantees.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        score_fn: Optional[NonconformityScore] = None
    ):
        """
        Initialize conformal predictor.

        Args:
            alpha: Miscoverage rate (e.g., 0.1 for 90% coverage target)
            score_fn: Nonconformity score function (default: MarginScore)
        """
        self.alpha = alpha
        self.score_fn = score_fn or MarginScore()
        self.threshold = None
        self.cal_scores = None
        self.is_calibrated = False

    def calibrate(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray
    ):
        """
        Calibrate conformal predictor on held-out data.

        Args:
            probabilities: [N] predicted probabilities for class 1
            labels: [N] true labels (0 or 1)
        """
        self.cal_scores = compute_nonconformity(
            probabilities, labels, self.score_fn
        )
        self.threshold = compute_quantile_threshold(
            self.cal_scores, self.alpha
        )
        self.is_calibrated = True

    def predict(self, probability: float) -> ConformalPrediction:
        """
        Make conformal prediction for a single sample.

        Args:
            probability: Predicted probability for class 1

        Returns:
            ConformalPrediction with prediction set
        """
        if not self.is_calibrated:
            raise RuntimeError("Must calibrate before prediction")

        # Compute scores for each possible label
        score_0 = self.score_fn(probability, 0)
        score_1 = self.score_fn(probability, 1)

        # Include in set if score <= threshold
        prediction_set = set()
        if score_0 <= self.threshold:
            prediction_set.add(0)
        if score_1 <= self.threshold:
            prediction_set.add(1)

        # Point prediction (highest probability class)
        point_prediction = 1 if probability >= 0.5 else 0

        return ConformalPrediction(
            prediction_set=prediction_set,
            point_prediction=point_prediction,
            probability=probability,
            set_size=len(prediction_set),
            threshold=self.threshold
        )

    def predict_batch(
        self,
        probabilities: np.ndarray
    ) -> List[ConformalPrediction]:
        """
        Make conformal predictions for multiple samples.

        Args:
            probabilities: [N] predicted probabilities

        Returns:
            List of ConformalPrediction objects
        """
        return [self.predict(p) for p in probabilities]

    def evaluate(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray
    ) -> ConformalMetrics:
        """
        Evaluate conformal predictor on test data.

        Args:
            probabilities: [N] predicted probabilities
            labels: [N] true labels

        Returns:
            ConformalMetrics with coverage, set size, etc.
        """
        predictions = self.predict_batch(probabilities)

        # Coverage: fraction where true label in prediction set
        covered = sum(
            1 for pred, y in zip(predictions, labels)
            if int(y) in pred.prediction_set
        )
        coverage = covered / len(labels)

        # Average set size
        avg_size = np.mean([p.set_size for p in predictions])

        # Singleton rate (informative predictions)
        singleton = sum(1 for p in predictions if p.set_size == 1)
        singleton_rate = singleton / len(predictions)

        # Empty rate (should be ~0)
        empty = sum(1 for p in predictions if p.set_size == 0)
        empty_rate = empty / len(predictions)

        return ConformalMetrics(
            empirical_coverage=coverage,
            average_set_size=avg_size,
            singleton_rate=singleton_rate,
            empty_rate=empty_rate,
            n_samples=len(labels)
        )


def coverage_vs_alpha(
    probabilities: np.ndarray,
    labels: np.ndarray,
    alphas: Optional[List[float]] = None,
    cal_fraction: float = 0.5
) -> List[Dict]:
    """
    Evaluate coverage at multiple alpha levels.

    Args:
        probabilities: [N] predicted probabilities
        labels: [N] true labels
        alphas: List of miscoverage rates to test
        cal_fraction: Fraction of data for calibration

    Returns:
        List of results for each alpha
    """
    if alphas is None:
        alphas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    # Split into calibration and test
    n = len(probabilities)
    n_cal = int(n * cal_fraction)
    indices = np.random.permutation(n)
    cal_idx = indices[:n_cal]
    test_idx = indices[n_cal:]

    cal_probs = probabilities[cal_idx]
    cal_labels = labels[cal_idx]
    test_probs = probabilities[test_idx]
    test_labels = labels[test_idx]

    results = []
    for alpha in alphas:
        predictor = FullConformalPredictor(alpha=alpha)
        predictor.calibrate(cal_probs, cal_labels)
        metrics = predictor.evaluate(test_probs, test_labels)

        results.append({
            'alpha': alpha,
            'target_coverage': 1 - alpha,
            'empirical_coverage': metrics.empirical_coverage,
            'coverage_gap': metrics.empirical_coverage - (1 - alpha),
            'average_set_size': metrics.average_set_size,
            'singleton_rate': metrics.singleton_rate,
            'threshold': predictor.threshold
        })

    return results
