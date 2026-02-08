"""
Adaptive Prediction Sets (APS).

An alternative to FCS that produces smaller prediction sets on average
while maintaining coverage guarantee.

Key difference: APS uses randomization to achieve exact coverage,
while FCS tends to overcover.
"""

from typing import List, Set, Tuple, Optional
from dataclasses import dataclass
import numpy as np

from .nonconformity import NonconformityScore, MarginScore
from .fcs import ConformalPrediction, ConformalMetrics


class AdaptivePredictionSets:
    """
    Adaptive Prediction Sets for more efficient conformal prediction.

    Uses randomization for tighter coverage and smaller sets.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        randomize: bool = True
    ):
        """
        Initialize APS predictor.

        Args:
            alpha: Miscoverage rate
            randomize: Whether to use randomization (recommended)
        """
        self.alpha = alpha
        self.randomize = randomize
        self.cal_scores = None
        self.threshold = None
        self.is_calibrated = False

    def _compute_aps_score(
        self,
        probability: float,
        target_label: int
    ) -> float:
        """
        Compute APS score for a target label.

        For binary: sum of probabilities of classes with higher prob than target.

        Args:
            probability: Prob of class 1
            target_label: Label to compute score for

        Returns:
            APS score
        """
        if target_label == 1:
            target_prob = probability
            other_prob = 1 - probability
        else:
            target_prob = 1 - probability
            other_prob = probability

        # Score = sum of probs for classes ranked higher + U * target_prob
        if self.randomize:
            u = np.random.uniform(0, 1)
        else:
            u = 1.0

        if other_prob > target_prob:
            # Other class ranked higher
            score = other_prob + u * target_prob
        else:
            # Target class ranked highest
            score = u * target_prob

        return score

    def calibrate(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray
    ):
        """
        Calibrate APS on held-out data.

        Args:
            probabilities: [N] predicted probabilities for class 1
            labels: [N] true labels
        """
        self.cal_scores = np.array([
            self._compute_aps_score(p, int(y))
            for p, y in zip(probabilities, labels)
        ])

        # Use finite-sample correction
        n = len(self.cal_scores)
        q = min((n + 1) * (1 - self.alpha) / n, 1.0)
        self.threshold = float(np.quantile(self.cal_scores, q))
        self.is_calibrated = True

    def predict(self, probability: float) -> ConformalPrediction:
        """
        Make APS prediction.

        Args:
            probability: Predicted probability for class 1

        Returns:
            ConformalPrediction
        """
        if not self.is_calibrated:
            raise RuntimeError("Must calibrate before prediction")

        # Compute scores for each label
        score_0 = self._compute_aps_score(probability, 0)
        score_1 = self._compute_aps_score(probability, 1)

        # Include if score <= threshold
        prediction_set = set()
        if score_0 <= self.threshold:
            prediction_set.add(0)
        if score_1 <= self.threshold:
            prediction_set.add(1)

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
        """Make APS predictions for multiple samples."""
        return [self.predict(p) for p in probabilities]

    def evaluate(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        n_trials: int = 1
    ) -> ConformalMetrics:
        """
        Evaluate APS predictor.

        Args:
            probabilities: [N] predicted probabilities
            labels: [N] true labels
            n_trials: Number of random trials (for randomized APS)

        Returns:
            ConformalMetrics
        """
        all_coverages = []
        all_sizes = []
        all_singletons = []
        all_empties = []

        for _ in range(n_trials):
            predictions = self.predict_batch(probabilities)

            covered = sum(
                1 for pred, y in zip(predictions, labels)
                if int(y) in pred.prediction_set
            )
            coverage = covered / len(labels)
            all_coverages.append(coverage)

            sizes = [p.set_size for p in predictions]
            all_sizes.extend(sizes)

            singleton = sum(1 for p in predictions if p.set_size == 1)
            all_singletons.append(singleton / len(predictions))

            empty = sum(1 for p in predictions if p.set_size == 0)
            all_empties.append(empty / len(predictions))

        return ConformalMetrics(
            empirical_coverage=np.mean(all_coverages),
            average_set_size=np.mean(all_sizes),
            singleton_rate=np.mean(all_singletons),
            empty_rate=np.mean(all_empties),
            n_samples=len(labels)
        )


def compare_fcs_aps(
    probabilities: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.1,
    cal_fraction: float = 0.5
) -> dict:
    """
    Compare FCS and APS on the same data.

    Args:
        probabilities: [N] predicted probabilities
        labels: [N] true labels
        alpha: Miscoverage rate
        cal_fraction: Fraction for calibration

    Returns:
        Comparison results
    """
    from .fcs import FullConformalPredictor

    # Split data
    n = len(probabilities)
    n_cal = int(n * cal_fraction)
    indices = np.random.permutation(n)
    cal_idx = indices[:n_cal]
    test_idx = indices[n_cal:]

    cal_probs = probabilities[cal_idx]
    cal_labels = labels[cal_idx]
    test_probs = probabilities[test_idx]
    test_labels = labels[test_idx]

    # FCS
    fcs = FullConformalPredictor(alpha=alpha)
    fcs.calibrate(cal_probs, cal_labels)
    fcs_metrics = fcs.evaluate(test_probs, test_labels)

    # APS
    aps = AdaptivePredictionSets(alpha=alpha, randomize=True)
    aps.calibrate(cal_probs, cal_labels)
    aps_metrics = aps.evaluate(test_probs, test_labels, n_trials=10)

    return {
        'alpha': alpha,
        'target_coverage': 1 - alpha,
        'fcs': {
            'coverage': fcs_metrics.empirical_coverage,
            'avg_set_size': fcs_metrics.average_set_size,
            'singleton_rate': fcs_metrics.singleton_rate,
            'threshold': fcs.threshold
        },
        'aps': {
            'coverage': aps_metrics.empirical_coverage,
            'avg_set_size': aps_metrics.average_set_size,
            'singleton_rate': aps_metrics.singleton_rate,
            'threshold': aps.threshold
        }
    }
