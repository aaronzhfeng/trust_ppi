"""
Trust Metrics for TrustPPI.

Helper functions for computing coverage, accuracy, and selective prediction metrics.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class SelectiveMetrics:
    """Metrics for selective prediction at a given coverage level."""
    coverage: float           # Fraction of samples accepted
    accuracy: float           # Accuracy on accepted samples
    precision: float          # Precision on accepted samples
    recall: float             # Recall on accepted samples
    f1: float                 # F1 score on accepted samples
    n_accepted: int           # Number of accepted samples
    n_total: int              # Total number of samples
    improvement: float        # Accuracy improvement over full coverage


def compute_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Compute accuracy."""
    return float(np.mean(predictions == labels))


def compute_precision_recall_f1(
    predictions: np.ndarray,
    labels: np.ndarray
) -> Tuple[float, float, float]:
    """Compute precision, recall, and F1 score."""
    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def compute_ece(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 15
) -> float:
    """
    Compute Expected Calibration Error.

    Args:
        confidences: [N] model confidence scores
        accuracies: [N] binary correct/incorrect indicators
        n_bins: Number of bins for calibration

    Returns:
        ECE value (lower is better)
    """
    bin_boundaries = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    total = len(confidences)

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.sum() / total

        if in_bin.sum() > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            ece += prop_in_bin * abs(avg_confidence - avg_accuracy)

    return ece


def compute_brier_score(
    probabilities: np.ndarray,
    labels: np.ndarray
) -> float:
    """
    Compute Brier score (mean squared error of probabilities).

    Lower is better. Perfect = 0.0.
    """
    return float(np.mean((probabilities - labels) ** 2))


def selective_accuracy_at_coverage(
    trust_scores: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    target_coverage: float
) -> SelectiveMetrics:
    """
    Compute accuracy when selecting top-k% by trust score.

    Args:
        trust_scores: [N] trust scores (higher = more trustworthy)
        predictions: [N] binary predictions
        labels: [N] ground truth labels
        target_coverage: Fraction of samples to accept (0.0 to 1.0)

    Returns:
        SelectiveMetrics with all computed metrics
    """
    n_total = len(trust_scores)
    n_accept = max(1, int(n_total * target_coverage))

    # Select top samples by trust
    top_indices = np.argsort(trust_scores)[-n_accept:]

    selected_preds = predictions[top_indices]
    selected_labels = labels[top_indices]

    # Compute metrics on selected subset
    accuracy = compute_accuracy(selected_preds, selected_labels)
    precision, recall, f1 = compute_precision_recall_f1(selected_preds, selected_labels)

    # Compute full coverage accuracy for comparison
    full_accuracy = compute_accuracy(predictions, labels)
    improvement = accuracy - full_accuracy

    return SelectiveMetrics(
        coverage=n_accept / n_total,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        n_accepted=n_accept,
        n_total=n_total,
        improvement=improvement
    )


def coverage_accuracy_curve(
    trust_scores: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    coverage_levels: Optional[List[float]] = None
) -> List[SelectiveMetrics]:
    """
    Compute accuracy at multiple coverage levels.

    Args:
        trust_scores: [N] trust scores
        predictions: [N] predictions
        labels: [N] labels
        coverage_levels: List of coverage fractions (default: 0.5 to 1.0)

    Returns:
        List of SelectiveMetrics, one per coverage level
    """
    if coverage_levels is None:
        coverage_levels = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    results = []
    for coverage in coverage_levels:
        metrics = selective_accuracy_at_coverage(
            trust_scores, predictions, labels, coverage
        )
        results.append(metrics)

    return results


def compute_auc_coverage_accuracy(
    trust_scores: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    n_points: int = 20
) -> float:
    """
    Compute area under the coverage-accuracy curve.

    Higher is better. Maximum = 1.0 if perfect selection.

    Args:
        trust_scores: [N] trust scores
        predictions: [N] predictions
        labels: [N] labels
        n_points: Number of coverage points to evaluate

    Returns:
        AUC value
    """
    coverages = np.linspace(0.1, 1.0, n_points)
    accuracies = []

    for cov in coverages:
        metrics = selective_accuracy_at_coverage(
            trust_scores, predictions, labels, cov
        )
        accuracies.append(metrics.accuracy)

    # Compute AUC using trapezoidal rule
    auc = np.trapz(accuracies, coverages) / (1.0 - 0.1)  # Normalize to [0, 1]
    return float(auc)


def risk_at_coverage(
    trust_scores: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    target_coverage: float
) -> float:
    """
    Compute risk (error rate) at a given coverage level.

    Risk = 1 - Accuracy

    Args:
        trust_scores: [N] trust scores
        predictions: [N] predictions
        labels: [N] labels
        target_coverage: Coverage level

    Returns:
        Risk (error rate) on accepted samples
    """
    metrics = selective_accuracy_at_coverage(
        trust_scores, predictions, labels, target_coverage
    )
    return 1.0 - metrics.accuracy


def optimal_threshold_for_coverage(
    trust_scores: np.ndarray,
    target_coverage: float
) -> float:
    """
    Find the trust threshold that achieves target coverage.

    Args:
        trust_scores: [N] trust scores
        target_coverage: Desired coverage fraction

    Returns:
        Threshold value (samples with trust >= threshold are accepted)
    """
    n_accept = int(len(trust_scores) * target_coverage)
    sorted_scores = np.sort(trust_scores)

    # Threshold is the n_accept-th highest score
    threshold_idx = len(sorted_scores) - n_accept
    return float(sorted_scores[threshold_idx])


def aggregate_trust_scores(
    trust_signals: Dict[str, np.ndarray],
    weights: Optional[Dict[str, float]] = None
) -> np.ndarray:
    """
    Aggregate multiple trust signals into a single score.

    Args:
        trust_signals: Dict mapping signal names to [N] arrays
        weights: Optional weights for each signal (default: equal)

    Returns:
        [N] aggregated trust scores
    """
    if weights is None:
        weights = {k: 1.0 / len(trust_signals) for k in trust_signals}

    # Normalize weights
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}

    # Weighted sum
    n_samples = len(next(iter(trust_signals.values())))
    aggregated = np.zeros(n_samples)

    for name, values in trust_signals.items():
        if name in weights:
            aggregated += weights[name] * values

    return aggregated
