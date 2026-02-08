"""
Threshold Search Optimization for TrustPPI.

Learn optimal trust thresholds from validation data instead of using fixed values.

Key approaches:
1. Grid search: Exhaustive search over threshold grid
2. Coverage-constrained: Find threshold that achieves target coverage
3. Pareto optimization: Trade-off between coverage and accuracy
"""

from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import numpy as np
from itertools import product
import logging

from .metrics import (
    SelectiveMetrics,
    selective_accuracy_at_coverage,
    coverage_accuracy_curve,
    aggregate_trust_scores,
    optimal_threshold_for_coverage
)

logger = logging.getLogger(__name__)


@dataclass
class ThresholdConfig:
    """Configuration for trust thresholds."""
    tau_accept: float          # Overall acceptance threshold on p_correct
    t_cal_weight: float        # Weight for calibration signal
    t_ood_weight: float        # Weight for OOD signal
    t_stab_weight: float       # Weight for stability signal
    t_sym_weight: float        # Weight for symmetry signal

    def to_dict(self) -> Dict[str, float]:
        return {
            'tau_accept': self.tau_accept,
            't_cal_weight': self.t_cal_weight,
            't_ood_weight': self.t_ood_weight,
            't_stab_weight': self.t_stab_weight,
            't_sym_weight': self.t_sym_weight
        }

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> 'ThresholdConfig':
        return cls(
            tau_accept=d.get('tau_accept', 0.5),
            t_cal_weight=d.get('t_cal_weight', 0.25),
            t_ood_weight=d.get('t_ood_weight', 0.25),
            t_stab_weight=d.get('t_stab_weight', 0.25),
            t_sym_weight=d.get('t_sym_weight', 0.25)
        )

    @classmethod
    def default(cls) -> 'ThresholdConfig':
        """Default configuration (equal weights)."""
        return cls(
            tau_accept=0.5,
            t_cal_weight=0.25,
            t_ood_weight=0.25,
            t_stab_weight=0.25,
            t_sym_weight=0.25
        )


@dataclass
class OptimizationResult:
    """Result of threshold optimization."""
    best_config: ThresholdConfig
    best_metrics: SelectiveMetrics
    all_results: List[Tuple[ThresholdConfig, SelectiveMetrics]]
    search_method: str


def compute_aggregated_trust(
    trust_signals: Dict[str, np.ndarray],
    config: ThresholdConfig
) -> np.ndarray:
    """
    Compute aggregated trust score using config weights.

    Args:
        trust_signals: Dict with 't_cal', 't_ood', 't_stab', 't_sym' arrays
        config: ThresholdConfig with weights

    Returns:
        [N] aggregated trust scores
    """
    weights = {
        't_cal': config.t_cal_weight,
        't_ood': config.t_ood_weight,
        't_stab': config.t_stab_weight,
        't_sym': config.t_sym_weight
    }
    return aggregate_trust_scores(trust_signals, weights)


def evaluate_config(
    config: ThresholdConfig,
    trust_signals: Dict[str, np.ndarray],
    predictions: np.ndarray,
    labels: np.ndarray,
    target_coverage: float
) -> SelectiveMetrics:
    """
    Evaluate a threshold configuration.

    Args:
        config: Threshold configuration to evaluate
        trust_signals: Dictionary of trust signal arrays
        predictions: [N] binary predictions
        labels: [N] ground truth labels
        target_coverage: Target coverage level

    Returns:
        SelectiveMetrics at the target coverage
    """
    # Compute aggregated trust
    trust_scores = compute_aggregated_trust(trust_signals, config)

    # Evaluate selective prediction
    return selective_accuracy_at_coverage(
        trust_scores, predictions, labels, target_coverage
    )


def grid_search_weights(
    trust_signals: Dict[str, np.ndarray],
    predictions: np.ndarray,
    labels: np.ndarray,
    target_coverage: float = 0.8,
    n_grid: int = 5,
    metric: str = 'accuracy'
) -> OptimizationResult:
    """
    Grid search over weight combinations.

    Searches over all combinations of weights that sum to 1.0.

    Args:
        trust_signals: Dictionary of trust signal arrays
        predictions: [N] predictions
        labels: [N] labels
        target_coverage: Target coverage level
        n_grid: Number of grid points per weight (total = n_grid^4 / normalizations)
        metric: 'accuracy', 'f1', or 'improvement'

    Returns:
        OptimizationResult with best configuration
    """
    # Generate weight grid
    weight_values = np.linspace(0.0, 1.0, n_grid)

    best_config = None
    best_metrics = None
    best_score = -np.inf
    all_results = []

    # Grid search over all weight combinations
    for w_cal, w_ood, w_stab, w_sym in product(weight_values, repeat=4):
        # Skip if all weights are zero
        total = w_cal + w_ood + w_stab + w_sym
        if total < 0.01:
            continue

        # Normalize weights
        w_cal /= total
        w_ood /= total
        w_stab /= total
        w_sym /= total

        config = ThresholdConfig(
            tau_accept=0.5,  # Not used in this mode
            t_cal_weight=w_cal,
            t_ood_weight=w_ood,
            t_stab_weight=w_stab,
            t_sym_weight=w_sym
        )

        metrics = evaluate_config(
            config, trust_signals, predictions, labels, target_coverage
        )

        all_results.append((config, metrics))

        # Select based on metric
        if metric == 'accuracy':
            score = metrics.accuracy
        elif metric == 'f1':
            score = metrics.f1
        elif metric == 'improvement':
            score = metrics.improvement
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score = score
            best_config = config
            best_metrics = metrics

    return OptimizationResult(
        best_config=best_config,
        best_metrics=best_metrics,
        all_results=all_results,
        search_method='grid_search_weights'
    )


def coverage_constrained_search(
    trust_signals: Dict[str, np.ndarray],
    predictions: np.ndarray,
    labels: np.ndarray,
    target_coverage: float = 0.8,
    coverage_tolerance: float = 0.05,
    n_grid: int = 5
) -> OptimizationResult:
    """
    Find best weights that achieve approximately target coverage.

    Args:
        trust_signals: Dictionary of trust signal arrays
        predictions: [N] predictions
        labels: [N] labels
        target_coverage: Target coverage level
        coverage_tolerance: Acceptable deviation from target
        n_grid: Grid resolution

    Returns:
        OptimizationResult with best configuration
    """
    weight_values = np.linspace(0.0, 1.0, n_grid)

    best_config = None
    best_metrics = None
    best_accuracy = -np.inf
    all_results = []

    for w_cal, w_ood, w_stab, w_sym in product(weight_values, repeat=4):
        total = w_cal + w_ood + w_stab + w_sym
        if total < 0.01:
            continue

        # Normalize
        w_cal /= total
        w_ood /= total
        w_stab /= total
        w_sym /= total

        config = ThresholdConfig(
            tau_accept=0.5,
            t_cal_weight=w_cal,
            t_ood_weight=w_ood,
            t_stab_weight=w_stab,
            t_sym_weight=w_sym
        )

        # Evaluate at target coverage
        metrics = evaluate_config(
            config, trust_signals, predictions, labels, target_coverage
        )

        all_results.append((config, metrics))

        # Check coverage constraint
        actual_coverage = metrics.coverage
        if abs(actual_coverage - target_coverage) <= coverage_tolerance:
            if metrics.accuracy > best_accuracy:
                best_accuracy = metrics.accuracy
                best_config = config
                best_metrics = metrics

    # If no config met constraint, use best overall
    if best_config is None:
        logger.warning("No config met coverage constraint, using best overall")
        for config, metrics in all_results:
            if metrics.accuracy > best_accuracy:
                best_accuracy = metrics.accuracy
                best_config = config
                best_metrics = metrics

    return OptimizationResult(
        best_config=best_config,
        best_metrics=best_metrics,
        all_results=all_results,
        search_method='coverage_constrained'
    )


def pareto_frontier_search(
    trust_signals: Dict[str, np.ndarray],
    predictions: np.ndarray,
    labels: np.ndarray,
    coverage_levels: Optional[List[float]] = None,
    n_grid: int = 5
) -> List[Tuple[ThresholdConfig, List[SelectiveMetrics]]]:
    """
    Find Pareto-optimal configurations across coverage levels.

    A configuration is Pareto-optimal if no other configuration
    dominates it (better on all coverage levels).

    Args:
        trust_signals: Dictionary of trust signal arrays
        predictions: [N] predictions
        labels: [N] labels
        coverage_levels: List of coverage levels to evaluate
        n_grid: Grid resolution

    Returns:
        List of (config, [metrics at each coverage]) for Pareto-optimal configs
    """
    if coverage_levels is None:
        coverage_levels = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    weight_values = np.linspace(0.0, 1.0, n_grid)
    all_configs = []

    # Evaluate all configs at all coverage levels
    for w_cal, w_ood, w_stab, w_sym in product(weight_values, repeat=4):
        total = w_cal + w_ood + w_stab + w_sym
        if total < 0.01:
            continue

        w_cal /= total
        w_ood /= total
        w_stab /= total
        w_sym /= total

        config = ThresholdConfig(
            tau_accept=0.5,
            t_cal_weight=w_cal,
            t_ood_weight=w_ood,
            t_stab_weight=w_stab,
            t_sym_weight=w_sym
        )

        trust_scores = compute_aggregated_trust(trust_signals, config)

        # Evaluate at all coverage levels
        metrics_list = coverage_accuracy_curve(
            trust_scores, predictions, labels, coverage_levels
        )

        all_configs.append((config, metrics_list))

    # Find Pareto frontier
    pareto_configs = []
    for i, (config_i, metrics_i) in enumerate(all_configs):
        is_dominated = False

        for j, (config_j, metrics_j) in enumerate(all_configs):
            if i == j:
                continue

            # Check if j dominates i (better on all coverage levels)
            dominates = all(
                m_j.accuracy >= m_i.accuracy
                for m_i, m_j in zip(metrics_i, metrics_j)
            ) and any(
                m_j.accuracy > m_i.accuracy
                for m_i, m_j in zip(metrics_i, metrics_j)
            )

            if dominates:
                is_dominated = True
                break

        if not is_dominated:
            pareto_configs.append((config_i, metrics_i))

    return pareto_configs


def optimize_thresholds(
    trust_signals: Dict[str, np.ndarray],
    predictions: np.ndarray,
    labels: np.ndarray,
    target_coverage: float = 0.8,
    method: str = 'grid',
    n_grid: int = 5,
    metric: str = 'accuracy'
) -> OptimizationResult:
    """
    Main entry point for threshold optimization.

    Args:
        trust_signals: Dict with 't_cal', 't_ood', 't_stab', 't_sym' [N] arrays
        predictions: [N] binary predictions
        labels: [N] ground truth labels
        target_coverage: Target coverage level (default: 0.8)
        method: 'grid' or 'coverage_constrained'
        n_grid: Grid resolution
        metric: Optimization metric ('accuracy', 'f1', 'improvement')

    Returns:
        OptimizationResult with best configuration and metrics
    """
    if method == 'grid':
        return grid_search_weights(
            trust_signals, predictions, labels,
            target_coverage=target_coverage,
            n_grid=n_grid,
            metric=metric
        )
    elif method == 'coverage_constrained':
        return coverage_constrained_search(
            trust_signals, predictions, labels,
            target_coverage=target_coverage,
            n_grid=n_grid
        )
    else:
        raise ValueError(f"Unknown method: {method}")


def compare_learned_vs_fixed(
    trust_signals: Dict[str, np.ndarray],
    predictions: np.ndarray,
    labels: np.ndarray,
    coverage_levels: Optional[List[float]] = None
) -> Dict[str, List[SelectiveMetrics]]:
    """
    Compare learned thresholds vs fixed equal weights.

    Args:
        trust_signals: Dictionary of trust signal arrays
        predictions: [N] predictions
        labels: [N] labels
        coverage_levels: List of coverage levels

    Returns:
        Dict with 'learned' and 'fixed' coverage-accuracy curves
    """
    if coverage_levels is None:
        coverage_levels = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Fixed equal weights
    fixed_config = ThresholdConfig.default()
    fixed_trust = compute_aggregated_trust(trust_signals, fixed_config)
    fixed_results = coverage_accuracy_curve(
        fixed_trust, predictions, labels, coverage_levels
    )

    # Learn optimal weights at 80% coverage
    opt_result = optimize_thresholds(
        trust_signals, predictions, labels,
        target_coverage=0.8,
        method='grid',
        n_grid=5
    )

    learned_trust = compute_aggregated_trust(trust_signals, opt_result.best_config)
    learned_results = coverage_accuracy_curve(
        learned_trust, predictions, labels, coverage_levels
    )

    return {
        'fixed': fixed_results,
        'learned': learned_results,
        'learned_config': opt_result.best_config.to_dict()
    }


# ============================================================================
# Bayesian Optimization (optional, for large search spaces)
# ============================================================================

def bayesian_optimization(
    trust_signals: Dict[str, np.ndarray],
    predictions: np.ndarray,
    labels: np.ndarray,
    target_coverage: float = 0.8,
    n_iterations: int = 50,
    metric: str = 'accuracy'
) -> OptimizationResult:
    """
    Bayesian optimization for threshold search.

    More efficient than grid search for large search spaces.
    Requires scikit-optimize.

    Args:
        trust_signals: Dictionary of trust signal arrays
        predictions: [N] predictions
        labels: [N] labels
        target_coverage: Target coverage level
        n_iterations: Number of optimization iterations
        metric: 'accuracy', 'f1', or 'improvement'

    Returns:
        OptimizationResult with best configuration
    """
    try:
        from skopt import gp_minimize
        from skopt.space import Real
    except ImportError:
        logger.warning("scikit-optimize not installed, falling back to grid search")
        return grid_search_weights(
            trust_signals, predictions, labels,
            target_coverage=target_coverage,
            metric=metric
        )

    def objective(params):
        w_cal, w_ood, w_stab, w_sym = params
        total = w_cal + w_ood + w_stab + w_sym

        if total < 0.01:
            return 0.0  # Invalid config

        config = ThresholdConfig(
            tau_accept=0.5,
            t_cal_weight=w_cal / total,
            t_ood_weight=w_ood / total,
            t_stab_weight=w_stab / total,
            t_sym_weight=w_sym / total
        )

        metrics = evaluate_config(
            config, trust_signals, predictions, labels, target_coverage
        )

        # Maximize accuracy -> minimize negative accuracy
        if metric == 'accuracy':
            return -metrics.accuracy
        elif metric == 'f1':
            return -metrics.f1
        else:
            return -metrics.improvement

    # Search space
    space = [
        Real(0.0, 1.0, name='w_cal'),
        Real(0.0, 1.0, name='w_ood'),
        Real(0.0, 1.0, name='w_stab'),
        Real(0.0, 1.0, name='w_sym')
    ]

    # Run optimization
    result = gp_minimize(objective, space, n_calls=n_iterations, random_state=42)

    # Extract best config
    w_cal, w_ood, w_stab, w_sym = result.x
    total = w_cal + w_ood + w_stab + w_sym

    best_config = ThresholdConfig(
        tau_accept=0.5,
        t_cal_weight=w_cal / total,
        t_ood_weight=w_ood / total,
        t_stab_weight=w_stab / total,
        t_sym_weight=w_sym / total
    )

    best_metrics = evaluate_config(
        best_config, trust_signals, predictions, labels, target_coverage
    )

    return OptimizationResult(
        best_config=best_config,
        best_metrics=best_metrics,
        all_results=[],  # Not tracked in Bayesian opt
        search_method='bayesian_optimization'
    )
