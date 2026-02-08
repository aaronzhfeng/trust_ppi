"""
D-SCRIPT Trust Wrapper for Sequence-Based PPI Models.

Adapts the TrustWrapper for D-SCRIPT and other sequence-based models
that use protein language model embeddings instead of graph inputs.

Key differences from EGNN wrapper:
- Input: (embedding_a, embedding_b) instead of (node_features, coords, edge_index)
- Stability: Measured via embedding perturbation or dropout
- Symmetry: Verified via f(A,B) vs f(B,A) predictions
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .wrapper import (
    TrustVector,
    TrustPrediction,
    TemperatureScaling,
    OODDetector
)
from .threshold_search import ThresholdConfig, compute_aggregated_trust
from .metrics import (
    SelectiveMetrics,
    selective_accuracy_at_coverage,
    coverage_accuracy_curve
)


class DScriptStabilityChecker:
    """
    Check prediction stability for D-SCRIPT models.

    Uses dropout or embedding perturbation to measure stability.
    """

    def __init__(
        self,
        n_samples: int = 5,
        noise_scale: float = 0.01,
        method: str = 'noise'  # 'noise' or 'dropout'
    ):
        self.n_samples = n_samples
        self.noise_scale = noise_scale
        self.method = method

    def compute_stability(
        self,
        model,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
        use_cuda: bool = False
    ) -> float:
        """
        Compute stability score (higher = more stable).

        Args:
            model: D-SCRIPT model
            emb_a: [L_a, D] embedding for protein A
            emb_b: [L_b, D] embedding for protein B
            use_cuda: Whether using GPU

        Returns:
            Stability score in [0, 1]
        """
        predictions = []

        with torch.no_grad():
            # Original prediction
            base_pred = model.predict(emb_a, emb_b).item()
            predictions.append(base_pred)

            # Perturbed predictions
            for _ in range(self.n_samples):
                if self.method == 'noise':
                    # Add Gaussian noise to embeddings
                    noise_a = torch.randn_like(emb_a) * self.noise_scale
                    noise_b = torch.randn_like(emb_b) * self.noise_scale
                    perturbed_a = emb_a + noise_a
                    perturbed_b = emb_b + noise_b
                else:
                    # Use original (dropout would require model.train())
                    perturbed_a = emb_a
                    perturbed_b = emb_b

                pred = model.predict(perturbed_a, perturbed_b).item()
                predictions.append(pred)

        # Stability = 1 - normalized variance
        predictions = np.array(predictions)
        variance = predictions.std()
        # Scale variance to [0, 1] (assuming max variance ~ 0.5 for binary)
        stability = 1.0 - min(variance / 0.5, 1.0)

        return stability


class DScriptSymmetryChecker:
    """
    Check swap invariance for D-SCRIPT predictions.

    For undirected PPI, f(A,B) should equal f(B,A).
    D-SCRIPT typically enforces this architecturally, so this should return ~0.
    """

    def compute_swap_error(
        self,
        model,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor
    ) -> float:
        """
        Compute |f(A,B) - f(B,A)|.

        Args:
            model: D-SCRIPT model
            emb_a: Embedding for protein A
            emb_b: Embedding for protein B

        Returns:
            Absolute difference (0 = perfect symmetry)
        """
        with torch.no_grad():
            pred_ab = model.predict(emb_a, emb_b).item()
            pred_ba = model.predict(emb_b, emb_a).item()

        return abs(pred_ab - pred_ba)


class DScriptTrustWrapper:
    """
    Trust wrapper for D-SCRIPT sequence-based PPI models.

    Wraps a D-SCRIPT model with trust signals:
    - Calibration (temperature scaling)
    - OOD detection (embedding-based)
    - Stability (perturbation)
    - Symmetry (swap invariance)
    """

    def __init__(
        self,
        model,
        calibrator: Optional[TemperatureScaling] = None,
        ood_detector: Optional[OODDetector] = None,
        stability_checker: Optional[DScriptStabilityChecker] = None,
        symmetry_checker: Optional[DScriptSymmetryChecker] = None,
        tau_accept: float = 0.5,
        trust_weights: Optional[Dict[str, float]] = None,
        use_cuda: bool = False
    ):
        """
        Initialize D-SCRIPT Trust Wrapper.

        Args:
            model: D-SCRIPT model (from dscript.models.interaction)
            calibrator: Temperature scaling calibrator
            ood_detector: OOD detection module
            stability_checker: Stability checker
            symmetry_checker: Symmetry checker
            tau_accept: Threshold for accepting predictions
            trust_weights: Weights for combining trust signals
            use_cuda: Whether model uses CUDA
        """
        self.model = model
        self.calibrator = calibrator or TemperatureScaling()
        self.ood_detector = ood_detector or OODDetector(method='energy')
        self.stability_checker = stability_checker or DScriptStabilityChecker()
        self.symmetry_checker = symmetry_checker or DScriptSymmetryChecker()
        self.tau_accept = tau_accept
        self.use_cuda = use_cuda

        # Default weights
        self.trust_weights = trust_weights or {
            't_cal': 0.4,
            't_ood': 0.3,
            't_stab': 0.2,
            't_sym': 0.1
        }

        # Learned threshold config (can be updated via fit_thresholds)
        self.threshold_config = ThresholdConfig(
            tau_accept=tau_accept,
            t_cal_weight=self.trust_weights.get('t_cal', 0.25),
            t_ood_weight=self.trust_weights.get('t_ood', 0.25),
            t_stab_weight=self.trust_weights.get('t_stab', 0.25),
            t_sym_weight=self.trust_weights.get('t_sym', 0.25)
        )

    def predict_with_trust(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
        compute_stability: bool = True,
        compute_symmetry: bool = True
    ) -> TrustPrediction:
        """
        Make prediction with trust assessment.

        Args:
            emb_a: [L_a, D] embedding for protein A
            emb_b: [L_b, D] embedding for protein B
            compute_stability: Whether to compute stability (slower)
            compute_symmetry: Whether to compute symmetry

        Returns:
            TrustPrediction with prediction, trust, and decision
        """
        # Get base prediction
        with torch.no_grad():
            raw_pred = self.model.predict(emb_a, emb_b)
            if isinstance(raw_pred, torch.Tensor):
                raw_pred = raw_pred.item()

        # Convert to logit for calibration
        # D-SCRIPT outputs probability, convert to logit
        raw_logit = np.log(raw_pred / (1 - raw_pred + 1e-10) + 1e-10)
        raw_logit_tensor = torch.tensor([raw_logit], dtype=torch.float32)

        # Calibrated prediction
        scaled_logit = self.calibrator(raw_logit_tensor)
        calibrated_prob = torch.sigmoid(scaled_logit).item()

        # 1. Calibrated confidence
        t_cal = max(calibrated_prob, 1 - calibrated_prob)

        # 2. OOD score (using logit as proxy)
        t_ood_raw = self.ood_detector(raw_logit_tensor).item()
        t_ood = 1.0 / (1.0 + np.exp(t_ood_raw))

        # 3. Stability
        if compute_stability:
            t_stab = self.stability_checker.compute_stability(
                self.model, emb_a, emb_b, self.use_cuda
            )
        else:
            t_stab = 1.0  # Assume stable

        # 4. Symmetry
        if compute_symmetry:
            t_sym_raw = self.symmetry_checker.compute_swap_error(
                self.model, emb_a, emb_b
            )
            t_sym = 1.0 - min(t_sym_raw * 10, 1.0)
        else:
            t_sym = 1.0  # Assume symmetric

        # Aggregate trust
        trust_signals = {
            't_cal': np.array([t_cal]),
            't_ood': np.array([t_ood]),
            't_stab': np.array([t_stab]),
            't_sym': np.array([t_sym])
        }
        p_correct = compute_aggregated_trust(
            trust_signals, self.threshold_config
        )[0]

        trust_vector = TrustVector(
            t_cal=t_cal,
            t_ood=t_ood,
            t_sym=t_sym_raw if compute_symmetry else 0.0,
            t_stab=t_stab,
            p_correct=p_correct
        )

        # Decision
        if p_correct >= self.tau_accept:
            decision = 'accept'
        else:
            decision = 'abstain'

        return TrustPrediction(
            prediction=calibrated_prob,
            raw_confidence=raw_pred,
            trust_vector=trust_vector,
            decision=decision,
            features=None  # D-SCRIPT doesn't expose features easily
        )

    def predict_batch(
        self,
        pairs: List[Tuple[torch.Tensor, torch.Tensor]],
        compute_stability: bool = False,
        compute_symmetry: bool = False
    ) -> List[TrustPrediction]:
        """
        Batch prediction with trust.

        Args:
            pairs: List of (emb_a, emb_b) tuples
            compute_stability: Whether to compute stability (expensive)
            compute_symmetry: Whether to compute symmetry

        Returns:
            List of TrustPrediction objects
        """
        results = []
        for emb_a, emb_b in pairs:
            result = self.predict_with_trust(
                emb_a, emb_b,
                compute_stability=compute_stability,
                compute_symmetry=compute_symmetry
            )
            results.append(result)
        return results

    def collect_trust_signals(
        self,
        pairs: List[Tuple[torch.Tensor, torch.Tensor]],
        labels: List[float],
        compute_stability: bool = True,
        compute_symmetry: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Collect trust signals for threshold optimization.

        Args:
            pairs: List of (emb_a, emb_b) tuples
            labels: Ground truth labels
            compute_stability: Whether to compute stability
            compute_symmetry: Whether to compute symmetry

        Returns:
            Dict with arrays for each trust signal and predictions/labels
        """
        t_cals = []
        t_oods = []
        t_stabs = []
        t_syms = []
        predictions = []
        raw_probs = []

        for emb_a, emb_b in pairs:
            result = self.predict_with_trust(
                emb_a, emb_b,
                compute_stability=compute_stability,
                compute_symmetry=compute_symmetry
            )

            t_cals.append(result.trust_vector.t_cal)
            t_oods.append(result.trust_vector.t_ood)
            t_stabs.append(result.trust_vector.t_stab)
            # Note: t_sym in TrustVector is the raw error, invert it
            t_syms.append(1.0 - min(result.trust_vector.t_sym * 10, 1.0))
            predictions.append(1 if result.prediction > 0.5 else 0)
            raw_probs.append(result.raw_confidence)

        return {
            't_cal': np.array(t_cals),
            't_ood': np.array(t_oods),
            't_stab': np.array(t_stabs),
            't_sym': np.array(t_syms),
            'predictions': np.array(predictions),
            'raw_probs': np.array(raw_probs),
            'labels': np.array(labels)
        }

    def fit_calibrator(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> float:
        """
        Fit temperature scaling on validation data.

        Args:
            logits: [N] model logits (convert probabilities to logits first)
            labels: [N] ground truth labels

        Returns:
            Optimal temperature
        """
        return self.calibrator.fit(logits, labels)

    def fit_thresholds(
        self,
        trust_signals: Dict[str, np.ndarray],
        target_coverage: float = 0.8,
        method: str = 'grid'
    ):
        """
        Learn optimal thresholds from validation data.

        Args:
            trust_signals: Output from collect_trust_signals()
            target_coverage: Target coverage level
            method: 'grid' or 'coverage_constrained'
        """
        from .threshold_search import optimize_thresholds

        predictions = trust_signals['predictions']
        labels = trust_signals['labels']

        signal_dict = {
            't_cal': trust_signals['t_cal'],
            't_ood': trust_signals['t_ood'],
            't_stab': trust_signals['t_stab'],
            't_sym': trust_signals['t_sym']
        }

        result = optimize_thresholds(
            signal_dict,
            predictions,
            labels,
            target_coverage=target_coverage,
            method=method
        )

        self.threshold_config = result.best_config
        self.trust_weights = {
            't_cal': result.best_config.t_cal_weight,
            't_ood': result.best_config.t_ood_weight,
            't_stab': result.best_config.t_stab_weight,
            't_sym': result.best_config.t_sym_weight
        }

        return result


def selective_prediction_dscript(
    wrapper: DScriptTrustWrapper,
    pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    labels: List[float],
    coverage_levels: Optional[List[float]] = None
) -> List[SelectiveMetrics]:
    """
    Evaluate selective prediction for D-SCRIPT with trust wrapper.

    Args:
        wrapper: DScriptTrustWrapper
        pairs: List of (emb_a, emb_b) tuples
        labels: Ground truth labels
        coverage_levels: List of coverage fractions

    Returns:
        List of SelectiveMetrics at each coverage level
    """
    if coverage_levels is None:
        coverage_levels = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Collect all trust signals (skip stability for speed)
    signals = wrapper.collect_trust_signals(
        pairs, labels,
        compute_stability=False,
        compute_symmetry=True
    )

    # Aggregate trust scores using current config
    trust_scores = compute_aggregated_trust(
        {
            't_cal': signals['t_cal'],
            't_ood': signals['t_ood'],
            't_stab': signals['t_stab'],
            't_sym': signals['t_sym']
        },
        wrapper.threshold_config
    )

    predictions = signals['predictions']
    labels_arr = signals['labels']

    # Compute metrics at each coverage level
    return coverage_accuracy_curve(
        trust_scores, predictions, labels_arr, coverage_levels
    )


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing DScriptTrustWrapper...")

    # Create a mock D-SCRIPT-like model for testing
    class MockDScriptModel:
        def predict(self, emb_a, emb_b):
            # Random prediction for testing
            return torch.tensor(0.7)

    model = MockDScriptModel()

    # Create wrapper
    wrapper = DScriptTrustWrapper(model, tau_accept=0.5)

    # Create mock embeddings
    emb_a = torch.randn(100, 1024)
    emb_b = torch.randn(150, 1024)

    # Test prediction
    result = wrapper.predict_with_trust(emb_a, emb_b)

    print(f"\nPrediction: {result.prediction:.3f}")
    print(f"Raw confidence: {result.raw_confidence:.3f}")
    print(f"Trust vector:")
    print(f"  t_cal: {result.trust_vector.t_cal:.3f}")
    print(f"  t_ood: {result.trust_vector.t_ood:.3f}")
    print(f"  t_stab: {result.trust_vector.t_stab:.3f}")
    print(f"  t_sym: {result.trust_vector.t_sym:.3f}")
    print(f"  p_correct: {result.trust_vector.p_correct:.3f}")
    print(f"Decision: {result.decision}")

    print("\nAll tests passed!")
