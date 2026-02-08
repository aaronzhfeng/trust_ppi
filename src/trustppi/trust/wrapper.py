"""
Trust Wrapper for PPI Predictions.

Wraps a base PPI model with multi-axis trust signals:
- Calibration (temperature scaling)
- OOD detection (MSP, Energy, Mahalanobis)
- Symmetry checking (swap invariance)
- Stability under perturbation

Enables selective prediction: abstain on low-trust samples.
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class TrustVector:
    """Multi-axis trust signals for a prediction."""
    t_cal: float      # Calibrated confidence
    t_ood: float      # OOD score (lower = more in-distribution)
    t_sym: float      # Symmetry violation (lower = better)
    t_stab: float     # Stability score (higher = more stable)
    p_correct: float  # Aggregated trust / probability of correctness


@dataclass
class TrustPrediction:
    """Complete prediction with trust information."""
    prediction: Union[float, torch.Tensor]  # Model output
    raw_confidence: float                    # Uncalibrated confidence
    trust_vector: TrustVector               # Trust signals
    decision: str                           # 'accept' or 'abstain'
    features: Optional[torch.Tensor] = None  # Internal features


# ============================================================================
# Trust Components
# ============================================================================

class TemperatureScaling(nn.Module):
    """
    Temperature scaling for calibration.

    Learns a single temperature parameter to rescale logits.
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by temperature."""
        return logits / self.temperature

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 100
    ) -> float:
        """
        Optimize temperature on validation set.

        Args:
            logits: [N] or [N, C] model logits
            labels: [N] ground truth labels
            lr: Learning rate
            max_iter: Maximum iterations

        Returns:
            Final temperature value
        """
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_fn():
            optimizer.zero_grad()
            scaled = self.forward(logits)
            loss = F.binary_cross_entropy_with_logits(scaled, labels.float())
            loss.backward()
            return loss

        optimizer.step(eval_fn)
        return self.temperature.item()


class OODDetector(nn.Module):
    """
    Out-of-Distribution detection using multiple methods.

    Methods:
    - MSP: Maximum Softmax Probability
    - Energy: Energy-based score
    - Mahalanobis: Distance from class centroids
    """

    def __init__(self, method: str = 'energy'):
        super().__init__()
        self.method = method

        # For Mahalanobis
        self.register_buffer('mean', None)
        self.register_buffer('precision', None)
        self.fitted = False

    def fit_mahalanobis(self, features: torch.Tensor, labels: torch.Tensor):
        """
        Fit Mahalanobis detector on training features.

        Args:
            features: [N, D] feature vectors
            labels: [N] class labels
        """
        # Compute class means
        unique_labels = labels.unique()
        means = []
        for label in unique_labels:
            mask = labels == label
            means.append(features[mask].mean(dim=0))
        self.mean = torch.stack(means)  # [C, D]

        # Compute shared covariance
        centered = features - self.mean[labels.long()]
        cov = centered.T @ centered / len(features)

        # Regularize and invert
        cov = cov + 1e-5 * torch.eye(cov.size(0), device=cov.device)
        self.precision = torch.linalg.inv(cov)

        self.fitted = True

    def forward(
        self,
        logits: Optional[torch.Tensor] = None,
        features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute OOD score (lower = more in-distribution).

        Args:
            logits: [N] or [N, C] model logits
            features: [N, D] feature vectors (for Mahalanobis)

        Returns:
            [N] OOD scores
        """
        if self.method == 'msp':
            # Maximum Softmax Probability
            # For binary: sigmoid gives probability
            probs = torch.sigmoid(logits)
            confidence = torch.max(probs, 1 - probs)
            return -confidence  # Negate so lower = more ID

        elif self.method == 'energy':
            # Energy score: -T * log(sum(exp(logits/T)))
            # For binary: just -|logits| approximately
            return -torch.abs(logits)  # Further from decision boundary = more confident

        elif self.method == 'mahalanobis':
            if not self.fitted or features is None:
                raise ValueError("Mahalanobis requires fitted mean/precision and features")

            # Compute distance to nearest class centroid
            dists = []
            for class_mean in self.mean:
                diff = features - class_mean
                dist = torch.sum(diff @ self.precision * diff, dim=-1)
                dists.append(dist)
            min_dist = torch.stack(dists).min(dim=0)[0]
            return min_dist

        else:
            raise ValueError(f"Unknown OOD method: {self.method}")


class StabilityChecker:
    """
    Check prediction stability under perturbations.

    Applies small perturbations to input and measures prediction variance.
    """

    def __init__(
        self,
        n_perturbations: int = 5,
        noise_scale: float = 0.1
    ):
        self.n_perturbations = n_perturbations
        self.noise_scale = noise_scale

    def compute_stability(
        self,
        model: nn.Module,
        coords: torch.Tensor,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        **kwargs
    ) -> float:
        """
        Compute stability score (higher = more stable).

        Adds noise to coordinates and measures prediction variance.

        Args:
            model: PPI model
            coords: [N, 3] coordinates
            Other args passed to model

        Returns:
            Stability score in [0, 1]
        """
        model.eval()
        predictions = []

        with torch.no_grad():
            # Original prediction
            out = model(node_features, coords, edge_index, edge_attr, **kwargs)
            base_pred = torch.sigmoid(out['logits']).mean().item()
            predictions.append(base_pred)

            # Perturbed predictions
            for _ in range(self.n_perturbations):
                noise = torch.randn_like(coords) * self.noise_scale
                perturbed_coords = coords + noise

                out = model(node_features, perturbed_coords, edge_index, edge_attr, **kwargs)
                pred = torch.sigmoid(out['logits']).mean().item()
                predictions.append(pred)

        # Stability = 1 - normalized variance
        predictions = np.array(predictions)
        variance = predictions.std()
        # Scale variance to [0, 1] (assuming max variance ~ 0.5 for binary)
        stability = 1.0 - min(variance / 0.5, 1.0)

        return stability


class SymmetryChecker:
    """
    Check swap invariance: f(A,B) should equal f(B,A).

    For architecturally symmetric models, this should always be 0.
    Non-zero indicates bugs or numerical instability.
    """

    def compute_swap_error(
        self,
        model: nn.Module,
        node_features: torch.Tensor,
        coords: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        chain_mask: torch.Tensor,
        batch: torch.Tensor
    ) -> float:
        """
        Compute swap error |f(A,B) - f(B,A)|.

        Args:
            model: PPI model
            chain_mask: [N] 0=chain A, 1=chain B
            batch: [N] batch indices
            Other args passed to model

        Returns:
            Absolute difference in predictions
        """
        model.eval()

        with torch.no_grad():
            # Original prediction
            out1 = model(
                node_features, coords, edge_index, edge_attr,
                chain_mask=chain_mask, batch=batch
            )
            pred1 = torch.sigmoid(out1['logits'])

            # Swapped prediction (flip chain mask)
            swapped_mask = 1 - chain_mask
            out2 = model(
                node_features, coords, edge_index, edge_attr,
                chain_mask=swapped_mask, batch=batch
            )
            pred2 = torch.sigmoid(out2['logits'])

            swap_error = torch.abs(pred1 - pred2).mean().item()

        return swap_error


# ============================================================================
# Trust Wrapper
# ============================================================================

class TrustWrapper(nn.Module):
    """
    Wrap a base PPI model with trust signals.

    Combines multiple trust axes into a single decision:
    - If trust is high enough (p_correct >= tau_accept): accept prediction
    - Otherwise: abstain (defer to human or more expensive method)
    """

    def __init__(
        self,
        base_model: nn.Module,
        calibrator: Optional[TemperatureScaling] = None,
        ood_detector: Optional[OODDetector] = None,
        stability_checker: Optional[StabilityChecker] = None,
        symmetry_checker: Optional[SymmetryChecker] = None,
        tau_accept: float = 0.7,
        trust_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize TrustWrapper.

        Args:
            base_model: Base PPI prediction model
            calibrator: Temperature scaling for calibration
            ood_detector: OOD detection module
            stability_checker: Stability under perturbation
            symmetry_checker: Swap invariance checker
            tau_accept: Threshold for accepting predictions
            trust_weights: Weights for combining trust signals
        """
        super().__init__()
        self.model = base_model
        self.calibrator = calibrator or TemperatureScaling()
        self.ood_detector = ood_detector or OODDetector()
        self.stability_checker = stability_checker or StabilityChecker()
        self.symmetry_checker = symmetry_checker or SymmetryChecker()
        self.tau_accept = tau_accept

        # Default weights for trust combination
        self.trust_weights = trust_weights or {
            't_cal': 0.4,    # Calibrated confidence
            't_ood': 0.3,    # OOD score (inverted)
            't_stab': 0.2,   # Stability
            't_sym': 0.1     # Symmetry (inverted)
        }

    def compute_trust_vector(
        self,
        logits: torch.Tensor,
        features: torch.Tensor,
        coords: torch.Tensor,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        chain_mask: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> TrustVector:
        """
        Compute all trust signals for a prediction.

        Args:
            logits: Model output logits
            features: Internal model features
            Other args for computing trust signals

        Returns:
            TrustVector with all signals
        """
        # 1. Calibrated confidence
        scaled_logits = self.calibrator(logits)
        t_cal = torch.sigmoid(scaled_logits).mean().item()

        # 2. OOD score
        if features is not None:
            t_ood_raw = self.ood_detector(logits, features).mean().item()
        else:
            t_ood_raw = self.ood_detector(logits).mean().item()
        # Normalize OOD to [0, 1] where 1 = in-distribution
        t_ood = 1.0 / (1.0 + np.exp(t_ood_raw))  # Sigmoid-ish normalization

        # 3. Stability
        t_stab = self.stability_checker.compute_stability(
            self.model, coords, node_features, edge_index, edge_attr,
            chain_mask=chain_mask, batch=batch
        )

        # 4. Symmetry (only for pair-level models)
        if chain_mask is not None and batch is not None:
            t_sym_raw = self.symmetry_checker.compute_swap_error(
                self.model, node_features, coords, edge_index, edge_attr,
                chain_mask, batch
            )
            t_sym = 1.0 - min(t_sym_raw * 10, 1.0)  # Scale and invert
        else:
            t_sym = 1.0  # Perfect symmetry assumed for node-level

        # 5. Aggregate into p_correct
        p_correct = (
            self.trust_weights['t_cal'] * t_cal +
            self.trust_weights['t_ood'] * t_ood +
            self.trust_weights['t_stab'] * t_stab +
            self.trust_weights['t_sym'] * t_sym
        )

        return TrustVector(
            t_cal=t_cal,
            t_ood=t_ood,
            t_sym=1.0 - t_sym_raw if chain_mask is not None else 0.0,
            t_stab=t_stab,
            p_correct=p_correct
        )

    def predict_with_trust(
        self,
        node_features: torch.Tensor,
        coords: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        chain_mask: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> TrustPrediction:
        """
        Make prediction with trust assessment.

        Args:
            Standard model inputs

        Returns:
            TrustPrediction with prediction, trust, and decision
        """
        self.model.eval()

        with torch.no_grad():
            # Get base prediction with features
            output = self.model(
                node_features, coords, edge_index, edge_attr,
                chain_mask=chain_mask, batch=batch,
                return_features=True
            )

            logits = output['logits']
            features = output.get('features')

            # Raw confidence
            raw_confidence = torch.sigmoid(logits).mean().item()

            # Compute trust vector
            trust_vector = self.compute_trust_vector(
                logits, features, coords, node_features,
                edge_index, edge_attr, chain_mask, batch
            )

            # Make decision
            if trust_vector.p_correct >= self.tau_accept:
                decision = 'accept'
            else:
                decision = 'abstain'

            # Calibrated prediction
            scaled_logits = self.calibrator(logits)
            prediction = torch.sigmoid(scaled_logits)

        return TrustPrediction(
            prediction=prediction,
            raw_confidence=raw_confidence,
            trust_vector=trust_vector,
            decision=decision,
            features=features
        )

    def fit_calibrator(
        self,
        val_logits: torch.Tensor,
        val_labels: torch.Tensor
    ) -> float:
        """Fit temperature scaling on validation data."""
        return self.calibrator.fit(val_logits, val_labels)

    def fit_ood_detector(
        self,
        train_features: torch.Tensor,
        train_labels: torch.Tensor
    ):
        """Fit Mahalanobis OOD detector on training features."""
        if self.ood_detector.method == 'mahalanobis':
            self.ood_detector.fit_mahalanobis(train_features, train_labels)


# ============================================================================
# Selective Prediction Evaluation
# ============================================================================

def selective_prediction_eval(
    trust_wrapper: TrustWrapper,
    test_data: List[Dict[str, torch.Tensor]],
    coverage_levels: List[float] = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
) -> List[Dict[str, float]]:
    """
    Evaluate selective prediction at different coverage levels.

    At each coverage, select top-k% samples by trust score and
    measure accuracy on selected samples.

    Args:
        trust_wrapper: TrustWrapper with fitted calibrator
        test_data: List of batched graph dictionaries
        coverage_levels: Fractions of data to keep

    Returns:
        List of dicts with coverage, accuracy, and other metrics
    """
    all_preds = []
    all_trusts = []
    all_labels = []

    trust_wrapper.eval()

    with torch.no_grad():
        for batch in test_data:
            result = trust_wrapper.predict_with_trust(
                batch['node_features'],
                batch['coords'],
                batch['edge_index'],
                batch['edge_attr'],
                batch.get('chain_mask'),
                batch.get('batch')
            )

            # For interface prediction, compare per-node
            pred = (result.prediction > 0.5).float()
            all_preds.append(pred)
            all_trusts.append(torch.full_like(pred, result.trust_vector.p_correct))
            all_labels.append(batch['labels'])

    all_preds = torch.cat(all_preds)
    all_trusts = torch.cat(all_trusts)
    all_labels = torch.cat(all_labels)

    # Compute metrics at each coverage level
    results = []
    n_total = len(all_preds)

    for coverage in coverage_levels:
        k = int(n_total * coverage)
        if k == 0:
            continue

        # Select top-k by trust
        _, top_idx = torch.topk(all_trusts, k)

        selected_preds = all_preds[top_idx]
        selected_labels = all_labels[top_idx]

        # Compute accuracy
        correct = (selected_preds == selected_labels).float()
        accuracy = correct.mean().item()

        # Also compute precision, recall for interface
        tp = ((selected_preds == 1) & (selected_labels == 1)).sum().item()
        fp = ((selected_preds == 1) & (selected_labels == 0)).sum().item()
        fn = ((selected_preds == 0) & (selected_labels == 1)).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results.append({
            'coverage': coverage,
            'n_samples': k,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })

    return results


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing TrustWrapper...")

    # Create dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(20, 1)

        def forward(self, node_features, coords, edge_index, edge_attr,
                    chain_mask=None, batch=None, return_features=False):
            h = node_features.mean(dim=0, keepdim=True).expand(node_features.size(0), -1)
            logits = self.fc(h).squeeze(-1)
            out = {'logits': logits}
            if return_features:
                out['features'] = h
            return out

    model = DummyModel()

    # Create wrapper
    wrapper = TrustWrapper(
        base_model=model,
        tau_accept=0.6
    )

    # Create dummy input
    n_nodes = 30
    node_features = torch.randn(n_nodes, 20)
    coords = torch.randn(n_nodes, 3)
    edge_index = torch.randint(0, n_nodes, (2, 100))
    edge_attr = torch.randn(100, 1).abs()

    # Test prediction with trust
    result = wrapper.predict_with_trust(
        node_features, coords, edge_index, edge_attr
    )

    print(f"\nPrediction shape: {result.prediction.shape}")
    print(f"Raw confidence: {result.raw_confidence:.3f}")
    print(f"Trust vector:")
    print(f"  t_cal: {result.trust_vector.t_cal:.3f}")
    print(f"  t_ood: {result.trust_vector.t_ood:.3f}")
    print(f"  t_stab: {result.trust_vector.t_stab:.3f}")
    print(f"  t_sym: {result.trust_vector.t_sym:.3f}")
    print(f"  p_correct: {result.trust_vector.p_correct:.3f}")
    print(f"Decision: {result.decision}")

    print("\nAll tests passed!")
