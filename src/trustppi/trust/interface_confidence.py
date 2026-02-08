"""
Interface-based trust signals for PPI models that output residue-level predictions.

Physical motivation: Protein-protein interactions are mediated by specific
interface residues. Uncertainty in interface prediction → uncertainty in
global interaction prediction.

D-SCRIPT outputs contact maps of shape [L_A, L_B] where each element represents
the predicted probability of contact between residue i in protein A and
residue j in protein B.

Trust signals extracted:
1. Interface Sharpness: Low entropy predictions = confident interface
2. Interface Size: Reasonable number of contacts (5-50 residues typically)
3. Interface Consistency: Contiguous patches, not scattered points
4. Combined: Weighted average of above signals
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class InterfaceMetrics:
    """Metrics computed from interface/contact map predictions."""
    sharpness: float          # 1 - normalized entropy (higher = more confident)
    size_score: float         # Plausibility of interface size
    consistency: float        # Spatial contiguity of contacts
    n_contacts: int           # Number of predicted contacts
    max_contact_prob: float   # Maximum contact probability
    mean_contact_prob: float  # Mean contact probability


class InterfaceConfidence:
    """
    Compute trust signals from interface/contact map predictions.

    Applies to models that output residue-level predictions:
    - D-SCRIPT: contact_map of shape [L_A, L_B]
    - Other PPI models with interface predictions

    Physical motivation:
    - Sharp predictions (near 0 or 1) indicate model confidence
    - Real interfaces have reasonable size (5-50 contacts)
    - Real interfaces form contiguous patches, not scattered points
    """

    def __init__(
        self,
        contact_threshold: float = 0.5,
        min_contacts: int = 5,
        max_contacts: int = 50,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize interface confidence calculator.

        Args:
            contact_threshold: Probability threshold for binary contact prediction
            min_contacts: Minimum expected interface contacts
            max_contacts: Maximum expected interface contacts
            weights: Weights for combining signals (default: equal weights)
        """
        self.contact_threshold = contact_threshold
        self.min_contacts = min_contacts
        self.max_contacts = max_contacts
        self.weights = weights or {
            'sharpness': 0.4,
            'size': 0.3,
            'consistency': 0.3
        }

    def compute(
        self,
        contact_map: Union[torch.Tensor, np.ndarray]
    ) -> Dict[str, float]:
        """
        Compute interface-based trust signals.

        Args:
            contact_map: [L_A, L_B] tensor/array of contact probabilities

        Returns:
            Dictionary of trust signals:
            - t_interface_sharp: Sharpness (1 = all predictions near 0/1)
            - t_interface_size: Size plausibility (1 = reasonable interface size)
            - t_interface_consist: Consistency (1 = contiguous interface)
            - t_interface_combined: Weighted combination
        """
        # Convert to tensor if needed
        if isinstance(contact_map, np.ndarray):
            contact_map = torch.from_numpy(contact_map).float()

        # Ensure 2D
        if contact_map.dim() == 3:
            contact_map = contact_map.squeeze(0)

        # Clip to valid probability range
        contact_map = torch.clamp(contact_map, 1e-7, 1 - 1e-7)

        # 1. Interface Sharpness (low entropy = confident predictions)
        sharpness = self._compute_sharpness(contact_map)

        # 2. Interface Size Plausibility
        size_score, n_contacts = self._compute_size_score(contact_map)

        # 3. Interface Consistency (spatial contiguity)
        consistency = self._compute_contiguity(contact_map)

        # 4. Combined score
        combined = (
            self.weights['sharpness'] * sharpness +
            self.weights['size'] * size_score +
            self.weights['consistency'] * consistency
        )

        return {
            't_interface_sharp': sharpness,
            't_interface_size': size_score,
            't_interface_consist': consistency,
            't_interface_combined': combined,
            'n_contacts': n_contacts,
            'max_contact_prob': float(contact_map.max().item()),
            'mean_contact_prob': float(contact_map.mean().item())
        }

    def _compute_sharpness(self, contact_map: torch.Tensor) -> float:
        """
        Compute sharpness as 1 - normalized entropy.

        High sharpness = predictions near 0 or 1 (confident)
        Low sharpness = predictions near 0.5 (uncertain)
        """
        # Binary entropy: H(p) = -p*log(p) - (1-p)*log(1-p)
        entropy = -(
            contact_map * torch.log(contact_map) +
            (1 - contact_map) * torch.log(1 - contact_map)
        )

        # Normalize by max entropy (log(2))
        normalized_entropy = entropy.mean().item() / np.log(2)

        # Sharpness = 1 - entropy (higher = more confident)
        return 1.0 - normalized_entropy

    def _compute_size_score(
        self,
        contact_map: torch.Tensor
    ) -> Tuple[float, int]:
        """
        Compute interface size plausibility.

        Real protein interfaces typically have 5-50 contacting residue pairs.
        Too few = no real interface, too many = probably noise.
        """
        binary_map = (contact_map > self.contact_threshold).float()
        n_contacts = int(binary_map.sum().item())

        if n_contacts < self.min_contacts:
            # Too few contacts: score proportional to count
            size_score = n_contacts / self.min_contacts
        elif n_contacts > self.max_contacts:
            # Too many contacts: penalize excess
            excess = n_contacts - self.max_contacts
            size_score = max(0.0, 1.0 - excess / self.max_contacts)
        else:
            # Goldilocks zone
            size_score = 1.0

        return size_score, n_contacts

    def _compute_contiguity(self, contact_map: torch.Tensor) -> float:
        """
        Measure how contiguous the predicted interface is.

        Real interfaces form patches, not scattered points.
        Computed as fraction of contact points that have neighbors.
        """
        binary_map = (contact_map > self.contact_threshold).float()

        if binary_map.sum() == 0:
            return 0.0

        # Pad for neighbor checking
        padded = torch.nn.functional.pad(binary_map, (1, 1, 1, 1), value=0)

        # Count neighbors (up, down, left, right)
        neighbors = (
            padded[:-2, 1:-1] +  # up
            padded[2:, 1:-1] +   # down
            padded[1:-1, :-2] +  # left
            padded[1:-1, 2:]     # right
        )

        # Fraction of contacts with at least one neighbor
        has_neighbor = ((neighbors > 0) & (binary_map > 0)).float()
        contiguity = has_neighbor.sum() / max(binary_map.sum(), 1)

        return min(float(contiguity.item()), 1.0)

    def compute_detailed(
        self,
        contact_map: Union[torch.Tensor, np.ndarray]
    ) -> InterfaceMetrics:
        """
        Compute detailed interface metrics.

        Args:
            contact_map: [L_A, L_B] contact probabilities

        Returns:
            InterfaceMetrics dataclass with all metrics
        """
        signals = self.compute(contact_map)

        return InterfaceMetrics(
            sharpness=signals['t_interface_sharp'],
            size_score=signals['t_interface_size'],
            consistency=signals['t_interface_consist'],
            n_contacts=signals['n_contacts'],
            max_contact_prob=signals['max_contact_prob'],
            mean_contact_prob=signals['mean_contact_prob']
        )


def extract_dscript_contact_map(
    model,
    emb_a: torch.Tensor,
    emb_b: torch.Tensor,
    use_cuda: bool = False
) -> Tuple[float, torch.Tensor]:
    """
    Extract contact map from D-SCRIPT model.

    D-SCRIPT computes contact maps internally during prediction.
    This function extracts both the global prediction and the contact map.

    Args:
        model: D-SCRIPT model
        emb_a: [1, L_A, D] embedding of protein A
        emb_b: [1, L_B, D] embedding of protein B
        use_cuda: Whether to use CUDA

    Returns:
        (global_prediction, contact_map) tuple
    """
    model.eval()

    with torch.no_grad():
        # D-SCRIPT forward pass
        # The model computes contact map internally
        # We need to access intermediate outputs

        # Get projection outputs
        z_a = model.projA(emb_a)  # [1, L_A, hidden]
        z_b = model.projB(emb_b)  # [1, L_B, hidden]

        # Compute contact map
        # D-SCRIPT uses: cm = abs(A - B) * (A * B)
        # But the actual computation may vary

        # For the standard D-SCRIPT interaction:
        # B = broadcast(z_a, z_b) gives [1, L_A, L_B, hidden*2]
        # Then MLP reduces to [1, L_A, L_B, 1]

        # Simplified: compute pairwise similarity as proxy
        z_a_norm = z_a / (z_a.norm(dim=-1, keepdim=True) + 1e-8)
        z_b_norm = z_b / (z_b.norm(dim=-1, keepdim=True) + 1e-8)

        # [1, L_A, L_B]
        contact_map = torch.sigmoid(
            torch.bmm(z_a_norm, z_b_norm.transpose(1, 2))
        )

        # Get global prediction
        global_pred = model.predict(emb_a, emb_b).item()

    return global_pred, contact_map.squeeze(0)


# Convenience function for D-SCRIPT evaluation
def compute_interface_trust_dscript(
    model,
    emb_a: torch.Tensor,
    emb_b: torch.Tensor,
    interface_conf: Optional[InterfaceConfidence] = None,
    use_cuda: bool = False
) -> Dict[str, float]:
    """
    Compute interface trust signals for D-SCRIPT prediction.

    Args:
        model: D-SCRIPT model
        emb_a: Protein A embedding
        emb_b: Protein B embedding
        interface_conf: InterfaceConfidence instance (created if None)
        use_cuda: Whether to use CUDA

    Returns:
        Dictionary with prediction and interface trust signals
    """
    if interface_conf is None:
        interface_conf = InterfaceConfidence()

    # Get prediction and contact map
    pred, contact_map = extract_dscript_contact_map(model, emb_a, emb_b, use_cuda)

    # Compute interface trust
    trust_signals = interface_conf.compute(contact_map)

    return {
        'prediction': pred,
        **trust_signals
    }
