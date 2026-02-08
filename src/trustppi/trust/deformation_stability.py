"""
Deformation stability trust signal for structure-based PPI models.

Physical motivation (GDL §3.3): Features learned by geometric models should
be stable under small deformations. A prediction that changes dramatically
under minor coordinate noise is untrustworthy.

This module provides:
1. DeformationStability: For structure-based models (EGNN, GVP, etc.)
2. SequenceDeformationStability: For sequence-based models via mutation

The key insight is that robust models should produce similar predictions
for similar inputs. Large prediction variance under small perturbations
indicates the model is operating in an unreliable region.
"""

import torch
import numpy as np
from typing import Dict, Callable, Optional, Union, List, Tuple
from dataclasses import dataclass


@dataclass
class DeformationMetrics:
    """Detailed metrics from deformation stability analysis."""
    stability: float           # 1 / (1 + std) - higher = more stable
    flip_rate: float          # Fraction of predictions that flip
    mean_change: float        # Mean absolute prediction change
    pred_std: float           # Standard deviation of predictions
    pred_range: float         # Range (max - min) of predictions
    lipschitz_ratio: float    # Prediction change / coordinate change


class DeformationStability:
    """
    Compute trust signals based on prediction stability under coordinate noise.

    Applies to structure-based models:
    - EGNN (Equivariant Graph Neural Networks)
    - GVP (Geometric Vector Perceptrons)
    - Any model taking 3D coordinates as input

    Physical motivation:
    - Real protein structures have some flexibility (~0.1-0.5 Å RMSD)
    - Predictions should be stable under this natural variation
    - Large changes under small perturbations indicate unreliable predictions
    """

    def __init__(
        self,
        noise_std: float = 0.1,
        n_samples: int = 10,
        seed: Optional[int] = 42
    ):
        """
        Initialize deformation stability calculator.

        Args:
            noise_std: Standard deviation of Gaussian noise in Angstroms
                      (typical protein flexibility is 0.1-0.5 Å)
            n_samples: Number of perturbed samples to evaluate
            seed: Random seed for reproducibility (None for random)
        """
        self.noise_std = noise_std
        self.n_samples = n_samples
        self.seed = seed

    def compute(
        self,
        model: Callable,
        coords_A: torch.Tensor,
        coords_B: torch.Tensor,
        **model_kwargs
    ) -> Dict[str, float]:
        """
        Compute deformation stability trust signals.

        Args:
            model: Callable that takes (coords_A, coords_B) and returns prediction
            coords_A: [N_A, 3] coordinates of protein A (Angstroms)
            coords_B: [N_B, 3] coordinates of protein B (Angstroms)
            **model_kwargs: Additional arguments for model

        Returns:
            Dictionary of trust signals:
            - t_deform_stable: 1/(1+std) - higher = more stable
            - t_deform_range: 1/(1+range) - higher = smaller prediction range
            - t_deform_flip: 1 - flip_rate - higher = fewer prediction flips
            - t_deform_lipschitz: 1/(1+ratio) - Lipschitz-style stability
            - t_deform_combined: Average of stability and flip signals
        """
        if self.seed is not None:
            torch.manual_seed(self.seed)

        # Ensure tensors are on the right device
        device = coords_A.device

        # Get base prediction
        with torch.no_grad():
            base_pred = self._get_prediction(model, coords_A, coords_B, **model_kwargs)

        # Get predictions under perturbations
        perturbed_preds = []
        for _ in range(self.n_samples):
            # Add Gaussian noise to coordinates
            noise_A = torch.randn_like(coords_A) * self.noise_std
            noise_B = torch.randn_like(coords_B) * self.noise_std

            with torch.no_grad():
                perturbed_pred = self._get_prediction(
                    model,
                    coords_A + noise_A,
                    coords_B + noise_B,
                    **model_kwargs
                )
                perturbed_preds.append(perturbed_pred)

        perturbed_preds = np.array(perturbed_preds)

        # Compute stability metrics
        pred_std = float(np.std(perturbed_preds))
        pred_range = float(np.max(perturbed_preds) - np.min(perturbed_preds))

        # Flip rate: how often does the binary prediction change?
        base_binary = base_pred > 0.5
        flip_rate = float(np.mean([p > 0.5 != base_binary for p in perturbed_preds]))

        # Lipschitz-inspired: prediction change / coordinate change
        mean_change = float(np.mean(np.abs(perturbed_preds - base_pred)))
        lipschitz_ratio = mean_change / self.noise_std if self.noise_std > 0 else 0.0

        # Convert to trust scores (higher = better)
        stability = 1.0 / (1.0 + pred_std)
        range_score = 1.0 / (1.0 + pred_range)
        flip_score = 1.0 - flip_rate
        lipschitz_score = 1.0 / (1.0 + lipschitz_ratio)

        return {
            't_deform_stable': stability,
            't_deform_range': range_score,
            't_deform_flip': flip_score,
            't_deform_lipschitz': lipschitz_score,
            't_deform_combined': (stability + flip_score) / 2,
            'base_prediction': base_pred,
            'pred_std': pred_std,
            'pred_range': pred_range,
            'flip_rate': flip_rate,
            'mean_change': mean_change
        }

    def _get_prediction(
        self,
        model: Callable,
        coords_A: torch.Tensor,
        coords_B: torch.Tensor,
        **kwargs
    ) -> float:
        """Extract scalar prediction from model output."""
        output = model(coords_A, coords_B, **kwargs)

        # Handle various output formats
        if isinstance(output, tuple):
            output = output[0]  # (pred, features) format

        if isinstance(output, torch.Tensor):
            if output.numel() == 1:
                return float(output.item())
            else:
                return float(output.mean().item())
        elif isinstance(output, (float, int)):
            return float(output)
        else:
            return float(output)

    def compute_detailed(
        self,
        model: Callable,
        coords_A: torch.Tensor,
        coords_B: torch.Tensor,
        **model_kwargs
    ) -> DeformationMetrics:
        """
        Compute detailed deformation metrics.

        Returns:
            DeformationMetrics dataclass with all computed values
        """
        signals = self.compute(model, coords_A, coords_B, **model_kwargs)

        return DeformationMetrics(
            stability=signals['t_deform_stable'],
            flip_rate=signals['flip_rate'],
            mean_change=signals['mean_change'],
            pred_std=signals['pred_std'],
            pred_range=signals['pred_range'],
            lipschitz_ratio=signals['mean_change'] / self.noise_std if self.noise_std > 0 else 0.0
        )


class SequenceDeformationStability:
    """
    Deformation stability for sequence-based models via mutation.

    For models that don't take coordinates (D-SCRIPT, TUnA), we test
    stability under single-residue mutations instead.

    Physical motivation:
    - Single point mutations rarely abolish interactions
    - Predictions should be stable under conservative mutations
    - Large changes under single mutations indicate unreliable predictions
    """

    AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

    # Conservative substitution groups (similar physicochemical properties)
    CONSERVATIVE_GROUPS = {
        'A': 'GS', 'C': 'S', 'D': 'EN', 'E': 'DQ', 'F': 'YW',
        'G': 'AS', 'H': 'NQ', 'I': 'LMV', 'K': 'R', 'L': 'IMV',
        'M': 'ILV', 'N': 'DHQ', 'P': 'A', 'Q': 'EHN', 'R': 'K',
        'S': 'ACT', 'T': 'S', 'V': 'ILM', 'W': 'FY', 'Y': 'FW'
    }

    def __init__(
        self,
        n_mutations: int = 10,
        mutation_type: str = 'random',
        seed: Optional[int] = 42
    ):
        """
        Initialize sequence deformation stability calculator.

        Args:
            n_mutations: Number of single-residue mutations to test
            mutation_type: 'random' or 'conservative'
            seed: Random seed for reproducibility
        """
        self.n_mutations = n_mutations
        self.mutation_type = mutation_type
        self.seed = seed

    def compute(
        self,
        model: Callable,
        seq_A: str,
        seq_B: str,
        mutate_both: bool = False
    ) -> Dict[str, float]:
        """
        Compute mutation stability trust signals.

        Args:
            model: Callable that takes (seq_A, seq_B) and returns prediction
            seq_A: Protein A sequence
            seq_B: Protein B sequence
            mutate_both: If True, mutate both proteins; if False, only A

        Returns:
            Dictionary of trust signals:
            - t_mut_stable: 1/(1+std) - higher = more stable
            - t_mut_flip: 1 - flip_rate - higher = fewer flips
            - t_mut_combined: Average of stability and flip
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        # Get base prediction
        base_pred = self._get_prediction(model, seq_A, seq_B)

        # Mutate and predict
        perturbed_preds = []
        for _ in range(self.n_mutations):
            # Mutate protein A
            mutant_A = self._mutate_sequence(seq_A)

            if mutate_both:
                mutant_B = self._mutate_sequence(seq_B)
            else:
                mutant_B = seq_B

            pred = self._get_prediction(model, mutant_A, mutant_B)
            perturbed_preds.append(pred)

        perturbed_preds = np.array(perturbed_preds)

        # Compute metrics
        pred_std = float(np.std(perturbed_preds))
        pred_range = float(np.max(perturbed_preds) - np.min(perturbed_preds))
        mean_change = float(np.mean(np.abs(perturbed_preds - base_pred)))

        # Flip rate
        base_binary = base_pred > 0.5
        flip_rate = float(np.mean([(p > 0.5) != base_binary for p in perturbed_preds]))

        # Trust scores
        stability = 1.0 / (1.0 + pred_std)
        flip_score = 1.0 - flip_rate

        return {
            't_mut_stable': stability,
            't_mut_flip': flip_score,
            't_mut_combined': (stability + flip_score) / 2,
            'base_prediction': base_pred,
            'pred_std': pred_std,
            'pred_range': pred_range,
            'flip_rate': flip_rate,
            'mean_change': mean_change
        }

    def _mutate_sequence(self, sequence: str) -> str:
        """Apply a single random mutation to the sequence."""
        if len(sequence) == 0:
            return sequence

        # Random position
        pos = np.random.randint(len(sequence))
        original_aa = sequence[pos]

        # Choose new amino acid
        if self.mutation_type == 'conservative' and original_aa in self.CONSERVATIVE_GROUPS:
            # Conservative substitution
            candidates = self.CONSERVATIVE_GROUPS[original_aa]
            new_aa = np.random.choice(list(candidates))
        else:
            # Random substitution (excluding original)
            candidates = [aa for aa in self.AMINO_ACIDS if aa != original_aa]
            new_aa = np.random.choice(candidates)

        # Create mutant
        return sequence[:pos] + new_aa + sequence[pos+1:]

    def _get_prediction(
        self,
        model: Callable,
        seq_A: str,
        seq_B: str
    ) -> float:
        """Extract scalar prediction from model output."""
        output = model(seq_A, seq_B)

        if hasattr(output, 'item'):
            return float(output.item())
        elif isinstance(output, torch.Tensor):
            return float(output.mean().item())
        else:
            return float(output)


class EmbeddingDeformationStability:
    """
    Deformation stability in embedding space.

    For models that use pre-computed embeddings (like D-SCRIPT with ESM),
    we can add noise to embeddings directly.

    This is faster than sequence mutation and more directly tests
    the PPI predictor's robustness (not the language model's).
    """

    def __init__(
        self,
        noise_std: float = 0.01,
        n_samples: int = 10,
        seed: Optional[int] = 42
    ):
        """
        Initialize embedding deformation stability calculator.

        Args:
            noise_std: Standard deviation of Gaussian noise (relative to embedding norm)
            n_samples: Number of perturbed samples
            seed: Random seed
        """
        self.noise_std = noise_std
        self.n_samples = n_samples
        self.seed = seed

    def compute(
        self,
        model: Callable,
        emb_A: torch.Tensor,
        emb_B: torch.Tensor,
        **model_kwargs
    ) -> Dict[str, float]:
        """
        Compute embedding deformation stability.

        Args:
            model: Model that takes (emb_A, emb_B) and returns prediction
            emb_A: Protein A embedding
            emb_B: Protein B embedding

        Returns:
            Dictionary of trust signals
        """
        if self.seed is not None:
            torch.manual_seed(self.seed)

        # Get base prediction
        with torch.no_grad():
            base_pred = self._get_prediction(model, emb_A, emb_B, **model_kwargs)

        # Perturb embeddings
        perturbed_preds = []
        for _ in range(self.n_samples):
            noise_A = torch.randn_like(emb_A) * self.noise_std
            noise_B = torch.randn_like(emb_B) * self.noise_std

            with torch.no_grad():
                pred = self._get_prediction(
                    model,
                    emb_A + noise_A,
                    emb_B + noise_B,
                    **model_kwargs
                )
                perturbed_preds.append(pred)

        perturbed_preds = np.array(perturbed_preds)

        # Metrics
        pred_std = float(np.std(perturbed_preds))
        flip_rate = float(np.mean([(p > 0.5) != (base_pred > 0.5) for p in perturbed_preds]))
        mean_change = float(np.mean(np.abs(perturbed_preds - base_pred)))

        stability = 1.0 / (1.0 + pred_std)
        flip_score = 1.0 - flip_rate

        return {
            't_emb_stable': stability,
            't_emb_flip': flip_score,
            't_emb_combined': (stability + flip_score) / 2,
            'base_prediction': base_pred,
            'pred_std': pred_std,
            'flip_rate': flip_rate,
            'mean_change': mean_change
        }

    def _get_prediction(
        self,
        model: Callable,
        emb_A: torch.Tensor,
        emb_B: torch.Tensor,
        **kwargs
    ) -> float:
        """Extract scalar prediction."""
        output = model(emb_A, emb_B, **kwargs)

        if hasattr(output, 'item'):
            return float(output.item())
        elif isinstance(output, tuple):
            return float(output[0].item()) if hasattr(output[0], 'item') else float(output[0])
        else:
            return float(output)


# Convenience functions for common use cases

def compute_structure_deformation_trust(
    model: Callable,
    coords_A: torch.Tensor,
    coords_B: torch.Tensor,
    noise_std: float = 0.1,
    n_samples: int = 10,
    **model_kwargs
) -> Dict[str, float]:
    """
    Convenience function for structure-based deformation stability.

    Args:
        model: Structure-based PPI model
        coords_A: Protein A coordinates [N, 3]
        coords_B: Protein B coordinates [M, 3]
        noise_std: Noise in Angstroms
        n_samples: Number of perturbations

    Returns:
        Dictionary of trust signals
    """
    deform = DeformationStability(noise_std=noise_std, n_samples=n_samples)
    return deform.compute(model, coords_A, coords_B, **model_kwargs)


def compute_sequence_deformation_trust(
    model: Callable,
    seq_A: str,
    seq_B: str,
    n_mutations: int = 10,
    mutation_type: str = 'random'
) -> Dict[str, float]:
    """
    Convenience function for sequence-based deformation stability.

    Args:
        model: Sequence-based PPI model
        seq_A: Protein A sequence
        seq_B: Protein B sequence
        n_mutations: Number of mutations to test
        mutation_type: 'random' or 'conservative'

    Returns:
        Dictionary of trust signals
    """
    deform = SequenceDeformationStability(
        n_mutations=n_mutations,
        mutation_type=mutation_type
    )
    return deform.compute(model, seq_A, seq_B)


def compute_embedding_deformation_trust(
    model: Callable,
    emb_A: torch.Tensor,
    emb_B: torch.Tensor,
    noise_std: float = 0.01,
    n_samples: int = 10,
    **model_kwargs
) -> Dict[str, float]:
    """
    Convenience function for embedding-based deformation stability.

    Args:
        model: Embedding-based PPI model
        emb_A: Protein A embedding
        emb_B: Protein B embedding
        noise_std: Relative noise level
        n_samples: Number of perturbations

    Returns:
        Dictionary of trust signals
    """
    deform = EmbeddingDeformationStability(noise_std=noise_std, n_samples=n_samples)
    return deform.compute(model, emb_A, emb_B, **model_kwargs)
