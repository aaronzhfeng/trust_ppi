"""
Experiment 14: Alternative OOD Detection Methods.

Tests different OOD detection approaches beyond energy-based.

Usage:
    python -m experiments.eval_ood_baselines --quick
    python -m experiments.eval_ood_baselines --data yeast --limit 200
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "external" / "D-SCRIPT"))

from src.trustppi.data import load_pairs, load_sequences_for_pairs, get_dataset_paths

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OODDetector:
    """Base class for OOD detection methods."""

    def fit(self, embeddings: np.ndarray, labels: np.ndarray = None):
        """Fit detector on in-distribution data."""
        raise NotImplementedError

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """Return OOD scores (higher = more OOD)."""
        raise NotImplementedError


class EnergyOOD(OODDetector):
    """Energy-based OOD detection."""

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
        self.mean_energy = None
        self.std_energy = None

    def fit(self, logits: np.ndarray, labels: np.ndarray = None):
        """Fit on training logits."""
        energies = -self.temperature * np.log(
            np.exp(logits / self.temperature).sum(axis=-1) + 1e-10
        )
        self.mean_energy = energies.mean()
        self.std_energy = energies.std()

    def score(self, logits: np.ndarray) -> np.ndarray:
        """Return normalized energy scores."""
        energies = -self.temperature * np.log(
            np.exp(logits / self.temperature).sum(axis=-1) + 1e-10
        )
        # Higher energy = more OOD
        return (energies - self.mean_energy) / (self.std_energy + 1e-10)


class MaxSoftmaxOOD(OODDetector):
    """Maximum softmax probability for OOD detection (baseline)."""

    def fit(self, logits: np.ndarray, labels: np.ndarray = None):
        pass  # No fitting needed

    def score(self, logits: np.ndarray) -> np.ndarray:
        """Lower max prob = more OOD."""
        probs = np.exp(logits) / (np.exp(logits).sum(axis=-1, keepdims=True) + 1e-10)
        max_probs = probs.max(axis=-1)
        # Invert so higher = more OOD
        return 1.0 - max_probs


class MahalanobisOOD(OODDetector):
    """Mahalanobis distance-based OOD detection."""

    def __init__(self):
        self.mean = None
        self.precision = None

    def fit(self, embeddings: np.ndarray, labels: np.ndarray = None):
        """Fit multivariate Gaussian to training embeddings."""
        self.mean = embeddings.mean(axis=0)
        centered = embeddings - self.mean
        covariance = np.cov(centered.T) + np.eye(embeddings.shape[1]) * 1e-6
        self.precision = np.linalg.inv(covariance)

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """Return Mahalanobis distances."""
        centered = embeddings - self.mean
        # Mahalanobis distance: sqrt((x-mu)^T * Sigma^-1 * (x-mu))
        distances = np.sqrt(
            np.sum(centered @ self.precision * centered, axis=1)
        )
        return distances


class KNNDistanceOOD(OODDetector):
    """K-nearest neighbor distance for OOD detection."""

    def __init__(self, k: int = 5):
        self.k = k
        self.train_embeddings = None

    def fit(self, embeddings: np.ndarray, labels: np.ndarray = None):
        self.train_embeddings = embeddings

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """Return mean distance to k nearest neighbors."""
        # Compute all pairwise distances
        distances = np.zeros(len(embeddings))

        for i, emb in enumerate(embeddings):
            dists = np.sqrt(((self.train_embeddings - emb) ** 2).sum(axis=1))
            # Get k smallest distances (excluding exact matches)
            k_smallest = np.sort(dists)[:self.k]
            distances[i] = k_smallest.mean()

        return distances


class EntropyOOD(OODDetector):
    """Predictive entropy for OOD detection."""

    def fit(self, logits: np.ndarray, labels: np.ndarray = None):
        pass

    def score(self, logits: np.ndarray) -> np.ndarray:
        """Higher entropy = more uncertain = potentially OOD."""
        probs = np.exp(logits) / (np.exp(logits).sum(axis=-1, keepdims=True) + 1e-10)
        # Entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=-1)
        return entropy


def load_dscript_model(device: str = "cpu"):
    """Load D-SCRIPT model."""
    from dscript.models.interaction import DSCRIPTModel
    use_cuda = device == "cuda" and torch.cuda.is_available()
    model = DSCRIPTModel.from_pretrained("samsl/topsy_turvy_human_v1", use_cuda=use_cuda)
    if use_cuda:
        model = model.cuda()
        model.use_cuda = True
    else:
        model = model.cpu()
        model.use_cuda = False
    model.eval()
    return model, use_cuda


def embed_sequence(sequence: str, use_cuda: bool = False) -> torch.Tensor:
    """Embed a protein sequence."""
    from dscript.language_model import lm_embed
    return lm_embed(sequence, use_cuda=use_cuda)


def collect_data_for_ood(
    model,
    pairs: List[Tuple[str, str, float]],
    sequences: Dict[str, str],
    use_cuda: bool = False
) -> Dict[str, np.ndarray]:
    """Collect predictions and embeddings for OOD evaluation."""
    predictions = []
    labels = []
    logits_list = []
    emb_cache = {}
    pair_embeddings = []

    for p_a, p_b, label in tqdm(pairs, desc="Collecting data"):
        p_a, p_b = str(p_a), str(p_b)

        if p_a not in sequences or p_b not in sequences:
            continue

        seq_a = sequences[p_a]
        seq_b = sequences[p_b]

        if seq_a not in emb_cache:
            emb_cache[seq_a] = embed_sequence(seq_a, use_cuda)
        if seq_b not in emb_cache:
            emb_cache[seq_b] = embed_sequence(seq_b, use_cuda)

        emb_a = emb_cache[seq_a]
        emb_b = emb_cache[seq_b]

        if use_cuda:
            emb_a = emb_a.cuda()
            emb_b = emb_b.cuda()

        with torch.no_grad():
            pred = model.predict(emb_a, emb_b).item()

        # Convert to logits format [logit_0, logit_1]
        # From probability p, logit = log(p / (1-p))
        eps = 1e-7
        pred_clipped = np.clip(pred, eps, 1 - eps)
        logit_1 = np.log(pred_clipped / (1 - pred_clipped))
        logits = np.array([0.0, logit_1])  # logit_0 = 0, logit_1 varies

        # Average pair embedding (simplified)
        emb_a_np = emb_a.mean(dim=1).cpu().numpy().flatten()
        emb_b_np = emb_b.mean(dim=1).cpu().numpy().flatten()
        pair_emb = (emb_a_np + emb_b_np) / 2

        predictions.append(pred)
        labels.append(float(label))
        logits_list.append(logits)
        pair_embeddings.append(pair_emb)

    return {
        'predictions': np.array(predictions),
        'labels': np.array(labels),
        'logits': np.array(logits_list),
        'embeddings': np.array(pair_embeddings)
    }


def create_ood_samples(
    in_dist_data: Dict[str, np.ndarray],
    method: str = 'noise'
) -> Dict[str, np.ndarray]:
    """Create synthetic OOD samples for evaluation."""
    n_samples = len(in_dist_data['predictions'])

    if method == 'noise':
        # Add noise to embeddings
        ood_embeddings = in_dist_data['embeddings'] + np.random.randn(
            *in_dist_data['embeddings'].shape
        ) * 2.0
        # Random logits
        ood_logits = np.random.randn(n_samples, 2) * 2
    elif method == 'uniform':
        # Completely random
        emb_dim = in_dist_data['embeddings'].shape[1]
        ood_embeddings = np.random.randn(n_samples, emb_dim) * 3
        ood_logits = np.random.randn(n_samples, 2) * 2
    elif method == 'boundary':
        # Near decision boundary (uncertain predictions)
        ood_embeddings = in_dist_data['embeddings'].copy()
        # Logits near 0 (50% probability)
        ood_logits = np.random.randn(n_samples, 2) * 0.1
    else:
        raise ValueError(f"Unknown OOD method: {method}")

    return {
        'embeddings': ood_embeddings,
        'logits': ood_logits,
        'is_ood': np.ones(n_samples)
    }


def evaluate_ood_methods(
    in_dist_data: Dict[str, np.ndarray],
    ood_data: Dict[str, np.ndarray],
    detectors: Dict[str, OODDetector]
) -> Dict[str, Dict]:
    """Evaluate all OOD detection methods."""
    results = {}

    # Combine data
    all_embeddings = np.vstack([in_dist_data['embeddings'], ood_data['embeddings']])
    all_logits = np.vstack([in_dist_data['logits'], ood_data['logits']])
    ood_labels = np.concatenate([
        np.zeros(len(in_dist_data['embeddings'])),  # in-distribution
        np.ones(len(ood_data['embeddings']))        # OOD
    ])

    for name, detector in detectors.items():
        try:
            # Fit on in-distribution data
            if name in ['mahalanobis', 'knn']:
                detector.fit(in_dist_data['embeddings'])
                ood_scores = detector.score(all_embeddings)
            else:
                detector.fit(in_dist_data['logits'])
                ood_scores = detector.score(all_logits)

            # Compute AUROC
            auroc = roc_auc_score(ood_labels, ood_scores)

            results[name] = {
                'auroc': float(auroc),
                'mean_in_dist': float(ood_scores[:len(in_dist_data['embeddings'])].mean()),
                'mean_ood': float(ood_scores[len(in_dist_data['embeddings']):].mean()),
                'std_in_dist': float(ood_scores[:len(in_dist_data['embeddings'])].std()),
                'std_ood': float(ood_scores[len(in_dist_data['embeddings']):].std())
            }
        except Exception as e:
            logger.warning(f"Failed for {name}: {e}")
            results[name] = {'auroc': 0.5, 'error': str(e)}

    return results


def main():
    parser = argparse.ArgumentParser(description="Alternative OOD Methods Evaluation")
    parser.add_argument('--data', type=str, default='yeast', help='Dataset')
    parser.add_argument('--limit', type=int, default=None, help='Limit samples')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--quick', action='store_true', help='Quick test (50 samples)')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory')

    args = parser.parse_args()

    if args.quick:
        args.limit = 50
        logger.info("Quick test mode (50 samples)")

    output_dir = args.output_dir or (PROJECT_ROOT / "experiments" / "results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info("Loading D-SCRIPT model...")
    model, use_cuda = load_dscript_model(device=args.device)

    # Load data
    logger.info(f"Loading {args.data} dataset...")
    pairs_file, seqs_file = get_dataset_paths(args.data)
    pairs = load_pairs(pairs_file, limit=args.limit, balanced=True)
    sequences = load_sequences_for_pairs(pairs, seqs_file)

    logger.info(f"  Pairs: {len(pairs)}, Sequences: {len(sequences)}")

    # Collect in-distribution data
    logger.info("Collecting in-distribution data...")
    in_dist_data = collect_data_for_ood(model, pairs, sequences, use_cuda)

    # Define OOD detectors
    detectors = {
        'energy': EnergyOOD(temperature=1.0),
        'max_softmax': MaxSoftmaxOOD(),
        'entropy': EntropyOOD(),
        'mahalanobis': MahalanobisOOD(),
        'knn': KNNDistanceOOD(k=min(5, len(pairs) // 2))
    }

    # Evaluate on different OOD types
    ood_types = ['noise', 'uniform', 'boundary']
    all_results = {}

    for ood_type in ood_types:
        logger.info(f"\nEvaluating on {ood_type} OOD samples...")
        ood_data = create_ood_samples(in_dist_data, method=ood_type)
        results = evaluate_ood_methods(in_dist_data, ood_data, detectors)
        all_results[ood_type] = results

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("ALTERNATIVE OOD METHODS RESULTS")
    logger.info("=" * 70)

    for ood_type, results in all_results.items():
        logger.info(f"\n{ood_type.upper()} OOD Type:")
        logger.info("-" * 50)
        logger.info("Method         | AUROC  | In-Dist Mean | OOD Mean")
        logger.info("-" * 50)

        for method, metrics in sorted(results.items(), key=lambda x: -x[1].get('auroc', 0)):
            if 'error' not in metrics:
                logger.info(
                    f"{method:14s} | {metrics['auroc']:.3f}  | "
                    f"{metrics['mean_in_dist']:+.3f}       | {metrics['mean_ood']:+.3f}"
                )
            else:
                logger.info(f"{method:14s} | ERROR: {metrics['error']}")

    # Overall comparison
    logger.info("\n" + "-" * 70)
    logger.info("Overall Method Ranking (Mean AUROC across OOD types):")
    logger.info("-" * 70)

    method_means = {}
    for method in detectors.keys():
        aurocs = [all_results[ood_type][method].get('auroc', 0.5) for ood_type in ood_types]
        method_means[method] = np.mean(aurocs)

    for method, mean_auroc in sorted(method_means.items(), key=lambda x: -x[1]):
        bar = "█" * int(mean_auroc * 20)
        logger.info(f"  {method:14s}: {mean_auroc:.3f} {bar}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = output_dir / f"alt_ood_eval_{args.data}_{timestamp}.json"

    output = {
        'experiment': 'alternative_ood_methods',
        'dataset': args.data,
        'n_samples': len(in_dist_data['predictions']),
        'timestamp': timestamp,
        'results_by_ood_type': all_results,
        'method_ranking': method_means
    }

    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
