"""
Experiment: Protein-Specific Trust Signals Evaluation.

Tests the protein-specific trust signals:
1. Interface Confidence - from D-SCRIPT contact maps
2. Deformation Stability - via embedding perturbation

Usage:
    python -m experiments.eval_dscript_trust --quick
    python -m experiments.eval_dscript_trust --data yeast --limit 100
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "external" / "D-SCRIPT"))

from src.trustppi.data import load_pairs, load_sequences_for_pairs, get_dataset_paths
from src.trustppi.trust.metrics import selective_accuracy_at_coverage
from src.trustppi.trust.interface_confidence import InterfaceConfidence
from src.trustppi.trust.deformation_stability import EmbeddingDeformationStability

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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


def compute_contact_map_dscript(model, emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
    """
    Extract the actual contact map from D-SCRIPT.

    D-SCRIPT's cpred() method returns the real contact map [b, 1, L_A, L_B].
    """
    with torch.no_grad():
        # Use D-SCRIPT's actual contact map prediction
        # cpred returns [batch, 1, L_A, L_B] sigmoid-activated contact probabilities
        contact_map = model.cpred(emb_a, emb_b)

    # Remove batch and channel dimensions: [1, 1, L_A, L_B] -> [L_A, L_B]
    return contact_map.squeeze(0).squeeze(0)


def collect_protein_trust_signals(
    model,
    pairs: List[Tuple[str, str, float]],
    sequences: Dict[str, str],
    use_cuda: bool = False,
    interface_conf: InterfaceConfidence = None,
    emb_deform: EmbeddingDeformationStability = None
) -> Dict[str, np.ndarray]:
    """Collect all protein-specific trust signals."""
    if interface_conf is None:
        interface_conf = InterfaceConfidence()
    if emb_deform is None:
        # Increased noise_std from 0.01 to 0.1 for better discrimination
        emb_deform = EmbeddingDeformationStability(noise_std=0.1, n_samples=10)

    predictions = []
    labels = []
    confidences = []

    # Interface signals
    interface_sharp = []
    interface_size = []
    interface_consist = []
    interface_combined = []

    # Deformation signals
    deform_stable = []
    deform_flip = []
    deform_combined = []

    emb_cache = {}

    for p_a, p_b, label in tqdm(pairs, desc="Collecting signals"):
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
            # Base prediction
            pred = model.predict(emb_a, emb_b).item()

        # Confidence
        conf = max(pred, 1 - pred)

        # Interface confidence signals
        try:
            contact_map = compute_contact_map_dscript(model, emb_a, emb_b)
            interface_signals = interface_conf.compute(contact_map)
            # Note: Interface signals empirically anti-correlate with correctness
            # for D-SCRIPT. This suggests that "sharper" interface predictions
            # are actually associated with MORE errors (overconfident wrong predictions).
            # We invert them so higher = more trustworthy.
            interface_sharp.append(1.0 - interface_signals['t_interface_sharp'])
            interface_size.append(1.0 - interface_signals['t_interface_size'])
            interface_consist.append(1.0 - interface_signals['t_interface_consist'])
            interface_combined.append(1.0 - interface_signals['t_interface_combined'])
        except Exception as e:
            logger.warning(f"Interface computation failed: {e}")
            interface_sharp.append(0.5)
            interface_size.append(0.5)
            interface_consist.append(0.5)
            interface_combined.append(0.5)

        # Embedding deformation signals
        try:
            def model_fn(ea, eb):
                return model.predict(ea, eb)

            deform_signals = emb_deform.compute(model_fn, emb_a, emb_b)
            deform_stable.append(deform_signals['t_emb_stable'])
            deform_flip.append(deform_signals['t_emb_flip'])
            deform_combined.append(deform_signals['t_emb_combined'])
        except Exception as e:
            logger.warning(f"Deformation computation failed: {e}")
            deform_stable.append(0.5)
            deform_flip.append(0.5)
            deform_combined.append(0.5)

        predictions.append(pred)
        labels.append(float(label))
        confidences.append(conf)

    return {
        'predictions': np.array(predictions),
        'labels': np.array(labels),
        'confidence': np.array(confidences),
        # Interface signals
        't_interface_sharp': np.array(interface_sharp),
        't_interface_size': np.array(interface_size),
        't_interface_consist': np.array(interface_consist),
        't_interface_combined': np.array(interface_combined),
        # Deformation signals
        't_deform_stable': np.array(deform_stable),
        't_deform_flip': np.array(deform_flip),
        't_deform_combined': np.array(deform_combined)
    }


def evaluate_signal_for_error_detection(
    signals: Dict[str, np.ndarray],
    signal_name: str
) -> Dict[str, float]:
    """Evaluate how well a signal detects errors."""
    predictions = (signals['predictions'] > 0.5).astype(int)
    labels = signals['labels'].astype(int)
    is_correct = (predictions == labels).astype(int)

    signal = signals[signal_name]

    # AUROC: Does higher signal correlate with correctness?
    try:
        auroc = roc_auc_score(is_correct, signal)
    except:
        auroc = 0.5

    # Correlation
    correlation = np.corrcoef(signal, is_correct)[0, 1]
    if np.isnan(correlation):
        correlation = 0.0

    return {
        'auroc': float(auroc),
        'correlation': float(correlation),
        'mean_correct': float(signal[is_correct == 1].mean()) if (is_correct == 1).sum() > 0 else 0,
        'mean_incorrect': float(signal[is_correct == 0].mean()) if (is_correct == 0).sum() > 0 else 0
    }


def evaluate_selective_prediction(
    signals: Dict[str, np.ndarray],
    signal_name: str,
    target_coverage: float = 0.8
) -> Dict[str, float]:
    """Evaluate selective prediction using a trust signal."""
    predictions = (signals['predictions'] > 0.5).astype(int)
    labels = signals['labels'].astype(int)
    signal = signals[signal_name]

    metrics = selective_accuracy_at_coverage(
        signal, predictions, labels, target_coverage=target_coverage
    )

    return {
        'coverage': metrics.coverage,
        'accuracy': metrics.accuracy,
        'improvement': metrics.improvement,
        'n_accepted': metrics.n_accepted
    }


def main():
    parser = argparse.ArgumentParser(description="Protein-Specific Trust Signals Evaluation")
    parser.add_argument('--data', type=str, default='yeast', help='Dataset')
    parser.add_argument('--limit', type=int, default=None, help='Limit samples')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--quick', action='store_true', help='Quick test (30 samples)')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    logger.info(f"Random seed: {args.seed}")

    if args.quick:
        args.limit = 30
        logger.info("Quick test mode (30 samples)")

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

    # Collect protein-specific trust signals
    logger.info("Collecting protein-specific trust signals...")
    signals = collect_protein_trust_signals(model, pairs, sequences, use_cuda)
    logger.info(f"  Got signals for {len(signals['predictions'])} samples")

    # Baseline
    predictions = (signals['predictions'] > 0.5).astype(int)
    labels = signals['labels'].astype(int)
    baseline_acc = (predictions == labels).mean()
    logger.info(f"\nBaseline accuracy: {baseline_acc:.3f}")

    # Evaluate each signal for error detection
    logger.info("\n" + "=" * 70)
    logger.info("ERROR DETECTION EVALUATION (AUROC)")
    logger.info("=" * 70)

    signal_names = [
        'confidence',
        't_interface_sharp', 't_interface_size', 't_interface_consist', 't_interface_combined',
        't_deform_stable', 't_deform_flip', 't_deform_combined'
    ]

    error_detection_results = {}
    for signal_name in signal_names:
        result = evaluate_signal_for_error_detection(signals, signal_name)
        error_detection_results[signal_name] = result
        logger.info(f"  {signal_name:25s}: AUROC={result['auroc']:.3f}, corr={result['correlation']:+.3f}")

    # Evaluate selective prediction
    logger.info("\n" + "=" * 70)
    logger.info("SELECTIVE PREDICTION EVALUATION (80% Coverage)")
    logger.info("=" * 70)

    selective_results = {}
    for signal_name in signal_names:
        result = evaluate_selective_prediction(signals, signal_name, target_coverage=0.8)
        selective_results[signal_name] = result
        logger.info(f"  {signal_name:25s}: Acc={result['accuracy']:.3f}, Imp={result['improvement']:+.1%}")

    # Combined signals evaluation
    logger.info("\n" + "=" * 70)
    logger.info("COMBINED SIGNALS EVALUATION")
    logger.info("=" * 70)

    # Combine generic + protein-specific
    combined_trust = (
        0.3 * signals['confidence'] +
        0.35 * signals['t_interface_combined'] +
        0.35 * signals['t_deform_combined']
    )

    # Evaluate combined
    combined_metrics = selective_accuracy_at_coverage(
        combined_trust, predictions, labels, target_coverage=0.8
    )
    logger.info(f"  Combined (conf+interface+deform): Acc={combined_metrics.accuracy:.3f}, Imp={combined_metrics.improvement:+.1%}")

    # Just protein-specific
    protein_only_trust = 0.5 * signals['t_interface_combined'] + 0.5 * signals['t_deform_combined']
    protein_metrics = selective_accuracy_at_coverage(
        protein_only_trust, predictions, labels, target_coverage=0.8
    )
    logger.info(f"  Protein-only (interface+deform): Acc={protein_metrics.accuracy:.3f}, Imp={protein_metrics.improvement:+.1%}")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    # Best protein signal
    protein_signals = ['t_interface_combined', 't_deform_combined']
    best_protein = max(protein_signals, key=lambda s: selective_results[s]['improvement'])
    best_protein_imp = selective_results[best_protein]['improvement']

    logger.info(f"  Baseline accuracy: {baseline_acc:.3f}")
    logger.info(f"  Best generic signal (confidence): {selective_results['confidence']['improvement']:+.1%}")
    logger.info(f"  Best protein signal ({best_protein}): {best_protein_imp:+.1%}")
    logger.info(f"  Combined improvement: {combined_metrics.improvement:+.1%}")

    # Success criteria
    protein_helps = combined_metrics.improvement > selective_results['confidence']['improvement']
    logger.info(f"\n  Protein signals add value over generic: {'✓ YES' if protein_helps else '✗ NO'}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = output_dir / f"protein_trust_eval_{args.data}_seed{args.seed}_{timestamp}.json"

    output = {
        'experiment': 'protein_specific_trust',
        'dataset': args.data,
        'seed': args.seed,
        'n_samples': len(signals['predictions']),
        'baseline_accuracy': float(baseline_acc),
        'timestamp': timestamp,
        'error_detection': error_detection_results,
        'selective_prediction': selective_results,
        'combined_improvement': combined_metrics.improvement,
        'protein_only_improvement': protein_metrics.improvement,
        'protein_adds_value': protein_helps
    }

    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
