"""
Trust-Guided Selection Comparison (Option B3).

Compares acquisition strategies for design-loop candidate selection:
  1. Random: Uniform selection
  2. Confidence: Select high-confidence predictions
  3. Deformation: Select predictions with high deformation stability
  4. Combined: Trust-weighted acquisition (confidence × deformation)

Measures: queries-to-top-10%, coverage, efficiency curves.

Usage:
    python -m experiments.eval_trust_selection --data yeast --quick
    python -m experiments.eval_trust_selection --data ecoli --limit 500 --seed 42
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.trustppi.data import get_dataset_paths, load_pairs, load_sequences_for_pairs

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# D-SCRIPT Prediction + Deformation
# ---------------------------------------------------------------------------

def load_dscript(device):
    """Load D-SCRIPT model."""
    from dscript.models.interaction import DSCRIPTModel
    use_cuda = device.type == "cuda" if hasattr(device, 'type') else str(device) == "cuda"
    model = DSCRIPTModel.from_pretrained("samsl/topsy_turvy_human_v1", use_cuda=use_cuda)
    if use_cuda:
        model = model.cuda()
        model.use_cuda = True
    else:
        model = model.cpu()
        model.use_cuda = False
    model.eval()
    return model


def collect_predictions(
    model,
    pairs: List[Tuple[str, str, float]],
    sequences: Dict[str, str],
    use_cuda: bool,
    noise_std: float = 0.1,
    n_deform: int = 10,
) -> Dict[str, np.ndarray]:
    """Collect predictions and trust signals for all pairs."""
    import torch
    from dscript.language_model import lm_embed

    predictions = []
    labels = []
    confidences = []
    deform_scores = []
    combined_scores = []

    for prot_a, prot_b, label in tqdm(pairs, desc="Collecting signals"):
        seq_a = sequences.get(str(prot_a))
        seq_b = sequences.get(str(prot_b))
        if seq_a is None or seq_b is None:
            continue

        try:
            with torch.no_grad():
                emb_a = lm_embed(seq_a, use_cuda=use_cuda)
                emb_b = lm_embed(seq_b, use_cuda=use_cuda)
                if use_cuda:
                    emb_a = emb_a.cuda()
                    emb_b = emb_b.cuda()

                prob = float(model.predict(emb_a, emb_b).item())

            # Confidence
            conf = max(prob, 1 - prob)

            # Deformation stability
            perturbed = []
            for _ in range(n_deform):
                na = torch.randn_like(emb_a) * noise_std
                nb = torch.randn_like(emb_b) * noise_std
                with torch.no_grad():
                    p = float(model.predict(emb_a + na, emb_b + nb).item())
                perturbed.append(p)

            pred_std = float(np.std(perturbed))
            flip_rate = float(np.mean([(p > 0.5) != (prob > 0.5) for p in perturbed]))
            stability = 1.0 / (1.0 + pred_std)
            flip_score = 1.0 - flip_rate
            deform = (stability + flip_score) / 2

            predictions.append(prob)
            labels.append(float(label))
            confidences.append(conf)
            deform_scores.append(deform)
            combined_scores.append(conf * deform)

        except Exception:
            continue

    return {
        'predictions': np.array(predictions),
        'labels': np.array(labels),
        'confidence': np.array(confidences),
        'deformation': np.array(deform_scores),
        'combined': np.array(combined_scores),
    }


# ---------------------------------------------------------------------------
# Selection Strategies
# ---------------------------------------------------------------------------

def random_selection(
    n_total: int,
    batch_size: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Random acquisition."""
    return rng.choice(n_total, size=min(batch_size, n_total), replace=False)


def score_based_selection(
    scores: np.ndarray,
    batch_size: int,
    already_selected: set,
) -> np.ndarray:
    """Select top-scoring candidates not yet selected."""
    available = np.array([i for i in range(len(scores)) if i not in already_selected])
    if len(available) == 0:
        return np.array([], dtype=int)

    available_scores = scores[available]
    n_select = min(batch_size, len(available))
    top_idx = np.argsort(available_scores)[-n_select:]
    return available[top_idx]


# ---------------------------------------------------------------------------
# Design Loop Simulation
# ---------------------------------------------------------------------------

def simulate_design_loop(
    signals: Dict[str, np.ndarray],
    strategy: str,
    n_rounds: int = 10,
    batch_size: int = 20,
    seed: int = 42,
) -> Dict:
    """
    Simulate adaptive design loop with a given selection strategy.

    Returns per-round metrics: coverage of true positives, accuracy, etc.
    """
    rng = np.random.RandomState(seed)
    n_total = len(signals['predictions'])
    labels = signals['labels']
    predictions = signals['predictions']

    # True positives: label==1 with correct prediction
    true_labels = labels.astype(int)

    selected = set()
    rounds = []

    for round_idx in range(n_rounds):
        if len(selected) >= n_total:
            break

        # Select batch
        if strategy == 'random':
            available = np.array([i for i in range(n_total) if i not in selected])
            if len(available) == 0:
                break
            n_select = min(batch_size, len(available))
            batch_idx = rng.choice(available, size=n_select, replace=False)
        else:
            score_key = strategy  # 'confidence', 'deformation', 'combined'
            batch_idx = score_based_selection(
                signals[score_key], batch_size, selected
            )

        if len(batch_idx) == 0:
            break

        selected.update(batch_idx.tolist())
        selected_arr = np.array(sorted(selected))

        # Metrics on selected set so far
        sel_preds = (predictions[selected_arr] > 0.5).astype(int)
        sel_labels = true_labels[selected_arr]
        accuracy = float((sel_preds == sel_labels).mean())

        # How many true positives found so far?
        total_positives = int(true_labels.sum())
        found_positives = int(sel_labels.sum())
        positive_recall = found_positives / max(total_positives, 1)

        # Precision among selected
        selected_positive_preds = (predictions[selected_arr] > 0.5).sum()
        precision = float(sel_labels[predictions[selected_arr] > 0.5].mean()) \
            if selected_positive_preds > 0 else 0.0

        rounds.append({
            'round': round_idx + 1,
            'n_selected': len(selected),
            'n_batch': len(batch_idx),
            'accuracy': accuracy,
            'positive_recall': positive_recall,
            'found_positives': found_positives,
            'total_positives': total_positives,
            'precision': precision,
        })

    # Efficiency metric: queries to find top-10% of true positives
    top10_target = max(1, int(0.1 * true_labels.sum()))
    queries_to_top10 = n_total  # default: need all

    cumulative_positives = 0
    for r in rounds:
        cumulative_positives = r['found_positives']
        if cumulative_positives >= top10_target:
            queries_to_top10 = r['n_selected']
            break

    return {
        'strategy': strategy,
        'rounds': rounds,
        'final_accuracy': rounds[-1]['accuracy'] if rounds else 0,
        'final_recall': rounds[-1]['positive_recall'] if rounds else 0,
        'queries_to_top10_pct': queries_to_top10,
        'efficiency': (n_total - queries_to_top10) / max(n_total, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Trust-Guided Selection Comparison"
    )
    parser.add_argument('--data', type=str, default='yeast',
                        choices=['yeast', 'human', 'ecoli', 'mouse', 'fly', 'worm'])
    parser.add_argument('--limit', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test (100 samples, 5 rounds)')
    parser.add_argument('--n-rounds', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--noise-std', type=float, default=0.1)
    parser.add_argument('--n-deform', type=int, default=10)
    parser.add_argument('--output-dir', type=Path,
                        default=PROJECT_ROOT / 'experiments' / 'results')

    args = parser.parse_args()

    import torch
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.quick:
        args.limit = 100
        args.n_rounds = 5
        args.n_deform = 5
        logger.info("Quick mode: 100 samples, 5 rounds")

    device = torch.device(
        args.device if args.device != 'cuda' or torch.cuda.is_available() else 'cpu'
    )
    use_cuda = device.type == 'cuda'

    # Load model and data
    logger.info("Loading D-SCRIPT model...")
    model = load_dscript(device)

    pairs_file, seqs_file = get_dataset_paths(args.data)
    pairs = load_pairs(pairs_file, limit=args.limit, balanced=True, seed=args.seed)
    sequences = load_sequences_for_pairs(pairs, seqs_file)
    logger.info(f"Loaded {len(pairs)} pairs")

    # Collect all signals
    logger.info("Collecting predictions and trust signals...")
    signals = collect_predictions(
        model, pairs, sequences, use_cuda,
        noise_std=args.noise_std,
        n_deform=args.n_deform,
    )
    logger.info(f"Collected signals for {len(signals['predictions'])} samples")

    # Simulate design loops
    strategies = ['random', 'confidence', 'deformation', 'combined']
    all_results = {}

    logger.info("\nSimulating design loops...")
    for strategy in strategies:
        result = simulate_design_loop(
            signals, strategy,
            n_rounds=args.n_rounds,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        all_results[strategy] = result
        logger.info(
            f"  {strategy:>12s}: queries_to_top10={result['queries_to_top10_pct']}, "
            f"efficiency={result['efficiency']:.1%}, "
            f"final_recall={result['final_recall']:.1%}"
        )

    # Print comparison table
    print(f"\n{'Strategy':>12s} {'Queries→10%':>12s} {'Efficiency':>10s} "
          f"{'Final Acc':>10s} {'Final Recall':>12s}")
    print("-" * 60)
    for strategy in strategies:
        r = all_results[strategy]
        print(f"{strategy:>12s} {r['queries_to_top10_pct']:>12d} "
              f"{r['efficiency']:>10.1%} "
              f"{r['final_accuracy']:>10.3f} "
              f"{r['final_recall']:>12.1%}")

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output_dir / f"trust_selection_{args.data}_seed{args.seed}_{timestamp}.json"

    output = {
        'experiment': 'trust_guided_selection',
        'dataset': args.data,
        'seed': args.seed,
        'n_samples': len(signals['predictions']),
        'n_rounds': args.n_rounds,
        'batch_size': args.batch_size,
        'timestamp': timestamp,
        'baseline_accuracy': float(((signals['predictions'] > 0.5).astype(int)
                                     == signals['labels'].astype(int)).mean())
                             if len(signals['predictions']) > 0 else 0,
        'selection_comparison': all_results,
        'notes': (
            f"Trust-guided selection comparison on D-SCRIPT ({args.data}). "
            f"Strategies: random, confidence, deformation stability, combined. "
            f"{args.n_rounds} rounds of {args.batch_size} candidates each."
        ),
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
