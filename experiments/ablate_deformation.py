"""
Ablation study for deformation stability parameters.

Grid searches over noise_std and n_perturbations to find optimal settings
and demonstrate parameter sensitivity.

Usage:
    python -m experiments.ablate_deformation --data yeast --quick
    python -m experiments.ablate_deformation --data ecoli --limit 200 --seed 42
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
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


# Ablation grid
NOISE_STDS = [0.01, 0.05, 0.1, 0.2, 0.5]
N_PERTURBATIONS = [3, 5, 10, 20, 50]


# ---------------------------------------------------------------------------
# D-SCRIPT Model Loading (default ablation target)
# ---------------------------------------------------------------------------

def load_dscript_model(device: torch.device):
    """Load D-SCRIPT model for ablation."""
    try:
        from dscript.models.interaction import DSCRIPTModel
    except ImportError:
        raise ImportError(
            "D-SCRIPT not installed. Run: pip install dscript\n"
            "Or use --model plm-interact for PLM-interact ablation."
        )

    use_cuda = device.type == "cuda"
    model = DSCRIPTModel.from_pretrained("samsl/topsy_turvy_human_v1", use_cuda=use_cuda)
    if use_cuda:
        model = model.cuda()
        model.use_cuda = True
    else:
        model = model.cpu()
        model.use_cuda = False
    model.eval()
    return model


def dscript_predict(model, seq_a: str, seq_b: str, use_cuda: bool) -> float:
    """Get D-SCRIPT prediction probability."""
    from dscript.language_model import lm_embed

    with torch.no_grad():
        emb_a = lm_embed(seq_a, use_cuda=use_cuda)
        emb_b = lm_embed(seq_b, use_cuda=use_cuda)
        if use_cuda:
            emb_a = emb_a.cuda()
            emb_b = emb_b.cuda()
        prob = model.predict(emb_a, emb_b).item()
    return prob


def dscript_deformation_stability(
    model,
    seq_a: str,
    seq_b: str,
    use_cuda: bool,
    noise_std: float,
    n_perturbations: int,
) -> Dict[str, float]:
    """Compute deformation stability with specific parameters."""
    from dscript.language_model import lm_embed

    with torch.no_grad():
        emb_a = lm_embed(seq_a, use_cuda=use_cuda)
        emb_b = lm_embed(seq_b, use_cuda=use_cuda)
        if use_cuda:
            emb_a = emb_a.cuda()
            emb_b = emb_b.cuda()

    # Base prediction
    with torch.no_grad():
        base_pred = model.predict(emb_a, emb_b).item()

    # Perturbed predictions
    perturbed = []
    for _ in range(n_perturbations):
        noise_a = torch.randn_like(emb_a) * noise_std
        noise_b = torch.randn_like(emb_b) * noise_std

        with torch.no_grad():
            p = model.predict(emb_a + noise_a, emb_b + noise_b).item()
        perturbed.append(p)

    perturbed = np.array(perturbed)

    pred_std = float(np.std(perturbed))
    flip_rate = float(np.mean([(p > 0.5) != (base_pred > 0.5) for p in perturbed]))

    stability = 1.0 / (1.0 + pred_std)
    flip_score = 1.0 - flip_rate
    combined = (stability + flip_score) / 2

    return {
        't_deform_stable': stability,
        't_deform_flip': flip_score,
        't_deform_combined': combined,
        'pred_std': pred_std,
        'flip_rate': flip_rate,
    }


# ---------------------------------------------------------------------------
# Ablation Runner
# ---------------------------------------------------------------------------

def run_single_ablation(
    model,
    pairs: List[Tuple[str, str, float]],
    sequences: Dict[str, str],
    use_cuda: bool,
    noise_std: float,
    n_perturbations: int,
) -> Dict[str, float]:
    """Run one ablation configuration and return metrics."""
    stabilities = []
    flip_scores = []
    combined_scores = []
    is_correct_list = []

    for prot_a, prot_b, label in pairs:
        seq_a = sequences.get(str(prot_a))
        seq_b = sequences.get(str(prot_b))
        if seq_a is None or seq_b is None:
            continue

        try:
            # Base prediction
            prob = dscript_predict(model, seq_a, seq_b, use_cuda)
            correct = int((prob > 0.5) == (label > 0.5))

            # Deformation
            deform = dscript_deformation_stability(
                model, seq_a, seq_b, use_cuda,
                noise_std=noise_std,
                n_perturbations=n_perturbations,
            )

            stabilities.append(deform['t_deform_stable'])
            flip_scores.append(deform['t_deform_flip'])
            combined_scores.append(deform['t_deform_combined'])
            is_correct_list.append(correct)

        except Exception as e:
            logger.debug(f"Skipping pair: {e}")
            continue

    if len(is_correct_list) < 10:
        return {'stable_auroc': 0.5, 'flip_auroc': 0.5, 'combined_auroc': 0.5, 'n': 0}

    is_correct = np.array(is_correct_list)

    def safe_auroc(scores):
        try:
            return float(roc_auc_score(is_correct, np.array(scores)))
        except Exception:
            return 0.5

    return {
        'stable_auroc': safe_auroc(stabilities),
        'flip_auroc': safe_auroc(flip_scores),
        'combined_auroc': safe_auroc(combined_scores),
        'mean_stability': float(np.mean(stabilities)),
        'mean_flip_score': float(np.mean(flip_scores)),
        'n': len(is_correct_list),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Ablation study for deformation stability parameters"
    )
    parser.add_argument('--data', type=str, default='yeast',
                        choices=['yeast', 'human', 'ecoli', 'mouse', 'fly', 'worm'])
    parser.add_argument('--limit', type=int, default=200,
                        help='Number of test samples')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test (50 samples, reduced grid)')
    parser.add_argument('--output-dir', type=Path,
                        default=PROJECT_ROOT / 'experiments' / 'results')

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    if args.quick:
        args.limit = 50
        noise_stds = [0.05, 0.1, 0.2]
        n_perts = [5, 10, 20]
        logger.info("Quick mode: reduced grid and 50 samples")
    else:
        noise_stds = NOISE_STDS
        n_perts = N_PERTURBATIONS

    device = torch.device(
        args.device if args.device != 'cuda' or torch.cuda.is_available() else 'cpu'
    )
    use_cuda = device.type == 'cuda'

    # Load model
    logger.info("Loading D-SCRIPT model...")
    model = load_dscript_model(device)

    # Load data
    pairs_file, seqs_file = get_dataset_paths(args.data)
    pairs = load_pairs(pairs_file, limit=args.limit, balanced=True, seed=args.seed)
    sequences = load_sequences_for_pairs(pairs, seqs_file)
    logger.info(f"Loaded {len(pairs)} pairs, {len(sequences)} sequences")

    # Run grid search
    results = []
    total = len(noise_stds) * len(n_perts)

    logger.info(f"Running {total} ablation configurations...")
    for noise_std, n_pert in tqdm(list(product(noise_stds, n_perts)),
                                   desc="Ablation grid"):
        logger.info(f"  noise_std={noise_std}, n_perturbations={n_pert}")

        metrics = run_single_ablation(
            model, pairs, sequences, use_cuda,
            noise_std=noise_std,
            n_perturbations=n_pert,
        )

        entry = {
            'noise_std': noise_std,
            'n_perturbations': n_pert,
            **metrics,
        }
        results.append(entry)

        logger.info(f"    combined_auroc={metrics['combined_auroc']:.3f}, "
                     f"n={metrics['n']}")

    # Find best configuration
    best = max(results, key=lambda x: x['combined_auroc'])
    logger.info(f"\nBest: noise_std={best['noise_std']}, "
                f"n_pert={best['n_perturbations']}, "
                f"AUROC={best['combined_auroc']:.3f}")

    # Default comparison
    default = next(
        (r for r in results if r['noise_std'] == 0.1 and r['n_perturbations'] == 10),
        None
    )

    # Print table
    print(f"\n{'noise_std':>10} {'n_pert':>8} {'stable':>8} {'flip':>8} {'combined':>10}")
    print("-" * 50)
    for r in results:
        marker = " *" if r == best else ""
        print(f"{r['noise_std']:>10.2f} {r['n_perturbations']:>8d} "
              f"{r['stable_auroc']:>8.3f} {r['flip_auroc']:>8.3f} "
              f"{r['combined_auroc']:>10.3f}{marker}")

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output_dir / f"deform_ablation_{args.data}_seed{args.seed}_{timestamp}.json"

    output = {
        'experiment': 'deformation_parameter_ablation',
        'dataset': args.data,
        'seed': args.seed,
        'n_samples': len(pairs),
        'timestamp': timestamp,
        'grid': {
            'noise_stds': noise_stds,
            'n_perturbations': n_perts,
        },
        'results': results,
        'best': best,
        'default': default,
        'notes': (
            f"Grid search over noise_std and n_perturbations for deformation stability. "
            f"Model: D-SCRIPT on {args.data} ({len(pairs)} pairs)."
        ),
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
