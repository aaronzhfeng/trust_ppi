"""
Experiment: TUnA Cross-Model Trust Evaluation.

Evaluates trust signals on TUnA (Transformer + Gaussian Process head),
which has native GP uncertainty. Key question: does our deformation
stability signal add value beyond TUnA's built-in GP variance?

Signals evaluated:
  - confidence: max(p, 1-p) — generic baseline
  - gp_variance: TUnA's native GP uncertainty (higher = less certain)
    For error detection we use 1-variance (higher = more trustworthy)
  - t_deform_stable, t_deform_flip, t_deform_combined: embedding
    perturbation stability (Gaussian noise on ESM2 embeddings)

Requires:
  - ESM2 embeddings for test proteins (generated on first run, cached)
  - TUnA pretrained weights (external/TUnA/results/xspecies/TUnA_seed47/output/model)
  - uncertaintyAwareDeepLearn package (VanillaRFFLayer)

Usage:
    python -m experiments.eval_tuna_trust --quick --seed 42
    python -m experiments.eval_tuna_trust --data yeast --limit 500 --seed 42
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
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# Add project root and TUnA source to path
PROJECT_ROOT = Path(__file__).parent.parent
TUNA_DIR = PROJECT_ROOT / "external" / "TUnA"
TUNA_SEED_DIR = TUNA_DIR / "results" / "xspecies" / "TUnA_seed47"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TUNA_SEED_DIR))

from src.trustppi.trust.metrics import selective_accuracy_at_coverage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# TUnA data paths
TUNA_DATA_PROCESSED = TUNA_DIR / "data" / "processed" / "xspecies"
TUNA_DATA_EMBEDDED = TUNA_DIR / "data" / "embedded" / "xspecies"
TUNA_MODEL_PATH = TUNA_SEED_DIR / "output" / "model"

# TUnA model config (from config.yaml)
TUNA_CONFIG = {
    'protein_embedding_dim': 640,
    'hid_dim': 256,
    'n_layers': 1,
    'n_heads': 8,
    'ff_dim': 1024,
    'dropout': 0.2,
    'max_sequence_length': 512,
    'activation_function': 'swish',
    'gp_rffs': 4096,
    'gp_out_targets': 1,
    'gp_cov_momentum': -1,
    'gp_ridge_penalty': 1,
    'gp_likelihood': 'binary_logistic',
    'random_seed': 47,
}

# Available test species
AVAILABLE_DATASETS = ['yeast', 'ecoli', 'human', 'mouse', 'fly', 'worm']


# ---------------------------------------------------------------------------
# ESM2 Embedding Generation
# ---------------------------------------------------------------------------

def generate_esm2_embeddings(
    dictionary_tsv: Path,
    output_pt: Path,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Generate ESM2 embeddings for proteins in a dictionary TSV file.

    Each line: protein_id<TAB>sequence

    Caches to output_pt for subsequent runs.
    """
    if output_pt.exists():
        logger.info(f"Loading cached embeddings from {output_pt}")
        return torch.load(output_pt, map_location='cpu', weights_only=False)

    import esm
    logger.info("Loading ESM2 model (esm2_t30_150M_UR50D)...")
    esm_model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    esm_model = esm_model.to(device)
    esm_model.eval()
    batch_converter = alphabet.get_batch_converter()

    # Read dictionary
    proteins = []
    with open(dictionary_tsv, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) == 2:
                proteins.append((parts[0], parts[1]))

    logger.info(f"Generating ESM2 embeddings for {len(proteins)} proteins...")
    protein_dict = {}
    for i, (prot_id, sequence) in enumerate(tqdm(proteins, desc="ESM2 embedding")):
        _, _, batch_tokens = batch_converter([("protein", sequence)])
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = esm_model(batch_tokens, repr_layers=[30], return_contacts=False)
            # Remove start and end tokens
            emb = results["representations"][30][0, 1:-1, :].cpu()

        protein_dict[prot_id] = emb

    # Cache
    output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(protein_dict, output_pt)
    logger.info(f"Saved embeddings to {output_pt}")

    # Free ESM model
    del esm_model
    torch.cuda.empty_cache()

    return protein_dict


# ---------------------------------------------------------------------------
# TUnA Model Loading
# ---------------------------------------------------------------------------

def load_tuna_model(device: torch.device):
    """Load the pretrained TUnA model."""
    from uncertaintyAwareDeepLearn import VanillaRFFLayer
    from model import IntraEncoder, InterEncoder, ProteinInteractionNet

    cfg = TUNA_CONFIG
    intra_encoder = IntraEncoder(
        cfg['protein_embedding_dim'], cfg['hid_dim'], cfg['n_layers'],
        cfg['n_heads'], cfg['ff_dim'], cfg['dropout'],
        cfg['activation_function'], device,
    )
    inter_encoder = InterEncoder(
        cfg['protein_embedding_dim'], cfg['hid_dim'], cfg['n_layers'],
        cfg['n_heads'], cfg['ff_dim'], cfg['dropout'],
        cfg['activation_function'], device,
    )
    gp_layer = VanillaRFFLayer(
        in_features=cfg['hid_dim'],
        RFFs=cfg['gp_rffs'],
        out_targets=cfg['gp_out_targets'],
        gp_cov_momentum=cfg['gp_cov_momentum'],
        gp_ridge_penalty=cfg['gp_ridge_penalty'],
        likelihood=cfg['gp_likelihood'],
        random_seed=cfg['random_seed'],
    )
    model = ProteinInteractionNet(intra_encoder, inter_encoder, gp_layer, device)
    model.load_state_dict(
        torch.load(str(TUNA_MODEL_PATH), map_location=device, weights_only=False)
    )
    model.eval()
    model.to(device)
    logger.info("TUnA model loaded successfully")
    return model


# ---------------------------------------------------------------------------
# TUnA Inference Helpers
# ---------------------------------------------------------------------------

def tuna_forward_single(
    model,
    emb_a: torch.Tensor,
    emb_b: torch.Tensor,
    device: torch.device,
    max_seq_len: int = 512,
    get_variance: bool = True,
) -> Tuple[float, Optional[float]]:
    """Run TUnA forward pass for a single protein pair.

    Args:
        model: ProteinInteractionNet
        emb_a: [L_A, 640] ESM2 embedding for protein A
        emb_b: [L_B, 640] ESM2 embedding for protein B
        device: torch device
        max_seq_len: max sequence length (truncate if longer)
        get_variance: whether to return GP variance

    Returns:
        (probability, variance) or (probability, None)
    """
    # Truncate to max_seq_len
    emb_a = emb_a[:max_seq_len]
    emb_b = emb_b[:max_seq_len]

    len_a = emb_a.shape[0]
    len_b = emb_b.shape[0]

    # Pad to batch of 1: [1, L, 640]
    prot_a = emb_a.unsqueeze(0).to(device)
    prot_b = emb_b.unsqueeze(0).to(device)

    with torch.no_grad():
        result = model.forward(
            prot_a, prot_b,
            [len_a], [len_b],
            len_a, len_b,
            last_epoch=get_variance,
            train=False,
        )

    if get_variance and isinstance(result, tuple):
        logit, var = result
        # Mean-field calibrated probability
        adjusted = torch.sigmoid(logit / torch.sqrt(1.0 + (np.pi / 8.0) * var))
        prob = float(adjusted.squeeze().cpu())
        variance = float(var.squeeze().cpu())
        return prob, variance
    else:
        logit = result
        prob = float(torch.sigmoid(logit).squeeze().cpu())
        return prob, None


def tuna_get_ppi_feature_vector(
    model,
    emb_a: torch.Tensor,
    emb_b: torch.Tensor,
    device: torch.device,
    max_seq_len: int = 512,
) -> torch.Tensor:
    """Get TUnA's internal PPI feature vector (pre-GP layer) for a pair.

    Used for deformation stability: we perturb ESM2 embeddings and see how
    the output changes.
    """
    emb_a = emb_a[:max_seq_len]
    emb_b = emb_b[:max_seq_len]

    len_a = emb_a.shape[0]
    len_b = emb_b.shape[0]

    prot_a = emb_a.unsqueeze(0).to(device)
    prot_b = emb_b.unsqueeze(0).to(device)

    with torch.no_grad():
        prot_a_mask = model.make_masks([len_a], len_a)
        prot_b_mask = model.make_masks([len_b], len_b)

        enc_a = model.intra_encoder(prot_a, prot_a_mask)
        enc_b = model.intra_encoder(prot_b, prot_b_mask)

        combined_mask_ab = model.combine_masks(prot_a_mask, prot_b_mask)
        combined_mask_ba = model.combine_masks(prot_b_mask, prot_a_mask)

        ab = model.inter_encoder(enc_a, enc_b, combined_mask_ab)
        ba = model.inter_encoder(enc_b, enc_a, combined_mask_ba)

        feat, _ = torch.max(torch.stack([ab, ba], dim=-1), dim=-1)

    return feat  # [1, hid_dim]


def tuna_predict_from_perturbed_embeddings(
    model,
    emb_a: torch.Tensor,
    emb_b: torch.Tensor,
    device: torch.device,
    max_seq_len: int = 512,
) -> float:
    """Get TUnA prediction probability from (possibly noisy) ESM2 embeddings."""
    emb_a = emb_a[:max_seq_len]
    emb_b = emb_b[:max_seq_len]

    len_a = emb_a.shape[0]
    len_b = emb_b.shape[0]

    prot_a = emb_a.unsqueeze(0).to(device)
    prot_b = emb_b.unsqueeze(0).to(device)

    with torch.no_grad():
        # Run forward without variance (simpler, faster for perturbation)
        logit = model.forward(
            prot_a, prot_b,
            [len_a], [len_b],
            len_a, len_b,
            last_epoch=False,
            train=False,
        )
        prob = float(torch.sigmoid(logit).squeeze().cpu())

    return prob


# ---------------------------------------------------------------------------
# Deformation Stability
# ---------------------------------------------------------------------------

def compute_deformation_stability(
    model,
    emb_a: torch.Tensor,
    emb_b: torch.Tensor,
    device: torch.device,
    noise_std: float = 0.1,
    n_samples: int = 10,
    max_seq_len: int = 512,
) -> Dict[str, float]:
    """Compute embedding deformation stability for TUnA.

    Adds Gaussian noise to ESM2 embeddings and measures prediction variance.
    """
    # Base prediction
    base_pred = tuna_predict_from_perturbed_embeddings(
        model, emb_a, emb_b, device, max_seq_len
    )

    perturbed_preds = []
    for _ in range(n_samples):
        noise_a = torch.randn_like(emb_a) * noise_std
        noise_b = torch.randn_like(emb_b) * noise_std

        pred = tuna_predict_from_perturbed_embeddings(
            model, emb_a + noise_a, emb_b + noise_b, device, max_seq_len
        )
        perturbed_preds.append(pred)

    perturbed_preds = np.array(perturbed_preds)

    pred_std = float(np.std(perturbed_preds))
    flip_rate = float(np.mean([(p > 0.5) != (base_pred > 0.5) for p in perturbed_preds]))
    mean_change = float(np.mean(np.abs(perturbed_preds - base_pred)))

    stability = 1.0 / (1.0 + pred_std)
    flip_score = 1.0 - flip_rate

    return {
        't_deform_stable': stability,
        't_deform_flip': flip_score,
        't_deform_combined': (stability + flip_score) / 2,
        'base_prediction': base_pred,
        'pred_std': pred_std,
        'flip_rate': flip_rate,
        'mean_change': mean_change,
    }


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_tuna_test_data(
    dataset: str,
    limit: Optional[int] = None,
    balanced: bool = True,
    seed: int = 42,
) -> Tuple[List[Tuple[str, str, int]], Dict[str, torch.Tensor], torch.device]:
    """Load TUnA's test interaction pairs.

    Returns:
        (pairs, protein_embeddings_needed_later)
        pairs: list of (prot_a_id, prot_b_id, label)
    """
    interaction_file = TUNA_DATA_PROCESSED / f"{dataset}_test_interaction.tsv"
    if not interaction_file.exists():
        raise FileNotFoundError(f"Interaction file not found: {interaction_file}")

    pairs = []
    with open(interaction_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) == 3:
                pairs.append((parts[0], parts[1], int(parts[2])))

    if balanced and limit:
        rng = np.random.RandomState(seed)
        pos = [p for p in pairs if p[2] == 1]
        neg = [p for p in pairs if p[2] == 0]
        n_each = min(limit // 2, len(pos), len(neg))
        rng.shuffle(pos)
        rng.shuffle(neg)
        pairs = pos[:n_each] + neg[:n_each]
        rng.shuffle(pairs)
    elif limit:
        rng = np.random.RandomState(seed)
        rng.shuffle(pairs)
        pairs = pairs[:limit]

    return pairs


# ---------------------------------------------------------------------------
# Evaluation Helpers
# ---------------------------------------------------------------------------

def evaluate_signal_for_error_detection(
    signals: Dict[str, np.ndarray],
    signal_name: str,
) -> Dict[str, float]:
    """Evaluate how well a signal detects errors."""
    predictions = (signals['predictions'] > 0.5).astype(int)
    labels = signals['labels'].astype(int)
    is_correct = (predictions == labels).astype(int)
    signal = signals[signal_name]

    try:
        auroc = roc_auc_score(is_correct, signal)
    except Exception:
        auroc = 0.5

    correlation = np.corrcoef(signal, is_correct)[0, 1]
    if np.isnan(correlation):
        correlation = 0.0

    return {
        'auroc': float(auroc),
        'correlation': float(correlation),
        'mean_correct': float(signal[is_correct == 1].mean()) if (is_correct == 1).sum() > 0 else 0,
        'mean_incorrect': float(signal[is_correct == 0].mean()) if (is_correct == 0).sum() > 0 else 0,
    }


def evaluate_selective_prediction(
    signals: Dict[str, np.ndarray],
    signal_name: str,
    target_coverage: float = 0.8,
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
        'n_accepted': metrics.n_accepted,
    }


# ---------------------------------------------------------------------------
# Signal Collection
# ---------------------------------------------------------------------------

def collect_signals(
    model,
    pairs: List[Tuple[str, str, int]],
    protein_dict: Dict[str, torch.Tensor],
    device: torch.device,
    noise_std: float = 0.1,
    n_deform_samples: int = 10,
    max_seq_len: int = 512,
) -> Dict[str, np.ndarray]:
    """Collect all trust signals for TUnA on test pairs."""
    predictions = []
    labels = []
    confidences = []
    gp_variances = []
    gp_trust = []  # 1 - normalized_variance (higher = more trustworthy)
    deform_stable = []
    deform_flip = []
    deform_combined = []
    skipped = 0

    for prot_a, prot_b, label in tqdm(pairs, desc="Collecting TUnA signals"):
        if prot_a not in protein_dict or prot_b not in protein_dict:
            skipped += 1
            continue

        emb_a = protein_dict[prot_a]
        emb_b = protein_dict[prot_b]

        # TUnA inference with GP variance
        prob, variance = tuna_forward_single(
            model, emb_a, emb_b, device, max_seq_len, get_variance=True
        )

        predictions.append(prob)
        labels.append(float(label))
        confidences.append(max(prob, 1 - prob))
        gp_variances.append(variance if variance is not None else 0.0)

        # Deformation stability
        try:
            deform = compute_deformation_stability(
                model, emb_a, emb_b, device,
                noise_std=noise_std,
                n_samples=n_deform_samples,
                max_seq_len=max_seq_len,
            )
            deform_stable.append(deform['t_deform_stable'])
            deform_flip.append(deform['t_deform_flip'])
            deform_combined.append(deform['t_deform_combined'])
        except Exception as e:
            logger.warning(f"Deformation failed for ({prot_a}, {prot_b}): {e}")
            deform_stable.append(0.5)
            deform_flip.append(0.5)
            deform_combined.append(0.5)

    if skipped > 0:
        logger.warning(f"Skipped {skipped} pairs (missing embeddings)")

    # Convert GP variance to trust signal: lower variance = higher trust
    gp_var_arr = np.array(gp_variances)
    # Use 1/(1+var) so higher = more trustworthy (matches other signals)
    gp_trust_arr = 1.0 / (1.0 + gp_var_arr)

    return {
        'predictions': np.array(predictions),
        'labels': np.array(labels),
        'confidence': np.array(confidences),
        'gp_variance': gp_trust_arr,  # transformed: higher = more trustworthy
        'gp_variance_raw': gp_var_arr,
        't_deform_stable': np.array(deform_stable),
        't_deform_flip': np.array(deform_flip),
        't_deform_combined': np.array(deform_combined),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TUnA Cross-Model Trust Evaluation")
    parser.add_argument('--data', type=str, default='yeast',
                        choices=AVAILABLE_DATASETS,
                        help='Test dataset')
    parser.add_argument('--limit', type=int, default=None, help='Limit test samples')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--quick', action='store_true', help='Quick test (30 samples)')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--noise-std', type=float, default=0.1,
                        help='Noise std for deformation stability')
    parser.add_argument('--n-deform-samples', type=int, default=10,
                        help='Number of perturbation samples')

    args = parser.parse_args()

    # Seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    logger.info(f"Random seed: {args.seed}")

    if args.quick:
        args.limit = 30
        logger.info("Quick test mode (30 samples)")

    device = torch.device(
        args.device if args.device != 'cuda' or torch.cuda.is_available() else 'cpu'
    )
    logger.info(f"Device: {device}")

    output_dir = args.output_dir or (PROJECT_ROOT / "experiments" / "results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Generate/load ESM2 embeddings
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info(f"LOADING ESM2 EMBEDDINGS — {args.data}")
    logger.info("=" * 70)

    dict_tsv = TUNA_DATA_PROCESSED / f"{args.data}_test_dictionary.tsv"
    emb_cache = TUNA_DATA_EMBEDDED / f"{args.data}_test_dictionary" / "protein_dictionary.pt"

    protein_dict = generate_esm2_embeddings(dict_tsv, emb_cache, device)
    logger.info(f"  {len(protein_dict)} protein embeddings available")

    # ------------------------------------------------------------------
    # 2. Load TUnA model
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("LOADING TUnA MODEL")
    logger.info("=" * 70)

    model = load_tuna_model(device)

    # ------------------------------------------------------------------
    # 3. Load test pairs
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info(f"LOADING TEST DATA — {args.data}")
    logger.info("=" * 70)

    pairs = load_tuna_test_data(
        args.data, limit=args.limit, balanced=True, seed=args.seed
    )
    logger.info(f"  {len(pairs)} test pairs")

    # ------------------------------------------------------------------
    # 4. Collect signals
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("COLLECTING TRUST SIGNALS")
    logger.info("=" * 70)

    signals = collect_signals(
        model, pairs, protein_dict, device,
        noise_std=args.noise_std,
        n_deform_samples=args.n_deform_samples,
    )
    n_samples = len(signals['predictions'])
    logger.info(f"  Collected signals for {n_samples} samples")

    # Sanity checks
    preds = signals['predictions']
    conf = signals['confidence']
    assert np.all((preds >= 0) & (preds <= 1)), "Probabilities out of [0,1]"
    assert np.all((conf >= 0.5) & (conf <= 1.0)), "Confidence out of [0.5,1]"

    # Baseline
    binary_preds = (signals['predictions'] > 0.5).astype(int)
    labels_arr = signals['labels'].astype(int)
    baseline_acc = float((binary_preds == labels_arr).mean())
    logger.info(f"\nBaseline accuracy: {baseline_acc:.3f}")

    # ------------------------------------------------------------------
    # 5. Error detection
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("ERROR DETECTION EVALUATION (AUROC)")
    logger.info("=" * 70)

    signal_names = [
        'confidence', 'gp_variance',
        't_deform_stable', 't_deform_flip', 't_deform_combined',
    ]

    error_detection_results = {}
    for name in signal_names:
        result = evaluate_signal_for_error_detection(signals, name)
        error_detection_results[name] = result
        logger.info(f"  {name:25s}: AUROC={result['auroc']:.3f}, corr={result['correlation']:+.3f}")

    # ------------------------------------------------------------------
    # 6. Selective prediction
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("SELECTIVE PREDICTION EVALUATION (80% Coverage)")
    logger.info("=" * 70)

    selective_results = {}
    for name in signal_names:
        result = evaluate_selective_prediction(signals, name, target_coverage=0.8)
        selective_results[name] = result
        logger.info(f"  {name:25s}: Acc={result['accuracy']:.3f}, Imp={result['improvement']:+.1%}")

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Baseline accuracy: {baseline_acc:.3f}")

    conf_auroc = error_detection_results['confidence']['auroc']
    gp_auroc = error_detection_results['gp_variance']['auroc']
    deform_auroc = error_detection_results['t_deform_combined']['auroc']

    logger.info(f"  Confidence AUROC:          {conf_auroc:.3f}")
    logger.info(f"  GP variance AUROC:         {gp_auroc:.3f}")
    logger.info(f"  Deformation AUROC:         {deform_auroc:.3f}")

    conf_imp = selective_results['confidence']['improvement']
    gp_imp = selective_results['gp_variance']['improvement']
    deform_imp = selective_results['t_deform_combined']['improvement']

    logger.info(f"  Confidence selective imp:   {conf_imp:+.1%}")
    logger.info(f"  GP variance selective imp:  {gp_imp:+.1%}")
    logger.info(f"  Deformation selective imp:  {deform_imp:+.1%}")

    deform_beats_gp = deform_auroc > gp_auroc
    logger.info(f"\n  Deformation adds value over GP: {'YES' if deform_beats_gp else 'NO'}")

    # GP variance stats
    raw_var = signals['gp_variance_raw']
    logger.info(f"\n  GP variance stats: mean={raw_var.mean():.4f}, "
                f"std={raw_var.std():.4f}, min={raw_var.min():.4f}, max={raw_var.max():.4f}")

    # ------------------------------------------------------------------
    # 8. Save JSON
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = output_dir / f"tuna_trust_{args.data}_seed{args.seed}_{timestamp}.json"

    output = {
        'experiment': 'tuna_cross_model_trust',
        'model': 'TUnA',
        'dataset': args.data,
        'seed': args.seed,
        'n_samples': n_samples,
        'baseline_accuracy': baseline_acc,
        'timestamp': timestamp,
        'error_detection': error_detection_results,
        'selective_prediction': selective_results,
        'deformation_adds_value_over_gp': deform_beats_gp,
        'notes': (
            f"TUnA (Transformer + GP head, seed47 checkpoint). "
            f"ESM2 embeddings (640d). "
            f"GP variance: native uncertainty from Gaussian Process layer. "
            f"Deformation stability via ESM2 embedding perturbation "
            f"(noise_std={args.noise_std}, n_samples={args.n_deform_samples})."
        ),
    }

    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
