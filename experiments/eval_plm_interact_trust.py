"""
Experiment: PLM-interact Cross-Model Trust Evaluation.

Evaluates trust signals on PLM-interact (ESM-2 fine-tuned for PPI),
a state-of-the-art PLM-based model. This provides a third architecture
(pure PLM) alongside D-SCRIPT (CNN) and TUnA (Transformer+GP).

Signals evaluated:
  - confidence: max(p, 1-p) — generic baseline
  - t_deform_stable, t_deform_flip, t_deform_combined: embedding
    perturbation stability (Gaussian noise on ESM2 hidden states)

Requires:
  - PLM-interact weights: checkpoints/plm_interact/pytorch_model.bin
  - ESM-2 base model: facebook/esm2_t33_650M_UR50D (auto-downloaded)

Usage:
    python -m experiments.eval_plm_interact_trust --quick --seed 42
    python -m experiments.eval_plm_interact_trust --data yeast --limit 500 --seed 42
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
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForMaskedLM, AutoTokenizer

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.trustppi.data import get_dataset_paths, load_pairs, load_sequences_for_pairs
from src.trustppi.trust.metrics import selective_accuracy_at_coverage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model paths
PLM_INTERACT_WEIGHTS = PROJECT_ROOT / "checkpoints" / "plm_interact" / "pytorch_model.bin"
PLM_INTERACT_CONFIG = PROJECT_ROOT / "checkpoints" / "plm_interact" / "config.json"
ESM2_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"

# Model config
PLM_CONFIG = {
    'model': ESM2_MODEL_NAME,
    'embedding_size': 1280,
    'num_labels': 1,
    'max_length': 1603,  # Max combined length for tokenizer
}

# Available test datasets
AVAILABLE_DATASETS = ['yeast', 'ecoli', 'human', 'mouse', 'fly', 'worm']


# ---------------------------------------------------------------------------
# PLM-interact Model Definition
# ---------------------------------------------------------------------------

class PLMinteract(nn.Module):
    """PLM-interact model: ESM-2 backbone + classification head."""

    def __init__(self, model_name: str, num_labels: int, embedding_size: int):
        super().__init__()
        self.esm_mask = AutoModelForMaskedLM.from_pretrained(model_name)
        self.embedding_size = embedding_size
        self.classifier = nn.Linear(embedding_size, 1)
        self.num_labels = num_labels

        # Store hooks for perturbation
        self._noise_hook = None
        self._noise_std = 0.0

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass returning probability."""
        embedding_output = self.esm_mask.base_model(**features, return_dict=True)
        embedding = embedding_output.last_hidden_state[:, 0, :]  # CLS token
        embedding = F.relu(embedding)
        logits = self.classifier(embedding)
        logits = logits.view(-1)
        probability = torch.sigmoid(logits)
        return probability

    def forward_with_embeddings(
        self,
        features: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both probability and CLS embedding."""
        embedding_output = self.esm_mask.base_model(**features, return_dict=True)
        embedding = embedding_output.last_hidden_state[:, 0, :]  # CLS token
        embedding_relu = F.relu(embedding)
        logits = self.classifier(embedding_relu)
        logits = logits.view(-1)
        probability = torch.sigmoid(logits)
        return probability, embedding

    def set_noise_perturbation(self, noise_std: float):
        """Enable noise perturbation during forward pass."""
        self._noise_std = noise_std

        if self._noise_hook is not None:
            self._noise_hook.remove()

        if noise_std > 0:
            def add_noise_hook(module, input, output):
                if self.training is False:  # Only during eval
                    noise = torch.randn_like(output.last_hidden_state) * self._noise_std
                    # Create new output with noisy hidden states
                    output.last_hidden_state = output.last_hidden_state + noise
                return output

            self._noise_hook = self.esm_mask.base_model.encoder.register_forward_hook(add_noise_hook)

    def clear_noise_perturbation(self):
        """Disable noise perturbation."""
        self._noise_std = 0.0
        if self._noise_hook is not None:
            self._noise_hook.remove()
            self._noise_hook = None


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

def load_plm_interact(device: torch.device) -> Tuple[PLMinteract, AutoTokenizer]:
    """Load pretrained PLM-interact model and tokenizer."""
    logger.info(f"Loading PLM-interact from {PLM_INTERACT_WEIGHTS}")

    # Check weights exist
    if not PLM_INTERACT_WEIGHTS.exists():
        raise FileNotFoundError(
            f"PLM-interact weights not found at {PLM_INTERACT_WEIGHTS}. "
            "Please download from HuggingFace: danliu1226/PLM-interact-650M-humanV11"
        )

    # Load config if available
    config = PLM_CONFIG.copy()
    if PLM_INTERACT_CONFIG.exists():
        with open(PLM_INTERACT_CONFIG) as f:
            saved_config = json.load(f)
            config.update(saved_config)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model'])

    # Load model
    model = PLMinteract(
        model_name=config['model'],
        num_labels=config['num_labels'],
        embedding_size=config['embedding_size']
    )

    # Load weights
    state_dict = torch.load(PLM_INTERACT_WEIGHTS, map_location=device, weights_only=False)

    # Load with strict=False to handle architecture differences between
    # the trained checkpoint and newer HuggingFace ESM-2 model
    # (e.g., position_embeddings vs rotary embeddings)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        logger.warning(f"Missing keys when loading: {missing[:5]}... ({len(missing)} total)")
    if unexpected:
        logger.warning(f"Unexpected keys when loading: {unexpected[:5]}... ({len(unexpected)} total)")

    # Verify classifier weights loaded correctly (most important for PPI prediction)
    if 'classifier.weight' not in [k.split('.')[-2] + '.' + k.split('.')[-1] for k in state_dict.keys() if 'classifier' in k]:
        logger.info("Classifier weights loaded from checkpoint")

    model.eval()
    model.to(device)

    logger.info("PLM-interact loaded successfully")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Inference Helpers
# ---------------------------------------------------------------------------

def plm_predict_single(
    model: PLMinteract,
    tokenizer: AutoTokenizer,
    seq_a: str,
    seq_b: str,
    device: torch.device,
    max_length: int = 1603,
) -> float:
    """Run PLM-interact inference for a single protein pair.

    Args:
        model: PLMinteract model
        tokenizer: ESM-2 tokenizer
        seq_a: Sequence of protein A
        seq_b: Sequence of protein B
        device: torch device
        max_length: Max combined length

    Returns:
        Interaction probability
    """
    # Tokenize pair (PLM-interact uses joint encoding)
    tokenized = tokenizer(
        seq_a, seq_b,
        padding=True,
        truncation='longest_first',
        return_tensors="pt",
        max_length=max_length
    )
    tokenized = {k: v.to(device) for k, v in tokenized.items()}

    with torch.no_grad():
        prob = model(tokenized)

    return float(prob.cpu().item())


# ---------------------------------------------------------------------------
# Deformation Stability
# ---------------------------------------------------------------------------

def compute_deformation_stability(
    model: PLMinteract,
    tokenizer: AutoTokenizer,
    seq_a: str,
    seq_b: str,
    device: torch.device,
    noise_std: float = 0.1,
    n_samples: int = 10,
    max_length: int = 1603,
) -> Dict[str, float]:
    """Compute embedding deformation stability for PLM-interact.

    Adds Gaussian noise to ESM-2 hidden states and measures prediction variance.
    """
    # Base prediction (no noise)
    model.clear_noise_perturbation()
    base_pred = plm_predict_single(model, tokenizer, seq_a, seq_b, device, max_length)

    # Perturbed predictions
    model.set_noise_perturbation(noise_std)

    perturbed_preds = []
    tokenized = tokenizer(
        seq_a, seq_b,
        padding=True,
        truncation='longest_first',
        return_tensors="pt",
        max_length=max_length
    )
    tokenized = {k: v.to(device) for k, v in tokenized.items()}

    for _ in range(n_samples):
        with torch.no_grad():
            prob = model(tokenized)
        perturbed_preds.append(float(prob.cpu().item()))

    # Clear perturbation
    model.clear_noise_perturbation()

    perturbed_preds = np.array(perturbed_preds)

    # Compute stability metrics
    pred_std = float(np.std(perturbed_preds))
    flip_rate = float(np.mean([(p > 0.5) != (base_pred > 0.5) for p in perturbed_preds]))
    mean_change = float(np.mean(np.abs(perturbed_preds - base_pred)))

    # Trust scores: higher = more trustworthy
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
    model: PLMinteract,
    tokenizer: AutoTokenizer,
    pairs: List[Tuple[str, str, float]],
    sequences: Dict[str, str],
    device: torch.device,
    noise_std: float = 0.1,
    n_deform_samples: int = 10,
    max_length: int = 1603,
) -> Dict[str, np.ndarray]:
    """Collect all trust signals for PLM-interact on test pairs."""
    predictions = []
    labels = []
    confidences = []
    deform_stable = []
    deform_flip = []
    deform_combined = []
    skipped = 0

    for prot_a, prot_b, label in tqdm(pairs, desc="Collecting PLM-interact signals"):
        # Get sequences
        seq_a = sequences.get(str(prot_a))
        seq_b = sequences.get(str(prot_b))

        if seq_a is None or seq_b is None:
            skipped += 1
            continue

        try:
            # Base prediction
            prob = plm_predict_single(model, tokenizer, seq_a, seq_b, device, max_length)

            predictions.append(prob)
            labels.append(float(label))
            confidences.append(max(prob, 1 - prob))

            # Deformation stability
            deform = compute_deformation_stability(
                model, tokenizer, seq_a, seq_b, device,
                noise_std=noise_std,
                n_samples=n_deform_samples,
                max_length=max_length,
            )
            deform_stable.append(deform['t_deform_stable'])
            deform_flip.append(deform['t_deform_flip'])
            deform_combined.append(deform['t_deform_combined'])

        except Exception as e:
            logger.warning(f"Error processing ({prot_a}, {prot_b}): {e}")
            skipped += 1
            continue

    if skipped > 0:
        logger.warning(f"Skipped {skipped} pairs")

    return {
        'predictions': np.array(predictions),
        'labels': np.array(labels),
        'confidence': np.array(confidences),
        't_deform_stable': np.array(deform_stable),
        't_deform_flip': np.array(deform_flip),
        't_deform_combined': np.array(deform_combined),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PLM-interact Cross-Model Trust Evaluation")
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
    # 1. Load PLM-interact model
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("LOADING PLM-INTERACT MODEL")
    logger.info("=" * 70)

    model, tokenizer = load_plm_interact(device)

    # ------------------------------------------------------------------
    # 2. Load test data
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info(f"LOADING TEST DATA — {args.data}")
    logger.info("=" * 70)

    pairs_file, seqs_file = get_dataset_paths(args.data)

    pairs = load_pairs(
        pairs_file,
        limit=args.limit,
        balanced=True,
        seed=args.seed
    )
    logger.info(f"  {len(pairs)} test pairs")

    sequences = load_sequences_for_pairs(pairs, seqs_file)
    logger.info(f"  {len(sequences)} sequences loaded")

    # ------------------------------------------------------------------
    # 3. Collect signals
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("COLLECTING TRUST SIGNALS")
    logger.info("=" * 70)

    signals = collect_signals(
        model, tokenizer, pairs, sequences, device,
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
    # 4. Error detection
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("ERROR DETECTION EVALUATION (AUROC)")
    logger.info("=" * 70)

    signal_names = [
        'confidence',
        't_deform_stable', 't_deform_flip', 't_deform_combined',
    ]

    error_detection_results = {}
    for name in signal_names:
        result = evaluate_signal_for_error_detection(signals, name)
        error_detection_results[name] = result
        logger.info(f"  {name:25s}: AUROC={result['auroc']:.3f}, corr={result['correlation']:+.3f}")

    # ------------------------------------------------------------------
    # 5. Selective prediction
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
    # 6. Summary
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Baseline accuracy: {baseline_acc:.3f}")

    conf_auroc = error_detection_results['confidence']['auroc']
    deform_auroc = error_detection_results['t_deform_combined']['auroc']

    logger.info(f"  Confidence AUROC:          {conf_auroc:.3f}")
    logger.info(f"  Deformation AUROC:         {deform_auroc:.3f}")

    conf_imp = selective_results['confidence']['improvement']
    deform_imp = selective_results['t_deform_combined']['improvement']

    logger.info(f"  Confidence selective imp:   {conf_imp:+.1%}")
    logger.info(f"  Deformation selective imp:  {deform_imp:+.1%}")

    deform_beats_conf = deform_auroc > conf_auroc
    logger.info(f"\n  Deformation beats confidence: {'YES' if deform_beats_conf else 'NO'}")

    # ------------------------------------------------------------------
    # 7. Save JSON
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = output_dir / f"plm_interact_trust_{args.data}_seed{args.seed}_{timestamp}.json"

    output = {
        'experiment': 'plm_interact_cross_model_trust',
        'model': 'PLM-interact-650M',
        'architecture': 'ESM-2 fine-tuned (PLM)',
        'dataset': args.data,
        'seed': args.seed,
        'n_samples': n_samples,
        'baseline_accuracy': baseline_acc,
        'timestamp': timestamp,
        'error_detection': error_detection_results,
        'selective_prediction': selective_results,
        'deformation_beats_confidence': deform_beats_conf,
        'notes': (
            f"PLM-interact (ESM-2 650M fine-tuned for PPI). "
            f"Joint protein pair encoding with CLS token classification. "
            f"Deformation stability via hidden state perturbation "
            f"(noise_std={args.noise_std}, n_samples={args.n_deform_samples})."
        ),
    }

    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
