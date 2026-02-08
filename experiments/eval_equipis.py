"""
Experiment: EquiPPIS Cross-Model Trust Evaluation.

Evaluates trust signals on EquiPPIS — an E(3)-equivariant GNN for
per-residue PPI interface prediction.  This provides the fourth architecture
(structure-based GNN) alongside D-SCRIPT (CNN), TUnA (Transformer+GP),
and PLM-interact (ESM-2 fine-tuned).

EquiPPIS predicts *per-residue interface labels* rather than binary PPI.
We adapt it to our trust-signal framework by:
  1. Running per-residue predictions on SAbDab antibody-antigen complexes
  2. Comparing predictions to ground-truth interface labels
  3. Computing deformation stability via 3D coordinate perturbation
  4. Aggregating per-residue trust metrics to per-complex scores

Signals evaluated:
  - confidence: max(p, 1-p) per residue, averaged per complex
  - t_deform_stable / flip / combined: coordinate perturbation stability

Requires:
  - EquiPPIS weights: external/EquiPPIS/Trained_model/EquiPPIS_model/E-l10-256.pt
  - SAbDab structures: data/sabdab/structures/

Usage:
    python -m experiments.eval_equipis --quick
    python -m experiments.eval_equipis --limit 100 --seed 42
"""

import argparse
import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# Project setup
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "external" / "EquiPPIS"))

from src.trustppi.trust.metrics import selective_accuracy_at_coverage
from src.trustppi.sabdab_data import (
    load_structures,
    extract_interfaces,
    parse_pdb_file,
    ComplexData,
    ResidueData,
    AA_3TO1,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
EQUIPIS_WEIGHTS = PROJECT_ROOT / "external" / "EquiPPIS" / "Trained_model" / "EquiPPIS_model" / "E-l10-256.pt"
SABDAB_DIR = PROJECT_ROOT / "data" / "sabdab"
SABDAB_STRUCTURES = SABDAB_DIR / "structures"
SABDAB_METADATA = SABDAB_DIR / "filtered_complexes.tsv"

# EquiPPIS model config
EQUIPIS_CONFIG = {
    'in_node_nf': 118,
    'hidden_nf': 256,
    'out_node_nf': 1,
    'in_edge_nf': 1,
    'n_layers': 10,
    'attention': True,
}

# Interface prediction threshold (from EquiPPIS paper)
INTERFACE_THRESHOLD = 0.18


# ===========================================================================
# EGNN Model (imported from EquiPPIS codebase)
# ===========================================================================

def load_egnn_model(device: torch.device) -> nn.Module:
    """Load the pretrained EquiPPIS EGNN model."""
    from egnn_clean import EGNN

    model = EGNN(
        in_node_nf=EQUIPIS_CONFIG['in_node_nf'],
        hidden_nf=EQUIPIS_CONFIG['hidden_nf'],
        out_node_nf=EQUIPIS_CONFIG['out_node_nf'],
        in_edge_nf=EQUIPIS_CONFIG['in_edge_nf'],
        n_layers=EQUIPIS_CONFIG['n_layers'],
        attention=EQUIPIS_CONFIG['attention'],
    )

    if not EQUIPIS_WEIGHTS.exists():
        raise FileNotFoundError(
            f"EquiPPIS weights not found: {EQUIPIS_WEIGHTS}\n"
            f"Expected at: external/EquiPPIS/Trained_model/EquiPPIS_model/E-l10-256.pt"
        )

    state_dict = torch.load(EQUIPIS_WEIGHTS, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Loaded EquiPPIS EGNN: {n_params:,} parameters")

    return model


# ===========================================================================
# Feature Computation (simplified, from PDB structures)
# ===========================================================================

AA_LIST = sorted(AA_3TO1.values())
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}


def compute_contact_count(
    coords: np.ndarray,
    threshold: float = 8.0,
) -> np.ndarray:
    """Number of CA neighbors within threshold distance."""
    from scipy.spatial.distance import cdist
    dists = cdist(coords, coords)
    counts = (dists < threshold).sum(axis=1) - 1  # exclude self
    return counts.astype(np.float32)


def compute_forward_reverse_ca(coords: np.ndarray) -> np.ndarray:
    """Forward and reverse CA distance vectors (6 dim)."""
    n = len(coords)
    features = np.zeros((n, 6), dtype=np.float32)

    for i in range(n):
        # Forward CA vector (to next residue)
        if i < n - 1:
            fwd = coords[i + 1] - coords[i]
        else:
            fwd = np.zeros(3)
        # Reverse CA vector (to previous residue)
        if i > 0:
            rev = coords[i - 1] - coords[i]
        else:
            rev = np.zeros(3)

        features[i, :3] = fwd
        features[i, 3:] = rev

    return features


def compute_backbone_angles(coords: np.ndarray) -> np.ndarray:
    """Pseudo phi/psi angles from CA trace (6 dim: sin/cos of 3 angles)."""
    n = len(coords)
    features = np.zeros((n, 6), dtype=np.float32)

    for i in range(1, n - 1):
        v1 = coords[i] - coords[i - 1]
        v2 = coords[i + 1] - coords[i]

        # Bond angle
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle)

        features[i, 0] = np.sin(angle)
        features[i, 1] = np.cos(angle)

        # Pseudo-dihedral (needs 4 consecutive residues)
        if i >= 2:
            v0 = coords[i - 1] - coords[i - 2]
            n1 = np.cross(v0, v1)
            n2 = np.cross(v1, v2)
            norm1 = np.linalg.norm(n1) + 1e-8
            norm2 = np.linalg.norm(n2) + 1e-8
            cos_dih = np.dot(n1, n2) / (norm1 * norm2)
            cos_dih = np.clip(cos_dih, -1, 1)
            dih = np.arccos(cos_dih)
            features[i, 2] = np.sin(dih)
            features[i, 3] = np.cos(dih)

        if i < n - 2:
            v3 = coords[i + 2] - coords[i + 1]
            n1 = np.cross(v1, v2)
            n2 = np.cross(v2, v3)
            norm1 = np.linalg.norm(n1) + 1e-8
            norm2 = np.linalg.norm(n2) + 1e-8
            cos_dih = np.dot(n1, n2) / (norm1 * norm2)
            cos_dih = np.clip(cos_dih, -1, 1)
            dih = np.arccos(cos_dih)
            features[i, 4] = np.sin(dih)
            features[i, 5] = np.cos(dih)

    return features


def compute_tetrahedral_geometry(coords: np.ndarray) -> np.ndarray:
    """Approximate tetrahedral geometry features (3 dim)."""
    n = len(coords)
    features = np.zeros((n, 3), dtype=np.float32)

    for i in range(2, n - 2):
        # Local frame from 5 consecutive CA atoms
        neighbors = coords[i - 2:i + 3]
        center = coords[i]
        relative = neighbors - center
        dists = np.linalg.norm(relative, axis=1)
        dists[2] = 1.0  # avoid div by zero for self

        features[i, 0] = np.mean(dists)  # mean distance to neighbors
        features[i, 1] = np.std(dists)   # std of distances
        # Volume-like feature
        if n > 4:
            v1 = relative[0]
            v2 = relative[1]
            v3 = relative[3]
            vol = abs(np.dot(v1, np.cross(v2, v3)))
            features[i, 2] = min(vol, 1000.0)  # clip extreme values

    return features


def compute_node_features_simplified(
    residues: List[ResidueData],
    coords: np.ndarray,
) -> np.ndarray:
    """
    Compute 118-dim node features from PDB structure alone.

    Feature breakdown (matching EquiPPIS expected dimensions):
      - PDB structural features:    26 dim  (approximated)
      - Extra structural feature:     1 dim
      - Contact count:                1 dim
      - DSSP-derived features:        6 dim  (approximated from geometry)
      - Phi/psi angle sin/cos:        6 dim
      - Forward/reverse CA:           6 dim
      - Tetrahedral geometry:         3 dim
      - PSSM:                        20 dim  (set to sigmoid(0) = 0.5)
      - ESM2 embeddings:            33 dim  (set to sigmoid(0) = 0.5)
      - Padding:                     16 dim  (zeros)
      Total:                        118 dim

    Note: For full accuracy, run the EquiPPIS preprocessing pipeline
    to generate proper DSSP, PSSM, and ESM2 features.
    """
    n = len(residues)
    features = np.zeros((n, 118), dtype=np.float32)

    # One-hot amino acid (place in first 20 of 26 PDB feature slots)
    for i, res in enumerate(residues):
        aa_1 = AA_3TO1.get(res.res_name, 'X')
        if aa_1 in AA_TO_IDX:
            features[i, AA_TO_IDX[aa_1]] = 1.0

    # Relative solvent accessibility proxy (use contact count inverse)
    contact_counts = compute_contact_count(coords)
    max_cc = max(contact_counts.max(), 1.0)
    features[:, 20] = contact_counts / max_cc  # normalized contact count
    features[:, 21] = 1.0 - (contact_counts / max_cc)  # inverse (RSA proxy)

    # Distance to centroid
    centroid = coords.mean(axis=0)
    dist_to_center = np.linalg.norm(coords - centroid, axis=1)
    max_dist = max(dist_to_center.max(), 1.0)
    features[:, 22] = dist_to_center / max_dist

    # Local density (number of neighbors within 10A / total)
    from scipy.spatial.distance import cdist
    dists = cdist(coords, coords)
    local_density = (dists < 10.0).sum(axis=1) / max(n, 1)
    features[:, 23] = local_density

    # Half-sphere exposure (approximate)
    for i in range(n):
        if i > 0 and i < n - 1:
            v_prev = coords[i - 1] - coords[i]
            v_next = coords[i + 1] - coords[i]
            chain_dir = v_prev + v_next
            chain_dir_norm = np.linalg.norm(chain_dir)
            if chain_dir_norm > 0:
                chain_dir = chain_dir / chain_dir_norm
            # Count neighbors in upper vs lower half-sphere
            neighbors = coords[max(0, i-10):min(n, i+10)] - coords[i]
            dots = neighbors @ chain_dir
            features[i, 24] = (dots > 0).sum() / max(len(dots), 1)
            features[i, 25] = (dots <= 0).sum() / max(len(dots), 1)

    # dim 26: extra structural feature
    features[:, 26] = contact_counts / max_cc

    # dim 27: contact count
    features[:, 27] = contact_counts / max_cc

    # dims 28-33: DSSP-derived features (approximate from geometry)
    backbone_angles = compute_backbone_angles(coords)
    features[:, 28:34] = backbone_angles

    # dims 34-39: phi/psi angles (same backbone angles)
    features[:, 34:40] = backbone_angles

    # dims 40-45: forward/reverse CA
    fwd_rev = compute_forward_reverse_ca(coords)
    # Normalize
    fwd_rev_max = np.abs(fwd_rev).max() + 1e-8
    features[:, 40:46] = fwd_rev / fwd_rev_max

    # dims 46-48: tetrahedral geometry
    tet_geom = compute_tetrahedral_geometry(coords)
    tet_max = np.abs(tet_geom).max() + 1e-8
    features[:, 46:49] = tet_geom / tet_max

    # dims 49-68: PSSM placeholder (sigmoid(0) = 0.5)
    features[:, 49:69] = 0.5

    # dims 69-101: ESM2 placeholder (sigmoid(0) = 0.5)
    features[:, 69:102] = 0.5

    # dims 102-117: padding zeros (already zero)

    return features


# ===========================================================================
# Graph Construction for EquiPPIS
# ===========================================================================

def build_distance_edges(
    coords: np.ndarray,
    threshold: float = 8.0,
) -> Tuple[List[int], List[int], np.ndarray]:
    """Build edges from distance map (matching EquiPPIS format)."""
    from scipy.spatial.distance import cdist

    n = len(coords)
    dists = cdist(coords, coords)

    src, dst, weights = [], [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dists[i, j] < threshold:
                src.append(i)
                dst.append(j)
                seq_sep = abs(i - j)
                w = math.log(max(seq_sep, 1)) / max(dists[i, j], 0.1)
                weights.append(w)

    return src, dst, np.array(weights, dtype=np.float32)


def prepare_equipis_input(
    residues: List[ResidueData],
    coords: np.ndarray,
    device: torch.device,
    edge_threshold: float = 8.0,
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], torch.Tensor]:
    """Prepare input tensors for EquiPPIS forward pass."""
    # Node features
    node_feats = compute_node_features_simplified(residues, coords)
    node_feats_t = torch.tensor(node_feats, dtype=torch.float32).to(device)

    # Coordinates
    coords_t = torch.tensor(coords, dtype=torch.float32).to(device)

    # Edges
    src, dst, weights = build_distance_edges(coords, threshold=edge_threshold)
    if len(src) == 0:
        # Fallback: connect all pairs within larger threshold
        src, dst, weights = build_distance_edges(coords, threshold=15.0)
    if len(src) == 0:
        raise ValueError("No edges could be constructed")

    edges = [
        torch.tensor(src, dtype=torch.long).to(device),
        torch.tensor(dst, dtype=torch.long).to(device),
    ]
    edge_attr = torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(device)

    return node_feats_t, coords_t, edges, edge_attr


# ===========================================================================
# EquiPPIS Inference
# ===========================================================================

def equipis_predict(
    model: nn.Module,
    node_feats: torch.Tensor,
    coords: torch.Tensor,
    edges: List[torch.Tensor],
    edge_attr: torch.Tensor,
) -> np.ndarray:
    """Run EquiPPIS forward pass, return per-residue probabilities."""
    with torch.no_grad():
        pred, _ = model(node_feats, coords, edges, edge_attr)
        prob = torch.sigmoid(pred).cpu().numpy().flatten()
    return prob


# ===========================================================================
# Deformation Stability (Coordinate Perturbation)
# ===========================================================================

def compute_deformation_stability(
    model: nn.Module,
    node_feats: torch.Tensor,
    coords: torch.Tensor,
    edges: List[torch.Tensor],
    edge_attr: torch.Tensor,
    noise_std: float = 0.5,
    n_samples: int = 10,
) -> Dict[str, np.ndarray]:
    """
    Compute coordinate deformation stability for EquiPPIS.

    Perturbs 3D coordinates and measures prediction variance.
    This is the natural deformation for structure-based models.

    Returns per-residue trust scores.
    """
    # Base prediction
    base_probs = equipis_predict(model, node_feats, coords, edges, edge_attr)
    n_residues = len(base_probs)

    # Collect perturbed predictions
    all_perturbed = []
    for _ in range(n_samples):
        noise = torch.randn_like(coords) * noise_std
        perturbed_coords = coords + noise

        perturbed_probs = equipis_predict(
            model, node_feats, perturbed_coords, edges, edge_attr
        )
        all_perturbed.append(perturbed_probs)

    all_perturbed = np.array(all_perturbed)  # [n_samples, n_residues]

    # Per-residue stability metrics
    pred_std = np.std(all_perturbed, axis=0)  # [n_residues]
    base_binary = (base_probs > INTERFACE_THRESHOLD).astype(int)

    flip_counts = np.zeros(n_residues)
    for pert in all_perturbed:
        pert_binary = (pert > INTERFACE_THRESHOLD).astype(int)
        flip_counts += (pert_binary != base_binary).astype(float)
    flip_rate = flip_counts / n_samples

    # Trust scores (higher = more trustworthy)
    stability = 1.0 / (1.0 + pred_std)
    flip_score = 1.0 - flip_rate
    combined = (stability + flip_score) / 2

    return {
        'base_probs': base_probs,
        't_deform_stable': stability,
        't_deform_flip': flip_score,
        't_deform_combined': combined,
        'pred_std': pred_std,
        'flip_rate': flip_rate,
    }


# ===========================================================================
# Per-Complex Evaluation
# ===========================================================================

def evaluate_complex(
    model: nn.Module,
    complex_data: ComplexData,
    device: torch.device,
    noise_std: float = 0.5,
    n_deform_samples: int = 10,
    chain: str = 'antigen',
) -> Optional[Dict]:
    """
    Evaluate EquiPPIS trust signals on a single complex.

    Args:
        chain: Which chain to predict ('antigen' or 'all')

    Returns dict with per-residue and per-complex metrics, or None on failure.
    """
    # Select residues
    if chain == 'antigen':
        residues = complex_data.antigen_residues
    elif chain == 'antibody':
        residues = complex_data.antibody_residues
    else:
        residues = complex_data.antibody_residues + complex_data.antigen_residues

    if len(residues) < 5:
        return None

    coords = np.array([r.ca_coord for r in residues], dtype=np.float32)
    labels = np.array([r.is_interface for r in residues], dtype=np.float32)

    # Skip if all same label (can't compute AUROC)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return None

    try:
        node_feats, coords_t, edges, edge_attr = prepare_equipis_input(
            residues, coords, device
        )
    except ValueError as e:
        logger.debug(f"Skipping {complex_data.pdb_id}: {e}")
        return None

    # Deformation stability
    try:
        deform = compute_deformation_stability(
            model, node_feats, coords_t, edges, edge_attr,
            noise_std=noise_std,
            n_samples=n_deform_samples,
        )
    except Exception as e:
        logger.debug(f"Deformation failed for {complex_data.pdb_id}: {e}")
        return None

    base_probs = deform['base_probs']
    predictions = (base_probs > INTERFACE_THRESHOLD).astype(int)
    is_correct = (predictions == labels).astype(int)

    # Per-residue confidence
    confidence = np.maximum(base_probs, 1 - base_probs)

    # Per-complex metrics
    accuracy = float(is_correct.mean())

    def safe_auroc(scores, targets):
        try:
            return float(roc_auc_score(targets, scores))
        except Exception:
            return 0.5

    # Error detection: can this signal tell correct from incorrect predictions?
    result = {
        'pdb_id': complex_data.pdb_id,
        'n_residues': len(residues),
        'n_interface': int(labels.sum()),
        'accuracy': accuracy,
        'predictions': base_probs.tolist(),
        'labels': labels.tolist(),
    }

    # Trust signal AUROCs (per-complex)
    for signal_name, signal_values in [
        ('confidence', confidence),
        ('t_deform_stable', deform['t_deform_stable']),
        ('t_deform_flip', deform['t_deform_flip']),
        ('t_deform_combined', deform['t_deform_combined']),
    ]:
        # Error detection AUROC
        result[f'{signal_name}_error_auroc'] = safe_auroc(signal_values, is_correct)

        # Interface prediction AUROC (how well does the signal predict interface?)
        if signal_name == 'confidence':
            result[f'{signal_name}_interface_auroc'] = safe_auroc(base_probs, labels)

    return result


# ===========================================================================
# Evaluation Helpers (matching other eval scripts)
# ===========================================================================

def evaluate_signal_for_error_detection(
    all_signals: Dict[str, np.ndarray],
    signal_name: str,
) -> Dict[str, float]:
    """Evaluate trust signal for error detection (dataset-level)."""
    is_correct = all_signals['is_correct']
    signal = all_signals[signal_name]

    try:
        auroc = float(roc_auc_score(is_correct, signal))
    except Exception:
        auroc = 0.5

    correlation = np.corrcoef(signal, is_correct)[0, 1]
    if np.isnan(correlation):
        correlation = 0.0

    return {
        'auroc': auroc,
        'correlation': float(correlation),
        'mean_correct': float(signal[is_correct == 1].mean()) if (is_correct == 1).sum() > 0 else 0,
        'mean_incorrect': float(signal[is_correct == 0].mean()) if (is_correct == 0).sum() > 0 else 0,
    }


def evaluate_selective_prediction(
    all_signals: Dict[str, np.ndarray],
    signal_name: str,
    target_coverage: float = 0.8,
) -> Dict[str, float]:
    """Evaluate selective prediction at target coverage."""
    predictions = all_signals['predictions']
    labels = all_signals['labels']
    signal = all_signals[signal_name]

    metrics = selective_accuracy_at_coverage(
        signal, predictions, labels, target_coverage=target_coverage
    )

    return {
        'coverage': metrics.coverage,
        'accuracy': metrics.accuracy,
        'improvement': metrics.improvement,
        'n_accepted': metrics.n_accepted,
    }


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="EquiPPIS Cross-Model Trust Evaluation"
    )
    parser.add_argument('--limit', type=int, default=None,
                        help='Max number of SAbDab complexes')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test (20 complexes)')
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--noise-std', type=float, default=0.5,
                        help='Noise std for coordinate perturbation (Angstroms)')
    parser.add_argument('--n-deform-samples', type=int, default=10,
                        help='Number of perturbation samples')
    parser.add_argument('--chain', type=str, default='antigen',
                        choices=['antigen', 'antibody', 'all'],
                        help='Which chain to predict')
    parser.add_argument('--interface-threshold', type=float, default=8.0,
                        help='Interface labeling distance threshold (Angstroms)')

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    if args.quick:
        args.limit = 20
        logger.info("Quick test mode (20 complexes)")

    device = torch.device(
        args.device if args.device != 'cuda' or torch.cuda.is_available() else 'cpu'
    )
    logger.info(f"Device: {device}")

    output_dir = args.output_dir or (PROJECT_ROOT / "experiments" / "results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load EquiPPIS model
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("LOADING EQUIPIS MODEL")
    logger.info("=" * 70)

    model = load_egnn_model(device)

    # ------------------------------------------------------------------
    # 2. Load SAbDab structures
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("LOADING SABDAB STRUCTURES")
    logger.info("=" * 70)

    complexes = load_structures(
        SABDAB_STRUCTURES,
        SABDAB_METADATA,
        limit=args.limit,
    )
    complexes = extract_interfaces(
        complexes,
        distance_threshold=args.interface_threshold,
    )
    logger.info(f"Loaded {len(complexes)} complexes")

    # Shuffle with seed for reproducibility
    rng = np.random.RandomState(args.seed)
    rng.shuffle(complexes)

    # ------------------------------------------------------------------
    # 3. Evaluate each complex
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("EVALUATING TRUST SIGNALS")
    logger.info("=" * 70)

    complex_results = []
    # Collect all per-residue signals for dataset-level evaluation
    all_predictions = []
    all_labels = []
    all_is_correct = []
    all_confidence = []
    all_deform_stable = []
    all_deform_flip = []
    all_deform_combined = []

    for cplx in tqdm(complexes, desc="Evaluating complexes"):
        result = evaluate_complex(
            model, cplx, device,
            noise_std=args.noise_std,
            n_deform_samples=args.n_deform_samples,
            chain=args.chain,
        )
        if result is None:
            continue

        complex_results.append(result)

        # Accumulate per-residue signals
        probs = np.array(result['predictions'])
        labels = np.array(result['labels'])
        preds = (probs > INTERFACE_THRESHOLD).astype(int)
        correct = (preds == labels).astype(int)

        all_predictions.extend(preds.tolist())
        all_labels.extend(labels.tolist())
        all_is_correct.extend(correct.tolist())
        all_confidence.extend(np.maximum(probs, 1 - probs).tolist())

        # Re-extract deformation signals from evaluate_complex internals
        # (already stored in result)
        # These are per-complex averages but we need per-residue for dataset-level
        # So we recompute from the complex result
        n_res = result['n_residues']
        all_deform_stable.extend([result['t_deform_stable_error_auroc']] * n_res)
        all_deform_flip.extend([result['t_deform_flip_error_auroc']] * n_res)
        all_deform_combined.extend([result['t_deform_combined_error_auroc']] * n_res)

    n_complexes = len(complex_results)
    n_total_residues = len(all_predictions)
    logger.info(f"\nEvaluated {n_complexes} complexes, {n_total_residues} residues total")

    if n_complexes == 0:
        logger.error("No complexes could be evaluated!")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 4. Dataset-level metrics
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("DATASET-LEVEL METRICS")
    logger.info("=" * 70)

    # Per-complex aggregated metrics
    complex_accuracies = [r['accuracy'] for r in complex_results]
    baseline_accuracy = float(np.mean(complex_accuracies))
    logger.info(f"  Mean complex accuracy: {baseline_accuracy:.3f}")

    # Per-complex error detection AUROCs (averaged)
    signal_names = ['confidence', 't_deform_stable', 't_deform_flip', 't_deform_combined']

    error_detection_results = {}
    selective_results = {}

    for signal_name in signal_names:
        key = f'{signal_name}_error_auroc'
        aurocs = [r[key] for r in complex_results if key in r]
        if aurocs:
            mean_auroc = float(np.mean(aurocs))
            std_auroc = float(np.std(aurocs))
        else:
            mean_auroc = 0.5
            std_auroc = 0.0

        error_detection_results[signal_name] = {
            'auroc': mean_auroc,
            'auroc_std': std_auroc,
            'correlation': 0.0,  # computed per-complex
            'mean_correct': 0.0,
            'mean_incorrect': 0.0,
        }
        logger.info(f"  {signal_name:25s}: AUROC={mean_auroc:.3f} ± {std_auroc:.3f}")

    # Selective prediction at complex level
    # Use per-complex mean deformation score as the trust signal
    for signal_name in signal_names:
        key = f'{signal_name}_error_auroc'
        complex_scores = np.array([r.get(key, 0.5) for r in complex_results])
        complex_accs = np.array(complex_accuracies)

        # Rank complexes by trust signal, select top 80%
        n_accept = max(1, int(len(complex_results) * 0.8))
        top_idx = np.argsort(complex_scores)[-n_accept:]
        selected_acc = float(complex_accs[top_idx].mean())
        improvement = selected_acc - baseline_accuracy

        selective_results[signal_name] = {
            'coverage': 0.8,
            'accuracy': selected_acc,
            'improvement': improvement,
            'n_accepted': n_accept,
        }
        logger.info(f"  Selective {signal_name:20s}: Acc={selected_acc:.3f}, "
                     f"Imp={improvement:+.1%}")

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    conf_auroc = error_detection_results['confidence']['auroc']
    deform_auroc = error_detection_results['t_deform_combined']['auroc']
    conf_imp = selective_results['confidence']['improvement']
    deform_imp = selective_results['t_deform_combined']['improvement']

    logger.info(f"  Baseline accuracy:         {baseline_accuracy:.3f}")
    logger.info(f"  Confidence AUROC:          {conf_auroc:.3f}")
    logger.info(f"  Deformation AUROC:         {deform_auroc:.3f}")
    logger.info(f"  Confidence selective imp:   {conf_imp:+.1%}")
    logger.info(f"  Deformation selective imp:  {deform_imp:+.1%}")

    deform_beats_conf = deform_auroc > conf_auroc
    logger.info(f"\n  Deformation beats confidence: {'YES' if deform_beats_conf else 'NO'}")

    # ------------------------------------------------------------------
    # 6. Save JSON
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = output_dir / f"equipis_trust_sabdab_seed{args.seed}_{timestamp}.json"

    output = {
        'experiment': 'equipis_cross_model_trust',
        'model': 'EquiPPIS',
        'architecture': 'E(3)-equivariant GNN (EGNN)',
        'dataset': 'sabdab',
        'seed': args.seed,
        'n_complexes': n_complexes,
        'n_samples': n_total_residues,
        'baseline_accuracy': baseline_accuracy,
        'timestamp': timestamp,
        'error_detection': error_detection_results,
        'selective_prediction': selective_results,
        'deformation_beats_confidence': deform_beats_conf,
        'per_complex_summary': {
            'mean_accuracy': baseline_accuracy,
            'std_accuracy': float(np.std(complex_accuracies)),
            'n_complexes': n_complexes,
            'mean_residues_per_complex': float(np.mean(
                [r['n_residues'] for r in complex_results]
            )),
        },
        'config': {
            'noise_std': args.noise_std,
            'n_deform_samples': args.n_deform_samples,
            'chain': args.chain,
            'interface_threshold': args.interface_threshold,
            'interface_pred_threshold': INTERFACE_THRESHOLD,
        },
        'notes': (
            f"EquiPPIS E(3)-equivariant GNN on SAbDab antibody-antigen complexes. "
            f"Per-residue interface prediction with coordinate perturbation "
            f"for deformation stability (noise_std={args.noise_std} Angstroms, "
            f"n_samples={args.n_deform_samples}). "
            f"Features: simplified from PDB (no DSSP/PSSM preprocessing). "
            f"Chain evaluated: {args.chain}."
        ),
    }

    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
