#!/usr/bin/env python3
"""
Cross-species label-free trust assessment.

For each (model, species) combo, loads or generates a per-pair CSV (from
analyze_per_pair.py), aggregates mean deformation score (label-free trust
proxy) and actual accuracy (ground-truth validation), then computes the
Pearson correlation between the two.

A strong correlation means deformation stability can rank model reliability
across species *without access to labels*.

Default species-model matrix:
  dscript:      yeast, ecoli, human
  tuna:         yeast, ecoli, fly
  plm_interact: yeast, ecoli, human

Usage:
    python scripts/analyze_cross_species_trust.py --quick
    python scripts/analyze_cross_species_trust.py
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_per_pair import (
    run_per_pair_analysis,
    identify_discordant_cases,
    save_per_pair_csv,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

TABLES_DIR = PROJECT_ROOT / "tables" / "csv"

# Default species-model matrix
DEFAULT_MATRIX = [
    ('dscript', 'yeast'),
    ('dscript', 'ecoli'),
    ('dscript', 'human'),
    ('tuna', 'yeast'),
    ('tuna', 'ecoli'),
    ('tuna', 'fly'),
    ('plm_interact', 'yeast'),
    ('plm_interact', 'ecoli'),
    ('plm_interact', 'human'),
]


# ============================================================================
# CSV loading
# ============================================================================

def load_per_pair_csv(path: Path) -> Dict:
    """Parse per-pair CSV and return aggregated stats.

    Returns:
        {'mean_deform_score': float, 'actual_accuracy': float, 'n_pairs': int}
    """
    deform_scores = []
    correct_count = 0
    total = 0

    with open(path, 'r') as f:
        header = f.readline().strip().split(',')
        deform_idx = header.index('deform_score')
        correct_idx = header.index('is_correct')

        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            deform_scores.append(float(parts[deform_idx]))
            correct_count += int(parts[correct_idx])
            total += 1

    if total == 0:
        return {'mean_deform_score': 0.0, 'actual_accuracy': 0.0, 'n_pairs': 0}

    return {
        'mean_deform_score': float(np.mean(deform_scores)),
        'actual_accuracy': correct_count / total,
        'n_pairs': total,
    }


# ============================================================================
# Ensure per-pair CSV exists
# ============================================================================

def ensure_per_pair_csv(
    model: str,
    species: str,
    device,
    quick: bool = False,
) -> Path:
    """Return path to per-pair CSV, generating it if needed."""
    csv_path = TABLES_DIR / f"per_pair_{model}_{species}.csv"

    if csv_path.exists():
        logger.info(f"  Found existing CSV: {csv_path.name}")
        return csv_path

    logger.info(f"  Generating per-pair CSV for {model}/{species}...")

    limit = 50 if quick else 500
    n_deform = 5 if quick else 10

    results, discordant = run_per_pair_analysis(
        model=model,
        dataset=species,
        limit=limit,
        seed=42,
        device=device,
        noise_std=0.1,
        n_deform=n_deform,
    )
    save_per_pair_csv(results, discordant, csv_path)
    return csv_path


# ============================================================================
# Correlation
# ============================================================================

def compute_and_print_correlation(rows: List[Dict]):
    """Compute Pearson r between mean_deform_score and actual_accuracy."""
    deform = np.array([r['mean_deform_score'] for r in rows])
    accuracy = np.array([r['actual_accuracy'] for r in rows])

    if len(rows) < 3:
        print("\n  Too few data points for meaningful correlation.")
        return

    try:
        from scipy.stats import pearsonr
        r, p = pearsonr(deform, accuracy)
        print(f"\n  Pearson r = {r:.3f}  (p = {p:.4f})")
    except ImportError:
        r = np.corrcoef(deform, accuracy)[0, 1]
        print(f"\n  Pearson r = {r:.3f}  (scipy not available; no p-value)")

    if abs(r) >= 0.7:
        print("  Interpretation: STRONG correlation — deformation stability")
        print("  reliably ranks model trustworthiness across species.")
    elif abs(r) >= 0.4:
        print("  Interpretation: MODERATE correlation — deformation stability")
        print("  provides a useful (but imperfect) label-free trust signal.")
    else:
        print("  Interpretation: WEAK correlation — deformation stability")
        print("  alone is insufficient as a cross-species trust proxy.")


# ============================================================================
# CSV output
# ============================================================================

def save_cross_species_csv(rows: List[Dict], path: Path):
    """Save cross-species summary CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)

    columns = ['model', 'species', 'mean_deform_score', 'actual_accuracy', 'n_pairs']

    with open(path, 'w') as f:
        f.write(','.join(columns) + '\n')
        for row in rows:
            values = [
                row['model'],
                row['species'],
                f"{row['mean_deform_score']:.4f}",
                f"{row['actual_accuracy']:.4f}",
                str(row['n_pairs']),
            ]
            f.write(','.join(values) + '\n')

    logger.info(f"Saved cross-species CSV: {path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cross-species label-free trust assessment"
    )
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode (limit=50, n_deform=5 for generation)')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    import torch
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    device = torch.device(
        args.device if args.device != 'cuda' or torch.cuda.is_available() else 'cpu'
    )

    print("=" * 60)
    print("CROSS-SPECIES LABEL-FREE TRUST ASSESSMENT")
    print("=" * 60)

    rows = []
    for model, species in DEFAULT_MATRIX:
        logger.info(f"Processing {model} / {species}...")
        csv_path = ensure_per_pair_csv(model, species, device, quick=args.quick)
        stats = load_per_pair_csv(csv_path)

        row = {
            'model': model,
            'species': species,
            'mean_deform_score': stats['mean_deform_score'],
            'actual_accuracy': stats['actual_accuracy'],
            'n_pairs': stats['n_pairs'],
        }
        rows.append(row)

        print(f"  {model:15s} {species:8s}  deform={stats['mean_deform_score']:.3f}  "
              f"acc={stats['actual_accuracy']:.3f}  n={stats['n_pairs']}")

    # Save summary
    csv_path = TABLES_DIR / "cross_species_trust_assessment.csv"
    save_cross_species_csv(rows, csv_path)

    # Correlation
    print(f"\n{'='*60}")
    print("CORRELATION ANALYSIS")
    print(f"{'='*60}")
    compute_and_print_correlation(rows)

    print(f"\n{'='*60}")
    print(f"Output: {csv_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
