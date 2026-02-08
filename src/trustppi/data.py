"""Data loading utilities for TrustPPI experiments."""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd


# Project root: /root/TrustPPI
# __file__ = /root/TrustPPI/src/trustppi/data.py
PROJECT_ROOT = Path(__file__).parent.parent.parent  # /root/TrustPPI

# Data paths
DATA_ROOT = PROJECT_ROOT / "data"
EXTERNAL_ROOT = PROJECT_ROOT / "external"

# PPI Data paths
PPI_PAIRS = DATA_ROOT / "ppi/pairs"
PPI_SEQS = DATA_ROOT / "ppi/seqs"

# Available pair files
ECOLI_TOY = PPI_PAIRS / "ecoli_toy.tsv"
ECOLI_TEST = PPI_PAIRS / "ecoli_test.tsv"
YEAST_TEST = PPI_PAIRS / "yeast_test.tsv"
HUMAN_TEST = PPI_PAIRS / "human_test.tsv"
HUMAN_TRAIN = PPI_PAIRS / "human_train.tsv"
MOUSE_TEST = PPI_PAIRS / "mouse_test.tsv"
FLY_TEST = PPI_PAIRS / "fly_test.tsv"
WORM_TEST = PPI_PAIRS / "worm_test.tsv"

# Sequence files
ECOLI_FASTA = PPI_SEQS / "ecoli.fasta"
YEAST_FASTA = PPI_SEQS / "yeast.fasta"
HUMAN_FASTA = PPI_SEQS / "human.fasta"
MOUSE_FASTA = PPI_SEQS / "mouse.fasta"
FLY_FASTA = PPI_SEQS / "fly.fasta"
WORM_FASTA = PPI_SEQS / "worm.fasta"


def load_pairs(
    pairs_file: Path,
    limit: Optional[int] = None,
    balanced: bool = False,
    seed: int = 42
) -> List[Tuple[str, str, float]]:
    """
    Load protein-protein interaction pairs from TSV file.

    Args:
        pairs_file: Path to TSV file with format: protein_a, protein_b, label
        limit: Maximum number of pairs to load (None for all)
        balanced: If True, load equal numbers of positive and negative pairs
        seed: Random seed for balanced sampling

    Returns:
        List of (protein_a_id, protein_b_id, label) tuples
    """
    df = pd.read_csv(pairs_file, sep='\t', header=None)
    df.columns = ['protein_a', 'protein_b', 'label']

    if balanced and limit is not None:
        # Sample equal numbers of positive and negative pairs
        positives = df[df['label'] == 1.0]
        negatives = df[df['label'] == 0.0]

        n_per_class = limit // 2

        # Sample from each class
        pos_sample = positives.sample(n=min(n_per_class, len(positives)), random_state=seed)
        neg_sample = negatives.sample(n=min(n_per_class, len(negatives)), random_state=seed)

        df = pd.concat([pos_sample, neg_sample]).sample(frac=1, random_state=seed)
    elif limit is not None:
        df = df.head(limit)

    pairs = [(row['protein_a'], row['protein_b'], row['label'])
             for _, row in df.iterrows()]
    return pairs


def load_sequences_for_pairs(
    pairs: List[Tuple[str, str, float]],
    fasta_file: Path
) -> Dict[str, str]:
    """
    Load sequences for proteins in the given pairs from a FASTA file.

    Args:
        pairs: List of (protein_a_id, protein_b_id, label) tuples
        fasta_file: Path to FASTA file with sequences

    Returns:
        Dictionary mapping protein IDs to sequences
    """
    # Get unique protein IDs needed
    protein_ids = set()
    for p_a, p_b, _ in pairs:
        protein_ids.add(str(p_a))
        protein_ids.add(str(p_b))

    # Parse FASTA file
    sequences = {}
    current_id = None
    current_seq = []

    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Save previous sequence
                if current_id is not None and current_id in protein_ids:
                    sequences[current_id] = ''.join(current_seq)
                # Start new sequence
                current_id = line[1:].split()[0]  # Take first word as ID
                current_seq = []
            else:
                current_seq.append(line)

        # Don't forget the last sequence
        if current_id is not None and current_id in protein_ids:
            sequences[current_id] = ''.join(current_seq)

    return sequences


def get_dataset_paths(dataset: str) -> Tuple[Path, Path]:
    """
    Get pairs and sequences file paths for a dataset.

    Args:
        dataset: One of 'ecoli_toy', 'ecoli', 'yeast', 'human'

    Returns:
        Tuple of (pairs_file, seqs_file)
    """
    dataset_map = {
        'ecoli_toy': (ECOLI_TOY, ECOLI_FASTA),
        'ecoli': (ECOLI_TEST, ECOLI_FASTA),
        'yeast': (YEAST_TEST, YEAST_FASTA),
        'human': (HUMAN_TEST, HUMAN_FASTA),
        'mouse': (MOUSE_TEST, MOUSE_FASTA),
        'fly': (FLY_TEST, FLY_FASTA),
        'worm': (WORM_TEST, WORM_FASTA),
    }

    if dataset not in dataset_map:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from {list(dataset_map.keys())}")

    return dataset_map[dataset]
