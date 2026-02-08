"""
SAbDab (Structural Antibody Database) Data Loader for TrustPPI Tier 2.

Loads antibody-antigen structural data for training geometric models.

Key functions:
- load_structures(): Parse PDB files and extract chain coordinates
- extract_interfaces(): Label interface residues (within distance threshold)
- create_graphs(): Build residue-level graphs for EGNN input
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
import logging

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ResidueData:
    """Single residue with coordinates and features."""
    chain_id: str
    res_id: int
    res_name: str
    ca_coord: np.ndarray  # Alpha carbon position [3]
    is_interface: bool = False


@dataclass
class ComplexData:
    """Antibody-antigen complex data."""
    pdb_id: str
    antibody_residues: List[ResidueData]
    antigen_residues: List[ResidueData]
    resolution: float = 0.0


@dataclass
class GraphData:
    """Graph representation for EGNN input."""
    node_features: torch.Tensor  # [N, feat_dim]
    coords: torch.Tensor         # [N, 3]
    edge_index: torch.Tensor     # [2, E]
    edge_attr: torch.Tensor      # [E, edge_dim]
    labels: torch.Tensor         # [N] interface labels
    chain_mask: torch.Tensor     # [N] 0=antibody, 1=antigen
    pdb_id: str


# ============================================================================
# PDB Parsing
# ============================================================================

# Standard amino acid 3-letter to 1-letter codes
AA_3TO1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
}

# One-hot encoding dimension
NUM_AMINO_ACIDS = 20
AA_TO_IDX = {aa: i for i, aa in enumerate(sorted(AA_3TO1.values()))}


def parse_pdb_file(
    pdb_path: Path,
    chains: List[str],
    atom_name: str = 'CA'
) -> Dict[str, List[ResidueData]]:
    """
    Parse PDB file and extract residue coordinates for specified chains.

    Args:
        pdb_path: Path to PDB file
        chains: List of chain IDs to extract
        atom_name: Atom to use for residue position (default: CA = alpha carbon)

    Returns:
        Dictionary mapping chain_id -> list of ResidueData
    """
    chain_residues: Dict[str, List[ResidueData]] = {c: [] for c in chains}
    seen_residues: Dict[str, set] = {c: set() for c in chains}

    try:
        with open(pdb_path, 'r') as f:
            for line in f:
                if not line.startswith('ATOM'):
                    continue

                chain_id = line[21]
                if chain_id not in chains:
                    continue

                atom = line[12:16].strip()
                if atom != atom_name:
                    continue

                res_name = line[17:20].strip()
                if res_name not in AA_3TO1:
                    continue  # Skip non-standard residues

                res_id = int(line[22:26])
                res_key = (chain_id, res_id)

                if res_key in seen_residues[chain_id]:
                    continue  # Skip alternate conformations
                seen_residues[chain_id].add(res_key)

                # Parse coordinates
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])

                chain_residues[chain_id].append(ResidueData(
                    chain_id=chain_id,
                    res_id=res_id,
                    res_name=res_name,
                    ca_coord=np.array([x, y, z], dtype=np.float32),
                    is_interface=False
                ))

    except Exception as e:
        logger.warning(f"Error parsing {pdb_path}: {e}")
        return {}

    return chain_residues


def load_structures(
    structures_dir: Path,
    metadata_file: Path,
    limit: Optional[int] = None,
    min_residues: int = 10
) -> List[ComplexData]:
    """
    Load antibody-antigen complex structures from SAbDab.

    Args:
        structures_dir: Directory containing PDB files
        metadata_file: Path to filtered_complexes.tsv
        limit: Maximum number of complexes to load
        min_residues: Minimum residues per chain to include

    Returns:
        List of ComplexData objects
    """
    import pandas as pd

    # Load metadata
    df = pd.read_csv(metadata_file, sep='\t')
    logger.info(f"Loaded metadata for {len(df)} complexes")

    # Filter for protein antigens only
    df = df[df['antigen_type'] == 'protein']
    logger.info(f"After filtering for protein antigens: {len(df)}")

    if limit:
        df = df.head(limit)

    complexes = []

    for _, row in df.iterrows():
        pdb_id = row['pdb'].lower()
        pdb_path = structures_dir / f"{pdb_id}.pdb"

        if not pdb_path.exists():
            continue

        # Get chain IDs - handle heavy+light chains
        ab_chains = []
        if pd.notna(row['Hchain']) and row['Hchain'] != 'NA':
            ab_chains.append(row['Hchain'])
        if pd.notna(row['Lchain']) and row['Lchain'] != 'NA':
            ab_chains.append(row['Lchain'])

        if not ab_chains:
            continue

        ag_chain = row['antigen_chain']
        if pd.isna(ag_chain) or ag_chain == 'NA':
            continue

        # Parse structure
        all_chains = ab_chains + [ag_chain]
        chain_data = parse_pdb_file(pdb_path, all_chains)

        if not chain_data:
            continue

        # Combine antibody chains
        ab_residues = []
        for chain in ab_chains:
            ab_residues.extend(chain_data.get(chain, []))

        ag_residues = chain_data.get(ag_chain, [])

        # Check minimum size
        if len(ab_residues) < min_residues or len(ag_residues) < min_residues:
            continue

        resolution = row['resolution'] if pd.notna(row['resolution']) else 0.0

        complexes.append(ComplexData(
            pdb_id=pdb_id,
            antibody_residues=ab_residues,
            antigen_residues=ag_residues,
            resolution=resolution
        ))

    logger.info(f"Successfully loaded {len(complexes)} complexes")
    return complexes


# ============================================================================
# Interface Extraction
# ============================================================================

def extract_interfaces(
    complexes: List[ComplexData],
    distance_threshold: float = 8.0
) -> List[ComplexData]:
    """
    Label interface residues based on distance between chains.

    A residue is labeled as interface if its CA atom is within
    distance_threshold of any CA atom in the partner chain.

    Args:
        complexes: List of ComplexData objects
        distance_threshold: Distance cutoff in Angstroms (default: 8.0)

    Returns:
        Same complexes with is_interface labels updated
    """
    threshold_sq = distance_threshold ** 2

    for complex_data in complexes:
        # Get coordinate arrays
        ab_coords = np.array([r.ca_coord for r in complex_data.antibody_residues])
        ag_coords = np.array([r.ca_coord for r in complex_data.antigen_residues])

        if len(ab_coords) == 0 or len(ag_coords) == 0:
            continue

        # Compute pairwise distances (squared)
        # [N_ab, 3] vs [N_ag, 3] -> [N_ab, N_ag]
        diff = ab_coords[:, None, :] - ag_coords[None, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1)

        # Label antibody interface residues
        ab_min_dist = np.min(dist_sq, axis=1)  # Min dist to any antigen residue
        for i, res in enumerate(complex_data.antibody_residues):
            res.is_interface = ab_min_dist[i] <= threshold_sq

        # Label antigen interface residues
        ag_min_dist = np.min(dist_sq, axis=0)  # Min dist to any antibody residue
        for i, res in enumerate(complex_data.antigen_residues):
            res.is_interface = ag_min_dist[i] <= threshold_sq

    return complexes


# ============================================================================
# Graph Construction
# ============================================================================

def residue_to_features(res: ResidueData) -> np.ndarray:
    """
    Convert residue to feature vector.

    Features:
    - One-hot amino acid encoding (20 dim)

    Returns:
        Feature vector [20]
    """
    one_hot = np.zeros(NUM_AMINO_ACIDS, dtype=np.float32)
    aa_1letter = AA_3TO1.get(res.res_name, 'X')
    if aa_1letter in AA_TO_IDX:
        one_hot[AA_TO_IDX[aa_1letter]] = 1.0
    return one_hot


def build_knn_edges(
    coords: np.ndarray,
    k: int = 10,
    cutoff: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build k-nearest neighbor edges with optional distance cutoff.

    Args:
        coords: Node coordinates [N, 3]
        k: Number of neighbors
        cutoff: Maximum distance for edges (Angstroms)

    Returns:
        edge_index [2, E], edge_distances [E]
    """
    from scipy.spatial.distance import cdist

    n = len(coords)
    k = min(k, n - 1)

    # Compute all pairwise distances
    dists = cdist(coords, coords)

    edges_src = []
    edges_dst = []
    edge_dists = []

    for i in range(n):
        # Get k nearest neighbors (excluding self)
        neighbor_dists = dists[i].copy()
        neighbor_dists[i] = np.inf  # Exclude self

        # Sort and take k nearest
        sorted_idx = np.argsort(neighbor_dists)[:k]

        for j in sorted_idx:
            d = neighbor_dists[j]
            if cutoff is None or d <= cutoff:
                edges_src.append(i)
                edges_dst.append(j)
                edge_dists.append(d)

    edge_index = np.array([edges_src, edges_dst], dtype=np.int64)
    edge_distances = np.array(edge_dists, dtype=np.float32)

    return edge_index, edge_distances


def create_graphs(
    complexes: List[ComplexData],
    k_neighbors: int = 10,
    distance_cutoff: float = 10.0
) -> List[GraphData]:
    """
    Convert complexes to graph representations for EGNN.

    Args:
        complexes: List of ComplexData with interface labels
        k_neighbors: Number of k-NN edges per node
        distance_cutoff: Maximum edge distance in Angstroms

    Returns:
        List of GraphData objects ready for EGNN
    """
    graphs = []

    for complex_data in complexes:
        all_residues = complex_data.antibody_residues + complex_data.antigen_residues
        n_ab = len(complex_data.antibody_residues)

        if len(all_residues) < 5:
            continue

        # Build node features and coordinates
        node_features = np.array([residue_to_features(r) for r in all_residues])
        coords = np.array([r.ca_coord for r in all_residues])
        labels = np.array([r.is_interface for r in all_residues], dtype=np.float32)

        # Chain mask: 0 = antibody, 1 = antigen
        chain_mask = np.zeros(len(all_residues), dtype=np.float32)
        chain_mask[n_ab:] = 1.0

        # Build edges
        edge_index, edge_distances = build_knn_edges(
            coords, k=k_neighbors, cutoff=distance_cutoff
        )

        if edge_index.size == 0:
            continue

        # Edge features: distance (can expand to RBF later)
        edge_attr = edge_distances[:, None]  # [E, 1]

        graphs.append(GraphData(
            node_features=torch.from_numpy(node_features),
            coords=torch.from_numpy(coords),
            edge_index=torch.from_numpy(edge_index),
            edge_attr=torch.from_numpy(edge_attr),
            labels=torch.from_numpy(labels),
            chain_mask=torch.from_numpy(chain_mask),
            pdb_id=complex_data.pdb_id
        ))

    logger.info(f"Created {len(graphs)} graphs from {len(complexes)} complexes")
    return graphs


# ============================================================================
# PyTorch Dataset
# ============================================================================

class SAbDabDataset(Dataset):
    """PyTorch Dataset for SAbDab antibody-antigen complexes."""

    def __init__(
        self,
        data_dir: Path,
        metadata_file: Optional[Path] = None,
        structures_dir: Optional[Path] = None,
        split: str = 'train',
        limit: Optional[int] = None,
        k_neighbors: int = 10,
        distance_cutoff: float = 10.0,
        interface_threshold: float = 8.0
    ):
        """
        Initialize SAbDab dataset.

        Args:
            data_dir: Base data directory (data/sabdab/)
            metadata_file: Path to metadata TSV (default: data_dir/filtered_complexes.tsv)
            structures_dir: Path to PDB structures (default: data_dir/structures/)
            split: 'train', 'val', or 'test'
            limit: Maximum number of complexes
            k_neighbors: KNN edges per node
            distance_cutoff: Maximum edge distance
            interface_threshold: Interface labeling distance
        """
        self.data_dir = Path(data_dir)
        self.metadata_file = metadata_file or self.data_dir / 'filtered_complexes.tsv'
        self.structures_dir = structures_dir or self.data_dir / 'structures'
        self.split = split
        self.k_neighbors = k_neighbors
        self.distance_cutoff = distance_cutoff
        self.interface_threshold = interface_threshold

        # Load and process data
        complexes = load_structures(
            self.structures_dir,
            self.metadata_file,
            limit=limit
        )

        # Extract interfaces
        complexes = extract_interfaces(complexes, distance_threshold=interface_threshold)

        # Create graphs
        self.graphs = create_graphs(
            complexes,
            k_neighbors=k_neighbors,
            distance_cutoff=distance_cutoff
        )

        # TODO: Implement proper train/val/test splits based on sequence clustering
        # For now, use simple ratio split
        n = len(self.graphs)
        train_end = int(0.8 * n)
        val_end = int(0.9 * n)

        if split == 'train':
            self.graphs = self.graphs[:train_end]
        elif split == 'val':
            self.graphs = self.graphs[train_end:val_end]
        elif split == 'test':
            self.graphs = self.graphs[val_end:]

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> GraphData:
        return self.graphs[idx]


def collate_graphs(batch: List[GraphData]) -> Dict[str, torch.Tensor]:
    """
    Collate multiple graphs into a batched graph.

    Combines graphs by offsetting edge indices and concatenating tensors.
    """
    if len(batch) == 0:
        return {}

    node_features = []
    coords = []
    edge_index = []
    edge_attr = []
    labels = []
    chain_mask = []
    batch_idx = []
    pdb_ids = []

    node_offset = 0

    for i, graph in enumerate(batch):
        n = graph.node_features.size(0)

        node_features.append(graph.node_features)
        coords.append(graph.coords)
        edge_index.append(graph.edge_index + node_offset)
        edge_attr.append(graph.edge_attr)
        labels.append(graph.labels)
        chain_mask.append(graph.chain_mask)
        batch_idx.append(torch.full((n,), i, dtype=torch.long))
        pdb_ids.append(graph.pdb_id)

        node_offset += n

    return {
        'node_features': torch.cat(node_features, dim=0),
        'coords': torch.cat(coords, dim=0),
        'edge_index': torch.cat(edge_index, dim=1),
        'edge_attr': torch.cat(edge_attr, dim=0),
        'labels': torch.cat(labels, dim=0),
        'chain_mask': torch.cat(chain_mask, dim=0),
        'batch': torch.cat(batch_idx, dim=0),
        'pdb_ids': pdb_ids
    }


def get_sabdab_dataloader(
    data_dir: Path,
    split: str = 'train',
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    **dataset_kwargs
) -> DataLoader:
    """
    Get DataLoader for SAbDab dataset.

    Args:
        data_dir: Base data directory
        split: 'train', 'val', or 'test'
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of data loading workers
        **dataset_kwargs: Additional arguments for SAbDabDataset

    Returns:
        DataLoader with graph batching
    """
    dataset = SAbDabDataset(data_dir, split=split, **dataset_kwargs)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_graphs
    )


# ============================================================================
# Download Utilities
# ============================================================================

def download_pdb_structures(
    pdb_ids: List[str],
    output_dir: Path,
    max_concurrent: int = 5
) -> List[str]:
    """
    Download PDB structure files from RCSB.

    Args:
        pdb_ids: List of PDB IDs to download
        output_dir: Directory to save PDB files
        max_concurrent: Maximum concurrent downloads

    Returns:
        List of successfully downloaded PDB IDs
    """
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor, as_completed

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def download_one(pdb_id: str) -> Optional[str]:
        pdb_id = pdb_id.lower()
        output_path = output_dir / f"{pdb_id}.pdb"

        if output_path.exists():
            return pdb_id  # Already downloaded

        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        try:
            urllib.request.urlretrieve(url, output_path)
            return pdb_id
        except Exception as e:
            logger.warning(f"Failed to download {pdb_id}: {e}")
            return None

    successful = []
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = {executor.submit(download_one, pdb): pdb for pdb in pdb_ids}

        for future in as_completed(futures):
            result = future.result()
            if result:
                successful.append(result)

    logger.info(f"Downloaded {len(successful)}/{len(pdb_ids)} PDB files")
    return successful


# ============================================================================
# Main / Testing
# ============================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    # Get project root
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data" / "sabdab"

    print(f"Data directory: {data_dir}")
    print(f"Metadata exists: {(data_dir / 'filtered_complexes.tsv').exists()}")
    print(f"Structures dir exists: {(data_dir / 'structures').exists()}")

    # Check for available structures
    structures_dir = data_dir / "structures"
    if structures_dir.exists():
        pdb_files = list(structures_dir.glob("*.pdb"))
        print(f"Found {len(pdb_files)} PDB files")
    else:
        print("No structures directory - run download first")
        sys.exit(0)

    # Test loading
    if len(list(structures_dir.glob("*.pdb"))) > 0:
        print("\nTesting data loading...")
        complexes = load_structures(
            structures_dir,
            data_dir / "filtered_complexes.tsv",
            limit=5
        )

        if complexes:
            print(f"Loaded {len(complexes)} complexes")

            # Extract interfaces
            complexes = extract_interfaces(complexes)

            for c in complexes:
                n_ab_interface = sum(1 for r in c.antibody_residues if r.is_interface)
                n_ag_interface = sum(1 for r in c.antigen_residues if r.is_interface)
                print(f"  {c.pdb_id}: Ab={len(c.antibody_residues)} ({n_ab_interface} interface), "
                      f"Ag={len(c.antigen_residues)} ({n_ag_interface} interface)")

            # Create graphs
            graphs = create_graphs(complexes)
            print(f"\nCreated {len(graphs)} graphs")

            if graphs:
                g = graphs[0]
                print(f"  First graph: {g.node_features.shape[0]} nodes, "
                      f"{g.edge_index.shape[1]} edges, "
                      f"{g.labels.sum().item():.0f} interface residues")
