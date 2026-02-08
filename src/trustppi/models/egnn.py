"""
E(n)-Equivariant Graph Neural Network for PPI Prediction.

Adapted from Satorras et al. (2021) EGNN implementation for
protein-protein interaction prediction on antibody-antigen complexes.

Key features:
- E(n)-equivariant message passing (rotation/translation invariant)
- Interface residue prediction (per-node classification)
- Symmetric pooling for pair-level prediction (swap invariant)
- Feature extraction for trust wrapper
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# EGNN Core Layers (adapted from external/egnn)
# ============================================================================

class E_GCL(nn.Module):
    """
    E(n) Equivariant Convolutional Layer.

    Uses distance-based edge features (SE(3)-invariant) and optionally
    updates coordinates in an equivariant manner.
    """

    def __init__(
        self,
        input_nf: int,
        output_nf: int,
        hidden_nf: int,
        edges_in_d: int = 0,
        act_fn: nn.Module = nn.SiLU(),
        residual: bool = True,
        attention: bool = False,
        normalize: bool = False,
        coords_agg: str = 'mean',
        tanh: bool = False,
        update_coords: bool = True
    ):
        super().__init__()
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.update_coords = update_coords
        self.epsilon = 1e-8

        input_edge = input_nf * 2
        edge_coords_nf = 1  # Distance feature

        # Edge MLP
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn
        )

        # Node MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf)
        )

        # Coordinate update MLP (optional)
        if update_coords:
            layer = nn.Linear(hidden_nf, 1, bias=False)
            nn.init.xavier_uniform_(layer.weight, gain=0.001)

            coord_mlp = [nn.Linear(hidden_nf, hidden_nf), act_fn, layer]
            if self.tanh:
                coord_mlp.append(nn.Tanh())
            self.coord_mlp = nn.Sequential(*coord_mlp)
        else:
            self.coord_mlp = None

        # Attention (optional)
        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid()
            )

    def edge_model(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        radial: torch.Tensor,
        edge_attr: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Compute edge features from node pairs and distance."""
        if edge_attr is None:
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)

        out = self.edge_mlp(out)

        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val

        return out

    def node_model(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_attr: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Aggregate edge features to nodes."""
        row, col = edge_index

        # Aggregate messages
        agg = self._unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))

        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)

        out = self.node_mlp(agg)

        if self.residual:
            out = x + out

        return out, agg

    def coord_model(
        self,
        coord: torch.Tensor,
        edge_index: torch.Tensor,
        coord_diff: torch.Tensor,
        edge_feat: torch.Tensor
    ) -> torch.Tensor:
        """Update coordinates in equivariant manner."""
        if self.coord_mlp is None:
            return coord

        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)

        if self.coords_agg == 'sum':
            agg = self._unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = self._unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise ValueError(f"Unknown coords_agg: {self.coords_agg}")

        return coord + agg

    def coord2radial(
        self,
        edge_index: torch.Tensor,
        coord: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute distance (invariant) and direction (equivariant) from coordinates."""
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff ** 2, dim=1, keepdim=True)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        coord: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        node_attr: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            h: Node features [N, hidden_nf]
            edge_index: Edge indices [2, E]
            coord: Node coordinates [N, 3]
            edge_attr: Edge features [E, edge_dim]
            node_attr: Additional node features

        Returns:
            Updated h, coord, edge_attr
        """
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, _ = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr

    @staticmethod
    def _unsorted_segment_sum(
        data: torch.Tensor,
        segment_ids: torch.Tensor,
        num_segments: int
    ) -> torch.Tensor:
        """Segment sum aggregation."""
        result = data.new_zeros((num_segments, data.size(1)))
        segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
        result.scatter_add_(0, segment_ids, data)
        return result

    @staticmethod
    def _unsorted_segment_mean(
        data: torch.Tensor,
        segment_ids: torch.Tensor,
        num_segments: int
    ) -> torch.Tensor:
        """Segment mean aggregation."""
        result = data.new_zeros((num_segments, data.size(1)))
        count = data.new_zeros((num_segments, data.size(1)))
        segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
        result.scatter_add_(0, segment_ids, data)
        count.scatter_add_(0, segment_ids, torch.ones_like(data))
        return result / count.clamp(min=1)


# ============================================================================
# EGNN Backbone
# ============================================================================

class EGNN(nn.Module):
    """
    E(n)-Equivariant Graph Neural Network.

    Stacks multiple E_GCL layers for deep message passing.
    """

    def __init__(
        self,
        in_node_nf: int,
        hidden_nf: int,
        out_node_nf: int,
        in_edge_nf: int = 0,
        n_layers: int = 4,
        residual: bool = True,
        attention: bool = False,
        normalize: bool = False,
        tanh: bool = False,
        update_coords: bool = True
    ):
        """
        Initialize EGNN.

        Args:
            in_node_nf: Input node feature dimension
            hidden_nf: Hidden dimension
            out_node_nf: Output node feature dimension
            in_edge_nf: Input edge feature dimension
            n_layers: Number of EGNN layers
            residual: Use residual connections
            attention: Use attention in message passing
            normalize: Normalize coordinate updates
            tanh: Apply tanh to coordinate updates
            update_coords: Whether to update coordinates
        """
        super().__init__()
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers

        # Input embedding
        self.embedding_in = nn.Linear(in_node_nf, hidden_nf)

        # EGNN layers
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(E_GCL(
                input_nf=hidden_nf,
                output_nf=hidden_nf,
                hidden_nf=hidden_nf,
                edges_in_d=in_edge_nf,
                residual=residual,
                attention=attention,
                normalize=normalize,
                tanh=tanh,
                update_coords=update_coords
            ))

        # Output embedding
        self.embedding_out = nn.Linear(hidden_nf, out_node_nf)

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            h: Node features [N, in_node_nf]
            x: Node coordinates [N, 3]
            edge_index: Edge indices [2, E]
            edge_attr: Edge features [E, edge_dim]

        Returns:
            Updated node features [N, out_node_nf], coordinates [N, 3]
        """
        h = self.embedding_in(h)

        for layer in self.layers:
            h, x, _ = layer(h, edge_index, x, edge_attr=edge_attr)

        h = self.embedding_out(h)
        return h, x


# ============================================================================
# PPI-Specific Model
# ============================================================================

class EGNN_PPI(nn.Module):
    """
    EGNN model for Protein-Protein Interaction prediction.

    Supports:
    - Per-residue interface classification
    - Pair-level binding prediction with swap-invariant pooling
    - Feature extraction for trust wrapper
    """

    def __init__(
        self,
        in_node_nf: int = 20,  # One-hot amino acids
        hidden_nf: int = 128,
        n_layers: int = 4,
        attention: bool = True,
        dropout: float = 0.1,
        task: str = 'interface'  # 'interface' or 'binding'
    ):
        """
        Initialize EGNN for PPI.

        Args:
            in_node_nf: Input feature dimension (20 for one-hot AA)
            hidden_nf: Hidden dimension
            n_layers: Number of EGNN layers
            attention: Use attention in message passing
            dropout: Dropout rate
            task: 'interface' for residue-level, 'binding' for pair-level
        """
        super().__init__()
        self.hidden_nf = hidden_nf
        self.task = task

        # EGNN backbone
        self.egnn = EGNN(
            in_node_nf=in_node_nf,
            hidden_nf=hidden_nf,
            out_node_nf=hidden_nf,
            in_edge_nf=1,  # Edge distance
            n_layers=n_layers,
            residual=True,
            attention=attention,
            normalize=True,
            update_coords=False  # Don't update coords for prediction
        )

        self.dropout = nn.Dropout(dropout)

        if task == 'interface':
            # Per-residue interface classifier
            self.classifier = nn.Sequential(
                nn.Linear(hidden_nf, hidden_nf // 2),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_nf // 2, 1)
            )
        elif task == 'binding':
            # Pair-level binding predictor with symmetric pooling
            self.pool_mlp = nn.Sequential(
                nn.Linear(hidden_nf * 2, hidden_nf),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_nf, 1)
            )
        else:
            raise ValueError(f"Unknown task: {task}")

    def forward(
        self,
        node_features: torch.Tensor,
        coords: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        chain_mask: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            node_features: [N, in_node_nf] node features
            coords: [N, 3] node coordinates
            edge_index: [2, E] edge indices
            edge_attr: [E, 1] edge distances
            chain_mask: [N] 0=chain A, 1=chain B
            batch: [N] batch assignment for each node
            return_features: Whether to return internal features

        Returns:
            Dictionary with:
            - 'logits': Prediction logits
            - 'features': (optional) Internal features for trust wrapper
        """
        # EGNN forward
        h, _ = self.egnn(node_features, coords, edge_index, edge_attr)
        h = self.dropout(h)

        output = {}

        if self.task == 'interface':
            # Per-residue interface prediction
            logits = self.classifier(h).squeeze(-1)
            output['logits'] = logits

        elif self.task == 'binding':
            # Symmetric pair-level pooling
            if chain_mask is None or batch is None:
                raise ValueError("binding task requires chain_mask and batch")

            # Pool each chain separately
            unique_batches = batch.unique()
            binding_logits = []

            for b in unique_batches:
                mask = batch == b
                h_b = h[mask]
                chain_b = chain_mask[mask]

                # Mean pool for each chain
                h_ab = h_b[chain_b == 0].mean(dim=0)  # Antibody
                h_ag = h_b[chain_b == 1].mean(dim=0)  # Antigen

                # Symmetric combination: concat sorted features
                # This ensures f(A,B) = f(B,A)
                h_pair = torch.stack([h_ab, h_ag], dim=0)  # [2, hidden]
                h_sorted, _ = torch.sort(h_pair, dim=0)  # Sort along chain dim
                h_sym = h_sorted.flatten()  # [hidden * 2]

                binding_logits.append(self.pool_mlp(h_sym))

            output['logits'] = torch.stack(binding_logits).squeeze(-1)

        if return_features:
            output['features'] = h

        return output

    def predict_proba(
        self,
        node_features: torch.Tensor,
        coords: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        chain_mask: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get probability predictions."""
        output = self.forward(
            node_features, coords, edge_index, edge_attr,
            chain_mask, batch
        )
        return torch.sigmoid(output['logits'])


# ============================================================================
# Loss Functions
# ============================================================================

class InterfaceLoss(nn.Module):
    """Weighted BCE loss for interface prediction with class imbalance."""

    def __init__(self, pos_weight: float = 5.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Compute weighted BCE loss.

        Args:
            logits: [N] predicted logits
            labels: [N] binary labels
            reduction: 'mean', 'sum', or 'none'
        """
        pos_weight = torch.tensor([self.pos_weight], device=logits.device)
        return F.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=pos_weight, reduction=reduction
        )


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing EGNN_PPI...")

    # Create dummy data
    n_nodes = 50
    n_edges = 200

    node_features = torch.randn(n_nodes, 20)
    coords = torch.randn(n_nodes, 3)
    edge_index = torch.randint(0, n_nodes, (2, n_edges))
    edge_attr = torch.randn(n_edges, 1).abs()
    chain_mask = torch.zeros(n_nodes)
    chain_mask[25:] = 1
    labels = torch.zeros(n_nodes)
    labels[20:30] = 1  # Some interface residues
    batch = torch.zeros(n_nodes, dtype=torch.long)

    # Test interface model
    print("\nInterface prediction model:")
    model = EGNN_PPI(task='interface')
    output = model(node_features, coords, edge_index, edge_attr, return_features=True)
    print(f"  Logits shape: {output['logits'].shape}")
    print(f"  Features shape: {output['features'].shape}")

    # Test loss
    loss_fn = InterfaceLoss()
    loss = loss_fn(output['logits'], labels)
    print(f"  Loss: {loss.item():.4f}")

    # Test binding model
    print("\nBinding prediction model:")
    model_binding = EGNN_PPI(task='binding')
    output_binding = model_binding(
        node_features, coords, edge_index, edge_attr,
        chain_mask=chain_mask, batch=batch
    )
    print(f"  Logits shape: {output_binding['logits'].shape}")

    print("\nAll tests passed!")
