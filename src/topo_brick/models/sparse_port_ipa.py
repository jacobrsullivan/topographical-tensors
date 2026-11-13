"""
WP-L3: Sparse Relative-Frame Graph Encoder & Discrete-Lattice Invariant Point Attention (IPA).

Provides both encoder backbones required for the controlled encoder ablation (Reviewer §5 & §9):
1. `RelativeFrameGNNEncoder`: Sparse graph attention over the Active Connection Frontier F_t
   using exact relative cubic rotations ΔR_uv in {0..23}, relative translations Δr_uv, and
   structural edge attributes.
2. `SparsePortIPAEncoder`: Full 3D Local-Frame Invariant Point Attention (IPA) operating on
   exact relative lattice transforms ΔT_uv = T_u^-1 ∘ T_v:
       p_{v -> u} = ΔR_uv * p_v + Δr_uv
   guaranteeing exact SE(3)_cubic invariance across all 24 3D orientations (upright + SNOT).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from topo_brick.interface.ldu_frames import SO3_CUBIC_MATRICES
from topo_brick.representation.interface_fields import FRONTIER_PORT_CHANNELS


class SparsePortAttentionLayer(nn.Module):
    """
    Single layer of sparse relative-frame attention over the Active Connection Frontier F_t,
    optionally augmented with 3D local-frame Invariant Point Attention (IPA).
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_query_points: int = 4,
        use_ipa_points: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.n_query_points = n_query_points
        self.use_ipa_points = use_ipa_points

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        # Factored relative frame embeddings: 24 proper cubic rotations + 3D relative offset + 6 edge attrs
        self.rot_emb = nn.Embedding(24, n_heads * self.d_head)
        self.rel_pos_mlp = nn.Sequential(
            nn.Linear(3 + 6, d_model),
            nn.SiLU(),
            nn.Linear(d_model, n_heads),
        )

        if use_ipa_points:
            # Local 3D query/key/value points in each port's intrinsic frame
            self.q_pts_proj = nn.Linear(d_model, n_heads * n_query_points * 3, bias=False)
            self.k_pts_proj = nn.Linear(d_model, n_heads * n_query_points * 3, bias=False)
            self.v_pts_proj = nn.Linear(d_model, n_heads * n_query_points * 3, bias=False)
            self.head_weights = nn.Parameter(torch.zeros(n_heads))
            out_in_dim = d_model + n_heads * n_query_points * 4
        else:
            out_in_dim = d_model

        self.out_proj = nn.Linear(out_in_dim, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

        # Register the 24 exact 3x3 rotation matrices of SO(3)_cubic as a buffer
        rot_tensor = torch.tensor(SO3_CUBIC_MATRICES, dtype=torch.float32)
        self.register_buffer("so3_matrices", rot_tensor, persistent=False)

    def forward(
        self,
        h: torch.Tensor,  # [N, d_model]
        edge_index: torch.Tensor,  # [2, E] int64
        edge_rel_rot_id: torch.Tensor,  # [E] int64
        edge_rel_pos: torch.Tensor,  # [E, 3] float32
        edge_attr: torch.Tensor,  # [E, 6] float32
    ) -> torch.Tensor:
        n_nodes = h.shape[0]
        if n_nodes == 0 or edge_index.shape[1] == 0:
            return h

        src = edge_index[0]  # u (receiver/query frame)
        dst = edge_index[1]  # v (sender/key frame)
        n_edges = src.shape[0]

        h_norm = self.norm1(h)
        q = self.q_proj(h_norm).view(n_nodes, self.n_heads, self.d_head)
        k = self.k_proj(h_norm).view(n_nodes, self.n_heads, self.d_head)
        v = self.v_proj(h_norm).view(n_nodes, self.n_heads, self.d_head)

        # Relative rotation key modulation r^{(K)}(ΔR_uv)
        r_k = self.rot_emb(edge_rel_rot_id).view(n_edges, self.n_heads, self.d_head)
        q_e = q[src]  # [E, H, D]
        k_e = k[dst] + r_k  # [E, H, D]

        # Content + relative frame dot product
        logits = (q_e * k_e).sum(dim=-1) / math.sqrt(self.d_head)  # [E, H]

        # Edge bias from relative position and topological edge attributes
        edge_feat = torch.cat([edge_rel_pos, edge_attr], dim=-1)  # [E, 9]
        logits = logits + self.rel_pos_mlp(edge_feat)  # [E, H]

        if self.use_ipa_points:
            q_pts = self.q_pts_proj(h_norm).view(n_nodes, self.n_heads, self.n_query_points, 3)
            k_pts = self.k_pts_proj(h_norm).view(n_nodes, self.n_heads, self.n_query_points, 3)
            v_pts = self.v_pts_proj(h_norm).view(n_nodes, self.n_heads, self.n_query_points, 3)

            # Transform k_pts[dst] from v's local frame into u's local frame via ΔT_uv = (ΔR_uv, Δr_uv)
            rot_uv = self.so3_matrices[edge_rel_rot_id]  # [E, 3, 3]
            k_pts_dst = k_pts[dst]  # [E, H, P, 3]
            k_pts_in_u = torch.einsum("eij,ehpj->ehpi", rot_uv, k_pts_dst) + edge_rel_pos.view(
                n_edges, 1, 1, 3
            )
            sq_dist = ((k_pts_in_u - q_pts[src]) ** 2).sum(dim=(-1, -2))  # [E, H]
            gamma = nn.functional.softplus(self.head_weights).view(1, self.n_heads)
            ipa_penalty = -0.5 * gamma * sq_dist / math.sqrt(3.0 * self.n_query_points)
            logits = logits + ipa_penalty

        # Sparse segment softmax over incoming edges for each node u in src
        max_per_node = torch.full(
            (n_nodes, self.n_heads), -1e9, dtype=logits.dtype, device=logits.device
        )
        max_per_node.scatter_reduce_(
            0, src.unsqueeze(-1).expand(-1, self.n_heads), logits, reduce="amax", include_self=True
        )
        exp_logits = torch.exp(logits - max_per_node[src])
        denom = torch.zeros(
            (n_nodes, self.n_heads), dtype=logits.dtype, device=logits.device
        )
        denom.scatter_add_(0, src.unsqueeze(-1).expand(-1, self.n_heads), exp_logits)
        attn = exp_logits / (denom[src] + 1e-8)  # [E, H]

        # Aggregate value vectors
        weighted_v = attn.unsqueeze(-1) * v[dst]  # [E, H, D]
        agg_v = torch.zeros(
            (n_nodes, self.n_heads, self.d_head), dtype=h.dtype, device=h.device
        )
        agg_v.scatter_add_(
            0,
            src.view(n_edges, 1, 1).expand(-1, self.n_heads, self.d_head),
            weighted_v,
        )
        out_feats = [agg_v.reshape(n_nodes, self.d_model)]

        if self.use_ipa_points:
            # Transform v_pts[dst] into u's local frame and aggregate
            v_pts_in_u = torch.einsum("eij,ehpj->ehpi", rot_uv, v_pts[dst]) + edge_rel_pos.view(
                n_edges, 1, 1, 3
            )
            weighted_pts = attn.view(n_edges, self.n_heads, 1, 1) * v_pts_in_u
            agg_pts = torch.zeros(
                (n_nodes, self.n_heads, self.n_query_points, 3),
                dtype=h.dtype,
                device=h.device,
            )
            agg_pts.scatter_add_(
                0,
                src.view(n_edges, 1, 1, 1).expand(-1, self.n_heads, self.n_query_points, 3),
                weighted_pts,
            )
            pts_norm = torch.sqrt((agg_pts**2).sum(dim=-1, keepdim=True) + 1e-8)
            out_feats.append(agg_pts.reshape(n_nodes, -1))
            out_feats.append(pts_norm.reshape(n_nodes, -1))

        combined = torch.cat(out_feats, dim=-1)
        h = h + self.out_proj(combined)
        h = h + self.ffn(self.norm2(h))
        return h


class SparsePortIPAEncoder(nn.Module):
    """
    3D Local-Frame Invariant Point Attention (IPA) trunk over the Active Connection Frontier.
    """

    def __init__(
        self,
        in_channels: int = FRONTIER_PORT_CHANNELS,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        n_query_points: int = 4,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )
        self.layers = nn.ModuleList(
            [
                SparsePortAttentionLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    n_query_points=n_query_points,
                    use_ipa_points=True,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        port_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_rel_rot_id: torch.Tensor,
        edge_rel_pos: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        h = self.input_proj(port_features)
        for layer in self.layers:
            h = layer(h, edge_index, edge_rel_rot_id, edge_rel_pos, edge_attr)
        return self.final_norm(h)


class RelativeFrameGNNEncoder(nn.Module):
    """
    Controlled ablation baseline (Reviewer §5 & §9): identical parameter structure and
    relative-frame edge features (ΔR_uv, Δr_uv, edge_attr) without 3D IPA point projections.
    """

    def __init__(
        self,
        in_channels: int = FRONTIER_PORT_CHANNELS,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )
        self.layers = nn.ModuleList(
            [
                SparsePortAttentionLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    use_ipa_points=False,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        port_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_rel_rot_id: torch.Tensor,
        edge_rel_pos: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        h = self.input_proj(port_features)
        for layer in self.layers:
            h = layer(h, edge_index, edge_rel_rot_id, edge_rel_pos, edge_attr)
        return self.final_norm(h)
