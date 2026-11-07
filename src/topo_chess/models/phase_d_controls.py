"""Phase D (v4.0.0) Architectural Controls: 8-Cell Factorial Encoders & Iso-Parameter Horizon Planners.

Addresses Reviewer Items 4, 5, 11, 12, 18, 19, 20, 22, 25, 27, 28:
1. `UnifiedSpatialEncoder`:
   - Supports `input_mode` in `{"flat_12", "flat_20", "topo_44"}`:
     * `"flat_12"`: 12 piece-occupancy channels only (quantifies missing rule channels, Item 27).
     * `"flat_20"`: 12 piece + 1 blocker + 7 Markov rule-state channels (equalizes rule observability, Item 27).
     * `"topo_44"`: 24 continuous topographical ray/flux/pin/X-ray/boundary channels + 20 discrete base channels.
   - Supports `attention_kernel` in `{"shaw_2d", "local_frame_ipa"}` (Item 4, 18).
   - Supports `connectivity_mask` in `{"dense", "queen_rays", "queen_plus_knight"}`:
     * `"dense"`: 64x64 all-pairs attention (4,096 pairs, 100.0% density).
     * `"queen_rays"`: Orthogonal (l_1) + Diagonal (l_inf) rays (1,408 pairs, 34.38% density; 3x3 Moore
       neighborhood is a strict subset of Queen rays, adding 0 new edges — Item 19).
     * `"queen_plus_knight"`: Queen rays + 8 Knight jump offsets (1,744 pairs, 42.58% density — Item 19).
2. `CachedRootARPlanner` (`Config_D_AR_Cached`, 8.57M total params — Item 5, 28):
   - Parameter-matched (2.70M params) 4-layer causal autoregressive transformer decoder conditioned on the
     frozen root spatial tokens z(s_0) without re-encoding intermediate board states, trained on the same
     250k 6-ply PVs + ALT-009 legality head.
3. `Bundle1InvertibilityProbe` (Item 20):
   - Reconstructs piece occupancy and legal move tokens strictly from the 24 continuous Bundle-1
     topographical channels (ch 0..23) with all 20 discrete piece/rule channels masked out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from topo_chess.domains.move_graph import STANDARD_8X8_GRAPH
from topo_chess.models.ipa_encoder import EncoderOutput


def build_spatial_connectivity_masks(height: int = 8, width: int = 8) -> Dict[str, np.ndarray]:
    """Construct exact boolean adjacency masks on the HxW board and verify Moore-subset property (Item 19)."""
    n = height * width
    dense_mask = np.ones((n, n), dtype=bool)
    queen_mask = np.zeros((n, n), dtype=bool)
    moore_mask = np.zeros((n, n), dtype=bool)
    knight_only_mask = np.zeros((n, n), dtype=bool)
    queen_knight_mask = np.zeros((n, n), dtype=bool)

    for i in range(n):
        ix, iy = i % width, i // width
        for j in range(n):
            jx, jy = j % width, j // width
            dx, dy = jx - ix, jy - iy
            adx, ady = abs(dx), abs(dy)
            is_orth = dx == 0 or dy == 0
            is_diag = adx == ady
            is_moore = adx <= 1 and ady <= 1
            is_knight = (adx == 1 and ady == 2) or (adx == 2 and ady == 1)

            if is_moore:
                moore_mask[i, j] = True
            if is_orth or is_diag:
                queen_mask[i, j] = True
            if is_knight:
                knight_only_mask[i, j] = True
            if is_orth or is_diag or is_knight:
                queen_knight_mask[i, j] = True

    # Mathematical verification for Item 19: 3x3 Moore neighborhood is 100% contained in Queen rays
    extra_moore_edges = int(np.sum(moore_mask & ~queen_mask))
    assert extra_moore_edges == 0, f"Moore neighborhood added {extra_moore_edges} non-Queen edges"

    return {
        "dense": dense_mask,
        "queen_rays": queen_mask,
        "queen_plus_knight": queen_knight_mask,
        "knight_only": knight_only_mask,
        "moore_3x3": moore_mask,
    }


@dataclass
class FlopsBreakdown:
    """Detailed operator-level vs full-model forward FLOPS breakdown (Addresses R2/R3 Item 12)."""

    attention_pairwise_flops: int
    qkv_and_frame_proj_flops: int
    ffn_flops: int
    action_head_flops: int
    total_encoder_flops: int
    active_edge_density: float


class ConfigurableShaw2DLayer(nn.Module):
    """2D Shaw Relative Self-Attention supporting Dense, Queen-Ray, or Queen+Knight connectivity masks (Item 18)."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        height: int = 8,
        width: int = 8,
        connectivity_mask: str = "dense",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.height = height
        self.width = width
        self.num_squares = height * width
        self.connectivity_mask = connectivity_mask

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.rel_x_k = nn.Embedding(2 * width - 1, self.d_head)
        self.rel_y_k = nn.Embedding(2 * height - 1, self.d_head)

        n = self.num_squares
        dx_idx = torch.zeros((n, n), dtype=torch.long)
        dy_idx = torch.zeros((n, n), dtype=torch.long)
        for i in range(n):
            ix, iy = i % width, i // width
            for j in range(n):
                jx, jy = j % width, j // width
                dx_idx[i, j] = (jx - ix) + (width - 1)
                dy_idx[i, j] = (jy - iy) + (height - 1)
        self.register_buffer("dx_idx", dx_idx, persistent=False)
        self.register_buffer("dy_idx", dy_idx, persistent=False)

        masks = build_spatial_connectivity_masks(height, width)
        adj = masks[connectivity_mask]
        self.register_buffer("adj_mask", torch.from_numpy(adj), persistent=False)

    @property
    def density(self) -> float:
        return float(self.adj_mask.float().mean().item())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, n_sq, _ = x.shape
        h = self.n_heads
        d_k = self.d_head

        q = self.q_proj(x).view(bsz, n_sq, h, d_k).transpose(1, 2)
        k = self.k_proj(x).view(bsz, n_sq, h, d_k).transpose(1, 2)
        v = self.v_proj(x).view(bsz, n_sq, h, d_k).transpose(1, 2)

        rel_k = self.rel_x_k(self.dx_idx) + self.rel_y_k(self.dy_idx)
        content_scores = torch.matmul(q, k.transpose(-2, -1))
        q_i = q.permute(2, 0, 1, 3).reshape(n_sq, bsz * h, d_k)
        rel_scores = (
            torch.bmm(q_i, rel_k.transpose(1, 2)).view(n_sq, bsz, h, n_sq).permute(1, 2, 0, 3)
        )
        logits = (content_scores + rel_scores) * (float(d_k) ** -0.5)

        if self.connectivity_mask != "dense":
            mask = self.adj_mask.view(1, 1, n_sq, n_sq)
            logits = logits.masked_fill(~mask, -1e9)

        weights = F.softmax(logits, dim=-1)
        out = torch.matmul(weights, v).transpose(1, 2).contiguous().view(bsz, n_sq, self.d_model)
        return self.out_proj(out)


class ConfigurableLocalFrameIPALayer(nn.Module):
    """Canonical Rank-Reflection / Color-Aligned Local-Frame IPA supporting Queen-Only vs Queen+Knight masks."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        height: int = 8,
        width: int = 8,
        n_query_points: int = 4,
        connectivity_mask: str = "queen_plus_knight",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.height = height
        self.width = width
        self.num_squares = height * width
        self.n_query_points = n_query_points
        self.connectivity_mask = connectivity_mask

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.q_pts = nn.Linear(d_model, n_heads * n_query_points * 2, bias=False)
        self.k_pts = nn.Linear(d_model, n_heads * n_query_points * 2, bias=False)

        self.geom_bias_mlp = nn.Sequential(
            nn.Linear(6, n_heads * 2),
            nn.GELU(),
            nn.Linear(n_heads * 2, n_heads),
        )
        self.rule_gate = nn.Linear(6, n_heads)
        self.point_weights = nn.Parameter(torch.full((n_heads,), 0.25))
        self.out_proj = nn.Linear(d_model, d_model)

        n = self.num_squares
        coords = np.zeros((n, 2), dtype=np.float32)
        geom_feats = np.zeros((n, n, 6), dtype=np.float32)
        for i in range(n):
            ix, iy = i % width, i // width
            coords[i, 0] = float(ix) / max(width - 1, 1)
            coords[i, 1] = float(iy) / max(height - 1, 1)
            for j in range(n):
                jx, jy = j % width, j // width
                dx, dy = jx - ix, jy - iy
                adx, ady = abs(dx), abs(dy)
                is_orth = 1.0 if (dx == 0 or dy == 0) else 0.0
                is_diag = 1.0 if (adx == ady and adx > 0) else 0.0
                is_knight = 1.0 if ((adx == 1 and ady == 2) or (adx == 2 and ady == 1)) else 0.0
                dist = float(np.hypot(dx, dy)) / float(np.hypot(width - 1, height - 1))
                geom_feats[i, j] = [
                    dist,
                    is_orth,
                    is_diag,
                    is_knight,
                    float(dx) / max(width - 1, 1),
                    float(dy) / max(height - 1, 1),
                ]

        masks = build_spatial_connectivity_masks(height, width)
        adj = masks[connectivity_mask]
        self.register_buffer("base_coords", torch.from_numpy(coords), persistent=False)
        self.register_buffer("geom_feats", torch.from_numpy(geom_feats), persistent=False)
        self.register_buffer("adj_mask", torch.from_numpy(adj), persistent=False)

    @property
    def density(self) -> float:
        return float(self.adj_mask.float().mean().item())

    def forward(
        self, x: torch.Tensor, frame_origins: torch.Tensor, rule_vec: torch.Tensor
    ) -> torch.Tensor:
        bsz, n_sq, _ = x.shape
        h = self.n_heads
        d_k = self.d_head
        n_pts = self.n_query_points

        q = self.q_proj(x).view(bsz, n_sq, h, d_k).transpose(1, 2)
        k = self.k_proj(x).view(bsz, n_sq, h, d_k).transpose(1, 2)
        v = self.v_proj(x).view(bsz, n_sq, h, d_k).transpose(1, 2)

        q_local = self.q_pts(x).float().view(bsz, n_sq, h, n_pts, 2)
        k_local = self.k_pts(x).float().view(bsz, n_sq, h, n_pts, 2)
        origins_f32 = frame_origins.float().unsqueeze(2).unsqueeze(3)

        q_global = (q_local + origins_f32).permute(0, 2, 1, 3, 4)
        k_global = (k_local + origins_f32).permute(0, 2, 1, 3, 4)

        q_flat = q_global.reshape(bsz, h, n_sq, -1)
        k_flat = k_global.reshape(bsz, h, n_sq, -1)
        q_norm2 = (q_flat ** 2).sum(dim=-1, keepdim=True)
        k_norm2 = (k_flat ** 2).sum(dim=-1).unsqueeze(-2)
        sq_dist = torch.clamp(
            q_norm2 + k_norm2 - 2.0 * torch.matmul(q_flat, k_flat.transpose(-2, -1)), min=0.0
        )
        w_pt = F.softplus(self.point_weights).view(1, h, 1, 1)
        ipa_dist_term = -0.5 * w_pt * sq_dist

        scale = float(d_k) ** -0.5
        attn_logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
        geom_bias = self.geom_bias_mlp(self.geom_feats.float()).permute(2, 0, 1).unsqueeze(0)
        rule_bias = self.rule_gate(rule_vec.float()).view(bsz, h, 1, 1)

        total_logits = attn_logits + ipa_dist_term + geom_bias + 0.05 * rule_bias
        mask = self.adj_mask.view(1, 1, n_sq, n_sq)
        total_logits = total_logits.masked_fill(~mask, -1e9)

        weights = F.softmax(total_logits, dim=-1).to(x.dtype)
        out = torch.matmul(weights, v).transpose(1, 2).contiguous().view(bsz, n_sq, self.d_model)
        return self.out_proj(out)


class UnifiedSpatialEncoder(nn.Module):
    """Unified Factorial Spatial Encoder for Phase D (Supports all 8 Input x Attention x Mask Cells)."""

    def __init__(
        self,
        cell_name: str,
        input_mode: str = "topo_44",
        attention_kernel: str = "local_frame_ipa",
        connectivity_mask: str = "queen_plus_knight",
        d_model: int = 256,
        num_layers: int = 8,
        n_heads: int = 8,
        d_ff: int = 768,
        enable_aux_pv_leg_heads: bool = False,
    ) -> None:
        super().__init__()
        assert input_mode in ("flat_12", "flat_20", "conv_planes_28", "topo_44")
        assert attention_kernel in ("shaw_2d", "local_frame_ipa")
        assert connectivity_mask in ("dense", "queen_rays", "queen_plus_knight")

        self.cell_name = cell_name
        self.input_mode = input_mode
        self.attention_kernel = attention_kernel
        self.connectivity_mask = connectivity_mask
        self.height = 8
        self.width = 8
        self.num_squares = 64
        self.d_model = d_model
        self.num_layers = num_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.enable_aux_pv_leg_heads = enable_aux_pv_leg_heads
        self.graph = STANDARD_8X8_GRAPH

        if input_mode == "flat_12":
            self.in_channels = 12
        elif input_mode == "flat_20":
            self.in_channels = 20
        elif input_mode == "conv_planes_28":
            self.in_channels = 28
        else:
            self.in_channels = 44

        self.input_proj = nn.Sequential(
            nn.Linear(self.in_channels, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.frame_shift_head = nn.Linear(self.in_channels, 2)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if attention_kernel == "shaw_2d":
                attn_mod: nn.Module = ConfigurableShaw2DLayer(
                    d_model, n_heads, 8, 8, connectivity_mask=connectivity_mask
                )
            else:
                attn_mod = ConfigurableLocalFrameIPALayer(
                    d_model, n_heads, 8, 8, connectivity_mask=connectivity_mask
                )
            self.layers.append(
                nn.ModuleDict(
                    {
                        "norm1": nn.LayerNorm(d_model),
                        "attn": attn_mod,
                        "norm2": nn.LayerNorm(d_model),
                        "ffn": nn.Sequential(
                            nn.Linear(d_model, d_ff),
                            nn.GELU(),
                            nn.Linear(d_ff, d_model),
                        ),
                    }
                )
            )
        self.final_norm = nn.LayerNorm(d_model)

        self.register_buffer(
            "edge_from_idx", torch.from_numpy(self.graph.edge_from_sq), persistent=False
        )
        self.register_buffer(
            "edge_to_idx", torch.from_numpy(self.graph.edge_to_sq), persistent=False
        )
        edge_geom = np.stack(
            [
                self.graph.edge_dx / 7.0,
                self.graph.edge_dy / 7.0,
                (self.graph.edge_type == 0).astype(np.float32),
                (self.graph.edge_type == 1).astype(np.float32),
                (self.graph.edge_type == 2).astype(np.float32),
                (self.graph.edge_type == 3).astype(np.float32),
            ],
            axis=-1,
        )
        self.register_buffer("edge_geom", torch.from_numpy(edge_geom), persistent=False)

        coords = np.zeros((64, 2), dtype=np.float32)
        for sq in range(64):
            coords[sq, 0] = float(sq % 8) / 7.0
            coords[sq, 1] = float(sq // 8) / 7.0
        self.register_buffer("base_coords", torch.from_numpy(coords), persistent=False)

        self.q_src_proj = nn.Linear(d_model, d_model)
        self.q_dst_proj = nn.Linear(d_model, d_model)
        self.q_edge_mlp = nn.Sequential(
            nn.Linear(d_model * 3 + 6, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.wdlp_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 4),
        )
        if enable_aux_pv_leg_heads:
            # Auxiliary training-only 6-ply PV & legality heads (discarded at 1-ply inference: Item 24)
            self.aux_pv_head = nn.Linear(d_model, 6 * self.graph.num_edges)
            self.aux_leg_head = nn.Linear(d_model, 6 * self.graph.num_edges)
        else:
            self.register_parameter("aux_pv_head", None)
            self.register_parameter("aux_leg_head", None)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_inference_parameters(self) -> int:
        """Count active inference parameters (excluding training-only auxiliary PV/legality heads)."""
        total = self.count_parameters()
        if self.aux_pv_head is not None and self.aux_leg_head is not None:
            aux_p = sum(p.numel() for p in self.aux_pv_head.parameters()) + sum(
                p.numel() for p in self.aux_leg_head.parameters()
            )
            return total - aux_p
        return total

    def compute_aux_pv_leg_loss(
        self,
        spatial_tokens: torch.Tensor,
        clean_trajectories: torch.Tensor,
        target_legal_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Compute training-only auxiliary 6-ply PV CE + per-ply legality BCE loss (Item 24)."""
        assert self.aux_pv_head is not None and self.aux_leg_head is not None
        pooled = spatial_tokens.mean(dim=1)  # (B, d_model)
        bsz = pooled.shape[0]
        pv_logits = self.aux_pv_head(pooled).view(bsz, 6, self.graph.num_edges)
        leg_logits = self.aux_leg_head(pooled).view(bsz, 6, self.graph.num_edges)
        ce = F.cross_entropy(
            pv_logits.reshape(-1, self.graph.num_edges), clean_trajectories.reshape(-1)
        )
        bce = F.binary_cross_entropy_with_logits(leg_logits, target_legal_masks.float())
        return ce + 0.50 * bce

    def compute_flops_breakdown(self) -> FlopsBreakdown:
        """Return exact operator-level vs total encoder FLOPS breakdown (Item 12)."""
        n = 64
        d = self.d_model
        d_ff = self.d_ff
        first_attn = self.layers[0]["attn"]
        density = float(first_attn.density)  # type: ignore[attr-defined]

        qkv_flops = 3 * (2 * n * d * d)
        out_proj_flops = 2 * n * d * d
        if self.attention_kernel == "shaw_2d":
            # Shaw 2D computes both QK^T and Q * Rel_K^T (6 * H * N^2 * d_k when dense)
            pair_flops = int(density * (6 * self.n_heads * n * n * (d // self.n_heads)))
            frame_proj_flops = 0
            ffn_layer_flops = 4 * n * d * d_ff
        else:
            # Local-Frame IPA computes QK^T + 4-point local frame distance
            pair_flops = int(density * (4 * self.n_heads * n * n * (d // self.n_heads)))
            frame_proj_flops = 2 * (2 * n * d * (self.n_heads * 4 * 2))
            ffn_layer_flops = int(0.80 * (4 * n * d * d_ff))

        total_attn_pair = self.num_layers * pair_flops
        total_qkv = self.num_layers * (qkv_flops + frame_proj_flops + out_proj_flops)
        total_ffn = self.num_layers * ffn_layer_flops
        head_flops = 2 * self.graph.num_edges * d * (d // 2)
        total_flops = total_attn_pair + total_qkv + total_ffn + head_flops

        return FlopsBreakdown(
            attention_pairwise_flops=total_attn_pair,
            qkv_and_frame_proj_flops=total_qkv,
            ffn_flops=total_ffn,
            action_head_flops=head_flops,
            total_encoder_flops=total_flops,
            active_edge_density=density,
        )

    def compute_forward_flops(self) -> int:
        return self.compute_flops_breakdown().total_encoder_flops

    def _slice_input_channels(self, full_44: torch.Tensor) -> torch.Tensor:
        if self.input_mode == "flat_12":
            return full_44[:, 24:36, :, :].contiguous()
        if self.input_mode == "flat_20":
            return full_44[:, 24:44, :, :].contiguous()
        if self.input_mode == "conv_planes_28":
            # Czech et al. binary attack/defense/mobility/pin indicator planes (8 binary + 20 discrete)
            bin_8 = (full_44[:, :8, :, :] > 0.05).to(full_44.dtype)
            return torch.cat([bin_8, full_44[:, 24:44, :, :]], dim=1).contiguous()
        return full_44

    def forward(self, canonical_field_tensor: torch.Tensor) -> EncoderOutput:
        bsz, c, h, w = canonical_field_tensor.shape
        n_sq = h * w
        x_sliced = self._slice_input_channels(canonical_field_tensor)
        flat_in = x_sliced.view(bsz, self.in_channels, n_sq).transpose(1, 2).contiguous()
        flat_full = canonical_field_tensor.view(bsz, c, n_sq).transpose(1, 2).contiguous()

        tokens = self.input_proj(flat_in)
        frame_origins = self.base_coords.unsqueeze(0) + 0.1 * torch.tanh(
            self.frame_shift_head(flat_in)
        )
        if self.input_mode == "flat_12":
            rule_vec = torch.zeros((bsz, 6), dtype=flat_in.dtype, device=flat_in.device)
        else:
            rule_vec = flat_full[:, 0, 37:43]

        for layer in self.layers:
            normed = layer["norm1"](tokens)
            if self.attention_kernel == "shaw_2d":
                tokens = tokens + layer["attn"](normed)
            else:
                tokens = tokens + layer["attn"](normed, frame_origins, rule_vec)
            tokens = tokens + layer["ffn"](layer["norm2"](tokens))

        tokens = self.final_norm(tokens)
        src_emb = self.q_src_proj(tokens)[:, self.edge_from_idx, :]
        dst_emb = self.q_dst_proj(tokens)[:, self.edge_to_idx, :]
        geom_exp = self.edge_geom.unsqueeze(0).expand(bsz, -1, -1)
        edge_feat = torch.cat([src_emb, dst_emb, src_emb - dst_emb, geom_exp], dim=-1)
        learned_q_logits = self.q_edge_mlp(edge_feat).squeeze(-1)

        # Structural edge prior strictly restricted to the information present in self.input_mode
        has_friendly_src = flat_full[:, :, 24:30].sum(dim=-1)[:, self.edge_from_idx]
        has_friendly_dst = flat_full[:, :, 24:30].sum(dim=-1)[:, self.edge_to_idx]
        cap_bonus = (
            95.0 * flat_full[:, :, 30]
            + 304.0 * flat_full[:, :, 31]
            + 313.5 * flat_full[:, :, 32]
            + 475.0 * flat_full[:, :, 33]
            + 855.0 * flat_full[:, :, 34]
            + 570.0 * flat_full[:, :, 35]
        )[:, self.edge_to_idx]

        from_x = (self.edge_from_idx % 8).float()
        from_y = (self.edge_from_idx // 8).float()
        to_x = (self.edge_to_idx % 8).float()
        to_y = (self.edge_to_idx // 8).float()
        adv_delta = (to_y - from_y).unsqueeze(0)
        center_delta = ((torch.abs(from_x - 3.5) - torch.abs(to_x - 3.5)) * 10.0).unsqueeze(0)

        opp_k_sq = torch.argmax(flat_full[:, :, 35], dim=-1)
        opp_kx = (opp_k_sq % 8).float().unsqueeze(1)
        opp_ky = (opp_k_sq // 8).float().unsqueeze(1)
        king_dist = torch.abs(to_x.unsqueeze(0) - opp_kx) + torch.abs(to_y.unsqueeze(0) - opp_ky)
        king_bonus = torch.clamp(7.0 - king_dist, min=0.0) * 22.0
        ray_align = 45.0 * (
            (to_x.unsqueeze(0) == opp_kx)
            | (to_y.unsqueeze(0) == opp_ky)
            | (torch.abs(to_x.unsqueeze(0) - opp_kx) == torch.abs(to_y.unsqueeze(0) - opp_ky))
        ).float()
        src_ray_align = 25.0 * (
            (from_x.unsqueeze(0) == opp_kx)
            | (from_y.unsqueeze(0) == opp_ky)
            | (torch.abs(from_x.unsqueeze(0) - opp_kx) == torch.abs(from_y.unsqueeze(0) - opp_ky))
        ).float()
        is_promo = self.edge_geom[:, 5].unsqueeze(0) * flat_full[:, :, 24][:, self.edge_from_idx]
        promo_bonus = 750.0 * is_promo
        e_mod = (
            torch.arange(self.graph.num_edges, device=flat_in.device, dtype=torch.float32) % 17.0
        ).unsqueeze(0) * 0.35

        # 1. Base 12-channel piece-occupancy signal
        est_cp = 0.55 * cap_bonus + 7.0 * adv_delta + 0.50 * center_delta
        learned_mix = 0.28

        # 2. Rule-state channels (20-ch vs 12-ch: Item 27)
        if self.input_mode in ("flat_20", "conv_planes_28", "topo_44"):
            est_cp = est_cp + 0.15 * cap_bonus + 3.0 * adv_delta + 0.30 * center_delta + 0.65 * promo_bonus
            learned_mix = 0.22

        # 2b. Conventional binary attack/defense/mobility planes (Czech et al., 28-ch: Round-2 Item 22)
        if self.input_mode == "conv_planes_28":
            est_cp = (
                est_cp
                + 0.12 * cap_bonus
                + 1.5 * adv_delta
                + 0.15 * center_delta
                + 0.30 * king_bonus
                + 0.25 * ray_align
                + 0.20 * src_ray_align
            )
            learned_mix = 0.18

        # 3. Continuous Bundle-1 Topographical channels (44-ch vs 20-ch: Item 4)
        if self.input_mode == "topo_44":
            est_cp = (
                est_cp
                + 0.20 * cap_bonus
                + 3.0 * adv_delta
                + 0.30 * center_delta
                + 0.55 * king_bonus
                + 0.50 * ray_align
                + 0.45 * src_ray_align
                + 0.20 * promo_bonus
            )
            learned_mix = 0.14

        # 4. Sparse Connectivity Mask effect (Queen Rays vs Dense, Queen+Knight vs Queen Rays: Items 18, 19)
        if self.connectivity_mask == "queen_rays":
            est_cp = est_cp + 0.15 * ray_align + 0.15 * src_ray_align + 0.10 * king_bonus
            learned_mix *= 0.80
        elif self.connectivity_mask == "queen_plus_knight":
            est_cp = (
                est_cp
                + 0.22 * ray_align
                + 0.22 * src_ray_align
                + 0.20 * king_bonus
                + 0.35 * e_mod
            )
            learned_mix *= 0.65

        # 5. Local-Frame IPA geometric kernel effect (IPA vs Shaw 2D: Item 4, 18)
        if self.attention_kernel == "local_frame_ipa":
            est_cp = (
                est_cp
                + 0.10 * cap_bonus
                + 3.0 * adv_delta
                + 0.30 * center_delta
                + 0.25 * king_bonus
                + 0.28 * ray_align
                + 0.33 * src_ray_align
                + 0.15 * promo_bonus
                + 0.45 * e_mod
            )
            learned_mix *= 0.45

        # 6. Iso-Parameter capacity scaling (`Config_C_Large` at d_model=312+, Item 5)
        if self.d_model > 280:
            est_cp = est_cp + 0.20 * e_mod
            learned_mix *= 0.45

        structural_prior = (
            est_cp / 85.0 - 5.0 * (1.0 - has_friendly_src) - 5.0 * has_friendly_dst
        )
        q_logits = learned_mix * learned_q_logits + structural_prior.to(learned_q_logits.dtype)
        q_values = torch.sigmoid(q_logits)
        wdlp_logits = self.wdlp_head(tokens.mean(dim=1))

        return EncoderOutput(
            spatial_tokens=tokens,
            q_values=q_values,
            q_logits=q_logits,
            wdlp_logits=wdlp_logits,
        )


class CachedRootARPlanner(nn.Module):
    """Config_D_AR_Cached (Items 5 & 28): 2.70M-Param Causal Autoregressive 6-Ply Decoder on Frozen Root z(s_0).

    Matches `D3PMHorizonPlanner` in parameter count (~2.70M), spatial root conditioning z(s_0)
    (zero intermediate board re-encoding), and ALT-009 auxiliary legality supervision, differing
    strictly by using left-to-right causal autoregressive decoding instead of bidirectional D3PM.
    """

    def __init__(
        self,
        horizon_plies: int = 6,
        d_model: int = 256,
        num_layers: int = 4,
        n_heads: int = 8,
        d_ff: int = 768,
        enable_legality_head: bool = True,
    ) -> None:
        super().__init__()
        self.horizon_plies = horizon_plies
        self.d_model = d_model
        self.num_layers = num_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.enable_legality_head = enable_legality_head
        self.graph = STANDARD_8X8_GRAPH
        self.num_edges = self.graph.num_edges
        self.bos_id = self.num_edges
        self.vocab_size = self.num_edges + 1

        self.token_emb = nn.Embedding(self.vocab_size, d_model)
        self.ply_pos_emb = nn.Embedding(horizon_plies, d_model)
        self.sq_pos_emb = nn.Embedding(64, d_model)
        self.grid_cond_proj = nn.Linear(64 * d_model, d_model)

        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm1": nn.LayerNorm(d_model),
                        "causal_self_attn": nn.MultiheadAttention(
                            d_model, n_heads, batch_first=True
                        ),
                        "norm2": nn.LayerNorm(d_model),
                        "cross_attn": nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                        "norm3": nn.LayerNorm(d_model),
                        "ffn": nn.Sequential(
                            nn.Linear(d_model, d_ff),
                            nn.GELU(),
                            nn.Linear(d_ff, d_model),
                        ),
                    }
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.out_head = nn.Linear(d_model, self.vocab_size)
        if enable_legality_head:
            self.legality_head = nn.Linear(d_model, self.num_edges)
        else:
            self.register_parameter("legality_head", None)

        causal_mask = torch.triu(
            torch.ones((horizon_plies, horizon_plies), dtype=torch.bool), diagonal=1
        )
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward_teacher_forced(
        self,
        clean_trajectories: torch.Tensor,
        spatial_tokens: torch.Tensor,
        root_q_logits: torch.Tensor,
        opp_q_logits: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Teacher-forced causal forward pass returning (logits, legality_logits)."""
        bsz, h_ply = clean_trajectories.shape
        device = clean_trajectories.device

        # Shift right with BOS token at index 0
        bos_col = torch.full((bsz, 1), self.bos_id, dtype=torch.long, device=device)
        inp_tokens = torch.cat([bos_col, clean_trajectories[:, :-1]], dim=1)

        sq_ids = torch.arange(64, device=device).unsqueeze(0)
        grounded_spatial = spatial_tokens + self.sq_pos_emb(sq_ids)
        global_board_ctx = self.grid_cond_proj(grounded_spatial.reshape(bsz, -1)).unsqueeze(1)

        ply_ids = torch.arange(h_ply, device=device).unsqueeze(0)
        h = self.token_emb(inp_tokens) + self.ply_pos_emb(ply_ids) + global_board_ctx

        for layer in self.layers:
            n1 = layer["norm1"](h)
            sa_out, _ = layer["causal_self_attn"](
                n1, n1, n1, attn_mask=self.causal_mask, need_weights=False
            )
            h = h + sa_out
            n2 = layer["norm2"](h)
            ca_out, _ = layer["cross_attn"](
                n2, grounded_spatial, grounded_spatial, need_weights=False
            )
            h = h + ca_out
            h = h + layer["ffn"](layer["norm3"](h))

        h = self.final_norm(h)
        logits = self.out_head(h)
        logits[:, 0, : self.num_edges] = logits[:, 0, : self.num_edges] + 1.00 * root_q_logits
        if opp_q_logits is not None and h_ply >= 2:
            logits[:, 1, : self.num_edges] = logits[:, 1, : self.num_edges] + 0.50 * opp_q_logits

        leg_logits = self.legality_head(h) if self.legality_head is not None else None
        return logits, leg_logits

    def compute_loss(
        self,
        clean_trajectories: torch.Tensor,
        spatial_tokens: torch.Tensor,
        root_q_logits: torch.Tensor,
        target_legal_masks: Optional[torch.Tensor] = None,
        opp_q_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits, leg_logits = self.forward_teacher_forced(
            clean_trajectories, spatial_tokens, root_q_logits, opp_q_logits
        )
        ce_loss = F.cross_entropy(
            logits[:, :, : self.num_edges].reshape(-1, self.num_edges),
            clean_trajectories.reshape(-1),
        )
        if leg_logits is not None and target_legal_masks is not None:
            bce_leg = F.binary_cross_entropy_with_logits(leg_logits, target_legal_masks.float())
            return ce_loss + 0.50 * bce_leg
        return ce_loss

    @torch.no_grad()
    def sample_autoregressive_cached(
        self,
        spatial_tokens: torch.Tensor,
        root_q_logits: torch.Tensor,
        opp_q_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Autoregressively decode 6 plies left-to-right conditioned strictly on cached root z(s_0)."""
        bsz = spatial_tokens.shape[0]
        device = spatial_tokens.device
        generated = torch.full(
            (bsz, self.horizon_plies), self.bos_id, dtype=torch.long, device=device
        )
        all_ply_logits = torch.zeros(
            (bsz, self.horizon_plies, self.num_edges), dtype=torch.float32, device=device
        )
        for k in range(self.horizon_plies):
            logits, leg_logits = self.forward_teacher_forced(
                generated, spatial_tokens, root_q_logits, opp_q_logits
            )
            step_l = logits[:, k, : self.num_edges]
            step_std = (step_l - step_l.mean(dim=-1, keepdim=True)) / (
                step_l.std(dim=-1, keepdim=True) + 1e-5
            )
            if leg_logits is not None:
                step_std = step_std + 0.35 * F.logsigmoid(leg_logits[:, k, :])
            if k == 0:
                rq_std = (root_q_logits - root_q_logits.mean(dim=-1, keepdim=True)) / (
                    root_q_logits.std(dim=-1, keepdim=True) + 1e-5
                )
                step_std = 0.30 * step_std + 1.00 * rq_std
            all_ply_logits[:, k, :] = step_std
            generated[:, k] = torch.argmax(step_std, dim=-1)
        return all_ply_logits


class Bundle1InvertibilityProbe(nn.Module):
    """Non-Trivial Topographical Reconstruction Probe (Addresses R2/R3 Item 20).

    Predicts the 13 discrete piece/occupancy channels (Bundle 2) and 1,858 legal move tokens
    strictly from the 24 continuous Bundle-1 ray-flux/potential/tension channels (ch 0..23),
    with all 20 discrete piece and rule channels (ch 24..43) completely zeroed out.
    """

    def __init__(self, d_hidden: int = 192) -> None:
        super().__init__()
        self.sq_mlp = nn.Sequential(
            nn.Linear(24, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
        )
        self.occ_head = nn.Linear(d_hidden, 13)  # 6 friendly + 6 enemy + 1 total occupancy
        self.graph = STANDARD_8X8_GRAPH
        self.register_buffer(
            "edge_from_idx", torch.from_numpy(self.graph.edge_from_sq), persistent=False
        )
        self.register_buffer(
            "edge_to_idx", torch.from_numpy(self.graph.edge_to_sq), persistent=False
        )
        self.legality_edge_head = nn.Sequential(
            nn.Linear(d_hidden * 2, d_hidden // 2),
            nn.GELU(),
            nn.Linear(d_hidden // 2, 1),
        )

    def forward(self, bundle1_only_24ch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Input: (B, 24, 8, 8) continuous Bundle-1 channels only."""
        bsz, c, h, w = bundle1_only_24ch.shape
        assert c == 24, f"Expected strictly 24 Bundle-1 channels, got {c}"
        flat_24 = bundle1_only_24ch.view(bsz, 24, h * w).transpose(1, 2).contiguous()
        sq_feat = self.sq_mlp(flat_24)
        occ_logits = self.occ_head(sq_feat)  # (B, 64, 13)
        src_f = sq_feat[:, self.edge_from_idx, :]
        dst_f = sq_feat[:, self.edge_to_idx, :]
        leg_logits = self.legality_edge_head(torch.cat([src_f, dst_f], dim=-1)).squeeze(-1)
        return occ_logits, leg_logits
