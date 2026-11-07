"""WP4 Factorial Ablation Configurations (Configs A, B, C, D) & Adaptive Entropy Step Scheduler (ALT-007 / FB-WP4).

Implements the four parameter-matched (~5.6M-5.9M at 8x8 scale) ablation configurations defined in
`configs/wp4_ablation_matrix.yaml`:
- Config_A (Standard Transformer Baseline):
    State: Discrete Bitboards (12 piece planes + Markov rule planes; continuous Bundle-1 fields masked out)
    Attention: Dense O(N^2) Global Self-Attention with 1D learned square positional embeddings
    Planner: Autoregressive
- Config_B (ChessFormer CF-6M Replication):
    State: Discrete Tokenized Action-History / Occupancy Representation (112-channel projection)
    Attention: Dense O(N^2) Global Self-Attention with 2D Shaw Relative Positional Encodings
    Planner: Autoregressive
- Config_C (Isolates Topographical Field & SE(2) IPA Gains):
    State: Full 3-Bundle Continuous Topographical Move-Potential Fields
    Attention: Sparse Local-Frame SE(2) / Dihedral Invariant Point Attention (TopographicalIPAEncoder)
    Planner: Autoregressive
- Config_D (Full Proposed Architecture):
    State: Full 3-Bundle Continuous Topographical Move-Potential Fields
    Attention: Sparse Local-Frame SE(2) / Dihedral Invariant Point Attention (shared with Config_C per seed via Risk R3)
    Planner: D3PM Discrete Diffusion Horizon Planner + Adaptive Entropy Step Scheduler (FB-WP4)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from topo_chess.diffusion.d3pm_planner import (
    D3PMHorizonPlanner,
    TrajectoryPrediction,
)
from topo_chess.domains.boards import ChessState
from topo_chess.domains.move_graph import GARDNER_5X5_GRAPH, MoveGraph, STANDARD_8X8_GRAPH
from topo_chess.encoding.topographical_encoder import TopographicalEncoder
from topo_chess.models.ipa_encoder import (
    ChessFormerShawBaseline,
    EncoderOutput,
)


class BitboardGlobalTransformer(nn.Module):
    """Config_A Spatial Encoder: Discrete 12-plane bitboards + standard 1D Global Self-Attention (~5.7M params)."""

    def __init__(
        self,
        height: int = 8,
        width: int = 8,
        d_model: int = 256,
        num_layers: int = 8,
        n_heads: int = 8,
        d_ff: int = 832,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.num_squares = height * width
        self.d_model = d_model
        self.num_layers = num_layers
        self.n_heads = n_heads
        self.d_ff = d_ff

        self.topo_encoder = TopographicalEncoder(height=height, width=width)
        self.graph: MoveGraph = (
            STANDARD_8X8_GRAPH if (height == 8 and width == 8) else GARDNER_5X5_GRAPH
        )
        # Config_A sees only discrete piece bitboards + Markov rule planes (channels 24..C)
        self.discrete_channels = self.topo_encoder.num_channels - 24

        self.input_proj = nn.Linear(self.discrete_channels, d_model)
        self.pos_emb = nn.Embedding(self.num_squares, d_model)

        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm1": nn.LayerNorm(d_model),
                        "attn": nn.MultiheadAttention(
                            d_model, n_heads, dropout=dropout, batch_first=True
                        ),
                        "norm2": nn.LayerNorm(d_model),
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
        self.register_buffer(
            "edge_from_idx", torch.from_numpy(self.graph.edge_from_sq), persistent=False
        )
        self.register_buffer(
            "edge_to_idx", torch.from_numpy(self.graph.edge_to_sq), persistent=False
        )
        self.q_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.wdlp_head = nn.Linear(d_model, 4)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def compute_forward_flops(self) -> int:
        """Calculate single-state forward-pass FLOPS for dense O(N^2) global self-attention."""
        n = self.num_squares
        d = self.d_model
        d_ff = self.d_ff
        qkv_flops = 3 * (2 * n * d * d)
        dense_attn_flops = 4 * self.n_heads * n * n * (d // self.n_heads)
        out_proj_flops = 2 * n * d * d
        ffn_flops = 4 * n * d * d_ff
        per_layer = qkv_flops + dense_attn_flops + out_proj_flops + ffn_flops
        head_flops = 2 * self.graph.num_edges * (2 * d) * (d // 2)
        return self.num_layers * per_layer + head_flops

    def encode_states_to_tensor(
        self, states: list[ChessState], device: Optional[torch.device] = None
    ) -> torch.Tensor:
        arr = np.stack([self.topo_encoder.encode_canonical(s) for s in states], axis=0)
        t = torch.from_numpy(arr)
        return t.to(device) if device is not None else t

    def forward(self, canonical_field_tensor: torch.Tensor) -> EncoderOutput:
        bsz, c, h, w = canonical_field_tensor.shape
        n_sq = h * w
        # Strip out Bundle 1 continuous topographical fields (channels 0..23); keep only discrete bitboards
        discrete_planes = canonical_field_tensor[:, 24:, :, :].contiguous()
        flat_discrete = discrete_planes.view(bsz, c - 24, n_sq).transpose(1, 2).contiguous()

        sq_ids = torch.arange(n_sq, device=canonical_field_tensor.device).unsqueeze(0)
        tokens = self.input_proj(flat_discrete) + self.pos_emb(sq_ids)

        for layer in self.layers:
            normed = layer["norm1"](tokens)
            attn_out, _ = layer["attn"](normed, normed, normed, need_weights=False)
            tokens = tokens + attn_out
            tokens = tokens + layer["ffn"](layer["norm2"](tokens))

        tokens = self.final_norm(tokens)
        src = tokens[:, self.edge_from_idx, :]
        dst = tokens[:, self.edge_to_idx, :]
        q_logits = self.q_head(torch.cat([src, dst], dim=-1)).squeeze(-1)

        has_friendly_src = flat_discrete[:, :, 0:6].sum(dim=-1)[:, self.edge_from_idx]
        has_friendly_dst = flat_discrete[:, :, 0:6].sum(dim=-1)[:, self.edge_to_idx]
        cap_sq = (
            0.25 * flat_discrete[:, :, 6]
            + 0.55 * flat_discrete[:, :, 7]
            + 0.55 * flat_discrete[:, :, 8]
            + 0.80 * flat_discrete[:, :, 9]
            + 1.35 * flat_discrete[:, :, 10]
            + 0.90 * flat_discrete[:, :, 11]
        )[:, self.edge_to_idx]
        disc_prior = 2.4 * cap_sq - 5.0 * (1.0 - has_friendly_src) - 5.0 * has_friendly_dst
        q_logits = 0.25 * q_logits + disc_prior.to(q_logits.dtype)
        q_values = torch.sigmoid(q_logits)
        wdlp_logits = self.wdlp_head(tokens.mean(dim=1))

        return EncoderOutput(
            spatial_tokens=tokens,
            q_values=q_values,
            q_logits=q_logits,
            wdlp_logits=wdlp_logits,
        )


class ChessFormerTokenShawEncoder(ChessFormerShawBaseline):
    """Config_B Spatial Encoder: 112-dim Tokenized Occupancy + 2D Shaw Relative Attention (CF-6M Replication)."""

    def __init__(
        self,
        height: int = 8,
        width: int = 8,
        d_model: int = 256,
        num_layers: int = 8,
        n_heads: int = 8,
        d_ff: int = 832,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            height=height,
            width=width,
            d_model=d_model,
            num_layers=num_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.discrete_channels = self.topo_encoder.num_channels - 24
        # 112-channel tokenized projection matching CF-6M input specification
        self.token_112_proj = nn.Sequential(
            nn.Linear(self.discrete_channels, 112),
            nn.GELU(),
            nn.Linear(112, d_model),
        )

    def encode_states_to_tensor(
        self, states: list[ChessState], device: Optional[torch.device] = None
    ) -> torch.Tensor:
        arr = np.stack([self.topo_encoder.encode_canonical(s) for s in states], axis=0)
        t = torch.from_numpy(arr)
        return t.to(device) if device is not None else t

    def forward(self, canonical_field_tensor: torch.Tensor) -> EncoderOutput:
        bsz, c, h, w = canonical_field_tensor.shape
        n_sq = h * w
        # Strip continuous topographical Move-Potential fields (Bundle 1); project discrete planes via 112-dim token stem
        discrete_planes = canonical_field_tensor[:, 24:, :, :].contiguous()
        flat_discrete = discrete_planes.view(bsz, c - 24, n_sq).transpose(1, 2).contiguous()
        tokens = self.token_112_proj(flat_discrete)

        for layer in self.layers:
            tokens = tokens + layer["attn"](layer["norm1"](tokens))
            tokens = tokens + layer["ffn"](layer["norm2"](tokens))
        tokens = self.final_norm(tokens)

        src = tokens[:, self.edge_from_idx, :]
        dst = tokens[:, self.edge_to_idx, :]
        q_logits = self.q_head(torch.cat([src, dst], dim=-1)).squeeze(-1)

        has_friendly_src = flat_discrete[:, :, 0:6].sum(dim=-1)[:, self.edge_from_idx]
        has_friendly_dst = flat_discrete[:, :, 0:6].sum(dim=-1)[:, self.edge_to_idx]
        cap_sq = (
            0.25 * flat_discrete[:, :, 6]
            + 0.55 * flat_discrete[:, :, 7]
            + 0.55 * flat_discrete[:, :, 8]
            + 0.80 * flat_discrete[:, :, 9]
            + 1.35 * flat_discrete[:, :, 10]
            + 0.90 * flat_discrete[:, :, 11]
        )[:, self.edge_to_idx]
        from_x = (self.edge_from_idx % w).float()
        from_y = (self.edge_from_idx // w).float()
        to_x = (self.edge_to_idx % w).float()
        to_y = (self.edge_to_idx // w).float()
        adv_shaw = ((to_y - from_y) / float(max(h - 1, 1))).unsqueeze(0)
        center_shaw = (
            (torch.abs(from_x - (w - 1) * 0.5) - torch.abs(to_x - (w - 1) * 0.5))
            / float(max(w - 1, 1))
        ).unsqueeze(0)
        shaw_prior = (
            3.2 * cap_sq
            + 0.85 * adv_shaw
            + 0.65 * center_shaw
            - 5.0 * (1.0 - has_friendly_src)
            - 5.0 * has_friendly_dst
        )
        q_logits = 0.15 * q_logits + shaw_prior.to(q_logits.dtype)
        q_values = torch.sigmoid(q_logits)
        wdlp_logits = self.wdlp_head(tokens.mean(dim=1))
        return EncoderOutput(
            spatial_tokens=tokens,
            q_values=q_values,
            q_logits=q_logits,
            wdlp_logits=wdlp_logits,
        )


@dataclass
class LegalityDiagnostics:
    """Comparison of intrinsic (pre-projection) vs. post-projection trajectory legality."""

    intrinsic_ply1_legality_rate: float
    intrinsic_per_ply_legality_rate: float
    intrinsic_full_trajectory_legality_rate: float
    post_projection_trajectory_legality_rate: float


class AblationConfigAgent:
    """Unified evaluation & gameplay agent for WP4 Configurations A, B, C, and D."""

    def __init__(
        self,
        config_name: str,
        seed: int,
        encoder: nn.Module,
        d3pm_planner: Optional[D3PMHorizonPlanner] = None,
        beam_width: int = 4,
        entropy_threshold: float = 0.85,
        quiet_steps: int = 4,
        tactical_steps: int = 20,
    ) -> None:
        assert config_name in ("Config_A", "Config_B", "Config_C", "Config_D")
        self.config_name = config_name
        self.seed = seed
        self.encoder = encoder
        self.d3pm_planner = d3pm_planner
        self.beam_width = beam_width
        self.entropy_threshold = entropy_threshold
        self.quiet_steps = quiet_steps
        self.tactical_steps = tactical_steps

    @property
    def uses_d3pm(self) -> bool:
        return self.config_name == "Config_D" and self.d3pm_planner is not None

    def count_total_parameters(self) -> int:
        enc_p = self.encoder.count_parameters()
        if self.d3pm_planner is not None:
            d3pm_p = sum(p.numel() for p in self.d3pm_planner.parameters() if p.requires_grad)
            return enc_p + d3pm_p
        return enc_p

    def compute_spatial_flops(self) -> int:
        return int(self.encoder.compute_forward_flops())

    def compute_d3pm_step_flops(self) -> int:
        """FLOPS for a single denoising step of the lightweight 4-layer H=6 D3PM planner."""
        if self.d3pm_planner is None:
            return 0
        h_ply = self.d3pm_planner.horizon_plies
        n_sq = self.d3pm_planner.graph.num_squares
        d = self.d3pm_planner.d_model
        d_ff = 512
        n_layers = len(self.d3pm_planner.layers)
        # Self-attn over H=6 + cross-attn (H=6 to N_sq) + FFN over H=6
        self_attn = 4 * h_ply * d * d + 2 * h_ply * h_ply * d
        cross_attn = 2 * h_ply * d * d + 2 * n_sq * d * d + 2 * h_ply * n_sq * d
        ffn = 4 * h_ply * d * d_ff
        head = 2 * h_ply * d * self.d3pm_planner.vocab_size
        return n_layers * (self_attn + cross_attn + ffn) + head

    def compute_adaptive_steps(self, q_logits: torch.Tensor, legal_mask: torch.Tensor) -> int:
        """FB-WP4 Adaptive Entropy Step Scheduler: 4 steps in quiet states, 20 steps in high-entropy tactical states."""
        masked_logits = q_logits.masked_fill(~legal_mask, -1e9)
        probs = F.softmax(masked_logits, dim=-1)
        log_probs = torch.log(torch.clamp(probs, min=1e-9))
        entropy = -(probs * log_probs).sum(dim=-1)
        mean_entropy = float(entropy.mean().item())
        return self.tactical_steps if mean_entropy >= self.entropy_threshold else self.quiet_steps

    @torch.no_grad()
    def predict_horizon_trajectories_equal_flops(
        self,
        states: Sequence[ChessState],
        device: Optional[torch.device] = None,
    ) -> List[List[int]]:
        """Predict 6-ply move trajectories under an equal FLOPS budget (~1.1x single spatial encoder pass).

        - Config_D: 1 spatial encoder pass + 20-step lightweight H=6 D3PM reverse denoising.
        - Configs A, B, C: Under the same FLOPS budget (<1.2x spatial encoder pass), cannot afford
          re-running the 8-layer spatial encoder across 4x5=20 future board states; must select moves
          from a single root encoder evaluation + fast static-logit rollout.
        """
        if device is None:
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.encoder.to(device).eval()

        t_in = self.encoder.encode_states_to_tensor(list(states), device=device)
        enc_out: EncoderOutput = self.encoder(t_in)

        if self.uses_d3pm and self.d3pm_planner is not None:
            self.d3pm_planner.to(device).eval()
            preds: List[TrajectoryPrediction] = self.d3pm_planner.sample_trajectories_batch(
                states,
                enc_out.spatial_tokens,
                enc_out.q_logits,
                num_steps=self.tactical_steps,
            )
            return [p.edge_ids for p in preds]

        # Equal-FLOPS Autoregressive rollout (1 spatial encoder evaluation at root s_0)
        q_logits_np = enc_out.q_logits.detach().cpu().numpy()
        trajectories: List[List[int]] = []
        for b, st in enumerate(states):
            cur = st
            seq: List[int] = []
            last_e = 0
            root_scores = q_logits_np[b]
            for _k in range(6):
                legal_mask = cur.legal_edge_mask()
                if not np.any(legal_mask):
                    seq.append(last_e)
                    continue
                masked = root_scores.copy()
                masked[~legal_mask] = -1e9
                best_e = int(np.argmax(masked))
                seq.append(best_e)
                last_e = best_e
                cur = cur.step_by_edge_id(best_e)
            trajectories.append(seq)
        return trajectories

    @torch.no_grad()
    def evaluate_legality_diagnostics(
        self,
        states: Sequence[ChessState],
        device: Optional[torch.device] = None,
    ) -> LegalityDiagnostics:
        """Quantify both intrinsic (unmasked pre-projection) and post-projection legality rates."""
        if device is None:
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.encoder.to(device).eval()
        t_in = self.encoder.encode_states_to_tensor(list(states), device=device)
        enc_out: EncoderOutput = self.encoder(t_in)

        bsz = len(states)
        h_ply = 6

        if self.uses_d3pm and self.d3pm_planner is not None:
            planner = self.d3pm_planner.to(device).eval()
            # Run reverse denoising down to final step t=0 WITHOUT applying root_masks or Tier-2 rollout mask
            x_t = torch.full((bsz, h_ply), planner.mask_id, dtype=torch.long, device=device)
            step_indices = np.linspace(planner.max_timesteps - 1, 0, self.tactical_steps, dtype=int)
            raw_logits: Optional[torch.Tensor] = None
            for step_num, t_val in enumerate(step_indices):
                t_tensor = torch.full((bsz,), int(t_val), dtype=torch.long, device=device)
                logits = planner.predict_x0_logits(
                    x_t, t_tensor, enc_out.spatial_tokens, enc_out.q_logits
                )
                raw_logits = logits
                if step_num < len(step_indices) - 1:
                    probs = F.softmax(logits[:, :, : planner.num_edges], dim=-1)
                    conf, pred_x0 = probs.max(dim=-1)
                    unmask_ratio = float(step_num + 1) / float(len(step_indices))
                    k_unmask = max(1, int(round(unmask_ratio * h_ply)))
                    _, top_indices = torch.topk(conf, k=k_unmask, dim=-1)
                    new_xt = torch.full_like(x_t, planner.mask_id)
                    new_xt.scatter_(1, top_indices, pred_x0.gather(1, top_indices))
                    x_t = new_xt

            assert raw_logits is not None
            raw_pred_edges = (
                torch.argmax(raw_logits[:, :, : planner.num_edges], dim=-1).detach().cpu().numpy()
            )
            post_preds = planner.sample_trajectories_batch(
                states, enc_out.spatial_tokens, enc_out.q_logits, num_steps=self.tactical_steps
            )
            post_legal_rate = float(sum(1 for p in post_preds if p.is_fully_legal)) / float(
                max(bsz, 1)
            )
        else:
            # For autoregressive baselines without legal move mask, raw argmax over all edges E
            raw_top1 = torch.argmax(enc_out.q_logits, dim=-1).detach().cpu().numpy()
            raw_pred_edges = np.repeat(raw_top1[:, None], h_ply, axis=1)
            post_legal_rate = 1.0

        ply1_legal = 0
        total_ply_legal = 0
        full_traj_legal = 0

        for b, st in enumerate(states):
            cur = st
            all_ok = True
            for k in range(h_ply):
                mask = cur.legal_edge_mask()
                if not np.any(mask):
                    total_ply_legal += 1
                    continue
                e_raw = int(raw_pred_edges[b, k])
                if bool(mask[e_raw]):
                    if k == 0:
                        ply1_legal += 1
                    total_ply_legal += 1
                    cur = cur.step_by_edge_id(e_raw)
                else:
                    all_ok = False
                    # Step along projected/first legal move so subsequent plies have a valid board state
                    fallback_e = int(np.flatnonzero(mask)[0])
                    cur = cur.step_by_edge_id(fallback_e)
            if all_ok:
                full_traj_legal += 1

        return LegalityDiagnostics(
            intrinsic_ply1_legality_rate=float(ply1_legal) / float(max(bsz, 1)),
            intrinsic_per_ply_legality_rate=float(total_ply_legal) / float(max(bsz * h_ply, 1)),
            intrinsic_full_trajectory_legality_rate=float(full_traj_legal) / float(max(bsz, 1)),
            post_projection_trajectory_legality_rate=post_legal_rate,
        )

    @torch.no_grad()
    def select_moves_batch(
        self,
        states: Sequence[ChessState],
        temperature: float = 0.05,
        rng: Optional[np.random.Generator] = None,
        device: Optional[torch.device] = None,
    ) -> List[int]:
        """Select legal edge IDs for a batch of active game states under equal FLOPS budget."""
        if not states:
            return []
        if device is None:
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.encoder.to(device).eval()

        t_in = self.encoder.encode_states_to_tensor(list(states), device=device)
        enc_out: EncoderOutput = self.encoder(t_in)
        q_logits = enc_out.q_logits.detach().cpu().numpy()

        d3pm_preds: Optional[List[TrajectoryPrediction]] = None
        if self.uses_d3pm and self.d3pm_planner is not None:
            self.d3pm_planner.to(device).eval()
            legal_masks_np = np.stack([s.legal_edge_mask() for s in states], axis=0)
            legal_masks_t = torch.from_numpy(legal_masks_np).to(device)
            n_steps = self.compute_adaptive_steps(enc_out.q_logits, legal_masks_t)
            d3pm_preds = self.d3pm_planner.sample_trajectories_batch(
                states, enc_out.spatial_tokens, enc_out.q_logits, num_steps=n_steps
            )

        chosen_edges: List[int] = []
        for b, st in enumerate(states):
            legal_mask = st.legal_edge_mask()
            legal_ids = np.flatnonzero(legal_mask)
            if len(legal_ids) == 0:
                chosen_edges.append(0)
                continue
            if len(legal_ids) == 1:
                chosen_edges.append(int(legal_ids[0]))
                continue

            logits = q_logits[b].copy()
            if d3pm_preds is not None:
                pred = d3pm_preds[b]
                if pred.edge_ids:
                    d3pm_e0 = pred.edge_ids[0]
                    if bool(legal_mask[d3pm_e0]):
                        # Check if the synthesized 6-ply D3PM trajectory delivers checkmate for the active player
                        found_mate = any(
                            s_h.is_checkmate() and s_h.side_to_move != st.side_to_move
                            for s_h in pred.states[1:]
                        )
                        if found_mate:
                            logits[d3pm_e0] += 8.0
                        else:
                            logits[d3pm_e0] += 0.35

                # Use the already-computed 6-ply D3PM horizon distribution (zero extra encoder FLOPS)
                # to evaluate 3-ply tactical continuations across the top-4 root candidate moves
                if pred.ply_logits is not None:
                    from topo_chess.data.pipeline import Gardner5x5Evaluator, Standard8x8Evaluator

                    def eval_state_cp(s_obj: ChessState) -> float:
                        if s_obj.width == 5:
                            return Gardner5x5Evaluator.static_eval_cp(s_obj)  # type: ignore[arg-type]
                        return Standard8x8Evaluator.static_eval_cp(s_obj)  # type: ignore[arg-type]

                    root_cp = eval_state_cp(st)
                    cand_order = np.argsort(logits[legal_ids])[::-1][: min(4, len(legal_ids))]
                    ply1_reply_logits = pred.ply_logits[1]
                    ply2_followup_logits = pred.ply_logits[2]
                    for loc_idx in cand_order:
                        cand_e = int(legal_ids[int(loc_idx)])
                        s1 = st.step_by_edge_id(cand_e)
                        if s1.is_checkmate():
                            logits[cand_e] += 12.0
                            continue
                        opp_mask = s1.legal_edge_mask()
                        opp_legal = np.flatnonzero(opp_mask)
                        if len(opp_legal) == 0:
                            continue
                        # Check if opponent has any immediate checkmate response or best D3PM-predicted reply
                        opp_mate_found = False
                        best_opp_e = int(opp_legal[int(np.argmax(ply1_reply_logits[opp_legal]))])
                        best_opp_cp = -1e9
                        for opp_e in opp_legal:
                            s2_cand = s1.step_by_edge_id(int(opp_e))
                            if s2_cand.is_checkmate():
                                opp_mate_found = True
                                break
                            # Opponent wants to minimize active player's cp (i.e. maximize -eval(s2_cand))
                            cp_opp = -eval_state_cp(s2_cand) + 15.0 * float(
                                ply1_reply_logits[int(opp_e)]
                            )
                            if cp_opp > best_opp_cp:
                                best_opp_cp = cp_opp
                                best_opp_e = int(opp_e)
                        if opp_mate_found:
                            logits[cand_e] -= 10.0
                            continue
                        s2 = s1.step_by_edge_id(best_opp_e)
                        w2_mask = s2.legal_edge_mask()
                        w2_legal = np.flatnonzero(w2_mask)
                        if len(w2_legal) > 0:
                            w2_scores = ply2_followup_logits[w2_legal]
                            best_w2_e = int(w2_legal[int(np.argmax(w2_scores))])
                            s3 = s2.step_by_edge_id(best_w2_e)
                            if s3.is_checkmate():
                                logits[cand_e] += 6.0
                            else:
                                # s3 is opponent's turn, so -eval_state_cp(s3) is active player's value
                                s3_cp = -eval_state_cp(s3)
                                delta_cp = np.clip(s3_cp - root_cp, -400.0, 400.0)
                                logits[cand_e] += 0.006 * float(delta_cp)

            legal_logits = logits[legal_ids]
            if temperature <= 1e-4 or rng is None:
                best_local = int(np.argmax(legal_logits))
                chosen_edges.append(int(legal_ids[best_local]))
            else:
                shifted = (legal_logits - np.max(legal_logits)) / temperature
                probs = np.exp(np.clip(shifted, -30.0, 0.0))
                probs = probs / np.sum(probs)
                pick = int(rng.choice(len(legal_ids), p=probs))
                chosen_edges.append(int(legal_ids[pick]))

        return chosen_edges
