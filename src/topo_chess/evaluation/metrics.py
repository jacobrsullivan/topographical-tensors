"""Evaluation & Stage-Gate 1-3 Verification Suite for Topographical Tensors and D3PM Chess."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import List, Optional, Sequence
import numpy as np
from scipy.stats import kendalltau
import torch
import torch.nn.functional as F

from topo_chess.data.pipeline import (
    PositionRecord,
    TacticalPuzzle,
)
from topo_chess.diffusion.d3pm_planner import AutoregressiveBeamBaseline, D3PMHorizonPlanner
from topo_chess.domains.boards import ChessState, Gardner5x5State, Standard8x8State
from topo_chess.models.ipa_encoder import TopographicalIPAEncoder


def compute_action_ranking_kendall_tau(
    encoder: TopographicalIPAEncoder,
    records: Sequence[PositionRecord],
    is_8x8: bool = False,
    device: Optional[torch.device] = None,
) -> float:
    """Compute mean Kendall rank correlation coefficient (tau) between model Q-values and ground truth."""
    encoder.eval()
    taus: List[float] = []

    states: List[ChessState] = [
        Standard8x8State.from_fen(r.fen) if is_8x8 else Gardner5x5State.from_fen(r.fen)
        for r in records
    ]
    with torch.no_grad():
        t_in = encoder.encode_states_to_tensor(states, device=device)
        out = encoder(t_in)
        pred_q = out.q_values.detach().cpu().numpy()

    for idx, rec in enumerate(records):
        edges = rec.legal_edge_ids
        if len(edges) < 2:
            continue
        gt_vals = np.array([rec.q_values[e] for e in edges], dtype=np.float64)
        pr_vals = np.array([pred_q[idx, e] for e in edges], dtype=np.float64)
        if np.allclose(gt_vals, gt_vals[0]):
            continue
        tau, _ = kendalltau(gt_vals, pr_vals)
        if not np.isnan(tau):
            taus.append(float(tau))

    return float(np.mean(taus)) if taus else 0.0


def verify_mirror_color_equivariance(
    encoder: TopographicalIPAEncoder,
    states: Sequence[ChessState],
    device: Optional[torch.device] = None,
) -> float:
    """Measure max absolute difference in action-value outputs between s and its mirrored/color-swapped twin."""
    encoder.eval()
    mirrored_states = [s.mirror_vertical_swap_colors() for s in states]
    with torch.no_grad():
        t_orig = encoder.encode_states_to_tensor(list(states), device=device)
        t_mirr = encoder.encode_states_to_tensor(mirrored_states, device=device)
        out_orig = encoder(t_orig)
        out_mirr = encoder(t_mirr)
        max_diff = float(torch.max(torch.abs(out_orig.q_values - out_mirr.q_values)).item())
    return max_diff


def train_wp2_ipa_on_dataset(
    encoder: TopographicalIPAEncoder,
    records: Sequence[PositionRecord],
    is_8x8: bool = False,
    epochs: int = 35,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
) -> List[float]:
    """Train the WP2 TopographicalIPAEncoder on supervised Q-value & WDLP targets."""
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    encoder.to(device)
    encoder.train()

    states: List[ChessState] = [
        Standard8x8State.from_fen(r.fen) if is_8x8 else Gardner5x5State.from_fen(r.fen)
        for r in records
    ]
    x_all = encoder.encode_states_to_tensor(states, device=device)
    num_samples = len(records)
    num_edges = encoder.graph.num_edges

    q_targets = torch.zeros((num_samples, num_edges), dtype=torch.float32, device=device)
    rank_targets = torch.zeros((num_samples, num_edges), dtype=torch.float32, device=device)
    q_masks = torch.zeros((num_samples, num_edges), dtype=torch.float32, device=device)
    wdlp_targets = torch.zeros((num_samples, 4), dtype=torch.float32, device=device)

    for i, rec in enumerate(records):
        e_list = list(rec.q_values.keys())
        vals = np.array([rec.q_values[e] for e in e_list], dtype=np.float32)
        # Compute normalized within-state rank targets in [0.05, 0.95] to amplify fine centipawn orderings
        if len(vals) > 1 and np.max(vals) > np.min(vals):
            order = np.argsort(np.argsort(vals)).astype(np.float32) / float(len(vals) - 1)
            r_vals = 0.05 + 0.90 * order
        else:
            r_vals = vals
        for idx_e, e_id in enumerate(e_list):
            q_targets[i, e_id] = float(vals[idx_e])
            rank_targets[i, e_id] = float(r_vals[idx_e])
            q_masks[i, e_id] = 1.0
        wdlp_targets[i] = torch.tensor(rec.wdlp_target, dtype=torch.float32, device=device)

    optimizer = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    epoch_losses: List[float] = []

    for _epoch in range(epochs):
        perm = torch.randperm(num_samples, device=device)
        total_loss = 0.0
        steps = 0
        for start in range(0, num_samples, batch_size):
            idx = perm[start : start + batch_size]
            bx = x_all[idx]
            bq = q_targets[idx]
            br = rank_targets[idx]
            bm = q_masks[idx]
            bw = wdlp_targets[idx]

            optimizer.zero_grad()
            out = encoder(bx)
            # Pairwise Bradley-Terry ranking loss (directly optimizes Kendall tau) + calibrated Q-value MSE
            prob_err = ((out.q_values - bq) ** 2) * bm
            mse_q_loss = prob_err.sum() / torch.clamp(bm.sum(), min=1.0)

            pair_losses: List[torch.Tensor] = []
            for b_i in range(bx.shape[0]):
                leg_idx = torch.nonzero(bm[b_i] > 0.5, as_tuple=False).squeeze(-1)
                if leg_idx.numel() < 2:
                    continue
                r_leg = br[b_i, leg_idx]
                l_leg = out.q_logits[b_i, leg_idx]
                dr = r_leg.unsqueeze(1) - r_leg.unsqueeze(0)
                valid_pairs = dr > 1e-5
                if torch.any(valid_pairs):
                    dl = l_leg.unsqueeze(1) - l_leg.unsqueeze(0)
                    pair_losses.append(F.softplus(-4.0 * dl[valid_pairs]).mean())

            rank_loss = (
                torch.stack(pair_losses).mean()
                if pair_losses
                else torch.tensor(0.0, device=device)
            )
            w_loss = F.mse_loss(torch.sigmoid(out.wdlp_logits), bw)
            loss = rank_loss + mse_q_loss + 0.1 * w_loss
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            steps += 1
        scheduler.step()
        epoch_losses.append(total_loss / max(steps, 1))

    encoder.eval()
    return epoch_losses


def train_wp3_d3pm_on_puzzles(
    encoder: TopographicalIPAEncoder,
    planner: D3PMHorizonPlanner,
    puzzles: Sequence[TacticalPuzzle],
    epochs: int = 45,
    batch_size: int = 32,
    lr: float = 2e-3,
    freeze_encoder: bool = True,
    device: Optional[torch.device] = None,
) -> List[float]:
    """Train the WP3 D3PMHorizonPlanner on 6-ply tactical trajectories (implementing Risk R3 encoder freeze)."""
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    encoder.to(device)
    planner.to(device)

    if freeze_encoder:
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad = False

    states = [p.initial_state for p in puzzles]
    with torch.no_grad():
        x_in = encoder.encode_states_to_tensor(states, device=device)
        enc_out = encoder(x_in)
        spatial_tokens = enc_out.spatial_tokens.detach()
        root_q_logits = enc_out.q_logits.detach()

    clean_traj = torch.tensor(
        [p.pv_edge_ids[: planner.horizon_plies] for p in puzzles],
        dtype=torch.long,
        device=device,
    )

    optimizer = torch.optim.AdamW(planner.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    num_samples = len(puzzles)
    losses: List[float] = []

    planner.train()
    for _epoch in range(epochs):
        perm = torch.randperm(num_samples, device=device)
        ep_loss = 0.0
        steps = 0
        for start in range(0, num_samples, batch_size):
            idx = perm[start : start + batch_size]
            b_traj = clean_traj[idx]
            b_tok = spatial_tokens[idx]
            b_q = root_q_logits[idx]

            optimizer.zero_grad()
            loss_out = planner.compute_loss(b_traj, b_tok, b_q)
            loss_out.total_loss.backward()
            optimizer.step()
            ep_loss += float(loss_out.total_loss.item())
            steps += 1
        scheduler.step()
        losses.append(ep_loss / max(steps, 1))

    if freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad = True
    planner.eval()
    return losses


@dataclass
class StageGate3Metrics:
    """Quantitative verification report for Stage-Gate 3 (D3PM Horizon Planner)."""

    legal_trajectory_rate: float
    mate_in_3_accuracy: float
    ply1_accuracy: float
    ply6_accuracy: float
    horizon_accuracy_drop: float
    d3pm_latency_sec: float
    beam_latency_sec: float
    latency_speedup_ratio: float


def evaluate_stage_gate_3(
    encoder: TopographicalIPAEncoder,
    planner: D3PMHorizonPlanner,
    puzzles: Sequence[TacticalPuzzle],
    legality_test_states: Sequence[ChessState],
    num_steps: int = 20,
    device: Optional[torch.device] = None,
) -> StageGate3Metrics:
    """Run all four Stage-Gate 3 metrics: legality, mate-in-3 accuracy, horizon drop, and latency speedup."""
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    encoder.to(device).eval()
    planner.to(device).eval()

    # 1. Legality test across test states
    legal_count = 0
    batch_sz = 128
    with torch.no_grad():
        for start in range(0, len(legality_test_states), batch_sz):
            batch_states = legality_test_states[start : start + batch_sz]
            t_in = encoder.encode_states_to_tensor(list(batch_states), device=device)
            enc_out = encoder(t_in)
            preds = planner.sample_trajectories_batch(
                batch_states, enc_out.spatial_tokens, enc_out.q_logits, num_steps=num_steps
            )
            legal_count += sum(1 for pr in preds if pr.is_fully_legal)
    legal_rate = float(legal_count) / float(max(len(legality_test_states), 1))

    # 2. Mate-in-3 puzzle evaluation & D3PM latency timing
    puzzle_states = [p.initial_state for p in puzzles]
    t0 = time.perf_counter()
    with torch.no_grad():
        p_in = encoder.encode_states_to_tensor(puzzle_states, device=device)
        p_enc = encoder(p_in)
        puzzle_preds = planner.sample_trajectories_batch(
            puzzle_states, p_enc.spatial_tokens, p_enc.q_logits, num_steps=num_steps
        )
    d3pm_elapsed = max(time.perf_counter() - t0, 1e-6)

    solved = 0
    ply1_hits = 0
    ply6_hits = 0
    for pz, pred in zip(puzzles, puzzle_preds):
        gt = pz.pv_edge_ids
        pr = pred.edge_ids
        if len(pr) >= 1 and pr[0] == gt[0]:
            ply1_hits += 1
        if len(pr) >= 6 and pr[5] == gt[5]:
            ply6_hits += 1
        # Solved if either the exact 5-ply PV matches or the resulting trajectory delivers checkmate
        delivered_mate = any(st.is_checkmate() for st in pred.states[1:])
        exact_pv = len(pr) >= 5 and pr[:5] == gt[:5]
        if exact_pv or delivered_mate:
            solved += 1

    n_pz = float(max(len(puzzles), 1))
    mate_acc = solved / n_pz
    ply1_acc = ply1_hits / n_pz
    ply6_acc = ply6_hits / n_pz
    horizon_drop = max(0.0, ply1_acc - ply6_acc)

    # 3. Autoregressive Beam Search Baseline latency timing on the same puzzle batch
    beam_baseline = AutoregressiveBeamBaseline(encoder=encoder, beam_width=4)
    t1 = time.perf_counter()
    beam_baseline.rollout_beam_search(puzzle_states, horizon_plies=6, device=device)
    beam_elapsed = max(time.perf_counter() - t1, 1e-6)

    return StageGate3Metrics(
        legal_trajectory_rate=legal_rate,
        mate_in_3_accuracy=mate_acc,
        ply1_accuracy=ply1_acc,
        ply6_accuracy=ply6_acc,
        horizon_accuracy_drop=horizon_drop,
        d3pm_latency_sec=d3pm_elapsed,
        beam_latency_sec=beam_elapsed,
        latency_speedup_ratio=beam_elapsed / d3pm_elapsed,
    )
