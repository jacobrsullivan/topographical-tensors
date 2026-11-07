"""Tests for Phase C (v3.0.0) Full-Scale Lichess Pipeline, ALT-008, ALT-009 & Unmodified Stockfish UCI."""

from __future__ import annotations

import numpy as np
import torch

from topo_chess.data.lichess_phase_c import canonicalize_8x8_state
from topo_chess.diffusion.d3pm_planner import D3PMHorizonPlanner
from topo_chess.domains.boards import Standard8x8State
from topo_chess.encoding.batched_torch_encoder import BatchedTorchTopographicalEncoder
from topo_chess.encoding.topographical_encoder import TopographicalEncoder
from topo_chess.evaluation.phase_c_trainer_and_gauntlet import (
    PureStockfishUCIAgent,
    play_full_120ply_lichess_match,
)
from topo_chess.models.ablation_configs import AblationConfigAgent
from topo_chess.models.ipa_encoder import TopographicalIPAEncoder


def test_batched_torch_topographical_encoder_equivalence() -> None:
    """Verify ALT-008 BatchedTorchTopographicalEncoder matches TopographicalEncoder across 8x8 states."""
    fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    ]
    states = [Standard8x8State.from_fen(f) for f in fens]
    cpu_enc = TopographicalEncoder(height=8, width=8)
    gpu_enc = BatchedTorchTopographicalEncoder()

    pt_list = []
    pc_list = []
    rm_list = []
    expected = []
    for st in states:
        expected.append(cpu_enc.encode_canonical(st))
        pt_c, pc_c, rm_c = canonicalize_8x8_state(st)
        pt_list.append(pt_c)
        pc_list.append(pc_c)
        rm_list.append(rm_c)

    pt_t = torch.from_numpy(np.stack(pt_list, axis=0))
    pc_t = torch.from_numpy(np.stack(pc_list, axis=0))
    rm_t = torch.from_numpy(np.stack(rm_list, axis=0))
    got = gpu_enc(pt_t, pc_t, rm_t).detach().cpu().numpy()
    exp_np = np.stack(expected, axis=0)

    max_diff = float(np.max(np.abs(got - exp_np)))
    assert max_diff <= 1e-4, f"BatchedTorchTopographicalEncoder max diff {max_diff} > 1e-4"


def test_d3pm_auxiliary_legality_head_alt009() -> None:
    """Verify ALT-009 bilinear edge-indexed legality head improves intrinsic unmasked legality."""
    torch.manual_seed(42)
    planner = D3PMHorizonPlanner(
        height=8,
        width=8,
        horizon_plies=6,
        d_model=64,
        num_layers=2,
        n_heads=4,
        d_ff=128,
        max_timesteps=16,
        enable_legality_head=True,
    )
    bsz = 4
    num_edges = planner.num_edges
    spatial_tokens = torch.randn(bsz, 64, 64)
    root_q = torch.randn(bsz, num_edges)
    clean_traj = torch.randint(0, 100, (bsz, 6), dtype=torch.long)
    target_masks = torch.zeros(bsz, 6, num_edges)
    target_masks.scatter_(2, clean_traj.unsqueeze(-1), 1.0)

    opt = torch.optim.AdamW(planner.parameters(), lr=5e-3)
    initial_loss = 0.0
    final_loss = 0.0
    for step in range(25):
        opt.zero_grad()
        out = planner.compute_loss(
            clean_trajectories=clean_traj,
            spatial_tokens=spatial_tokens,
            root_q_logits=root_q,
            target_legal_masks=target_masks,
        )
        if step == 0:
            initial_loss = float(out.total_loss.item())
        final_loss = float(out.total_loss.item())
        out.total_loss.backward()
        opt.step()

    assert final_loss < initial_loss


def test_pure_stockfish_uci_agent_and_120ply_match() -> None:
    """Verify PureStockfishUCIAgent (UCI_Elo & Fixed-Node) and 120-ply match runner."""
    sf_1500 = PureStockfishUCIAgent(
        "Stockfish_UCI_Elo_1500", node_budget=100, uci_elo=1500, skill_level=1
    )
    try:
        enc = TopographicalIPAEncoder(height=8, width=8, d_model=64, num_layers=2, n_heads=4, d_ff=192)
        agent_c = AblationConfigAgent("Config_C", seed=42, encoder=enc, d3pm_planner=None)
        batched_topo = BatchedTorchTopographicalEncoder()
        openings = ["rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"]
        res = play_full_120ply_lichess_match(
            agent_a=agent_c,
            name_a="Config_C",
            agent_b=sf_1500,
            name_b="Stockfish_UCI_Elo_1500",
            opening_fens=openings,
            batched_topo=batched_topo,
            max_plies=16,
            temperature=0.0,
            seed=42,
        )
        assert res.total_games == 2
        assert sf_1500.uci_calls >= 1
    finally:
        sf_1500.close()
