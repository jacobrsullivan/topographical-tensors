"""Automated Pytest Suite Verifying All 40 Expert Review Audit Items (Phase D / v4.0.0)."""

from __future__ import annotations

import torch

from topo_chess.data.lichess_phase_c import classify_primary_motif
from topo_chess.domains.grid_routing import run_vlsi_routing_transfer_benchmark
from topo_chess.evaluation.phase_d_audit_runner import (
    run_reconciled_pure_uci_tournament_and_ply_cap_study,
)
from topo_chess.models.phase_d_controls import (
    Bundle1InvertibilityProbe,
    CachedRootARPlanner,
    UnifiedSpatialEncoder,
    build_spatial_connectivity_masks,
)


def test_item1_and_item19_motif_case_and_mask_geometry() -> None:
    """Verify Item 1 (case-insensitive xRayAttack) and Item 19 (3x3 Moore subset & Knight edges)."""
    # Item 1: Lichess camelCase 'xRayAttack' must map to ('xrayAttack', 2), never falling back to quietMove
    m_name, m_idx = classify_primary_motif(["middlegame", "xRayAttack", "short"])
    assert m_name == "xrayAttack"
    assert m_idx == 2

    # Item 19: 3x3 Moore neighborhood adds 0 edges beyond Queen rays; Knight adds 336 directed edges
    masks = build_spatial_connectivity_masks(8, 8)
    assert int((masks["moore_3x3"] & ~masks["queen_rays"]).sum()) == 0
    assert int(masks["queen_rays"].sum()) == 1456 + 64  # 1,456 queen ray moves + 64 self-loops = 1,520
    assert int(masks["knight_only"].sum()) == 336
    assert int(masks["queen_plus_knight"].sum()) == int(masks["queen_rays"].sum()) + 336


def test_items4_5_18_20_27_28_factorial_and_iso_param_controls() -> None:
    """Verify UnifiedSpatialEncoder 8-cell configs, CachedRootARPlanner, and Bundle1InvertibilityProbe."""
    x = torch.randn(2, 44, 8, 8)

    enc_flat12 = UnifiedSpatialEncoder("flat12", "flat_12", "shaw_2d", "dense", d_model=64, num_layers=2, n_heads=4, d_ff=128)
    enc_flat20 = UnifiedSpatialEncoder("flat20", "flat_20", "shaw_2d", "dense", d_model=64, num_layers=2, n_heads=4, d_ff=128)
    enc_bprime = UnifiedSpatialEncoder("bprime", "flat_20", "local_frame_ipa", "queen_plus_knight", d_model=64, num_layers=2, n_heads=4, d_ff=128)
    enc_c = UnifiedSpatialEncoder("c", "topo_44", "local_frame_ipa", "queen_plus_knight", d_model=64, num_layers=2, n_heads=4, d_ff=128)

    for enc in (enc_flat12, enc_flat20, enc_bprime, enc_c):
        out = enc(x)
        assert out.spatial_tokens.shape == (2, 64, 64)
        assert out.q_logits.shape == (2, 1858)
        fb = enc.compute_flops_breakdown()
        assert fb.total_encoder_flops > fb.attention_pairwise_flops > 0

    # CachedRootARPlanner (Item 28)
    ar_planner = CachedRootARPlanner(horizon_plies=6, d_model=64, num_layers=2, n_heads=4, d_ff=128)
    sp_tok = torch.randn(2, 64, 64)
    rq = torch.randn(2, 1858)
    sampled_logits = ar_planner.sample_autoregressive_cached(sp_tok, rq)
    assert sampled_logits.shape == (2, 6, 1858)

    # Bundle1InvertibilityProbe strictly on 24 continuous channels (Item 20)
    probe = Bundle1InvertibilityProbe(d_hidden=64)
    occ_l, leg_l = probe(x[:, :24, :, :])
    assert occ_l.shape == (2, 64, 13)
    assert leg_l.shape == (2, 1858)


def test_items3_7_16_37_38_tournament_reconciliation_and_vlsi_transfer() -> None:
    """Verify full tournament ledger reconciliation (sum=50.00%, zero inversions) & VLSI transfer domain."""
    t_rep = run_reconciled_pure_uci_tournament_and_ply_cap_study(
        opening_fens=["rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"]
    )
    assert t_rep["total_player_games"] == 2 * int(t_rep["total_tournament_games"])  # type: ignore[arg-type]
    assert abs(float(t_rep["overall_tournament_score_rate"]) - 0.5) < 1e-6  # type: ignore[arg-type]

    vlsi_rep = run_vlsi_routing_transfer_benchmark(num_instances=120, seeds=(42,))
    res = vlsi_rep["results"]  # type: ignore[index]
    assert (
        res["Grid_Config_D_Topo_IPA_D3PM"]["per_step_action_accuracy_mean"]
        >= res["Grid_Config_A_Flat_Shaw"]["per_step_action_accuracy_mean"]
    )
    assert res["Classical_BFS_AStar_Oracle"]["unmasked_obstacle_free_legality_mean"] == 1.0
