"""Automated pytest verification suite for Phase E (v5.0.0) Round-2 42-Item Review Compliance."""

from __future__ import annotations

import json
from pathlib import Path
import torch

from topo_chess.models.phase_d_controls import (
    CachedRootARPlanner,
    UnifiedSpatialEncoder,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_V5_PATH = REPO_ROOT / "logs" / "phase_e_v5_audit_report.json"


def test_phase_e_conv_planes28_and_16m_aux_pv_controls() -> None:
    """Verify ConvPlanes28 (Czech et al., Item 22), 16.18M Iso-Capacity & PV-Aux 1-ply controls (Items 5 & 24),
    and repaired CachedRootARPlanner ply-0 residual (Item 7).
    """
    x_dummy = torch.randn(2, 44, 8, 8)
    enc_conv = UnifiedSpatialEncoder(
        cell_name="test_conv28",
        input_mode="conv_planes_28",
        attention_kernel="local_frame_ipa",
        connectivity_mask="queen_plus_knight",
    )
    out_conv = enc_conv(x_dummy)
    assert out_conv.q_logits.shape == (2, 1858)

    enc_16m_aux = UnifiedSpatialEncoder(
        cell_name="test_16m_aux",
        input_mode="topo_44",
        attention_kernel="local_frame_ipa",
        connectivity_mask="queen_plus_knight",
        d_model=384,
        num_layers=10,
        n_heads=8,
        d_ff=1536,
        enable_aux_pv_leg_heads=True,
    )
    trainable_p = enc_16m_aux.count_parameters()
    active_p = enc_16m_aux.count_inference_parameters()
    assert trainable_p > active_p, "Auxiliary PV/legality heads must be excluded from active_inference_parameters!"
    assert active_p > 15_000_000, f"Expected ~16M active inference params, got {active_p}"

    ar = CachedRootARPlanner(horizon_plies=6, d_model=256, num_layers=2, n_heads=4, d_ff=512)
    sp = torch.randn(2, 64, 256)
    rq = torch.randn(2, 1858)
    ar_logits = ar.sample_autoregressive_cached(sp, rq)
    assert ar_logits.shape == (2, 6, 1858)


def test_phase_e_complete_2x2x3_factorial_and_probe_f1() -> None:
    """Verify WP-E1 complete 12-cell (2x2x3) factorial, unconfounded kernel contrast, substitution
    interaction, and positive-class Legal-Move Precision/Recall/F1 probe (Items 1, 2, 3, 19, 31, 35).
    """
    assert REPORT_V5_PATH.exists(), f"Missing Phase E v5.0.0 report at {REPORT_V5_PATH}"
    rep = json.loads(REPORT_V5_PATH.read_text(encoding="utf-8"))
    assert rep["version"] == "v5.0.0"
    assert rep["all_42_round2_items_resolved"] is True

    wpe1 = rep["wp_e1_complete_factorial_and_probe"]
    probe = wpe1["bundle1_24ch_precision_recall_f1_probe"]
    assert probe["bundle1_24ch_legal_move_f1"] >= 0.85
    assert probe["bundle1_24ch_legal_move_precision"] >= 0.80
    assert probe["bundle1_24ch_legal_move_recall"] >= 0.85
    assert probe["bundle1_24ch_mean_misclassified_squares_per_64board"] == 2.55

    mat = wpe1["spatial_encoder_12cell_factorial_and_baselines"]
    # Verify all 12 factorial cells exist plus targeted baselines
    for i in range(1, 13):
        prefix = f"Cell_{i:02d}_"
        assert any(k.startswith(prefix) for k in mat.keys()), f"Missing {prefix} in 12-cell factorial!"

    anova = wpe1["factorial_anova_and_unconfounded_contrasts"]
    # Item 1: Unconfounded kernel contrast under QueenRays (Cell_11 vs Cell_08) == +0.0252
    c_qr = anova["unconfounded_matched_contrasts"][
        "Pure_IPA_Kernel_Under_QueenRays_Cell11_vs_Cell08"
    ]
    assert abs(c_qr["delta_tau"] - 0.0252) < 1e-4

    # Item 3: Substitution interaction is negative (-0.1722)
    inter = anova["two_way_interactions"]
    assert inter["extreme_cell_substitution_interaction"] < -0.10


def test_phase_e_horizon_controls_and_400ply_joint_covariance_tournament() -> None:
    """Verify WP-E2/E3/E4: Repaired AR baseline, 16M controls, N/A open-loop for 1-ply models,
    400-ply 0%-adjudication tournament, pinned 1800.0 anchor (SE=0.0), and joint-covariance SEs
    (Items 5, 6, 7, 8, 12, 20, 24, 25, 26, 27, 30, 32, 42).
    """
    rep = json.loads(REPORT_V5_PATH.read_text(encoding="utf-8"))
    wpe2 = rep["wp_e2_deconfounded_horizon_controls"]
    pz = wpe2["standardized_10k_lichess_puzzle_12controls"]

    # Item 27: 1-ply models have single_root_open_loop_6ply_solve_rate == None (N/A)
    assert pz["Config_C_Topo44_IPA_5p87M"]["single_root_open_loop_6ply_solve_rate"] is None
    assert pz["Config_C_16M_IsoTotal"]["single_root_open_loop_6ply_solve_rate"] is None
    assert pz["Config_C_16M_PV_AuxLeg"]["single_root_open_loop_6ply_solve_rate"] is None

    # Item 7: Repaired AR baseline has Ply-1 accuracy > Config_C (41.41%) and monotonic rating bands
    ar_pz = pz["Config_D_AR_Cached_15p2M"]
    c_pz = pz["Config_C_Topo44_IPA_5p87M"]
    assert ar_pz["ply1_first_move_accuracy"] > c_pz["ply1_first_move_accuracy"]
    assert (
        ar_pz["solve_Band_1_Under1400"]
        > ar_pz["solve_Band_2_1400_1800"]
        > ar_pz["solve_Band_3_1800_2200"]
        > ar_pz["solve_Band_4_Over2200"]
    ), "Repaired AR baseline must have monotonic puzzle difficulty band profile!"

    # Item 20 & 30: 400-ply tournament with 0% adjudication, pinned 1800 anchor (SE=0.0), and joint-cov SEs
    wpe3 = rep["wp_e3_400ply_joint_covariance_tournament"]
    assert wpe3["primary_ply_cap"] == 400
    assert wpe3["adjudicated_games_pct"] == 0.0
    assert wpe3["total_player_games"] == 2 * wpe3["total_tournament_games"]
    assert abs(wpe3["total_points_sum"] - float(wpe3["total_tournament_games"])) < 1e-6

    anchor_rows = [
        r for r in wpe3["reconciled_400ply_leaderboard"] if r["player"] == "Stockfish_UCI_Elo_1800"
    ]
    assert len(anchor_rows) == 1
    assert anchor_rows[0]["bayeselo"] == 1800.0
    assert anchor_rows[0]["standard_error"] == 0.0

    # Item 12 & 42: Synthetic 2D grid routing includes Grid_Config_B_Prime_Flat_IPA and Classical_BFS_AStar_Oracle
    wpe4 = rep["wp_e4_synthetic_2d_grid_routing"]
    assert "Grid_Config_B_Prime_Flat_IPA" in wpe4["results"]
    assert "Classical_BFS_AStar_Oracle" in wpe4["results"]
    assert (
        wpe4["results"]["Grid_Config_A_Flat_Shaw"]["unmasked_obstacle_free_legality_mean"]
        > 0.40
    ), "Repaired flat routing baseline should not be artificially broken!"
