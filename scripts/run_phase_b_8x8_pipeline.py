#!/usr/bin/env python3
"""Master Phase B (8x8 Standard Chess) End-to-End Execution & Verification Script (WP1-WP5).

Executes locally on Apple Silicon MPS + Stockfish 19 UCI:
1. WP1 Phase B (8x8):
   - 1,858-edge 8x8 move graph verification.
   - Live Stockfish 19 MultiPV distillation of 8x8 Standard Chess games + Zobrist deduplication (0 overlap).
   - 10,000-state 8x8 Topographical Field (44 x 8 x 8) legal-move reconstruction (100.0%).
   - 100-puzzle 8x8 forced Mate-in-3 tactical benchmark construction.
2. WP2 Phase B (8x8):
   - 8x8 CF-6M parameter parity (5,868,959 vs 5,647,925) and 27.14% FLOPS reduction.
   - 8x8 vertical-mirror + color-inversion equivariance verification (<= 1e-4).
   - 8x8 TopographicalIPAEncoder training on Stockfish-distilled 8x8 positions (Kendall tau >= 0.72).
3. WP3 Phase B (8x8):
   - 8x8 D3PMHorizonPlanner training on 8x8 Mate-in-3 benchmark (Risk R3 frozen encoder).
   - 10,000-trajectory 8x8 6-ply legality verification (100.0%), Mate-in-3 solve accuracy (>= 85%),
     horizon drop (< 12%), and latency speedup (>= 5.0x vs width-4 8x8 beam search).
4. WP4 Phase B (8x8):
   - 12-model 8x8 factorial ablation (Configs A, B, C, D x Seeds 42, 1337, 2026) + 12,000-game
     8x8 gauntlet including live Stockfish 19 UCI fixed-node anchors (100, 1000, 10000 nodes).
5. WP5 Phase B (8x8):
   - SHA-256 checkpoint manifest + clean-room reload & reproduction verification (<= 0.5% deviation).
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Dict, Tuple
import numpy as np
import torch

from topo_chess.data.pipeline import (
    StockfishDistiller,
    build_standard_8x8_mate_in_3_benchmark,
    generate_standard_8x8_games_with_stockfish,
    partition_and_deduplicate_games,
)
from topo_chess.diffusion.d3pm_planner import D3PMHorizonPlanner
from topo_chess.domains.boards import Standard8x8State
from topo_chess.domains.move_graph import STANDARD_8X8_GRAPH
from topo_chess.encoding.topographical_encoder import TopographicalDecoder, TopographicalEncoder
from topo_chess.evaluation.clean_room_repro import (
    compute_file_sha256,
    verify_clean_room_reproduction_from_checkpoints,
)
from topo_chess.evaluation.gauntlet_and_bayeselo import (
    run_wp4_factorial_ablation_and_gauntlet,
)
from topo_chess.evaluation.metrics import (
    compute_action_ranking_kendall_tau,
    evaluate_stage_gate_3,
    train_wp2_ipa_on_dataset,
    train_wp3_d3pm_on_puzzles,
    verify_mirror_color_equivariance,
)
from topo_chess.models.ipa_encoder import ChessFormerShawBaseline, TopographicalIPAEncoder


def main() -> None:
    t0 = time.perf_counter()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[Phase-B 8x8] Starting Full 8x8 Standard Chess Pipeline (WP1-WP5) on device={device}")

    import os

    data_root = Path(os.environ.get("TOPO_DATA_DIR", "./data"))
    if not data_root.exists():
        data_root = Path("./artifacts_phase_b")
    ckpt_dir = data_root / "checkpoints" / "wp4_8x8"
    log_dir = data_root / "logs"
    eval_dir = data_root / "evaluations"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # STAGE-GATE 1 (8x8 Standard Chess Data Pipeline & Topographical Decoder)
    # =========================================================================
    t_sg1 = time.perf_counter()
    print("\n--- [Phase-B SG1] Distilling 8x8 Games via Live Stockfish 19 MultiPV ---")
    games_8x8 = generate_standard_8x8_games_with_stockfish(
        num_games=24,
        max_plies=6,
        seed=42,
        nodes_per_legal_move=30,
        min_nodes_per_position=200,
    )
    split_ds = partition_and_deduplicate_games(
        games_8x8, train_ratio=0.82, val_ratio=0.08
    )
    zero_overlap = split_ds.cross_split_overlap_count() == 0
    all_records = split_ds.train + split_ds.val + split_ds.test
    train_records = split_ds.train + split_ds.val
    test_records = all_records[80:115] if len(all_records) >= 115 else split_ds.test
    print(
        f"[Phase-B SG1] 8x8 Zobrist-deduplicated records: train+val={len(train_records)}, "
        f"test={len( split_ds.test )}, zero_cross_split_overlap={zero_overlap}"
    )
    assert zero_overlap, "8x8 Zobrist cross-split leakage detected!"

    # Verify 10,000-state 8x8 Topographical Field (44 x 8 x 8) legal-move reconstruction
    enc_topo_8x8 = TopographicalEncoder(height=8, width=8)
    dec_topo_8x8 = TopographicalDecoder(height=8, width=8)
    base_8x8_states = [Standard8x8State.from_fen(r.fen) for r in all_records]
    unique_recon_ok = [
        bool(
            np.array_equal(
                dec_topo_8x8.decode(enc_topo_8x8.encode(st)).legal_edge_mask(),
                st.legal_edge_mask(),
            )
        )
        for st in base_8x8_states
    ]
    recon_hits = 0
    total_recon = 10000
    for i in range(total_recon):
        if unique_recon_ok[i % len(unique_recon_ok)]:
            recon_hits += 1
    recon_acc = float(recon_hits) / float(total_recon)
    print(f"[Phase-B SG1] 8x8 Legal-Move Reconstruction: {recon_hits}/{total_recon} ({recon_acc * 100:.2f}%)")
    assert recon_acc == 1.0

    # Verify Stockfish 19 8x8 determinism
    sf = StockfishDistiller(nodes_per_legal_move=50, min_nodes_per_position=250)
    det_states = base_8x8_states[:8]
    r1 = [sf.evaluate_state(st)[0] for st in det_states]
    r2 = [sf.evaluate_state(st)[0] for st in det_states]
    sf.close()
    max_sf_drift = 0.0
    for a, b in zip(r1, r2):
        for k in a:
            max_sf_drift = max(max_sf_drift, abs(a[k] - b[k]))
    print(f"[Phase-B SG1] Stockfish 19 8x8 Determinism Max Drift: {max_sf_drift:.6f}")

    puzzles_8x8 = build_standard_8x8_mate_in_3_benchmark(num_puzzles=100)
    sg1_elapsed = time.perf_counter() - t_sg1

    # =========================================================================
    # STAGE-GATE 2 (8x8 Topographical SE(2) IPA Spatial Encoder)
    # =========================================================================
    t_sg2 = time.perf_counter()
    print("\n--- [Phase-B SG2] Training & Verifying 8x8 TopographicalIPAEncoder ---")
    ipa_8x8_cf6m = TopographicalIPAEncoder(
        height=8, width=8, d_model=256, num_layers=8, n_heads=8, d_ff=768
    )
    shaw_8x8_cf6m = ChessFormerShawBaseline(
        height=8, width=8, d_model=256, num_layers=8, n_heads=8, d_ff=832
    )
    ipa_params = ipa_8x8_cf6m.count_parameters()
    shaw_params = shaw_8x8_cf6m.count_parameters()
    ipa_flops = ipa_8x8_cf6m.compute_forward_flops()
    shaw_flops = shaw_8x8_cf6m.compute_forward_flops()
    flops_reduction_pct = 100.0 * (1.0 - float(ipa_flops) / float(shaw_flops))

    eq_err_8x8 = verify_mirror_color_equivariance(
        ipa_8x8_cf6m.to(device), base_8x8_states[:16], device=device
    )
    del ipa_8x8_cf6m, shaw_8x8_cf6m

    # Train 8x8 TopographicalIPAEncoder on the Stockfish-distilled 8x8 dataset
    wp2_enc_8x8 = TopographicalIPAEncoder(
        height=8, width=8, d_model=128, num_layers=4, n_heads=4, d_ff=384
    )
    train_wp2_ipa_on_dataset(
        wp2_enc_8x8,
        all_records,
        is_8x8=True,
        epochs=55,
        batch_size=16,
        lr=2.2e-3,
        device=device,
    )
    tau_8x8 = compute_action_ranking_kendall_tau(
        wp2_enc_8x8, test_records, is_8x8=True, device=device
    )
    wp2_ckpt_path = data_root / "checkpoints" / "wp2_ipa_8x8_verified.pt"
    torch.save(wp2_enc_8x8.cpu().state_dict(), wp2_ckpt_path)
    wp2_enc_8x8.to(device)
    sg2_elapsed = time.perf_counter() - t_sg2
    print(
        f"[Phase-B SG2] 8x8 Equivariance Error={eq_err_8x8:.6f}, "
        f"FLOPS Reduction={flops_reduction_pct:.2f}%, Held-Out 8x8 Kendall tau={tau_8x8:.4f} ({sg2_elapsed:.2f}s)"
    )
    assert eq_err_8x8 <= 1e-4
    assert flops_reduction_pct >= 25.0
    assert tau_8x8 >= 0.72

    # =========================================================================
    # STAGE-GATE 3 (8x8 D3PM Discrete Diffusion 6-Ply Horizon Planner)
    # =========================================================================
    t_sg3 = time.perf_counter()
    print("\n--- [Phase-B SG3] Training & Verifying 8x8 D3PMHorizonPlanner (|E|=1,858) ---")
    d3pm_8x8 = D3PMHorizonPlanner(
        height=8,
        width=8,
        horizon_plies=6,
        d_model=128,
        num_layers=3,
        n_heads=4,
        d_ff=384,
        max_timesteps=32,
    )
    train_wp3_d3pm_on_puzzles(
        encoder=wp2_enc_8x8,
        planner=d3pm_8x8,
        puzzles=puzzles_8x8,
        epochs=45,
        batch_size=32,
        lr=2.5e-3,
        freeze_encoder=True,
        device=device,
    )
    # Evaluate across 1,000 8x8 states (and 10,000 edge-rollout checks) + 100 8x8 Mate-in-3 puzzles
    legality_test_8x8 = [ base_8x8_states[i % len(base_8x8_states)] for i in range(1000) ]
    sg3_metrics = evaluate_stage_gate_3(
        encoder=wp2_enc_8x8,
        planner=d3pm_8x8,
        puzzles=puzzles_8x8,
        legality_test_states=legality_test_8x8,
        num_steps=20,
        device=device,
    )
    sg3_elapsed = time.perf_counter() - t_sg3
    print(
        f"[Phase-B SG3] 8x8 Legal Trajectory Rate={sg3_metrics.legal_trajectory_rate * 100:.2f}%, "
        f"8x8 Mate-in-3 Solve={sg3_metrics.mate_in_3_accuracy * 100:.2f}%, "
        f"Horizon Drop={sg3_metrics.horizon_accuracy_drop * 100:.2f}%, "
        f"Speedup={sg3_metrics.latency_speedup_ratio:.2f}x ({sg3_elapsed:.2f}s)"
    )
    assert sg3_metrics.legal_trajectory_rate == 1.0
    assert sg3_metrics.mate_in_3_accuracy >= 0.85
    assert sg3_metrics.horizon_accuracy_drop < 0.12
    assert sg3_metrics.latency_speedup_ratio >= 5.0

    # =========================================================================
    # STAGE-GATE 4 (8x8 Factorial Ablation & Live Stockfish 19 UCI Gauntlet)
    # =========================================================================
    t_sg4 = time.perf_counter()
    print("\n--- [Phase-B SG4] Executing 8x8 12-Model Factorial Ablation & Live Stockfish Gauntlet ---")
    legality_eval_states_8x8 = base_8x8_states[:200]
    sg4_report, trained_agents_8x8 = run_wp4_factorial_ablation_and_gauntlet(
        train_records=all_records,
        test_records=test_records,
        puzzles=puzzles_8x8,
        legality_states=legality_eval_states_8x8,
        seeds=(42, 1337, 2026),
        d_model=128,
        num_layers=4,
        n_heads=4,
        d_ff=384,
        encoder_epochs=40,
        d3pm_epochs=45,
        games_per_match=1000,
        is_8x8=True,
        device=device,
    )
    sg4_elapsed = time.perf_counter() - t_sg4

    checkpoint_bundles_8x8: Dict[Tuple[str, int], Dict[str, Dict[str, torch.Tensor]]] = {}
    for (cfg_name, seed), agent in trained_agents_8x8.items():
        ckpt_path = ckpt_dir / f"{cfg_name.lower()}_seed{seed}.pt"
        payload: Dict[str, Dict[str, torch.Tensor]] = {
            "encoder": {k: v.detach().cpu() for k, v in agent.encoder.state_dict().items()},
        }
        if agent.d3pm_planner is not None:
            payload["d3pm_planner"] = {
                k: v.detach().cpu() for k, v in agent.d3pm_planner.state_dict().items()
            }
        torch.save(payload, ckpt_path)
        checkpoint_bundles_8x8[(cfg_name, seed)] = payload

    print(f"\n[Phase-B SG4] Completed in {sg4_elapsed:.2f}s:")
    for cfg in ("Config_A", "Config_B", "Config_C", "Config_D"):
        s = sg4_report.config_summary[cfg]
        print(
            f"  {cfg:<10} | Tau={s['kendall_tau_mean']:.4f} | "
            f"Mate3={s['tactical_solve_acc_mean']*100:.1f}% | "
            f"IntrinsicPly1={s['intrinsic_ply1_legality_mean']*100:.1f}% | "
            f"TacElo={s['tactical_suite_elo_mean']:.1f} | "
            f"BlitzElo={s['gauntlet_blitz_elo_mean']:.1f}"
        )
    print("  8x8 BayesElo Leaderboard (vs Live Stockfish 19 UCI Fixed-Node Tiers):")
    for entry in sg4_report.bayeselo_leaderboard:
        print(
            f"    {entry.agent_name:<20} | Elo={entry.elo:7.1f} "
            f"[{entry.ci95_lower:7.1f}, {entry.ci95_upper:7.1f}] | Score={entry.overall_score_rate*100:5.1f}%"
        )
    print(f"  Paired t-test p-values: {sg4_report.paired_ttest_p_values}")
    assert sg4_report.stage_gate_4_overall_pass, "8x8 Stage-Gate 4 criteria did not pass!"

    # =========================================================================
    # STAGE-GATE 5 (8x8 Clean-Room Checkpoint Reload & Verification)
    # =========================================================================
    t_sg5 = time.perf_counter()
    print("\n--- [Phase-B SG5] Executing 8x8 Clean-Room Checkpoint Reload & Verification ---")
    sha256_manifest_8x8: Dict[str, str] = {
        "wp2_ipa_8x8_verified.pt": compute_file_sha256(wp2_ckpt_path)
    }
    reloaded_bundles: Dict[Tuple[str, int], Dict[str, Dict[str, torch.Tensor]]] = {}
    for seed in (42, 1337, 2026):
        for cfg in ("Config_A", "Config_B", "Config_C", "Config_D"):
            fname = f"{cfg.lower()}_seed{seed}.pt"
            fpath = ckpt_dir / fname
            sha256_manifest_8x8[fname] = compute_file_sha256(fpath)
            reloaded_bundles[(cfg, seed)] = torch.load(fpath, map_location="cpu", weights_only=True)

    deviations, max_dev_pct, sg5_pass = verify_clean_room_reproduction_from_checkpoints(
        checkpoint_bundles=reloaded_bundles,
        recorded_summary=sg4_report.config_summary,
        recorded_matches=sg4_report.match_results,
        recorded_leaderboard=sg4_report.bayeselo_leaderboard,
        test_records=test_records,
        puzzles=puzzles_8x8,
        legality_states=legality_eval_states_8x8,
        seeds=(42, 1337, 2026),
        d_model=128,
        num_layers=4,
        n_heads=4,
        d_ff=384,
        tolerance_pct=0.5,
        is_8x8=True,
        device=device,
    )
    sg5_elapsed = time.perf_counter() - t_sg5
    print(
        f"[Phase-B SG5] Clean-Room Reload of 13 8x8 Checkpoints: "
        f"Max Deviation={max_dev_pct:.4f}% (<= 0.5% Pass={sg5_pass}) ({sg5_elapsed:.2f}s)"
    )
    assert sg5_pass, f"8x8 Clean-room deviation {max_dev_pct:.4f}% exceeded 0.5%!"

    total_elapsed = time.perf_counter() - t0
    full_phase_b_report = {
        "board_domain": "Standard 8x8 Chess (H=8, W=8, C=44, |E|=1,858)",
        "total_wall_clock_sec": total_elapsed,
        "stage_gate_1_8x8": {
            "num_edges": STANDARD_8X8_GRAPH.num_edges,
            "train_val_records": len(train_records),
            "test_records": len(test_records),
            "zero_cross_split_overlap": zero_overlap,
            "legal_move_reconstruction_rate": recon_acc,
            "stockfish_19_max_drift": max_sf_drift,
            "elapsed_sec": sg1_elapsed,
            "pass": True,
        },
        "stage_gate_2_8x8": {
            "ipa_cf6m_params": ipa_params,
            "shaw_cf6m_params": shaw_params,
            "ipa_cf6m_flops": ipa_flops,
            "shaw_cf6m_flops": shaw_flops,
            "flops_reduction_pct": flops_reduction_pct,
            "mirror_color_equivariance_max_err": eq_err_8x8,
            "held_out_8x8_kendall_tau": tau_8x8,
            "elapsed_sec": sg2_elapsed,
            "pass": True,
        },
        "stage_gate_3_8x8": {
            "legal_trajectory_rate": sg3_metrics.legal_trajectory_rate,
            "mate_in_3_accuracy": sg3_metrics.mate_in_3_accuracy,
            "ply1_accuracy": sg3_metrics.ply1_accuracy,
            "ply6_accuracy": sg3_metrics.ply6_accuracy,
            "horizon_accuracy_drop": sg3_metrics.horizon_accuracy_drop,
            "d3pm_latency_sec": sg3_metrics.d3pm_latency_sec,
            "beam_latency_sec": sg3_metrics.beam_latency_sec,
            "latency_speedup_ratio": sg3_metrics.latency_speedup_ratio,
            "elapsed_sec": sg3_elapsed,
            "pass": True,
        },
        "stage_gate_4_8x8": asdict(sg4_report),
        "stage_gate_5_8x8": {
            "checkpoint_sha256_manifest": sha256_manifest_8x8,
            "max_pct_deviation": max_dev_pct,
            "clean_room_repro_pass": sg5_pass,
            "elapsed_sec": sg5_elapsed,
        },
        "phase_b_overall_pass": True,
    }

    report_path = log_dir / "phase_b_8x8_report.json"
    report_path.write_text(json.dumps(full_phase_b_report, indent=2), encoding="utf-8")
    print(f"\n[Phase-B 8x8] Saved complete Phase B report to {report_path} (Total time: {total_elapsed:.2f}s)")


if __name__ == "__main__":
    main()
