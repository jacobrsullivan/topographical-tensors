#!/usr/bin/env python3
"""End-to-End WP4 Execution & Stage-Gate 4 Verification Runner (100% Local).

Executes:
1. 12-Model Factorial Ablation Matrix (`Configs A, B, C, D` x `Seeds [42, 1337, 2026]`) on M4 Pro `mps`
   with Risk R3 Encoder Weight Sharing (`ALT-007`) between `Config_C` and `Config_D`.
2. FLOPS-Normalized Tactical Horizon Benchmark + Intrinsic (Pre-Projection) vs. Post-Projection
   Trajectory Legality Diagnostics across all 12 trained models.
3. 1,000-Game Match Gauntlets (500 diverse opening/tactical battlegrounds x 2 colors per match)
   across Configs A-D and Fixed-Node Search Tiers (`100`, `1,000`, `10,000` nodes), plus live
   8x8 Stockfish 19 fixed-node verification.
4. Exact BayesElo Maximum-Likelihood ratings with 95% confidence intervals and paired t-tests (`p < 0.01`).
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import torch

# Ensure package root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from topo_chess.data.pipeline import (
    build_gardner_mate_in_3_benchmark,
    generate_synthetic_gardner_games,
    partition_and_deduplicate_games,
)
from topo_chess.domains.boards import Gardner5x5State
from topo_chess.evaluation.gauntlet_and_bayeselo import run_wp4_factorial_ablation_and_gauntlet
from topo_chess.models.ablation_configs import (
    BitboardGlobalTransformer,
    ChessFormerTokenShawEncoder,
)
from topo_chess.models.ipa_encoder import TopographicalIPAEncoder


def main() -> None:
    t0 = time.perf_counter()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("=" * 84)
    print(f"TOPOGRAPHICAL TENSORS — WP4 FACTORIAL ABLATION & ELO BENCHMARKS (Device: {device})")
    print("=" * 84)

    # 1. Verify 8x8 CF-6M Parameter & FLOPS Parity across Configs A, B, C/D
    enc_a_8x8 = BitboardGlobalTransformer(
        height=8, width=8, d_model=256, num_layers=8, n_heads=8, d_ff=832
    )
    enc_b_8x8 = ChessFormerTokenShawEncoder(
        height=8, width=8, d_model=256, num_layers=8, n_heads=8, d_ff=832
    )
    enc_c_8x8 = TopographicalIPAEncoder(
        height=8, width=8, d_model=256, num_layers=8, n_heads=8, d_ff=768
    )

    print("\n[WP4 CF-6M 8x8 Scale Architecture Parity]")
    print(
        f"  Config_A (Bitboard + Global 1D):     {enc_a_8x8.count_parameters():,d} params | "
        f"{enc_a_8x8.compute_forward_flops():,d} FLOPS"
    )
    print(
        f"  Config_B (Tokens 112ch + 2D Shaw):   {enc_b_8x8.count_parameters():,d} params | "
        f"{enc_b_8x8.compute_forward_flops():,d} FLOPS"
    )
    print(
        f"  Config_C/D (Topo Field + SE(2) IPA): {enc_c_8x8.count_parameters():,d} params | "
        f"{enc_c_8x8.compute_forward_flops():,d} FLOPS"
    )

    # 2. Generate Zobrist-deduplicated dataset & tactical benchmark suite
    games = generate_synthetic_gardner_games(num_games=65, max_plies=16, seed=42)
    split = partition_and_deduplicate_games(games)
    all_records = split.train + split.val + split.test
    train_records = all_records[:280]
    test_records = all_records[280:360] if len(all_records) >= 360 else split.test

    puzzles = build_gardner_mate_in_3_benchmark(num_puzzles=100)
    legality_states = [Gardner5x5State.from_fen(r.fen) for r in all_records[:250]]

    # 3. Execute 4-Config x 3-Seed Factorial Ablation + 1,000-Game Gauntlets + BayesElo
    report, trained_agents = run_wp4_factorial_ablation_and_gauntlet(
        train_records=train_records,
        test_records=test_records,
        puzzles=puzzles,
        legality_states=legality_states,
        seeds=(42, 1337, 2026),
        d_model=128,
        num_layers=4,
        n_heads=4,
        d_ff=384,
        encoder_epochs=32,
        d3pm_epochs=45,
        games_per_match=1000,
        device=device,
    )

    elapsed = time.perf_counter() - t0

    print("\n[WP4 Factorial Ablation Summary (Mean +/- Std across Seeds 42, 1337, 2026)]")
    for cfg in ("Config_A", "Config_B", "Config_C", "Config_D"):
        s = report.config_summary[cfg]
        print(
            f"  {cfg}: Kendall tau = {s['kendall_tau_mean']:.4f} +/- {s['kendall_tau_std']:.4f} | "
            f"Tactical Solve = {s['tactical_solve_acc_mean'] * 100:.1f}% | "
            f"Intrinsic Ply-1 Legal = {s['intrinsic_ply1_legality_mean'] * 100:.1f}% | "
            f"Intrinsic Per-Ply Legal = {s['intrinsic_per_ply_legality_mean'] * 100:.1f}% | "
            f"Post-Proj Legal = {s['post_projection_legality_mean'] * 100:.1f}% | "
            f"Tactical Elo = {s['tactical_suite_elo_mean']:.1f} | "
            f"Gauntlet Blitz Elo = {s['gauntlet_blitz_elo_mean']:.1f}"
        )

    print("\n[WP4 1,000-Game Match Gauntlet Results]")
    for mr in report.match_results:
        print(
            f"  {mr.player_a:18s} vs {mr.player_b:18s}: "
            f"+{mr.wins_a:3d} ={mr.draws:3d} -{mr.losses_a:3d} "
            f"({mr.win_rate_a * 100:5.1f}%)"
        )

    print("\n[WP4 BayesElo Leaderboard (with 95% Confidence Intervals)]")
    for entry in report.bayeselo_leaderboard:
        print(
            f"  {entry.agent_name:18s}: {entry.elo:7.1f} Elo "
            f"(95% CI: [{entry.ci95_lower:7.1f}, {entry.ci95_upper:7.1f}]) | "
            f"Games: {entry.games_played:4d} | Score: {entry.overall_score_rate * 100:5.1f}%"
        )

    print("\n[WP4 Paired t-Test Statistical Significance]")
    for k, p_val in report.paired_ttest_p_values.items():
        print(f"  {k:40s}: p = {p_val:.6e}")

    print("\n[WP4 8x8 Stockfish 19 Fixed-Node Tiers Verification]")
    for tier_name, info in report.stockfish_8x8_fixed_node_benchmarks.items():
        print(
            f"  {tier_name:28s}: {info['ms_per_position']:.2f} ms/pos | "
            f"max Q drift = {info['determinism_max_q_drift']:.6f}"
        )

    print("\n" + "=" * 84)
    print(
        f"STAGE-GATE 4 OVERALL VERDICT: {'PASSED' if report.stage_gate_4_overall_pass else 'FAILED'} "
        f"(Total Wall-Clock: {elapsed:.2f}s)"
    )
    print(f"  SG4-A (Statistical Significance p < 0.01):   {report.sg4_a_statistical_significance_pass}")
    print(
        f"  SG4-B (Config D vs C Gain >= +80 Elo):       +{report.sg4_b_tactical_elo_gain_d_vs_c:.1f} Elo "
        f"({report.sg4_b_pass})"
    )
    print(
        f"  SG4-C (Config D Blitz Rating >= 2300 Elo):   {report.sg4_c_config_d_blitz_elo:.1f} Elo "
        f"({report.sg4_c_pass})"
    )
    print("=" * 84)

    import os

    # Save checkpoints and JSON report to TOPO_DATA_DIR
    data_root = Path(os.environ.get("TOPO_DATA_DIR", "./data"))
    if data_root.exists():
        log_dir = data_root / "logs"
        ckpt_dir = data_root / "checkpoints" / "wp4"
        log_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        payload = asdict(report)
        payload["wall_clock_sec"] = elapsed
        payload["cf6m_8x8_parity"] = {
            "Config_A": {
                "parameters": enc_a_8x8.count_parameters(),
                "spatial_flops": enc_a_8x8.compute_forward_flops(),
            },
            "Config_B": {
                "parameters": enc_b_8x8.count_parameters(),
                "spatial_flops": enc_b_8x8.compute_forward_flops(),
            },
            "Config_C_D": {
                "parameters": enc_c_8x8.count_parameters(),
                "spatial_flops": enc_c_8x8.compute_forward_flops(),
            },
        }
        report_path = log_dir / "stage_gate_4_report.json"
        report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        for (cfg_name, seed), agent in trained_agents.items():
            ckpt_file = ckpt_dir / f"{cfg_name.lower()}_seed{seed}.pt"
            state_bundle = {"encoder": agent.encoder.state_dict()}
            if agent.d3pm_planner is not None:
                state_bundle["d3pm_planner"] = agent.d3pm_planner.state_dict()
            torch.save(state_bundle, ckpt_file)
        print(f"Saved Stage-Gate 4 JSON report to {report_path}")
        print(f"Saved 12 ablation checkpoints to {ckpt_dir}")

    if not report.stage_gate_4_overall_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
