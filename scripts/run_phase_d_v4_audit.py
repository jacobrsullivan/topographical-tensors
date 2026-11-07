#!/usr/bin/env python3
"""CLI Runner for Phase D (v4.0.0) 40-Item Reviewer Audit & Full Factorial Control Suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from topo_chess.evaluation.phase_d_audit_runner import run_phase_d_full_audit


def format_phase_d_markdown_tables(report: Dict[str, object]) -> str:
    """Format complete v4.0.0 Manuscript Tables from the Phase D JSON telemetry report."""
    report["wp_d1_forensic_hygiene"]  # type: ignore[index]
    wp2 = report["wp_d2_architectural_controls"]  # type: ignore[index]
    wp3 = report["wp_d3_mechanistic_and_transfer"]  # type: ignore[index]
    wp4 = report["wp_d4_reconciled_tournament"]  # type: ignore[index]

    enc_mat = wp2["spatial_encoder_8cell_matrix"]  # type: ignore[index]
    pz_ctrl = wp2["standardized_10k_lichess_puzzle_controls"]  # type: ignore[index]
    sc_12 = wp2["sample_complexity_12point_scaling_item40"]  # type: ignore[index]
    vlsi = wp3["vlsi_2d_grid_routing_transfer_item7"]  # type: ignore[index]
    ledger = wp4["reconciled_leaderboard_ledger"]  # type: ignore[index]
    wald = wp4["key_wald_elo_comparisons"]  # type: ignore[index]
    ply_cap = wp4["ply_cap_sensitivity_study"]  # type: ignore[index]

    lines = [
        "# Manuscript Tables (`v4.0.0` — Phase D 40-Item Reviewer Audit & Complete Factorial Controls)",
        "",
        f"**Proposed Title:** *{report['title']}*",
        "",
        "## Table 1: Complete 8-Cell Spatial Encoder Factorial Matrix + Iso-Parameter `Config_C_Large` (Items 1, 2, 4, 12, 17, 18, 19, 27, 39)",
        "",
        "| Cell ID | Input Channels | Attention Kernel | Connectivity Mask | Params | Pairwise Attn FLOPS | Total Encoder FLOPS | Held-Out $\\tau$ (`40,634` States) | Stockfish MultiPV $\\tau$ (`2k`) | Top-1 Legal Acc | Pin $\\tau$ | X-Ray $\\tau$ | Discovered $\\tau$ | Fork $\\tau$ | Quiet $\\tau$ |",
        "| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for c_name, c_data in enc_mat.items():
        m = c_data["mean_metrics"]
        fb = c_data["flops_breakdown"]
        lines.append(
            f"| `{c_name}` | `{c_data['input_mode']}` | `{c_data['attention_kernel']}` | `{c_data['connectivity_mask']}` ({fb['active_edge_density']*100:.1f}%) | `{c_data['parameters']:,}` | `{fb['attention_pairwise_flops']/1e6:.2f}M` | `{fb['total_encoder_flops']/1e6:.2f}M` | `{c_data['overall_tau_mean']:.4f} ± {c_data['overall_tau_sd']:.4f}` | `{m['tau_stockfish_multipv_2k']:.4f}` | `{m['top1_legal_accuracy']*100:.2f}%` | `{m['tau_pin']:.4f}` | `{m['tau_xrayAttack']:.4f}` | `{m['tau_discoveredAttack']:.4f}` | `{m['tau_fork']:.4f}` | `{m['tau_quietMove']:.4f}` |"
        )

    lines.extend(
        [
            "",
            "## Table 2: Iso-Parameter (`8.57M`) & Iso-Supervision (`250k` PVs) Horizon Controls on `10,000` External Held-Out Lichess Puzzles (Items 5, 25, 28, 32)",
            "",
            "| Model Configuration | Total Params | Denoising Steps $T$ | Root Ply-1 Accuracy | Interactive Multi-Turn Solve Rate | Single-Root $z(s_0)$ Solve Rate | `<1400` | `1400–1800` | `1800–2200` | `2200+` | `mateIn3` | `pin` | `xrayAttack` | `discoveredAttack` |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    for m_name, p_data in pz_ctrl.items():
        lines.append(
            f"| `{m_name}` | `{p_data['total_parameters']:,}` | `{p_data['denoising_steps_T']}` | `{p_data['ply1_first_move_accuracy']*100:.2f}%` | `{p_data['interactive_multi_turn_solve_rate']*100:.2f}%` | `{p_data['single_root_z0_horizon_solve_rate']*100:.2f}%` | `{p_data['solve_Band_1_Under1400']*100:.2f}%` | `{p_data['solve_Band_2_1400_1800']*100:.2f}%` | `{p_data['solve_Band_3_1800_2200']*100:.2f}%` | `{p_data['solve_Band_4_Over2200']*100:.2f}%` | `{p_data['solve_motif_mateIn3']*100:.2f}%` | `{p_data['solve_motif_pin']*100:.2f}%` | `{p_data['solve_motif_xrayAttack']*100:.2f}%` | `{p_data['solve_motif_discoveredAttack']*100:.2f}%` |"
        )

    lines.extend(
        [
            "",
            "## Table 3: 12-Point Sample-Complexity Legality Phase Transition at Fixed Optimization Budget (Items 6, 10, 25, 29, 30, 40)",
            "",
            "| Dataset Size $N$ | Fixed Opt Steps | Unmasked Per-Ply Legality (With `ALT-009`) | Unmasked Full 6-Ply Rollout (With `ALT-009`) | Unmasked Per-Ply Legality (No `ALT-009`) | Unmasked Full 6-Ply Rollout (No `ALT-009`) | Post-`ALT-002` Safety Projection |",
            "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    with_a9 = sc_12["with_alt009_legality_head"]
    no_a9 = sc_12["without_alt009_legality_head"]
    for n_sz in sc_12["sample_sizes"]:
        k_str = f"N_{n_sz}"
        w_row = with_a9[k_str]
        n_row = no_a9[k_str]
        lines.append(
            f"| `{n_sz:,}` | `{w_row['fixed_optimization_steps']}` | `{w_row['unmasked_mean_per_ply_legality']*100:.2f}%` | `{w_row['unmasked_full_6ply_rollout_legality']*100:.2f}%` | `{n_row['unmasked_mean_per_ply_legality']*100:.2f}%` | `{n_row['unmasked_full_6ply_rollout_legality']*100:.2f}%` | `100.00%` |"
        )

    lines.extend(
        [
            "",
            "## Table 4: Fully Reconciled 14-Participant Connected Tournament Ledger vs. Unmodified Stockfish 19 UCI (Items 3, 16, 34, 37, 38)",
            "",
            f"*Total Pairings: `{wp4['total_pairings']}` | Games per Pairing: `{wp4['games_per_pairing']}` | Total Games: `{wp4['total_tournament_games']:,}` | Total Player-Games: `{wp4['total_player_games']:,}` | Total Points Sum: `{wp4['total_points_sum']:,.1f}` | Overall Mean Score Rate: `{wp4['overall_tournament_score_rate']*100:.4f}%`*",
            "",
            "| Rank | Participant | Category | Calibrated BayesElo | 95% CI | SE | Total Games | Total Points | Overall Score Rate | Games vs SF Anchors | Score vs SF Anchors | Games vs Neural Pool | Score vs Neural Pool |",
            "| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    for idx_r, row in enumerate(ledger, start=1):
        sf_sc = (
            f"{row['score_rate_vs_stockfish_anchors']*100:.2f}%"
            if row["score_rate_vs_stockfish_anchors"] is not None
            else "—"
        )
        np_sc = (
            f"{row['score_rate_vs_neural_pool']*100:.2f}%"
            if row["score_rate_vs_neural_pool"] is not None
            else "—"
        )
        lines.append(
            f"| {idx_r} | `{row['player']}` | `{row['category']}` | **`{row['bayeselo']:.1f}`** | `[{row['ci_low']:.1f}, {row['ci_high']:.1f}]` | `{row['standard_error']:.2f}` | `{row['total_games']:,}` | `{row['total_points']:,.1f}` | `{row['overall_score_rate']*100:.2f}%` | `{row['games_vs_stockfish_anchors']:,}` | `{sf_sc}` | `{row['games_vs_neural_pool']:,}` | `{np_sc}` |"
        )

    lines.extend(
        [
            "",
            "## Table 5: Exact Fisher Covariance Wald $z$-Test Comparisons (Item 38)",
            "",
            "| Comparison | $\\Delta\\text{Elo}$ | $\\text{SE}_{\\text{diff}}$ | 95% CI of $\\Delta\\text{Elo}$ | $z$-Statistic | Two-Sided Wald $p$-Value |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    for c_key, c_val in wald.items():
        lines.append(
            f"| `{c_key}` | `{c_val['delta_elo']:+.2f}` | `{c_val['se_diff']:.2f}` | `[{c_val['ci95_diff_low']:+.2f}, {c_val['ci95_diff_high']:+.2f}]` | `{c_val['z_stat']:.3f}` | `{c_val['two_sided_wald_p_value']:.3e}` |"
        )

    lines.extend(
        [
            "",
            "## Table 6: Game-Length Ply-Cap Sensitivity Study (Item 21)",
            "",
            "| Ply Cap (`max_plies`) | Natural Checkmate | Stalemate / Insuff. | Threefold Repetition | 50-Move Rule | Capped & Adjudicated | `Config_D` Elo | `Config_C` Elo | $\\Delta\\text{Elo}(\\text{D vs C})$ |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    for cap_k, cap_v in ply_cap.items():
        lines.append(
            f"| `{cap_k}` (`{cap_v['max_plies']}` plies) | `{cap_v['natural_checkmate_pct']:.1f}%` | `{cap_v['stalemate_or_insufficient_pct']:.1f}%` | `{cap_v['threefold_repetition_pct']:.1f}%` | `{cap_v['fifty_move_rule_pct']:.1f}%` | `{cap_v['reached_ply_cap_adjudicated_pct']:.1f}%` | `{cap_v['config_d_elo']:.1f}` | `{cap_v['config_c_elo']:.1f}` | `{cap_v['delta_elo_d_vs_c']:+.1f}` |"
        )

    lines.extend(
        [
            "",
            "## Table 7: Secondary Transfer Domain — 2D Octilinear VLSI / Grid Multi-Pin Obstacle Routing (Items 7, 13)",
            "",
            "| Routing Architecture | Unmasked Obstacle-Free Path Legality | Via + Sink Full Route Completion | Per-Step Action Accuracy |",
            "| :--- | :---: | :---: | :---: |",
        ]
    )

    for g_name, g_res in vlsi["results"].items():
        lines.append(
            f"| `{g_name}` | `{g_res['unmasked_obstacle_free_legality_mean']*100:.2f}% ± {g_res['unmasked_obstacle_free_legality_std']*100:.2f}%` | `{g_res['full_route_completion_rate_mean']*100:.2f}% ± {g_res['full_route_completion_rate_std']*100:.2f}%` | `{g_res['per_step_action_accuracy_mean']*100:.2f}%` |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    import os

    repo_root = Path(__file__).resolve().parents[1]
    default_data_dir = Path(os.environ.get("TOPO_DATA_DIR", str(repo_root / "data")))
    parser = argparse.ArgumentParser(description="Run Phase D v4.0.0 40-Item Audit & Control Suite")
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=default_data_dir / "raw" / "lichess_db_puzzle.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_data_dir,
    )
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=repo_root,
    )
    parser.add_argument("--num-positions", type=int, default=1_000_000)
    parser.add_argument("--num-puzzles", type=int, default=250_000)
    parser.add_argument("--external-test-puzzles", type=int, default=10_000)
    parser.add_argument("--live-stockfish-states", type=int, default=2_000)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--encoder-steps", type=int, default=45)
    args = parser.parse_args()

    ckpt_dir = args.repo_dir / "checkpoints" / "wp4_phase_d"
    report = run_phase_d_full_audit(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        checkpoint_dir=ckpt_dir,
        num_positions=args.num_positions,
        num_puzzles=args.num_puzzles,
        external_test_puzzles=args.external_test_puzzles,
        live_stockfish_states=args.live_stockfish_states,
        num_workers=args.num_workers,
        encoder_steps=args.encoder_steps,
    )

    for base_d in (args.output_dir, args.repo_dir):
        log_d = base_d / "logs"
        eval_d = base_d / "evaluations"
        log_d.mkdir(parents=True, exist_ok=True)
        eval_d.mkdir(parents=True, exist_ok=True)
        (log_d / "phase_d_v4_audit_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        (eval_d / "manuscript_tables_v4.0_phase_d.md").write_text(
            format_phase_d_markdown_tables(report), encoding="utf-8"
        )

    print(
        f"[Phase D v4.0.0 Complete] all_40_audit_items_resolved={report['all_40_audit_items_resolved']} in {report['total_wall_clock_sec']}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
