"""Phase B (8x8 Standard Chess) End-to-End Verification Tests (WP1-WP5)."""

from __future__ import annotations

from typing import Dict, Tuple
import torch

from topo_chess.data.pipeline import (
    build_standard_8x8_mate_in_3_benchmark,
    generate_standard_8x8_games_with_stockfish,
    partition_and_deduplicate_games,
)
from topo_chess.domains.boards import Standard8x8State
from topo_chess.evaluation.clean_room_repro import verify_clean_room_reproduction_from_checkpoints
from topo_chess.evaluation.gauntlet_and_bayeselo import run_wp4_factorial_ablation_and_gauntlet


def test_standard_8x8_mate_in_3_puzzles_and_distillation() -> None:
    """Verify 8x8 forced Mate-in-3 puzzles end in checkmate and 8x8 Zobrist deduplication has 0 overlap."""
    puzzles = build_standard_8x8_mate_in_3_benchmark(num_puzzles=12)
    assert len(puzzles) == 12
    for pz in puzzles:
        cur = pz.initial_state
        assert cur.width == 8 and cur.height == 8
        reached_mate = False
        for e_id in pz.pv_edge_ids[:5]:
            cur = cur.step_by_edge_id(e_id)
            if cur.is_checkmate():
                reached_mate = True
        assert reached_mate, f"8x8 puzzle {pz.puzzle_id} did not end in checkmate!"

    games = generate_standard_8x8_games_with_stockfish(
        num_games=8, max_plies=6, seed=42, nodes_per_legal_move=20, min_nodes_per_position=100
    )
    split_ds = partition_and_deduplicate_games(games)
    assert split_ds.cross_split_overlap_count() == 0


def test_phase_b_8x8_ablation_and_clean_room_repro() -> None:
    """Verify 8x8 WP4 factorial ablation + live Stockfish UCI gauntlet + WP5 8x8 clean-room reload."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    games = generate_standard_8x8_games_with_stockfish(
        num_games=12, max_plies=6, seed=42, nodes_per_legal_move=20, min_nodes_per_position=100
    )
    split_ds = partition_and_deduplicate_games(games)
    all_records = split_ds.train + split_ds.val + split_ds.test
    test_records = split_ds.test if len(split_ds.test) >= 4 else all_records[:8]
    puzzles = build_standard_8x8_mate_in_3_benchmark(num_puzzles=16)
    legality_states = [Standard8x8State.from_fen(r.fen) for r in all_records[:16]]

    report, agents = run_wp4_factorial_ablation_and_gauntlet(
        train_records=all_records,
        test_records=test_records,
        puzzles=puzzles,
        legality_states=legality_states,
        seeds=(42, 1337),
        d_model=64,
        num_layers=2,
        n_heads=4,
        d_ff=128,
        encoder_epochs=12,
        d3pm_epochs=20,
        games_per_match=16,
        is_8x8=True,
        device=device,
    )
    assert len(report.bayeselo_leaderboard) == 7
    assert report.config_summary["Config_D"]["post_projection_legality_mean"] == 1.0

    bundles: Dict[Tuple[str, int], Dict[str, Dict[str, torch.Tensor]]] = {}
    for (cfg_name, seed), agent in agents.items():
        b: Dict[str, Dict[str, torch.Tensor]] = {
            "encoder": {k: v.detach().cpu().clone() for k, v in agent.encoder.state_dict().items()}
        }
        if agent.d3pm_planner is not None:
            b["d3pm_planner"] = {
                k: v.detach().cpu().clone() for k, v in agent.d3pm_planner.state_dict().items()
            }
        bundles[(cfg_name, seed)] = b

    _, max_dev, all_ok = verify_clean_room_reproduction_from_checkpoints(
        checkpoint_bundles=bundles,
        recorded_summary=report.config_summary,
        recorded_matches=report.match_results,
        recorded_leaderboard=report.bayeselo_leaderboard,
        test_records=test_records,
        puzzles=puzzles,
        legality_states=legality_states,
        seeds=(42, 1337),
        d_model=64,
        num_layers=2,
        n_heads=4,
        d_ff=128,
        tolerance_pct=0.5,
        is_8x8=True,
        device=device,
    )
    assert all_ok
    assert max_dev <= 0.5
