"""Stage-Gate 5 Verification Tests: WP5 Clean-Room Reproduction (SG5-A) & Editorial/NeurIPS Audit (SG5-B)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
import torch

from topo_chess.data.pipeline import (
    build_gardner_mate_in_3_benchmark,
    generate_synthetic_gardner_games,
    partition_and_deduplicate_games,
)
from topo_chess.domains.boards import Gardner5x5State
from topo_chess.evaluation.clean_room_repro import (
    audit_editorial_and_neurips_compliance,
    verify_clean_room_reproduction_from_checkpoints,
)
from topo_chess.evaluation.gauntlet_and_bayeselo import run_wp4_factorial_ablation_and_gauntlet


def test_sg5_b_editorial_neurips_and_risk_r4_compliance() -> None:
    """Verify SG5-B: Formal proofs, hyperparams/datasheet, Risk R4 dual reporting, Dockerfile & checklists."""
    repo_root = Path(__file__).resolve().parents[1]
    audit = audit_editorial_and_neurips_compliance(repo_root)

    assert audit.proofs_complete
    assert audit.hyperparameters_and_datasheet_complete
    assert audit.risk_r4_dual_benchmark_reported
    assert audit.neurips_checklist_complete
    assert audit.ste100_compliance_complete
    assert audit.dockerfile_present
    assert audit.flight_recorder_complete
    assert audit.checklist_items_verified == 16
    assert audit.passed


def test_sg5_a_clean_room_checkpoint_reproduction() -> None:
    """Verify SG5-A: Reloading saved checkpoints in a clean room reproduces all metrics within <= +/-0.5%."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    games = generate_synthetic_gardner_games(num_games=30, max_plies=10, seed=42)
    split = partition_and_deduplicate_games(games)
    all_records = split.train + split.val + split.test
    train_records = all_records[:120]
    test_records = all_records[120:160]

    puzzles = build_gardner_mate_in_3_benchmark(num_puzzles=30)
    legality_states = [Gardner5x5State.from_fen(r.fen) for r in all_records[:40]]

    seeds = (42, 1337, 2026)
    report, trained_agents = run_wp4_factorial_ablation_and_gauntlet(
        train_records=train_records,
        test_records=test_records,
        puzzles=puzzles,
        legality_states=legality_states,
        seeds=seeds,
        d_model=64,
        num_layers=2,
        n_heads=4,
        d_ff=128,
        encoder_epochs=10,
        d3pm_epochs=18,
        games_per_match=40,
        device=device,
    )

    # Extract raw state_dict bundles as saved to disk
    ckpt_bundles: Dict[Tuple[str, int], Dict[str, Dict[str, torch.Tensor]]] = {}
    for (cfg_name, seed), agent in trained_agents.items():
        bundle = {"encoder": {k: v.detach().clone() for k, v in agent.encoder.state_dict().items()}}
        if agent.d3pm_planner is not None:
            bundle["d3pm_planner"] = {
                k: v.detach().clone() for k, v in agent.d3pm_planner.state_dict().items()
            }
        ckpt_bundles[(cfg_name, seed)] = bundle

    deviations, max_dev, all_ok = verify_clean_room_reproduction_from_checkpoints(
        checkpoint_bundles=ckpt_bundles,
        recorded_summary=report.config_summary,
        recorded_matches=report.match_results,
        recorded_leaderboard=report.bayeselo_leaderboard,
        test_records=test_records,
        puzzles=puzzles,
        legality_states=legality_states,
        seeds=seeds,
        d_model=64,
        num_layers=2,
        n_heads=4,
        d_ff=128,
        tolerance_pct=0.5,
        device=device,
    )

    assert len(deviations) >= 20
    assert max_dev <= 0.5
    assert all_ok
