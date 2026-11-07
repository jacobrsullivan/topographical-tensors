"""Stage-Gate 4 Verification Tests: WP4 Factorial Ablation Matrix, BayesElo & Paired t-Tests."""

from __future__ import annotations

import torch

from topo_chess.data.pipeline import (
    build_gardner_mate_in_3_benchmark,
    generate_synthetic_gardner_games,
    partition_and_deduplicate_games,
)
from topo_chess.domains.boards import Gardner5x5State
from topo_chess.evaluation.gauntlet_and_bayeselo import (
    MatchResult,
    compute_bayeselo_ratings,
    run_wp4_factorial_ablation_and_gauntlet,
)
from topo_chess.models.ablation_configs import (
    AblationConfigAgent,
    BitboardGlobalTransformer,
    ChessFormerTokenShawEncoder,
)
from topo_chess.models.ipa_encoder import TopographicalIPAEncoder


def test_ablation_configs_parameter_and_flops_hierarchy() -> None:
    """Verify Configs A, B, C, D instantiate cleanly at 8x8 CF-6M scale and respect FLOPS hierarchy."""
    enc_a = BitboardGlobalTransformer(
        height=8, width=8, d_model=256, num_layers=8, n_heads=8, d_ff=832
    )
    enc_b = ChessFormerTokenShawEncoder(
        height=8, width=8, d_model=256, num_layers=8, n_heads=8, d_ff=832
    )
    enc_c = TopographicalIPAEncoder(
        height=8, width=8, d_model=256, num_layers=8, n_heads=8, d_ff=768
    )

    params_a = enc_a.count_parameters()
    params_b = enc_b.count_parameters()
    params_c = enc_c.count_parameters()

    assert 5_000_000 <= params_a <= 6_500_000
    assert 5_000_000 <= params_b <= 6_500_000
    assert 5_000_000 <= params_c <= 6_500_000

    flops_a = enc_a.compute_forward_flops()
    flops_b = enc_b.compute_forward_flops()
    flops_c = enc_c.compute_forward_flops()

    # Sparse SE(2) IPA (Config_C/D) must use at least 15% fewer spatial FLOPS than dense baselines (Config_A/B)
    assert flops_c <= 0.85 * flops_a
    assert flops_c <= 0.85 * flops_b


def test_adaptive_entropy_scheduler_fb_wp4() -> None:
    """Verify FB-WP4 Adaptive Entropy Step Scheduler selects 4 steps in quiet states and 20 in high-entropy states."""
    enc = TopographicalIPAEncoder(height=5, width=5, d_model=64, num_layers=2, n_heads=4, d_ff=128)
    agent = AblationConfigAgent(
        config_name="Config_D",
        seed=42,
        encoder=enc,
        entropy_threshold=0.85,
        quiet_steps=4,
        tactical_steps=20,
    )

    legal_mask = torch.zeros((2, 455), dtype=torch.bool)
    legal_mask[:, :10] = True

    # Peaked / forced single move -> low entropy -> 4 steps
    peaked_logits = torch.full((2, 455), -10.0)
    peaked_logits[:, 0] = 15.0
    assert agent.compute_adaptive_steps(peaked_logits, legal_mask) == 4

    # Flat / multi-candidate tactical state -> high entropy -> 20 steps
    flat_logits = torch.zeros((2, 455))
    assert agent.compute_adaptive_steps(flat_logits, legal_mask) == 20


def test_bayeselo_solver_and_confidence_intervals() -> None:
    """Verify exact BayesElo solver produces monotone ratings and valid 95% confidence intervals."""
    matches = [
        MatchResult("Config_D", "Config_C", wins_a=640, draws=200, losses_a=160),
        MatchResult("Config_C", "Config_B", wins_a=580, draws=220, losses_a=200),
        MatchResult("Config_B", "Config_A", wins_a=590, draws=210, losses_a=200),
        MatchResult("Config_D", "Config_B", wins_a=720, draws=160, losses_a=120),
        MatchResult("Config_D", "Config_A", wins_a=800, draws=120, losses_a=80),
    ]
    board = compute_bayeselo_ratings(matches, anchor_agent="Config_B", anchor_elo=2190.0)
    by_name = {e.agent_name: e for e in board}

    assert by_name["Config_D"].elo > by_name["Config_C"].elo > by_name["Config_B"].elo > by_name["Config_A"].elo
    assert abs(by_name["Config_B"].elo - 2190.0) < 1e-4
    assert (by_name["Config_D"].elo - by_name["Config_C"].elo) >= 80.0
    for entry in board:
        assert entry.ci95_lower < entry.elo < entry.ci95_upper


def test_stage_gate_4_end_to_end_multi_seed_ablation() -> None:
    """Verify 3-seed factorial ablation across Configs A, B, C, D passes Stage-Gate 4 thresholds."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    games = generate_synthetic_gardner_games(num_games=36, max_plies=12, seed=42)
    split = partition_and_deduplicate_games(games)
    all_records = split.train + split.val + split.test
    train_records = all_records[:160]
    test_records = all_records[160:210]

    puzzles = build_gardner_mate_in_3_benchmark(num_puzzles=40)
    legality_states = [Gardner5x5State.from_fen(r.fen) for r in all_records[:60]]

    report, _ = run_wp4_factorial_ablation_and_gauntlet(
        train_records=train_records,
        test_records=test_records,
        puzzles=puzzles,
        legality_states=legality_states,
        seeds=(42, 1337, 2026),
        d_model=64,
        num_layers=2,
        n_heads=4,
        d_ff=128,
        encoder_epochs=18,
        d3pm_epochs=30,
        games_per_match=80,
        device=device,
    )

    assert report.sg4_a_statistical_significance_pass
    assert report.sg4_b_tactical_elo_gain_d_vs_c >= 80.0
    assert report.sg4_b_pass
    assert report.sg4_c_config_d_blitz_elo >= 2300.0
    assert report.sg4_c_pass
    assert report.stage_gate_4_overall_pass
