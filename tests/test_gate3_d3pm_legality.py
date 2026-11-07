"""Stage-Gate 3 Verification Suite: D3PM Transition Matrix, 100% Move Legality, Mate-in-3 & Latency."""

from __future__ import annotations

import random
from typing import List
import torch

from topo_chess.data.pipeline import build_gardner_mate_in_3_benchmark
from topo_chess.diffusion.d3pm_planner import AbsorbingMaskSchedule, D3PMHorizonPlanner
from topo_chess.domains.boards import Gardner5x5State
from topo_chess.evaluation.metrics import evaluate_stage_gate_3, train_wp3_d3pm_on_puzzles
from topo_chess.models.ipa_encoder import TopographicalIPAEncoder


def test_sg3_d3pm_absorbing_markov_matrix_properties() -> None:
    """Verify Q_t is strictly row-stochastic and [MASK] is an absorbing state."""
    sched = AbsorbingMaskSchedule(num_edges=455, max_timesteps=32)
    for t in (0, 15, 31):
        qt = sched.build_transition_matrix_qt(t)
        assert qt.shape == (456, 456)
        row_sums = qt.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)
        # Absorbing state property: P(x_t = MASK | x_{t-1} = MASK) == 1.0
        assert float(qt[455, 455].item()) == 1.0
        assert float(qt[455, :455].sum().item()) == 0.0


def test_sg3_legality_tactical_coherence_horizon_and_latency() -> None:
    """SG3-A..D: Verify 100% move legality, >=85% mate-in-3 accuracy, <12% horizon drop, and >=5x speedup."""
    torch.manual_seed(42)
    rng = random.Random(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    puzzles = build_gardner_mate_in_3_benchmark(num_puzzles=60)
    train_puzzles = puzzles[:40]
    test_puzzles = puzzles[40:]

    # Generate random test positions for 6-ply legality verification
    legality_states: List[Gardner5x5State] = []
    st = Gardner5x5State.starting_state()
    for _ in range(200):
        legality_states.append(st)
        mvs = st.legal_moves()
        if not mvs or st.is_game_over():
            st = Gardner5x5State.starting_state()
        else:
            st = st.step(rng.choice(mvs))

    encoder = TopographicalIPAEncoder(
        height=5, width=5, d_model=128, num_layers=4, n_heads=4, d_ff=256
    )
    planner = D3PMHorizonPlanner(
        height=5,
        width=5,
        horizon_plies=6,
        d_model=128,
        num_layers=3,
        n_heads=4,
        d_ff=256,
        max_timesteps=32,
        aux_ce_lambda=0.1,
        rollout_mask_final_steps=3,
    )

    losses = train_wp3_d3pm_on_puzzles(
        encoder=encoder,
        planner=planner,
        puzzles=train_puzzles,
        epochs=60,
        batch_size=20,
        lr=3e-3,
        freeze_encoder=True,
        device=device,
    )
    assert losses[-1] < losses[0]

    metrics = evaluate_stage_gate_3(
        encoder=encoder,
        planner=planner,
        puzzles=test_puzzles,
        legality_test_states=legality_states,
        num_steps=20,
        device=device,
    )

    assert metrics.legal_trajectory_rate == 1.0, (
        f"SG3-A failed: legal_trajectory_rate={metrics.legal_trajectory_rate}"
    )
    assert metrics.mate_in_3_accuracy >= 0.85, (
        f"SG3-B failed: mate_in_3_accuracy={metrics.mate_in_3_accuracy:.2%}"
    )
    assert metrics.horizon_accuracy_drop < 0.12, (
        f"SG3-C failed: horizon_accuracy_drop={metrics.horizon_accuracy_drop:.2%}"
    )
    assert metrics.latency_speedup_ratio >= 5.0, (
        f"SG3-D failed: speedup={metrics.latency_speedup_ratio:.2f}x < 5.0x"
    )
