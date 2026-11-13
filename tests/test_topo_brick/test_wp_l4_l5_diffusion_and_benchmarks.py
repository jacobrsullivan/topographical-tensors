"""
WP-L4 & WP-L5 Verification Tests:
1. Candidate Interaction Graph (`G_cand`) construction:
   - Unary orphan-port penalty under restricted inventory (`1x6` span without `1x1` bricks).
   - Pure pairwise domino-parity trap (`2x3` grid with `1x2` bricks where `unary_orphan == 0`
     for every single candidate, but incompatible corner pairs have `W_{AB} < 0`).
   - Cross-track seam-alignment penalty vs. staggered seam bonus.
   - Counter-balancing cantilever torque synergy (`W_{overhang, counter} > 0`).
2. Four-Planner Controlled Comparison across Benchmarks A1, A2, B, and C:
   - `one_shot_greedy` (`Planner_1`)
   - `one_shot_milp` (`Planner_2`, exact additive HiGHS MILP)
   - `autoregressive_beam` (`Planner_3`, beam search B=4)
   - `binary_diffusion` (`Planner_4` / `Config_D_Brick`, S=6 binary candidate-pool diffusion)
"""

from __future__ import annotations

import numpy as np

from topo_brick.evaluation.benchmark_suite import (
    PLANNER_TYPES,
    run_benchmark_a1_1x6_parity_packing,
    run_benchmark_a2_2x3_domino_trap,
    run_benchmark_b_seam_staggering,
    run_benchmark_c_counterbalanced_cantilever,
    run_stage2_full_comparison,
)
from topo_brick.interface.ldu_frames import LDUBox, LDUFrame
from topo_brick.interface.port_manifold import AssemblyState
from topo_brick.models.motif_diffusion import (
    CandidateGraphDenoiser,
    build_candidate_interaction_graph,
)
from topo_brick.representation.interface_fields import (
    TargetShapeBoxes,
    build_frontier_and_candidate_tensors,
)


def test_candidate_interaction_graph_2x3_domino_pairwise_trap() -> None:
    """
    Verify that on a 2x3 target grid with only 1x2 bricks (`["3004"]`), EVERY valid
    in-target 1x2 candidate has zero unary orphan penalty (`unary_orphan == 0`),
    because any single 1x2 brick on a 2x3 grid can still be completed to 100% tiling.
    However, selecting the two opposite vertical corner 1x2 bricks together leaves two
    isolated 1x1 corners, which `build_candidate_interaction_graph` detects via a strictly
    negative pairwise synergy entry `W_{A, B} < -5.0`.
    """
    state = AssemblyState()
    state.place_root_part("3002", LDUFrame(0, (0, 0, 0)), is_grounded=True)  # 2x3 base
    target_shape = TargetShapeBoxes(boxes=(LDUBox(-10, -10, 24, 30, 50, 48),))

    frontier = state.get_active_frontier()
    pool = state.generate_candidate_pool(allowed_part_ids=["3004"], frontier=frontier)
    batch = build_frontier_and_candidate_tensors(
        state=state, frontier=frontier, pool=pool, target_shape=target_shape
    )
    interaction = build_candidate_interaction_graph(
        state=state,
        frontier=frontier,
        pool=pool,
        batch=batch,
        target_shape=target_shape,
        allowed_part_ids=["3004"],
    )

    in_target_cids = [
        cid for cid in range(len(pool)) if batch.candidate_features[cid, 13] >= 0.99
    ]
    assert len(in_target_cids) == 7  # 7 distinct 1x2 domino placements on a 2x3 grid
    for cid in in_target_cids:
        assert interaction.unary_orphan_penalty[cid] == 0.0

    # Find opposite vertical corner 1x2 candidates A and B
    cid_a = -1
    cid_b = -1
    cid_parallel_partner = -1
    for cid in in_target_cids:
        b = pool.candidates[cid].world_boxes[0]
        if (b.xmin, b.ymin, b.xmax, b.ymax) == (-10, -10, 10, 30):
            cid_a = cid
        elif (b.xmin, b.ymin, b.xmax, b.ymax) == (10, 10, 30, 50):
            cid_b = cid
        elif (b.xmin, b.ymin, b.xmax, b.ymax) == (10, -10, 30, 30):
            cid_parallel_partner = cid

    assert cid_a >= 0 and cid_b >= 0 and cid_parallel_partner >= 0
    # Pair (A, B) strands two 1x1 corners -> strongly negative pairwise interaction W_{AB}
    assert interaction.synergy_matrix[cid_a, cid_b] < -5.0
    # Pair (A, parallel_partner) touches flush and leaves a fillable 1x2 strip -> positive synergy
    assert interaction.synergy_matrix[cid_a, cid_parallel_partner] > 0.0


def test_benchmark_a1_and_a2_parity_and_domino_trap() -> None:
    """
    Verify Benchmark A1 (1x6 parity packing) and Benchmark A2 (2x3 domino trap):
    - On A1, `one_shot_greedy` picks the center 1x4 brick first and strands two 1x1 end studs
      (`IoU == 4/6 == 0.667`), whereas `binary_diffusion` achieves `IoU == 1.000`.
    - On A2, because `unary_orphan == 0` for every candidate, BOTH `one_shot_greedy` AND
      `one_shot_milp` fall into the pairwise domino trap (`IoU == 4/6 == 0.667`), whereas
      `binary_diffusion` resolves the pairwise conflict/synergy graph to achieve `IoU == 1.000`.
    """
    iou_a1_greedy, _ = run_benchmark_a1_1x6_parity_packing("one_shot_greedy")
    iou_a1_diff, _ = run_benchmark_a1_1x6_parity_packing("binary_diffusion")
    assert abs(iou_a1_greedy - (4.0 / 6.0)) < 1e-4
    assert abs(iou_a1_diff - 1.0) < 1e-4

    iou_a2_greedy, _ = run_benchmark_a2_2x3_domino_trap("one_shot_greedy")
    iou_a2_milp, _ = run_benchmark_a2_2x3_domino_trap("one_shot_milp")
    iou_a2_diff, _ = run_benchmark_a2_2x3_domino_trap("binary_diffusion")

    assert abs(iou_a2_greedy - (4.0 / 6.0)) < 1e-4
    assert abs(iou_a2_milp - (4.0 / 6.0)) < 1e-4
    assert abs(iou_a2_diff - 1.0) < 1e-4


def test_benchmark_b_seam_staggering_and_c_counterbalanced_cantilever() -> None:
    """
    Verify Benchmark B (cross-track seam staggering) and Benchmark C (counter-balanced cantilever):
    - On Benchmark B, `binary_diffusion` achieves a strictly higher quadratic course objective
      and higher biconnected structural ratio than unary `one_shot_greedy` and `one_shot_milp`.
    - On Benchmark C, unary `one_shot_greedy` and `one_shot_milp` pair the overhang with the
      higher-unary decoy, which fails static equilibrium and backs off (`completed == False`),
      whereas `binary_diffusion` selects the counter-balancing pair (`completed == True`) with
      stable static equilibrium (`peak_utilization <= 1.0`).
    """
    obj_greedy, bic_greedy, _ = run_benchmark_b_seam_staggering("one_shot_greedy")
    obj_milp, bic_milp, _ = run_benchmark_b_seam_staggering("one_shot_milp")
    obj_diff, bic_diff, _ = run_benchmark_b_seam_staggering("binary_diffusion")

    assert obj_diff > obj_greedy + 1.0
    assert obj_diff > obj_milp + 1.0
    assert bic_diff >= bic_greedy

    comp_greedy, _, m_greedy = run_benchmark_c_counterbalanced_cantilever("one_shot_greedy")
    comp_milp, _, m_milp = run_benchmark_c_counterbalanced_cantilever("one_shot_milp")
    comp_beam, _, m_beam = run_benchmark_c_counterbalanced_cantilever("autoregressive_beam")
    comp_diff, eta_diff, m_diff = run_benchmark_c_counterbalanced_cantilever("binary_diffusion")

    assert not comp_greedy
    assert not comp_milp
    assert not comp_beam
    assert m_greedy.prefix_backoff_steps >= 1
    assert m_milp.prefix_backoff_steps >= 1
    assert m_beam.prefix_backoff_steps >= 1

    assert comp_diff
    assert m_diff.prefix_backoff_steps == 0
    assert m_diff.equilibrium_result.is_stable
    assert eta_diff <= 1.0


def test_stage2_full_four_planner_suite_and_denoiser_gradients() -> None:
    """
    Run the full 4-planner comparison (`run_stage2_full_comparison`) and verify that
    `CandidateGraphDenoiser` supports end-to-end gradient backpropagation.
    """
    results = run_stage2_full_comparison(seed=42)
    assert len(results) == len(PLANNER_TYPES)
    by_name = {r.planner_name: r for r in results}

    diff_res = by_name["binary_diffusion"]
    assert abs(diff_res.benchmark_a1_1x6_parity_iou - 1.0) < 1e-4
    assert abs(diff_res.benchmark_a2_2x3_domino_trap_iou - 1.0) < 1e-4
    assert diff_res.benchmark_c_cantilever_completed

    # Verify CandidateGraphDenoiser backward pass
    import torch

    denoiser = CandidateGraphDenoiser(d_cand=18, d_hidden=32)
    cand_x = torch.randn(10, 18, requires_grad=True)
    probs = torch.rand(10)
    syn = torch.randn(10)
    conf = torch.rand(10)
    out = denoiser(cand_x, probs, syn, conf, step_idx=3)
    assert out.shape == (10,)
    loss = out.pow(2).sum()
    loss.backward()
    assert cand_x.grad is not None
    assert float(np.abs(cand_x.grad.detach().numpy()).sum()) > 0.0
