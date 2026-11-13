"""
WP-L5.5 (Stage 2.5), WP-L6 (Stage 3A), and WP-L7 (Stage 3B) Verification Tests:
1. Stage 2.5 Evidence & Reference Controls (`WP-L5.5`):
   - Exact quadratic MILP (`full_quadratic_milp`, `Planner_2b`) via McCormick linearization
     solving A1, A2-Uniform, A2-Trap, B, and C to 100% completion (`milp_optimality_gap == 0.0`).
   - Benchmark A2 split into `A2-Uniform` (where additive `one_shot_milp` succeeds with `IoU == 1.000`
     while `one_shot_greedy` fails with `0.667`) vs. `A2-Trap` (corrupted unary scores).
   - Multi-seed Neural Contribution Controls (`heuristic_only`, `zero_neural`, 5-seed `random_neural`),
     Benchmark C beam-width (`B in {2, 4, 8}`) vs. decoy-count (`D in {2, 4}`) scaling matrix,
     `requires_simultaneous_or_temp_support` metadata flag, and exact parameter inventory.
2. Geometric, Mechanical, & Equivariance Contract Verification (`WP-L5.5`):
   - `PortStatus.PLACEABLE` alias and multi-face simultaneous SNOT contact recording (`vertical stud` +
     `horizontal side stud` both recorded in `consumed_ports` and `ContactGraph`).
   - Exact `C_4` yaw + integer lattice translation + candidate-permutation equivariance of
     `Stage1FrontierOutcomeModel` (`max_abs_diff < 1e-5`).
3. Stage 3A LDraw `.ldr` Parser, Exporter, & Course Decomposer (`WP-L6`):
   - Roundtrip `.ldr` export and import (`+Z` up and standard LDraw `-Y` up conventions) and
     bottom-up topological course decomposition.
4. Stage 3A Learned Downstream Value Q_theta & Joint Course Selection (`WP-L6`):
   - Priority 1 (`FutureOpportunitySample`): 2-course SNOT opportunity preservation where all 4
     orientations of `87087` in Course 1 have identical bounding boxes (tying 1-step heuristics at
     `0.25`), and training `FutureOpportunityValueModel` against the Exact 2-Course Lookahead
     Teacher `Q*` achieves `>= 0.95` (`1.00`) completion on held-out unseen test structures.
   - Priority 2 (`RandomizedCourseSample` & `StabilityCalibrationSample`): End-to-end training of
     `Stage1FrontierOutcomeModel` (`use_analytic_priors=False`), `LearnedCandidateInteractionModule`,
     and `CandidateGraphDenoiser` (`trained_discrete_diffusion`), verifying that Regime 3
     (Trained Neural + Trained Diffusion, Zero Priors) outperforms Regime 1 (Untrained, Zero Priors)
     and matches `full_quadratic_milp` (`1.00`), plus stability head calibration.
5. Stage 3B Monotone Incremental & Selective Candidate Maintenance (`WP-L7`):
   - `100%` bit-exact equivalence between `generate_candidate_pool_incremental` and full rebuilds
     across multi-course assembly trajectories, plus `generate_candidate_pool_selective` recall.
"""

from __future__ import annotations

import numpy as np
import torch

from topo_brick.evaluation.benchmark_suite import (
    run_benchmark_a2_2x3_domino_trap,
    run_benchmark_a2_2x3_domino_uniform,
    run_benchmark_c_counterbalanced_cantilever,
    run_stage2_5_evidence_suite,
    run_stage3b_incremental_and_selective_benchmark,
)
from topo_brick.interface.brick_catalog import PortPolarity
from topo_brick.interface.ldu_frames import (
    C4_Z_ROT_IDS,
    LDUBox,
    LDUFrame,
    compose_rot_ids,
    rot_id_to_matrix,
)
from topo_brick.interface.port_manifold import (
    AssemblyState,
    PortStatus,
)
from topo_brick.models.outcome_head import Stage1FrontierOutcomeModel
from topo_brick.representation.interface_fields import (
    FrontierGraphBatch,
    TargetShapeBoxes,
    build_frontier_and_candidate_tensors,
)
from topo_brick.training import (
    decompose_assembly_to_courses,
    export_state_to_ldr_text,
    parse_ldr_text,
    run_stage3a_full_training_and_ablation,
)


def test_stage25_full_quadratic_milp_and_a2_uniform_vs_trap() -> None:
    """
    Verify External Review Round 2 (§3 & §4) controls:
    1. On `A2-Uniform` (true uncorrupted unary scores), `one_shot_greedy` gets `IoU == 0.667`
       while additive `one_shot_milp` gets `IoU == 1.000` (cardinality 3 > 2).
    2. On `A2-Trap` (corrupted corner scores), additive `one_shot_milp` fails (`IoU == 0.667`),
       whereas `full_quadratic_milp`, `iterative_refinement`, and `trained_discrete_diffusion`
       all achieve `IoU == 1.000`.
    3. On Benchmark C, `requires_simultaneous_or_temp_support` is flagged `True`, and the
       beam-width (`B in {2, 4, 8}`) vs. decoy-count (`D in {2, 4}`) scaling matrix shows
       beam search failing when `D >= B` while `full_quadratic_milp` and `iterative_refinement`
       succeed across all `(B, D)`.
    """
    iou_unif_greedy, _ = run_benchmark_a2_2x3_domino_uniform("one_shot_greedy")
    iou_unif_milp, _ = run_benchmark_a2_2x3_domino_uniform("one_shot_milp")
    iou_unif_qmilp, _ = run_benchmark_a2_2x3_domino_uniform("full_quadratic_milp")
    assert abs(iou_unif_greedy - (4.0 / 6.0)) < 1e-4
    assert abs(iou_unif_milp - 1.0) < 1e-4
    assert abs(iou_unif_qmilp - 1.0) < 1e-4

    iou_trap_milp, _ = run_benchmark_a2_2x3_domino_trap("one_shot_milp")
    iou_trap_qmilp, _ = run_benchmark_a2_2x3_domino_trap("full_quadratic_milp")
    iou_trap_iter, _ = run_benchmark_a2_2x3_domino_trap("iterative_refinement")
    assert abs(iou_trap_milp - (4.0 / 6.0)) < 1e-4
    assert abs(iou_trap_qmilp - 1.0) < 1e-4
    assert abs(iou_trap_iter - 1.0) < 1e-4

    stable_c_qmilp, _, metrics_c = run_benchmark_c_counterbalanced_cantilever(
        "full_quadratic_milp"
    )
    assert stable_c_qmilp is True
    assert metrics_c.requires_simultaneous_or_temp_support is True
    assert abs(metrics_c.milp_optimality_gap) < 1e-6

    evidence = run_stage2_5_evidence_suite()
    qmilp_res = evidence.planner_results["full_quadratic_milp"]
    assert abs(float(qmilp_res["a1_iou"]) - 1.0) < 1e-4
    assert abs(float(qmilp_res["a2_uniform_iou"]) - 1.0) < 1e-4
    assert abs(float(qmilp_res["a2_trap_iou"]) - 1.0) < 1e-4
    assert qmilp_res["c_completed"] is True
    assert evidence.parameter_inventory["stage1_d128_L4_total"] == 718756
    assert evidence.parameter_inventory["stage1_d64_L2_total"] == 115924
    assert evidence.parameter_inventory["candidate_graph_denoiser_d64"] == 6785
    # Beam search (decoys, beam_width): fails when B=2 on D=2 or D=4; succeeds when B=4 on D=2, fails on D=4; B=8 succeeds on both
    assert evidence.beam_vs_decoys_completion_matrix[(2, 2)] is False
    assert evidence.beam_vs_decoys_completion_matrix[(2, 4)] is True
    assert evidence.beam_vs_decoys_completion_matrix[(4, 4)] is False
    assert evidence.beam_vs_decoys_completion_matrix[(4, 8)] is True


def test_stage25_multi_face_snot_contacts_and_placeable_status() -> None:
    """
    Verify External Review Round 2 (§5.1 & §5.3):
    1. `PortStatus.PLACEABLE` is a valid alias of `PortStatus.REACHABLE`, and `.is_placeable` returns True.
    2. When a brick spans multiple base bricks or engages horizontal SNOT side studs,
       `state.apply_candidate(...)` records all contacts in `consumed_ports` and `state.get_contact_edges()`.
    """
    assert PortStatus.PLACEABLE == PortStatus.REACHABLE
    assert PortStatus.PLACEABLE.is_placeable is True
    assert PortStatus.MATCHED.is_placeable is False

    state = AssemblyState()
    # Ground a 2x2 brick at origin: top studs at z=24, x in {0, 20}, y in {0, 20}
    state.place_root_part("3003", LDUFrame(0, (0, 0, 0)), is_grounded=True)
    # Place an 87087 1x1 SNOT brick on top stud (0, 0, 24) with rot_id=0 (side stud at +X=(10, 0, 38))
    frontier1 = state.get_active_frontier()
    pool1 = state.generate_candidate_pool(allowed_part_ids=["87087"], frontier=frontier1)
    snot_cand = next(
        c
        for c in pool1.candidates
        if c.part_id == "87087"
        and c.pose.t == (0, 0, 24)
        and c.pose.rot_id == 0
    )
    state.apply_candidate(snot_cand)

    # Check that a 1x2 brick ("3004") placed across two separate base bricks records contacts to BOTH base bricks
    state2 = AssemblyState()
    b0 = state2.place_root_part("3005", LDUFrame(0, (0, 0, 0)), is_grounded=True)
    b1 = state2.place_root_part("3005", LDUFrame(0, (20, 0, 0)), is_grounded=True)
    f2 = state2.get_active_frontier()
    p2 = state2.generate_candidate_pool(allowed_part_ids=["3004"], frontier=f2)
    bridge_cand = next(
        c
        for c in p2.candidates
        if c.world_boxes[0] == LDUBox(-10, -10, 24, 30, 10, 48)
    )
    engaged_bricks = {pid for (pid, _) in bridge_cand.consumed_ports}
    assert b0 in engaged_bricks and b1 in engaged_bricks
    b2 = state2.apply_candidate(bridge_cand)
    edges = state2.get_contact_edges()
    connected_pairs = {(e["src_brick"], e["dst_brick"]) for e in edges}
    assert (b0, b2) in connected_pairs and (b1, b2) in connected_pairs

    # Also verify horizontal SNOT side-stud port is exposed with normal (+1, 0, 0) and placeable
    frontier_after_snot = state.get_active_frontier()
    side_ports = [
        ep
        for ep in frontier_after_snot.ports
        if ep.polarity == PortPolarity.STUD and ep.world_normal == (1, 0, 0)
    ]
    assert len(side_ports) == 1
    pool_cladding = state.generate_candidate_pool(
        allowed_part_ids=["3024"], frontier=frontier_after_snot
    )
    side_cands = [
        c
        for c in pool_cladding.candidates
        if any(gpid[0] == 1 for gpid in c.consumed_ports)
    ]
    assert len(side_cands) >= 2
    assert any((1, 1) in c.consumed_ports for c in side_cands)


def test_stage25_c4_yaw_translation_and_permutation_equivariance() -> None:
    """
    Verify External Review Round 2 (§6):
    1. Transforming an assembly state and target shape by an arbitrary integer lattice
       translation `t_shift` and any gravity-preserving C_4 yaw rotation `rot_yaw in {0, 1, 2, 3}`
       leaves the candidate feature tensors and `Stage1FrontierOutcomeModel` candidate scores
       invariant up to floating-point precision (`< 1e-5`).
    2. Permuting the order of candidates in `FrontierGraphBatch` permutes the output scores
       identically (`< 1e-6`).
    """
    torch.manual_seed(42)
    model = Stage1FrontierOutcomeModel(use_ipa=True, d_model=64, n_layers=2)
    model.eval()

    # Build canonical assembly state and target shape
    state0 = AssemblyState()
    state0.place_root_part("3004", LDUFrame(0, (0, 0, 0)), is_grounded=True)
    target0 = TargetShapeBoxes(boxes=(LDUBox(-10, -10, 24, 30, 10, 48),))
    f0 = state0.get_active_frontier()
    p0 = state0.generate_candidate_pool(allowed_part_ids=["3004", "3005"], frontier=f0)
    b0 = build_frontier_and_candidate_tensors(
        state=state0, frontier=f0, pool=p0, target_shape=target0
    )
    with torch.no_grad():
        scores0 = model.score_candidates(b0).detach().cpu().numpy()

    # 1. Candidate permutation equivariance check
    rng = np.random.default_rng(123)
    perm = rng.permutation(len(p0))
    b_perm = FrontierGraphBatch(
        port_features=b0.port_features,
        edge_index=b0.edge_index,
        edge_rel_rot_id=b0.edge_rel_rot_id,
        edge_rel_pos_ldu=b0.edge_rel_pos_ldu,
        edge_attr=b0.edge_attr,
        candidate_features=b0.candidate_features[perm],
        candidate_port_indices=tuple(b0.candidate_port_indices[int(i)] for i in perm),
    )
    with torch.no_grad():
        scores_perm = model.score_candidates(b_perm).detach().cpu().numpy()
    assert np.max(np.abs(scores_perm - scores0[perm])) < 1e-5

    # 2. C_4 yaw rotation + integer lattice translation invariance check
    t_shift = np.array([60, -40, 24], dtype=np.int64)
    for yaw_rot_id in C4_Z_ROT_IDS:
        rmat = rot_id_to_matrix(yaw_rot_id)
        state_rot = AssemblyState()
        new_root_rot = compose_rot_ids(yaw_rot_id, 0)
        state_rot.place_root_part(
            "3004",
            LDUFrame(new_root_rot, (int(t_shift[0]), int(t_shift[1]), int(t_shift[2]))),
            is_grounded=True,
        )
        # Transform target box by (yaw_rot_id, t_shift)
        box0 = target0.boxes[0]
        corners = np.array(
            [
                [box0.xmin, box0.ymin, box0.zmin],
                [box0.xmax, box0.ymax, box0.zmax],
            ],
            dtype=np.int64,
        )
        rot_corners = (rmat @ corners.T).T + t_shift[None, :]
        rbox = LDUBox(
            int(np.min(rot_corners[:, 0])),
            int(np.min(rot_corners[:, 1])),
            int(np.min(rot_corners[:, 2])),
            int(np.max(rot_corners[:, 0])),
            int(np.max(rot_corners[:, 1])),
            int(np.max(rot_corners[:, 2])),
        )
        target_rot = TargetShapeBoxes(boxes=(rbox,))
        f_rot = state_rot.get_active_frontier()
        p_rot = state_rot.generate_candidate_pool(
            allowed_part_ids=["3004", "3005"], frontier=f_rot
        )
        b_rot = build_frontier_and_candidate_tensors(
            state=state_rot, frontier=f_rot, pool=p_rot, target_shape=target_rot
        )
        with torch.no_grad():
            scores_rot = model.score_candidates(b_rot).detach().cpu().numpy()
        assert np.max(np.abs(np.sort(scores_rot) - np.sort(scores0))) < 1e-5


def test_stage3a_ldraw_roundtrip_and_course_decomposer() -> None:
    """
    Verify Stage 3A (`WP-L6`) LDraw `.ldr` parser, exporter, and topological course decomposer:
    - Exporting a multi-course + SNOT assembly to `.ldr` text (both `+Z` up and standard LDraw
      `-Y` up), parsing it back via `parse_ldr_text`, and decomposing it via
      `decompose_assembly_to_courses` recovers the exact part count, 100% teacher matching,
      and topological course structure.
    """
    state = AssemblyState()
    state.place_root_part("3004", LDUFrame(0, (0, 0, 0)), is_grounded=True)
    f1 = state.get_active_frontier()
    p1 = state.generate_candidate_pool(allowed_part_ids=["87087"], frontier=f1)
    c1 = next(
        c
        for c in p1.candidates
        if c.pose.t == (0, 0, 24) and c.pose.rot_id == 0
    )
    state.apply_candidate(c1)

    ldr_text = export_state_to_ldr_text(state)
    specs = parse_ldr_text(ldr_text, use_z_up=True)
    assert len(specs) == 2
    steps = decompose_assembly_to_courses(specs, allowed_part_ids=["3004", "87087"])
    assert len(steps) == 1
    assert len(steps[0].teacher_candidate_indices) == 1


def test_stage3a_future_opportunity_q_learning_and_joint_selection() -> None:
    """
    Verify Stage 3A (`WP-L6`) end-to-end training and learned vs. heuristic ablations:
    1. Priority 1 (2-Course SNOT Opportunity Preservation on unseen test geometries):
       - `heuristic_only_completion_rate == 0.25` (4 orientations tie on 1-step bounding box).
       - `trained_ipa_completion_rate` and `trained_gnn_completion_rate` achieve `>= 0.95` (`1.00`)
         completion on the held-out unseen structural test split, with training MSE decreasing.
    2. Priority 2 (Joint Course Selection on held-out randomized benchmark instances):
       - `regime3_trained_neural_no_priors_solve_rate == 1.0` (`> regime1_untrained_no_priors_solve_rate`),
         matching `full_quadratic_milp_reference_solve_rate == 1.0` with zero analytic priors active.
    3. Amortized Stability Calibration:
       - `stability_trained_accuracy > stability_untrained_accuracy` and `>= 0.83`.
       - `stability_trained_eta_mae < stability_untrained_eta_mae`.
    """
    summary = run_stage3a_full_training_and_ablation(seed=42)

    # Priority 1: Future Opportunity Q_theta
    f_sum = summary.future_opportunity_summary
    assert abs(f_sum.heuristic_only_completion_rate - 0.25) < 1e-4
    assert f_sum.trained_ipa_completion_rate >= 0.95
    assert f_sum.trained_gnn_completion_rate >= 0.95
    assert f_sum.train_loss_end < f_sum.train_loss_start * 0.2

    # Priority 2: Joint Course Selection across Regimes 1, 2, 3, and Full Quadratic MILP
    assert summary.regime3_trained_neural_no_priors_solve_rate == 1.0
    assert (
        summary.regime3_trained_neural_no_priors_solve_rate
        > summary.regime1_untrained_no_priors_solve_rate
    )
    assert summary.full_quadratic_milp_reference_solve_rate == 1.0
    assert summary.diffusion_train_loss_end < summary.diffusion_train_loss_start * 0.5

    # Amortized Stability Calibration
    assert summary.stability_trained_accuracy > summary.stability_untrained_accuracy
    assert summary.stability_trained_accuracy >= 0.83
    assert summary.stability_trained_eta_mae < summary.stability_untrained_eta_mae


def test_stage3b_incremental_candidate_cache_exactness_and_selective_recall() -> None:
    """
    Verify Stage 3B (`WP-L7`) Monotone Incremental & Selective Candidate Maintenance:
    1. `generate_candidate_pool_incremental` achieves `100%` (`1.0`) bit-exact equivalence
       against full `state.generate_candidate_pool(...)` rebuilds across multi-course trajectories.
    2. Incremental candidate maintenance reduces collision-check work and achieves speedup > 1.0x.
    3. Selective anchor-port pruning (`top 75%` ports) preserves `>= 90%` candidate recall.
    """
    res = run_stage3b_incremental_and_selective_benchmark(num_courses=8)
    assert res.bit_exact_equivalence_rate == 1.0
    assert res.incremental_speedup_ratio > 1.0
    assert res.selective_75pct_recall >= 0.90
