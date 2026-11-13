"""
Verification Test Suite for Stage 3.5 (WP-L7.5a-d) and Stage 4 (WP-L8 & WP-L9).

Tests:
1. `test_wp_l75a_exact_two_course_teacher_and_non_myopic_trap_plus_5seed_opportunity`
2. `test_wp_l75b_strategy_mixing_inclusion_vs_conditional_and_joint_values`
3. `test_wp_l75c_paired_d3pm_denoising_vs_min_over_modes_and_s1_ablation`
4. `test_wp_l75d_stability_boundary_ridge_and_four_way_candidate_engine`
5. `test_wp_l8a_and_l8b_ldraw_strict_contract_and_integral_grid_equivalence`
6. `test_wp_l8c_and_l9_stage4_sequential_complete_build_experiment`
"""

from __future__ import annotations

from topo_brick.interface.ldu_frames import LDUBox
from topo_brick.representation.interface_fields import TargetShapeBoxes
from topo_brick.training.complete_build_benchmark import (
    run_stage4_sequential_complete_build_study,
)
from topo_brick.training.ldraw_decomposer import parse_ldr_text_with_report
from topo_brick.training.stage35_tightening import (
    build_non_myopic_two_course_trap,
    run_four_way_candidate_engine_study,
    run_paired_d3pm_denoising_ablation,
    run_stability_boundary_and_ridge_study,
    run_strategy_mixing_value_study,
    run_wp_l75a_opportunity_and_teacher_study,
    solve_exact_two_course_teacher,
)


def test_wp_l75a_exact_two_course_teacher_and_non_myopic_trap_plus_5seed_opportunity() -> None:
    # 1. Verify exact 2-course enumeration teacher on deliberate non-myopic trap
    state, target_shape, root_ids, inv = build_non_myopic_two_course_trap()
    sol = solve_exact_two_course_teacher(
        state=state,
        target_shape=target_shape,
        root_ids=root_ids,
        allowed_parts_c1=["3010", "87087"],
        allowed_parts_c2=["3010", "3024"],
        max_bricks_c1=1,
        max_bricks_c2=2,
        inventory_counts=inv,
    )
    assert sol.is_strictly_non_myopic is True
    assert sol.optimal_course1_indices != sol.myopic_course1_indices
    assert sol.v2_optimal_iou > sol.myopic_two_course_iou + 0.10

    # 2. Verify 5-seed opportunity & continuation-count comparator study across 32 unseen instances
    summary = run_wp_l75a_opportunity_and_teacher_study()
    assert summary.non_myopic_trap_verified is True
    assert abs(summary.heuristic_1step_rate - 0.25) < 1e-6
    assert summary.continuation_count_snot_only_rate >= 0.99
    assert summary.continuation_count_trap_rate <= 0.55
    assert summary.trained_gnn_5seed_mean_rate >= 0.95
    assert summary.trained_ipa_5seed_mean_rate >= 0.95
    assert summary.trained_gnn_ci95[0] >= 0.90
    assert len(summary.seed_by_instance_matrix_gnn) == 5
    assert len(summary.seed_by_instance_matrix_gnn[0]) == 32


def test_wp_l75b_strategy_mixing_inclusion_vs_conditional_and_joint_values() -> None:
    summary = run_strategy_mixing_value_study()
    assert summary.marginal_q2_incl_values == (10.0, 10.0, 10.0, 10.0)
    assert summary.additive_q2_incl_coherent_rate <= 0.70
    assert summary.conditional_partial_q_coherent_rate == 1.0
    assert summary.whole_course_set_v_coherent_rate == 1.0
    assert summary.joint_pairwise_w_coherent_rate == 1.0


def test_wp_l75c_paired_d3pm_denoising_vs_min_over_modes_and_s1_ablation() -> None:
    summary = run_paired_d3pm_denoising_ablation()
    assert summary.paired_d3pm_valid_course_rate_by_steps[4] == 1.0
    assert summary.paired_d3pm_valid_course_rate_by_steps[6] == 1.0
    assert summary.paired_d3pm_conditioned_mode_recall == 1.0
    assert summary.paired_d3pm_mode_diversity_ratio == 1.0
    assert summary.paired_d3pm_conditioned_mode_recall > summary.state_only_s1_conditioned_mode_recall


def test_wp_l75d_stability_boundary_ridge_and_four_way_candidate_engine() -> None:
    stab = run_stability_boundary_and_ridge_study()
    assert stab.trained_overall_acc >= stab.ridge_overall_acc
    assert stab.trained_eta_mae < stab.ridge_eta_mae
    assert stab.trained_eta_mae < stab.untrained_eta_mae
    assert stab.accepted_unstable_rate_with_exact_gate == 0.0

    cand_eng = run_four_way_candidate_engine_study(port_keep_ratio=0.50)
    assert cand_eng.incremental_bit_exact_rate == 1.0
    assert cand_eng.incremental_mode.preserved_objective_gap_delta_j == 0.0
    assert cand_eng.selective_mode.setup_brick_retention_rate == 1.0
    assert cand_eng.incremental_selective_mode.setup_brick_retention_rate == 1.0
    assert cand_eng.selective_mode.speedup_vs_full > 1.0
    assert cand_eng.incremental_selective_mode.speedup_vs_full > 1.0


def test_wp_l8a_and_l8b_ldraw_strict_contract_and_integral_grid_equivalence() -> None:
    # 1. WP-L8a LDraw subset validation contract
    ldr_sample = "\n".join(
        [
            "0 Test LDraw Contract",
            "1 15 0 0 0 1 0 0 0 1 0 0 0 1 3001.dat",         # line 2: valid
            "1 15 0 0 24 1 0 0 0 1 0 0 0 1 99999.dat",        # line 3: unknown part
            "1 15 10.5 0 24 1 0 0 0 1 0 0 0 1 3004.dat",      # line 4: fractional LDU coord
            "1 15 0 0 24 2 0 0 0 1 0 0 0 1 3004.dat",         # line 5: scaled/non-cubic rot
        ]
    )
    rep = parse_ldr_text_with_report(ldr_sample, use_z_up=True, strict=False)
    assert rep.total_type1_lines == 4
    assert len(rep.accepted_specs) == 1
    assert rep.rejected_unknown_part_lines == (3,)
    assert rep.rejected_off_lattice_coord_lines == (4,)
    assert rep.rejected_non_cubic_rotation_lines == (5,)

    # 2. WP-L8b IntegralTargetGrid O(1) bit-exact equivalence with pairwise reference
    boxes = [
        LDUBox(0, 0, 0, 40, 20, 24),
        LDUBox(40, 0, 0, 80, 40, 48),
        LDUBox(-20, -20, 24, 0, 20, 48),
        LDUBox(10, 10, 48, 30, 30, 56),
    ]
    t_shape = TargetShapeBoxes(boxes=boxes)
    assert t_shape.integral_grid.can_use_grid is True

    queries = [
        [LDUBox(-10, -10, 0, 50, 30, 36)],
        [LDUBox(0, 0, 0, 80, 40, 56), LDUBox(-30, -30, 12, 20, 20, 48)],
        [LDUBox(12, 12, 48, 28, 28, 56)],
    ]
    for q_list in queries:
        v_grid = t_shape.intersection_volume(q_list)
        v_ref = t_shape.intersection_volume_pairwise_reference(q_list)
        assert v_grid == v_ref
        assert v_grid > 0


def test_wp_l8c_and_l9_stage4_sequential_complete_build_experiment() -> None:
    summary = run_stage4_sequential_complete_build_study()

    # Step 1: 5-Planner Comparison on Full Candidate Pools (Tier 1: 50-100 bricks)
    s1 = summary.step1_planner_stats
    assert s1["analytic_heuristic_with_cheap_lookahead"].completion_rate == 0.0
    assert s1["trained_scorer_greedy"].completion_rate == 0.0
    assert s1["trained_discrete_diffusion"].completion_rate == 1.0
    assert s1["full_quadratic_milp"].completion_rate == 1.0
    assert (
        s1["trained_discrete_diffusion"].mean_final_iou
        > s1["analytic_heuristic_with_cheap_lookahead"].mean_final_iou + 0.05
    )

    # Step 2: 4 Candidate-Engine Modes with `trained_discrete_diffusion`
    s2 = summary.step2_candidate_engine_stats
    assert s2["full"].completion_rate == 1.0
    assert s2["incremental"].completion_rate == 1.0
    assert s2["selective"].completion_rate == 1.0
    assert s2["incremental_selective"].completion_rate == 1.0
    assert s2["incremental_selective"].setup_brick_retention_rate == 1.0

    # Step 3: 5-Seed Tier 1 & Tier 2 (250-500 bricks, 274 bricks/build) Verification
    assert summary.step3_tier1_5seed_completion_rate == 1.0
    assert summary.step3_tier1_sequential_only_completion_rate == 1.0
    assert summary.step3_tier1_temp_support_completion_rate == 1.0
    assert summary.step3_tier2_250_500_diffusion_inc_sel_stats.completion_rate == 1.0
    assert summary.step3_tier2_250_500_heuristic_full_stats.completion_rate == 0.0
    assert summary.deployed_params_relative_frame_gnn < summary.deployed_params_sparse_port_ipa
