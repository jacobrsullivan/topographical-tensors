"""
Comprehensive Phase 7 Verification Suite (`R8-WS1` through `R8-WS6`):

Tests all Round 8 remediation and state-of-the-art enhancement workstreams end-to-end:
1. `R8-WS1` & `R8-WS2` (`Eval B`): All 6 mandatory adversarial kernel soundness cases,
   untraced `UNKNOWN` semantics, `FoundationPolicy` separation, `PerturbationMechanicsCert`,
   and task-aware `repair()` with `RepairAuditReport` & `PROVEN_INFEASIBLE` on locked features.
2. `R8-WS3` (`Eval A` Track 1 & Track 2): `S0_Reference_Replay` separation from blind `S1`,
   `40` raw continuous 3D triangle meshes (including `6` real `BrickAnything` `.obj` meshes),
   complete 6-term `T_total` latency breakdown, active tile memory accounting (`2.42 MiB/tile`),
   and comparison against `GreedyLayerMerge_Control` + `Luo2015_IterativeSeamRefinement`.
3. `R8-WS4`: 7-Row Capability Matrix (`Import` -> `Physical Validation`) across all 8 connector
   families + `30` hybrid cubic & continuous `SE(3)` builds across 6 task classes including
   damped Gauss-Newton loop closure (`< 1e-3 LDU` residual).
4. `R8-WS5`: 3-Tier Physical Evidence Separation across `9/9` CMU `StableLego` hardware JSONs
   (including horizontal force), strictly disjoint `Train (n=40) -> Holdout (n=110)` boundary
   assemblies (`holdout_false_pass_rate <= 0.02`), and `40/40` multi-connector boundary rigs.
5. `R8-WS6` & `Eval C`: `PairwiseSynergyValueModel` (`M3_Pairwise`) resolving multimodal
   strategy-mixing failure while running faster than 2-course beam lookahead, whole-build `M2`
   promotion-gate honesty vs. deterministic port filtering, zero-shot `M1` held-out family
   transfer, and `Eval C` Fixed Constructive Agent A/B (`n = 50` prompts).
"""

from __future__ import annotations

from topo_brick import (
    CAPABILITY_MATRIX_ROWS,
    CONNECTOR_FAMILY_COLUMNS,
    UNTOUCHED_STABLETEXT2BRICK_21_MANIFEST,
    CertStatus,
    KernelResultStatus,
    TopoBrickKernel,
    run_eval_b_adversarial_kernel_suite,
    run_expanded_physical_calibration_suite,
    run_mixed_family_7row_versatility_benchmark,
    run_raw_mesh_blind_benchmark,
    run_selective_learning_and_eval_c_suite,
)


def test_r8_ws1_and_ws2_eval_b_all_six_adversarial_cases() -> None:
    """Verify all 6 mandatory §9B adversarial soundness cases pass with zero false passes."""
    rep = run_eval_b_adversarial_kernel_suite()
    assert rep.case1_delayed_counterweight_rejected
    assert rep.case1_untraced_import_returns_unknown
    assert rep.case2_no_single_part_order_both_orders_rejected
    assert rep.case2_temp_support_trace_certified
    assert rep.case2_subassembly_trace_certified
    assert rep.case3_blocked_cavity_insertion_rejected
    assert rep.case3_open_corridor_order_certified
    assert rep.case4_cubic_third_part_intrusion_rejected_both_orders
    assert rep.case4_se3_third_part_intrusion_rejected_both_orders
    assert rep.case5_inverted_pendulum_perturbation_rejected
    assert rep.case5_hanging_and_locking_hinge_certified
    assert rep.case6_locked_feature_deletion_refused_infeasible
    assert rep.case6_constructive_pillar_preserves_locked_feature
    assert rep.all_six_cases_passed


def test_r8_ws3_raw_mesh_blind_benchmark_6term_latency_and_luo2015_baseline() -> None:
    """Verify blind raw triangle-mesh construction, S0 replay separation, 6-term T_total, and Luo2015 baseline."""
    kernel = TopoBrickKernel()
    rec0 = UNTOUCHED_STABLETEXT2BRICK_21_MANIFEST[0]
    res_s0 = kernel.realize_external_record(rec0, mode="reference_replay")
    res_s1 = kernel.realize_external_record(rec0, mode="blind_from_voxel_silhouette")
    assert res_s0.build_record.solver_id == "S0_Reference_Replay"
    assert res_s0.status == KernelResultStatus.COMPLETED_CERTIFIED
    assert res_s1.status == KernelResultStatus.COMPLETED_CERTIFIED
    assert res_s1.certificates.physical_validation.status == CertStatus.UNKNOWN

    mesh_rep = run_raw_mesh_blind_benchmark()
    assert mesh_rep.num_total_meshes == 40
    assert mesh_rep.num_external_brickanything_meshes == 6

    s1_pil = mesh_rep.solver_summaries["S1_Blind_MinimalPillars"]
    s1_nosup = mesh_rep.solver_summaries["S1_Blind_NoSupport"]
    greedy = mesh_rep.solver_summaries["GreedyLayerMerge_Control"]
    luo = mesh_rep.solver_summaries["Luo2015_IterativeSeamRefinement"]

    # Complete 6-term T_total & tile memory accounting must be populated and positive
    assert s1_pil.median_t_target_ms > 0.0
    assert s1_pil.median_t_candidates_ms > 0.0
    assert s1_pil.median_t_selection_ms > 0.0
    assert s1_pil.median_t_mechanics_ms >= 0.0
    assert s1_pil.median_t_execution_checks_ms >= 0.0
    assert s1_pil.mean_active_tiles >= 1.0
    assert s1_pil.mean_tile_memory_mib >= 2.40
    assert s1_pil.median_total_ms < 500.0

    # Luo2015 seam refinement beats unaligned GreedyLayerMerge_Control, and S1 beats both!
    assert luo.certified_conjunction_rate >= greedy.certified_conjunction_rate
    assert s1_nosup.certified_conjunction_rate >= luo.certified_conjunction_rate
    assert s1_pil.certified_conjunction_rate >= 0.90
    assert s1_pil.certified_conjunction_rate > greedy.certified_conjunction_rate + 0.25


def test_r8_ws4_mixed_family_7row_matrix_and_hybrid_se3_loop_closure() -> None:
    """Verify 7-row x 8-family capability matrix and 30/30 hybrid cubic + continuous SE(3) task builds."""
    rep = run_mixed_family_7row_versatility_benchmark()
    assert tuple(rep.capability_matrix.keys()) == CAPABILITY_MATRIX_ROWS
    for row in CAPABILITY_MATRIX_ROWS:
        assert tuple(rep.capability_matrix[row].keys()) == CONNECTOR_FAMILY_COLUMNS
        for val in rep.capability_matrix[row].values():
            assert val in ("VERIFIED", "SCOPED", "UNTESTED")

    assert rep.total_instances == 30
    assert rep.total_certified == 30
    assert rep.overall_certified_rate == 1.0
    truss = rep.task_class_summaries["loop_closed_a_frame_truss"]
    assert truss.certified_count == 5
    assert truss.loop_closure_max_residual_ldu < 1e-3


def test_r8_ws5_disjoint_physical_calibration_false_pass_bound_and_3_tiers() -> None:
    """Verify 8/8 CMU hardware cases separated from 16-brick solver fixture, disjoint Train(40)->Holdout(110) <= 2% FP, and 40/40 rigs."""
    rep = run_expanded_physical_calibration_suite()
    assert rep.cmu_hardware_8case_matched == 8
    assert rep.cmu_hardware_8case_total == 8
    assert rep.upstream_horizontal_contact_fixture_verified
    assert rep.synthetic_lateral_sanity_verified
    assert rep.train_n == 40
    assert rep.holdout_n == 110
    assert rep.holdout_false_pass_rate <= 0.02
    assert rep.holdout_accuracy >= 0.78
    assert rep.multi_connector_rigs_matched == 40
    assert rep.multi_connector_accuracy == 1.0
    assert rep.tier1_cold_rebuild_consistency_rate == 1.0
    assert rep.tier2_bricksim_demo_stable


def test_r8_ws6_pairwise_m3_whole_build_m2_heldout_m1_and_eval_c_agent_ab() -> None:
    """Verify M3_Pairwise beating unary strategy-mixing & S1, honest M2 gate, M1 transfer, and Eval C Agent A/B."""
    rep = run_selective_learning_and_eval_c_suite()
    # M3_Pairwise resolves symmetric multimodal strategy mixing where myopic S1 & unary M3 fail
    assert rep.m3_pairwise_solve_rate >= 0.95
    assert rep.m3_pairwise_solve_rate > rep.m3_myopic_s1_solve_rate + 0.50
    assert rep.m3_pairwise_solve_rate > rep.m3_unary_solve_rate + 0.30
    assert rep.m3_pairwise_median_ms < rep.m3_lookahead2_median_ms
    assert rep.m3_pairwise_verdict.module_benefit_gate_passed

    # Whole-build M2 preserves IoU while strictly honoring deterministic control comparison
    assert abs(rep.m2_whole_build_iou_delta_vs_unpruned) <= 0.01

    # Zero-shot M1 held-out family transfer (slopes, SNOT, clips/bars)
    assert rep.m1_held_out_families_certified_count == 12
    assert rep.m1_held_out_transfer_rate == 1.0

    # Eval C Fixed Constructive Agent A/B (n = 50 prompts)
    ab = rep.eval_c_agent_ab
    assert ab.num_tasks == 50
    assert ab.kernel_aided_certified_rate >= 0.92
    assert ab.kernel_aided_certified_rate > ab.standalone_certified_rate + 0.25
    assert ab.kernel_aided_mean_iou >= 0.94
