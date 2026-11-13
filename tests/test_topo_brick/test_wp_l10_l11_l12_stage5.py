"""
Automated Verification Suite for Stage 5 (`WP-L10`, `WP-L11`, `WP-L12`):
  - `WP-L10` (`Gap 1`): Continuation-Preserving Selective Candidate Expansion & Per-Port Incremental Cache
  - `WP-L11` (`Gap 5` & `§6.1`): Continuous-y_ij McCormick MILP [R32] & Transparent Hybrid Course Router
  - `WP-L12` (`§5` & `§2`): External Shape-to-Brick Corpus Benchmark (Zero Course Spoilers) & LDraw Attrition
"""

from __future__ import annotations

from topo_brick.training.complete_build_benchmark import _train_complete_build_models
from topo_brick.training.stage5_external_corpus_and_hybrid_benchmark import (
    build_stage5_external_shape_corpus,
    run_wp_l10_continuation_preserving_study,
    run_wp_l11_hybrid_router_and_continuous_milp_study,
    run_wp_l12_external_shape_to_brick_corpus_study,
)


def test_wp_l10_l11_l12_stage5_end_to_end() -> None:
    """
    Verify all Stage 5 (`WP-L10`, `WP-L11`, `WP-L12`) mathematical, algorithmic, and empirical contracts.
    """
    # Train shared compact deployed RelativeFrameGNN model suite (~110.7K params) once
    s1_model, q_model, inter_mod, denoiser = _train_complete_build_models(
        seed=40, use_ipa=False
    )

    # ------------------------------------------------------------------
    # 1. WP-L10: Continuation-Preserving Selective Expansion & Opportunity Loss
    # ------------------------------------------------------------------
    r10 = run_wp_l10_continuation_preserving_study(
        s1_model=s1_model, q_model=q_model, inter_mod=inter_mod, denoiser=denoiser
    )
    assert r10.mean_opportunity_loss_delta_h == 0.0
    assert r10.max_opportunity_loss_delta_h == 0.0
    assert r10.bipartite_multi_stud_recall == 1.0
    assert r10.snot_continuation_recall == 1.0

    for mode in ("full", "incremental", "selective", "incremental_selective"):
        st = r10.engine_stats_tier1[mode]
        assert st.completion_rate == 1.0
        assert abs(st.mean_final_iou - 1.0) < 1e-5
        assert st.fallback_trigger_rate == 0.0

    # Verify Tier 2 (257 bricks, 84 courses) recovers 1.000 IoU and 0.0% fallback rate
    assert r10.tier2_incremental_selective_stats.completion_rate == 1.0
    assert abs(r10.tier2_incremental_selective_stats.mean_final_iou - 1.0) < 1e-5
    assert r10.tier2_incremental_selective_stats.fallback_trigger_rate == 0.0
    assert (
        r10.engine_stats_tier1["incremental_selective"].mean_cand_gen_ms
        < r10.engine_stats_tier1["full"].mean_cand_gen_ms
    )

    # ------------------------------------------------------------------
    # 2. WP-L11: Continuous-y_ij McCormick MILP & Transparent Hybrid Router
    # ------------------------------------------------------------------
    r11 = run_wp_l11_hybrid_router_and_continuous_milp_study(
        s1_model=s1_model, q_model=q_model, inter_mod=inter_mod, denoiser=denoiser
    )
    assert r11.continuous_vs_binary_milp_max_obj_diff < 1e-4
    assert r11.continuous_milp_mean_int_vars < r11.binary_milp_mean_int_vars
    assert r11.tier1_hybrid_router_stats.completion_rate == 1.0
    assert abs(r11.tier1_hybrid_router_stats.mean_final_iou - 1.0) < 1e-5
    assert r11.tier2_hybrid_router_stats.completion_rate == 1.0
    assert abs(r11.tier2_hybrid_router_stats.mean_final_iou - 1.0) < 1e-5
    assert (
        r11.tier1_hybrid_router_stats.mean_planner_ms
        < r11.tier1_pure_diffusion_stats.mean_planner_ms
    )

    # ------------------------------------------------------------------
    # 3. WP-L12: Staged External Shape-to-Brick Corpus & LDraw Attrition
    # ------------------------------------------------------------------
    train_objs, test_objs = build_stage5_external_shape_corpus()
    assert len(train_objs) == 8
    assert len(test_objs) == 12
    train_ids = {o.object_id for o in train_objs}
    test_ids = {o.object_id for o in test_objs}
    assert train_ids.isdisjoint(test_ids)

    r12 = run_wp_l12_external_shape_to_brick_corpus_study(
        s1_model=s1_model, q_model=q_model, inter_mod=inter_mod, denoiser=denoiser
    )
    assert r12.planner_stats["analytic_heuristic_with_cheap_lookahead"].completion_rate == 0.0
    assert r12.planner_stats["trained_scorer_greedy"].completion_rate == 0.0

    for strong_planner in (
        "trained_conditional_ar_or_recurrent",
        "trained_discrete_diffusion",
        "hybrid_diffusion_router",
        "full_quadratic_milp",
    ):
        pst = r12.planner_stats[strong_planner]
        assert pst.completion_rate == 1.0
        assert abs(pst.mean_final_iou - 1.0) < 1e-5
        assert pst.unseen_category_completion_rate == 1.0

    att = r12.ldraw_attrition_summary
    assert att.num_compliant_ldr_models == 8
    assert att.compliant_accepted_lines == att.compliant_total_type1_lines
    assert att.compliant_decomposition_success_rate == 1.0
    assert att.compliant_closed_loop_completion_rate == 1.0
    assert att.noisy_rejected_unknown_part_count == 3
    assert att.noisy_rejected_off_lattice_count == 2
    assert att.noisy_rejected_non_cubic_rot_count == 2
