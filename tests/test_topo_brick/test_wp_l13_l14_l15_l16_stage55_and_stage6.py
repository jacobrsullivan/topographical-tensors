"""
Stage 5.5 (`WP-L13`, `WP-L14`) & Stage 6 (`WP-L15`, `WP-L16`) Verification Test Suite.

Verifies:
  1. `WP-L13` (Step 1 — Auditable Run Bundle & Reporting Corrections):
     - Zero train/test overlap on near-boundary stability instances (`boundary_train_test_overlap_count == 0`).
     - Constant-1.0 predictor boundary MAE (`0.0800 <= 0.15`), with trained GNN boundary MAE strictly
       beating both the Constant-1.0 predictor and 24-ch Ridge on the disjoint boundary test split,
       and beating Constant-1.0, Ridge, and Untrained on the wide-range test split.
     - Explicit per-proposal LP gate trace confirming 100% of accepted proposals pass an exact LP.
     - Reconciled McCormick MILP active vs. unfiltered pair variable ledger and large-active-M (`M >= 44`)
       zero-gap equivalence.
     - Matched Tier 1 and Tier 2 LP call accounting and Object-Level vs. Seed-Robustness Clopper-Pearson CIs.
  2. `WP-L14` (Step 2 — Differential Target Segmentation & Per-Port Cache Invariants):
     - Exact 4-way target partition invariance (`max_partition_iou_diff == 0.0` across Unmerged,
       Canonical Voxels, Merged Maximal Boxes, and Randomly Split/Shuffled Boxes).
     - `100.0%` completion under common unconstrained catalog (`use_common_catalog=True`).
     - Exact differential cache equivalence under deferred port reactivation, selective-to-full
       fallback, neighbour contact updates, and odd-LDU `IntegralTargetGrid` fallback.
     - Brick-by-brick sequential constructibility prefix witness verification.
  3. `WP-L15` (Step 3 — Frozen Independently Sourced External `AvaLovelace/StableText2Brick` Benchmark):
     - Cryptographic SHA-256 verification of all 16 locked external `StableText2Brick` objects
       (`28–151` bricks across 4 ShapeNet categories: `02843684`, `03467517`, `02876657`, `04460130`).
     - Separation of population catalog coverage vs. blind whole-target reconstruction.
     - Demonstration that `frontier_reachable_z` scheduling (`100.0%`, `16/16`) resolves the `4`
       external objects with hanging eaves/overhangs where `strict_lowest_z` stalls (`75.0%`, `12/16`),
       and coordinated lookahead outperforms `myopic_unary_no_lookahead`.
  4. `WP-L16` (Step 4 — Controlled 3-Family Coordinated Planner & Terminal-Prior Diffusion Study):
     - Explicit comparison of `unary_greedy`, `conditional_autoregressive_greedy`,
       `synchronous_recurrent_refinement_s2`, `d3pm_diffusion_s4_terminal_prior`, and
       `continuous_mccormick_milp` across Full vs. Selective expansion.
     - True terminal-prior autonomous multi-modal generation (`3/3 = 100.0%` mode coverage with zero
       clean-solution bits vs. `1/3` single-mode collapse for deterministic refinement).
"""

from __future__ import annotations

from topo_brick.training.complete_build_benchmark import _train_complete_build_models
from topo_brick.training.stage55_and_stage6_audit_and_external_eval import (
     run_wp_l13_auditable_run_bundle,
    run_wp_l14_target_and_cache_invariants_study,
    run_wp_l15_frozen_external_benchmark_study,
    run_wp_l16_coordinated_planner_comparison_study,
)


def test_wp_l13_l14_l15_l16_stage55_and_stage6_end_to_end() -> None:
    s1_model, q_model, inter_mod, denoiser = _train_complete_build_models(seed=42, use_ipa=False)

    # 1. WP-L13: Auditable Run Bundle & Immediate Reporting Corrections
    s13 = run_wp_l13_auditable_run_bundle(
        s1_model=s1_model,
        q_model=q_model,
        inter_mod=inter_mod,
        denoiser=denoiser,
        seed=42,
    )
    stab = s13.stability_calibration
    assert stab.boundary_train_test_overlap_count == 0
    assert stab.constant_one_boundary_mae <= 0.15
    assert stab.trained_boundary_eta_mae < stab.constant_one_boundary_mae
    assert stab.trained_boundary_eta_mae < stab.ridge_boundary_eta_mae
    assert stab.trained_eta_mae < stab.constant_one_wide_mae
    assert stab.trained_eta_mae < stab.ridge_eta_mae
    assert stab.accepted_unstable_rate_with_exact_gate == 0.0
    assert all(
        entry["lp_called"] is True and entry["exact_lp_certificate_eta"] <= 1.0
        for entry in stab.lp_gate_proposal_trace
        if entry["accepted"]
    )

    assert s13.milp_variable_accounting.mean_active_instantiated_pair_vars > 0.0
    assert (
        s13.milp_variable_accounting.mean_active_instantiated_pair_vars
        < s13.milp_variable_accounting.mean_unfiltered_synergy_pairs
    )
    assert s13.large_active_m_candidates >= 44
    assert s13.large_active_m_instantiated_pairs >= 400
    assert s13.large_active_m_max_obj_diff < 1e-4

    assert s13.tier1_lp_calls_per_build == 16
    assert s13.tier2_final_only_lp_calls == 1
    assert s13.tier2_every_course_lp_calls == 84
    assert round(s13.object_level_ci_8_tier1_builds[0], 3) == 0.631
    assert round(s13.object_level_ci_12_internal_shapes[0], 3) == 0.735
    assert round(s13.object_level_ci_16_external_stabletext2brick[0], 3) == 0.794

    # 2. WP-L14: Target Partition & Per-Port Cache Differential Invariants
    s14 = run_wp_l14_target_and_cache_invariants_study(
        s1_model=s1_model,
        q_model=q_model,
        inter_mod=inter_mod,
        denoiser=denoiser,
        seed=42,
    )
    assert s14.unmerged_mean_iou == 1.0
    assert s14.canonical_voxel_mean_iou == 1.0
    assert s14.merged_maximal_mean_iou == 1.0
    assert s14.random_split_shuffled_mean_iou == 1.0
    assert s14.max_partition_iou_diff == 0.0
    assert s14.common_catalog_completion_rate == 1.0
    assert s14.common_catalog_mean_iou == 1.0
    assert s14.initial_foundation_part_count == 6
    assert s14.initial_foundation_target_iou_contribution == 0.0
    assert s14.deferred_port_reactivation_exact_match is True
    assert s14.selective_to_full_fallback_exact_match is True
    assert s14.neighbour_contact_update_exact_match is True
    assert s14.odd_ldu_grid_alignment_exact_match is True
    assert s14.sequential_constructibility_witness_verified_rate == 1.0

    # 3. WP-L15: Frozen Independently Sourced External `AvaLovelace/StableText2Brick` Benchmark
    s15 = run_wp_l15_frozen_external_benchmark_study()
    assert s15.locked_external_object_count == 16
    assert s15.locked_large_100plus_brick_object_count == 4
    assert s15.all_sha256_hashes_verified is True
    assert s15.ldraw_unsupported_valid_part_lines == 3
    assert s15.ldraw_unsupported_noncubic_pose_lines == 2
    assert s15.ldraw_malformed_syntax_lines == 1
    assert s15.coordinated_frontier_reachable_completion_rate == 1.0
    assert s15.coordinated_frontier_reachable_mean_iou == 1.0
    assert s15.coordinated_large_100plus_completion_rate == 1.0
    assert (
        s15.coordinated_frontier_reachable_completion_rate
        > s15.strict_lowest_z_scheduler_completion_rate
    )
    assert (
        s15.coordinated_frontier_reachable_completion_rate > s15.myopic_no_lookahead_completion_rate
    )
    assert s15.peak_raw_candidates_m_observed >= 190
    assert s15.peak_active_candidates_m_observed >= 50

    # 4. WP-L16: Controlled 3-Family Coordinated Planner & Terminal-Prior Diffusion Study
    s16 = run_wp_l16_coordinated_planner_comparison_study(
        s1_model=s1_model,
        q_model=q_model,
        inter_mod=inter_mod,
        denoiser=denoiser,
        seed=42,
    )
    assert s16.planner_metrics["unary_greedy"].completion_rate == 0.0
    assert s16.planner_metrics["conditional_autoregressive_greedy"].completion_rate == 0.0
    assert s16.planner_metrics["synchronous_recurrent_refinement_s2"].completion_rate == 1.0
    assert s16.planner_metrics["d3pm_diffusion_s4_terminal_prior"].completion_rate == 1.0
    assert s16.planner_metrics["continuous_mccormick_milp"].completion_rate == 1.0
    assert s16.terminal_prior_validity_rate_by_steps[4] == 1.0
    assert s16.terminal_prior_distinct_modes_covered_d3pm == 3
    assert s16.terminal_prior_distinct_modes_covered_deterministic_s2 == 1
    assert (
        s16.terminal_prior_mode_coverage_ratio_d3pm
        > s16.terminal_prior_mode_coverage_ratio_deterministic_s2
    )
