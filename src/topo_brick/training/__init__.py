"""Stage 3A, Stage 3.5 (WP-L7.5), & Stage 4 (WP-L8/L9) Training and Complete-Build Modules (PEP 562 lazy exports)."""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Tuple

_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    "CompleteBuildRolloutMetrics": ("topo_brick.training.complete_build_benchmark", "CompleteBuildRolloutMetrics"),
    "ComposedCompleteBuild": ("topo_brick.training.complete_build_benchmark", "ComposedCompleteBuild"),
    "ComposedCourseSpec": ("topo_brick.training.complete_build_benchmark", "ComposedCourseSpec"),
    "PlannerAggregateStats": ("topo_brick.training.complete_build_benchmark", "PlannerAggregateStats"),
    "Stage4SequentialExperimentSummary": ("topo_brick.training.complete_build_benchmark", "Stage4SequentialExperimentSummary"),
    "build_stage4_composed_corpus": ("topo_brick.training.complete_build_benchmark", "build_stage4_composed_corpus"),
    "execute_complete_build_rollout": ("topo_brick.training.complete_build_benchmark", "execute_complete_build_rollout"),
    "run_stage4_sequential_complete_build_study": ("topo_brick.training.complete_build_benchmark", "run_stage4_sequential_complete_build_study"),
    "FutureOpportunitySample": ("topo_brick.training.dataset_builder", "FutureOpportunitySample"),
    "RandomizedCourseSample": ("topo_brick.training.dataset_builder", "RandomizedCourseSample"),
    "StabilityCalibrationSample": ("topo_brick.training.dataset_builder", "StabilityCalibrationSample"),
    "build_future_opportunity_dataset": ("topo_brick.training.dataset_builder", "build_future_opportunity_dataset"),
    "build_randomized_course_dataset": ("topo_brick.training.dataset_builder", "build_randomized_course_dataset"),
    "build_stability_calibration_dataset": ("topo_brick.training.dataset_builder", "build_stability_calibration_dataset"),
    "extract_future_port_opportunity_features": ("topo_brick.training.dataset_builder", "extract_future_port_opportunity_features"),
    "CourseTrajectoryStep": ("topo_brick.training.ldraw_decomposer", "CourseTrajectoryStep"),
    "LDrawParseReport": ("topo_brick.training.ldraw_decomposer", "LDrawParseReport"),
    "LDrawPartSpec": ("topo_brick.training.ldraw_decomposer", "LDrawPartSpec"),
    "decompose_assembly_to_courses": ("topo_brick.training.ldraw_decomposer", "decompose_assembly_to_courses"),
    "export_state_to_ldr_text": ("topo_brick.training.ldraw_decomposer", "export_state_to_ldr_text"),
    "parse_ldr_text": ("topo_brick.training.ldraw_decomposer", "parse_ldr_text"),
    "parse_ldr_text_with_report": ("topo_brick.training.ldraw_decomposer", "parse_ldr_text_with_report"),
    "CandidateEngineModeMetrics": ("topo_brick.training.stage35_tightening", "CandidateEngineModeMetrics"),
    "FourWayCandidateEngineSummary": ("topo_brick.training.stage35_tightening", "FourWayCandidateEngineSummary"),
    "OpportunityTighteningSummary": ("topo_brick.training.stage35_tightening", "OpportunityTighteningSummary"),
    "PairedD3PMAblationSummary": ("topo_brick.training.stage35_tightening", "PairedD3PMAblationSummary"),
    "StabilityBoundaryCalibrationSummary": ("topo_brick.training.stage35_tightening", "StabilityBoundaryCalibrationSummary"),
    "StrategyMixingDiagnosticSummary": ("topo_brick.training.stage35_tightening", "StrategyMixingDiagnosticSummary"),
    "TwoCourseTeacherSolution": ("topo_brick.training.stage35_tightening", "TwoCourseTeacherSolution"),
    "build_non_myopic_opportunity_trap_dataset": ("topo_brick.training.stage35_tightening", "build_non_myopic_opportunity_trap_dataset"),
    "build_non_myopic_two_course_trap": ("topo_brick.training.stage35_tightening", "build_non_myopic_two_course_trap"),
    "clopper_pearson_ci_95": ("topo_brick.training.stage35_tightening", "clopper_pearson_ci_95"),
    "run_four_way_candidate_engine_study": ("topo_brick.training.stage35_tightening", "run_four_way_candidate_engine_study"),
    "run_paired_d3pm_denoising_ablation": ("topo_brick.training.stage35_tightening", "run_paired_d3pm_denoising_ablation"),
    "run_stability_boundary_and_ridge_study": ("topo_brick.training.stage35_tightening", "run_stability_boundary_and_ridge_study"),
    "run_strategy_mixing_value_study": ("topo_brick.training.stage35_tightening", "run_strategy_mixing_value_study"),
    "run_wp_l75a_opportunity_and_teacher_study": ("topo_brick.training.stage35_tightening", "run_wp_l75a_opportunity_and_teacher_study"),
    "solve_exact_two_course_teacher": ("topo_brick.training.stage35_tightening", "solve_exact_two_course_teacher"),
    "ExternalCorpusPlannerStats": ("topo_brick.training.stage5_external_corpus_and_hybrid_benchmark", "ExternalCorpusPlannerStats"),
    "ExternalShapeObjectSpec": ("topo_brick.training.stage5_external_corpus_and_hybrid_benchmark", "ExternalShapeObjectSpec"),
    "LDrawAttritionAndReconstructionSummary": ("topo_brick.training.stage5_external_corpus_and_hybrid_benchmark", "LDrawAttritionAndReconstructionSummary"),
    "SpoilerFreeRolloutMetrics": ("topo_brick.training.stage5_external_corpus_and_hybrid_benchmark", "SpoilerFreeRolloutMetrics"),
    "WPL10ContinuationPreservingSummary": ("topo_brick.training.stage5_external_corpus_and_hybrid_benchmark", "WPL10ContinuationPreservingSummary"),
    "WPL11HybridRouterAndMILPSummary": ("topo_brick.training.stage5_external_corpus_and_hybrid_benchmark", "WPL11HybridRouterAndMILPSummary"),
    "WPL12ExternalCorpusStudySummary": ("topo_brick.training.stage5_external_corpus_and_hybrid_benchmark", "WPL12ExternalCorpusStudySummary"),
    "build_stage5_external_shape_corpus": ("topo_brick.training.stage5_external_corpus_and_hybrid_benchmark", "build_stage5_external_shape_corpus"),
    "execute_spoiler_free_shape_to_brick_rollout": ("topo_brick.training.stage5_external_corpus_and_hybrid_benchmark", "execute_spoiler_free_shape_to_brick_rollout"),
    "run_wp_l10_continuation_preserving_study": ("topo_brick.training.stage5_external_corpus_and_hybrid_benchmark", "run_wp_l10_continuation_preserving_study"),
    "run_wp_l11_hybrid_router_and_continuous_milp_study": ("topo_brick.training.stage5_external_corpus_and_hybrid_benchmark", "run_wp_l11_hybrid_router_and_continuous_milp_study"),
    "run_wp_l12_external_shape_to_brick_corpus_study": ("topo_brick.training.stage5_external_corpus_and_hybrid_benchmark", "run_wp_l12_external_shape_to_brick_corpus_study"),
    "to_canonical_voxel_target_shape": ("topo_brick.training.stage5_external_corpus_and_hybrid_benchmark", "to_canonical_voxel_target_shape"),
    "to_merged_maximal_target_shape": ("topo_brick.training.stage5_external_corpus_and_hybrid_benchmark", "to_merged_maximal_target_shape"),
    "to_random_split_shuffled_target_shape": ("topo_brick.training.stage5_external_corpus_and_hybrid_benchmark", "to_random_split_shuffled_target_shape"),
    "LOCKED_STABLETEXT2BRICK_MANIFEST": ("topo_brick.training.stage55_and_stage6_audit_and_external_eval", "LOCKED_STABLETEXT2BRICK_MANIFEST"),
    "CoordinatedPlannerFamilyMetrics": ("topo_brick.training.stage55_and_stage6_audit_and_external_eval", "CoordinatedPlannerFamilyMetrics"),
    "ExternalObjectRolloutResult": ("topo_brick.training.stage55_and_stage6_audit_and_external_eval", "ExternalObjectRolloutResult"),
    "LockedStableText2BrickRecord": ("topo_brick.training.stage55_and_stage6_audit_and_external_eval", "LockedStableText2BrickRecord"),
    "WPL13AuditableRunBundleSummary": ("topo_brick.training.stage55_and_stage6_audit_and_external_eval", "WPL13AuditableRunBundleSummary"),
    "WPL14TargetAndCacheInvariantsSummary": ("topo_brick.training.stage55_and_stage6_audit_and_external_eval", "WPL14TargetAndCacheInvariantsSummary"),
    "WPL15FrozenExternalBenchmarkSummary": ("topo_brick.training.stage55_and_stage6_audit_and_external_eval", "WPL15FrozenExternalBenchmarkSummary"),
    "WPL16CoordinatedPlannerComparisonSummary": ("topo_brick.training.stage55_and_stage6_audit_and_external_eval", "WPL16CoordinatedPlannerComparisonSummary"),
    "compute_clopper_pearson_ci": ("topo_brick.training.stage55_and_stage6_audit_and_external_eval", "compute_clopper_pearson_ci"),
    "execute_locked_stabletext2brick_rollout": ("topo_brick.training.stage55_and_stage6_audit_and_external_eval", "execute_locked_stabletext2brick_rollout"),
    "parse_stabletext2brick_line": ("topo_brick.training.stage55_and_stage6_audit_and_external_eval", "parse_stabletext2brick_line"),
    "run_wp_l13_auditable_run_bundle": ("topo_brick.training.stage55_and_stage6_audit_and_external_eval", "run_wp_l13_auditable_run_bundle"),
    "run_wp_l14_target_and_cache_invariants_study": ("topo_brick.training.stage55_and_stage6_audit_and_external_eval", "run_wp_l14_target_and_cache_invariants_study"),
    "run_wp_l15_frozen_external_benchmark_study": ("topo_brick.training.stage55_and_stage6_audit_and_external_eval", "run_wp_l15_frozen_external_benchmark_study"),
    "run_wp_l16_coordinated_planner_comparison_study": ("topo_brick.training.stage55_and_stage6_audit_and_external_eval", "run_wp_l16_coordinated_planner_comparison_study"),
    "verify_sequential_constructibility_witness": ("topo_brick.training.stage55_and_stage6_audit_and_external_eval", "verify_sequential_constructibility_witness"),
    "FutureOpportunityEvalSummary": ("topo_brick.training.trainer", "FutureOpportunityEvalSummary"),
    "FutureOpportunityValueModel": ("topo_brick.training.trainer", "FutureOpportunityValueModel"),
    "Stage3AGeneralizationSummary": ("topo_brick.training.trainer", "Stage3AGeneralizationSummary"),
    "run_stage3a_full_training_and_ablation": ("topo_brick.training.trainer", "run_stage3a_full_training_and_ablation"),
    "train_and_evaluate_future_opportunity_q": ("topo_brick.training.trainer", "train_and_evaluate_future_opportunity_q"),
}

__all__: List[str] = list(_LAZY_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name, attr_name = target
    module = importlib.import_module(mod_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> List[str]:
    return sorted(set(globals().keys()) | set(__all__))
