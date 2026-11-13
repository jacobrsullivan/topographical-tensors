"""Evaluation, static equilibrium, physical validation, and gate verification modules for topo_brick."""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Tuple

_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    "EquilibriumResult": ("topo_brick.evaluation.static_equilibrium", "EquilibriumResult"),
    "SparseStaticEquilibriumValidator": ("topo_brick.evaluation.static_equilibrium", "SparseStaticEquilibriumValidator"),
    "SubStudEquilibriumResult": ("topo_brick.evaluation.physical_validation", "SubStudEquilibriumResult"),
    "PhysicalCaseEval": ("topo_brick.evaluation.physical_validation", "PhysicalCaseEval"),
    "ModelAgreementReport": ("topo_brick.evaluation.physical_validation", "ModelAgreementReport"),
    "CalibratedSubStudPolygonValidator": ("topo_brick.evaluation.physical_validation", "CalibratedSubStudPolygonValidator"),
    "load_physical_validation_assets": ("topo_brick.evaluation.physical_validation", "load_physical_validation_assets"),
    "evaluate_bricksim_demo_topology": ("topo_brick.evaluation.physical_validation", "evaluate_bricksim_demo_topology"),
    "run_physical_and_model_agreement_suite": ("topo_brick.evaluation.physical_validation", "run_physical_and_model_agreement_suite"),
    "PLANNER_TYPES": ("topo_brick.evaluation.benchmark_suite", "PLANNER_TYPES"),
    "STAGE2_5_PLANNER_TYPES": ("topo_brick.evaluation.benchmark_suite", "STAGE2_5_PLANNER_TYPES"),
    "Stage1BenchmarkSummary": ("topo_brick.evaluation.benchmark_suite", "Stage1BenchmarkSummary"),
    "Stage2PlannerBenchmarkResult": ("topo_brick.evaluation.benchmark_suite", "Stage2PlannerBenchmarkResult"),
    "Stage25EvidenceSummary": ("topo_brick.evaluation.benchmark_suite", "Stage25EvidenceSummary"),
    "Stage3BScalingSummary": ("topo_brick.evaluation.benchmark_suite", "Stage3BScalingSummary"),
    "run_target_wall_reconstruction": ("topo_brick.evaluation.benchmark_suite", "run_target_wall_reconstruction"),
    "run_benchmark_a1_1x6_parity_packing": ("topo_brick.evaluation.benchmark_suite", "run_benchmark_a1_1x6_parity_packing"),
    "run_benchmark_a2_2x3_domino_trap": ("topo_brick.evaluation.benchmark_suite", "run_benchmark_a2_2x3_domino_trap"),
    "run_benchmark_a2_2x3_domino_uniform": ("topo_brick.evaluation.benchmark_suite", "run_benchmark_a2_2x3_domino_uniform"),
    "run_benchmark_b_seam_staggering": ("topo_brick.evaluation.benchmark_suite", "run_benchmark_b_seam_staggering"),
    "run_benchmark_c_counterbalanced_cantilever": ("topo_brick.evaluation.benchmark_suite", "run_benchmark_c_counterbalanced_cantilever"),
    "run_stage2_full_comparison": ("topo_brick.evaluation.benchmark_suite", "run_stage2_full_comparison"),
    "run_stage2_5_evidence_suite": ("topo_brick.evaluation.benchmark_suite", "run_stage2_5_evidence_suite"),
    "run_stage3b_incremental_and_selective_benchmark": ("topo_brick.evaluation.benchmark_suite", "run_stage3b_incremental_and_selective_benchmark"),
}

__all__: List[str] = [
    "EquilibriumResult",
    "SparseStaticEquilibriumValidator",
    "SubStudEquilibriumResult",
    "PhysicalCaseEval",
    "ModelAgreementReport",
    "CalibratedSubStudPolygonValidator",
    "load_physical_validation_assets",
    "evaluate_bricksim_demo_topology",
    "run_physical_and_model_agreement_suite",
    "PLANNER_TYPES",
    "STAGE2_5_PLANNER_TYPES",
    "Stage1BenchmarkSummary",
    "Stage2PlannerBenchmarkResult",
    "Stage25EvidenceSummary",
    "Stage3BScalingSummary",
    "run_target_wall_reconstruction",
    "run_benchmark_a1_1x6_parity_packing",
    "run_benchmark_a2_2x3_domino_trap",
    "run_benchmark_a2_2x3_domino_uniform",
    "run_benchmark_b_seam_staggering",
    "run_benchmark_c_counterbalanced_cantilever",
    "run_stage2_full_comparison",
    "run_stage2_5_evidence_suite",
    "run_stage3b_incremental_and_selective_benchmark",
]


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
