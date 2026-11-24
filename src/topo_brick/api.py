"""
Curated Consumer API Surface for `topo_brick` (`v5.4.1`, `Phase 15–16.1`).

This module exposes the supported ~45-symbol downstream consumer interface for deterministic
3D digital brick synthesis, repair, action-grammar constraint, non-certificate design
diagnostics, cross-route comparison, and zero-kernel-import independent verification.

Key Architectural Guarantees:
  1. **Zero Eager `torch` Dependency:** Importing `topo_brick.api` or resolving any symbol in
     `SUPPORTED_CONSUMER_EXPORTS` (including `TopoBrickKernel`, `voxelize_triangle_mesh_to_target_field`,
     and `verify_certificate_independently`) keeps `"torch" not in sys.modules`. Optional neural
     planners (`M1`/`M2`/`M3`, `D3PM`, `BrickGrammarLogitsProcessor`) remain available under
     `topo_brick` and `topo_brick.kernel.learned_bottleneck_controls`.
  2. **Explicit Verification Boundary:** Certificate verification (`verify_certificate_independently`,
     `build_relying_party_obligations`) enforces zero `topo_brick.kernel.*` imports and defaults
     relying-party target-bound builders to `verification_scope="assembly_and_target"`.
  3. **Diagnostic vs. Certificate Separation:** Design reports (`CatalogFunnelAndShortlistReport`,
     `RegionalAppearanceReport`, `BilateralSymmetryReport`) are planner telemetry only and never
     enter `CertificateBundle` or independent mechanical/geometric replay verdicts.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Tuple

_API_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    # 1. Core Construction Kernel, Policy & Certificates
    "TopoBrickKernel": ("topo_brick.kernel.construction_kernel", "TopoBrickKernel"),
    "RealizeRequestSpec": ("topo_brick.kernel.realize_policy", "RealizeRequestSpec"),
    "ResolvedRealizePolicy": ("topo_brick.kernel.realize_policy", "ResolvedRealizePolicy"),
    "RealizeRouteProvenance": ("topo_brick.kernel.kernel_contracts", "RealizeRouteProvenance"),
    "KernelRealizeResult": ("topo_brick.kernel.kernel_contracts", "KernelRealizeResult"),
    "KernelRepairResult": ("topo_brick.kernel.kernel_contracts", "KernelRepairResult"),
    "KernelConstrainResult": ("topo_brick.kernel.kernel_contracts", "KernelConstrainResult"),
    "KernelResultStatus": ("topo_brick.kernel.kernel_contracts", "KernelResultStatus"),
    "CertStatus": ("topo_brick.kernel.kernel_contracts", "CertStatus"),
    "CertificateBundle": ("topo_brick.kernel.kernel_contracts", "CertificateBundle"),
    "FoundationPolicy": ("topo_brick.evaluation.static_equilibrium", "FoundationPolicy"),
    "ValidatedConstraintSpec": ("topo_brick.kernel.kernel_contracts", "ValidatedConstraintSpec"),
    "SingleCoreAuditReport": ("topo_brick.kernel.kernel_contracts", "SingleCoreAuditReport"),
    "enforce_and_audit_single_core": (
        "topo_brick.kernel.kernel_contracts",
        "enforce_and_audit_single_core",
    ),
    # 2. Targets, Frames & Assembly Representations
    "SparseTiledTargetField": (
        "topo_brick.representation.interface_fields",
        "SparseTiledTargetField",
    ),
    "TargetShapeBoxes": ("topo_brick.representation.interface_fields", "TargetShapeBoxes"),
    "AdaptiveSurfaceTarget": (
        "topo_brick.kernel.adaptive_surface_and_macro_planner",
        "AdaptiveSurfaceTarget",
    ),
    "voxelize_triangle_mesh_to_target_field": (
        "topo_brick.representation.interface_fields",
        "voxelize_triangle_mesh_to_target_field",
    ),
    "AssemblyState": ("topo_brick.interface.port_manifold", "AssemblyState"),
    "ContinuousAssemblyState": (
        "topo_brick.interface.se3_kinematics",
        "ContinuousAssemblyState",
    ),
    "LDUFrame": ("topo_brick.interface.ldu_frames", "LDUFrame"),
    "LDUBox": ("topo_brick.interface.ldu_frames", "LDUBox"),
    "ContinuousSE3Pose": ("topo_brick.interface.se3_kinematics", "ContinuousSE3Pose"),
    # 3. Inventory, Compositional `.mpd`, View Priority & Official LDraw Palette
    "ColoredInventorySpec": ("topo_brick.interface.brick_catalog", "ColoredInventorySpec"),
    "HollowShellPolicy": (
        "topo_brick.kernel.adaptive_surface_and_macro_planner",
        "HollowShellPolicy",
    ),
    "ViewPrioritySpec": ("topo_brick.kernel.view_priority", "ViewPrioritySpec"),
    "SubmodelNodeSpec": ("topo_brick.kernel.subassembly_realizer", "SubmodelNodeSpec"),
    "LandmarkAttachmentRequest": (
        "topo_brick.kernel.subassembly_realizer",
        "LandmarkAttachmentRequest",
    ),
    "LDRAW_FULL_PALETTE_RECORDS": (
        "topo_brick.interface.ldraw_palette",
        "LDRAW_FULL_PALETTE_RECORDS",
    ),
    "flatten_omr_mpd_to_se3": (
        "topo_brick.interface.se3_kinematics",
        "flatten_omr_mpd_to_se3",
    ),
    # 4. Non-Certificate Planner Design Diagnostics
    "CatalogFunnelAndShortlistReport": (
        "topo_brick.kernel.design_diagnostics",
        "CatalogFunnelAndShortlistReport",
    ),
    "RegionalAppearanceReport": (
        "topo_brick.kernel.design_diagnostics",
        "RegionalAppearanceReport",
    ),
    "BilateralSymmetryReport": (
        "topo_brick.kernel.design_diagnostics",
        "BilateralSymmetryReport",
    ),
    "evaluate_bilateral_construction_symmetry": (
        "topo_brick.kernel.design_diagnostics",
        "evaluate_bilateral_construction_symmetry",
    ),
    # 5. Independent Verification & Cross-Route Evaluation
    "VerificationObligations": (
        "topo_brick.evaluation.independent_verifier",
        "VerificationObligations",
    ),
    "MechanicsVerificationProfile": (
        "topo_brick.evaluation.independent_verifier",
        "MechanicsVerificationProfile",
    ),
    "VerifierResourceLimits": (
        "topo_brick.evaluation.independent_verifier",
        "VerifierResourceLimits",
    ),
    "IndependentVerificationReport": (
        "topo_brick.evaluation.independent_verifier",
        "IndependentVerificationReport",
    ),
    "VersionedCertificateArtifact": (
        "topo_brick.evaluation.independent_verifier",
        "VersionedCertificateArtifact",
    ),
    "export_certificate_artifact": (
        "topo_brick.evaluation.independent_verifier",
        "export_certificate_artifact",
    ),
    "verify_certificate_independently": (
        "topo_brick.evaluation.independent_verifier",
        "verify_certificate_independently",
    ),
    "verify_certificate_json_file": (
        "topo_brick.evaluation.independent_verifier",
        "verify_certificate_json_file",
    ),
    "verify_certificate_in_isolated_worker": (
        "topo_brick.evaluation.independent_verifier",
        "verify_certificate_in_isolated_worker",
    ),
    "build_relying_party_obligations": (
        "topo_brick.evaluation.independent_verifier",
        "build_relying_party_obligations",
    ),
    "compare_realize_routes_on_target": (
        "topo_brick.evaluation.route_comparison",
        "compare_realize_routes_on_target",
    ),
    "CrossRouteComparisonReport": (
        "topo_brick.evaluation.route_comparison",
        "CrossRouteComparisonReport",
    ),
    "SingleRouteEvaluationEntry": (
        "topo_brick.evaluation.route_comparison",
        "SingleRouteEvaluationEntry",
    ),
    "MPDRoundTripVerificationReport": (
        "topo_brick.evaluation.route_comparison",
        "MPDRoundTripVerificationReport",
    ),
    "verify_exported_mpd_roundtrip": (
        "topo_brick.evaluation.route_comparison",
        "verify_exported_mpd_roundtrip",
    ),
}

SUPPORTED_CONSUMER_EXPORTS: Tuple[str, ...] = tuple(_API_LAZY_EXPORTS.keys())

__all__: List[str] = ["SUPPORTED_CONSUMER_EXPORTS", *SUPPORTED_CONSUMER_EXPORTS]


def __getattr__(name: str) -> Any:
    target = _API_LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name, attr_name = target
    module = importlib.import_module(mod_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> List[str]:
    return sorted(set(globals().keys()) | set(__all__))
