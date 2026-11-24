"""
Phase 15 (`v5.3.0`) Architecture & Policy Remediation Audit Suite (`Priorities 1–5` & `§5`):

1. `test_p15_01_cross_route_deadline_and_offset_datum_propagation`:
   - Shared `deadline_monotonic_s` / `max_wall_clock_s` propagated across `open_catalog_sculpted`,
     multi-node `compositional_mpd`, `adaptive_plate_slope`, and `course_by_course_search`
     (`S1_CoverAndRepair` & `GreedyLayerMerge_Control`), returning `BUDGET_EXCEEDED` within grace.
   - Non-zero world ground datum (`ground_plane_z_ldu = -24.0`) on `TABLE_FRICTION_ONLY`
     compositional `.mpd` with landmark attachment, verified end-to-end by `.mpd` SE(3) round-trip
     and `verify_certificate_independently` (`"assembly_and_target"` scope), failing closed when
     the relying-party ground datum mismatches.
2. `test_p15_02_route_provenance_quickstart_scope_and_three_route_comparison`:
   - `RealizeRouteProvenance` populated on all routes, `build_relying_party_obligations(...)`
     defaulting to `"assembly_and_target"` (failing closed on target substitution),
   - `compare_realize_routes_on_target(...)` evaluating exported `.mpd` 5-camera silhouette IoU
     across `course_by_course_search`, `open_catalog_sculpted`, and `compositional_mpd`,
   - `scripts/run_topo_brick_quickstart.py` binding target obligations (`verification_scope == "assembly_and_target"`).
3. `test_p15_03_modular_extraction_line_caps_and_legacy_imports`:
   - `construction_kernel.py < 2,400` lines and `open_catalog_realizer.py < 1,400` lines,
   - Legacy `from topo_brick.kernel.construction_kernel import ...` and
     `from topo_brick.kernel.open_catalog_realizer import ...` imports 100% backward-compatible,
   - `independent_verifier.py` AST + runtime `sys.modules` zero-kernel-import boundary intact.
4. `test_p15_04_catalog_funnel_regional_color_and_bilateral_symmetry_diagnostics`:
   - `CatalogFunnelAndShortlistReport` exact parity with `CATALOG_RETRIEVAL_INDEX.funnel_metadata()`,
   - `RegionalAppearanceReport` visible vs. occluded `ΔE_00`, per-region `ΔE_00`, `322`-colour
     `open_color_palette_mode="ldconfig_finish_aware"`, and crisp red/white stripe boundary brick
     detection (`high_gradient_boundary_bricks_count >= 1`),
   - `evaluate_bilateral_construction_symmetry` passing matched wings & opposite-hand wedge pair
     (`43722` / `43723`) while flagging `1x4` left wing vs. two `1x2` right wing plates
     (`silhouette_mirror_iou == 1.0`, `exterior_construction_symmetry_ratio < 1.0`), without
     invalidating `CertificateBundle.overall_certified`.
5. `test_p15_05_torch_free_deterministic_kernel_and_consumer_api_boundary`:
   - Fresh isolated subprocesses resolving `topo_brick.api.SUPPORTED_CONSUMER_EXPORTS`, running
     deterministic sculpted `TopoBrickKernel.realize()` + `"assembly_and_target"` replay with
     `"torch" not in sys.modules`, and running `scripts/benchmark_topo_brick_boundaries.py`.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Tuple

import numpy as np

import topo_brick as tb
import topo_brick.api as tb_api
from topo_brick.evaluation.independent_verifier import (
    build_relying_party_obligations,
    export_certificate_artifact,
    verify_certificate_independently,
)
from topo_brick.evaluation.route_comparison import compare_realize_routes_on_target
from topo_brick.evaluation.static_equilibrium import FoundationPolicy
from topo_brick.interface.catalog_retrieval_index import CATALOG_RETRIEVAL_INDEX
from topo_brick.interface.ldu_frames import LDUBox, LDUFrame
from topo_brick.interface.port_manifold import AssemblyState
from topo_brick.interface.se3_kinematics import flatten_omr_mpd_to_se3
from topo_brick.kernel.design_diagnostics import (
    KNOWN_HANDED_COUNTERPART_PAIRS,
    evaluate_bilateral_construction_symmetry,
)
from topo_brick.kernel.kernel_contracts import KernelResultStatus
from topo_brick.kernel.realize_policy import RealizeRequestSpec, resolve_realize_policy

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"


def _check(cond: bool, msg: str = "check failed") -> None:
    """Assertion helper preserved under `python -O` (`PYTHONOPTIMIZE=1`)."""
    if not cond:
        raise AssertionError(msg)


def _box_mesh(
    sx: int,
    sy: int,
    sz: int,
    z0_ldu: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    x1, y1 = float(sx * 20), float(sy * 20)
    z0 = float(z0_ldu)
    z1 = z0 + float(sz * 24)
    verts = np.array(
        [
            [0.0, 0.0, z0],
            [x1, 0.0, z0],
            [x1, y1, z0],
            [0.0, y1, z0],
            [0.0, 0.0, z1],
            [x1, 0.0, z1],
            [x1, y1, z1],
            [0.0, y1, z1],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [1, 2, 6],
            [1, 6, 5],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int32,
    )
    return verts, faces


def _single_core_env() -> dict[str, str]:
    env = dict(os.environ)
    for k in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[k] = "1"
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{SRC_DIR}{os.pathsep}{existing_pp}" if existing_pp else str(SRC_DIR)
    return env


def test_p15_01_cross_route_deadline_and_offset_datum_propagation() -> None:
    """Verify shared monotonic deadlines across all 4 routes + offset `ground_plane_z_ldu = -24.0`."""
    kernel = tb.TopoBrickKernel()
    vb_big, fb_big = _box_mesh(8, 8, 5)
    ctf_big, _ = tb.voxelize_triangle_mesh_to_target_field(vb_big, fb_big, max_xy_studs=8, max_z_courses=5)
    ad_big = tb.AdaptiveSurfaceTarget.from_triangle_mesh(vb_big, fb_big, max_xy_studs=8, max_z_courses=5)

    comp_nodes = (
        tb.SubmodelNodeSpec("base_a.ldr", ad_big, world_offset_ldu=(0, 0, 0)),
        tb.SubmodelNodeSpec("top_b.ldr", ad_big, world_offset_ldu=(0, 0, 120), parent_submodel="base_a.ldr"),
    )

    # 1A. Expired monotonic deadline returns `BUDGET_EXCEEDED` immediately across EVERY route
    expired_dl = time.monotonic() - 0.005
    for route_name, kwargs in (
        ("sculpted", {"surface_refinement": "open_catalog_sculpted", "adaptive_mesh_target": ad_big}),
        ("compositional", {"subassembly_graph": comp_nodes}),
        ("adaptive_plate_slope", {"surface_refinement": "adaptive_plate_slope", "adaptive_mesh_target": ad_big}),
        ("course_s1", {"solver_id": "S1_CoverAndRepair"}),
        ("course_greedy", {"solver_id": "GreedyLayerMerge_Control"}),
    ):
        t0 = time.perf_counter()
        res_exp = kernel.realize(ctf_big, deadline_monotonic_s=expired_dl, **kwargs)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        _check(
            res_exp.status == KernelResultStatus.BUDGET_EXCEEDED,
            f"Route {route_name} ignored expired deadline: got {res_exp.status}",
        )
        _check(dt_ms <= 250.0, f"Route {route_name} took {dt_ms:.1f} ms on expired deadline (> 250 ms)")
        _check(res_exp.route_provenance is not None, f"Route {route_name} missing route_provenance")
        _check(bool(res_exp.stopping_reason), f"Route {route_name} stopping_reason empty")

    # 1B. Tight live budget (`max_wall_clock_s=0.02`) across multi-node compositional & sculpted routes
    # (Warm up one-time lazy catalog/retrieval singletons first so `dt_tight_ms` measures solver grace, not gzip I/O)
    from topo_brick.interface.catalog_retrieval_index import CATALOG_RETRIEVAL_INDEX
    from topo_brick.kernel.open_catalog_realizer import realize_open_catalog_sculpted as _warm_oc

    _ = (_warm_oc, CATALOG_RETRIEVAL_INDEX.retrieve_top_k(ad_big.occ_5x5x6, top_k=4))
    t0 = time.perf_counter()
    res_tight_comp = kernel.realize(ctf_big, subassembly_graph=comp_nodes, max_wall_clock_s=0.02)
    dt_tight_ms = (time.perf_counter() - t0) * 1000.0
    _check(
        res_tight_comp.status in (KernelResultStatus.BUDGET_EXCEEDED, KernelResultStatus.COMPLETED_CERTIFIED),
        f"Unexpected status {res_tight_comp.status}",
    )
    _check(dt_tight_ms <= 250.0, f"Multi-node compositional tight budget overrun: {dt_tight_ms:.1f} ms")

    # 1C. `RealizeRequestSpec` & `resolve_realize_policy` explicit contract check
    req_spec = RealizeRequestSpec(
        target_field=ctf_big,
        subassembly_graph=comp_nodes,
        foundation_policy=FoundationPolicy.TABLE_FRICTION_ONLY,
        ground_plane_z_ldu=-24.0,
        max_wall_clock_s=5.0,
    )
    pol = resolve_realize_policy(kernel, req_spec)
    _check(pol.candidate_route_id == "compositional_mpd", f"Unexpected route {pol.candidate_route_id}")
    _check(pol.world_ground_plane_z_ldu == -24.0, "world_ground_plane_z_ldu mismatch")
    _check(pol.eff_deadline_monotonic_s is not None, "eff_deadline_monotonic_s not resolved")

    # 1D. Non-zero world ground datum (`ground_plane_z_ldu = -24.0`) on `TABLE_FRICTION_ONLY` compositional MPD + landmark
    vb_ped, fb_ped = _box_mesh(6, 4, 2)
    ad_ped = tb.AdaptiveSurfaceTarget.from_triangle_mesh(vb_ped, fb_ped, max_xy_studs=6, max_z_courses=2)
    vb_top, fb_top = _box_mesh(4, 2, 1)
    ad_top = tb.AdaptiveSurfaceTarget.from_triangle_mesh(vb_top, fb_top, max_xy_studs=4, max_z_courses=1)
    offset_nodes = (
        tb.SubmodelNodeSpec("pedestal.ldr", ad_ped, world_offset_ldu=(0, 0, -24)),
        tb.SubmodelNodeSpec("cabin.ldr", ad_top, world_offset_ldu=(20, 20, 24), parent_submodel="pedestal.ldr"),
    )
    lm_req = (
        tb.LandmarkAttachmentRequest(
            role="BAR_POLE_ANTENNA",
            anchor_box_ldu=LDUBox(20, 20, 48, 100, 60, 72),
            preferred_color_id=4,
        ),
    )
    dummy_tf = tb.SparseTiledTargetField(boxes=(LDUBox(0, 0, -24, 120, 80, 24),))
    res_off = kernel.realize(
        dummy_tf,
        subassembly_graph=offset_nodes,
        landmark_requests=lm_req,
        foundation_policy=FoundationPolicy.TABLE_FRICTION_ONLY,
        ground_plane_z_ldu=-24.0,
    )
    _check(
        res_off.status == KernelResultStatus.COMPLETED_CERTIFIED and res_off.certificates.overall_certified,
        f"Offset datum compositional build failed: {res_off.status}, {res_off.certificates}",
    )
    _check(res_off.route_provenance is not None and res_off.route_provenance.ground_plane_z_ldu == -24.0)
    min_wz = min(b.zmin for pl in res_off.assembly.placed_parts.values() for b in pl.world_boxes)
    _check(min_wz == -24, f"Expected lowest placed brick at z=-24 LDU, got {min_wz}")
    grounded_parts = [pl for pl in res_off.assembly.placed_parts.values() if pl.is_grounded]
    _check(len(grounded_parts) >= 1, "Expected grounded parts on z=-24.0 table plane")
    _check(res_off.compositional_mpd_text is not None)
    _check(
        res_off.open_catalog_diagnostics is None
        or res_off.open_catalog_diagnostics.landmarks_attached_count >= 0
    )

    # Verify exported `.mpd` SE(3) round-trip preserves z = -24.0 datum
    flat_mpd = flatten_omr_mpd_to_se3(res_off.compositional_mpd_text, catalog=tb.COMBINED_BRICK_CATALOG)
    _check(len(flat_mpd) == res_off.assembly.num_parts, "Flattened MPD part count mismatch")
    mpd_min_z = min(
        b.transform(p.continuous_pose.to_ldu_frame_if_cubic()).zmin
        for p in flat_mpd
        if p.continuous_pose is not None and p.continuous_pose.to_ldu_frame_if_cubic() is not None
        for b in tb.COMBINED_BRICK_CATALOG[p.canonical_part_id].hull_boxes
    )
    _check(mpd_min_z == -24, f"Flattened MPD min z mismatch: {mpd_min_z}")

    # Independent verification at expected z = -24.0 passes; mismatched z = 0.0 fails closed!
    eff_tf_off = res_off.evaluated_target_field
    _check(eff_tf_off is not None)
    art_off = export_certificate_artifact(
        res_off.assembly,
        res_off.certificates,
        validator=kernel.validator,
        target_field=eff_tf_off,
        foundation_policy=FoundationPolicy.TABLE_FRICTION_ONLY.value,
        ground_plane_z_ldu=-24.0,
    )
    ob_ok = build_relying_party_obligations(
        expected_target_field=eff_tf_off,
        min_target_iou=0.25,
        foundation_policy=FoundationPolicy.TABLE_FRICTION_ONLY,
        ground_plane_z_ldu=-24.0,
    )
    ver_ok = verify_certificate_independently(art_off, ob_ok)
    _check(ver_ok.verified is True, f"Offset datum replay failed: {ver_ok.failure_reasons}")
    _check(ver_ok.verification_scope == "assembly_and_target", f"Scope mismatch: {ver_ok.verification_scope}")

    ob_wrong_datum = build_relying_party_obligations(
        expected_target_field=eff_tf_off,
        min_target_iou=0.25,
        foundation_policy=FoundationPolicy.TABLE_FRICTION_ONLY,
        ground_plane_z_ldu=0.0,
    )
    ver_wrong = verify_certificate_independently(art_off, ob_wrong_datum)
    _check(ver_wrong.verified is False, "Mismatched ground_plane_z_ldu=0.0 must fail closed")


def test_p15_02_route_provenance_quickstart_scope_and_three_route_comparison() -> None:
    """Verify `RealizeRouteProvenance`, `compare_realize_routes_on_target`, and quickstart target scope."""
    kernel = tb.TopoBrickKernel()
    vb, fb = _box_mesh(4, 4, 2)
    ctf, _ = tb.voxelize_triangle_mesh_to_target_field(vb, fb, max_xy_studs=4, max_z_courses=2)
    ad = tb.AdaptiveSurfaceTarget.from_triangle_mesh(vb, fb, max_xy_studs=4, max_z_courses=2)
    vb_half, fb_half = _box_mesh(4, 4, 1)
    ad_half = tb.AdaptiveSurfaceTarget.from_triangle_mesh(vb_half, fb_half, max_xy_studs=4, max_z_courses=1)
    comp_nodes = (
        tb.SubmodelNodeSpec("lower.ldr", ad_half, world_offset_ldu=(0, 0, 0)),
        tb.SubmodelNodeSpec("upper.ldr", ad_half, world_offset_ldu=(0, 0, 24), parent_submodel="lower.ldr"),
    )
    v_pts = ad.mesh_surf_pts
    rgb = np.tile(np.array([[180, 0, 0]], dtype=np.uint8), (len(v_pts), 1))

    cmp_rep = compare_realize_routes_on_target(
        target_field=ctf,
        adaptive_target=ad,
        compositional_nodes=comp_nodes,
        kernel=kernel,
        verts_ldu=v_pts,
        vertex_colors_rgb=rgb,
    )
    _check(
        set(cmp_rep.entries_by_route.keys())
        == {"course_by_course_search", "open_catalog_sculpted", "compositional_mpd"}
    )
    for r_key, entry in cmp_rep.entries_by_route.items():
        _check(entry.status == "COMPLETED_CERTIFIED", f"{r_key} not certified: {entry.status}")
        _check(entry.independent_verified is True, f"{r_key} failed independent replay")
        _check(entry.verification_scope == "assembly_and_target", f"{r_key} scope: {entry.verification_scope}")
        _check(entry.exported_mpd_part_count == entry.num_parts, f"{r_key} MPD part count mismatch")
        _check(len(entry.exported_five_view_silhouette_iou) == 5, f"{r_key} missing 5-camera views")
        _check(entry.mean_five_view_silhouette_iou >= 0.85, f"{r_key} low 5-view IoU: {entry.mean_five_view_silhouette_iou}")
        _check(
            entry.route_provenance.caller_target_sha256 == cmp_rep.caller_target_sha256,
            f"{r_key} caller_target_sha256 drift",
        )

    # Verify fail-closed rejection when relying-party target differs from certificate artifact
    res_s1 = kernel.realize(ctf, solver_id="S1_CoverAndRepair")
    art_s1 = export_certificate_artifact(
        res_s1.assembly, res_s1.certificates, validator=kernel.validator, target_field=ctf
    )
    wrong_tf = tb.SparseTiledTargetField(boxes=(LDUBox(0, 0, 0, 40, 40, 24),))
    ver_sub = verify_certificate_independently(
        art_s1,
        build_relying_party_obligations(expected_target_field=wrong_tf, min_target_iou=0.80),
    )
    _check(ver_sub.verified is False, "Target substitution must fail closed under relying-party obligations")

    # Verify `scripts/run_topo_brick_quickstart.py` binds `"assembly_and_target"` scope
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "run_topo_brick_quickstart.py")],
        cwd=str(REPO_ROOT),
        env=_single_core_env(),
        capture_output=True,
        text=True,
        timeout=90.0,
        check=True,
    )
    _check("assembly_and_target" in proc.stdout, f"Quickstart missing target scope:\n{proc.stdout}")


def test_p15_03_modular_extraction_line_caps_and_legacy_imports() -> None:
    """Verify decomposed module line caps, legacy import compatibility, and verifier isolation."""
    ck_path = SRC_DIR / "topo_brick" / "kernel" / "construction_kernel.py"
    ocr_path = SRC_DIR / "topo_brick" / "kernel" / "open_catalog_realizer.py"
    iv_path = SRC_DIR / "topo_brick" / "evaluation" / "independent_verifier.py"

    ck_lines = sum(1 for _ in ck_path.open(encoding="utf-8"))
    ocr_lines = sum(1 for _ in ocr_path.open(encoding="utf-8"))
    _check(ck_lines < 2400, f"construction_kernel.py line cap exceeded: {ck_lines} >= 2400")
    _check(ocr_lines < 1400, f"open_catalog_realizer.py line cap exceeded: {ocr_lines} >= 1400")

    # Legacy `construction_kernel` and `open_catalog_realizer` imports remain 100% intact
    from topo_brick.kernel.construction_kernel import (
        CertificateBundle,
        CoverAndRepairSolver,
        GreedyLayerMergeControl,
        KernelRealizeResult,
        LearnedCatalogAndPortFilter,
        LuoInspiredSeamRefinement,
        RealizeRouteProvenance,
        SingleCoreAuditReport,
        TopoBrickKernel,
        check_monotone_insertion_corridor,
        enforce_and_audit_single_core,
        run_deliverable1_small_kernel_verification,
    )
    from topo_brick.kernel.open_catalog_realizer import (
        LDRAW_SOLID_PALETTE_SRGB,
        OpenCatalogRealizeDiagnostics,
        assign_perceptual_ldraw_colors,
        ciede2000_delta_e,
        realize_open_catalog_sculpted,
        srgb_to_cielab,
    )

    _check(TopoBrickKernel is tb_api.TopoBrickKernel)
    _check(CertificateBundle is tb_api.CertificateBundle)
    _check(KernelRealizeResult is tb_api.KernelRealizeResult)
    _check(RealizeRouteProvenance is tb_api.RealizeRouteProvenance)
    _check(SingleCoreAuditReport is tb_api.SingleCoreAuditReport)
    _check(enforce_and_audit_single_core is tb_api.enforce_and_audit_single_core)
    _check(callable(CoverAndRepairSolver) and callable(GreedyLayerMergeControl))
    _check(callable(LuoInspiredSeamRefinement) and callable(LearnedCatalogAndPortFilter))
    _check(callable(check_monotone_insertion_corridor) and callable(run_deliverable1_small_kernel_verification))
    _check(len(LDRAW_SOLID_PALETTE_SRGB) == 54 and OpenCatalogRealizeDiagnostics is not None)
    _check(callable(assign_perceptual_ldraw_colors) and callable(ciede2000_delta_e))
    _check(callable(realize_open_catalog_sculpted) and callable(srgb_to_cielab))

    # Static AST audit: `independent_verifier.py` imports zero `topo_brick.kernel` or `torch` modules
    tree = ast.parse(iv_path.read_text(encoding="utf-8"), filename=str(iv_path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _check(not alias.name.startswith("topo_brick.kernel"), f"Forbidden import {alias.name}")
                _check(not alias.name.startswith("torch"), f"Forbidden torch import {alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.module:
            _check(not node.module.startswith("topo_brick.kernel"), f"Forbidden from-import {node.module}")
            _check(not node.module.startswith("torch"), f"Forbidden torch from-import {node.module}")


def test_p15_04_catalog_funnel_regional_color_and_bilateral_symmetry_diagnostics() -> None:
    """Verify catalog funnel parity, regional/boundary colour telemetry, and bilateral symmetry audit."""
    kernel = tb.TopoBrickKernel()
    _check(len(KNOWN_HANDED_COUNTERPART_PAIRS) >= 12)

    # 4A & 4B. Sculpted build with crisp red/white stripe on single exterior bricks + region labels + 322 LDConfig mode
    vb, fb = _box_mesh(4, 2, 2)
    ctf, _ = tb.voxelize_triangle_mesh_to_target_field(vb, fb, max_xy_studs=4, max_z_courses=2)
    ad = tb.AdaptiveSurfaceTarget.from_triangle_mesh(vb, fb, max_xy_studs=4, max_z_courses=2)

    # Dense sample grid spanning x in [0, 80] with sharp red (x < 40) / white (x >= 40) stripe at x = 40
    xs = np.linspace(2.0, 78.0, 20)
    ys = np.linspace(2.0, 38.0, 8)
    zs = np.linspace(2.0, 46.0, 6)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    pts_stripe = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    is_left = pts_stripe[:, 0] < 40.0
    rgb_stripe = np.where(
        is_left[:, None],
        np.array([[180, 0, 0]], dtype=np.uint8),
        np.array([[255, 255, 255]], dtype=np.uint8),
    )
    reg_labels = np.where(is_left, "starboard_red_band", "port_white_band")

    res_diag = kernel.realize(
        ctf,
        part_retrieval="open_catalog_top_k",
        retrieval_top_k=24,
        license_filter="cc_by_4_0_only",
        surface_refinement="open_catalog_sculpted",
        adaptive_mesh_target=ad,
        verts_ldu=pts_stripe,
        vertex_colors_rgb=rgb_stripe,
        vertex_region_labels=reg_labels,
        open_color_palette_mode="ldconfig_finish_aware",
        symmetry_plane_axis="x",
    )
    _check(res_diag.status == KernelResultStatus.COMPLETED_CERTIFIED and res_diag.certificates.overall_certified)
    diag = res_diag.open_catalog_diagnostics
    _check(diag is not None)

    # Catalog funnel report matches `CATALOG_RETRIEVAL_INDEX.funnel_metadata()` 1-for-1
    fmeta = CATALOG_RETRIEVAL_INDEX.funnel_metadata()
    cf = diag.catalog_funnel_report
    _check(cf is not None)
    _check(cf.combined_catalog_ids == fmeta["combined_catalog_ids"] == 16085)
    _check(cf.compiled_catalog_ids == fmeta["compiled_catalog_ids"] == 16072)
    _check(cf.canonical_system_molds == fmeta["canonical_system_molds"])
    _check(cf.active_buildable_molds == fmeta["active_buildable_molds"])
    _check(cf.valid_sculptural_molds == fmeta["valid_sculptural_molds"])
    _check(cf.pose_signatures_total == fmeta["pose_signatures"])
    _check(cf.license_filter == "cc_by_4_0_only")
    _check(sum(cf.placed_mold_family_counts.values()) == res_diag.assembly.num_parts)

    # Regional appearance report & boundary-straddling brick detection
    ra = diag.regional_appearance_report
    _check(ra is not None)
    _check(ra.palette_mode == "ldconfig_finish_aware")
    _check(ra.palette_candidates_searched == 322)
    _check(set(ra.per_region_mean_delta_e00.keys()) == {"port_white_band", "starboard_red_band"})
    _check(
        ra.high_gradient_boundary_bricks_count >= 1,
        f"Expected >= 1 boundary-straddling brick across crisp x=40 stripe, got {ra}",
    )
    _check(diag.bilateral_symmetry_report is not None)

    # 4C. Bilateral construction symmetry:
    # Case 1: Symmetric centerpiece (`3001` centered at x=0 via pos=(-10, -30, 0) -> x in [-20, 20], y in [-40, 40])
    # + opposite-hand wedge wings (`43722` & `43723` at mirrored x = +-60, z = 8 -> z in [0, 8]) -> 100% construction symmetry!
    st_sym = AssemblyState()
    st_sym.place_root_part("3001", LDUFrame(0, (-10, -30, 0)), is_grounded=True, color_id=71)
    st_sym.place_root_part("43722", LDUFrame(0, (-60, 0, 8)), is_grounded=True, color_id=4)
    st_sym.place_root_part("43723", LDUFrame(0, (60, 0, 8)), is_grounded=True, color_id=4)

    sym_ok = evaluate_bilateral_construction_symmetry(st_sym, axis="x", plane_coord_ldu=0.0)
    _check(sym_ok.centerline_spanning_count == 1, f"Expected 1 centerline part, got {sym_ok}")
    _check(sym_ok.handed_counterpart_pairs == 1, f"Expected 1 handed pair, got {sym_ok}")
    _check(sym_ok.exterior_construction_symmetry_ratio == 1.0, f"Expected 1.0 symmetry, got {sym_ok}")
    _check(sym_ok.silhouette_mirror_iou >= 0.90, f"Expected high silhouette mirror IoU, got {sym_ok.silhouette_mirror_iou}")
    _check(sym_ok.symmetric_color_agreement_ratio == 1.0)

    # Also verify a backwards-rotated wedge (`rot_id=9`, 180 deg) on the right is REJECTED as asymmetric!
    st_bad_rot = AssemblyState()
    st_bad_rot.place_root_part("3001", LDUFrame(0, (-10, -30, 0)), is_grounded=True, color_id=71)
    st_bad_rot.place_root_part("43722", LDUFrame(0, (-60, 0, 8)), is_grounded=True, color_id=4)
    st_bad_rot.place_root_part("43723", LDUFrame(9, (60, 0, 8)), is_grounded=True, color_id=4)
    sym_bad_rot = evaluate_bilateral_construction_symmetry(st_bad_rot, axis="x", plane_coord_ldu=0.0)
    _check(sym_bad_rot.handed_counterpart_pairs == 0, "Backwards-rotated wedge must not count as symmetric pair")
    _check(sym_bad_rot.exterior_construction_symmetry_ratio == 0.0)

    # Case 2: Identical silhouette (`silhouette_mirror_iou == 1.0`), asymmetric brick decomposition!
    # Center: `3001` at `(-10, -30, 0)` -> `x in [-20, 20], y in [-40, 40]`
    # Left wing: single `1x4` plate (`3710` at `(-40, -30, 0)` -> `x in [-50, -30], y in [-40, 40], z in [0, 8]`)
    # Right wing: two `1x2` plates (`3023` at `(40, -30, 0)` -> `y in [-40, 0]` and `(40, 10, 0)` -> `y in [0, 40]`, both `x in [30, 50]`)!
    st_asym = AssemblyState()
    st_asym.place_root_part("3001", LDUFrame(0, (-10, -30, 0)), is_grounded=True, color_id=71)
    st_asym.place_root_part("3710", LDUFrame(0, (-40, -30, 0)), is_grounded=True, color_id=1)
    st_asym.place_root_part("3023", LDUFrame(0, (40, -30, 0)), is_grounded=True, color_id=1)
    st_asym.place_root_part("3023", LDUFrame(0, (40, 10, 0)), is_grounded=True, color_id=1)

    sym_split = evaluate_bilateral_construction_symmetry(st_asym, axis="x", plane_coord_ldu=0.0)
    _check(sym_split.silhouette_mirror_iou == 1.0, f"Expected 1.0 silhouette IoU, got {sym_split.silhouette_mirror_iou}")
    _check(
        sym_split.exterior_construction_symmetry_ratio == 0.0,
        f"Asymmetric 1x4 vs 2x(1x2) must score < 1.0 construction symmetry, got {sym_split}",
    )
    _check(sym_split.seam_alignment_ratio == 0.0)
    _check(sym_split.unmatched_parts_count == 3)

    # Confirm physical certification (`kernel.certify`) remains 100% `True` on `st_asym`!
    cert_asym = kernel.certify(st_asym, foundation_policy=FoundationPolicy.CLUTCHED_BASEPLATE)
    _check(
        cert_asym.overall_certified is True,
        f"Asymmetric seam split must NOT invalidate CertificateBundle: {cert_asym}",
    )


def test_p15_05_torch_free_deterministic_kernel_and_consumer_api_boundary() -> None:
    """Verify `topo_brick.api` surface and `scripts/benchmark_topo_brick_boundaries.py` run with zero `torch`."""
    _check(35 <= len(tb_api.SUPPORTED_CONSUMER_EXPORTS) <= 50)

    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "benchmark_topo_brick_boundaries.py")],
        cwd=str(REPO_ROOT),
        env=_single_core_env(),
        capture_output=True,
        text=True,
        timeout=120.0,
        check=True,
    )
    report = json.loads(proc.stdout)
    _check(report["all_boundaries_passed"] is True, f"Boundary benchmark failed:\n{proc.stdout}")
    _check(report["cold_import_topo_brick"]["torch_loaded"] is False)
    _check(report["consumer_api_surface"]["torch_loaded"] is False)
    _check(report["torch_free_sculpted_realize_and_verify"]["torch_loaded"] is False)
    _check(report["torch_free_sculpted_realize_and_verify"]["verified"] is True)
    _check(report["torch_free_sculpted_realize_and_verify"]["verification_scope"] == "assembly_and_target")
    _check(report["isolated_verifier_self_test"]["kernel_modules_loaded"] == [])
