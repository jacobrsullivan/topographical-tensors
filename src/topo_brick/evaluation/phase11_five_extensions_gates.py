"""
Phase 11 (`v5.2.0`): Five Extensions to `topo_brick` for Constrained Generative & Multimodal Synthesis
Verification & Promotion Gate Suite (`Gates 9A, 9B, 9C, 9D`).

Implements strict single-core (`OMP_NUM_THREADS=1`) falsifiable acceptance benchmarks:
- `Gate 9A` (`run_gate9a_colored_finite_inventory_and_sparse_kit_benchmark`):
  * `n = 40` colored 3D target meshes realized strictly under retail set `10698-1`
    (`SET_10698_COLORED_INVENTORY`: `790` pieces across `33` LDraw colors) + `n = 20` adversarial
    zero-`1x1` odd-parity finite-inventory targets + `Schema v1.1.0` tampered color-overrun mutant
    rejection (`100%`) + `InventoryShortfallReport` infeasibility diagnosis.
- `Gate 9B` (`run_gate9b_ribbed_hollow_shell_scaling_benchmark`):
  * `n = 30` medium/large architectural & statue meshes (`16x16x10` to `24x24x16` studs) paired
    `HollowShellPolicy(enabled=True)` vs. solid fill under `FoundationPolicy.TABLE_FRICTION_ONLY`,
    verifying `100%` certified + `independent_verifier` pass (`0` temp supports), `>= 45%` mean
    brick/volume reduction, and `|exterior_crust_iou_5x5x6_hollow - solid| <= 0.015`.
- `Gate 9C` (`run_gate9c_compositional_mpd_and_landmarks_benchmark`):
  * `n = 25` multi-submodel `.mpd` architectural scenes across all `8` `LANDMARK_DETAIL_ROLES`
    under both `license_filter="all"` and `"cc_by_4_0_only"`, verifying `100%` recursive
    `PLACE_SUBASSEMBLY` `subtrace` replay + `100%` tampered `subtrace` mutant rejection + `100%`
    exact `SE(3)` `.mpd` round-trip via `flatten_omr_mpd_to_se3` + `0` `CC-BY-SA-4.0` leaks.
- `Gate 9D` (`run_gate9d_view_prioritized_scarcity_study`):
  * Paired ablation (`ViewPrioritySpec` ON vs. OFF) across `n = 30` scarce-color (`30%` matching
    budget) single-view sculpted/relief mesh targets under `FoundationPolicy.TABLE_FRICTION_ONLY`,
    verifying `100%` certified + `>= +0.04` `silhouette_iou_2d` lift + `>= 4.0` `ΔE_00` reduction.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from topo_brick.evaluation.independent_verifier import (
    CERTIFICATE_SCHEMA_VERSION,
    compute_catalog_sha256,
    export_certificate_artifact,
    verify_certificate_independently,
)
from topo_brick.evaluation.phase7_sota_benchmarks import voxelize_triangle_mesh_to_target_field
from topo_brick.evaluation.phase8_round9_gates import (
    SingleCoreAuditReport,
    enforce_and_audit_single_core,
)
from topo_brick.evaluation.static_equilibrium import FoundationPolicy
from topo_brick.interface.brick_catalog import (
    EXTENDED_BRICK_CATALOG,
    SET_10698_COLORED_INVENTORY,
    ColoredInventorySpec,
    ColoredPartKey,
)
from topo_brick.interface.catalog_retrieval_index import CATALOG_RETRIEVAL_INDEX
from topo_brick.interface.landmark_detail_index import (
    LANDMARK_DETAIL_INDEX,
    LANDMARK_DETAIL_ROLES,
)
from topo_brick.interface.ldraw_shadow_catalog_compiler import COMBINED_BRICK_CATALOG
from topo_brick.interface.ldu_frames import LDUBox
from topo_brick.interface.se3_kinematics import flatten_omr_mpd_to_se3
from topo_brick.kernel.adaptive_surface_and_macro_planner import (
    AdaptiveSurfaceTarget,
    HollowShellPolicy,
    evaluate_exterior_crust_fidelity,
)
from topo_brick.kernel.construction_kernel import (
    CertStatus,
    KernelResultStatus,
    TopoBrickKernel,
)
from topo_brick.kernel.subassembly_realizer import (
    LandmarkAttachmentRequest,
    SubmodelNodeSpec,
)
from topo_brick.kernel.view_priority import (
    ViewPrioritySpec,
    evaluate_camera_reprojection_fidelity,
)


# ---------------------------------------------------------------------------
# Report Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Gate9AColoredInventoryReport:
    """Audit report for `Gate 9A`: Colored Finite Inventory & Sparse-Kit Synthesis (`n = 60`)."""

    single_core_audit: SingleCoreAuditReport
    catalog_sha256_v110: str
    set_10698_num_targets: int
    set_10698_certified_count: int
    set_10698_verifier_passed_count: int
    set_10698_color_overrun_count: int
    set_10698_skin_mean_delta_e00: float
    set_10698_fifo_skin_mean_delta_e00: float
    set_10698_skin_delta_e00_reduction_ratio: float
    set_10698_total_core_parts_hidden: int
    set_10698_total_chromatic_splits: int
    sparse_odd_parity_num_targets: int
    sparse_odd_parity_certified_count: int
    sparse_odd_parity_verifier_passed_count: int
    sparse_odd_parity_one_by_one_emitted: int
    total_targets: int
    total_certified_count: int
    total_verifier_passed_count: int
    tampered_color_mutants_tested: int
    tampered_color_mutants_rejected: int
    shortfall_infeasible_detected: bool
    shortfall_missing_volume_ldu3: int
    realize_p50_wall_ms: float
    realize_p90_wall_ms: float
    wall_clock_ms: float
    gate9a_passed: bool
    evidence_kind: str = "rebrickable_10698_1_exact_manifest_plus_synthetic_chromatic_and_odd_parity_targets"


@dataclass(frozen=True, slots=True)
class Gate9BHollowShellReport:
    """Audit report for `Gate 9B`: Ribbed Hollow-Shell Scaling (`n = 30`, `TABLE_FRICTION_ONLY`)."""

    single_core_audit: SingleCoreAuditReport
    num_targets: int
    hollow_certified_count: int
    solid_certified_count: int
    hollow_insertion_pass_count: int
    hollow_temp_supports_emitted: int
    independent_verifier_passed_count: int
    mean_solid_parts: float
    mean_hollow_parts: float
    mean_brick_reduction_ratio: float
    mean_volume_savings_ratio: float
    mean_solid_crust_iou_5x5x6: float
    mean_hollow_crust_iou_5x5x6: float
    mean_abs_crust_iou_diff: float
    max_abs_crust_iou_diff: float
    hollow_p50_wall_ms: float
    wall_clock_ms: float
    gate9b_passed: bool
    evidence_kind: str = "synthetic_closed_mesh_hollow_shell_benchmark_table_friction_only"


@dataclass(frozen=True, slots=True)
class Gate9CCompositionalMPDReport:
    """Audit report for `Gate 9C`: Compositional Multi-Block `.mpd` & 8-Role Landmarks (`n = 25`)."""

    single_core_audit: SingleCoreAuditReport
    num_scenes: int
    all_license_scenes: int
    cc_by_4_0_scenes: int
    certified_count: int
    independent_verifier_passed_count: int
    mpd_se3_roundtrip_exact_count: int
    mpd_max_position_error_ldu: float
    cc_by_4_0_share_alike_leaks: int
    landmark_roles_exercised_count: int
    landmark_roles_exercised: Tuple[str, ...]
    total_landmarks_attached: int
    tampered_subtrace_mutants_tested: int
    tampered_subtrace_mutants_rejected: int
    wall_clock_ms: float
    gate9c_passed: bool
    evidence_kind: str = "synthetic_compositional_mpd_submodel_and_8role_landmark_scenes"


@dataclass(frozen=True, slots=True)
class Gate9DViewPriorityReport:
    """Audit report for `Gate 9D`: View-Prioritized Scarcity Study (`n = 30`, `TABLE_FRICTION_ONLY`)."""

    single_core_audit: SingleCoreAuditReport
    num_targets: int
    view_on_certified_count: int
    view_off_certified_count: int
    independent_verifier_passed_count: int
    mean_silhouette_iou_2d_off: float
    mean_silhouette_iou_2d_on: float
    mean_silhouette_iou_2d_lift: float
    mean_visible_delta_e00_off: float
    mean_visible_delta_e00_on: float
    mean_visible_delta_e00_reduction: float
    mean_occluded_rear_delta_e00_on: float
    wall_clock_ms: float
    gate9d_passed: bool
    evidence_kind: str = "synthetic_camera_view_prioritized_scarcity_ablation"


@dataclass(frozen=True, slots=True)
class Phase11CompleteGateSummary:
    """Combined Phase 11 (`v5.2.0`) verification ledger across `Gates 9A, 9B, 9C, 9D`."""

    schema_version: str
    catalog_sha256_v110: str
    landmark_index_sha256: str
    gate9a: Gate9AColoredInventoryReport
    gate9b: Gate9BHollowShellReport
    gate9c: Gate9CCompositionalMPDReport
    gate9d: Gate9DViewPriorityReport
    all_phase11_gates_passed: bool


# ---------------------------------------------------------------------------
# Deterministic Parametric Mesh Builders (Watertight Closed Triangle Meshes)
# ---------------------------------------------------------------------------


def _append_box_mesh(
    verts_list: List[np.ndarray],
    faces_list: List[np.ndarray],
    colors_list: List[np.ndarray],
    x0: float,
    y0: float,
    z0: float,
    x1: float,
    y1: float,
    z1: float,
    rgb: Tuple[float, float, float] = (0.85, 0.85, 0.85),
) -> None:
    """Append an axis-aligned watertight cuboid (with dense surface vertices for color interpolation)."""
    nx = max(2, int(round((x1 - x0) / 20.0)) + 1)
    ny = max(2, int(round((y1 - y0) / 20.0)) + 1)
    nz = max(2, int(round((z1 - z0) / 24.0)) + 1)
    base_idx = sum(v.shape[0] for v in verts_list)

    # 8 corners for exact watertight triangle faces + surface grid points for color K-D tree fidelity
    corners = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=np.float64,
    )
    tri = (
        np.array(
            [
                [0, 2, 1],
                [0, 3, 2],
                [4, 5, 6],
                [4, 6, 7],
                [0, 1, 5],
                [0, 5, 4],
                [2, 3, 7],
                [2, 7, 6],
                [0, 4, 7],
                [0, 7, 3],
                [1, 2, 6],
                [1, 6, 5],
            ],
            dtype=np.int64,
        )
        + base_idx
    )

    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    zs = np.linspace(z0, z1, nz)
    extra_pts: List[List[float]] = []
    for xv in xs:
        for yv in ys:
            extra_pts.append([float(xv), float(yv), float(z0)])
            extra_pts.append([float(xv), float(yv), float(z1)])
    for xv in xs:
        for zv in zs:
            extra_pts.append([float(xv), float(y0), float(zv)])
            extra_pts.append([float(xv), float(y1), float(zv)])
    for yv in ys:
        for zv in zs:
            extra_pts.append([float(x0), float(yv), float(zv)])
            extra_pts.append([float(x1), float(yv), float(zv)])

    v_block = np.vstack([corners, np.asarray(extra_pts, dtype=np.float64)])
    c_block = np.tile(np.asarray(rgb, dtype=np.float64), (v_block.shape[0], 1))
    verts_list.append(v_block)
    faces_list.append(tri)
    colors_list.append(c_block)


def _append_gable_prism_mesh(
    verts_list: List[np.ndarray],
    faces_list: List[np.ndarray],
    colors_list: List[np.ndarray],
    x0: float,
    y0: float,
    z0: float,
    x1: float,
    y1: float,
    z_peak: float,
    along_x: bool = True,
    rgb: Tuple[float, float, float] = (0.78, 0.12, 0.10),
) -> None:
    """Append a watertight pitched gable roof prism."""
    base_idx = sum(v.shape[0] for v in verts_list)
    if along_x:
        ym = 0.5 * (y0 + y1)
        pts = np.array(
            [
                [x0, y0, z0],
                [x1, y0, z0],
                [x1, y1, z0],
                [x0, y1, z0],
                [x0, ym, z_peak],
                [x1, ym, z_peak],
            ],
            dtype=np.float64,
        )
    else:
        xm = 0.5 * (x0 + x1)
        pts = np.array(
            [
                [x0, y0, z0],
                [x1, y0, z0],
                [x1, y1, z0],
                [x0, y1, z0],
                [xm, y0, z_peak],
                [xm, y1, z_peak],
            ],
            dtype=np.float64,
        )
    if along_x:
        tri = (
            np.array(
                [
                    [0, 2, 1],
                    [0, 3, 2],
                    [0, 1, 5],
                    [0, 5, 4],
                    [2, 3, 4],
                    [2, 4, 5],
                    [0, 4, 3],
                    [1, 2, 5],
                ],
                dtype=np.int64,
            )
            + base_idx
        )
    else:
        tri = (
            np.array(
                [
                    [0, 2, 1],
                    [0, 3, 2],
                    [0, 4, 5],
                    [0, 5, 3],
                    [1, 2, 5],
                    [1, 5, 4],
                    [0, 1, 4],
                    [3, 5, 2],
                ],
                dtype=np.int64,
            )
            + base_idx
        )
    # Dense samples on roof slopes for accurate color lookup
    ts = np.linspace(0.0, 1.0, 5)
    us = np.linspace(0.0, 1.0, 5)
    slope_pts: List[np.ndarray] = []
    for tv in ts:
        for uv in us:
            if along_x:
                xv = x0 + uv * (x1 - x0)
                slope_pts.append(np.array([xv, y0 + tv * (ym - y0), z0 + tv * (z_peak - z0)]))
                slope_pts.append(np.array([xv, y1 + tv * (ym - y1), z0 + tv * (z_peak - z0)]))
            else:
                yv = y0 + uv * (y1 - y0)
                slope_pts.append(np.array([x0 + tv * (xm - x0), yv, z0 + tv * (z_peak - z0)]))
                slope_pts.append(np.array([x1 + tv * (xm - x1), yv, z0 + tv * (z_peak - z0)]))
    v_all = np.vstack([pts, np.asarray(slope_pts, dtype=np.float64)])
    c_all = np.tile(np.asarray(rgb, dtype=np.float64), (v_all.shape[0], 1))
    verts_list.append(v_all)
    faces_list.append(tri)
    colors_list.append(c_all)


def _finalize_mesh(
    verts_list: List[np.ndarray],
    faces_list: List[np.ndarray],
    colors_list: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.vstack(verts_list).astype(np.float64),
        np.vstack(faces_list).astype(np.int64),
        np.vstack(colors_list).astype(np.float64),
    )


# ---------------------------------------------------------------------------
# Gate 9A: Colored Finite Inventory (`10698-1`) & Odd-Parity Sparse Kits
# ---------------------------------------------------------------------------

# Primary palette sRGB tuples present in SET_10698_COLORED_INVENTORY (so exterior skin has low-ΔE matches,
# while clashing pink/lime/purple/orange stock in 10698-1 must be routed into OCCLUDED_CORE!)
_GATE9A_PRIMARY_PALETTES: Tuple[Tuple[Tuple[float, float, float], Tuple[float, float, float]], ...] = (
    ((0.95, 0.95, 0.95), (0.77, 0.10, 0.09)),  # White walls + Red roof/trim
    ((0.95, 0.80, 0.12), (0.02, 0.33, 0.75)),  # Yellow body + Blue trim
    ((0.63, 0.65, 0.66), (0.42, 0.43, 0.43)),  # Light/Dark Bluish Gray keep
    ((0.89, 0.80, 0.62), (0.35, 0.16, 0.07)),  # Tan temple + Reddish Brown cap
    ((0.14, 0.47, 0.25), (0.95, 0.95, 0.95)),  # Green garden pavilion + White cap
    ((0.02, 0.33, 0.75), (0.95, 0.95, 0.95)),  # Blue tower + White battlements
    ((0.77, 0.10, 0.09), (0.95, 0.80, 0.12)),  # Red citadel + Yellow crown
    ((0.42, 0.43, 0.43), (0.77, 0.10, 0.09)),  # Dark Gray fortress + Red roof
)


def _build_gate9a_colored_mesh(idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Build deterministic colored 3D mesh `idx in 0..39` with rich interior core + two-tone exterior."""
    wall_rgb, roof_rgb = _GATE9A_PRIMARY_PALETTES[idx % len(_GATE9A_PRIMARY_PALETTES)]
    archetype = idx % 5
    vl: List[np.ndarray] = []
    fl: List[np.ndarray] = []
    cl: List[np.ndarray] = []

    if archetype == 0:
        # Gabled cottage (`6x6x4` or `8x6x4` studs -> thick interior core + sloped gable)
        wx = 6 if (idx // 5) % 2 == 0 else 8
        wy = 6
        wz = 4
        _append_box_mesh(vl, fl, cl, 0.0, 0.0, 0.0, wx * 20.0, wy * 20.0, 72.0, wall_rgb)
        _append_gable_prism_mesh(
            vl, fl, cl, 0.0, 0.0, 72.0, wx * 20.0, wy * 20.0, 96.0, along_x=(idx % 2 == 0), rgb=roof_rgb
        )
    elif archetype == 1:
        # Stepped ziggurat monument (`6x6x4` studs)
        wx, wy, wz = 6, 6, 4
        _append_box_mesh(vl, fl, cl, 0.0, 0.0, 0.0, 120.0, 120.0, 48.0, wall_rgb)
        _append_box_mesh(vl, fl, cl, 20.0, 20.0, 48.0, 100.0, 100.0, 96.0, roof_rgb)
    elif archetype == 2:
        # Watchtower keep with upper crown (`6x6x5` studs)
        wx, wy, wz = 6, 6, 5
        _append_box_mesh(vl, fl, cl, 0.0, 0.0, 0.0, 120.0, 120.0, 96.0, wall_rgb)
        _append_box_mesh(vl, fl, cl, 20.0, 20.0, 96.0, 100.0, 100.0, 120.0, roof_rgb)
    elif archetype == 3:
        # L-shaped manor wing with thick corner block (`8x6x4` studs)
        wx, wy, wz = 8, 6, 4
        _append_box_mesh(vl, fl, cl, 0.0, 0.0, 0.0, 160.0, 80.0, 72.0, wall_rgb)
        _append_box_mesh(vl, fl, cl, 0.0, 0.0, 0.0, 80.0, 120.0, 72.0, wall_rgb)
        _append_box_mesh(vl, fl, cl, 0.0, 0.0, 72.0, 80.0, 80.0, 96.0, roof_rgb)
    else:
        # Twin-buttress shrine (`8x6x4` studs)
        wx, wy, wz = 8, 6, 4
        _append_box_mesh(vl, fl, cl, 0.0, 0.0, 0.0, 160.0, 120.0, 48.0, wall_rgb)
        _append_box_mesh(vl, fl, cl, 20.0, 20.0, 48.0, 140.0, 100.0, 72.0, wall_rgb)
        _append_gable_prism_mesh(vl, fl, cl, 20.0, 20.0, 72.0, 140.0, 100.0, 96.0, along_x=True, rgb=roof_rgb)

    v, f, c = _finalize_mesh(vl, fl, cl)
    return v, f, c, wx, wz


def _build_zero_1x1_odd_parity_target(idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, ColoredInventorySpec]:
    """Build odd-parity (`5x7`, `7x5`, `7x7`, `5x9`, `9x5`) target + strict zero-`1x1` `ColoredInventorySpec`."""
    dims = ((5, 7, 3), (7, 5, 3), (7, 7, 3), (5, 9, 3), (9, 5, 3))[idx % 5]
    wx, wy, wz = dims
    vl: List[np.ndarray] = []
    fl: List[np.ndarray] = []
    cl: List[np.ndarray] = []
    # Odd-dimensioned footprints (`5x7`, `7x7`, `5x9`) cannot be tiled with even-parity `2x2`/`2x4` alone unless
    # parity-aware morphological snap or alternating `1x2`/`2x2` MILP packing resolves the odd strip!
    if idx % 2 == 0:
        _append_box_mesh(vl, fl, cl, 0.0, 0.0, 0.0, wx * 20.0, wy * 20.0, wz * 24.0, (0.95, 0.95, 0.95))
    else:
        _append_box_mesh(vl, fl, cl, 0.0, 0.0, 0.0, wx * 20.0, wy * 20.0, 48.0, (0.95, 0.95, 0.95))
        _append_box_mesh(vl, fl, cl, 20.0, 20.0, 48.0, (wx - 1) * 20.0, (wy - 1) * 20.0, 72.0, (0.77, 0.10, 0.09))
    v, f, c = _finalize_mesh(vl, fl, cl)

    # Zero `1x1` bricks (`3005=0`) and zero `1x1` plates (`3024=0`); strictly `2x4`, `2x2`, `1x4`, `1x2`
    counts: Dict[ColoredPartKey, int] = {
        ("3001", 15): 24,
        ("3001", 4): 12,
        ("3001", 28): 12,
        ("3003", 15): 24,
        ("3003", 4): 12,
        ("3010", 15): 20,
        ("3004", 15): 36,
        ("3004", 4): 18,
        ("3020", 15): 16,
        ("3022", 15): 16,
        ("3023", 15): 24,
    }
    return v, f, c, max(wx, wy), wz, ColoredInventorySpec(counts=counts)


def run_gate9a_colored_finite_inventory_and_sparse_kit_benchmark(
    n_set_10698: int = 40,
    n_odd_parity: int = 20,
    max_p50_wall_ms: float = 250.0,
) -> Gate9AColoredInventoryReport:
    """Execute `Gate 9A` (`Ext 1 & 2`) under strict `OMP_NUM_THREADS=1` single-core auditing."""
    t0_suite = time.perf_counter()
    sc_audit = enforce_and_audit_single_core()
    kernel = TopoBrickKernel()
    cat_sha = compute_catalog_sha256(CERTIFICATE_SCHEMA_VERSION)

    s10698_cert = 0
    s10698_iv = 0
    s10698_overruns = 0
    skin_de_list: List[float] = []
    fifo_de_list: List[float] = []
    tot_core_hidden = 0
    tot_splits = 0
    wall_times_ms: List[float] = []
    sample_valid_artifacts: List[Any] = []

    for i in range(n_set_10698):
        v, f, c, max_xy, max_z = _build_gate9a_colored_mesh(i)
        tf, _ = voxelize_triangle_mesh_to_target_field(v, f, max_xy_studs=max_xy, max_z_courses=max_z)
        ast = AdaptiveSurfaceTarget.from_triangle_mesh(v, f, max_xy_studs=max_xy, max_z_courses=max_z)
        t_r0 = time.perf_counter()
        res = kernel.realize(
            tf,
            adaptive_mesh_target=ast,
            part_retrieval="open_catalog_top_k",
            colored_inventory=SET_10698_COLORED_INVENTORY,
            verts_ldu=v,
            vertex_colors_rgb=c,
            object_id=f"gate9a_10698_{i:02d}",
        )
        wall_times_ms.append((time.perf_counter() - t_r0) * 1000.0)

        if res.status == KernelResultStatus.COMPLETED_CERTIFIED and res.certificates.overall_certified:
            s10698_cert += 1
        # Audit `(part_id, color_id)` stock strictly
        used_pc: Dict[Tuple[str, int], int] = {}
        for pl in res.assembly.placed_parts.values():
            key_pc = (pl.part_id, int(pl.color_id))
            used_pc[key_pc] = used_pc.get(key_pc, 0) + 1
        for (pid_u, cid_u), cnt_u in used_pc.items():
            if cnt_u > SET_10698_COLORED_INVENTORY.get_count(pid_u, cid_u):
                s10698_overruns += 1

        art = export_certificate_artifact(
            res.assembly,
            res.certificates,
            validator=kernel.validator,
            colored_inventory=SET_10698_COLORED_INVENTORY,
        )
        iv_rep = verify_certificate_independently(art)
        if iv_rep.verified and art.schema_version == "1.1.0":
            s10698_iv += 1
            if len(sample_valid_artifacts) < 10:
                sample_valid_artifacts.append(art)

        inv_cert = res.certificates.inventory
        skin_de_list.append(float(inv_cert.skin_mean_delta_e00))
        fifo_de_list.append(float(inv_cert.fifo_baseline_skin_mean_delta_e00))
        tot_core_hidden += int(inv_cert.core_parts_count)
        tot_splits += int(inv_cert.chromatic_splits_accepted)

    odd_cert = 0
    odd_iv = 0
    odd_1x1_emitted = 0
    for j in range(n_odd_parity):
        v_o, f_o, c_o, mxy_o, mz_o, odd_inv = _build_zero_1x1_odd_parity_target(j)
        tf_o, _ = voxelize_triangle_mesh_to_target_field(v_o, f_o, max_xy_studs=mxy_o, max_z_courses=mz_o)
        ast_o = AdaptiveSurfaceTarget.from_triangle_mesh(v_o, f_o, max_xy_studs=mxy_o, max_z_courses=mz_o)
        t_r0 = time.perf_counter()
        res_o = kernel.realize(
            tf_o,
            adaptive_mesh_target=ast_o,
            part_retrieval="open_catalog_top_k",
            colored_inventory=odd_inv,
            verts_ldu=v_o,
            vertex_colors_rgb=c_o,
            object_id=f"gate9a_odd_{j:02d}",
        )
        wall_times_ms.append((time.perf_counter() - t_r0) * 1000.0)

        if res_o.status == KernelResultStatus.COMPLETED_CERTIFIED and res_o.certificates.overall_certified:
            odd_cert += 1
        for pl_o in res_o.assembly.placed_parts.values():
            if pl_o.part_id in ("3005", "3024"):
                odd_1x1_emitted += 1
        art_o = export_certificate_artifact(
            res_o.assembly,
            res_o.certificates,
            validator=kernel.validator,
            colored_inventory=odd_inv,
        )
        iv_o = verify_certificate_independently(art_o)
        if iv_o.verified:
            odd_iv += 1

    # Tampered color-overrun mutant falsification (`10` mutants)
    mutants_tested = len(sample_valid_artifacts)
    mutants_rejected = 0
    for m_idx, valid_art in enumerate(sample_valid_artifacts):
        tampered_dict = valid_art.to_json_dict()
        if tampered_dict["placed_parts"]:
            # Mutate placed part 0's color_id to 999 (unstocked color) on even indices, or overrun stock on odd indices
            if m_idx % 2 == 0:
                tampered_dict["placed_parts"][0]["color_id"] = 999
            else:
                p0_id = str(tampered_dict["placed_parts"][0]["part_id"])
                c0_id = int(tampered_dict["placed_parts"][0]["color_id"])
                tampered_dict["colored_inventory_spec"][f"{p0_id}:{c0_id}"] = 0
        tampered_art = valid_art.from_json_dict(copy.deepcopy(tampered_dict))
        iv_m = verify_certificate_independently(tampered_art)
        if not iv_m.verified and any("colored_inventory_exceeded" in r for r in iv_m.failure_reasons):
            mutants_rejected += 1

    # Volume-insufficient shortfall diagnosis check
    vl_sf: List[np.ndarray] = []
    fl_sf: List[np.ndarray] = []
    cl_sf: List[np.ndarray] = []
    _append_box_mesh(vl_sf, fl_sf, cl_sf, 0.0, 0.0, 0.0, 160.0, 160.0, 96.0, (0.9, 0.1, 0.1))
    v_sf, f_sf, c_sf = _finalize_mesh(vl_sf, fl_sf, cl_sf)
    tf_sf, _ = voxelize_triangle_mesh_to_target_field(v_sf, f_sf, max_xy_studs=8, max_z_courses=4)
    tiny_inv = ColoredInventorySpec(counts={("3001", 4): 2})
    res_sf = kernel.realize(tf_sf, colored_inventory=tiny_inv, verts_ldu=v_sf, vertex_colors_rgb=c_sf)
    shortfall_missing = (
        max(0, int(res_sf.inventory_shortfall.required_target_volume_ldu3 - res_sf.inventory_shortfall.available_volume_ldu3))
        if res_sf.inventory_shortfall is not None
        else 0
    )
    shortfall_ok = (
        res_sf.status == KernelResultStatus.PROVEN_INFEASIBLE
        and res_sf.inventory_shortfall is not None
        and not res_sf.inventory_shortfall.is_volume_sufficient
        and shortfall_missing > 0
    )

    mean_skin_de = float(np.mean(skin_de_list)) if skin_de_list else 0.0
    mean_fifo_de = float(np.mean(fifo_de_list)) if fifo_de_list else 1.0
    de_red_ratio = (mean_fifo_de - mean_skin_de) / max(1e-6, mean_fifo_de)
    p50_ms = float(np.percentile(wall_times_ms, 50)) if wall_times_ms else 0.0
    p90_ms = float(np.percentile(wall_times_ms, 90)) if wall_times_ms else 0.0

    tot_n = n_set_10698 + n_odd_parity
    tot_cert = s10698_cert + odd_cert
    tot_iv = s10698_iv + odd_iv

    gate9a_ok = (
        sc_audit.strictly_single_core_verified
        and tot_cert == tot_n
        and tot_iv == tot_n
        and s10698_overruns == 0
        and odd_1x1_emitted == 0
        and de_red_ratio >= 0.35
        and tot_core_hidden > 0
        and mutants_tested >= 8
        and mutants_rejected == mutants_tested
        and shortfall_ok
        and p50_ms <= max_p50_wall_ms
    )

    return Gate9AColoredInventoryReport(
        single_core_audit=sc_audit,
        catalog_sha256_v110=cat_sha,
        set_10698_num_targets=n_set_10698,
        set_10698_certified_count=s10698_cert,
        set_10698_verifier_passed_count=s10698_iv,
        set_10698_color_overrun_count=s10698_overruns,
        set_10698_skin_mean_delta_e00=round(mean_skin_de, 4),
        set_10698_fifo_skin_mean_delta_e00=round(mean_fifo_de, 4),
        set_10698_skin_delta_e00_reduction_ratio=round(de_red_ratio, 6),
        set_10698_total_core_parts_hidden=tot_core_hidden,
        set_10698_total_chromatic_splits=tot_splits,
        sparse_odd_parity_num_targets=n_odd_parity,
        sparse_odd_parity_certified_count=odd_cert,
        sparse_odd_parity_verifier_passed_count=odd_iv,
        sparse_odd_parity_one_by_one_emitted=odd_1x1_emitted,
        total_targets=tot_n,
        total_certified_count=tot_cert,
        total_verifier_passed_count=tot_iv,
        tampered_color_mutants_tested=mutants_tested,
        tampered_color_mutants_rejected=mutants_rejected,
        shortfall_infeasible_detected=shortfall_ok,
        shortfall_missing_volume_ldu3=shortfall_missing,
        realize_p50_wall_ms=round(p50_ms, 2),
        realize_p90_wall_ms=round(p90_ms, 2),
        wall_clock_ms=round((time.perf_counter() - t0_suite) * 1000.0, 2),
        gate9a_passed=gate9a_ok,
    )


# ---------------------------------------------------------------------------
# Gate 9B: Ribbed Hollow-Shell Megastructure Scaling (`n = 30`, TABLE_FRICTION_ONLY)
# ---------------------------------------------------------------------------


def _build_gate9b_megastructure_mesh(idx: int) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Build `16x16x10` to `24x24x16` architectural/statue mesh `idx in 0..29`."""
    # Span `16x16x10` through `24x24x16` deterministically across 6 structural archetypes
    scales: Tuple[Tuple[int, int, int], ...] = (
        (16, 16, 10),
        (18, 16, 11),
        (18, 18, 12),
        (20, 18, 12),
        (20, 20, 14),
        (24, 24, 16),
    )
    wx, wy, wz = scales[idx % len(scales)]
    archetype = (idx // len(scales)) % 5
    vl: List[np.ndarray] = []
    fl: List[np.ndarray] = []
    cl: List[np.ndarray] = []

    if archetype == 0:
        # Monumental citadel keep
        _append_box_mesh(vl, fl, cl, 0.0, 0.0, 0.0, wx * 20.0, wy * 20.0, wz * 24.0)
    elif archetype == 1:
        # Two-tier stepped ziggurat temple
        h1 = (wz - 3) * 24.0
        _append_box_mesh(vl, fl, cl, 0.0, 0.0, 0.0, wx * 20.0, wy * 20.0, h1)
        _append_box_mesh(vl, fl, cl, 40.0, 40.0, h1, (wx - 2) * 20.0, (wy - 2) * 20.0, wz * 24.0)
    elif archetype == 2:
        # Gabled grand cathedral hall (full-footprint nave + 45°-pitched clerestory gable roof)
        h_wall = (wz - 3) * 24.0
        yn0 = float(((wy - 6) // 2) * 20)
        yn1 = yn0 + 120.0
        _append_box_mesh(vl, fl, cl, 0.0, 0.0, 0.0, wx * 20.0, wy * 20.0, h_wall)
        _append_gable_prism_mesh(vl, fl, cl, 0.0, yn0, h_wall, wx * 20.0, yn1, wz * 24.0, along_x=True)
    elif archetype == 3:
        # Plinth + monumental statue pedestal
        h_plinth = 4 * 24.0
        _append_box_mesh(vl, fl, cl, 0.0, 0.0, 0.0, wx * 20.0, wy * 20.0, h_plinth)
        _append_box_mesh(vl, fl, cl, 20.0, 20.0, h_plinth, (wx - 1) * 20.0, (wy - 1) * 20.0, wz * 24.0)
    else:
        # Stepped fortress bastion with upper lantern block
        h_base = (wz - 4) * 24.0
        h_mid = (wz - 1) * 24.0
        _append_box_mesh(vl, fl, cl, 0.0, 0.0, 0.0, wx * 20.0, wy * 20.0, h_base)
        _append_box_mesh(vl, fl, cl, 20.0, 20.0, h_base, (wx - 1) * 20.0, (wy - 1) * 20.0, h_mid)
        _append_box_mesh(vl, fl, cl, 40.0, 40.0, h_mid, (wx - 2) * 20.0, (wy - 2) * 20.0, wz * 24.0)

    v, f, _ = _finalize_mesh(vl, fl, cl)
    return v, f, max(wx, wy), wz


def run_gate9b_ribbed_hollow_shell_scaling_benchmark(
    num_targets: int = 30,
    target_indices: Optional[Sequence[int]] = None,
) -> Gate9BHollowShellReport:
    """Execute `Gate 9B` (`Ext 3`) across `n = 30` medium/large meshes under `TABLE_FRICTION_ONLY`."""
    t0_suite = time.perf_counter()
    sc_audit = enforce_and_audit_single_core()
    kernel = TopoBrickKernel()
    shell_policy = HollowShellPolicy(enabled=True)
    eval_indices = tuple(int(i) for i in target_indices) if target_indices is not None else tuple(range(num_targets))
    eff_n = len(eval_indices)

    hollow_cert = 0
    solid_cert = 0
    hollow_ins_ok = 0
    temp_supports_cnt = 0
    iv_ok_cnt = 0

    solid_parts_list: List[int] = []
    hollow_parts_list: List[int] = []
    brick_reds: List[float] = []
    vol_reds: List[float] = []
    solid_crust_ious: List[float] = []
    hollow_crust_ious: List[float] = []
    crust_diffs: List[float] = []
    hollow_ms_list: List[float] = []

    for i in eval_indices:
        v, f, max_xy, max_z = _build_gate9b_megastructure_mesh(i)
        tf, _ = voxelize_triangle_mesh_to_target_field(v, f, max_xy_studs=max_xy, max_z_courses=max_z)
        ast = AdaptiveSurfaceTarget.from_triangle_mesh(v, f, max_xy_studs=max_xy, max_z_courses=max_z)

        res_solid = kernel.realize(
            tf,
            adaptive_mesh_target=ast,
            part_retrieval="open_catalog_top_k",
            foundation_policy=FoundationPolicy.TABLE_FRICTION_ONLY,
            object_id=f"gate9b_solid_{i:02d}",
        )
        t_h0 = time.perf_counter()
        res_hollow = kernel.realize(
            tf,
            adaptive_mesh_target=ast,
            part_retrieval="open_catalog_top_k",
            foundation_policy=FoundationPolicy.TABLE_FRICTION_ONLY,
            hollow_shell_policy=shell_policy,
            object_id=f"gate9b_hollow_{i:02d}",
        )
        hollow_ms_list.append((time.perf_counter() - t_h0) * 1000.0)

        if res_solid.status == KernelResultStatus.COMPLETED_CERTIFIED and res_solid.certificates.overall_certified:
            solid_cert += 1
        if (
            res_hollow.status == KernelResultStatus.COMPLETED_CERTIFIED
            and res_hollow.certificates.overall_certified
            and res_hollow.certificates.foundation_policy == "table_friction_only"
        ):
            hollow_cert += 1
        if res_hollow.certificates.insertion.status == CertStatus.PASS:
            hollow_ins_ok += 1

        eff_tr = res_hollow.certificates.execution_trace
        if eff_tr is not None:
            for st in eff_tr.steps:
                if "TEMP_SUPPORT" in str(st.op_type):
                    temp_supports_cnt += 1

        art_h = export_certificate_artifact(res_hollow.assembly, res_hollow.certificates, validator=kernel.validator)
        iv_h = verify_certificate_independently(art_h)
        if iv_h.verified:
            iv_ok_cnt += 1

        n_sol = max(1, res_solid.assembly.num_parts)
        n_hol = res_hollow.assembly.num_parts
        solid_parts_list.append(n_sol)
        hollow_parts_list.append(n_hol)
        brick_reds.append(1.0 - (float(n_hol) / float(n_sol)))

        cs_iou, _ = evaluate_exterior_crust_fidelity(res_solid.assembly, ast.occ_5x5x6)
        ch_iou, v_red = evaluate_exterior_crust_fidelity(res_hollow.assembly, ast.occ_5x5x6)
        solid_crust_ious.append(cs_iou)
        hollow_crust_ious.append(ch_iou)
        vol_reds.append(v_red)
        crust_diffs.append(abs(ch_iou - cs_iou))

    mean_b_red = float(np.mean(brick_reds)) if brick_reds else 0.0
    mean_v_red = float(np.mean(vol_reds)) if vol_reds else 0.0
    mean_c_diff = float(np.mean(crust_diffs)) if crust_diffs else 1.0
    max_c_diff = float(np.max(crust_diffs)) if crust_diffs else 1.0
    h_p50_ms = float(np.percentile(hollow_ms_list, 50)) if hollow_ms_list else 0.0

    gate9b_ok = (
        sc_audit.strictly_single_core_verified
        and hollow_cert == eff_n
        and solid_cert == eff_n
        and hollow_ins_ok == eff_n
        and temp_supports_cnt == 0
        and iv_ok_cnt == eff_n
        and mean_b_red >= 0.45
        and mean_v_red >= 0.45
        and mean_c_diff <= 0.015
    )

    return Gate9BHollowShellReport(
        single_core_audit=sc_audit,
        num_targets=eff_n,
        hollow_certified_count=hollow_cert,
        solid_certified_count=solid_cert,
        hollow_insertion_pass_count=hollow_ins_ok,
        hollow_temp_supports_emitted=temp_supports_cnt,
        independent_verifier_passed_count=iv_ok_cnt,
        mean_solid_parts=round(float(np.mean(solid_parts_list)), 2),
        mean_hollow_parts=round(float(np.mean(hollow_parts_list)), 2),
        mean_brick_reduction_ratio=round(mean_b_red, 6),
        mean_volume_savings_ratio=round(mean_v_red, 6),
        mean_solid_crust_iou_5x5x6=round(float(np.mean(solid_crust_ious)), 6),
        mean_hollow_crust_iou_5x5x6=round(float(np.mean(hollow_crust_ious)), 6),
        mean_abs_crust_iou_diff=round(mean_c_diff, 6),
        max_abs_crust_iou_diff=round(max_c_diff, 6),
        hollow_p50_wall_ms=round(h_p50_ms, 2),
        wall_clock_ms=round((time.perf_counter() - t0_suite) * 1000.0, 2),
        gate9b_passed=gate9b_ok,
    )


# ---------------------------------------------------------------------------
# Gate 9C: Compositional `.mpd` Subassemblies & 8-Role Landmarks (`n = 25`)
# ---------------------------------------------------------------------------


def _build_box_ast(
    nx_studs: int,
    ny_studs: int,
    nz_courses: int,
    rgb: Tuple[float, float, float] = (0.85, 0.85, 0.85),
) -> Tuple[AdaptiveSurfaceTarget, np.ndarray, np.ndarray]:
    vl: List[np.ndarray] = []
    fl: List[np.ndarray] = []
    cl: List[np.ndarray] = []
    _append_box_mesh(vl, fl, cl, 0.0, 0.0, 0.0, nx_studs * 20.0, ny_studs * 20.0, nz_courses * 24.0, rgb)
    v, f, c = _finalize_mesh(vl, fl, cl)
    ast = AdaptiveSurfaceTarget.from_triangle_mesh(
        v, f, max_xy_studs=max(nx_studs, ny_studs), max_z_courses=nz_courses
    )
    return ast, v, c


def _build_gate9c_scene_spec(
    scene_idx: int,
) -> Tuple[List[SubmodelNodeSpec], List[LandmarkAttachmentRequest], str]:
    """
    Construct multi-module scene `scene_idx in 0..24` (`13` `"all"` + `12` `"cc_by_4_0_only"`)
    spanning courtyards, curtain walls, twin watchtowers, cupolas, and all `8` landmark roles.
    """
    lic = "cc_by_4_0_only" if scene_idx < 12 else "all"
    style = "smooth_studless" if (scene_idx % 3 == 0) else None

    role_primary = LANDMARK_DETAIL_ROLES[scene_idx % len(LANDMARK_DETAIL_ROLES)]
    role_secondary = LANDMARK_DETAIL_ROLES[(scene_idx + 3) % len(LANDMARK_DETAIL_ROLES)]

    pattern = scene_idx % 4
    nodes: List[SubmodelNodeSpec] = []
    landmarks: List[LandmarkAttachmentRequest] = []

    if pattern == 0:
        # Plinth courtyard (`8x6x2`) + twin stacked watchtowers (`2x2x3` left & right)
        ast_base, vb, cb = _build_box_ast(8, 6, 2, (0.63, 0.65, 0.66))
        ast_tw, vt, ct = _build_box_ast(2, 2, 3, (0.95, 0.95, 0.95))
        nodes.append(
            SubmodelNodeSpec(
                submodel_name="courtyard_base",
                adaptive_target=ast_base,
                world_offset_ldu=(0, 0, 0),
                semantic_style_hint=style,
                verts_ldu=vb,
                vertex_colors_rgb=cb,
            )
        )
        nodes.append(
            SubmodelNodeSpec(
                submodel_name="west_watchtower",
                adaptive_target=ast_tw,
                world_offset_ldu=(0, 20, 48),
                parent_submodel="courtyard_base",
                verts_ldu=vt,
                vertex_colors_rgb=ct,
            )
        )
        nodes.append(
            SubmodelNodeSpec(
                submodel_name="east_watchtower",
                adaptive_target=ast_tw,
                world_offset_ldu=(120, 20, 48),
                parent_submodel="courtyard_base",
                verts_ldu=vt,
                vertex_colors_rgb=ct,
            )
        )
        top_z = 120
        landmarks.append(
            LandmarkAttachmentRequest(
                role=role_primary,
                anchor_box_ldu=LDUBox(-20, 0, 40, 180, 120, top_z + 36),
                preferred_color_id=4,
                count=1,
            )
        )
        landmarks.append(
            LandmarkAttachmentRequest(
                role=role_secondary,
                anchor_box_ldu=LDUBox(-20, 0, 40, 180, 120, top_z + 36),
                preferred_color_id=14,
                count=1,
            )
        )
    elif pattern == 1:
        # Stepped 3-level pagoda/campanile (`6x6x2` -> `4x4x2` -> `2x2x2`)
        ast_0, v0, c0 = _build_box_ast(6, 6, 2, (0.89, 0.80, 0.62))
        ast_1, v1, c1 = _build_box_ast(4, 4, 2, (0.77, 0.10, 0.09))
        ast_2, v2, c2 = _build_box_ast(2, 2, 2, (0.95, 0.80, 0.12))
        nodes.append(
            SubmodelNodeSpec(
                submodel_name="pagoda_plinth",
                adaptive_target=ast_0,
                world_offset_ldu=(0, 0, 0),
                semantic_style_hint=style,
                verts_ldu=v0,
                vertex_colors_rgb=c0,
            )
        )
        nodes.append(
            SubmodelNodeSpec(
                submodel_name="pagoda_tier1",
                adaptive_target=ast_1,
                world_offset_ldu=(20, 20, 48),
                parent_submodel="pagoda_plinth",
                semantic_style_hint=style,
                verts_ldu=v1,
                vertex_colors_rgb=c1,
            )
        )
        nodes.append(
            SubmodelNodeSpec(
                submodel_name="pagoda_lantern",
                adaptive_target=ast_2,
                world_offset_ldu=(40, 40, 96),
                parent_submodel="pagoda_tier1",
                verts_ldu=v2,
                vertex_colors_rgb=c2,
            )
        )
        landmarks.append(
            LandmarkAttachmentRequest(
                role=role_primary,
                anchor_box_ldu=LDUBox(-10, -10, 40, 130, 130, 168),
                preferred_color_id=14,
                count=1,
            )
        )
    elif pattern == 2:
        # Gatehouse podium (`8x4x2`) + twin gate piers (`2x4x2`) + bridging barbican (`6x2x1`)
        ast_pod, vp, cp = _build_box_ast(8, 4, 2, (0.42, 0.43, 0.43))
        ast_pier, vpi, cpi = _build_box_ast(2, 4, 2, (0.63, 0.65, 0.66))
        ast_cap, vc, cc = _build_box_ast(8, 2, 1, (0.77, 0.10, 0.09))
        nodes.append(
            SubmodelNodeSpec(
                submodel_name="gate_podium",
                adaptive_target=ast_pod,
                world_offset_ldu=(0, 0, 0),
                semantic_style_hint=style,
                verts_ldu=vp,
                vertex_colors_rgb=cp,
            )
        )
        nodes.append(
            SubmodelNodeSpec(
                submodel_name="left_pier",
                adaptive_target=ast_pier,
                world_offset_ldu=(0, 0, 48),
                parent_submodel="gate_podium",
                verts_ldu=vpi,
                vertex_colors_rgb=cpi,
            )
        )
        nodes.append(
            SubmodelNodeSpec(
                submodel_name="right_pier",
                adaptive_target=ast_pier,
                world_offset_ldu=(120, 0, 48),
                parent_submodel="gate_podium",
                verts_ldu=vpi,
                vertex_colors_rgb=cpi,
            )
        )
        nodes.append(
            SubmodelNodeSpec(
                submodel_name="barbican_lintel",
                adaptive_target=ast_cap,
                world_offset_ldu=(0, 20, 96),
                parent_submodel="left_pier",
                verts_ldu=vc,
                vertex_colors_rgb=cc,
            )
        )
        landmarks.append(
            LandmarkAttachmentRequest(
                role=role_primary,
                anchor_box_ldu=LDUBox(-10, -10, 40, 170, 90, 144),
                preferred_color_id=15,
                count=1,
            )
        )
    else:
        # Citadel hall (`6x6x2`) + corner keep (`3x3x3`)
        ast_hall, vh, ch = _build_box_ast(6, 6, 2, (0.95, 0.95, 0.95))
        ast_keep, vk, ck = _build_box_ast(3, 3, 3, (0.02, 0.33, 0.75))
        nodes.append(
            SubmodelNodeSpec(
                submodel_name="citadel_hall",
                adaptive_target=ast_hall,
                world_offset_ldu=(0, 0, 0),
                semantic_style_hint=style,
                verts_ldu=vh,
                vertex_colors_rgb=ch,
            )
        )
        nodes.append(
            SubmodelNodeSpec(
                submodel_name="corner_keep",
                adaptive_target=ast_keep,
                world_offset_ldu=(0, 0, 48),
                parent_submodel="citadel_hall",
                verts_ldu=vk,
                vertex_colors_rgb=ck,
            )
        )
        landmarks.append(
            LandmarkAttachmentRequest(
                role=role_primary,
                anchor_box_ldu=LDUBox(-10, -10, 40, 130, 130, 150),
                preferred_color_id=4,
                count=1,
            )
        )
        landmarks.append(
            LandmarkAttachmentRequest(
                role=role_secondary,
                anchor_box_ldu=LDUBox(-10, -10, 40, 130, 130, 150),
                preferred_color_id=15,
                count=1,
            )
        )

    return nodes, landmarks, lic


def _verify_mpd_se3_roundtrip(assembly: Any, mpd_text: str) -> Tuple[bool, float]:
    """Flatten `.mpd` via `flatten_omr_mpd_to_se3` and match 1-to-1 against `assembly.placed_parts`."""
    flat = flatten_omr_mpd_to_se3(mpd_text, catalog=COMBINED_BRICK_CATALOG)
    if len(flat) != assembly.num_parts:
        return False, float("inf")

    from topo_brick.interface.ldu_frames import rot_id_to_matrix

    remaining = list(range(len(flat)))
    max_pos_err = 0.0
    for pl in sorted(assembly.placed_parts.values(), key=lambda x: x.instance_id):
        r_mat = rot_id_to_matrix(pl.pose.rot_id)
        p_vec = np.asarray(pl.pose.pos, dtype=np.float64)
        best_j: Optional[int] = None
        best_err = float("inf")
        for j in remaining:
            fp = flat[j]
            if fp.canonical_part_id != pl.part_id or int(fp.color_code) != int(pl.color_id) or fp.continuous_pose is None:
                continue
            pos_err = float(np.max(np.abs(np.asarray(fp.continuous_pose.pos, dtype=np.float64) - p_vec)))
            rot_err = float(np.max(np.abs(np.asarray(fp.continuous_pose.rot_matrix, dtype=np.float64) - r_mat)))
            if pos_err <= 1e-3 and rot_err <= 1e-4 and pos_err < best_err:
                best_err = pos_err
                best_j = j
                if pos_err <= 1e-6 and rot_err <= 1e-6:
                    break
        if best_j is None:
            return False, float("inf")
        remaining.remove(best_j)
        if best_err > max_pos_err:
            max_pos_err = best_err
    return True, max_pos_err


def _is_part_cc_by_sa(part_id: str) -> bool:
    """Return True if `part_id` carries `CC-BY-SA-4.0` ShareAlike licensing (`not CC-BY-4.0`)."""
    if part_id in EXTENDED_BRICK_CATALOG:
        return False
    rec_lm = LANDMARK_DETAIL_INDEX.get_record(part_id)
    if rec_lm is not None:
        return rec_lm.license_tier != "cc_by_4_0"
    can_id = CATALOG_RETRIEVAL_INDEX.resolve_canonical_mold_id(part_id) or part_id
    ent = CATALOG_RETRIEVAL_INDEX.get_entry(can_id)
    if ent is not None:
        return ent.license_tier != "cc_by_4_0"
    comp_spec = COMBINED_BRICK_CATALOG.get(part_id)
    if comp_spec is not None and hasattr(comp_spec, "license_tier"):
        return str(getattr(comp_spec, "license_tier")) != "cc_by_4_0"
    return True


def run_gate9c_compositional_mpd_and_landmarks_benchmark(
    num_scenes: int = 25,
) -> Gate9CCompositionalMPDReport:
    """Execute `Gate 9C` (`Ext 4A & 4B`) across `n = 25` multi-module `.mpd` architectural scenes."""
    t0_suite = time.perf_counter()
    sc_audit = enforce_and_audit_single_core()
    kernel = TopoBrickKernel()

    n_all_lic = 0
    n_cc4_lic = 0
    cert_cnt = 0
    iv_cnt = 0
    mpd_rt_cnt = 0
    max_mpd_pos_err = 0.0
    cc4_sa_leaks = 0
    roles_hit: Set[str] = set()
    tot_lm_attached = 0
    subtrace_artifacts: List[Any] = []

    landmark_role_by_pid: Dict[str, Tuple[str, ...]] = {}
    for r_name in LANDMARK_DETAIL_ROLES:
        for rec in LANDMARK_DETAIL_INDEX.query_role(r_name, license_filter="all"):
            landmark_role_by_pid.setdefault(rec.part_id, ())
            landmark_role_by_pid[rec.part_id] = tuple(sorted(set(landmark_role_by_pid[rec.part_id]) | {r_name}))

    dummy_vl: List[np.ndarray] = []
    dummy_fl: List[np.ndarray] = []
    dummy_cl: List[np.ndarray] = []
    _append_box_mesh(dummy_vl, dummy_fl, dummy_cl, 0.0, 0.0, 0.0, 40.0, 40.0, 24.0)
    dv, df, _ = _finalize_mesh(dummy_vl, dummy_fl, dummy_cl)
    dummy_tf, _ = voxelize_triangle_mesh_to_target_field(dv, df, max_xy_studs=2, max_z_courses=1)

    for s_idx in range(num_scenes):
        nodes, lm_reqs, lic = _build_gate9c_scene_spec(s_idx)
        if lic == "cc_by_4_0_only":
            n_cc4_lic += 1
        else:
            n_all_lic += 1

        res = kernel.realize(
            dummy_tf,
            subassembly_graph=nodes,
            landmark_requests=lm_reqs,
            license_filter=lic,
            object_id=f"gate9c_scene_{s_idx:02d}",
        )
        if res.status == KernelResultStatus.COMPLETED_CERTIFIED and res.certificates.overall_certified:
            cert_cnt += 1

        art = export_certificate_artifact(res.assembly, res.certificates, validator=kernel.validator)
        iv = verify_certificate_independently(art)
        if iv.verified and any(op.get("op_type") == "PLACE_SUBASSEMBLY" and op.get("subtrace") for op in art.execution_operations):
            iv_cnt += 1
            if len(subtrace_artifacts) < 3:
                subtrace_artifacts.append(art)

        if res.compositional_mpd_text:
            rt_ok, p_err = _verify_mpd_se3_roundtrip(res.assembly, res.compositional_mpd_text)
            if rt_ok:
                mpd_rt_cnt += 1
                max_mpd_pos_err = max(max_mpd_pos_err, p_err)

        for pl in res.assembly.placed_parts.values():
            if lic == "cc_by_4_0_only" and _is_part_cc_by_sa(pl.part_id):
                cc4_sa_leaks += 1
            if pl.part_id in landmark_role_by_pid:
                tot_lm_attached += 1
                for r_hit in landmark_role_by_pid[pl.part_id]:
                    roles_hit.add(r_hit)

    # Tampered `subtrace` mutant falsification (`9` mutants: colliding CSG, floating unstable, non-rigid offset)
    mut_tested = 0
    mut_rejected = 0
    for base_art in subtrace_artifacts:
        for mode in ("collision", "floating", "non_rigid"):
            d_mut = copy.deepcopy(base_art.to_json_dict())
            sub_op = next(
                op for op in d_mut["execution_operations"] if op.get("op_type") == "PLACE_SUBASSEMBLY" and op.get("subtrace")
            )
            if mode == "collision" and len(sub_op["subtrace"]) >= 2:
                # Force step 1 onto exact same local coordinates as step 0 -> 2-LDU CSG collision inside subtrace!
                sub_op["subtrace"][1]["pos"] = list(sub_op["subtrace"][0]["pos"])
            elif mode == "floating":
                # Float a non-grounded subtrace step 120 LDU up in local Z -> local workbench equilibrium fails!
                target_st = sub_op["subtrace"][-1]
                target_st["is_grounded"] = False
                target_st["pos"][2] = int(target_st["pos"][2]) + 120
            else:
                # Shift one local subtrace step by +20 LDU in X without moving world placed_parts -> non-rigid offset!
                sub_op["subtrace"][0]["pos"][0] = int(sub_op["subtrace"][0]["pos"][0]) + 20
            mut_tested += 1
            iv_mut = verify_certificate_independently(base_art.from_json_dict(d_mut))
            if not iv_mut.verified:
                mut_rejected += 1

    gate9c_ok = (
        sc_audit.strictly_single_core_verified
        and cert_cnt == num_scenes
        and iv_cnt == num_scenes
        and mpd_rt_cnt == num_scenes
        and cc4_sa_leaks == 0
        and len(roles_hit) == len(LANDMARK_DETAIL_ROLES)
        and mut_tested >= 6
        and mut_rejected == mut_tested
    )

    return Gate9CCompositionalMPDReport(
        single_core_audit=sc_audit,
        num_scenes=num_scenes,
        all_license_scenes=n_all_lic,
        cc_by_4_0_scenes=n_cc4_lic,
        certified_count=cert_cnt,
        independent_verifier_passed_count=iv_cnt,
        mpd_se3_roundtrip_exact_count=mpd_rt_cnt,
        mpd_max_position_error_ldu=round(max_mpd_pos_err, 6),
        cc_by_4_0_share_alike_leaks=cc4_sa_leaks,
        landmark_roles_exercised_count=len(roles_hit),
        landmark_roles_exercised=tuple(sorted(roles_hit)),
        total_landmarks_attached=tot_lm_attached,
        tampered_subtrace_mutants_tested=mut_tested,
        tampered_subtrace_mutants_rejected=mut_rejected,
        wall_clock_ms=round((time.perf_counter() - t0_suite) * 1000.0, 2),
        gate9c_passed=gate9c_ok,
    )


# ---------------------------------------------------------------------------
# Gate 9D: View-Prioritized Scarcity Study (`n = 30`, TABLE_FRICTION_ONLY)
# ---------------------------------------------------------------------------


def _build_gate9d_single_view_case(
    idx: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, ViewPrioritySpec, ColoredInventorySpec]:
    """
    Construct a single-view relief/silhouette target mesh (`idx in 0..29`) with thin (`8–10 LDU`)
    camera-contour relief features (battlements, stepped crest, side wings, spire) backed by a
    3D counterweight body, plus a `30%`-matching / `70%`-clashing `ColoredInventorySpec`.
    """
    # Alternate camera directions & orthographic vs perspective cameras
    cam_mode = idx % 5
    if cam_mode == 0:
        # Camera looking along +Y (at y = 0 front facade)
        v_spec = ViewPrioritySpec(view_dir_ldu=(0.0, 1.0, 0.0))
    elif cam_mode == 1:
        # Camera looking along -Y (at y = max front facade)
        v_spec = ViewPrioritySpec(view_dir_ldu=(0.0, -1.0, 0.0))
    elif cam_mode == 2:
        # Slightly elevated front orthographic camera
        v_spec = ViewPrioritySpec(view_dir_ldu=(0.0, 1.0, -0.12))
    elif cam_mode == 3:
        # Perspective front camera
        v_spec = ViewPrioritySpec(view_dir_ldu=(0.0, 1.0, 0.0), camera_pos_ldu=(80.0, -260.0, 48.0))
    else:
        # Perspective -Y camera
        v_spec = ViewPrioritySpec(view_dir_ldu=(0.0, -1.0, 0.0), camera_pos_ldu=(80.0, 340.0, 48.0))

    # Color pairs with high ΔE_00 contrast between matching target color (`~30%` stock) & clashing filler (`~70%` stock)
    palette_choices: Tuple[Tuple[Tuple[float, float, float], int, int], ...] = (
        ((0.95, 0.95, 0.95), 15, 4),   # White target (`15`) vs bright Red filler (`4`)
        ((0.02, 0.33, 0.75), 1, 14),   # Blue target (`1`) vs bright Yellow filler (`14`)
        ((0.95, 0.80, 0.12), 14, 1),   # Yellow target (`14`) vs Blue filler (`1`)
        ((0.77, 0.10, 0.09), 4, 27),   # Red target (`4`) vs Lime filler (`27`)
        ((0.14, 0.47, 0.25), 2, 25),   # Green target (`2`) vs Orange filler (`25`)
    )
    tgt_rgb, match_cid, clash_cid = palette_choices[(idx // 5) % len(palette_choices)]

    vl: List[np.ndarray] = []
    fl: List[np.ndarray] = []
    cl: List[np.ndarray] = []

    wx, wz = 8, 4
    # Solid counterweight main body: x in [20, 140] (6 studs wide), y in [0, 80] (4 studs deep), z in [0, 72] (3 courses)
    _append_box_mesh(vl, fl, cl, 20.0, 0.0, 0.0, 140.0, 80.0, 72.0, tgt_rgb)

    # Thin (`9 LDU` deep in Y — < 50% of a 20-LDU stud!) silhouette contour wings & battlements:
    # Under isotropic 3D voxelization (`val_c < 24`), a 9-LDU-deep slice has only 2/4 sub-voxels in Y so
    # isotropic 3D realization drops it to avoid 3D rear air overfill, losing the 2D silhouette contour!
    # Under ViewPrioritySpec, rear air sits in `air_occluded_behind` (weight ~0.12) while front silhouette
    # pixels carry `3.5` weight, so 20-LDU bricks preserve the exact 2D silhouette cleanly!
    front_y0, front_y1 = (0.0, 6.5) if v_spec.view_dir_ldu[1] > 0 else (73.5, 80.0)
    motif = idx % 3
    if motif == 0:
        # Left & right heraldic side wings + 2 rooftop merlons
        _append_box_mesh(vl, fl, cl, 0.0, front_y0, 0.0, 20.0, front_y1, 72.0, tgt_rgb)
        _append_box_mesh(vl, fl, cl, 140.0, front_y0, 0.0, 160.0, front_y1, 72.0, tgt_rgb)
        _append_box_mesh(vl, fl, cl, 20.0, front_y0, 72.0, 60.0, front_y1, 96.0, tgt_rgb)
        _append_box_mesh(vl, fl, cl, 100.0, front_y0, 72.0, 140.0, front_y1, 96.0, tgt_rgb)
    elif motif == 1:
        # Stepped facade crown + wide side buttresses
        _append_box_mesh(vl, fl, cl, 0.0, front_y0, 0.0, 20.0, front_y1, 48.0, tgt_rgb)
        _append_box_mesh(vl, fl, cl, 140.0, front_y0, 0.0, 160.0, front_y1, 48.0, tgt_rgb)
        _append_box_mesh(vl, fl, cl, 40.0, front_y0, 72.0, 120.0, front_y1, 96.0, tgt_rgb)
    else:
        # Full crenellated parapet + full-height side pilasters
        _append_box_mesh(vl, fl, cl, 0.0, front_y0, 0.0, 20.0, front_y1, 96.0, tgt_rgb)
        _append_box_mesh(vl, fl, cl, 140.0, front_y0, 0.0, 160.0, front_y1, 96.0, tgt_rgb)
        _append_box_mesh(vl, fl, cl, 60.0, front_y0, 72.0, 100.0, front_y1, 96.0, tgt_rgb)

    v, f, c = _finalize_mesh(vl, fl, cl)

    # Genuine ~30% matching-color stock (~8 matching bricks vs ~35 clashing filler bricks for a ~25-brick build)
    scarce_counts: Dict[ColoredPartKey, int] = {
        ("3001", match_cid): 2,
        ("3001", clash_cid): 12,
        ("3003", match_cid): 2,
        ("3003", clash_cid): 10,
        ("3010", match_cid): 2,
        ("3010", clash_cid): 10,
        ("3004", match_cid): 2,
        ("3004", clash_cid): 14,
        ("3005", match_cid): 2,
        ("3005", clash_cid): 12,
        ("3020", match_cid): 1,
        ("3020", clash_cid): 6,
        ("3710", match_cid): 2,
        ("3710", clash_cid): 8,
        ("3023", match_cid): 2,
        ("3023", clash_cid): 10,
    }
    inv = ColoredInventorySpec(counts=scarce_counts)
    return v, f, c, wx, wz, v_spec, inv


def run_gate9d_view_prioritized_scarcity_study(
    num_targets: int = 30,
) -> Gate9DViewPriorityReport:
    """Execute `Gate 9D` (`Ext 5`) paired ablation (`ViewPrioritySpec` ON vs. OFF) under `TABLE_FRICTION_ONLY`."""
    t0_suite = time.perf_counter()
    sc_audit = enforce_and_audit_single_core()
    kernel = TopoBrickKernel()

    on_cert = 0
    off_cert = 0
    iv_on_ok = 0

    sil_off_list: List[float] = []
    sil_on_list: List[float] = []
    vis_de_off_list: List[float] = []
    vis_de_on_list: List[float] = []
    rear_de_on_list: List[float] = []

    for i in range(num_targets):
        v, f, c, wx, wz, v_spec, scarce_inv = _build_gate9d_single_view_case(i)
        tf, _ = voxelize_triangle_mesh_to_target_field(v, f, max_xy_studs=wx, max_z_courses=wz)
        ast = AdaptiveSurfaceTarget.from_triangle_mesh(v, f, max_xy_studs=wx, max_z_courses=wz)

        res_off = kernel.realize(
            tf,
            adaptive_mesh_target=ast,
            surface_refinement="open_catalog_sculpted",
            colored_inventory=scarce_inv,
            verts_ldu=v,
            vertex_colors_rgb=c,
            foundation_policy=FoundationPolicy.TABLE_FRICTION_ONLY,
            view_priority=None,
            object_id=f"gate9d_off_{i:02d}",
        )
        res_on = kernel.realize(
            tf,
            adaptive_mesh_target=ast,
            surface_refinement="open_catalog_sculpted",
            colored_inventory=scarce_inv,
            verts_ldu=v,
            vertex_colors_rgb=c,
            foundation_policy=FoundationPolicy.TABLE_FRICTION_ONLY,
            view_priority=v_spec,
            object_id=f"gate9d_on_{i:02d}",
        )

        if res_off.status == KernelResultStatus.COMPLETED_CERTIFIED and res_off.certificates.overall_certified:
            off_cert += 1
        if (
            res_on.status == KernelResultStatus.COMPLETED_CERTIFIED
            and res_on.certificates.overall_certified
            and res_on.certificates.foundation_policy == "table_friction_only"
        ):
            on_cert += 1

        art_on = export_certificate_artifact(
            res_on.assembly,
            res_on.certificates,
            validator=kernel.validator,
            colored_inventory=scarce_inv,
        )
        if verify_certificate_independently(art_on).verified:
            iv_on_ok += 1

        m_off = evaluate_camera_reprojection_fidelity(
            res_off.assembly, ast.occ_5x5x6, v_spec, verts_ldu=v, vertex_colors_rgb=c
        )
        m_on = evaluate_camera_reprojection_fidelity(
            res_on.assembly, ast.occ_5x5x6, v_spec, verts_ldu=v, vertex_colors_rgb=c
        )
        sil_off_list.append(m_off.silhouette_iou_2d)
        sil_on_list.append(m_on.silhouette_iou_2d)
        vis_de_off_list.append(m_off.visible_pixel_mean_delta_e00)
        vis_de_on_list.append(m_on.visible_pixel_mean_delta_e00)
        rear_de_on_list.append(m_on.occluded_rear_mean_delta_e00)

    mean_sil_off = float(np.mean(sil_off_list)) if sil_off_list else 0.0
    mean_sil_on = float(np.mean(sil_on_list)) if sil_on_list else 0.0
    sil_lift = mean_sil_on - mean_sil_off

    mean_de_off = float(np.mean(vis_de_off_list)) if vis_de_off_list else 0.0
    mean_de_on = float(np.mean(vis_de_on_list)) if vis_de_on_list else 0.0
    de_red = mean_de_off - mean_de_on

    gate9d_ok = (
        sc_audit.strictly_single_core_verified
        and on_cert == num_targets
        and off_cert == num_targets
        and iv_on_ok == num_targets
        and sil_lift >= 0.04
        and de_red >= 4.0
    )

    return Gate9DViewPriorityReport(
        single_core_audit=sc_audit,
        num_targets=num_targets,
        view_on_certified_count=on_cert,
        view_off_certified_count=off_cert,
        independent_verifier_passed_count=iv_on_ok,
        mean_silhouette_iou_2d_off=round(mean_sil_off, 6),
        mean_silhouette_iou_2d_on=round(mean_sil_on, 6),
        mean_silhouette_iou_2d_lift=round(sil_lift, 6),
        mean_visible_delta_e00_off=round(mean_de_off, 4),
        mean_visible_delta_e00_on=round(mean_de_on, 4),
        mean_visible_delta_e00_reduction=round(de_red, 4),
        mean_occluded_rear_delta_e00_on=round(float(np.mean(rear_de_on_list)), 4),
        wall_clock_ms=round((time.perf_counter() - t0_suite) * 1000.0, 2),
        gate9d_passed=gate9d_ok,
    )


# ---------------------------------------------------------------------------
# Full Phase 11 Gate Suite Runner & CLI
# ---------------------------------------------------------------------------


def run_all_phase11_gates(
    output_json_path: Optional[Path] = None,
    max_gate9a_p50_ms: float = 250.0,
) -> Phase11CompleteGateSummary:
    """Execute all four Phase 11 promotion gates (`9A, 9B, 9C, 9D`) under strict `1`-core auditing."""
    g9a = run_gate9a_colored_finite_inventory_and_sparse_kit_benchmark(max_p50_wall_ms=max_gate9a_p50_ms)
    g9b = run_gate9b_ribbed_hollow_shell_scaling_benchmark()
    g9c = run_gate9c_compositional_mpd_and_landmarks_benchmark()
    g9d = run_gate9d_view_prioritized_scarcity_study()

    summary = Phase11CompleteGateSummary(
        schema_version=CERTIFICATE_SCHEMA_VERSION,
        catalog_sha256_v110=compute_catalog_sha256(CERTIFICATE_SCHEMA_VERSION),
        landmark_index_sha256=LANDMARK_DETAIL_INDEX.sha256,
        gate9a=g9a,
        gate9b=g9b,
        gate9c=g9c,
        gate9d=g9d,
        all_phase11_gates_passed=bool(
            g9a.gate9a_passed and g9b.gate9b_passed and g9c.gate9c_passed and g9d.gate9d_passed
        ),
    )
    if output_json_path is not None:
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 11 Five-Extensions Promotion Gates (9A–9D).")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON ledger output path.")
    parser.add_argument("--max-p50-ms", type=float, default=250.0, help="Max allowed Gate 9A p50 wall ms.")
    args = parser.parse_args()

    out_path = Path(args.output) if args.output else None
    summary = run_all_phase11_gates(output_json_path=out_path, max_gate9a_p50_ms=float(args.max_p50_ms))
    print(
        json.dumps(
            {
                "all_phase11_gates_passed": summary.all_phase11_gates_passed,
                "schema_version": summary.schema_version,
                "gate9a_passed": summary.gate9a.gate9a_passed,
                "gate9a_certified": f"{summary.gate9a.total_certified_count}/{summary.gate9a.total_targets}",
                "gate9a_skin_de00_reduction_ratio": summary.gate9a.set_10698_skin_delta_e00_reduction_ratio,
                "gate9a_p50_ms": summary.gate9a.realize_p50_wall_ms,
                "gate9b_passed": summary.gate9b.gate9b_passed,
                "gate9b_certified": f"{summary.gate9b.hollow_certified_count}/{summary.gate9b.num_targets}",
                "gate9b_brick_reduction": summary.gate9b.mean_brick_reduction_ratio,
                "gate9b_vol_reduction": summary.gate9b.mean_volume_savings_ratio,
                "gate9b_crust_iou_diff": summary.gate9b.mean_abs_crust_iou_diff,
                "gate9c_passed": summary.gate9c.gate9c_passed,
                "gate9c_certified": f"{summary.gate9c.certified_count}/{summary.gate9c.num_scenes}",
                "gate9c_roles_exercised": summary.gate9c.landmark_roles_exercised_count,
                "gate9c_cc4_sa_leaks": summary.gate9c.cc_by_4_0_share_alike_leaks,
                "gate9d_passed": summary.gate9d.gate9d_passed,
                "gate9d_certified": f"{summary.gate9d.view_on_certified_count}/{summary.gate9d.num_targets}",
                "gate9d_silhouette_lift": summary.gate9d.mean_silhouette_iou_2d_lift,
                "gate9d_visible_de00_reduction": summary.gate9d.mean_visible_delta_e00_reduction,
            },
            indent=2,
        )
    )
    if not summary.all_phase11_gates_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
