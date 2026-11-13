"""
Phase 10 (`P10-WS1` .. `P10-WS4`): Open-Catalog Semantic & Geometric Part Retrieval
(`16,085` Combined Catalog -> `242` Watertight Buildable System Molds -> `0.92+` Original-Mesh IoU)
Verification & Promotion Gate Suite (`Gate 8A` & `Gate 8B`).

Implements falsifiable single-core (`OMP_NUM_THREADS=1`) acceptance gates:
- `Gate 8A` (`run_gate8a_open_catalog_blind_mesh_benchmark`):
  * Evaluates `TopoBrickKernel.realize(..., part_retrieval="open_catalog_top_k", surface_refinement="open_catalog_sculpted")`
    across all `80` frozen unseen continuous triangle meshes from `Gate 2` using `Gate 2`'s
    100% untouched `5x5x6 LDU` `evaluate_assembly_against_original_mesh` judge + `5x5x2 LDU`
    sub-plate IoU + planner-free `verify_certificate_independently`.
- `Gate 8B` (`run_gate8b_retrieval_ablation_license_and_style_study`):
  * Paired `n = 40` stratified `K`-ablation (`Fixed-10` vs. `Fixed-27` `Gate 7A` vs. `Open-Catalog Top-16`,
    `Top-32`, `Top-64`), strict `license_filter="cc_by_4_0_only"` commercial license audit (`0` `CC-BY-SA-4.0`
    parts emitted), and Option 3B perceptual `CIEDE2000` (`ΔE_00 <= 8.0`) LDraw color + `"smooth_studless"`
    stud-reduction (`>= 60%`) study across `n = 20` multi-region colored meshes.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from topo_brick.evaluation.independent_verifier import (
    compute_catalog_sha256,
    export_certificate_artifact,
    verify_certificate_independently,
)
from topo_brick.evaluation.phase7_sota_benchmarks import voxelize_triangle_mesh_to_target_field
from topo_brick.evaluation.phase8_round9_gates import (
    SingleCoreAuditReport,
    _rasterize_fine_mesh_5x5x6,
    _scale_mesh_to_ldu,
    enforce_and_audit_single_core,
    evaluate_assembly_against_original_mesh,
    load_frozen_80_unseen_mesh_corpus,
)
from topo_brick.interface.brick_catalog import (
    EXTENDED_10PART_RECT_KIT,
    EXTENDED_BRICK_CATALOG,
    get_declarative_part_spec,
)
from topo_brick.interface.catalog_retrieval_index import (
    CATALOG_RETRIEVAL_INDEX,
    SCULPTURAL_TAXONOMY_CLASSES,
)
from topo_brick.interface.ldraw_shadow_catalog_compiler import COMBINED_BRICK_CATALOG
from topo_brick.interface.ldu_frames import LDUBox
from topo_brick.kernel.adaptive_surface_and_macro_planner import (
    ADAPTIVE_SURFACE_KIT,
    AdaptiveSurfaceTarget,
    evaluate_exact_5x5x2_mesh_iou,
)
from topo_brick.kernel.construction_kernel import KernelResultStatus, TopoBrickKernel
from topo_brick.kernel.open_catalog_realizer import (
    LDRAW_SOLID_PALETTE_SRGB,
    ciede2000_delta_e,
    count_exposed_upward_studs,
    srgb_to_cielab,
)
from topo_brick.representation.interface_fields import SparseTiledTargetField
from topo_brick.training import parse_ldr_text


@dataclass(frozen=True, slots=True)
class Gate8ABandSummary:
    """Per-band metrics on `Gate 2`'s frozen 80-mesh unseen corpus (`20` meshes / band)."""

    size_band: str
    num_meshes: int
    certified_count: int
    openings_clear_count: int
    coarse_mean_iou_5x5x6: float
    gate7a_mean_iou_5x5x6: float
    open_catalog_mean_iou_5x5x6: float
    open_catalog_median_iou_5x5x6: float
    open_catalog_min_iou_5x5x6: float
    open_catalog_mean_iou_5x5x2: float
    open_catalog_mean_chamfer_ldu: float
    qualified_at_085_rate: float
    qualified_at_090_rate: float
    p50_wall_ms: float
    p90_wall_ms: float


@dataclass(frozen=True, slots=True)
class Gate8AOpenCatalogBenchmarkReport:
    """Audit report for `Gate 8A`: Open-Catalog Blind Mesh Benchmark (`n = 80`)."""

    single_core_audit: SingleCoreAuditReport
    corpus_sha256: str
    catalog_sha256: str
    retrieval_index_sha256: str
    retrieval_top_k: int
    num_meshes: int
    certified_count: int
    certified_rate: float
    openings_clear_count: int
    independent_verifier_tested: int
    independent_verifier_passed: int
    coarse_mean_iou_5x5x6: float
    gate7a_mean_iou_5x5x6: float
    open_catalog_mean_iou_5x5x6: float
    open_catalog_median_iou_5x5x6: float
    bands_2_to_4_mean_iou_5x5x6: float
    bands_2_to_4_median_iou_5x5x6: float
    bands_2_to_4_qualified_at_090_rate: float
    band1_mean_iou_5x5x6: float
    overall_qualified_at_085_rate: float
    overall_qualified_at_080_rate: float
    open_catalog_mean_iou_5x5x2: float
    discretization_loss_reduction_vs_coarse: float
    distinct_parts_placed_count: int
    distinct_parts_placed: Tuple[str, ...]
    sculptural_classes_exercised: Tuple[str, ...]
    retrieval_query_p50_ms: float
    retrieval_query_p90_ms: float
    realize_only_p50_ms: float
    end_to_end_p50_ms: float
    end_to_end_p90_ms: float
    by_band: Tuple[Gate8ABandSummary, ...]
    wall_clock_ms: float
    gate8a_passed: bool


@dataclass(frozen=True, slots=True)
class RetrievalKAblationArm:
    """Summary of one arm in `Gate 8B`'s stratified `K`-ablation."""

    arm_name: str
    top_k: int
    num_meshes: int
    certified_count: int
    mean_iou_5x5x6: float
    median_iou_5x5x6: float
    distinct_parts_placed_count: int
    p50_wall_ms: float


@dataclass(frozen=True, slots=True)
class Gate8BAblationLicenseStyleReport:
    """Audit report for `Gate 8B`: Retrieval `K`-Ablation, License Filter & Chromatic/Style Study."""

    single_core_audit: SingleCoreAuditReport
    ablation_num_meshes: int
    ablation_arms: Tuple[RetrievalKAblationArm, ...]
    monotonic_topk_verified: bool
    nested_topk_subsets_verified: bool
    lift_top64_over_gate7a: float
    lift_top64_over_fixed10: float
    commercial_cc_by_4_num_meshes: int
    commercial_cc_by_4_certified_count: int
    commercial_cc_by_4_share_alike_parts_emitted: int
    commercial_cc_by_4_mean_iou_5x5x6: float
    chromatic_style_num_meshes: int
    chromatic_mean_delta_e00: float
    chromatic_max_delta_e00: float
    ldraw_mpd_roundtrip_exact_count: int
    default_exposed_upward_studs_total: int
    smooth_studless_exposed_upward_studs_total: int
    exposed_upward_stud_reduction_ratio: float
    smooth_studless_certified_count: int
    smooth_studless_mean_iou_delta: float
    wall_clock_ms: float
    gate8b_passed: bool


@dataclass(frozen=True, slots=True)
class Phase10CompleteGateSummary:
    """Unified Phase 10 promotion gate summary (`Gate 8A` + `Gate 8B`)."""

    gate8a: Gate8AOpenCatalogBenchmarkReport
    gate8b: Gate8BAblationLicenseStyleReport
    all_phase10_gates_passed: bool


def _warm_kernel_and_index(kernel: TopoBrickKernel) -> None:
    """Warm up catalog retrieval index and LP/kernel caches prior to latency measurement."""
    CATALOG_RETRIEVAL_INDEX._ensure_loaded()
    warm_tf = SparseTiledTargetField(boxes=(LDUBox(0, 0, 0, 40, 40, 24),))
    _ = kernel.realize(warm_tf, allowed_part_ids=EXTENDED_10PART_RECT_KIT)
    _ = kernel.realize(
        warm_tf,
        part_retrieval="open_catalog_top_k",
        retrieval_top_k=24,
        surface_refinement="open_catalog_sculpted",
    )


def run_gate8a_open_catalog_blind_mesh_benchmark(
    retrieval_top_k: int = 48,
    verify_all_independently: bool = True,
    max_p50_wall_ms: float = 220.0,
) -> Gate8AOpenCatalogBenchmarkReport:
    """
    Execute `Gate 8A` across all `80` frozen `Gate 2` unseen continuous triangle meshes
    under strict `OMP_NUM_THREADS=1` single-core execution.
    """
    t0_suite = time.perf_counter()
    sc_audit = enforce_and_audit_single_core()
    corpus, corpus_sha = load_frozen_80_unseen_mesh_corpus()
    kernel = TopoBrickKernel()
    _warm_kernel_and_index(kernel)

    cat_sha = compute_catalog_sha256()
    idx_sha = CATALOG_RETRIEVAL_INDEX.index_sha256

    coarse_ious: List[float] = []
    gate7a_ious: List[float] = []
    oc_ious_6: List[float] = []
    oc_ious_2: List[float] = []
    oc_chamfers: List[float] = []
    ret_ms_list: List[float] = []
    realize_ms_list: List[float] = []
    e2e_ms_list: List[float] = []

    cert_cnt = 0
    open_cnt = 0
    indep_tested = 0
    indep_passed = 0
    qual_085 = 0
    qual_080 = 0

    used_parts: set[str] = set()
    used_sculpt_classes: set[str] = set()

    # Per-band accumulator: (c_iou, g7a_iou, oc_iou6, oc_iou2, chamfer, is_cert, is_open, e2e_ms)
    by_band_rows: Dict[str, List[Tuple[float, float, float, float, float, bool, bool, float]]] = {}

    for spec in corpus:
        verts = spec["verts"]
        faces = spec["faces"]
        max_xy_studs = int(spec["max_xy_studs"])
        max_z_courses = int(spec["max_z_courses"])
        req_openings = tuple(spec["required_openings_ldu"])
        size_band = str(spec["size_band"])

        v_ldu = _scale_mesh_to_ldu(verts, max_xy_studs, max_z_courses)
        fine_occ, surf_pts = _rasterize_fine_mesh_5x5x6(v_ldu, faces)
        coarse_tf, t_vox_ms = voxelize_triangle_mesh_to_target_field(
            verts, faces, max_xy_studs=max_xy_studs, max_z_courses=max_z_courses
        )

        # Reference 1: Phase 8 coarse `20x20x24 LDU` S1
        res_c = kernel.realize(
            coarse_tf,
            solver_id="S1_CoverAndRepair",
            allowed_part_ids=EXTENDED_10PART_RECT_KIT,
            support_policy="minimal_permanent_pillars",
            forbidden_support_boxes=req_openings,
            t_target_init_ms=t_vox_ms,
        )
        fid_c = evaluate_assembly_against_original_mesh(
            fine_occ, surf_pts, coarse_tf, res_c.assembly, required_openings_ldu=req_openings
        )
        c_iou = float(fid_c.original_mesh_iou)
        coarse_ious.append(c_iou)

        # Reference 2: Phase 9 `Gate 7A` Fixed-27 adaptive kit
        ad_ref = AdaptiveSurfaceTarget.from_triangle_mesh(
            verts, faces, max_xy_studs=max_xy_studs, max_z_courses=max_z_courses
        )
        res_7a = kernel.realize(
            coarse_tf,
            solver_id="S1_CoverAndRepair",
            allowed_part_ids=ADAPTIVE_SURFACE_KIT,
            support_policy="minimal_permanent_pillars",
            forbidden_support_boxes=req_openings,
            surface_refinement="adaptive_plate_slope",
            adaptive_mesh_target=ad_ref,
            t_target_init_ms=ad_ref.rasterize_ms,
        )
        fid_7a = evaluate_assembly_against_original_mesh(
            fine_occ, surf_pts, coarse_tf, res_7a.assembly, required_openings_ldu=req_openings
        )
        g7a_iou = float(fid_7a.original_mesh_iou)
        gate7a_ious.append(g7a_iou)

        # Primary: Phase 10 Open-Catalog Online Top-K Retrieval + Sculpted Realization
        t_e2e_0 = time.perf_counter()
        ad_oc = AdaptiveSurfaceTarget.from_triangle_mesh(
            verts, faces, max_xy_studs=max_xy_studs, max_z_courses=max_z_courses
        )
        t_r0 = time.perf_counter()
        res_oc = kernel.realize(
            coarse_tf,
            solver_id="S1_CoverAndRepair",
            part_retrieval="open_catalog_top_k",
            retrieval_top_k=retrieval_top_k,
            surface_refinement="open_catalog_sculpted",
            adaptive_mesh_target=ad_oc,
            support_policy="minimal_permanent_pillars",
            forbidden_support_boxes=req_openings,
            t_target_init_ms=ad_oc.rasterize_ms,
        )
        dt_realize_ms = (time.perf_counter() - t_r0) * 1000.0
        dt_e2e_ms = (time.perf_counter() - t_e2e_0) * 1000.0

        fid_oc = evaluate_assembly_against_original_mesh(
            fine_occ, surf_pts, coarse_tf, res_oc.assembly, required_openings_ldu=req_openings
        )
        iou_6 = float(fid_oc.original_mesh_iou)
        iou_2 = float(evaluate_exact_5x5x2_mesh_iou(ad_oc, res_oc.assembly))
        cham_v = float(fid_oc.chamfer_l2_ldu)

        is_cert = res_oc.status == KernelResultStatus.COMPLETED_CERTIFIED
        is_open = bool(fid_oc.required_opening_clear)
        if is_cert:
            cert_cnt += 1
        if is_open:
            open_cnt += 1
        if is_cert and is_open and iou_6 >= 0.85:
            qual_085 += 1
        if is_cert and is_open and iou_6 >= 0.80:
            qual_080 += 1

        if verify_all_independently:
            indep_tested += 1
            art = export_certificate_artifact(
                res_oc.assembly,
                res_oc.certificates,
                validator=kernel.validator,
                target_field=ad_oc.to_fine_target_field(),
                support_policy="minimal_permanent_pillars",
            )
            vr = verify_certificate_independently(art)
            if vr.verified:
                indep_passed += 1

        diag = res_oc.open_catalog_diagnostics
        if diag is not None:
            ret_ms_list.append(float(diag.retrieval_result.query_wall_ms))
        for pl in res_oc.assembly.placed_parts.values():
            used_parts.add(pl.part_id)
            ent = CATALOG_RETRIEVAL_INDEX.get_entry(pl.part_id)
            if ent is not None and ent.taxonomy_class in SCULPTURAL_TAXONOMY_CLASSES:
                used_sculpt_classes.add(ent.taxonomy_class)

        oc_ious_6.append(iou_6)
        oc_ious_2.append(iou_2)
        oc_chamfers.append(cham_v)
        realize_ms_list.append(dt_realize_ms)
        e2e_ms_list.append(dt_e2e_ms)
        by_band_rows.setdefault(size_band, []).append(
            (c_iou, g7a_iou, iou_6, iou_2, cham_v, is_cert, is_open, dt_e2e_ms)
        )

    n_tot = len(corpus)
    band_summaries: List[Gate8ABandSummary] = []
    b24_ious: List[float] = []
    b24_qual90 = 0
    b24_cnt = 0
    b1_ious: List[float] = []

    for b_name in sorted(by_band_rows.keys()):
        rows = by_band_rows[b_name]
        nb = len(rows)
        b_oc6 = [r[2] for r in rows]
        b_q85 = sum(1 for r in rows if r[5] and r[6] and r[2] >= 0.85) / float(max(1, nb))
        b_q90 = sum(1 for r in rows if r[5] and r[6] and r[2] >= 0.90) / float(max(1, nb))
        if b_name.startswith("Band1"):
            b1_ious.extend(b_oc6)
        else:
            b24_ious.extend(b_oc6)
            b24_cnt += nb
            b24_qual90 += sum(1 for r in rows if r[5] and r[6] and r[2] >= 0.90)
        band_summaries.append(
            Gate8ABandSummary(
                size_band=b_name,
                num_meshes=nb,
                certified_count=sum(1 for r in rows if r[5]),
                openings_clear_count=sum(1 for r in rows if r[6]),
                coarse_mean_iou_5x5x6=round(float(np.mean([r[0] for r in rows])), 6),
                gate7a_mean_iou_5x5x6=round(float(np.mean([r[1] for r in rows])), 6),
                open_catalog_mean_iou_5x5x6=round(float(np.mean(b_oc6)), 6),
                open_catalog_median_iou_5x5x6=round(float(np.median(b_oc6)), 6),
                open_catalog_min_iou_5x5x6=round(float(np.min(b_oc6)), 6),
                open_catalog_mean_iou_5x5x2=round(float(np.mean([r[3] for r in rows])), 6),
                open_catalog_mean_chamfer_ldu=round(float(np.mean([r[4] for r in rows])), 4),
                qualified_at_085_rate=round(float(b_q85), 6),
                qualified_at_090_rate=round(float(b_q90), 6),
                p50_wall_ms=round(float(np.median([r[7] for r in rows])), 2),
                p90_wall_ms=round(float(np.percentile([r[7] for r in rows], 90)), 2),
            )
        )

    c_mean = float(np.mean(coarse_ious))
    g7a_mean = float(np.mean(gate7a_ious))
    oc_mean = float(np.mean(oc_ious_6))
    oc_med = float(np.median(oc_ious_6))
    b24_mean = float(np.mean(b24_ious)) if b24_ious else 0.0
    b24_med = float(np.median(b24_ious)) if b24_ious else 0.0
    b24_q90_rate = float(b24_qual90) / float(max(1, b24_cnt))
    b1_mean = float(np.mean(b1_ious)) if b1_ious else 0.0
    q85_rate = float(qual_085) / float(max(1, n_tot))
    q80_rate = float(qual_080) / float(max(1, n_tot))
    loss_red = (oc_mean - c_mean) / max(1e-9, 1.0 - c_mean)

    ret_p50 = float(np.median(ret_ms_list)) if ret_ms_list else 0.0
    ret_p90 = float(np.percentile(ret_ms_list, 90)) if ret_ms_list else 0.0
    real_p50 = float(np.median(realize_ms_list))
    e2e_p50 = float(np.median(e2e_ms_list))
    e2e_p90 = float(np.percentile(e2e_ms_list, 90))

    gate8a_ok = (
        sc_audit.strictly_single_core_verified
        and cert_cnt == n_tot
        and open_cnt == n_tot
        and indep_passed == indep_tested
        and b24_mean >= 0.920
        and b24_q90_rate >= 0.75
        and oc_med >= 0.925
        and oc_mean >= 0.905
        and b1_mean >= 0.820
        and loss_red >= 0.62
        and q85_rate >= 0.80
        and len(used_parts) >= 45
        and len(used_sculpt_classes) >= 5
        and ret_p50 < 5.0
        and e2e_p50 < max_p50_wall_ms
    )

    return Gate8AOpenCatalogBenchmarkReport(
        single_core_audit=sc_audit,
        corpus_sha256=corpus_sha,
        catalog_sha256=cat_sha,
        retrieval_index_sha256=idx_sha,
        retrieval_top_k=retrieval_top_k,
        num_meshes=n_tot,
        certified_count=cert_cnt,
        certified_rate=round(float(cert_cnt) / float(max(1, n_tot)), 6),
        openings_clear_count=open_cnt,
        independent_verifier_tested=indep_tested,
        independent_verifier_passed=indep_passed,
        coarse_mean_iou_5x5x6=round(c_mean, 6),
        gate7a_mean_iou_5x5x6=round(g7a_mean, 6),
        open_catalog_mean_iou_5x5x6=round(oc_mean, 6),
        open_catalog_median_iou_5x5x6=round(oc_med, 6),
        bands_2_to_4_mean_iou_5x5x6=round(b24_mean, 6),
        bands_2_to_4_median_iou_5x5x6=round(b24_med, 6),
        bands_2_to_4_qualified_at_090_rate=round(b24_q90_rate, 6),
        band1_mean_iou_5x5x6=round(b1_mean, 6),
        overall_qualified_at_085_rate=round(q85_rate, 6),
        overall_qualified_at_080_rate=round(q80_rate, 6),
        open_catalog_mean_iou_5x5x2=round(float(np.mean(oc_ious_2)), 6),
        discretization_loss_reduction_vs_coarse=round(loss_red, 6),
        distinct_parts_placed_count=len(used_parts),
        distinct_parts_placed=tuple(sorted(used_parts)),
        sculptural_classes_exercised=tuple(sorted(used_sculpt_classes)),
        retrieval_query_p50_ms=round(ret_p50, 4),
        retrieval_query_p90_ms=round(ret_p90, 4),
        realize_only_p50_ms=round(real_p50, 2),
        end_to_end_p50_ms=round(e2e_p50, 2),
        end_to_end_p90_ms=round(e2e_p90, 2),
        by_band=tuple(band_summaries),
        wall_clock_ms=round((time.perf_counter() - t0_suite) * 1000.0, 2),
        gate8a_passed=gate8a_ok,
    )


def _select_stratified_meshes(
    corpus: Sequence[Dict[str, Any]], per_band: int
) -> List[Dict[str, Any]]:
    """Deterministically select `per_band` meshes from each of the `4` size bands."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for spec in corpus:
        grouped.setdefault(str(spec["size_band"]), []).append(spec)
    selected: List[Dict[str, Any]] = []
    for b_name in sorted(grouped.keys()):
        rows = grouped[b_name]
        selected.extend(rows[:per_band])
    return selected


def _generate_multi_region_lego_vertex_colors(
    v_ldu: np.ndarray, mesh_idx: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assign deterministic multi-region LEGO sRGB colors (`3` distinct architectural/sculptural
    color zones along height Z + slight realistic lighting jitter `±4/255`) to `v_ldu`, returning
    `(noisy_vertex_rgb_u8, ground_truth_clean_rgb_u8)`.
    """
    palette_subset = (
        LDRAW_SOLID_PALETTE_SRGB[4],   # Red (4)
        LDRAW_SOLID_PALETTE_SRGB[1],   # Blue (1)
        LDRAW_SOLID_PALETTE_SRGB[14],  # Yellow (14)
        LDRAW_SOLID_PALETTE_SRGB[15],  # White (15)
        LDRAW_SOLID_PALETTE_SRGB[2],   # Green (2)
        LDRAW_SOLID_PALETTE_SRGB[21],  # Orange (25)
        LDRAW_SOLID_PALETTE_SRGB[18],  # Tan (19)
        LDRAW_SOLID_PALETTE_SRGB[31],  # Dark Bluish Gray (72)
        LDRAW_SOLID_PALETTE_SRGB[45],  # Dark Azur (321)
        LDRAW_SOLID_PALETTE_SRGB[49],  # Olive Green (330)
    )
    c_low = np.array(palette_subset[(3 * mesh_idx) % len(palette_subset)][2], dtype=np.float64)
    c_mid = np.array(palette_subset[(3 * mesh_idx + 3) % len(palette_subset)][2], dtype=np.float64)
    c_top = np.array(palette_subset[(3 * mesh_idx + 7) % len(palette_subset)][2], dtype=np.float64)

    z_vals = v_ldu[:, 2]
    z_min = float(np.min(z_vals))
    z_span = max(1.0, float(np.max(z_vals)) - z_min)
    z_norm = (z_vals - z_min) / z_span

    clean_rgb = np.zeros((v_ldu.shape[0], 3), dtype=np.float64)
    for i in range(v_ldu.shape[0]):
        if z_norm[i] < 0.34:
            clean_rgb[i] = c_low
        elif z_norm[i] < 0.68:
            clean_rgb[i] = c_mid
        else:
            clean_rgb[i] = c_top

    rng = np.random.default_rng(10000 + mesh_idx)
    jitter = rng.uniform(-3.5, 3.5, size=clean_rgb.shape)
    noisy_rgb = np.clip(np.round(clean_rgb + jitter), 0.0, 255.0).astype(np.uint8)
    return noisy_rgb, clean_rgb.astype(np.uint8)


def run_gate8b_retrieval_ablation_license_and_style_study() -> Gate8BAblationLicenseStyleReport:
    """
    Execute `Gate 8B` under strict `OMP_NUM_THREADS=1` single-core execution:
      1. Paired `n = 40` stratified (`10` per band) retrieval `K`-ablation (`Fixed-10`, `Fixed-27`
         `Gate 7A`, `Open-Catalog Top-16`, `Top-32`, `Top-64`).
      2. Strict commercial `license_filter="cc_by_4_0_only"` audit across all `40` meshes (`0`
         `CC-BY-SA-4.0` parts emitted).
      3. Option 3B perceptual `CIEDE2000` (`ΔE_00 <= 8.0`) LDraw color + `"smooth_studless"`
         stud-reduction (`>= 60%`) study across `n = 20` multi-region colored meshes.
    """
    t0_suite = time.perf_counter()
    sc_audit = enforce_and_audit_single_core()
    corpus, _ = load_frozen_80_unseen_mesh_corpus()
    kernel = TopoBrickKernel()
    _warm_kernel_and_index(kernel)

    ablation_meshes = _select_stratified_meshes(corpus, per_band=10)
    n_abl = len(ablation_meshes)

    arm_names = ("Fixed_10_Coarse", "Fixed_27_Gate7A", "OpenCatalog_Top16", "OpenCatalog_Top32", "OpenCatalog_Top64")
    arm_ious: Dict[str, List[float]] = {k: [] for k in arm_names}
    arm_dts: Dict[str, List[float]] = {k: [] for k in arm_names}
    arm_certs: Dict[str, int] = {k: 0 for k in arm_names}
    arm_parts: Dict[str, set[str]] = {k: set() for k in arm_names}

    nested_ok = True
    cc4_certs = 0
    cc4_sa_emitted = 0
    cc4_ious: List[float] = []

    for spec in ablation_meshes:
        verts = spec["verts"]
        faces = spec["faces"]
        max_xy_studs = int(spec["max_xy_studs"])
        max_z_courses = int(spec["max_z_courses"])
        req_openings = tuple(spec["required_openings_ldu"])

        v_ldu = _scale_mesh_to_ldu(verts, max_xy_studs, max_z_courses)
        fine_occ, surf_pts = _rasterize_fine_mesh_5x5x6(v_ldu, faces)
        coarse_tf, t_vox_ms = voxelize_triangle_mesh_to_target_field(
            verts, faces, max_xy_studs=max_xy_studs, max_z_courses=max_z_courses
        )
        ad_t = AdaptiveSurfaceTarget.from_triangle_mesh(
            verts, faces, max_xy_studs=max_xy_studs, max_z_courses=max_z_courses
        )

        # Verify nested retrieval kits `Top-16 ⊂ Top-32 ⊂ Top-64`
        r16 = set(CATALOG_RETRIEVAL_INDEX.retrieve_top_k(ad_t.occ_5x5x6, top_k=16).retrieved_part_ids)
        r32 = set(CATALOG_RETRIEVAL_INDEX.retrieve_top_k(ad_t.occ_5x5x6, top_k=32).retrieved_part_ids)
        r64 = set(CATALOG_RETRIEVAL_INDEX.retrieve_top_k(ad_t.occ_5x5x6, top_k=64).retrieved_part_ids)
        if not (r16.issubset(r32) and r32.issubset(r64)):
            nested_ok = False

        # Arm 1: Fixed-10 Coarse
        t_0 = time.perf_counter()
        res_10 = kernel.realize(
            coarse_tf,
            solver_id="S1_CoverAndRepair",
            allowed_part_ids=EXTENDED_10PART_RECT_KIT,
            support_policy="minimal_permanent_pillars",
            forbidden_support_boxes=req_openings,
            t_target_init_ms=t_vox_ms,
        )
        arm_dts["Fixed_10_Coarse"].append((time.perf_counter() - t_0) * 1000.0 + t_vox_ms)
        fid_10 = evaluate_assembly_against_original_mesh(
            fine_occ, surf_pts, coarse_tf, res_10.assembly, required_openings_ldu=req_openings
        )
        arm_ious["Fixed_10_Coarse"].append(float(fid_10.original_mesh_iou))
        if res_10.status == KernelResultStatus.COMPLETED_CERTIFIED and fid_10.required_opening_clear:
            arm_certs["Fixed_10_Coarse"] += 1
        for p in res_10.assembly.placed_parts.values():
            arm_parts["Fixed_10_Coarse"].add(p.part_id)

        # Arm 2: Fixed-27 Gate 7A
        t_0 = time.perf_counter()
        res_27 = kernel.realize(
            coarse_tf,
            solver_id="S1_CoverAndRepair",
            allowed_part_ids=ADAPTIVE_SURFACE_KIT,
            support_policy="minimal_permanent_pillars",
            forbidden_support_boxes=req_openings,
            surface_refinement="adaptive_plate_slope",
            adaptive_mesh_target=ad_t,
            t_target_init_ms=ad_t.rasterize_ms,
        )
        arm_dts["Fixed_27_Gate7A"].append((time.perf_counter() - t_0) * 1000.0 + ad_t.rasterize_ms)
        fid_27 = evaluate_assembly_against_original_mesh(
            fine_occ, surf_pts, coarse_tf, res_27.assembly, required_openings_ldu=req_openings
        )
        arm_ious["Fixed_27_Gate7A"].append(float(fid_27.original_mesh_iou))
        if res_27.status == KernelResultStatus.COMPLETED_CERTIFIED and fid_27.required_opening_clear:
            arm_certs["Fixed_27_Gate7A"] += 1
        for p in res_27.assembly.placed_parts.values():
            arm_parts["Fixed_27_Gate7A"].add(p.part_id)

        # Arms 3..5: Open-Catalog Top-16, Top-32, Top-64
        for arm_k, k_val in (("OpenCatalog_Top16", 16), ("OpenCatalog_Top32", 32), ("OpenCatalog_Top64", 64)):
            t_0 = time.perf_counter()
            res_k = kernel.realize(
                coarse_tf,
                solver_id="S1_CoverAndRepair",
                part_retrieval="open_catalog_top_k",
                retrieval_top_k=k_val,
                surface_refinement="open_catalog_sculpted",
                adaptive_mesh_target=ad_t,
                support_policy="minimal_permanent_pillars",
                forbidden_support_boxes=req_openings,
                t_target_init_ms=ad_t.rasterize_ms,
            )
            arm_dts[arm_k].append((time.perf_counter() - t_0) * 1000.0 + ad_t.rasterize_ms)
            fid_k = evaluate_assembly_against_original_mesh(
                fine_occ, surf_pts, coarse_tf, res_k.assembly, required_openings_ldu=req_openings
            )
            arm_ious[arm_k].append(float(fid_k.original_mesh_iou))
            if res_k.status == KernelResultStatus.COMPLETED_CERTIFIED and fid_k.required_opening_clear:
                arm_certs[arm_k] += 1
            for p in res_k.assembly.placed_parts.values():
                arm_parts[arm_k].add(p.part_id)

        # Commercial CC-BY-4.0-only audit (`0` CC-BY-SA-4.0 Shadow parts allowed)
        res_cc4 = kernel.realize(
            coarse_tf,
            solver_id="S1_CoverAndRepair",
            part_retrieval="open_catalog_top_k",
            retrieval_top_k=32,
            license_filter="cc_by_4_0_only",
            surface_refinement="open_catalog_sculpted",
            adaptive_mesh_target=ad_t,
            support_policy="minimal_permanent_pillars",
            forbidden_support_boxes=req_openings,
            t_target_init_ms=ad_t.rasterize_ms,
        )
        fid_cc4 = evaluate_assembly_against_original_mesh(
            fine_occ, surf_pts, coarse_tf, res_cc4.assembly, required_openings_ldu=req_openings
        )
        if res_cc4.status == KernelResultStatus.COMPLETED_CERTIFIED and fid_cc4.required_opening_clear:
            cc4_certs += 1
        cc4_ious.append(float(fid_cc4.original_mesh_iou))
        for p in res_cc4.assembly.placed_parts.values():
            if p.part_id not in EXTENDED_BRICK_CATALOG:
                dspec = get_declarative_part_spec(p.part_id)
                conn_lic = dspec.provenance.connector_license if dspec.provenance is not None else ""
                if "SA" in conn_lic.upper():
                    cc4_sa_emitted += 1

    ablation_summaries: List[RetrievalKAblationArm] = []
    arm_k_map = {
        "Fixed_10_Coarse": 10,
        "Fixed_27_Gate7A": 27,
        "OpenCatalog_Top16": 16,
        "OpenCatalog_Top32": 32,
        "OpenCatalog_Top64": 64,
    }
    for a_name in arm_names:
        ablation_summaries.append(
            RetrievalKAblationArm(
                arm_name=a_name,
                top_k=arm_k_map[a_name],
                num_meshes=n_abl,
                certified_count=arm_certs[a_name],
                mean_iou_5x5x6=round(float(np.mean(arm_ious[a_name])), 6),
                median_iou_5x5x6=round(float(np.median(arm_ious[a_name])), 6),
                distinct_parts_placed_count=len(arm_parts[a_name]),
                p50_wall_ms=round(float(np.median(arm_dts[a_name])), 2),
            )
        )

    m10 = float(np.mean(arm_ious["Fixed_10_Coarse"]))
    m27 = float(np.mean(arm_ious["Fixed_27_Gate7A"]))
    m16 = float(np.mean(arm_ious["OpenCatalog_Top16"]))
    m32 = float(np.mean(arm_ious["OpenCatalog_Top32"]))
    m64 = float(np.mean(arm_ious["OpenCatalog_Top64"]))
    monotonic_ok = (m16 <= m32 + 1e-6) and (m32 <= m64 + 1e-6)
    lift_vs_7a = m64 - m27
    lift_vs_10 = m64 - m10

    # 3. Option 3B Chromatic (`CIEDE2000`) & `"smooth_studless"` Study on `n = 20` Stratified Meshes (`5` / band)
    style_meshes = _select_stratified_meshes(corpus, per_band=5)
    n_style = len(style_meshes)
    palette_id_to_rgb = {cid: np.array(rgb, dtype=np.float64) for cid, _, rgb in LDRAW_SOLID_PALETTE_SRGB}

    gt_de00_list: List[float] = []
    mpd_roundtrip_ok = 0
    default_studs_tot = 0
    smooth_studs_tot = 0
    smooth_certs = 0
    iou_deltas_smooth: List[float] = []

    for m_idx, spec in enumerate(style_meshes):
        verts = spec["verts"]
        faces = spec["faces"]
        max_xy_studs = int(spec["max_xy_studs"])
        max_z_courses = int(spec["max_z_courses"])
        req_openings = tuple(spec["required_openings_ldu"])

        v_ldu = _scale_mesh_to_ldu(verts, max_xy_studs, max_z_courses)
        fine_occ, surf_pts = _rasterize_fine_mesh_5x5x6(v_ldu, faces)
        coarse_tf, _ = voxelize_triangle_mesh_to_target_field(
            verts, faces, max_xy_studs=max_xy_studs, max_z_courses=max_z_courses
        )
        ad_s = AdaptiveSurfaceTarget.from_triangle_mesh(
            verts, faces, max_xy_studs=max_xy_studs, max_z_courses=max_z_courses
        )
        noisy_rgb, clean_rgb = _generate_multi_region_lego_vertex_colors(v_ldu, m_idx)

        # Baseline structural core/plate realization (`no smooth_studless decomposition`, `ADAPTIVE_SURFACE_KIT` exposed studs)
        res_base_studs = kernel.realize(
            coarse_tf,
            solver_id="S1_CoverAndRepair",
            allowed_part_ids=ADAPTIVE_SURFACE_KIT,
            surface_refinement="adaptive_plate_slope",
            adaptive_mesh_target=ad_s,
            support_policy="minimal_permanent_pillars",
            forbidden_support_boxes=req_openings,
        )
        default_studs_tot += count_exposed_upward_studs(res_base_studs.assembly)

        # Default Open-Catalog (`semantic_style_hint=None`) for exact IoU comparison against `"smooth_studless"`
        res_oc_def = kernel.realize(
            coarse_tf,
            solver_id="S1_CoverAndRepair",
            part_retrieval="open_catalog_top_k",
            retrieval_top_k=36,
            surface_refinement="open_catalog_sculpted",
            adaptive_mesh_target=ad_s,
            support_policy="minimal_permanent_pillars",
            forbidden_support_boxes=req_openings,
        )
        fid_def = evaluate_assembly_against_original_mesh(
            fine_occ, surf_pts, coarse_tf, res_oc_def.assembly, required_openings_ldu=req_openings
        )

        # Option 3B: `semantic_style_hint="smooth_studless"` + `vertex_colors_rgb=noisy_rgb`
        res_sm = kernel.realize(
            coarse_tf,
            solver_id="S1_CoverAndRepair",
            part_retrieval="open_catalog_top_k",
            retrieval_top_k=36,
            semantic_style_hint="smooth_studless",
            vertex_colors_rgb=noisy_rgb,
            verts_ldu=v_ldu,
            surface_refinement="open_catalog_sculpted",
            adaptive_mesh_target=ad_s,
            support_policy="minimal_permanent_pillars",
            forbidden_support_boxes=req_openings,
        )
        fid_sm = evaluate_assembly_against_original_mesh(
            fine_occ, surf_pts, coarse_tf, res_sm.assembly, required_openings_ldu=req_openings
        )
        if res_sm.status == KernelResultStatus.COMPLETED_CERTIFIED and fid_sm.required_opening_clear:
            smooth_certs += 1
        smooth_studs_tot += count_exposed_upward_studs(res_sm.assembly)
        iou_deltas_smooth.append(float(fid_sm.original_mesh_iou) - float(fid_def.original_mesh_iou))

        # Evaluate perceptual `CIEDE2000` error of each placed part's assigned LDraw color vs. clean ground-truth vertex color
        clean_lab = srgb_to_cielab(clean_rgb.astype(np.float64))
        part_errs: List[float] = []
        for pl in res_sm.assembly.placed_parts.values():
            cx = 0.5 * (min(b.xmin for b in pl.world_boxes) + max(b.xmax for b in pl.world_boxes))
            cy = 0.5 * (min(b.ymin for b in pl.world_boxes) + max(b.ymax for b in pl.world_boxes))
            cz = 0.5 * (min(b.zmin for b in pl.world_boxes) + max(b.zmax for b in pl.world_boxes))
            d2 = np.sum((v_ldu - np.array([[cx, cy, cz]], dtype=np.float64)) ** 2, axis=1)
            nn_idx = np.argsort(d2)[: min(8, v_ldu.shape[0])]
            gt_lab_mean = np.mean(clean_lab[nn_idx], axis=0)
            assigned_rgb = palette_id_to_rgb.get(pl.color_id, np.array([244.0, 244.0, 244.0]))
            assigned_lab = srgb_to_cielab(assigned_rgb)
            part_errs.append(float(ciede2000_delta_e(assigned_lab, gt_lab_mean)))
        gt_de00_list.append(float(np.mean(part_errs)) if part_errs else 0.0)

        # Verify LDraw `.mpd` Type-1 export round-trips part count & assigned color IDs
        mpd_str = res_sm.assembly.to_ldraw_mpd(model_name=f"style_{m_idx}.ldr")
        parsed_specs = parse_ldr_text(mpd_str, use_z_up=False, catalog=COMBINED_BRICK_CATALOG)  # type: ignore[arg-type]
        exported_colors = sorted(pl.color_id for pl in res_sm.assembly.placed_parts.values())
        parsed_colors = sorted(sp.color_code for sp in parsed_specs)
        if len(parsed_specs) == res_sm.assembly.num_parts and exported_colors == parsed_colors:
            mpd_roundtrip_ok += 1

    stud_red_ratio = 1.0 - (float(smooth_studs_tot) / float(max(1, default_studs_tot)))
    mean_de00 = float(np.mean(gt_de00_list))
    max_de00 = float(np.max(gt_de00_list))
    mean_sm_iou_delta = float(np.mean(iou_deltas_smooth))

    gate8b_ok = (
        sc_audit.strictly_single_core_verified
        and all(arm_certs[a] == n_abl for a in arm_names)
        and nested_ok
        and monotonic_ok
        and lift_vs_7a >= 0.055
        and lift_vs_10 >= 0.140
        and cc4_certs == n_abl
        and cc4_sa_emitted == 0
        and mean_de00 <= 8.0
        and mpd_roundtrip_ok == n_style
        and smooth_certs == n_style
        and stud_red_ratio >= 0.60
        and abs(mean_sm_iou_delta) <= 0.005
    )

    return Gate8BAblationLicenseStyleReport(
        single_core_audit=sc_audit,
        ablation_num_meshes=n_abl,
        ablation_arms=tuple(ablation_summaries),
        monotonic_topk_verified=monotonic_ok,
        nested_topk_subsets_verified=nested_ok,
        lift_top64_over_gate7a=round(lift_vs_7a, 6),
        lift_top64_over_fixed10=round(lift_vs_10, 6),
        commercial_cc_by_4_num_meshes=n_abl,
        commercial_cc_by_4_certified_count=cc4_certs,
        commercial_cc_by_4_share_alike_parts_emitted=cc4_sa_emitted,
        commercial_cc_by_4_mean_iou_5x5x6=round(float(np.mean(cc4_ious)), 6),
        chromatic_style_num_meshes=n_style,
        chromatic_mean_delta_e00=round(mean_de00, 4),
        chromatic_max_delta_e00=round(max_de00, 4),
        ldraw_mpd_roundtrip_exact_count=mpd_roundtrip_ok,
        default_exposed_upward_studs_total=default_studs_tot,
        smooth_studless_exposed_upward_studs_total=smooth_studs_tot,
        exposed_upward_stud_reduction_ratio=round(stud_red_ratio, 6),
        smooth_studless_certified_count=smooth_certs,
        smooth_studless_mean_iou_delta=round(mean_sm_iou_delta, 6),
        wall_clock_ms=round((time.perf_counter() - t0_suite) * 1000.0, 2),
        gate8b_passed=gate8b_ok,
    )


def run_all_phase10_gates(
    output_json_path: Optional[Path] = None,
    max_p50_wall_ms: float = 220.0,
) -> Phase10CompleteGateSummary:
    """Run both `Gate 8A` and `Gate 8B` under strict `1`-core auditing and optionally emit JSON."""
    g8a = run_gate8a_open_catalog_blind_mesh_benchmark(
        retrieval_top_k=48,
        verify_all_independently=True,
        max_p50_wall_ms=max_p50_wall_ms,
    )
    g8b = run_gate8b_retrieval_ablation_license_and_style_study()
    summary = Phase10CompleteGateSummary(
        gate8a=g8a,
        gate8b=g8b,
        all_phase10_gates_passed=bool(g8a.gate8a_passed and g8b.gate8b_passed),
    )
    if output_json_path is not None:
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 10 Open-Catalog Retrieval Gates (8A & 8B).")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path.")
    parser.add_argument("--max-p50-ms", type=float, default=220.0, help="Max allowed p50 wall ms.")
    args = parser.parse_args()

    out_path = Path(args.output) if args.output else None
    summary = run_all_phase10_gates(output_json_path=out_path, max_p50_wall_ms=float(args.max_p50_ms))
    print(
        json.dumps(
            {
                "all_phase10_gates_passed": summary.all_phase10_gates_passed,
                "gate8a_passed": summary.gate8a.gate8a_passed,
                "gate8a_mean_iou_5x5x6": summary.gate8a.open_catalog_mean_iou_5x5x6,
                "gate8a_median_iou_5x5x6": summary.gate8a.open_catalog_median_iou_5x5x6,
                "gate8a_bands_2_to_4_mean_iou": summary.gate8a.bands_2_to_4_mean_iou_5x5x6,
                "gate8a_band1_mean_iou": summary.gate8a.band1_mean_iou_5x5x6,
                "gate8a_distinct_parts": summary.gate8a.distinct_parts_placed_count,
                "gate8a_sculptural_classes": summary.gate8a.sculptural_classes_exercised,
                "gate8a_retrieval_p50_ms": summary.gate8a.retrieval_query_p50_ms,
                "gate8a_e2e_p50_ms": summary.gate8a.end_to_end_p50_ms,
                "gate8b_passed": summary.gate8b.gate8b_passed,
                "gate8b_lift_over_gate7a": summary.gate8b.lift_top64_over_gate7a,
                "gate8b_cc4_sa_emitted": summary.gate8b.commercial_cc_by_4_share_alike_parts_emitted,
                "gate8b_mean_delta_e00": summary.gate8b.chromatic_mean_delta_e00,
                "gate8b_stud_reduction_ratio": summary.gate8b.exposed_upward_stud_reduction_ratio,
            },
            indent=2,
        )
    )
    if not summary.all_phase10_gates_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
