"""
Phase 10 (`Gate 8A` & `Gate 8B`) Pytest Verification Suite (`6` tests):

1. `test_p10_catalog_retrieval_index_deduplication_latency_and_nested_topk`
2. `test_p10_cc_by_4_0_only_license_filter_emits_zero_share_alike_parts`
3. `test_p10_option_3b_terminal_tile_swap_smooth_studless_and_cielab_colors`
4. `test_p10_independent_verifier_ast_isolation_and_index_sha256_tamper`
5. `test_p10_gate8a_open_catalog_80_mesh_benchmark_passes_all_thresholds`
6. `test_p10_gate8b_k_ablation_commercial_license_and_chromatic_study`
"""

from __future__ import annotations

import ast
from pathlib import Path
import time

import numpy as np

from topo_brick import (
    CATALOG_RETRIEVAL_INDEX,
    COMBINED_BRICK_CATALOG,
    MOLD_TAXONOMY_CLASSES,
    SCULPTURAL_TAXONOMY_CLASSES,
    EXTENDED_BRICK_CATALOG,
    KernelResultStatus,
    LDUBox,
    SparseTiledTargetField,
    TopoBrickKernel,
    VersionedCertificateArtifact,
    compute_catalog_sha256,
    count_exposed_upward_studs,
    export_certificate_artifact,
    get_declarative_part_spec,
    parse_ldr_text,
    run_gate8a_open_catalog_blind_mesh_benchmark,
    run_gate8b_retrieval_ablation_license_and_style_study,
    verify_certificate_independently,
)
import topo_brick.evaluation.independent_verifier as iv_mod


def test_p10_catalog_retrieval_index_deduplication_latency_and_nested_topk() -> None:
    """
    Verify `CATALOG_RETRIEVAL_INDEX` (`compiled_mold_retrieval_index.npz`) deduplicates
    `COMBINED_BRICK_CATALOG` (`16,085` parts) into canonical System buildable molds across all
    `12` taxonomy classes, executes online `retrieve_top_k(...)` in `< 5.0 ms` `p50` single-core,
    and guarantees strictly nested `Top-16 ⊂ Top-32 ⊂ Top-64` candidate sets.
    """
    assert CATALOG_RETRIEVAL_INDEX.is_available
    assert CATALOG_RETRIEVAL_INDEX.num_molds >= 200
    assert len(CATALOG_RETRIEVAL_INDEX.index_sha256) == 64

    present_classes = {e.taxonomy_class for e in CATALOG_RETRIEVAL_INDEX.entries}
    for cls_name in MOLD_TAXONOMY_CLASSES:
        assert cls_name in present_classes

    # Synthetic sloped + stepped `5x5x6 LDU` target occupancy
    occ6 = np.zeros((24, 16, 16), dtype=bool)
    for ix in range(24):
        for iy in range(16):
            z_lim = max(2, 16 - (ix // 2))
            occ6[ix, iy, :z_lim] = True

    # Warm once, then measure p50 over 25 online queries
    _ = CATALOG_RETRIEVAL_INDEX.retrieve_top_k(occ6, top_k=32)
    latencies_ms = []
    for _ in range(25):
        t0 = time.perf_counter()
        r32 = CATALOG_RETRIEVAL_INDEX.retrieve_top_k(occ6, top_k=32)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        assert len(r32.retrieved_part_ids) == 32

    assert float(np.median(latencies_ms)) < 5.0

    r16 = CATALOG_RETRIEVAL_INDEX.retrieve_top_k(occ6, top_k=16)
    r32 = CATALOG_RETRIEVAL_INDEX.retrieve_top_k(occ6, top_k=32)
    r64 = CATALOG_RETRIEVAL_INDEX.retrieve_top_k(occ6, top_k=64)
    s16 = set(r16.retrieved_part_ids)
    s32 = set(r32.retrieved_part_ids)
    s64 = set(r64.retrieved_part_ids)
    assert len(s16) == 16 and len(s32) == 32 and len(s64) == 64
    assert s16.issubset(s32)
    assert s32.issubset(s64)

    sculpt_in_32 = {
        CATALOG_RETRIEVAL_INDEX.get_entry(pid).taxonomy_class  # type: ignore[union-attr]
        for pid in s32
        if CATALOG_RETRIEVAL_INDEX.get_entry(pid) is not None
        and CATALOG_RETRIEVAL_INDEX.get_entry(pid).taxonomy_class in SCULPTURAL_TAXONOMY_CLASSES  # type: ignore[union-attr]
    }
    assert len(sculpt_in_32) >= 4


def test_p10_cc_by_4_0_only_license_filter_emits_zero_share_alike_parts() -> None:
    """
    Verify `license_filter="cc_by_4_0_only"` strictly filters out `TIER_B` (`CC-BY-SA-4.0`)
    parts in both `CATALOG_RETRIEVAL_INDEX.retrieve_top_k(...)` and `TopoBrickKernel.realize(...)`.
    """
    occ6 = np.ones((16, 12, 12), dtype=bool)
    occ6[8:, :, 6:] = False
    ret_cc4 = CATALOG_RETRIEVAL_INDEX.retrieve_top_k(
        occ6, top_k=40, license_filter="cc_by_4_0_only"
    )
    for pid in ret_cc4.retrieved_part_ids:
        if pid not in EXTENDED_BRICK_CATALOG:
            spec = get_declarative_part_spec(pid)
            assert "CC-BY-SA" not in spec.provenance.connector_license
            assert spec.provenance.redistribution_status != "SHARE_ALIKE_ADAPTATION"

    kernel = TopoBrickKernel()
    tf = SparseTiledTargetField(
        boxes=(
            LDUBox(0, 0, 0, 80, 60, 24),
            LDUBox(0, 0, 24, 40, 60, 48),
        )
    )
    res = kernel.realize(
        tf,
        part_retrieval="open_catalog_top_k",
        retrieval_top_k=36,
        license_filter="cc_by_4_0_only",
        surface_refinement="open_catalog_sculpted",
    )
    assert res.status == KernelResultStatus.COMPLETED_CERTIFIED
    for pl in res.assembly.placed_parts.values():
        if pl.part_id not in EXTENDED_BRICK_CATALOG:
            spec = get_declarative_part_spec(pl.part_id)
            assert "CC-BY-SA" not in spec.provenance.connector_license


def test_p10_option_3b_terminal_tile_swap_smooth_studless_and_cielab_colors() -> None:
    """
    Verify Option 3B Chromatic & Studless-Tile Skinning:
    1. `semantic_style_hint="smooth_studless"` decomposes exposed flat-topped `24 LDU` courses
       into cross-bonded `8 LDU` plates + smooth tiles, cutting exposed upward studs by `>= 60%`.
    2. `vertex_colors_rgb` assigns CIE Lab (`CIEDE2000`) LDraw palette IDs and round-trips
       cleanly through `AssemblyState.to_ldraw_mpd(...)` and `parse_ldr_text(...)`.
    """
    kernel = TopoBrickKernel()
    tf = SparseTiledTargetField(
        boxes=(
            LDUBox(0, 0, 0, 120, 80, 24),
            LDUBox(20, 0, 24, 100, 80, 48),
        )
    )
    res_default = kernel.realize(
        tf,
        part_retrieval="open_catalog_top_k",
        retrieval_top_k=36,
        surface_refinement="open_catalog_sculpted",
    )
    assert res_default.status == KernelResultStatus.COMPLETED_CERTIFIED
    studs_default = count_exposed_upward_studs(res_default.assembly)
    assert studs_default > 0

    # Build 2-region vertex color cloud (Bright Red lower course, Bright Blue upper course)
    pts = []
    rgbs = []
    for x in range(5, 120, 15):
        for y in range(5, 80, 15):
            pts.append([float(x), float(y), 12.0])
            rgbs.append([201.0, 26.0, 9.0])  # LDraw Bright Red (id=4)
            pts.append([float(x), float(y), 36.0])
            rgbs.append([0.0, 85.0, 191.0])  # LDraw Bright Blue (id=1)
    verts_ldu = np.asarray(pts, dtype=np.float64)
    vertex_colors_rgb = np.asarray(rgbs, dtype=np.float64)

    res_smooth = kernel.realize(
        tf,
        part_retrieval="open_catalog_top_k",
        retrieval_top_k=36,
        semantic_style_hint="smooth_studless",
        vertex_colors_rgb=vertex_colors_rgb,
        verts_ldu=verts_ldu,
        surface_refinement="open_catalog_sculpted",
    )
    assert res_smooth.status == KernelResultStatus.COMPLETED_CERTIFIED
    studs_smooth = count_exposed_upward_studs(res_smooth.assembly)
    assert studs_smooth <= int(0.40 * studs_default)

    assigned_colors = {pl.color_id for pl in res_smooth.assembly.placed_parts.values()}
    assert len(assigned_colors) >= 2

    mpd_text = res_smooth.assembly.to_ldraw_mpd(model_name="option3b_test.ldr")
    parsed = parse_ldr_text(mpd_text, use_z_up=False, catalog=COMBINED_BRICK_CATALOG)  # type: ignore[arg-type]
    assert len(parsed) == res_smooth.assembly.num_parts
    assert sorted(sp.color_code for sp in parsed) == sorted(
        pl.color_id for pl in res_smooth.assembly.placed_parts.values()
    )


def test_p10_independent_verifier_ast_isolation_and_index_sha256_tamper() -> None:
    """
    Verify `independent_verifier.py` maintains strict AST import isolation (`zero` imports from
    `topo_brick.kernel`, `open_catalog_realizer`, or `catalog_retrieval_index`), binds
    `compiled_mold_retrieval_index.npz` into `compute_catalog_sha256()`, and independently
    verifies an Open-Catalog sculpted build while rejecting tampered `catalog_sha256`.
    """
    iv_path = Path(iv_mod.__file__).resolve()
    tree = ast.parse(iv_path.read_text(encoding="utf-8"), filename=str(iv_path))
    forbidden_prefixes = (
        "topo_brick.kernel",
        "topo_brick.model",
        "topo_brick.interface.catalog_retrieval_index",
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith(forbidden_prefixes), f"Forbidden import: {alias.name}"
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith(forbidden_prefixes), f"Forbidden from-import: {node.module}"

    kernel = TopoBrickKernel()
    tf = SparseTiledTargetField(
        boxes=(
            LDUBox(0, 0, 0, 80, 40, 24),
            LDUBox(20, 0, 24, 60, 40, 40),
        )
    )
    res = kernel.realize(
        tf,
        part_retrieval="open_catalog_top_k",
        retrieval_top_k=32,
        surface_refinement="open_catalog_sculpted",
    )
    assert res.status == KernelResultStatus.COMPLETED_CERTIFIED
    art = export_certificate_artifact(
        res.assembly,
        res.certificates,
        validator=kernel.validator,
        target_field=res.evaluated_target_field,
    )
    assert art.catalog_sha256 == compute_catalog_sha256()
    vr = verify_certificate_independently(art)
    assert vr.verified is True

    d_bad = art.to_json_dict()
    d_bad["catalog_sha256"] = "f" * 64
    vr_bad = verify_certificate_independently(VersionedCertificateArtifact.from_json_dict(d_bad))
    assert vr_bad.verified is False
    assert vr_bad.catalog_sha256_valid is False


def test_p10_gate8a_open_catalog_80_mesh_benchmark_passes_all_thresholds() -> None:
    """
    Verify `Gate 8A` across all `80` frozen unseen meshes on `Gate 2`'s untouched `5x5x6 LDU`
    `evaluate_assembly_against_original_mesh` evaluator:
    - `80 / 80` `COMPLETED_CERTIFIED`, `80 / 80` openings clear, `80 / 80` independent verifier pass;
    - `Bands 2–4` mean `5x5x6` IoU `>= 0.920` and `qualified@0.90 >= 75%`;
    - Overall `80`-mesh median `>= 0.925`, mean `>= 0.905`, `Band 1` mean `>= 0.820`,
      `qualified@0.85 >= 80%`, discretization loss reduction `>= 62%`;
    - `>= 45` distinct part types automatically retrieved & placed across `>= 5` sculptural classes;
    - `< 5.0 ms` p50 retrieval query and `< 220.0 ms` p50 single-core end-to-end latency.
    """
    report = run_gate8a_open_catalog_blind_mesh_benchmark(
        retrieval_top_k=48,
        verify_all_independently=True,
        max_p50_wall_ms=220.0,
    )
    assert report.gate8a_passed is True
    assert report.num_meshes == 80
    assert report.certified_count == 80
    assert report.openings_clear_count == 80
    assert report.independent_verifier_passed == report.independent_verifier_tested == 80
    assert report.bands_2_to_4_mean_iou_5x5x6 >= 0.920
    assert report.bands_2_to_4_qualified_at_090_rate >= 0.75
    assert report.open_catalog_median_iou_5x5x6 >= 0.925
    assert report.open_catalog_mean_iou_5x5x6 >= 0.905
    assert report.band1_mean_iou_5x5x6 >= 0.820
    assert report.overall_qualified_at_085_rate >= 0.80
    assert report.discretization_loss_reduction_vs_coarse >= 0.62
    assert report.distinct_parts_placed_count >= 45
    assert len(report.sculptural_classes_exercised) >= 5
    assert report.retrieval_query_p50_ms < 5.0
    assert report.end_to_end_p50_ms < 220.0


def test_p10_gate8b_k_ablation_commercial_license_and_chromatic_study() -> None:
    """
    Verify `Gate 8B` paired `K`-ablation (`Fixed-10`, `Fixed-27 Gate 7A`, `Top-16`, `Top-32`,
    `Top-64`), commercial `CC-BY-4.0`-only license filter (`0` `CC-BY-SA-4.0` parts across `40`
    meshes), and Option 3B chromatic (`ΔE_00 <= 8.0`, `20 / 20` `.mpd` round-trip) &
    `smooth_studless` (`>= 60%` stud reduction, `<= 0.005` IoU delta, `20 / 20` certified) study.
    """
    report = run_gate8b_retrieval_ablation_license_and_style_study()
    assert report.gate8b_passed is True
    assert report.nested_topk_subsets_verified is True
    assert report.monotonic_topk_verified is True
    assert report.lift_top64_over_gate7a >= 0.055
    assert report.lift_top64_over_fixed10 >= 0.140
    assert report.commercial_cc_by_4_certified_count == report.commercial_cc_by_4_num_meshes == 40
    assert report.commercial_cc_by_4_share_alike_parts_emitted == 0
    assert report.chromatic_mean_delta_e00 <= 8.0
    assert report.ldraw_mpd_roundtrip_exact_count == report.chromatic_style_num_meshes == 20
    assert report.smooth_studless_certified_count == 20
    assert report.exposed_upward_stud_reduction_ratio >= 0.60
    assert abs(report.smooth_studless_mean_iou_delta) <= 0.005
