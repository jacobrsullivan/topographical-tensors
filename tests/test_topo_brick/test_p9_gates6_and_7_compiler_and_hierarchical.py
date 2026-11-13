"""
Phase 9 (`Gate 6`, `Gate 7A`, `Gate 7B`) Pytest Verification Suite (`6` tests):

1. `test_p9_gate6_compiled_ldraw_catalog_size_provenance_and_coacd_sat`
2. `test_p9_gate6_omr_26set_99pct_coverage_and_500part_kinematic_audit`
3. `test_p9_gate7a_adaptive_plate_slope_skinning_80mesh_benchmark`
4. `test_p9_gate7b_hierarchical_macro_blocks_36_megastructures_subsecond`
5. `test_p9_hollow_socket_rim_mates_while_internal_wall_intrusion_rejected`
6. `test_p9_catalog_sha256_includes_compiled_archive_and_tamper_fails`
"""

from __future__ import annotations

import pytest

from topo_brick import (
    COMBINED_BRICK_CATALOG,
    COMPILED_LDRAW_CATALOG,
    EXTENDED_10PART_RECT_KIT,
    EXTENDED_BRICK_CATALOG,
    AssemblyState,
    LDUBox,
    LDUFrame,
    SparseTiledTargetField,
    TopoBrickKernel,
    VersionedCertificateArtifact,
    compute_catalog_sha256,
    export_certificate_artifact,
    get_declarative_part_spec,
    run_gate6_compiled_ldraw_shadow_catalog_audit,
    run_gate7a_adaptive_surface_skinning_benchmark,
    run_gate7b_hierarchical_macro_block_megastructure_benchmark,
    verify_certificate_independently,
)
from topo_brick.evaluation.phase9_compiler_and_hierarchical_gates import (
    _verify_coacd_concave_and_solid_sat,
)


def test_p9_gate6_compiled_ldraw_catalog_size_provenance_and_coacd_sat() -> None:
    """
    Verify compiled LDraw `2610` + LDCad Shadow + `CoACD` catalog scale (`>= 14,450` standalone,
    surpassing `BrickBench = 14,441` and `BrickNet = 14,442`), `>= 2,500` `CoACD` non-convex
    polytope parts, 3-tier provenance, `CC-BY-4.0` vs. `CC-BY-SA-4.0` license separation, and
    `CoACD` SAT concave-pocket clearance + solid-overlap rejection.
    """
    assert COMPILED_LDRAW_CATALOG.is_available
    n_compiled = len(COMPILED_LDRAW_CATALOG)
    assert n_compiled >= 14450
    assert n_compiled > 14441  # Exceeds BrickBench (14,441)
    assert n_compiled > 14442  # Exceeds BrickNet (14,442)
    assert len(COMBINED_BRICK_CATALOG) > n_compiled

    manifest = COMPILED_LDRAW_CATALOG.manifest
    assert int(manifest["coacd_decomposed_parts"]) >= 2500
    assert int(manifest["coacd_unique_shells"]) >= 200
    tiers = manifest["tier_counts"]
    assert int(tiers.get("TIER_B_LDCAD_SHADOW_DAG", 0)) >= 14000
    assert int(tiers.get("TIER_C_LDRAW_PRIMITIVE_GEOMETRIC", 0)) >= 500

    # Verify explicit license separation between TIER_B (Shadow CC-BY-SA-4.0) and TIER_C (LDraw CC-BY-4.0)
    COMPILED_LDRAW_CATALOG._ensure_loaded()
    tier_b_pid = next(
        pid
        for pid, r in COMPILED_LDRAW_CATALOG._raw_parts.items()
        if str(r["t"]).startswith("TIER_B") and pid not in EXTENDED_BRICK_CATALOG
    )
    tier_c_pid = next(
        pid
        for pid, r in COMPILED_LDRAW_CATALOG._raw_parts.items()
        if str(r["t"]).startswith("TIER_C") and pid not in EXTENDED_BRICK_CATALOG
    )
    spec_b = get_declarative_part_spec(tier_b_pid)
    spec_c = get_declarative_part_spec(tier_c_pid)
    assert "CC-BY-4.0" in spec_b.provenance.geometry_license
    assert "CC-BY-SA-4.0" in spec_b.provenance.connector_license
    assert spec_b.provenance.redistribution_status == "SHARE_ALIKE_ADAPTATION"
    assert "CC-BY-4.0" in spec_c.provenance.connector_license
    assert spec_c.provenance.redistribution_status == "COMMERCIALLY_USABLE_WITH_ATTRIBUTION"

    concave_ok, solid_ok = _verify_coacd_concave_and_solid_sat()
    assert concave_ok is True
    assert solid_ok is True


def test_p9_gate6_omr_26set_99pct_coverage_and_500part_kinematic_audit() -> None:
    """
    Verify `Gate 6` end-to-end: `500`-part stratified mating & solid-wall collision rejection
    (`100%` wall-overlap rejection) plus `26`-set human Official Model Repository (`.mpd`)
    coverage (`>= 99.0%` aggregate part coverage, `26 / 26 >= 98%`, `100%` SE(3) round-trip).
    """
    report = run_gate6_compiled_ldraw_shadow_catalog_audit(sample_n=500)
    assert report.gate6_passed is True
    assert report.wall_overlap_rejected_count == report.kinematic_sample_count == 500
    assert report.wall_overlap_rejection_rate == pytest.approx(1.0)
    assert report.exterior_mate_accept_rate >= 0.94
    assert report.omr_compiled_part_coverage >= 0.99
    assert report.omr_compiled_part_coverage > report.omr_baseline_part_coverage + 0.05
    assert report.omr_sets_ge_98pct == report.omr_total_sets == 26
    assert report.omr_se3_roundtrip_exact_rate >= 0.999
    assert len(report.omr_connector_families) >= 5


def test_p9_gate7a_adaptive_plate_slope_skinning_80mesh_benchmark() -> None:
    """
    Verify `Gate 7A` across all `80` frozen unseen meshes (`Gate 2` untouched `5x5x6 LDU`
    evaluator): `80 / 80` `COMPLETED_CERTIFIED`, `80 / 80` openings clear, `80 / 80`
    `verify_certificate_independently` pass, `mean_IoU_5x5x6 >= 0.85` (up from `0.7531`),
    `qualified@0.80 >= 80%`, `>= 18` distinct brick+plate+slope parts, and `< 250 ms` p50.
    """
    report = run_gate7a_adaptive_surface_skinning_benchmark(verify_all_independently=True)
    assert report.gate7a_passed is True
    assert report.total_meshes == 80
    assert report.adaptive_certified_count == 80
    assert report.adaptive_openings_clear_count == 80
    assert report.independent_verified_count == report.independent_tested_count == 80
    assert report.adaptive_mean_iou_5x5x6 >= 0.85
    assert report.mean_iou_5x5x6_lift >= 0.08
    assert report.adaptive_qualified_at_080_rate >= 0.80
    assert report.discretization_loss_reduction_ratio >= 0.35
    assert report.distinct_part_types_used >= 18
    assert report.p50_total_ms < 250.0


def test_p9_gate7b_hierarchical_macro_blocks_36_megastructures_subsecond() -> None:
    """
    Verify `Gate 7B` across all `36` unseen `250–1,100`-piece megastructures (`OMP=1`):
    `36 / 36` `COMPLETED_CERTIFIED`, `< 1,000 ms` p50 in `751–1,100`-piece band, `< 600 ms`
    overall p50, `>= 8x` paired speedup vs. `macro_block_mode='off'`, and `12 / 12`
    `verify_certificate_independently` pass.
    """
    report = run_gate7b_hierarchical_macro_block_megastructure_benchmark(
        verify_independently_sample=12,
        paired_ablation_count=1,
    )
    assert report.gate7b_passed is True
    assert report.total_megastructures == 36
    assert report.certified_count == 36
    assert report.overall_min_parts >= 250
    assert report.overall_max_parts >= 900
    assert report.m3_751_1100_p50_total_ms < 1000.0
    assert report.overall_p50_total_ms < 600.0
    assert report.overall_mean_target_iou >= 0.99
    assert report.paired_speedup_factor >= 8.0
    assert report.independent_verified_count == report.independent_tested_count == 12


def test_p9_hollow_socket_rim_mates_while_internal_wall_intrusion_rejected() -> None:
    """
    Verify bore-carved hollow-rim parts (`3899` Minifig Cup) legally mate on both their bottom
    anti-stud socket and side `3.18 mm` bar handle without false self-hull blockage, WHILE true
    internal-wall intrusion (`3899` rim wall collision + `99780` bracket perpendicular wall hit
    by `87087` SNOT brick, `Eval B-Extended` Case 8) remains strictly rejected.
    """
    st_cup = AssemblyState()
    st_cup.place_root_part("3899", LDUFrame(0, (0, 0, 0)), is_grounded=True)
    pool_cup, _ = st_cup.generate_candidate_pool_incremental(
        allowed_part_ids=("3024", "4085d"),
        build_conflict_graph=False,
    )
    mated_parts = {c.part_id for c in pool_cup.candidates}
    assert "3024" in mated_parts
    assert "4085d" in mated_parts

    cand_3024 = next(c for c in pool_cup.candidates if c.part_id == "3024")
    st_cup.commit_candidate(cand_3024, is_grounded=False)
    assert len(st_cup.matched_port_pairs) >= 2
    assert st_cup.verify_state_integrity() is True

    # Shifting 3024 laterally into 3899's carved rim wall MUST fail collision check
    st_cup_wall = AssemblyState()
    st_cup_wall.place_root_part("3899", LDUFrame(0, (0, 0, 0)), is_grounded=True)
    with pytest.raises(ValueError):
        st_cup_wall.place_root_part("3024", LDUFrame(0, (8, 0, -4)), is_grounded=False)

    # Eval B-Extended Case 8B: 87087 mated under 99780 at (0, 0, 24) embeds side stud into hanging wall
    st_bad = AssemblyState()
    st_bad.place_root_part("99780", LDUFrame(0, (0, 0, 24)), is_grounded=True)
    with pytest.raises(ValueError):
        st_bad.place_root_part("87087", LDUFrame(0, (0, 0, 0)), is_grounded=True)


def test_p9_catalog_sha256_includes_compiled_archive_and_tamper_fails() -> None:
    """
    Verify `compute_catalog_sha256()` binds `COMPILED_LDRAW_CATALOG.archive_sha256` into every
    exported `VersionedCertificateArtifact`, and tampering with `catalog_sha256` fails
    `verify_certificate_independently`.
    """
    assert COMPILED_LDRAW_CATALOG.archive_sha256 != "not_compiled"
    cat_sha = compute_catalog_sha256()
    assert len(cat_sha) == 64

    kernel = TopoBrickKernel()
    tf = SparseTiledTargetField(boxes=(LDUBox(0, 0, 0, 80, 40, 48),))
    res = kernel.realize(tf, allowed_part_ids=EXTENDED_10PART_RECT_KIT)
    art = export_certificate_artifact(
        res.assembly,
        res.certificates,
        validator=kernel.validator,
        target_field=tf,
    )
    assert art.catalog_sha256 == cat_sha
    rep_ok = verify_certificate_independently(art)
    assert rep_ok.verified is True
    assert rep_ok.catalog_sha256_valid is True

    d_tampered = art.to_json_dict()
    d_tampered["catalog_sha256"] = "0" * 64
    art_tampered = VersionedCertificateArtifact.from_json_dict(d_tampered)
    rep_bad = verify_certificate_independently(art_tampered)
    assert rep_bad.catalog_sha256_valid is False
    assert "catalog_sha256_mismatch" in rep_bad.failure_reasons
    assert rep_bad.verified is False
