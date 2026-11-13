#!/usr/bin/env python3
"""Deterministic end-to-end quickstart and independent verification smoke test for topo_brick."""

from __future__ import annotations

import time
import numpy as np

from topo_brick import (
    AdaptiveSurfaceTarget,
    KernelResultStatus,
    TopoBrickKernel,
    enforce_and_audit_single_core,
    evaluate_exact_5x5x2_mesh_iou,
    export_certificate_artifact,
    get_declarative_part_spec,
    load_frozen_80_unseen_mesh_corpus,
    verify_certificate_independently,
    voxelize_triangle_mesh_to_target_field,
)


def main() -> int:
    sc = enforce_and_audit_single_core()
    corpus, _ = load_frozen_80_unseen_mesh_corpus()
    # Select an unseen sloped portal/shrine triangle mesh with a required clear arch opening
    spec = next(m for m in corpus if len(m["required_openings_ldu"]) > 0)

    verts = spec["verts"]
    faces = spec["faces"]
    max_xy = int(spec["max_xy_studs"])
    max_z = int(spec["max_z_courses"])
    req_openings = tuple(spec["required_openings_ldu"])

    t0 = time.perf_counter()
    coarse_tf, _ = voxelize_triangle_mesh_to_target_field(
        verts, faces, max_xy_studs=max_xy, max_z_courses=max_z
    )
    adaptive_target = AdaptiveSurfaceTarget.from_triangle_mesh(
        verts, faces, max_xy_studs=max_xy, max_z_courses=max_z
    )

    # Synthesize a 2-band architectural sRGB surface palette for CIEDE2000 quantization
    v_ldu = adaptive_target.mesh_surf_pts
    z_norm = (v_ldu[:, 2] - float(v_ldu[:, 2].min())) / max(1.0, float(np.ptp(v_ldu[:, 2])))
    v_rgb = np.where(
        z_norm[:, None] < 0.5,
        np.array([[160, 165, 169]], dtype=np.uint8),  # Light Bluish Gray stonework
        np.array([[180, 0, 0]], dtype=np.uint8),      # Dark/Bright Red sloped roof
    )

    kernel = TopoBrickKernel()
    res = kernel.realize(
        coarse_tf,
        solver_id="S1_CoverAndRepair",
        part_retrieval="open_catalog_top_k",
        retrieval_top_k=32,
        license_filter="cc_by_4_0_only",
        semantic_style_hint="smooth_studless",
        surface_refinement="open_catalog_sculpted",
        adaptive_mesh_target=adaptive_target,
        support_policy="minimal_permanent_pillars",
        forbidden_support_boxes=req_openings,
        vertex_colors_rgb=v_rgb,
        verts_ldu=v_ldu,
        t_target_init_ms=adaptive_target.rasterize_ms,
    )
    wall_ms = (time.perf_counter() - t0) * 1000.0

    iou_5x5x2 = float(evaluate_exact_5x5x2_mesh_iou(adaptive_target, res.assembly))
    mpd_text = res.assembly.to_ldraw_mpd(model_name="quickstart_shrine.ldr")
    cert_artifact = export_certificate_artifact(
        res.assembly,
        res.certificates,
        validator=kernel.validator,
        target_field=coarse_tf,
        support_policy="minimal_permanent_pillars",
    )
    indep_report = verify_certificate_independently(cert_artifact)

    # Verify zero CC-BY-SA-4.0 parts emitted under license_filter="cc_by_4_0_only"
    for placed in res.assembly.placed_parts.values():
        p_spec = get_declarative_part_spec(placed.part_id)
        assert p_spec.provenance.redistribution_status != "SHARE_ALIKE_ADAPTATION", (
            f"Unexpected ShareAlike part: {placed.part_id}"
        )

    assert res.status == KernelResultStatus.COMPLETED_CERTIFIED, f"Unexpected status: {res.status}"
    assert indep_report.verified, f"Independent replay failed: {indep_report.failure_reasons}"
    assert iou_5x5x2 >= 0.80, f"Unexpectedly low sculpted mesh IoU: {iou_5x5x2:.4f}"

    distinct_molds = len({p.part_id for p in res.assembly.placed_parts.values()})
    print("=" * 72)
    print("topo_brick Quickstart & Independent Certificate Verification PASSED")
    print("=" * 72)
    print(f"Single-core threads: {sc.torch_num_threads} (strictly_single_core={sc.strictly_single_core_verified})")
    print(f"Mesh ID:             {spec['mesh_id']} ({spec['size_band']}, clear arch openings={len(req_openings)})")
    print(f"Certified parts:     {res.assembly.num_parts} bricks ({distinct_molds} distinct molds)")
    print("License filter:      cc_by_4_0_only (0 CC-BY-SA-4.0 parts emitted)")
    print(f"5x5x2 LDU Mesh IoU:  {iou_5x5x2:.4f}")
    print(f"End-to-end latency:  {wall_ms:.1f} ms")
    print(f"LDraw .mpd lines:    {len(mpd_text.splitlines())} lines")
    print(f"Independent replay:  VERIFIED (catalog_sha256={cert_artifact.catalog_sha256[:16]}...)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
