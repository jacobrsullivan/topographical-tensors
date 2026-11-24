"""
Phase 16 (`v5.4.0`) Architecture, Coordinate-Frame, Export & Diagnostic Remediation Audit Suite.

Validates the `Phase 16` (`v5.4.0+`, `docs/system-architecture-and-interfaces.md`) coordinate-frame,
policy, export, diagnostic, and packaging contracts
under both standard Python and `python -O` (`_check(...)` runtime assertions):
  1. `Decision 1` (`P0`): World-coordinate invariance across negative/shifted coordinates (`x < 0`,
     `z_ground != 0`) + explicit `UNSUPPORTED_SPEC` boundary reporting on unrepresentable frames.
  2. `Decision 2` (`P1`): Single-owner `resolve_realize_policy` option accounting (`consumed_options`,
     `ignored_nondefault_options`), compositional world `forbidden_support_boxes` + colour/symmetry
     forwarding, `< 2,100`-line `eroded_core_and_snot_stage.py`, and `SessionCommitSnapshot` invariants.
  3. `Decision 3` (`P1`): Fail-closed delivered `.mpd` round-trip verification (`verify_exported_mpd_roundtrip`)
     and 5-mode `.mpd` tamper falsification (deleted, shifted, recoloured, unknown part, colliding duplicate).
  4. `Decision 4` (`P2`): Honest `None` diagnostic sentinels when colour samples or comparable pairs
     are empty, `LDConfig.ldr` finish penalties, `ViewPrioritySpec` directional open-palette weighting,
     exterior-skin vs. internal-core bilateral symmetry separation, and runtime `actual_candidate_poses_scored`.
  5. `§5` Packaging: Lightweight `[project].dependencies` (`0` `torch`/`transformers`/`chess` in base
     install) + hard `sys.meta_path` Torch/Transformers/Chess-blocked subprocess workflow.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tomllib
from typing import Dict, Tuple
import numpy as np

import topo_brick as tb
import topo_brick.api as tb_api
from topo_brick.evaluation.independent_verifier import (
    build_relying_party_obligations,
    export_certificate_artifact,
    verify_certificate_independently,
)
from topo_brick.evaluation.route_comparison import (
    compare_realize_routes_on_target,
    verify_exported_mpd_roundtrip,
)
from topo_brick.evaluation.static_equilibrium import FoundationPolicy
from topo_brick.interface.ldu_frames import LDUBox, LDUFrame
from topo_brick.interface.port_manifold import AssemblyState
from topo_brick.kernel.design_diagnostics import (
    evaluate_bilateral_construction_symmetry,
)
from topo_brick.kernel.kernel_contracts import KernelResultStatus
from topo_brick.kernel.sculpted_primitives import assign_perceptual_ldraw_colors
from topo_brick.kernel.sculpted_stages.commit_session import SculptedLiveSession
from topo_brick.kernel.view_priority import ViewPrioritySpec

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = REPO_ROOT / "src"


def _check(cond: bool, msg: str = "Phase 16 audit assertion failed") -> None:
    if not cond:
        raise AssertionError(msg)


def _single_core_env() -> Dict[str, str]:
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


def _shifted_box_mesh(
    x0: float, y0: float, z0: float, dx: float, dy: float, dz: float
) -> Tuple[np.ndarray, np.ndarray]:
    x1, y1, z1 = x0 + dx, y0 + dy, z0 + dz
    verts = np.array(
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


def test_p16_01_world_frame_invariance_across_negative_and_shifted_coordinates() -> None:
    """Verify lossless world-coordinate preservation on negative/shifted targets and fail-closed frame bounds."""
    kernel = tb.TopoBrickKernel()

    # 1A. Exact v2-assessment probe: 2x4 brick target swept across x in (-80, -60, -40, -20, 0, +80)
    for x0 in (-80, -60, -40, -20, 0, 80):
        caller_tf = tb.TargetShapeBoxes(boxes=(LDUBox(x0, 0, 0, x0 + 80, 40, 24),))
        res = kernel.realize(
            caller_tf,
            part_retrieval="open_catalog_top_k",
            surface_refinement="open_catalog_sculpted",
        )
        _check(
            res.status == KernelResultStatus.COMPLETED_CERTIFIED and res.certificates.overall_certified,
            f"Shifted sculpted realize failed at x0={x0}: {res.status}",
        )
        rp = res.route_provenance
        _check(rp is not None, f"Missing route_provenance at x0={x0}")
        _check(rp.achieved_caller_target_iou >= 0.99, f"Caller IoU collapsed at x0={x0}: {rp.achieved_caller_target_iou}")
        _check(rp.achieved_evaluated_iou >= 0.99, f"Evaluated IoU collapsed at x0={x0}: {rp.achieved_evaluated_iou}")
        _check(rp.workbench_origin_ldu == (x0, 0, 0), f"Unexpected workbench_origin_ldu at x0={x0}: {rp.workbench_origin_ldu}")

        # Relying-party verification bound to ORIGINAL caller_tf (not just evaluated_target_field) MUST pass!
        art = export_certificate_artifact(
            res.assembly,
            res.certificates,
            validator=kernel.validator,
            target_field=caller_tf,
        )
        ver_caller = verify_certificate_independently(
            art,
            build_relying_party_obligations(expected_target_field=caller_tf, min_target_iou=0.95),
        )
        _check(ver_caller.verified is True, f"Caller-target replay failed at x0={x0}: {ver_caller.failure_reasons}")

    # 1B. `preserve_world_origin=True` on shifted mesh + `adaptive_plate_slope` + `z_ground = -24.0`
    v_sh, f_sh = _shifted_box_mesh(-60.0, 20.0, -24.0, 80.0, 40.0, 24.0)
    ctf_sh, _ = tb.voxelize_triangle_mesh_to_target_field(
        v_sh, f_sh, preserve_world_origin=True, ground_plane_z_ldu=-24.0
    )
    ad_sh = tb.AdaptiveSurfaceTarget.from_triangle_mesh(
        v_sh, f_sh, preserve_world_origin=True, ground_plane_z_ldu=-24.0
    )
    _check(ad_sh.origin_ldu == (-60, 20, -24), f"Unexpected mesh origin_ldu: {ad_sh.origin_ldu}")

    for surf_mode in ("open_catalog_sculpted", "adaptive_plate_slope"):
        res_m = kernel.realize(
            ctf_sh,
            surface_refinement=surf_mode,
            adaptive_mesh_target=ad_sh,
            foundation_policy=FoundationPolicy.TABLE_FRICTION_ONLY,
            ground_plane_z_ldu=-24.0,
        )
        _check(
            res_m.status == KernelResultStatus.COMPLETED_CERTIFIED and res_m.certificates.overall_certified,
            f"{surf_mode} failed on world-origin mesh at (-60, 20, -24): {res_m.status}",
        )
        _check(res_m.route_provenance.achieved_caller_target_iou >= 0.95)
        min_z = min(b.zmin for p in res_m.assembly.placed_parts.values() for b in p.world_boxes)
        min_x = min(b.xmin for p in res_m.assembly.placed_parts.values() for b in p.world_boxes)
        _check(min_z == -24 and min_x == -60, f"World bounds drifted on {surf_mode}: min_x={min_x}, min_z={min_z}")

    # 1C. Unrepresentable frame boundaries fail closed with `UNSUPPORTED_SPEC` (zero silent clamping!)
    below_ground_tf = tb.TargetShapeBoxes(boxes=(LDUBox(0, 0, -24, 80, 40, 0),))
    res_bg = kernel.realize(
        below_ground_tf,
        surface_refinement="open_catalog_sculpted",
        ground_plane_z_ldu=0.0,
    )
    _check(
        res_bg.status == KernelResultStatus.UNSUPPORTED_SPEC,
        f"Below-ground target must return UNSUPPORTED_SPEC, got {res_bg.status}",
    )

    odd_datum_tf = tb.TargetShapeBoxes(boxes=(LDUBox(0, 0, 2, 80, 40, 26),))
    res_odd_gz = kernel.realize(
        odd_datum_tf,
        surface_refinement="open_catalog_sculpted",
        foundation_policy=FoundationPolicy.TABLE_FRICTION_ONLY,
        ground_plane_z_ldu=1.0,  # Odd LDU ground datum violates 2-LDU parity
    )
    _check(
        res_odd_gz.status == KernelResultStatus.UNSUPPORTED_SPEC,
        f"Odd-LDU ground datum must return UNSUPPORTED_SPEC, got {res_odd_gz.status}",
    )


def test_p16_02_single_owner_policy_accounting_and_compositional_constraint_forwarding() -> None:
    """Verify `resolve_realize_policy` option accounting, compositional forwarding, and stage invariants."""
    kernel = tb.TopoBrickKernel()

    # 2A. Non-default options unsupported by `course_by_course_search` & `adaptive_plate_slope`
    # are explicitly recorded in `ignored_nondefault_options`
    box_tf = tb.TargetShapeBoxes(boxes=(LDUBox(0, 0, 0, 80, 40, 24),))
    res_c = kernel.realize(
        box_tf,
        solver_id="S1_CoverAndRepair",
        open_color_palette_mode="ldconfig_finish_aware",
        symmetry_plane_axis="x",
        semantic_style_hint="smooth_studless",
    )
    rp_c = res_c.route_provenance
    _check(rp_c is not None)
    _check(rp_c.candidate_route_id == "course_by_course_search")
    _check("open_color_palette_mode" in rp_c.ignored_nondefault_options)
    _check("symmetry_plane_axis" in rp_c.ignored_nondefault_options)
    _check("semantic_style_hint" in rp_c.ignored_nondefault_options)
    _check("solver_id" in rp_c.consumed_options)

    ad_box = tb.AdaptiveSurfaceTarget.from_target_boxes(box_tf)
    res_ap = kernel.realize(
        box_tf,
        surface_refinement="adaptive_plate_slope",
        adaptive_mesh_target=ad_box,
        view_priority=ViewPrioritySpec(view_dir_ldu=(0.0, 1.0, 0.0)),
    )
    _check(res_ap.route_provenance.candidate_route_id == "adaptive_plate_slope")
    _check("view_priority" in res_ap.route_provenance.ignored_nondefault_options)

    # 2B. Compositional `.mpd` forwards world `forbidden_support_boxes`, colour/region arrays, and symmetry audit
    v_sub, f_sub = _shifted_box_mesh(0.0, 0.0, 0.0, 40.0, 40.0, 24.0)
    ad_sub = tb.AdaptiveSurfaceTarget.from_triangle_mesh(v_sub, f_sub, max_xy_studs=2, max_z_courses=1)
    # Left pillar at x in [-60, -20], right pillar at x in [20, 60], forbidden opening in center + landmark box
    forb_world = (LDUBox(-18, 0, 0, 18, 40, 48),)
    comp_nodes = (
        tb.SubmodelNodeSpec("left_wing.ldr", ad_sub, world_offset_ldu=(-60, 0, 0)),
        tb.SubmodelNodeSpec("right_wing.ldr", ad_sub, world_offset_ldu=(20, 0, 0)),
    )
    pts_w = np.array([[-40.0, 20.0, 12.0], [40.0, 20.0, 12.0]], dtype=np.float64)
    rgb_w = np.array([[180, 0, 0], [180, 0, 0]], dtype=np.uint8)
    reg_w = ["port_wing", "starboard_wing"]
    caller_comp_tf = tb.SparseTiledTargetField(
        boxes=(LDUBox(-60, 0, 0, -20, 40, 24), LDUBox(20, 0, 0, 60, 40, 24))
    )
    res_comp = kernel.realize(
        caller_comp_tf,
        subassembly_graph=comp_nodes,
        forbidden_support_boxes=forb_world,
        verts_ldu=pts_w,
        vertex_colors_rgb=rgb_w,
        vertex_region_labels=reg_w,
        symmetry_plane_axis="x",
        symmetry_plane_coord_ldu=0.0,
    )
    _check(res_comp.status == KernelResultStatus.COMPLETED_CERTIFIED and res_comp.certificates.overall_certified)
    _check(
        all(
            not wb.intersects(fb)
            for pl in res_comp.assembly.placed_parts.values()
            for wb in pl.world_boxes
            for fb in forb_world
        ),
        "Compositional build violated world forbidden_support_boxes",
    )
    rp_comp = res_comp.route_provenance
    _check("forbidden_support_boxes" in rp_comp.consumed_options)
    _check("symmetry_plane_axis" in rp_comp.consumed_options)
    _check(len(rp_comp.ignored_nondefault_options) == 0, f"Unexpected ignored options: {rp_comp.ignored_nondefault_options}")
    _check(res_comp.open_catalog_diagnostics is not None)
    _check(res_comp.open_catalog_diagnostics.bilateral_symmetry_report is not None)
    _check(res_comp.open_catalog_diagnostics.regional_appearance_report is not None)

    # 2C. Module decomposition line caps & `SessionCommitSnapshot` tamper detection
    eroded_p = SRC_DIR / "topo_brick" / "kernel" / "sculpted_stages" / "eroded_core_and_snot_stage.py"
    snot_p = SRC_DIR / "topo_brick" / "kernel" / "sculpted_stages" / "snot_and_jumper_stage.py"
    eroded_lines = sum(1 for _ in eroded_p.open(encoding="utf-8"))
    snot_lines = sum(1 for _ in snot_p.open(encoding="utf-8"))
    _check(eroded_lines < 2100, f"eroded_core_and_snot_stage.py exceeded 2,100 lines: {eroded_lines}")
    _check(300 < snot_lines < 1600, f"Unexpected snot_and_jumper_stage.py size: {snot_lines}")

    class _DummyPrep:
        occ6 = np.ones((2, 2, 2), dtype=bool)
        max_inv_snapshot = {"3005": 2}

    dummy_sess = SculptedLiveSession(
        kernel=kernel,
        prep=_DummyPrep(),
        core_plan=None,
        coarse_target_field=box_tf,
        adaptive_target=ad_sub,
        allowed_part_ids=None,
        part_retrieval="disabled",
        retrieval_top_k=16,
        license_filter="all",
        semantic_style_hint=None,
        inventory_counts={"3005": 2},
        support_policy="none",
        foundation_policy=FoundationPolicy.CLUTCHED_BASEPLATE,
        forbidden_support_boxes=(),
        colored_inventory=None,
        hollow_shell_policy=None,
        view_priority=None,
        suppress_studless_capping_boxes=(),
        deadline_monotonic_s=None,
        ground_plane_z_ldu=0.0,
        forced_plan=None,
        eval_only=False,
        max_waves=1,
        occ6_live=np.zeros((2, 2, 2), dtype=bool),
        rem_inv={"3005": 2},
    )
    snap0 = dummy_sess.capture_invariant_snapshot()
    dummy_sess.verify_commit_transition(snap0)
    dummy_sess.inter_live = 5  # Desync tracked inter_live vs occ6_live!
    caught_drift = False
    try:
        dummy_sess.verify_commit_transition(snap0)
    except RuntimeError:
        caught_drift = True
    _check(caught_drift, "verify_commit_transition must raise RuntimeError on occ6_live drift")


def test_p16_03_delivered_mpd_roundtrip_and_tamper_falsification() -> None:
    """Verify bijective `.mpd` round-trip parity across all 3 routes and 5 tamper modes."""
    kernel = tb.TopoBrickKernel()
    v_b, f_b = _shifted_box_mesh(0.0, 0.0, 0.0, 80.0, 40.0, 48.0)
    ctf, _ = tb.voxelize_triangle_mesh_to_target_field(v_b, f_b, max_xy_studs=4, max_z_courses=2)
    ad = tb.AdaptiveSurfaceTarget.from_triangle_mesh(v_b, f_b, max_xy_studs=4, max_z_courses=2)
    v_h, f_h = _shifted_box_mesh(0.0, 0.0, 0.0, 80.0, 40.0, 24.0)
    ad_h = tb.AdaptiveSurfaceTarget.from_triangle_mesh(v_h, f_h, max_xy_studs=4, max_z_courses=1)
    comp_nodes = (
        tb.SubmodelNodeSpec("base.ldr", ad_h, world_offset_ldu=(0, 0, 0)),
        tb.SubmodelNodeSpec("roof.ldr", ad_h, world_offset_ldu=(0, 0, 24), parent_submodel="base.ldr"),
    )

    cmp_rep = compare_realize_routes_on_target(
        target_field=ctf,
        adaptive_target=ad,
        compositional_nodes=comp_nodes,
        kernel=kernel,
    )
    for r_lbl, ent in cmp_rep.entries_by_route.items():
        _check(ent.mpd_roundtrip_verified is True, f"Route {r_lbl} failed MPD round-trip: {ent.mpd_roundtrip_report}")
        _check(ent.mpd_roundtrip_report is not None and ent.mpd_roundtrip_report.dropped_instances_count == 0)
        _check(ent.mpd_roundtrip_report.exact_part_pose_color_parity is True)

    # Build a clean 2-brick reference assembly + certificate artifact, then falsify all 5 tamper modes!
    st_ref = AssemblyState()
    st_ref.place_root_part("3001", LDUFrame(0, (10, 10, 0)), is_grounded=True, color_id=4)
    st_ref.place_root_part("3001", LDUFrame(0, (10, 10, 24)), is_grounded=False, color_id=14)
    certs_ref = kernel.certify(st_ref)
    art_ref = export_certificate_artifact(st_ref, certs_ref, validator=kernel.validator)
    mpd_clean = st_ref.to_ldraw_mpd(model_name="clean.ldr")

    _, rt_ok = verify_exported_mpd_roundtrip(
        mpd_clean, expected_assembly=st_ref, certificate_artifact=art_ref
    )
    _check(rt_ok.roundtrip_verified is True and rt_ok.exact_part_pose_color_parity is True)

    mpd_lines = mpd_clean.splitlines()
    type1_indices = [i for i, ln in enumerate(mpd_lines) if ln.strip().startswith("1 ")]
    _check(len(type1_indices) == 2)

    # Mode 1: Deleted Type-1 line
    mpd_deleted = "\n".join(ln for i, ln in enumerate(mpd_lines) if i != type1_indices[1]) + "\n"
    _, rt_del = verify_exported_mpd_roundtrip(mpd_deleted, expected_assembly=st_ref, certificate_artifact=art_ref)
    _check(rt_del.roundtrip_verified is False and rt_del.exact_part_pose_color_parity is False)

    # Mode 2: Shifted Type-1 X coordinate by +20 LDU
    toks_s = mpd_lines[type1_indices[1]].split()
    toks_s[2] = f"{float(toks_s[2]) + 20.0:.6g}"
    mpd_shifted_lines = list(mpd_lines)
    mpd_shifted_lines[type1_indices[1]] = " ".join(toks_s)
    _, rt_shift = verify_exported_mpd_roundtrip(
        "\n".join(mpd_shifted_lines) + "\n", expected_assembly=st_ref, certificate_artifact=art_ref
    )
    _check(rt_shift.roundtrip_verified is False and rt_shift.exact_part_pose_color_parity is False)

    # Mode 3: Recoloured Type-1 line (color 14 -> color 1)
    toks_c = mpd_lines[type1_indices[1]].split()
    toks_c[1] = "1"
    mpd_recol_lines = list(mpd_lines)
    mpd_recol_lines[type1_indices[1]] = " ".join(toks_c)
    _, rt_col = verify_exported_mpd_roundtrip(
        "\n".join(mpd_recol_lines) + "\n", expected_assembly=st_ref, certificate_artifact=art_ref
    )
    _check(rt_col.roundtrip_verified is False and rt_col.exact_part_pose_color_parity is False)

    # Mode 4: Unknown part reference (`999999bad.dat`)
    toks_u = mpd_lines[type1_indices[1]].split()
    toks_u[14] = "999999bad.dat"
    mpd_unk_lines = list(mpd_lines)
    mpd_unk_lines[type1_indices[1]] = " ".join(toks_u)
    _, rt_unk = verify_exported_mpd_roundtrip(
        "\n".join(mpd_unk_lines) + "\n", expected_assembly=st_ref, certificate_artifact=art_ref
    )
    _check(rt_unk.roundtrip_verified is False and len(rt_unk.unknown_part_refs) == 1)

    # Mode 5: Duplicate overlapping Type-1 line (collision on replay)
    mpd_dup_lines = list(mpd_lines[:-1]) + [mpd_lines[type1_indices[0]], mpd_lines[-1]]
    _, rt_dup = verify_exported_mpd_roundtrip("\n".join(mpd_dup_lines) + "\n", expected_assembly=st_ref)
    _check(rt_dup.roundtrip_verified is False and rt_dup.collision_failures >= 1)


def test_p16_04_honest_none_diagnostics_finish_penalties_view_weighting_and_skin_symmetry() -> None:
    """Verify `None` sentinels, finish penalties, camera colour weighting, and exterior-vs-core symmetry."""
    kernel = tb.TopoBrickKernel()

    # 4A. Uncoloured sculpted build -> `RegionalAppearanceReport` returns `None` (not `0.0`)!
    box_tf = tb.TargetShapeBoxes(boxes=(LDUBox(0, 0, 0, 80, 40, 48),))
    res_nocol = kernel.realize(
        box_tf,
        part_retrieval="open_catalog_top_k",
        surface_refinement="open_catalog_sculpted",
        symmetry_plane_axis="x",
    )
    _check(res_nocol.status == KernelResultStatus.COMPLETED_CERTIFIED)
    diag_nc = res_nocol.open_catalog_diagnostics
    _check(diag_nc is not None and diag_nc.regional_appearance_report is not None)
    ra_nc = diag_nc.regional_appearance_report
    _check(ra_nc.color_samples_available is False)
    _check(ra_nc.visible_skin_mean_delta_e00 is None, f"Expected None without colour samples, got {ra_nc.visible_skin_mean_delta_e00}")
    _check(ra_nc.occluded_core_mean_delta_e00 is None, f"Expected None without colour samples, got {ra_nc.occluded_core_mean_delta_e00}")
    _check(diag_nc.catalog_funnel_report is not None and diag_nc.catalog_funnel_report.actual_candidate_poses_scored >= 0)

    # Also verify runtime `actual_candidate_poses_scored > 0` on a tapered mesh exercising skin scoring
    verts_tap = np.array(
        [
            [0.0, 0.0, 0.0],
            [80.0, 0.0, 0.0],
            [80.0, 40.0, 0.0],
            [0.0, 40.0, 0.0],
            [20.0, 0.0, 48.0],
            [60.0, 0.0, 48.0],
            [60.0, 40.0, 48.0],
            [20.0, 40.0, 48.0],
        ],
        dtype=np.float64,
    )
    faces_tap = np.array(
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
    ctf_tap, _ = tb.voxelize_triangle_mesh_to_target_field(verts_tap, faces_tap, max_xy_studs=4, max_z_courses=2)
    ad_tap = tb.AdaptiveSurfaceTarget.from_triangle_mesh(verts_tap, faces_tap, max_xy_studs=4, max_z_courses=2)
    res_tap = kernel.realize(
        ctf_tap,
        part_retrieval="open_catalog_top_k",
        surface_refinement="open_catalog_sculpted",
        adaptive_mesh_target=ad_tap,
    )
    _check(res_tap.open_catalog_diagnostics.catalog_funnel_report.actual_candidate_poses_scored > 0)

    # Empty & zero-pair assemblies -> `symmetric_color_agreement_ratio is None`!
    sym_empty = evaluate_bilateral_construction_symmetry(AssemblyState(), axis="x")
    _check(sym_empty.exterior_construction_symmetry_ratio is None)
    _check(sym_empty.symmetric_color_agreement_ratio is None)
    _check(sym_empty.comparable_pairs_count == 0)

    st_center_only = AssemblyState()
    st_center_only.place_root_part("3001", LDUFrame(0, (-10, -30, 0)), is_grounded=True, color_id=4)
    sym_center = evaluate_bilateral_construction_symmetry(st_center_only, axis="x", plane_coord_ldu=0.0)
    _check(sym_center.comparable_pairs_count == 0)
    _check(sym_center.symmetric_color_agreement_ratio is None, "Zero pairs must report None color agreement")
    _check(sym_center.exterior_construction_symmetry_ratio == 1.0)

    # 4B. `ldconfig_finish_aware` finish penalties (`15 White` SOLID beats `47 Trans_Clear` TRANSPARENT on opaque white!)
    st_col = AssemblyState()
    st_col.place_root_part("3001", LDUFrame(0, (10, 10, 0)), is_grounded=True)
    pts_white = np.array([[20.0, 20.0, 12.0], [20.0, 60.0, 12.0]], dtype=np.float64)
    rgb_white = np.array([[252, 252, 252], [252, 252, 252]], dtype=np.uint8)
    assign_perceptual_ldraw_colors(
        st_col, pts_white, rgb_white, open_color_palette_mode="ldconfig_finish_aware"
    )
    _check(
        st_col.placed_parts[0].color_id == 15,
        f"Opaque white should pick SOLID 15 White over 47 Trans_Clear, got {st_col.placed_parts[0].color_id}",
    )
    assign_perceptual_ldraw_colors(
        st_col,
        pts_white,
        rgb_white,
        open_color_palette_mode="ldconfig_finish_aware",
        preferred_finish="TRANSPARENT",
    )
    _check(
        st_col.placed_parts[0].color_id == 47,
        f"preferred_finish='TRANSPARENT' should pick 47 Trans_Clear, got {st_col.placed_parts[0].color_id}",
    )

    # Directional `ViewPrioritySpec` weighting on a two-sided brick (`x in [0, 40]`, Red on -X face, Blue on +X face)
    pts_two_sided = np.array(
        [
            [1.0, 10.0, 12.0],
            [1.0, 30.0, 12.0],
            [1.0, 50.0, 12.0],
            [39.0, 10.0, 12.0],
            [39.0, 30.0, 12.0],
            [39.0, 50.0, 12.0],
        ],
        dtype=np.float64,
    )
    rgb_two_sided = np.array(
        [
            [180, 0, 0],
            [180, 0, 0],
            [180, 0, 0],
            [30, 90, 168],
            [30, 90, 168],
            [30, 90, 168],
        ],
        dtype=np.uint8,
    )
    assign_perceptual_ldraw_colors(
        st_col,
        pts_two_sided,
        rgb_two_sided,
        view_priority=ViewPrioritySpec(view_dir_ldu=(1.0, 0.0, 0.0)),  # Camera at -X looking +X (sees Red)
    )
    col_from_neg_x = st_col.placed_parts[0].color_id
    assign_perceptual_ldraw_colors(
        st_col,
        pts_two_sided,
        rgb_two_sided,
        view_priority=ViewPrioritySpec(view_dir_ldu=(-1.0, 0.0, 0.0)),  # Camera at +X looking -X (sees Blue)
    )
    col_from_pos_x = st_col.placed_parts[0].color_id
    _check(col_from_neg_x == 4 and col_from_pos_x == 1, f"Expected (4 Red, 1 Blue), got ({col_from_neg_x}, {col_from_pos_x})")

    # 4C. Exterior visible skin vs. completely enclosed internal core filler symmetry separation!
    # Build a 4x4 symmetric outer brick ring (`x in [-40, 40], y in [-40, 40], z in [0, 24]`) capped on top
    # (`z in [24, 32]`) by two symmetric `2x4` plates (`3020`), completely enclosing the `2x2` interior core
    # (`x in [-20, 20], y in [-20, 20], z in [0, 24]`) on all 6 sides (`z=0` rests on solid baseplate!).
    # Inside the enclosed `2x2` core: left half (`x in [-20, 0]`) uses ONE `1x2` brick (`3004`),
    # right half (`x in [0, 20]`) uses TWO `1x1` bricks (`3005`)!
    st_shell = AssemblyState()
    # Left & right outer walls (`1x4` brick `3010`: `x` width 20, `y` length 80 -> at rot_id=0, hull is [0..20]x[0..80] or centered: check via LDUFrame)
    # Using `3005` (`1x1`), `3004` (`1x2`), `3010` (`1x4`), `3020` (`2x4` plate):
    # Let's place using exact `hull_boxes` offsets so coordinates are crystal clear!
    def _place_box_min(st_t: AssemblyState, pid: str, rid: int, xmin_w: int, ymin_w: int, zmin_w: int, col: int = 71) -> int:
        prim = tb.get_brick_primitive(pid)
        rbs = [b.transform(LDUFrame(rid, (0, 0, 0))) for b in prim.hull_boxes]
        pose = LDUFrame(
            rid,
            (
                xmin_w - min(b.xmin for b in rbs),
                ymin_w - min(b.ymin for b in rbs),
                zmin_w - min(b.zmin for b in rbs),
            ),
        )
        return st_t.place_root_part(pid, pose, is_grounded=(zmin_w == 0), color_id=col)

    # Outer symmetric walls (`z in [0, 24]`):
    _place_box_min(st_shell, "3010", 0, -40, -40, 0, col=71)  # Left wall: x in [-40, -20], y in [-40, 40]
    _place_box_min(st_shell, "3010", 0, 20, -40, 0, col=71)   # Right wall: x in [20, 40], y in [-40, 40]
    _place_box_min(st_shell, "3004", 9, -20, -40, 0, col=71)  # Front centerline wall: x in [-20, 20], y in [-40, -20]
    _place_box_min(st_shell, "3004", 9, -20, 20, 0, col=71)   # Back centerline wall: x in [-20, 20], y in [20, 40]
    # Asymmetric enclosed interior core (`x in [-20, 20], y in [-20, 20], z in [0, 24]`):
    _place_box_min(st_shell, "3004", 0, -20, -20, 0, col=4)   # Left interior: ONE 1x2 brick
    _place_box_min(st_shell, "3005", 0, 0, -20, 0, col=14)    # Right interior 1: 1x1 brick
    _place_box_min(st_shell, "3005", 0, 0, 0, 0, col=14)      # Right interior 2: 1x1 brick
    # Symmetric top roof plates (`z in [24, 32]`) sealing the interior core completely:
    _place_box_min(st_shell, "3020", 0, -40, -40, 24, col=71)  # Left roof 2x4 plate: x in [-40, 0], y in [-40, 40]
    _place_box_min(st_shell, "3020", 0, 0, -40, 24, col=71)    # Right roof 2x4 plate: x in [0, 40], y in [-40, 40]

    sym_shell = evaluate_bilateral_construction_symmetry(st_shell, axis="x", plane_coord_ldu=0.0)
    _check(
        sym_shell.exterior_construction_symmetry_ratio == 1.0,
        f"Exterior skin is 100% symmetric (enclosed core must be excluded), got {sym_shell}",
    )
    _check(
        sym_shell.overall_construction_symmetry_ratio is not None
        and sym_shell.overall_construction_symmetry_ratio < 1.0,
        f"Overall symmetry (including internal core filler) must be < 1.0, got {sym_shell}",
    )
    _check(sym_shell.orthographic_silhouette_mirror_iou == 1.0)
    _check(sym_shell.volumetric_mirror_iou_5x5x2 == 1.0)
    _check(sym_shell.symmetric_color_agreement_ratio == 1.0)


def test_p16_05_lightweight_core_packaging_and_torch_blocked_subprocess() -> None:
    """Verify `pyproject.toml` lightweight base dependencies, `>= 5.4.0` version parity, and Torch-blocked workflow."""
    pyproj = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    proj = pyproj["project"]
    ver_tuple = tuple(int(x) for x in str(proj["version"]).split("."))
    _check(
        ver_tuple >= (5, 4, 0) and proj["version"] == tb.__version__,
        f"Version mismatch: {proj['version']} vs {tb.__version__}",
    )

    base_deps = [d.lower() for d in proj["dependencies"]]
    for heavy_pkg in ("torch", "transformers", "chess"):
        _check(
            not any(d.startswith(heavy_pkg) for d in base_deps),
            f"Heavy optional package {heavy_pkg!r} must not be in base [project].dependencies: {base_deps}",
        )
    opt_deps = proj["optional-dependencies"]
    for extra_name in ("neural", "chess", "all", "dev", "compiler"):
        _check(extra_name in opt_deps, f"Missing optional extra [{extra_name}] in pyproject.toml")
    _check(any(d.startswith("torch>=2.10.0") for d in opt_deps["neural"]))
    _check(any(d.startswith("torch>=2.10.0") for d in opt_deps["dev"]))
    _check(35 <= len(tb_api.SUPPORTED_CONSUMER_EXPORTS) <= 50)

    # Execute full boundary benchmark CLI (including Probe 7: `sys.meta_path` Torch/Transformers/Chess blocker)
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "benchmark_topo_brick_boundaries.py")],
        cwd=str(REPO_ROOT),
        env=_single_core_env(),
        capture_output=True,
        text=True,
        timeout=120.0,
        check=True,
    )
    import json

    rep = json.loads(proc.stdout)
    _check(rep["all_boundaries_passed"] is True, f"Boundary benchmark failed:\n{proc.stdout}")
    p7 = rep["torch_blocked_lightweight_core_workflow"]
    _check(p7["certified"] is True)
    _check(p7["mpd_roundtrip_verified"] is True)
    _check(p7["independent_verified"] is True)
    _check(p7["blocked_loaded"] == [])
