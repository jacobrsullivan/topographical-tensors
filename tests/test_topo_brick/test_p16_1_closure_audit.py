"""
Phase 16.1 (`v5.4.1`) Assessment Closure Audit Suite.

Verifies complete closure of the `Phase 16.1` (`v5.4.1`, `docs/system-architecture-and-interfaces.md`)
MPD transform/colour, world-frame camera reprojection, option-accounting, compositional carving,
and fast-path/diagnostic contracts:
  1. `test_p16_1_01_mpd_submodel_scale_rejection_and_color16_inheritance`
  2. `test_p16_1_02_five_view_silhouette_world_translation_and_datum_invariance`
  3. `test_p16_1_03_complete_prerequisite_and_fallback_option_accounting`
  4. `test_p16_1_04_compositional_partial_and_full_forbidden_volume_carving`
  5. `test_p16_1_05_portal_fast_path_invariants_and_uncoloured_summary_none`

All checks use explicit `_check(...)` guards so the entire file remains active under `python -O`.
"""

from __future__ import annotations

import math
from pathlib import Path
import tomllib
from typing import Dict, Tuple
import numpy as np
import pytest

import topo_brick as tb
from topo_brick.evaluation.independent_verifier import CERTIFICATE_SCHEMA_VERSION
from topo_brick.evaluation.route_comparison import (
    compare_realize_routes_on_target,
    verify_exported_mpd_roundtrip,
)
from topo_brick.evaluation.static_equilibrium import FoundationPolicy
from topo_brick.interface.ldu_frames import LDUBox, LDUFrame
from topo_brick.interface.port_manifold import AssemblyState
from topo_brick.kernel.adaptive_surface_and_macro_planner import HierarchicalMacroBlockPlanner
from topo_brick.kernel.kernel_contracts import KernelResultStatus
from topo_brick.kernel.realize_policy import RealizeRequestSpec, compute_route_option_accounting
from topo_brick.kernel.sculpted_stages.commit_session import (
    verify_standalone_sculpted_state_invariants,
)
from topo_brick.kernel.view_priority import ViewPrioritySpec, compute_subvoxel_view_weights

REPO_ROOT = Path(__file__).resolve().parents[2]


def _check(cond: bool, msg: str = "Phase 16.1 closure check failed") -> None:
    if not cond:
        raise AssertionError(msg)


def _shifted_box_mesh(
    xmin: float,
    ymin: float,
    zmin: float,
    xmax: float,
    ymax: float,
    zmax: float,
) -> Tuple[np.ndarray, np.ndarray]:
    verts = np.array(
        [
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmax, ymax, zmin],
            [xmin, ymax, zmin],
            [xmin, ymin, zmax],
            [xmax, ymin, zmax],
            [xmax, ymax, zmax],
            [xmin, ymax, zmax],
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
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int32,
    )
    return verts, faces


def test_p16_1_01_mpd_submodel_scale_rejection_and_color16_inheritance() -> None:
    """Closure test for Finding 1: reject non-rigid MPD transforms and resolve inherited colour 16."""
    kernel = tb.TopoBrickKernel()
    ref_state = AssemblyState()
    ref_state.place_root_part("3005", LDUFrame(0, (0, 0, 0)), is_grounded=True, color_id=4)
    certs = kernel.certify(ref_state)
    cert_art = tb.export_certificate_artifact(ref_state, certs, validator=kernel.validator)

    exported_ldr = ref_state.to_ldraw_mpd(model_name="control.ldr")
    leaf_lines = [ln.strip() for ln in exported_ldr.splitlines() if ln.strip().startswith("1 ")]
    _check(len(leaf_lines) == 1, f"Expected 1 exported Type-1 line, got {leaf_lines}")
    leaf_tokens = leaf_lines[0].split()
    leaf_red_line = " ".join(leaf_tokens)
    leaf_inherit_line = " ".join(["1", "16"] + leaf_tokens[2:])
    leaf_edge24_line = " ".join(["1", "24"] + leaf_tokens[2:])
    leaf_neg_line = " ".join(["1", "-1"] + leaf_tokens[2:])

    # 1A. Identity control with explicit red leaf (`color=4`)
    mpd_control = (
        "0 FILE main.ldr\n"
        "1 16 0 0 0 1 0 0 0 1 0 0 0 1 child.ldr\n"
        "0 NOFILE\n"
        "0 FILE child.ldr\n"
        f"{leaf_red_line}\n"
        "0 NOFILE\n"
    )
    for ref_kwargs in ({"expected_assembly": ref_state}, {"certificate_artifact": cert_art}):
        _, rep_ctrl = verify_exported_mpd_roundtrip(mpd_control, **ref_kwargs)
        _check(rep_ctrl.roundtrip_verified is True, f"Control failed: {rep_ctrl.failure_reasons}")
        _check(rep_ctrl.exact_part_pose_color_parity is True)
        _check(rep_ctrl.non_rigid_transform_failures == 0)
        _check(rep_ctrl.unresolved_color_failures == 0)

    # 1B. Non-rigid parent matrices: 2x uniform scale, non-uniform scale, shear, det=-1 reflection
    non_rigid_parent_matrices = (
        ("uniform_2x_scale", "2 0 0 0 2 0 0 0 2"),
        ("nonuniform_scale", "1 0 0 0 2 0 0 0 1"),
        ("shear_xy", "1 0.5 0 0 1 0 0 0 1"),
        ("improper_reflection", "-1 0 0 0 1 0 0 0 1"),
    )
    for label, mat_str in non_rigid_parent_matrices:
        mpd_bad = (
            "0 FILE main.ldr\n"
            f"1 16 0 0 0 {mat_str} child.ldr\n"
            "0 NOFILE\n"
            "0 FILE child.ldr\n"
            f"{leaf_red_line}\n"
            "0 NOFILE\n"
        )
        for ref_kwargs in ({"expected_assembly": ref_state}, {"certificate_artifact": cert_art}):
            _, rep_bad = verify_exported_mpd_roundtrip(mpd_bad, **ref_kwargs)
            _check(rep_bad.roundtrip_verified is False, f"{label} was not rejected!")
            _check(rep_bad.exact_part_pose_color_parity is False)
            _check(
                rep_bad.non_rigid_transform_failures >= 1,
                f"Expected non_rigid_transform_failures >= 1 for {label}, got {rep_bad}",
            )

    # Even an empty scaled child submodel fails closed with non_rigid_transform_failures >= 1
    mpd_empty_scaled = (
        "0 FILE main.ldr\n"
        f"{leaf_red_line}\n"
        "1 16 0 0 0 2 0 0 0 2 0 0 0 2 empty_sub.ldr\n"
        "0 NOFILE\n"
        "0 FILE empty_sub.ldr\n"
        "0 STEP\n"
        "0 NOFILE\n"
    )
    _, rep_es = verify_exported_mpd_roundtrip(mpd_empty_scaled, expected_assembly=ref_state)
    _check(rep_es.roundtrip_verified is False and rep_es.non_rigid_transform_failures >= 1)

    # Duplicate `0 FILE` block shadowing fails closed
    mpd_dup_block = (
        "0 FILE main.ldr\n"
        "1 4 0 0 0 1 0 0 0 1 0 0 0 1 child.ldr\n"
        "0 NOFILE\n"
        "0 FILE child.ldr\n"
        "0 STEP\n"
        "0 NOFILE\n"
        "0 FILE child.ldr\n"
        f"{leaf_inherit_line}\n"
        "0 NOFILE\n"
    )
    _, rep_dup = verify_exported_mpd_roundtrip(mpd_dup_block, expected_assembly=ref_state)
    _check(
        rep_dup.roundtrip_verified is False
        and any("duplicate_mpd_file_block" in r for r in rep_dup.failure_reasons)
    )

    # 1C. Exporter rounding tolerance: slightly rounded proper rotation (~1e-4 rad) + inherited colour
    eps_ang = 1e-4
    ce, se = math.cos(eps_ang), math.sin(eps_ang)
    mpd_rounded = (
        "0 FILE main.ldr\n"
        f"1 4 0 0 0 {ce:.6f} 0 {se:.6f} 0 1 0 {-se:.6f} 0 {ce:.6f} child.ldr\n"
        "0 NOFILE\n"
        "0 FILE child.ldr\n"
        f"{leaf_inherit_line}\n"
        "0 NOFILE\n"
    )
    _, rep_round = verify_exported_mpd_roundtrip(
        mpd_rounded, expected_assembly=ref_state, certificate_artifact=cert_art
    )
    _check(
        rep_round.roundtrip_verified is True,
        f"Exporter rounding tolerance rejected valid pose: {rep_round.failure_reasons}",
    )

    # 1D. Inherited colour 16: 1-level (`parent=4 -> leaf=16`) and 2-level (`root=4 -> mid=16 -> leaf=16`)
    mpd_inherit_1 = (
        "0 FILE main.ldr\n"
        "1 4 0 0 0 1 0 0 0 1 0 0 0 1 child.ldr\n"
        "0 NOFILE\n"
        "0 FILE child.ldr\n"
        f"{leaf_inherit_line}\n"
        "0 NOFILE\n"
    )
    mpd_inherit_2 = (
        "0 FILE main.ldr\n"
        "1 4 0 0 0 1 0 0 0 1 0 0 0 1 mid.ldr\n"
        "0 NOFILE\n"
        "0 FILE mid.ldr\n"
        "1 16 0 0 0 1 0 0 0 1 0 0 0 1 leaf.ldr\n"
        "0 NOFILE\n"
        "0 FILE leaf.ldr\n"
        f"{leaf_inherit_line}\n"
        "0 NOFILE\n"
    )
    for mpd_inh in (mpd_inherit_1, mpd_inherit_2):
        for ref_kwargs in ({"expected_assembly": ref_state}, {"certificate_artifact": cert_art}):
            st_inh, rep_inh = verify_exported_mpd_roundtrip(mpd_inh, **ref_kwargs)
            _check(rep_inh.roundtrip_verified is True, f"Inherited colour 16 failed: {rep_inh.failure_reasons}")
            _check(rep_inh.exact_part_pose_color_parity is True)
            _check(rep_inh.unresolved_color_failures == 0)
            _check(next(iter(st_inh.placed_parts.values())).color_id == 4)

    # 1E. Unresolved root colour 16 (`parent=16 -> leaf=16`), edge colour 24, negative colour, & cycle fail closed
    for bad_leaf in (leaf_inherit_line, leaf_edge24_line, leaf_neg_line):
        mpd_ucol = (
            "0 FILE main.ldr\n"
            "1 16 0 0 0 1 0 0 0 1 0 0 0 1 child.ldr\n"
            "0 NOFILE\n"
            "0 FILE child.ldr\n"
            f"{bad_leaf}\n"
            "0 NOFILE\n"
        )
        _, rep_uc = verify_exported_mpd_roundtrip(mpd_ucol, expected_assembly=ref_state)
        _check(rep_uc.roundtrip_verified is False and rep_uc.unresolved_color_failures >= 1)

    mpd_cycle = (
        "0 FILE a.ldr\n"
        "1 4 0 0 0 1 0 0 0 1 0 0 0 1 b.ldr\n"
        "0 NOFILE\n"
        "0 FILE b.ldr\n"
        "1 16 0 0 0 1 0 0 0 1 0 0 0 1 a.ldr\n"
        f"{leaf_inherit_line}\n"
        "0 NOFILE\n"
    )
    _, rep_cyc = verify_exported_mpd_roundtrip(mpd_cycle, expected_assembly=ref_state)
    _check(
        rep_cyc.roundtrip_verified is False
        and any("submodel_cycle_detected" in r for r in rep_cyc.failure_reasons)
    )


def test_p16_1_02_five_view_silhouette_world_translation_and_datum_invariance() -> None:
    """Closure test for Finding 2: 5-view silhouette & colour scores match across world translations & datums."""
    kernel = tb.TopoBrickKernel()

    v_loc, f_loc = _shifted_box_mesh(0.0, 0.0, 0.0, 80.0, 40.0, 24.0)
    ad_loc = tb.AdaptiveSurfaceTarget.from_triangle_mesh(v_loc, f_loc, max_xy_studs=4, max_z_courses=1)

    cases = (
        ((0, 0, 0), 0.0, FoundationPolicy.CLUTCHED_BASEPLATE),
        ((-80, -40, 0), 0.0, FoundationPolicy.CLUTCHED_BASEPLATE),
        ((80, 40, 0), 0.0, FoundationPolicy.CLUTCHED_BASEPLATE),
        ((-60, 20, -24), -24.0, FoundationPolicy.TABLE_FRICTION_ONLY),
        ((40, -60, -8), -8.0, FoundationPolicy.TABLE_FRICTION_ONLY),  # 1-plate (-8 LDU, non-multiple of 6!) datum
    )

    baseline_views: Dict[str, Dict[str, float]] = {}
    baseline_col_de00: Dict[str, Dict[str, float]] = {}
    for (ox, oy, oz), gz, fp in cases:
        box = LDUBox(ox, oy, oz, ox + 80, oy + 40, oz + 24)
        tf = tb.SparseTiledTargetField(boxes=(box,))
        ad_world = tb.AdaptiveSurfaceTarget.from_target_boxes((box,), ground_plane_z_ldu=gz)
        comp_graph = (tb.SubmodelNodeSpec("body.ldr", ad_loc, world_offset_ldu=(ox, oy, oz)),)
        verts_w = np.array(
            [
                [ox + 20.0, oy + 20.0, oz + 12.0],
                [ox + 60.0, oy + 20.0, oz + 12.0],
            ],
            dtype=np.float64,
        )
        vcols_rgb = np.array([[180, 0, 0], [180, 0, 0]], dtype=np.uint8)

        cmp_rep = compare_realize_routes_on_target(
            target_field=tf,
            adaptive_target=ad_world,
            compositional_nodes=comp_graph,
            kernel=kernel,
            foundation_policy=fp,
            ground_plane_z_ldu=gz,
            verts_ldu=verts_w,
            vertex_colors_rgb=vcols_rgb,
        )
        _check(len(cmp_rep.entries_by_route) == 3)
        for route_name, entry in cmp_rep.entries_by_route.items():
            _check(
                entry.status == KernelResultStatus.COMPLETED_CERTIFIED.value,
                f"Route {route_name} at {(ox, oy, oz)} failed: {entry.status}",
            )
            _check(entry.mpd_roundtrip_verified and entry.independent_verified)
            _check(abs(entry.caller_target_iou - 1.0) <= 1e-6, f"caller_target_iou={entry.caller_target_iou}")
            _check(abs(entry.mesh_5x5x2_iou - 1.0) <= 1e-6, f"mesh_5x5x2_iou={entry.mesh_5x5x2_iou}")
            _check(
                abs(entry.mean_five_view_silhouette_iou - 1.0) <= 1e-6,
                f"Route {route_name} at {(ox, oy, oz)} mean_five_view_silhouette_iou={entry.mean_five_view_silhouette_iou}",
            )
            _check(len(entry.exported_five_view_silhouette_iou) == 5)
            for v_name, v_iou in entry.exported_five_view_silhouette_iou.items():
                _check(
                    abs(v_iou - 1.0) <= 1e-6,
                    f"Route {route_name} at {(ox, oy, oz)} view {v_name} IoU={v_iou}",
                )
            if route_name not in baseline_views:
                baseline_views[route_name] = dict(entry.exported_five_view_silhouette_iou)
            else:
                for v_name, ref_val in baseline_views[route_name].items():
                    _check(abs(entry.exported_five_view_silhouette_iou[v_name] - ref_val) <= 1e-6)

            if route_name in ("open_catalog_sculpted", "compositional_mpd"):
                if route_name not in baseline_col_de00:
                    baseline_col_de00[route_name] = {
                        k: float(v) for k, v in entry.exported_five_view_color_delta_e00.items() if v is not None
                    }
                else:
                    for v_name, ref_de in baseline_col_de00[route_name].items():
                        cur_de = entry.exported_five_view_color_delta_e00[v_name]
                        _check(
                            cur_de is not None and abs(float(cur_de) - ref_de) <= 1e-6,
                            f"Colour reprojection ΔE00 drifted for {route_name}/{v_name} at {(ox, oy, oz)}: {cur_de} vs {ref_de}",
                        )

    # Perspective camera_pos_ldu translation invariance on compute_subvoxel_view_weights
    occ_ref = np.ones((8, 8, 4), dtype=bool)
    vp_origin = ViewPrioritySpec(view_dir_ldu=(0.0, 1.0, -0.35), camera_pos_ldu=(20.0, -80.0, 40.0))
    vp_shifted = ViewPrioritySpec(view_dir_ldu=(0.0, 1.0, -0.35), camera_pos_ldu=(-60.0, -120.0, 16.0))
    w_orig = compute_subvoxel_view_weights(occ_ref, vp_origin, origin_ldu=(0, 0, 0))
    w_shift = compute_subvoxel_view_weights(occ_ref, vp_shifted, origin_ldu=(-80, -40, -24))
    _check(np.allclose(w_orig, w_shift, atol=1e-9), "Perspective view weights drifted under world translation!")


def test_p16_1_03_complete_prerequisite_and_fallback_option_accounting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Closure test for Finding 3: every nondefault option accounted for across prerequisites & fallbacks."""
    kernel = tb.TopoBrickKernel()
    box_tf = tb.TargetShapeBoxes(boxes=(LDUBox(0, 0, 0, 80, 40, 24),))

    # 3A. Exact Finding 3 reproducer on BOTH course_by_course_search AND open_catalog_sculpted (without allowed_part_ids!)
    for surf_mode, exp_route in (
        ("none", "course_by_course_search"),
        ("open_catalog_sculpted", "open_catalog_sculpted"),
    ):
        res_f3 = kernel.realize(
            box_tf,
            surface_refinement=surf_mode,
            retrieval_top_k=3,
            symmetry_plane_coord_ldu=20.0,
        )
        rp_f3 = res_f3.route_provenance
        _check(rp_f3 is not None and rp_f3.effective_route_id == exp_route)
        _check("retrieval_top_k" in rp_f3.ignored_nondefault_options, f"{exp_route}: {rp_f3.ignored_nondefault_options}")
        _check("symmetry_plane_coord_ldu" in rp_f3.ignored_nondefault_options)
        _check("retrieval_top_k" not in rp_f3.consumed_options)
        _check("symmetry_plane_coord_ldu" not in rp_f3.consumed_options)
        _check(set(rp_f3.consumed_options).isdisjoint(rp_f3.ignored_nondefault_options))

    # 3B. open_color_palette_mode ignored when colored_inventory is active, and kit_name accounting
    pts_only = np.array([[20.0, 20.0, 12.0]], dtype=np.float64)
    rgb_match = np.array([[180, 0, 0]], dtype=np.uint8)
    res_cinv = kernel.realize(
        box_tf,
        surface_refinement="open_catalog_sculpted",
        colored_inventory=tb.SET_10698_COLORED_INVENTORY,
        open_color_palette_mode="ldconfig_finish_aware",
        kit_name="EXTENDED_10PART_RECT_KIT",
        verts_ldu=pts_only,
        vertex_colors_rgb=rgb_match,
    )
    rp_cinv = res_cinv.route_provenance
    _check("colored_inventory" in rp_cinv.consumed_options)
    _check("open_color_palette_mode" in rp_cinv.ignored_nondefault_options)
    _check("kit_name" in rp_cinv.ignored_nondefault_options)

    # Named rectangular kit consumed on course_by_course_search when not overridden, ignored when overridden
    res_kit_ok = kernel.realize(box_tf, kit_name="EXTENDED_10PART_RECT_KIT")
    _check("kit_name" in res_kit_ok.route_provenance.consumed_options)
    _check("kit_name" not in res_kit_ok.route_provenance.ignored_nondefault_options)

    res_kit_ov = kernel.realize(box_tf, kit_name="EXTENDED_10PART_RECT_KIT", allowed_part_ids=("3001", "3003"))
    _check("kit_name" in res_kit_ov.route_provenance.ignored_nondefault_options)

    # 3C. Satisfied prerequisites on open_catalog_sculpted -> consumed, zero ignored
    res_oc_ok = kernel.realize(
        box_tf,
        surface_refinement="open_catalog_sculpted",
        part_retrieval="open_catalog_top_k",
        retrieval_top_k=12,
        symmetry_plane_axis="x",
        symmetry_plane_coord_ldu=40.0,
        open_color_palette_mode="ldconfig_finish_aware",
        verts_ldu=pts_only,
        vertex_colors_rgb=rgb_match,
    )
    rp_oc_ok = res_oc_ok.route_provenance
    _check(len(rp_oc_ok.ignored_nondefault_options) == 0, f"Unexpected ignored: {rp_oc_ok.ignored_nondefault_options}")
    for active_opt in (
        "surface_refinement",
        "part_retrieval",
        "retrieval_top_k",
        "symmetry_plane_axis",
        "symmetry_plane_coord_ldu",
        "open_color_palette_mode",
        "verts_ldu",
        "vertex_colors_rgb",
    ):
        _check(active_opt in rp_oc_ok.consumed_options, f"Missing {active_opt} from consumed_options")

    # 3D. Direct dispatch vs. fallback on hierarchical_macro_block -> course_by_course_search,
    # plus adaptive_plate_slope / open_catalog_sculpted -> course_by_course_search accounting transition
    ad_box = tb.AdaptiveSurfaceTarget.from_target_boxes(box_tf)
    req_sculpt_fb = RealizeRequestSpec(
        target_field=box_tf,
        surface_refinement="adaptive_plate_slope",
        adaptive_mesh_target=ad_box,
        view_priority=ViewPrioritySpec(view_dir_ldu=(0.0, 1.0, 0.0)),
    )
    c_oc, ig_oc = compute_route_option_accounting(req_sculpt_fb, "open_catalog_sculpted")
    _check("adaptive_mesh_target" in c_oc and "view_priority" in c_oc)
    c_ap, ig_ap = compute_route_option_accounting(req_sculpt_fb, "adaptive_plate_slope")
    _check("adaptive_mesh_target" in c_ap and "view_priority" in ig_ap)
    c_fb, ig_fb = compute_route_option_accounting(req_sculpt_fb, "course_by_course_search")
    _check("adaptive_mesh_target" in ig_fb and "view_priority" in ig_fb and "surface_refinement" in ig_fb)
    _check(set(c_fb).isdisjoint(ig_fb))

    res_macro_direct = kernel.realize(
        box_tf,
        macro_block_mode="force",
        solver_id="S2_PairSeededLocalSearch",
        port_filter_keep_ratio=0.75,
    )
    rp_md = res_macro_direct.route_provenance
    _check(rp_md.candidate_route_id == "hierarchical_macro_block")
    _check(rp_md.effective_route_id == "hierarchical_macro_block")
    _check(rp_md.fallback_from_route_id is None)
    _check("macro_block_mode" in rp_md.consumed_options)
    _check("solver_id" in rp_md.ignored_nondefault_options)
    _check("port_filter_keep_ratio" in rp_md.ignored_nondefault_options)

    monkeypatch.setattr(HierarchicalMacroBlockPlanner, "should_activate", staticmethod(lambda _tf, _mode: True))
    monkeypatch.setattr(
        HierarchicalMacroBlockPlanner,
        "realize_macro_blocks",
        staticmethod(lambda **_kw: (None, 0, 0)),
    )
    res_macro_fb = kernel.realize(
        box_tf,
        macro_block_mode="auto",
        solver_id="S2_PairSeededLocalSearch",
        port_filter_keep_ratio=0.75,
    )
    rp_fb = res_macro_fb.route_provenance
    _check(rp_fb.candidate_route_id == "hierarchical_macro_block")
    _check(rp_fb.effective_route_id == "course_by_course_search")
    _check(rp_fb.fallback_from_route_id == "hierarchical_macro_block")
    _check("solver_id" in rp_fb.consumed_options and "solver_id" not in rp_fb.ignored_nondefault_options)
    _check(
        "port_filter_keep_ratio" in rp_fb.consumed_options
        and "port_filter_keep_ratio" not in rp_fb.ignored_nondefault_options
    )
    _check("macro_block_mode" in rp_fb.ignored_nondefault_options and "macro_block_mode" not in rp_fb.consumed_options)
    _check(set(rp_fb.consumed_options).isdisjoint(rp_fb.ignored_nondefault_options))


def test_p16_1_04_compositional_partial_and_full_forbidden_volume_carving() -> None:
    """Closure test for Finding 4: partial/full world exclusions on translated compositional node without phantom box."""
    kernel = tb.TopoBrickKernel()
    v_sub, f_sub = _shifted_box_mesh(0.0, 0.0, 0.0, 40.0, 40.0, 24.0)
    ad_sub = tb.AdaptiveSurfaceTarget.from_triangle_mesh(v_sub, f_sub, max_xy_studs=2, max_z_courses=1)

    # Translated single node at (80, 40, 0) -> world box [80..120, 40..80, 0..24]
    node_trans = (tb.SubmodelNodeSpec("pod.ldr", ad_sub, world_offset_ldu=(80, 40, 0)),)
    caller_tf = tb.SparseTiledTargetField(boxes=(LDUBox(80, 40, 0, 120, 80, 24),))

    # 4A. Full intersection: forbidden volume covers entire translated node -> PROVEN_INFEASIBLE + empty evaluated target
    res_full = kernel.realize(
        caller_tf,
        subassembly_graph=node_trans,
        forbidden_support_boxes=(LDUBox(80, 40, 0, 120, 80, 24),),
    )
    _check(res_full.assembly.num_parts == 0)
    _check(res_full.status == KernelResultStatus.PROVEN_INFEASIBLE, f"Got status {res_full.status}")
    _check(res_full.stopping_reason == "empty_compositional_target_after_forbidden_exclusions")
    _check(res_full.evaluated_target_field is not None)
    _check(res_full.evaluated_target_field.boxes == (), f"Phantom boxes leaked: {res_full.evaluated_target_field.boxes}")
    _check(res_full.evaluated_target_field.total_volume_ldu3 == 0)
    _check(res_full.route_provenance.achieved_evaluated_iou == 0.0)

    # 4B. Partial Y-half intersection: forbidden volume [80..120, 40..60, 0..24] carves to exact 40x20x24 at (80, 60, 0)
    forb_y_half = (LDUBox(80, 40, 0, 120, 60, 24),)
    res_part_y = kernel.realize(
        caller_tf,
        subassembly_graph=node_trans,
        forbidden_support_boxes=forb_y_half,
    )
    _check(res_part_y.status == KernelResultStatus.COMPLETED_CERTIFIED and res_part_y.certificates.overall_certified)
    _check(res_part_y.assembly.num_parts > 0)
    for pl in res_part_y.assembly.placed_parts.values():
        for wb in pl.world_boxes:
            _check(not wb.intersects(forb_y_half[0]), f"Placed box {wb} intersected forbidden Y-half")
            _check(wb.xmin >= 80 and wb.xmax <= 120 and wb.ymin >= 60 and wb.ymax <= 80)
    _check(res_part_y.evaluated_target_field.total_volume_ldu3 == 40 * 20 * 24)
    _check(res_part_y.evaluated_target_field.occupied_bounds_ldu() == (80, 60, 0, 120, 80, 24))

    # Partial X-half intersection: forbidden volume [80..100, 40..80, 0..24] carves to exact 20x40x24 at (100, 40, 0)
    forb_x_half = (LDUBox(80, 40, 0, 100, 80, 24),)
    res_part_x = kernel.realize(
        caller_tf,
        subassembly_graph=node_trans,
        forbidden_support_boxes=forb_x_half,
    )
    _check(res_part_x.status == KernelResultStatus.COMPLETED_CERTIFIED and res_part_x.certificates.overall_certified)
    _check(res_part_x.evaluated_target_field.total_volume_ldu3 == 20 * 40 * 24)
    _check(res_part_x.evaluated_target_field.occupied_bounds_ldu() == (100, 40, 0, 120, 80, 24))

    # 4C. Caller-supplied empty submodel target without exclusions -> UNSUPPORTED_SPEC + empty target
    empty_ad = tb.AdaptiveSurfaceTarget(
        nx_studs=2,
        ny_studs=2,
        nz_courses=1,
        occ_5x5x2=np.zeros((8, 8, 12), dtype=bool),
        occ_5x5x6=np.zeros((8, 8, 4), dtype=bool),
        mesh_surf_pts=np.zeros((0, 3), dtype=np.float64),
        rasterize_ms=0.0,
    )
    res_empty_in = kernel.realize(
        caller_tf,
        subassembly_graph=(tb.SubmodelNodeSpec("empty.ldr", empty_ad, world_offset_ldu=(80, 40, 0)),),
    )
    _check(res_empty_in.status == KernelResultStatus.UNSUPPORTED_SPEC)
    _check(res_empty_in.stopping_reason == "empty_compositional_submodel_target")
    _check(res_empty_in.evaluated_target_field.boxes == ())
    _check(res_empty_in.evaluated_target_field.total_volume_ldu3 == 0)


def test_p16_1_05_portal_fast_path_invariants_and_uncoloured_summary_none() -> None:
    """Closure test for Residual Table Items: fast-path invariants, None colour summary, and v5.4.1 metadata."""
    kernel = tb.TopoBrickKernel()

    # 5A. End-to-end portal bridge fast-path build + standalone invariant tamper checks
    portal_boxes = (
        LDUBox(0, 0, 0, 20, 40, 24),
        LDUBox(60, 0, 0, 80, 40, 24),
        LDUBox(0, 0, 24, 80, 40, 48),
    )
    portal_forb = (LDUBox(20, 0, 0, 60, 40, 24),)
    portal_tf = tb.SparseTiledTargetField(boxes=portal_boxes)
    res_portal = kernel.realize(
        portal_tf,
        surface_refinement="open_catalog_sculpted",
        part_retrieval="open_catalog_top_k",
        forbidden_support_boxes=portal_forb,
    )
    _check(res_portal.status == KernelResultStatus.COMPLETED_CERTIFIED)
    _check(res_portal.open_catalog_diagnostics is not None)
    _check(res_portal.open_catalog_diagnostics.mean_color_delta_e00 is None)
    _check(res_portal.open_catalog_diagnostics.visible_skin_mean_delta_e00 is None)
    _check(res_portal.open_catalog_diagnostics.color_samples_available is False)

    class _PrepStub:
        max_inv_snapshot = {"3005": 1}
        occ6 = np.ones((4, 4, 4), dtype=bool)  # [0..20, 0..20, 0..24]

    st_ok = AssemblyState()
    st_ok.place_root_part("3005", LDUFrame(0, (10, 10, 0)), is_grounded=True)
    verify_standalone_sculpted_state_invariants(st_ok, _PrepStub(), ground_plane_z_ldu=0.0)

    # Out-of-workbench bounds caught
    st_oob = AssemblyState()
    st_oob.place_root_part("3005", LDUFrame(0, (50, 10, 0)), is_grounded=True)
    caught_oob = False
    try:
        verify_standalone_sculpted_state_invariants(st_oob, _PrepStub(), ground_plane_z_ldu=0.0)
    except RuntimeError:
        caught_oob = True
    _check(caught_oob, "Fast-path invariant check missed out-of-bounds hull box")

    # Forbidden-box intrusion caught
    caught_forb = False
    try:
        verify_standalone_sculpted_state_invariants(
            st_ok, _PrepStub(), forbidden_support_boxes=(LDUBox(0, 0, 0, 20, 20, 24),)
        )
    except RuntimeError:
        caught_forb = True
    _check(caught_forb, "Fast-path invariant check missed forbidden_support_boxes overlap")

    # Below-ground datum caught
    caught_datum = False
    try:
        verify_standalone_sculpted_state_invariants(st_ok, _PrepStub(), ground_plane_z_ldu=24.0)
    except RuntimeError:
        caught_datum = True
    _check(caught_datum, "Fast-path invariant check missed below-ground hull box")

    # Stock overrun caught
    class _WidePrepStub:
        max_inv_snapshot = {"3005": 1}
        occ6 = np.ones((12, 4, 4), dtype=bool)

    st_over = AssemblyState()
    st_over.place_root_part("3005", LDUFrame(0, (10, 10, 0)), is_grounded=True)
    st_over.place_root_part("3005", LDUFrame(0, (30, 10, 0)), is_grounded=True)
    caught_stock = False
    try:
        verify_standalone_sculpted_state_invariants(st_over, _WidePrepStub(), ground_plane_z_ldu=0.0)
    except RuntimeError:
        caught_stock = True
    _check(caught_stock, "Fast-path invariant check missed finite stock overrun")

    # Trace desync caught
    st_ok._recorded_step_ids.append(999)
    caught_trace = False
    try:
        verify_standalone_sculpted_state_invariants(st_ok, _PrepStub(), ground_plane_z_ldu=0.0)
    except RuntimeError:
        caught_trace = True
    _check(caught_trace, "Fast-path invariant check missed recorded_step_ids desync")

    # 5B. Missing-colour summary returns None (`color_samples_available=False`, `color_sample_count=0`)
    # on both open_catalog_sculpted and compositional_mpd (plain & symmetry-enabled)
    box_tf = tb.SparseTiledTargetField(boxes=(LDUBox(0, 0, 0, 40, 40, 24),))
    res_oc_nocol = kernel.realize(box_tf, surface_refinement="open_catalog_sculpted")
    diag_oc_nc = res_oc_nocol.open_catalog_diagnostics
    _check(diag_oc_nc is not None)
    _check(diag_oc_nc.mean_color_delta_e00 is None, f"Expected None, got {diag_oc_nc.mean_color_delta_e00}")
    _check(diag_oc_nc.visible_skin_mean_delta_e00 is None)
    _check(diag_oc_nc.color_samples_available is False and diag_oc_nc.color_sample_count == 0)

    v_sub, f_sub = _shifted_box_mesh(0.0, 0.0, 0.0, 40.0, 40.0, 24.0)
    ad_sub = tb.AdaptiveSurfaceTarget.from_triangle_mesh(v_sub, f_sub, max_xy_studs=2, max_z_courses=1)
    comp_nodes = (tb.SubmodelNodeSpec("unit.ldr", ad_sub, world_offset_ldu=(0, 0, 0)),)
    for sym_ax in (None, "x"):
        res_comp_nc = kernel.realize(box_tf, subassembly_graph=comp_nodes, symmetry_plane_axis=sym_ax)
        diag_c_nc = res_comp_nc.open_catalog_diagnostics
        _check(diag_c_nc is not None)
        _check(diag_c_nc.mean_color_delta_e00 is None, f"Expected None, got {diag_c_nc.mean_color_delta_e00}")
        _check(diag_c_nc.visible_skin_mean_delta_e00 is None)
        _check(diag_c_nc.color_samples_available is False and diag_c_nc.color_sample_count == 0)

    # Coloured contrast: numeric float & color_samples_available=True on both routes
    pts_c = np.array([[10.0, 10.0, 12.0], [30.0, 30.0, 12.0]], dtype=np.float64)
    rgb_c = np.array([[180, 0, 0], [180, 0, 0]], dtype=np.uint8)
    res_oc_col = kernel.realize(
        box_tf,
        surface_refinement="open_catalog_sculpted",
        verts_ldu=pts_c,
        vertex_colors_rgb=rgb_c,
    )
    _check(res_oc_col.open_catalog_diagnostics.color_samples_available is True)
    _check(res_oc_col.open_catalog_diagnostics.color_sample_count == 2)
    _check(isinstance(res_oc_col.open_catalog_diagnostics.mean_color_delta_e00, float))
    _check(isinstance(res_oc_col.open_catalog_diagnostics.visible_skin_mean_delta_e00, float))

    res_comp_col = kernel.realize(
        box_tf,
        subassembly_graph=comp_nodes,
        verts_ldu=pts_c,
        vertex_colors_rgb=rgb_c,
    )
    _check(res_comp_col.open_catalog_diagnostics.color_samples_available is True)
    _check(res_comp_col.open_catalog_diagnostics.color_sample_count == 2)
    _check(isinstance(res_comp_col.open_catalog_diagnostics.mean_color_delta_e00, float))
    _check(isinstance(res_comp_col.open_catalog_diagnostics.visible_skin_mean_delta_e00, float))

    # 5C. v5.4.1 version parity & Schema 1.1.1 retention
    pyproj = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    cff_txt = (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    _check(pyproj["project"]["version"] == "5.4.1" == tb.__version__)
    _check("version: 5.4.1" in cff_txt)
    _check(CERTIFICATE_SCHEMA_VERSION == "1.1.1")
