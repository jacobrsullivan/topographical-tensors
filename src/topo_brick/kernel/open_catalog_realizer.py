"""
Open-Catalog Sculpted Realizer Pipeline (`Phase 15` modularized stage orchestrator).

Stages (`< 300` lines total):
  1. `TargetPrepStage.prepare` (`topo_brick.kernel.open_catalog_stages`)
  2. `PortalFrustumFastPathStage.try_realize` (`topo_brick.kernel.sculpted_stages`)
  3. `ErodedCorePlanningStage.plan` + `build_and_commit_core_session`
  4. `SculpturalSkinAndPruneStage.run`
  5. `SmoothTileAndSlabPortfolioStage` (`apply_smooth_tile_swap` + `run_portfolio_and_capping_refinement`)
  6. `attach_semantic_landmarks` + `ChromaticAllocationStage.run`
  7. `TraceCertifyStage.finalize` (including `CatalogFunnelAndShortlistReport`,
     `RegionalAppearanceReport`, and `BilateralSymmetryReport`)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from topo_brick.evaluation.static_equilibrium import FoundationPolicy
from topo_brick.interface.catalog_retrieval_index import CatalogRetrievalResult
from topo_brick.interface.ldu_frames import LDUBox
from topo_brick.interface.port_manifold import AssemblyState
from topo_brick.kernel.adaptive_surface_and_macro_planner import AdaptiveSurfaceTarget
from topo_brick.kernel.open_catalog_stages import (
    ChromaticAllocationStage,
    TargetPrepStage,
    TraceCertifyStage,
)
from topo_brick.kernel.sculpted_primitives import (
    BRICK_TO_PLATE_AND_TILE_MAP,
    LDRAW_SOLID_PALETTE_LAB,
    LDRAW_SOLID_PALETTE_SRGB,
    PLATE_TO_SMOOTH_TILE_MAP,
    _box_hits_forbidden,
    _build_fine_target_field_from_occ6,
    _cached_local_csg_2ldu,
    _fast_rel_mask6,
    _fast_slope_grid_templates,
    _fast_upward_stud_templates,
    _rect_templates,
    _tile_boolean_layer_with_rects,
    _world_csg_keys_even,
    assign_perceptual_ldraw_colors,
    ciede2000_delta_e,
    count_exposed_upward_studs,
    srgb_to_cielab,
)
from topo_brick.kernel.sculpted_stages import (
    ErodedCorePlanningStage,
    PortalFrustumFastPathStage,
    SculptedLiveSession,
    SculpturalSkinAndPruneStage,
    SmoothTileAndSlabPortfolioStage,
    build_and_commit_core_session,
)
from topo_brick.representation.interface_fields import SparseTiledTargetField

__all__ = [
    "BRICK_TO_PLATE_AND_TILE_MAP",
    "LDRAW_SOLID_PALETTE_LAB",
    "LDRAW_SOLID_PALETTE_SRGB",
    "OpenCatalogRealizeDiagnostics",
    "PLATE_TO_SMOOTH_TILE_MAP",
    "_box_hits_forbidden",
    "_build_fine_target_field_from_occ6",
    "_cached_local_csg_2ldu",
    "_fast_rel_mask6",
    "_fast_slope_grid_templates",
    "_fast_upward_stud_templates",
    "_rect_templates",
    "_tile_boolean_layer_with_rects",
    "_world_csg_keys_even",
    "assign_perceptual_ldraw_colors",
    "ciede2000_delta_e",
    "count_exposed_upward_studs",
    "realize_open_catalog_sculpted",
    "srgb_to_cielab",
]


@dataclass(frozen=True, slots=True)
class OpenCatalogRealizeDiagnostics:
    """Telemetry produced by `realize_open_catalog_sculpted(...)` (`Phase 10–15`)."""

    retrieval_result: CatalogRetrievalResult
    distinct_parts_placed: Tuple[str, ...]
    sculptural_parts_placed_count: int
    jumper_parts_placed_count: int
    snot_parts_placed_count: int
    smooth_tiles_placed_count: int
    exposed_upward_studs_count: int
    mean_color_delta_e00: Optional[float] = None
    visible_skin_mean_delta_e00: Optional[float] = None
    color_samples_available: bool = False
    color_sample_count: int = 0
    chromatic_allocation: Optional[Any] = None
    exterior_crust_iou_5x5x6: float = 0.0
    volume_savings_ratio: float = 0.0
    camera_reprojection: Optional[Any] = None
    landmarks_attached_count: int = 0
    stage_timings_ms: Dict[str, float] = field(default_factory=dict)
    sculpt_substage_timings_ms: Dict[str, float] = field(default_factory=dict)
    catalog_funnel_report: Optional[Any] = None
    regional_appearance_report: Optional[Any] = None
    bilateral_symmetry_report: Optional[Any] = None


def realize_open_catalog_sculpted(
    kernel: Any,
    coarse_target_field: SparseTiledTargetField | Any,
    adaptive_target: AdaptiveSurfaceTarget,
    allowed_part_ids: Optional[Sequence[str]] = None,
    part_retrieval: str = "open_catalog_top_k",
    retrieval_top_k: int = 32,
    license_filter: str = "all",
    semantic_style_hint: Optional[str] = None,
    vertex_colors_rgb: Optional[np.ndarray] = None,
    verts_ldu: Optional[np.ndarray] = None,
    inventory_counts: Optional[Dict[str, int]] = None,
    support_policy: str = "minimal_permanent_pillars",
    foundation_policy: FoundationPolicy = FoundationPolicy.CLUTCHED_BASEPLATE,
    forbidden_support_boxes: Sequence[LDUBox] = (),
    colored_inventory: Optional[Any] = None,
    hollow_shell_policy: Optional[Any] = None,
    view_priority: Optional[Any] = None,
    suppress_studless_capping_boxes: Sequence[LDUBox] = (),
    landmark_requests: Sequence[Any] = (),
    deadline_monotonic_s: Optional[float] = None,
    ground_plane_z_ldu: float = 0.0,
    vertex_region_labels: Optional[np.ndarray] = None,
    open_color_palette_mode: str = "solid_54_default",
    symmetry_plane_axis: Optional[str] = None,
    symmetry_plane_coord_ldu: Optional[float] = None,
    _forced_plan: Optional[Tuple[List[Tuple[int, int]], Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]], int]] = None,
    _pre_ret: Optional[CatalogRetrievalResult] = None,
    _eval_only: bool = False,
    _max_waves: int = 2,
) -> Tuple[AssemblyState, SparseTiledTargetField, OpenCatalogRealizeDiagnostics]:
    """Synthesize a 6-DoF-certified open-catalog sculpted LEGO assembly through explicit stages."""
    if getattr(adaptive_target, "origin_ldu", (0, 0, 0)) != (0, 0, 0) or abs(float(ground_plane_z_ldu)) > 1e-6:
        from topo_brick.kernel.workbench_frame import WorkbenchFrameTransform

        wb_in = WorkbenchFrameTransform.to_local_workbench(
            coarse_target_field=coarse_target_field,
            adaptive_target=adaptive_target,
            world_ground_plane_z_ldu=ground_plane_z_ldu,
            forbidden_support_boxes=forbidden_support_boxes,
            suppress_studless_capping_boxes=suppress_studless_capping_boxes,
            landmark_requests=landmark_requests,
            verts_ldu=verts_ldu,
            symmetry_plane_axis=symmetry_plane_axis,
            symmetry_plane_coord_ldu=symmetry_plane_coord_ldu,
        )
        if wb_in.origin_ldu != (0, 0, 0):
            loc_vp = view_priority
            if view_priority is not None and getattr(view_priority, "camera_pos_ldu", None) is not None:
                from dataclasses import replace as _dc_replace

                cpos = view_priority.camera_pos_ldu
                loc_vp = _dc_replace(
                    view_priority,
                    camera_pos_ldu=(
                        float(cpos[0]) - float(wb_in.origin_ldu[0]),
                        float(cpos[1]) - float(wb_in.origin_ldu[1]),
                        float(cpos[2]) - float(wb_in.origin_ldu[2]),
                    ),
                )
            loc_st, loc_fine_tf, loc_diag = realize_open_catalog_sculpted(
                kernel=kernel,
                coarse_target_field=wb_in.local_coarse_target_field,
                adaptive_target=wb_in.local_adaptive_target,
                allowed_part_ids=allowed_part_ids,
                part_retrieval=part_retrieval,
                retrieval_top_k=retrieval_top_k,
                license_filter=license_filter,
                semantic_style_hint=semantic_style_hint,
                vertex_colors_rgb=vertex_colors_rgb,
                verts_ldu=wb_in.local_verts_ldu,
                inventory_counts=inventory_counts,
                support_policy=support_policy,
                foundation_policy=foundation_policy,
                forbidden_support_boxes=wb_in.local_forbidden_support_boxes,
                colored_inventory=colored_inventory,
                hollow_shell_policy=hollow_shell_policy,
                view_priority=loc_vp,
                suppress_studless_capping_boxes=wb_in.local_suppress_capping_boxes,
                landmark_requests=wb_in.local_landmark_requests,
                deadline_monotonic_s=deadline_monotonic_s,
                ground_plane_z_ldu=wb_in.local_ground_plane_z_ldu,
                vertex_region_labels=vertex_region_labels,
                open_color_palette_mode=open_color_palette_mode,
                symmetry_plane_axis=symmetry_plane_axis,
                symmetry_plane_coord_ldu=wb_in.local_symmetry_plane_coord_ldu,
                _forced_plan=_forced_plan,
                _pre_ret=_pre_ret,
                _eval_only=_eval_only,
                _max_waves=_max_waves,
            )
            return WorkbenchFrameTransform.to_world_frame(
                local_state=loc_st,
                local_fine_target_field=loc_fine_tf,
                local_diagnostics=loc_diag,
                origin_ldu=wb_in.origin_ldu,
                world_ground_plane_z_ldu=ground_plane_z_ldu,
            )

    prep = TargetPrepStage.prepare(
        adaptive_target=adaptive_target,
        allowed_part_ids=allowed_part_ids,
        part_retrieval=part_retrieval,
        retrieval_top_k=retrieval_top_k,
        license_filter=license_filter,
        semantic_style_hint=semantic_style_hint,
        inventory_counts=inventory_counts,
        forbidden_support_boxes=forbidden_support_boxes,
        colored_inventory=colored_inventory,
        hollow_shell_policy=hollow_shell_policy,
        view_priority=view_priority,
        pre_ret=_pre_ret,
        deadline_monotonic_s=deadline_monotonic_s,
    )
    eff_shell_policy = hollow_shell_policy or adaptive_target.hollow_shell_policy
    t_sculpt0 = time.perf_counter()

    def _early_expired() -> Tuple[AssemblyState, SparseTiledTargetField, OpenCatalogRealizeDiagnostics]:
        return TraceCertifyStage.finalize(
            state=AssemblyState(),
            occ6=prep.occ6,
            solid_occ6_ref=prep.solid_occ6_ref,
            adaptive_target=adaptive_target,
            verts_ldu=verts_ldu,
            vertex_colors_rgb=vertex_colors_rgb,
            view_priority=view_priority,
            ret_res=prep.ret_res,
            mean_de00=0.0,
            chrom_res=None,
            landmarks_attached_count=0,
            plate_to_tile_values=set(PLATE_TO_SMOOTH_TILE_MAP.values()),
            build_fine_tf_fn=(lambda _occ: SparseTiledTargetField(boxes=())),
            count_exposed_studs_fn=count_exposed_upward_studs,
            diagnostics_cls=OpenCatalogRealizeDiagnostics,
            target_prep_ms=prep.target_prep_ms,
            retrieval_ms=prep.retrieval_ms,
            sculpt_synthesis_ms=(time.perf_counter() - t_sculpt0) * 1000.0,
            color_allocation_ms=0.0,
            license_filter=license_filter,
            allowed_part_ids=prep.inv_allowed_for_ret,
            colored_inventory=colored_inventory,
            vertex_region_labels=vertex_region_labels,
            open_color_palette_mode=open_color_palette_mode,
            symmetry_plane_axis=symmetry_plane_axis,
            symmetry_plane_coord_ldu=symmetry_plane_coord_ldu,
        )

    if deadline_monotonic_s is not None and time.monotonic() >= deadline_monotonic_s:
        return _early_expired()

    fast_res = PortalFrustumFastPathStage.try_realize(
        kernel=kernel,
        prep=prep,
        adaptive_target=adaptive_target,
        foundation_policy=foundation_policy,
        forbidden_support_boxes=forbidden_support_boxes,
        suppress_studless_capping_boxes=suppress_studless_capping_boxes,
        view_priority=view_priority,
        landmark_requests=landmark_requests,
        verts_ldu=verts_ldu,
        vertex_colors_rgb=vertex_colors_rgb,
        colored_inventory=colored_inventory,
        license_filter=license_filter,
        allowed_part_ids=prep.inv_allowed_for_ret,
        vertex_region_labels=vertex_region_labels,
        open_color_palette_mode=open_color_palette_mode,
        symmetry_plane_axis=symmetry_plane_axis,
        symmetry_plane_coord_ldu=symmetry_plane_coord_ldu,
        forced_plan=_forced_plan,
        eval_only=_eval_only,
        diagnostics_cls=OpenCatalogRealizeDiagnostics,
        t_sculpt0=t_sculpt0,
        deadline_monotonic_s=deadline_monotonic_s,
        ground_plane_z_ldu=ground_plane_z_ldu,
    )
    if fast_res is not None:
        return fast_res

    t_core0 = time.perf_counter()
    core_plan = ErodedCorePlanningStage.plan(
        prep=prep,
        adaptive_target=adaptive_target,
        eff_shell_policy=eff_shell_policy,
        support_policy=support_policy,
        foundation_policy=foundation_policy,
        forbidden_support_boxes=forbidden_support_boxes,
        license_filter=license_filter,
        view_priority=view_priority,
        deadline_monotonic_s=deadline_monotonic_s,
        _forced_plan=_forced_plan,
    )
    if core_plan is None:
        return _early_expired()

    sess = SculptedLiveSession(
        kernel=kernel,
        prep=prep,
        core_plan=core_plan,
        coarse_target_field=coarse_target_field,
        adaptive_target=adaptive_target,
        allowed_part_ids=allowed_part_ids,
        part_retrieval=part_retrieval,
        retrieval_top_k=retrieval_top_k,
        license_filter=license_filter,
        semantic_style_hint=semantic_style_hint,
        inventory_counts=inventory_counts,
        support_policy=support_policy,
        foundation_policy=foundation_policy,
        forbidden_support_boxes=forbidden_support_boxes,
        colored_inventory=colored_inventory,
        hollow_shell_policy=hollow_shell_policy,
        view_priority=view_priority,
        suppress_studless_capping_boxes=suppress_studless_capping_boxes,
        deadline_monotonic_s=deadline_monotonic_s,
        ground_plane_z_ldu=ground_plane_z_ldu,
        forced_plan=_forced_plan,
        eval_only=_eval_only,
        max_waves=_max_waves,
        verts_ldu=verts_ldu,
    )
    snap_core = sess.capture_invariant_snapshot()
    build_and_commit_core_session(sess)
    sess.verify_commit_transition(snap_core, allow_pruning=False)
    eroded_core_ms = (time.perf_counter() - t_core0) * 1000.0

    t_skin0 = time.perf_counter()
    snap_skin = sess.capture_invariant_snapshot()
    gx6, gy6, gz6 = SculpturalSkinAndPruneStage.run(sess)
    sess.verify_commit_transition(snap_skin, allow_pruning=True)
    sculptural_skin_ms = (time.perf_counter() - t_skin0) * 1000.0

    t_tile0 = time.perf_counter()
    snap_tile = sess.capture_invariant_snapshot()
    SmoothTileAndSlabPortfolioStage.apply_smooth_tile_swap(sess)
    sess.verify_commit_transition(snap_tile, allow_pruning=False)
    smooth_tile_ms = (time.perf_counter() - t_tile0) * 1000.0

    if _eval_only:
        return (
            sess.state,
            coarse_target_field,
            OpenCatalogRealizeDiagnostics(
                retrieval_result=prep.ret_res,
                distinct_parts_placed=(),
                sculptural_parts_placed_count=0,
                jumper_parts_placed_count=0,
                snot_parts_placed_count=0,
                smooth_tiles_placed_count=0,
                exposed_upward_studs_count=0,
                mean_color_delta_e00=None,
                color_samples_available=False,
                color_sample_count=0,
            ),
        )

    t_port0 = time.perf_counter()
    snap_port = sess.capture_invariant_snapshot()
    SmoothTileAndSlabPortfolioStage.run_portfolio_and_capping_refinement(
        sess,
        gx6=gx6,
        gy6=gy6,
        gz6=gz6,
        recursive_realize_fn=realize_open_catalog_sculpted,
    )
    sess.verify_commit_transition(snap_port, allow_pruning=True)
    portfolio_refine_ms = (time.perf_counter() - t_port0) * 1000.0

    t_lm0 = time.perf_counter()
    lm_attached_cnt = 0
    is_expired = sess.deadline_expired() if sess.deadline_expired is not None else False
    if landmark_requests and not is_expired:
        from topo_brick.kernel.subassembly_realizer import attach_semantic_landmarks

        lm_res = attach_semantic_landmarks(
            kernel=kernel,
            state=sess.state,
            requests=landmark_requests,
            default_license_filter=license_filter,
            allowed_part_ids=prep.inv_allowed_for_ret,
            rem_inv=sess.rem_inv,
            foundation_policy=foundation_policy,
            ground_plane_z_ldu=ground_plane_z_ldu,
            deadline_monotonic_s=deadline_monotonic_s,
        )
        lm_attached_cnt = len(lm_res)
    landmark_ms = (time.perf_counter() - t_lm0) * 1000.0

    sculpt_elapsed_ms = (time.perf_counter() - t_sculpt0) * 1000.0
    is_expired_after = sess.deadline_expired() if sess.deadline_expired is not None else False
    if is_expired_after:
        mean_de00, chrom_res, color_elapsed_ms = 0.0, None, 0.0
        eff_fine_tf_fn = lambda _occ: SparseTiledTargetField(boxes=())
    else:
        mean_de00, chrom_res, color_elapsed_ms = ChromaticAllocationStage.run(
            state=sess.state,
            adaptive_target=adaptive_target,
            verts_ldu=verts_ldu,
            vertex_colors_rgb=vertex_colors_rgb,
            colored_inventory=colored_inventory,
            validator=kernel.validator,
            foundation_policy=foundation_policy,
            view_weight_grid_6=prep.view_weight_grid_6,
            view_priority=view_priority,
            perceptual_color_fn=assign_perceptual_ldraw_colors,
            open_color_palette_mode=open_color_palette_mode,
        )
        eff_fine_tf_fn = _build_fine_target_field_from_occ6

    return TraceCertifyStage.finalize(
        state=sess.state,
        occ6=prep.occ6,
        solid_occ6_ref=prep.solid_occ6_ref,
        adaptive_target=adaptive_target,
        verts_ldu=verts_ldu,
        vertex_colors_rgb=vertex_colors_rgb,
        view_priority=view_priority,
        ret_res=prep.ret_res,
        mean_de00=mean_de00,
        chrom_res=chrom_res,
        landmarks_attached_count=lm_attached_cnt,
        plate_to_tile_values=set(PLATE_TO_SMOOTH_TILE_MAP.values()),
        build_fine_tf_fn=eff_fine_tf_fn,
        count_exposed_studs_fn=count_exposed_upward_studs,
        diagnostics_cls=OpenCatalogRealizeDiagnostics,
        target_prep_ms=prep.target_prep_ms,
        retrieval_ms=prep.retrieval_ms,
        sculpt_synthesis_ms=sculpt_elapsed_ms,
        color_allocation_ms=color_elapsed_ms,
        license_filter=license_filter,
        allowed_part_ids=prep.inv_allowed_for_ret,
        colored_inventory=colored_inventory,
        vertex_region_labels=vertex_region_labels,
        open_color_palette_mode=open_color_palette_mode,
        symmetry_plane_axis=symmetry_plane_axis,
        symmetry_plane_coord_ldu=symmetry_plane_coord_ldu,
        actual_candidate_poses_scored=sess.candidate_poses_scored,
        extra_stage_timings_ms={
            "eroded_core_ms": eroded_core_ms,
            "sculptural_skin_ms": sculptural_skin_ms,
            "smooth_tile_ms": smooth_tile_ms,
            "portfolio_refine_ms": portfolio_refine_ms,
            "landmark_ms": landmark_ms,
        },
    )
