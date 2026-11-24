"""
Typed Modular Stage Pipeline for `realize_open_catalog_sculpted` (`Phase 12 F12` & `§5.3`):

Decomposes target preparation, mold retrieval, chromatic inventory allocation, landmark
attachment, and diagnostics/timing finalization into cohesive, independently testable stages
with explicit stage-separated wall-clock timings (`stage_timings_ms`).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np

from topo_brick.evaluation.static_equilibrium import FoundationPolicy
from topo_brick.interface.brick_catalog import EXTENDED_BRICK_CATALOG
from topo_brick.interface.catalog_retrieval_index import (
    CATALOG_RETRIEVAL_INDEX,
    SCULPTURAL_TAXONOMY_CLASSES,
    CatalogRetrievalResult,
)
from topo_brick.interface.ldu_frames import LDUBox, LDUFrame
from topo_brick.interface.port_manifold import AssemblyState
from topo_brick.kernel.adaptive_surface_and_macro_planner import (
    AdaptiveSurfaceTarget,
    apply_hollow_shell_policy,
    evaluate_exterior_crust_fidelity,
)
from topo_brick.representation.interface_fields import SparseTiledTargetField


@dataclass(slots=True)
class PreparedOpenCatalogTarget:
    """Typed output of `TargetPrepStage.prepare(...)`."""

    nx_s: int
    ny_s: int
    nz_c: int
    n_plates_z: int
    s6x: int
    s6y: int
    s6z: int
    occ6: np.ndarray
    occ2: np.ndarray
    ad_occ6: np.ndarray
    solid_occ6_ref: Optional[np.ndarray]
    solid_occ6_padded: Optional[np.ndarray]
    gz6_nonempty_limit: int
    occ6_csum: np.ndarray
    eff_inv_map: Optional[Dict[str, int]]
    inv_allowed_for_ret: Optional[Sequence[str]]
    ret_res: CatalogRetrievalResult
    kit_ids: List[str]
    kit_set: Set[str]
    rem_inv: Optional[Dict[str, int]]
    max_inv_snapshot: Optional[Dict[str, int]]
    smooth_mode: bool
    view_weight_grid_6: Optional[np.ndarray]
    is_hollow_shell_active: bool
    target_prep_ms: float
    retrieval_ms: float


class TargetPrepStage:
    """Stage 1 & 2: Target grid padding, opening masking, shell policy, and Top-K mold retrieval."""

    @staticmethod
    def prepare(
        adaptive_target: AdaptiveSurfaceTarget,
        allowed_part_ids: Optional[Sequence[str]],
        part_retrieval: str,
        retrieval_top_k: int,
        license_filter: str,
        semantic_style_hint: Optional[str],
        inventory_counts: Optional[Dict[str, int]],
        forbidden_support_boxes: Sequence[LDUBox],
        colored_inventory: Optional[Any],
        hollow_shell_policy: Optional[Any],
        view_priority: Optional[Any],
        pre_ret: Optional[CatalogRetrievalResult] = None,
        deadline_monotonic_s: Optional[float] = None,
    ) -> PreparedOpenCatalogTarget:
        t0 = time.perf_counter()
        eff_shell_policy = hollow_shell_policy or adaptive_target.hollow_shell_policy
        solid_occ6_ref: Optional[np.ndarray] = adaptive_target.solid_occ_5x5x6
        src_occ2 = adaptive_target.occ_5x5x2
        src_occ6 = adaptive_target.occ_5x5x6
        if (
            eff_shell_policy is not None
            and getattr(eff_shell_policy, "enabled", False)
            and solid_occ6_ref is None
        ):
            solid_occ6_ref = adaptive_target.occ_5x5x6.copy()
            src_occ2, src_occ6 = apply_hollow_shell_policy(
                src_occ2, src_occ6, eff_shell_policy, forbidden_support_boxes=forbidden_support_boxes
            )

        nx_s = adaptive_target.nx_studs + 1
        ny_s = adaptive_target.ny_studs + 1
        nz_c = adaptive_target.nz_courses + 1
        n_plates_z = 3 * nz_c

        occ6 = np.zeros((4 * nx_s, 4 * ny_s, 4 * nz_c), dtype=bool)
        s6x, s6y, s6z = src_occ6.shape
        occ6[:s6x, :s6y, :s6z] = src_occ6

        solid_occ6_padded: Optional[np.ndarray] = None
        if solid_occ6_ref is not None:
            solid_occ6_padded = np.zeros_like(occ6)
            r6x, r6y, r6z = solid_occ6_ref.shape
            solid_occ6_padded[
                : min(occ6.shape[0], r6x),
                : min(occ6.shape[1], r6y),
                : min(occ6.shape[2], r6z),
            ] = solid_occ6_ref[
                : min(occ6.shape[0], r6x),
                : min(occ6.shape[1], r6y),
                : min(occ6.shape[2], r6z),
            ]

        occ2 = np.zeros((4 * nx_s, 4 * ny_s, 12 * nz_c), dtype=bool)
        s2x, s2y, s2z = src_occ2.shape
        occ2[:s2x, :s2y, :s2z] = src_occ2

        for fb in forbidden_support_boxes:
            fx0 = max(0, int(math.floor(fb.xmin / 5.0)))
            fx1 = min(occ6.shape[0], int(math.ceil(fb.xmax / 5.0)))
            fy0 = max(0, int(math.floor(fb.ymin / 5.0)))
            fy1 = min(occ6.shape[1], int(math.ceil(fb.ymax / 5.0)))
            fz6_0 = max(0, int(math.floor(fb.zmin / 6.0)))
            fz6_1 = min(occ6.shape[2], int(math.ceil(fb.zmax / 6.0)))
            fz2_0 = max(0, int(math.floor(fb.zmin / 2.0)))
            fz2_1 = min(occ2.shape[2], int(math.ceil(fb.zmax / 2.0)))
            if fx1 > fx0 and fy1 > fy0:
                if fz6_1 > fz6_0:
                    occ6[fx0:fx1, fy0:fy1, fz6_0:fz6_1] = False
                if fz2_1 > fz2_0:
                    occ2[fx0:fx1, fy0:fy1, fz2_0:fz2_1] = False

        z6_any = np.any(occ6, axis=(0, 1))
        gz6_nonempty_limit = (
            int(np.max(np.nonzero(z6_any)[0]) + 1) if np.any(z6_any) else occ6.shape[2]
        )
        if deadline_monotonic_s is not None and time.monotonic() >= deadline_monotonic_s:
            occ6_csum = np.zeros((1, 1, 1), dtype=np.int32)
        else:
            occ6_csum = np.zeros(
                (occ6.shape[0] + 1, occ6.shape[1] + 1, occ6.shape[2] + 1), dtype=np.int32
            )
            occ6_csum[1:, 1:, 1:] = np.cumsum(
                np.cumsum(np.cumsum(occ6.astype(np.int32), axis=0), axis=1), axis=2
            )

        view_weight_grid_6: Optional[np.ndarray] = None
        if view_priority is not None:
            from topo_brick.kernel.view_priority import compute_subvoxel_view_weights

            view_weight_grid_6 = compute_subvoxel_view_weights(
                occ6,
                view_priority,
                origin_ldu=getattr(adaptive_target, "origin_ldu", (0, 0, 0)),
            )

        target_prep_ms = (time.perf_counter() - t0) * 1000.0

        t_ret0 = time.perf_counter()
        eff_inv_map: Optional[Dict[str, int]] = None
        if colored_inventory is not None:
            eff_inv_map = colored_inventory.marginal_mold_counts()
            if inventory_counts is not None:
                for k_ic, v_ic in inventory_counts.items():
                    eff_inv_map[k_ic] = min(eff_inv_map.get(k_ic, int(v_ic)), int(v_ic))
        elif inventory_counts is not None:
            eff_inv_map = dict(inventory_counts)

        inv_allowed_for_ret: Optional[Sequence[str]] = allowed_part_ids
        if inv_allowed_for_ret is None and eff_inv_map is not None:
            inv_allowed_for_ret = tuple(sorted(pid for pid, q in eff_inv_map.items() if int(q) > 0))

        if deadline_monotonic_s is not None and time.monotonic() >= deadline_monotonic_s and pre_ret is None:
            ret_res = CatalogRetrievalResult(
                retrieved_part_ids=("3005",),
                sculptural_part_ids=(),
                backbone_part_ids=("3005",),
                class_counts={"CORE_BRICK": 1},
                motif_weights={},
                license_filter=license_filter,
                query_wall_ms=0.0,
            )
        else:
            ret_res = (
                pre_ret
                if pre_ret is not None
                else CATALOG_RETRIEVAL_INDEX.retrieve_top_k(
                    occ6,
                    top_k=retrieval_top_k,
                    license_filter=license_filter,
                    semantic_style_hint=semantic_style_hint,
                    allowed_part_ids=inv_allowed_for_ret,
                )
            )
        if part_retrieval == "open_catalog_top_k":
            kit_ids = list(ret_res.retrieved_part_ids)
        elif allowed_part_ids is not None:
            kit_ids = list(allowed_part_ids)
        else:
            kit_ids = list(ret_res.retrieved_part_ids)

        if license_filter == "cc_by_4_0_only":
            filtered_kit: List[str] = []
            for pid in kit_ids:
                if pid in EXTENDED_BRICK_CATALOG:
                    filtered_kit.append(pid)
                    continue
                can_id = CATALOG_RETRIEVAL_INDEX.resolve_canonical_mold_id(pid) or pid
                ent = CATALOG_RETRIEVAL_INDEX.get_entry(can_id)
                if ent is not None and ent.license_tier == "cc_by_4_0":
                    filtered_kit.append(pid)
            kit_ids = filtered_kit

        kit_set = set(kit_ids)
        rem_inv = dict(eff_inv_map) if eff_inv_map is not None else None
        max_inv_snapshot = dict(eff_inv_map) if eff_inv_map is not None else None
        smooth_mode = semantic_style_hint == "smooth_studless"
        ad_occ6 = occ6[:s6x, :s6y, :s6z]

        is_hollow_shell_active = bool(
            eff_shell_policy is not None and getattr(eff_shell_policy, "enabled", False)
        )
        if is_hollow_shell_active and allowed_part_ids is None and eff_inv_map is None:
            for girder_id in (
                "6111", "3006", "3007", "3008", "2456", "3009", "3001", "3010",
                "4477", "3034", "3460", "3666", "3020", "3710",
            ):
                if girder_id not in kit_set:
                    kit_ids.append(girder_id)
                    kit_set.add(girder_id)

        retrieval_ms = (time.perf_counter() - t_ret0) * 1000.0
        return PreparedOpenCatalogTarget(
            nx_s=nx_s,
            ny_s=ny_s,
            nz_c=nz_c,
            n_plates_z=n_plates_z,
            s6x=s6x,
            s6y=s6y,
            s6z=s6z,
            occ6=occ6,
            occ2=occ2,
            ad_occ6=ad_occ6,
            solid_occ6_ref=solid_occ6_ref,
            solid_occ6_padded=solid_occ6_padded,
            gz6_nonempty_limit=gz6_nonempty_limit,
            occ6_csum=occ6_csum,
            eff_inv_map=eff_inv_map,
            inv_allowed_for_ret=inv_allowed_for_ret,
            ret_res=ret_res,
            kit_ids=kit_ids,
            kit_set=kit_set,
            rem_inv=rem_inv,
            max_inv_snapshot=max_inv_snapshot,
            smooth_mode=smooth_mode,
            view_weight_grid_6=view_weight_grid_6,
            is_hollow_shell_active=is_hollow_shell_active,
            target_prep_ms=target_prep_ms,
            retrieval_ms=retrieval_ms,
        )


class ChromaticAllocationStage:
    """Stage 6: Perceptual CIE Lab (`CIEDE2000`) or Two-Zone Finite-Inventory Chromatic Flow."""

    @staticmethod
    def run(
        state: AssemblyState,
        adaptive_target: AdaptiveSurfaceTarget,
        verts_ldu: Optional[np.ndarray],
        vertex_colors_rgb: Optional[np.ndarray],
        colored_inventory: Optional[Any],
        validator: Any,
        foundation_policy: FoundationPolicy,
        view_weight_grid_6: Optional[np.ndarray],
        view_priority: Optional[Any],
        perceptual_color_fn: Callable[..., float],
        open_color_palette_mode: str = "solid_54_default",
    ) -> Tuple[float, Optional[Any], float]:
        t0 = time.perf_counter()
        mean_de00 = 0.0
        chrom_res = None
        v_pts_color = verts_ldu if verts_ldu is not None else adaptive_target.mesh_surf_pts
        if colored_inventory is not None:
            from topo_brick.kernel.colored_inventory_allocator import (
                allocate_inventory_colors_min_cost,
                apply_chromatic_same_footprint_splits,
            )

            chrom_res = allocate_inventory_colors_min_cost(
                state=state,
                colored_inventory=colored_inventory,
                verts_ldu=v_pts_color,
                vertex_colors_rgb=vertex_colors_rgb,
                view_weight_grid_6=view_weight_grid_6,
                view_priority=view_priority,
            )
            chrom_res = apply_chromatic_same_footprint_splits(
                state=state,
                alloc=chrom_res,
                colored_inventory=colored_inventory,
                verts_ldu=v_pts_color,
                vertex_colors_rgb=vertex_colors_rgb,
                validator=validator,
                foundation_policy=foundation_policy,
                view_weight_grid_6=view_weight_grid_6,
                view_priority=view_priority,
            )
            mean_de00 = float(chrom_res.skin_mean_delta_e00)
        elif vertex_colors_rgb is not None:
            if open_color_palette_mode != "solid_54_default" or view_priority is not None:
                mean_de00 = float(
                    perceptual_color_fn(
                        state,
                        v_pts_color,
                        vertex_colors_rgb,
                        open_color_palette_mode=open_color_palette_mode,
                        view_priority=view_priority,
                    )
                )
            else:
                mean_de00 = float(perceptual_color_fn(state, v_pts_color, vertex_colors_rgb))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return mean_de00, chrom_res, elapsed_ms


class TraceCertifyStage:
    """Stage 7: Landmark attachment, camera/crust fidelity, fine target packing, and stage timings."""

    @staticmethod
    def finalize(
        state: AssemblyState,
        occ6: np.ndarray,
        solid_occ6_ref: Optional[np.ndarray],
        adaptive_target: AdaptiveSurfaceTarget,
        verts_ldu: Optional[np.ndarray],
        vertex_colors_rgb: Optional[np.ndarray],
        view_priority: Optional[Any],
        ret_res: CatalogRetrievalResult,
        mean_de00: float,
        chrom_res: Optional[Any],
        landmarks_attached_count: int,
        plate_to_tile_values: Set[str],
        build_fine_tf_fn: Callable[[np.ndarray], SparseTiledTargetField],
        count_exposed_studs_fn: Callable[[AssemblyState], int],
        diagnostics_cls: Any,
        target_prep_ms: float,
        retrieval_ms: float,
        sculpt_synthesis_ms: float,
        color_allocation_ms: float,
        *,
        license_filter: str = "all",
        allowed_part_ids: Optional[Sequence[str]] = None,
        colored_inventory: Optional[Any] = None,
        vertex_region_labels: Optional[Sequence[Any]] = None,
        open_color_palette_mode: str = "solid_54_default",
        symmetry_plane_axis: Optional[str] = None,
        symmetry_plane_coord_ldu: Optional[float] = None,
        actual_candidate_poses_scored: int = 0,
        extra_stage_timings_ms: Optional[Dict[str, float]] = None,
    ) -> Tuple[AssemblyState, SparseTiledTargetField, Any]:
        t0 = time.perf_counter()
        v_pts_color = verts_ldu if verts_ldu is not None else adaptive_target.mesh_surf_pts
        crust_iou_val, vol_sav_val = (
            evaluate_exterior_crust_fidelity(solid_occ6_ref, state)
            if solid_occ6_ref is not None
            else (0.0, 0.0)
        )
        cam_reproj = None
        if view_priority is not None:
            from topo_brick.kernel.view_priority import evaluate_camera_reprojection_fidelity

            cam_reproj = evaluate_camera_reprojection_fidelity(
                state=state,
                occ_5x5x6=occ6,
                spec=view_priority,
                verts_ldu=v_pts_color,
                vertex_colors_rgb=vertex_colors_rgb,
                origin_ldu=getattr(adaptive_target, "origin_ldu", (0, 0, 0)),
            )

        fine_tf = build_fine_tf_fn(occ6)
        distinct_placed = tuple(sorted({p.part_id for p in state.placed_parts.values()}))
        sculpt_cnt = 0
        jump_cnt = 0
        snot_cnt = 0
        tile_cnt = 0
        for p in state.placed_parts.values():
            can_p = CATALOG_RETRIEVAL_INDEX.resolve_canonical_mold_id(p.part_id) or p.part_id
            ent = CATALOG_RETRIEVAL_INDEX.get_entry(can_p)
            cls_p = ent.taxonomy_class if ent is not None else ""
            if cls_p in SCULPTURAL_TAXONOMY_CLASSES:
                sculpt_cnt += 1
            if cls_p == "JUMPER_HALF_STUD":
                jump_cnt += 1
            if cls_p == "SNOT_ANCHOR" or LDUFrame(p.pose.rot_id, (0, 0, 0)).z_axis[2] == 0:
                snot_cnt += 1
            if cls_p == "SMOOTH_TILE" or p.part_id in plate_to_tile_values:
                tile_cnt += 1

        from topo_brick.kernel.design_diagnostics import (
            build_catalog_funnel_and_shortlist_report,
            evaluate_bilateral_construction_symmetry,
            evaluate_regional_appearance_diagnostics,
        )

        funnel_rep = build_catalog_funnel_and_shortlist_report(
            state=state,
            retrieval_result=ret_res,
            license_filter=license_filter,
            allowed_part_ids=allowed_part_ids,
            plate_to_tile_values=tuple(plate_to_tile_values),
            actual_candidate_poses_scored=actual_candidate_poses_scored,
        )
        reg_app_rep = evaluate_regional_appearance_diagnostics(
            state=state,
            verts_ldu=v_pts_color,
            vertex_colors_rgb=vertex_colors_rgb,
            vertex_region_labels=vertex_region_labels,
            view_priority=view_priority,
            colored_inventory=colored_inventory,
            open_color_palette_mode=open_color_palette_mode,
        )
        sym_rep = (
            evaluate_bilateral_construction_symmetry(
                state,
                axis=symmetry_plane_axis,
                plane_coord_ldu=symmetry_plane_coord_ldu,
                verts_ldu=v_pts_color,
                vertex_region_labels=vertex_region_labels,
            )
            if symmetry_plane_axis is not None
            else None
        )

        finalize_ms = (time.perf_counter() - t0) * 1000.0
        stage_timings: Dict[str, float] = {
            "target_prep_ms": round(target_prep_ms, 4),
            "retrieval_ms": round(retrieval_ms, 4),
            "sculpt_synthesis_ms": round(sculpt_synthesis_ms, 4),
            "color_allocation_ms": round(color_allocation_ms, 4),
            "diagnostics_finalize_ms": round(finalize_ms, 4),
        }
        substage_timings: Dict[str, float] = (
            {str(k_st): round(float(v_st), 4) for k_st, v_st in extra_stage_timings_ms.items()}
            if extra_stage_timings_ms
            else {}
        )
        has_col_samples = bool(reg_app_rep.color_samples_available)
        n_col_samples = (
            int(len(vertex_colors_rgb))
            if (has_col_samples and vertex_colors_rgb is not None)
            else 0
        )
        vis_skin_de00 = (
            round(float(reg_app_rep.visible_skin_mean_delta_e00), 4)
            if (has_col_samples and reg_app_rep.visible_skin_mean_delta_e00 is not None)
            else None
        )
        diag = diagnostics_cls(
            retrieval_result=ret_res,
            distinct_parts_placed=distinct_placed,
            sculptural_parts_placed_count=sculpt_cnt,
            jumper_parts_placed_count=jump_cnt,
            snot_parts_placed_count=snot_cnt,
            smooth_tiles_placed_count=tile_cnt,
            exposed_upward_studs_count=count_exposed_studs_fn(state),
            mean_color_delta_e00=round(float(mean_de00), 4) if has_col_samples else None,
            visible_skin_mean_delta_e00=vis_skin_de00,
            color_samples_available=has_col_samples,
            color_sample_count=n_col_samples,
            chromatic_allocation=chrom_res,
            exterior_crust_iou_5x5x6=crust_iou_val,
            volume_savings_ratio=vol_sav_val,
            camera_reprojection=cam_reproj,
            landmarks_attached_count=landmarks_attached_count,
            stage_timings_ms=stage_timings,
            sculpt_substage_timings_ms=substage_timings,
            catalog_funnel_report=funnel_rep,
            regional_appearance_report=reg_app_rep,
            bilateral_symmetry_report=sym_rep,
        )
        return state, fine_tf, diag
