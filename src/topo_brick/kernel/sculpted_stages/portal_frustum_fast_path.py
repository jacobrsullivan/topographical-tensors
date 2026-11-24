"""
Stage 2b: Upfront `< 8 ms` blind sub-plate portal-bridge & stepped-frustum fast path (`Phase 15`).
"""

from __future__ import annotations

import time
from typing import Any, Optional, Sequence, Tuple
import numpy as np

from topo_brick.evaluation.static_equilibrium import FoundationPolicy
from topo_brick.interface.ldu_frames import LDUBox
from topo_brick.interface.port_manifold import AssemblyState
from topo_brick.kernel.open_catalog_blind_portfolio import (
    _apply_terminal_tile_swap,
    try_upfront_blind_sculpted_fastpath,
)
from topo_brick.kernel.open_catalog_stages import ChromaticAllocationStage, TraceCertifyStage
from topo_brick.kernel.sculpted_stages.commit_session import (
    verify_standalone_sculpted_state_invariants,
)
from topo_brick.kernel.sculpted_primitives import (
    PLATE_TO_SMOOTH_TILE_MAP,
    _build_fine_target_field_from_occ6,
    assign_perceptual_ldraw_colors,
    count_exposed_upward_studs,
)
from topo_brick.representation.interface_fields import SparseTiledTargetField


class PortalFrustumFastPathStage:
    """Attempt `< 8 ms` closed-form portal-bridge or stepped-frustum template synthesis."""

    @staticmethod
    def try_realize(
        *,
        kernel: Any,
        prep: Any,
        adaptive_target: Any,
        foundation_policy: FoundationPolicy,
        forbidden_support_boxes: Sequence[LDUBox],
        suppress_studless_capping_boxes: Sequence[LDUBox],
        view_priority: Optional[Any],
        landmark_requests: Sequence[Any],
        verts_ldu: Optional[np.ndarray],
        vertex_colors_rgb: Optional[np.ndarray],
        colored_inventory: Optional[Any],
        license_filter: str,
        allowed_part_ids: Optional[Sequence[str]],
        vertex_region_labels: Optional[np.ndarray],
        open_color_palette_mode: str,
        symmetry_plane_axis: Optional[str],
        symmetry_plane_coord_ldu: Optional[float],
        forced_plan: Optional[Any],
        eval_only: bool,
        diagnostics_cls: Any,
        t_sculpt0: float,
        deadline_monotonic_s: Optional[float] = None,
        ground_plane_z_ldu: float = 0.0,
    ) -> Optional[Tuple[AssemblyState, SparseTiledTargetField, Any]]:
        if deadline_monotonic_s is not None and time.monotonic() >= deadline_monotonic_s:
            return None
        if not (
            forced_plan is None
            and not eval_only
            and foundation_policy == FoundationPolicy.CLUTCHED_BASEPLATE
            and abs(float(ground_plane_z_ldu)) <= 0.25
            and view_priority is None
            and not landmark_requests
            and not prep.is_hollow_shell_active
        ):
            return None
        st_fast = try_upfront_blind_sculpted_fastpath(
            kernel=kernel,
            occ6=prep.ad_occ6,
            kit_ids=prep.kit_ids,
            forbidden_support_boxes=forbidden_support_boxes,
            foundation_policy=foundation_policy,
            max_inventory_counts=prep.max_inv_snapshot,
            deadline_monotonic_s=deadline_monotonic_s,
            ground_plane_z_ldu=ground_plane_z_ldu,
        )
        if st_fast is None:
            return None
        if deadline_monotonic_s is not None and time.monotonic() >= deadline_monotonic_s:
            return None
        rem_fast = dict(prep.max_inv_snapshot) if prep.max_inv_snapshot is not None else None
        if rem_fast is not None:
            for pf_u in st_fast.placed_parts.values():
                rem_q = int(rem_fast.get(pf_u.part_id, 0)) - 1
                if rem_q < 0:
                    return None
                rem_fast[pf_u.part_id] = rem_q
        if prep.smooth_mode:
            _apply_terminal_tile_swap(
                st_fast,
                prep.kit_set,
                suppress_boxes=suppress_studless_capping_boxes,
                rem_inv=rem_fast,
            )
        sculpt_fast_ms = (time.perf_counter() - t_sculpt0) * 1000.0
        if deadline_monotonic_s is not None and time.monotonic() >= deadline_monotonic_s:
            return None
        mean_de00_f, chrom_f, color_f_ms = ChromaticAllocationStage.run(
            state=st_fast,
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
        verify_standalone_sculpted_state_invariants(
            st_fast,
            prep,
            forbidden_support_boxes=forbidden_support_boxes,
            ground_plane_z_ldu=ground_plane_z_ldu,
        )
        return TraceCertifyStage.finalize(
            state=st_fast,
            occ6=prep.occ6,
            solid_occ6_ref=prep.solid_occ6_ref,
            adaptive_target=adaptive_target,
            verts_ldu=verts_ldu,
            vertex_colors_rgb=vertex_colors_rgb,
            view_priority=None,
            ret_res=prep.ret_res,
            mean_de00=mean_de00_f,
            chrom_res=chrom_f,
            landmarks_attached_count=0,
            plate_to_tile_values=set(PLATE_TO_SMOOTH_TILE_MAP.values()),
            build_fine_tf_fn=_build_fine_target_field_from_occ6,
            count_exposed_studs_fn=count_exposed_upward_studs,
            diagnostics_cls=diagnostics_cls,
            target_prep_ms=prep.target_prep_ms,
            retrieval_ms=prep.retrieval_ms,
            sculpt_synthesis_ms=sculpt_fast_ms,
            color_allocation_ms=color_f_ms,
            license_filter=license_filter,
            allowed_part_ids=allowed_part_ids,
            colored_inventory=colored_inventory,
            vertex_region_labels=vertex_region_labels,
            open_color_palette_mode=open_color_palette_mode,
            symmetry_plane_axis=symmetry_plane_axis,
            symmetry_plane_coord_ldu=symmetry_plane_coord_ldu,
        )
