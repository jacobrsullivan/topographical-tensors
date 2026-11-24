"""
Stage 5a & 5b: Option 3B zero-cost terminal smooth-tile swap and portfolio jumper-phase /
single-leg horizontal-slab refinement (`Phase 15`).
"""

from __future__ import annotations

from collections import deque
import math
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import numpy as np

from topo_brick.evaluation.static_equilibrium import FoundationPolicy
from topo_brick.interface.brick_catalog import (
    EXTENDED_BRICK_CATALOG,
    ConnectorType,
    PortPolarity,
    get_brick_primitive,
)
from topo_brick.interface.catalog_retrieval_index import CATALOG_RETRIEVAL_INDEX
from topo_brick.interface.ldu_frames import C4_Z_ROT_IDS, LDUBox, LDUFrame
from topo_brick.interface.port_manifold import AssemblyState, _get_anchor_templates
from topo_brick.kernel.insertion_corridors import check_monotone_insertion_corridor
from topo_brick.kernel.open_catalog_blind_portfolio import (
    _apply_terminal_tile_swap,
    refine_open_catalog_blind_portfolio,
)
from topo_brick.kernel.sculpted_primitives import (
    BRICK_TO_PLATE_AND_TILE_MAP,
    PLATE_TO_SMOOTH_TILE_MAP,
    _box_hits_forbidden,
)
from topo_brick.kernel.sculpted_stages.commit_session import SculptedLiveSession


class SmoothTileAndSlabPortfolioStage:
    """Apply terminal smooth-tile swaps and evaluate portfolio phase/slab candidates."""

    @staticmethod
    def apply_smooth_tile_swap(sess: SculptedLiveSession) -> None:
        kit_set = sess.prep.kit_set
        smooth_mode = sess.prep.smooth_mode
        license_filter = sess.license_filter

        # 5. Option 3B Zero-Cost Terminal Smooth-Tile Swap (`0.000 IoU` delta, `0` extra parts!)
        matched_now = sess.state.matched_port_pairs
        for iid in sorted(sess.state.placed_parts.keys()):
            pl = sess.state.placed_parts[iid]
            tile_pid = PLATE_TO_SMOOTH_TILE_MAP.get(pl.part_id)
            if tile_pid is None:
                continue
            # In smooth_studless mode, allow all standard smooth tiles if license-valid; otherwise require in kit_set
            if not smooth_mode and tile_pid not in kit_set:
                continue
            if license_filter == "cc_by_4_0_only" and tile_pid not in EXTENDED_BRICK_CATALOG:
                ent_t = CATALOG_RETRIEVAL_INDEX.get_entry(tile_pid)
                if ent_t is None or ent_t.license_tier != "cc_by_4_0":
                    continue
            can_tl = CATALOG_RETRIEVAL_INDEX.resolve_canonical_mold_id(tile_pid) or tile_pid
            if sess.rem_inv is not None and sess.rem_inv.get(can_tl, 0) <= 0:
                continue
            prim_p = get_brick_primitive(pl.part_id)
            # Only swap upward-oriented horizontal plates (`zmax - zmin == 8`) with ZERO matched upward studs
            if (max(b.zmax for b in pl.world_boxes) - min(b.zmin for b in pl.world_boxes)) != 8:
                continue
            has_matched_top = any(
                (iid, p.port_id) in matched_now
                for p in prim_p.ports
                if p.polarity == PortPolarity.STUD
            )
            if has_matched_top:
                continue
            # Align `tile_pid` across `C4_Z_ROT_IDS` so LDraw-centered tiles (`e.g. 69729`) match `pl.world_boxes`
            prim_t = get_brick_primitive(tile_pid)
            if len(prim_t.hull_boxes) != 1 or len(pl.world_boxes) != 1:
                continue
            pb_w = pl.world_boxes[0]
            matched_tile_pose: Optional[LDUFrame] = None
            for rid_t in C4_Z_ROT_IDS:
                tb_l = prim_t.hull_boxes[0].transform(LDUFrame(rid_t, (0, 0, 0)))
                if (tb_l.xmax - tb_l.xmin) == (pb_w.xmax - pb_w.xmin) and (tb_l.ymax - tb_l.ymin) == (pb_w.ymax - pb_w.ymin):
                    cand_tp = LDUFrame(rid_t, (pb_w.xmin - tb_l.xmin, pb_w.ymin - tb_l.ymin, pb_w.zmin - tb_l.zmin))
                    if tuple(b.transform(cand_tp) for b in prim_t.hull_boxes) == pl.world_boxes:
                        matched_tile_pose = cand_tp
                        break
            if matched_tile_pose is None:
                continue
            try:
                sess.state.reorient_part(iid, matched_tile_pose, new_part_id=tile_pid)
                sess.plumb_hop_cache.pop(iid, None)
                if sess.rem_inv is not None:
                    can_pl = CATALOG_RETRIEVAL_INDEX.resolve_canonical_mold_id(pl.part_id) or pl.part_id
                    sess.rem_inv[can_pl] = sess.rem_inv.get(can_pl, 0) + 1
                    sess.rem_inv[can_tl] = sess.rem_inv.get(can_tl, 0) - 1
            except ValueError:
                continue


    @staticmethod
    def run_portfolio_and_capping_refinement(
        sess: SculptedLiveSession,
        *,
        gx6: int,
        gy6: int,
        gz6: int,
        recursive_realize_fn: Callable[..., Any],
    ) -> None:
        kernel = sess.kernel
        coarse_target_field = sess.coarse_target_field
        adaptive_target = sess.adaptive_target
        allowed_part_ids = sess.allowed_part_ids
        part_retrieval = sess.part_retrieval
        retrieval_top_k = sess.retrieval_top_k
        license_filter = sess.license_filter
        semantic_style_hint = sess.semantic_style_hint
        inventory_counts = sess.inventory_counts
        support_policy = sess.support_policy
        foundation_policy = sess.foundation_policy
        forbidden_support_boxes = sess.forbidden_support_boxes
        hollow_shell_policy = sess.hollow_shell_policy
        view_priority = sess.view_priority
        suppress_studless_capping_boxes = sess.suppress_studless_capping_boxes
        deadline_monotonic_s = sess.deadline_monotonic_s
        ground_plane_z_ldu = sess.ground_plane_z_ldu
        _forced_plan = sess.forced_plan
        _eval_only = sess.eval_only
        verts_ldu = sess.verts_ldu
        is_megastructure_footprint = sess.is_megastructure_footprint

        occ6 = sess.prep.occ6
        ad_occ6 = sess.prep.ad_occ6
        n_plates_z = sess.prep.n_plates_z
        ret_res = sess.prep.ret_res
        kit_ids = sess.prep.kit_ids
        kit_set = sess.prep.kit_set
        eff_inv_map = sess.prep.eff_inv_map
        max_inv_snapshot = sess.prep.max_inv_snapshot
        smooth_mode = sess.prep.smooth_mode
        is_hollow_shell_active = sess.prep.is_hollow_shell_active

        best_ground_phase = sess.core_plan.best_ground_phase
        layer_phase = sess.core_plan.layer_phase
        phase_grids = sess.core_plan.phase_grids
        is_thin_regime = sess.core_plan.is_thin_regime
        is_legged_frame = sess.core_plan.is_legged_frame
        core_cell_thresh = sess.core_plan.core_cell_thresh
        _fast_est_plan_occ6_iou = sess.core_plan.fast_est_plan_occ6_iou

        mesh_sub_total = sess.mesh_sub_total
        _eval_boxes_delta_iou = sess.eval_boxes_delta_iou
        _deadline_expired = sess.deadline_expired
        _try_commit_part = sess.try_commit_part
        realize_open_catalog_sculpted = recursive_realize_fn
        assert _eval_boxes_delta_iou is not None
        assert _deadline_expired is not None
        assert _try_commit_part is not None

        def _exact_certified_occ6_iou(
            st_chk: AssemblyState,
        ) -> Tuple[float, int, int, np.ndarray]:
            if not st_chk.placed_parts or not any(p.is_grounded for p in st_chk.placed_parts.values()):
                return -1.0, 0, 0, sess.occ6_live
            # Verify single connected component across matched port pairs
            if len(st_chk.placed_parts) > 1:
                adj_c: Dict[int, Set[int]] = {k_c: set() for k_c in st_chk.placed_parts}
                for (ia_c, _), (ib_c, _) in st_chk.matched_port_pairs.items():
                    if ia_c != ib_c and ia_c in adj_c and ib_c in adj_c:
                        adj_c[ia_c].add(ib_c)
                        adj_c[ib_c].add(ia_c)
                roots_c = [k_c for k_c, p_c in st_chk.placed_parts.items() if p_c.is_grounded]
                vis_c: Set[int] = set(roots_c)
                dq_c = deque(roots_c)
                while dq_c:
                    cur_c = dq_c.popleft()
                    for nb_c in adj_c[cur_c]:
                        if nb_c not in vis_c:
                            vis_c.add(nb_c)
                            dq_c.append(nb_c)
                if len(vis_c) != len(st_chk.placed_parts):
                    return -1.0, 0, 0, sess.occ6_live
            m_chk = np.zeros_like(occ6)
            for pl_chk in st_chk.placed_parts.values():
                for b_chk in pl_chk.world_boxes:
                    if _box_hits_forbidden(b_chk, forbidden_support_boxes):
                        return -1.0, 0, 0, sess.occ6_live
                    ux0_c = int(math.floor(b_chk.xmin / 5.0))
                    ux1_c = int(math.ceil(b_chk.xmax / 5.0))
                    uy0_c = int(math.floor(b_chk.ymin / 5.0))
                    uy1_c = int(math.ceil(b_chk.ymax / 5.0))
                    uz0_c = int(math.floor(b_chk.zmin / 6.0))
                    uz1_c = int(math.ceil(b_chk.zmax / 6.0))
                    if (
                        ux0_c < 0
                        or uy0_c < 0
                        or uz0_c < 0
                        or ux1_c > gx6
                        or uy1_c > gy6
                        or uz1_c > gz6
                    ):
                        return -1.0, 0, 0, sess.occ6_live
                    if ux1_c > ux0_c and uy1_c > uy0_c and uz1_c > uz0_c:
                        m_chk[ux0_c:ux1_c, uy0_c:uy1_c, uz0_c:uz1_c] = True
            if not kernel.validator.evaluate(st_chk, foundation_policy=foundation_policy).is_stable:
                return -1.0, 0, 0, sess.occ6_live
            in_chk = int(np.count_nonzero(m_chk & occ6))
            ov_chk = int(np.count_nonzero(m_chk & (~occ6)))
            return float(in_chk) / float(max(1, mesh_sub_total + ov_chk)), in_chk, ov_chk, m_chk

        # Evaluate upper-setback `15573` jumper phase-shift plans & compact thin-frame phase/threshold
        # candidates against exact `5x5x6 LDU` `occ6` IoU + 6-DoF static stability (`monotonically >= Plan 0`!)
        if (
            _forced_plan is None
            and inventory_counts is None
            and foundation_policy == FoundationPolicy.CLUTCHED_BASEPLATE
            and mesh_sub_total > 0
        ):

            best_plan_iou, inter_0, over_0, mask_0 = _exact_certified_occ6_iou(sess.state)
            if best_plan_iou >= 0.0:
                sess.inter_live, sess.over_live, sess.occ6_live = inter_0, over_0, mask_0

            cand_forced_plans: List[
                Tuple[List[Tuple[int, int]], Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]], int]
            ] = []

            if "15573" in kit_set and not forbidden_support_boxes and n_plates_z >= (5 if is_thin_regime else 7) and mesh_sub_total < 15000:
                def _slice_align_sc(ph_chk: Tuple[int, int], p0_c: int, p1_c: int) -> float:
                    cg_s = phase_grids[ph_chk][:, :, p0_c:p1_c]
                    if is_thin_regime:
                        sol_s = cg_s >= 16
                        strad_s = (cg_s >= 6) & (cg_s < 16)
                        return float(np.sum(2 * cg_s[sol_s].astype(np.int32) - 24)) - 8.0 * float(
                            np.count_nonzero(strad_s)
                        )
                    sol_s = cg_s >= 38
                    strad_s = (cg_s >= 16) & (cg_s < 38)
                    return float(np.sum(2 * cg_s[sol_s].astype(np.int32) - 64)) - 12.0 * float(
                        np.count_nonzero(strad_s)
                    )

                jb_scored: List[
                    Tuple[
                        float,
                        Tuple[int, int],
                        List[Tuple[int, int]],
                        Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]],
                    ]
                ] = []
                for p_jb_c in range(1, n_plates_z - 2):
                    for ph_up_c in ((10, 0), (0, 10), (10, 10)):
                        if ph_up_c == best_ground_phase:
                            continue
                        two_ax = (ph_up_c[0] != best_ground_phase[0]) and (ph_up_c[1] != best_ground_phase[1])
                        p_up0_c = p_jb_c + (2 if two_ax else 1)
                        if p_up0_c + 2 > n_plates_z:
                            continue
                        p_imm_end = min(n_plates_z, p_up0_c + 3)
                        imm_gain_c = _slice_align_sc(ph_up_c, p_up0_c, p_imm_end) - _slice_align_sc(
                            best_ground_phase, p_up0_c, p_imm_end
                        )
                        if imm_gain_c <= (8.0 if is_thin_regime else 20.0):
                            continue
                        up_gain_c = _slice_align_sc(ph_up_c, p_up0_c, n_plates_z) - _slice_align_sc(
                            best_ground_phase, p_up0_c, n_plates_z
                        )
                        if up_gain_c <= (24.0 if is_thin_regime else 80.0):
                            continue
                        jb_loss_c = (0.10 if is_thin_regime else 0.15) * max(
                            0.0, _slice_align_sc(best_ground_phase, p_jb_c, p_up0_c)
                        ) + (8.0 if is_thin_regime else 30.0) * (2.0 if two_ax else 1.0)
                        # Penalize placing a jumper bridge when the layer above the jumper is still wider on `best_ground_phase`
                        g_any_jb = int(
                            np.count_nonzero(
                                phase_grids[best_ground_phase][:, :, p_up0_c] >= (16 if is_thin_regime else 38)
                            )
                        )
                        u_sol_up = int(
                            np.count_nonzero(
                                phase_grids[ph_up_c][:, :, p_up0_c] >= (16 if is_thin_regime else 38)
                            )
                        )
                        if not is_thin_regime and g_any_jb >= u_sol_up + 4:
                            jb_loss_c += 35.0 * float(g_any_jb - u_sol_up)
                        gain_f_c = up_gain_c - jb_loss_c
                        if gain_f_c <= (10.0 if is_thin_regime else 30.0):
                            continue
                        l_ph_cand = [best_ground_phase] * p_up0_c + [ph_up_c] * (n_plates_z - p_up0_c)
                        if two_ax:
                            ph_mid_c = (ph_up_c[0], best_ground_phase[1])
                            jb_cand = {
                                p_jb_c: (best_ground_phase, ph_mid_c),
                                p_jb_c + 1: (ph_mid_c, ph_up_c),
                            }
                        else:
                            jb_cand = {p_jb_c: (best_ground_phase, ph_up_c)}
                        jb_scored.append((gain_f_c, ph_up_c, l_ph_cand, jb_cand))

                jb_scored.sort(key=lambda item: -item[0])
                if jb_scored:
                    base_est_jb = _fast_est_plan_occ6_iou(layer_phase, {}, core_cell_thresh)
                    jb_est_ranked: List[
                        Tuple[
                            float,
                            float,
                            Tuple[int, int],
                            List[Tuple[int, int]],
                            Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]],
                        ]
                    ] = []
                    seen_ph_est: Set[Tuple[int, int]] = set()
                    top_jb_pool: List[
                        Tuple[
                            float,
                            Tuple[int, int],
                            List[Tuple[int, int]],
                            Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]],
                        ]
                    ] = []
                    for g_val, ph_u_s, l_ph_s, jb_s in jb_scored:
                        if ph_u_s not in seen_ph_est:
                            seen_ph_est.add(ph_u_s)
                            top_jb_pool.append((g_val, ph_u_s, l_ph_s, jb_s))
                    for g_val, ph_u_s, l_ph_s, jb_s in jb_scored:
                        if len(top_jb_pool) >= 6:
                            break
                        if all(jb_s != ex[3] for ex in top_jb_pool):
                            top_jb_pool.append((g_val, ph_u_s, l_ph_s, jb_s))
                    for g_val, ph_u_s, l_ph_s, jb_s in top_jb_pool:
                        est_j = _fast_est_plan_occ6_iou(l_ph_s, jb_s, core_cell_thresh)
                        if est_j >= base_est_jb - (0.030 if is_thin_regime else 0.020):
                            jb_est_ranked.append((est_j, g_val, ph_u_s, l_ph_s, jb_s))
                    jb_est_ranked.sort(key=lambda item: (-item[0], -item[1]))
                    max_jb_try = 2 if mesh_sub_total < 1200 else (1 if mesh_sub_total < 6000 else 0)
                    for _, _, _, l_ph_s, jb_s in jb_est_ranked[:max_jb_try]:
                        cand_forced_plans.append((l_ph_s, jb_s, core_cell_thresh))

            if forbidden_support_boxes and not is_thin_regime and mesh_sub_total < 4500 and best_plan_iou < 0.905:
                cand_forced_plans.append(([best_ground_phase] * n_plates_z, {}, 46))

            if not is_thin_regime and not forbidden_support_boxes and 2800 <= mesh_sub_total < 25000 and best_plan_iou < 0.9245:
                thr_hi = (
                    42
                    if (
                        6000 <= mesh_sub_total < 12000
                        and best_plan_iou < 0.916
                        and _fast_est_plan_occ6_iou([best_ground_phase] * n_plates_z, {}, 42)
                        >= _fast_est_plan_occ6_iou([best_ground_phase] * n_plates_z, {}, 38) - 0.004
                    )
                    else 38
                )
                cand_forced_plans.append(([best_ground_phase] * n_plates_z, {}, thr_hi))

            if (
                not is_thin_regime
                and not forbidden_support_boxes
                and 15000 <= mesh_sub_total < 26000
                and best_plan_iou < 0.931
                and best_ground_phase != (0, 10)
                and _fast_est_plan_occ6_iou([(0, 10)] * n_plates_z, {}, 28)
                >= _fast_est_plan_occ6_iou(layer_phase, {}, core_cell_thresh) - 0.015
            ):
                cand_forced_plans.append(([(0, 10)] * n_plates_z, {}, 28))

            if mesh_sub_total < 1600 and not forbidden_support_boxes:
                alt_small_specs: Tuple[Tuple[Tuple[int, int], int], ...] = (
                    (
                        ((0, 0), 8),
                        ((0, 10), 12),
                        ((10, 0), 14),
                        ((0, 10), 14),
                        ((0, 0), 18),
                    )
                    if is_legged_frame
                    else (
                        (
                            ((0, 0), 14),
                            ((0, 0), 24),
                            ((0, 10), 8),
                            ((10, 0), 8),
                            ((10, 0), 14),
                            ((0, 10), 14),
                            ((0, 0), 18),
                        )
                        if (is_thin_regime and mesh_sub_total < 900)
                        else (
                            ((10, 10), 28),
                            ((10, 0), 14),
                            ((0, 10), 14),
                            ((0, 0), 18),
                        )
                    )
                )
                max_small_try = 2 if is_legged_frame else 1
                ranked_alt_specs = sorted(
                    (
                        (_fast_est_plan_occ6_iou([ph_alt] * n_plates_z, {}, thr_alt), ph_alt, thr_alt)
                        for ph_alt, thr_alt in alt_small_specs
                        if not (ph_alt == best_ground_phase and thr_alt == core_cell_thresh)
                    ),
                    key=lambda item: -item[0],
                )
                for _, ph_alt, thr_alt in ranked_alt_specs[:max_small_try]:
                    cand_forced_plans.append(([ph_alt] * n_plates_z, {}, thr_alt))

            # Fast `< 4 ms` Single-Leg Horizontal-Slab Portfolio Candidate for elevated thin-tabletop meshes
            won_single_leg_slab = False
            if (
                is_thin_regime
                and not forbidden_support_boxes
                and mesh_sub_total < 1200
                and best_plan_iou < 0.55
                and "3005" in kit_set
                and "3024" in kit_set
                and "3023" in kit_set
            ):
                plates_rect_sl = [
                    p_r
                    for p_r in ("3795", "3020", "3021", "3022", "3666", "3710", "3623", "3023", "3024")
                    if p_r in kit_set
                ]
                best_p_slab = -1
                best_slab_tp = 0
                for p_sl in range(4, n_plates_z):
                    uz0_sl = (8 * p_sl) // 6
                    uz1_sl = min(gz6, (8 * (p_sl + 1) + 5) // 6)
                    tp_sl = int(np.count_nonzero(occ6[:, :, uz0_sl:uz1_sl]))
                    if tp_sl > best_slab_tp and tp_sl >= int(0.42 * mesh_sub_total):
                        best_slab_tp = tp_sl
                        best_p_slab = p_sl
                if best_p_slab >= 4 and plates_rect_sl:
                    uz0_sl = (8 * best_p_slab) // 6
                    uz1_sl = min(gz6, (8 * (best_p_slab + 1) + 5) // 6)
                    z_sl = 8 * best_p_slab
                    for ox_sl, oy_sl in ((0, 0), (10, 0), (0, 10), (10, 10)):
                        slab_cells: Set[Tuple[int, int]] = set()
                        for ix_sl in range((gx6 * 5 - ox_sl) // 20):
                            for iy_sl in range((gy6 * 5 - oy_sl) // 20):
                                cx_sl = (ix_sl * 20 + ox_sl) // 5
                                cy_sl = (iy_sl * 20 + oy_sl) // 5
                                if int(np.count_nonzero(occ6[cx_sl : cx_sl + 4, cy_sl : cy_sl + 4, uz0_sl:uz1_sl])) >= 11:
                                    slab_cells.add((ix_sl, iy_sl))
                        if len(slab_cells) < 6:
                            continue
                        xs_sl = [c_s[0] for c_s in slab_cells]
                        ys_sl = [c_s[1] for c_s in slab_cells]
                        leg_cands_sl: List[Tuple[int, int, int]] = []
                        for lx_s, ly_s in slab_cells:
                            cx_sl = (lx_s * 20 + ox_sl) // 5
                            cy_sl = (ly_s * 20 + oy_sl) // 5
                            if int(np.count_nonzero(occ6[cx_sl : cx_sl + 4, cy_sl : cy_sl + 4, 0])) > 0:
                                leg_cands_sl.append(
                                    (int(np.count_nonzero(occ6[cx_sl : cx_sl + 4, cy_sl : cy_sl + 4, :uz0_sl])), lx_s, ly_s)
                                )
                        leg_cands_sl.sort(reverse=True)
                        for _, lx_s, ly_s in leg_cands_sl[:2]:
                            st_sl = AssemblyState()
                            px_sl = lx_s * 20 + ox_sl + 10
                            py_sl = ly_s * 20 + oy_sl + 10
                            zc_sl = 0
                            try:
                                while zc_sl + 24 <= z_sl:
                                    st_sl.place_root_part("3005", LDUFrame(0, (px_sl, py_sl, zc_sl)), is_grounded=(zc_sl == 0))
                                    zc_sl += 24
                                while zc_sl + 8 <= z_sl:
                                    st_sl.place_root_part("3024", LDUFrame(0, (px_sl, py_sl, zc_sl)), is_grounded=(zc_sl == 0))
                                    zc_sl += 8
                                rem_sl = set(slab_cells)
                                cov_sl: Set[Tuple[int, int]] = set()
                                first_p_sl = None
                                for pid_r in plates_rect_sl:
                                    prim_r = get_brick_primitive(pid_r)
                                    for rid_r in (0, 9):
                                        b0_r = prim_r.hull_boxes[0].transform(LDUFrame(rid_r, (0, 0, 0)))
                                        sx_r = (b0_r.xmax - b0_r.xmin) // 20
                                        sy_r = (b0_r.ymax - b0_r.ymin) // 20
                                        for ix0_r in range(min(xs_sl), max(xs_sl) - sx_r + 2):
                                            for iy0_r in range(min(ys_sl), max(ys_sl) - sy_r + 2):
                                                cs_r = {(ix0_r + dx_r, iy0_r + dy_r) for dx_r in range(sx_r) for dy_r in range(sy_r)}
                                                if (lx_s, ly_s) in cs_r and cs_r <= rem_sl:
                                                    if first_p_sl is None or len(cs_r) > first_p_sl[0]:
                                                        first_p_sl = (
                                                            len(cs_r),
                                                            pid_r,
                                                            rid_r,
                                                            ix0_r * 20 + ox_sl - b0_r.xmin,
                                                            iy0_r * 20 + oy_sl - b0_r.ymin,
                                                            cs_r,
                                                        )
                                if first_p_sl is None:
                                    continue
                                _, pid_r0, rid_r0, wx_r0, wy_r0, cs_r0 = first_p_sl
                                st_sl.place_root_part(pid_r0, LDUFrame(rid_r0, (wx_r0, wy_r0, z_sl)), is_grounded=False)
                                rem_sl -= cs_r0
                                cov_sl |= cs_r0
                                for _ in range(5):
                                    if not rem_sl:
                                        break
                                    best_ext_sl = None
                                    for pid_e in plates_rect_sl:
                                        prim_e = get_brick_primitive(pid_e)
                                        for rid_e in (0, 9):
                                            b0_e = prim_e.hull_boxes[0].transform(LDUFrame(rid_e, (0, 0, 0)))
                                            sx_e = (b0_e.xmax - b0_e.xmin) // 20
                                            sy_e = (b0_e.ymax - b0_e.ymin) // 20
                                            for ix0_e in range(min(xs_sl), max(xs_sl) - sx_e + 2):
                                                for iy0_e in range(min(ys_sl), max(ys_sl) - sy_e + 2):
                                                    cs_e = {(ix0_e + dx_e, iy0_e + dy_e) for dx_e in range(sx_e) for dy_e in range(sy_e)}
                                                    if cs_e <= rem_sl:
                                                        tie_pose_sl = None
                                                        best_tie_tp = -1
                                                        for ce_s in cs_e:
                                                            for ct_s in cov_sl:
                                                                if abs(ce_s[0] - ct_s[0]) + abs(ce_s[1] - ct_s[1]) == 1 and ct_s != (lx_s, ly_s):
                                                                    rid_u = 9 if abs(ce_s[0] - ct_s[0]) == 1 else 0
                                                                    b0_u = get_brick_primitive("3023").hull_boxes[0].transform(LDUFrame(rid_u, (0, 0, 0)))
                                                                    ixu = min(ce_s[0], ct_s[0])
                                                                    iyu = min(ce_s[1], ct_s[1])
                                                                    wbu = LDUBox(
                                                                        ixu * 20 + ox_sl,
                                                                        iyu * 20 + oy_sl,
                                                                        z_sl - 8,
                                                                        (max(ce_s[0], ct_s[0]) + 1) * 20 + ox_sl,
                                                                        (max(ce_s[1], ct_s[1]) + 1) * 20 + oy_sl,
                                                                        z_sl,
                                                                    )
                                                                    if not st_sl.check_boxes_collision((wbu,)):
                                                                        ttp = int(
                                                                            np.count_nonzero(
                                                                                occ6[
                                                                                    wbu.xmin // 5 : (wbu.xmax + 4) // 5,
                                                                                    wbu.ymin // 5 : (wbu.ymax + 4) // 5,
                                                                                    wbu.zmin // 6 : (wbu.zmax + 5) // 6,
                                                                                ]
                                                                            )
                                                                        )
                                                                        if ttp > best_tie_tp:
                                                                            best_tie_tp = ttp
                                                                            tie_pose_sl = LDUFrame(
                                                                                rid_u,
                                                                                (ixu * 20 + ox_sl - b0_u.xmin, iyu * 20 + oy_sl - b0_u.ymin, z_sl - 8),
                                                                            )
                                                        if tie_pose_sl is not None:
                                                            sc_e = (len(cs_e), best_tie_tp)
                                                            if best_ext_sl is None or sc_e > best_ext_sl[0]:
                                                                best_ext_sl = (
                                                                    sc_e,
                                                                    pid_e,
                                                                    rid_e,
                                                                    ix0_e * 20 + ox_sl - b0_e.xmin,
                                                                    iy0_e * 20 + oy_sl - b0_e.ymin,
                                                                    cs_e,
                                                                    tie_pose_sl,
                                                                )
                                    if best_ext_sl is None:
                                        break
                                    _, pid_e0, rid_e0, wx_e0, wy_e0, cs_e0, tie_pose_0 = best_ext_sl
                                    st_sl.place_root_part("3023", tie_pose_0, is_grounded=False)
                                    st_sl.place_root_part(pid_e0, LDUFrame(rid_e0, (wx_e0, wy_e0, z_sl)), is_grounded=False)
                                    rem_sl -= cs_e0
                                    cov_sl |= cs_e0
                            except ValueError:
                                continue
                            iou_sl, in_sl, ov_sl, mask_sl = _exact_certified_occ6_iou(st_sl)
                            if iou_sl > best_plan_iou + 1.0e-4:
                                best_plan_iou = iou_sl
                                sess.state, sess.inter_live, sess.over_live, sess.occ6_live = st_sl, in_sl, ov_sl, mask_sl
                                sess.base_witness = None
                                sess.base_lp_eta = 0.0
                                sess.plumb_deltas.clear()
                                sess.plumb_hop_cache.clear()
                                won_single_leg_slab = True

            ad_origin = getattr(adaptive_target, "origin_ldu", (0, 0, 0))
            if view_priority is not None:
                from topo_brick.kernel.view_priority import evaluate_camera_reprojection_fidelity

                best_view_sil = evaluate_camera_reprojection_fidelity(
                    sess.state, ad_occ6, view_priority, origin_ldu=ad_origin
                ).silhouette_iou_2d
            else:
                best_view_sil = -1.0
            for f_plan_try in (() if (won_single_leg_slab or is_hollow_shell_active or is_megastructure_footprint or _deadline_expired()) else cand_forced_plans):
                if _deadline_expired():
                    break
                st_try, _, _ = realize_open_catalog_sculpted(
                    kernel=kernel,
                    coarse_target_field=coarse_target_field,
                    adaptive_target=adaptive_target,
                    allowed_part_ids=allowed_part_ids,
                    part_retrieval=part_retrieval,
                    retrieval_top_k=retrieval_top_k,
                    license_filter=license_filter,
                    semantic_style_hint=semantic_style_hint,
                    vertex_colors_rgb=None,
                    verts_ldu=verts_ldu,
                    inventory_counts=max_inv_snapshot,
                    support_policy=support_policy,
                    foundation_policy=foundation_policy,
                    forbidden_support_boxes=forbidden_support_boxes,
                    hollow_shell_policy=hollow_shell_policy,
                    view_priority=view_priority,
                    deadline_monotonic_s=deadline_monotonic_s,
                    ground_plane_z_ldu=ground_plane_z_ldu,
                    _forced_plan=f_plan_try,
                    _pre_ret=ret_res,
                    _eval_only=True,
                    _max_waves=(2 if mesh_sub_total < 2200 else 1),
                )
                iou_try, in_try, ov_try, mask_try = _exact_certified_occ6_iou(st_try)
                if view_priority is not None:
                    sil_try = evaluate_camera_reprojection_fidelity(
                        st_try, ad_occ6, view_priority, origin_ldu=ad_origin
                    ).silhouette_iou_2d
                    better_try = (sil_try + 0.15 * iou_try) > (best_view_sil + 0.15 * best_plan_iou + 1.0e-4)
                else:
                    sil_try = -1.0
                    better_try = iou_try > best_plan_iou + 1.0e-4
                if better_try:
                    best_plan_iou = iou_try
                    best_view_sil = sil_try
                    sess.state, sess.inter_live, sess.over_live, sess.occ6_live = st_try, in_try, ov_try, mask_try
                    if max_inv_snapshot is not None:
                        sess.rem_inv = dict(max_inv_snapshot)
                        for pf_u in sess.state.placed_parts.values():
                            sess.rem_inv[pf_u.part_id] = max(0, sess.rem_inv.get(pf_u.part_id, 0) - 1)
                    sess.base_witness = None
                    sess.base_lp_eta = 0.0
                    sess.plumb_deltas.clear()
                    sess.plumb_hop_cache.clear()

        if not _eval_only and (sess.rem_inv is None or max_inv_snapshot is not None) and not is_hollow_shell_active and not is_megastructure_footprint and not _deadline_expired():
            st_port = refine_open_catalog_blind_portfolio(
                kernel=kernel,
                state=sess.state,
                occ6=ad_occ6,
                kit_ids=tuple(kit_ids),
                forbidden_support_boxes=forbidden_support_boxes,
                foundation_policy=foundation_policy,
                max_inventory_counts=max_inv_snapshot,
            )
            if st_port is not sess.state:
                accept_port = True
                if view_priority is not None:
                    from topo_brick.kernel.view_priority import evaluate_camera_reprojection_fidelity

                    ad_origin_pf = getattr(adaptive_target, "origin_ldu", (0, 0, 0))
                    s_cur_2d = evaluate_camera_reprojection_fidelity(
                        sess.state, ad_occ6, view_priority, origin_ldu=ad_origin_pf
                    ).silhouette_iou_2d
                    s_pf_2d = evaluate_camera_reprojection_fidelity(
                        st_port, ad_occ6, view_priority, origin_ldu=ad_origin_pf
                    ).silhouette_iou_2d
                    accept_port = s_pf_2d > s_cur_2d + 1.0e-4
                if accept_port:
                    iou_pf, in_pf, ov_pf, mask_pf = _exact_certified_occ6_iou(st_port)
                    if iou_pf >= 0.0:
                        sess.state, sess.inter_live, sess.over_live, sess.occ6_live = st_port, in_pf, ov_pf, mask_pf
                        if max_inv_snapshot is not None:
                            sess.rem_inv = dict(max_inv_snapshot)
                            for pf_u in sess.state.placed_parts.values():
                                sess.rem_inv[pf_u.part_id] = max(0, sess.rem_inv.get(pf_u.part_id, 0) - 1)
                        sess.base_witness = None
                        sess.base_lp_eta = 0.0
                        sess.plumb_deltas.clear()
                        sess.plumb_hop_cache.clear()
            if smooth_mode:
                _apply_terminal_tile_swap(
                    sess.state, kit_set, suppress_boxes=suppress_studless_capping_boxes, rem_inv=sess.rem_inv
                )
                sess.plumb_hop_cache.clear()
            elif sess.rem_inv is None:
                present_molds = {p.part_id for p in sess.state.placed_parts.values()}
                matched_now_t = sess.state.matched_port_pairs
                for iid_t in sorted(sess.state.placed_parts.keys()):
                    pl_t = sess.state.placed_parts[iid_t]
                    tl_cand = PLATE_TO_SMOOTH_TILE_MAP.get(pl_t.part_id)
                    if tl_cand is None or tl_cand not in kit_set or tl_cand in present_molds or len(pl_t.world_boxes) != 1:
                        continue
                    prim_pt = get_brick_primitive(pl_t.part_id)
                    if any((iid_t, pt.port_id) in matched_now_t for pt in prim_pt.ports if pt.polarity == PortPolarity.STUD):
                        continue
                    _apply_terminal_tile_swap(
                        sess.state, {tl_cand}, target_iids=(iid_t,), suppress_boxes=suppress_studless_capping_boxes
                    )
                    if sess.state.placed_parts[iid_t].part_id == tl_cand:
                        present_molds.add(tl_cand)

        # (Step 5b) Zero-IoU-delta inline decomposition of completely top-exposed, 100%-supported `24 LDU`
        #           rectangular bricks into `(8 LDU plate + 8 LDU plate + 8 LDU smooth tile)` of identical footprint
        #           (all eligible roof bricks in `smooth_studless`; at most 1 brick per unused tile mold in default mode).
        if not _eval_only and sess.rem_inv is None and (smooth_mode or len(sess.state.placed_parts) <= 90):
            matched_cur = sess.state.matched_port_pairs
            present_molds_b = {p.part_id for p in sess.state.placed_parts.values()}
            decomp_targets: Dict[int, Tuple[List[Tuple[str, LDUFrame]], bool]] = {}
            b_3024_loc = get_brick_primitive("3024").hull_boxes[0]
            b_3070b_loc = get_brick_primitive("3070b").hull_boxes[0]
            for iid_b in sorted(sess.state.placed_parts.keys()):
                pl_b = sess.state.placed_parts[iid_b]
                pt_pair = BRICK_TO_PLATE_AND_TILE_MAP.get(pl_b.part_id)
                if pt_pair is None or len(pl_b.world_boxes) != 1:
                    continue
                wb_b = pl_b.world_boxes[0]
                if (wb_b.zmax - wb_b.zmin) != 24:
                    continue
                if suppress_studless_capping_boxes and any(wb_b.intersects(sb) for sb in suppress_studless_capping_boxes):
                    continue
                pl_pid, tl_pid = pt_pair
                if not smooth_mode:
                    if pl_pid not in kit_set or tl_pid not in kit_set or tl_pid in present_molds_b:
                        continue
                if license_filter == "cc_by_4_0_only":
                    ok_lic = True
                    for chk_pid in (pl_pid, tl_pid, "3024", "3070b"):
                        if chk_pid not in EXTENDED_BRICK_CATALOG:
                            ent_chk = CATALOG_RETRIEVAL_INDEX.get_entry(chk_pid)
                            if ent_chk is None or ent_chk.license_tier != "cc_by_4_0":
                                ok_lic = False
                    if not ok_lic:
                        continue
                prim_bk = get_brick_primitive(pl_b.part_id)
                up_stud_ports = [
                    p_bk
                    for p_bk in prim_bk.ports
                    if p_bk.polarity == PortPolarity.STUD and pl_b.pose.compose(p_bk.local_frame).z_axis == (0, 0, 1)
                ]
                if not up_stud_ports:
                    continue
                n_mated_up = sum(1 for p_bk in up_stud_ports if (iid_b, p_bk.port_id) in matched_cur)
                if not smooth_mode and n_mated_up > 0:
                    continue
                if smooth_mode and n_mated_up >= len(up_stud_ports):
                    continue
                if not pl_b.is_grounded:
                    n_anti = sum(1 for p_bk in prim_bk.ports if p_bk.polarity == PortPolarity.ANTI_STUD)
                    n_mated_anti = sum(
                        1
                        for p_bk in prim_bk.ports
                        if p_bk.polarity == PortPolarity.ANTI_STUD and (iid_b, p_bk.port_id) in matched_cur
                    )
                    if n_anti == 0 or n_mated_anti < n_anti:
                        continue
                # Match orientation `(0 vs 9)` of `pl_pid` and `tl_pid` so `world_boxes` cover `wb_b` bit-for-bit
                found_seq: Optional[List[Tuple[str, LDUFrame]]] = None
                for rid_pl in (0, 9):
                    b_pl0 = get_brick_primitive(pl_pid).hull_boxes[0].transform(LDUFrame(rid_pl, (0, 0, 0)))
                    if (b_pl0.xmax - b_pl0.xmin) == (wb_b.xmax - wb_b.xmin) and (b_pl0.ymax - b_pl0.ymin) == (
                        wb_b.ymax - wb_b.ymin
                    ):
                        f_p0 = LDUFrame(rid_pl, (wb_b.xmin - b_pl0.xmin, wb_b.ymin - b_pl0.ymin, wb_b.zmin - b_pl0.zmin))
                        f_p1 = LDUFrame(
                            rid_pl, (wb_b.xmin - b_pl0.xmin, wb_b.ymin - b_pl0.ymin, wb_b.zmin + 8 - b_pl0.zmin)
                        )
                        if n_mated_up == 0:
                            for rid_tl in (0, 9):
                                b_tl0 = get_brick_primitive(tl_pid).hull_boxes[0].transform(LDUFrame(rid_tl, (0, 0, 0)))
                                if (b_tl0.xmax - b_tl0.xmin) == (wb_b.xmax - wb_b.xmin) and (
                                    b_tl0.ymax - b_tl0.ymin
                                ) == (wb_b.ymax - wb_b.ymin):
                                    f_t2 = LDUFrame(
                                        rid_tl,
                                        (wb_b.xmin - b_tl0.xmin, wb_b.ymin - b_tl0.ymin, wb_b.zmin + 16 - b_tl0.zmin),
                                    )
                                    found_seq = [(pl_pid, f_p0), (pl_pid, f_p1), (tl_pid, f_t2)]
                                    break
                        else:
                            seq_p: List[Tuple[str, LDUFrame]] = [(pl_pid, f_p0), (pl_pid, f_p1)]
                            for p_bk in sorted(up_stud_ports, key=lambda ps: pl_b.pose.compose(ps.local_frame).pos):
                                sx_w, sy_w, _ = pl_b.pose.compose(p_bk.local_frame).pos
                                if (iid_b, p_bk.port_id) in matched_cur:
                                    seq_p.append(
                                        (
                                            "3024",
                                            LDUFrame(
                                                0,
                                                (
                                                    sx_w - 10 - b_3024_loc.xmin,
                                                    sy_w - 10 - b_3024_loc.ymin,
                                                    wb_b.zmin + 16 - b_3024_loc.zmin,
                                                ),
                                            ),
                                        )
                                    )
                                else:
                                    seq_p.append(
                                        (
                                            "3070b",
                                            LDUFrame(
                                                0,
                                                (
                                                    sx_w - 10 - b_3070b_loc.xmin,
                                                    sy_w - 10 - b_3070b_loc.ymin,
                                                    wb_b.zmin + 16 - b_3070b_loc.zmin,
                                                ),
                                            ),
                                        )
                                    )
                            found_seq = seq_p
                        break
                if found_seq is not None:
                    decomp_targets[iid_b] = (found_seq, pl_b.is_grounded)
                    present_molds_b.add(tl_pid)

            if decomp_targets:
                trial_sm = AssemblyState()
                replay_ok = True
                for iid_orig in sorted(sess.state.placed_parts.keys()):
                    if iid_orig in decomp_targets:
                        seq_d, is_gr_d = decomp_targets[iid_orig]
                        try:
                            for idx_sd, (pid_sd, f_sd) in enumerate(seq_d):
                                if not check_monotone_insertion_corridor(trial_sm, pid_sd, f_sd):
                                    replay_ok = False
                                    break
                                trial_sm.place_root_part(pid_sd, f_sd, is_grounded=(is_gr_d and idx_sd == 0))
                            if not replay_ok:
                                break
                        except ValueError:
                            replay_ok = False
                            break
                    else:
                        po = sess.state.placed_parts[iid_orig]
                        try:
                            if not check_monotone_insertion_corridor(trial_sm, po.part_id, po.pose):
                                replay_ok = False
                                break
                            trial_sm.place_root_part(po.part_id, po.pose, is_grounded=po.is_grounded)
                        except ValueError:
                            replay_ok = False
                            break
                if replay_ok and not _deadline_expired() and kernel.validator.evaluate(
                    trial_sm,
                    foundation_policy=foundation_policy,
                    ground_plane_z_ldu=ground_plane_z_ldu,
                    deadline_monotonic_s=deadline_monotonic_s,
                ).is_stable:
                    sess.state = trial_sm
                    sess.base_witness = None
                    sess.base_lp_eta = 0.0
                    sess.plumb_deltas.clear()
                    sess.plumb_hop_cache.clear()

        if smooth_mode:
            tile_cap_ids = [t for t in ("87079", "3068b", "2431", "63864", "3069b", "3070b") if (t in kit_set or eff_inv_map is None)]
            fr_sm = sess.state.get_active_frontier()
            exp_up = [
                ep
                for ep in fr_sm.ports
                if ep.status.is_exposed
                and ep.polarity == PortPolarity.STUD
                and ep.world_normal == (0, 0, 1)
                and ep.connector_type in (ConnectorType.STANDARD_STUD, ConnectorType.HOLLOW_STUD)
            ]
            iou_before_cap = float(sess.inter_live) / float(max(1, mesh_sub_total + sess.over_live))
            for ep in exp_up:
                ux, uy, uz = ep.world_frame.pos
                if suppress_studless_capping_boxes and any(
                    sb.xmin <= ux <= sb.xmax and sb.ymin <= uy <= sb.ymax and sb.zmin <= uz <= sb.zmax
                    for sb in suppress_studless_capping_boxes
                ):
                    continue
                # Skip if already matched in an earlier multi-stud tile placement
                if ep.global_port_id in sess.state.matched_port_pairs:
                    continue
                for t_pid in tile_cap_ids:
                    placed_t = False
                    for tmpl in _get_anchor_templates(ep.world_frame.rot_id, ep.polarity, ep.connector_type, t_pid):
                        rid_t, dx_t, dy_t, dz_t, rboxes_t, rports_t, _, _ = tmpl
                        pos_t = (ux + dx_t, uy + dy_t, uz + dz_t)
                        wbs_t = tuple(
                            LDUBox(b[0] + ux, b[1] + uy, b[2] + uz, b[3] + ux, b[4] + uy, b[5] + uz)
                            for b in rboxes_t
                        )
                        if suppress_studless_capping_boxes and any(wb_c.intersects(sb) for wb_c in wbs_t for sb in suppress_studless_capping_boxes):
                            continue
                        if sess.state.check_boxes_collision(wbs_t):
                            continue
                        # Ensure ALL anti-studs of `t_pid` mate with exposed upward studs (so it doesn't overhang in midair!)
                        all_anti_mated = True
                        for _, _, apx, apy, apz, apol, _ in rports_t:
                            if apol == PortPolarity.ANTI_STUD:
                                w_apos = (ux + apx, uy + apy, uz + apz)
                                if not any(
                                    ep2.world_frame.pos == w_apos and ep2.global_port_id not in sess.state.matched_port_pairs
                                    for ep2 in exp_up
                                ):
                                    all_anti_mated = False
                                    break
                        if not all_anti_mated:
                            continue
                        d_iou_t, d_in_t, d_ov_t = _eval_boxes_delta_iou(wbs_t, enforce_min_precision=False)
                        proj_iou = float(sess.inter_live + d_in_t) / float(max(1, mesh_sub_total + sess.over_live + d_ov_t))
                        if proj_iou >= iou_before_cap - 0.0020:
                            if _try_commit_part(t_pid, LDUFrame(rid_t, pos_t), grounded_c=False, check_csg_strict=True):
                                placed_t = True
                                break
                    if placed_t:
                        break

        sess.resync_live_metrics()

