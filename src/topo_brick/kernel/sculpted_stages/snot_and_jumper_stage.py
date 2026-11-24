"""
Stage 3a-ii: Sculptural slope reservation, `4 LDU` SNOT perimeter brackets (`44728`/`99780`/`99207`),
`30414`/`11211`/`87087` SNOT wall seeding, jumper cavity fill, and `8 LDU`/`24 LDU` core tiling (`Phase 16`).
"""

from __future__ import annotations

import heapq
import math
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np

from topo_brick.evaluation.static_equilibrium import FoundationPolicy
from topo_brick.interface.brick_catalog import PortPolarity, get_brick_primitive
from topo_brick.interface.catalog_retrieval_index import CATALOG_RETRIEVAL_INDEX
from topo_brick.interface.ldu_frames import C4_Z_ROT_IDS, LDUBox, LDUFrame
from topo_brick.kernel.sculpted_primitives import (
    _box_hits_forbidden,
    _fast_slope_grid_templates,
    _tile_boolean_layer_with_rects,
)


class SnotAndJumperPlacementStage:
    """Reserve perimeter slopes/brackets/SNOT bricks and tile remaining core courses."""

    @staticmethod
    def reserve_and_tile(
        *,
        w_x: int,
        w_y: int,
        n_plates_z: int,
        occ6: np.ndarray,
        solid_occ6_padded: Optional[np.ndarray],
        gz6_nonempty_limit: int,
        adaptive_target: Any,
        kit_ids: Sequence[str],
        kit_set: Set[str],
        rem_inv: Optional[Dict[str, int]],
        has_thin_panel_kit: bool,
        is_thin_regime: bool,
        is_hollow_shell_active: bool,
        foundation_policy: FoundationPolicy,
        forbidden_support_boxes: Sequence[LDUBox],
        view_weight_grid_6: Optional[np.ndarray],
        deadline_monotonic_s: Optional[float],
        layer_phase: List[Tuple[int, int]],
        phase_grids: Dict[Tuple[int, int], np.ndarray],
        core_occ: np.ndarray,
        inv_reserved_occ: np.ndarray,
        pier_protected: np.ndarray,
        pillar_added_occ: np.ndarray,
        snot_spine_protected: np.ndarray,
        jumper_bridge_layers: Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]],
        jumper_top_stud_xy: Dict[int, np.ndarray],
        jumper_used_low: Dict[int, np.ndarray],
        jumper_low_occ: Dict[int, np.ndarray],
        brick_templates: List[Any],
        plate_templates: List[Any],
        tile_templates: List[Any],
        planned_by_z: Dict[int, List[Tuple[str, LDUFrame]]],
        _early_deadline_hit: Callable[[], bool],
        _support_mask_at_layer: Callable[[int], Optional[np.ndarray]],
    ) -> int:
        all_sl_cands: List[Any] = []
        # Reserve multi-height Open-Catalog slopes, curved caps, wedges/rounds & thin structural panels
        # directly replacing coarse rectangular `core_occ` cells wherever exact global `occ6` `IoU` improves!
        slope_reserved_occ = np.zeros((w_x, w_y, n_plates_z), dtype=bool)
        must_be_smooth_tile_occ = np.zeros((w_x, w_y, n_plates_z), dtype=bool)
        has_3070b_for_corner = ("3070b" in kit_set) and (rem_inv is None or rem_inv.get("3070b", 0) > 0)
        has_top_stud_at_p = core_occ.copy()
        for jb_p, jb_mask in jumper_top_stud_xy.items():
            has_top_stud_at_p[:, :, jb_p] = jb_mask

        # Compute exact initial `occ6` intersection & overshoot of `core_occ` so `top_stud_slope_molds`
        # optimizes true global `ΔIoU` (`never carving dense material out of sparse/thin meshes!`)
        core_occ6_est = np.zeros_like(occ6)
        for p_est in range(n_plates_z):
            if p_est in jumper_bridge_layers or not np.any(core_occ[:, :, p_est]):
                continue
            ph_e0 = layer_phase[p_est]
            z6_e0 = max(0, (p_est * 8) // 6)
            z6_e1 = min(occ6.shape[2], int(math.ceil(((p_est + 1) * 8) / 6.0)))
            for ix_e0 in range(w_x):
                for iy_e0 in range(w_y):
                    if core_occ[ix_e0, iy_e0, p_est]:
                        x6_e0 = (ix_e0 * 20 + ph_e0[0]) // 5
                        y6_e0 = (iy_e0 * 20 + ph_e0[1]) // 5
                        core_occ6_est[x6_e0 : x6_e0 + 4, y6_e0 : y6_e0 + 4, z6_e0:z6_e1] = True
        mesh_sub_est = max(1, int(np.count_nonzero(occ6)))
        core_inter_est = int(np.count_nonzero(core_occ6_est & occ6))
        core_over_est = int(np.count_nonzero(core_occ6_est & (~occ6)))
        base_iou_est = float(core_inter_est) / float(max(1, mesh_sub_est + core_over_est))

        allowed_top_slope_classes = (
            ("PLANAR_SLOPE", "STEEP_SLOPE", "XY_WEDGE_ROUND", "CURVED_SLOPE", "THIN_PANEL")
            if is_thin_regime
            else ("PLANAR_SLOPE", "STEEP_SLOPE", "XY_WEDGE_ROUND", "CURVED_SLOPE")
        )
        top_stud_slope_molds = [
            pid
            for pid in kit_ids
            if (ent_ts := CATALOG_RETRIEVAL_INDEX.get_entry(pid)) is not None
            and ent_ts.taxonomy_class in allowed_top_slope_classes
            and 0.20 <= ent_ts.fill_ratio <= 0.94
            and ent_ts.height_ldu in (8, 16, 24, 32, 48, 72)
            and max(ent_ts.studs_x, ent_ts.studs_y) >= 1
            and not (
                is_thin_regime
                and ent_ts.taxonomy_class != "THIN_PANEL"
                and ent_ts.studs_x >= 2
                and ent_ts.studs_y >= 2
                and ent_ts.height_ldu >= 24
            )
            and (
                len(get_brick_primitive(pid).hull_boxes) >= 2
                or (
                    ent_ts.taxonomy_class == "THIN_PANEL"
                    and len(get_brick_primitive(pid).hull_boxes) == 1
                    and min(
                        get_brick_primitive(pid).hull_boxes[0].xmax - get_brick_primitive(pid).hull_boxes[0].xmin,
                        get_brick_primitive(pid).hull_boxes[0].ymax - get_brick_primitive(pid).hull_boxes[0].ymin,
                    ) <= 12
                )
            )
            and not all(
                (b.xmax - b.xmin) % 20 == 0 and (b.ymax - b.ymin) % 20 == 0 and (b.zmax - b.zmin) % 24 == 0
                for b in get_brick_primitive(pid).hull_boxes
            )
        ]
        if top_stud_slope_molds:
            all_sl_cands: List[
                Tuple[
                    float,
                    float,
                    str,
                    LDUFrame,
                    int,
                    int,
                    int,
                    int,
                    int,
                    int,
                    Set[Tuple[int, int]],
                    Set[Tuple[int, int]],
                    Set[Tuple[int, int]],
                ]
            ] = []
            base_cg_min = 4 if is_thin_regime else 20
            def _canonical_s3_templates(pid_s3: str) -> Tuple[Any, ...]:
                """Keep 1 canonical anchor `(drx0 == 0, dry0 == 0)` per rotation so Step 3 avoids 5x redundant anti-stud scans."""
                raw_t = _fast_slope_grid_templates(pid_s3)
                seen_r: Set[int] = set()
                dedup: List[Any] = []
                for t_item in raw_t:
                    if t_item[4] == 0 and t_item[6] == 0 and t_item[0] not in seen_r:
                        seen_r.add(t_item[0])
                        dedup.append(t_item)
                if len(dedup) < len({t_item[0] for t_item in raw_t}):
                    for t_item in raw_t:
                        if t_item[0] not in seen_r:
                            seen_r.add(t_item[0])
                            dedup.append(t_item)
                return tuple(dedup)

            fp_groups_s3: Dict[Tuple[int, int, int, int, int], List[Tuple[str, bool, str, Any]]] = {}
            for pid_sm in top_stud_slope_molds:
                ent_sm = CATALOG_RETRIEVAL_INDEX.get_entry(pid_sm)
                cls_sm = ent_sm.taxonomy_class if ent_sm is not None else ""
                is_tp_sm = cls_sm == "THIN_PANEL"
                for t_sm in _canonical_s3_templates(pid_sm):
                    fp_groups_s3.setdefault((t_sm[4], t_sm[5], t_sm[6], t_sm[7], t_sm[8]), []).append(
                        (pid_sm, is_tp_sm, cls_sm, t_sm)
                    )
            fp_group_list_s3 = tuple((k_fp, tuple(v_fp), any(it[1] for it in v_fp)) for k_fp, v_fp in fp_groups_s3.items())
            uniform_phase_s3 = len(set(layer_phase)) <= 1
            has_jb_s3 = bool(jumper_bridge_layers)
            step3_hard_block = inv_reserved_occ | snot_spine_protected
            has_hard_block_s3 = bool(step3_hard_block.any())
            has_pier_s3 = bool(pier_protected.any())
            core_zb = [[sum((1 << p_k) for p_k, v_k in enumerate(col_k) if v_k) for col_k in row_k] for row_k in core_occ.tolist()]
            hb_zb = (
                [[sum((1 << p_k) for p_k, v_k in enumerate(col_k) if v_k) for col_k in row_k] for row_k in step3_hard_block.tolist()]
                if has_hard_block_s3
                else None
            )
            pier_zb = (
                [[sum((1 << p_k) for p_k, v_k in enumerate(col_k) if v_k) for col_k in row_k] for row_k in pier_protected.tolist()]
                if has_pier_s3
                else None
            )
            inv_zb = [[sum((1 << p_k) for p_k, v_k in enumerate(col_k) if v_k) for col_k in row_k] for row_k in inv_reserved_occ.tolist()]
            pil_zb = [[sum((1 << p_k) for p_k, v_k in enumerate(col_k) if v_k) for col_k in row_k] for row_k in pillar_added_occ.tolist()]
            snot_zb = [[sum((1 << p_k) for p_k, v_k in enumerate(col_k) if v_k) for col_k in row_k] for row_k in snot_spine_protected.tolist()]
            perim_offsets = ((1, 0), (-1, 0), (0, 1), (0, -1), (2, 0), (-2, 0), (0, 2), (0, -2))
            skip_s3_box_megastructure = (w_x * w_y) >= 169 and adaptive_target.mesh_surf_pts.shape[0] == 0
            for p_sl0 in (() if skip_s3_box_megastructure else range(n_plates_z)):
                if deadline_monotonic_s is not None and time.monotonic() >= deadline_monotonic_s:
                    break
                if p_sl0 in jumper_bridge_layers or (p_sl0 == 0 and foundation_policy != FoundationPolicy.CLUTCHED_BASEPLATE):
                    continue
                ph_s = layer_phase[p_sl0]
                cg_p0_l = phase_grids[ph_s][:, :, p_sl0].tolist()
                sup_s = _support_mask_at_layer(p_sl0)
                sup_s_l = sup_s.tolist() if sup_s is not None else None
                z0_s = p_sl0 * 8
                r_zmod3 = p_sl0 % 3
                bit_p0 = 1 << p_sl0
                bit_pm1 = (1 << (p_sl0 - 1)) if p_sl0 >= 1 else 0
                bit_pm2 = (1 << (p_sl0 - 2)) if p_sl0 >= 2 else 0
                bit_roof_chk = 1 << min(n_plates_z - 1, p_sl0 + 3)
                seen_sl_poses: Set[Tuple[str, int, int, int]] = set()

                for ix_r in range(w_x):
                    core_zb_ix = core_zb[ix_r]
                    cg_ix = cg_p0_l[ix_r]
                    sup_ix = sup_s_l[ix_r] if sup_s_l is not None else None
                    for iy_r in range(w_y):
                        cell_in_core = (core_zb_ix[iy_r] & bit_p0) != 0
                        cg_val_r = cg_ix[iy_r]
                        cell_in_mesh_s3 = cg_val_r >= base_cg_min or (
                            not is_thin_regime
                            and cg_val_r >= 12
                            and (
                                (ix_r + 1 < w_x and (core_zb[ix_r + 1][iy_r] & bit_p0) != 0)
                                or (iy_r + 1 < w_y and (core_zb_ix[iy_r + 1] & bit_p0) != 0)
                            )
                        )
                        if not cell_in_core and not cell_in_mesh_s3 and not (is_thin_regime and has_thin_panel_kit):
                            continue
                        if sup_ix is not None and not sup_ix[iy_r]:
                            continue
                        # Candidate must touch or sit within 2 cells of course/roof boundary (`or be on thin regime`)
                        if not is_thin_regime:
                            is_near_perim = (not cell_in_core) or (p_sl0 + 3 >= n_plates_z) or ((core_zb_ix[iy_r] & bit_roof_chk) == 0)
                            if not is_near_perim:
                                for dx_p, dy_p in perim_offsets:
                                    nx_p, ny_p = ix_r + dx_p, iy_r + dy_p
                                    if nx_p < 0 or nx_p >= w_x or ny_p < 0 or ny_p >= w_y or (core_zb[nx_p][ny_p] & bit_p0) == 0:
                                        is_near_perim = True
                                        break
                            if not is_near_perim:
                                continue
                        px_r = ix_r * 20 + ph_s[0] + 10
                        py_r = iy_r * 20 + ph_s[1] + 10
                        for (drx0, drx1, dry0, dry1, h_pl_s), grp_items_s3, grp_has_tp in fp_group_list_s3:
                            if not cell_in_core and not cell_in_mesh_s3 and not grp_has_tp:
                                continue
                            rx0_s = ix_r + drx0
                            rx1_s = ix_r + drx1
                            ry0_s = iy_r + dry0
                            ry1_s = iy_r + dry1
                            if rx0_s < 0 or ry0_s < 0 or rx1_s > w_x or ry1_s > w_y:
                                continue
                            p_top_next = p_sl0 + h_pl_s
                            if p_top_next > n_plates_z:
                                continue
                            z_span_m = ((1 << h_pl_s) - 1) << p_sl0
                            if not uniform_phase_s3 or has_jb_s3:
                                ph_bad = False
                                for q_h in range(h_pl_s):
                                    pq_h = p_sl0 + q_h
                                    if layer_phase[pq_h] != ph_s or (
                                        pq_h in jumper_bridge_layers
                                        and bool(jumper_used_low[pq_h][rx0_s:rx1_s, ry0_s:ry1_s].any())
                                    ):
                                        ph_bad = True
                                        break
                                if ph_bad:
                                    continue
                            if hb_zb is not None:
                                hb_hit = False
                                for bx_h in range(rx0_s, rx1_s):
                                    hb_row = hb_zb[bx_h]
                                    for by_h in range(ry0_s, ry1_s):
                                        if hb_row[by_h] & z_span_m:
                                            hb_hit = True
                                            break
                                    if hb_hit:
                                        break
                                if hb_hit:
                                    continue
                            pier_hit_fp = False
                            if pier_zb is not None:
                                for bx_p in range(rx0_s, rx1_s):
                                    p_row = pier_zb[bx_p]
                                    for by_p in range(ry0_s, ry1_s):
                                        if p_row[by_p] & z_span_m:
                                            pier_hit_fp = True
                                            break
                                    if pier_hit_fp:
                                        break
                                if pier_hit_fp and not (is_thin_regime and grp_has_tp and not forbidden_support_boxes):
                                    continue
                            sx0_q = (rx0_s * 20 + ph_s[0]) // 5
                            sx1_q = (rx1_s * 20 + ph_s[0]) // 5
                            sy0_q = (ry0_s * 20 + ph_s[1]) // 5
                            sy1_q = (ry1_s * 20 + ph_s[1]) // 5
                            sz0_q = z0_s // 6
                            sz1_q = int(math.ceil((p_top_next * 8) / 6.0))
                            tgt_blk = (
                                solid_occ6_padded[sx0_q:sx1_q, sy0_q:sy1_q, sz0_q:sz1_q]
                                if solid_occ6_padded is not None
                                else occ6[sx0_q:sx1_q, sy0_q:sy1_q, sz0_q:sz1_q]
                            )
                            cached_co_full: Optional[Tuple[int, int]] = None

                            for pid_sm, is_thin_panel_mold, cls_sm, t_sm in grp_items_s3:
                                if not cell_in_core and not cell_in_mesh_s3 and not is_thin_panel_mold:
                                    continue
                                if pier_hit_fp and not (is_thin_regime and is_thin_panel_mold and not forbidden_support_boxes):
                                    continue
                                if rem_inv is not None and rem_inv.get(pid_sm, 0) <= 0:
                                    continue
                                (
                                    rid_sm,
                                    dx_sm,
                                    dy_sm,
                                    dz_sm,
                                    _,
                                    _,
                                    _,
                                    _,
                                    _,
                                    rboxes_sm,
                                    rel_tstuds,
                                    rel_banti,
                                    rel_stud_clash,
                                    rel_occ_cells,
                                    masks_z3,
                                ) = t_sm
                                wx_sm = px_r + dx_sm
                                wy_sm = py_r + dy_sm
                                skey = (pid_sm, rid_sm, wx_sm, wy_sm)
                                if skey in seen_sl_poses:
                                    continue
                                seen_sl_poses.add(skey)
                                pos_sm = (wx_sm, wy_sm, z0_s + dz_sm)
                                banti_xy_s: Set[Tuple[int, int]] = {
                                    (ix_r + dbx, iy_r + dby) for dbx, dby in rel_banti
                                }
                                clash_xy_s: Set[Tuple[int, int]] = {
                                    (ix_r + dcx, iy_r + dcy) for dcx, dcy in rel_stud_clash
                                }
                                # Require full downward support on anti-stud cells (`or >= 50% on 15573 jumper bridges`);
                                # on chamfered corner cells in `clash_xy_s`, allow capping with `3070b` so zero stud clash occurs!
                                if sup_s_l is not None:
                                    on_jb_s = (p_sl0 - 1) in jumper_top_stud_xy
                                    if on_jb_s:
                                        sup_ok = (
                                            sum(1 for bx_a, by_a in banti_xy_s if sup_s_l[bx_a][by_a])
                                            >= max(1, (len(banti_xy_s) + 1) // 2)
                                        ) and not any(sup_s_l[bx_c][by_c] for bx_c, by_c in clash_xy_s)
                                    else:
                                        sup_ok = True
                                        for bx_a in range(rx0_s, rx1_s):
                                            sup_row_a = sup_s_l[bx_a]
                                            for by_a in range(ry0_s, ry1_s):
                                                if (bx_a, by_a) in banti_xy_s:
                                                    if not sup_row_a[by_a]:
                                                        sup_ok = False
                                                        break
                                                elif (bx_a, by_a) in clash_xy_s and p_sl0 > 0 and (core_zb[bx_a][by_a] & bit_pm1) != 0:
                                                    can_smooth_cap = (
                                                        has_3070b_for_corner
                                                        and (inv_zb[bx_a][by_a] & bit_pm1) == 0
                                                        and (p_sl0 == 1 or (core_zb[bx_a][by_a] & bit_pm2) != 0)
                                                    )
                                                    if not can_smooth_cap:
                                                        sup_ok = False
                                                        break
                                            if not sup_ok:
                                                break
                                    if not sup_ok:
                                        continue
                                top_stud_xy_s: Set[Tuple[int, int]] = {
                                    (ix_r + dtx, iy_r + dty) for dtx, dty in rel_tstuds
                                }
                                # Require every anti-stud base cell at `p_sl0` to lie inside mesh/core (`never spear empty air`),
                                # except grounded `p_sl0 == 0` thin structural wall panels where >= 1 base stud anchors a leg/pier!
                                if (
                                    is_thin_regime
                                    and is_thin_panel_mold
                                    and p_sl0 == 0
                                    and foundation_policy == FoundationPolicy.CLUTCHED_BASEPLATE
                                ):
                                    base_ok = any(
                                        (core_zb[bx_a][by_a] & 1) != 0 or cg_p0_l[bx_a][by_a] >= base_cg_min
                                        for bx_a, by_a in banti_xy_s
                                    )
                                else:
                                    has_core_back_stud = False
                                    if top_stud_xy_s:
                                        for tx_s, ty_s in top_stud_xy_s:
                                            if (core_zb[tx_s][ty_s] & bit_p0) != 0 and (pil_zb[tx_s][ty_s] & bit_p0) == 0:
                                                has_core_back_stud = True
                                                break
                                    base_ok = True
                                    for bx_a, by_a in banti_xy_s:
                                        in_co = (
                                            (core_zb[bx_a][by_a] & bit_p0) != 0
                                            if (is_thin_regime and is_thin_panel_mold)
                                            else ((core_zb[bx_a][by_a] & bit_p0) != 0 and (pil_zb[bx_a][by_a] & bit_p0) == 0)
                                        )
                                        min_cg_cell = (
                                            12
                                            if (has_core_back_stud and (bx_a, by_a) not in top_stud_xy_s)
                                            else base_cg_min
                                        )
                                        if not (in_co or cg_p0_l[bx_a][by_a] >= min_cg_cell):
                                            base_ok = False
                                            break
                                if not base_ok:
                                    continue

                                sm_blk = masks_z3[r_zmod3]
                                if sm_blk.shape != tgt_blk.shape:
                                    continue
                                # Prevent tall slopes (`>= 32 LDU` = `h_pl_s >= 4`) from spearing empty sky at their apex!
                                if not is_thin_regime and not is_thin_panel_mold and h_pl_s >= 4 and sm_blk.shape[2] >= 2:
                                    top_sl_m = sm_blk[:, :, -2:]
                                    top_sl_tot = int(np.count_nonzero(top_sl_m))
                                    if top_sl_tot > 0 and int(np.count_nonzero(top_sl_m & tgt_blk[:, :, -2:])) * 2 < top_sl_tot:
                                        continue

                                # Precomputed XY grid cells touched by `sm_blk` (`preserves untouched inner cells on L-shaped/macaroni `3063a`!`)
                                occ_cells_s = tuple((ix_r + domx, iy_r + domy) for domx, domy in rel_occ_cells)
                                if not occ_cells_s:
                                    continue
                                is_partial_xy_mold = len(occ_cells_s) < (rx1_s - rx0_s) * (ry1_s - ry0_s)

                                upper_blocked = False
                                sh_clear_list: List[Tuple[int, int, int]] = []
                                if p_top_next < n_plates_z:
                                    bit_ptop = 1 << p_top_next
                                    if p_top_next in jumper_bridge_layers:
                                        j_req = jumper_used_low.get(p_top_next)
                                        if j_req is not None:
                                            for ux_k, uy_k in occ_cells_s:
                                                if j_req[ux_k, uy_k] and (ux_k, uy_k) not in top_stud_xy_s:
                                                    upper_blocked = True
                                                    break
                                    elif layer_phase[p_top_next] == ph_s:
                                        for ux_k, uy_k in occ_cells_s:
                                            if (ux_k, uy_k) in top_stud_xy_s:
                                                continue
                                            if (inv_zb[ux_k][uy_k] & bit_ptop) != 0:
                                                upper_blocked = True
                                                break
                                            if (core_zb[ux_k][uy_k] & bit_ptop) != 0:
                                                can_clear_lip = (
                                                    not is_thin_regime
                                                    and h_pl_s >= 2
                                                    and (pier_zb is None or (pier_zb[ux_k][uy_k] & bit_ptop) == 0)
                                                    and (snot_zb[ux_k][uy_k] & bit_ptop) == 0
                                                    and (
                                                        (p_top_next + 1) >= n_plates_z
                                                        or (
                                                            (core_zb[ux_k][uy_k] >> (p_top_next + 1)) == 0
                                                            and (inv_zb[ux_k][uy_k] >> (p_top_next + 1)) == 0
                                                            and (
                                                                (p_top_next + 1) not in jumper_used_low
                                                                or not jumper_used_low[p_top_next + 1][ux_k, uy_k]
                                                            )
                                                        )
                                                    )
                                                )
                                                if can_clear_lip:
                                                    sh_clear_list.append((ux_k, uy_k, p_top_next))
                                                else:
                                                    upper_blocked = True
                                                    break
                                    else:
                                        ph_u = layer_phase[p_top_next]
                                        for ux_k, uy_k in occ_cells_s:
                                            if (ux_k, uy_k) in top_stud_xy_s:
                                                continue
                                            xb0_k = ux_k * 20 + ph_s[0]
                                            yb0_k = uy_k * 20 + ph_s[1]
                                            for bx_u in range(max(0, ux_k - 1), min(w_x, ux_k + 2)):
                                                for by_u in range(max(0, uy_k - 1), min(w_y, uy_k + 2)):
                                                    if (core_zb[bx_u][by_u] & bit_ptop) != 0:
                                                        if (
                                                            min(xb0_k + 20, bx_u * 20 + ph_u[0] + 20)
                                                            > max(xb0_k, bx_u * 20 + ph_u[0])
                                                            and min(yb0_k + 20, by_u * 20 + ph_u[1] + 20)
                                                            > max(yb0_k, by_u * 20 + ph_u[1])
                                                        ):
                                                            upper_blocked = True
                                                            break
                                                if upper_blocked:
                                                    break
                                            if upper_blocked:
                                                break
                                if upper_blocked:
                                    continue

                                if sz1_q > gz6_nonempty_limit:
                                    sky_sl = sm_blk[:, :, max(0, gz6_nonempty_limit - sz0_q) :]
                                    if int(np.count_nonzero(sky_sl)) > max(
                                        4, int(0.08 * int(np.count_nonzero(sm_blk & tgt_blk)))
                                    ):
                                        continue
                                # Prevent multi-cell slopes (`e.g. 3040a`) from rotating 90° parallel to an inset cliff
                                # and shoving their full-height solid back-stud column (`top_stud_xy_s`) into empty sky!
                                if not is_thin_regime and not is_thin_panel_mold and top_stud_xy_s:
                                    p_back_top = p_sl0 + h_pl_s - 1
                                    if any(
                                        int(phase_grids[ph_s][tx_b, ty_b, p_back_top]) < 46
                                        for tx_b, ty_b in top_stud_xy_s
                                    ):
                                        continue
                                # Never let a studless (`len(rel_tstuds) == 0`) slope cap substantial uncovered mesh above it
                                if (
                                    len(rel_tstuds) == 0
                                    and sz1_q < occ6.shape[2]
                                    and not (is_thin_panel_mold and (sz1_q >= gz6_nonempty_limit - 1 or h_pl_s >= 3))
                                ):
                                    uz2_q = min(occ6.shape[2], sz1_q + 4)
                                    above_mesh_q = int(np.count_nonzero(occ6[sx0_q:sx1_q, sy0_q:sy1_q, sz1_q:uz2_q]))
                                    min_uncov_cap = (
                                        3 if is_thin_regime else 10 * max(1, len(occ_cells_s))
                                    )
                                    if above_mesh_q >= min_uncov_cap:
                                        continue

                                sm_in = int(np.count_nonzero(sm_blk & tgt_blk))
                                sm_tot = int(np.count_nonzero(sm_blk))
                                if is_partial_xy_mold:
                                    co_in = 0
                                    co_tot = 0
                                    for ux_oc, uy_oc in occ_cells_s:
                                        cx_oc = (ux_oc * 20 + ph_s[0]) // 5
                                        cy_oc = (uy_oc * 20 + ph_s[1]) // 5
                                        c_b_oc = core_occ6_est[cx_oc : cx_oc + 4, cy_oc : cy_oc + 4, sz0_q:sz1_q]
                                        t_b_oc = occ6[cx_oc : cx_oc + 4, cy_oc : cy_oc + 4, sz0_q:sz1_q]
                                        co_in += int(np.count_nonzero(c_b_oc & t_b_oc))
                                        co_tot += int(np.count_nonzero(c_b_oc))
                                else:
                                    if cached_co_full is None:
                                        co_blk = core_occ6_est[sx0_q:sx1_q, sy0_q:sy1_q, sz0_q:sz1_q]
                                        cached_co_full = (int(np.count_nonzero(co_blk & tgt_blk)), int(np.count_nonzero(co_blk)))
                                    co_in, co_tot = cached_co_full
                                if sh_clear_list:
                                    sz1_sh = min(occ6.shape[2], int(math.ceil(((p_top_next + 1) * 8) / 6.0)))
                                    if sz1_sh > sz1_q:
                                        for ux_cl, uy_cl, _ in sh_clear_list:
                                            cx6_cl = (ux_cl * 20 + ph_s[0]) // 5
                                            cy6_cl = (uy_cl * 20 + ph_s[1]) // 5
                                            c_sub_cl = core_occ6_est[cx6_cl : cx6_cl + 4, cy6_cl : cy6_cl + 4, sz1_q:sz1_sh]
                                            t_sub_cl = occ6[cx6_cl : cx6_cl + 4, cy6_cl : cy6_cl + 4, sz1_q:sz1_sh]
                                            co_in += int(np.count_nonzero(c_sub_cl & t_sub_cl))
                                            co_tot += int(np.count_nonzero(c_sub_cl))
                                if is_thin_regime and not is_thin_panel_mold:
                                    if h_pl_s >= 4 and sm_tot >= 40:
                                        continue
                                    if sm_tot >= 28 and sm_in < int(0.55 * (sm_tot - sm_in)):
                                        continue
                                d_in_sw = sm_in - co_in
                                d_ov_sw = (sm_tot - sm_in) - (co_tot - co_in)
                                new_iou_sw = float(max(0, core_inter_est + d_in_sw)) / float(
                                    max(1, mesh_sub_est + core_over_est + d_ov_sw)
                                )
                                val_diff = (new_iou_sw - base_iou_est) * float(mesh_sub_est + core_over_est)
                                min_vd_req = (
                                    (0.75 if is_thin_panel_mold else 1.2)
                                    if is_thin_regime
                                    else (0.70 if (len(rel_tstuds) == 0 and h_pl_s == 2 and p_top_next in jumper_bridge_layers) else 1.8)
                                )
                                is_curved_step_tie = (
                                    cls_sm in ("CURVED_SLOPE", "XY_WEDGE_ROUND")
                                    and not sh_clear_list
                                    and sm_in >= 12
                                    and sm_in >= co_in
                                    and (sm_tot - sm_in) <= (co_tot - co_in)
                                    and (d_in_sw > 0 or d_ov_sw < 0)
                                )
                                if sm_in >= 4 and (val_diff >= min_vd_req or is_curved_step_tie):
                                    if forbidden_support_boxes and any(
                                        _box_hits_forbidden(
                                            LDUBox(b[0] + px_r, b[1] + py_r, b[2] + z0_s, b[3] + px_r, b[4] + py_r, b[5] + z0_s),
                                            forbidden_support_boxes,
                                        )
                                        for b in rboxes_sm
                                    ):
                                        continue
                                    area_xy_s = max(1, len(occ_cells_s))
                                    taper_area_s = (
                                        float(area_xy_s)
                                        if len(top_stud_xy_s) == 0
                                        else float(max(1, len(occ_cells_s) - len(top_stud_xy_s)))
                                    )
                                    eff_vol_s = taper_area_s * (
                                        (float(h_pl_s) / 2.5)
                                        if (h_pl_s >= 4 and not is_thin_panel_mold and not forbidden_support_boxes)
                                        else 1.0
                                    )
                                    mold_h_s = sum(ord(ch) * (i_ch + 3) for i_ch, ch in enumerate(pid_sm))
                                    mesh_h_s = w_x * 17 + w_y * 31 + n_plates_z * 13 + (mesh_sub_est % 29) * 7 + p_sl0 * 5
                                    m_rot_tie = 0.014 * float(
                                        ((mesh_h_s ^ (mold_h_s * 19)) + mold_h_s * (1 + (mesh_h_s % 7))) % 11
                                    )
                                    wide_scale_s = (4.0 / float(area_xy_s)) if area_xy_s > 4 else 1.0
                                    vd_norm_s = val_diff * wide_scale_s
                                    ev_norm_s = max(0.5, eff_vol_s * wide_scale_s)
                                    sc_sm = (
                                        (0.15 + 0.08 * float(min(4, area_xy_s)) + m_rot_tie)
                                        if is_curved_step_tie
                                        else ((vd_norm_s / (ev_norm_s ** 0.55)) + 0.42 * vd_norm_s + m_rot_tie)
                                    )
                                    if view_weight_grid_6 is not None:
                                        vw_blk = view_weight_grid_6[sx0_q:sx1_q, sy0_q:sy1_q, sz0_q:sz1_q]
                                        if vw_blk.shape == sm_blk.shape and vw_blk.size > 0:
                                            co_w_sl = core_occ6_est[sx0_q:sx1_q, sy0_q:sy1_q, sz0_q:sz1_q]
                                            carved_sp = (~sm_blk) & co_w_sl & (~tgt_blk)
                                            if np.any(carved_sp):
                                                vw_mult = float(np.mean(vw_blk[carved_sp]))
                                            elif np.any(sm_blk):
                                                vw_mult = float(np.mean(vw_blk[sm_blk]))
                                            else:
                                                vw_mult = float(np.mean(vw_blk))
                                            sc_sm = sc_sm * max(0.05, vw_mult)
                                    all_sl_cands.append(
                                        (
                                            sc_sm,
                                            0.0 if is_curved_step_tie else val_diff,
                                            pid_sm,
                                            LDUFrame(rid_sm, pos_sm),
                                            p_sl0,
                                            h_pl_s,
                                            rx0_s,
                                            rx1_s,
                                            ry0_s,
                                            ry1_s,
                                            top_stud_xy_s,
                                            banti_xy_s,
                                            clash_xy_s,
                                            tuple(sh_clear_list),
                                            occ_cells_s,
                                        )
                                    )

            reserved_mold_counts: Dict[str, int] = {}
            rep_pen_sl = 0.08

            def _mold_s3_first_bonus(pid_b: str, vd_b: float) -> float:
                ent_b = CATALOG_RETRIEVAL_INDEX.get_entry(pid_b)
                if ent_b is not None and ent_b.taxonomy_class in ("CURVED_SLOPE", "XY_WEDGE_ROUND", "THIN_PANEL"):
                    return 0.12 if vd_b <= 1e-6 else 0.10
                return 0.05

            sl_heap = [
                (
                    -(c[0] + _mold_s3_first_bonus(c[2], c[1])),
                    -c[1],
                    0,
                    c[2],
                    c[3].rot_id,
                    c[3].pos,
                    c[3],
                    c[4],
                    c[5],
                    c[6],
                    c[7],
                    c[8],
                    c[9],
                    c[10],
                    c[11],
                    c[12],
                    c[13],
                    c[14],
                )
                for c in all_sl_cands
            ]
            heapq.heapify(sl_heap)
            sl_res_zb = [[0] * w_y for _ in range(w_x)]
            sm_tile_zb = [[sum((1 << p_k) for p_k, v_k in enumerate(col_k) if v_k) for col_k in row_k] for row_k in must_be_smooth_tile_occ.tolist()]
            while sl_heap:
                (
                    neg_sc,
                    neg_vd,
                    used_seen,
                    pid_cs,
                    rid_cs_k,
                    pos_cs_k,
                    pose_cs,
                    p_cs,
                    h_cs,
                    rx0_c,
                    rx1_c,
                    ry0_c,
                    ry1_c,
                    tstud_c,
                    banti_c,
                    clash_c,
                    sh_clear_c,
                    occ_cells_c,
                ) = heapq.heappop(sl_heap)
                cur_used = reserved_mold_counts.get(pid_cs, 0)
                if -neg_vd <= 1e-6 and cur_used >= 1:
                    continue
                if cur_used != used_seen:
                    fb_s = _mold_s3_first_bonus(pid_cs, -neg_vd)
                    base_sc = -neg_sc - (fb_s if used_seen == 0 else 0.0) + rep_pen_sl * float(min(12, used_seen))
                    heapq.heappush(
                        sl_heap,
                        (
                            -(base_sc + (fb_s if cur_used == 0 else 0.0) - rep_pen_sl * float(min(12, cur_used))),
                            neg_vd,
                            cur_used,
                            pid_cs,
                            rid_cs_k,
                            pos_cs_k,
                            pose_cs,
                            p_cs,
                            h_cs,
                            rx0_c,
                            rx1_c,
                            ry0_c,
                            ry1_c,
                            tstud_c,
                            banti_c,
                            clash_c,
                            sh_clear_c,
                            occ_cells_c,
                        ),
                    )
                    continue
                if rem_inv is not None and rem_inv.get(pid_cs, 0) <= 0:
                    continue
                z_cs_m = ((1 << h_cs) - 1) << p_cs
                if any(
                    (sl_res_zb[ux_o][uy_o] | inv_zb[ux_o][uy_o] | sm_tile_zb[ux_o][uy_o]) & z_cs_m
                    for ux_o, uy_o in occ_cells_c
                ):
                    continue
                if sh_clear_c and any(
                    ((sl_res_zb[ux_s][uy_s] | inv_zb[ux_s][uy_s] | sm_tile_zb[ux_s][uy_s]) & (1 << p_s)) != 0
                    or ((p_s + 1) < n_plates_z and (sl_res_zb[ux_s][uy_s] >> (p_s + 1)) != 0)
                    for ux_s, uy_s, p_s in sh_clear_c
                ):
                    continue
                # Dynamic full downward support check against already-reserved lower slopes
                smooth_cap_needed_xy: List[Tuple[int, int]] = []
                if p_cs > 0:
                    if (p_cs - 1) in jumper_top_stud_xy:
                        sup_dyn = jumper_top_stud_xy[p_cs - 1]
                    elif layer_phase[p_cs - 1] == layer_phase[p_cs]:
                        sup_dyn = has_top_stud_at_p[:, :, p_cs - 1]
                    else:
                        sup_dyn = _support_mask_at_layer(p_cs)
                    if sup_dyn is not None:
                        if (p_cs - 1) in jumper_top_stud_xy:
                            dyn_ok = (
                                sum(1 for bx_d, by_d in banti_c if sup_dyn[bx_d, by_d])
                                >= max(1, (len(banti_c) + 1) // 2)
                            ) and not any(sup_dyn[bx_d, by_d] for bx_d, by_d in clash_c if (bx_d, by_d) in occ_cells_c)
                        else:
                            dyn_ok = True
                            bit_csm1 = 1 << (p_cs - 1)
                            for bx_d, by_d in occ_cells_c:
                                if (bx_d, by_d) in banti_c:
                                    if not sup_dyn[bx_d, by_d]:
                                        dyn_ok = False
                                        break
                                elif (bx_d, by_d) in clash_c and has_top_stud_at_p[bx_d, by_d, p_cs - 1]:
                                    can_cap_now = (
                                        has_3070b_for_corner
                                        and (core_zb[bx_d][by_d] & bit_csm1) != 0
                                        and ((sl_res_zb[bx_d][by_d] | inv_zb[bx_d][by_d]) & bit_csm1) == 0
                                        and (p_cs == 1 or bool(has_top_stud_at_p[bx_d, by_d, p_cs - 2]))
                                    )
                                    if can_cap_now:
                                        smooth_cap_needed_xy.append((bx_d, by_d))
                                    else:
                                        dyn_ok = False
                                        break
                        if not dyn_ok:
                            continue
                # Dynamic upward support check: never starve an already-reserved upper piece, jumper, or non-cleared `core_occ`
                sh_clear_set_c = set(sh_clear_c)
                if (p_cs + h_cs) < n_plates_z:
                    up_starved = False
                    p_top_c = p_cs + h_cs
                    bit_top_c = 1 << p_top_c
                    j_req_c = jumper_used_low.get(p_top_c)
                    for ux_d, uy_d in occ_cells_c:
                        needs_stud = (
                            bool(j_req_c[ux_d, uy_d])
                            if j_req_c is not None
                            else bool(
                                (
                                    (core_zb[ux_d][uy_d] & bit_top_c) != 0
                                    and (ux_d, uy_d, p_top_c) not in sh_clear_set_c
                                )
                                or ((sl_res_zb[ux_d][uy_d] | inv_zb[ux_d][uy_d]) & bit_top_c) != 0
                            )
                        )
                        if needs_stud and (ux_d, uy_d) not in tstud_c:
                            up_starved = True
                            break
                    if up_starved:
                        continue

                for ux_s, uy_s, p_s in sh_clear_c:
                    core_occ[ux_s, uy_s, p_s] = False
                    core_zb[ux_s][uy_s] &= ~(1 << p_s)
                    has_top_stud_at_p[ux_s, uy_s, p_s] = False

                for sc_x, sc_y in smooth_cap_needed_xy:
                    must_be_smooth_tile_occ[sc_x, sc_y, p_cs - 1] = True
                    sm_tile_zb[sc_x][sc_y] |= 1 << (p_cs - 1)
                    has_top_stud_at_p[sc_x, sc_y, p_cs - 1] = False

                for ux_o, uy_o in occ_cells_c:
                    slope_reserved_occ[ux_o, uy_o, p_cs : p_cs + h_cs] = True
                    sl_res_zb[ux_o][uy_o] |= z_cs_m
                    has_top_stud_at_p[ux_o, uy_o, p_cs : p_cs + h_cs] = False
                for sx_t, sy_t in tstud_c:
                    if rx0_c <= sx_t < rx1_c and ry0_c <= sy_t < ry1_c:
                        has_top_stud_at_p[sx_t, sy_t, p_cs + h_cs - 1] = True
                planned_by_z.setdefault(p_cs * 8, []).append((pid_cs, pose_cs))
                reserved_mold_counts[pid_cs] = cur_used + 1
                if rem_inv is not None:
                    rem_inv[pid_cs] = rem_inv.get(pid_cs, 0) - 1

        # Cap any remaining exposed terrace cells alongside `15573` jumper bridge layers with smooth tiles
        for p_jb in sorted(jumper_bridge_layers.keys()):
            ph_from_jb, _ = jumper_bridge_layers[p_jb]
            z0_jb = p_jb * 8
            z6_t0 = min(occ6.shape[2], int(math.ceil(z0_jb / 6.0)))
            z6_t1 = min(occ6.shape[2], int(math.ceil((z0_jb + 8) / 6.0)))
            if z6_t1 <= z6_t0:
                continue
            low_sup_jb = (
                has_top_stud_at_p[:, :, p_jb - 1]
                if (p_jb > 0 and (p_jb - 1) not in jumper_bridge_layers)
                else jumper_low_occ[p_jb]
            )
            u_low_jb = jumper_used_low[p_jb]
            cg_ter = phase_grids[ph_from_jb][:, :, p_jb]
            terrace_occ = (cg_ter >= 42) & low_sup_jb & (~u_low_jb) & (~slope_reserved_occ[:, :, p_jb])
            for tx_jb in range(w_x):
                for ty_jb in range(w_y):
                    if terrace_occ[tx_jb, ty_jb]:
                        x6_tb = (tx_jb * 20 + ph_from_jb[0]) // 5
                        y6_tb = (ty_jb * 20 + ph_from_jb[1]) // 5
                        tb_sl = occ6[x6_tb : x6_tb + 4, y6_tb : y6_tb + 4, z6_t0:z6_t1]
                        if tb_sl.size == 0 or 2 * int(np.count_nonzero(tb_sl)) < int(tb_sl.size):
                            terrace_occ[tx_jb, ty_jb] = False
            if np.any(terrace_occ) and tile_templates:
                t_pls = _tile_boolean_layer_with_rects(
                    terrace_occ,
                    z0=z0_jb,
                    ox=ph_from_jb[0],
                    oy=ph_from_jb[1],
                    templates=tile_templates,
                    prefer_x_major=(p_jb % 2 == 1),
                    parity_shift=((p_jb // 2) % 2),
                    rem_inv=rem_inv,
                    supported_below_xy=low_sup_jb & (~u_low_jb),
                )
                if t_pls:
                    planned_by_z.setdefault(z0_jb, []).extend(t_pls)

        rem_core = core_occ & (~inv_reserved_occ) & (~slope_reserved_occ)

        # Phase 11 Ext 5: When `view_weight_grid_6` is active, prune un-sloped topmost `rem_core` cells
        # whose weighted 2D silhouette background spill exceeds their weighted mesh intersection (`ov_w > 1.25 * in_w`),
        # operating at `1`-plate granularity when `plate_templates` are available and `3`-plate (`24 LDU`) course
        # granularity when the finite kit contains only `24 LDU` bricks!
        if view_weight_grid_6 is not None:
            has_avail_plates = bool(plate_templates) and (
                rem_inv is None or any(rem_inv.get(tp[0], 0) > 0 for tp in plate_templates)
            )
            step_z_vp = 1 if has_avail_plates else 3
            start_p_vp = (n_plates_z - 1) if step_z_vp == 1 else (3 * ((n_plates_z // 3) - 1))
            for p_vp in range(start_p_vp, 0, -step_z_vp):
                p_vp_end = min(n_plates_z, p_vp + step_z_vp)
                if any((p_vp + q_v) in jumper_bridge_layers for q_v in range(p_vp_end - p_vp)):
                    continue
                ph_vp = layer_phase[p_vp]
                if any(layer_phase[p_vp + q_v] != ph_vp for q_v in range(p_vp_end - p_vp)):
                    continue
                z6_v0 = max(0, (p_vp * 8) // 6)
                z6_v1 = min(occ6.shape[2], int(math.ceil((p_vp_end * 8) / 6.0)))
                if z6_v1 <= z6_v0:
                    continue
                j_need_up = jumper_used_low.get(p_vp_end)
                for ix_v in range(w_x):
                    for iy_v in range(w_y):
                        if not np.any(rem_core[ix_v, iy_v, p_vp:p_vp_end]) or np.any(pier_protected[ix_v, iy_v, p_vp:p_vp_end]):
                            continue
                        if j_need_up is not None and bool(j_need_up[ix_v, iy_v]):
                            continue
                        if p_vp_end < n_plates_z and (
                            rem_core[ix_v, iy_v, p_vp_end:].any()
                            or slope_reserved_occ[ix_v, iy_v, p_vp_end:].any()
                            or inv_reserved_occ[ix_v, iy_v, p_vp_end:].any()
                        ):
                            continue
                        x6_v = (ix_v * 20 + ph_vp[0]) // 5
                        y6_v = (iy_v * 20 + ph_vp[1]) // 5
                        t_sl_v = occ6[x6_v : x6_v + 4, y6_v : y6_v + 4, z6_v0:z6_v1]
                        w_sl_v = view_weight_grid_6[x6_v : x6_v + 4, y6_v : y6_v + 4, z6_v0:z6_v1]
                        in_w_v = float(np.sum(w_sl_v[t_sl_v]))
                        ov_w_v = float(np.sum(w_sl_v[~t_sl_v]))
                        if ov_w_v > 1.25 * in_w_v:
                            rem_core[ix_v, iy_v, p_vp:p_vp_end] = False

        # Place flush `4 LDU`-thin vertical perimeter brackets (`44728`, `99780`, `99207` — all `CC-BY-4.0`)
        # on `1x2` perimeter plate runs whenever a `4 LDU` (`1` sub-voxel) outer wall/taper slice strictly improves `occ6` IoU!
        bracket_cands = (
            []
            if ((w_x * w_y) >= 169 and adaptive_target.mesh_surf_pts.shape[0] == 0)
            else [bpid for bpid in ("44728", "99780", "99207") if bpid in kit_set]
        )
        if bracket_cands:
            planned_blade_boxes: List[LDUBox] = []
            for bpid in bracket_cands:
                prim_br = get_brick_primitive(bpid)
                if len(prim_br.hull_boxes) != 2:
                    continue
                for p_br in range(n_plates_z):
                    if p_br in jumper_bridge_layers or not np.any(rem_core[:, :, p_br]):
                        continue
                    ph_br = layer_phase[p_br]
                    sup_br = _support_mask_at_layer(p_br)
                    z0_br = p_br * 8
                    for rid_br in C4_Z_ROT_IDS:
                        if rem_inv is not None and rem_inv.get(bpid, 0) <= 0:
                            break
                        b_base = prim_br.hull_boxes[0].transform(LDUFrame(rid_br, (0, 0, 0)))
                        b_blade = prim_br.hull_boxes[1].transform(LDUFrame(rid_br, (0, 0, 0)))
                        wx_br = (b_base.xmax - b_base.xmin) // 20
                        wy_br = (b_base.ymax - b_base.ymin) // 20
                        if wx_br <= 0 or wy_br <= 0:
                            continue
                        odx_b = 1 if b_blade.xmin >= b_base.xmax else (-1 if b_blade.xmax <= b_base.xmin else 0)
                        ody_b = 1 if b_blade.ymin >= b_base.ymax else (-1 if b_blade.ymax <= b_base.ymin else 0)
                        if abs(odx_b) + abs(ody_b) != 1:
                            continue
                        zb_min = z0_br - b_base.zmin + b_blade.zmin
                        zb_max = z0_br - b_base.zmin + b_blade.zmax
                        if zb_min < 0 or int(math.ceil(zb_max / 6.0)) > gz6_nonempty_limit:
                            continue
                        p_chk0 = max(0, zb_min // 8)
                        p_chk1 = n_plates_z if zb_min < z0_br else min(n_plates_z, int(math.ceil(zb_max / 8.0)))
                        for ix_b in range(w_x - wx_br + 1):
                            for iy_b in range(w_y - wy_br + 1):
                                if rem_inv is not None and rem_inv.get(bpid, 0) <= 0:
                                    break
                                if not np.all(rem_core[ix_b : ix_b + wx_br, iy_b : iy_b + wy_br, p_br]):
                                    continue
                                if np.any(must_be_smooth_tile_occ[ix_b : ix_b + wx_br, iy_b : iy_b + wy_br, p_br]):
                                    continue
                                if sup_br is not None and not np.all(sup_br[ix_b : ix_b + wx_br, iy_b : iy_b + wy_br]):
                                    continue
                                # For upward blade (`zb_max > z0_br + 8`), avoid crossing phase-shift layers above `p_br`
                                if zb_max > z0_br + 8 and any(
                                    layer_phase[pq_b] != ph_br or pq_b in jumper_bridge_layers
                                    for pq_b in range(p_br, min(n_plates_z, zb_max // 8))
                                ):
                                    continue
                                # Ensure outer cell column is clear across `p_chk0 .. p_chk1` (and below `p_br` for downward blades)
                                outer_clear = True
                                for dx_k in range(wx_br):
                                    for dy_k in range(wy_br):
                                        ox_k = ix_b + dx_k + odx_b
                                        oy_k = iy_b + dy_k + ody_b
                                        if 0 <= ox_k < w_x and 0 <= oy_k < w_y:
                                            if (
                                                np.any(core_occ[ox_k, oy_k, p_chk0:p_chk1])
                                                or np.any(inv_reserved_occ[ox_k, oy_k, p_chk0:p_chk1])
                                                or np.any(slope_reserved_occ[ox_k, oy_k, p_chk0:p_chk1])
                                            ):
                                                outer_clear = False
                                                break
                                    if not outer_clear:
                                        break
                                if not outer_clear:
                                    continue
                                fx0_b = ix_b * 20 + ph_br[0]
                                fy0_b = iy_b * 20 + ph_br[1]
                                pose_br = LDUFrame(rid_br, (fx0_b - b_base.xmin, fy0_b - b_base.ymin, z0_br - b_base.zmin))
                                wb_base_w = prim_br.hull_boxes[0].transform(pose_br)
                                wb_blade_w = prim_br.hull_boxes[1].transform(pose_br)
                                if (
                                    _box_hits_forbidden(wb_base_w, forbidden_support_boxes)
                                    or _box_hits_forbidden(wb_blade_w, forbidden_support_boxes)
                                    or any(wb_blade_w.intersects(eb) for eb in planned_blade_boxes)
                                ):
                                    continue
                                bx0_6 = int(math.floor(wb_blade_w.xmin / 5.0))
                                bx1_6 = int(math.ceil(wb_blade_w.xmax / 5.0))
                                by0_6 = int(math.floor(wb_blade_w.ymin / 5.0))
                                by1_6 = int(math.ceil(wb_blade_w.ymax / 5.0))
                                bz0_6 = int(math.floor(wb_blade_w.zmin / 6.0))
                                bz1_6 = int(math.ceil(wb_blade_w.zmax / 6.0))
                                if (
                                    bx0_6 < 0
                                    or by0_6 < 0
                                    or bz0_6 < 0
                                    or bx1_6 > occ6.shape[0]
                                    or by1_6 > occ6.shape[1]
                                    or bz1_6 > occ6.shape[2]
                                ):
                                    continue
                                bl_sl = occ6[bx0_6:bx1_6, by0_6:by1_6, bz0_6:bz1_6]
                                bl_tot = int(bl_sl.size)
                                bl_tp = int(np.count_nonzero(bl_sl))
                                bl_fp = bl_tot - bl_tp
                                min_bl_tp = 8 if is_thin_regime else 18
                                if bl_tp < min_bl_tp or bl_tp < (bl_fp + 2 if is_thin_regime else 2 * bl_fp + 4):
                                    continue
                                planned_by_z.setdefault(z0_br, []).append((bpid, pose_br))
                                planned_blade_boxes.append(wb_blade_w)
                                rem_core[ix_b : ix_b + wx_br, iy_b : iy_b + wy_br, p_br] = False
                                if rem_inv is not None:
                                    rem_inv[bpid] = rem_inv.get(bpid, 0) - 1

        # Fuse aligned 3-plate (`24 LDU`) runs where phase is identical, all 3 layers are True, and top isn't reserved smooth-tile
        p_idx = 0
        while p_idx + 2 < n_plates_z:
            if _early_deadline_hit():
                return None
            if (
                p_idx not in jumper_bridge_layers
                and (p_idx + 1) not in jumper_bridge_layers
                and (p_idx + 2) not in jumper_bridge_layers
                and layer_phase[p_idx] == layer_phase[p_idx + 1] == layer_phase[p_idx + 2]
            ):
                fuse_mask = (
                    rem_core[:, :, p_idx]
                    & rem_core[:, :, p_idx + 1]
                    & rem_core[:, :, p_idx + 2]
                    & (~must_be_smooth_tile_occ[:, :, p_idx])
                    & (~must_be_smooth_tile_occ[:, :, p_idx + 1])
                    & (~must_be_smooth_tile_occ[:, :, p_idx + 2])
                )
                sup_f_chk = _support_mask_at_layer(p_idx)
                if sup_f_chk is not None and bool(
                    np.any(rem_core[:, :, p_idx] & (~sup_f_chk) & (~fuse_mask))
                ):
                    p_idx += 1
                    continue
                if np.any(fuse_mask):
                    ph_f = layer_phase[p_idx]
                    c_idx = p_idx // 3
                    sup_f = sup_f_chk
                    z0_f = p_idx * 8
                    z6_f0 = max(0, z0_f // 6)
                    z6_f1 = min(occ6.shape[2], (z0_f + 24) // 6)
                    # Seed protrusion-safe calibrated straight-edge (`30414`, `11211`, `87087`)
                    # outward SNOT core bricks across ALL 4 carved frustum/terrace walls!
                    snot_core_cands = [spid for spid in ("30414", "11211", "87087") if spid in kit_set]
                    if snot_core_cands:
                        def _cell_dir_snot_hits(cx_i: int, cy_i: int, odx: int, ody: int) -> int:
                            ox_c, oy_c = cx_i + odx, cy_i + ody
                            if 0 <= ox_c < w_x and 0 <= oy_c < w_y:
                                if (
                                    np.any(core_occ[ox_c, oy_c, p_idx : p_idx + 3])
                                    or np.any(inv_reserved_occ[ox_c, oy_c, p_idx : p_idx + 3])
                                    or np.any(slope_reserved_occ[ox_c, oy_c, p_idx : p_idx + 3])
                                ):
                                    return 0
                            fx0_c = cx_i * 20 + ph_f[0]
                            fx1_c = (cx_i + 1) * 20 + ph_f[0]
                            fy0_c = cy_i * 20 + ph_f[1]
                            fy1_c = (cy_i + 1) * 20 + ph_f[1]
                            if odx == 1:
                                mx0, mx1, my0, my1 = fx1_c // 5, fx1_c // 5 + 2, fy0_c // 5, fy1_c // 5
                            elif odx == -1:
                                mx0, mx1, my0, my1 = fx0_c // 5 - 2, fx0_c // 5, fy0_c // 5, fy1_c // 5
                            elif ody == 1:
                                mx0, mx1, my0, my1 = fx0_c // 5, fx1_c // 5, fy1_c // 5, fy1_c // 5 + 2
                            else:
                                mx0, mx1, my0, my1 = fx0_c // 5, fx1_c // 5, fy0_c // 5 - 2, fy0_c // 5
                            if mx0 < 0 or my0 < 0 or mx1 > occ6.shape[0] or my1 > occ6.shape[1]:
                                return 0
                            m_strip = occ6[mx0:mx1, my0:my1, z6_f0:z6_f1]
                            n_hits_s = int(np.count_nonzero(m_strip))
                            if m_strip.size == 0 or n_hits_s < int(0.18 * m_strip.size):
                                return 0
                            prot_box = LDUBox(
                                min(fx0_c, mx0 * 5),
                                min(fy0_c, my0 * 5),
                                z0_f,
                                max(fx1_c, mx1 * 5),
                                max(fy1_c, my1 * 5),
                                z0_f + 24,
                            )
                            return n_hits_s if not _box_hits_forbidden(prot_box, forbidden_support_boxes) else 0

                        dir_list_4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
                        cell_valid_dirs: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
                        for ix_d in range(w_x):
                            for iy_d in range(w_y):
                                if not fuse_mask[ix_d, iy_d]:
                                    continue
                                if sup_f is not None and not sup_f[ix_d, iy_d]:
                                    continue
                                vds_hits = [
                                    (h_v, d_v)
                                    for d_v in dir_list_4
                                    if (h_v := _cell_dir_snot_hits(ix_d, iy_d, d_v[0], d_v[1])) > 0
                                ]
                                if vds_hits:
                                    if is_thin_regime:
                                        vds_hits.sort(key=lambda item: -item[0])
                                        cell_valid_dirs[(ix_d, iy_d)] = [d_v for _, d_v in vds_hits]
                                    else:
                                        vds = [d_v for _, d_v in vds_hits]
                                        rot_d = (ix_d + iy_d + c_idx) % len(vds)
                                        cell_valid_dirs[(ix_d, iy_d)] = vds[rot_d:] + vds[:rot_d]

                        # Pass 1: Place multi-stud `30414` (`1x4`) and `11211` (`1x2`) ONLY along interior straight wall
                        # segments (`len(cell_valid_dirs) == 1`) so 2-exposed corner cells remain free to alternate walls!
                        for spid in [s_m for s_m in ("30414", "11211") if s_m in snot_core_cands]:
                            prim_sn = get_brick_primitive(spid)
                            lat_ps = [pt for pt in prim_sn.ports if pt.polarity == PortPolarity.STUD and pt.local_frame.z_axis[2] == 0]
                            if not lat_ps:
                                continue
                            for rid_sn in C4_Z_ROT_IDS:
                                if rem_inv is not None and rem_inv.get(spid, 0) <= 0:
                                    break
                                b_sn = prim_sn.hull_boxes[0].transform(LDUFrame(rid_sn, (0, 0, 0)))
                                wx_sn = (b_sn.xmax - b_sn.xmin) // 20
                                wy_sn = (b_sn.ymax - b_sn.ymin) // 20
                                f_norm = LDUFrame(rid_sn, (-b_sn.xmin, -b_sn.ymin, -b_sn.zmin))
                                wf_0 = f_norm.compose(lat_ps[0].local_frame)
                                odx_m, ody_m = int(wf_0.z_axis[0]), int(wf_0.z_axis[1])
                                for ix_s in range(w_x - wx_sn + 1):
                                    for iy_s in range(w_y - wy_sn + 1):
                                        if rem_inv is not None and rem_inv.get(spid, 0) <= 0:
                                            break
                                        if not np.all(fuse_mask[ix_s : ix_s + wx_sn, iy_s : iy_s + wy_sn]):
                                            continue
                                        if sup_f is not None and not np.all(sup_f[ix_s : ix_s + wx_sn, iy_s : iy_s + wy_sn]):
                                            continue
                                        # Require every covered cell to need `(odx_m, ody_m)`; allow at most 1 corner on 4-stud `30414` on wide footprints
                                        seg_ok = True
                                        corner_used_sn = 0
                                        max_corners_sn = 1 if (wx_sn * wy_sn >= 4 and w_x * w_y >= 24) else 0
                                        for dx_k in range(wx_sn):
                                            for dy_k in range(wy_sn):
                                                v_k = cell_valid_dirs.get((ix_s + dx_k, iy_s + dy_k), [])
                                                if (odx_m, ody_m) not in v_k:
                                                    seg_ok = False
                                                    break
                                                if len(v_k) > 1:
                                                    corner_used_sn += 1
                                                    if corner_used_sn > max_corners_sn:
                                                        seg_ok = False
                                                        break
                                            if not seg_ok:
                                                break
                                        if not seg_ok:
                                            continue
                                        fx0_l = ix_s * 20 + ph_f[0]
                                        fy0_l = iy_s * 20 + ph_f[1]
                                        pose_sn = LDUFrame(rid_sn, (fx0_l - b_sn.xmin, fy0_l - b_sn.ymin, z0_f - b_sn.zmin))
                                        planned_by_z.setdefault(z0_f, []).append((spid, pose_sn))
                                        fuse_mask[ix_s : ix_s + wx_sn, iy_s : iy_s + wy_sn] = False
                                        rem_core[ix_s : ix_s + wx_sn, iy_s : iy_s + wy_sn, p_idx : p_idx + 3] = False
                                        if rem_inv is not None:
                                            rem_inv[spid] = rem_inv.get(spid, 0) - 1

                        # Pass 2: Place `87087` (`1x1` SNOT brick) at every remaining perimeter/corner cell in `cell_valid_dirs`,
                        # alternating preferred direction `(ix + iy + c_idx) % len(vds)` so ALL 4 carved walls get lateral studs!
                        if "87087" in snot_core_cands:
                            prim_87 = get_brick_primitive("87087")
                            lat_87 = [pt for pt in prim_87.ports if pt.polarity == PortPolarity.STUD and pt.local_frame.z_axis[2] == 0]
                            dir_to_87: Dict[Tuple[int, int], Tuple[int, LDUBox]] = {}
                            if lat_87:
                                for rid_87 in C4_Z_ROT_IDS:
                                    b_87 = prim_87.hull_boxes[0].transform(LDUFrame(rid_87, (0, 0, 0)))
                                    wf_87 = LDUFrame(rid_87, (-b_87.xmin, -b_87.ymin, -b_87.zmin)).compose(lat_87[0].local_frame)
                                    dir_to_87[(int(wf_87.z_axis[0]), int(wf_87.z_axis[1]))] = (rid_87, b_87)
                            for (ix_s, iy_s), vds_c in sorted(cell_valid_dirs.items()):
                                if rem_inv is not None and rem_inv.get("87087", 0) <= 0:
                                    break
                                if not fuse_mask[ix_s, iy_s]:
                                    continue
                                pref_dir = vds_c[0]
                                if pref_dir not in dir_to_87:
                                    continue
                                rid_87, b_87 = dir_to_87[pref_dir]
                                fx0_l = ix_s * 20 + ph_f[0]
                                fy0_l = iy_s * 20 + ph_f[1]
                                pose_87 = LDUFrame(rid_87, (fx0_l - b_87.xmin, fy0_l - b_87.ymin, z0_f - b_87.zmin))
                                planned_by_z.setdefault(z0_f, []).append(("87087", pose_87))
                                fuse_mask[ix_s, iy_s] = False
                                rem_core[ix_s, iy_s, p_idx : p_idx + 3] = False
                                if rem_inv is not None:
                                    rem_inv["87087"] = rem_inv.get("87087", 0) - 1
                    if (p_idx - 1) in jumper_bridge_layers:
                        j_from, j_to = jumper_bridge_layers[p_idx - 1]
                        pref_x = j_to[0] != j_from[0]
                        par_sh = 0
                    else:
                        pref_x = c_idx % 2 == 0
                        par_sh = c_idx % 2
                    b_pls = _tile_boolean_layer_with_rects(
                        fuse_mask,
                        z0=z0_f,
                        ox=ph_f[0],
                        oy=ph_f[1],
                        templates=brick_templates,
                        prefer_x_major=pref_x,
                        parity_shift=par_sh,
                        rem_inv=rem_inv,
                        supported_below_xy=sup_f,
                        shell_running_bond=is_hollow_shell_active,
                    )
                    if b_pls:
                        planned_by_z.setdefault(z0_f, []).extend(b_pls)
                        for pid_b, pose_b in b_pls:
                            wb_b = get_brick_primitive(pid_b).hull_boxes[0].transform(pose_b)
                            ix0 = (wb_b.xmin - ph_f[0]) // 20
                            ix1 = (wb_b.xmax - ph_f[0]) // 20
                            iy0 = (wb_b.ymin - ph_f[1]) // 20
                            iy1 = (wb_b.ymax - ph_f[1]) // 20
                            rem_core[ix0:ix1, iy0:iy1, p_idx : p_idx + 3] = False
                    p_idx += 3
                    continue
            p_idx += 1

        # Fill interior solid mesh cavities around `15573` jumper bridges at `p in jumper_bridge_layers`
        # (`using studless smooth tiles at `p*8`, or `8 LDU plate + 8 LDU smooth tile` across 2-layer bridges)
        if jumper_bridge_layers and tile_templates and not is_thin_regime:
            all_planned_boxes: List[LDUBox] = [
                b_pl.transform(pose_pl)
                for z_list in planned_by_z.values()
                for pid_pl, pose_pl in z_list
                for b_pl in get_brick_primitive(pid_pl).hull_boxes
            ]
            for p_jb_f in sorted(jumper_bridge_layers.keys()):
                if p_jb_f <= 0 or (p_jb_f - 1) in jumper_bridge_layers:
                    continue
                ph_low_f, _ = jumper_bridge_layers[p_jb_f]
                z0_jb = p_jb_f * 8
                z6_jb0 = min(occ6.shape[2], int(math.ceil(z0_jb / 6.0)))
                z6_jb1 = min(occ6.shape[2], int(math.ceil((z0_jb + 8) / 6.0)))
                if z6_jb1 <= z6_jb0:
                    continue
                is_two_jb = (p_jb_f + 1) in jumper_bridge_layers
                z6_jb2 = min(occ6.shape[2], int(math.ceil((z0_jb + 16) / 6.0)))
                fill_1_mask = np.zeros((w_x, w_y), dtype=bool)
                fill_2_mask = np.zeros((w_x, w_y), dtype=bool)
                for ix_f in range(w_x):
                    for iy_f in range(w_y):
                        if not has_top_stud_at_p[ix_f, iy_f, p_jb_f - 1] or jumper_used_low[p_jb_f][ix_f, iy_f]:
                            continue
                        bx0_f = ix_f * 20 + ph_low_f[0]
                        by0_f = iy_f * 20 + ph_low_f[1]
                        wb1_f = LDUBox(bx0_f, by0_f, z0_jb, bx0_f + 20, by0_f + 20, z0_jb + 8)
                        if _box_hits_forbidden(wb1_f, forbidden_support_boxes):
                            continue
                        if any(
                            b_ex.xmin < wb1_f.xmax
                            and b_ex.xmax > wb1_f.xmin
                            and b_ex.ymin < wb1_f.ymax
                            and b_ex.ymax > wb1_f.ymin
                            and b_ex.zmin < wb1_f.zmax
                            and b_ex.zmax > wb1_f.zmin
                            for b_ex in all_planned_boxes
                        ):
                            continue
                        sx6_f, sy6_f = bx0_f // 5, by0_f // 5
                        m1_sl = occ6[sx6_f : sx6_f + 4, sy6_f : sy6_f + 4, z6_jb0:z6_jb1]
                        if m1_sl.size == 0 or int(np.count_nonzero(m1_sl)) < int(0.58 * m1_sl.size):
                            continue
                        if is_two_jb and plate_templates and z6_jb2 > z6_jb1:
                            wb2_f = LDUBox(bx0_f, by0_f, z0_jb + 8, bx0_f + 20, by0_f + 20, z0_jb + 16)
                            m2_sl = occ6[sx6_f : sx6_f + 4, sy6_f : sy6_f + 4, z6_jb1:z6_jb2]
                            clear_2 = (
                                not _box_hits_forbidden(wb2_f, forbidden_support_boxes)
                                and m2_sl.size > 0
                                and int(np.count_nonzero(m2_sl)) >= int(0.62 * m2_sl.size)
                                and not any(
                                    b_ex.xmin < wb2_f.xmax
                                    and b_ex.xmax > wb2_f.xmin
                                    and b_ex.ymin < wb2_f.ymax
                                    and b_ex.ymax > wb2_f.ymin
                                    and b_ex.zmin < wb2_f.zmax
                                    and b_ex.zmax > wb2_f.zmin
                                    for b_ex in all_planned_boxes
                                )
                            )
                            if clear_2:
                                fill_2_mask[ix_f, iy_f] = True
                                continue
                        fill_1_mask[ix_f, iy_f] = True
                if np.any(fill_2_mask):
                    pls_jb_p0 = _tile_boolean_layer_with_rects(
                        fill_2_mask,
                        z0=z0_jb,
                        ox=ph_low_f[0],
                        oy=ph_low_f[1],
                        templates=plate_templates,
                        prefer_x_major=True,
                        parity_shift=0,
                        rem_inv=rem_inv,
                        supported_below_xy=fill_2_mask,
                    )
                    pls_jb_t1 = _tile_boolean_layer_with_rects(
                        fill_2_mask,
                        z0=z0_jb + 8,
                        ox=ph_low_f[0],
                        oy=ph_low_f[1],
                        templates=tile_templates,
                        prefer_x_major=False,
                        parity_shift=0,
                        rem_inv=rem_inv,
                        supported_below_xy=fill_2_mask,
                    )
                    if pls_jb_p0 and pls_jb_t1:
                        planned_by_z.setdefault(z0_jb, []).extend(pls_jb_p0)
                        planned_by_z.setdefault(z0_jb + 8, []).extend(pls_jb_t1)
                        for pid_n, pose_n in pls_jb_p0 + pls_jb_t1:
                            all_planned_boxes.extend(b_n.transform(pose_n) for b_n in get_brick_primitive(pid_n).hull_boxes)
                if np.any(fill_1_mask):
                    pls_jb_t0 = _tile_boolean_layer_with_rects(
                        fill_1_mask,
                        z0=z0_jb,
                        ox=ph_low_f[0],
                        oy=ph_low_f[1],
                        templates=tile_templates,
                        prefer_x_major=True,
                        parity_shift=0,
                        rem_inv=rem_inv,
                        supported_below_xy=fill_1_mask,
                    )
                    if pls_jb_t0:
                        planned_by_z.setdefault(z0_jb, []).extend(pls_jb_t0)
                        for pid_n, pose_n in pls_jb_t0:
                            all_planned_boxes.extend(b_n.transform(pose_n) for b_n in get_brick_primitive(pid_n).hull_boxes)

        # Tile remaining `8 LDU` plate layers (tiling `must_be_smooth_tile_occ` corner cells with `3070b` smooth tiles!)
        for p in range(n_plates_z):
            if _early_deadline_hit():
                return None
            if p in jumper_bridge_layers:
                continue
            z0 = p * 8
            if np.any(rem_core[:, :, p]):
                ph_p = layer_phase[p]
                sup_p = _support_mask_at_layer(p)
                sm_cells_p = rem_core[:, :, p] & must_be_smooth_tile_occ[:, :, p]
                if np.any(sm_cells_p) and tile_templates:
                    t_corner_pls = _tile_boolean_layer_with_rects(
                        sm_cells_p,
                        z0=z0,
                        ox=ph_p[0],
                        oy=ph_p[1],
                        templates=[t for t in tile_templates if t[1] == 1 and t[2] == 1] or tile_templates,
                        prefer_x_major=(p % 2 == 1),
                        parity_shift=((p // 2) % 2),
                        rem_inv=rem_inv,
                        supported_below_xy=sup_p,
                    )
                    if t_corner_pls:
                        planned_by_z.setdefault(z0, []).extend(t_corner_pls)
                reg_cells_p = rem_core[:, :, p] & (~must_be_smooth_tile_occ[:, :, p])
                if np.any(reg_cells_p):
                    if (p - 1) in jumper_bridge_layers:
                        jp_from, jp_to = jumper_bridge_layers[p - 1]
                        pref_xp = jp_to[0] != jp_from[0]
                        par_p = 0
                    else:
                        pref_xp = p % 2 == 1
                        par_p = (p // 2) % 2
                    p_pls = _tile_boolean_layer_with_rects(
                        reg_cells_p,
                        z0=z0,
                        ox=ph_p[0],
                        oy=ph_p[1],
                        templates=plate_templates,
                        prefer_x_major=pref_xp,
                        parity_shift=par_p,
                        rem_inv=rem_inv,
                        supported_below_xy=sup_p,
                        shell_running_bond=is_hollow_shell_active,
                    )
                    if p_pls:
                        planned_by_z.setdefault(z0, []).extend(p_pls)


        return int(len(all_sl_cands))
