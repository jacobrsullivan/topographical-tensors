"""
Stage 3a: Phase-aware `8 LDU` plate-layer + `+10 LDU` jumper eroded core, hollow interior girders,
and `30414`/`11211`/`87087` SNOT perimeter backing planner (`Phase 15`).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np

from topo_brick.evaluation.static_equilibrium import FoundationPolicy
from topo_brick.interface.brick_catalog import (
    EXTENDED_BRICK_CATALOG,
    PortPolarity,
    get_brick_primitive,
)
from topo_brick.interface.catalog_retrieval_index import (
    CATALOG_RETRIEVAL_INDEX,
    SCULPTURAL_TAXONOMY_CLASSES,
)
from topo_brick.interface.ldu_frames import C4_Z_ROT_IDS, LDUBox, LDUFrame
from topo_brick.kernel.sculpted_primitives import (
    _box_hits_forbidden,
    _fast_upward_stud_templates,
    _rect_templates,
)


@dataclass
class ErodedCorePlan:
    """Planned bottom-up structural & SNOT courses produced by `ErodedCorePlanningStage`."""

    planned_by_z: Dict[int, List[Tuple[str, LDUFrame]]]
    layer_phase: List[Tuple[int, int]]
    best_ground_phase: Tuple[int, int]
    phase_grids: Dict[Tuple[int, int], np.ndarray]
    core_occ_pre_sculpt: np.ndarray
    is_thin_regime: bool
    is_legged_frame: bool
    eff_gz6_sky_limit: int
    w_x: int
    w_y: int
    core_cell_thresh: int
    fast_est_plan_occ6_iou: Callable[..., float]
    s3_poses_scored: int = 0


class ErodedCorePlanningStage:
    """Plan phase-aware eroded core layers, hollow interior girders, and SNOT backing."""

    @staticmethod
    def plan(
        *,
        prep: Any,
        adaptive_target: Any,
        eff_shell_policy: Optional[Any],
        support_policy: str,
        foundation_policy: FoundationPolicy,
        forbidden_support_boxes: Sequence[LDUBox],
        license_filter: str,
        view_priority: Optional[Any],
        deadline_monotonic_s: Optional[float],
        _forced_plan: Optional[Any],
    ) -> Optional[ErodedCorePlan]:
        nx_s = prep.nx_s
        ny_s = prep.ny_s
        n_plates_z = prep.n_plates_z
        occ6 = prep.occ6
        occ2 = prep.occ2
        solid_occ6_padded = prep.solid_occ6_padded
        gz6_nonempty_limit = prep.gz6_nonempty_limit
        eff_inv_map = prep.eff_inv_map
        kit_ids = prep.kit_ids
        kit_set = prep.kit_set
        rem_inv = dict(prep.rem_inv) if prep.rem_inv is not None else None
        view_weight_grid_6 = prep.view_weight_grid_6
        is_hollow_shell_active = prep.is_hollow_shell_active

        def _early_deadline_hit() -> bool:
            return deadline_monotonic_s is not None and time.monotonic() >= deadline_monotonic_s

        # 3. Phase-Aware `8 LDU` Plate-Layer + `+10 LDU` Jumper Eroded Core Synthesis
        #    Check ground sub-stud phase `(ox0, oy0)` on `CLUTCHED_BASEPLATE` (`(0,0), (10,0), (0,10), (10,10)`)
        def _cell_counts_at_phase(ox: int, oy: int) -> np.ndarray:
            """Return `(nx_s - 1, ny_s - 1, n_plates_z)` int16 sub-voxel counts (`0..64`) on `occ2`."""
            sx_0 = ox // 5
            sy_0 = oy // 5
            w_x = nx_s - 1
            w_y = ny_s - 1
            sub = occ2[sx_0 : sx_0 + 4 * w_x, sy_0 : sy_0 + 4 * w_y, : 4 * n_plates_z]
            return np.sum(
                sub.reshape(w_x, 4, w_y, 4, n_plates_z, 4), axis=(1, 3, 5), dtype=np.int16
            )

        phase_grids: Dict[Tuple[int, int], np.ndarray] = {
            (ox, oy): _cell_counts_at_phase(ox, oy)
            for ox in (0, 10)
            for oy in (0, 10)
        }

        # Detect thin-wireframe / thin-slab / narrow-blade regime purely from sub-voxel occupancy density
        all_g_phases = (
            [(0, 0), (10, 0), (0, 10), (10, 10)]
            if foundation_policy == FoundationPolicy.CLUTCHED_BASEPLATE
            else [(0, 0)]
        )
        max_cg_48 = max(int(np.count_nonzero(phase_grids[ph] >= 48)) for ph in all_g_phases)
        max_cg_32 = max(int(np.count_nonzero(phase_grids[ph] >= 32)) for ph in all_g_phases)
        max_cg_16 = max(int(np.count_nonzero(phase_grids[ph] >= 16)) for ph in all_g_phases)
        ground_c24 = int(np.count_nonzero(phase_grids[(0, 0)][:, :, 0] >= 24))
        peak_c24 = max(
            (int(np.count_nonzero(phase_grids[(0, 0)][:, :, q_i] >= 24)) for q_i in range(n_plates_z)),
            default=0,
        )
        ground_c16 = int(np.count_nonzero(phase_grids[(0, 0)][:, :, 0] >= 16))
        peak_c16 = max(
            (int(np.count_nonzero(phase_grids[(0, 0)][:, :, q_i] >= 16)) for q_i in range(n_plates_z)),
            default=0,
        )
        mesh_sub_total_init = int(np.count_nonzero(occ6))
        is_legged_frame = (
            (
                (peak_c24 >= 6 and ground_c24 <= int(0.26 * peak_c24))
                or (peak_c16 >= 6 and ground_c16 <= max(1, int(0.28 * peak_c16)))
            )
            and (not forbidden_support_boxes)
        )
        is_thin_regime = (
            is_legged_frame
            or (max_cg_48 <= 4)
            or (max_cg_32 < max(8, int(0.38 * max_cg_16)))
            or (mesh_sub_total_init < 720 and max_cg_48 <= 16)
        )
        eff_gz6_sky_limit = gz6_nonempty_limit
        if not is_thin_regime and gz6_nonempty_limit >= 10:
            cnt_sky_top = int(np.count_nonzero(occ6[:, :, gz6_nonempty_limit - 1]))
            cnt_sky_bel = int(np.count_nonzero(occ6[:, :, gz6_nonempty_limit - 2]))
            if cnt_sky_bel >= 32 and cnt_sky_top * 6 <= cnt_sky_bel:
                eff_gz6_sky_limit = gz6_nonempty_limit - 1
        candidates_g_phase = all_g_phases

        # Choose ground phase `(ox_cur, oy_cur)`: on thick multi-tier meshes inspect Course 0 (`z = 0..24`)
        # so shifted upper tiers never bias the baseplate; on thin-frame / narrow-wall `.obj` meshes align the whole frame!
        # Penalize orphaned sub-voxels in `occ2` falling outside `[ox, ox + 20*w_x] x [oy, oy + 20*w_y]`.
        best_ground_phase = (0, 0)
        best_g_score = -1e9
        wx4 = 4 * (nx_s - 1)
        wy4 = 4 * (ny_s - 1)
        is_narrow_xy = min(nx_s - 1, ny_s - 1) <= 2
        for ph in candidates_g_phase:
            sx0_p = ph[0] // 5
            sy0_p = ph[1] // 5
            if is_thin_regime or is_narrow_xy:
                z_lim2 = min(occ2.shape[2], 4 * n_plates_z)
                orphan_cnt = (
                    int(np.count_nonzero(occ2[:sx0_p, :, :z_lim2]))
                    + int(np.count_nonzero(occ2[sx0_p + wx4 :, :, :z_lim2]))
                    + int(np.count_nonzero(occ2[sx0_p : sx0_p + wx4, :sy0_p, :z_lim2]))
                    + int(np.count_nonzero(occ2[sx0_p : sx0_p + wx4, sy0_p + wy4 :, :z_lim2]))
                )
                cg = phase_grids[ph]
                active_thin = cg >= 14
                sc = (
                    float(np.sum(2 * cg[active_thin].astype(np.int32) - 24))
                    - 3.5 * float(orphan_cnt)
                    - (8.0 if ph != (0, 0) else 0.0)
                )
            else:
                z_lim2 = min(occ2.shape[2], 4 * min(3, n_plates_z))
                orphan_cnt = (
                    int(np.count_nonzero(occ2[:sx0_p, :, :z_lim2]))
                    + int(np.count_nonzero(occ2[sx0_p + wx4 :, :, :z_lim2]))
                    + int(np.count_nonzero(occ2[sx0_p : sx0_p + wx4, :sy0_p, :z_lim2]))
                    + int(np.count_nonzero(occ2[sx0_p : sx0_p + wx4, sy0_p + wy4 :, :z_lim2]))
                )
                cg = phase_grids[ph][:, :, : min(3, n_plates_z)]
                solid = cg >= 36
                ambig = (cg >= 16) & (cg < 36)
                sc = (
                    float(np.sum(2 * cg[solid].astype(np.int32) - 64))
                    - 12.0 * float(np.count_nonzero(ambig))
                    - 3.0 * float(orphan_cnt)
                    - (4.0 if ph != (0, 0) else 0.0)
                )
            if sc > best_g_score:
                best_g_score = sc
                best_ground_phase = ph

        w_x = nx_s - 1
        w_y = ny_s - 1
        has_snot_core = any(sp in kit_set for sp in ("30414", "11211", "87087", "44728", "4070", "99780"))
        has_thin_panel_kit = any(tp in kit_set for tp in ("4865a", "23950", "30413", "93095", "4864a", "2362a", "4215a"))

        def _build_initial_core_occ(
            l_phases: Sequence[Tuple[int, int]],
            j_bridges: Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]],
            thr_use: int,
        ) -> np.ndarray:
            c_occ = np.zeros((w_x, w_y, n_plates_z), dtype=bool)
            for p in range(n_plates_z):
                if p in j_bridges:
                    continue
                ph_use = l_phases[p]
                cg_p = phase_grids[ph_use][:, :, p]
                z6_prev_p = min(occ6.shape[2], int(math.ceil((8 * p) / 6.0)))
                z6_cur_p = min(occ6.shape[2], int(math.ceil((8 * (p + 1)) / 6.0)))
                for ix in range(w_x):
                    for iy in range(w_y):
                        val_c = int(cg_p[ix, iy])
                        take_cell = False
                        if view_weight_grid_6 is not None and val_c >= 12:
                            x6_vw = (ix * 20 + ph_use[0]) // 5
                            y6_vw = (iy * 20 + ph_use[1]) // 5
                            z6_vw0 = max(0, (8 * p) // 6)
                            z6_vw1 = min(occ6.shape[2], int(math.ceil((8 * (p + 1)) / 6.0)))
                            if z6_vw1 > z6_vw0:
                                t_vw = occ6[x6_vw : x6_vw + 4, y6_vw : y6_vw + 4, z6_vw0:z6_vw1]
                                w_vw = view_weight_grid_6[x6_vw : x6_vw + 4, y6_vw : y6_vw + 4, z6_vw0:z6_vw1]
                                in_vw = float(np.sum(w_vw[t_vw]))
                                ov_vw = float(np.sum(w_vw[~t_vw]))
                                has_vis_fg = bool(np.any(t_vw) and float(np.max(w_vw[t_vw])) >= 1.5)
                                if has_vis_fg and in_vw >= 0.95 * ov_vw:
                                    take_cell = True
                                elif val_c >= thr_use and ov_vw <= 1.45 * in_vw:
                                    take_cell = True
                        elif val_c >= thr_use:
                            take_cell = True
                        elif (
                            p > 0
                            and (p - 1) not in j_bridges
                            and l_phases[p - 1] == ph_use
                            and c_occ[ix, iy, p - 1]
                            and val_c >= 12
                            and z6_cur_p > z6_prev_p
                        ):
                            # Top-cap partial-plate recovery (`e.g. z=[96..100]` where `d_in >= d_ov` on `occ6`)
                            x6_c = (ix * 20 + ph_use[0]) // 5
                            y6_c = (iy * 20 + ph_use[1]) // 5
                            blk_cap = occ6[x6_c : x6_c + 4, y6_c : y6_c + 4, z6_prev_p:z6_cur_p]
                            is_roof_cap_cell = (p + 1) >= n_plates_z or int(phase_grids[ph_use][ix, iy, p + 1]) < 8
                            cap_frac_req = 0.50 if is_roof_cap_cell else max(0.50, float(thr_use) / 64.0)
                            if blk_cap.size > 0 and float(np.count_nonzero(blk_cap)) >= cap_frac_req * float(blk_cap.size):
                                take_cell = True
                        elif is_thin_regime and not is_legged_frame and thr_use <= 32 and val_c >= 8:
                            col_z = phase_grids[ph_use][ix, iy, :]
                            has_solid_below = p > 0 and bool(np.any(col_z[:p] >= 24))
                            has_solid_above = (p + 1) < n_plates_z and bool(np.any(col_z[p + 1 :] >= 24))
                            if has_solid_below and has_solid_above:
                                take_cell = True
                        if not take_cell and is_thin_regime and thr_use <= 32 and p >= 3 and val_c >= 6:
                            # Capture elevated horizontal thin slabs (`e.g. thin tabletops`) including perimeter/corner cells
                            sx0_c = (ix * 20 + ph_use[0]) // 5
                            sy0_c = (iy * 20 + ph_use[1]) // 5
                            sub_xy = occ2[sx0_c : sx0_c + 4, sy0_c : sy0_c + 4, 4 * p : 4 * (p + 1)]
                            x_span_5 = int(np.count_nonzero(np.any(sub_xy, axis=(1, 2))))
                            y_span_5 = int(np.count_nonzero(np.any(sub_xy, axis=(0, 2))))
                            nbr_12 = sum(
                                1
                                for dx_n, dy_n in ((1, 0), (-1, 0), (0, 1), (0, -1))
                                if 0 <= ix + dx_n < w_x
                                and 0 <= iy + dy_n < w_y
                                and int(cg_p[ix + dx_n, iy + dy_n]) >= 8
                            )
                            if x_span_5 >= 2 and y_span_5 >= 2 and nbr_12 >= 1 and int(np.count_nonzero(cg_p >= 8)) >= 6:
                                take_cell = True
                        if take_cell:
                            wb = LDUBox(
                                ix * 20 + ph_use[0],
                                iy * 20 + ph_use[1],
                                p * 8,
                                (ix + 1) * 20 + ph_use[0],
                                (iy + 1) * 20 + ph_use[1],
                                (p + 1) * 8,
                            )
                            if not _box_hits_forbidden(wb, forbidden_support_boxes):
                                c_occ[ix, iy, p] = True

            if is_thin_regime:
                for p_sl in range(5, n_plates_z - 1):
                    if p_sl in j_bridges or (p_sl + 1) in j_bridges or l_phases[p_sl] != l_phases[p_sl + 1]:
                        continue
                    ph_sl = l_phases[p_sl]
                    n_a = int(np.count_nonzero(c_occ[:, :, p_sl]))
                    n_b = int(np.count_nonzero(c_occ[:, :, p_sl + 1]))
                    if n_a >= 4 and n_b >= 4:
                        def _layer_occ6_mask(p_eval: int) -> np.ndarray:
                            m6 = np.zeros_like(occ6)
                            z6_0 = max(0, int(math.floor((p_eval * 8) / 6.0)))
                            z6_1 = min(occ6.shape[2], int(math.ceil(((p_eval + 1) * 8) / 6.0)))
                            for ix_e in range(w_x):
                                for iy_e in range(w_y):
                                    if c_occ[ix_e, iy_e, p_eval]:
                                        x6_0 = (ix_e * 20 + ph_sl[0]) // 5
                                        y6_0 = (iy_e * 20 + ph_sl[1]) // 5
                                        m6[x6_0 : x6_0 + 4, y6_0 : y6_0 + 4, z6_0:z6_1] = True
                            return m6

                        ma = _layer_occ6_mask(p_sl)
                        mb = _layer_occ6_mask(p_sl + 1)
                        mab = ma | mb
                        net_a = 2 * int(np.count_nonzero(ma & occ6)) - int(np.count_nonzero(ma))
                        net_b = 2 * int(np.count_nonzero(mb & occ6)) - int(np.count_nonzero(mb))
                        net_ab = 2 * int(np.count_nonzero(mab & occ6)) - int(np.count_nonzero(mab))
                        if max(net_a, net_b) >= net_ab + 16:
                            if net_a >= net_b:
                                c_occ[:, :, p_sl + 1] = False
                            else:
                                c_occ[:, :, p_sl] = False
            return c_occ

        # Pure-NumPy `(layer_phase, jumper_bridge_layers, core_cell_thresh)` model selector directly on `occ6` (`< 3 ms`)
        default_thr = 22 if is_thin_regime else (24 if max_cg_48 < 14 else 30)
        if (
            not forbidden_support_boxes
            and foundation_policy == FoundationPolicy.CLUTCHED_BASEPLATE
            and (is_thin_regime or is_narrow_xy)
        ):
            cand_phases_sel = all_g_phases
        else:
            cand_phases_sel = [best_ground_phase]

        if is_thin_regime:
            cand_threshs_sel = [16, 22, 30]
        elif not forbidden_support_boxes:
            cand_threshs_sel = [22, 28, 33]
        else:
            cand_threshs_sel = [default_thr]

        def _est_core_occ6_iou(
            c_arr: np.ndarray,
            l_phases_c: Sequence[Tuple[int, int]],
            j_bridges_c: Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]],
        ) -> float:
            co6 = np.zeros_like(occ6)
            for p_c0 in range(n_plates_z):
                if p_c0 in j_bridges_c or not np.any(c_arr[:, :, p_c0]):
                    continue
                ph_c0 = l_phases_c[p_c0]
                z6_c0 = max(0, (p_c0 * 8) // 6)
                z6_c1 = min(occ6.shape[2], int(math.ceil(((p_c0 + 1) * 8) / 6.0)))
                for ix_c0 in range(w_x):
                    for iy_c0 in range(w_y):
                        if c_arr[ix_c0, iy_c0, p_c0]:
                            x6_c0 = (ix_c0 * 20 + ph_c0[0]) // 5
                            y6_c0 = (iy_c0 * 20 + ph_c0[1]) // 5
                            co6[x6_c0 : x6_c0 + 4, y6_c0 : y6_c0 + 4, z6_c0:z6_c1] = True
            m_sub = max(1, int(np.count_nonzero(occ6)))
            c_in = int(np.count_nonzero(co6 & occ6))
            c_ov = int(np.count_nonzero(co6 & (~occ6)))
            return float(c_in) / float(max(1, m_sub + c_ov))

        def _fast_est_plan_occ6_iou(
            l_phases_e: Sequence[Tuple[int, int]],
            j_bridges_e: Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]],
            thr_e: int,
        ) -> float:
            c_e = _build_initial_core_occ(l_phases_e, j_bridges_e, thr_e)
            if is_thin_regime:
                marg_r = 0.20 if has_thin_panel_kit else 0.35
            else:
                iou_c0 = _est_core_occ6_iou(c_e, l_phases_e, j_bridges_e)
                marg_r = max(0.455, iou_c0 / (1.0 + iou_c0))
            for p_pr in range(n_plates_z - 1, 0, -1):
                if p_pr in j_bridges_e or (p_pr - 1) in j_bridges_e or l_phases_e[p_pr] != l_phases_e[p_pr - 1]:
                    continue
                ph_e = l_phases_e[p_pr]
                z6_p0 = min(occ6.shape[2], int(math.ceil((8 * p_pr) / 6.0)))
                z6_p1 = min(occ6.shape[2], int(math.ceil((8 * (p_pr + 1)) / 6.0)))
                if z6_p1 <= z6_p0:
                    continue
                for ix_p in range(w_x):
                    for iy_p in range(w_y):
                        if c_e[ix_p, iy_p, p_pr] and c_e[ix_p, iy_p, p_pr - 1] and not (
                            (p_pr + 1) < n_plates_z and np.any(c_e[ix_p, iy_p, p_pr + 1 :])
                        ):
                            x6_p = (ix_p * 20 + ph_e[0]) // 5
                            y6_p = (iy_p * 20 + ph_e[1]) // 5
                            bm = occ6[x6_p : x6_p + 4, y6_p : y6_p + 4, z6_p0:z6_p1]
                            if bm.size > 0 and float(np.count_nonzero(bm)) < marg_r * float(bm.size):
                                c_e[ix_p, iy_p, p_pr] = False
            if has_snot_core and not is_thin_regime and not forbidden_support_boxes:
                for c_l in range((n_plates_z // 3) - 1, -1, -1):
                    p0_e = 3 * c_l
                    if any((p0_e + q_e) in j_bridges_e for q_e in range(min(4, n_plates_z - p0_e))):
                        continue
                    ph_e = l_phases_e[p0_e]
                    if l_phases_e[p0_e + 1] != ph_e or l_phases_e[p0_e + 2] != ph_e:
                        continue
                    carve_e: Set[Tuple[int, int]] = set()
                    for ix_e in range(w_x):
                        for iy_e in range(w_y):
                            if not np.any(c_e[ix_e, iy_e, p0_e : p0_e + 3]):
                                continue
                            if (p0_e + 3) < n_plates_z and np.any(c_e[ix_e, iy_e, p0_e + 3 :]):
                                continue
                            sx_e = (ix_e * 20 + ph_e[0]) // 5
                            sy_e = (iy_e * 20 + ph_e[1]) // 5
                            bl = occ2[sx_e : sx_e + 4, sy_e : sy_e + 4, 4 * p0_e : 4 * (p0_e + 3)]
                            for dx_o, dy_o in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                                ox_i, oy_i = ix_e + dx_o, iy_e + dy_o
                                in_x, in_y = ix_e - dx_o, iy_e - dy_o
                                if 0 <= ox_i < w_x and 0 <= oy_i < w_y and np.any(c_e[ox_i, oy_i, p0_e : p0_e + 3]):
                                    continue
                                if not (0 <= in_x < w_x and 0 <= in_y < w_y and np.all(c_e[in_x, in_y, p0_e : p0_e + 3])):
                                    continue
                                if (in_x, in_y) in carve_e:
                                    continue
                                if dx_o == 1:
                                    op0 = int(np.count_nonzero(bl[2:, :, :4]))
                                    oc, ic = int(np.count_nonzero(bl[2:, :, :])), int(np.count_nonzero(bl[:2, :, :]))
                                elif dx_o == -1:
                                    op0 = int(np.count_nonzero(bl[:2, :, :4]))
                                    oc, ic = int(np.count_nonzero(bl[:2, :, :])), int(np.count_nonzero(bl[2:, :, :]))
                                elif dy_o == 1:
                                    op0 = int(np.count_nonzero(bl[:, 2:, :4]))
                                    oc, ic = int(np.count_nonzero(bl[:, 2:, :])), int(np.count_nonzero(bl[:, :2, :]))
                                else:
                                    op0 = int(np.count_nonzero(bl[:, :2, :4]))
                                    oc, ic = int(np.count_nonzero(bl[:, :2, :])), int(np.count_nonzero(bl[:, 2:, :]))
                                if op0 <= 6 and oc <= 18 and ic >= 28:
                                    carve_e.add((ix_e, iy_e))
                                    break
                    for cx_e, cy_e in carve_e:
                        c_e[cx_e, cy_e, p0_e : p0_e + 3] = False

            m6_e = np.zeros_like(occ6)
            j_top_e: Dict[int, np.ndarray] = {}
            for p_e in range(n_plates_z):
                z6_a = max(0, (p_e * 8) // 6)
                z6_b = min(occ6.shape[2], int(math.ceil(((p_e + 1) * 8) / 6.0)))
                if p_e in j_bridges_e:
                    ph_f_j, ph_t_j = j_bridges_e[p_e]
                    sh_x = ph_t_j[0] != ph_f_j[0]
                    low_j = (
                        j_top_e[p_e - 1]
                        if (p_e - 1) in j_top_e
                        else (c_e[:, :, p_e - 1] if p_e > 0 else np.zeros((w_x, w_y), dtype=bool))
                    )
                    p_u_e = p_e + 1
                    while p_u_e < n_plates_z and p_u_e in j_bridges_e:
                        p_u_e += 1
                    if p_u_e < n_plates_z:
                        ph_fin_e = l_phases_e[p_u_e]
                        raw_u_e = c_e[:, :, p_u_e]
                        if ph_fin_e == ph_t_j:
                            up_j = raw_u_e
                        else:
                            up_j = np.zeros((w_x, w_y), dtype=bool)
                            uxs_e, uys_e = np.nonzero(raw_u_e)
                            uys_list_e = sorted(set(uys_e.tolist()))
                            trim_uy_e: Set[int] = set()
                            if sh_x and len(uys_list_e) >= 4:
                                cg_le = phase_grids[ph_f_j][:, :, p_e]
                                if ((uys_list_e[0] * 20 + ph_fin_e[1] - ph_t_j[1]) // 20) == 0 and int(np.max(cg_le[:, 0])) < 54:
                                    trim_uy_e.add(uys_list_e[0])
                                if (((uys_list_e[-1] * 20 + ph_fin_e[1] - ph_t_j[1]) // 20) + 1) == (w_y - 1) and int(np.max(cg_le[:, w_y - 1])) < 54:
                                    trim_uy_e.add(uys_list_e[-1])
                            for uxf_e, uyf_e in zip(uxs_e.tolist(), uys_e.tolist()):
                                if sh_x:
                                    if uyf_e in trim_uy_e:
                                        continue
                                    iym_e = (uyf_e * 20 + ph_fin_e[1] - ph_t_j[1]) // 20
                                    for iyc_e in (iym_e, iym_e + 1):
                                        if 0 <= iyc_e < w_y:
                                            up_j[uxf_e, iyc_e] = True
                                else:
                                    ixm_e = (uxf_e * 20 + ph_fin_e[0] - ph_t_j[0]) // 20
                                    for ixc_e in (ixm_e, ixm_e + 1):
                                        if 0 <= ixc_e < w_x:
                                            up_j[ixc_e, uyf_e] = True
                    else:
                        up_j = np.zeros((w_x, w_y), dtype=bool)
                    top_j = np.zeros((w_x, w_y), dtype=bool)
                    used_l_j = np.zeros((w_x, w_y), dtype=bool)
                    for ux_j in range(w_x):
                        for uy_j in range(w_y):
                            if not up_j[ux_j, uy_j]:
                                continue
                            if sh_x:
                                w_cx = ux_j * 20 + ph_t_j[0] + 10
                                ix_l = (w_cx - 10 - ph_f_j[0]) // 20
                                if (
                                    0 <= ix_l
                                    and (ix_l + 1) < w_x
                                    and not used_l_j[ix_l, uy_j]
                                    and not used_l_j[ix_l + 1, uy_j]
                                    and low_j[ix_l, uy_j]
                                    and low_j[ix_l + 1, uy_j]
                                ):
                                    x6_j = (ix_l * 20 + ph_f_j[0]) // 5
                                    y6_j = (uy_j * 20 + ph_f_j[1]) // 5
                                    m6_e[x6_j : x6_j + 8, y6_j : y6_j + 4, z6_a:z6_b] = True
                                    used_l_j[ix_l, uy_j] = True
                                    used_l_j[ix_l + 1, uy_j] = True
                                    top_j[ux_j, uy_j] = True
                            else:
                                w_cy = uy_j * 20 + ph_t_j[1] + 10
                                iy_l = (w_cy - 10 - ph_f_j[1]) // 20
                                if (
                                    0 <= iy_l
                                    and (iy_l + 1) < w_y
                                    and not used_l_j[ux_j, iy_l]
                                    and not used_l_j[ux_j, iy_l + 1]
                                    and low_j[ux_j, iy_l]
                                    and low_j[ux_j, iy_l + 1]
                                ):
                                    x6_j = (ux_j * 20 + ph_f_j[0]) // 5
                                    y6_j = (iy_l * 20 + ph_f_j[1]) // 5
                                    m6_e[x6_j : x6_j + 4, y6_j : y6_j + 8, z6_a:z6_b] = True
                                    used_l_j[ux_j, iy_l] = True
                                    used_l_j[ux_j, iy_l + 1] = True
                                    top_j[ux_j, uy_j] = True
                    # Include terrace smooth-tile caps on exposed lower-tier shoulders
                    cg_ter_e = phase_grids[ph_f_j][:, :, p_e]
                    for tx_e in range(w_x):
                        for ty_e in range(w_y):
                            if low_j[tx_e, ty_e] and not used_l_j[tx_e, ty_e] and int(cg_ter_e[tx_e, ty_e]) >= 42:
                                x6_t = (tx_e * 20 + ph_f_j[0]) // 5
                                y6_t = (ty_e * 20 + ph_f_j[1]) // 5
                                t_sl_e = occ6[x6_t : x6_t + 4, y6_t : y6_t + 4, z6_a:z6_b]
                                if t_sl_e.size > 0 and 2 * int(np.count_nonzero(t_sl_e)) >= int(t_sl_e.size):
                                    m6_e[x6_t : x6_t + 4, y6_t : y6_t + 4, z6_a:z6_b] = True
                    j_top_e[p_e] = top_j
                    continue
                if not np.any(c_e[:, :, p_e]):
                    continue
                ph_e = l_phases_e[p_e]
                for ix_e in range(w_x):
                    for iy_e in range(w_y):
                        x6_e = (ix_e * 20 + ph_e[0]) // 5
                        y6_e = (iy_e * 20 + ph_e[1]) // 5
                        if c_e[ix_e, iy_e, p_e]:
                            m6_e[x6_e : x6_e + 4, y6_e : y6_e + 4, z6_a:z6_b] = True
                        elif (
                            is_thin_regime
                            and has_thin_panel_kit
                            and p_e >= 2
                            and np.any(c_e[ix_e, iy_e, :p_e])
                            and x6_e + 4 <= occ6.shape[0]
                            and y6_e + 4 <= occ6.shape[1]
                        ):
                            # Truthfully score 5 LDU perimeter thin-panel walls above supported base cells
                            for wx0, wx1, wy0, wy1 in (
                                (x6_e, x6_e + 1, y6_e, y6_e + 4),
                                (x6_e + 3, x6_e + 4, y6_e, y6_e + 4),
                                (x6_e, x6_e + 4, y6_e, y6_e + 1),
                                (x6_e, x6_e + 4, y6_e + 3, y6_e + 4),
                            ):
                                w_sl = occ6[wx0:wx1, wy0:wy1, z6_a:z6_b]
                                w_in = int(np.count_nonzero(w_sl))
                                w_ov = int(w_sl.size) - w_in
                                if w_in >= 2 and w_in * 3 >= w_ov:
                                    m6_e[wx0:wx1, wy0:wy1, z6_a:z6_b] = True
                                    break
            if has_snot_core and not is_thin_regime:
                for c_l in range(n_plates_z // 3):
                    p0_e = 3 * c_l
                    if any((p0_e + q_e) in j_bridges_e for q_e in range(3)):
                        continue
                    ph_e = l_phases_e[p0_e]
                    if l_phases_e[p0_e + 1] != ph_e or l_phases_e[p0_e + 2] != ph_e:
                        continue
                    z6_f0 = max(0, (p0_e * 8) // 6)
                    z6_f1 = min(occ6.shape[2], (p0_e * 8 + 24) // 6)
                    for ix_e in range(w_x):
                        for iy_e in range(w_y):
                            if not np.all(c_e[ix_e, iy_e, p0_e : p0_e + 3]):
                                continue
                            fx0_e, fx1_e = ix_e * 20 + ph_e[0], (ix_e + 1) * 20 + ph_e[0]
                            fy0_e, fy1_e = iy_e * 20 + ph_e[1], (iy_e + 1) * 20 + ph_e[1]
                            for dx_o, dy_o, mx0, mx1, my0, my1 in (
                                (1, 0, fx1_e // 5, fx1_e // 5 + 2, fy0_e // 5, fy1_e // 5),
                                (-1, 0, fx0_e // 5 - 2, fx0_e // 5, fy0_e // 5, fy1_e // 5),
                                (0, 1, fx0_e // 5, fx1_e // 5, fy1_e // 5, fy1_e // 5 + 2),
                                (0, -1, fx0_e // 5, fx1_e // 5, fy0_e // 5 - 2, fy0_e // 5),
                            ):
                                ox_i, oy_i = ix_e + dx_o, iy_e + dy_o
                                if 0 <= ox_i < w_x and 0 <= oy_i < w_y and np.any(c_e[ox_i, oy_i, p0_e : p0_e + 3]):
                                    continue
                                if 0 <= mx0 and 0 <= my0 and mx1 <= occ6.shape[0] and my1 <= occ6.shape[1]:
                                    ms_e = occ6[mx0:mx1, my0:my1, z6_f0:z6_f1]
                                    if ms_e.size > 0 and int(np.count_nonzero(ms_e)) >= int(0.40 * ms_e.size):
                                        m6_e[mx0:mx1, my0:my1, z6_f0:z6_f1] = True
            if view_weight_grid_6 is not None:
                inter_we = float(np.sum(view_weight_grid_6[m6_e & occ6]))
                union_we = float(np.sum(view_weight_grid_6[m6_e | occ6]))
                return inter_we / max(1.0, union_we)
            inter_e = int(np.count_nonzero(m6_e & occ6))
            union_e = int(np.count_nonzero(m6_e | occ6))
            return float(inter_e) / float(max(1, union_e))

        if _forced_plan is not None:
            layer_phase = list(_forced_plan[0])
            jumper_bridge_layers = dict(_forced_plan[1])
            core_cell_thresh = int(_forced_plan[2])
        else:
            best_sel_iou = -1.0
            core_cell_thresh = default_thr
            is_box_megastructure = (w_x * w_y) >= 169 and adaptive_target.mesh_surf_pts.shape[0] == 0
            if is_box_megastructure and view_priority is None:
                eff_phases_sel = (best_ground_phase,)
                eff_threshs_sel = (default_thr,)
            else:
                eff_phases_sel = ((0, 0), (10, 0), (0, 10), (10, 10)) if view_priority is not None else cand_phases_sel
                eff_threshs_sel = (18, 24, 30, 36, 42) if view_priority is not None else cand_threshs_sel
            if len(eff_phases_sel) == 1 and len(eff_threshs_sel) == 1:
                best_ground_phase = eff_phases_sel[0]
                core_cell_thresh = eff_threshs_sel[0]
            else:
                for ph_try in eff_phases_sel:
                    for thr_try in eff_threshs_sel:
                        est_iou = _fast_est_plan_occ6_iou([ph_try] * n_plates_z, {}, thr_try)
                        if ph_try == best_ground_phase:
                            est_iou += 2.0e-4
                        if thr_try == default_thr:
                            est_iou += 1.0e-4
                        if est_iou > best_sel_iou:
                            best_sel_iou = est_iou
                            best_ground_phase = ph_try
                            core_cell_thresh = thr_try

            layer_phase = [best_ground_phase] * n_plates_z
            jumper_bridge_layers = {}

        core_occ = _build_initial_core_occ(layer_phase, jumper_bridge_layers, core_cell_thresh)
        pier_protected = np.zeros((w_x, w_y, n_plates_z), dtype=bool)
        pillar_added_occ = np.zeros((w_x, w_y, n_plates_z), dtype=bool)

        # Pre-protect pier columns bordering forbidden architectural openings so `inv_arch_molds` never carves portal piers
        if forbidden_support_boxes:
            for fb in forbidden_support_boxes:
                p_fb_max = min(n_plates_z, int(math.ceil(fb.zmax / 8.0)) + 1)
                for p_fb in range(p_fb_max):
                    ph_fb = layer_phase[p_fb]
                    fx0_i = max(0, (fb.xmin - ph_fb[0]) // 20 - 1)
                    fx1_i = min(w_x, (fb.xmax - ph_fb[0] + 19) // 20 + 1)
                    fy0_i = max(0, (fb.ymin - ph_fb[1]) // 20 - 1)
                    fy1_i = min(w_y, (fb.ymax - ph_fb[1] + 19) // 20 + 1)
                    pier_protected[fx0_i:fx1_i, fy0_i:fy1_i, p_fb] = True

        # Reserve bottom-up Open-Catalog `INVERTED_SLOPE` & `ARCH` molds (`3660, 3665, 4287c,
        # 3659, 15254, 3308`, etc. with `num_top_studs >= 1`) on outward-flaring overhangs BEFORE dropping pillars!
        inv_reserved_occ = np.zeros((w_x, w_y, n_plates_z), dtype=bool)
        planned_inv_by_z: Dict[int, List[Tuple[str, LDUFrame]]] = {}
        inv_arch_molds = [
            pid
            for pid in kit_ids
            if CATALOG_RETRIEVAL_INDEX.get_entry(pid) is not None
            and CATALOG_RETRIEVAL_INDEX.get_entry(pid).taxonomy_class in ("INVERTED_SLOPE", "ARCH")  # type: ignore[union-attr]
            and CATALOG_RETRIEVAL_INDEX.get_entry(pid).num_top_studs >= 1  # type: ignore[union-attr]
            and len(get_brick_primitive(pid).hull_boxes) >= 2
        ]
        if inv_arch_molds:
            has_openings_inv = bool(forbidden_support_boxes)
            cand_p0_inv = (
                list(range(0, max(0, n_plates_z - 2)))
                if (foundation_policy == FoundationPolicy.CLUTCHED_BASEPLATE and has_openings_inv)
                else [3 * c_i for c_i in range(1, n_plates_z // 3)]
            )
            for p0_i in cand_p0_inv:
                if _early_deadline_hit():
                    return None
                if (p0_i > 0 and (p0_i - 1) in jumper_bridge_layers) or any(
                    (p0_i + q_i) in jumper_bridge_layers for q_i in range(3) if (p0_i + q_i) < n_plates_z
                ):
                    continue
                ph_i = layer_phase[p0_i]
                if p0_i > 0 and layer_phase[p0_i - 1] != ph_i:
                    continue
                z0_i = p0_i * 8
                inv_cands_course: List[Tuple[float, str, LDUFrame, List[Tuple[int, int, int]], List[Tuple[int, int, int]]]] = []
                seen_inv_poses: Set[Tuple[str, int, Tuple[int, int, int]]] = set()
                sup_xy_i = (
                    np.ones((w_x, w_y), dtype=bool)
                    if p0_i == 0
                    else core_occ[:, :, p0_i - 1]
                )
                if not sup_xy_i.any():
                    continue
                # Probe perimeter cells of `sup_xy_i` (or cells adjacent to `core_occ[:,:,p0_i]` portals when `p0_i == 0`)
                ref_perim_xy = core_occ[:, :, p0_i] if p0_i == 0 else sup_xy_i
                for ix_a in range(w_x):
                    for iy_a in range(w_y):
                        if not ref_perim_xy[ix_a, iy_a]:
                            continue
                        is_perim_a = any(
                            not (0 <= ix_a + dx_a < w_x and 0 <= iy_a + dy_a < w_y and ref_perim_xy[ix_a + dx_a, iy_a + dy_a])
                            for dx_a, dy_a in ((1, 0), (-1, 0), (0, 1), (0, -1))
                        )
                        if not is_perim_a:
                            continue
                        px_a = ix_a * 20 + ph_i[0] + 10
                        py_a = iy_a * 20 + ph_i[1] + 10
                        for pid_ia in inv_arch_molds:
                            if rem_inv is not None and rem_inv.get(pid_ia, 0) <= 0:
                                continue
                            prim_ia = get_brick_primitive(pid_ia)
                            ent_ia = CATALOG_RETRIEVAL_INDEX.get_entry(pid_ia)
                            is_arch_mold = ent_ia is not None and ent_ia.taxonomy_class == "ARCH"
                            if p0_i == 0 and not is_arch_mold:
                                continue
                            if (p0_i % 3) != 0 and not (has_openings_inv and is_arch_mold):
                                continue
                            for rid_ia, dx_ia, dy_ia, dz_ia, rboxes_ia in _fast_upward_stud_templates(pid_ia):
                                pos_ia = (px_a + dx_ia, py_a + dy_ia, z0_i + dz_ia)
                                ikey = (pid_ia, rid_ia, pos_ia)
                                if ikey in seen_inv_poses:
                                    continue
                                seen_inv_poses.add(ikey)
                                w_zmin = min(b[2] for b in rboxes_ia) + z0_i
                                w_zmax = max(b[5] for b in rboxes_ia) + z0_i
                                h_ldu_ia = w_zmax - w_zmin
                                if w_zmin != z0_i or h_ldu_ia <= 0 or (h_ldu_ia % 8) != 0:
                                    continue
                                h_pl = h_ldu_ia // 8
                                if p0_i + h_pl > n_plates_z:
                                    continue
                                if any(
                                    layer_phase[p0_i + q_h] != ph_i or (p0_i + q_h) in jumper_bridge_layers
                                    for q_h in range(h_pl)
                                ):
                                    continue
                                w_xmin = min(b[0] for b in rboxes_ia) + px_a
                                w_xmax = max(b[3] for b in rboxes_ia) + px_a
                                w_ymin = min(b[1] for b in rboxes_ia) + py_a
                                w_ymax = max(b[4] for b in rboxes_ia) + py_a
                                if (
                                    (w_xmin - ph_i[0]) % 20 != 0
                                    or (w_xmax - ph_i[0]) % 20 != 0
                                    or (w_ymin - ph_i[1]) % 20 != 0
                                    or (w_ymax - ph_i[1]) % 20 != 0
                                ):
                                    continue
                                rx0_i = (w_xmin - ph_i[0]) // 20
                                rx1_i = (w_xmax - ph_i[0]) // 20
                                ry0_i = (w_ymin - ph_i[1]) // 20
                                ry1_i = (w_ymax - ph_i[1]) // 20
                                if rx0_i < 0 or ry0_i < 0 or rx1_i > w_x or ry1_i > w_y:
                                    continue
                                if inv_reserved_occ[rx0_i:rx1_i, ry0_i:ry1_i, p0_i : p0_i + h_pl].any():
                                    continue
                                pose_ia = LDUFrame(rid_ia, pos_ia)
                                # Verify ALL bottom anti-studs rest on `sup_xy_i` (and `core_occ[:,:,p0_i]` if `p0_i == 0`) and >= 1 top stud bridges/overhangs!
                                all_bot_sup = True
                                n_bot = 0
                                top_stud_cells: List[Tuple[int, int, int]] = []
                                has_overhang = False
                                for pt_ia in prim_ia.ports:
                                    wf_pt = pose_ia.compose(pt_ia.local_frame)
                                    if pt_ia.polarity == PortPolarity.ANTI_STUD and wf_pt.z_axis == (0, 0, 1) and wf_pt.pos[2] == z0_i:
                                        bx_p = (wf_pt.pos[0] - ph_i[0]) // 20
                                        by_p = (wf_pt.pos[1] - ph_i[1]) // 20
                                        bot_ok = (
                                            (0 <= bx_p < w_x and 0 <= by_p < w_y and core_occ[bx_p, by_p, 0])
                                            if p0_i == 0
                                            else (0 <= bx_p < w_x and 0 <= by_p < w_y and sup_xy_i[bx_p, by_p])
                                        )
                                        if not bot_ok:
                                            all_bot_sup = False
                                            break
                                        n_bot += 1
                                    elif pt_ia.polarity == PortPolarity.STUD and wf_pt.z_axis == (0, 0, 1) and wf_pt.pos[2] == w_zmax:
                                        tx_p = (wf_pt.pos[0] - ph_i[0]) // 20
                                        ty_p = (wf_pt.pos[1] - ph_i[1]) // 20
                                        if 0 <= tx_p < w_x and 0 <= ty_p < w_y:
                                            top_stud_cells.append((tx_p, ty_p, p0_i + h_pl - 1))
                                            if (not core_occ[tx_p, ty_p, 0]) if p0_i == 0 else (not sup_xy_i[tx_p, ty_p]):
                                                has_overhang = True
                                if not all_bot_sup or n_bot == 0 or not has_overhang:
                                    continue
                                if (
                                    is_thin_regime
                                    and len(top_stud_cells) < (rx1_i - rx0_i) * (ry1_i - ry0_i)
                                ):
                                    uz_i0 = min(occ6.shape[2], int(math.ceil(w_zmax / 6.0)))
                                    uz_i1 = min(occ6.shape[2], uz_i0 + 4)
                                    sx_i0 = max(0, w_xmin // 5)
                                    sx_i1 = min(occ6.shape[0], w_xmax // 5)
                                    sy_i0 = max(0, w_ymin // 5)
                                    sy_i1 = min(occ6.shape[1], w_ymax // 5)
                                    if uz_i1 > uz_i0 and int(np.count_nonzero(occ6[sx_i0:sx_i1, sy_i0:sy_i1, uz_i0:uz_i1])) >= 3:
                                        continue
                                wbs_ia = tuple(
                                    LDUBox(b[0] + px_a, b[1] + py_a, b[2] + z0_i, b[3] + px_a, b[4] + py_a, b[5] + z0_i)
                                    for b in rboxes_ia
                                )
                                if any(_box_hits_forbidden(wb_k, forbidden_support_boxes) for wb_k in wbs_ia):
                                    continue
                                # Compare `occ6` net sub-voxel gain (`2*d_in - tot`) of `wbs_ia` vs. current `core_occ`
                                sx0_q = max(0, w_xmin // 5)
                                sx1_q = min(occ6.shape[0], w_xmax // 5)
                                sy0_q = max(0, w_ymin // 5)
                                sy1_q = min(occ6.shape[1], w_ymax // 5)
                                sz0_q = max(0, z0_i // 6)
                                sz1_q = min(occ6.shape[2], int(math.ceil(w_zmax / 6.0)))
                                if sx1_q <= sx0_q or sy1_q <= sy0_q or sz1_q <= sz0_q:
                                    continue
                                tgt_blk = occ6[sx0_q:sx1_q, sy0_q:sy1_q, sz0_q:sz1_q]
                                ia_blk = np.zeros_like(tgt_blk)
                                for wb_k in wbs_ia:
                                    kx0 = max(sx0_q, int(math.floor(wb_k.xmin / 5.0))) - sx0_q
                                    kx1 = min(sx1_q, int(math.ceil(wb_k.xmax / 5.0))) - sx0_q
                                    ky0 = max(sy0_q, int(math.floor(wb_k.ymin / 5.0))) - sy0_q
                                    ky1 = min(sy1_q, int(math.ceil(wb_k.ymax / 5.0))) - sy0_q
                                    kz0 = max(sz0_q, int(math.floor(wb_k.zmin / 6.0))) - sz0_q
                                    kz1 = min(sz1_q, int(math.ceil(wb_k.zmax / 6.0))) - sz0_q
                                    if kx1 > kx0 and ky1 > ky0 and kz1 > kz0:
                                        ia_blk[kx0:kx1, ky0:ky1, kz0:kz1] = True
                                core_blk = np.zeros_like(tgt_blk)
                                for rx_q in range(rx0_i, rx1_i):
                                    for ry_q in range(ry0_i, ry1_i):
                                        for pq in range(p0_i, p0_i + h_pl):
                                            if core_occ[rx_q, ry_q, pq]:
                                                cx0_q = max(sx0_q, (rx_q * 20 + ph_i[0]) // 5) - sx0_q
                                                cx1_q = min(sx1_q, ((rx_q + 1) * 20 + ph_i[0]) // 5) - sx0_q
                                                cy0_q = max(sy0_q, (ry_q * 20 + ph_i[1]) // 5) - sy0_q
                                                cy1_q = min(sy1_q, ((ry_q + 1) * 20 + ph_i[1]) // 5) - sy0_q
                                                cz0_q = max(sz0_q, int(math.floor((pq * 8) / 6.0))) - sz0_q
                                                cz1_q = min(sz1_q, int(math.ceil(((pq + 1) * 8) / 6.0))) - sz0_q
                                                if cx1_q > cx0_q and cy1_q > cy0_q and cz1_q > cz0_q:
                                                    core_blk[cx0_q:cx1_q, cy0_q:cy1_q, cz0_q:cz1_q] = True
                                ia_in = int(np.count_nonzero(ia_blk & tgt_blk))
                                ia_tot = int(np.count_nonzero(ia_blk))
                                co_in = int(np.count_nonzero(core_blk & tgt_blk))
                                co_tot = int(np.count_nonzero(core_blk))
                                ia_net = 2 * ia_in - ia_tot
                                co_net = 2 * co_in - co_tot
                                exact_tie_consolidation = (
                                    ia_in >= 12
                                    and ia_in >= co_in
                                    and (ia_tot - ia_in) <= (co_tot - co_in)
                                )
                                if ia_net > 4 and (ia_net >= co_net + 2 or exact_tie_consolidation):
                                    res_cells = [
                                        (rx_q, ry_q, pq)
                                        for rx_q in range(rx0_i, rx1_i)
                                        for ry_q in range(ry0_i, ry1_i)
                                        for pq in range(p0_i, p0_i + h_pl)
                                    ]
                                    used_inv_already = any(
                                        it_iv[0] == pid_ia for v_iv in planned_inv_by_z.values() for it_iv in v_iv
                                    )
                                    sc_ia = (
                                        float(ia_net - co_net)
                                        + (2.5 if is_arch_mold else 0.8)
                                        + (3.8 if not used_inv_already else 0.0)
                                        + 0.85 * float(rx1_i - rx0_i) * float(ry1_i - ry0_i)
                                    )
                                    if view_weight_grid_6 is not None:
                                        vw_ia = view_weight_grid_6[sx0_q:sx1_q, sy0_q:sy1_q, sz0_q:sz1_q]
                                        if vw_ia.size > 0:
                                            sc_ia = sc_ia * max(0.05, float(np.mean(vw_ia)))
                                    inv_cands_course.append((sc_ia, pid_ia, pose_ia, res_cells, top_stud_cells))

                inv_cands_course.sort(key=lambda x: (-x[0], x[1], x[2].rot_id, x[2].pos))
                for _, pid_ch, pose_ch, r_cells, t_cells in inv_cands_course:
                    if rem_inv is not None and rem_inv.get(pid_ch, 0) <= 0:
                        continue
                    if any(inv_reserved_occ[rc[0], rc[1], rc[2]] for rc in r_cells):
                        continue
                    for rc in r_cells:
                        inv_reserved_occ[rc[0], rc[1], rc[2]] = True
                        core_occ[rc[0], rc[1], rc[2]] = False
                        pier_protected[rc[0], rc[1], : rc[2] + 1] = True
                    for tc in t_cells:
                        core_occ[tc[0], tc[1], tc[2]] = True
                        pier_protected[tc[0], tc[1], tc[2]] = True
                    planned_inv_by_z.setdefault(z0_i, []).append((pid_ch, pose_ch))
                    if rem_inv is not None:
                        rem_inv[pid_ch] = rem_inv.get(pid_ch, 0) - 1

        # Ensure every elevated component (`e.g. thin tabletops & canopies`) has minimal vertical pier support
        pier_cover_radius = 4 if is_legged_frame else (3 if is_thin_regime else 2)
        if is_hollow_shell_active:
            pier_cover_radius = max(
                pier_cover_radius,
                max(3, (int(getattr(eff_shell_policy, "diaphragm_pitch_studs", 6)) + 1) // 2),
            )
        if support_policy == "minimal_permanent_pillars":
            for p in range(n_plates_z - 1, 0, -1):
                if _early_deadline_hit():
                    return None
                if p in jumper_bridge_layers or not np.any(core_occ[:, :, p]):
                    continue
                p_below = p - 1
                while p_below > 0 and p_below in jumper_bridge_layers:
                    p_below -= 1
                ph_p = layer_phase[p]
                vis = np.zeros((w_x, w_y), dtype=bool)
                for sx_i in range(w_x):
                    for sy_i in range(w_y):
                        if not core_occ[sx_i, sy_i, p] or vis[sx_i, sy_i]:
                            continue
                        comp: List[Tuple[int, int]] = []
                        dq = deque([(sx_i, sy_i)])
                        vis[sx_i, sy_i] = True
                        while dq:
                            cx_i, cy_i = dq.popleft()
                            comp.append((cx_i, cy_i))
                            for dx_n, dy_n in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                                nx_i, ny_i = cx_i + dx_n, cy_i + dy_n
                                if 0 <= nx_i < w_x and 0 <= ny_i < w_y and core_occ[nx_i, ny_i, p] and not vis[nx_i, ny_i]:
                                    vis[nx_i, ny_i] = True
                                    dq.append((nx_i, ny_i))

                        supported_pts: List[Tuple[int, int]] = []
                        ph_b = layer_phase[p_below]
                        for cx_i, cy_i in comp:
                            xb0 = cx_i * 20 + ph_p[0]
                            yb0 = cy_i * 20 + ph_p[1]
                            has_b = False
                            for bx_i in range(max(0, cx_i - 1), min(w_x, cx_i + 2)):
                                for by_i in range(max(0, cy_i - 1), min(w_y, cy_i + 2)):
                                    if not core_occ[bx_i, by_i, p_below]:
                                        continue
                                    if (
                                        min(xb0 + 20, bx_i * 20 + ph_b[0] + 20) > max(xb0, bx_i * 20 + ph_b[0])
                                        and min(yb0 + 20, by_i * 20 + ph_b[1] + 20) > max(yb0, by_i * 20 + ph_b[1])
                                    ):
                                        has_b = True
                                        break
                                if has_b:
                                    break
                            if has_b:
                                supported_pts.append((cx_i, cy_i))

                        # Skip extra pillars when >= 30% (`and >= 3 studs`) of the span rests on solid piers within <= 4 studs
                        # (counterbalanced cantilevers bridged by monolithic 2x6 / 1x6 / 2x8 girders!)
                        if not is_thin_regime and len(supported_pts) >= max(3, int(0.30 * len(comp))):
                            max_cant_d = max(
                                min(abs(pt[0] - sp[0]) + abs(pt[1] - sp[1]) for sp in supported_pts)
                                for pt in comp
                            )
                            if max_cant_d <= 4:
                                continue

                        # Consolidate ultra-thin corner/perimeter legs (`col_mesh_sum < 18 * p`, >3x empty overshoot per shaft)
                        # under an elevated slab (`4..26` studs) into a single optimal central shaft + plate capital!
                        if (
                            is_legged_frame
                            and foundation_policy == FoundationPolicy.CLUTCHED_BASEPLATE
                            and 4 <= len(comp) <= 26
                            and p >= 4
                            and (
                                len(supported_pts) >= 2
                                or (
                                    len(supported_pts) == 1
                                    and len(comp) >= 12
                                    and max(abs(pt[0] - supported_pts[0][0]) + abs(pt[1] - supported_pts[0][1]) for pt in comp) > 4
                                )
                            )
                            and p_below == p - 1
                            and layer_phase[p_below] == ph_p
                            and all(
                                not np.any(pier_protected[sx_l, sy_l, :p])
                                and sum(int(phase_grids[layer_phase[q_l]][sx_l, sy_l, q_l]) for q_l in range(p)) < 18 * p
                                for sx_l, sy_l in supported_pts
                            )
                        ):
                            for sx_l, sy_l in supported_pts:
                                core_occ[sx_l, sy_l, :p] = False
                            supported_pts = []

                        eff_pier_rad = pier_cover_radius
                        placed_thin_pedestal = False
                        while True:
                            unsup = [
                                pt
                                for pt in comp
                                if not supported_pts
                                or min(abs(pt[0] - sp[0]) + abs(pt[1] - sp[1]) for sp in supported_pts) > eff_pier_rad
                            ]
                            if not unsup:
                                break
                            best_anc: Optional[Tuple[int, int]] = None
                            best_col_sc = -1e9
                            best_col_mesh = 0
                            cx_mean = sum(pt[0] for pt in comp) / float(max(1, len(comp)))
                            cy_mean = sum(pt[1] for pt in comp) / float(max(1, len(comp)))
                            # Check column mesh support under `comp` to avoid phantom empty-air shafts (`col_mesh_sum == 0`)
                            max_comp_col_mesh = 0
                            if is_legged_frame and p >= 4:
                                for ux_m, uy_m in comp:
                                    cm_s = sum(
                                        int(
                                            phase_grids[layer_phase[q_m]][
                                                max(0, min(w_x - 1, (ux_m * 20 + ph_p[0] - layer_phase[q_m][0]) // 20)),
                                                max(0, min(w_y - 1, (uy_m * 20 + ph_p[1] - layer_phase[q_m][1]) // 20)),
                                                q_m,
                                            ]
                                        )
                                        for q_m in range(p)
                                    )
                                    if cm_s > max_comp_col_mesh:
                                        max_comp_col_mesh = cm_s
                            prefer_center_pedestal = (
                                is_legged_frame and len(comp) <= 26 and p >= 4 and max_comp_col_mesh < 18 * p
                            )
                            for ux, uy in comp:
                                if (ux, uy) in supported_pts:
                                    continue
                                clear_col = True
                                col_mesh_sum = 0
                                for q_dn in range(p):
                                    ph_q = layer_phase[q_dn]
                                    qx = max(0, min(w_x - 1, (ux * 20 + ph_p[0] - ph_q[0]) // 20))
                                    qy = max(0, min(w_y - 1, (uy * 20 + ph_p[1] - ph_q[1]) // 20))
                                    wb_q = LDUBox(
                                        qx * 20 + ph_q[0],
                                        qy * 20 + ph_q[1],
                                        q_dn * 8,
                                        (qx + 1) * 20 + ph_q[0],
                                        (qy + 1) * 20 + ph_q[1],
                                        (q_dn + 1) * 8,
                                    )
                                    if _box_hits_forbidden(wb_q, forbidden_support_boxes):
                                        clear_col = False
                                        break
                                    col_mesh_sum += int(phase_grids[ph_q][qx, qy, q_dn])
                                if not clear_col:
                                    continue
                                if is_legged_frame and max_comp_col_mesh >= 8 and col_mesh_sum == 0 and len(comp) <= 15:
                                    continue
                                rad_chk = 10 if prefer_center_pedestal else eff_pier_rad
                                cov_gain = sum(
                                    1 for pt_u in unsup if abs(pt_u[0] - ux) + abs(pt_u[1] - uy) <= rad_chk
                                )
                                cent_w = (
                                    42.0
                                    if (prefer_center_pedestal and len(comp) >= 12)
                                    else (
                                        8.0
                                        if (prefer_center_pedestal and max_comp_col_mesh >= 16)
                                        else (38.0 if prefer_center_pedestal else 4.0)
                                    )
                                )
                                cent_pen = cent_w * (
                                    abs(float(ux) - cx_mean) + abs(float(uy) - cy_mean)
                                )
                                sc_a = (
                                    (160.0 * float(cov_gain) + (18.0 if prefer_center_pedestal else 12.0) * float(col_mesh_sum) - cent_pen)
                                    if is_legged_frame
                                    else (24.0 * float(col_mesh_sum) + 12.0 * float(cov_gain))
                                )
                                if sc_a > best_col_sc:
                                    best_col_sc = sc_a
                                    best_anc = (ux, uy)
                                    best_col_mesh = col_mesh_sum
                            if best_anc is None:
                                break
                            ax_i, ay_i = best_anc
                            supported_pts.append(best_anc)
                            pier_protected[ax_i, ay_i, p] = True
                            for q_dn in range(p - 1, -1, -1):
                                if q_dn in jumper_bridge_layers:
                                    continue
                                ph_q = layer_phase[q_dn]
                                qx = max(0, min(w_x - 1, (ax_i * 20 + ph_p[0] - ph_q[0]) // 20))
                                qy = max(0, min(w_y - 1, (ay_i * 20 + ph_p[1] - ph_q[1]) // 20))
                                pier_protected[qx, qy, q_dn] = True
                                if not core_occ[qx, qy, q_dn]:
                                    pillar_added_occ[qx, qy, q_dn] = True
                                core_occ[qx, qy, q_dn] = True
                            # When an elevated slab sits over ultra-thin legs where each `20x20` brick shaft overshoots >3x,
                            # give the shaft a compact `1x2` (`or symmetric 1x3 on >=16-stud wide tabletops`) anti-tipping plate capital at `p - 1`!
                            if (
                                is_legged_frame
                                and foundation_policy == FoundationPolicy.CLUTCHED_BASEPLATE
                                and not placed_thin_pedestal
                                and p >= 4
                                and best_col_mesh < 18 * p
                                and (p - 1) not in jumper_bridge_layers
                                and layer_phase[p - 1] == ph_p
                            ):
                                sx_cap = 1 if (ax_i + 1 < w_x and (float(ax_i) <= cx_mean or ax_i == 0)) else (-1 if ax_i - 1 >= 0 else 0)
                                sy_cap = 1 if (ay_i + 1 < w_y and (float(ay_i) <= cy_mean or ay_i == 0)) else (-1 if ay_i - 1 >= 0 else 0)
                                if len(comp) >= 10 and w_x >= 4 and sy_cap != 0 and 0 <= ay_i + sy_cap < w_y:
                                    x_c0 = int(round(cx_mean)) - 1
                                    x_c0 = max(1 if w_x >= 5 else 0, min(w_x - (4 if w_x >= 5 else 3), x_c0))
                                    if ax_i < x_c0:
                                        x_c0 = ax_i
                                    elif ax_i > x_c0 + 2:
                                        x_c0 = ax_i - 2
                                    x_c0 = max(0, min(w_x - 3, x_c0))
                                    x_mid = x_c0 + 1
                                    cap_offsets = [
                                        (x_c0 - ax_i, 0),
                                        (x_c0 + 1 - ax_i, 0),
                                        (x_c0 + 2 - ax_i, 0),
                                        (x_mid - ax_i, sy_cap),
                                        (0, sy_cap),
                                    ]
                                    cap_offsets = list(dict.fromkeys(cap_offsets))
                                elif len(comp) >= 10 and w_y >= 4 and sx_cap != 0 and 0 <= ax_i + sx_cap < w_x:
                                    y_c0 = int(round(cy_mean)) - 1
                                    y_c0 = max(1 if w_y >= 5 else 0, min(w_y - (4 if w_y >= 5 else 3), y_c0))
                                    if ay_i < y_c0:
                                        y_c0 = ay_i
                                    elif ay_i > y_c0 + 2:
                                        y_c0 = ay_i - 2
                                    y_c0 = max(0, min(w_y - 3, y_c0))
                                    y_mid = y_c0 + 1
                                    cap_offsets = [
                                        (0, y_c0 - ay_i),
                                        (0, y_c0 + 1 - ay_i),
                                        (0, y_c0 + 2 - ay_i),
                                        (sx_cap, y_mid - ay_i),
                                        (sx_cap, 0),
                                    ]
                                    cap_offsets = list(dict.fromkeys(cap_offsets))
                                elif len(comp) >= 10 and 0 < ax_i < w_x - 1:
                                    cap_offsets = [(-1, 0), (0, 0), (1, 0)]
                                elif len(comp) >= 10 and 0 < ay_i < w_y - 1:
                                    cap_offsets = [(0, -1), (0, 0), (0, 1)]
                                elif len(comp) >= 8 and sx_cap != 0 and sy_cap != 0 and 0 <= ax_i + 2 * sx_cap < w_x and 0 <= ay_i + sy_cap < w_y:
                                    cap_offsets = [(0, 0), (0, sy_cap), (sx_cap, sy_cap), (2 * sx_cap, sy_cap)]
                                elif len(comp) >= 8 and sx_cap != 0 and sy_cap != 0 and 0 <= ax_i + sx_cap < w_x and 0 <= ay_i + 2 * sy_cap < w_y:
                                    cap_offsets = [(0, 0), (sx_cap, 0), (sx_cap, sy_cap), (sx_cap, 2 * sy_cap)]
                                else:
                                    cap_offsets = [(0, 0), (sx_cap, 0) if sx_cap != 0 else (0, sy_cap)]
                                for dcx, dcy in cap_offsets:
                                    cx_h, cy_h = ax_i + dcx, ay_i + dcy
                                    if 0 <= cx_h < w_x and 0 <= cy_h < w_y:
                                        core_occ[cx_h, cy_h, p - 1] = True
                                        pier_protected[cx_h, cy_h, p - 1] = True
                                        if (cx_h, cy_h) in comp:
                                            pier_protected[cx_h, cy_h, p] = True
                                            if (cx_h, cy_h) not in supported_pts:
                                                supported_pts.append((cx_h, cy_h))
                                placed_thin_pedestal = True
                                eff_pier_rad = 10

        # Protect pier-top lintel anchors within Manhattan distance <= 2 of genuine unsupported bridge spans
        # so arches, doorways, tunnels, and cantilevers NEVER have their supporting pier shoulders carved!
        for p_br in range(1, n_plates_z):
            if p_br in jumper_bridge_layers or not np.any(core_occ[:, :, p_br]):
                continue
            p_bel = p_br - 1
            while p_bel > 0 and p_bel in jumper_bridge_layers:
                p_bel -= 1
            if layer_phase[p_br] != layer_phase[p_bel]:
                continue
            unsup_xy = core_occ[:, :, p_br] & (~core_occ[:, :, p_bel])
            n_unsup = int(np.count_nonzero(unsup_xy))
            if n_unsup == 0 or (not forbidden_support_boxes and not is_legged_frame and n_unsup < 3):
                continue
            uxs, uys = np.nonzero(unsup_xy)
            for ux_i, uy_i in zip(uxs.tolist(), uys.tolist()):
                for ax_i in range(max(0, ux_i - 2), min(w_x, ux_i + 3)):
                    for ay_i in range(max(0, uy_i - 2), min(w_y, uy_i + 3)):
                        md_a = abs(ax_i - ux_i) + abs(ay_i - uy_i)
                        if md_a <= 2:
                            pier_protected[ax_i, ay_i, : p_br + 1] = True

        # Ensure ground seed exists even on ultra-thin wireframes (`e.g. 5 LDU blades`)
        if not np.any(core_occ[:, :, 0]):
            ph0 = layer_phase[0]
            cg0 = phase_grids[ph0][:, :, 0]
            best_seed_val = 0
            best_seed_xy = (0, 0)
            for ix in range(w_x):
                for iy in range(w_y):
                    wb0 = LDUBox(ix * 20 + ph0[0], iy * 20 + ph0[1], 0, (ix + 1) * 20 + ph0[0], (iy + 1) * 20 + ph0[1], 8)
                    if not _box_hits_forbidden(wb0, forbidden_support_boxes) and int(cg0[ix, iy]) > best_seed_val:
                        best_seed_val = int(cg0[ix, iy])
                        best_seed_xy = (ix, iy)
            if best_seed_val > 0:
                seeded = False
                for ix in range(w_x):
                    for iy in range(w_y):
                        wb0 = LDUBox(ix * 20 + ph0[0], iy * 20 + ph0[1], 0, (ix + 1) * 20 + ph0[0], (iy + 1) * 20 + ph0[1], 8)
                        if not _box_hits_forbidden(wb0, forbidden_support_boxes) and int(cg0[ix, iy]) >= 14:
                            core_occ[ix, iy, 0] = True
                            seeded = True
                if not seeded:
                    core_occ[best_seed_xy[0], best_seed_xy[1], 0] = True

        core_occ_pre_sculpt = core_occ.copy()

        # Exact `occ6` marginal sub-voxel terminal-plate prune (top-down): prune any topmost `core_occ[ix, iy, p]`
        # sitting on `core_occ[ix, iy, p - 1]` whose marginal `5x5x6 LDU` slice (`above ceil(8*p / 6)`) adds more
        # empty overshoot than mesh intersection (`in_marg < 0.45 * tot_marg` on thick meshes, `< 0.35` on thin frames).
        for _prune_pass in range(2):
            if _early_deadline_hit():
                return None
            if is_thin_regime:
                marg_ratio = 0.20 if has_thin_panel_kit else 0.35
            else:
                iou_cur_pr = _est_core_occ6_iou(core_occ, layer_phase, jumper_bridge_layers)
                marg_ratio = max(0.455, iou_cur_pr / (1.0 + iou_cur_pr))
            for p_pr in range(n_plates_z - 1, 0, -1):
                if p_pr in jumper_bridge_layers or (p_pr - 1) in jumper_bridge_layers:
                    continue
                ph_pr = layer_phase[p_pr]
                if layer_phase[p_pr - 1] != ph_pr:
                    continue
                z6_prev = min(occ6.shape[2], int(math.ceil((8 * p_pr) / 6.0)))
                z6_cur = min(occ6.shape[2], int(math.ceil((8 * (p_pr + 1)) / 6.0)))
                if z6_cur <= z6_prev:
                    continue
                for ix in range(w_x):
                    for iy in range(w_y):
                        if not core_occ[ix, iy, p_pr] or not core_occ[ix, iy, p_pr - 1]:
                            continue
                        if pier_protected[ix, iy, p_pr] or inv_reserved_occ[ix, iy, p_pr]:
                            continue
                        if (p_pr + 1) < n_plates_z and (
                            core_occ[ix, iy, p_pr + 1 :].any() or inv_reserved_occ[ix, iy, p_pr + 1 :].any()
                        ):
                            continue
                        x6_p = (ix * 20 + ph_pr[0]) // 5
                        y6_p = (iy * 20 + ph_pr[1]) // 5
                        blk_m = occ6[x6_p : x6_p + 4, y6_p : y6_p + 4, z6_prev:z6_cur]
                        if blk_m.size == 0:
                            continue
                        in_m_cnt = int(np.count_nonzero(blk_m))
                        if view_weight_grid_6 is not None:
                            w_blk_m = view_weight_grid_6[x6_p : x6_p + 4, y6_p : y6_p + 4, z6_prev:z6_cur]
                            in_w_m = float(np.sum(w_blk_m[blk_m]))
                            ov_w_m = float(np.sum(w_blk_m[~blk_m]))
                            if ov_w_m > 1.15 * in_w_m:
                                core_occ[ix, iy, p_pr] = False
                        elif float(in_m_cnt) < marg_ratio * float(blk_m.size):
                            core_occ[ix, iy, p_pr] = False

        snot_spine_protected = np.zeros((w_x, w_y, n_plates_z), dtype=bool)
        snot_channel_carved = np.zeros((w_x, w_y, n_plates_z), dtype=bool)
        has_sculptural_kit = any(
            (CATALOG_RETRIEVAL_INDEX.get_entry(pid) is not None and CATALOG_RETRIEVAL_INDEX.get_entry(pid).taxonomy_class in SCULPTURAL_TAXONOMY_CLASSES)
            for pid in kit_ids
        )
        if has_sculptural_kit and not is_thin_regime:
            # Build phase-aligned 3-plate brick course starts within each maximal non-jumper phase segment
            # so upper tiers above `15573` jumper bridges (`even when p_up0 % 3 != 0`!) get full perimeter slope erosion!
            phase_course_starts: List[int] = []
            seg_s = 0
            while seg_s < n_plates_z:
                if seg_s in jumper_bridge_layers:
                    seg_s += 1
                    continue
                seg_ph = layer_phase[seg_s]
                seg_e = seg_s + 1
                while seg_e < n_plates_z and seg_e not in jumper_bridge_layers and layer_phase[seg_e] == seg_ph:
                    seg_e += 1
                p_cur = seg_s
                while p_cur + 3 <= seg_e:
                    phase_course_starts.append(p_cur)
                    p_cur += 3
                seg_s = seg_e

            for p0 in reversed(phase_course_starts):
                if _early_deadline_hit():
                    return None
                if p0 == 0 and foundation_policy != FoundationPolicy.CLUTCHED_BASEPLATE:
                    continue
                p_top = p0 + 2
                if (p_top + 1) in jumper_bridge_layers:
                    continue
                ph_c = layer_phase[p0]
                cg_c = phase_grids[ph_c]
                for ix in range(w_x):
                    for iy in range(w_y):
                        if np.any(pier_protected[ix, iy, p0 : p_top + 1]) or np.any(inv_reserved_occ[ix, iy, p0 : p_top + 1]):
                            continue
                        if (p_top + 1) < n_plates_z and (
                            np.any(core_occ[ix, iy, p_top + 1 :]) or np.any(cg_c[ix, iy, p_top + 1 :] >= 16)
                        ):
                            continue
                        if not np.any(core_occ[ix, iy, p0 : p_top + 1]):
                            continue
                        v0 = int(cg_c[ix, iy, p0])
                        v1 = int(cg_c[ix, iy, p0 + 1])
                        v2 = int(cg_c[ix, iy, p0 + 2])
                        is_perim = any(
                            not (0 <= ix + dx_p < w_x and 0 <= iy + dy_p < w_y and core_occ[ix + dx_p, iy + dy_p, p0])
                            for dx_p, dy_p in ((1, 0), (-1, 0), (0, 1), (0, -1))
                        )
                        if is_perim and v0 <= 22 and v1 <= 14 and v2 <= 8 and view_weight_grid_6 is None:
                            core_occ[ix, iy, p0 : p_top + 1] = False

            # Lateral half-stud stepped-terrace carving (top-down across phase-aligned courses):
            # carve outer perimeter cells whose outer `10 LDU` half is nearly empty (`outer_p0 <= 5` and `outer_cnt <= 16`)
            # while their inner `10 LDU` half is present (`>= 26 / 96`), exposing outward SNOT spines for `8 LDU` tiles!
            if has_snot_core:
                for p0_l in reversed(phase_course_starts):
                    if _early_deadline_hit():
                        return None
                    if p0_l == 0 and foundation_policy != FoundationPolicy.CLUTCHED_BASEPLATE:
                        continue
                    if (p0_l + 3) in jumper_bridge_layers:
                        continue
                    ph_l = layer_phase[p0_l]
                    to_carve_lat: Dict[Tuple[int, int], Tuple[int, int, bool]] = {}
                    for ix in range(w_x):
                        for iy in range(w_y):
                            if not np.any(core_occ[ix, iy, p0_l : p0_l + 3]):
                                continue
                            if np.any(pier_protected[ix, iy, p0_l : p0_l + 3]) or np.any(
                                inv_reserved_occ[ix, iy, p0_l : p0_l + 3]
                            ):
                                continue
                            if (p0_l + 3) < n_plates_z and np.any(core_occ[ix, iy, p0_l + 3 :]):
                                continue
                            sx0_l = (ix * 20 + ph_l[0]) // 5
                            sy0_l = (iy * 20 + ph_l[1]) // 5
                            blk_l = occ2[sx0_l : sx0_l + 4, sy0_l : sy0_l + 4, 4 * p0_l : 4 * (p0_l + 3)]
                            for dx_o, dy_o in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                                ox_i, oy_i = ix + dx_o, iy + dy_o
                                in_x, in_y = ix - dx_o, iy - dy_o
                                if 0 <= ox_i < w_x and 0 <= oy_i < w_y and np.any(core_occ[ox_i, oy_i, p0_l : p0_l + 3]):
                                    continue
                                if not (0 <= in_x < w_x and 0 <= in_y < w_y and np.all(core_occ[in_x, in_y, p0_l : p0_l + 3])):
                                    continue
                                if (in_x, in_y) in to_carve_lat:
                                    continue
                                in2_x, in2_y = ix - 2 * dx_o, iy - 2 * dy_o
                                has_in2 = (
                                    0 <= in2_x < w_x
                                    and 0 <= in2_y < w_y
                                    and bool(np.all(core_occ[in2_x, in2_y, p0_l : p0_l + 3]))
                                    and (in2_x, in2_y) not in to_carve_lat
                                )
                                if not has_in2:
                                    perp_dx, perp_dy = abs(dy_o), abs(dx_o)
                                    has_perp_spine = any(
                                        0 <= in_x + sgn * perp_dx < w_x
                                        and 0 <= in_y + sgn * perp_dy < w_y
                                        and bool(np.all(core_occ[in_x + sgn * perp_dx, in_y + sgn * perp_dy, p0_l : p0_l + 3]))
                                        and (in_x + sgn * perp_dx, in_y + sgn * perp_dy) not in to_carve_lat
                                        for sgn in (-1, 1)
                                    )
                                    if not has_perp_spine:
                                        continue
                                if dx_o == 1:
                                    outer_p0 = int(np.count_nonzero(blk_l[2:, :, :4]))
                                    outer_cnt = int(np.count_nonzero(blk_l[2:, :, :]))
                                    inner_cnt = int(np.count_nonzero(blk_l[:2, :, :]))
                                elif dx_o == -1:
                                    outer_p0 = int(np.count_nonzero(blk_l[:2, :, :4]))
                                    outer_cnt = int(np.count_nonzero(blk_l[:2, :, :]))
                                    inner_cnt = int(np.count_nonzero(blk_l[2:, :, :]))
                                elif dy_o == 1:
                                    outer_p0 = int(np.count_nonzero(blk_l[:, 2:, :4]))
                                    outer_cnt = int(np.count_nonzero(blk_l[:, 2:, :]))
                                    inner_cnt = int(np.count_nonzero(blk_l[:, :2, :]))
                                else:
                                    outer_p0 = int(np.count_nonzero(blk_l[:, :2, :4]))
                                    outer_cnt = int(np.count_nonzero(blk_l[:, :2, :]))
                                    inner_cnt = int(np.count_nonzero(blk_l[:, 2:, :]))
                                if outer_p0 <= 6 and outer_cnt <= 18 and inner_cnt >= 28:
                                    to_carve_lat[(ix, iy)] = (
                                        in_x,
                                        in_y,
                                        bool(outer_cnt <= 14 and inner_cnt >= 52),
                                    )
                                    break
                    for (cx_l, cy_l), (sp_x, sp_y, is_vert_hs) in to_carve_lat.items():
                        core_occ[cx_l, cy_l, p0_l : p0_l + 3] = False
                        snot_channel_carved[cx_l, cy_l, p0_l : p0_l + 3] = True
                        if is_vert_hs and not (
                            (p0_l + 3) >= n_plates_z - 1
                            and int(phase_grids[ph_l][sp_x, sp_y, p0_l + 2]) < 44
                        ):
                            snot_spine_protected[sp_x, sp_y, p0_l : p0_l + 3] = True

        brick_templates = _rect_templates(kit_ids, 24)
        plate_templates = _rect_templates(kit_ids, 8)
        tile_templates = _rect_templates(kit_ids, 8, studless_tiles=True)
        if not plate_templates and (eff_inv_map is None or eff_inv_map.get("3024", 0) > 0):
            plate_templates = [("3024", 1, 1)]
        if not brick_templates and (eff_inv_map is None or eff_inv_map.get("3005", 0) > 0):
            brick_templates = [("3005", 1, 1)]
        if not plate_templates and brick_templates:
            # When an inventory contains bricks only (zero 8-LDU plates), reuse brick_templates on 24-LDU bands
            plate_templates = []
        if not tile_templates and eff_inv_map is None:
            tile_templates = [
                ("87079", 2, 4),
                ("26603", 2, 3),
                ("3068b", 2, 2),
                ("2431", 1, 4),
                ("63864", 1, 3),
                ("3069b", 1, 2),
                ("3070b", 1, 1),
            ]
            if license_filter == "cc_by_4_0_only":
                tile_templates = [
                    t
                    for t in tile_templates
                    if t[0] in EXTENDED_BRICK_CATALOG
                    or (
                        CATALOG_RETRIEVAL_INDEX.get_entry(t[0]) is not None
                        and CATALOG_RETRIEVAL_INDEX.get_entry(t[0]).license_tier == "cc_by_4_0"  # type: ignore[union-attr]
                    )
                ]

        planned_by_z: Dict[int, List[Tuple[str, LDUFrame]]] = {
            z_k: list(v_k) for z_k, v_k in planned_inv_by_z.items()
        }

        # Plan `15573` jumper transition layers FIRST so `jumper_top_stud_xy[p]` records the exact
        # shifted cells `(ux, uy)` at `p + 1` resting directly on a `15573` top stud!
        jumper_top_stud_xy: Dict[int, np.ndarray] = {}
        jumper_used_low: Dict[int, np.ndarray] = {}
        jumper_low_occ: Dict[int, np.ndarray] = {}
        planned_jb_y_targets: Dict[int, np.ndarray] = {}

        def _disjoint_target_indices(
            u_line: np.ndarray, low_line: np.ndarray, ph_to_ax: int, ph_from_ax: int, alt_parity: int = 0
        ) -> List[int]:
            """Select `100%` disjoint `15573` upper-tier stud indices covering both perimeter endpoints."""
            occ_u = np.nonzero(u_line)[0].tolist()
            if not occ_u:
                return []
            runs: List[Tuple[int, int]] = []
            r_s = occ_u[0]
            r_e = occ_u[0]
            for u_v in occ_u[1:]:
                if u_v == r_e + 1:
                    r_e = u_v
                else:
                    runs.append((r_s, r_e))
                    r_s = u_v
                    r_e = u_v
            runs.append((r_s, r_e))

            def _pair_supported(u_idx: int) -> bool:
                w_c = u_idx * 20 + ph_to_ax + 10
                i0 = (w_c - 10 - ph_from_ax) // 20
                if not (0 <= i0 and (i0 + 1) < low_line.shape[0]):
                    return False
                return bool(low_line[i0] or low_line[i0 + 1]) if is_thin_regime else bool(low_line[i0] and low_line[i0 + 1])

            chosen: List[int] = []
            for u0, u1 in runs:
                span_w = u1 - u0 + 1
                if span_w % 2 == 1:
                    if (
                        span_w == 3
                        and (alt_parity % 3 == 1)
                        and _pair_supported(u0 + 1)
                    ):
                        cands_r = [u0 + 1]
                    else:
                        cands_r = list(range(u0, u1 + 1, 2))
                elif span_w >= 4 and _pair_supported(u0) and _pair_supported(u1):
                    cands_r = list(range(u0, u1 - 2, 2)) + [u1]
                elif span_w == 2 and _pair_supported(u0) and _pair_supported(u1):
                    cands_r = [u1] if (alt_parity % 2 == 1) else [u0]
                elif _pair_supported(u0):
                    cands_r = list(range(u0, u1 + 1, 2))
                else:
                    cands_r = list(range(u0 + 1, u1 + 1, 2))
                for u_c in cands_r:
                    if _pair_supported(u_c):
                        chosen.append(u_c)
            return chosen

        for p in sorted(jumper_bridge_layers.keys()):
            z0 = p * 8
            ph_from, ph_to = jumper_bridge_layers[p]
            shift_x = ph_to[0] != ph_from[0]
            p_low = p - 1
            if p_low in jumper_top_stud_xy:
                low_occ = jumper_top_stud_xy[p_low]
            elif p_low >= 0:
                low_occ = core_occ[:, :, p_low]
            elif foundation_policy == FoundationPolicy.CLUTCHED_BASEPLATE:
                low_occ = np.ones((w_x, w_y), dtype=bool)
            else:
                low_occ = np.zeros((w_x, w_y), dtype=bool)
            p_up_start = p + 1
            while p_up_start < n_plates_z and p_up_start in jumper_bridge_layers:
                p_up_start += 1
            ph_final = layer_phase[p_up_start] if p_up_start < n_plates_z else ph_to
            raw_up = np.zeros((w_x, w_y), dtype=bool)
            for q_u in range(p_up_start, min(n_plates_z, p_up_start + 3)):
                if layer_phase[q_u] == ph_final:
                    raw_up |= core_occ[:, :, q_u]

            cg_jb_low = phase_grids[ph_from][:, :, p]
            if ph_final == ph_to:
                if p in planned_jb_y_targets:
                    up_occ_union = planned_jb_y_targets[p]
                elif not is_thin_regime:
                    up_occ_union = raw_up.copy()
                    if shift_x:
                        if w_y >= 6:
                            if int(np.max(cg_jb_low[:, 0])) < 54 and up_occ_union[:, 1:].any():
                                up_occ_union[:, 0] = False
                            if int(np.max(cg_jb_low[:, w_y - 1])) < 54 and up_occ_union[:, : w_y - 1].any():
                                up_occ_union[:, w_y - 1] = False
                        occ_ux1 = np.nonzero(up_occ_union.any(axis=1))[0].tolist()
                        if len(occ_ux1) >= 5:
                            ix_0 = ((occ_ux1[0] * 20 + ph_to[0] + 10) - 10 - ph_from[0]) // 20
                            ix_1 = ((occ_ux1[-1] * 20 + ph_to[0] + 10) - 10 - ph_from[0]) // 20
                            if ix_0 == 0 and int(np.max(cg_jb_low[0, :])) < 56 and up_occ_union[occ_ux1[0] + 1 :, :].any():
                                up_occ_union[occ_ux1[0], :] = False
                            if (ix_1 + 1) == (w_x - 1) and int(np.max(cg_jb_low[w_x - 1, :])) < 56 and up_occ_union[: occ_ux1[-1], :].any():
                                up_occ_union[occ_ux1[-1], :] = False
                    else:
                        if w_x >= 6:
                            if int(np.max(cg_jb_low[0, :])) < 54 and up_occ_union[1:, :].any():
                                up_occ_union[0, :] = False
                            if int(np.max(cg_jb_low[w_x - 1, :])) < 54 and up_occ_union[: w_x - 1, :].any():
                                up_occ_union[w_x - 1, :] = False
                        occ_uy1 = np.nonzero(up_occ_union.any(axis=0))[0].tolist()
                        if len(occ_uy1) >= 5:
                            iy_0 = ((occ_uy1[0] * 20 + ph_to[1] + 10) - 10 - ph_from[1]) // 20
                            iy_1 = ((occ_uy1[-1] * 20 + ph_to[1] + 10) - 10 - ph_from[1]) // 20
                            if iy_0 == 0 and int(np.max(cg_jb_low[:, 0])) < 56 and up_occ_union[:, occ_uy1[0] + 1 :].any():
                                up_occ_union[:, occ_uy1[0]] = False
                            if (iy_1 + 1) == (w_y - 1) and int(np.max(cg_jb_low[:, w_y - 1])) < 56 and up_occ_union[:, : occ_uy1[-1]].any():
                                up_occ_union[:, occ_uy1[-1]] = False
                else:
                    up_occ_union = raw_up
            elif shift_x and (p + 1) in jumper_bridge_layers:
                # Top-down 2-layer `(X-shift at p, Y-shift at p + 1)` jumper bridge planning:
                # 1) Trim tapered outer X-columns (`ix == 0`, `ix == w_x - 1`), then select disjoint X-columns `u_cols_x`.
                # 2) On each `k_col, ux_c in enumerate(u_cols_x)`, select parity-alternated interior Y-targets `uy_t` at `p + 1`.
                # 3) Populate `up_occ_union` at `p` with ONLY the exact intermediate `(ux_c, iy_m), (ux_c, iy_m + 1)` cells!
                low_pair_ok_x = np.zeros(w_x, dtype=bool)
                for ix_chk in range(w_x - 1):
                    if bool((low_occ[ix_chk, :] & low_occ[ix_chk + 1, :]).any()):
                        low_pair_ok_x[ix_chk] = True
                        low_pair_ok_x[ix_chk + 1] = True
                raw_x_line = raw_up.any(axis=1).copy()
                occ_ux = np.nonzero(raw_x_line)[0].tolist()
                if len(occ_ux) >= 3:
                    trim_x = raw_x_line.copy()
                    ix_x0 = ((occ_ux[0] * 20 + ph_to[0] + 10) - 10 - ph_from[0]) // 20
                    ix_x1 = ((occ_ux[-1] * 20 + ph_to[0] + 10) - 10 - ph_from[0]) // 20
                    if ix_x0 == 0 and int(np.max(cg_jb_low[0, :])) < 58:
                        trim_x[occ_ux[0]] = False
                    if (ix_x1 + 1) == (w_x - 1) and int(np.max(cg_jb_low[w_x - 1, :])) < 58:
                        trim_x[occ_ux[-1]] = False
                    if trim_x.any() and _disjoint_target_indices(trim_x, low_pair_ok_x, ph_to[0], ph_from[0], alt_parity=0):
                        raw_x_line = trim_x
                u_cols_x = _disjoint_target_indices(
                    raw_x_line, low_pair_ok_x, ph_to[0], ph_from[0], alt_parity=0
                )
                y_target_mask = np.zeros((w_x, w_y), dtype=bool)
                up_occ_union = np.zeros((w_x, w_y), dtype=bool)
                for k_col, ux_c in enumerate(u_cols_x):
                    w_cx_c = ux_c * 20 + ph_to[0] + 10
                    ix_c = (w_cx_c - 10 - ph_from[0]) // 20
                    if ix_c < 0 or ix_c + 1 >= w_x:
                        continue
                    low_y_for_col = low_occ[ix_c, :] & low_occ[ix_c + 1, :]
                    u_line_y = raw_up[ux_c, :].copy()
                    occ_uy = np.nonzero(u_line_y)[0].tolist()
                    if len(occ_uy) >= 4:
                        u0_y, u1_y = occ_uy[0], occ_uy[-1]
                        iym_0 = (u0_y * 20 + ph_final[1] - ph_to[1]) // 20
                        iym_1 = (u1_y * 20 + ph_final[1] - ph_to[1]) // 20
                        # When outer Tier-1 Y-walls (`iy == 0` or `iy == w_y - 1`) taper (`cg < 54`), keep jumper
                        # bridges strictly interior along Y so ALL 4 Tier-1 perimeter walls stay free for slopes!
                        trimmed_y = u_line_y.copy()
                        if iym_0 == 0 and max(int(cg_jb_low[ix_c, 0]), int(cg_jb_low[ix_c + 1, 0])) < 54:
                            trimmed_y[u0_y] = False
                        if (iym_1 + 1) == (w_y - 1) and max(
                            int(cg_jb_low[ix_c, w_y - 1]), int(cg_jb_low[ix_c + 1, w_y - 1])
                        ) < 54:
                            trimmed_y[u1_y] = False
                        if trimmed_y.any():
                            cand_trim = _disjoint_target_indices(
                                trimmed_y, low_y_for_col, ph_final[1], ph_to[1], alt_parity=k_col
                            )
                            if cand_trim:
                                u_line_y = trimmed_y
                    uy_list = _disjoint_target_indices(
                        u_line_y, low_y_for_col, ph_final[1], ph_to[1], alt_parity=k_col
                    )
                    for uy_c in uy_list:
                        iy_m = (uy_c * 20 + ph_final[1] - ph_to[1]) // 20
                        if 0 <= iy_m and (iy_m + 1) < w_y and low_y_for_col[iy_m] and low_y_for_col[iy_m + 1]:
                            y_target_mask[ux_c, uy_c] = True
                            up_occ_union[ux_c, iy_m] = True
                            up_occ_union[ux_c, iy_m + 1] = True
                planned_jb_y_targets[p + 1] = y_target_mask
            else:
                up_occ_union = np.zeros((w_x, w_y), dtype=bool)
                uxs_r, uys_r = np.nonzero(raw_up)
                for uxf, uyf in zip(uxs_r.tolist(), uys_r.tolist()):
                    if shift_x:
                        iy_m = (uyf * 20 + ph_final[1] - ph_to[1]) // 20
                        for iyc in (iy_m, iy_m + 1):
                            if 0 <= iyc < w_y:
                                up_occ_union[uxf, iyc] = True
                    else:
                        ix_m = (uxf * 20 + ph_final[0] - ph_to[0]) // 20
                        for ixc in (ix_m, ix_m + 1):
                            if 0 <= ixc < w_x:
                                up_occ_union[ixc, uyf] = True

            used_low = np.zeros((w_x, w_y), dtype=bool)
            stud_mask_up = np.zeros((w_x, w_y), dtype=bool)
            prim_j = get_brick_primitive("15573")

            for rid_j in C4_Z_ROT_IDS:
                jf = LDUFrame(rid_j, (0, 0, 0))
                apts = [jf.compose(pt.local_frame).pos for pt in prim_j.anti_stud_ports]
                dx_a = max(a[0] for a in apts) - min(a[0] for a in apts)
                dy_a = max(a[1] for a in apts) - min(a[1] for a in apts)
                if (shift_x and dx_a == 20 and dy_a == 0) or ((not shift_x) and dy_a == 20 and dx_a == 0):
                    min_ax = min(a[0] for a in apts)
                    min_ay = min(a[1] for a in apts)
                    act_line_k = 0
                    use_alt_p = ((p + 1) not in jumper_bridge_layers) and (p not in planned_jb_y_targets)
                    if shift_x:
                        is_preplanned_2layer = (p + 1) in planned_jb_y_targets
                        for iy in range(w_y):
                            u_targets = (
                                np.nonzero(up_occ_union[:, iy])[0].tolist()
                                if is_preplanned_2layer
                                else _disjoint_target_indices(
                                    up_occ_union[:, iy],
                                    low_occ[:, iy],
                                    ph_to[0],
                                    ph_from[0],
                                    alt_parity=(act_line_k if use_alt_p else 0),
                                )
                            )
                            if u_targets:
                                act_line_k += 1
                            for ux_t in u_targets:
                                w_cx = ux_t * 20 + ph_to[0] + 10
                                ix = (w_cx - 10 - ph_from[0]) // 20
                                nx2, ny2 = ix + 1, iy
                                if ix < 0 or nx2 >= w_x or used_low[ix, iy] or used_low[nx2, ny2]:
                                    continue
                                if not (
                                    (low_occ[ix, iy] or low_occ[nx2, ny2])
                                    if is_thin_regime
                                    else (low_occ[ix, iy] and low_occ[nx2, ny2])
                                ):
                                    continue
                                if rem_inv is not None and rem_inv.get("15573", 0) <= 0:
                                    continue
                                tx = ix * 20 + ph_from[0] + 10 - min_ax
                                ty = iy * 20 + ph_from[1] + 10 - min_ay
                                tz = z0 - apts[0][2]
                                pose_j = LDUFrame(rid_j, (tx, ty, tz))
                                wb_j = prim_j.hull_boxes[0].transform(pose_j)
                                if _box_hits_forbidden(wb_j, forbidden_support_boxes):
                                    continue
                                planned_by_z.setdefault(z0, []).append(("15573", pose_j))
                                used_low[ix, iy] = True
                                used_low[nx2, ny2] = True
                                for pt_s in prim_j.ports:
                                    if pt_s.polarity == PortPolarity.STUD:
                                        sp = pose_j.compose(pt_s.local_frame).pos
                                        ux_s = (sp[0] - ph_to[0]) // 20
                                        uy_s = (sp[1] - ph_to[1]) // 20
                                        if 0 <= ux_s < w_x and 0 <= uy_s < w_y:
                                            stud_mask_up[ux_s, uy_s] = True
                                if rem_inv is not None:
                                    rem_inv["15573"] = rem_inv.get("15573", 0) - 1
                    else:
                        is_preplanned_y = p in planned_jb_y_targets
                        for ix in range(w_x):
                            u_targets = (
                                np.nonzero(up_occ_union[ix, :])[0].tolist()
                                if is_preplanned_y
                                else _disjoint_target_indices(
                                    up_occ_union[ix, :],
                                    low_occ[ix, :],
                                    ph_to[1],
                                    ph_from[1],
                                    alt_parity=(act_line_k if use_alt_p else 0),
                                )
                            )
                            if u_targets:
                                act_line_k += 1
                            for uy_t in u_targets:
                                w_cy = uy_t * 20 + ph_to[1] + 10
                                iy = (w_cy - 10 - ph_from[1]) // 20
                                nx2, ny2 = ix, iy + 1
                                if iy < 0 or ny2 >= w_y or used_low[ix, iy] or used_low[nx2, ny2]:
                                    continue
                                if not (
                                    (low_occ[ix, iy] or low_occ[nx2, ny2])
                                    if is_thin_regime
                                    else (low_occ[ix, iy] and low_occ[nx2, ny2])
                                ):
                                    continue
                                if rem_inv is not None and rem_inv.get("15573", 0) <= 0:
                                    continue
                                tx = ix * 20 + ph_from[0] + 10 - min_ax
                                ty = iy * 20 + ph_from[1] + 10 - min_ay
                                tz = z0 - apts[0][2]
                                pose_j = LDUFrame(rid_j, (tx, ty, tz))
                                wb_j = prim_j.hull_boxes[0].transform(pose_j)
                                if _box_hits_forbidden(wb_j, forbidden_support_boxes):
                                    continue
                                planned_by_z.setdefault(z0, []).append(("15573", pose_j))
                                used_low[ix, iy] = True
                                used_low[nx2, ny2] = True
                                for pt_s in prim_j.ports:
                                    if pt_s.polarity == PortPolarity.STUD:
                                        sp = pose_j.compose(pt_s.local_frame).pos
                                        ux_s = (sp[0] - ph_to[0]) // 20
                                        uy_s = (sp[1] - ph_to[1]) // 20
                                        if 0 <= ux_s < w_x and 0 <= uy_s < w_y:
                                            stud_mask_up[ux_s, uy_s] = True
                                if rem_inv is not None:
                                    rem_inv["15573"] = rem_inv.get("15573", 0) - 1
                    break

            jumper_top_stud_xy[p] = stud_mask_up
            jumper_used_low[p] = used_low.copy()
            jumper_low_occ[p] = low_occ.copy()

        def _support_mask_at_layer(p_layer: int) -> Optional[np.ndarray]:
            if p_layer <= 0:
                return None
            if (p_layer - 1) in jumper_top_stud_xy:
                return jumper_top_stud_xy[p_layer - 1]
            ph_t = layer_phase[p_layer]
            p_b = p_layer - 1
            while p_b > 0 and p_b in jumper_bridge_layers:
                p_b -= 1
            ph_b = layer_phase[p_b]
            if ph_b == ph_t:
                return core_occ[:, :, p_b]
            sup_m = np.zeros((w_x, w_y), dtype=bool)
            for ix in range(w_x):
                for iy in range(w_y):
                    xb0 = ix * 20 + ph_t[0]
                    yb0 = iy * 20 + ph_t[1]
                    for bx in range(max(0, ix - 1), min(w_x, ix + 2)):
                        for by in range(max(0, iy - 1), min(w_y, iy + 2)):
                            if core_occ[bx, by, p_b]:
                                if (
                                    min(xb0 + 20, bx * 20 + ph_b[0] + 20) > max(xb0, bx * 20 + ph_b[0])
                                    and min(yb0 + 20, by * 20 + ph_b[1] + 20) > max(yb0, by * 20 + ph_b[1])
                                ):
                                    sup_m[ix, iy] = True
                                    break
                        if sup_m[ix, iy]:
                            break
            return sup_m

        from topo_brick.kernel.sculpted_stages.snot_and_jumper_stage import (
            SnotAndJumperPlacementStage,
        )

        s3_poses_scored = SnotAndJumperPlacementStage.reserve_and_tile(
            w_x=w_x,
            w_y=w_y,
            n_plates_z=n_plates_z,
            occ6=occ6,
            solid_occ6_padded=solid_occ6_padded,
            gz6_nonempty_limit=gz6_nonempty_limit,
            adaptive_target=adaptive_target,
            kit_ids=kit_ids,
            kit_set=kit_set,
            rem_inv=rem_inv,
            has_thin_panel_kit=has_thin_panel_kit,
            is_thin_regime=is_thin_regime,
            is_hollow_shell_active=is_hollow_shell_active,
            foundation_policy=foundation_policy,
            forbidden_support_boxes=forbidden_support_boxes,
            view_weight_grid_6=view_weight_grid_6,
            deadline_monotonic_s=deadline_monotonic_s,
            layer_phase=layer_phase,
            phase_grids=phase_grids,
            core_occ=core_occ,
            inv_reserved_occ=inv_reserved_occ,
            pier_protected=pier_protected,
            pillar_added_occ=pillar_added_occ,
            snot_spine_protected=snot_spine_protected,
            jumper_bridge_layers=jumper_bridge_layers,
            jumper_top_stud_xy=jumper_top_stud_xy,
            jumper_used_low=jumper_used_low,
            jumper_low_occ=jumper_low_occ,
            brick_templates=brick_templates,
            plate_templates=plate_templates,
            tile_templates=tile_templates,
            planned_by_z=planned_by_z,
            _early_deadline_hit=_early_deadline_hit,
            _support_mask_at_layer=_support_mask_at_layer,
        )
        # Commit planned core bottom-up while maintaining live `occ6_live`, incremental 6-DoF witness,
        # monotone insertion corridors, and `2 LDU` CSG sets!

        return ErodedCorePlan(
            planned_by_z=planned_by_z,
            layer_phase=layer_phase,
            best_ground_phase=best_ground_phase,
            phase_grids=phase_grids,
            core_occ_pre_sculpt=core_occ_pre_sculpt,
            is_thin_regime=bool(is_thin_regime),
            is_legged_frame=bool(is_legged_frame),
            eff_gz6_sky_limit=int(eff_gz6_sky_limit),
            w_x=int(w_x),
            w_y=int(w_y),
            core_cell_thresh=int(core_cell_thresh),
            fast_est_plan_occ6_iou=_fast_est_plan_occ6_iou,
            s3_poses_scored=int(s3_poses_scored),
        )
