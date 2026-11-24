"""
Stage 4 & 4b: Exact global `ΔIoU` open-catalog sculptural, jumper & 24-orientation SNOT skinning
plus top-surface / leaf overfill pruning (`Phase 15`).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Set, Tuple
import numpy as np

from topo_brick.evaluation.static_equilibrium import FoundationPolicy
from topo_brick.interface.brick_catalog import (
    EXTENDED_BRICK_CATALOG,
    ConnectorType,
    PortPolarity,
    get_brick_primitive,
)
from topo_brick.interface.catalog_retrieval_index import (
    CATALOG_RETRIEVAL_INDEX,
    SCULPTURAL_TAXONOMY_CLASSES,
)
from topo_brick.interface.ldu_frames import LDUBox, LDUFrame
from topo_brick.interface.port_manifold import AssemblyState, _get_anchor_templates
from topo_brick.kernel.insertion_corridors import check_monotone_insertion_corridor
from topo_brick.kernel.sculpted_primitives import _fast_upward_stud_templates
from topo_brick.kernel.sculpted_stages.commit_session import SculptedLiveSession


class SculpturalSkinAndPruneStage:
    """Attach sculptural slopes/wedges/curves/SNOT skin and prune overfilled leaf caps."""

    @staticmethod
    def run(sess: SculptedLiveSession) -> Tuple[int, int, int]:
        kernel = sess.kernel
        occ6 = sess.prep.occ6
        kit_ids = sess.prep.kit_ids
        kit_set = sess.prep.kit_set
        smooth_mode = sess.prep.smooth_mode
        foundation_policy = sess.foundation_policy
        deadline_monotonic_s = sess.deadline_monotonic_s
        ground_plane_z_ldu = sess.ground_plane_z_ldu
        _max_waves = sess.max_waves

        is_thin_regime = sess.core_plan.is_thin_regime
        layer_phase = sess.core_plan.layer_phase
        w_x = sess.core_plan.w_x
        w_y = sess.core_plan.w_y
        is_megastructure_footprint = sess.is_megastructure_footprint
        mesh_sub_total = sess.mesh_sub_total
        _eval_fast_pose_delta_iou = sess.eval_fast_pose_delta_iou
        _eval_boxes_delta_iou = sess.eval_boxes_delta_iou
        _protrusion_clear_against_existing = sess.protrusion_clear_against_existing
        _deadline_expired = sess.deadline_expired
        _try_commit_part = sess.try_commit_part
        assert _eval_fast_pose_delta_iou is not None
        assert _eval_boxes_delta_iou is not None
        assert _protrusion_clear_against_existing is not None
        assert _deadline_expired is not None
        assert _try_commit_part is not None

        # 4. Exact Global `ΔIoU` Open-Catalog Sculptural, Jumper & 24-Orientation SNOT Skinning
        #    Build skinning candidate part list from `kit_ids`:
        sculptural_molds: List[Tuple[str, str, bool]] = []
        snot_anchor_molds: List[str] = []
        snot_side_skin_molds: List[str] = []
        jumper_cap_molds: List[str] = []

        for pid in kit_ids:
            ent = CATALOG_RETRIEVAL_INDEX.get_entry(pid)
            cls_name = ent.taxonomy_class if ent is not None else "CORE_PLATE"
            in_comp = pid not in EXTENDED_BRICK_CATALOG
            if cls_name == "SNOT_ANCHOR":
                snot_anchor_molds.append(pid)
                if ent is not None and ent.studs_x * ent.studs_y <= 4:
                    sculptural_molds.append((pid, cls_name, in_comp))
                if ent is not None and ent.studs_x * ent.studs_y <= 2:
                    jumper_cap_molds.append(pid)
            elif cls_name == "JUMPER_HALF_STUD":
                continue
            elif cls_name in SCULPTURAL_TAXONOMY_CLASSES:
                if not is_thin_regime and pid == "6231":
                    continue
                prim_sc = get_brick_primitive(pid)
                n_hbox = len(prim_sc.hull_boxes)
                all_coarse_20x24 = n_hbox >= 1 and all(
                    (b.xmax - b.xmin) % 20 == 0 and (b.ymax - b.ymin) % 20 == 0 and (b.zmax - b.zmin) % 24 == 0
                    for b in prim_sc.hull_boxes
                )
                # Exclude unsliced 1-box, >0.90-fill, or coarse 20x20x24 cuboids masquerading as slopes/arches/panels
                is_genuine_thin_wall = (
                    cls_name == "THIN_PANEL"
                    and n_hbox == 1
                    and min(
                        prim_sc.hull_boxes[0].xmax - prim_sc.hull_boxes[0].xmin,
                        prim_sc.hull_boxes[0].ymax - prim_sc.hull_boxes[0].ymin,
                    ) <= 12
                )
                if (
                    n_hbox == 1
                    and not is_genuine_thin_wall
                    and (
                        cls_name in ("ARCH", "INVERTED_SLOPE", "CURVED_SLOPE", "PLANAR_SLOPE", "STEEP_SLOPE", "THIN_PANEL")
                        or (cls_name == "XY_WEDGE_ROUND" and ent is not None and ent.studs_x * ent.studs_y > 1)
                    )
                ) or (
                    ent is not None
                    and (
                        (ent.fill_ratio > 0.90 and ent.studs_x * ent.studs_y > 1)
                        or (all_coarse_20x24 and cls_name in ("STEEP_SLOPE", "PLANAR_SLOPE", "CURVED_SLOPE"))
                        or (is_thin_regime and cls_name != "THIN_PANEL" and ent.height_ldu >= 32 and ent.fill_ratio > 0.60)
                    )
                ):
                    continue
                sculptural_molds.append((pid, cls_name, in_comp))
                if ent is not None and ent.height_ldu <= 16 and ent.studs_x * ent.studs_y <= 12:
                    snot_side_skin_molds.append(pid)
                if (
                    ent is not None
                    and ent.studs_x * ent.studs_y <= (2 if is_thin_regime else 4)
                    and (
                        ent.height_ldu <= 24
                        or (cls_name == "THIN_PANEL" and ent.height_ldu <= 72)
                    )
                ):
                    jumper_cap_molds.append(pid)
            elif cls_name == "SMOOTH_TILE" and ent is not None and ent.studs_x * ent.studs_y <= 12:
                snot_side_skin_molds.append(pid)
            elif cls_name == "CORE_PLATE" and ent is not None and ent.studs_x * ent.studs_y <= 12:
                if ent.studs_x * ent.studs_y <= (8 if is_thin_regime else 6):
                    sculptural_molds.append((pid, cls_name, in_comp))
                snot_side_skin_molds.append(pid)
                if ent.studs_x * ent.studs_y <= 4:
                    jumper_cap_molds.append(pid)
            elif cls_name == "CORE_BRICK" and ent is not None and ent.studs_x * ent.studs_y <= 6:
                sculptural_molds.append((pid, cls_name, in_comp))
                if pid in ("3005", "3004", "3622"):
                    jumper_cap_molds.append(pid)

        def _cap_cls_rank(pid_j: str) -> int:
            ent_j = CATALOG_RETRIEVAL_INDEX.get_entry(pid_j)
            if ent_j is None:
                return 5
            c_j = ent_j.taxonomy_class
            if c_j == "THIN_PANEL":
                return 0
            if c_j in ("CURVED_SLOPE", "PLANAR_SLOPE", "STEEP_SLOPE", "XY_WEDGE_ROUND"):
                return 1
            if c_j == "CORE_PLATE":
                return 2
            return 3

        jumper_cap_molds.sort(key=lambda p_j: (_cap_cls_rank(p_j), p_j))
        snot_side_skin_molds.sort(
            key=lambda p_s: (
                0
                if (ent_s := CATALOG_RETRIEVAL_INDEX.get_entry(p_s)) is not None and ent_s.taxonomy_class == "SMOOTH_TILE"
                else (1 if ent_s is not None and ent_s.taxonomy_class == "CORE_PLATE" else 2),
                -(ent_s.studs_x * ent_s.studs_y if ent_s is not None else 1),
                p_s,
            )
        )

        # Helper: count uncovered mesh sub-voxels in tight local port neighborhood so we rank true boundary ports first
        def _port_uncovered_count(pos: Tuple[int, int, int], w_norm: Tuple[int, int, int]) -> int:
            gx, gy, gz = occ6.shape
            px, py, pz = pos
            r_xy = 22 if w_norm == (0, 0, 1) else 28
            x0 = max(0, (px - r_xy) // 5)
            x1 = min(gx, (px + r_xy) // 5 + 1)
            y0 = max(0, (py - r_xy) // 5)
            y1 = min(gy, (py + r_xy) // 5 + 1)
            z0 = max(0, (pz - (20 if w_norm[2] <= 0 else 0)) // 6)
            z1 = min(gz, (pz + (26 if w_norm == (0, 0, 1) else 32)) // 6 + 1)
            if x1 <= x0 or y1 <= y0 or z1 <= z0:
                return 0
            return int(np.count_nonzero(occ6[x0:x1, y0:y1, z0:z1] & (~sess.occ6_live[x0:x1, y0:y1, z0:z1])))

        gx6, gy6, gz6 = occ6.shape
        mold_has_top_stud: Dict[str, bool] = {
            pid_m: any(
                pt_m.polarity == PortPolarity.STUD and pt_m.local_frame.z_axis == (0, 0, 1)
                for pt_m in get_brick_primitive(pid_m).ports
            )
            for pid_m in {*(m[0] for m in sculptural_molds), *jumper_cap_molds}
        }

        def _caps_uncovered_upper_mesh(
            wbs_chk: Sequence[LDUBox],
            cls_chk: str = "",
            pid_chk: str = "",
            pose_chk: Optional[LDUFrame] = None,
        ) -> bool:
            """Return True if placing a studless (or partially studless on thin frames) mold at `wbs_chk` caps uncovered upper mesh."""
            if is_thin_regime and cls_chk in ("THIN_PANEL", "SNOT_ANCHOR"):
                return False
            ux0 = max(0, min(b.xmin for b in wbs_chk) // 5)
            ux1 = min(gx6, int(math.ceil(max(b.xmax for b in wbs_chk) / 5.0)))
            uy0 = max(0, min(b.ymin for b in wbs_chk) // 5)
            uy1 = min(gy6, int(math.ceil(max(b.ymax for b in wbs_chk) / 5.0)))
            uz1 = min(gz6, int(math.ceil(max(b.zmax for b in wbs_chk) / 6.0)))
            uz2 = min(gz6, uz1 + 3)
            if uz2 <= uz1 or ux1 <= ux0 or uy1 <= uy0:
                return False
            if not mold_has_top_stud.get(pid_chk, False):
                above_uncov = int(
                    np.count_nonzero(occ6[ux0:ux1, uy0:uy1, uz1:uz2] & (~sess.occ6_live[ux0:ux1, uy0:uy1, uz1:uz2]))
                )
                return above_uncov >= (3 if is_thin_regime else 8)
            if is_thin_regime and pid_chk and pose_chk is not None and cls_chk in ("XY_WEDGE_ROUND", "CURVED_SLOPE", "PLANAR_SLOPE"):
                above_sl = (occ6[ux0:ux1, uy0:uy1, uz1:uz2] & (~sess.occ6_live[ux0:ux1, uy0:uy1, uz1:uz2])).copy()
                if not above_sl.any():
                    return False
                for pt_u in get_brick_primitive(pid_chk).ports:
                    if pt_u.polarity == PortPolarity.STUD:
                        wf_u = pose_chk.compose(pt_u.local_frame)
                        if wf_u.z_axis == (0, 0, 1):
                            sx0_u = max(ux0, (wf_u.pos[0] - 10) // 5) - ux0
                            sx1_u = min(ux1, (wf_u.pos[0] + 10) // 5) - ux0
                            sy0_u = max(uy0, (wf_u.pos[1] - 10) // 5) - uy0
                            sy1_u = min(uy1, (wf_u.pos[1] + 10) // 5) - uy0
                            if sx1_u > sx0_u and sy1_u > sy0_u:
                                above_sl[sx0_u:sx1_u, sy0_u:sy1_u, :] = False
                return int(np.count_nonzero(above_sl)) >= 3
            return False

        # Grounded perimeter pass (`z = 0`): place grounded sculptural/plate/brick/SNOT molds on uncovered ground cells
        if foundation_policy == FoundationPolicy.CLUTCHED_BASEPLATE:
            ground_cands: List[Tuple[float, str, LDUFrame, bool]] = []
            seen_ground: Set[Tuple[str, int, Tuple[int, int, int]]] = set()
            ph0 = layer_phase[0]
            for ix in range(w_x):
                for iy in range(w_y):
                    cx0 = (ix * 20 + ph0[0]) // 5
                    cy0 = (iy * 20 + ph0[1]) // 5
                    if cx0 + 4 > gx6 or cy0 + 4 > gy6:
                        continue
                    if not np.any(occ6[cx0 : cx0 + 4, cy0 : cy0 + 4, :4] & (~sess.occ6_live[cx0 : cx0 + 4, cy0 : cy0 + 4, :4])):
                        continue
                    px_g = ix * 20 + ph0[0] + 10
                    py_g = iy * 20 + ph0[1] + 10
                    for pid_s, cls_s, is_comp_s in sculptural_molds:
                        for rid_t, dx_t, dy_t, dz_t, rboxes_t in _fast_upward_stud_templates(pid_s):
                            pos_g = (px_g + dx_t, py_g + dy_t, dz_t)
                            gkey = (pid_s, rid_t, pos_g)
                            if gkey in seen_ground:
                                continue
                            seen_ground.add(gkey)
                            d_iou = _eval_fast_pose_delta_iou(
                                pid_s, rid_t, pos_g[0], pos_g[1], pos_g[2], require_ground_z0=True
                            )
                            if d_iou <= 2.5e-5:
                                continue
                            wbs_t = tuple(
                                LDUBox(
                                    b[0] + px_g,
                                    b[1] + py_g,
                                    b[2],
                                    b[3] + px_g,
                                    b[4] + py_g,
                                    b[5],
                                )
                                for b in rboxes_t
                            )
                            if (not mold_has_top_stud.get(pid_s, True) or is_thin_regime) and _caps_uncovered_upper_mesh(
                                wbs_t, cls_s, pid_s, LDUFrame(rid_t, pos_g)
                            ):
                                continue
                            if sess.state.check_boxes_collision(wbs_t):
                                continue
                            eps = (1.5e-5 if cls_s in SCULPTURAL_TAXONOMY_CLASSES else 0.0) - 1.5e-5 * float(
                                min(6, sess.build_mold_counts.get(pid_s, 0))
                            ) + (
                                3.0e-5
                                if sess.build_mold_counts.get(pid_s, 0) == 0
                                and cls_s in ("CURVED_SLOPE", "XY_WEDGE_ROUND", "THIN_PANEL", "SNOT_ANCHOR")
                                else 0.0
                            )
                            ground_cands.append((d_iou + eps, pid_s, LDUFrame(rid_t, pos_g), is_comp_s))
            ground_cands.sort(key=lambda x: (-x[0], sess.build_mold_counts.get(x[1], 0), x[1], x[2].rot_id, x[2].pos))
            for _, pid_g, pose_g, _ in ground_cands:
                d_now = _eval_fast_pose_delta_iou(
                    pid_g, pose_g.rot_id, pose_g.pos[0], pose_g.pos[1], pose_g.pos[2], require_ground_z0=True
                )
                if d_now <= 2.5e-5:
                    continue
                wbs_gn = tuple(b.transform(pose_g) for b in get_brick_primitive(pid_g).hull_boxes)
                if sess.state.check_boxes_collision(wbs_gn):
                    continue
                _try_commit_part(pid_g, pose_g, grounded_c=True, check_csg_strict=True, budget_lp=True)

        # Precompute upward-stud templates & overhang inverted-slope molds once per build
        mold_templates_s4 = [
            (pid_s, cls_s, is_comp_s, _fast_upward_stud_templates(pid_s))
            for pid_s, cls_s, is_comp_s in sculptural_molds
        ]
        overhang_inv_molds = [
            (pid_s, cls_s, is_comp_s)
            for pid_s, cls_s, is_comp_s in sculptural_molds
            if cls_s in ("INVERTED_SLOPE", "ARCH")
            or pid_s in ("3024", "3023", "3623", "3710", "3022", "24307", "24299")
        ]

        # Multi-wave frontier sculptural + jumper + SNOT skinning (`up to eff_max_waves fast waves; early-exits when 0 committed`)
        eff_max_waves = 0 if is_megastructure_footprint else min(_max_waves, 2)
        for _wave in range(eff_max_waves):
            snap_parts_w = [(pl_s.part_id, pl_s.pose, pl_s.is_grounded) for _, pl_s in sorted(sess.state.placed_parts.items())]
            snap_occ6_w = sess.occ6_live.copy()
            snap_inter_w = sess.inter_live
            snap_over_w = sess.over_live
            snap_counts_w = dict(sess.build_mold_counts)
            snap_rem_w = dict(sess.rem_inv) if sess.rem_inv is not None else None
            snap_witness_w = sess.base_witness
            snap_eta_w = sess.base_lp_eta
            snap_deltas_w = dict(sess.plumb_deltas)
            sess.deferred_lp_in_batch = False
            sess.batch_new_commits = []

            frontier = sess.state.get_active_frontier()
            ranked_ports: List[Tuple[int, object]] = []
            for ep in frontier.ports:
                if ep.status.is_exposed and ep.connector_type in (ConnectorType.STANDARD_STUD, ConnectorType.HOLLOW_STUD):
                    uc_cnt = _port_uncovered_count(ep.world_frame.pos, ep.world_normal)
                    if uc_cnt > 0:
                        ranked_ports.append((uc_cnt, ep))
            if not ranked_ports:
                break
            ranked_ports.sort(
                key=lambda item: (
                    0 if item[1].world_normal[2] == 0 else 1,
                    -item[0],
                    item[1].world_frame.pos[2],
                    item[1].world_frame.pos[0],
                    item[1].world_frame.pos[1],
                )
            )
            port_cap = (56 if _wave == 0 else 28) if is_thin_regime else ((128 if mesh_sub_total >= 5000 else 80) if _wave == 0 else 48)
            active_ports = [item[1] for item in ranked_ports[:port_cap]]

            scored_single: List[Tuple[float, str, LDUFrame, bool]] = []
            seen_poses: Set[Tuple[str, int, Tuple[int, int, int]]] = set()

            for ep in active_ports:
                ux, uy, uz = ep.world_frame.pos
                u_rid = ep.world_frame.rot_id
                u_pol = ep.polarity
                u_ct = ep.connector_type
                wn = ep.world_normal

                if wn == (0, 0, 1) and u_pol == PortPolarity.STUD:
                    for pid_s, cls_s, is_comp_s, tmpls_s in mold_templates_s4:
                        if sess.rem_inv is not None and sess.rem_inv.get(pid_s, 0) <= 0:
                            continue
                        for rid_t, dx_t, dy_t, dz_t, rboxes_t in tmpls_s:
                            pos_w = (ux + dx_t, uy + dy_t, uz + dz_t)
                            pkey = (pid_s, rid_t, pos_w)
                            if pkey in seen_poses:
                                continue
                            seen_poses.add(pkey)
                            d_iou = _eval_fast_pose_delta_iou(pid_s, rid_t, pos_w[0], pos_w[1], pos_w[2])
                            if d_iou <= 2.5e-5:
                                continue
                            wbs_t = tuple(
                                LDUBox(b[0] + ux, b[1] + uy, b[2] + uz, b[3] + ux, b[4] + uy, b[5] + uz)
                                for b in rboxes_t
                            )
                            if _wave == 0 and (not mold_has_top_stud.get(pid_s, True) or is_thin_regime) and _caps_uncovered_upper_mesh(
                                wbs_t, cls_s, pid_s, LDUFrame(rid_t, pos_w)
                            ):
                                continue
                            if sess.state.check_boxes_collision(wbs_t):
                                continue
                            eps = -2.0e-5 * float(min(10, sess.build_mold_counts.get(pid_s, 0)))
                            if sess.build_mold_counts.get(pid_s, 0) == 0 and cls_s in (
                                "CURVED_SLOPE",
                                "PLANAR_SLOPE",
                                "STEEP_SLOPE",
                                "XY_WEDGE_ROUND",
                                "THIN_PANEL",
                                "SNOT_ANCHOR",
                                "INVERTED_SLOPE",
                                "ARCH",
                                "SMOOTH_TILE",
                            ):
                                eps += 3.5e-5
                            if cls_s in ("CURVED_SLOPE", "XY_WEDGE_ROUND", "INVERTED_SLOPE", "ARCH", "STEEP_SLOPE", "THIN_PANEL"):
                                eps += 2.0e-5
                            elif cls_s == "PLANAR_SLOPE":
                                eps += 1.5e-5
                            elif cls_s == "SMOOTH_TILE":
                                eps += 3.0e-5 if smooth_mode else 1.5e-5
                            scored_single.append((d_iou + eps, pid_s, LDUFrame(rid_t, pos_w), is_comp_s))
                elif u_pol == PortPolarity.ANTI_STUD and ep.world_frame.z_axis == (0, 0, 1) and uz >= 16:
                    for pid_s, cls_s, is_comp_s in overhang_inv_molds:
                        if sess.rem_inv is not None and sess.rem_inv.get(pid_s, 0) <= 0:
                            continue
                        for tmpl in _get_anchor_templates(u_rid, u_pol, u_ct, pid_s):
                            rid_t, dx_t, dy_t, dz_t, rboxes_t, _, _, _ = tmpl
                            if LDUFrame(rid_t, (0, 0, 0)).z_axis != (0, 0, 1):
                                continue
                            pos_w = (ux + dx_t, uy + dy_t, uz + dz_t)
                            pkey = (pid_s, rid_t, pos_w)
                            if pkey in seen_poses:
                                continue
                            seen_poses.add(pkey)
                            d_iou = _eval_fast_pose_delta_iou(pid_s, rid_t, pos_w[0], pos_w[1], pos_w[2])
                            if d_iou <= 2.5e-5:
                                continue
                            wbs_t = tuple(
                                LDUBox(b[0] + ux, b[1] + uy, b[2] + uz, b[3] + ux, b[4] + uy, b[5] + uz)
                                for b in rboxes_t
                            )
                            _, s_din, s_dov = _eval_boxes_delta_iou(wbs_t)
                            if s_din < s_dov:
                                continue
                            if sess.state.check_boxes_collision(wbs_t):
                                continue
                            ov_first_b = 3.5e-5 if sess.build_mold_counts.get(pid_s, 0) == 0 else 0.0
                            scored_single.append(
                                (
                                    d_iou + 3.0e-5 + ov_first_b - 2.0e-5 * float(min(10, sess.build_mold_counts.get(pid_s, 0))),
                                    pid_s,
                                    LDUFrame(rid_t, pos_w),
                                    is_comp_s,
                                )
                            )
                elif wn[2] == 0 and u_pol == PortPolarity.STUD:
                    for pid_s in snot_side_skin_molds:
                        if sess.rem_inv is not None and sess.rem_inv.get(pid_s, 0) <= 0:
                            continue
                        is_comp_s = pid_s not in EXTENDED_BRICK_CATALOG
                        for tmpl in _get_anchor_templates(u_rid, u_pol, u_ct, pid_s):
                            rid_t, dx_t, dy_t, dz_t, rboxes_t, _, _, _ = tmpl
                            pos_w = (ux + dx_t, uy + dy_t, uz + dz_t)
                            pkey = (pid_s, rid_t, pos_w)
                            if pkey in seen_poses:
                                continue
                            seen_poses.add(pkey)
                            d_iou = _eval_fast_pose_delta_iou(pid_s, rid_t, pos_w[0], pos_w[1], pos_w[2])
                            if d_iou <= 2.5e-5:
                                continue
                            wbs_t = tuple(
                                LDUBox(b[0] + ux, b[1] + uy, b[2] + uz, b[3] + ux, b[4] + uy, b[5] + uz)
                                for b in rboxes_t
                            )
                            if sess.state.check_boxes_collision(wbs_t):
                                continue
                            scored_single.append(
                                (
                                    d_iou + 8.0e-4 - 4.0e-5 * float(min(10, sess.build_mold_counts.get(pid_s, 0))),
                                    pid_s,
                                    LDUFrame(rid_t, pos_w),
                                    is_comp_s,
                                )
                            )

            # Evaluate 2-part `(15573 jumper + half-stud cap)` & `(SNOT anchor + sideways tile/plate)` micro-motifs
            # IN THE SAME UNIFIED `ΔIoU` PRIORITY QUEUE as single parts so high-precision half-stud tier ledges
            # (`b4_unseen_01`, `b3_unseen_01`, `b2_unseen_02`, `b1_ext_e8`) win BEFORE low-precision smooth tiles cap studs!
            micro_pairs: List[Tuple[float, str, LDUFrame, str, LDUFrame, bool]] = []
            run_micro_wave = (_wave == 0) or (_wave == 1 and is_thin_regime)
            if run_micro_wave and (
                ("15573" in kit_set and jumper_cap_molds) or (snot_anchor_molds and snot_side_skin_molds)
            ):
                up_ports = [
                    ep_u
                    for ep_u in active_ports
                    if ep_u.polarity == PortPolarity.STUD and ep_u.world_normal == (0, 0, 1)
                ][: (44 if mesh_sub_total >= 5000 else 28)]

                if _wave in (0, 1) and "15573" in kit_set and jumper_cap_molds:
                    j_anc_cap = 28 if mesh_sub_total >= 2500 else 18
                    j_mold_cap = 22 if (is_thin_regime or mesh_sub_total < 4500) else 14
                    cand_j_anchors: List[Tuple[int, int, int, bool]] = [
                        (ep.world_frame.pos[0], ep.world_frame.pos[1], ep.world_frame.pos[2], False)
                        for ep in up_ports[:j_anc_cap]
                    ]
                    if _wave == 0 and foundation_policy == FoundationPolicy.CLUTCHED_BASEPLATE and min(w_x, w_y) <= 2:
                        ph0_m = layer_phase[0]
                        for ix_g in range(w_x):
                            for iy_g in range(w_y):
                                cand_j_anchors.append((ix_g * 20 + ph0_m[0] + 10, iy_g * 20 + ph0_m[1] + 10, 0, True))
                    for ux, uy, uz, is_g_j in cand_j_anchors:
                        for rid_j, dx_j, dy_j, dz_j, rboxes_j in _fast_upward_stud_templates("15573"):
                            pos_j = (ux + dx_j, uy + dy_j, uz + dz_j)
                            wbs_j = tuple(
                                LDUBox(b[0] + ux, b[1] + uy, b[2] + uz, b[3] + ux, b[4] + uy, b[5] + uz)
                                for b in rboxes_j
                            )
                            if min(b.zmin for b in wbs_j) < 0 or sess.state.check_boxes_collision(wbs_j):
                                continue
                            jf_w = LDUFrame(rid_j, pos_j)
                            s_ports = [
                                jf_w.compose(pt_s.local_frame)
                                for pt_s in get_brick_primitive("15573").ports
                                if pt_s.polarity == PortPolarity.STUD
                            ]
                            if not s_ports:
                                continue
                            wx_s, wy_s, wz_s = s_ports[0].pos
                            for cap_pid in jumper_cap_molds[:j_mold_cap]:
                                ent_cap = CATALOG_RETRIEVAL_INDEX.get_entry(cap_pid)
                                cls_cap = ent_cap.taxonomy_class if ent_cap is not None else "CORE_PLATE"
                                for rid_c, dx_c, dy_c, dz_c, rboxes_c in _fast_upward_stud_templates(cap_pid):
                                    pos_c = (wx_s + dx_c, wy_s + dy_c, wz_s + dz_c)
                                    if (
                                        _eval_fast_pose_delta_iou(
                                            cap_pid,
                                            rid_c,
                                            pos_c[0],
                                            pos_c[1],
                                            pos_c[2],
                                            enforce_min_precision=(not is_thin_regime),
                                        )
                                        <= 0.0
                                    ):
                                        continue
                                    wbs_c = tuple(
                                        LDUBox(b[0] + wx_s, b[1] + wy_s, b[2] + wz_s, b[3] + wx_s, b[4] + wy_s, b[5] + wz_s)
                                        for b in rboxes_c
                                    )
                                    if sess.state.check_boxes_collision(wbs_c) or (
                                        (not mold_has_top_stud.get(cap_pid, True) or is_thin_regime)
                                        and _caps_uncovered_upper_mesh(wbs_j + wbs_c, cls_cap, cap_pid, LDUFrame(rid_c, pos_c))
                                    ):
                                        continue
                                    d_pair, _, _ = _eval_boxes_delta_iou(
                                        wbs_j + wbs_c, enforce_min_precision=(not is_thin_regime)
                                    )
                                    if d_pair > 1.5e-4:
                                        c_first_b = (
                                            3.5e-5
                                            if sess.build_mold_counts.get(cap_pid, 0) == 0
                                            and cls_cap in ("CURVED_SLOPE", "XY_WEDGE_ROUND", "THIN_PANEL", "SNOT_ANCHOR")
                                            else 0.0
                                        )
                                        micro_pairs.append(
                                            (d_pair + c_first_b, "15573", jf_w, cap_pid, LDUFrame(rid_c, pos_c), is_g_j)
                                        )

                if snot_anchor_molds and snot_side_skin_molds and (_wave == 0 or (is_thin_regime and _wave <= 1)) and (is_thin_regime or mesh_sub_total < 4500):
                    anc_cap_n = 4 if (is_thin_regime and mesh_sub_total < 900) else 3
                    sk_cap_n = 8
                    anc_try = [a for a in ("4070", "44728", "99780", "87087", "11211", "30414") if a in snot_anchor_molds][:anc_cap_n]
                    cand_s_anchors: List[Tuple[int, int, int, bool]] = [
                        (ep.world_frame.pos[0], ep.world_frame.pos[1], ep.world_frame.pos[2], False)
                        for ep in up_ports[: (14 if is_thin_regime else 6)]
                    ]
                    if (
                        is_thin_regime
                        and foundation_policy == FoundationPolicy.CLUTCHED_BASEPLATE
                        and min(w_x, w_y) <= 3
                    ):
                        ph0_s = layer_phase[0]
                        for ix_gs in range(w_x):
                            for iy_gs in range(w_y):
                                cx0_g = (ix_gs * 20 + ph0_s[0]) // 5
                                cy0_g = (iy_gs * 20 + ph0_s[1]) // 5
                                if (
                                    cx0_g + 4 <= gx6
                                    and cy0_g + 4 <= gy6
                                    and int(np.count_nonzero(occ6[cx0_g : cx0_g + 4, cy0_g : cy0_g + 4, 0])) >= 4
                                    and (occ6[cx0_g : cx0_g + 4, cy0_g : cy0_g + 4, :6] & (~sess.occ6_live[cx0_g : cx0_g + 4, cy0_g : cy0_g + 4, :6])).any()
                                ):
                                    cand_s_anchors.append((ix_gs * 20 + ph0_s[0] + 10, iy_gs * 20 + ph0_s[1] + 10, 0, True))
                    for ux, uy, uz, is_g_s in cand_s_anchors[: (16 if is_thin_regime else 8)]:
                        for anc_pid in anc_try:
                            prim_anc = get_brick_primitive(anc_pid)
                            for rid_a, dx_a, dy_a, dz_a, rboxes_a in _fast_upward_stud_templates(anc_pid):
                                pos_a = (ux + dx_a, uy + dy_a, uz + dz_a)
                                if is_g_s:
                                    if min(b[2] for b in rboxes_a) != 0:
                                        continue
                                    if _eval_fast_pose_delta_iou(anc_pid, rid_a, pos_a[0], pos_a[1], pos_a[2], require_ground_z0=True) <= 0.0:
                                        continue
                                wbs_a = tuple(
                                    LDUBox(b[0] + ux, b[1] + uy, b[2] + uz, b[3] + ux, b[4] + uy, b[5] + uz)
                                    for b in rboxes_a
                                )
                                if min(b.zmin for b in wbs_a) < 0 or sess.state.check_boxes_collision(wbs_a):
                                    continue
                                af_w = LDUFrame(rid_a, pos_a)
                                lat_ports = [
                                    (af_w.compose(pt_l.local_frame), pt_l.polarity, pt_l.connector_type)
                                    for pt_l in prim_anc.ports
                                    if pt_l.polarity == PortPolarity.STUD and af_w.compose(pt_l.local_frame).z_axis[2] == 0
                                ]
                                if not lat_ports:
                                    continue
                                for lf_w, l_pol, l_ct in lat_ports[: (4 if is_thin_regime else 2)]:
                                    wx_l, wy_l, wz_l = lf_w.pos
                                    for sk_pid in snot_side_skin_molds[:sk_cap_n]:
                                        for tmpl_k in _get_anchor_templates(lf_w.rot_id, l_pol, l_ct, sk_pid):
                                            rid_k, dx_k, dy_k, dz_k, rboxes_k, _, _, _ = tmpl_k
                                            pos_k = (wx_l + dx_k, wy_l + dy_k, wz_l + dz_k)
                                            if _eval_fast_pose_delta_iou(sk_pid, rid_k, pos_k[0], pos_k[1], pos_k[2]) <= 1.0e-4:
                                                continue
                                            wbs_k = tuple(
                                                LDUBox(b[0] + wx_l, b[1] + wy_l, b[2] + wz_l, b[3] + wx_l, b[4] + wy_l, b[5] + wz_l)
                                                for b in rboxes_k
                                            )
                                            if min(b.zmin for b in wbs_k) < 0 or sess.state.check_boxes_collision(wbs_k):
                                                continue
                                            if any(ba.intersects(bk) for ba in wbs_a for bk in wbs_k):
                                                continue
                                            d_pair, _, _ = _eval_boxes_delta_iou(
                                                wbs_a + wbs_k, enforce_min_precision=(not is_thin_regime)
                                            )
                                            if d_pair > 2.5e-4:
                                                anc_first_b = 2.2e-4 if sess.build_mold_counts.get(anc_pid, 0) == 0 else 0.0
                                                micro_pairs.append(
                                                    (d_pair + 1.5e-4 + anc_first_b, anc_pid, af_w, sk_pid, LDUFrame(rid_k, pos_k), is_g_s)
                                                )

            # Unified greedy `ΔIoU` commit queue over both 1-part and 2-part frontier actions
            unified_actions: List[Tuple[float, int, str, LDUFrame, Optional[str], Optional[LDUFrame], bool]] = [
                (sc_1, 1, p_1, f_1, None, None, False) for sc_1, p_1, f_1, _ in scored_single
            ]
            if micro_pairs:
                micro_pairs.sort(key=lambda x: (-x[0], x[1], x[2].rot_id, x[2].pos, x[3], x[4].rot_id, x[4].pos))
                anc_pair_cnt: Dict[Tuple[str, Tuple[int, int, int]], int] = {}
                added_mp = 0
                for sc_2, p1_m, f1_m, p2_m, f2_m, g1_m in micro_pairs:
                    akey_m = (p1_m, f1_m.pos)
                    c_ak = anc_pair_cnt.get(akey_m, 0)
                    if c_ak >= 2:
                        continue
                    anc_pair_cnt[akey_m] = c_ak + 1
                    unified_actions.append((sc_2, 2, p1_m, f1_m, p2_m, f2_m, g1_m))
                    added_mp += 1
                    if added_mp >= 56:
                        break
            unified_actions.sort(
                key=lambda x: (-x[0], x[1], sess.build_mold_counts.get(x[2], 0), x[2], x[3].rot_id, x[3].pos)
            )

            committed_in_wave = 0
            for _, kind_a, p1_a, f1_a, p2_a, f2_a, g1_a in unified_actions:
                if kind_a == 1:
                    d_now = _eval_fast_pose_delta_iou(
                        p1_a, f1_a.rot_id, f1_a.pos[0], f1_a.pos[1], f1_a.pos[2]
                    )
                    if d_now <= 2.5e-5:
                        continue
                    wbs_now = tuple(b.transform(f1_a) for b in get_brick_primitive(p1_a).hull_boxes)
                    if sess.state.check_boxes_collision(wbs_now):
                        continue
                    if _try_commit_part(
                        p1_a, f1_a, grounded_c=False, check_csg_strict=True, budget_lp=True, defer_lp=True
                    ):
                        committed_in_wave += 1
                elif p2_a is not None and f2_a is not None:
                    wb1 = tuple(b.transform(f1_a) for b in get_brick_primitive(p1_a).hull_boxes)
                    wb2 = tuple(b.transform(f2_a) for b in get_brick_primitive(p2_a).hull_boxes)
                    if sess.state.check_boxes_collision(wb1) or sess.state.check_boxes_collision(wb2):
                        continue
                    if not check_monotone_insertion_corridor(sess.state, p2_a, f2_a) or not _protrusion_clear_against_existing(p2_a, f2_a):
                        continue
                    dn_pair, _, _ = _eval_boxes_delta_iou(wb1 + wb2, enforce_min_precision=(not is_thin_regime))
                    if dn_pair <= 1.2e-4:
                        continue
                    if _try_commit_part(p1_a, f1_a, grounded_c=g1_a, check_csg_strict=True, budget_lp=True, defer_lp=True):
                        if _try_commit_part(p2_a, f2_a, grounded_c=False, check_csg_strict=True, budget_lp=True, defer_lp=True):
                            committed_in_wave += 2

            if sess.deferred_lp_in_batch:
                sess.skin_lp_calls += 1
                eq_wave = kernel.validator.evaluate(
                    sess.state,
                    foundation_policy=foundation_policy,
                    ground_plane_z_ldu=ground_plane_z_ldu,
                    deadline_monotonic_s=deadline_monotonic_s,
                )
                if eq_wave.is_stable:
                    sess.base_witness = eq_wave.primal_witness
                    sess.base_lp_eta = float(eq_wave.peak_utilization)
                    sess.plumb_deltas.clear()
                elif not eq_wave.budget_exceeded:
                    # Roll back speculative batch and replay per-part with exact individual LP certification
                    sess.state = AssemblyState()
                    for p_rb, f_rb, g_rb in snap_parts_w:
                        sess.state.place_root_part(p_rb, f_rb, is_grounded=g_rb)
                    sess.occ6_live[:] = snap_occ6_w
                    sess.inter_live = snap_inter_w
                    sess.over_live = snap_over_w
                    sess.build_mold_counts = snap_counts_w
                    if sess.rem_inv is not None and snap_rem_w is not None:
                        sess.rem_inv.clear()
                        sess.rem_inv.update(snap_rem_w)
                    sess.base_witness = snap_witness_w
                    sess.base_lp_eta = snap_eta_w
                    sess.plumb_deltas.clear()
                    sess.plumb_deltas.update(snap_deltas_w)
                    sess.plumb_hop_cache.clear()
                    sess.deferred_lp_in_batch = False
                    committed_in_wave = 0
                    for p_rp, f_rp, g_rp in sess.batch_new_commits:
                        if _deadline_expired():
                            break
                        if _try_commit_part(
                            p_rp, f_rp, grounded_c=g_rp, check_csg_strict=True, budget_lp=True, defer_lp=False
                        ):
                            committed_in_wave += 1

            if committed_in_wave == 0 or _deadline_expired():
                break

        # 4b. Exact Global `ΔIoU` Top-Surface & Leaf Part Prune (`~0.3 ms`): greedily prune any top-surface or
        #     degree-1 part that is net-negative at the final assembly IoU (up to 2 top-down passes), verifying
        #     monotone insertion corridors, full graph connectivity, and 6-DoF static stability on every removal.
        if len(sess.state.placed_parts) > 1 and mesh_sub_total > 0 and not is_megastructure_footprint and not _deadline_expired():
            def _try_build_pruned_state(skip_iids: Set[int]) -> Optional[AssemblyState]:
                if _deadline_expired():
                    return None
                cand_st = AssemblyState()
                for iid_r in sorted(sess.state.placed_parts.keys()):
                    if iid_r in skip_iids:
                        continue
                    pl_r = sess.state.placed_parts[iid_r]
                    if not check_monotone_insertion_corridor(cand_st, pl_r.part_id, pl_r.pose):
                        return None
                    prev_m_len = len(cand_st.matched_port_pairs)
                    try:
                        cand_st.place_root_part(pl_r.part_id, pl_r.pose, is_grounded=pl_r.is_grounded)
                    except ValueError:
                        return None
                    if not pl_r.is_grounded and len(cand_st.matched_port_pairs) == prev_m_len:
                        return None
                if not cand_st.placed_parts:
                    return None
                eq_pr = kernel.validator.evaluate(
                    cand_st,
                    foundation_policy=foundation_policy,
                    ground_plane_z_ldu=ground_plane_z_ldu,
                    deadline_monotonic_s=deadline_monotonic_s,
                )
                return cand_st if eq_pr.is_stable else None

            for _pr_pass in range(16):
                if len(sess.state.placed_parts) <= 1 or _deadline_expired():
                    break
                adj_insts: Dict[int, Set[int]] = {k_i: set() for k_i in sess.state.placed_parts}
                inst_zmax: Dict[int, int] = {
                    k_i: max(b.zmax for b in pl_i.world_boxes) for k_i, pl_i in sess.state.placed_parts.items()
                }
                has_higher_nbr: Set[int] = set()
                for (ia_m, _), (ib_m, _) in sess.state.matched_port_pairs.items():
                    if ia_m != ib_m and ia_m in adj_insts and ib_m in adj_insts:
                        adj_insts[ia_m].add(ib_m)
                        adj_insts[ib_m].add(ia_m)
                        if inst_zmax[ib_m] > inst_zmax[ia_m]:
                            has_higher_nbr.add(ia_m)
                        elif inst_zmax[ia_m] > inst_zmax[ib_m]:
                            has_higher_nbr.add(ib_m)
                leaf_cands = [
                    iid_l
                    for iid_l, pl_l in sess.state.placed_parts.items()
                    if (not pl_l.is_grounded and (len(adj_insts[iid_l]) == 1 or iid_l not in has_higher_nbr))
                    or (pl_l.is_grounded and len(adj_insts[iid_l]) == 0)
                ]
                if not leaf_cands:
                    break

                occ6_cnt = np.zeros(occ6.shape, dtype=np.uint16)
                inst_masks: Dict[int, Tuple[int, int, int, int, int, int, np.ndarray]] = {}
                for iid_p, pl_p in sess.state.placed_parts.items():
                    wbs_p = pl_p.world_boxes
                    ux0 = max(0, int(math.floor(min(b.xmin for b in wbs_p) / 5.0)))
                    ux1 = min(gx6, int(math.ceil(max(b.xmax for b in wbs_p) / 5.0)))
                    uy0 = max(0, int(math.floor(min(b.ymin for b in wbs_p) / 5.0)))
                    uy1 = min(gy6, int(math.ceil(max(b.ymax for b in wbs_p) / 5.0)))
                    uz0 = max(0, int(math.floor(min(b.zmin for b in wbs_p) / 6.0)))
                    uz1 = min(gz6, int(math.ceil(max(b.zmax for b in wbs_p) / 6.0)))
                    if ux1 <= ux0 or uy1 <= uy0 or uz1 <= uz0:
                        continue
                    loc_m = np.zeros((ux1 - ux0, uy1 - uy0, uz1 - uz0), dtype=bool)
                    for b in wbs_p:
                        bx0 = max(ux0, int(math.floor(b.xmin / 5.0))) - ux0
                        bx1 = min(ux1, int(math.ceil(b.xmax / 5.0))) - ux0
                        by0 = max(uy0, int(math.floor(b.ymin / 5.0))) - uy0
                        by1 = min(uy1, int(math.ceil(b.ymax / 5.0))) - uy0
                        bz0 = max(uz0, int(math.floor(b.zmin / 6.0))) - uz0
                        bz1 = min(uz1, int(math.ceil(b.zmax / 6.0))) - uz0
                        if bx1 > bx0 and by1 > by0 and bz1 > bz0:
                            loc_m[bx0:bx1, by0:by1, bz0:bz1] = True
                    occ6_cnt[ux0:ux1, uy0:uy1, uz0:uz1] += loc_m.astype(np.uint16)
                    inst_masks[iid_p] = (ux0, ux1, uy0, uy1, uz0, uz1, loc_m)

                neg_cands: List[Tuple[float, int]] = []
                iou_base = float(sess.inter_live) / float(max(1, mesh_sub_total + sess.over_live))
                for iid_l in leaf_cands:
                    m_info = inst_masks.get(iid_l)
                    if m_info is None:
                        continue
                    ux0, ux1, uy0, uy1, uz0, uz1, loc_m = m_info
                    uniq_m = loc_m & (occ6_cnt[ux0:ux1, uy0:uy1, uz0:uz1] == 1)
                    if not np.any(uniq_m):
                        continue
                    tgt_u = occ6[ux0:ux1, uy0:uy1, uz0:uz1]
                    lost_in = int(np.count_nonzero(uniq_m & tgt_u))
                    saved_ov = int(np.count_nonzero(uniq_m & (~tgt_u)))
                    if sess.inter_live - lost_in <= 0:
                        continue
                    iou_after = float(sess.inter_live - lost_in) / float(max(1, mesh_sub_total + sess.over_live - saved_ov))
                    if iou_after > iou_base + 1.0e-5:
                        neg_cands.append((iou_after - iou_base, iid_l))

                if not neg_cands:
                    break
                neg_cands.sort(key=lambda x: (-x[0], -x[1]))

                # Fast batch try first; if blocked by a bridge dependency, verify each candidate one-by-one!
                batch_ids = {iid_n for _, iid_n in neg_cands}
                batch_st = _try_build_pruned_state(batch_ids) if len(batch_ids) > 1 else None
                if batch_st is not None:
                    occ6_new = np.zeros_like(occ6)
                    for pl_k in batch_st.placed_parts.values():
                        for b in pl_k.world_boxes:
                            occ6_new[
                                max(0, b.xmin // 5) : min(gx6, (b.xmax + 4) // 5),
                                max(0, b.ymin // 5) : min(gy6, (b.ymax + 4) // 5),
                                max(0, b.zmin // 6) : min(gz6, (b.zmax + 5) // 6),
                            ] = True
                    in_new = int(np.count_nonzero(occ6_new & occ6))
                    ov_new = int(np.count_nonzero(occ6_new & (~occ6)))
                    if in_new > 0 and (float(in_new) / float(max(1, mesh_sub_total + ov_new))) > iou_base + 1.0e-5:
                        if sess.rem_inv is not None:
                            for iid_d in batch_ids:
                                pid_d = sess.state.placed_parts[iid_d].part_id
                                can_d = CATALOG_RETRIEVAL_INDEX.resolve_canonical_mold_id(pid_d) or pid_d
                                sess.rem_inv[can_d] = sess.rem_inv.get(can_d, 0) + 1
                        sess.state = batch_st
                        sess.plumb_hop_cache.clear()
                        sess.occ6_live = occ6_new
                        sess.inter_live = in_new
                        sess.over_live = ov_new
                        continue

                dropped_any = False
                for _, iid_one in neg_cands[:8]:
                    if iid_one not in sess.state.placed_parts:
                        continue
                    m_info = inst_masks.get(iid_one)
                    if m_info is None:
                        continue
                    ux0, ux1, uy0, uy1, uz0, uz1, loc_m = m_info
                    uniq_m = loc_m & (occ6_cnt[ux0:ux1, uy0:uy1, uz0:uz1] == 1)
                    if not np.any(uniq_m):
                        continue
                    tgt_u = occ6[ux0:ux1, uy0:uy1, uz0:uz1]
                    lost_in = int(np.count_nonzero(uniq_m & tgt_u))
                    saved_ov = int(np.count_nonzero(uniq_m & (~tgt_u)))
                    if sess.inter_live - lost_in <= 0:
                        continue
                    iou_cur_p = float(sess.inter_live) / float(max(1, mesh_sub_total + sess.over_live))
                    iou_nxt_p = float(sess.inter_live - lost_in) / float(max(1, mesh_sub_total + sess.over_live - saved_ov))
                    if iou_nxt_p <= iou_cur_p + 1.0e-5:
                        continue
                    one_st = _try_build_pruned_state({iid_one})
                    if one_st is not None:
                        pid_d = sess.state.placed_parts[iid_one].part_id
                        can_d = CATALOG_RETRIEVAL_INDEX.resolve_canonical_mold_id(pid_d) or pid_d
                        if sess.rem_inv is not None:
                            sess.rem_inv[can_d] = sess.rem_inv.get(can_d, 0) + 1
                        # Note: `place_root_part` re-indexes `inst_id` sequentially `0..N-1`; rebuild `occ6_cnt` on next pass
                        sess.state = one_st
                        sess.plumb_hop_cache.clear()
                        occ6_cnt[ux0:ux1, uy0:uy1, uz0:uz1] -= loc_m.astype(np.uint16)
                        sess.inter_live -= lost_in
                        sess.over_live -= saved_ov
                        dropped_any = True
                        break
                if not dropped_any:
                    break
                # Sync `occ6_live` if single-item prune mutated `state` (`so any subsequent logic sees exact state`)
                sess.occ6_live.fill(False)
                for pl_s in sess.state.placed_parts.values():
                    for b in pl_s.world_boxes:
                        sess.occ6_live[
                            max(0, b.xmin // 5) : min(gx6, (b.xmax + 4) // 5),
                            max(0, b.ymin // 5) : min(gy6, (b.ymax + 4) // 5),
                            max(0, b.zmin // 6) : min(gz6, (b.zmax + 5) // 6),
                        ] = True
                sess.inter_live = int(np.count_nonzero(sess.occ6_live & occ6))
                sess.over_live = int(np.count_nonzero(sess.occ6_live & (~occ6)))


        return int(gx6), int(gy6), int(gz6)
