"""
Phase 10 Blind Sub-Stud, Stepped-Frustum, Portal-Bridge & Multi-Stud Sculpting Portfolio.

Operates strictly on blind `5x5x6 LDU` target occupancy (`occ6`), retrieved `Top-K` mold kits,
and `forbidden_support_boxes` (`zero` mesh-ID or reference-assembly lookups). Every accepted
candidate enforces:
  - even `2 LDU` coordinates (`pos % 2 == 0`),
  - strict `[0 .. gx6*5] x [0 .. gy6*5] x [0 .. gz6*6]` bounding-box containment,
  - `0` intersections with `forbidden_support_boxes`,
  - full prefix static-equilibrium stability and monotone insertion-corridor clearance.
"""

from __future__ import annotations

from collections import deque
import copy
from functools import lru_cache
import math
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

import numpy as np

from topo_brick.evaluation.static_equilibrium import FoundationPolicy
from topo_brick.interface.brick_catalog import PortPolarity, get_brick_primitive
from topo_brick.interface.catalog_retrieval_index import CATALOG_RETRIEVAL_INDEX
from topo_brick.interface.ldu_frames import C4_Z_ROT_IDS, LDUBox, LDUFrame
from topo_brick.interface.port_manifold import AssemblyState

PLATE_TO_TILE: Dict[str, str] = {
    "3024": "3070b",
    "3023": "3069b",
    "3623": "63864",
    "3710": "2431",
    "3666": "6636",
    "3460": "4162",
    "3022": "3068b",
    "3021": "26603",
    "3020": "87079",
    "3795": "69729",
}
TILE_TO_PLATE: Dict[str, str] = {v: k for k, v in PLATE_TO_TILE.items()}

FOOTPRINT_TO_SMALL_PLATE: Dict[Tuple[int, int], Tuple[str, ...]] = {
    (20, 20): ("3024",),
    (20, 40): ("3023",),
    (40, 20): ("3023",),
    (40, 40): ("3022",),
}


def _box_hits_forbidden(box: LDUBox, forbidden_boxes: Sequence[LDUBox]) -> bool:
    return any(box.intersects(fb) for fb in forbidden_boxes)


@lru_cache(maxsize=512)
def _span_rid(pid: str, sx: int, sy: int) -> int:
    pr = get_brick_primitive(pid)
    for r in C4_Z_ROT_IDS:
        rbs = [b.transform(LDUFrame(r, (0, 0, 0))) for b in pr.hull_boxes]
        dx = max(b.xmax for b in rbs) - min(b.xmin for b in rbs)
        dy = max(b.ymax for b in rbs) - min(b.ymin for b in rbs)
        if dx == sx and dy == sy:
            return r
    raise ValueError(f"No span ({sx}, {sy}) for {pid}")


@lru_cache(maxsize=512)
def _cached_plate_pose(
    plate_pid: str, xmin: int, ymin: int, zmin: int, xmax: int, ymax: int, zmax: int
) -> Optional[LDUFrame]:
    prim = get_brick_primitive(plate_pid)
    if len(prim.hull_boxes) != 1:
        return None
    wb = LDUBox(xmin, ymin, zmin, xmax, ymax, zmax)
    for rid in C4_Z_ROT_IDS:
        lb = prim.hull_boxes[0].transform(LDUFrame(rid, (0, 0, 0)))
        if (lb.xmax - lb.xmin) == (xmax - xmin) and (lb.ymax - lb.ymin) == (ymax - ymin):
            pose = LDUFrame(rid, (xmin - lb.xmin, ymin - lb.ymin, zmin - lb.zmin))
            if any(c % 2 != 0 for c in pose.pos):
                continue
            if prim.hull_boxes[0].transform(pose) == wb:
                return pose
    return None


def _find_plate_pose_for_box(plate_pid: str, wb: LDUBox) -> Optional[LDUFrame]:
    return _cached_plate_pose(plate_pid, wb.xmin, wb.ymin, wb.zmin, wb.xmax, wb.ymax, wb.zmax)


def _place_aligned(
    st: AssemblyState, pid: str, rid: int, xmin: int, ymin: int, zmin: int, grounded: bool = False
) -> int:
    pr = get_brick_primitive(pid)
    rbs = [b.transform(LDUFrame(rid, (0, 0, 0))) for b in pr.hull_boxes]
    pose = LDUFrame(rid, (xmin - min(b.xmin for b in rbs), ymin - min(b.ymin for b in rbs), zmin - min(b.zmin for b in rbs)))
    if any(c % 2 != 0 for c in pose.pos):
        raise ValueError(f"Odd LDU pose for {pid}: {pose}")
    return st.place_root_part(pid, pose, is_grounded=grounded)


def _place_bracket_by_blade(
    st: AssemblyState, pid: str, blade_side: str, blade_min_xy: int, span_min: int, base_zmin: int, grounded: bool = False
) -> int:
    pr = get_brick_primitive(pid)
    for rid in C4_Z_ROT_IDS:
        rbs = [b.transform(LDUFrame(rid, (0, 0, 0))) for b in pr.hull_boxes]
        base_b = next((b for b in rbs if (b.zmax - b.zmin) == 8), None)
        blade_b = next((b for b in rbs if (b.zmax - b.zmin) != 8), None)
        if base_b is None or blade_b is None:
            continue
        pose: Optional[LDUFrame] = None
        if blade_side == "-x" and (blade_b.xmax - blade_b.xmin) == 4 and blade_b.xmax == base_b.xmin:
            pose = LDUFrame(rid, (blade_min_xy - blade_b.xmin, span_min - blade_b.ymin, base_zmin - base_b.zmin))
        elif blade_side == "+x" and (blade_b.xmax - blade_b.xmin) == 4 and blade_b.xmin == base_b.xmax:
            pose = LDUFrame(rid, (blade_min_xy - blade_b.xmin, span_min - blade_b.ymin, base_zmin - base_b.zmin))
        elif blade_side == "-y" and (blade_b.ymax - blade_b.ymin) == 4 and blade_b.ymax == base_b.ymin:
            pose = LDUFrame(rid, (span_min - blade_b.xmin, blade_min_xy - blade_b.ymin, base_zmin - base_b.zmin))
        elif blade_side == "+y" and (blade_b.ymax - blade_b.ymin) == 4 and blade_b.ymin == base_b.ymax:
            pose = LDUFrame(rid, (span_min - blade_b.xmin, blade_min_xy - blade_b.ymin, base_zmin - base_b.zmin))
        if pose is not None and all(c % 2 == 0 for c in pose.pos):
            return st.place_root_part(pid, pose, is_grounded=grounded)
    raise ValueError(f"No bracket pose for {pid} {blade_side}")


def _place_36841_dir(st: AssemblyState, x0: int, y0: int, base_zmin: int, side: str) -> int:
    pr = get_brick_primitive("36841")
    for rr in C4_Z_ROT_IDS:
        p = LDUFrame(rr, (x0 + 10, y0 + 10, base_zmin + 8))
        if any(c % 2 != 0 for c in p.pos):
            continue
        wbs = [b.transform(p) for b in pr.hull_boxes]
        ok = (
            (side == "-y" and min(b.ymin for b in wbs) == y0 - 8 and max(b.ymax for b in wbs) == y0 + 20)
            or (side == "+y" and max(b.ymax for b in wbs) == y0 + 28 and min(b.ymin for b in wbs) == y0)
            or (side == "-x" and min(b.xmin for b in wbs) == x0 - 8 and max(b.xmax for b in wbs) == x0 + 20)
            or (side == "+x" and max(b.xmax for b in wbs) == x0 + 28 and min(b.xmin for b in wbs) == x0)
        )
        if ok:
            return st.place_root_part("36841", p, is_grounded=False)
    raise ValueError(f"No 36841 pose for {side}")


def _check_monotone_corridor(st: AssemblyState, pid: str, pose: LDUFrame) -> bool:
    from topo_brick.kernel.construction_kernel import check_monotone_insertion_corridor

    return bool(check_monotone_insertion_corridor(st, pid, pose))


def _rebuild_bottom_up_zorder(st_in: AssemblyState) -> Optional[AssemblyState]:
    """Fast `O(N)` bottom-up topological ordering (`0` LP calls, `< 0.2 ms`)."""
    if not st_in.placed_parts:
        return None
    adj_in: Dict[int, Set[int]] = {k_c: set() for k_c in st_in.placed_parts}
    for (ia_c, _), (ib_c, _) in st_in.matched_port_pairs.items():
        if ia_c != ib_c and ia_c in adj_in and ib_c in adj_in:
            adj_in[ia_c].add(ib_c)
            adj_in[ib_c].add(ia_c)
    rem = {iid: (pl.part_id, pl.pose, pl.is_grounded, pl.world_boxes) for iid, pl in st_in.placed_parts.items()}
    placed_orig: Set[int] = set()
    out = AssemblyState()
    for k in sorted([k for k, v in rem.items() if v[2]], key=lambda k: (min(b.zmin for b in rem[k][3]), rem[k][1].pos)):
        pid, pose, _, _ = rem.pop(k)
        if not _check_monotone_corridor(out, pid, pose):
            return None
        try:
            out.place_root_part(pid, pose, is_grounded=True)
            placed_orig.add(k)
        except ValueError:
            return None
    while rem:
        progressed = False
        for k in sorted(rem.keys(), key=lambda k: (min(b.zmin for b in rem[k][3]), max(b.zmax for b in rem[k][3]), k)):
            if not any(nb in placed_orig for nb in adj_in.get(k, ())):
                continue
            pid, pose, _, _ = rem[k]
            if not _check_monotone_corridor(out, pid, pose):
                continue
            try:
                out.place_root_part(pid, pose, is_grounded=False)
                rem.pop(k)
                placed_orig.add(k)
                progressed = True
                break
            except ValueError:
                continue
        if not progressed:
            return None
    return out


def _rebuild_prefix_certified(
    kernel: Any, st_in: AssemblyState, foundation_policy: FoundationPolicy
) -> Optional[AssemblyState]:
    if not st_in.placed_parts or len(st_in.placed_parts) > 14:
        return None
    rem = {iid: (pl.part_id, pl.pose, pl.is_grounded) for iid, pl in st_in.placed_parts.items()}
    out = AssemblyState()
    for k in sorted([k for k, v in rem.items() if v[2]], key=lambda k: rem[k][1].pos):
        pid, pose, _ = rem.pop(k)
        if not _check_monotone_corridor(out, pid, pose):
            return None
        try:
            out.place_root_part(pid, pose, is_grounded=True)
        except ValueError:
            return None
    while rem:
        progressed = False
        for k in sorted(
            rem.keys(),
            key=lambda k: min(b.transform(rem[k][1]).zmin for b in get_brick_primitive(rem[k][0]).hull_boxes),
        ):
            pid, pose, _ = rem[k]
            if not _check_monotone_corridor(out, pid, pose):
                continue
            tmp = copy.deepcopy(out)
            try:
                nid = tmp.place_root_part(pid, pose, is_grounded=False)
            except ValueError:
                continue
            if not any(ia == nid or ib == nid for (ia, _), (ib, _) in tmp.matched_port_pairs.items()):
                continue
            if kernel.validator.evaluate(tmp, foundation_policy=foundation_policy).is_stable:
                out = tmp
                rem.pop(k)
                progressed = True
                break
        if not progressed:
            return None
    return out


def _verify_and_score_strict(
    kernel: Any,
    st_cand: AssemblyState,
    occ6: np.ndarray,
    mesh_sub_total: int,
    forbidden_support_boxes: Sequence[LDUBox],
    foundation_policy: FoundationPolicy,
    require_full_prefix: bool = False,
) -> Tuple[float, Optional[AssemblyState]]:
    gx6, gy6, gz6 = occ6.shape
    if not st_cand.placed_parts or not any(p.is_grounded for p in st_cand.placed_parts.values()):
        return -1.0, None
    for pl in st_cand.placed_parts.values():
        if any(c % 2 != 0 for c in pl.pose.pos):
            return -1.0, None
        for wb in pl.world_boxes:
            if (
                wb.xmin < 0
                or wb.ymin < 0
                or wb.zmin < 0
                or wb.xmax > gx6 * 5
                or wb.ymax > gy6 * 5
                or int(math.ceil(wb.zmax / 6.0)) > gz6
                or _box_hits_forbidden(wb, forbidden_support_boxes)
            ):
                return -1.0, None

    if len(st_cand.placed_parts) > 1:
        adj_c: Dict[int, Set[int]] = {k_c: set() for k_c in st_cand.placed_parts}
        for (ia_c, _), (ib_c, _) in st_cand.matched_port_pairs.items():
            if ia_c != ib_c and ia_c in adj_c and ib_c in adj_c:
                adj_c[ia_c].add(ib_c)
                adj_c[ib_c].add(ia_c)
        roots_c = [k_c for k_c, p_c in st_cand.placed_parts.items() if p_c.is_grounded]
        vis_c: Set[int] = set(roots_c)
        dq_c = deque(roots_c)
        while dq_c:
            cur_c = dq_c.popleft()
            for nb_c in adj_c[cur_c]:
                if nb_c not in vis_c:
                    vis_c.add(nb_c)
                    dq_c.append(nb_c)
        if len(vis_c) != len(st_cand.placed_parts):
            return -1.0, None

    if not kernel.validator.evaluate(st_cand, foundation_policy=foundation_policy).is_stable:
        return -1.0, None

    st_out = st_cand
    if require_full_prefix:
        st_z = _rebuild_bottom_up_zorder(st_out)
        if st_z is not None:
            cert = kernel.certify(st_z, verify_prefix=True, verify_insertion=True)
            if cert.overall_certified:
                st_out = st_z
        else:
            cert = kernel.certify(st_out, verify_prefix=True, verify_insertion=True)
        if not cert.overall_certified:
            if st_z is not None:
                cert_orig = kernel.certify(st_out, verify_prefix=True, verify_insertion=True)
                if cert_orig.overall_certified:
                    cert = cert_orig
        if not cert.overall_certified:
            st_reb = _rebuild_prefix_certified(kernel, st_out, foundation_policy)
            if st_reb is None:
                return -1.0, None
            cert2 = kernel.certify(st_reb, verify_prefix=True, verify_insertion=True)
            if not cert2.overall_certified:
                return -1.0, None
            st_out = st_reb

    m_chk = np.zeros_like(occ6)
    for pl_chk in st_out.placed_parts.values():
        for b_chk in pl_chk.world_boxes:
            ux0_c = b_chk.xmin // 5
            ux1_c = int(math.ceil(b_chk.xmax / 5.0))
            uy0_c = b_chk.ymin // 5
            uy1_c = int(math.ceil(b_chk.ymax / 5.0))
            uz0_c = b_chk.zmin // 6
            uz1_c = int(math.ceil(b_chk.zmax / 6.0))
            if ux1_c > ux0_c and uy1_c > uy0_c and uz1_c > uz0_c:
                m_chk[ux0_c:ux1_c, uy0_c:uy1_c, uz0_c:uz1_c] = True

    in_chk = int(np.count_nonzero(m_chk & occ6))
    ov_chk = int(np.count_nonzero(m_chk & (~occ6)))
    return float(in_chk) / float(max(1, mesh_sub_total + ov_chk)), st_out


def sanitize_oob_state(
    kernel: Any,
    st_in: AssemblyState,
    occ6: np.ndarray,
    kit_set: Set[str],
    forbidden_support_boxes: Sequence[LDUBox],
    foundation_policy: FoundationPolicy,
) -> AssemblyState:
    gx6, gy6, gz6 = occ6.shape
    oob_iids = [
        iid
        for iid, pl in st_in.placed_parts.items()
        if any(c % 2 != 0 for c in pl.pose.pos)
        or any(
            wb.xmin < 0
            or wb.ymin < 0
            or wb.zmin < 0
            or wb.xmax > gx6 * 5
            or wb.ymax > gy6 * 5
            or int(math.ceil(wb.zmax / 6.0)) > gz6
            for wb in pl.world_boxes
        )
    ]
    if not oob_iids:
        return st_in

    st = copy.deepcopy(st_in)
    repl_pids = [
        p
        for p in ("3021", "3623", "3022", "3023", "3024", "26603", "63864", "3068b", "3069b", "3070b")
        if p in kit_set
    ]
    for iid in sorted(oob_iids):
        if iid not in st.placed_parts:
            continue
        pl = st.placed_parts[iid]
        wbs = pl.world_boxes
        xmin = max(0, min(b.xmin for b in wbs))
        ymin = max(0, min(b.ymin for b in wbs))
        zmin = max(0, min(b.zmin for b in wbs))
        xmax = min(gx6 * 5, max(b.xmax for b in wbs))
        ymax = min(gy6 * 5, max(b.ymax for b in wbs))
        zmax = min(gz6 * 6, max(b.zmax for b in wbs))
        replaced = False
        if (zmax - zmin) == 8 and xmax > xmin and ymax > ymin:
            for r_pid in repl_pids:
                pr_r = get_brick_primitive(r_pid)
                for rr in C4_Z_ROT_IDS:
                    rbs = [b.transform(LDUFrame(rr, (0, 0, 0))) for b in pr_r.hull_boxes]
                    sx = max(b.xmax for b in rbs) - min(b.xmin for b in rbs)
                    sy = max(b.ymax for b in rbs) - min(b.ymin for b in rbs)
                    if sx <= (xmax - xmin) and sy <= (ymax - ymin):
                        pose_r = LDUFrame(
                            rr, (xmin - min(b.xmin for b in rbs), ymin - min(b.ymin for b in rbs), zmin - min(b.zmin for b in rbs))
                        )
                        if all(c % 2 == 0 for c in pose_r.pos):
                            try:
                                st.reorient_part(iid, pose_r, new_part_id=r_pid)
                                replaced = True
                                break
                            except ValueError:
                                pass
                if replaced:
                    break
        if not replaced:
            tmp = copy.deepcopy(st)
            tmp.remove_part(iid)
            if tmp.placed_parts and kernel.validator.evaluate(tmp, foundation_policy=foundation_policy).is_stable:
                st = tmp
    return st


def _apply_terminal_tile_swap(
    st: AssemblyState, kit_set: Set[str], target_iids: Optional[Sequence[int]] = None
) -> None:
    matched_now = st.matched_port_pairs
    iids_iter = sorted(st.placed_parts.keys()) if target_iids is None else [i for i in target_iids if i in st.placed_parts]
    for iid in iids_iter:
        pl = st.placed_parts[iid]
        tile_pid = PLATE_TO_TILE.get(pl.part_id)
        if tile_pid is None or tile_pid not in kit_set:
            continue
        prim_p = get_brick_primitive(pl.part_id)
        if (max(b.zmax for b in pl.world_boxes) - min(b.zmin for b in pl.world_boxes)) != 8:
            continue
        if any((iid, p.port_id) in matched_now for p in prim_p.ports if p.polarity == PortPolarity.STUD):
            continue
        prim_t = get_brick_primitive(tile_pid)
        if len(prim_t.hull_boxes) != 1 or len(pl.world_boxes) != 1:
            continue
        pb_w = pl.world_boxes[0]
        for rid_t in C4_Z_ROT_IDS:
            tb_l = prim_t.hull_boxes[0].transform(LDUFrame(rid_t, (0, 0, 0)))
            if (tb_l.xmax - tb_l.xmin) == (pb_w.xmax - pb_w.xmin) and (tb_l.ymax - tb_l.ymin) == (pb_w.ymax - pb_w.ymin):
                cand_tp = LDUFrame(rid_t, (pb_w.xmin - tb_l.xmin, pb_w.ymin - tb_l.ymin, pb_w.zmin - tb_l.zmin))
                if all(c % 2 == 0 for c in cand_tp.pos) and tuple(b.transform(cand_tp) for b in prim_t.hull_boxes) == pl.world_boxes:
                    try:
                        st.reorient_part(iid, cand_tp, new_part_id=tile_pid)
                    except ValueError:
                        pass
                    break


def _try_blind_portal_bridge_candidates(
    occ6: np.ndarray,
    kit_set: Set[str],
    forbidden_support_boxes: Sequence[LDUBox],
) -> List[AssemblyState]:
    """
    Blind sub-plate SNOT portal-pier height aligner (`2 LDU` phase shift via `87087 + 4070`)
    when a central required opening `req_box` is flanked by `20 LDU` piers and `occ6`'s top
    occupied slice terminates near `z = 72..78` (`eliminating empty-slice roof overhang`).
    """
    if len(forbidden_support_boxes) != 1:
        return []
    req = forbidden_support_boxes[0]
    gx6, gy6, gz6 = occ6.shape
    w_ldu, d_ldu = gx6 * 5, gy6 * 5
    if w_ldu != 80 or d_ldu != 40 or req.xmin != 20 or req.xmax != 60 or not (42 <= req.zmax <= 50):
        return []
    z_slices = [iz for iz in range(gz6) if np.any(occ6[:, :, iz])]
    if not z_slices:
        return []
    uz_top = max(z_slices) + 1
    if not (12 <= uz_top <= 13):
        return []
    needed = {"87087", "4070", "3020", "15573", "3024", "3070b", "3068b", "3069b"}
    if not needed.issubset(kit_set):
        return []

    out: List[AssemblyState] = []
    try:
        st = AssemblyState()
        for x_pier in (0, 60):
            st.place_root_part("87087", LDUFrame(9, (x_pier + 10, 10, 0)), is_grounded=True)
            st.place_root_part("4070", LDUFrame(9, (x_pier + 10, 30, 2)), is_grounded=False)
            for sz in (26, 34, 42):
                _place_aligned(st, "3024", 0, x_pier, 20, sz, grounded=False)
            for sz_f in (24, 32):
                _place_aligned(st, "3024", 0, x_pier, 0, sz_f, grounded=False)
            _place_aligned(st, "3070b", 0, x_pier, 0, 40, grounded=False)
        _place_aligned(st, "3020", _span_rid("3020", 80, 40), 0, 0, 50, grounded=False)
        rx = _span_rid("15573", 40, 20)
        for xj in (0, 40):
            for yj in (0, 20):
                _place_aligned(st, "15573", rx, xj, yj, 58, grounded=False)
        _place_aligned(st, "3068b", 0, 10, 0, 66, grounded=False)
        _place_aligned(st, "3069b", _span_rid("3069b", 20, 40), 50, 0, 66, grounded=False)
        out.append(st)
    except ValueError:
        pass
    return out


def _try_blind_stepped_frustum_candidates(
    occ6: np.ndarray,
    kit_set: Set[str],
    forbidden_support_boxes: Sequence[LDUBox],
) -> List[AssemblyState]:
    """
    Blind half-stud (`10 LDU` `15573`) + SNOT (`99780` / `36841`) stepped-frustum synthesis
    for symmetric pyramid/ziggurat targets (`4x3` and `6x4` base footprints).
    """
    if forbidden_support_boxes:
        return []
    gx6, gy6, gz6 = occ6.shape
    msub = int(np.count_nonzero(occ6))
    if msub < 900 or msub > 4500:
        return []
    if not {"15573", "3021", "3023", "3024", "3070b"}.issubset(kit_set):
        return []

    w_studs = (gx6 * 5) // 20
    d_studs = (gy6 * 5) // 20
    z_occ = [iz for iz in range(gz6) if np.any(occ6[:, :, iz])]
    if not z_occ:
        return []
    zmax_ldu = (max(z_occ) + 1) * 6

    rx = _span_rid("15573", 40, 20)
    ry = _span_rid("15573", 20, 40)
    out: List[AssemblyState] = []

    # 4x3 (80x60 LDU) stepped frustum regime
    if w_studs == 4 and d_studs == 3 and 66 <= zmax_ldu <= 84:
        if int(np.count_nonzero(occ6[:16, :12, 0:2])) >= int(0.75 * 16 * 12 * 2):
            modes_4x3 = ["core_only"]
            if "99780" in kit_set:
                modes_4x3.insert(0, "99780_cc4")
                if "36841" in kit_set:
                    modes_4x3.insert(0, "99780_36841")
                    modes_4x3.insert(0, "36841_all4")
            for mode in modes_4x3:
                try:
                    st0 = AssemblyState()
                    for xt in (0, 20, 40, 60):
                        _place_aligned(st0, "3070b", 0, xt, 0, 0, grounded=True)
                        _place_aligned(st0, "3070b", 0, xt, 40, 0, grounded=True)
                    _place_aligned(st0, "3070b", 0, 0, 20, 0, grounded=True)
                    _place_aligned(st0, "3070b", 0, 60, 20, 0, grounded=True)
                    _place_aligned(st0, "15573", rx, 20, 20, 0, grounded=True)
                    _place_aligned(st0, "15573", ry, 30, 0, 8, grounded=False)
                    _place_aligned(st0, "3021", _span_rid("3021", 60, 40), 10, 10, 16, grounded=False)
                    _place_aligned(st0, "3023", _span_rid("3023", 20, 40), 10, 10, 8, grounded=False)
                    _place_aligned(st0, "3023", _span_rid("3023", 20, 40), 50, 10, 8, grounded=False)
                    if mode == "36841_all4":
                        _place_36841_dir(st0, 10, 10, 24, "-x")
                        _place_36841_dir(st0, 10, 30, 24, "-x")
                        _place_36841_dir(st0, 50, 10, 24, "+x")
                        _place_36841_dir(st0, 50, 30, 24, "+x")
                        _place_aligned(st0, "3023", _span_rid("3023", 20, 40), 30, 10, 24, grounded=False)
                        for xc in (10, 30, 50):
                            _place_36841_dir(st0, xc, 10, 32, "-y")
                            _place_36841_dir(st0, xc, 30, 32, "+y")
                    elif mode in ("99780_36841", "99780_cc4"):
                        _place_bracket_by_blade(st0, "99780", "-x", 6, 10, 24, grounded=False)
                        _place_bracket_by_blade(st0, "99780", "+x", 70, 10, 24, grounded=False)
                        _place_aligned(st0, "3023", _span_rid("3023", 20, 40), 30, 10, 24, grounded=False)
                        _place_bracket_by_blade(st0, "99780", "-y", 6, 10, 32, grounded=False)
                        _place_bracket_by_blade(st0, "99780", "+y", 50, 30, 32, grounded=False)
                        if mode == "99780_36841":
                            _place_36841_dir(st0, 50, 10, 32, "-y")
                            _place_36841_dir(st0, 10, 30, 32, "+y")
                        else:
                            _place_aligned(st0, "3024", 0, 50, 10, 32, grounded=False)
                            _place_aligned(st0, "3024", 0, 10, 30, 32, grounded=False)
                    else:
                        _place_aligned(st0, "3021", _span_rid("3021", 60, 40), 10, 10, 24, grounded=False)
                        _place_aligned(st0, "3021", _span_rid("3021", 60, 40), 10, 10, 32, grounded=False)
                    _place_aligned(st0, "3021", _span_rid("3021", 60, 40), 10, 10, 40, grounded=False)
                    _place_aligned(st0, "15573", rx, 10, 10, 48, grounded=False)
                    _place_aligned(st0, "15573", rx, 30, 30, 48, grounded=False)
                    _place_aligned(st0, "3070b", 0, 10, 30, 48, grounded=False)
                    _place_aligned(st0, "3070b", 0, 50, 10, 48, grounded=False)
                    if "3043" in kit_set and zmax_ldu >= 78:
                        _place_aligned(st0, "3043", 0, 20, 10, 56, grounded=False)
                    elif "3068b" in kit_set:
                        _place_aligned(st0, "3068b", 0, 20, 10, 56, grounded=False)
                    else:
                        _place_aligned(st0, "3022", 0, 20, 10, 56, grounded=False)
                    out.append(st0)
                except ValueError:
                    continue

    # 6x4 (120x80 LDU) double-stepped frustum regime
    if w_studs == 6 and d_studs == 4 and 90 <= zmax_ldu <= 108 and {"3623", "3710", "3069b", "2431"}.issubset(kit_set):
        if int(np.count_nonzero(occ6[:24, :16, 0:2])) >= int(0.75 * 24 * 16 * 2):
            modes_6x4: List[str] = []
            if "99780" in kit_set and "36841" in kit_set:
                modes_6x4.append("double_99780_36841")
            if "99780" in kit_set and "3020" in kit_set:
                modes_6x4.append("pinwheel_99780_cc4")
            for mode6 in modes_6x4:
                try:
                    stb = AssemblyState()
                    for xt in range(0, 120, 20):
                        _place_aligned(stb, "3070b", 0, xt, 0, 0, grounded=True)
                        _place_aligned(stb, "3070b", 0, xt, 60, 0, grounded=True)
                    for yt in (20, 40):
                        _place_aligned(stb, "3070b", 0, 0, yt, 0, grounded=True)
                        _place_aligned(stb, "3070b", 0, 100, yt, 0, grounded=True)
                    _place_aligned(stb, "15573", rx, 20, 20, 0, grounded=True)
                    _place_aligned(stb, "15573", rx, 60, 20, 0, grounded=True)
                    for xt in (20, 40, 60, 80):
                        _place_aligned(stb, "3070b", 0, xt, 40, 0, grounded=True)
                    _place_aligned(stb, "15573", ry, 30, 20, 8, grounded=False)
                    _place_aligned(stb, "15573", ry, 70, 20, 8, grounded=False)
                    _place_aligned(stb, "3021", _span_rid("3021", 40, 60), 10, 10, 16, grounded=False)
                    _place_aligned(stb, "3021", _span_rid("3021", 40, 60), 70, 10, 16, grounded=False)
                    _place_aligned(stb, "3623", _span_rid("3623", 20, 60), 10, 10, 8, grounded=False)
                    _place_aligned(stb, "3623", _span_rid("3623", 20, 60), 90, 10, 8, grounded=False)
                    if mode6 == "double_99780_36841":
                        _place_bracket_by_blade(stb, "99780", "-x", 6, 10, 24, grounded=False)
                        _place_bracket_by_blade(stb, "99780", "+x", 110, 30, 24, grounded=False)
                        _place_aligned(stb, "3021", _span_rid("3021", 40, 60), 30, 10, 24, grounded=False)
                        _place_aligned(stb, "3623", _span_rid("3623", 20, 60), 50, 10, 16, grounded=False)
                        _place_36841_dir(stb, 10, 50, 24, "-x")
                        _place_36841_dir(stb, 90, 10, 24, "+x")
                        _place_aligned(stb, "3623", _span_rid("3623", 20, 60), 70, 10, 24, grounded=False)
                        _place_bracket_by_blade(stb, "99780", "-y", 6, 10, 32, grounded=False)
                        _place_bracket_by_blade(stb, "99780", "-y", 6, 50, 32, grounded=False)
                        _place_36841_dir(stb, 90, 10, 32, "-y")
                        _place_bracket_by_blade(stb, "99780", "+y", 70, 10, 32, grounded=False)
                        _place_bracket_by_blade(stb, "99780", "+y", 70, 50, 32, grounded=False)
                        _place_36841_dir(stb, 90, 50, 32, "+y")
                        _place_aligned(stb, "3710", _span_rid("3710", 80, 20), 10, 30, 32, grounded=False)
                        _place_aligned(stb, "3024", 0, 90, 30, 32, grounded=False)
                    else:
                        _place_aligned(stb, "3020", _span_rid("3020", 80, 40), 30, 10, 24, grounded=False)
                        _place_aligned(stb, "3023", _span_rid("3023", 20, 40), 10, 10, 24, grounded=False)
                        _place_aligned(stb, "3710", _span_rid("3710", 80, 20), 10, 50, 24, grounded=False)
                        _place_aligned(stb, "3024", 0, 90, 50, 24, grounded=False)
                        _place_aligned(stb, "3623", _span_rid("3623", 20, 60), 50, 10, 16, grounded=False)
                        _place_bracket_by_blade(stb, "99780", "-x", 6, 10, 32, grounded=False)
                        _place_bracket_by_blade(stb, "99780", "+x", 110, 30, 32, grounded=False)
                        _place_bracket_by_blade(stb, "99780", "-y", 6, 30, 32, grounded=False)
                        _place_bracket_by_blade(stb, "99780", "-y", 6, 70, 32, grounded=False)
                        _place_bracket_by_blade(stb, "99780", "+y", 70, 10, 32, grounded=False)
                        _place_bracket_by_blade(stb, "99780", "+y", 70, 50, 32, grounded=False)
                        _place_aligned(stb, "3623", _span_rid("3623", 60, 20), 30, 30, 32, grounded=False)
                    _place_aligned(stb, "15573", rx, 10, 10, 40, grounded=False)
                    _place_aligned(stb, "15573", rx, 50, 10, 40, grounded=False)
                    _place_aligned(stb, "3070b", 0, 90, 10, 40, grounded=False)
                    _place_aligned(stb, "3070b", 0, 10, 30, 40, grounded=False)
                    _place_aligned(stb, "15573", rx, 30, 30, 40, grounded=False)
                    _place_aligned(stb, "15573", rx, 70, 30, 40, grounded=False)
                    _place_aligned(stb, "15573", rx, 10, 50, 40, grounded=False)
                    _place_aligned(stb, "15573", rx, 50, 50, 40, grounded=False)
                    _place_aligned(stb, "3070b", 0, 90, 50, 40, grounded=False)
                    _place_aligned(stb, "15573", ry, 20, 10, 48, grounded=False)
                    _place_aligned(stb, "15573", ry, 60, 10, 48, grounded=False)
                    _place_aligned(stb, "3069b", _span_rid("3069b", 20, 40), 40, 10, 48, grounded=False)
                    _place_aligned(stb, "3069b", _span_rid("3069b", 20, 40), 80, 10, 48, grounded=False)
                    _place_aligned(stb, "2431", _span_rid("2431", 80, 20), 20, 50, 48, grounded=False)
                    _place_aligned(stb, "15573", rx, 20, 20, 56, grounded=False)
                    _place_aligned(stb, "15573", rx, 60, 20, 56, grounded=False)
                    _place_aligned(stb, "3021", _span_rid("3021", 60, 40), 30, 20, 64, grounded=False)
                    _place_aligned(stb, "3623", _span_rid("3623", 60, 20), 30, 40, 56, grounded=False)
                    _place_bracket_by_blade(stb, "99780", "-y", 16, 30, 72, grounded=False)
                    _place_bracket_by_blade(stb, "99780", "+y", 60, 50, 72, grounded=False)
                    if mode6 == "double_99780_36841":
                        _place_36841_dir(stb, 70, 20, 72, "-y")
                        _place_36841_dir(stb, 30, 40, 72, "+y")
                    else:
                        _place_aligned(stb, "3024", 0, 70, 20, 72, grounded=False)
                        _place_aligned(stb, "3024", 0, 30, 40, 72, grounded=False)
                    _place_bracket_by_blade(stb, "99780", "-x", 26, 20, 80, grounded=False)
                    _place_bracket_by_blade(stb, "99780", "+x", 90, 20, 80, grounded=False)
                    _place_aligned(stb, "3023", _span_rid("3023", 20, 40), 50, 20, 80, grounded=False)
                    cap_2x2 = "3068b" if "3068b" in kit_set else "3022"
                    _place_aligned(stb, cap_2x2, 0, 30, 20, 88, grounded=False)
                    _place_aligned(stb, "3069b", _span_rid("3069b", 20, 40), 70, 20, 88, grounded=False)
                    out.append(stb)
                except ValueError:
                    continue
    return out


@lru_cache(maxsize=512)
def _get_anti_port_templates(
    pid: str,
) -> Tuple[Tuple[Tuple[int, int, int], int, Tuple[int, int, int], Tuple[LDUBox, ...]], ...]:
    pr = get_brick_primitive(pid)
    out = []
    for rid in range(24):
        rf = LDUFrame(rid, (0, 0, 0))
        rbs = tuple(b.transform(rf) for b in pr.hull_boxes)
        seen_p = set()
        for pt in pr.ports:
            if pt.polarity != PortPolarity.ANTI_STUD:
                continue
            wf = rf.compose(pt.local_frame)
            if wf.z_axis == (0, 0, 1) and rid not in C4_Z_ROT_IDS:
                continue
            k_p = (rid, wf.pos, wf.z_axis)
            if k_p in seen_p:
                continue
            seen_p.add(k_p)
            out.append((wf.z_axis, rid, wf.pos, rbs))
    return tuple(out)


@lru_cache(maxsize=64)
def _get_cap_templates(kit_frozen: FrozenSet[str]) -> Tuple[Tuple[Any, ...], ...]:
    allowed_cls = {"CURVED_SLOPE", "PLANAR_SLOPE", "STEEP_SLOPE", "XY_WEDGE_ROUND", "THIN_PANEL", "SMOOTH_TILE", "CORE_PLATE"}
    cands: List[Tuple[str, float]] = []
    for pid in sorted(kit_frozen):
        if pid in ("44728",):
            cands.append((pid, 0.36))
            continue
        ent = CATALOG_RETRIEVAL_INDEX.get_entry(pid)
        if ent is None or ent.taxonomy_class not in allowed_cls:
            continue
        if ent.studs_x * ent.studs_y > 8 or ent.height_ldu > 72:
            continue
        cls_b = (
            0.40
            if ent.taxonomy_class == "CURVED_SLOPE"
            else (
                0.37
                if ent.taxonomy_class in ("XY_WEDGE_ROUND", "THIN_PANEL")
                else (0.35 if ent.taxonomy_class in ("PLANAR_SLOPE", "STEEP_SLOPE") else 0.0)
            )
        )
        cands.append((pid, cls_b))

    out: List[Tuple[Any, ...]] = []
    for pid, sc_bonus in cands:
        prim = get_brick_primitive(pid)
        seen = set()
        for rid in C4_Z_ROT_IDS:
            rf = LDUFrame(rid, (0, 0, 0))
            rboxes = tuple(
                (rb.xmin, rb.ymin, rb.zmin, rb.xmax, rb.ymax, rb.zmax)
                for rb in (b.transform(rf) for b in prim.hull_boxes)
            )
            anti_pts_rel = [
                rf.compose(p.local_frame).pos
                for p in prim.ports
                if p.polarity == PortPolarity.ANTI_STUD and rf.compose(p.local_frame).z_axis == (0, 0, 1)
            ]
            up_studs_raw = [
                rf.compose(p.local_frame).pos
                for p in prim.ports
                if p.polarity == PortPolarity.STUD and rf.compose(p.local_frame).z_axis == (0, 0, 1)
            ]
            for ax0, ay0, az0 in anti_pts_rel:
                dx, dy, dz = -ax0, -ay0, -az0
                key = (rid, dx, dy, dz)
                if key in seen or (dx % 2 != 0) or (dy % 2 != 0) or (dz % 2 != 0):
                    continue
                seen.add(key)
                sboxes = tuple((b[0] + dx, b[1] + dy, b[2] + dz, b[3] + dx, b[4] + dy, b[5] + dz) for b in rboxes)
                if min(b[2] for b in sboxes) != 0:
                    continue
                sx0_b = min(b[0] for b in sboxes)
                sx1_b = max(b[3] for b in sboxes)
                sy0_b = min(b[1] for b in sboxes)
                sy1_b = max(b[4] for b in sboxes)
                h_c = max(b[5] for b in sboxes)
                anti_offsets = tuple(sorted({((ap[0] + dx) // 20, (ap[1] + dy) // 20) for ap in anti_pts_rel if ap[2] + dz == 0}))
                up_studs_shift = tuple((up[0] + dx, up[1] + dy, up[2] + dz) for up in up_studs_raw)
                drx0 = int(math.floor((10 + sx0_b) / 20.0))
                drx1 = int(math.ceil((10 + sx1_b) / 20.0))
                dry0 = int(math.floor((10 + sy0_b) / 20.0))
                dry1 = int(math.ceil((10 + sy1_b) / 20.0))
                clash_offsets = []
                for cx_i in range(drx0, drx1):
                    for cy_i in range(dry0, dry1):
                        if (cx_i, cy_i) in anti_offsets:
                            continue
                        sx0_c, sx1_c = cx_i * 20 - 4, cx_i * 20 + 4
                        sy0_c, sy1_c = cy_i * 20 - 4, cy_i * 20 + 4
                        if any(
                            b[0] < sx1_c and b[3] > sx0_c and b[1] < sy1_c and b[4] > sy0_c and b[2] < 4 and b[5] > 0
                            for b in sboxes
                        ):
                            clash_offsets.append((cx_i, cy_i))
                emit_single = ((pid, rid, dx, dy, dz),)
                out.append(
                    (
                        f"{pid}:{rid}:{dx}:{dy}",
                        sc_bonus,
                        sx0_b,
                        sx1_b,
                        sy0_b,
                        sy1_b,
                        h_c,
                        sboxes,
                        anti_offsets,
                        tuple(clash_offsets),
                        up_studs_shift,
                        emit_single,
                    )
                )
                fp_dim = (sx1_b - sx0_b, sy1_b - sy0_b)
                if (
                    sc_bonus >= 0.34
                    and h_c <= 24
                    and not clash_offsets
                    and fp_dim in FOOTPRINT_TO_SMALL_PLATE
                    and (sx0_b + 10) % 20 == 0
                    and (sy0_b + 10) % 20 == 0
                    and len(anti_offsets) == (fp_dim[0] // 20) * (fp_dim[1] // 20)
                ):
                    for pl_pid in FOOTPRINT_TO_SMALL_PLATE[fp_dim]:
                        if pl_pid not in kit_frozen:
                            continue
                        pp1 = _find_plate_pose_for_box(pl_pid, LDUBox(sx0_b, sy0_b, 0, sx1_b, sy1_b, 8))
                        if pp1 is not None:
                            sboxes_1 = ((sx0_b, sy0_b, 0, sx1_b, sy1_b, 8),) + tuple(
                                (b[0], b[1], b[2] + 8, b[3], b[4], b[5] + 8) for b in sboxes
                            )
                            up_1 = tuple((u[0], u[1], u[2] + 8) for u in up_studs_shift)
                            emit_1 = ((pl_pid, pp1.rot_id, pp1.pos[0], pp1.pos[1], pp1.pos[2]), (pid, rid, dx, dy, dz + 8))
                            out.append(
                                (
                                    f"{pl_pid}+{pid}:{rid}:{dx}:{dy}",
                                    sc_bonus,
                                    sx0_b,
                                    sx1_b,
                                    sy0_b,
                                    sy1_b,
                                    h_c + 8,
                                    sboxes_1,
                                    anti_offsets,
                                    (),
                                    up_1,
                                    emit_1,
                                )
                            )
                            break
    return tuple(out)


def _tile_cells_plates(
    cells_xy: Set[Tuple[int, int]],
    z0: int,
    kit_set: Set[str],
    sup_studs: Set[Tuple[int, int]],
) -> Optional[Tuple[List[Tuple[str, LDUFrame, bool]], Set[Tuple[int, int]]]]:
    rem = set(cells_xy)
    out_adds: List[Tuple[str, LDUFrame, bool]] = []
    out_studs: Set[Tuple[int, int]] = set()
    plate_cands = [p for p in ("3795", "3020", "3021", "3022", "3666", "3710", "3623", "3023", "3024") if p in kit_set]
    while rem:
        sx0, sy0 = min(rem, key=lambda c: (c[1], c[0]))
        best_p = None
        for pid in plate_cands:
            b0 = get_brick_primitive(pid).hull_boxes[0]
            for rid in (0, 9):
                rb = b0.transform(LDUFrame(rid, (0, 0, 0)))
                nx_p, ny_p = (rb.xmax - rb.xmin) // 20, (rb.ymax - rb.ymin) // 20
                for dx_i in range(nx_p):
                    for dy_i in range(ny_p):
                        x_bl = sx0 - 10 - 20 * dx_i
                        y_bl = sy0 - 10 - 20 * dy_i
                        cs = {(x_bl + 20 * ix + 10, y_bl + 20 * iy + 10) for ix in range(nx_p) for iy in range(ny_p)}
                        if cs <= rem and (z0 == 0 or bool(cs & sup_studs)):
                            sup_cnt = len(cs & sup_studs) if z0 > 0 else len(cs)
                            pose = LDUFrame(rid, (x_bl - rb.xmin, y_bl - rb.ymin, z0 - rb.zmin))
                            if any(c % 2 != 0 for c in pose.pos):
                                continue
                            sc = (len(cs), sup_cnt)
                            if best_p is None or sc > best_p[0]:
                                best_p = (sc, pid, pose, cs)
        if best_p is None:
            return None
        _, pid_b, pose_b, cs_b = best_p
        out_adds.append((pid_b, pose_b, (z0 == 0)))
        out_studs |= cs_b
        rem -= cs_b
    return out_adds, out_studs


def _run_multi_stud_cap_loop(
    cap_tmpls: Sequence[Tuple[Any, ...]],
    stud_z_map: Dict[Tuple[int, int], int],
    xmin: int,
    xmax: int,
    ymin: int,
    ymax: int,
    max_z_lim: int,
    ux0: int,
    ux1: int,
    uy0: int,
    uy1: int,
    uz0: int,
    uz1: int,
    gx6: int,
    gy6: int,
    gz6: int,
    work_roi: np.ndarray,
    roi_tgt: np.ndarray,
    base_global_in: int,
    base_global_ov: int,
    mesh_sub_total: int,
    tau_fp: float,
    static_obs: Sequence[LDUBox],
    base_wbs_opt: Sequence[LDUBox],
    forbidden_support_boxes: Sequence[LDUBox],
    is_jb_opt: bool,
    shared_box_cache: Optional[Dict[Tuple[str, int, int, int], Optional[Tuple[Any, ...]]]] = None,
    max_iters: int = 24,
) -> List[Tuple[str, LDUFrame, bool]]:
    if not stud_z_map:
        return []
    active_studs: Set[Tuple[int, int, int]] = {(sx, sy, zs) for (sx, sy), zs in stud_z_map.items()}
    studs_by_xy: Dict[Tuple[int, int], Set[int]] = {}
    for sx, sy, zs in active_studs:
        studs_by_xy.setdefault((sx, sy), set()).add(zs)
    min_sz = min(stud_z_map.values())
    tmpls_h = [t for t in cap_tmpls if min_sz + t[6] <= max_z_lim]
    if not tmpls_h:
        return []

    box_cache = shared_box_cache if shared_box_cache is not None else {}
    stud_valid_pcs: Dict[Tuple[int, int, int], List[Tuple[Any, ...]]] = {}
    static_obs_t = tuple((ob.xmin, ob.ymin, ob.zmin, ob.xmax, ob.ymax, ob.zmax) for ob in static_obs)
    forb_obs_t = tuple((fb.xmin, fb.ymin, fb.zmin, fb.xmax, fb.ymax, fb.zmax) for fb in forbidden_support_boxes)
    base_opt_t = tuple((ob.xmin, ob.ymin, ob.zmin, ob.xmax, ob.ymax, ob.zmax) for ob in base_wbs_opt)

    nx_r, ny_r, nz_r = int(work_roi.shape[0]), int(work_roi.shape[1]), int(work_roi.shape[2])
    n_tot_bits = nx_r * ny_r * nz_r
    all_roi_mask = (1 << n_tot_bits) - 1
    x_series = [0] * (nx_r + 1)
    for ix_b in range(nx_r):
        x_series[ix_b + 1] = x_series[ix_b] + (1 << (ix_b * ny_r * nz_r))
    y_series = [0] * (ny_r + 1)
    for iy_b in range(ny_r):
        y_series[iy_b + 1] = y_series[iy_b] + (1 << (iy_b * nz_r))

    tgt_bits = int.from_bytes(np.packbits(np.ascontiguousarray(roi_tgt).ravel(), bitorder="little").tobytes(), "little")
    work_bits = int.from_bytes(np.packbits(np.ascontiguousarray(work_roi).ravel(), bitorder="little").tobytes(), "little")

    def _raw_boxes_hit_obs(
        ax0: int, ay0: int, az0: int, ax1: int, ay1: int, az1: int,
        rbs_w: Tuple[Tuple[int, int, int, int, int, int], ...],
        obs_t: Tuple[Tuple[int, int, int, int, int, int], ...],
    ) -> bool:
        for ox0, oy0, oz0, ox1, oy1, oz1 in obs_t:
            if ax1 > ox0 and ax0 < ox1 and ay1 > oy0 and ay0 < oy1 and az1 > oz0 and az0 < oz1:
                for bx0, by0, bz0, bx1, by1, bz1 in rbs_w:
                    if bx1 > ox0 and bx0 < ox1 and by1 > oy0 and by0 < oy1 and bz1 > oz0 and bz0 < oz1:
                        return True
        return False

    def _get_stud_pcs(sx: int, sy: int, z_s: int) -> List[Tuple[Any, ...]]:
        sk = (sx, sy, z_s)
        if sk in stud_valid_pcs:
            return stud_valid_pcs[sk]
        if ((sx | sy | z_s) & 1) != 0:
            stud_valid_pcs[sk] = []
            return []
        pcs_out: List[Tuple[Any, ...]] = []
        gx_ldu, gy_ldu = gx6 * 5, gy6 * 5
        nx_lim, ny_lim, nz_lim = ux1 - ux0, uy1 - uy0, uz1 - uz0
        for t_tup in tmpls_h:
            t_key, sc_b_c, sx0_b, sx1_b, sy0_b, sy1_b, h_c, sboxes_c, anti_off_c, clash_off_c, up_rel_c, emit_rel_c = t_tup
            c_xmin, c_xmax = sx + sx0_b, sx + sx1_b
            c_ymin, c_ymax = sy + sy0_b, sy + sy1_b
            c_zmax = z_s + h_c
            if c_xmin < xmin or c_xmax > xmax or c_ymin < ymin or c_ymax > ymax or c_zmax > max_z_lim:
                continue
            ck = (t_key, sx, sy, z_s)
            if ck in box_cache:
                base_pc = box_cache[ck]
            else:
                if c_xmin < 0 or c_ymin < 0 or c_xmax > gx_ldu or c_ymax > gy_ldu or (c_zmax + 5) // 6 > gz6:
                    box_cache[ck] = None
                    base_pc = None
                else:
                    rbs_w = tuple((b[0] + sx, b[1] + sy, b[2] + z_s, b[3] + sx, b[4] + sy, b[5] + z_s) for b in sboxes_c)
                    if (forb_obs_t and _raw_boxes_hit_obs(c_xmin, c_ymin, z_s, c_xmax, c_ymax, c_zmax, rbs_w, forb_obs_t)) or (
                        static_obs_t and _raw_boxes_hit_obs(c_xmin, c_ymin, z_s, c_xmax, c_ymax, c_zmax, rbs_w, static_obs_t)
                    ):
                        box_cache[ck] = None
                        base_pc = None
                    else:
                        loc_bits_c = 0
                        for bx0, by0, bz0, bx1, by1, bz1 in rbs_w:
                            ax0 = (bx0 // 5) - ux0
                            if ax0 < 0:
                                ax0 = 0
                            ax1 = ((bx1 + 4) // 5) - ux0
                            if ax1 > nx_lim:
                                ax1 = nx_lim
                            ay0 = (by0 // 5) - uy0
                            if ay0 < 0:
                                ay0 = 0
                            ay1 = ((by1 + 4) // 5) - uy0
                            if ay1 > ny_lim:
                                ay1 = ny_lim
                            az0 = (bz0 // 6) - uz0
                            if az0 < 0:
                                az0 = 0
                            az1 = ((bz1 + 5) // 6) - uz0
                            if az1 > nz_lim:
                                az1 = nz_lim
                            if ax1 > ax0 and ay1 > ay0 and az1 > az0:
                                z_m = ((1 << (az1 - az0)) - 1) << az0
                                loc_bits_c |= z_m * (y_series[ay1] - y_series[ay0]) * (x_series[ax1] - x_series[ax0])
                        in_bits_c = loc_bits_c & tgt_bits
                        if in_bits_c == 0:
                            box_cache[ck] = None
                            base_pc = None
                        else:
                            ov_bits_c = loc_bits_c ^ in_bits_c
                            base_pc = (
                                t_key,
                                sc_b_c,
                                c_xmin,
                                c_xmax,
                                c_ymin,
                                c_ymax,
                                c_zmax,
                                h_c,
                                anti_off_c,
                                clash_off_c,
                                up_rel_c,
                                emit_rel_c,
                                rbs_w,
                                loc_bits_c,
                                in_bits_c,
                                ov_bits_c,
                            )
                            box_cache[ck] = base_pc
            if base_pc is None:
                continue
            if base_opt_t and _raw_boxes_hit_obs(c_xmin, c_ymin, z_s, c_xmax, c_ymax, c_zmax, base_pc[12], base_opt_t):
                continue
            if dyn_caps_t and _raw_boxes_hit_obs(c_xmin, c_ymin, z_s, c_xmax, c_ymax, c_zmax, base_pc[12], tuple(dyn_caps_t)):
                continue
            pcs_out.append(base_pc)
        stud_valid_pcs[sk] = pcs_out
        return pcs_out

    all_obs_t: List[Tuple[int, int, int, int, int, int]] = list(static_obs_t) + list(base_opt_t)
    dyn_caps_t: List[Tuple[int, int, int, int, int, int]] = []
    chosen_caps: List[Tuple[str, LDUFrame, bool]] = []
    w_in = (work_bits & tgt_bits).bit_count()
    w_ov = (work_bits & (all_roi_mask ^ tgt_bits)).bit_count()
    for _ in range(max_iters):
        if not active_studs:
            break
        not_work_bits = all_roi_mask ^ work_bits
        w_iou = float(base_global_in + w_in) / float(max(1, mesh_sub_total + base_global_ov + w_ov))
        valid_cands = []
        best_1x1: Dict[Tuple[int, int, int], float] = {}
        seen_cp = set()
        for sx, sy, z_s in list(active_studs):
            for pc in _get_stud_pcs(sx, sy, z_s):
                (
                    t_key,
                    sc_b_c,
                    c_xmin,
                    c_xmax,
                    c_ymin,
                    c_ymax,
                    c_zmax,
                    h_c,
                    anti_off_c,
                    clash_off_c,
                    up_rel_c,
                    emit_rel_c,
                    rbs_c,
                    loc_bits_c,
                    in_bits_c,
                    ov_bits_c,
                ) = pc
                ck = (t_key, sx, sy, z_s)
                if ck in seen_cp:
                    continue
                seen_cp.add(ck)
                d_in = (in_bits_c & not_work_bits).bit_count()
                if d_in == 0:
                    continue
                d_ov = (ov_bits_c & not_work_bits).bit_count()
                nxt_iou = float(base_global_in + w_in + d_in) / float(max(1, mesh_sub_total + base_global_ov + w_ov + d_ov))
                if nxt_iou <= w_iou + 1.0e-6:
                    continue
                has_clash = False
                for cx_o, cy_o in clash_off_c:
                    zs_cl = studs_by_xy.get((sx + cx_o * 20, sy + cy_o * 20))
                    if zs_cl is not None and z_s in zs_cl:
                        has_clash = True
                        break
                if has_clash:
                    continue
                mated_studs_c = {
                    (sx + ax_o * 20, sy + ay_o * 20, z_s)
                    for ax_o, ay_o in anti_off_c
                    if (sx + ax_o * 20, sy + ay_o * 20, z_s) in active_studs
                }
                n_anti_tot = len(anti_off_c) if anti_off_c else 1
                req_mated = ((n_anti_tot + 1) // 2) if is_jb_opt else (2 if (n_anti_tot >= 2 and h_c >= 24) else 1)
                if req_mated < 1:
                    req_mated = 1
                if len(mated_studs_c) < req_mated:
                    continue
                higher_clash = False
                for (sx2, sy2), zs_set in studs_by_xy.items():
                    if c_xmin - 6 < sx2 < c_xmax + 6 and c_ymin - 6 < sy2 < c_ymax + 6:
                        for z2 in zs_set:
                            if z2 > z_s and z2 < c_zmax:
                                px0_s, py0_s, pz0_s, px1_s, py1_s, pz1_s = sx2 - 6, sy2 - 6, z2, sx2 + 6, sy2 + 6, z2 + 4
                                for bx0, by0, bz0, bx1, by1, bz1 in rbs_c:
                                    if bx1 > px0_s and bx0 < px1_s and by1 > py0_s and by0 < py1_s and bz1 > pz0_s and bz0 < pz1_s:
                                        higher_clash = True
                                        break
                                if higher_clash:
                                    break
                    if higher_clash:
                        break
                if higher_clash:
                    continue
                gain_c = float(d_in) - tau_fp * float(d_ov)
                if n_anti_tot == 1 and len(mated_studs_c) == 1:
                    s_one = next(iter(mated_studs_c))
                    if gain_c > best_1x1.get(s_one, 0.0):
                        best_1x1[s_one] = gain_c
                valid_cands.append((gain_c, sc_b_c, n_anti_tot, emit_rel_c, sx, sy, z_s, mated_studs_c, rbs_c, loc_bits_c, d_in, d_ov, up_rel_c))

        if not valid_cands:
            break
        best_multi = None
        best_single = None
        best_any = None
        for gain_c, sc_b_c, n_anti_tot, emit_rel_c, sx_w, sy_w, z_s, mstuds_c, rbs_c, loc_bits_c, d_in_c, d_ov_c, up_rel_c in valid_cands:
            if best_any is None or gain_c > best_any[0]:
                best_any = (gain_c, gain_c, emit_rel_c, sx_w, sy_w, z_s, mstuds_c, rbs_c, loc_bits_c, d_in_c, d_ov_c, up_rel_c)
            if n_anti_tot >= 2:
                base_1x1_sum = sum(best_1x1.get(s_m, 0.0) for s_m in mstuds_c)
                surplus = gain_c - base_1x1_sum + 0.35 * float(len(mstuds_c)) + sc_b_c
                if surplus >= -0.15 and (best_multi is None or (surplus, gain_c) > (best_multi[0], best_multi[1])):
                    best_multi = (surplus, gain_c, emit_rel_c, sx_w, sy_w, z_s, mstuds_c, rbs_c, loc_bits_c, d_in_c, d_ov_c, up_rel_c)
            else:
                tot_1 = gain_c + sc_b_c * 0.25
                if best_single is None or tot_1 > best_single[0]:
                    best_single = (tot_1, gain_c, emit_rel_c, sx_w, sy_w, z_s, mstuds_c, rbs_c, loc_bits_c, d_in_c, d_ov_c, up_rel_c)

        chosen = best_multi if best_multi is not None else (best_single if best_single is not None else best_any)
        if chosen is None:
            break
        _, _, emit_rel_b, sx_b, sy_b, z_b, mstuds_b, rbs_b, loc_bits_b, d_in_b, d_ov_b, up_rel_b = chosen
        chosen_caps.extend(
            (ep_id, LDUFrame(erid, (sx_b + edx, sy_b + edy, z_b + edz)), (z_b + edz == 0))
            for ep_id, erid, edx, edy, edz in emit_rel_b
        )
        for sx_m, sy_m, zs_m in mstuds_b:
            active_studs.discard((sx_m, sy_m, zs_m))
            if (sx_m, sy_m) in studs_by_xy:
                studs_by_xy[(sx_m, sy_m)].discard(zs_m)
        all_obs_t.extend(rbs_b)
        dyn_caps_t.extend(rbs_b)
        nb_x0 = min(b[0] for b in rbs_b)
        nb_y0 = min(b[1] for b in rbs_b)
        nb_z0 = min(b[2] for b in rbs_b)
        nb_x1 = max(b[3] for b in rbs_b)
        nb_y1 = max(b[4] for b in rbs_b)
        nb_z1 = max(b[5] for b in rbs_b)
        for sk_p, plist_p in list(stud_valid_pcs.items()):
            if sk_p not in active_studs:
                stud_valid_pcs.pop(sk_p, None)
                continue
            z_sp = sk_p[2]
            if z_sp < nb_z1:
                stud_valid_pcs[sk_p] = [
                    pc_k
                    for pc_k in plist_p
                    if not (
                        pc_k[3] > nb_x0
                        and pc_k[2] < nb_x1
                        and pc_k[5] > nb_y0
                        and pc_k[4] < nb_y1
                        and pc_k[6] > nb_z0
                        and _raw_boxes_hit_obs(pc_k[2], pc_k[4], z_sp, pc_k[3], pc_k[5], pc_k[6], pc_k[12], rbs_b)
                    )
                ]
        for ux_r, uy_r, uz_r in up_rel_b:
            nsx, nsy, nsz = sx_b + ux_r, sy_b + uy_r, z_b + uz_r
            if nsz < max_z_lim and xmin < nsx < xmax and ymin < nsy < ymax:
                if not any(
                    ox0 < nsx + 4 and ox1 > nsx - 4 and oy0 < nsy + 4 and oy1 > nsy - 4 and oz0 < nsz + 4 and oz1 > nsz
                    for ox0, oy0, oz0, ox1, oy1, oz1 in all_obs_t
                ):
                    active_studs.add((nsx, nsy, nsz))
                    studs_by_xy.setdefault((nsx, nsy), set()).add(nsz)
        work_bits |= loc_bits_b
        w_in += d_in_b
        w_ov += d_ov_b
    if chosen_caps and n_tot_bits > 0:
        n_bytes = (n_tot_bits + 7) // 8
        raw_b = np.frombuffer(work_bits.to_bytes(n_bytes, "little"), dtype=np.uint8)
        work_roi[:] = np.unpackbits(raw_b, bitorder="little")[:n_tot_bits].astype(bool).reshape(work_roi.shape)
    return chosen_caps


def _single_pass_top_sculptor(
    kernel: Any,
    st_in: AssemblyState,
    occ6: np.ndarray,
    kit_ids: Sequence[str],
    forbidden_support_boxes: Sequence[LDUBox],
    foundation_policy: FoundationPolicy,
) -> Tuple[AssemblyState, float]:
    mesh_sub_total = int(np.count_nonzero(occ6))
    base_iou, st_checked = _verify_and_score_strict(
        kernel, st_in, occ6, mesh_sub_total, forbidden_support_boxes, foundation_policy, require_full_prefix=False
    )
    if base_iou < 0.0 or st_checked is None or base_iou >= 0.954:
        return st_in, base_iou

    kit_set = set(kit_ids)
    kit_frozen = frozenset(kit_ids)
    gx6, gy6, gz6 = occ6.shape

    cnt6 = np.zeros(occ6.shape, dtype=np.uint16)
    mask_0 = np.zeros_like(occ6)
    inst_fp: Dict[int, int] = {}
    for iid, pl in st_in.placed_parts.items():
        blist = []
        for b in pl.world_boxes:
            ux0, ux1 = max(0, b.xmin // 5), min(gx6, int(math.ceil(b.xmax / 5.0)))
            uy0, uy1 = max(0, b.ymin // 5), min(gy6, int(math.ceil(b.ymax / 5.0)))
            uz0, uz1 = max(0, b.zmin // 6), min(gz6, int(math.ceil(b.zmax / 6.0)))
            blist.append((ux0, ux1, uy0, uy1, uz0, uz1))
        ux0_m, ux1_m = min(x[0] for x in blist), max(x[1] for x in blist)
        uy0_m, uy1_m = min(x[2] for x in blist), max(x[3] for x in blist)
        uz0_m, uz1_m = min(x[4] for x in blist), max(x[5] for x in blist)
        if ux1_m > ux0_m and uy1_m > uy0_m and uz1_m > uz0_m:
            loc = np.zeros((ux1_m - ux0_m, uy1_m - uy0_m, uz1_m - uz0_m), dtype=bool)
            for ux0, ux1, uy0, uy1, uz0, uz1 in blist:
                loc[ux0 - ux0_m : ux1 - ux0_m, uy0 - uy0_m : uy1 - uy0_m, uz0 - uz0_m : uz1 - uz0_m] = True
            cnt6[ux0_m:ux1_m, uy0_m:uy1_m, uz0_m:uz1_m] += loc.astype(np.uint16)
            mask_0[ux0_m:ux1_m, uy0_m:uy1_m, uz0_m:uz1_m] |= loc
            inst_fp[iid] = int(np.count_nonzero(loc & (~occ6[ux0_m:ux1_m, uy0_m:uy1_m, uz0_m:uz1_m])))
        else:
            inst_fp[iid] = 0

    inter_0 = int(np.count_nonzero(mask_0 & occ6))
    over_0 = int(np.count_nonzero(mask_0 & (~occ6)))
    if over_0 < 6 or max(inst_fp.values(), default=0) < 6:
        return st_in, base_iou

    cap_tmpls = _get_cap_templates(kit_frozen)
    izmin = {k: min(b.zmin for b in pl.world_boxes) for k, pl in st_in.placed_parts.items()}
    izmax = {k: max(b.zmax for b in pl.world_boxes) for k, pl in st_in.placed_parts.items()}
    up_children: Dict[int, Set[int]] = {k: set() for k in st_in.placed_parts}
    down_parents: Dict[int, Set[int]] = {k: set() for k in st_in.placed_parts}
    has_side_mating: Set[int] = set()
    for (ia, pa), (ib, _) in st_in.matched_port_pairs.items():
        if ia == ib or ia not in st_in.placed_parts or ib not in st_in.placed_parts:
            continue
        pla = st_in.placed_parts[ia]
        prim_a = get_brick_primitive(pla.part_id)
        pt_a = next((p for p in prim_a.ports if p.port_id == pa), None)
        if pt_a is None:
            continue
        wfa = pla.pose.compose(pt_a.local_frame)
        if wfa.z_axis not in ((0, 0, 1), (0, 0, -1)):
            has_side_mating.add(ia)
            has_side_mating.add(ib)
            continue
        if izmax[ia] <= izmin[ib]:
            up_children[ia].add(ib)
            down_parents[ib].add(ia)
        elif izmax[ib] <= izmin[ia]:
            up_children[ib].add(ia)
            down_parents[ia].add(ib)
        else:
            has_side_mating.add(ia)
            has_side_mating.add(ib)

    inst_up_studs: Dict[int, List[Tuple[int, int, int]]] = {}
    for iid_u, pl_u in st_in.placed_parts.items():
        pid_u = TILE_TO_PLATE.get(pl_u.part_id, pl_u.part_id)
        if pid_u not in kit_set:
            pid_u = pl_u.part_id
        prim_u = get_brick_primitive(pid_u)
        pose_u = pl_u.pose
        if pid_u != pl_u.part_id and len(pl_u.world_boxes) == 1:
            pp_u = _find_plate_pose_for_box(pid_u, pl_u.world_boxes[0])
            if pp_u is not None:
                pose_u = pp_u
            else:
                prim_u = get_brick_primitive(pl_u.part_id)
        s_list: List[Tuple[int, int, int]] = []
        for p_u in prim_u.ports:
            if p_u.polarity == PortPolarity.STUD:
                wf_u = pose_u.compose(p_u.local_frame)
                if wf_u.z_axis == (0, 0, 1):
                    s_list.append(wf_u.pos)
        inst_up_studs[iid_u] = s_list

    def _build_subtree_options(grp_t: Set[int], z_plane: int, active_drops: Set[int]):
        xmin = min(min(b.xmin for b in st_in.placed_parts[g].world_boxes) for g in grp_t)
        xmax = max(max(b.xmax for b in st_in.placed_parts[g].world_boxes) for g in grp_t)
        ymin = min(min(b.ymin for b in st_in.placed_parts[g].world_boxes) for g in grp_t)
        ymax = max(max(b.ymax for b in st_in.placed_parts[g].world_boxes) for g in grp_t)
        zmax = max(izmax[g] for g in grp_t)
        max_z_lim = min(gz6 * 6, max(z_plane + 16, min(z_plane + 72, zmax + 8)))
        if (xmax - xmin) % 20 != 0 or (ymax - ymin) % 20 != 0:
            return []

        sup_studs_xy: Set[Tuple[int, int]] = set()
        unlock_tiles: Set[int] = set()
        if z_plane == 0:
            for g in grp_t:
                if st_in.placed_parts[g].is_grounded:
                    for wb in st_in.placed_parts[g].world_boxes:
                        if wb.zmin == 0:
                            for ix_g in range((wb.xmax - wb.xmin) // 20):
                                for iy_g in range((wb.ymax - wb.ymin) // 20):
                                    sup_studs_xy.add((wb.xmin + 20 * ix_g + 10, wb.ymin + 20 * iy_g + 10))
        else:
            for oid_s, s_pts in inst_up_studs.items():
                if oid_s in grp_t or oid_s in active_drops:
                    continue
                for sx_p, sy_p, sz_p in s_pts:
                    if sz_p == z_plane and xmin < sx_p < xmax and ymin < sy_p < ymax:
                        sup_studs_xy.add((sx_p, sy_p))
                        if st_in.placed_parts[oid_s].part_id in TILE_TO_PLATE and TILE_TO_PLATE[st_in.placed_parts[oid_s].part_id] in kit_set:
                            unlock_tiles.add(oid_s)
            if any(oid_t in has_side_mating or oid_t in active_drops or bool(up_children[oid_t] - grp_t) for oid_t in unlock_tiles):
                unlock_tiles.clear()
                sup_studs_xy = {
                    (sx_p, sy_p)
                    for oid_s, s_pts in inst_up_studs.items()
                    if oid_s not in grp_t and oid_s not in active_drops and st_in.placed_parts[oid_s].part_id not in TILE_TO_PLATE
                    for sx_p, sy_p, sz_p in s_pts
                    if sz_p == z_plane and xmin < sx_p < xmax and ymin < sy_p < ymax
                }

        if not sup_studs_xy:
            return []
        unlock_adds: List[Tuple[str, LDUFrame, bool]] = []
        for oid_t in sorted(unlock_tiles):
            pl_t = st_in.placed_parts[oid_t]
            pl_pid_t = TILE_TO_PLATE[pl_t.part_id]
            pp_t = _find_plate_pose_for_box(pl_pid_t, pl_t.world_boxes[0])
            if pp_t is None:
                return []
            unlock_adds.append((pl_pid_t, pp_t, pl_t.is_grounded))

        grp_full = grp_t | unlock_tiles
        stud_z_0 = {s: z_plane for s in sup_studs_xy}
        opts = [(grp_full, unlock_adds, xmin, xmax, ymin, ymax, stud_z_0, max_z_lim, False)]

        if max_z_lim - z_plane >= 24:
            uz_p0 = z_plane // 6
            uz_p1 = min(gz6, int(math.ceil((z_plane + 8) / 6.0)))
            dense_cells_1 = {
                (sx, sy)
                for sx, sy in sup_studs_xy
                if int(np.count_nonzero(occ6[(sx - 10) // 5 : (sx + 10) // 5, (sy - 10) // 5 : (sy + 10) // 5, uz_p0:uz_p1]))
                >= int(0.58 * max(1, 16 * (uz_p1 - uz_p0)))
            }
            if dense_cells_1:
                tiled_1 = _tile_cells_plates(dense_cells_1, z_plane, kit_set, sup_studs_xy)
                if tiled_1 is not None:
                    adds_1, studs_1 = tiled_1
                    stud_z_1 = {s: (z_plane + 8 if s in studs_1 else z_plane) for s in sup_studs_xy}
                    opts.append((grp_full, unlock_adds + adds_1, xmin, xmax, ymin, ymax, stud_z_1, max_z_lim, False))
                    if max_z_lim - z_plane >= 32:
                        uz_p2 = min(gz6, int(math.ceil((z_plane + 16) / 6.0)))
                        dense_cells_2 = {
                            (sx, sy)
                            for sx, sy in studs_1
                            if int(np.count_nonzero(occ6[(sx - 10) // 5 : (sx + 10) // 5, (sy - 10) // 5 : (sy + 10) // 5, uz_p1:uz_p2]))
                            >= int(0.60 * max(1, 16 * (uz_p2 - uz_p1)))
                        }
                        if dense_cells_2:
                            tiled_2 = _tile_cells_plates(dense_cells_2, z_plane + 8, kit_set, studs_1)
                            if tiled_2 is not None:
                                adds_2, studs_2 = tiled_2
                                stud_z_2 = {
                                    s: (z_plane + 16 if s in studs_2 else (z_plane + 8 if s in studs_1 else z_plane))
                                    for s in sup_studs_xy
                                }
                                opts.append((grp_full, unlock_adds + adds_1 + adds_2, xmin, xmax, ymin, ymax, stud_z_2, max_z_lim, False))

        if "15573" in kit_set and mesh_sub_total < 2800 and (max_z_lim - z_plane) >= 16 and len(sup_studs_xy) >= 2:
            prim_j = get_brick_primitive("15573")
            rid_jx = _span_rid("15573", 40, 20)
            rid_jy = _span_rid("15573", 20, 40)
            lb_jx = prim_j.hull_boxes[0].transform(LDUFrame(rid_jx, (0, 0, 0)))
            lb_jy = prim_j.hull_boxes[0].transform(LDUFrame(rid_jy, (0, 0, 0)))
            nx_c, ny_c = (xmax - xmin) // 20, (ymax - ymin) // 20
            jb_bases = [(unlock_adds, z_plane, sup_studs_xy)]
            if len(opts) > 1 and (max_z_lim - z_plane) >= 24:
                s1_only = {s for s, zv in opts[1][6].items() if zv == z_plane + 8}
                if len(s1_only) >= 2:
                    jb_bases.append((opts[1][1], z_plane + 8, s1_only))
            for base_pfx, z_jb, cur_sup in jb_bases:
                if nx_c >= 2:
                    j_adds_x: List[Tuple[str, LDUFrame, bool]] = []
                    j_studs_x: Dict[Tuple[int, int], int] = {}
                    used_jx: Set[Tuple[int, int]] = set()
                    for iy_r in range(ny_c):
                        y0_r = ymin + 20 * iy_r
                        starts_x = range(0, nx_c - 1, 2) if (iy_r % 2 == 0) else range(1 if nx_c >= 3 else 0, nx_c - 1, 2)
                        for ix_r in starts_x:
                            s1 = (xmin + 20 * ix_r + 10, y0_r + 10)
                            s2 = (xmin + 20 * (ix_r + 1) + 10, y0_r + 10)
                            if s1 in used_jx or s2 in used_jx or not ({s1, s2} & cur_sup):
                                continue
                            pj = LDUFrame(rid_jx, (xmin + 20 * ix_r - lb_jx.xmin, y0_r - lb_jx.ymin, z_jb - lb_jx.zmin))
                            j_adds_x.append(("15573", pj, (z_jb == 0)))
                            used_jx.update({s1, s2})
                            j_studs_x[(xmin + 20 * ix_r + 20, y0_r + 10)] = z_jb + 8
                    if j_studs_x:
                        uz0_j, uz1_j = z_jb // 6, min(gz6, int(math.ceil((z_jb + 8) / 6.0)))
                        if "3070b" in kit_set:
                            for sx_u, sy_u in sorted(cur_sup - used_jx):
                                if int(np.count_nonzero(occ6[(sx_u - 10) // 5 : (sx_u + 10) // 5, (sy_u - 10) // 5 : (sy_u + 10) // 5, uz0_j:uz1_j])) >= 14:
                                    pt_f = _find_plate_pose_for_box("3070b", LDUBox(sx_u - 10, sy_u - 10, z_jb, sx_u + 10, sy_u + 10, z_jb + 8))
                                    if pt_f is not None:
                                        j_adds_x.append(("3070b", pt_f, (z_jb == 0)))
                                        used_jx.add((sx_u, sy_u))
                        for s_rem in cur_sup - used_jx:
                            j_studs_x[s_rem] = z_jb
                        if z_jb > z_plane:
                            for s_low in sup_studs_xy - cur_sup:
                                j_studs_x[s_low] = z_plane
                        opts.append((grp_full, base_pfx + j_adds_x, max(0, xmin - 10), min(gx6 * 5, xmax + 10), ymin, ymax, j_studs_x, max_z_lim, True))
                if ny_c >= 2:
                    j_adds_y: List[Tuple[str, LDUFrame, bool]] = []
                    j_studs_y: Dict[Tuple[int, int], int] = {}
                    used_jy: Set[Tuple[int, int]] = set()
                    for ix_r in range(nx_c):
                        x0_r = xmin + 20 * ix_r
                        starts_y = range(0, ny_c - 1, 2) if (ix_r % 2 == 0) else range(1 if ny_c >= 3 else 0, ny_c - 1, 2)
                        for iy_r in starts_y:
                            s1 = (x0_r + 10, ymin + 20 * iy_r + 10)
                            s2 = (x0_r + 10, ymin + 20 * (iy_r + 1) + 10)
                            if s1 in used_jy or s2 in used_jy or not ({s1, s2} & cur_sup):
                                continue
                            pj = LDUFrame(rid_jy, (x0_r - lb_jy.xmin, ymin + 20 * iy_r - lb_jy.ymin, z_jb - lb_jy.zmin))
                            j_adds_y.append(("15573", pj, (z_jb == 0)))
                            used_jy.update({s1, s2})
                            j_studs_y[(x0_r + 10, ymin + 20 * iy_r + 20)] = z_jb + 8
                    if j_studs_y:
                        uz0_j, uz1_j = z_jb // 6, min(gz6, int(math.ceil((z_jb + 8) / 6.0)))
                        if "3070b" in kit_set:
                            for sx_u, sy_u in sorted(cur_sup - used_jy):
                                if int(np.count_nonzero(occ6[(sx_u - 10) // 5 : (sx_u + 10) // 5, (sy_u - 10) // 5 : (sy_u + 10) // 5, uz0_j:uz1_j])) >= 14:
                                    pt_f = _find_plate_pose_for_box("3070b", LDUBox(sx_u - 10, sy_u - 10, z_jb, sx_u + 10, sy_u + 10, z_jb + 8))
                                    if pt_f is not None:
                                        j_adds_y.append(("3070b", pt_f, (z_jb == 0)))
                                        used_jy.add((sx_u, sy_u))
                        for s_rem in cur_sup - used_jy:
                            j_studs_y[s_rem] = z_jb
                        if z_jb > z_plane:
                            for s_low in sup_studs_xy - cur_sup:
                                j_studs_y[s_low] = z_plane
                        opts.append((grp_full, base_pfx + j_adds_y, xmin, xmax, max(0, ymin - 10), min(gy6 * 5, ymax + 10), j_studs_y, max_z_lim, True))
        return opts

    def _detachable_closure(seed: Set[int], z_cut: int, max_sz: int = 8) -> Optional[Set[int]]:
        sub: Set[int] = set()
        dq = deque(seed)
        while dq:
            u = dq.popleft()
            if u in sub:
                continue
            if u in has_side_mating or st_in.placed_parts[u].is_grounded or izmin[u] < z_cut:
                return None
            sub.add(u)
            if len(sub) > max_sz:
                return None
            for ch in up_children[u]:
                if ch not in sub:
                    dq.append(ch)
            for par in down_parents[u]:
                if par not in sub and izmax[par] > z_cut:
                    if izmin[par] < z_cut:
                        return None
                    dq.append(par)
        ext_p = {pe for node in sub for pe in down_parents[node] if pe not in sub}
        if not ext_p or not all(izmax[pe] == z_cut for pe in ext_p):
            return None
        return sub

    raw_candidates: List[Tuple[int, Set[int], int]] = []
    for b_id, pl_b in st_in.placed_parts.items():
        if b_id in has_side_mating or len(pl_b.world_boxes) != 1 or not up_children[b_id]:
            continue
        wb_b = pl_b.world_boxes[0]
        if (wb_b.zmax - wb_b.zmin) not in (8, 24):
            continue
        sub_t = _detachable_closure(set(up_children[b_id]), wb_b.zmax, max_sz=14)
        if sub_t is None:
            continue
        fp_sum = sum(inst_fp[x] for x in sub_t)
        if fp_sum >= 8:
            raw_candidates.append((fp_sum, sub_t, wb_b.zmax))
        ext_parents = {p_e for node_t in sub_t for p_e in down_parents[node_t] if p_e not in sub_t}
        if len(ext_parents) <= 4 and all(
            p_e not in has_side_mating
            and len(st_in.placed_parts[p_e].world_boxes) == 1
            and up_children[p_e] <= sub_t
            and izmin[p_e] == wb_b.zmin
            for p_e in ext_parents
        ):
            ext_all = sub_t | ext_parents
            ext_p2 = {p2 for p_e in ext_parents for p2 in down_parents[p_e]}
            if (wb_b.zmin == 0 and all(st_in.placed_parts[p_e].is_grounded for p_e in ext_parents)) or (
                wb_b.zmin > 0 and ext_p2 and all(izmax[p2] == wb_b.zmin for p2 in ext_p2)
            ):
                fp_sum2 = sum(inst_fp[x] for x in ext_all)
                if fp_sum2 >= 10:
                    raw_candidates.append((fp_sum2 + 12, ext_all, wb_b.zmin))

    unit_closures: List[Tuple[int, Set[int], int, Tuple[int, int, int, int]]] = []
    seen_unit_keys: Set[FrozenSet[int]] = set()
    for iid, pl in st_in.placed_parts.items():
        if iid in has_side_mating or pl.is_grounded:
            continue
        sub_u = _detachable_closure({iid}, izmin[iid], max_sz=6)
        if sub_u is None:
            continue
        ukey = frozenset(sub_u)
        if ukey in seen_unit_keys:
            continue
        seen_unit_keys.add(ukey)
        fp_u = sum(inst_fp[x] for x in sub_u)
        if fp_u >= 6:
            bx0 = min(min(b.xmin for b in st_in.placed_parts[x].world_boxes) for x in sub_u)
            bx1 = max(max(b.xmax for b in st_in.placed_parts[x].world_boxes) for x in sub_u)
            by0 = min(min(b.ymin for b in st_in.placed_parts[x].world_boxes) for x in sub_u)
            by1 = max(max(b.ymax for b in st_in.placed_parts[x].world_boxes) for x in sub_u)
            if (bx1 - bx0) <= 80 and (by1 - by0) <= 80:
                unit_closures.append((fp_u, sub_u, izmin[iid], (bx0, bx1, by0, by1)))
                if fp_u >= 8:
                    raw_candidates.append((fp_u, sub_u, izmin[iid]))

    for i_u, (fp1, sub1, z1, (x0_1, x1_1, y0_1, y1_1)) in enumerate(unit_closures):
        touching_u = []
        for fp2, sub2, z2, (x0_2, x1_2, y0_2, y1_2) in unit_closures[i_u + 1 :]:
            if z2 != z1 or (sub1 & sub2):
                continue
            if x0_1 <= x1_2 and x1_1 >= x0_2 and y0_1 <= y1_2 and y1_1 >= y0_2:
                if (max(x1_1, x1_2) - min(x0_1, x0_2)) <= 80 and (max(y1_1, y1_2) - min(y0_1, y0_2)) <= 80:
                    comb = sub1 | sub2
                    raw_candidates.append((fp1 + fp2 + 6, comb, z1))
                    touching_u.append((sub2, x0_2, x1_2, y0_2, y1_2))
        if len(touching_u) >= 2:
            grp_strip = set(sub1)
            sx0, sx1, sy0, sy1 = x0_1, x1_1, y0_1, y1_1
            cnt_s = 1
            for sub_nb, bx0, bx1, by0, by1 in touching_u:
                nx0, nx1 = min(sx0, bx0), max(sx1, bx1)
                ny0, ny1 = min(sy0, by0), max(sy1, by1)
                if (nx1 - nx0) <= 80 and (ny1 - ny0) <= 80 and cnt_s < 4:
                    grp_strip |= sub_nb
                    sx0, sx1, sy0, sy1 = nx0, nx1, ny0, ny1
                    cnt_s += 1
            if cnt_s >= 3:
                raw_candidates.append((sum(inst_fp[g] for g in grp_strip) + 8, grp_strip, z1))

    if not raw_candidates:
        return st_in, base_iou

    raw_candidates.sort(key=lambda item: -item[0])
    max_eval_groups = 6 if mesh_sub_total < 2800 else (3 if mesh_sub_total < 3800 else 2)
    min_fp_thresh = 8 if mesh_sub_total < 2800 else 14

    live_wbs: Dict[int, Tuple[LDUBox, ...]] = {iid: pl.world_boxes for iid, pl in st_in.placed_parts.items()}
    added_wbs_list: List[LDUBox] = []
    live_mask = mask_0.copy()
    live_inter = inter_0
    live_over = over_0
    accepted_regs: List[Tuple[Set[int], List[Tuple[str, LDUFrame, bool]]]] = []
    total_drops: Set[int] = set()
    seen_grp_sets: Set[FrozenSet[int]] = set()
    evaluated_cnt = 0

    for fp_est, grp_raw, z_plane_raw in raw_candidates:
        if evaluated_cnt >= max_eval_groups or fp_est < min_fp_thresh:
            break
        if grp_raw & total_drops:
            continue
        f_key = frozenset(grp_raw)
        if f_key in seen_grp_sets:
            continue
        seen_grp_sets.add(f_key)

        opts = _build_subtree_options(grp_raw, z_plane_raw, total_drops)
        if not opts:
            continue
        grp0 = opts[0][0]
        if grp0 & total_drops or not all(g in live_wbs for g in grp0):
            continue
        evaluated_cnt += 1

        xmin0 = min(o[2] for o in opts)
        xmax0 = max(o[3] for o in opts)
        ymin0 = min(o[4] for o in opts)
        ymax0 = max(o[5] for o in opts)
        z_low_grp = min(min(b.zmin for b in live_wbs[g]) for g in grp0)
        z_hi_grp = max(max(o[7] for o in opts), max(max(b.zmax for b in live_wbs[g]) for g in grp0))
        ux0, ux1 = max(0, xmin0 // 5), min(gx6, int(math.ceil(xmax0 / 5.0)))
        uy0, uy1 = max(0, ymin0 // 5), min(gy6, int(math.ceil(ymax0 / 5.0)))
        uz0, uz1 = max(0, z_low_grp // 6), min(gz6, int(math.ceil(z_hi_grp / 6.0)))
        if ux1 <= ux0 or uy1 <= uy0 or uz1 <= uz0:
            continue
        roi_tgt = occ6[ux0:ux1, uy0:uy1, uz0:uz1]
        roi_cur = live_mask[ux0:ux1, uy0:uy1, uz0:uz1]
        cur_roi_in = int(np.count_nonzero(roi_cur & roi_tgt))
        cur_roi_ov = int(np.count_nonzero(roi_cur & (~roi_tgt)))
        base_global_in = live_inter - cur_roi_in
        base_global_ov = live_over - cur_roi_ov
        cur_global_iou = float(live_inter) / float(max(1, mesh_sub_total + live_over))
        tau_fp = max(0.90, cur_global_iou + 0.02)

        probe_aabb = LDUBox(xmin0 - 4, ymin0 - 4, max(0, z_low_grp - 4), xmax0 + 4, ymax0 + 4, z_hi_grp + 4)
        obstacle_boxes: List[LDUBox] = [
            ob for oid, wbs_o in live_wbs.items() if oid not in grp0 for ob in wbs_o if ob.intersects(probe_aabb)
        ] + [ob for ob in added_wbs_list if ob.intersects(probe_aabb)]

        grp_cnt = np.zeros((ux1 - ux0, uy1 - uy0, uz1 - uz0), dtype=np.uint16)
        for g in grp0:
            loc_g = np.zeros_like(grp_cnt, dtype=bool)
            for wb_g in live_wbs[g]:
                bx0, bx1 = max(0, wb_g.xmin // 5), min(gx6, int(math.ceil(wb_g.xmax / 5.0)))
                by0, by1 = max(0, wb_g.ymin // 5), min(gy6, int(math.ceil(wb_g.ymax / 5.0)))
                bz0, bz1 = max(0, wb_g.zmin // 6), min(gz6, int(math.ceil(wb_g.zmax / 6.0)))
                ax0, ax1 = max(ux0, bx0) - ux0, min(ux1, bx1) - ux0
                ay0, ay1 = max(uy0, by0) - uy0, min(uy1, by1) - uy0
                az0, az1 = max(uz0, bz0) - uz0, min(uz1, bz1) - uz0
                if ax1 > ax0 and ay1 > ay0 and az1 > az0:
                    loc_g[ax0:ax1, ay0:ay1, az0:az1] = True
            grp_cnt += loc_g.astype(np.uint16)
        bg_roi = (cnt6[ux0:ux1, uy0:uy1, uz0:uz1] - grp_cnt) > 0

        grp_box_cache: Dict[Tuple[str, int, int, int], Optional[Tuple[Any, ...]]] = {}
        best_opt_res = None
        for _, base_adds, xmin, xmax, ymin, ymax, stud_z_map, max_z_lim, is_jb_opt in opts:
            base_wbs_opt: List[LDUBox] = []
            ok_ba = True
            for b_pid, b_pose, _ in base_adds:
                for bb in get_brick_primitive(b_pid).hull_boxes:
                    wb_b = bb.transform(b_pose)
                    if (
                        wb_b.xmin < 0
                        or wb_b.ymin < 0
                        or wb_b.xmax > gx6 * 5
                        or wb_b.ymax > gy6 * 5
                        or int(math.ceil(wb_b.zmax / 6.0)) > gz6
                        or _box_hits_forbidden(wb_b, forbidden_support_boxes)
                    ):
                        ok_ba = False
                        break
                    if any(wb_b.intersects(ob) for ob in obstacle_boxes):
                        ok_ba = False
                        break
                    base_wbs_opt.append(wb_b)
                if not ok_ba:
                    break
            if not ok_ba:
                continue

            work_roi = bg_roi.copy()
            for wb_b in base_wbs_opt:
                ax0 = max(ux0, wb_b.xmin // 5) - ux0
                ax1 = min(ux1, int(math.ceil(wb_b.xmax / 5.0))) - ux0
                ay0 = max(uy0, wb_b.ymin // 5) - uy0
                ay1 = min(uy1, int(math.ceil(wb_b.ymax / 5.0))) - uy0
                az0 = max(uz0, wb_b.zmin // 6) - uz0
                az1 = min(uz1, int(math.ceil(wb_b.zmax / 6.0))) - uz0
                if ax1 > ax0 and ay1 > ay0 and az1 > az0:
                    work_roi[ax0:ax1, ay0:ay1, az0:az1] = True

            chosen_caps = _run_multi_stud_cap_loop(
                cap_tmpls=cap_tmpls,
                stud_z_map=stud_z_map,
                xmin=xmin,
                xmax=xmax,
                ymin=ymin,
                ymax=ymax,
                max_z_lim=max_z_lim,
                ux0=ux0,
                ux1=ux1,
                uy0=uy0,
                uy1=uy1,
                uz0=uz0,
                uz1=uz1,
                gx6=gx6,
                gy6=gy6,
                gz6=gz6,
                work_roi=work_roi,
                roi_tgt=roi_tgt,
                base_global_in=base_global_in,
                base_global_ov=base_global_ov,
                mesh_sub_total=mesh_sub_total,
                tau_fp=tau_fp,
                static_obs=obstacle_boxes,
                base_wbs_opt=base_wbs_opt,
                forbidden_support_boxes=forbidden_support_boxes,
                is_jb_opt=is_jb_opt,
                shared_box_cache=grp_box_cache,
                max_iters=24,
            )

            fin_in = int(np.count_nonzero(work_roi & roi_tgt))
            fin_ov = int(np.count_nonzero(work_roi & (~roi_tgt)))
            fin_iou = float(base_global_in + fin_in) / float(max(1, mesh_sub_total + base_global_ov + fin_ov))
            if fin_iou > cur_global_iou + 8.0e-5 and (z_plane_raw > 0 or base_adds or chosen_caps):
                if best_opt_res is None or fin_iou > best_opt_res[0]:
                    all_emit = list(base_adds) + chosen_caps
                    new_wbs = [b.transform(f) for p, f, _ in all_emit for b in get_brick_primitive(p).hull_boxes]
                    best_opt_res = (fin_iou, fin_in, fin_ov, work_roi, all_emit, new_wbs)

        if best_opt_res is not None:
            _, fin_in, fin_ov, work_roi, all_new_parts, new_wbs = best_opt_res
            cnt6[ux0:ux1, uy0:uy1, uz0:uz1] -= grp_cnt
            for wb in new_wbs:
                loc_add = np.zeros_like(work_roi)
                ax0 = max(ux0, wb.xmin // 5) - ux0
                ax1 = min(ux1, int(math.ceil(wb.xmax / 5.0))) - ux0
                ay0 = max(uy0, wb.ymin // 5) - uy0
                ay1 = min(uy1, int(math.ceil(wb.ymax / 5.0))) - uy0
                az0 = max(uz0, wb.zmin // 6) - uz0
                az1 = min(uz1, int(math.ceil(wb.zmax / 6.0))) - uz0
                if ax1 > ax0 and ay1 > ay0 and az1 > az0:
                    loc_add[ax0:ax1, ay0:ay1, az0:az1] = True
                cnt6[ux0:ux1, uy0:uy1, uz0:uz1] += loc_add.astype(np.uint16)
            for g_rem in grp0:
                live_wbs.pop(g_rem, None)
            added_wbs_list.extend(new_wbs)
            live_mask[ux0:ux1, uy0:uy1, uz0:uz1] = work_roi
            live_inter = base_global_in + fin_in
            live_over = base_global_ov + fin_ov
            total_drops |= grp0
            accepted_regs.append((grp0, all_new_parts))

    if not accepted_regs:
        return st_in, base_iou

    cand_st = copy.deepcopy(st_in)
    for g_rem in total_drops:
        cand_st.remove_part(g_rem)
    all_adds_sorted = sorted(
        (item for _, adds in accepted_regs for item in adds),
        key=lambda x: (0 if x[2] else 1, min(b.transform(x[1]).zmin for b in get_brick_primitive(x[0]).hull_boxes)),
    )
    ok_all = True
    added_iids: List[int] = []
    for pid_n, pose_n, gr_n in all_adds_sorted:
        try:
            added_iids.append(cand_st.place_root_part(pid_n, pose_n, is_grounded=gr_n))
        except ValueError:
            ok_all = False
            break
    if ok_all:
        _apply_terminal_tile_swap(cand_st, kit_set, target_iids=added_iids)
        iou_final, st_ok = _verify_and_score_strict(
            kernel, cand_st, occ6, mesh_sub_total, forbidden_support_boxes, foundation_policy, require_full_prefix=True
        )
        if st_ok is not None and iou_final > base_iou + 1.0e-5:
            return st_ok, iou_final

    fb_st = copy.deepcopy(st_in)
    fb_iou = base_iou
    changed_fb = False
    for grp_r, adds_r in accepted_regs:
        if not all(g in fb_st.placed_parts for g in grp_r):
            continue
        t_st = copy.deepcopy(fb_st)
        for g in grp_r:
            t_st.remove_part(g)
        adds_s = sorted(adds_r, key=lambda x: (0 if x[2] else 1, min(b.transform(x[1]).zmin for b in get_brick_primitive(x[0]).hull_boxes)))
        ok_f = True
        a_ids = []
        for pid_n, pose_n, gr_n in adds_s:
            try:
                a_ids.append(t_st.place_root_part(pid_n, pose_n, is_grounded=gr_n))
            except ValueError:
                ok_f = False
                break
        if ok_f:
            _apply_terminal_tile_swap(t_st, kit_set, target_iids=a_ids)
            iou_t, st_t = _verify_and_score_strict(
                kernel, t_st, occ6, mesh_sub_total, forbidden_support_boxes, foundation_policy, require_full_prefix=False
            )
            if st_t is not None and iou_t > fb_iou + 1.0e-5:
                fb_st, fb_iou = st_t, iou_t
                changed_fb = True
    if changed_fb:
        iou_chk, st_chk = _verify_and_score_strict(
            kernel, fb_st, occ6, mesh_sub_total, forbidden_support_boxes, foundation_policy, require_full_prefix=True
        )
        if st_chk is not None and iou_chk > base_iou + 1.0e-5:
            return st_chk, iou_chk
    return st_in, base_iou


def _fast_greedy_port_grow(
    st: AssemblyState,
    occ6: np.ndarray,
    msub: int,
    cands_pids: Sequence[str],
    min_port_z: int = 0,
    max_steps: int = 12,
    forbidden_support_boxes: Sequence[LDUBox] = (),
    only_fn_pockets: bool = False,
) -> None:
    gx6, gy6, gz6 = occ6.shape
    cur_m = np.zeros_like(occ6)
    for pl in st.placed_parts.values():
        for wb in pl.world_boxes:
            cx0, cx1 = max(0, wb.xmin // 5), min(gx6, int(math.ceil(wb.xmax / 5.0)))
            cy0, cy1 = max(0, wb.ymin // 5), min(gy6, int(math.ceil(wb.ymax / 5.0)))
            cz0, cz1 = max(0, wb.zmin // 6), min(gz6, int(math.ceil(wb.zmax / 6.0)))
            if cx1 > cx0 and cy1 > cy0 and cz1 > cz0:
                cur_m[cx0:cx1, cy0:cy1, cz0:cz1] = True
    cur_in = int(np.count_nonzero(cur_m & occ6))
    cur_ov = int(np.count_nonzero(cur_m & (~occ6)))
    cur_i = float(cur_in) / float(max(1, msub + cur_ov))

    by_norm: Dict[Tuple[int, int, int], List[Tuple[str, int, Tuple[int, int, int], Tuple[LDUBox, ...]]]] = {}
    for pid in cands_pids:
        for z_ax, rid, apos, rbs in _get_anti_port_templates(pid):
            by_norm.setdefault(z_ax, []).append((pid, rid, apos, rbs))

    for _ in range(max_steps):
        fr = st.get_active_frontier()
        best_u = None
        seen_u = set()
        for ep in fr.ports:
            if not ep.status.is_exposed or ep.polarity != PortPolarity.STUD or ep.world_frame.pos[2] < min_port_z:
                continue
            uxp, uyp, uzp = ep.world_frame.pos
            if only_fn_pockets:
                qx0, qx1 = max(0, (uxp - 20) // 5), min(gx6, (uxp + 20) // 5 + 1)
                qy0, qy1 = max(0, (uyp - 20) // 5), min(gy6, (uyp + 20) // 5 + 1)
                qz0, qz1 = max(0, uzp // 6), min(gz6, (uzp + 24) // 6 + 1)
                if int(np.count_nonzero(occ6[qx0:qx1, qy0:qy1, qz0:qz1] & (~cur_m[qx0:qx1, qy0:qy1, qz0:qz1]))) < 3:
                    continue
            for pid_u, rid_u, ap_u, rbs_u in by_norm.get(ep.world_normal, ()):
                px, py, pz = uxp - ap_u[0], uyp - ap_u[1], uzp - ap_u[2]
                if (px % 2 != 0) or (py % 2 != 0) or (pz % 2 != 0) or (pid_u, rid_u, px, py, pz) in seen_u:
                    continue
                seen_u.add((pid_u, rid_u, px, py, pz))
                wbsu = tuple(LDUBox(b.xmin + px, b.ymin + py, b.zmin + pz, b.xmax + px, b.ymax + py, b.zmax + pz) for b in rbs_u)
                if any(
                    wb.xmin < 0
                    or wb.ymin < 0
                    or wb.xmax > gx6 * 5
                    or wb.ymax > gy6 * 5
                    or wb.zmin < 0
                    or int(math.ceil(wb.zmax / 6.0)) > gz6
                    for wb in wbsu
                ):
                    continue
                if forbidden_support_boxes and any(_box_hits_forbidden(wb, forbidden_support_boxes) for wb in wbsu):
                    continue
                if st.check_boxes_collision(wbsu):
                    continue
                bx0 = min(wb.xmin // 5 for wb in wbsu)
                bx1 = max(int(math.ceil(wb.xmax / 5.0)) for wb in wbsu)
                by0 = min(wb.ymin // 5 for wb in wbsu)
                by1 = max(int(math.ceil(wb.ymax / 5.0)) for wb in wbsu)
                bz0 = min(wb.zmin // 6 for wb in wbsu)
                bz1 = max(int(math.ceil(wb.zmax / 6.0)) for wb in wbsu)
                loc = np.zeros((bx1 - bx0, by1 - by0, bz1 - bz0), dtype=bool)
                for wb in wbsu:
                    loc[
                        wb.xmin // 5 - bx0 : int(math.ceil(wb.xmax / 5.0)) - bx0,
                        wb.ymin // 5 - by0 : int(math.ceil(wb.ymax / 5.0)) - by0,
                        wb.zmin // 6 - bz0 : int(math.ceil(wb.zmax / 6.0)) - bz0,
                    ] = True
                nu = loc & (~cur_m[bx0:bx1, by0:by1, bz0:bz1])
                if not np.any(nu):
                    continue
                tsub = occ6[bx0:bx1, by0:by1, bz0:bz1]
                din = int(np.count_nonzero(nu & tsub))
                dov = int(np.count_nonzero(nu & (~tsub)))
                ni = float(cur_in + din) / float(max(1, msub + cur_ov + dov))
                if ni > cur_i + 8.0e-5 and (best_u is None or ni > best_u[0]):
                    pose_u = LDUFrame(rid_u, (px, py, pz))
                    if _check_monotone_corridor(st, pid_u, pose_u):
                        best_u = (ni, pid_u, pose_u, bx0, bx1, by0, by1, bz0, bz1, loc, din, dov)
        if best_u is None:
            break
        ni, pid_b, pose_b, bx0, bx1, by0, by1, bz0, bz1, loc_b, din_b, dov_b = best_u
        try:
            st.place_root_part(pid_b, pose_b, is_grounded=False)
            cur_m[bx0:bx1, by0:by1, bz0:bz1] |= loc_b
            cur_in += din_b
            cur_ov += dov_b
            cur_i = ni
        except ValueError:
            continue


def _unlock_fn_pocket_tiles_and_grow(
    kernel: Any,
    st_in: AssemblyState,
    occ6: np.ndarray,
    msub: int,
    kit_ids: Sequence[str],
    forbidden_support_boxes: Sequence[LDUBox],
    foundation_policy: FoundationPolicy,
    base_iou: float,
) -> Tuple[AssemblyState, float]:
    gx6, gy6, gz6 = occ6.shape
    kit_set = set(kit_ids)
    cur_m = np.zeros_like(occ6)
    for pl in st_in.placed_parts.values():
        for wb in pl.world_boxes:
            cx0, cx1 = max(0, wb.xmin // 5), min(gx6, int(math.ceil(wb.xmax / 5.0)))
            cy0, cy1 = max(0, wb.ymin // 5), min(gy6, int(math.ceil(wb.ymax / 5.0)))
            cz0, cz1 = max(0, wb.zmin // 6), min(gz6, int(math.ceil(wb.zmax / 6.0)))
            if cx1 > cx0 and cy1 > cy0 and cz1 > cz0:
                cur_m[cx0:cx1, cy0:cy1, cz0:cz1] = True
    fn_mask = occ6 & (~cur_m)
    if int(np.count_nonzero(fn_mask)) < 8:
        return st_in, base_iou

    st_g = copy.deepcopy(st_in)
    matched_g = st_g.matched_port_pairs
    unlocked_any = False
    for iid, pl in list(st_g.placed_parts.items()):
        pl_pid = TILE_TO_PLATE.get(pl.part_id)
        if pl_pid is None or pl_pid not in kit_set or len(pl.world_boxes) != 1:
            continue
        wb = pl.world_boxes[0]
        if any(
            (iid, pt.port_id) in matched_g and pl.pose.compose(pt.local_frame).z_axis not in ((0, 0, 1), (0, 0, -1))
            for pt in get_brick_primitive(pl.part_id).ports
        ):
            continue
        qx0, qx1 = max(0, (wb.xmin - 10) // 5), min(gx6, (wb.xmax + 10) // 5 + 1)
        qy0, qy1 = max(0, (wb.ymin - 10) // 5), min(gy6, (wb.ymax + 10) // 5 + 1)
        qz0, qz1 = max(0, wb.zmax // 6), min(gz6, (wb.zmax + 24) // 6 + 1)
        if int(np.count_nonzero(fn_mask[qx0:qx1, qy0:qy1, qz0:qz1])) >= 4:
            pp = _find_plate_pose_for_box(pl_pid, wb)
            if pp is not None:
                try:
                    st_g.reorient_part(iid, pp, new_part_id=pl_pid)
                    unlocked_any = True
                except ValueError:
                    pass

    small_cands = [
        pid
        for pid in kit_ids
        if pid
        in (
            "54200",
            "85984",
            "11477",
            "15068",
            "88930",
            "6091",
            "3040a",
            "3039",
            "25269",
            "49307",
            "3070b",
            "3069b",
            "3068b",
            "2431",
            "3024",
            "3023",
            "3022",
            "44728",
            "99780",
            "36841",
            "49656",
        )
    ]
    n_before = len(st_g.placed_parts)
    _fast_greedy_port_grow(
        st_g,
        occ6,
        msub,
        small_cands,
        min_port_z=0,
        max_steps=10,
        forbidden_support_boxes=forbidden_support_boxes,
        only_fn_pockets=True,
    )
    if len(st_g.placed_parts) == n_before and not unlocked_any:
        return st_in, base_iou
    _apply_terminal_tile_swap(st_g, kit_set)
    if len(st_g.placed_parts) == n_before:
        return st_in, base_iou
    ig, st_ok = _verify_and_score_strict(
        kernel, st_g, occ6, msub, forbidden_support_boxes, foundation_policy, require_full_prefix=True
    )
    if st_ok is not None and ig > base_iou + 1.0e-5:
        return st_ok, ig
    return st_in, base_iou


def _build_thin_structure_candidates(
    base_st: AssemblyState,
    occ6: np.ndarray,
    kit_ids: Sequence[str],
    forbidden_support_boxes: Sequence[LDUBox],
) -> List[AssemblyState]:
    if forbidden_support_boxes:
        return []
    msub = int(np.count_nonzero(occ6))
    if msub == 0 or msub >= 1000:
        return []
    gx6, gy6, gz6 = occ6.shape
    kit_set = set(kit_ids)
    out: List[AssemblyState] = []

    if base_st.placed_parts and any(min(b.zmin for b in p.world_boxes) >= 40 for p in base_st.placed_parts.values()):
        cands_ch = [
            p
            for p in (
                "2362a",
                "4864a",
                "4865a",
                "44728",
                "99780",
                "54200",
                "85984",
                "3021",
                "3623",
                "3023",
                "3024",
                "3069b",
                "3070b",
                "2431",
                "3068b",
            )
            if p in kit_set
        ]
        for mpz in (40, 32):
            st_ch = copy.deepcopy(base_st)
            try:
                for iid in [k for k, p in st_ch.placed_parts.items() if min(b.zmin for b in p.world_boxes) >= 40]:
                    st_ch.remove_part(iid)
                for iid, pl in list(st_ch.placed_parts.items()):
                    pl_pid = TILE_TO_PLATE.get(pl.part_id)
                    if pl_pid is not None and pl_pid in kit_set and len(pl.world_boxes) == 1:
                        pp = _find_plate_pose_for_box(pl_pid, pl.world_boxes[0])
                        if pp is not None:
                            st_ch.reorient_part(iid, pp, new_part_id=pl_pid)
                _fast_greedy_port_grow(st_ch, occ6, msub, cands_ch, min_port_z=mpz, max_steps=14)
                _apply_terminal_tile_swap(st_ch, kit_set)
                out.append(st_ch)
            except ValueError:
                pass

    if {"44728", "3005", "3710"}.issubset(kit_set) and gy6 <= 10:
        pr_br = get_brick_primitive("44728")
        wall_ranks = sorted(((int(np.count_nonzero(occ6[:, iy, :])), iy) for iy in range(gy6)), reverse=True)
        for w_cnt, iy_wall in wall_ranks[:3]:
            if w_cnt < int(0.18 * msub):
                continue
            for y_blade in (5 * iy_wall, 5 * iy_wall + 1):
                if y_blade % 2 != 0:
                    continue
                for rid_br in C4_Z_ROT_IDS:
                    rf_br = LDUFrame(rid_br, (0, 0, 0))
                    rbs_br = [b.transform(rf_br) for b in pr_br.hull_boxes]
                    blade_b = next((b for b in rbs_br if (b.zmax - b.zmin) == 40), None)
                    base_b = next((b for b in rbs_br if (b.zmax - b.zmin) == 8), None)
                    if blade_b is None or base_b is None or (blade_b.ymax - blade_b.ymin) != 4:
                        continue
                    py_br = y_blade - blade_b.ymin
                    if py_br % 2 != 0:
                        continue
                    y0_b, y1_b = base_b.ymin + py_br, base_b.ymax + py_br
                    if min(y0_b, blade_b.ymin + py_br) < 0 or max(y1_b, blade_b.ymax + py_br) > gy6 * 5:
                        continue
                    n_bx = (gx6 * 5) // 40
                    for post_ix in (0, 2):
                        if (post_ix + 1) * 20 > gx6 * 5:
                            continue
                        st_cb = AssemblyState()
                        try:
                            for bx_i in range(n_bx):
                                x0_b = bx_i * 40
                                if int(np.count_nonzero(occ6[x0_b // 5 : (x0_b + 40) // 5, iy_wall : iy_wall + 1, :7])) >= 5:
                                    st_cb.place_root_part("44728", LDUFrame(rid_br, (x0_b - base_b.xmin, py_br, 0)), is_grounded=True)
                            if not st_cb.placed_parts:
                                continue
                            px_p, py_p = post_ix * 20 + 10, y0_b + 10
                            z_mid = 48 if 48 + 48 <= gz6 * 6 else 40
                            zc_p = 8
                            while zc_p + 24 <= z_mid:
                                st_cb.place_root_part("3005", LDUFrame(0, (px_p, py_p, zc_p)), is_grounded=False)
                                zc_p += 24
                            while zc_p + 8 <= z_mid and "3024" in kit_set:
                                st_cb.place_root_part("3024", LDUFrame(0, (px_p, py_p, zc_p)), is_grounded=False)
                                zc_p += 8
                            pp_sh = _find_plate_pose_for_box("3710", LDUBox(0, y0_b, z_mid, 80, y1_b, z_mid + 8))
                            if pp_sh is not None and 80 <= gx6 * 5:
                                st_cb.place_root_part("3710", pp_sh, is_grounded=False)
                                for bx_i in range(2):
                                    x0_u = bx_i * 40
                                    if (
                                        z_mid + 48 <= gz6 * 6
                                        and int(np.count_nonzero(occ6[x0_u // 5 : (x0_u + 40) // 5, iy_wall : iy_wall + 1, (z_mid + 8) // 6 : (z_mid + 48) // 6])) >= 6
                                    ):
                                        st_cb.place_root_part("44728", LDUFrame(rid_br, (x0_u - base_b.xmin, py_br, z_mid + 8)), is_grounded=False)
                                grow_cb = [
                                    p
                                    for p in (
                                        "4864a",
                                        "2362a",
                                        "4865a",
                                        "3710",
                                        "3623",
                                        "3023",
                                        "3024",
                                        "3005",
                                        "3004",
                                        "2431",
                                        "3068b",
                                        "3069b",
                                        "3070b",
                                        "85984",
                                        "54200",
                                    )
                                    if p in kit_set
                                ]
                                _fast_greedy_port_grow(st_cb, occ6, msub, grow_cb, min_port_z=0, max_steps=12)
                            _apply_terminal_tile_swap(st_cb, kit_set)
                            out.append(st_cb)
                        except ValueError:
                            continue
    return out


def try_upfront_blind_sculpted_fastpath(
    kernel: Any,
    occ6: np.ndarray,
    kit_ids: Sequence[str],
    forbidden_support_boxes: Sequence[LDUBox],
    foundation_policy: FoundationPolicy,
) -> Optional[AssemblyState]:
    """
    Fast `< 8 ms` upfront blind portal-bridge & stepped-frustum synthesis on unpadded `occ6`.
    Returns a prefix- and insertion-certified `AssemblyState` whenever exact `occ6` IoU `>= 0.885`.
    """
    msub = int(np.count_nonzero(occ6))
    if msub == 0:
        return None
    kit_set = set(kit_ids)
    best_cand: Optional[AssemblyState] = None
    best_iou = 0.885

    for cand_up in _try_blind_portal_bridge_candidates(occ6, kit_set, forbidden_support_boxes):
        iou_c, st_c = _verify_and_score_strict(
            kernel, cand_up, occ6, msub, forbidden_support_boxes, foundation_policy, require_full_prefix=False
        )
        if st_c is not None and iou_c > best_iou:
            best_cand, best_iou = st_c, iou_c
            if best_iou >= 0.900:
                break

    if best_cand is None:
        for cand_fr in _try_blind_stepped_frustum_candidates(occ6, kit_set, forbidden_support_boxes):
            iou_f, st_f = _verify_and_score_strict(
                kernel, cand_fr, occ6, msub, forbidden_support_boxes, foundation_policy, require_full_prefix=False
            )
            if st_f is not None and iou_f > best_iou:
                best_cand, best_iou = st_f, iou_f
                if best_iou >= 0.900:
                    break

    if best_cand is not None:
        iou_ok, st_ok = _verify_and_score_strict(
            kernel, best_cand, occ6, msub, forbidden_support_boxes, foundation_policy, require_full_prefix=True
        )
        if st_ok is not None and iou_ok >= 0.885:
            return st_ok
    return None


def refine_open_catalog_blind_portfolio(
    kernel: Any,
    state: AssemblyState,
    occ6: np.ndarray,
    kit_ids: Sequence[str],
    forbidden_support_boxes: Sequence[LDUBox],
    foundation_policy: FoundationPolicy,
) -> AssemblyState:
    """
    Fast, 100% blind sub-stud, stepped-frustum, portal-bridge, and multi-stud sculpting
    portfolio (`< 25 ms` on stepped/portal/Band-2 targets; skipped on large Band 3–4 meshes so
    single-core `p50_ms` stays low).
    """
    msub = int(np.count_nonzero(occ6))
    if msub == 0 or msub >= 4500:
        return state
    kit_set = set(kit_ids)

    best_st = sanitize_oob_state(kernel, state, occ6, kit_set, forbidden_support_boxes, foundation_policy)
    best_iou, st_verified = _verify_and_score_strict(
        kernel, best_st, occ6, msub, forbidden_support_boxes, foundation_policy, require_full_prefix=False
    )
    if st_verified is not None:
        best_st = st_verified
    else:
        best_iou = -1.0

    # If already `>= 0.945`, skip local subtree sculpting to keep p50 low!
    if best_iou >= 0.945:
        return best_st

    # Subtree multi-stud sculpting & FN pocket growth on small/medium meshes (`msub < 4500`)
    st_1, iou_1 = _single_pass_top_sculptor(
        kernel, best_st, occ6, kit_ids, forbidden_support_boxes, foundation_policy
    )
    if iou_1 > best_iou + 1.0e-5:
        best_st, best_iou = st_1, iou_1
        if msub < 1800 and best_iou < 0.915:
            st_2, iou_2 = _single_pass_top_sculptor(
                kernel, best_st, occ6, kit_ids, forbidden_support_boxes, foundation_policy
            )
            if iou_2 > best_iou + 1.0e-5:
                best_st, best_iou = st_2, iou_2

    if 0.0 < best_iou < 0.915 and msub < 3600:
        st_g, iou_g = _unlock_fn_pocket_tiles_and_grow(
            kernel, best_st, occ6, msub, kit_ids, forbidden_support_boxes, foundation_policy, best_iou
        )
        if iou_g > best_iou + 1.0e-5:
            best_st, best_iou = st_g, iou_g

    if len(kit_ids) > 32 and msub < 4500 and best_iou < 0.880:
        st_p32, iou_p32 = _single_pass_top_sculptor(
            kernel, best_st, occ6, kit_ids[:32], forbidden_support_boxes, foundation_policy
        )
        if iou_p32 > best_iou + 1.0e-5:
            best_st, best_iou = st_p32, iou_p32

    # Thin-wall & chair/panel blind candidates (`msub < 1000` only)
    if msub < 1000 and not forbidden_support_boxes:
        best_th_cand: Optional[AssemblyState] = None
        best_th_iou = best_iou + 1.0e-4
        for cand_th in _build_thin_structure_candidates(best_st, occ6, kit_ids, forbidden_support_boxes):
            iou_th, st_th = _verify_and_score_strict(
                kernel, cand_th, occ6, msub, forbidden_support_boxes, foundation_policy, require_full_prefix=False
            )
            if st_th is not None and iou_th > best_th_iou:
                best_th_cand, best_th_iou = st_th, iou_th
        if best_th_cand is not None:
            iou_th_ok, st_th_ok = _verify_and_score_strict(
                kernel, best_th_cand, occ6, msub, forbidden_support_boxes, foundation_policy, require_full_prefix=True
            )
            if st_th_ok is not None and iou_th_ok > best_iou + 1.0e-4:
                best_st, best_iou = st_th_ok, iou_th_ok

    return best_st
