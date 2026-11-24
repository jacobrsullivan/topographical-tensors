"""
Continuous 6-axis cardinal swept-volume insertion corridor & course atomic ordering checks (`Phase 15`).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from topo_brick.evaluation.static_equilibrium import SparseStaticEquilibriumValidator
from topo_brick.interface.brick_catalog import get_brick_primitive
from topo_brick.interface.ldu_frames import LDUBox, LDUFrame
from topo_brick.interface.port_manifold import AssemblyState, CandidatePool, TrialPartRollbackToken


def _boxes_have_continuous_cardinal_corridor(
    new_boxes: Sequence[LDUBox],
    existing_boxes: Sequence[LDUBox],
) -> bool:
    """
    Exact continuous 1D swept-volume half-ray clearance (`t in (0, +inf)`) along any of the
    6 cardinal world directions `(+Z, -Z, +X, -X, +Y, -Y)` (`F10`: zero intermediate tunneling
    and zero `48 LDU` horizon cutoff).
    """
    if not new_boxes or not existing_boxes:
        return True
    # 1. +Z extraction (t > 0): collides iff XY strictly overlap AND eb.zmax > nb.zmin
    if not any(
        nb.xmin < eb.xmax
        and nb.xmax > eb.xmin
        and nb.ymin < eb.ymax
        and nb.ymax > eb.ymin
        and eb.zmax > nb.zmin
        for nb in new_boxes
        for eb in existing_boxes
    ):
        return True
    # 2. -Z extraction (t > 0 along -Z): collides iff XY strictly overlap AND eb.zmin < nb.zmax
    if not any(
        nb.xmin < eb.xmax
        and nb.xmax > eb.xmin
        and nb.ymin < eb.ymax
        and nb.ymax > eb.ymin
        and eb.zmin < nb.zmax
        for nb in new_boxes
        for eb in existing_boxes
    ):
        return True
    # 3. +X extraction
    if not any(
        nb.ymin < eb.ymax
        and nb.ymax > eb.ymin
        and nb.zmin < eb.zmax
        and nb.zmax > eb.zmin
        and eb.xmax > nb.xmin
        for nb in new_boxes
        for eb in existing_boxes
    ):
        return True
    # 4. -X extraction
    if not any(
        nb.ymin < eb.ymax
        and nb.ymax > eb.ymin
        and nb.zmin < eb.zmax
        and nb.zmax > eb.zmin
        and eb.xmin < nb.xmax
        for nb in new_boxes
        for eb in existing_boxes
    ):
        return True
    # 5. +Y extraction
    if not any(
        nb.xmin < eb.xmax
        and nb.xmax > eb.xmin
        and nb.zmin < eb.zmax
        and nb.zmax > eb.zmin
        and eb.ymax > nb.ymin
        for nb in new_boxes
        for eb in existing_boxes
    ):
        return True
    # 6. -Y extraction
    if not any(
        nb.xmin < eb.xmax
        and nb.xmax > eb.xmin
        and nb.zmin < eb.zmax
        and nb.zmax > eb.zmin
        and eb.ymin < nb.ymax
        for nb in new_boxes
        for eb in existing_boxes
    ):
        return True
    return False


def _swept_obb_hits_along_ray(
    obb_moving: Any,
    obb_static: Any,
    ray_dir: np.ndarray,
    allowed_penetration_ldu: float = 0.25,
) -> bool:
    """
    Exact continuous 1D Minkowski-ray SAT intersection for `obb_moving + t * ray_dir` (`t in (0, +inf)`)
    against `obb_static` (`F10` continuous $SE(3)$ swept-OBB check).
    """
    ax_a = np.asarray(obb_moving.axes_rows, dtype=np.float64)
    he_a = np.asarray(obb_moving.half_extents, dtype=np.float64)
    ct_a = np.asarray(obb_moving.center, dtype=np.float64)
    ax_b = np.asarray(obb_static.axes_rows, dtype=np.float64)
    he_b = np.asarray(obb_static.half_extents, dtype=np.float64)
    ct_b = np.asarray(obb_static.center, dtype=np.float64)
    delta0 = ct_a - ct_b
    d_v = np.asarray(ray_dir, dtype=np.float64)

    cand_axes: List[np.ndarray] = [ax_a[0], ax_a[1], ax_a[2], ax_b[0], ax_b[1], ax_b[2]]
    for i in range(3):
        for j in range(3):
            cr = np.cross(ax_a[i], ax_b[j])
            nrm = float(np.linalg.norm(cr))
            if nrm > 1e-8:
                cand_axes.append(cr / nrm)
        cr_da = np.cross(d_v, ax_a[i])
        n_da = float(np.linalg.norm(cr_da))
        if n_da > 1e-8:
            cand_axes.append(cr_da / n_da)
        cr_db = np.cross(d_v, ax_b[i])
        n_db = float(np.linalg.norm(cr_db))
        if n_db > 1e-8:
            cand_axes.append(cr_db / n_db)

    t_lo = 0.0
    t_hi = float("inf")
    eps = float(allowed_penetration_ldu)
    for u in cand_axes:
        r_sum = float(np.sum(np.abs(ax_a @ u) * he_a) + np.sum(np.abs(ax_b @ u) * he_b)) - eps
        if r_sum <= 0.0:
            return False
        alpha = float(np.dot(d_v, u))
        beta = float(np.dot(delta0, u))
        if abs(alpha) <= 1e-12:
            if abs(beta) >= r_sum:
                return False
        else:
            t1 = (-beta - r_sum) / alpha
            t2 = (-beta + r_sum) / alpha
            lo_u = t1 if t1 < t2 else t2
            hi_u = t2 if t1 < t2 else t1
            if lo_u > t_lo:
                t_lo = lo_u
            if hi_u < t_hi:
                t_hi = hi_u
            if t_lo >= t_hi - 1e-7 or t_hi <= 1e-6:
                return False
    return t_lo < t_hi - 1e-7 and t_hi > 1e-6


def check_monotone_insertion_corridor(
    state_before: AssemblyState,
    part_id: str,
    pose: LDUFrame,
) -> bool:
    """
    Check whether placing `(part_id, pose)` into `state_before` has an exact continuous half-ray
    (`t in (0, +inf)`) swept-box insertion corridor along any of the 6 cardinal world axes
    `(+Z, -Z, +X, -X, +Y, -Y)` (`F10`: zero tunneling, zero `48 LDU` horizon cutoff).
    """
    if not state_before._parts:
        return True
    prim = get_brick_primitive(part_id)
    base_boxes = tuple(b.transform(pose) for b in prim.hull_boxes)
    existing_boxes = [eb for p in state_before._parts.values() for eb in p.world_boxes]
    return _boxes_have_continuous_cardinal_corridor(base_boxes, existing_boxes)


def check_rigid_group_insertion_corridor(
    state_before: AssemblyState,
    group_boxes: Sequence[LDUBox],
) -> bool:
    """
    Check whether a pre-assembled rigid subassembly (`PLACE_SUBASSEMBLY`) with combined
    hull boxes `group_boxes` has an exact continuous half-ray (`t in (0, +inf)`) insertion
    corridor into `state_before` along any of the 6 cardinal world directions (`F10`).
    """
    if not group_boxes or not state_before._parts:
        return True
    existing_boxes = [eb for p in state_before._parts.values() for eb in p.world_boxes]
    return _boxes_have_continuous_cardinal_corridor(group_boxes, existing_boxes)


def _order_course_atomic_1by1(
    state: AssemblyState,
    pool: CandidatePool,
    chosen_cids: Sequence[int],
    validator: SparseStaticEquilibriumValidator,
    prior_peak_utilization: Optional[float] = None,
) -> Optional[List[int]]:
    """
    Schedule a verified 1-by-1 atomic `PLACE_PART` order for `chosen_cids` (`R8-WS1`)
    Assuming `state + chosen_cids` as a whole is already static-equilibrium stable:
      - Orders central / plumb-supported bricks first so stabilizing normal preload is applied
        before overhanging sibling bricks.
      - Uses exact Minkowski convex-cone subadditivity (`can_extend_plumb_primal_witness`) on
        plumb steps and full 6-DoF `validator.evaluate` LP on overhanging intermediate steps,
        retrying alternative sibling orderings (Kahn greedy swap) if an overhanging brick needs
        a sibling counterweight placed first.
      - Returns `None` ONLY when NO 1-by-1 single-part permutation is statically stable at every
        intermediate step (e.g. a genuine mutual-counterweight or arch course requiring
        `PLACE_SUBASSEMBLY` or `ADD_TEMP_SUPPORT`).
    """
    cids = list(chosen_cids)
    if len(cids) <= 1:
        return cids

    grounded_pts = [
        (0.5 * (b.xmin + b.xmax), 0.5 * (b.ymin + b.ymax))
        for p in state.placed_parts.values()
        if p.is_grounded
        for b in p.world_boxes
    ]
    gx_mean = float(np.mean([pt[0] for pt in grounded_pts])) if grounded_pts else 0.0
    gy_mean = float(np.mean([pt[1] for pt in grounded_pts])) if grounded_pts else 0.0

    def _support_priority(cid: int) -> Tuple[int, int, float]:
        cand = pool.candidates[cid]
        c_z0 = min(wb.zmin for wb in cand.world_boxes)
        bottom_mates = sum(
            1
            for gp in cand.consumed_ports
            if gp[0] in state._parts
            and max(bb.zmax for bb in state._parts[gp[0]].world_boxes) <= c_z0
        )
        cx = float(cand.pose.pos[0])
        cy = float(cand.pose.pos[1])
        dist2 = (cx - gx_mean) ** 2 + (cy - gy_mean) ** 2
        return (1 if bottom_mates > 0 else 0, bottom_mates, -dist2)

    remaining = sorted(cids, key=_support_priority, reverse=True)
    base_witness: Any = None
    if prior_peak_utilization is not None:
        base_eta = float(prior_peak_utilization)
    elif state.num_parts == 0:
        base_eta = 0.0
    else:
        eq0 = validator.evaluate(state)
        if not eq0.is_stable:
            return None
        base_eta = float(eq0.peak_utilization)
        base_witness = eq0.primal_witness

    plumb_deltas: Dict[Tuple[int, int], Tuple[float, float, float]] = {}
    ordered: List[int] = []
    tokens: List[TrialPartRollbackToken] = []
    try:
        while remaining:
            if len(remaining) == 1:
                # Final brick of `chosen_cids`: whole-course equilibrium was already verified!
                ordered.append(remaining.pop(0))
                break

            placed_step = False
            for idx_r, cid_try in enumerate(remaining):
                cand_t = pool.candidates[cid_try]
                cand_grounded = all(gp[0] == -1 for gp in cand_t.consumed_ports)
                try:
                    tok = state.push_trial_part(
                        cand_t.part_id, cand_t.pose, is_grounded=cand_grounded
                    )
                except ValueError:
                    continue
                if cand_grounded:
                    tokens.append(tok)
                    ordered.append(cid_try)
                    remaining.pop(idx_r)
                    placed_step = True
                    break

                snap_deltas = dict(plumb_deltas)
                can_ext, peak_d = validator.can_extend_plumb_primal_witness(
                    state, tok.inst_id, plumb_deltas, base_witness=base_witness
                )
                if can_ext and (base_eta + peak_d) <= 1.0 - 1e-6:
                    tokens.append(tok)
                    ordered.append(cid_try)
                    remaining.pop(idx_r)
                    placed_step = True
                    break

                plumb_deltas = snap_deltas
                eq_try = validator.evaluate(state)
                if eq_try.is_stable:
                    base_eta = float(eq_try.peak_utilization)
                    base_witness = eq_try.primal_witness
                    plumb_deltas.clear()
                    tokens.append(tok)
                    ordered.append(cid_try)
                    remaining.pop(idx_r)
                    placed_step = True
                    break

                state.pop_trial_part(tok)

            if not placed_step:
                return None
        return ordered
    finally:
        for tok in reversed(tokens):
            state.pop_trial_part(tok)

