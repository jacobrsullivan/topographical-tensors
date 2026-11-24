"""
Task-aware constrained repair implementation (`TopoBrickKernel.repair`, `Phase 15` extracted module).
"""

from __future__ import annotations

import copy
import time
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple
import numpy as np

from topo_brick.evaluation.static_equilibrium import FoundationPolicy
from topo_brick.interface.brick_catalog import (
    get_brick_primitive,
    is_known_part_id,
)
from topo_brick.interface.ldu_frames import LDUBox, LDUFrame
from topo_brick.interface.port_manifold import (
    AssemblyState,
    CandidatePlacement,
    PortPolarity,
)
from topo_brick.kernel.insertion_corridors import check_monotone_insertion_corridor
from topo_brick.kernel.kernel_contracts import (
    CertStatus,
    KernelRepairResult,
    KernelResultStatus,
    RepairAuditReport,
    RepairEditType,
    ValidatedConstraintSpec,
)
from topo_brick.representation.interface_fields import SparseTiledTargetField, TargetShapeBoxes


def execute_constrained_repair(
    self,
    assembly: AssemblyState,
    target_field: Optional[SparseTiledTargetField | TargetShapeBoxes] = None,
    allowed_part_ids: Optional[Sequence[str]] = None,
    allowed_edits: Optional[FrozenSet[str]] = None,
    locked_instance_ids: FrozenSet[int] = frozenset(),
    locked_world_boxes: Tuple[LDUBox, ...] = (),
    inventory_counts: Optional[Dict[str, int]] = None,
    colored_inventory: Optional[Any] = None,
    forbidden_support_boxes: Sequence[LDUBox] = (),
    support_policy: str = "none",
    min_retained_iou: Optional[float] = None,
    max_deleted_volume_ratio: Optional[float] = None,
    foundation_policy: FoundationPolicy | str = FoundationPolicy.CLUTCHED_BASEPLATE,
    time_budget_ms: float = 1200.0,
    max_constructive_steps: int = 1,
) -> KernelRepairResult:
    """
    Task-Aware Constrained Repair (`R8-WS2`, `R9-WS3` & `Phase 12 F3` contract):
    Threads `ValidatedConstraintSpec` (`inventory_counts`, `colored_inventory`,
    `forbidden_support_boxes`, `allowed_part_ids`, `foundation_policy`) uniformly across
    initial certification and Stages 1–3, computing `RepairAuditReport` booleans from
    verified state rather than literal `True`.
    """
    # Ensure lazy compiled catalog is loaded before starting `time_budget_ms` clock
    _ = is_known_part_id("3005")
    t0 = time.perf_counter()
    cspec = ValidatedConstraintSpec.from_args(
        allowed_part_ids=allowed_part_ids,
        inventory_counts=inventory_counts,
        colored_inventory=colored_inventory,
        foundation_policy=foundation_policy,
        support_policy=support_policy,
        forbidden_support_boxes=forbidden_support_boxes,
        min_retained_iou=min_retained_iou,
        max_deleted_volume_ratio=max_deleted_volume_ratio,
    )
    kit = cspec.allowed_part_ids if cspec.allowed_part_ids is not None else self.default_kit
    explicit_kit_set: Optional[Set[str]] = set(cspec.allowed_part_ids) if cspec.allowed_part_ids is not None else None
    fp_enum = cspec.foundation_policy

    orig_parts = list(assembly.placed_parts.values())
    orig_total_vol = max(sum(b.volume_ldu3 for p in orig_parts for b in p.world_boxes), 1)
    locked_ids_early: Set[int] = set(locked_instance_ids)
    if locked_world_boxes:
        for p in orig_parts:
            if any(pb.intersects(lb) for pb in p.world_boxes for lb in locked_world_boxes):
                locked_ids_early.add(p.instance_id)
    locked_part_poses: Set[Tuple[str, Any]] = {
        (p.part_id, p.pose) for p in orig_parts if p.instance_id in locked_ids_early
    }

    def _violates_forbidden_or_kit(st_chk: AssemblyState) -> bool:
        for pl_c in st_chk.placed_parts.values():
            if (
                explicit_kit_set is not None
                and pl_c.part_id not in explicit_kit_set
                and (pl_c.part_id, pl_c.pose) not in locked_part_poses
            ):
                return True
            if cspec.forbidden_support_boxes and any(
                wb.intersects(fb) for wb in pl_c.world_boxes for fb in cspec.forbidden_support_boxes
            ):
                return True
        return False

    init_certs = self.certify(
        assembly,
        target_field=target_field,
        verify_prefix=True,
        verify_insertion=True,
        foundation_policy=fp_enum,
        inventory_counts=cspec.inventory_counts,
        colored_inventory=cspec.colored_inventory,
    )
    init_iou = float(init_certs.geometry.target_iou)
    init_inv_ok = init_certs.inventory.status in (CertStatus.PASS, CertStatus.NOT_APPLICABLE)

    if init_certs.overall_certified and not _violates_forbidden_or_kit(assembly):
        audit_ok = RepairAuditReport(
            initial_iou=init_iou,
            repaired_iou=init_iou,
            retained_volume_ratio=1.0,
            deleted_volume_ratio=0.0,
            locked_features_preserved=True,
            inventory_respected=init_inv_ok,
            support_volume_added_ldu3=0,
            applied_edit_types=(),
        )
        return KernelRepairResult(
            status=KernelResultStatus.COMPLETED_CERTIFIED,
            repaired_assembly=copy.deepcopy(assembly),
            initial_certificates=init_certs,
            repaired_certificates=init_certs,
            parts_removed=0,
            parts_added=0,
            parts_reoriented=0,
            edit_volume_ldu3=0,
            wall_clock_ms=(time.perf_counter() - t0) * 1000.0,
            audit=audit_ok,
        )

    locked_ids: Set[int] = locked_ids_early

    eff_min_iou = (
        float(cspec.min_retained_iou)
        if cspec.min_retained_iou is not None
        else (0.85 * init_iou if target_field is not None else 0.0)
    )
    eff_total_inv: Optional[Dict[str, int]] = (
        dict(cspec.inventory_counts)
        if cspec.inventory_counts is not None
        else (
            cspec.colored_inventory.marginal_mold_counts()
            if cspec.colored_inventory is not None and hasattr(cspec.colored_inventory, "marginal_mold_counts")
            else None
        )
    )
    rem_inv: Optional[Dict[str, int]] = None
    if eff_total_inv is not None:
        used_orig: Dict[str, int] = {}
        for p_u in orig_parts:
            used_orig[p_u.part_id] = used_orig.get(p_u.part_id, 0) + 1
        rem_inv = {
            pid_k: max(0, int(lim_v) - used_orig.get(pid_k, 0))
            for pid_k, lim_v in eff_total_inv.items()
        }

    def _can_use_edit(edit_name: str) -> bool:
        return allowed_edits is None or edit_name in allowed_edits

    def _within_budget() -> bool:
        return ((time.perf_counter() - t0) * 1000.0) < float(time_budget_ms)

    stage1_cap_ms = (
        0.55 * float(time_budget_ms)
        if cspec.support_policy == "minimal_permanent_pillars"
        else 0.85 * float(time_budget_ms)
    )

    def _within_stage1_budget() -> bool:
        return ((time.perf_counter() - t0) * 1000.0) < stage1_cap_ms

    def _prove_repair_domain_infeasible() -> Optional[str]:
        """
        Return an analytical impossibility proof tag ONLY when NO assembly in the declared
        `(locked_ids, allowed_edits, inventory_counts, support_policy, eff_min_iou,
        max_deleted_volume_ratio)` domain can satisfy `overall_certified` (`R9-WS3` §5.6 + `F3`).
        """
        if len(orig_parts) > 0 and eff_total_inv is not None and sum(max(0, v) for v in eff_total_inv.values()) == 0:
            return "zero_total_inventory_budget"
        if eff_total_inv is not None and locked_ids:
            locked_mold_used: Dict[str, int] = {}
            for p_l in orig_parts:
                if p_l.instance_id in locked_ids:
                    locked_mold_used[p_l.part_id] = locked_mold_used.get(p_l.part_id, 0) + 1
            if any(cnt > int(eff_total_inv.get(pid_l, 0)) for pid_l, cnt in locked_mold_used.items()):
                return "locked_parts_exceed_inventory_budget"
        if cspec.forbidden_support_boxes and any(
            p_l.instance_id in locked_ids
            and any(wb.intersects(fb) for wb in p_l.world_boxes for fb in cspec.forbidden_support_boxes)
            for p_l in orig_parts
        ):
            return "locked_parts_intersect_forbidden_boxes"
        # 1. Mutual collision between two locked parts cannot be resolved by any allowed edit
        if any(
            u in locked_ids and v in locked_ids
            for u, v in init_certs.geometry.violating_pairs
        ):
            return "locked_parts_mutually_colliding"
        if (
            not init_certs.geometry.collision_free
            and cspec.max_deleted_volume_ratio is not None
            and float(cspec.max_deleted_volume_ratio) <= 0.0
            and not _can_use_edit(RepairEditType.REORIENT_PART.value)
        ):
            return "collision_present_with_zero_deletion_budget"

        can_add_cw = _can_use_edit(RepairEditType.ADD_COUNTERWEIGHT.value) or _can_use_edit(
            RepairEditType.REMERGE_SEAM.value
        )
        can_add_sup = (
            support_policy == "minimal_permanent_pillars"
            and _can_use_edit(RepairEditType.ADD_EXTERNAL_SUPPORT.value)
            and "3005" in kit
        )
        spare_inv_zero = rem_inv is not None and sum(max(0, v) for v in rem_inv.values()) == 0
        if (not can_add_cw and not can_add_sup) or spare_inv_zero:
            if not _can_use_edit(RepairEditType.DELETE_NONLOCKED_PART.value):
                return "all_constructive_and_pruning_edits_disallowed_or_zero_inventory"
            # Check if even the locked subset + grounded anchors cannot stand without constructive additions
            if locked_ids:
                grounded_or_locked = [p for p in orig_parts if p.is_grounded or p.instance_id in locked_ids]
                grounded_or_locked.sort(key=lambda p: (min(b.zmin for b in p.world_boxes), p.instance_id))
                st_gl = AssemblyState()
                gl_ok = True
                for p_gl in grounded_or_locked:
                    try:
                        st_gl.place_root_part(p_gl.part_id, p_gl.pose, is_grounded=p_gl.is_grounded)
                    except ValueError:
                        gl_ok = False
                        break
                if not gl_ok or any(d >= 999 for d in st_gl.compute_support_depths().values()) or not self.validator.evaluate(st_gl, foundation_policy=fp_enum).is_stable:
                    if not can_add_cw and not can_add_sup:
                        return "constructive_edits_disallowed_and_locked_subset_unstable"
                    if spare_inv_zero:
                        return "zero_spare_inventory_and_locked_subset_unstable"

        # 2. Global rigid-body ground moment equilibrium vs. target-IoU external volume bound
        if (
            locked_ids
            and support_policy == "none"
            and fp_enum != FoundationPolicy.FIXED_ANCHOR
            and target_field is not None
            and eff_min_iou > 1e-6
        ):
            grounded_parts = [p for p in orig_parts if p.is_grounded]
            if not grounded_parts:
                return "no_grounded_anchor_and_external_support_disallowed"
            gx0 = min(b.xmin for p in grounded_parts for b in p.world_boxes)
            gx1 = max(b.xmax for p in grounded_parts for b in p.world_boxes)
            gy0 = min(b.ymin for p in grounded_parts for b in p.world_boxes)
            gy1 = max(b.ymax for p in grounded_parts for b in p.world_boxes)
            n_g_studs = max(
                1,
                sum(
                    max(1, ((b.xmax - b.xmin) // 20) * ((b.ymax - b.ymin) // 20))
                    for p in grounded_parts
                    for b in p.world_boxes
                ),
            )
            f_cap = float(self.validator.clutch_force_capacity_grams)
            m_arm = float(self.validator.moment_arm_capacity_ldu)
            t_vol = float(target_field.total_volume_ldu3)
            t_bounds = target_field.occupied_bounds_ldu()
            if t_bounds is not None and t_vol > 0:
                tx0, ty0, _, tx1, ty1, _ = t_bounds
                v_ext_iou = max(0.0, t_vol * (1.0 / min(eff_min_iou, 1.0) - 1.0))
                if rem_inv is not None:
                    v_ext_inv = float(
                        sum(
                            max(0, cnt) * sum(b.volume_ldu3 for b in get_brick_primitive(pid).hull_boxes)
                            for pid, cnt in rem_inv.items()
                            if is_known_part_id(pid)
                        )
                    )
                    v_ext_max = min(v_ext_iou, v_ext_inv)
                else:
                    v_ext_max = v_ext_iou
                # Maximum catalog brick mass density (g per LDU^3): max(mass / vol) across `kit`
                max_density = max(
                    (
                        float(get_brick_primitive(pid).mass_grams)
                        / float(max(sum(b.volume_ldu3 for b in get_brick_primitive(pid).hull_boxes), 1))
                        for pid in kit
                        if is_known_part_id(pid)
                    ),
                    default=0.45 / 9600.0,
                )
                m_ext_max = v_ext_max * max_density
                max_part_span = max(
                    (
                        max(b.xmax - b.xmin, b.ymax - b.ymin)
                        for pid in kit
                        if is_known_part_id(pid)
                        for b in get_brick_primitive(pid).hull_boxes
                    ),
                    default=80,
                )
                min_part_vol = min(
                    (
                        sum(b.volume_ldu3 for b in get_brick_primitive(pid).hull_boxes)
                        for pid in kit
                        if is_known_part_id(pid)
                    ),
                    default=9600,
                )
                n_ext_bricks_ub = int(v_ext_max // max(min_part_vol, 1))
                reach_ub = float(n_ext_bricks_ub * max_part_span)
                tz0, tz1 = int(t_bounds[2]), int(t_bounds[5])

                locked_parts = [p for p in orig_parts if p.instance_id in locked_ids]
                # Exact Farkas dual inequality of the 6-DoF static LP around each support hull edge:
                # sum_{i in L} m_i max(0, coord_i - edge) <= N_g * f_cap * (m_arm + 0.5 * span_g)
                #   + M_restoring_in_target + M_restoring_external
                edge_specs = (
                    (
                        sum(
                            get_brick_primitive(p.part_id).mass_grams
                            * max(0.0, 0.5 * float(min(b.ymin for b in p.world_boxes) + max(b.ymax for b in p.world_boxes)) - float(gy1))
                            for p in locked_parts
                        ),
                        float(max(0, gy1 - ty0)),
                        float(gy1 - gy0),
                        float(max(0, tx1 - tx0) * max(0, gy1 - ty0) * max(0, tz1 - tz0)),
                    ),
                    (
                        sum(
                            get_brick_primitive(p.part_id).mass_grams
                            * max(0.0, float(gy0) - 0.5 * float(min(b.ymin for b in p.world_boxes) + max(b.ymax for b in p.world_boxes)))
                            for p in locked_parts
                        ),
                        float(max(0, ty1 - gy0)),
                        float(gy1 - gy0),
                        float(max(0, tx1 - tx0) * max(0, ty1 - gy0) * max(0, tz1 - tz0)),
                    ),
                    (
                        sum(
                            get_brick_primitive(p.part_id).mass_grams
                            * max(0.0, 0.5 * float(min(b.xmin for b in p.world_boxes) + max(b.xmax for b in p.world_boxes)) - float(gx1))
                            for p in locked_parts
                        ),
                        float(max(0, gx1 - tx0)),
                        float(gx1 - gx0),
                        float(max(0, gx1 - tx0) * max(0, ty1 - ty0) * max(0, tz1 - tz0)),
                    ),
                    (
                        sum(
                            get_brick_primitive(p.part_id).mass_grams
                            * max(0.0, float(gx0) - 0.5 * float(min(b.xmin for b in p.world_boxes) + max(b.xmax for b in p.world_boxes)))
                            for p in locked_parts
                        ),
                        float(max(0, tx1 - gx0)),
                        float(gx1 - gx0),
                        float(max(0, tx1 - gx0) * max(0, ty1 - ty0) * max(0, tz1 - tz0)),
                    ),
                )
                for m_over, in_arm_ub, span_g, v_rest_box in edge_specs:
                    m_ground_farkas = float(n_g_studs) * f_cap * (m_arm + 0.5 * span_g)
                    m_in_rest = min(t_vol, v_rest_box) * max_density * (0.5 * in_arm_ub)
                    m_ext_rest = m_ext_max * (in_arm_ub + reach_ub) if n_ext_bricks_ub > 0 else 0.0
                    if m_over > m_ground_farkas + m_in_rest + m_ext_rest + 1e-6:
                        return "farkas_dual_moment_equilibrium_vs_fidelity_domain_impossible"
        return None

    def _fallback_unsolved_status() -> Tuple[KernelResultStatus, Optional[str]]:
        proof = _prove_repair_domain_infeasible()
        if proof is not None:
            return KernelResultStatus.PROVEN_INFEASIBLE, proof
        if not _within_budget():
            return KernelResultStatus.PARTIAL_OR_TIMEOUT, None
        return KernelResultStatus.INCONCLUSIVE, None

    # ------------------------------------------------------------------
    # Stage 1: Constructive Tie-Plate / Counterweight Addition (0 deletions)
    # ------------------------------------------------------------------
    if (
        init_certs.geometry.collision_free
        and init_certs.geometry.protrusion_collision_free
        and (_can_use_edit(RepairEditType.ADD_COUNTERWEIGHT.value) or _can_use_edit(RepairEditType.REMERGE_SEAM.value))
        and _within_budget()
    ):
        max_c_steps = max(1, int(max_constructive_steps))
        grounded_xy = [
            (0.5 * float(b.xmin + b.xmax), 0.5 * float(b.ymin + b.ymax))
            for p in orig_parts
            if p.is_grounded
            for b in p.world_boxes
        ]
        all_com_x = float(np.mean([0.5 * (b.xmin + b.xmax) for p in orig_parts for b in p.world_boxes]))
        all_com_y = float(np.mean([0.5 * (b.ymin + b.ymax) for p in orig_parts for b in p.world_boxes]))
        gx_c = float(np.mean([pt[0] for pt in grounded_xy])) if grounded_xy else all_com_x
        gy_c = float(np.mean([pt[1] for pt in grounded_xy])) if grounded_xy else all_com_y
        # Vector pointing opposite the overhang direction (counterweight lever direction)
        cw_dir_x = gx_c - all_com_x
        cw_dir_y = gy_c - all_com_y

        beam_nodes: List[Tuple[float, AssemblyState, Optional[Dict[str, int]], int, int, Tuple[str, ...]]] = [
            (
                float(init_certs.static_mechanics.peak_utilization),
                copy.deepcopy(assembly),
                dict(rem_inv) if rem_inv is not None else None,
                0,
                0,
                (),
            )
        ]
        for _step_c in range(max_c_steps):
            if not _within_stage1_budget() or not beam_nodes:
                break
            next_beam: List[Tuple[float, AssemblyState, Optional[Dict[str, int]], int, int, Tuple[str, ...]]] = []
            for _, cur_st, cur_inv, cur_added, cur_vol, cur_tags in beam_nodes:
                if not _within_stage1_budget():
                    break
                cand_kit = tuple(
                    pid for pid in kit if (cur_inv is None or cur_inv.get(pid, 0) > 0)
                )
                if not cand_kit:
                    continue
                fr_cw = cur_st.get_active_frontier()
                pool_cw = cur_st.generate_candidate_pool(allowed_part_ids=cand_kit, frontier=fr_cw)
                scored_cw: List[Tuple[float, CandidatePlacement]] = []
                for c_cw in pool_cw.candidates:
                    vol_c = sum(b.volume_ldu3 for b in c_cw.world_boxes)
                    inter_c = (
                        float(target_field.intersection_volume(c_cw.world_boxes))
                        if target_field is not None
                        else 0.0
                    )
                    seam_b = 20000.0 if c_cw.closes_structural_seam else 0.0
                    cyc_b = 15000.0 if c_cw.creates_biconnected_cycle else 0.0
                    cx_p = 0.5 * float(min(b.xmin for b in c_cw.world_boxes) + max(b.xmax for b in c_cw.world_boxes))
                    cy_p = 0.5 * float(min(b.ymin for b in c_cw.world_boxes) + max(b.ymax for b in c_cw.world_boxes))
                    cw_align = (cx_p - gx_c) * cw_dir_x + (cy_p - gy_c) * cw_dir_y
                    cw_bonus = max(0.0, cw_align) * (float(vol_c) / 9600.0)
                    sc_cw = inter_c + seam_b + cyc_b + cw_bonus - 0.05 * float(vol_c)
                    scored_cw.append((sc_cw, c_cw))
                scored_cw.sort(key=lambda x: x[0], reverse=True)

                for _, c_cw in scored_cw[:16]:
                    if not _within_stage1_budget():
                        break
                    if cur_inv is not None and cur_inv.get(c_cw.part_id, 0) <= 0:
                        continue
                    if not check_monotone_insertion_corridor(cur_st, c_cw.part_id, c_cw.pose):
                        continue
                    t_next = copy.deepcopy(cur_st)
                    try:
                        t_next.commit_candidate(c_cw, is_grounded=False)
                    except ValueError:
                        continue
                    c_chk = self.certify(
                        t_next,
                        target_field=target_field,
                        verify_prefix=True,
                        verify_insertion=True,
                        foundation_policy=fp_enum,
                        inventory_counts=cspec.inventory_counts,
                        colored_inventory=cspec.colored_inventory,
                    )
                    if c_chk.static_mechanics.is_stable and not c_chk.overall_certified:
                        # Synthesize a prefix-stable, monotone-insertable assembly order (e.g. placing
                        # base + counterweights BEFORE overhanging cantilevers)!
                        orig_max_iid = max(assembly.placed_parts.keys(), default=-1)
                        ord_st = AssemblyState()
                        placed_old: Set[int] = set()
                        rem_ord = sorted(
                            t_next.placed_parts.values(),
                            key=lambda p_o: (
                                0 if p_o.is_grounded else (1 if p_o.instance_id > orig_max_iid else 2),
                                min(b.zmin for b in p_o.world_boxes),
                                p_o.instance_id,
                            ),
                        )
                        prog_ord = True
                        while rem_ord and prog_ord:
                            prog_ord = False
                            nxt_rem_ord = []
                            for idx_r, p_r in enumerate(rem_ord):
                                conn_r = bool(
                                    p_r.is_grounded
                                    or (t_next._part_adj.get(p_r.instance_id, set()) & placed_old)
                                )
                                if not conn_r or not check_monotone_insertion_corridor(
                                    ord_st, p_r.part_id, p_r.pose
                                ):
                                    nxt_rem_ord.append(p_r)
                                    continue
                                tr_ord = copy.deepcopy(ord_st)
                                try:
                                    tr_ord.place_root_part(
                                        p_r.part_id,
                                        p_r.pose,
                                        is_grounded=p_r.is_grounded,
                                        color_id=p_r.color_id,
                                    )
                                except ValueError:
                                    nxt_rem_ord.append(p_r)
                                    continue
                                if self.validator.evaluate(tr_ord, foundation_policy=fp_enum).is_stable:
                                    ord_st = tr_ord
                                    placed_old.add(p_r.instance_id)
                                    prog_ord = True
                                    nxt_rem_ord.extend(rem_ord[idx_r + 1 :])
                                    break
                                nxt_rem_ord.append(p_r)
                            rem_ord = nxt_rem_ord
                        if not rem_ord and ord_st.num_parts == t_next.num_parts:
                            c_ord = self.certify(
                                ord_st,
                                target_field=target_field,
                                verify_prefix=True,
                                verify_insertion=True,
                                foundation_policy=fp_enum,
                                inventory_counts=cspec.inventory_counts,
                                colored_inventory=cspec.colored_inventory,
                            )
                            if c_ord.overall_certified:
                                t_next = ord_st
                                c_chk = c_ord
                    new_added = cur_added + 1
                    new_vol = cur_vol + sum(b.volume_ldu3 for b in c_cw.world_boxes)
                    edit_tag = (
                        RepairEditType.REMERGE_SEAM.value
                        if c_cw.closes_structural_seam
                        else RepairEditType.ADD_COUNTERWEIGHT.value
                    )
                    new_tags = tuple(dict.fromkeys(cur_tags + (edit_tag,)))
                    if (
                        c_chk.overall_certified
                        and not _violates_forbidden_or_kit(t_next)
                        and c_chk.geometry.target_iou >= eff_min_iou - 1e-6
                    ):
                        audit_cw = RepairAuditReport(
                            initial_iou=init_iou,
                            repaired_iou=float(c_chk.geometry.target_iou),
                            retained_volume_ratio=1.0,
                            deleted_volume_ratio=0.0,
                            locked_features_preserved=True,
                            inventory_respected=(
                                c_chk.inventory.status in (CertStatus.PASS, CertStatus.NOT_APPLICABLE)
                            ),
                            support_volume_added_ldu3=new_vol,
                            applied_edit_types=new_tags,
                        )
                        return KernelRepairResult(
                            status=KernelResultStatus.COMPLETED_CERTIFIED,
                            repaired_assembly=t_next,
                            initial_certificates=init_certs,
                            repaired_certificates=c_chk,
                            parts_removed=0,
                            parts_added=new_added,
                            parts_reoriented=0,
                            edit_volume_ldu3=new_vol,
                            wall_clock_ms=(time.perf_counter() - t0) * 1000.0,
                            audit=audit_cw,
                        )
                    if _step_c + 1 < max_c_steps and c_chk.geometry.target_iou >= eff_min_iou - 1e-6:
                        nxt_inv = dict(cur_inv) if cur_inv is not None else None
                        if nxt_inv is not None:
                            nxt_inv[c_cw.part_id] = max(0, nxt_inv.get(c_cw.part_id, 0) - 1)
                        next_beam.append(
                            (
                                float(c_chk.static_mechanics.peak_utilization),
                                t_next,
                                nxt_inv,
                                new_added,
                                new_vol,
                                new_tags,
                            )
                        )
            next_beam.sort(key=lambda item: (item[0], item[4]))
            beam_nodes = next_beam[:4]

    # ------------------------------------------------------------------
    # Stage 2: Minimal Permanent Support Pillars (`ADD_EXTERNAL_SUPPORT`, 0 deletions)
    # ------------------------------------------------------------------
    if (
        support_policy == "minimal_permanent_pillars"
        and _can_use_edit(RepairEditType.ADD_EXTERNAL_SUPPORT.value)
        and "3005" in kit
        and init_certs.geometry.collision_free
        and init_certs.geometry.protrusion_collision_free
        and _within_budget()
    ):
        trial_sup = copy.deepcopy(assembly)
        min_z_s = min(min(b.zmin for b in p.world_boxes) for p in orig_parts)
        sup_added = 0
        sup_vol = 0
        rem_sup_inv = dict(rem_inv) if rem_inv is not None else None
        for _sr in range(10):
            if not _within_budget():
                break
            # Canonicalize bottom-up order and check BOTH final static AND 1-by-1 prefix stability!
            ordered_sup = AssemblyState()
            rem_p = sorted(
                trial_sup.placed_parts.values(),
                key=lambda p: (min(b.zmin for b in p.world_boxes), 0 if p.is_grounded else 1, p.instance_id),
            )
            for p_s in rem_p:
                try:
                    ordered_sup.place_root_part(
                        p_s.part_id,
                        p_s.pose,
                        is_grounded=p_s.is_grounded,
                        color_id=p_s.color_id,
                    )
                except ValueError:
                    break
            c_sup_iter = self.certify(
                ordered_sup,
                target_field=target_field,
                verify_prefix=True,
                verify_insertion=True,
                foundation_policy=fp_enum,
                inventory_counts=cspec.inventory_counts,
                colored_inventory=cspec.colored_inventory,
            )
            if (
                sup_added > 0
                and ordered_sup.num_parts == trial_sup.num_parts
                and c_sup_iter.overall_certified
                and not _violates_forbidden_or_kit(ordered_sup)
                and c_sup_iter.geometry.target_iou >= eff_min_iou - 1e-6
            ):
                audit_sup = RepairAuditReport(
                    initial_iou=init_iou,
                    repaired_iou=float(c_sup_iter.geometry.target_iou),
                    retained_volume_ratio=1.0,
                    deleted_volume_ratio=0.0,
                    locked_features_preserved=True,
                    inventory_respected=(
                        c_sup_iter.inventory.status in (CertStatus.PASS, CertStatus.NOT_APPLICABLE)
                    ),
                    support_volume_added_ldu3=sup_vol,
                    applied_edit_types=(RepairEditType.ADD_EXTERNAL_SUPPORT.value,),
                )
                return KernelRepairResult(
                    status=KernelResultStatus.COMPLETED_CERTIFIED,
                    repaired_assembly=ordered_sup,
                    initial_certificates=init_certs,
                    repaired_certificates=c_sup_iter,
                    parts_removed=0,
                    parts_added=sup_added,
                    parts_reoriented=0,
                    edit_volume_ldu3=sup_vol,
                    wall_clock_ms=(time.perf_counter() - t0) * 1000.0,
                    audit=audit_sup,
                )

            # Prioritize open anti-studs on the first unstable prefix step if prefix failed!
            prio_z: Optional[int] = None
            f_step = c_sup_iter.prefix_mechanics.first_unstable_step
            if f_step is not None and f_step < len(rem_p):
                prio_z = min(b.zmin for b in rem_p[f_step].world_boxes)

            open_antis: List[Tuple[int, int, int]] = []
            for (px, py, pz), entries in trial_sup._ports_by_pos.items():
                if pz <= min_z_s:
                    continue
                if any(
                    gp not in trial_sup._matched_ports and pol == PortPolarity.ANTI_STUD
                    for gp, _, pol, _ in entries
                ):
                    open_antis.append((px - 10, py - 10, pz))
            open_antis.sort(
                key=lambda xyz: (
                    1 if (prio_z is not None and xyz[2] == prio_z) else 0,
                    xyz[2],
                    abs(xyz[0]) + abs(xyz[1]),
                ),
                reverse=True,
            )
            placed_col = False
            for x0_s, y0_s, z_top_s in open_antis:
                col_boxes = [LDUBox(x0_s, y0_s, zp, x0_s + 20, y0_s + 20, zp + 24) for zp in range(min_z_s, z_top_s, 24)]
                if cspec.forbidden_support_boxes and any(
                    cb_f.intersects(fb) for cb_f in col_boxes for fb in cspec.forbidden_support_boxes
                ):
                    continue
                if all(not trial_sup.check_boxes_collision((cb_f,)) for cb_f in col_boxes):
                    n_need = (z_top_s - min_z_s) // 24
                    if rem_sup_inv is not None and rem_sup_inv.get("3005", 0) < n_need:
                        continue
                    for zp in range(min_z_s, z_top_s, 24):
                        trial_sup.place_root_part(
                            "3005",
                            LDUFrame(0, (x0_s + 10, y0_s + 10, zp)),
                            is_grounded=(zp <= min_z_s),
                        )
                        sup_added += 1
                        sup_vol += 9600
                        if rem_sup_inv is not None:
                            rem_sup_inv["3005"] -= 1
                    placed_col = True
                    break
            if not placed_col:
                break

    # ------------------------------------------------------------------
    # Stage 3: Constrained Non-Locked Pruning + Re-Fill (`DELETE_NONLOCKED_PART`)
    # ------------------------------------------------------------------
    if not _can_use_edit(RepairEditType.DELETE_NONLOCKED_PART.value):
        fb_status, proof_tag = _fallback_unsolved_status()
        audit_dis = RepairAuditReport(
            initial_iou=init_iou,
            repaired_iou=init_iou,
            retained_volume_ratio=1.0,
            deleted_volume_ratio=0.0,
            locked_features_preserved=True,
            inventory_respected=init_inv_ok,
            support_volume_added_ldu3=0,
            applied_edit_types=(),
            rejection_reason=(
                proof_tag
                or "delete_nonlocked_part_disallowed_and_bounded_constructive_search_exhausted"
            ),
        )
        return KernelRepairResult(
            status=fb_status,
            repaired_assembly=copy.deepcopy(assembly),
            initial_certificates=init_certs,
            repaired_certificates=init_certs,
            parts_removed=0,
            parts_added=0,
            parts_reoriented=0,
            edit_volume_ldu3=0,
            wall_clock_ms=(time.perf_counter() - t0) * 1000.0,
            audit=audit_dis,
        )

    orig_parts.sort(key=lambda p: (min(b.zmin for b in p.world_boxes), p.instance_id))
    repaired = AssemblyState()
    deferred_parts: List[Any] = []
    removed_vol = 0
    parts_removed = 0
    locked_failed = False
    retained_mold_used: Dict[str, int] = {}
    retained_color_used: Dict[Tuple[str, int], int] = {}

    for p in orig_parts:
        trial = copy.deepcopy(repaired)
        try:
            if (
                explicit_kit_set is not None
                and p.part_id not in explicit_kit_set
                and p.instance_id not in locked_ids
            ):
                raise ValueError("Part outside allowed_part_ids")
            if cspec.forbidden_support_boxes and any(
                wb.intersects(fb) for wb in p.world_boxes for fb in cspec.forbidden_support_boxes
            ):
                raise ValueError("Intersects forbidden_support_boxes")
            if eff_total_inv is not None and retained_mold_used.get(p.part_id, 0) + 1 > int(eff_total_inv.get(p.part_id, 0)):
                raise ValueError("Exceeds inventory_counts")
            if cspec.colored_inventory is not None and hasattr(cspec.colored_inventory, "get_count"):
                ck_p = (p.part_id, int(p.color_id))
                if retained_color_used.get(ck_p, 0) + 1 > int(cspec.colored_inventory.get_count(p.part_id, int(p.color_id))):
                    raise ValueError("Exceeds colored_inventory")
            trial.place_root_part(
                p.part_id, p.pose, is_grounded=p.is_grounded, color_id=p.color_id
            )
            depths = trial.compute_support_depths()
            if any(d >= 999 for d in depths.values()):
                raise ValueError("Floating part")
            if not self.validator.evaluate(trial, foundation_policy=fp_enum).is_stable:
                raise ValueError("Unstable")
            repaired = trial
            retained_mold_used[p.part_id] = retained_mold_used.get(p.part_id, 0) + 1
            ck_ok = (p.part_id, int(p.color_id))
            retained_color_used[ck_ok] = retained_color_used.get(ck_ok, 0) + 1
        except ValueError:
            if p.instance_id in locked_ids:
                locked_failed = True
                break
            deferred_parts.append(p)
            parts_removed += 1
            removed_vol += sum(b.volume_ldu3 for b in p.world_boxes)

    if locked_failed:
        fb_status, proof_tag = _fallback_unsolved_status()
        audit_lf = RepairAuditReport(
            initial_iou=init_iou,
            repaired_iou=init_iou,
            retained_volume_ratio=1.0,
            deleted_volume_ratio=0.0,
            locked_features_preserved=False,
            inventory_respected=init_inv_ok,
            support_volume_added_ldu3=0,
            applied_edit_types=(),
            rejection_reason=(
                proof_tag
                or "bounded_repair_search_exhausted_without_deleting_locked_feature"
            ),
        )
        return KernelRepairResult(
            status=fb_status,
            repaired_assembly=copy.deepcopy(assembly),
            initial_certificates=init_certs,
            repaired_certificates=init_certs,
            parts_removed=0,
            parts_added=0,
            parts_reoriented=0,
            edit_volume_ldu3=0,
            wall_clock_ms=(time.perf_counter() - t0) * 1000.0,
            audit=audit_lf,
        )

    stage3_rem_inv: Optional[Dict[str, int]] = (
        {
            pid_k: max(0, int(lim_v) - retained_mold_used.get(pid_k, 0))
            for pid_k, lim_v in eff_total_inv.items()
        }
        if eff_total_inv is not None
        else None
    )
    parts_added = 0
    added_vol = 0
    if (target_field is not None or deferred_parts) and repaired.num_parts > 0 and _within_budget():
        effective_target = (
            target_field
            if target_field is not None
            else TargetShapeBoxes(boxes=tuple(b for dp in deferred_parts for b in dp.world_boxes))
        )
        frontier = repaired.get_active_frontier()
        pool = repaired.generate_candidate_pool(allowed_part_ids=kit, frontier=frontier)
        sel_indices, _ = self.s1_solver.solve_course(
            repaired,
            pool,
            effective_target,
            max_bricks=max(4, len(deferred_parts)),
            inventory_counts=stage3_rem_inv,
        )
        for idx in sel_indices:
            cand = pool.candidates[idx]
            if stage3_rem_inv is not None and stage3_rem_inv.get(cand.part_id, 0) <= 0:
                continue
            if cspec.forbidden_support_boxes and any(
                wb.intersects(fb) for wb in cand.world_boxes for fb in cspec.forbidden_support_boxes
            ):
                continue
            trial = copy.deepcopy(repaired)
            try:
                trial.commit_candidate(cand, is_grounded=False)
                if self.validator.evaluate(trial, foundation_policy=fp_enum).is_stable:
                    repaired = trial
                    parts_added += 1
                    added_vol += sum(b.volume_ldu3 for b in cand.world_boxes)
                    if stage3_rem_inv is not None:
                        stage3_rem_inv[cand.part_id] -= 1
            except ValueError:
                continue

    if len(orig_parts) > 0 and repaired.num_parts == 0:
        fb_status, proof_tag = _fallback_unsolved_status()
        audit_empty = RepairAuditReport(
            initial_iou=init_iou,
            repaired_iou=0.0,
            retained_volume_ratio=0.0,
            deleted_volume_ratio=1.0,
            locked_features_preserved=(len(locked_ids) == 0),
            inventory_respected=init_inv_ok,
            support_volume_added_ldu3=0,
            applied_edit_types=(RepairEditType.DELETE_NONLOCKED_PART.value,),
            rejection_reason=proof_tag or "pruning_eliminated_all_parts_in_assembly",
        )
        return KernelRepairResult(
            status=fb_status,
            repaired_assembly=copy.deepcopy(assembly),
            initial_certificates=init_certs,
            repaired_certificates=init_certs,
            parts_removed=0,
            parts_added=0,
            parts_reoriented=0,
            edit_volume_ldu3=0,
            wall_clock_ms=(time.perf_counter() - t0) * 1000.0,
            audit=audit_empty,
        )

    rep_certs = self.certify(
        repaired,
        target_field=target_field,
        verify_prefix=True,
        verify_insertion=True,
        foundation_policy=fp_enum,
        inventory_counts=cspec.inventory_counts,
        colored_inventory=cspec.colored_inventory,
    )
    rep_inv_ok = rep_certs.inventory.status in (CertStatus.PASS, CertStatus.NOT_APPLICABLE)
    del_ratio = float(removed_vol) / float(orig_total_vol)
    ret_ratio = max(0.0, 1.0 - del_ratio)
    fidelity_ok = float(rep_certs.geometry.target_iou) >= eff_min_iou - 1e-6
    del_budget_ok = (
        cspec.max_deleted_volume_ratio is None or del_ratio <= float(cspec.max_deleted_volume_ratio) + 1e-6
    )
    if not fidelity_ok or not del_budget_ok:
        fb_status, proof_tag = _fallback_unsolved_status()
        audit_rej = RepairAuditReport(
            initial_iou=init_iou,
            repaired_iou=float(rep_certs.geometry.target_iou),
            retained_volume_ratio=ret_ratio,
            deleted_volume_ratio=del_ratio,
            locked_features_preserved=True,
            inventory_respected=rep_inv_ok,
            support_volume_added_ldu3=0,
            applied_edit_types=(RepairEditType.DELETE_NONLOCKED_PART.value,),
            rejection_reason=(
                proof_tag
                or "bounded_pruning_exceeded_fidelity_or_deletion_budget"
            ),
        )
        return KernelRepairResult(
            status=fb_status,
            repaired_assembly=repaired,
            initial_certificates=init_certs,
            repaired_certificates=rep_certs,
            parts_removed=parts_removed,
            parts_added=parts_added,
            parts_reoriented=0,
            edit_volume_ldu3=removed_vol + added_vol,
            wall_clock_ms=(time.perf_counter() - t0) * 1000.0,
            audit=audit_rej,
        )

    if rep_certs.overall_certified and not _violates_forbidden_or_kit(repaired):
        status = KernelResultStatus.COMPLETED_CERTIFIED
    else:
        status, _ = _fallback_unsolved_status()
    applied_ops: List[str] = []
    if parts_removed > 0:
        applied_ops.append(RepairEditType.DELETE_NONLOCKED_PART.value)
    if parts_added > 0:
        applied_ops.append(RepairEditType.REMERGE_SEAM.value)
    audit_final = RepairAuditReport(
        initial_iou=init_iou,
        repaired_iou=float(rep_certs.geometry.target_iou),
        retained_volume_ratio=ret_ratio,
        deleted_volume_ratio=del_ratio,
        locked_features_preserved=True,
        inventory_respected=rep_inv_ok,
        support_volume_added_ldu3=0,
        applied_edit_types=tuple(applied_ops),
    )
    return KernelRepairResult(
        status=status,
        repaired_assembly=repaired,
        initial_certificates=init_certs,
        repaired_certificates=rep_certs,
        parts_removed=parts_removed,
        parts_added=parts_added,
        parts_reoriented=0,
        edit_volume_ldu3=removed_vol + added_vol,
        wall_clock_ms=(time.perf_counter() - t0) * 1000.0,
        audit=audit_final,
    )
