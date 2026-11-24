"""
Deterministic course-packing solvers (`S1` `CoverAndRepairSolver`, `S2` `PairSeededLocalSearch`),
experimental batch samplers (`S3`, `S4`), and internal layer-merge/seam-refinement controls (`Phase 15`).
"""

from __future__ import annotations

import time
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix

from topo_brick.evaluation.static_equilibrium import (
    EquilibriumResult,
    FoundationPolicy,
    SparseStaticEquilibriumValidator,
)
from topo_brick.interface.brick_catalog import (
    STABLETEXT2BRICK_8PART_KIT,
    ConnectorType,
    get_brick_primitive,
    is_known_part_id,
)
from topo_brick.interface.ldu_frames import C4_Z_ROT_IDS, LDUBox, LDUFrame
from topo_brick.interface.port_manifold import (
    AssemblyState,
    CandidatePlacement,
    CandidatePool,
    GlobalPortId,
    SpatialBoxHashIndex,
    TrialPartRollbackToken,
)
from topo_brick.kernel.insertion_corridors import _order_course_atomic_1by1
from topo_brick.representation.interface_fields import SparseTiledTargetField, TargetShapeBoxes


# ---------------------------------------------------------------------------
# 4. Solvers: S1 (CoverAndRepairSolver Default), S2, S3, S4, and Legolization
# ---------------------------------------------------------------------------


_SHARED_HIGHS_MIP: Any = None


def _solve_binary_mip_fast(
    c_obj: np.ndarray,
    a_mat: Optional[csc_matrix],
    lbs: Sequence[float],
    ubs: Sequence[float],
    time_limit_s: float = 1.0,
    mip_rel_gap: float = 0.01,
) -> Optional[np.ndarray]:
    """
    Solve binary set-packing MIP `min c_obj^T x, x in {0, 1}^K, lbs <= A x <= ubs` on 1 CPU core
    using native `highspy` (`threads=1`, `mip_max_leaves=64`, `random_seed=0`) with deterministic
    leaf budget so results are clock-speed-invariant across CPU hosts (`R9-WS1`).
    """
    global _SHARED_HIGHS_MIP
    k_vars = int(c_obj.shape[0])
    if k_vars == 0:
        return None
    try:
        import highspy

        if _SHARED_HIGHS_MIP is None:
            h = highspy.Highs()
            h.silent()
            h.setOptionValue("threads", 1)
            _SHARED_HIGHS_MIP = h
        else:
            h = _SHARED_HIGHS_MIP
            h.clearSolver()
        h.setOptionValue("threads", 1)
        h.setOptionValue("random_seed", 0)
        h.setOptionValue("time_limit", float(time_limit_s))
        h.setOptionValue("mip_rel_gap", float(mip_rel_gap))
        h.setOptionValue("mip_max_leaves", 64)
        col_lower = np.zeros(k_vars, dtype=np.float64)
        col_upper = np.ones(k_vars, dtype=np.float64)
        if a_mat is not None and a_mat.shape[0] > 0:
            a_csc = a_mat.tocsc()
            inf = float(highspy.kHighsInf)
            r_lb = np.array([-inf if np.isneginf(v) else float(v) for v in lbs], dtype=np.float64)
            r_ub = np.array([inf if np.isposinf(v) else float(v) for v in ubs], dtype=np.float64)
            lp = highspy.HighsLp()
            lp.num_col_ = k_vars
            lp.num_row_ = int(a_csc.shape[0])
            lp.col_cost_ = np.ascontiguousarray(c_obj, dtype=np.float64)
            lp.col_lower_ = col_lower
            lp.col_upper_ = col_upper
            lp.row_lower_ = r_lb
            lp.row_upper_ = r_ub
            lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
            lp.a_matrix_.start_ = np.ascontiguousarray(a_csc.indptr, dtype=np.int32)
            lp.a_matrix_.index_ = np.ascontiguousarray(a_csc.indices, dtype=np.int32)
            lp.a_matrix_.value_ = np.ascontiguousarray(a_csc.data, dtype=np.float64)
            lp.integrality_ = [highspy.HighsVarType.kInteger] * k_vars
            h.passModel(lp)
        else:
            return (c_obj < 0.0).astype(np.float64)
        h.run()
        info = h.getInfo()
        if getattr(info, "primal_solution_status", 0) == int(highspy.SolutionStatus.kSolutionStatusFeasible):
            sol = np.asarray(h.getSolution().col_value, dtype=np.float64)
            if sol.shape[0] == k_vars:
                return sol
        return None
    except Exception:
        constraints = (
            [LinearConstraint(a_mat, np.array(lbs), np.array(ubs))]
            if a_mat is not None and a_mat.shape[0] > 0
            else []
        )
        res = milp(
            c=c_obj,
            integrality=np.ones(k_vars, dtype=np.int64),
            bounds=Bounds(np.zeros(k_vars), np.ones(k_vars)),
            constraints=constraints,
            options={"time_limit": float(time_limit_s), "mip_rel_gap": float(mip_rel_gap)},
        )
        return res.x if (res.success and res.x is not None) else None


def build_canonical_nogood_cut(
    num_vars: int,
    cand_col_offset: int,
    selected_indices: Sequence[int],
    num_candidates: int,
) -> Tuple[np.ndarray, float, float]:
    """
    Round 7 Critique 5C Fix:
    Construct the exact canonical binary no-good cut excluding ONLY the specific binary
    assignment x^(r) in {0, 1}^M:
        sum_{i in S} (1 - x_i) + sum_{i not in S} x_i >= 1
    which rearranges linearly to:
        sum_{i in S} x_i - sum_{i not in S} x_i <= |S| - 1.
    Unlike `sum_{i in S} (1 - x_i) >= 1`, this canonical cut NEVER cuts off strict supersets
    S' supset S (such as adding a counterbalancing brick that restores static equilibrium!).
    """
    row = np.zeros(num_vars, dtype=np.float64)
    sel_set = set(selected_indices)
    for cid in range(num_candidates):
        if cid in sel_set:
            row[cand_col_offset + cid] = 1.0
        else:
            row[cand_col_offset + cid] = -1.0
    rhs_ub = float(len(sel_set) - 1)
    return row, -np.inf, rhs_ub


class CoverAndRepairSolver:
    """
    `S1` Provisional Default Solver (`CoverAndRepairSolver`, Zero Learned Parameters).

    Solves course-level or subproblem cell-cover MILP over `CandidatePool` with:
    1. Exact port-capacity and hull/protrusion conflict constraints.
    2. Explicit target cell coverage indicators + structural seam-bridging and cycle bonuses.
    3. Conservative canonical binary no-good cuts:
           sum_{i: x_i^(r)=1} x_i - sum_{i: x_i^(r)=0} x_i <= |x^(r)| - 1
       when a candidate course violates 6-DoF static equilibrium.
    4. Declared lexicographic bounded-subtree fallback repair (`conservative_nogood_cut_and_lexicographic_repair`).
    """

    solver_id: str = "S1_CoverAndRepair"

    def __init__(
        self,
        validator: Optional[SparseStaticEquilibriumValidator] = None,
        max_nogood_rounds: int = 8,
        seam_bonus: float = 0.25,
        cycle_bonus: float = 0.20,
        part_count_regularization: float = 0.55,
    ) -> None:
        self.validator = validator or SparseStaticEquilibriumValidator(
            clutch_force_capacity_grams=99.9
        )
        self.max_nogood_rounds = int(max_nogood_rounds)
        self.seam_bonus = float(seam_bonus)
        self.cycle_bonus = float(cycle_bonus)
        self.part_count_regularization = float(part_count_regularization)

    def solve_course(
        self,
        state: AssemblyState,
        pool: CandidatePool,
        target_field: SparseTiledTargetField | TargetShapeBoxes,
        max_bricks: Optional[int] = None,
        inventory_counts: Optional[Dict[str, int]] = None,
        candidate_bonus_scores: Optional[np.ndarray] = None,
        pairwise_synergy_matrix: Optional[np.ndarray] = None,
        require_atomic_1by1_order: bool = True,
        prior_peak_utilization: Optional[float] = None,
        foundation_policy: FoundationPolicy | str = FoundationPolicy.CLUTCHED_BASEPLATE,
        ground_plane_z_ldu: float = 0.0,
        deadline_monotonic_s: Optional[float] = None,
    ) -> Tuple[List[int], int]:
        """
        Select a compatible, static-equilibrium-certified course `selected_indices` from `pool`.
        Returns `(selected_indices, nogood_cuts_added)`.
        """
        if deadline_monotonic_s is not None and time.monotonic() >= deadline_monotonic_s:
            return [], 0
        m = len(pool)
        if m == 0:
            return [], 0

        # Filter to candidates that have high target precision (>= 0.95)
        valid_cids: List[int] = []
        cand_rewards = np.full(m, -100.0, dtype=np.float64)
        reuse_inter = pool.filtered_target_id != 0 and pool.filtered_target_id == id(target_field)
        for cid, cand in enumerate(pool.candidates):
            vol = float(sum(b.volume_ldu3 for b in cand.world_boxes))
            if reuse_inter and cand.target_intersection_ldu3 >= 0:
                inter = float(cand.target_intersection_ldu3)
            else:
                inter = float(target_field.intersection_volume(cand.world_boxes))
            prec = inter / max(vol, 1.0)
            if prec < 0.95:
                continue
            if inventory_counts is not None and inventory_counts.get(cand.part_id, 0) <= 0:
                continue
            valid_cids.append(cid)
            # Reward proportional to target volume covered (in 1x1 brick units = 9600 LDU^3)
            # minus small per-part regularization to prefer fewer, larger interlocking bricks
            reward = (inter / 9600.0) - self.part_count_regularization
            if cand.closes_structural_seam:
                reward += self.seam_bonus
            if cand.creates_biconnected_cycle:
                reward += self.cycle_bonus
            if candidate_bonus_scores is not None and cid < len(candidate_bonus_scores):
                reward += float(candidate_bonus_scores[cid])
            # Deterministic canonical tie-breaker across MIP solver versions
            reward += 1e-6 * (float(cid) / float(max(m, 1)))
            cand_rewards[cid] = reward

        if not valid_cids:
            return [], 0

        k_valid = len(valid_cids)
        # Build sparse linear constraints directly in COO triplets + cell-clique tightening
        coo_rows: List[int] = []
        coo_cols: List[int] = []
        coo_vals: List[float] = []
        lbs: List[float] = []
        ubs: List[float] = []
        n_rows = 0

        valid_pos_map = {cid: idx for idx, cid in enumerate(valid_cids)}

        # 1. Port capacity constraints: sum_{i ni u} z_i <= 1
        for _, c_list in pool.port_to_candidates.items():
            rel = [valid_pos_map[c] for c in c_list if c in valid_pos_map]
            if len(rel) >= 2:
                for col_i in rel:
                    coo_rows.append(n_rows)
                    coo_cols.append(col_i)
                    coo_vals.append(1.0)
                lbs.append(-np.inf)
                ubs.append(1.0)
                n_rows += 1

        # 2. Spatial 20x20x24 cell cliques over candidate hull boxes (tightens MILP LP relaxation
        #    and replaces O(K^2) overhanging pairwise conflict rows with a single clique row!)
        cell_to_local: Dict[Tuple[int, int, int], List[int]] = {}
        cand_cells: List[FrozenSet[Tuple[int, int, int]]] = []
        for idx, cid in enumerate(valid_cids):
            cand_v = pool.candidates[cid]
            seen_cells: Set[Tuple[int, int, int]] = set()
            for wb in cand_v.world_boxes:
                if (wb.xmax - wb.xmin) >= 20 and (wb.ymax - wb.ymin) >= 20 and (wb.zmax - wb.zmin) >= 8:
                    for cx in range((wb.xmin - 10) // 20, (wb.xmax + 10) // 20 + 1):
                        for cy in range((wb.ymin - 10) // 20, (wb.ymax + 10) // 20 + 1):
                            for cz in range(wb.zmin // 24, (wb.zmax - 1) // 24 + 1):
                                # Strict interior sample points prevent false cliques on flush boundaries
                                x_s1, y_s1 = cx * 20 + 10, cy * 20 + 10
                                if wb.xmin < x_s1 < wb.xmax and wb.ymin < y_s1 < wb.ymax:
                                    seen_cells.add((cx * 2 + 1, cy * 2 + 1, cz))
                                x_s0, y_s0 = cx * 20, cy * 20
                                if wb.xmin < x_s0 < wb.xmax and wb.ymin < y_s0 < wb.ymax:
                                    seen_cells.add((cx * 2, cy * 2, cz))
            cand_cells.append(frozenset(seen_cells))
            for sc in seen_cells:
                cell_to_local.setdefault(sc, []).append(idx)

        seen_clique_sigs: Set[Tuple[int, ...]] = set()
        for c_members in cell_to_local.values():
            if len(c_members) >= 2:
                sig = tuple(c_members)
                if sig in seen_clique_sigs:
                    continue
                seen_clique_sigs.add(sig)
                for col_i in c_members:
                    coo_rows.append(n_rows)
                    coo_cols.append(col_i)
                    coo_vals.append(1.0)
                lbs.append(-np.inf)
                ubs.append(1.0)
                n_rows += 1

        # 2b. Any remaining sub-cell / protrusion disjoint-port conflict edges: z_i + z_j <= 1
        for cid_a, cid_b in pool.conflict_edges:
            if cid_a in valid_pos_map and cid_b in valid_pos_map:
                la, lb = valid_pos_map[cid_a], valid_pos_map[cid_b]
                if not cand_cells[la].isdisjoint(cand_cells[lb]):
                    continue
                coo_rows.append(n_rows)
                coo_cols.append(la)
                coo_vals.append(1.0)
                coo_rows.append(n_rows)
                coo_cols.append(lb)
                coo_vals.append(1.0)
                lbs.append(-np.inf)
                ubs.append(1.0)
                n_rows += 1

        # 3. Optional max_bricks constraint
        if max_bricks is not None:
            for col_i in range(k_valid):
                coo_rows.append(n_rows)
                coo_cols.append(col_i)
                coo_vals.append(1.0)
            lbs.append(-np.inf)
            ubs.append(float(max_bricks))
            n_rows += 1

        # 4. Optional per-part inventory constraints
        if inventory_counts is not None:
            by_part: Dict[str, List[int]] = {}
            for idx, cid in enumerate(valid_cids):
                by_part.setdefault(pool.candidates[cid].part_id, []).append(idx)
            for pid, idx_list in by_part.items():
                cap = int(inventory_counts.get(pid, 0))
                if len(idx_list) > cap:
                    for col_i in idx_list:
                        coo_rows.append(n_rows)
                        coo_cols.append(col_i)
                        coo_vals.append(1.0)
                    lbs.append(-np.inf)
                    ubs.append(float(cap))
                    n_rows += 1

        c_base = -np.array([cand_rewards[cid] for cid in valid_cids], dtype=np.float64)
        aux_costs: List[float] = []
        if pairwise_synergy_matrix is not None and pairwise_synergy_matrix.shape == (m, m):
            for la in range(k_valid):
                cid_a = valid_cids[la]
                for lb in range(la + 1, k_valid):
                    cid_b = valid_cids[lb]
                    if cid_b in pool.conflict_adjacency[cid_a]:
                        continue
                    q_ab = float(
                        pairwise_synergy_matrix[cid_a, cid_b] + pairwise_synergy_matrix[cid_b, cid_a]
                    ) * 0.5
                    if abs(q_ab) <= 1e-4:
                        continue
                    y_col = k_valid + len(aux_costs)
                    aux_costs.append(-q_ab)
                    if q_ab > 0.0:
                        # Maximize q_ab * y_ab: y_ab <= z_a and y_ab <= z_b
                        coo_rows.extend([n_rows, n_rows])
                        coo_cols.extend([y_col, la])
                        coo_vals.extend([1.0, -1.0])
                        lbs.append(-np.inf)
                        ubs.append(0.0)
                        n_rows += 1
                        coo_rows.extend([n_rows, n_rows])
                        coo_cols.extend([y_col, lb])
                        coo_vals.extend([1.0, -1.0])
                        lbs.append(-np.inf)
                        ubs.append(0.0)
                        n_rows += 1
                    else:
                        # Penalize incompatible pair (q_ab < 0): y_ab >= z_a + z_b - 1  <=> z_a + z_b - y_ab <= 1
                        coo_rows.extend([n_rows, n_rows, n_rows])
                        coo_cols.extend([la, lb, y_col])
                        coo_vals.extend([1.0, 1.0, -1.0])
                        lbs.append(-np.inf)
                        ubs.append(1.0)
                        n_rows += 1

        total_mip_vars = k_valid + len(aux_costs)
        c_obj = (
            np.concatenate([c_base, np.asarray(aux_costs, dtype=np.float64)])
            if aux_costs
            else c_base
        )
        extra_cut_rows: List[np.ndarray] = []

        nogood_cuts_used = 0
        best_fallback_subset: List[int] = []

        base_csc = (
            csc_matrix((coo_vals, (coo_rows, coo_cols)), shape=(n_rows, total_mip_vars))
            if n_rows > 0
            else None
        )

        for _round in range(self.max_nogood_rounds):
            if deadline_monotonic_s is not None and time.monotonic() >= deadline_monotonic_s:
                break
            if base_csc is not None and not extra_cut_rows:
                a_mat: Optional[csc_matrix] = base_csc
            elif extra_cut_rows:
                from scipy.sparse import vstack as sp_vstack

                cut_csc = csc_matrix(np.vstack(extra_cut_rows))
                a_mat = sp_vstack([base_csc, cut_csc], format="csc") if base_csc is not None else cut_csc
            else:
                a_mat = None

            mip_tl = 0.06
            if deadline_monotonic_s is not None:
                rem_mip = deadline_monotonic_s - time.monotonic()
                if rem_mip <= 0.0:
                    break
                mip_tl = min(0.06, max(0.005, rem_mip))
            x_sol = _solve_binary_mip_fast(
                c_obj=c_obj,
                a_mat=a_mat,
                lbs=lbs,
                ubs=ubs,
                time_limit_s=mip_tl,
                mip_rel_gap=0.002,
            )
            if x_sol is not None:
                chosen_local = [i for i in range(k_valid) if x_sol[i] >= 0.5]
                if not chosen_local:
                    break
                chosen_cids = [valid_cids[i] for i in chosen_local]
                if not pool.is_selection_compatible(chosen_cids):
                    chosen_cids = pool.decode_compatible_course(
                        cand_rewards, max_bricks=max_bricks, score_threshold=0.0
                    )
            elif _round == 0:
                # Timeout before first MIP incumbent on large floor: fall back to greedy set packing
                chosen_cids = pool.decode_compatible_course(
                    cand_rewards, max_bricks=max_bricks, score_threshold=0.0
                )
                if inventory_counts is not None:
                    rem_local: Dict[str, int] = dict(inventory_counts)
                    filtered_cids: List[int] = []
                    for cid_g in chosen_cids:
                        pid_g = pool.candidates[cid_g].part_id
                        if rem_local.get(pid_g, 0) > 0:
                            filtered_cids.append(cid_g)
                            rem_local[pid_g] -= 1
                    chosen_cids = filtered_cids
                chosen_local = [valid_pos_map[c] for c in chosen_cids if c in valid_pos_map]
                if not chosen_cids:
                    break
            else:
                break

            # Grounded base-course pool (virtual ground ports `(-1, k)`) is grounded by construction
            is_ground_pool = all(
                gp[0] == -1 for cid in chosen_cids for gp in pool.candidates[cid].consumed_ports
            )
            if is_ground_pool:
                return chosen_cids, nogood_cuts_used

            # Verify full 6-DoF static equilibrium LP unconditionally on EVERY non-ground course
            # using O(1) in-place push/pop trial tokens (zero `copy.deepcopy(state)`).
            pushed_tokens: List[TrialPartRollbackToken] = []
            try:
                if not pool.is_selection_compatible(chosen_cids):
                    raise ValueError("Incompatible selection")
                for cid in chosen_cids:
                    cand_c = pool.candidates[cid]
                    cand_grounded = all(gp[0] == -1 for gp in cand_c.consumed_ports)
                    pushed_tokens.append(
                        state.push_trial_part(cand_c.part_id, cand_c.pose, is_grounded=cand_grounded)
                    )
                eq = self.validator.evaluate(
                    state,
                    foundation_policy=foundation_policy,
                    ground_plane_z_ldu=ground_plane_z_ldu,
                    deadline_monotonic_s=deadline_monotonic_s,
                )
            except ValueError:
                eq = EquilibriumResult(
                    is_stable=False,
                    peak_utilization=float("inf"),
                    safety_margin=-float("inf"),
                    contact_utilizations={},
                    weakest_contact=None,
                    solve_time_ms=0.0,
                    status_message="Course collision",
                )
            finally:
                for tok in reversed(pushed_tokens):
                    state.pop_trial_part(tok)

            if eq.budget_exceeded:
                return [], nogood_cuts_used
            if eq.is_stable:
                if require_atomic_1by1_order and len(chosen_cids) > 1:
                    ordered_1by1 = _order_course_atomic_1by1(
                        state,
                        pool,
                        chosen_cids,
                        self.validator,
                        prior_peak_utilization=prior_peak_utilization,
                    )
                    if ordered_1by1 is not None:
                        return ordered_1by1, nogood_cuts_used
                else:
                    return chosen_cids, nogood_cuts_used

            # Append canonical binary no-good cut (Critique 5C fix)
            cut_row, cut_lb, cut_ub = build_canonical_nogood_cut(
                num_vars=total_mip_vars,
                cand_col_offset=0,
                selected_indices=chosen_local,
                num_candidates=k_valid,
            )
            extra_cut_rows.append(cut_row)
            lbs.append(cut_lb)
            ubs.append(cut_ub)
            nogood_cuts_used += 1
            if not best_fallback_subset:
                best_fallback_subset = chosen_cids

        # Lexicographic bounded-subtree fallback repair: retain highest-reward stable prefix
        # and greedily fill any vacated cells with non-conflicting stable candidates from `valid_cids`.
        ordered_fallback = sorted(best_fallback_subset, key=lambda c: cand_rewards[c], reverse=True)
        seen_fb = set(ordered_fallback)
        extra_fill = [
            c
            for c in sorted(valid_cids, key=lambda c: cand_rewards[c], reverse=True)
            if c not in seen_fb
        ]
        stable_prefix: List[int] = []
        blocked_fb: Set[int] = set()
        used_inv_fb: Dict[str, int] = {}
        rep_tokens: List[TrialPartRollbackToken] = []
        consec_fb_fails = 0
        try:
            for cid in ordered_fallback + extra_fill:
                if deadline_monotonic_s is not None and time.monotonic() >= deadline_monotonic_s:
                    break
                if cid in blocked_fb:
                    continue
                if max_bricks is not None and len(stable_prefix) >= max_bricks:
                    break
                if cid not in seen_fb and consec_fb_fails >= 24:
                    break
                cand_fb = pool.candidates[cid]
                if inventory_counts is not None:
                    if used_inv_fb.get(cand_fb.part_id, 0) >= inventory_counts.get(cand_fb.part_id, 0):
                        continue
                try:
                    tok = state.push_trial_part(cand_fb.part_id, cand_fb.pose, is_grounded=False)
                except ValueError:
                    continue
                eq_fb = self.validator.evaluate(
                    state,
                    foundation_policy=foundation_policy,
                    ground_plane_z_ldu=ground_plane_z_ldu,
                    deadline_monotonic_s=deadline_monotonic_s,
                )
                if eq_fb.budget_exceeded:
                    state.pop_trial_part(tok)
                    break
                if eq_fb.is_stable:
                    consec_fb_fails = 0
                    rep_tokens.append(tok)
                    stable_prefix.append(cid)
                    blocked_fb.add(cid)
                    blocked_fb.update(pool.conflict_adjacency[cid])
                    used_inv_fb[cand_fb.part_id] = used_inv_fb.get(cand_fb.part_id, 0) + 1
                else:
                    consec_fb_fails += 1
                    state.pop_trial_part(tok)
        finally:
            for tok in reversed(rep_tokens):
                state.pop_trial_part(tok)
        return stable_prefix, nogood_cuts_used


class PairSeededLocalSearch:
    """`S2` Fast Greedy + 1-Out/2-In Local Swap Search Baseline."""

    solver_id: str = "S2_PairSeededLocalSearch"

    def __init__(self, validator: Optional[SparseStaticEquilibriumValidator] = None) -> None:
        self.validator = validator or SparseStaticEquilibriumValidator(
            clutch_force_capacity_grams=99.9
        )

    def solve_course(
        self,
        state: AssemblyState,
        pool: CandidatePool,
        target_field: SparseTiledTargetField | TargetShapeBoxes,
        max_bricks: Optional[int] = None,
        inventory_counts: Optional[Dict[str, int]] = None,
    ) -> List[int]:
        m = len(pool)
        if m == 0:
            return []
        scores = np.full(m, -100.0, dtype=np.float64)
        for cid, cand in enumerate(pool.candidates):
            if inventory_counts is not None and inventory_counts.get(cand.part_id, 0) <= 0:
                continue
            vol = float(sum(b.volume_ldu3 for b in cand.world_boxes))
            inter = float(target_field.intersection_volume(cand.world_boxes))
            if inter / max(vol, 1.0) >= 0.95:
                scores[cid] = (inter / 9600.0) - 0.55 + (0.20 if cand.closes_structural_seam else 0.0)
        greedy_sel = pool.decode_compatible_course(scores, max_bricks=max_bricks, score_threshold=0.0)
        if inventory_counts is not None:
            rem_g: Dict[str, int] = dict(inventory_counts)
            filt_g: List[int] = []
            for cg in greedy_sel:
                pg = pool.candidates[cg].part_id
                if rem_g.get(pg, 0) > 0:
                    filt_g.append(cg)
                    rem_g[pg] -= 1
            greedy_sel = filt_g
        # 1-out / 2-in local swap improvement over candidates unblocked by removing `rem_idx`
        best_sel = list(greedy_sel)
        best_val = float(sum(scores[i] for i in best_sel))
        for rem_idx in list(best_sel):
            if rem_idx not in best_sel:
                continue
            base_set = [i for i in best_sel if i != rem_idx]
            blocked: Set[int] = set(base_set)
            for b_i in base_set:
                blocked.update(pool.conflict_adjacency[b_i])
            # Only candidates conflicting with `rem_idx` become newly available when `rem_idx` is removed!
            avail = [
                c
                for c in pool.conflict_adjacency[rem_idx]
                if scores[c] > 0.0 and c not in blocked and c != rem_idx
            ]
            if len(avail) > 24:
                avail.sort(key=lambda c: float(scores[c]), reverse=True)
                avail = avail[:24]
            base_val = float(sum(scores[i] for i in base_set))
            for a_i in range(len(avail)):
                c1 = avail[a_i]
                for b_i in range(a_i + 1, len(avail)):
                    c2 = avail[b_i]
                    if c2 in pool.conflict_adjacency[c1]:
                        continue
                    if max_bricks is not None and len(base_set) + 2 > max_bricks:
                        continue
                    if inventory_counts is not None:
                        used_cnt: Dict[str, int] = {}
                        inv_ok = True
                        for ci in base_set + [c1, c2]:
                            pi = pool.candidates[ci].part_id
                            used_cnt[pi] = used_cnt.get(pi, 0) + 1
                            if used_cnt[pi] > inventory_counts.get(pi, 0):
                                inv_ok = False
                                break
                        if not inv_ok:
                            continue
                    val = base_val + float(scores[c1]) + float(scores[c2])
                    if val > best_val + 1e-4 and pool.is_selection_compatible(base_set + [c1, c2]):
                        best_sel = base_set + [c1, c2]
                        best_val = val

        if not best_sel:
            return []
        is_ground_pool = all(
            gp[0] == -1 for cid in best_sel for gp in pool.candidates[cid].consumed_ports
        )
        if is_ground_pool:
            return best_sel

        pushed: List[TrialPartRollbackToken] = []
        whole_stable = False
        try:
            for cid in best_sel:
                cand_c = pool.candidates[cid]
                pushed.append(state.push_trial_part(cand_c.part_id, cand_c.pose, is_grounded=False))
            whole_stable = self.validator.evaluate(state).is_stable
        except ValueError:
            whole_stable = False
        finally:
            for tok in reversed(pushed):
                state.pop_trial_part(tok)

        if whole_stable:
            if len(best_sel) <= 1:
                return best_sel
            ordered_1by1 = _order_course_atomic_1by1(state, pool, best_sel, self.validator)
            if ordered_1by1 is not None:
                return ordered_1by1

        # Greedy 1-by-1 equilibrium-certified fallback over positive-scoring candidates
        seen_s2 = set(best_sel)
        cand_order = sorted(best_sel, key=lambda c: float(scores[c]), reverse=True) + [
            c
            for c in sorted(range(m), key=lambda i: float(scores[i]), reverse=True)
            if scores[c] > 0.0 and c not in seen_s2
        ]
        stable_s2: List[int] = []
        blocked_s2: Set[int] = set()
        used_s2: Dict[str, int] = {}
        toks_s2: List[TrialPartRollbackToken] = []
        try:
            for cid in cand_order:
                if cid in blocked_s2:
                    continue
                if max_bricks is not None and len(stable_s2) >= max_bricks:
                    break
                cand_f = pool.candidates[cid]
                if inventory_counts is not None:
                    if used_s2.get(cand_f.part_id, 0) >= inventory_counts.get(cand_f.part_id, 0):
                        continue
                try:
                    tok = state.push_trial_part(cand_f.part_id, cand_f.pose, is_grounded=False)
                except ValueError:
                    continue
                if self.validator.evaluate(state).is_stable:
                    toks_s2.append(tok)
                    stable_s2.append(cid)
                    blocked_s2.add(cid)
                    blocked_s2.update(pool.conflict_adjacency[cid])
                    used_s2[cand_f.part_id] = used_s2.get(cand_f.part_id, 0) + 1
                else:
                    state.pop_trial_part(tok)
        finally:
            for tok in reversed(toks_s2):
                state.pop_trial_part(tok)
        return stable_s2


class RecurrentBatchRefiner:
    """`S3` Experimental Recurrent / Iterative Coordinate-Ascent Course Refiner."""

    solver_id: str = "S3_RecurrentBatchRefiner"

    def __init__(self, num_steps: int = 4) -> None:
        self.num_steps = int(num_steps)

    def solve_course(
        self,
        state: AssemblyState,
        pool: CandidatePool,
        target_field: SparseTiledTargetField | TargetShapeBoxes,
        max_bricks: Optional[int] = None,
    ) -> List[int]:
        m = len(pool)
        if m == 0:
            return []
        base_scores = np.full(m, -100.0, dtype=np.float64)
        for cid, cand in enumerate(pool.candidates):
            vol = float(sum(b.volume_ldu3 for b in cand.world_boxes))
            inter = float(target_field.intersection_volume(cand.world_boxes))
            if inter / max(vol, 1.0) >= 0.95:
                base_scores[cid] = (
                    (inter / 9600.0)
                    - 0.55
                    + (0.25 if cand.closes_structural_seam else 0.0)
                    + (0.20 if cand.creates_biconnected_cycle else 0.0)
                )
        logits = np.copy(base_scores)
        best_sel: List[int] = []
        best_score = -float("inf")
        for _ in range(self.num_steps):
            sel = pool.decode_compatible_course(logits, max_bricks=max_bricks, score_threshold=0.0)
            sc = float(sum(base_scores[i] for i in sel))
            if sc > best_score:
                best_score = sc
                best_sel = sel
            # Boost non-conflicting complements of currently selected course
            selected_set = set(sel)
            blocked: Set[int] = set()
            for i in sel:
                blocked.update(pool.conflict_adjacency[i])
            for cid in range(m):
                if base_scores[cid] > 0:
                    if cid in selected_set or cid not in blocked:
                        logits[cid] = 0.7 * logits[cid] + 0.3 * (base_scores[cid] + 0.15)
        return best_sel


class D3PMBatchSampler:
    """
    `S4` Opt-In Experimental Multi-Modal Discrete Denoising Sampler (`D3PMBatchSampler`).
    Kept outside the default production path per Round 7 Decision 4; used only when a caller
    explicitly requests multi-sample (`K > 1`) diverse course realizations.
    """

    solver_id: str = "S4_D3PMBatchSampler"

    def __init__(self, num_samples: int = 6, num_steps: int = 4, temperature: float = 0.35) -> None:
        self.num_samples = int(num_samples)
        self.num_steps = int(num_steps)
        self.temperature = float(temperature)

    def sample_diverse_courses(
        self,
        pool: CandidatePool,
        target_field: SparseTiledTargetField | TargetShapeBoxes,
        max_bricks: Optional[int] = None,
        seed: int = 0,
    ) -> List[Tuple[int, ...]]:
        m = len(pool)
        if m == 0:
            return []
        rng = np.random.default_rng(seed)
        base_scores = np.full(m, -100.0, dtype=np.float64)
        for cid, cand in enumerate(pool.candidates):
            vol = float(sum(b.volume_ldu3 for b in cand.world_boxes))
            inter = float(target_field.intersection_volume(cand.world_boxes))
            if inter / max(vol, 1.0) >= 0.95:
                base_scores[cid] = (inter / 9600.0) + (0.20 if cand.closes_structural_seam else 0.0)

        modes: List[Tuple[int, ...]] = []
        seen: Set[Tuple[int, ...]] = set()
        for k in range(self.num_samples):
            noise = rng.normal(0.0, self.temperature, size=m) if k > 0 else np.zeros(m)
            noisy_scores = np.where(base_scores > 0.0, base_scores + noise, -100.0)
            sel = tuple(sorted(pool.decode_compatible_course(noisy_scores, max_bricks=max_bricks, score_threshold=0.0)))
            if sel and sel not in seen:
                seen.add(sel)
                modes.append(sel)
        return modes


class GreedyLayerMergeControl:
    """
    Fast `1x1`/`1x2` Alternating-Axis Greedy Layer-Merge Control (`GreedyLayerMerge_Control`;
    legacy alias `Legolization_Baseline` retained for backward compatibility, `R8-WS3` fix).
    Initializes occupied target cells as `1x1` bricks (`3005`) and greedily merges adjacent
    pairs into `1x2` bricks (`3004`), alternating dominant merge axis (`X` vs `Y`) across courses
    without global optimization or iterative seam refinement.
    """

    solver_id: str = "GreedyLayerMerge_Control"

    @staticmethod
    def _x_merge_rot_id() -> int:
        b3004 = get_brick_primitive("3004").hull_boxes[0]
        for rid in C4_Z_ROT_IDS:
            rb = b3004.transform(LDUFrame(rid, (0, 0, 0)))
            if rb.xmax - rb.xmin == 40 and rb.xmin == -10:
                return rid
        return C4_Z_ROT_IDS[1]

    @classmethod
    def realize_blind(
        cls,
        target_field: SparseTiledTargetField | TargetShapeBoxes,
        allowed_part_ids: Sequence[str] = STABLETEXT2BRICK_8PART_KIT,
        validator: Optional[SparseStaticEquilibriumValidator] = None,
        deadline_monotonic_s: Optional[float] = None,
        ground_plane_z_ldu: float = 0.0,
    ) -> Tuple[AssemblyState, float]:
        """
        Genuinely blind `1x1`/`1x2` greedy layer-merge control operating strictly on
        `target_field` with zero reference LDraw specs.
        """
        t0 = time.perf_counter()
        _ = validator
        state = AssemblyState()
        bounds = target_field.occupied_bounds_ldu()
        if bounds is None or target_field.total_volume_ldu3 <= 0:
            return state, 0.0
        xmin, ymin, zmin, xmax, ymax, zmax = bounds
        z_levels = [
            z0
            for z0 in range(zmin, zmax, 24)
            if target_field.band_volume_ldu3(z0, z0 + 24) > 0
        ]
        if not z_levels:
            return state, 0.0
        min_z = z_levels[0]
        x_rot = cls._x_merge_rot_id()

        is_g_z = float(ground_plane_z_ldu)
        for layer_idx, z0 in enumerate(z_levels):
            if deadline_monotonic_s is not None and time.monotonic() >= deadline_monotonic_s:
                break
            occupied_cells: Set[Tuple[int, int]] = set()
            for cx in range(xmin + 10, xmax, 20):
                for cy in range(ymin + 10, ymax, 20):
                    if target_field.contains_point((cx, cy, z0 + 12)):
                        occupied_cells.add((cx, cy))

            prefer_y = (layer_idx % 2 == 0)
            unmerged = set(occupied_cells)
            for cx, cy in sorted(occupied_cells):
                if (cx, cy) not in unmerged:
                    continue
                placed_ok = False
                if prefer_y and (cx, cy + 20) in unmerged and "3004" in allowed_part_ids:
                    try:
                        state.place_root_part("3004", LDUFrame(0, (cx, cy, z0)), is_grounded=(z0 <= min_z or abs(float(z0) - is_g_z) <= 0.25))
                        unmerged.remove((cx, cy))
                        unmerged.remove((cx, cy + 20))
                        placed_ok = True
                    except ValueError:
                        pass
                elif (not prefer_y) and (cx + 20, cy) in unmerged and "3004" in allowed_part_ids:
                    try:
                        state.place_root_part(
                            "3004", LDUFrame(x_rot, (cx, cy, z0)), is_grounded=(z0 <= min_z or abs(float(z0) - is_g_z) <= 0.25)
                        )
                        unmerged.remove((cx, cy))
                        unmerged.remove((cx + 20, cy))
                        placed_ok = True
                    except ValueError:
                        pass
                if not placed_ok and "3005" in allowed_part_ids:
                    try:
                        state.place_root_part("3005", LDUFrame(0, (cx, cy, z0)), is_grounded=(z0 <= min_z or abs(float(z0) - is_g_z) <= 0.25))
                        unmerged.remove((cx, cy))
                    except ValueError:
                        pass

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return state, elapsed_ms

    @classmethod
    def realize_from_reference_specs(
        cls,
        specs_by_z: Dict[int, List[Any]],
        target_field: SparseTiledTargetField,
        allowed_part_ids: Sequence[str] = STABLETEXT2BRICK_8PART_KIT,
        validator: Optional[SparseStaticEquilibriumValidator] = None,
    ) -> Tuple[AssemblyState, float]:
        _ = specs_by_z
        return cls.realize_blind(
            target_field=target_field,
            allowed_part_ids=allowed_part_ids,
            validator=validator,
        )


LegolizationBaseline = GreedyLayerMergeControl


class LuoInspiredSeamRefinement:
    """
    Mechanics-Guided Multi-Brick Merge + Iterative Seam Refinement Internal Comparator
    (`LuoInspiredSeamRefinement`; inspired by `Luo et al., SIGGRAPH Asia 2015`, `R9-WS2` §4.1;
    legacy alias `IterativeSeamRefinementLuo2015` retained for backward compatibility):
      1. Greedily merges occupied `1x1` cells in each layer into larger rectangular bricks
         (`2x8`, `2x6`, `2x4`, `2x3`, `2x2`, `1x8`, `1x6`, `1x4`, `1x3`, `1x2`, `1x1`)
         filtered to `allowed_part_ids`, alternating dominant axis and parity stagger across layers.
      2. Evaluates 6-DoF static equilibrium + graph connectivity and runs up to `max_rounds`
         of targeted weak-joint / seam-crossing re-merging on overloaded or disconnected layers.
    """

    solver_id: str = "LuoInspiredSeamRefinement"

    @staticmethod
    def _catalog_rect_templates(
        allowed_part_ids: Sequence[str],
    ) -> List[Tuple[int, str, int, int, int, int, int]]:
        templates: List[Tuple[int, str, int, int, int, int, int]] = []
        seen: Set[Tuple[str, int, int]] = set()
        for pid in allowed_part_ids:
            if not is_known_part_id(pid):
                continue
            prim = get_brick_primitive(pid)
            if len(prim.hull_boxes) != 1 or prim.is_snot:
                continue
            b0 = prim.hull_boxes[0]
            if (b0.zmax - b0.zmin) != 24:
                continue
            for rid in C4_Z_ROT_IDS:
                rb = b0.transform(LDUFrame(rid, (0, 0, 0)))
                nx = (rb.xmax - rb.xmin) // 20
                ny = (rb.ymax - rb.ymin) // 20
                if nx <= 0 or ny <= 0 or (pid, nx, ny) in seen:
                    continue
                seen.add((pid, nx, ny))
                templates.append((nx * ny, pid, rid, nx, ny, -rb.xmin, -rb.ymin))
        templates.sort(key=lambda t: (t[0], min(t[3], t[4])), reverse=True)
        return templates

    @classmethod
    def _tile_single_layer(
        cls,
        occupied_origins: Set[Tuple[int, int]],
        z0: int,
        templates: Sequence[Tuple[int, str, int, int, int, int, int]],
        prefer_x: bool,
        stagger_shift: int,
        prev_seams_x: Set[int],
        prev_seams_y: Set[int],
    ) -> List[Tuple[str, LDUFrame]]:
        unmerged = set(occupied_origins)
        placements: List[Tuple[str, LDUFrame]] = []
        ordered_templates = sorted(
            templates,
            key=lambda t: (
                t[0],
                (t[3] >= t[4]) if prefer_x else (t[4] >= t[3]),
            ),
            reverse=True,
        )
        scan_cells = sorted(
            occupied_origins,
            key=lambda xy: (
                ((xy[0] + stagger_shift) % 40 != 0),
                xy[0] if prefer_x else xy[1],
                xy[1] if prefer_x else xy[0],
            ),
        )
        for x0, y0 in scan_cells:
            if (x0, y0) not in unmerged:
                continue
            best_choice: Optional[Tuple[float, str, LDUFrame, List[Tuple[int, int]]]] = None
            for area, pid, rid, nx, ny, ox_off, oy_off in ordered_templates:
                cells = [
                    (x0 + ix * 20, y0 + iy * 20)
                    for ix in range(nx)
                    for iy in range(ny)
                ]
                if not all(c in unmerged for c in cells):
                    continue
                cross_bonus = 0.0
                for ix in range(1, nx):
                    if (x0 + ix * 20) in prev_seams_x:
                        cross_bonus += 0.5
                for iy in range(1, ny):
                    if (y0 + iy * 20) in prev_seams_y:
                        cross_bonus += 0.5
                orient_bonus = 0.15 if ((nx > ny) == prefer_x) else 0.0
                score = float(area) + cross_bonus + orient_bonus
                pose = LDUFrame(rid, (x0 + ox_off, y0 + oy_off, z0))
                if best_choice is None or score > best_choice[0]:
                    best_choice = (score, pid, pose, cells)
            if best_choice is not None:
                _, pid_b, pose_b, cells_b = best_choice
                placements.append((pid_b, pose_b))
                for c in cells_b:
                    unmerged.discard(c)
        return placements

    @classmethod
    def realize_blind(
        cls,
        target_field: SparseTiledTargetField | TargetShapeBoxes,
        allowed_part_ids: Sequence[str] = STABLETEXT2BRICK_8PART_KIT,
        validator: Optional[SparseStaticEquilibriumValidator] = None,
        max_refinement_rounds: int = 8,
        deadline_monotonic_s: Optional[float] = None,
        ground_plane_z_ldu: float = 0.0,
    ) -> Tuple[AssemblyState, float]:
        t0 = time.perf_counter()
        is_g_z = float(ground_plane_z_ldu)
        val = validator or SparseStaticEquilibriumValidator(clutch_force_capacity_grams=99.9)
        bounds = target_field.occupied_bounds_ldu()
        if bounds is None or target_field.total_volume_ldu3 <= 0:
            return AssemblyState(), 0.0
        xmin, ymin, zmin, xmax, ymax, zmax = bounds
        z_levels = [
            z0
            for z0 in range(zmin, zmax, 24)
            if target_field.band_volume_ldu3(z0, z0 + 24) > 0
        ]
        if not z_levels:
            return AssemblyState(), 0.0
        min_z = z_levels[0]
        templates = cls._catalog_rect_templates(allowed_part_ids)
        if not templates:
            return GreedyLayerMergeControl.realize_blind(target_field, allowed_part_ids, val)

        layer_cells: List[Set[Tuple[int, int]]] = []
        for z0 in z_levels:
            occ: Set[Tuple[int, int]] = set()
            for x0 in range(xmin, xmax, 20):
                for y0 in range(ymin, ymax, 20):
                    if target_field.contains_point((x0 + 10, y0 + 10, z0 + 12)):
                        occ.add((x0, y0))
            layer_cells.append(occ)

        layer_configs = [(idx % 2 == 1, (idx % 2) * 20) for idx in range(len(z_levels))]
        best_state: Optional[AssemblyState] = None
        best_peak_u = float("inf")

        for _r in range(max_refinement_rounds):
            if deadline_monotonic_s is not None and time.monotonic() >= deadline_monotonic_s:
                break
            st = AssemblyState()
            prev_sx: Set[int] = set()
            prev_sy: Set[int] = set()
            for idx, z0 in enumerate(z_levels):
                pref_x, stag = layer_configs[idx]
                layer_pl = cls._tile_single_layer(
                    layer_cells[idx], z0, templates, pref_x, stag, prev_sx, prev_sy
                )
                cur_sx: Set[int] = set()
                cur_sy: Set[int] = set()
                for pid_p, pose_p in layer_pl:
                    try:
                        iid = st.place_root_part(pid_p, pose_p, is_grounded=(z0 <= min_z or abs(float(z0) - is_g_z) <= 0.25))
                        wb = st.placed_parts[iid].world_boxes[0]
                        cur_sx.add(wb.xmin)
                        cur_sx.add(wb.xmax)
                        cur_sy.add(wb.ymin)
                        cur_sy.add(wb.ymax)
                    except ValueError:
                        continue
                prev_sx, prev_sy = cur_sx, cur_sy

            depths = st.compute_support_depths()
            disconnected_ids = [iid for iid, d in depths.items() if d >= 999]
            if not disconnected_ids:
                eq = val.evaluate(
                    st,
                    ground_plane_z_ldu=ground_plane_z_ldu,
                    deadline_monotonic_s=deadline_monotonic_s,
                )
                if eq.budget_exceeded:
                    break
                if eq.peak_utilization < best_peak_u:
                    best_peak_u = float(eq.peak_utilization)
                    best_state = st
                if eq.is_stable:
                    return st, (time.perf_counter() - t0) * 1000.0
                weak_pair = eq.weakest_contact
                if weak_pair is not None and weak_pair[0][0] in st.placed_parts:
                    wz = min(b.zmin for b in st.placed_parts[weak_pair[0][0]].world_boxes)
                else:
                    wz = z_levels[min(1, len(z_levels) - 1)]
            else:
                if best_state is None:
                    best_state = st
                wz = min(
                    min(b.zmin for b in st.placed_parts[iid].world_boxes)
                    for iid in disconnected_ids
                )

            # Flip dominant orientation & shift stagger phase around the weakest layer `wz`
            if wz in z_levels:
                w_idx = z_levels.index(wz)
                for adj in (w_idx, max(0, w_idx - 1), min(len(z_levels) - 1, w_idx + 1)):
                    old_px, old_st = layer_configs[adj]
                    layer_configs[adj] = (not old_px, (old_st + 20) % 40)
            else:
                break

        return (best_state if best_state is not None else AssemblyState()), (
            time.perf_counter() - t0
        ) * 1000.0


IterativeSeamRefinementLuo2015 = LuoInspiredSeamRefinement


def _build_blind_grounded_base_pool(
    target_field: SparseTiledTargetField | TargetShapeBoxes,
    z0: int,
    dz: int,
    allowed_part_ids: Sequence[str],
) -> CandidatePool:
    """
    Enumerate all in-silhouette (`prec >= 0.95`) grounded base placements on `[z0, z0 + dz)`
    strictly from `target_field` (zero reference part specs), constructing virtual ground-cell
    ports `(-1, cell_idx)` and exact conflict hyperedges `CandidatePool`.
    """
    bounds = target_field.occupied_bounds_ldu()
    if bounds is None:
        return CandidatePool(
            candidates=(),
            port_to_candidates={},
            conflict_edges=(),
            conflict_adjacency=(),
        )
    xmin, ymin, _, xmax, ymax, _ = bounds
    cell_to_port_id: Dict[Tuple[int, int], GlobalPortId] = {}
    occupied_origins: List[Tuple[int, int]] = []
    if (
        xmin % 10 == 0
        and ymin % 10 == 0
        and (xmax - xmin) % 20 == 0
        and (ymax - ymin) % 20 == 0
    ):
        for x0 in range(xmin, xmax, 20):
            for y0 in range(ymin, ymax, 20):
                if target_field.contains_point((x0 + 10, y0 + 10, z0 + dz // 2)):
                    gpid: GlobalPortId = (-1, len(occupied_origins))
                    cell_to_port_id[(x0, y0)] = gpid
                    occupied_origins.append((x0, y0))
    else:
        min_cell_vol = int(0.95 * 400 * dz)
        best_cells: List[Tuple[int, int]] = []
        for px, py in ((0, 0), (10, 10), (0, 10), (10, 0)):
            gx0 = ((xmin - px) // 20) * 20 + px
            gy0 = ((ymin - py) // 20) * 20 + py
            cur_cells: List[Tuple[int, int]] = []
            for x0 in range(gx0, xmax, 20):
                for y0 in range(gy0, ymax, 20):
                    if target_field.intersection_volume(
                        (LDUBox(x0, y0, z0, x0 + 20, y0 + 20, z0 + dz),)
                    ) >= min_cell_vol:
                        cur_cells.append((x0, y0))
            if len(cur_cells) > len(best_cells):
                best_cells = cur_cells
        for x0, y0 in best_cells:
            gpid = (-1, len(occupied_origins))
            cell_to_port_id[(x0, y0)] = gpid
            occupied_origins.append((x0, y0))

    if not occupied_origins:
        return CandidatePool(
            candidates=(),
            port_to_candidates={},
            conflict_edges=(),
            conflict_adjacency=(),
        )

    candidates: List[CandidatePlacement] = []
    seen_keys: Set[Tuple[str, Tuple[LDUBox, ...], int]] = set()
    for part_id in allowed_part_ids:
        prim = get_brick_primitive(part_id)
        part_vol = sum(b.volume_ldu3 for b in prim.hull_boxes)
        min_inter = int(0.95 * part_vol)
        seen_rot_shapes: Set[Tuple[Tuple[Tuple[int, int, int, int, int, int], ...], int]] = set()
        for rot_id in C4_Z_ROT_IDS:
            r_hulls = tuple(b.transform(LDUFrame(rot_id, (0, 0, 0))) for b in prim.hull_boxes)
            rb0 = r_hulls[0]
            if (rb0.zmax - rb0.zmin) > dz:
                continue
            has_asym_ports = prim.is_snot or any(
                p.connector_type != ConnectorType.STANDARD_STUD or abs(int(p.local_frame.z_axis[2])) != 1
                for p in prim.ports
            )
            dedup_rot = rot_id if has_asym_ports else 0
            rel_offsets = tuple(
                (
                    rb.xmin - rb0.xmin,
                    rb.ymin - rb0.ymin,
                    rb.zmin - rb0.zmin,
                    rb.xmax - rb0.xmin,
                    rb.ymax - rb0.ymin,
                    rb.zmax - rb0.zmin,
                )
                for rb in r_hulls
            )
            shape_sig = (rel_offsets, dedup_rot)
            if shape_sig in seen_rot_shapes:
                continue
            seen_rot_shapes.add(shape_sig)
            is_single_box = len(rel_offsets) == 1
            dx0_single = rel_offsets[0][3] if is_single_box else 0
            dy0_single = rel_offsets[0][4] if is_single_box else 0
            dz0_single = rel_offsets[0][5] if is_single_box else 0
            ox_off, oy_off, oz_off = -rb0.xmin, -rb0.ymin, -rb0.zmin
            for x0, y0 in occupied_origins:
                if is_single_box:
                    w_boxes = (LDUBox(x0, y0, z0, x0 + dx0_single, y0 + dy0_single, z0 + dz0_single),)
                else:
                    w_boxes = tuple(
                        LDUBox(x0 + ox0, y0 + oy0, z0 + oz0, x0 + ox1, y0 + oy1, z0 + oz1)
                        for ox0, oy0, oz0, ox1, oy1, oz1 in rel_offsets
                    )
                d_key = (part_id, w_boxes, dedup_rot)
                if d_key in seen_keys:
                    continue
                seen_keys.add(d_key)
                inter_b = int(target_field.intersection_volume(w_boxes))
                if inter_b < min_inter:
                    continue
                pose = LDUFrame(rot_id, (x0 + ox_off, y0 + oy_off, z0 + oz_off))
                consumed: Set[GlobalPortId] = set()
                for wb in w_boxes:
                    for cx0 in range(wb.xmin, wb.xmax, 20):
                        for cy0 in range(wb.ymin, wb.ymax, 20):
                            gpid_c = cell_to_port_id.get((cx0, cy0))
                            if gpid_c is not None:
                                consumed.add(gpid_c)
                if not consumed:
                    continue
                candidates.append(
                    CandidatePlacement(
                        candidate_id=len(candidates),
                        part_id=part_id,
                        pose=pose,
                        consumed_ports=frozenset(consumed),
                        matched_pairs=tuple((0, gp) for gp in sorted(consumed)),
                        world_boxes=w_boxes,
                        bridged_instances=frozenset(),
                        closes_structural_seam=(len(consumed) >= 2),
                        creates_biconnected_cycle=False,
                        gravity_local=(0, 0, -1),
                        target_intersection_ldu3=inter_b,
                    )
                )

    m = len(candidates)
    port_to_cands: Dict[GlobalPortId, List[int]] = {}
    for cand in candidates:
        for gp in cand.consumed_ports:
            port_to_cands.setdefault(gp, []).append(cand.candidate_id)

    conflict_adj: List[Set[int]] = [set() for _ in range(m)]
    for c_list in port_to_cands.values():
        if len(c_list) >= 2:
            for ca in c_list:
                conflict_adj[ca].update(c_list)
    for idx_self in range(m):
        conflict_adj[idx_self].discard(idx_self)

    base_cells: Dict[Tuple[int, int, int], List[int]] = {}
    for i in range(m):
        for b in candidates[i].world_boxes:
            for ck in SpatialBoxHashIndex.box_cells(b):
                base_cells.setdefault(ck, []).append(i)

    base_nbrs: List[Set[int]] = [set() for _ in range(m)]
    for bucket in base_cells.values():
        bl = len(bucket)
        for ia in range(bl):
            ca = bucket[ia]
            for ib in range(ia + 1, bl):
                cb = bucket[ib]
                if cb not in conflict_adj[ca]:
                    if ca < cb:
                        base_nbrs[ca].add(cb)
                    else:
                        base_nbrs[cb].add(ca)

    hull_conflicts: List[Tuple[int, int]] = []
    for i in range(m):
        bi = candidates[i].world_boxes
        for j in sorted(base_nbrs[i]):
            bj = candidates[j].world_boxes
            if any(ba.intersects(bb) for ba in bi for bb in bj):
                hull_conflicts.append((i, j))
                conflict_adj[i].add(j)
                conflict_adj[j].add(i)

    return CandidatePool(
        candidates=tuple(candidates),
        port_to_candidates={k: tuple(v) for k, v in port_to_cands.items()},
        conflict_edges=tuple(hull_conflicts),
        conflict_adjacency=tuple(frozenset(s) for s in conflict_adj),
        filtered_target_id=id(target_field),
    )
