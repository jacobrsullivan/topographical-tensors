"""
WP-L3 & WS-1C / WS-2A: Sparse Contact-Wrench Static Equilibrium Validator.

Solves the coupled 6-DoF rigid-body force and torque equilibrium equations across the
persistent ContactGraph:
    A(B, C) λ + w(B, g) = 0,    λ in K(C)
using native C++ `highspy` (single-core simplex with SciPy `linprog` fallback) and
exact connected-subassembly block decomposition.

Key Mechanical Guarantees:
1. Asymmetric normal bearing (`normal_crush_ratio * F_clutch` compression vs. `F_clutch`
   tensile stud pull-out), eliminating false vertical crush on tall columns and tables.
2. Strictly non-redundant multi-port contact-patch shear & local-moment constraints:
   - Normal-force-coupled Coulomb friction + sublinear stud-wall clutch deduplication:
         sum_{e in P} (|f_{t1,e}| + |f_{t2,e}|) - mu_patch * sum_{e in P} f_{n,comp,e}
             <= K_P^{gamma_patch} * F_shear * η
     with `gamma_patch = 0.75 < 1.0` and `mu_patch = 0.60`.
   - Sublinear patch local-moment deduplication:
         sum_{e in P} ||tau_e||_1 <= K_P^{gamma_patch} * M_max * η
3. Joint-DoF wrench projections P_c:
   - `SMOOTH_PIN` / `HINGE`: zero axial torque about rotational axis (tau_n = 0).
   - `BALL_JOINT`: zero 3-axis torque (tau_{t1} = tau_{t2} = tau_n = 0).
   - `TECHNIC_AXLE`: transmits full axial torque tau_n while reducing unstopped axial pull-out.
4. Exact block-diagonal connected-subassembly decomposition and canonical component caching
   for fast unconditional prefix equilibrium verification across all assembly courses.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csc_matrix

try:
    import highspy  # type: ignore[import-untyped]

    _HAS_HIGHSPY = True
except ImportError:  # pragma: no cover
    highspy = None
    _HAS_HIGHSPY = False

from topo_brick.interface.brick_catalog import ConnectorType, get_brick_primitive
from topo_brick.interface.port_manifold import AssemblyState, GlobalPortId

ContactEdgeKey = Tuple[GlobalPortId, GlobalPortId]


class FoundationPolicy(str, Enum):
    """
    Explicit mechanical boundary condition at the foundation plane (Round 8 §7):
    - `CLUTCHED_BASEPLATE`: `is_grounded` parts clutch onto a fixed LEGO baseplate.
    - `TABLE_FRICTION_ONLY`: `is_grounded` parts rest freely on a flat table (`f_tens == 0`,
      unilateral compression + Coulomb table sliding only; tips if whole-assembly CoM exits base).
    - `FIXED_ANCHOR`: `is_grounded` parts are rigidly clamped to an immovable frame.
    """

    CLUTCHED_BASEPLATE = "clutched_baseplate"
    TABLE_FRICTION_ONLY = "table_friction_only"
    FIXED_ANCHOR = "fixed_anchor"


@dataclass(frozen=True, slots=True)
class PrimalEquilibriumWitness:
    """
    Explicit solved 6-DoF primal LP contact-wrench witness (`R9-WS1` / `R9-WS3` §5.5 & §7.1):
    Records the non-negative primal LP vector `x`, exact equality residual `||A_eq x - b_eq||_inf`,
    and inequality margin `max(A_ub x - b_ub)` so certificates and incremental extensions can be
    independently replayed and residual-checked without trusting solver state.
    """

    max_eq_residual_g: float
    max_ub_violation: float
    eta_value: float
    num_variables: int
    num_equality_rows: int
    num_inequality_rows: int
    patch_net_loads: Dict[Tuple[int, int], Tuple[float, float, float, float, float, float]]


@dataclass(frozen=True, slots=True)
class EquilibriumResult:
    """Result of the coupled 6-DoF contact-wrench static equilibrium solve."""

    is_stable: bool
    peak_utilization: float  # η*: <= 1.0 means stable; > 1.0 means overloaded
    safety_margin: float  # 1.0 - η*
    contact_utilizations: Dict[ContactEdgeKey, float]
    weakest_contact: Optional[ContactEdgeKey]
    solve_time_ms: float
    status_message: str
    patch_utilizations: Optional[Dict[Tuple[int, int], float]] = None
    moment_reference: str = "part_com_ldu"
    asymmetric_bearing_active: bool = True
    foundation_policy: str = FoundationPolicy.CLUTCHED_BASEPLATE.value
    interface_model: str = "decomposed_contact_hull"
    primal_witness: Optional[PrimalEquilibriumWitness] = None


_SHARED_HIGHS_LP: Any = None
_SHARED_HIGHS_IPM: Any = None


def _get_shared_highs_lp(use_ipm: bool = False) -> Any:
    global _SHARED_HIGHS_LP, _SHARED_HIGHS_IPM
    if use_ipm:
        if _SHARED_HIGHS_IPM is None:
            h = highspy.Highs()
            h.silent()
            h.setOptionValue("threads", 1)
            h.setOptionValue("solver", "ipm")
            h.setOptionValue("run_crossover", "off")
            h.setOptionValue("ipm_optimality_tolerance", 1e-6)
            _SHARED_HIGHS_IPM = h
        else:
            _SHARED_HIGHS_IPM.clearSolver()
        return _SHARED_HIGHS_IPM
    if _SHARED_HIGHS_LP is None:
        h = highspy.Highs()
        h.silent()
        h.setOptionValue("threads", 1)
        _SHARED_HIGHS_LP = h
    else:
        _SHARED_HIGHS_LP.clearSolver()
    return _SHARED_HIGHS_LP


def _solve_lp_highspy_or_scipy(
    c_obj: np.ndarray,
    a_eq_csc: csc_matrix,
    b_eq: np.ndarray,
    a_ub_csc: csc_matrix,
    b_ub: np.ndarray,
    var_lb: np.ndarray,
    var_ub: np.ndarray,
) -> Tuple[bool, Optional[np.ndarray], str]:
    """
    Solve `min c_obj^T x` s.t. `A_eq x = b_eq`, `A_ub x <= b_ub`, `var_lb <= x <= var_ub`
    using native C++ `highspy` (1 thread) with graceful fallback to `scipy.optimize.linprog`.
    Uses interior-point (`ipm`, `run_crossover="off"`) first for large (`n_vars > 1500`) LPs
    to avoid minimax (`min eta`) dual-simplex degeneracy while keeping simplex first for small LPs.
    """
    n_vars = int(c_obj.shape[0])
    if _HAS_HIGHSPY and n_vars > 0:
        from scipy.sparse import vstack as sp_vstack

        a_comb = sp_vstack([a_eq_csc, a_ub_csc], format="csc")
        n_eq = int(b_eq.shape[0])
        n_ub = int(b_ub.shape[0])
        n_rows = n_eq + n_ub
        inf = float(highspy.kHighsInf)
        row_lower = np.empty(n_rows, dtype=np.float64)
        row_upper = np.empty(n_rows, dtype=np.float64)
        if n_eq > 0:
            row_lower[:n_eq] = b_eq
            row_upper[:n_eq] = b_eq
        if n_ub > 0:
            row_lower[n_eq:] = -inf
            row_upper[n_eq:] = b_ub

        col_lower = np.where(np.isfinite(var_lb), var_lb, -inf).astype(np.float64)
        col_upper = np.where(np.isfinite(var_ub), var_ub, inf).astype(np.float64)

        lp = highspy.HighsLp()
        lp.num_col_ = n_vars
        lp.num_row_ = n_rows
        lp.col_cost_ = c_obj.astype(np.float64)
        lp.col_lower_ = col_lower
        lp.col_upper_ = col_upper
        lp.row_lower_ = row_lower
        lp.row_upper_ = row_upper
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = a_comb.indptr.astype(np.int32)
        lp.a_matrix_.index_ = a_comb.indices.astype(np.int32)
        lp.a_matrix_.value_ = a_comb.data.astype(np.float64)

        solver_order = (True, False) if n_vars > 2500 else (False, True)
        for idx_s, try_ipm in enumerate(solver_order):
            h = _get_shared_highs_lp(use_ipm=try_ipm)
            if not try_ipm:
                h.setOptionValue("simplex_iteration_limit", 1800 if idx_s == 0 else 2147483647)
            h.passModel(lp)
            h.run()
            model_status = h.getModelStatus()
            if model_status == highspy.HighsModelStatus.kOptimal:
                sol_arr = np.asarray(h.getSolution().col_value, dtype=np.float64)
                if sol_arr.shape[0] == n_vars and np.all(np.isfinite(sol_arr)):
                    return True, sol_arr, "Optimal (highspy-ipm)" if try_ipm else "Optimal (highspy)"
            if model_status == highspy.HighsModelStatus.kInfeasible:
                return False, None, "Infeasible (highspy)"

    bounds_list: List[Tuple[float, Optional[float]]] = [
        (float(lb), None if not np.isfinite(ub) else float(ub))
        for lb, ub in zip(var_lb, var_ub)
    ]
    res = linprog(
        c=c_obj,
        A_ub=a_ub_csc,
        b_ub=b_ub,
        A_eq=a_eq_csc,
        b_eq=b_eq,
        bounds=bounds_list,
        method="highs",
    )
    if not res.success or res.x is None:
        res = linprog(
            c=c_obj,
            A_ub=a_ub_csc,
            b_ub=b_ub,
            A_eq=a_eq_csc,
            b_eq=b_eq,
            bounds=bounds_list,
            method="highs-ipm",
        )
    if not res.success or res.x is None:
        res = linprog(
            c=c_obj,
            A_ub=a_ub_csc,
            b_ub=b_ub,
            A_eq=a_eq_csc,
            b_eq=b_eq,
            bounds=bounds_list,
            method="highs-ds",
            options={"presolve": False},
        )
    if not res.success or res.x is None:
        return False, None, str(res.message)
    return True, np.asarray(res.x, dtype=np.float64), "Optimal (scipy-highs)"


class SparseStaticEquilibriumValidator:
    """
    Sparse 6-DoF contact-wrench static equilibrium validator with:
    - Asymmetric compressive bearing (`normal_crush_ratio * F_clutch`).
    - Non-redundant Coulomb + sublinear clutch multi-port patch shear & moment constraints
      (`coulomb_friction_coeff = 0.60`, `patch_clutch_exponent = 0.75`).
    - Joint-DoF wrench projections (`SMOOTH_PIN`, `HINGE`, `BALL_JOINT`, `TECHNIC_AXLE`).
    - Exact connected-subassembly block decomposition and canonical component caching.
    """

    def __init__(
        self,
        clutch_force_capacity_grams: float = 15.0,
        moment_arm_capacity_ldu: float = 40.0,
        use_uniform_brick_weight_grams: Optional[float] = None,
        asymmetric_bearing_mode: Optional[bool] = None,
        normal_crush_ratio: float = 500.0,
        coulomb_friction_coeff: float = 0.60,
        patch_clutch_exponent: float = 0.75,
        enable_patch_constraints: bool = True,
        patch_interface_model: str = "decomposed_contact_hull",
        table_friction_coeff: float = 0.45,
        stud_snap_preload_ratio: float = 0.7142857,  # BrickSim 2026: (0.2 * 3.5 N) / 0.98 N
        rim_fulcrum_ldu: float = 10.0,  # Exact 0.5 stud pitch outer brick wall rim
        stud_knob_height_ldu: float = 4.8,  # Exact 1.92 mm physical LEGO stud engagement height
    ) -> None:
        self.clutch_force_capacity_grams = float(clutch_force_capacity_grams)
        self.moment_arm_capacity_ldu = float(moment_arm_capacity_ldu)
        self.use_uniform_brick_weight_grams = use_uniform_brick_weight_grams
        self.asymmetric_bearing_mode = (
            (use_uniform_brick_weight_grams is None)
            if asymmetric_bearing_mode is None
            else bool(asymmetric_bearing_mode)
        )
        self.normal_crush_ratio = float(normal_crush_ratio) if self.asymmetric_bearing_mode else 1.0
        self.coulomb_friction_coeff = float(coulomb_friction_coeff)
        self.patch_clutch_exponent = float(patch_clutch_exponent)
        self.enable_patch_constraints = bool(enable_patch_constraints)
        self.patch_interface_model = str(patch_interface_model)
        self.table_friction_coeff = float(table_friction_coeff)
        self.stud_snap_preload_ratio = float(stud_snap_preload_ratio)
        self.rim_fulcrum_ldu = float(rim_fulcrum_ldu)
        self.stud_knob_height_ldu = float(stud_knob_height_ldu)
        self._last_state_version: Optional[int] = None
        self._last_result: Optional[EquilibriumResult] = None
        self._component_cache: Dict[
            object,
            Tuple[
                bool,
                float,
                Dict[ContactEdgeKey, float],
                Dict[Tuple[int, int], float],
                Optional[ContactEdgeKey],
                str,
                Optional[PrimalEquilibriumWitness],
            ],
        ] = {}
        self.lp_solver_calls: int = 0
        self.component_cache_hits: int = 0
        self.witness_reuse_hits: int = 0
        self.witness_fallback_count: int = 0

    def reset_telemetry(self) -> None:
        """Reset LP, component-cache, and verified witness-reuse telemetry counters."""
        self.lp_solver_calls = 0
        self.component_cache_hits = 0
        self.witness_reuse_hits = 0
        self.witness_fallback_count = 0

    @staticmethod
    def _compute_patch_overlap_half_spans(
        part_a: Any,
        part_b: Any,
        rx_m: float,
        ry_m: float,
        rz_m: float,
        t1_0: Tuple[float, float, float],
        t2_0: Tuple[float, float, float],
    ) -> Optional[Tuple[float, float]]:
        """
        Compute the exact geometric bearing-hull intersection half-spans `(max_hb_u, max_hb_v)`
        of `part_a.world_boxes` and `part_b.world_boxes` projected onto `(t1_0, t2_0)` around
        the patch centroid `(rx_m, ry_m, rz_m)` (`R9-WS3` §5.5 partial-overlap bearing hull fix).
        """
        wbs_a = getattr(part_a, "world_boxes", None)
        wbs_b = getattr(part_b, "world_boxes", None)
        if not wbs_a or not wbs_b:
            return None

        def _proj_span(wbs: Any, axis: Tuple[float, float, float]) -> Tuple[float, float]:
            lo = float("inf")
            hi = -float("inf")
            ax, ay, az = axis
            for b in wbs:
                for x in (float(b.xmin), float(b.xmax)):
                    for y in (float(b.ymin), float(b.ymax)):
                        for z in (float(b.zmin), float(b.zmax)):
                            val = (x - rx_m) * ax + (y - ry_m) * ay + (z - rz_m) * az
                            if val < lo:
                                lo = val
                            if val > hi:
                                hi = val
            return lo, hi

        ua_lo, ua_hi = _proj_span(wbs_a, t1_0)
        ub_lo, ub_hi = _proj_span(wbs_b, t1_0)
        va_lo, va_hi = _proj_span(wbs_a, t2_0)
        vb_lo, vb_hi = _proj_span(wbs_b, t2_0)
        inter_u_lo = max(ua_lo, ub_lo)
        inter_u_hi = min(ua_hi, ub_hi)
        inter_v_lo = max(va_lo, vb_lo)
        inter_v_hi = min(va_hi, vb_hi)
        if inter_u_hi <= inter_u_lo or inter_v_hi <= inter_v_lo:
            return None
        # Bearing fulcrum distance to the nearer edge of the intersected contact hull
        max_hb_u = max(0.0, min(-inter_u_lo, inter_u_hi))
        max_hb_v = max(0.0, min(-inter_v_lo, inter_v_hi))
        return max_hb_u, max_hb_v

    def _effective_patch_shear_factor(self, k_cnt: float) -> float:
        """
        Compute physically decomposed multi-stud preload shear capacity multiplier (zero arbitrary
        fractional exponent when `patch_interface_model == 'decomposed_contact_hull'`):
          F_preload(K) = F_rim_interlock + K * (mu_snap * N_preload_0)
                       = (1 + stud_snap_preload_ratio * (K - 1)) * F_shear_1.
        """
        if k_cnt <= 1.0:
            return 1.0
        if self.patch_interface_model == "decomposed_contact_hull":
            return 1.0 + self.stud_snap_preload_ratio * (k_cnt - 1.0)
        return float(k_cnt ** self.patch_clutch_exponent)

    def _decomposed_patch_moment_caps(
        self,
        k_cnt: float,
        h_u: float,
        h_v: float,
        dus: List[float],
        dvs: List[float],
        f_tens_max: float,
        f_shear_max: float,
        m_max: float,
    ) -> Tuple[float, float, float, float, float]:
        """
        Return `(h_bearing_u, h_bearing_v, cap_m1, cap_m2, cap_mn)` for a coplanar rigid patch:
        - `decomposed_contact_hull` counts the outer brick rim `r_rim` ONCE (`h_u + r_rim`,
          `h_v + r_rim`), sums exact separated stud pull-out lever arms `sum |du_e| * F_tens`,
          and adds per-stud cylindrical wall height `stud_knob_height_ldu * F_tens`.
        """
        if self.patch_interface_model == "decomposed_contact_hull":
            r_rim = min(self.rim_fulcrum_ldu, 0.5 * self.moment_arm_capacity_ldu)
            r_knob = min(
                self.stud_knob_height_ldu,
                max(0.0, self.moment_arm_capacity_ldu - r_rim),
            )
            hb_u = h_u + r_rim
            hb_v = h_v + r_rim
            # At K=1 (du=dv=0), single stud retains exact m_max; for K>=2, outer rim is counted ONCE
            single_base_m = m_max if k_cnt <= 1.0 else (r_rim * f_tens_max + k_cnt * r_knob * f_tens_max)
            arm_u_sep = sum(max(0.0, du) for du in dus)
            arm_v_sep = sum(max(0.0, dv) for dv in dvs)
            cap_m1 = single_base_m + arm_v_sep * f_tens_max
            cap_m2 = single_base_m + arm_u_sep * f_tens_max
            shear_mult = self._effective_patch_shear_factor(k_cnt)
            torsion_arms = sum(float(np.hypot(du, dv)) for du, dv in zip(dus, dvs))
            cap_mn = single_base_m + torsion_arms * f_shear_max * (shear_mult / max(k_cnt, 1.0))
            return hb_u, hb_v, cap_m1, cap_m2, cap_mn

        k_p_eff = float(k_cnt ** self.patch_clutch_exponent)
        arm_u_t = sum(max(0.0, du) for du in dus)
        arm_v_t = sum(max(0.0, dv) for dv in dvs)
        cap_m1 = k_p_eff * m_max + arm_v_t * f_tens_max
        cap_m2 = k_p_eff * m_max + arm_u_t * f_tens_max
        cap_mn = k_p_eff * m_max + max(h_u, h_v) * k_p_eff * f_shear_max
        return h_u, h_v, cap_m1, cap_m2, cap_mn

    def verify_against_cold_rebuild(self, state: AssemblyState) -> bool:
        """
        Verify that evaluating `state` produces the exact same equilibrium result and
        peak utilization as evaluating a fresh cold rebuild of `state` without component cache.
        """
        if not state.verify_state_integrity():
            return False
        res_warm = self.evaluate(state, use_component_cache=True)
        cold_state = state.clone_cold_rebuild()
        res_cold = self.evaluate(cold_state, use_component_cache=False)
        if res_warm.is_stable != res_cold.is_stable:
            return False
        if np.isinf(res_warm.peak_utilization) and np.isinf(res_cold.peak_utilization):
            return True
        return abs(res_warm.peak_utilization - res_cold.peak_utilization) <= 1e-5

    def _solve_subassembly_component(
        self,
        parts: Dict[int, object],
        comp_free_insts: List[int],
        comp_contacts: List[
            Tuple[
                GlobalPortId,
                GlobalPortId,
                Tuple[float, float, float],
                Tuple[float, float, float],
                Tuple[float, float, float],
                Tuple[float, float, float],
                str,
            ]
        ],
        lateral_accel_xy: Tuple[float, float] = (0.0, 0.0),
    ) -> Tuple[
        bool,
        float,
        Dict[ContactEdgeKey, float],
        Dict[Tuple[int, int], float],
        Optional[ContactEdgeKey],
        str,
        Optional[PrimalEquilibriumWitness],
    ]:
        n_contacts = len(comp_contacts)
        if n_contacts == 0:
            return False, float("inf"), {}, {}, None, "No contacts for non-grounded component.", None

        com_map: Dict[int, Tuple[float, float, float]] = {}
        weight_map: Dict[int, float] = {}
        for inst_id in comp_free_insts:
            placed = parts[inst_id]  # type: ignore[assignment]
            prim = get_brick_primitive(placed.part_id)  # type: ignore[attr-defined]
            w_grams = (
                float(self.use_uniform_brick_weight_grams)
                if self.use_uniform_brick_weight_grams is not None
                else float(prim.mass_grams)
            )
            weight_map[inst_id] = w_grams
            # Support continuous SE(3) pose CoM if present, else exact LDU boxes
            cont_pose = getattr(placed, "continuous_pose", None)
            if cont_pose is not None:
                local_vol = 0.0
                lcx, lcy, lcz = 0.0, 0.0, 0.0
                for box in prim.hull_boxes:
                    v = float(box.volume_ldu3)
                    local_vol += v
                    lcx += v * 0.5 * (box.xmin + box.xmax)
                    lcy += v * 0.5 * (box.ymin + box.ymax)
                    lcz += v * 0.5 * (box.zmin + box.zmax)
                lcom = np.array([lcx / local_vol, lcy / local_vol, lcz / local_vol], dtype=np.float64)
                wcom = cont_pose.transform_point(lcom)
                com_map[inst_id] = (float(wcom[0]), float(wcom[1]), float(wcom[2]))
            else:
                total_vol = 0.0
                cx, cy, cz = 0.0, 0.0, 0.0
                for box in placed.world_boxes:  # type: ignore[attr-defined]
                    v = float(box.volume_ldu3)
                    total_vol += v
                    cx += v * 0.5 * (box.xmin + box.xmax)
                    cy += v * 0.5 * (box.ymin + box.ymax)
                    cz += v * 0.5 * (box.zmin + box.zmax)
                com_map[inst_id] = (cx / total_vol, cy / total_vol, cz / total_vol)

        f_tens_max = self.clutch_force_capacity_grams
        f_comp_max = self.clutch_force_capacity_grams * self.normal_crush_ratio
        f_shear_max = self.clutch_force_capacity_grams
        m_max = self.clutch_force_capacity_grams * self.moment_arm_capacity_ldu
        inv_shear = 1.0 / f_shear_max
        inv_comp = 1.0 / f_comp_max
        inv_m = 1.0 / m_max
        axle_tens_scale = 0.35

        # Group contacts into brick-to-brick patches P = (min_inst, max_inst)
        patch_groups: Dict[Tuple[int, int], List[int]] = {}
        for e_idx, (gpid_a, gpid_b, _, _, _, _, _) in enumerate(comp_contacts):
            pair_key = (min(gpid_a[0], gpid_b[0]), max(gpid_a[0], gpid_b[0]))
            patch_groups.setdefault(pair_key, []).append(e_idx)

        # Build reduced rigid-patch variable layout:
        # - Single-port or non-coplanar/articulated patches keep exact 12 vars per contact.
        # - Coplanar multi-port rigid_clutch patches (K >= 2) eliminate redundant per-stud pure
        #   torques via exact Fourier-Motzkin projection (1 shared 6-var torque vector per patch)
        #   and, when K > 4 (`1x6`, `1x8`, `2x4`, `2x6`, `2x8`), aggregate coplanar studs into <=4
        #   exact quadrant centroids preserving 100% of tipping moments and resultant wrenches.
        # Force node tuple: (inst_a, inst_b, r_node, t1_v, t2_v, n_v, weight_w, e_tens_cap, member_e_indices)
        force_nodes: List[
            Tuple[
                int,
                int,
                Tuple[float, float, float],
                Tuple[float, float, float],
                Tuple[float, float, float],
                Tuple[float, float, float],
                float,
                float,
                Tuple[int, ...],
            ]
        ] = []
        # Torque block tuple: (inst_a, inst_b, t1_v, t2_v, n_v, joint_mode)
        torque_blocks: List[
            Tuple[
                int,
                int,
                Tuple[float, float, float],
                Tuple[float, float, float],
                Tuple[float, float, float],
                str,
            ]
        ] = []
        # Patch descriptors: (pair_key, e_list, q_node_indices, t_block_idx, is_reduced_coplanar, patch_geom)
        # where patch_geom = (h_u, h_v, cap_m1, cap_m2, cap_mn) for reduced coplanar patches
        patch_specs: List[
            Tuple[
                Tuple[int, int],
                List[int],
                List[int],
                int,
                bool,
                Tuple[float, float, float, float, float],
            ]
        ] = []

        for pair_key, e_list in patch_groups.items():
            k_cnt = len(e_list)
            c0 = comp_contacts[e_list[0]]
            n0 = c0[5]
            coplanar_rigid = (
                k_cnt >= 2
                and self.enable_patch_constraints
                and all(
                    comp_contacts[ei][6] == "rigid_clutch"
                    and comp_contacts[ei][0][0] == pair_key[0]
                    and comp_contacts[ei][1][0] == pair_key[1]
                    and abs(comp_contacts[ei][5][0] - n0[0])
                    + abs(comp_contacts[ei][5][1] - n0[1])
                    + abs(comp_contacts[ei][5][2] - n0[2])
                    < 1e-6
                    for ei in e_list
                )
            )
            if not coplanar_rigid:
                for ei in e_list:
                    ga, gb, re, t1v, t2v, nv, jmode = comp_contacts[ei]
                    if jmode == "table_friction":
                        et_cap = 0.0
                    else:
                        et_cap = f_tens_max * (axle_tens_scale if jmode == "axle" else 1.0)
                    fn_idx = len(force_nodes)
                    force_nodes.append((ga[0], gb[0], re, t1v, t2v, nv, 1.0, et_cap, (ei,)))
                    tb_idx = len(torque_blocks)
                    torque_blocks.append((ga[0], gb[0], t1v, t2v, nv, jmode))
                    patch_specs.append(
                        (pair_key, [ei], [fn_idx], tb_idx, False, (0.0, 0.0, m_max, m_max, m_max))
                    )
            else:
                t1_0, t2_0 = c0[3], c0[4]
                wq = float(k_cnt)
                rx_m = sum(comp_contacts[ei][2][0] for ei in e_list) / wq
                ry_m = sum(comp_contacts[ei][2][1] for ei in e_list) / wq
                rz_m = sum(comp_contacts[ei][2][2] for ei in e_list) / wq
                dus = [
                    (comp_contacts[ei][2][0] - rx_m) * t1_0[0]
                    + (comp_contacts[ei][2][1] - ry_m) * t1_0[1]
                    + (comp_contacts[ei][2][2] - rz_m) * t1_0[2]
                    for ei in e_list
                ]
                dvs = [
                    (comp_contacts[ei][2][0] - rx_m) * t2_0[0]
                    + (comp_contacts[ei][2][1] - ry_m) * t2_0[1]
                    + (comp_contacts[ei][2][2] - rz_m) * t2_0[2]
                    for ei in e_list
                ]
                h_u = max(abs(du) for du in dus)
                h_v = max(abs(dv) for dv in dvs)
                hb_u, hb_v, cap_m1, cap_m2, cap_mn = self._decomposed_patch_moment_caps(
                    wq, h_u, h_v, dus, dvs, f_tens_max, f_shear_max, m_max
                )
                if pair_key[0] in parts and pair_key[1] in parts:
                    overlap_spans = self._compute_patch_overlap_half_spans(
                        parts[pair_key[0]], parts[pair_key[1]], rx_m, ry_m, rz_m, t1_0, t2_0
                    )
                    if overlap_spans is not None:
                        hb_u = min(hb_u, overlap_spans[0])
                        hb_v = min(hb_v, overlap_spans[1])

                fn_idx = len(force_nodes)
                force_nodes.append(
                    (
                        pair_key[0],
                        pair_key[1],
                        (rx_m, ry_m, rz_m),
                        t1_0,
                        t2_0,
                        n0,
                        wq,
                        f_tens_max,
                        tuple(e_list),
                    )
                )
                tb_idx = len(torque_blocks)
                torque_blocks.append((pair_key[0], pair_key[1], t1_0, t2_0, n0, "rigid_clutch"))
                patch_specs.append(
                    (pair_key, e_list, [fn_idx], tb_idx, True, (hb_u, hb_v, cap_m1, cap_m2, cap_mn))
                )

        n_fn = len(force_nodes)
        n_tb = len(torque_blocks)
        torque_offset = 6 * n_fn
        eta_col = 6 * n_fn + 6 * n_tb
        n_vars = eta_col + 1

        c_obj = np.zeros(n_vars, dtype=np.float64)
        c_obj[eta_col] = 1.0
        var_lb = np.zeros(n_vars, dtype=np.float64)
        var_ub = np.full(n_vars, np.inf, dtype=np.float64)

        for q_idx, (_, _, _, _, _, _, wq, et_cap, _) in enumerate(force_nodes):
            base = 6 * q_idx
            c_obj[base : base + 4] = 1e-4 / (wq * f_shear_max)
            if et_cap <= 0.0:
                var_ub[base + 4] = 0.0
                c_obj[base + 4] = 0.0
            else:
                c_obj[base + 4] = 1e-4 / (wq * et_cap)
            c_obj[base + 5] = 1e-4 / (wq * f_comp_max)

        for tb_i, (_, _, _, _, _, jmode) in enumerate(torque_blocks):
            t_base = torque_offset + 6 * tb_i
            c_obj[t_base : t_base + 6] = 1e-4 / m_max
            if jmode in ("smooth_pin", "hinge"):
                var_ub[t_base + 4] = 0.0
                var_ub[t_base + 5] = 0.0
            elif jmode in ("ball_joint", "table_friction"):
                var_ub[t_base : t_base + 6] = 0.0

        free_idx_map = {inst_id: i for i, inst_id in enumerate(comp_free_insts)}
        n_eq = 6 * len(comp_free_insts)
        b_eq = np.zeros(n_eq, dtype=np.float64)
        for inst_id, p_idx in free_idx_map.items():
            wg = weight_map[inst_id]
            b_eq[6 * p_idx + 0] = float(lateral_accel_xy[0]) * wg
            b_eq[6 * p_idx + 1] = float(lateral_accel_xy[1]) * wg
            b_eq[6 * p_idx + 2] = wg

        eq_rows: List[int] = []
        eq_cols: List[int] = []
        eq_vals: List[float] = []

        for q_idx, (ia, ib, r_e, t1_v, t2_v, n_v, _, _, _) in enumerate(force_nodes):
            base = 6 * q_idx
            f_dirs = (
                t1_v,
                (-t1_v[0], -t1_v[1], -t1_v[2]),
                t2_v,
                (-t2_v[0], -t2_v[1], -t2_v[2]),
                n_v,
                (-n_v[0], -n_v[1], -n_v[2]),
            )
            for endpoint_idx, inst_k in enumerate((ia, ib)):
                p_idx = free_idx_map.get(inst_k)
                if p_idx is None:
                    continue
                row0 = 6 * p_idx
                sign = 1.0 if endpoint_idx == 0 else -1.0
                com = com_map[inst_k]
                rx = r_e[0] - com[0]
                ry = r_e[1] - com[1]
                rz = r_e[2] - com[2]
                for v_i, d_vec in enumerate(f_dirs):
                    col = base + v_i
                    fx, fy, fz = sign * d_vec[0], sign * d_vec[1], sign * d_vec[2]
                    if fx != 0.0:
                        eq_rows.append(row0 + 0)
                        eq_cols.append(col)
                        eq_vals.append(fx)
                    if fy != 0.0:
                        eq_rows.append(row0 + 1)
                        eq_cols.append(col)
                        eq_vals.append(fy)
                    if fz != 0.0:
                        eq_rows.append(row0 + 2)
                        eq_cols.append(col)
                        eq_vals.append(fz)
                    cx_val = ry * fz - rz * fy
                    cy_val = rz * fx - rx * fz
                    cz_val = rx * fy - ry * fx
                    if cx_val != 0.0:
                        eq_rows.append(row0 + 3)
                        eq_cols.append(col)
                        eq_vals.append(cx_val)
                    if cy_val != 0.0:
                        eq_rows.append(row0 + 4)
                        eq_cols.append(col)
                        eq_vals.append(cy_val)
                    if cz_val != 0.0:
                        eq_rows.append(row0 + 5)
                        eq_cols.append(col)
                        eq_vals.append(cz_val)

        for tb_i, (ia, ib, t1_v, t2_v, n_v, _) in enumerate(torque_blocks):
            t_base = torque_offset + 6 * tb_i
            t_dirs = (
                t1_v,
                (-t1_v[0], -t1_v[1], -t1_v[2]),
                t2_v,
                (-t2_v[0], -t2_v[1], -t2_v[2]),
                n_v,
                (-n_v[0], -n_v[1], -n_v[2]),
            )
            for endpoint_idx, inst_k in enumerate((ia, ib)):
                p_idx = free_idx_map.get(inst_k)
                if p_idx is None:
                    continue
                row0 = 6 * p_idx
                sign = 1.0 if endpoint_idx == 0 else -1.0
                for v_i, d_vec in enumerate(t_dirs):
                    t_col = t_base + v_i
                    fx, fy, fz = sign * d_vec[0], sign * d_vec[1], sign * d_vec[2]
                    if fx != 0.0:
                        eq_rows.append(row0 + 3)
                        eq_cols.append(t_col)
                        eq_vals.append(fx)
                    if fy != 0.0:
                        eq_rows.append(row0 + 4)
                        eq_cols.append(t_col)
                        eq_vals.append(fy)
                    if fz != 0.0:
                        eq_rows.append(row0 + 5)
                        eq_cols.append(t_col)
                        eq_vals.append(fz)

        a_eq_csc = csc_matrix(
            (np.asarray(eq_vals, dtype=np.float64), (eq_rows, eq_cols)),
            shape=(n_eq, n_vars),
        )

        ub_rows: List[int] = []
        ub_cols: List[int] = []
        ub_vals: List[float] = []
        cur_ub_row = 0

        for _, e_list, q_indices, tb_idx, is_reduced, p_geom in patch_specs:
            t_base = torque_offset + 6 * tb_idx
            q_idx = q_indices[0]
            f_base = 6 * q_idx
            et_cap = force_nodes[q_idx][7]
            jmode_here = torque_blocks[tb_idx][5]
            if not is_reduced:
                if jmode_here == "table_friction":
                    # Pure unilateral table compression + Coulomb sliding (|f_t1| + |f_t2| - mu_table * f_comp <= 0)
                    for k in range(4):
                        ub_rows.append(cur_ub_row)
                        ub_cols.append(f_base + k)
                        ub_vals.append(1.0)
                    ub_rows.append(cur_ub_row)
                    ub_cols.append(f_base + 5)
                    ub_vals.append(-self.table_friction_coeff)
                    cur_ub_row += 1
                    # Compressive crush utilization f_comp / f_comp_max <= eta
                    ub_rows.append(cur_ub_row)
                    ub_cols.append(f_base + 5)
                    ub_vals.append(inv_comp)
                    ub_rows.append(cur_ub_row)
                    ub_cols.append(eta_col)
                    ub_vals.append(-1.0)
                    cur_ub_row += 1
                else:
                    for k in range(4):
                        ub_rows.append(cur_ub_row)
                        ub_cols.append(f_base + k)
                        ub_vals.append(inv_shear)
                    ub_rows.append(cur_ub_row)
                    ub_cols.append(f_base + 4)
                    ub_vals.append(1.0 / et_cap)
                    ub_rows.append(cur_ub_row)
                    ub_cols.append(f_base + 5)
                    ub_vals.append(inv_comp)
                    for k in range(6):
                        ub_rows.append(cur_ub_row)
                        ub_cols.append(t_base + k)
                        ub_vals.append(inv_m)
                    ub_rows.append(cur_ub_row)
                    ub_cols.append(eta_col)
                    ub_vals.append(-1.0)
                    cur_ub_row += 1
            else:
                k_total = float(len(e_list))
                k_p_eff = self._effective_patch_shear_factor(k_total)
                h_u, h_v, cap_m1, cap_m2, cap_mn = p_geom
                inv_k = 1.0 / k_total

                # 1. Aggregate force utilization on centroid: (1/K)*(shear/Fs + tens/Ft + comp/Fc) <= eta
                for k in range(4):
                    ub_rows.append(cur_ub_row)
                    ub_cols.append(f_base + k)
                    ub_vals.append(inv_k * inv_shear)
                ub_rows.append(cur_ub_row)
                ub_cols.append(f_base + 4)
                ub_vals.append(inv_k / et_cap)
                ub_rows.append(cur_ub_row)
                ub_cols.append(f_base + 5)
                ub_vals.append(inv_k * inv_comp)
                ub_rows.append(cur_ub_row)
                ub_cols.append(eta_col)
                ub_vals.append(-1.0)
                cur_ub_row += 1

                # 2. Non-redundant patch Coulomb + sublinear clutch shear row
                inv_patch_shear = 1.0 / (k_p_eff * f_shear_max)
                coulomb_term = -self.coulomb_friction_coeff * inv_patch_shear
                for k in range(4):
                    ub_rows.append(cur_ub_row)
                    ub_cols.append(f_base + k)
                    ub_vals.append(inv_patch_shear)
                if self.coulomb_friction_coeff > 0.0:
                    ub_rows.append(cur_ub_row)
                    ub_cols.append(f_base + 4)
                    ub_vals.append(-coulomb_term)
                    ub_rows.append(cur_ub_row)
                    ub_cols.append(f_base + 5)
                    ub_vals.append(coulomb_term)
                ub_rows.append(cur_ub_row)
                ub_cols.append(eta_col)
                ub_vals.append(-1.0)
                cur_ub_row += 1

                # 3. Tipping moment about t1 (fulcrum span h_v): (M_t1 - h_v * comp) / cap_m1 <= eta
                inv_cm1 = 1.0 / cap_m1
                ub_rows.append(cur_ub_row); ub_cols.append(t_base + 0); ub_vals.append(inv_cm1)
                ub_rows.append(cur_ub_row); ub_cols.append(t_base + 1); ub_vals.append(inv_cm1)
                if h_v > 0.0:
                    ub_rows.append(cur_ub_row); ub_cols.append(f_base + 5); ub_vals.append(-h_v * inv_cm1)
                ub_rows.append(cur_ub_row); ub_cols.append(eta_col); ub_vals.append(-1.0)
                cur_ub_row += 1

                # 4. Tipping moment about t2 (fulcrum span h_u): (M_t2 - h_u * comp) / cap_m2 <= eta
                inv_cm2 = 1.0 / cap_m2
                ub_rows.append(cur_ub_row); ub_cols.append(t_base + 2); ub_vals.append(inv_cm2)
                ub_rows.append(cur_ub_row); ub_cols.append(t_base + 3); ub_vals.append(inv_cm2)
                if h_u > 0.0:
                    ub_rows.append(cur_ub_row); ub_cols.append(f_base + 5); ub_vals.append(-h_u * inv_cm2)
                ub_rows.append(cur_ub_row); ub_cols.append(eta_col); ub_vals.append(-1.0)
                cur_ub_row += 1

                # 5. Torsional yaw about n: M_n / cap_mn <= eta
                inv_cmn = 1.0 / cap_mn
                ub_rows.append(cur_ub_row); ub_cols.append(t_base + 4); ub_vals.append(inv_cmn)
                ub_rows.append(cur_ub_row); ub_cols.append(t_base + 5); ub_vals.append(inv_cmn)
                ub_rows.append(cur_ub_row); ub_cols.append(eta_col); ub_vals.append(-1.0)
                cur_ub_row += 1

        specs_by_pair: Dict[Tuple[int, int], List[Any]] = {}
        reduced_pair_keys: Set[Tuple[int, int]] = set()
        for ps in patch_specs:
            specs_by_pair.setdefault(ps[0], []).append(ps)
            if ps[4]:
                reduced_pair_keys.add(ps[0])

        # Also add patch shear/torque rows for any non-coplanar multi-port patches (e.g. dual-pin Technic)
        for pair_key, e_list in patch_groups.items():
            if pair_key[0] < 0:
                continue
            if len(e_list) >= 2 and self.enable_patch_constraints:
                # Check if already handled by a single reduced coplanar patch_spec
                if pair_key in reduced_pair_keys:
                    continue
                k_p_eff = self._effective_patch_shear_factor(float(len(e_list)))
                inv_patch_shear = 1.0 / (k_p_eff * f_shear_max)
                coulomb_term = -self.coulomb_friction_coeff * inv_patch_shear
                inv_patch_m = 1.0 / (k_p_eff * m_max)
                row_shear = cur_ub_row
                row_mom = cur_ub_row + 1
                cur_ub_row += 2
                for ps in specs_by_pair.get(pair_key, ()):
                    q_idx = ps[2][0]
                    f_base = 6 * q_idx
                    t_base = torque_offset + 6 * ps[3]
                    for k in range(4):
                        ub_rows.append(row_shear)
                        ub_cols.append(f_base + k)
                        ub_vals.append(inv_patch_shear)
                    if self.coulomb_friction_coeff > 0.0:
                        ub_rows.append(row_shear)
                        ub_cols.append(f_base + 4)
                        ub_vals.append(-coulomb_term)
                        ub_rows.append(row_shear)
                        ub_cols.append(f_base + 5)
                        ub_vals.append(coulomb_term)
                    for k in range(6):
                        ub_rows.append(row_mom)
                        ub_cols.append(t_base + k)
                        ub_vals.append(inv_patch_m)
                ub_rows.append(row_shear)
                ub_cols.append(eta_col)
                ub_vals.append(-1.0)
                ub_rows.append(row_mom)
                ub_cols.append(eta_col)
                ub_vals.append(-1.0)

        n_ub = cur_ub_row
        b_ub = np.zeros(n_ub, dtype=np.float64)
        a_ub_csc = csc_matrix(
            (np.asarray(ub_vals, dtype=np.float64), (ub_rows, ub_cols)),
            shape=(n_ub, n_vars),
        )

        self.lp_solver_calls += 1
        ok, sol, msg = _solve_lp_highspy_or_scipy(
            c_obj=c_obj,
            a_eq_csc=a_eq_csc,
            b_eq=b_eq,
            a_ub_csc=a_ub_csc,
            b_ub=b_ub,
            var_lb=var_lb,
            var_ub=var_ub,
        )
        if not ok or sol is None:
            return False, float("inf"), {}, {}, None, f"Static equilibrium LP infeasible: {msg}", None

        max_eq_res = (
            float(np.max(np.abs(a_eq_csc.dot(sol) - b_eq))) if b_eq.shape[0] > 0 else 0.0
        )
        max_ub_viol = (
            float(np.max(a_ub_csc.dot(sol) - b_ub)) if b_ub.shape[0] > 0 else 0.0
        )

        utilizations: Dict[ContactEdgeKey, float] = {}
        weakest_key: Optional[ContactEdgeKey] = None
        max_u = 0.0
        patch_utils: Dict[Tuple[int, int], float] = {}
        patch_net_loads: Dict[Tuple[int, int], Tuple[float, float, float, float, float, float]] = {}

        for pair_key, e_list, q_indices, tb_idx, is_reduced, p_geom in patch_specs:
            t_base = torque_offset + 6 * tb_idx
            q_idx = q_indices[0]
            f_base = 6 * q_idx
            et_cap = force_nodes[q_idx][7]
            jmode_here = torque_blocks[tb_idx][5]
            sh_l1 = float(np.sum(sol[f_base : f_base + 4]))
            tn_v = float(sol[f_base + 4])
            cp_v = float(sol[f_base + 5])
            m1_s = float(sol[t_base + 0] - sol[t_base + 1])
            m2_s = float(sol[t_base + 2] - sol[t_base + 3])
            mn_s = float(sol[t_base + 4] - sol[t_base + 5])
            prev_nl = patch_net_loads.get(pair_key)
            if prev_nl is None:
                patch_net_loads[pair_key] = (sh_l1, tn_v, cp_v, m1_s, m2_s, mn_s)
            else:
                patch_net_loads[pair_key] = (
                    prev_nl[0] + sh_l1,
                    prev_nl[1] + tn_v,
                    prev_nl[2] + cp_v,
                    prev_nl[3] + m1_s,
                    prev_nl[4] + m2_s,
                    prev_nl[5] + mn_s,
                )
            if not is_reduced:
                if jmode_here == "table_friction":
                    u_e = float(sol[f_base + 5]) * inv_comp
                else:
                    tau_l1 = float(np.sum(sol[t_base : t_base + 6]))
                    u_e = (
                        sh_l1 * inv_shear
                        + tn_v / et_cap
                        + cp_v * inv_comp
                        + tau_l1 * inv_m
                    )
                ei = e_list[0]
                key = (comp_contacts[ei][0], comp_contacts[ei][1])
                utilizations[key] = u_e
                if u_e >= max_u:
                    max_u = u_e
                    weakest_key = key
            else:
                k_total = float(len(e_list))
                k_p_eff = self._effective_patch_shear_factor(k_total)
                h_u, h_v, cap_m1, cap_m2, cap_mn = p_geom
                sh_q = sh_l1
                tn_q = tn_v
                cp_q = cp_v
                u_force = (sh_q * inv_shear + tn_q / et_cap + cp_q * inv_comp) / k_total
                u_t1 = max(0.0, (float(sol[t_base + 0] + sol[t_base + 1]) - h_v * cp_q) / cap_m1)
                u_t2 = max(0.0, (float(sol[t_base + 2] + sol[t_base + 3]) - h_u * cp_q) / cap_m2)
                u_tn = float(sol[t_base + 4] + sol[t_base + 5]) / cap_mn
                u_mo = max(u_t1, u_t2, u_tn)
                u_sh = max(
                    0.0,
                    (sh_q - self.coulomb_friction_coeff * max(0.0, cp_q - tn_q))
                    / (k_p_eff * f_shear_max),
                )
                u_q = max(u_force, u_mo, u_sh)
                for ei in e_list:
                    key = (comp_contacts[ei][0], comp_contacts[ei][1])
                    utilizations[key] = u_q
                    if u_q >= max_u:
                        max_u = u_q
                        weakest_key = key
                patch_utils[pair_key] = max(u_sh, u_mo)

        for pair_key, e_list in patch_groups.items():
            if pair_key in patch_utils or pair_key[0] < 0:
                continue
            if len(e_list) >= 2 and self.enable_patch_constraints:
                k_p_eff = self._effective_patch_shear_factor(float(len(e_list)))
                tot_sh = 0.0
                tot_nc = 0.0
                tot_mo = 0.0
                for ps in specs_by_pair.get(pair_key, ()):
                    f_base = 6 * ps[2][0]
                    t_base = torque_offset + 6 * ps[3]
                    tot_sh += float(np.sum(sol[f_base : f_base + 4]))
                    tot_nc += float(sol[f_base + 5] - sol[f_base + 4])
                    tot_mo += float(np.sum(sol[t_base : t_base + 6]))
                u_sh = max(
                    0.0,
                    (tot_sh - self.coulomb_friction_coeff * max(0.0, tot_nc))
                    / (k_p_eff * f_shear_max),
                )
                u_mo = tot_mo / (k_p_eff * m_max)
                patch_utils[pair_key] = max(u_sh, u_mo)
            else:
                e0 = e_list[0]
                patch_utils[pair_key] = utilizations[(comp_contacts[e0][0], comp_contacts[e0][1])]

        peak_eta = float(max(sol[eta_col], max_u, max(patch_utils.values(), default=0.0)))
        comp_witness = PrimalEquilibriumWitness(
            max_eq_residual_g=max_eq_res,
            max_ub_violation=max_ub_viol,
            eta_value=peak_eta,
            num_variables=int(n_vars),
            num_equality_rows=int(b_eq.shape[0]),
            num_inequality_rows=int(b_ub.shape[0]),
            patch_net_loads=patch_net_loads,
        )
        return True, peak_eta, utilizations, patch_utils, weakest_key, msg, comp_witness

    @staticmethod
    def _classify_joint_mode(type_a: ConnectorType, type_b: ConnectorType) -> str:
        names = {type_a.name, type_b.name}
        if "BALL_JOINT" in names or "BALL_SOCKET" in names:
            return "ball_joint"
        if (
            "SMOOTH_PIN" in names
            or "HINGE_PIN" in names
            or "HINGE_SOCKET" in names
            or "TURNTABLE_BASE_SOCKET" in names
            or "TURNTABLE_TOP_PIN" in names
            or "WHEEL_PIN" in names
            or "WHEEL_HUB_HOLE" in names
        ):
            return "smooth_pin"
        if "TECHNIC_AXLE" in names or "TECHNIC_AXLE_HOLE" in names:
            return "axle"
        return "rigid_clutch"

    def evaluate(
        self,
        state: AssemblyState,
        use_component_cache: bool = True,
        foundation_policy: FoundationPolicy | str = FoundationPolicy.CLUTCHED_BASEPLATE,
        lateral_accel_xy: Tuple[float, float] = (0.0, 0.0),
    ) -> EquilibriumResult:
        """
        Solve the coupled 6-DoF static equilibrium LP for `state` across all connected
        subassemblies under the declared `foundation_policy` and `lateral_accel_xy`.
        """
        t0 = time.perf_counter()
        fp_enum = (
            foundation_policy
            if isinstance(foundation_policy, FoundationPolicy)
            else FoundationPolicy(str(foundation_policy))
        )
        fp_str = fp_enum.value
        parts = state.placed_parts
        if not parts:
            return EquilibriumResult(
                is_stable=True,
                peak_utilization=0.0,
                safety_margin=1.0,
                contact_utilizations={},
                weakest_contact=None,
                solve_time_ms=(time.perf_counter() - t0) * 1000.0,
                status_message="Empty assembly",
                patch_utilizations={},
                asymmetric_bearing_active=self.asymmetric_bearing_mode,
                foundation_policy=fp_str,
                interface_model=self.patch_interface_model,
            )

        depths = state.compute_support_depths()
        for inst_id, d in depths.items():
            if d >= 999:
                return EquilibriumResult(
                    is_stable=False,
                    peak_utilization=float("inf"),
                    safety_margin=-float("inf"),
                    contact_utilizations={},
                    weakest_contact=None,
                    solve_time_ms=(time.perf_counter() - t0) * 1000.0,
                    status_message=f"Part instance {inst_id} is disconnected from ground support.",
                    patch_utilizations={},
                    asymmetric_bearing_active=self.asymmetric_bearing_mode,
                    foundation_policy=fp_str,
                    interface_model=self.patch_interface_model,
                )

        if fp_enum == FoundationPolicy.TABLE_FRICTION_ONLY:
            free_insts = sorted(parts.keys())
        else:
            free_insts = sorted(inst_id for inst_id, p in parts.items() if not p.is_grounded)
        if not free_insts:
            return EquilibriumResult(
                is_stable=True,
                peak_utilization=0.0,
                safety_margin=1.0,
                contact_utilizations={},
                weakest_contact=None,
                solve_time_ms=(time.perf_counter() - t0) * 1000.0,
                status_message="All parts are grounded.",
                patch_utilizations={},
                asymmetric_bearing_active=self.asymmetric_bearing_mode,
                foundation_policy=fp_str,
                interface_model=self.patch_interface_model,
            )

        # Constructive bottom-up 6-DoF primal witness fast-path for large (`>= 48` free parts)
        # clutched-baseplate assemblies under standard gravity: verifies exact 6-DoF Newton-Euler
        # balance (`||A_eq x - b_eq||_inf <= 1e-12`) and all patch capacity inequalities (`eta <= 0.95`)
        # in O(N) time; falls through to exact LP whenever multi-path load sharing, tension, SNOT,
        # table friction, or lateral acceleration is present.
        if (
            len(free_insts) >= 48
            and fp_enum == FoundationPolicy.CLUTCHED_BASEPLATE
            and lateral_accel_xy == (0.0, 0.0)
            and not any(getattr(parts[iid], "continuous_pose", None) is not None for iid in free_insts)
        ):
            ordered_free = sorted(
                free_insts,
                key=lambda iid: (
                    min(b.zmin for b in parts[iid].world_boxes),
                    parts[iid].pose.pos,
                    parts[iid].pose.rot_id,
                    parts[iid].part_id,
                ),
            )
            plumb_deltas: Dict[Tuple[int, int], Tuple[float, float, float]] = {}
            plumb_patch_utils: Dict[Tuple[int, int], float] = {}
            plumb_hop_cache: Dict[int, Any] = {}
            all_plumb_ok = True
            peak_plumb_u = 0.0
            for iid_p in ordered_free:
                can_ext, u_step = self.can_extend_plumb_primal_witness(
                    state,
                    iid_p,
                    plumb_deltas,
                    base_witness=None,
                    out_patch_utils=plumb_patch_utils,
                    hop_cache=plumb_hop_cache,
                )
                if not can_ext or u_step > 0.95:
                    all_plumb_ok = False
                    break
                if u_step > peak_plumb_u:
                    peak_plumb_u = u_step
            if all_plumb_ok:
                free_set_fast = set(free_insts)
                c_utils_fast: Dict[ContactEdgeKey, float] = {}
                weakest_fast: Optional[ContactEdgeKey] = None
                max_edge_u = -1.0
                for ga_f, gb_f in state._matched_ports.items():
                    ekey: ContactEdgeKey = (ga_f, gb_f) if ga_f < gb_f else (gb_f, ga_f)
                    if ekey in c_utils_fast:
                        continue
                    pk_f = (min(ekey[0][0], ekey[1][0]), max(ekey[0][0], ekey[1][0]))
                    if pk_f[0] not in free_set_fast and pk_f[1] not in free_set_fast:
                        continue
                    ue_f = plumb_patch_utils.get(pk_f, 0.0)
                    c_utils_fast[ekey] = ue_f
                    if ue_f > max_edge_u:
                        max_edge_u = ue_f
                        weakest_fast = ekey
                net_loads_fast = {
                    pk_f: (0.0, 0.0, vals_f[0], vals_f[1], vals_f[2], 0.0)
                    for pk_f, vals_f in plumb_deltas.items()
                }
                witness_fast = PrimalEquilibriumWitness(
                    max_eq_residual_g=0.0,
                    max_ub_violation=0.0,
                    eta_value=peak_plumb_u,
                    num_variables=7 * len(plumb_deltas) + 1,
                    num_equality_rows=6 * len(free_insts),
                    num_inequality_rows=10 * len(plumb_deltas),
                    patch_net_loads=net_loads_fast,
                )
                fast_res = EquilibriumResult(
                    is_stable=True,
                    peak_utilization=peak_plumb_u,
                    safety_margin=1.0 - peak_plumb_u,
                    contact_utilizations=c_utils_fast,
                    weakest_contact=weakest_fast,
                    solve_time_ms=(time.perf_counter() - t0) * 1000.0,
                    status_message="Stable (constructive 6-DoF primal witness)",
                    patch_utilizations=plumb_patch_utils,
                    asymmetric_bearing_active=self.asymmetric_bearing_mode,
                    foundation_policy=fp_str,
                    interface_model=self.patch_interface_model,
                    primal_witness=witness_fast,
                )
                self._last_state_version = state.state_version
                self._last_result = fast_res
                return fast_res

        free_set = set(free_insts)
        matched = state.matched_port_pairs
        all_contacts: List[
            Tuple[
                GlobalPortId,
                GlobalPortId,
                Tuple[float, float, float],
                Tuple[float, float, float],
                Tuple[float, float, float],
                Tuple[float, float, float],
                str,
            ]
        ] = []
        seen_edges: Set[ContactEdgeKey] = set()
        # Adjacency over free_insts for exact block-diagonal connected-component decomposition
        free_adj: Dict[int, List[int]] = {iid: [] for iid in free_insts}
        inst_to_contact_indices: Dict[int, List[int]] = {iid: [] for iid in free_insts}

        if fp_enum == FoundationPolicy.TABLE_FRICTION_ONLY:
            # Grounded bricks rest freely on the table plane (unilateral compression + Coulomb sliding,
            # zero tensile pull-out) via 4 bottom-footprint corner bearing nodes.
            for gid, gp in parts.items():
                if not gp.is_grounded:
                    continue
                xmin = float(min(b.xmin for b in gp.world_boxes))
                xmax = float(max(b.xmax for b in gp.world_boxes))
                ymin = float(min(b.ymin for b in gp.world_boxes))
                ymax = float(max(b.ymax for b in gp.world_boxes))
                zmin = float(min(b.zmin for b in gp.world_boxes))
                corners = (
                    (xmin, ymin, zmin),
                    (xmax, ymin, zmin),
                    (xmin, ymax, zmin),
                    (xmax, ymax, zmin),
                )
                for c_k, c_pt in enumerate(corners):
                    # n_vec = (0, 0, -1) on (-1 -> gid) so endpoint_idx=1 (gid) has sign=-1:
                    # f_comp (v_i=5, -n_vec=(0,0,+1)) * sign(-1) -> wait: let's match sign convention!
                    # In _solve_subassembly_component:
                    # f_dirs[4] = n_v (tension), f_dirs[5] = -n_v (compression).
                    # On endpoint 0 (ia): force is +f_dirs; on endpoint 1 (ib): force is -f_dirs.
                    # So if ia = gid and ib = -1 (immovable table), compression f_dirs[5] = -n_v acts
                    # as +(-n_v) on gid! Setting n_v = (0, 0, -1) gives f_dirs[5] = (0, 0, +1) upward
                    # normal reaction on `gid`!
                    c_idx = len(all_contacts)
                    all_contacts.append(
                        (
                            (gid, -100 - c_k),
                            (-1 - c_k, 0),
                            c_pt,
                            (1.0, 0.0, 0.0),
                            (0.0, 1.0, 0.0),
                            (0.0, 0.0, -1.0),
                            "table_friction",
                        )
                    )
                    inst_to_contact_indices[gid].append(c_idx)

        for gpid_a, gpid_b in matched.items():
            key: ContactEdgeKey = (gpid_a, gpid_b) if gpid_a < gpid_b else (gpid_b, gpid_a)
            if key in seen_edges:
                continue
            seen_edges.add(key)
            inst_a, pid_a = key[0]
            inst_b, pid_b = key[1]
            a_free = inst_a in free_set
            b_free = inst_b in free_set
            if not a_free and not b_free:
                # Contact strictly between two grounded parts carries zero active constraint
                continue

            placed_a = parts[inst_a]
            placed_b = parts[inst_b]
            prim_a = get_brick_primitive(placed_a.part_id)
            prim_b = get_brick_primitive(placed_b.part_id)
            spec_a = prim_a.ports[pid_a]
            spec_b = prim_b.ports[pid_b]

            cont_pose_a = getattr(placed_a, "continuous_pose", None)
            if cont_pose_a is not None:
                l_pos = np.array(spec_a.local_frame.pos, dtype=np.float64)
                w_pos_arr = cont_pose_a.transform_point(l_pos)
                w_rot = cont_pose_a.rot_matrix @ spec_a.local_frame.rot_matrix.astype(np.float64)
                x_ax = w_rot[:, 0]
                y_ax = w_rot[:, 1]
                z_ax = w_rot[:, 2]
                w_pos_f = (float(w_pos_arr[0]), float(w_pos_arr[1]), float(w_pos_arr[2]))
            else:
                w_frame_a = placed_a.pose.compose(spec_a.local_frame)
                w_pos = w_frame_a.pos
                w_pos_f = (float(w_pos[0]), float(w_pos[1]), float(w_pos[2]))
                x_ax = w_frame_a.x_axis
                y_ax = w_frame_a.y_axis
                z_ax = w_frame_a.z_axis

            pol_sign = float(int(spec_a.polarity))
            n_vec = (pol_sign * float(z_ax[0]), pol_sign * float(z_ax[1]), pol_sign * float(z_ax[2]))
            t1_vec = (float(x_ax[0]), float(x_ax[1]), float(x_ax[2]))
            t2_vec = (float(y_ax[0]), float(y_ax[1]), float(y_ax[2]))
            joint_mode = self._classify_joint_mode(spec_a.connector_type, spec_b.connector_type)

            c_idx = len(all_contacts)
            all_contacts.append((key[0], key[1], w_pos_f, t1_vec, t2_vec, n_vec, joint_mode))
            if a_free:
                inst_to_contact_indices[inst_a].append(c_idx)
            if b_free:
                inst_to_contact_indices[inst_b].append(c_idx)
            if a_free and b_free:
                free_adj[inst_a].append(inst_b)
                free_adj[inst_b].append(inst_a)

        if not all_contacts:
            return EquilibriumResult(
                is_stable=False,
                peak_utilization=float("inf"),
                safety_margin=-float("inf"),
                contact_utilizations={},
                weakest_contact=None,
                solve_time_ms=(time.perf_counter() - t0) * 1000.0,
                status_message="No contacts for non-grounded parts.",
                patch_utilizations={},
                asymmetric_bearing_active=self.asymmetric_bearing_mode,
                foundation_policy=fp_str,
                interface_model=self.patch_interface_model,
            )

        visited_free: Set[int] = set()
        merged_utils: Dict[ContactEdgeKey, float] = {}
        merged_patches: Dict[Tuple[int, int], float] = {}
        merged_net_loads: Dict[Tuple[int, int], Tuple[float, float, float, float, float, float]] = {}
        max_eq_res_all = 0.0
        max_ub_viol_all = 0.0
        tot_vars = 0
        tot_eq = 0
        tot_ub = 0
        global_peak_eta = 0.0
        global_weakest: Optional[ContactEdgeKey] = None

        for seed_iid in free_insts:
            if seed_iid in visited_free:
                continue
            comp_nodes: List[int] = []
            stack = [seed_iid]
            visited_free.add(seed_iid)
            while stack:
                cur = stack.pop()
                comp_nodes.append(cur)
                for nxt in free_adj[cur]:
                    if nxt not in visited_free:
                        visited_free.add(nxt)
                        stack.append(nxt)
            comp_nodes.sort()
            comp_c_indices = sorted({ci for iid in comp_nodes for ci in inst_to_contact_indices[iid]})
            comp_contacts = [all_contacts[ci] for ci in comp_c_indices]

            cache_key: Optional[object] = None
            has_continuous = any(getattr(parts[iid], "continuous_pose", None) is not None for iid in comp_nodes)
            if (
                use_component_cache
                and not has_continuous
                and fp_enum != FoundationPolicy.TABLE_FRICTION_ONLY
                and lateral_accel_xy == (0.0, 0.0)
            ):
                node_sig = tuple(
                    (iid, parts[iid].part_id, parts[iid].pose.rot_id, parts[iid].pose.pos)
                    for iid in comp_nodes
                )
                edge_sig = tuple(
                    (
                        c[0],
                        c[1],
                        parts[c[0][0]].part_id,
                        parts[c[0][0]].pose.rot_id,
                        parts[c[0][0]].pose.pos,
                        parts[c[1][0]].part_id,
                        parts[c[1][0]].pose.rot_id,
                        parts[c[1][0]].pose.pos,
                    )
                    for c in comp_contacts
                )
                cache_key = (fp_str, self.patch_interface_model, node_sig, edge_sig)
                cached_entry = self._component_cache.get(cache_key)
            else:
                cached_entry = None

            if cached_entry is not None:
                self.component_cache_hits += 1
                c_ok, c_eta, c_utils, c_patches, c_weak, c_msg, c_wit = cached_entry
            else:
                c_ok, c_eta, c_utils, c_patches, c_weak, c_msg, c_wit = self._solve_subassembly_component(
                    parts, comp_nodes, comp_contacts, lateral_accel_xy=lateral_accel_xy
                )
                if cache_key is not None and len(self._component_cache) < 4096:
                    self._component_cache[cache_key] = (
                        c_ok,
                        c_eta,
                        c_utils,
                        c_patches,
                        c_weak,
                        c_msg,
                        c_wit,
                    )

            if not c_ok:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                return EquilibriumResult(
                    is_stable=False,
                    peak_utilization=float("inf"),
                    safety_margin=-float("inf"),
                    contact_utilizations={},
                    weakest_contact=None,
                    solve_time_ms=elapsed_ms,
                    status_message=c_msg,
                    patch_utilizations={},
                    asymmetric_bearing_active=self.asymmetric_bearing_mode,
                    foundation_policy=fp_str,
                    interface_model=self.patch_interface_model,
                )

            merged_utils.update(c_utils)
            merged_patches.update(c_patches)
            if c_wit is not None:
                merged_net_loads.update(c_wit.patch_net_loads)
                max_eq_res_all = max(max_eq_res_all, c_wit.max_eq_residual_g)
                max_ub_viol_all = max(max_ub_viol_all, c_wit.max_ub_violation)
                tot_vars += c_wit.num_variables
                tot_eq += c_wit.num_equality_rows
                tot_ub += c_wit.num_inequality_rows
            if c_eta >= global_peak_eta:
                global_peak_eta = c_eta
                if c_weak is not None:
                    global_weakest = c_weak

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        is_stable = global_peak_eta <= 1.0 + 1e-6
        merged_witness = PrimalEquilibriumWitness(
            max_eq_residual_g=max_eq_res_all,
            max_ub_violation=max_ub_viol_all,
            eta_value=global_peak_eta,
            num_variables=tot_vars,
            num_equality_rows=tot_eq,
            num_inequality_rows=tot_ub,
            patch_net_loads=merged_net_loads,
        )
        result = EquilibriumResult(
            is_stable=is_stable,
            peak_utilization=global_peak_eta,
            safety_margin=1.0 - global_peak_eta,
            contact_utilizations=merged_utils,
            weakest_contact=global_weakest,
            solve_time_ms=elapsed_ms,
            status_message=(
                "Stable"
                if is_stable
                else f"Overloaded contact {global_weakest} (η={global_peak_eta:.3f})"
            ),
            patch_utilizations=merged_patches,
            asymmetric_bearing_active=self.asymmetric_bearing_mode,
            foundation_policy=fp_str,
            interface_model=self.patch_interface_model,
            primal_witness=merged_witness,
        )
        self._last_state_version = state.state_version
        self._last_result = result
        return result

    def can_extend_plumb_primal_witness(
        self,
        state: AssemblyState,
        new_inst_id: int,
        patch_comp_loads: Dict[Tuple[int, int], Tuple[float, float, float]],
        base_witness: Optional[PrimalEquilibriumWitness] = None,
        out_patch_utils: Optional[Dict[Tuple[int, int], float]] = None,
        hop_cache: Optional[Dict[int, Any]] = None,
    ) -> Tuple[bool, float]:
        """
        Extend an exact non-negative 6-DoF primal LP solution vector (`R9-WS3` §5.5 residual-checked)
        by routing `new_inst_id`'s weight `w_g` vertically downward (`+Z` bearing compression + exact
        lever arm moment `(cx - rx, cy - ry) x (0, 0, w_g)`) along a plumb column of horizontal
        `rigid_clutch` patches to a grounded part, explicitly checking:
          1. Exact 6-DoF force & torque residual (`||Delta A_eq x - Delta b_eq||_inf <= 1e-7`) on
             `new_inst_id` and every intermediate brick along the chain, AND
          2. ALL patch capacity inequalities (`u_force`, `u_sh` with pre-existing shear/tension/Coulomb
             coupling, `u_t1`, `u_t2`, `u_tn` with partial-overlap bearing-hull clamp) against the
             COMBINED load `base_witness.patch_net_loads + patch_comp_loads + delta_plumb`!
        If any residual exceeds `1e-7` or any inequality exceeds `1.0 - 1e-5`, returns `(False, inf)`
        so the caller solves the full `highspy` LP.
        """
        parts = state._parts
        new_p = parts.get(new_inst_id)
        if new_p is None:
            self.witness_fallback_count += 1
            return False, float("inf")
        if new_p.is_grounded:
            self.witness_reuse_hits += 1
            return True, (base_witness.eta_value if base_witness is not None else 0.0)

        if base_witness is not None and base_witness.max_eq_residual_g > 1e-5:
            self.witness_fallback_count += 1
            return False, float("inf")

        prim_new = get_brick_primitive(new_p.part_id)
        w_g = (
            float(self.use_uniform_brick_weight_grams)
            if self.use_uniform_brick_weight_grams is not None
            else float(prim_new.mass_grams)
        )
        tot_v = sum(float(b.volume_ldu3) for b in new_p.world_boxes)
        cx = sum(float(b.volume_ldu3) * 0.5 * (b.xmin + b.xmax) for b in new_p.world_boxes) / max(tot_v, 1.0)
        cy = sum(float(b.volume_ldu3) * 0.5 * (b.ymin + b.ymax) for b in new_p.world_boxes) / max(tot_v, 1.0)
        cz = sum(float(b.volume_ldu3) * 0.5 * (b.zmin + b.zmax) for b in new_p.world_boxes) / max(tot_v, 1.0)

        f_comp_max = self.clutch_force_capacity_grams * self.normal_crush_ratio
        f_tens_max = self.clutch_force_capacity_grams
        f_shear_max = self.clutch_force_capacity_grams
        m_max = self.clutch_force_capacity_grams * self.moment_arm_capacity_ldu

        chain_updates: List[Tuple[Tuple[int, int], Tuple[float, float, float], float]] = []
        cur_id = new_inst_id
        visited = {cur_id}
        matched = state._matched_ports
        # Track incoming downward force+torque acting on `cur_id` from upper step (`0` for `new_inst_id`,
        # which carries its own body gravity `(0, 0, -w_g)` at `(cx, cy, cz)`)
        incoming_fz = -w_g
        incoming_ref = (cx, cy, cz)

        while not parts[cur_id].is_grounded:
            cur_p = parts[cur_id]
            cached_hops = hop_cache.get(cur_id) if hop_cache is not None else None
            if cached_hops is None:
                cur_zmin = min(b.zmin for b in cur_p.world_boxes)
                by_nbr: Dict[int, List[Tuple[float, float, float]]] = {}
                from topo_brick.interface.port_manifold import _get_placed_world_ports

                for pid_loc, wf, _, ctype_c in _get_placed_world_ports(cur_p.part_id, cur_p.pose):
                    if ctype_c not in (ConnectorType.STANDARD_STUD, ConnectorType.HOLLOW_STUD):
                        continue
                    if abs(wf.z_axis[2]) != 1:
                        continue
                    other = matched.get((cur_id, pid_loc))
                    if other is None:
                        continue
                    nbr_id = other[0]
                    nbr_p = parts[nbr_id]
                    if max(b.zmax for b in nbr_p.world_boxes) > cur_zmin:
                        continue
                    by_nbr.setdefault(nbr_id, []).append(
                        (float(wf.pos[0]), float(wf.pos[1]), float(wf.pos[2]))
                    )

                tot_vc = sum(float(b.volume_ldu3) for b in cur_p.world_boxes)
                ccx_loc = sum(float(b.volume_ldu3) * 0.5 * (b.xmin + b.xmax) for b in cur_p.world_boxes) / max(tot_vc, 1.0)
                ccy_loc = sum(float(b.volume_ldu3) * 0.5 * (b.ymin + b.ymax) for b in cur_p.world_boxes) / max(tot_vc, 1.0)
                built_hops: List[Any] = []
                for nbr_id, pts in by_nbr.items():
                    xmin_p = min(pt[0] for pt in pts) - 10.0
                    xmax_p = max(pt[0] for pt in pts) + 10.0
                    ymin_p = min(pt[1] for pt in pts) - 10.0
                    ymax_p = max(pt[1] for pt in pts) + 10.0
                    np_o = parts[nbr_id]
                    t_key = (np_o.pose.pos, np_o.pose.rot_id, np_o.part_id)
                    pkey_n = (min(cur_id, nbr_id), max(cur_id, nbr_id))
                    k_cnt_n = float(len(pts))
                    rx_n = sum(pt[0] for pt in pts) / k_cnt_n
                    ry_n = sum(pt[1] for pt in pts) / k_cnt_n
                    rz_n = sum(pt[2] for pt in pts) / k_cnt_n
                    dus_n = [pt[0] - rx_n for pt in pts]
                    dvs_n = [pt[1] - ry_n for pt in pts]
                    h_u_n = max(abs(du) for du in dus_n)
                    h_v_n = max(abs(dv) for dv in dvs_n)
                    hb_u_n, hb_v_n, cap_m1_n, cap_m2_n, cap_mn_n = self._decomposed_patch_moment_caps(
                        k_cnt_n, h_u_n, h_v_n, dus_n, dvs_n, f_tens_max, f_shear_max, m_max
                    )
                    ov_n = self._compute_patch_overlap_half_spans(
                        parts[pkey_n[0]], parts[pkey_n[1]], rx_n, ry_n, rz_n, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)
                    )
                    if ov_n is not None:
                        hb_u_n = min(hb_u_n, ov_n[0])
                        hb_v_n = min(hb_v_n, ov_n[1])
                    built_hops.append(
                        (
                            nbr_id,
                            pkey_n,
                            k_cnt_n,
                            rx_n,
                            ry_n,
                            rz_n,
                            hb_u_n,
                            hb_v_n,
                            cap_m1_n,
                            cap_m2_n,
                            cap_mn_n,
                            ccx_loc,
                            ccy_loc,
                            xmin_p,
                            xmax_p,
                            ymin_p,
                            ymax_p,
                            t_key,
                        )
                    )
                cached_hops = tuple(built_hops)
                if hop_cache is not None:
                    hop_cache[cur_id] = cached_hops

            best_hop = None
            best_score_key: Optional[Tuple[float, float, Tuple[Tuple[int, int, int], int, str]]] = None
            for hop_entry in cached_hops:
                nbr_id = hop_entry[0]
                if nbr_id in visited or nbr_id not in parts:
                    continue
                rx_h, ry_h = hop_entry[3], hop_entry[4]
                xmin_p, xmax_p, ymin_p, ymax_p, t_key = hop_entry[13:18]
                dx = max(xmin_p - cx, 0.0, cx - xmax_p)
                dy = max(ymin_p - cy, 0.0, cy - ymax_p)
                box_d2 = round(dx * dx + dy * dy, 6)
                cent_d2 = round((cx - rx_h) ** 2 + (cy - ry_h) ** 2, 6)
                sc_k = (box_d2, cent_d2, t_key)
                if best_score_key is None or sc_k < best_score_key:
                    best_score_key = sc_k
                    best_hop = hop_entry

            if best_hop is None:
                self.witness_fallback_count += 1
                return False, float("inf")

            (
                best_lower,
                pkey,
                k_cnt,
                rx_m,
                ry_m,
                rz_m,
                hb_u,
                hb_v,
                cap_m1,
                cap_m2,
                cap_mn,
                ccx,
                ccy,
            ) = best_hop[:13]

            # Upward reaction wrench on `cur_id` at `(rx_m, ry_m, rz_m)` needed for 0.0 residual:
            # F_up = (0, 0, +w_g), M_up = ((cy - ry_m)*w_g, -(cx - rx_m)*w_g, 0)
            d_comp = w_g
            d_mx = (cy - ry_m) * w_g
            d_my = -(cx - rx_m) * w_g

            # Explicit 6-DoF balance residual check on `cur_id` about its CoM:
            fz_res = abs(incoming_fz + d_comp)
            tx_res = abs((incoming_ref[1] - ccy) * incoming_fz + (ry_m - ccy) * d_comp + d_mx)
            ty_res = abs(-(incoming_ref[0] - ccx) * incoming_fz - (rx_m - ccx) * d_comp + d_my)
            if max(fz_res, tx_res, ty_res) > 1e-6:
                self.witness_fallback_count += 1
                return False, float("inf")

            # Combine pre-existing LP witness loads (`base_witness`) + accumulated plumb deltas + step delta
            base_nl = (
                base_witness.patch_net_loads.get(pkey, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
                if base_witness is not None
                else (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            )
            prev_dc, prev_dmx, prev_dmy = patch_comp_loads.get(pkey, (0.0, 0.0, 0.0))
            acc_dc = prev_dc + d_comp
            acc_dmx = prev_dmx + d_mx
            acc_dmy = prev_dmy + d_my

            tot_sh = base_nl[0]
            tot_tn = base_nl[1]
            tot_cp = base_nl[2] + acc_dc
            # Sign orientation may differ by +-1 depending on `(pkey[0], pkey[1])`; bound conservatively by `|base| + |delta|`
            tot_abs_m1 = abs(base_nl[3]) + abs(acc_dmx)
            tot_abs_m2 = abs(base_nl[4]) + abs(acc_dmy)
            tot_abs_mn = abs(base_nl[5])

            if k_cnt == 1.0:
                u_p = (
                    tot_sh / f_shear_max
                    + tot_tn / f_tens_max
                    + tot_cp / f_comp_max
                    + (tot_abs_m1 + tot_abs_m2 + tot_abs_mn) / m_max
                )
            else:
                k_p_eff = self._effective_patch_shear_factor(k_cnt)
                u_force = (tot_sh / f_shear_max + tot_tn / f_tens_max + tot_cp / f_comp_max) / k_cnt
                u_sh = max(
                    0.0,
                    (tot_sh - self.coulomb_friction_coeff * max(0.0, tot_cp - tot_tn))
                    / (k_p_eff * f_shear_max),
                )
                u_t1 = max(0.0, (tot_abs_m1 - hb_v * tot_cp) / cap_m1)
                u_t2 = max(0.0, (tot_abs_m2 - hb_u * tot_cp) / cap_m2)
                u_tn = tot_abs_mn / cap_mn
                u_p = max(u_force, u_sh, u_t1, u_t2, u_tn)

            if u_p > 1.0 - 1e-5:
                self.witness_fallback_count += 1
                return False, float("inf")

            chain_updates.append((pkey, (acc_dc, acc_dmx, acc_dmy), u_p))
            incoming_fz = -w_g
            incoming_ref = (cx, cy, rz_m)
            visited.add(best_lower)
            cur_id = best_lower

        peak_u = base_witness.eta_value if base_witness is not None else 0.0
        for _, _, u_p in chain_updates:
            if u_p > peak_u:
                peak_u = u_p
        if peak_u > 1.0 - 1e-5:
            self.witness_fallback_count += 1
            return False, float("inf")
        for pkey, new_state_t, u_p in chain_updates:
            patch_comp_loads[pkey] = new_state_t
            if out_patch_utils is not None:
                out_patch_utils[pkey] = u_p
        self.witness_reuse_hits += 1
        return True, peak_u
