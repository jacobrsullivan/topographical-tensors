"""
WS-2B: External Physical Stability Calibration & Model-Agreement Suite
(`StableLego` Physical Hardware Boundary Cases + `BrickSim` Topology + `StableText2Brick` Scores).

Implements an exact, Gurobi-free open-source `highspy` Linear Programming formulation of the
CMU Intelligent Control Lab sub-stud contact-polygon & knob-wall mechanics model (`StableLego`,
Liu et al., IEEE RA-L 2024; `BrickSim`, Pun et al., ICCV 2025):
- Models 3-point (`2xN` center-tube + side-wall) and 4-point (`1xN` perimeter rib) sub-stud
  contact polygons (`±0.25` / `±0.125` stud-pitch fulcrum offsets = `±5.0` / `±2.5 LDU`),
  lateral knob-press couples (`H / 2 = 12 LDU`), and flush side-wall normal compression.
- Exploits the exact LP complementarity property: because `n_up[k] - f_down[k]` enters
  5-DoF/6-DoF rigid-body equilibrium linearly while `f_down[k]` and `n_up[k]` carry strictly
  positive objective weights, `min(n_up[k], f_down[k]) == 0` holds at every LP optimum
  without non-convex bilinear constraints (`n_up * f_down == 0`) or commercial Gurobi licenses.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from scipy.sparse import csc_matrix

from topo_brick.evaluation.static_equilibrium import _solve_lp_highspy_or_scipy
from topo_brick.interface.brick_catalog import get_brick_primitive
from topo_brick.interface.port_manifold import AssemblyState


_ASSETS_PATH = Path(__file__).with_name("physical_validation_assets.json")


@dataclass(frozen=True, slots=True)
class SubStudEquilibriumResult:
    """Result of the calibrated sub-stud contact-polygon physical stability LP."""

    is_stable: bool
    peak_utilization: float  # max_{b, k} f_down[b, k] / T_max (<= 1.0 + tol & eq_residual == 0)
    equilibrium_residual_n: float
    per_brick_utilization: Tuple[float, ...]
    solve_time_ms: float


@dataclass(frozen=True, slots=True)
class PhysicalCaseEval:
    """Verification outcome on a single physical hardware benchmark structure."""

    case_name: str
    num_bricks: int
    expected_physically_stable: bool
    predicted_stable: bool
    peak_utilization: float
    equilibrium_residual_n: float
    solve_time_ms: float
    matched_ground_truth: bool


@dataclass(frozen=True, slots=True)
class ModelAgreementReport:
    """
    Separate verification report distinguishing:
    1. `physical_hardware_suite_accuracy`: exact discrimination on the 8 CMU `StableLego`
       physical boundary assemblies (`stair_19/20/20_good`, `stick_light/heavy/heavy_good`,
       `external_weight_good/fail`) + `BrickSim` topology demo.
    2. `s2b_reference_agreement_rate`: binary stability agreement against `AvaLovelace/StableText2Brick`
       ground-truth Gurobi `stability_scores` (`max(score) < 1.0`).
    """

    num_physical_cases: int
    physical_cases_matched: int
    physical_hardware_accuracy: float
    physical_case_results: Tuple[PhysicalCaseEval, ...]
    bricksim_demo_certified_stable: bool
    bricksim_demo_peak_utilization: float
    s2b_num_evaluated: int
    s2b_substud_agreement_rate: float
    s2b_kernel_lp_agreement_rate: float
    s2b_mean_max_score_mae: float


class CalibratedSubStudPolygonValidator:
    """
    Calibrated sub-stud contact-polygon & side-wall LP validator (`T = 100 gf` per sub-stud
    clutch point, `g = 9.8 m/s^2`, `L_stud = 7.8 mm`, `H_brick = 9.6 mm`).
    """

    def __init__(
        self,
        clutch_capacity_grams: float = 100.0,
        gravity_m_s2: float = 9.8,
        brick_unit_height_m: float = 0.0096,
        brick_unit_length_m: float = 0.0078,
        alpha: float = 1e-3,
        beta: float = 1e-6,
    ) -> None:
        self.clutch_capacity_grams = float(clutch_capacity_grams)
        self.gravity_m_s2 = float(gravity_m_s2)
        self.t_max_newtons = (self.clutch_capacity_grams / 1000.0) * self.gravity_m_s2
        self.brick_unit_height_m = float(brick_unit_height_m)
        self.brick_unit_length_m = float(brick_unit_length_m)
        self.alpha = float(alpha)
        self.beta = float(beta)

    def evaluate_grid_bricks(
        self,
        bricks: Sequence[Tuple[int, int, int, int, int, float]],
        enable_side_wall_press: bool = True,
    ) -> SubStudEquilibriumResult:
        """
        Evaluate a list of cuboid bricks `(x0, y0, z0, hx, wy, mass_kg)` on the integer unit
        grid (`z0 >= 0`, where `z0 == 0` rests on the grounded baseplate).
        """
        t0 = time.perf_counter()
        n_bricks = len(bricks)
        if n_bricks == 0:
            return SubStudEquilibriumResult(
                is_stable=True,
                peak_utilization=0.0,
                equilibrium_residual_n=0.0,
                per_brick_utilization=(),
                solve_time_ms=0.0,
            )

        # Map occupied unit cells (x, y, z) -> brick_idx
        cell_to_brick: Dict[Tuple[int, int, int], int] = {}
        cell_four_pt: Dict[Tuple[int, int, int], bool] = {}
        for b_idx, (bx, by, bz, hx, wy, _mass) in enumerate(bricks):
            is_4pt = min(hx, wy) < 2
            for ix in range(bx, bx + hx):
                for iy in range(by, by + wy):
                    cell_to_brick[(ix, iy, bz)] = b_idx
                    cell_four_pt[(ix, iy, bz)] = is_4pt

        # Allocate LP decision variables dynamically
        col_costs: List[float] = []

        def _add_var(cost: float = 0.0) -> int:
            vid = len(col_costs)
            col_costs.append(cost)
            return vid

        # 1. Per-brick equilibrium slack variables |F_x|, |F_y|, |F_z|, |tau_1|, |tau_2|
        #    represented via non-negative splits r^+ - r^- (cost = 1.0 each so eq_obj = sum(r^+ + r^-))
        eq_slack_pos: np.ndarray = np.zeros((n_bricks, 5), dtype=np.int32)
        eq_slack_neg: np.ndarray = np.zeros((n_bricks, 5), dtype=np.int32)
        brick_max_f_down_var: np.ndarray = np.zeros(n_bricks, dtype=np.int32)
        for b_idx in range(n_bricks):
            for d in range(5):
                eq_slack_pos[b_idx, d] = _add_var(1.0)
                eq_slack_neg[b_idx, d] = _add_var(1.0)
            brick_max_f_down_var[b_idx] = _add_var(self.alpha)

        # 2. Adjacent horizontal side-wall normal presses between distinct bricks at same z
        #    Ordered pair of cells ((x, y, z), (nx, ny, z)) with (x, y) < (nx, ny)
        horiz_press_x: Dict[Tuple[int, int, int], int] = {}  # between (x, y, z) and (x+1, y, z)
        horiz_press_y: Dict[Tuple[int, int, int], int] = {}  # between (x, y, z) and (x, y+1, z)
        if enable_side_wall_press:
            for (ix, iy, iz), b_a in cell_to_brick.items():
                b_bx = cell_to_brick.get((ix + 1, iy, iz))
                if b_bx is not None and b_bx != b_a:
                    horiz_press_x[(ix, iy, iz)] = _add_var(1e-9)
                b_by = cell_to_brick.get((ix, iy + 1, iz))
                if b_by is not None and b_by != b_a:
                    horiz_press_y[(ix, iy, iz)] = _add_var(1e-9)

        # 3. Vertical stud-to-cavity connections at bottom of every cell `(ix, iy, iz)` that is
        #    either at `iz == 0` (grounded baseplate) or has an occupied cell at `(ix, iy, iz - 1)`!
        #    Note: in StableLego, variables on top of `(ix, iy, iz - 1)` equal variables on bottom
        #    of `(ix, iy, iz)` (`top_x_pos == bottom_x_neg`, `f_up == f_down`, `n_down == n_up`),
        #    so allocating ONE canonical variable set per engaged cavity `(ix, iy, iz)` enforces
        #    Newton's 3rd Law identically with half the variables and constraints!
        cavity_vars: Dict[Tuple[int, int, int], Tuple[int, int, int, int, Tuple[int, ...], Tuple[int, ...]]] = {}
        for (ix, iy, iz), b_upper in cell_to_brick.items():
            has_support = (iz == 0) or ((ix, iy, iz - 1) in cell_to_brick)
            if not has_support:
                continue
            bx_pos = _add_var(1e-9)
            bx_neg = _add_var(1e-9)
            by_pos = _add_var(1e-9)
            by_neg = _add_var(1e-9)
            n_pts = 4 if cell_four_pt[(ix, iy, iz)] else 3
            f_down_pts = tuple(_add_var(self.beta) for _ in range(n_pts))
            n_up_pts = tuple(_add_var(1e-9) for _ in range(n_pts))
            cavity_vars[(ix, iy, iz)] = (bx_pos, bx_neg, by_pos, by_neg, f_down_pts, n_up_pts)

        # Build equality rows (5 per brick: Fx, Fy, Fz, Torque_1, Torque_2) and inequality rows
        # (`f_down[k] - brick_max_f_down[b] <= 0`)
        eq_rows: List[int] = []
        eq_cols: List[int] = []
        eq_vals: List[float] = []
        b_eq = np.zeros(5 * n_bricks, dtype=np.float64)

        ub_rows: List[int] = []
        ub_cols: List[int] = []
        ub_vals: List[float] = []
        n_ub = 0

        half_h = 0.5 * self.brick_unit_height_m
        L_u = self.brick_unit_length_m

        def _add_eq(row: int, col: int, val: float) -> None:
            if val != 0.0:
                eq_rows.append(row)
                eq_cols.append(col)
                eq_vals.append(val)

        for b_idx, (bx, by, bz, hx, wy, mass_kg) in enumerate(bricks):
            r_fx = 5 * b_idx + 0
            r_fy = 5 * b_idx + 1
            r_fz = 5 * b_idx + 2
            r_t1 = 5 * b_idx + 3
            r_t2 = 5 * b_idx + 4

            # Equilibrium slack r^+ - r^- moves residual to 0 when physically equilibratable
            for d in range(5):
                _add_eq(5 * b_idx + d, int(eq_slack_pos[b_idx, d]), -1.0)
                _add_eq(5 * b_idx + d, int(eq_slack_neg[b_idx, d]), 1.0)

            brick_weight = float(mass_kg) * self.gravity_m_s2
            b_eq[r_fz] = brick_weight
            center_x = bx + (hx - 1) / 2.0
            center_y = by + (wy - 1) / 2.0

            for ix in range(bx, bx + hx):
                for iy in range(by, by + wy):
                    # Side-wall normal contacts along X
                    v_xm = horiz_press_x.get((ix - 1, iy, bz))
                    if v_xm is not None:
                        # Neighbor at ix-1 pushes +X on (ix, iy, bz)
                        _add_eq(r_fx, v_xm, 1.0)
                        _add_eq(r_t2, v_xm, -half_h)
                    v_xp = horiz_press_x.get((ix, iy, bz))
                    if v_xp is not None:
                        # Neighbor at ix+1 pushes -X on (ix, iy, bz)
                        _add_eq(r_fx, v_xp, -1.0)
                        _add_eq(r_t2, v_xp, half_h)

                    # Side-wall normal contacts along Y
                    v_ym = horiz_press_y.get((ix, iy - 1, bz))
                    if v_ym is not None:
                        _add_eq(r_fy, v_ym, 1.0)
                        _add_eq(r_t1, v_ym, half_h)
                    v_yp = horiz_press_y.get((ix, iy, bz))
                    if v_yp is not None:
                        _add_eq(r_fy, v_yp, -1.0)
                        _add_eq(r_t1, v_yp, -half_h)

                    # Top knob of (ix, iy, bz) engaged with cavity at (ix, iy, bz + 1)
                    cav_top = cavity_vars.get((ix, iy, bz + 1))
                    if cav_top is not None:
                        bx_pos_t, bx_neg_t, by_pos_t, by_neg_t, f_down_t, n_up_t = cav_top
                        # Recall Newton's 3rd Law from StableLego:
                        # top_x_pos = bottom_x_neg, top_x_neg = bottom_x_pos,
                        # top_y_pos = bottom_y_neg, top_y_neg = bottom_y_pos,
                        # f_up = f_down, n_down = n_up
                        _add_eq(r_fx, bx_neg_t, 1.0)
                        _add_eq(r_fx, bx_pos_t, -1.0)
                        _add_eq(r_fy, by_neg_t, 1.0)
                        _add_eq(r_fy, by_pos_t, -1.0)
                        _add_eq(r_t1, by_pos_t, half_h)
                        _add_eq(r_t1, by_neg_t, -half_h)
                        _add_eq(r_t2, bx_neg_t, half_h)
                        _add_eq(r_t2, bx_pos_t, -half_h)

                        if len(f_down_t) == 4:
                            offsets = ((0.0, -0.25), (-0.25, 0.0), (0.0, 0.25), (0.25, 0.0))
                        else:
                            offsets = ((0.125, -0.125), (-0.25, 0.0), (0.125, 0.125))
                        for k_pt, (ox, oy) in enumerate(offsets):
                            fd_v = f_down_t[k_pt]
                            nu_v = n_up_t[k_pt]
                            _add_eq(r_fz, fd_v, 1.0)
                            _add_eq(r_fz, nu_v, -1.0)
                            dy_m = (iy - center_y + oy) * L_u
                            dx_m = (ix - center_x + ox) * L_u
                            _add_eq(r_t1, fd_v, dy_m)
                            _add_eq(r_t1, nu_v, -dy_m)
                            _add_eq(r_t2, fd_v, -dx_m)
                            _add_eq(r_t2, nu_v, dx_m)

                    # Bottom cavity of (ix, iy, bz) engaged with ground or (ix, iy, bz - 1)
                    cav_bot = cavity_vars.get((ix, iy, bz))
                    if cav_bot is not None:
                        bx_pos_b, bx_neg_b, by_pos_b, by_neg_b, f_down_b, n_up_b = cav_bot
                        _add_eq(r_fx, bx_pos_b, 1.0)
                        _add_eq(r_fx, bx_neg_b, -1.0)
                        _add_eq(r_fy, by_pos_b, 1.0)
                        _add_eq(r_fy, by_neg_b, -1.0)
                        _add_eq(r_t1, by_pos_b, half_h)
                        _add_eq(r_t1, by_neg_b, -half_h)
                        _add_eq(r_t2, bx_neg_b, half_h)
                        _add_eq(r_t2, bx_pos_b, -half_h)

                        if len(f_down_b) == 4:
                            offsets = ((0.0, -0.25), (-0.25, 0.0), (0.0, 0.25), (0.25, 0.0))
                        else:
                            offsets = ((0.125, -0.125), (-0.25, 0.0), (0.125, 0.125))
                        for k_pt, (ox, oy) in enumerate(offsets):
                            fd_v = f_down_b[k_pt]
                            nu_v = n_up_b[k_pt]
                            _add_eq(r_fz, nu_v, 1.0)
                            _add_eq(r_fz, fd_v, -1.0)
                            dy_m = (iy - center_y + oy) * L_u
                            dx_m = (ix - center_x + ox) * L_u
                            _add_eq(r_t1, nu_v, dy_m)
                            _add_eq(r_t1, fd_v, -dy_m)
                            _add_eq(r_t2, nu_v, -dx_m)
                            _add_eq(r_t2, fd_v, dx_m)
                            # Bound brick_max_f_down[b_idx] >= f_down_b[k_pt]
                            ub_rows.append(n_ub)
                            ub_cols.append(fd_v)
                            ub_vals.append(1.0)
                            ub_rows.append(n_ub)
                            ub_cols.append(int(brick_max_f_down_var[b_idx]))
                            ub_vals.append(-1.0)
                            n_ub += 1

        n_vars = len(col_costs)
        c_obj = np.asarray(col_costs, dtype=np.float64)
        var_lb = np.zeros(n_vars, dtype=np.float64)
        var_ub = np.full(n_vars, np.inf, dtype=np.float64)
        a_eq_csc = csc_matrix(
            (np.asarray(eq_vals, dtype=np.float64), (eq_rows, eq_cols)),
            shape=(5 * n_bricks, n_vars),
        )
        a_ub_csc = csc_matrix(
            (np.asarray(ub_vals, dtype=np.float64), (ub_rows, ub_cols)),
            shape=(n_ub, n_vars),
        )
        b_ub = np.zeros(n_ub, dtype=np.float64)

        ok, sol, _msg = _solve_lp_highspy_or_scipy(
            c_obj=c_obj,
            a_eq_csc=a_eq_csc,
            b_eq=b_eq,
            a_ub_csc=a_ub_csc,
            b_ub=b_ub,
            var_lb=var_lb,
            var_ub=var_ub,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if not ok or sol is None:
            return SubStudEquilibriumResult(
                is_stable=False,
                peak_utilization=float("inf"),
                equilibrium_residual_n=float("inf"),
                per_brick_utilization=tuple(float("inf") for _ in range(n_bricks)),
                solve_time_ms=elapsed_ms,
            )

        eq_res = float(
            np.sum(sol[eq_slack_pos.ravel()]) + np.sum(sol[eq_slack_neg.ravel()])
        )
        per_brick_u: List[float] = []
        for b_idx, (bx, by, bz, hx, wy, _m) in enumerate(bricks):
            b_res = float(
                np.sum(sol[eq_slack_pos[b_idx]]) + np.sum(sol[eq_slack_neg[b_idx]])
            )
            max_fd = 0.0
            has_any_conn = False
            for ix in range(bx, bx + hx):
                for iy in range(by, by + wy):
                    cav = cavity_vars.get((ix, iy, bz))
                    if cav is not None:
                        has_any_conn = True
                        for fd_v in cav[4]:
                            max_fd = max(max_fd, float(sol[fd_v]))
                    cav_t = cavity_vars.get((ix, iy, bz + 1))
                    if cav_t is not None:
                        has_any_conn = True
                        for fd_v in cav_t[4]:
                            max_fd = max(max_fd, float(sol[fd_v]))
            if b_res > 1e-5 or not has_any_conn:
                per_brick_u.append(float("inf"))
            else:
                per_brick_u.append(max_fd / self.t_max_newtons)

        finite_u = [u for u in per_brick_u if np.isfinite(u)]
        peak_u = max(finite_u, default=0.0) if eq_res <= 1e-5 else float("inf")
        is_stable = (eq_res <= 1e-5) and (peak_u < 1.0 - 1e-6)
        return SubStudEquilibriumResult(
            is_stable=is_stable,
            peak_utilization=peak_u,
            equilibrium_residual_n=eq_res,
            per_brick_utilization=tuple(per_brick_u),
            solve_time_ms=elapsed_ms,
        )

    def evaluate_assembly_state(self, state: AssemblyState) -> SubStudEquilibriumResult:
        """
        Evaluate an `AssemblyState` composed of rectangular cubic bricks using the calibrated
        sub-stud contact-polygon LP.
        """
        parts = state.placed_parts
        if not parts:
            return SubStudEquilibriumResult(True, 0.0, 0.0, (), 0.0)

        all_boxes = [b for p in parts.values() for b in p.world_boxes]
        xmin = min(b.xmin for b in all_boxes)
        ymin = min(b.ymin for b in all_boxes)
        zmin = min(b.zmin for b in all_boxes)

        grid_bricks: List[Tuple[int, int, int, int, int, float]] = []
        for iid in sorted(parts.keys()):
            p = parts[iid]
            prim = get_brick_primitive(p.part_id)
            bx0 = min(b.xmin for b in p.world_boxes)
            bx1 = max(b.xmax for b in p.world_boxes)
            by0 = min(b.ymin for b in p.world_boxes)
            by1 = max(b.ymax for b in p.world_boxes)
            bz0 = min(b.zmin for b in p.world_boxes)
            gx = (bx0 - xmin) // 20
            gy = (by0 - ymin) // 20
            gz = (bz0 - zmin) // 24
            hx = max(1, (bx1 - bx0) // 20)
            wy = max(1, (by1 - by0) // 20)
            grid_bricks.append((gx, gy, gz, hx, wy, float(prim.mass_grams) / 1000.0))
        return self.evaluate_grid_bricks(grid_bricks)


def load_physical_validation_assets() -> Dict[str, Any]:
    return json.loads(_ASSETS_PATH.read_text(encoding="utf-8"))


def evaluate_bricksim_demo_topology(
    validator: Optional[CalibratedSubStudPolygonValidator] = None,
) -> SubStudEquilibriumResult:
    """Reconstruct and certify `BrickSim`'s `demo_assembly_structure1.json` (`lego_topology@2`)."""
    assets = load_physical_validation_assets()
    topo = assets["bricksim_demo_topology"]["topology"]
    val = validator or CalibratedSubStudPolygonValidator()

    parts_by_id = {int(p["id"]): p["payload"] for p in topo["parts"]}
    # Reconstruct grid poses from pose_hints + relative local connections (`stud_id`, `hole_id`, `offset`, `yaw`)
    # Track `(origin_x, origin_y, z_plates, world_yaw)` where world_yaw in {0,1,2,3} maps local (u, v) in [0..L) x [0..W)
    placed_frames: Dict[int, Tuple[int, int, int, int]] = {}
    placed_boxes: Dict[int, Tuple[int, int, int, int, int]] = {}

    def _local_box_to_world(ox: int, oy: int, yaw: int, l_dim: int, w_dim: int) -> Tuple[int, int, int, int]:
        # Transform corners (0,0) and (l_dim, w_dim) by C4 yaw
        if yaw == 0:
            return ox, oy, l_dim, w_dim
        if yaw == 1:
            return ox - w_dim, oy, w_dim, l_dim
        if yaw == 2:
            return ox - l_dim, oy - w_dim, l_dim, w_dim
        return ox, oy - l_dim, w_dim, l_dim

    def _local_vec_to_world(du: int, dv: int, yaw: int) -> Tuple[int, int]:
        if yaw == 0:
            return du, dv
        if yaw == 1:
            return -dv, du
        if yaw == 2:
            return -du, -dv
        return dv, -du

    p0 = parts_by_id[0]
    placed_frames[0] = (0, 0, 0, 0)
    placed_boxes[0] = (0, 0, 0, int(p0["L"]), int(p0["W"]))

    remaining = list(topo["connections"])
    for _ in range(len(remaining) + 1):
        for conn in list(remaining):
            sid = int(conn["stud_id"])
            hid = int(conn["hole_id"])
            if sid in placed_frames and hid not in placed_frames:
                sox, soy, sz_p, syaw = placed_frames[sid]
                s_h = int(parts_by_id[sid]["H"])
                hp = parts_by_id[hid]
                rel_yaw = int(conn.get("yaw", 0)) % 4
                hyaw = (syaw + rel_yaw) % 4
                du, dv = int(conn["offset"][0]), int(conn["offset"][1])
                wx, wy_off = _local_vec_to_world(du, dv, syaw)
                hox, hoy = sox + wx, soy + wy_off
                bx, by, hx, wy = _local_box_to_world(hox, hoy, hyaw, int(hp["L"]), int(hp["W"]))
                placed_frames[hid] = (hox, hoy, sz_p + s_h, hyaw)
                placed_boxes[hid] = (bx, by, sz_p + s_h, hx, wy)
                remaining.remove(conn)

    # Shift to non-negative grid origin
    min_gx = min(v[0] for k, v in placed_boxes.items() if k != 0)
    min_gy = min(v[1] for k, v in placed_boxes.items() if k != 0)
    mass_table = {(2, 4): 0.00216, (2, 6): 0.00323, (2, 2): 0.00115}
    grid_bricks: List[Tuple[int, int, int, int, int, float]] = []
    for pid in sorted(placed_boxes.keys()):
        if pid == 0:
            continue
        gx, gy, zp, hx, wy = placed_boxes[pid]
        gz = (zp - 1) // 3
        m_kg = mass_table.get((min(hx, wy), max(hx, wy)), 0.00027 * hx * wy)
        grid_bricks.append((gx - min_gx, gy - min_gy, gz, hx, wy, m_kg))
    return val.evaluate_grid_bricks(grid_bricks)


def run_physical_and_model_agreement_suite(
    s2b_records: Sequence[Dict[str, Any]] = (),
) -> ModelAgreementReport:
    """
    Execute:
    1. All 8 CMU `StableLego` physical hardware benchmark cases (`stair_19`, `stair_20`,
       `stair_20_good`, `stick_light`, `stick_heavy`, `stick_heavy_good`,
       `external_weight_good`, `external_weight_fail`).
    2. `BrickSim` `demo_assembly_structure1.json` (`lego_topology@2`).
    3. Optional `AvaLovelace/StableText2Brick` `stability_scores` agreement check across `s2b_records`.
    """
    assets = load_physical_validation_assets()
    cfg = assets["config"]
    lib = assets["lego_library"]
    val = CalibratedSubStudPolygonValidator(
        clutch_capacity_grams=float(cfg["T"]),
        gravity_m_s2=float(cfg["g"]),
        brick_unit_height_m=float(cfg["Brick_Unit_Height"]),
        brick_unit_length_m=float(cfg["Brick_Unit_Length"]),
        alpha=float(cfg["Alpha"]),
        beta=float(cfg["Beta"]),
    )

    case_results: List[PhysicalCaseEval] = []
    for case_name, cinfo in assets["stable_lego_physical_cases"].items():
        raw_bricks = cinfo["bricks"]
        # Preserve original StableLego insertion key order
        grid_list: List[Tuple[int, int, int, int, int, float]] = []
        for _k, b in raw_bricks.items():
            bid = str(b["brick_id"])
            ori = int(b["ori"])
            lib_entry = lib[bid]
            if ori == 0:
                hx = int(lib_entry["height"])
                wy = int(lib_entry["width"])
            else:
                wy = int(lib_entry["height"])
                hx = int(lib_entry["width"])
            mass_kg = float(lib_entry["mass"])
            grid_list.append((int(b["x"]), int(b["y"]), int(b["z"]), hx, wy, mass_kg))

        res = val.evaluate_grid_bricks(grid_list)
        exp = bool(cinfo["expected_physically_stable"])
        case_results.append(
            PhysicalCaseEval(
                case_name=case_name,
                num_bricks=len(grid_list),
                expected_physically_stable=exp,
                predicted_stable=res.is_stable,
                peak_utilization=res.peak_utilization,
                equilibrium_residual_n=res.equilibrium_residual_n,
                solve_time_ms=res.solve_time_ms,
                matched_ground_truth=(res.is_stable == exp),
            )
        )

    bs_res = evaluate_bricksim_demo_topology(val)

    # Optional StableText2Brick reference stability_scores agreement
    s2b_n = 0
    substud_agree = 0
    kernel_agree = 0
    score_diffs: List[float] = []

    if s2b_records:
        from topo_brick.evaluation.static_equilibrium import SparseStaticEquilibriumValidator
        from topo_brick.kernel.construction_kernel import parse_external_record_specs
        from topo_brick.interface.brick_catalog import STABLETEXT2BRICK_8PART_KIT

        s2b_mass_map = {
            (1, 1): 0.00043,
            (1, 2): 0.00081,
            (1, 4): 0.00157,
            (1, 6): 0.00228,
            (1, 8): 0.00303,
            (2, 2): 0.00115,
            (2, 4): 0.00216,
            (2, 6): 0.00323,
        }
        k_val = SparseStaticEquilibriumValidator(clutch_force_capacity_grams=99.9)
        for rec in s2b_records:
            ref_min = rec.get("reference_min_stability_score")
            ref_max = rec.get("reference_max_stability_score")
            if ref_min is not None:
                # AvaLovelace/StableText2Brick stores safety margin = 1.0 - utilization in (20,20,20)
                ref_stable = float(ref_min) > 0.0
                ref_util = 1.0 - float(ref_min)
            elif ref_max is not None:
                ref_stable = float(ref_max) < 1.0
                ref_util = float(ref_max)
            else:
                continue
            ldr_text = str(rec["ldr_text"])
            specs, _ = parse_external_record_specs(ldr_text, allowed_part_ids=STABLETEXT2BRICK_8PART_KIT)
            if not specs:
                continue
            s2b_n += 1
            st = AssemblyState()
            min_z = min(
                min(b.transform(sp.pose).zmin for b in get_brick_primitive(sp.part_id).hull_boxes)
                for sp in specs
            )
            grid_b: List[Tuple[int, int, int, int, int, float]] = []
            for sp in specs:
                prim = get_brick_primitive(sp.part_id)
                wbs = tuple(b.transform(sp.pose) for b in prim.hull_boxes)
                st.place_root_part(sp.part_id, sp.pose, is_grounded=(min(b.zmin for b in wbs) <= min_z))
                bx0, bx1 = min(b.xmin for b in wbs), max(b.xmax for b in wbs)
                by0, by1 = min(b.ymin for b in wbs), max(b.ymax for b in wbs)
                bz0 = min(b.zmin for b in wbs)
                hx, wy = (bx1 - bx0) // 20, (by1 - by0) // 20
                m_kg = s2b_mass_map.get((min(hx, wy), max(hx, wy)), float(prim.mass_grams) / 1000.0)
                grid_b.append((bx0 // 20, by0 // 20, (bz0 - min_z) // 24, hx, wy, m_kg))

            r_sub = val.evaluate_grid_bricks(grid_b)
            r_ker = k_val.evaluate(st)
            if r_sub.is_stable == ref_stable:
                substud_agree += 1
            if r_ker.is_stable == ref_stable:
                kernel_agree += 1
            if np.isfinite(r_sub.peak_utilization) and np.isfinite(ref_util):
                score_diffs.append(abs(r_sub.peak_utilization - ref_util))

    matched_cnt = sum(1 for c in case_results if c.matched_ground_truth)
    return ModelAgreementReport(
        num_physical_cases=len(case_results),
        physical_cases_matched=matched_cnt,
        physical_hardware_accuracy=float(matched_cnt / max(len(case_results), 1)),
        physical_case_results=tuple(case_results),
        bricksim_demo_certified_stable=bs_res.is_stable,
        bricksim_demo_peak_utilization=bs_res.peak_utilization,
        s2b_num_evaluated=s2b_n,
        s2b_substud_agreement_rate=float(substud_agree / max(s2b_n, 1)) if s2b_n > 0 else 1.0,
        s2b_kernel_lp_agreement_rate=float(kernel_agree / max(s2b_n, 1)) if s2b_n > 0 else 1.0,
        s2b_mean_max_score_mae=float(np.mean(score_diffs)) if score_diffs else 0.0,
    )


def evaluate_upstream_horizontal_contact_fixture() -> Dict[str, Any]:
    """
    Evaluate the exact 16-brick upstream `StableLego` solver fixture
    `stick_heavy_good_test_horizontal_force.json` (git blob `e5c1175606c447e10600f80a89f04b3d99374a82`),
    reporting its true provenance (`upstream_stablelego_solver_fixture_adjacent_brick_horizontal_press`,
    `is_confirmed_hardware_experiment=False`) separately from the 8 hardware-labeled cases.
    """
    assets = load_physical_validation_assets()
    cfg = assets["config"]
    lib = assets["lego_library"]
    fx = assets["stable_lego_solver_fixtures"]["stick_heavy_good_test_horizontal_force"]
    grid_list: List[Tuple[int, int, int, int, int, float]] = []
    for _k, b in fx["bricks"].items():
        le = lib[str(b["brick_id"])]
        ori = int(b["ori"])
        hx, wy = (
            (int(le["height"]), int(le["width"]))
            if ori == 0
            else (int(le["width"]), int(le["height"]))
        )
        grid_list.append((int(b["x"]), int(b["y"]), int(b["z"]), hx, wy, float(le["mass"])))

    val = CalibratedSubStudPolygonValidator(
        clutch_capacity_grams=float(cfg["T"]),
        gravity_m_s2=float(cfg["g"]),
        brick_unit_height_m=float(cfg["Brick_Unit_Height"]),
        brick_unit_length_m=float(cfg["Brick_Unit_Length"]),
        alpha=float(cfg["Alpha"]),
        beta=float(cfg["Beta"]),
    )
    res_with_side_press = val.evaluate_grid_bricks(grid_list, enable_side_wall_press=True)
    res_without_side_press = val.evaluate_grid_bricks(grid_list, enable_side_wall_press=False)
    return {
        "fixture_name": "stick_heavy_good_test_horizontal_force.json",
        "git_blob_sha1": str(fx["git_blob_sha1"]),
        "sha256": str(fx["sha256"]),
        "num_bricks": len(grid_list),
        "label_source": str(fx["label_source"]),
        "is_confirmed_hardware_experiment": bool(fx["is_confirmed_hardware_experiment"]),
        "stable_with_side_wall_press": bool(res_with_side_press.is_stable),
        "peak_utilization_with_side_wall_press": float(res_with_side_press.peak_utilization),
        "stable_without_side_wall_press": bool(res_without_side_press.is_stable),
        "peak_utilization_without_side_wall_press": float(res_without_side_press.peak_utilization),
    }
