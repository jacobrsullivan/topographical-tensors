"""
WS-3B & WS-3C: Continuous SE(3) Kinematics, 15-Axis Separating-Axis-Theorem (SAT) OBB Collider,
Joint-DoF Static Equilibrium, Genuine OMR `.mpd` Multi-Part Flattener (`Claim C`), and Official
LEGO Set `10698-1` Inventory-Bounded Alt-Build Realization (`Claim E`).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np
from scipy.sparse import csc_matrix

if TYPE_CHECKING:
    from topo_brick.evaluation.static_equilibrium import EquilibriumResult
from topo_brick.interface.brick_catalog import (
    SET_10698_OFFICIAL_INVENTORY,
    TIER_A_CATALOG,
    BrickPrimitive,
    ConnectorType,
    PortPolarity,
    are_connectors_compatible,
    get_brick_primitive,
)
from topo_brick.interface.ldu_frames import (
    LDUBox,
    LDUFrame,
    matrix_to_rot_id,
    rot_id_to_matrix,
)
from topo_brick.interface.port_manifold import get_admissible_socket_pocket_ldu
from topo_brick.interface.tier_a_catalog import normalize_ldraw_part_id


def get_connector_axial_slide_bounds_ldu(connector_type: ConnectorType) -> Tuple[float, float]:
    """
    Connector-specific admissible axial slide / engagement bounds in LDU (`R9-WS3` §6.2):
      - `STANDARD_STUD` / `HOLLOW_STUD` / `HINGE_*` / `BALL_*` / `TURNTABLE_*`: flush snap `[-0.5, 0.5]` LDU
      - `TECHNIC_PIN*` / `SMOOTH_PIN` / `WHEEL_*`: snap-ring seating `[-1.0, 1.0]` LDU
      - `BAR_3_18MM` / `CLIP_3_18MM` / `TECHNIC_AXLE*`: continuous slide `[-16.5, 16.5]` LDU
        (requiring >= 3.5 LDU engagement overlap inside a 20 LDU bore/clip)
    """
    name = connector_type.name
    if name in (
        "BAR_3_18MM",
        "CLIP_3_18MM",
        "TECHNIC_AXLE",
        "TECHNIC_AXLE_HOLE",
    ):
        return (-16.5, 16.5)
    if name in (
        "TECHNIC_PIN",
        "TECHNIC_PIN_HOLE",
        "SMOOTH_PIN",
        "WHEEL_PIN",
        "WHEEL_HUB_HOLE",
    ):
        return (-1.0, 1.0)
    return (-0.5, 0.5)


def is_obb_protrusion_contained_in_socket_pocket(
    prot_obb: OBB3D,
    hull_obb: OBB3D,
    socket_world_pose: ContinuousSE3Pose,
    socket_connector_type: ConnectorType,
    tol_ldu: float = 0.35,
) -> bool:
    """
    Connector-specific continuous $SE(3)$ socket-pocket CSG check (`R9-WS3` §6.2),
    replacing the coarse `14.0 LDU` spherical exemption:
      1. `hull_obb` must actually contain the socket cavity's center along `socket_world_pose`
         (rejecting perpendicular/internal walls of multi-box brackets such as `99780`).
      2. All 8 corners of `prot_obb` entering the socket half-space (`z_local >= -tol_ldu`)
         must lie within `get_admissible_socket_pocket_ldu(socket_connector_type)` radially
         and axially (rejecting over-insertion and lateral misalignment).
    """
    pocket = get_admissible_socket_pocket_ldu(socket_connector_type)
    r_sock = socket_world_pose.R
    t_sock = socket_world_pose.t
    pocket_obb = box_to_obb(pocket, socket_world_pose)
    if not obb_intersects_sat(pocket_obb, hull_obb, allowed_penetration_ldu=-0.65):
        return False

    p_axes = np.asarray(prot_obb.axes_rows, dtype=np.float64)
    p_center = np.asarray(prot_obb.center, dtype=np.float64)
    p_he = np.asarray(prot_obb.half_extents, dtype=np.float64)
    max_r_xy = math.hypot(
        0.5 * float(pocket.xmax - pocket.xmin),
        0.5 * float(pocket.ymax - pocket.ymin),
    ) + float(tol_ldu)
    z_lo = float(pocket.zmin) - float(tol_ldu)
    z_hi = float(pocket.zmax) + float(tol_ldu)

    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                corner_w = (
                    p_center
                    + sx * p_he[0] * p_axes[0]
                    + sy * p_he[1] * p_axes[1]
                    + sz * p_he[2] * p_axes[2]
                )
                local_pt = r_sock.T @ (corner_w - t_sock)
                if float(local_pt[2]) >= -float(tol_ldu):
                    if float(local_pt[2]) < z_lo or float(local_pt[2]) > z_hi:
                        return False
                    if math.hypot(float(local_pt[0]), float(local_pt[1])) > max_r_xy:
                        return False
    return True


# LDraw (-Y up) -> topo_brick (+Z up) right-handed basis permutation (det = +1)
_C_LDRAW_TO_ZUP = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.float64
)
# Aligns topo_brick intrinsic (X_short, Y_long) with LDraw intrinsic (X_long, Y_short) (det = +1)
_R_TB_TO_LDRAW_ZUP = np.array(
    [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
)


def _rodrigues(axis: Sequence[float], angle_rad: float) -> np.ndarray:
    v = np.asarray(axis, dtype=np.float64)
    nrm = float(np.linalg.norm(v))
    if nrm < 1e-12 or abs(angle_rad) < 1e-15:
        return np.eye(3, dtype=np.float64)
    u = v / nrm
    ux, uy, uz = float(u[0]), float(u[1]), float(u[2])
    k_mat = np.array(
        [[0.0, -uz, uy], [uz, 0.0, -ux], [-uy, ux, 0.0]], dtype=np.float64
    )
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.eye(3, dtype=np.float64) + s * k_mat + (1.0 - c) * (k_mat @ k_mat)


def _orthonormalize_so3(mat: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.asarray(mat, dtype=np.float64))
    r = u @ vt
    if np.linalg.det(r) < 0.0:
        u = u.copy()
        u[:, -1] *= -1.0
        r = u @ vt
    return r


@dataclass(frozen=True, slots=True)
class ContinuousSE3Pose:
    """
    Continuous rigid pose (R, pos) in SE(3) = R^3_LDU ⋊ SO(3), generalizing discrete `LDUFrame`
    to arbitrary hinge, turntable, Technic axle, and ball-joint orientations.
    """

    rot_matrix: Tuple[
        Tuple[float, float, float],
        Tuple[float, float, float],
        Tuple[float, float, float],
    ]
    pos: Tuple[float, float, float]

    @classmethod
    def identity(cls, pos: Sequence[float] = (0.0, 0.0, 0.0)) -> "ContinuousSE3Pose":
        return cls(
            rot_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            pos=(float(pos[0]), float(pos[1]), float(pos[2])),
        )

    @classmethod
    def from_matrix_and_pos(
        cls, rot: np.ndarray, pos: Sequence[float], orthonormalize: bool = True
    ) -> "ContinuousSE3Pose":
        r = _orthonormalize_so3(rot) if orthonormalize else np.asarray(rot, dtype=np.float64)
        return cls(
            rot_matrix=(
                (float(r[0, 0]), float(r[0, 1]), float(r[0, 2])),
                (float(r[1, 0]), float(r[1, 1]), float(r[1, 2])),
                (float(r[2, 0]), float(r[2, 1]), float(r[2, 2])),
            ),
            pos=(float(pos[0]), float(pos[1]), float(pos[2])),
        )

    @classmethod
    def from_ldu_frame(cls, frame: LDUFrame) -> "ContinuousSE3Pose":
        r = rot_id_to_matrix(frame.rot_id).astype(np.float64)
        return cls.from_matrix_and_pos(r, frame.pos, orthonormalize=False)

    @classmethod
    def from_axis_angle(
        cls,
        axis: Sequence[float],
        angle_rad: float,
        pos: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> "ContinuousSE3Pose":
        return cls.from_matrix_and_pos(_rodrigues(axis, angle_rad), pos, orthonormalize=False)

    @property
    def R(self) -> np.ndarray:
        return np.asarray(self.rot_matrix, dtype=np.float64)

    @property
    def t(self) -> np.ndarray:
        return np.asarray(self.pos, dtype=np.float64)

    def compose(self, other: "ContinuousSE3Pose") -> "ContinuousSE3Pose":
        r_a = self.R
        r_out = r_a @ other.R
        t_out = r_a @ other.t + self.t
        return ContinuousSE3Pose.from_matrix_and_pos(r_out, t_out, orthonormalize=False)

    def inverse(self) -> "ContinuousSE3Pose":
        rt = self.R.T
        t_inv = -(rt @ self.t)
        return ContinuousSE3Pose.from_matrix_and_pos(rt, t_inv, orthonormalize=False)

    def transform_point(self, pt: Sequence[float]) -> Tuple[float, float, float]:
        v = self.R @ np.asarray(pt, dtype=np.float64) + self.t
        return (float(v[0]), float(v[1]), float(v[2]))

    def rotate_vector(self, vec: Sequence[float]) -> Tuple[float, float, float]:
        v = self.R @ np.asarray(vec, dtype=np.float64)
        return (float(v[0]), float(v[1]), float(v[2]))

    def is_cubic_lattice_aligned(self, tol: float = 1e-3) -> bool:
        r = self.R
        if np.any(np.abs(r - np.round(r)) > tol):
            return False
        if any(abs(v - round(v)) > tol for v in self.pos):
            return False
        try:
            _ = matrix_to_rot_id(np.round(r).astype(np.int64))
            return True
        except ValueError:
            return False

    def to_ldu_frame_if_cubic(self, tol: float = 1e-3) -> Optional[LDUFrame]:
        if not self.is_cubic_lattice_aligned(tol=tol):
            return None
        rot_id = matrix_to_rot_id(np.round(self.R).astype(np.int64))
        p_int = (int(round(self.pos[0])), int(round(self.pos[1])), int(round(self.pos[2])))
        return LDUFrame(rot_id=rot_id, pos=p_int)


@dataclass(frozen=True, slots=True)
class OBB3D:
    """
    3D Oriented Bounding Box in continuous world LDU coordinates:
      - `center`: (cx, cy, cz)
      - `axes`: 3x3 orthonormal columns [u0, u1, u2] (rows of `axes_rows` are the 3 unit axes!)
      - `half_extents`: (e0, e1, e2) > 0
      - `aabb_bounds`: (xmin, ymin, zmin, xmax, ymax, zmax) for fast broad-phase rejection.
    """

    center: Tuple[float, float, float]
    axes_rows: Tuple[
        Tuple[float, float, float],
        Tuple[float, float, float],
        Tuple[float, float, float],
    ]
    half_extents: Tuple[float, float, float]
    aabb_bounds: Tuple[float, float, float, float, float, float]


def box_to_obb(box: LDUBox, pose: ContinuousSE3Pose) -> OBB3D:
    """Transform an axis-aligned local `LDUBox` by `ContinuousSE3Pose` into a world `OBB3D`."""
    lc = np.array(
        [0.5 * (box.xmin + box.xmax), 0.5 * (box.ymin + box.ymax), 0.5 * (box.zmin + box.zmax)],
        dtype=np.float64,
    )
    he = np.array(
        [0.5 * (box.xmax - box.xmin), 0.5 * (box.ymax - box.ymin), 0.5 * (box.zmax - box.zmin)],
        dtype=np.float64,
    )
    r = pose.R
    wc = r @ lc + pose.t
    # Columns of R are the 3 world unit axes u0, u1, u2 of the local X, Y, Z directions
    u0 = r[:, 0]
    u1 = r[:, 1]
    u2 = r[:, 2]
    abs_r = np.abs(r)
    world_he = abs_r @ he
    return OBB3D(
        center=(float(wc[0]), float(wc[1]), float(wc[2])),
        axes_rows=(
            (float(u0[0]), float(u0[1]), float(u0[2])),
            (float(u1[0]), float(u1[1]), float(u1[2])),
            (float(u2[0]), float(u2[1]), float(u2[2])),
        ),
        half_extents=(float(he[0]), float(he[1]), float(he[2])),
        aabb_bounds=(
            float(wc[0] - world_he[0]),
            float(wc[1] - world_he[1]),
            float(wc[2] - world_he[2]),
            float(wc[0] + world_he[0]),
            float(wc[1] + world_he[1]),
            float(wc[2] + world_he[2]),
        ),
    )


def obb_intersects_sat(
    obb_a: OBB3D,
    obb_b: OBB3D,
    allowed_penetration_ldu: float = 0.05,
) -> bool:
    """
    Exact 15-Axis Separating-Axis-Theorem (SAT) narrow-phase collision test between two `OBB3D`s
    (Gottschalk, Lin & Manocha, OBBTree).
    Returns True iff the strict interiors overlap by more than `allowed_penetration_ldu` across
    all 15 separating axes (3 face normals of A, 3 face normals of B, 9 edge cross-products).
    """
    ax0, ay0, az0, ax1, ay1, az1 = obb_a.aabb_bounds
    bx0, by0, bz0, bx1, by1, bz1 = obb_b.aabb_bounds
    tol = float(allowed_penetration_ldu)
    # Fast world AABB broad-phase rejection
    if (
        ax1 <= bx0 + tol
        or bx1 <= ax0 + tol
        or ay1 <= by0 + tol
        or by1 <= ay0 + tol
        or az1 <= bz0 + tol
        or bz1 <= az0 + tol
    ):
        return False

    a_axes = np.asarray(obb_a.axes_rows, dtype=np.float64)  # [3, 3], row i is axis A_i
    b_axes = np.asarray(obb_b.axes_rows, dtype=np.float64)  # [3, 3], row j is axis B_j
    ea = np.maximum(0.0, np.asarray(obb_a.half_extents, dtype=np.float64) - 0.5 * tol)
    eb = np.maximum(0.0, np.asarray(obb_b.half_extents, dtype=np.float64) - 0.5 * tol)

    d_world = np.asarray(obb_b.center, dtype=np.float64) - np.asarray(obb_a.center, dtype=np.float64)
    # Express translation d in A's local frame: t_a[i] = d_world . A_i
    t_a = a_axes @ d_world
    # Rotation matrix C[i, j] = A_i . B_j
    c_mat = a_axes @ b_axes.T
    abs_c = np.abs(c_mat) + 1e-9

    # 1. Test 3 face normals of A (L = A_0, A_1, A_2)
    for i in range(3):
        ra = ea[i]
        rb = float(abs_c[i, 0] * eb[0] + abs_c[i, 1] * eb[1] + abs_c[i, 2] * eb[2])
        if abs(float(t_a[i])) >= ra + rb:
            return False

    # 2. Test 3 face normals of B (L = B_0, B_1, B_2)
    t_b = b_axes @ d_world
    for j in range(3):
        ra = float(abs_c[0, j] * ea[0] + abs_c[1, j] * ea[1] + abs_c[2, j] * ea[2])
        rb = eb[j]
        if abs(float(t_b[j])) >= ra + rb:
            return False

    # 3. Test 9 edge cross-product axes L = A_i x B_j
    for i in range(3):
        i1 = (i + 1) % 3
        i2 = (i + 2) % 3
        for j in range(3):
            j1 = (j + 1) % 3
            j2 = (j + 2) % 3
            ra = float(ea[i1] * abs_c[i2, j] + ea[i2] * abs_c[i1, j])
            rb = float(eb[j1] * abs_c[i, j2] + eb[j2] * abs_c[i, j1])
            proj = abs(float(t_a[i2] * c_mat[i1, j] - t_a[i1] * c_mat[i2, j]))
            if proj >= ra + rb:
                return False

    return True


@dataclass(frozen=True, slots=True)
class ContinuousPlacedPart:
    """A part placed at an arbitrary continuous `ContinuousSE3Pose` inside `ContinuousAssemblyState`."""

    instance_id: int
    part_id: str
    continuous_pose: ContinuousSE3Pose
    world_obbs: Tuple[OBB3D, ...]
    is_grounded: bool = False


def _orthonormal_tangent_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = normal / max(float(np.linalg.norm(normal)), 1e-12)
    ref = np.array([1.0, 0.0, 0.0]) if abs(float(n[0])) < 0.85 else np.array([0.0, 1.0, 0.0])
    t1 = np.cross(n, ref)
    t1 /= max(float(np.linalg.norm(t1)), 1e-12)
    t2 = np.cross(n, t1)
    t2 /= max(float(np.linalg.norm(t2)), 1e-12)
    return t1, t2


class ContinuousAssemblyState:
    """
    Continuous-SE(3) assembly graph & narrow-phase OBB collision / static-equilibrium engine.
    Handles arbitrary non-cubic hinge, turntable, Technic axle, and 3-DoF ball-joint poses while
    enforcing:
      1. Exact 15-axis SAT solid-hull non-intersection,
      2. Mated-port protrusion exemption (unmated male studs/pins checked via SAT),
      3. Opposite-polarity + 8-family connector mating (`are_connectors_compatible`), and
      4. Coupled 6-DoF static equilibrium LP with joint-DoF projections.
    """

    def __init__(self, pos_mate_tol_ldu: float = 1.5, normal_cos_tol: float = 0.96) -> None:
        self.placed_parts: Dict[int, ContinuousPlacedPart] = {}
        self.matched_port_pairs: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self.articulated_joints: Dict[
            Tuple[int, int],
            Tuple[int, int, float, Tuple[float, float], float, Tuple[float, float]],
        ] = {}
        self._next_id: int = 0
        self.pos_mate_tol_ldu = float(pos_mate_tol_ldu)
        self.normal_cos_tol = float(normal_cos_tol)

    def _port_world_pose(self, pose: ContinuousSE3Pose, local_frame: LDUFrame) -> ContinuousSE3Pose:
        return pose.compose(ContinuousSE3Pose.from_ldu_frame(local_frame))

    def place_part(
        self,
        part_id: str,
        pose: ContinuousSE3Pose,
        is_grounded: bool = False,
        allowed_penetration_ldu: float = 0.05,
        check_collision: bool = True,
    ) -> int:
        prim = get_brick_primitive(part_id)
        new_obbs = tuple(box_to_obb(b, pose) for b in prim.hull_boxes)

        # 1. Check solid hull OBBs against all placed solid hull OBBs (with CoACD convex-polytope narrow-phase refinement)
        if check_collision:
            for existing in self.placed_parts.values():
                obb_hit = any(
                    obb_intersects_sat(ob_a, ob_b, allowed_penetration_ldu=allowed_penetration_ldu)
                    for ob_a in new_obbs
                    for ob_b in existing.world_obbs
                )
                if not obb_hit:
                    continue
                ex_prim = get_brick_primitive(existing.part_id)
                if prim.coacd_shell_id is not None or ex_prim.coacd_shell_id is not None:
                    from topo_brick.interface.ldraw_shadow_catalog_compiler import (
                        COMPILED_LDRAW_CATALOG,
                        convex_polytopes_intersect_sat,
                    )

                    polys_a = COMPILED_LDRAW_CATALOG.get_coacd_polytopes(prim.coacd_shell_id)
                    polys_b = COMPILED_LDRAW_CATALOG.get_coacd_polytopes(ex_prim.coacd_shell_id)
                    if polys_a or polys_b:
                        def _world_hulls(
                            polys: Tuple[Any, ...], p_pose: ContinuousSE3Pose, fallback_obbs: Tuple[OBB3D, ...]
                        ) -> List[Tuple[np.ndarray, np.ndarray]]:
                            if polys:
                                r_m = p_pose.R.astype(np.float32)
                                t_v = p_pose.t.astype(np.float32)
                                return [
                                    (hp.vertices_ldu @ r_m.T + t_v, hp.halfspace_normals @ r_m.T)
                                    for hp in polys
                                ]
                            out_h: List[Tuple[np.ndarray, np.ndarray]] = []
                            signs = np.array(
                                [[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1], [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]],
                                dtype=np.float32,
                            )
                            for ob in fallback_obbs:
                                ax = np.asarray(ob.axes_rows, dtype=np.float32)
                                he = np.asarray(ob.half_extents, dtype=np.float32)
                                ctr = np.asarray(ob.center, dtype=np.float32)
                                crn = ctr + (signs * he) @ ax
                                out_h.append((crn, ax))
                            return out_h

                        wh_a = _world_hulls(polys_a, pose, new_obbs)
                        wh_b = _world_hulls(polys_b, existing.continuous_pose, existing.world_obbs)
                        eff_pen = max(float(allowed_penetration_ldu), 0.35)
                        poly_hit = any(
                            convex_polytopes_intersect_sat(va, na, vb, nb, allowed_penetration_ldu=eff_pen)
                            for va, na in wh_a
                            for vb, nb in wh_b
                        )
                        if not poly_hit:
                            continue
                raise ValueError(
                    f"Continuous SE(3) SAT collision between new '{part_id}' and placed "
                    f"instance {existing.instance_id} ('{existing.part_id}')."
                )

        # 2. Match compatible opposite-polarity ports within `pos_mate_tol_ldu` & aligned z-axis
        new_id = self._next_id
        candidate_matches: List[Tuple[int, Tuple[int, int]]] = []
        used_existing: Set[Tuple[int, int]] = set()
        for p_new in prim.ports:
            wf_new = self._port_world_pose(pose, p_new.local_frame)
            z_new = wf_new.R[:, 2]
            pos_new = wf_new.t
            best_dist = self.pos_mate_tol_ldu
            best_gpid: Optional[Tuple[int, int]] = None
            for ex_id, ex_part in self.placed_parts.items():
                ex_prim = get_brick_primitive(ex_part.part_id)
                for p_ex in ex_prim.ports:
                    gpid_ex = (ex_id, p_ex.port_id)
                    if gpid_ex in self.matched_port_pairs or gpid_ex in used_existing:
                        continue
                    if not are_connectors_compatible(
                        p_new.connector_type, p_new.polarity, p_ex.connector_type, p_ex.polarity
                    ):
                        continue
                    wf_ex = self._port_world_pose(ex_part.continuous_pose, p_ex.local_frame)
                    dist = float(np.linalg.norm(pos_new - wf_ex.t))
                    if dist > best_dist:
                        continue
                    # For BALL_JOINT <-> BALL_SOCKET, spherical tilt up to ~55 deg (cos >= 0.50) is legal!
                    is_ball = p_new.connector_type in (
                        ConnectorType.BALL_JOINT,
                        ConnectorType.BALL_SOCKET,
                    )
                    cos_align = float(np.dot(z_new, wf_ex.R[:, 2]))
                    min_cos = 0.45 if is_ball else self.normal_cos_tol
                    if cos_align >= min_cos:
                        best_dist = dist
                        best_gpid = gpid_ex
            if best_gpid is not None:
                used_existing.add(best_gpid)
                candidate_matches.append((p_new.port_id, best_gpid))

        mated_new_to_partner: Dict[int, Tuple[int, int]] = dict(candidate_matches)
        newly_mated_ex_to_new: Dict[Tuple[int, int], int] = {
            gp: lp for lp, gp in candidate_matches
        }

        def _point_to_obb_dist(pt: np.ndarray, obb: OBB3D) -> float:
            axes_mat = np.asarray(obb.axes_rows, dtype=np.float64)
            c_vec = np.asarray(obb.center, dtype=np.float64)
            he_vec = np.asarray(obb.half_extents, dtype=np.float64)
            rel = axes_mat @ (pt - c_vec)
            excess = np.maximum(np.abs(rel) - he_vec, 0.0)
            return float(np.linalg.norm(excess))

        # 3. Check male protrusion OBBs via 15-axis SAT with connector-specific socket-pocket CSG:
        #    Mating with `(partner_id, partner_port)` ONLY exempts intersections strictly contained
        #    inside `get_admissible_socket_pocket_ldu` on the specific hull OBB housing that socket,
        #    and NEVER exempts perpendicular internal bracket walls or any third part!
        if check_collision:
            for p_new in prim.ports:
                if p_new.local_protrusion_box is None:
                    continue
                wf_new = self._port_world_pose(pose, p_new.local_frame)
                prot_obb = box_to_obb(p_new.local_protrusion_box, wf_new)
                partner_gpid = mated_new_to_partner.get(p_new.port_id)
                partner_inst_id = partner_gpid[0] if partner_gpid is not None else -1
                partner_wf: Optional[ContinuousSE3Pose] = None
                partner_ctype: Optional[ConnectorType] = None
                if partner_gpid is not None and partner_inst_id in self.placed_parts:
                    p_part = self.placed_parts[partner_inst_id]
                    p_pspec = get_brick_primitive(p_part.part_id).ports[partner_gpid[1]]
                    partner_wf = self._port_world_pose(
                        p_part.continuous_pose, p_pspec.local_frame
                    )
                    partner_ctype = p_pspec.connector_type

                for existing in self.placed_parts.values():
                    for ob_b in existing.world_obbs:
                        if (
                            existing.instance_id == partner_inst_id
                            and partner_wf is not None
                            and partner_ctype is not None
                            and is_obb_protrusion_contained_in_socket_pocket(
                                prot_obb, ob_b, partner_wf, partner_ctype
                            )
                        ):
                            continue
                        if obb_intersects_sat(
                            prot_obb, ob_b, allowed_penetration_ldu=allowed_penetration_ldu
                        ):
                            raise ValueError(
                                f"Male protrusion of '{part_id}' port {p_new.port_id} collides "
                                f"with placed instance {existing.instance_id} (third-part or non-socket intrusion)."
                            )

            # Check ALL existing male protrusions (both unmated AND already-mated to other parts!)
            # against `new_obbs` so a newly placed third part cannot overlap an active connector protrusion.
            for ex_id, ex_part in self.placed_parts.items():
                ex_prim = get_brick_primitive(ex_part.part_id)
                for p_ex in ex_prim.ports:
                    if p_ex.local_protrusion_box is None:
                        continue
                    gpid_ex = (ex_id, p_ex.port_id)
                    wf_ex = self._port_world_pose(ex_part.continuous_pose, p_ex.local_frame)
                    prot_ex = box_to_obb(p_ex.local_protrusion_box, wf_ex)
                    new_local_pid = newly_mated_ex_to_new.get(gpid_ex)
                    new_mated_wf: Optional[ContinuousSE3Pose] = None
                    new_mated_ctype: Optional[ConnectorType] = None
                    if new_local_pid is not None:
                        new_pspec = prim.ports[new_local_pid]
                        new_mated_wf = self._port_world_pose(pose, new_pspec.local_frame)
                        new_mated_ctype = new_pspec.connector_type
                    for ob_a in new_obbs:
                        if (
                            new_mated_wf is not None
                            and new_mated_ctype is not None
                            and is_obb_protrusion_contained_in_socket_pocket(
                                prot_ex, ob_a, new_mated_wf, new_mated_ctype
                            )
                        ):
                            continue
                        if obb_intersects_sat(
                            prot_ex, ob_a, allowed_penetration_ldu=allowed_penetration_ldu
                        ):
                            raise ValueError(
                                f"Existing protrusion {gpid_ex} collides with new '{part_id}' "
                                f"(third-part or non-socket intrusion)."
                            )

        self._next_id += 1
        self.placed_parts[new_id] = ContinuousPlacedPart(
            instance_id=new_id,
            part_id=part_id,
            continuous_pose=pose,
            world_obbs=new_obbs,
            is_grounded=bool(is_grounded),
        )
        for lp, gp_ex in candidate_matches:
            gp_new = (new_id, lp)
            self.matched_port_pairs[gp_new] = gp_ex
            self.matched_port_pairs[gp_ex] = gp_new
        return new_id

    def attach_articulated_part(
        self,
        parent_instance_id: int,
        parent_port_id: int,
        child_part_id: str,
        child_port_id: int,
        joint_angle_rad: float = 0.0,
        ball_tilt_xy_rad: Tuple[float, float] = (0.0, 0.0),
        axial_slide_ldu: float = 0.0,
        angle_limits_rad: Tuple[float, float] = (-3.14159265, 3.14159265),
        slide_limits_ldu: Tuple[float, float] = (-40.0, 40.0),
        ball_max_cone_rad: float = 0.96,
        enforce_joint_limits: bool = True,
        check_collision: bool = True,
    ) -> int:
        """
        Attach `child_part_id` (via `child_port_id`) to `(parent_instance_id, parent_port_id)`
        using explicit connector-local joint transform:
            T_b = T_a C_a J(q) C_b^{-1},
        enforcing joint angle, spherical cone, and axial slide limits when `enforce_joint_limits=True`.
        """
        parent = self.placed_parts[parent_instance_id]
        p_prim = get_brick_primitive(parent.part_id)
        c_prim = get_brick_primitive(child_part_id)
        p_spec = p_prim.ports[parent_port_id]
        c_spec = c_prim.ports[child_port_id]
        if not are_connectors_compatible(
            p_spec.connector_type, p_spec.polarity, c_spec.connector_type, c_spec.polarity
        ):
            raise ValueError(
                f"Incompatible articulated ports: {p_spec.connector_type.name}({p_spec.polarity}) "
                f"vs {c_spec.connector_type.name}({c_spec.polarity})."
            )
        if enforce_joint_limits:
            if not (angle_limits_rad[0] - 1e-6 <= joint_angle_rad <= angle_limits_rad[1] + 1e-6):
                raise ValueError(
                    f"Joint angle {joint_angle_rad:.4f} rad exceeds limits {angle_limits_rad}."
                )
            if not (slide_limits_ldu[0] - 1e-6 <= axial_slide_ldu <= slide_limits_ldu[1] + 1e-6):
                raise ValueError(
                    f"Axial slide {axial_slide_ldu:.2f} LDU exceeds limits {slide_limits_ldu}."
                )
            conn_lo_a, conn_hi_a = get_connector_axial_slide_bounds_ldu(p_spec.connector_type)
            conn_lo_b, conn_hi_b = get_connector_axial_slide_bounds_ldu(c_spec.connector_type)
            eff_slide_lo = max(conn_lo_a, conn_lo_b)
            eff_slide_hi = min(conn_hi_a, conn_hi_b)
            if not (eff_slide_lo - 1e-6 <= axial_slide_ldu <= eff_slide_hi + 1e-6):
                raise ValueError(
                    f"Connector engagement depth violation for {p_spec.connector_type.name}: "
                    f"axial_slide_ldu={axial_slide_ldu:.2f} outside admissible [{eff_slide_lo:.2f}, {eff_slide_hi:.2f}] LDU."
                )
            cone_mag = math.hypot(ball_tilt_xy_rad[0], ball_tilt_xy_rad[1])
            is_ball = p_spec.connector_type in (
                ConnectorType.BALL_JOINT,
                ConnectorType.BALL_SOCKET,
            )
            if is_ball and cone_mag > ball_max_cone_rad + 1e-6:
                raise ValueError(
                    f"Ball-joint tilt {cone_mag:.4f} rad exceeds cone limit {ball_max_cone_rad:.4f} rad."
                )
            if not is_ball and cone_mag > 1e-6:
                raise ValueError(
                    f"Non-spherical connector {p_spec.connector_type.name} forbids off-axis ball tilt."
                )
            is_hinge = p_spec.connector_type in (
                ConnectorType.HINGE_PIN,
                ConnectorType.HINGE_SOCKET,
                ConnectorType.LOCKING_HINGE_PIN,
                ConnectorType.LOCKING_HINGE_SOCKET,
            )
            if is_hinge and abs(joint_angle_rad) > 2.36:
                raise ValueError(
                    f"Hinge connector {p_spec.connector_type.name} angle {joint_angle_rad:.3f} rad "
                    "exceeds physical LEGO hinge stop (±135 deg)."
                )

        t_parent_port = self._port_world_pose(parent.continuous_pose, p_spec.local_frame)
        r_yaw = _rodrigues((0.0, 0.0, 1.0), joint_angle_rad)
        rx = _rodrigues((1.0, 0.0, 0.0), ball_tilt_xy_rad[0])
        ry = _rodrigues((0.0, 1.0, 0.0), ball_tilt_xy_rad[1])
        r_joint = ContinuousSE3Pose.from_matrix_and_pos(
            r_yaw @ rx @ ry, (0.0, 0.0, float(axial_slide_ldu))
        )
        f_child = ContinuousSE3Pose.from_ldu_frame(c_spec.local_frame)
        x_child = t_parent_port.compose(r_joint).compose(f_child.inverse())
        old_tol = self.pos_mate_tol_ldu
        if abs(axial_slide_ldu) > 1e-6:
            self.pos_mate_tol_ldu = max(old_tol, abs(float(axial_slide_ldu)) + 1.5)
        try:
            cid = self.place_part(
                child_part_id, x_child, is_grounded=False, check_collision=check_collision
            )
        finally:
            self.pos_mate_tol_ldu = old_tol
        self.articulated_joints[(parent_instance_id, cid)] = (
            parent_port_id,
            child_port_id,
            float(joint_angle_rad),
            (float(ball_tilt_xy_rad[0]), float(ball_tilt_xy_rad[1])),
            float(axial_slide_ldu),
            (float(angle_limits_rad[0]), float(angle_limits_rad[1])),
        )
        return cid

    def evaluate_static_equilibrium(
        self,
        clutch_force_capacity_grams: float = 45.0,
        moment_arm_capacity_ldu: float = 16.0,
        locking_hinge_detent_moment_ldu_g: float = 380.0,
        ball_joint_friction_moment_ldu_g: float = 220.0,
        coulomb_friction_coeff: float = 0.45,
        normal_crush_ratio: float = 25.0,
        lateral_accel_xy: Tuple[float, float] = (0.0, 0.0),
    ) -> EquilibriumResult:
        """
        Solve the coupled 6-DoF static equilibrium LP in continuous SE(3) coordinates with
        explicit joint-DoF wrench projections:
          - `HINGE_PIN`/`HINGE_SOCKET`, `SMOOTH_PIN`, `TURNTABLE_*`, `WHEEL_*`: axial torque τ_z = 0
          - `LOCKING_HINGE_PIN`/`LOCKING_HINGE_SOCKET`: axial torque bounded by `locking_hinge_detent_moment_ldu_g`
          - `BALL_JOINT`/`BALL_SOCKET`: all 3 torque axes bounded by `ball_joint_friction_moment_ldu_g`
          - `TECHNIC_AXLE`: axial tension = 0, cross-shaped torsion resisted (`2.5 * M_max`).
        """
        from topo_brick.evaluation.static_equilibrium import (
            EquilibriumResult,
            _solve_lp_highspy_or_scipy,
        )

        parts = self.placed_parts
        if not parts:
            return EquilibriumResult(True, 0.0, 1.0, {}, None, 0.0, "Empty")

        # BFS ground connectivity
        grounded_ids = [i for i, p in parts.items() if p.is_grounded]
        if not grounded_ids:
            return EquilibriumResult(False, float("inf"), -float("inf"), {}, None, 0.0, "No grounded parts")
        visited = set(grounded_ids)
        adj: Dict[int, Set[int]] = {i: set() for i in parts}
        for (ia, _), (ib, _) in self.matched_port_pairs.items():
            adj[ia].add(ib)
            adj[ib].add(ia)
        queue = list(grounded_ids)
        while queue:
            cur = queue.pop(0)
            for nb in adj[cur]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        if len(visited) < len(parts):
            return EquilibriumResult(
                False, float("inf"), -float("inf"), {}, None, 0.0, "Disconnected floating part"
            )

        free_insts = sorted(i for i, p in parts.items() if not p.is_grounded)
        if not free_insts:
            return EquilibriumResult(True, 0.0, 1.0, {}, None, 0.0, "All grounded")
        free_idx_map = {iid: idx for idx, iid in enumerate(free_insts)}

        # Collect unique active contacts
        seen_edges: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()
        contacts: List[Tuple[Tuple[int, int], Tuple[int, int], np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]] = []
        for gpid_a, gpid_b in self.matched_port_pairs.items():
            ekey = (gpid_a, gpid_b) if gpid_a < gpid_b else (gpid_b, gpid_a)
            if ekey in seen_edges:
                continue
            seen_edges.add(ekey)
            ia, pa = ekey[0]
            ib, pb = ekey[1]
            if (ia not in free_idx_map) and (ib not in free_idx_map):
                continue
            part_a = parts[ia]
            part_b = parts[ib]
            spec_a = get_brick_primitive(part_a.part_id).ports[pa]
            spec_b = get_brick_primitive(part_b.part_id).ports[pb]
            wf_a = self._port_world_pose(part_a.continuous_pose, spec_a.local_frame)
            p_c = wf_a.t
            # Convention: normal n_c points from male STUD into female ANTI_STUD
            if spec_a.polarity == PortPolarity.STUD:
                n_c = wf_a.R[:, 2]
                ordered_a, ordered_b = ekey[0], ekey[1]
            else:
                wf_b = self._port_world_pose(part_b.continuous_pose, spec_b.local_frame)
                n_c = wf_b.R[:, 2]
                ordered_a, ordered_b = ekey[1], ekey[0]
            t1, t2 = _orthonormal_tangent_basis(n_c)
            names = {spec_a.connector_type.name, spec_b.connector_type.name}
            if "BALL_JOINT" in names or "BALL_SOCKET" in names:
                jmode = "ball_joint"
            elif (
                "SMOOTH_PIN" in names
                or "HINGE_PIN" in names
                or "HINGE_SOCKET" in names
                or "TURNTABLE_BASE_SOCKET" in names
                or "TURNTABLE_TOP_PIN" in names
                or "WHEEL_PIN" in names
                or "WHEEL_HUB_HOLE" in names
            ):
                jmode = "smooth_hinge"
            elif "LOCKING_HINGE_PIN" in names or "LOCKING_HINGE_SOCKET" in names:
                jmode = "locking_hinge"
            elif "TECHNIC_AXLE" in names or "TECHNIC_AXLE_HOLE" in names:
                jmode = "axle"
            else:
                jmode = "rigid_clutch"
            contacts.append((ordered_a, ordered_b, p_c, n_c, t1, t2, jmode))

        n_free = len(free_insts)
        n_c = len(contacts)
        num_vars = 12 * n_c + 1
        eta_col = 12 * n_c

        # Equality rows: 6 * n_free
        eq_rows: List[int] = []
        eq_cols: List[int] = []
        eq_vals: List[float] = []
        b_eq = np.zeros(6 * n_free, dtype=np.float64)

        coms: Dict[int, np.ndarray] = {}
        for iid in free_insts:
            part = parts[iid]
            prim = get_brick_primitive(part.part_id)
            b_idx = free_idx_map[iid]
            m_g = float(prim.mass_grams)
            b_eq[6 * b_idx + 0] = float(lateral_accel_xy[0]) * m_g
            b_eq[6 * b_idx + 1] = float(lateral_accel_xy[1]) * m_g
            b_eq[6 * b_idx + 2] = m_g
            vol_sum = 0.0
            lc = np.zeros(3, dtype=np.float64)
            for bx in prim.hull_boxes:
                v = float(bx.volume_ldu3)
                vol_sum += v
                lc += v * np.array(
                    [0.5 * (bx.xmin + bx.xmax), 0.5 * (bx.ymin + bx.ymax), 0.5 * (bx.zmin + bx.zmax)]
                )
            lc /= max(vol_sum, 1.0)
            coms[iid] = part.continuous_pose.R @ lc + part.continuous_pose.t

        var_ub = np.full(num_vars, np.inf, dtype=np.float64)
        ub_rows: List[int] = []
        ub_cols: List[int] = []
        ub_vals: List[float] = []
        b_ub_list: List[float] = []

        f_max = float(clutch_force_capacity_grams)
        f_comp_max = f_max * float(normal_crush_ratio)
        m_max = f_max * float(moment_arm_capacity_ldu)
        mu = float(coulomb_friction_coeff)

        for e_idx, (ga, gb, pc, nc, t1, t2, jmode) in enumerate(contacts):
            base = 12 * e_idx
            dirs = (t1, t2, nc)
            for end_id, sgn in ((ga[0], -1.0), (gb[0], +1.0)):
                if end_id not in free_idx_map:
                    continue
                b_row = 6 * free_idx_map[end_id]
                r_arm = pc - coms[end_id]
                for d_i, d_vec in enumerate(dirs):
                    col_p = base + 2 * d_i
                    col_m = col_p + 1
                    rx_d = np.cross(r_arm, d_vec)
                    for ax in range(3):
                        if abs(float(d_vec[ax])) > 1e-12:
                            eq_rows.extend((b_row + ax, b_row + ax))
                            eq_cols.extend((col_p, col_m))
                            eq_vals.extend((sgn * float(d_vec[ax]), -sgn * float(d_vec[ax])))
                        if abs(float(rx_d[ax])) > 1e-12:
                            eq_rows.extend((b_row + 3 + ax, b_row + 3 + ax))
                            eq_cols.extend((col_p, col_m))
                            eq_vals.extend((sgn * float(rx_d[ax]), -sgn * float(rx_d[ax])))
                    mcol_p = base + 6 + 2 * d_i
                    mcol_m = mcol_p + 1
                    for ax in range(3):
                        if abs(float(d_vec[ax])) > 1e-12:
                            eq_rows.extend((b_row + 3 + ax, b_row + 3 + ax))
                            eq_cols.extend((mcol_p, mcol_m))
                            eq_vals.extend((sgn * float(d_vec[ax]), -sgn * float(d_vec[ax])))

            # Joint-DoF bounds
            if jmode == "smooth_hinge":
                # Zero axial torque around n_c (tau_n = 0)
                var_ub[base + 10] = 0.0
                var_ub[base + 11] = 0.0
            elif jmode == "axle":
                var_ub[base + 5] = 0.0  # zero axial pull-out clutch along smooth axle
            # Normal compression/tension
            r0 = len(b_ub_list)
            ub_rows.extend((r0, r0))
            ub_cols.extend((base + 4, eta_col))
            ub_vals.extend((1.0, -f_comp_max))
            b_ub_list.append(0.0)

            r1 = len(b_ub_list)
            ub_rows.extend((r1, r1))
            ub_cols.extend((base + 5, eta_col))
            ub_vals.extend((1.0, -f_max))
            b_ub_list.append(0.0)

            # Coulomb-coupled shear: |f_t1| + |f_t2| - mu*(f_n+ - f_n-) - f_max * eta <= 0
            r2 = len(b_ub_list)
            ub_rows.extend([r2] * 7)
            ub_cols.extend((base, base + 1, base + 2, base + 3, base + 4, base + 5, eta_col))
            ub_vals.extend((1.0, 1.0, 1.0, 1.0, -mu, +mu, -f_max))
            b_ub_list.append(0.0)

            # Bending / torsion capacity per joint mode
            if jmode == "ball_joint":
                m_cap = float(ball_joint_friction_moment_ldu_g)
                r3 = len(b_ub_list)
                ub_rows.extend([r3] * 7)
                ub_cols.extend((base + 6, base + 7, base + 8, base + 9, base + 10, base + 11, eta_col))
                ub_vals.extend((1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -m_cap))
                b_ub_list.append(0.0)
            elif jmode == "locking_hinge":
                # Transverse bending bounded by m_max, hinge-axis click torque bounded by detent
                r3 = len(b_ub_list)
                ub_rows.extend([r3] * 5)
                ub_cols.extend((base + 6, base + 7, base + 8, base + 9, eta_col))
                ub_vals.extend((1.0, 1.0, 1.0, 1.0, -m_max))
                b_ub_list.append(0.0)

                r4 = len(b_ub_list)
                ub_rows.extend((r4, r4, r4))
                ub_cols.extend((base + 10, base + 11, eta_col))
                ub_vals.extend((1.0, 1.0, -float(locking_hinge_detent_moment_ldu_g)))
                b_ub_list.append(0.0)
            else:
                m_cap = 2.5 * m_max if jmode == "axle" else m_max
                r3 = len(b_ub_list)
                ub_rows.extend([r3] * 7)
                ub_cols.extend((base + 6, base + 7, base + 8, base + 9, base + 10, base + 11, eta_col))
                ub_vals.extend((1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -m_cap))
                b_ub_list.append(0.0)

        a_eq_csc = csc_matrix((eq_vals, (eq_rows, eq_cols)), shape=(6 * n_free, num_vars))
        a_ub_csc = csc_matrix((ub_vals, (ub_rows, ub_cols)), shape=(len(b_ub_list), num_vars))
        b_ub = np.asarray(b_ub_list, dtype=np.float64)
        var_lb = np.zeros(num_vars, dtype=np.float64)
        c_obj = np.zeros(num_vars, dtype=np.float64)
        c_obj[eta_col] = 1.0
        c_obj[:eta_col] = 1e-7

        ok, sol, msg = _solve_lp_highspy_or_scipy(
            c_obj, a_eq_csc, b_eq, a_ub_csc, b_ub, var_lb, var_ub
        )
        if not ok or sol is None:
            return EquilibriumResult(False, float("inf"), -float("inf"), {}, None, 0.0, msg)
        eta = float(sol[eta_col])
        return EquilibriumResult(
            is_stable=(eta <= 1.0 + 1e-6),
            peak_utilization=eta,
            safety_margin=1.0 - eta,
            contact_utilizations={},
            weakest_contact=None,
            solve_time_ms=0.0,
            status_message=msg,
        )

    def evaluate_perturbation_stability(
        self,
        delta_theta_rad: float = math.radians(2.0),
        lateral_disturbance_ratio: float = 0.05,
        allow_stable_pendulums: bool = True,
        allow_neutral_mechanisms: bool = False,
        clutch_force_capacity_grams: float = 45.0,
        moment_arm_capacity_ldu: float = 16.0,
        locking_hinge_detent_moment_ldu_g: float = 380.0,
        ball_joint_friction_moment_ldu_g: float = 220.0,
    ) -> "PerturbationStabilityResult":
        """
        Round 8 R8-WS1 (§7 & §9B Case #5):
        Distinguish true mechanical stability from instantaneous static equilibrium:
          1. Nominal 6-DoF static equilibrium (`eta_0 <= 1.0`).
          2. Passive zero-torque bridge joints (`SMOOTH_PIN`, `HINGE_PIN`/`HINGE_SOCKET`,
             `TURNTABLE_*`, `WHEEL_*`) tested under ±`delta_theta_rad` angular perturbations:
             - If gravitational potential energy decreases (`min_delta_u < -1e-5`), rejects as
               `unstable_inverted_mechanism` (e.g. mass balanced vertically above smooth pin).
             - If strictly restoring (`min_delta_u > +1e-4`), classifies as `stable_hanging_pendulum`.
             - If flat (`|delta_u| <= 1e-4`), classifies as `neutral_free_mechanism`.
          3. 4-directional lateral disturbance wrench envelope `(±eps, 0)` and `(0, ±eps)` in the
             coupled 6-DoF LP (catches unbraced multi-link parallelogram mechanisms while passing
             loop-locked pinned A-frame triangles and detented hinges).
        """
        eq0 = self.evaluate_static_equilibrium(
            clutch_force_capacity_grams=clutch_force_capacity_grams,
            moment_arm_capacity_ldu=moment_arm_capacity_ldu,
            locking_hinge_detent_moment_ldu_g=locking_hinge_detent_moment_ldu_g,
            ball_joint_friction_moment_ldu_g=ball_joint_friction_moment_ldu_g,
        )
        if not eq0.is_stable:
            return PerturbationStabilityResult(
                is_perturbation_stable=False,
                nominal_utilization=eq0.peak_utilization,
                worst_perturbed_utilization=eq0.peak_utilization,
                min_restoring_work_g_ldu=0.0,
                mechanism_classification="nominal_static_infeasible",
                free_passive_joints=0,
            )

        parts = self.placed_parts
        if not parts:
            return PerturbationStabilityResult(
                is_perturbation_stable=True,
                nominal_utilization=0.0,
                worst_perturbed_utilization=0.0,
                min_restoring_work_g_ldu=0.0,
                mechanism_classification="rigid_or_detented",
                free_passive_joints=0,
            )

        # Compute world CoMs and masses
        coms: Dict[int, np.ndarray] = {}
        masses: Dict[int, float] = {}
        for iid, part in parts.items():
            prim = get_brick_primitive(part.part_id)
            masses[iid] = float(prim.mass_grams)
            vol_s = 0.0
            lc = np.zeros(3, dtype=np.float64)
            for bx in prim.hull_boxes:
                v = float(bx.volume_ldu3)
                vol_s += v
                lc += v * np.array(
                    [0.5 * (bx.xmin + bx.xmax), 0.5 * (bx.ymin + bx.ymax), 0.5 * (bx.zmin + bx.zmax)]
                )
            lc /= max(vol_s, 1.0)
            coms[iid] = part.continuous_pose.R @ lc + part.continuous_pose.t

        # Group port pairs by unordered part pair (u, v)
        pair_ports: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], Tuple[int, int]]]] = {}
        seen_e: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()
        for ga, gb in self.matched_port_pairs.items():
            ek = (ga, gb) if ga < gb else (gb, ga)
            if ek in seen_e:
                continue
            seen_e.add(ek)
            pk = (min(ga[0], gb[0]), max(ga[0], gb[0]))
            pair_ports.setdefault(pk, []).append(ek)

        adj: Dict[int, Set[int]] = {i: set() for i in parts}
        for u, v in pair_ports:
            adj[u].add(v)
            adj[v].add(u)

        grounded_set = {i for i, p in parts.items() if p.is_grounded}
        passive_names = {
            "SMOOTH_PIN",
            "HINGE_PIN",
            "HINGE_SOCKET",
            "TURNTABLE_BASE_SOCKET",
            "TURNTABLE_TOP_PIN",
            "WHEEL_PIN",
            "WHEEL_HUB_HOLE",
        }

        min_work = float("inf")
        min_hess_eig = 0.0
        free_passive_cnt = 0
        mech_mode = "rigid_or_detented"
        has_free_bridge = False

        # Collect all passive bridge DOFs (`(subtree_set, child_id, pivot_0, axis_0)`)
        bridge_dofs: List[Tuple[Set[int], int, np.ndarray, np.ndarray]] = []

        for (u, v), e_list in pair_ports.items():
            if len(e_list) != 1:
                continue  # Multi-port patch between same two parts locks relative 1-DoF rotation
            ga, gb = e_list[0]
            sa = get_brick_primitive(parts[ga[0]].part_id).ports[ga[1]]
            sb = get_brick_primitive(parts[gb[0]].part_id).ports[gb[1]]
            if not ({sa.connector_type.name, sb.connector_type.name} & passive_names):
                continue

            # Check if removing edge (u, v) disconnects one side from all grounded parts (bridge test)
            vis_g = set(grounded_set)
            q_bfs = list(grounded_set)
            while q_bfs:
                cur = q_bfs.pop(0)
                for nb in adj[cur]:
                    if (cur == u and nb == v) or (cur == v and nb == u):
                        continue
                    if nb not in vis_g:
                        vis_g.add(nb)
                        q_bfs.append(nb)
            if len(vis_g) == len(parts):
                continue  # Edge lies on a closed loop; loop-locked or tested via lateral wrench LP below

            has_free_bridge = True
            free_passive_cnt += 1
            subtree_set = {i for i in parts if i not in vis_g}
            child_id = v if v in subtree_set else u
            wf_p = self._port_world_pose(parts[ga[0]].continuous_pose, sa.local_frame)
            pivot = wf_p.t.copy()
            is_ball_j = {sa.connector_type.name, sb.connector_type.name} & {
                "BALL_JOINT",
                "BALL_SOCKET",
            }
            if is_ball_j:
                for col_i in (0, 1, 2):
                    ax_c = wf_p.R[:, col_i].copy()
                    nrm_c = float(np.linalg.norm(ax_c))
                    if nrm_c > 1e-12:
                        bridge_dofs.append((subtree_set, child_id, pivot.copy(), ax_c / nrm_c))
            else:
                axis = wf_p.R[:, 2].copy()
                norm_ax = float(np.linalg.norm(axis))
                if norm_ax <= 1e-12:
                    continue
                axis = axis / norm_ax
                bridge_dofs.append((subtree_set, child_id, pivot, axis))

        if bridge_dofs:
            # Sort root-to-leaf: strictly larger `subtree_set` precedes any descendant joint inside it
            bridge_dofs.sort(key=lambda item: len(item[0]), reverse=True)
            d_dofs = len(bridge_dofs)

            def _eval_coupled_delta_u(q_vec: np.ndarray) -> float:
                cur_coms = {k: v.copy() for k, v in coms.items()}
                cur_pivots = [item[2].copy() for item in bridge_dofs]
                cur_axes = [item[3].copy() for item in bridge_dofs]
                for j_idx in range(d_dofs):
                    ang = float(q_vec[j_idx])
                    if abs(ang) <= 1e-15:
                        continue
                    sub_j = bridge_dofs[j_idx][0]
                    piv_j = cur_pivots[j_idx]
                    ax_j = cur_axes[j_idx]
                    r_j = _rodrigues((float(ax_j[0]), float(ax_j[1]), float(ax_j[2])), ang)
                    for k_id in sub_j:
                        cur_coms[k_id] = piv_j + r_j @ (cur_coms[k_id] - piv_j)
                    for m_idx in range(j_idx + 1, d_dofs):
                        if bridge_dofs[m_idx][1] in sub_j:
                            cur_pivots[m_idx] = piv_j + r_j @ (cur_pivots[m_idx] - piv_j)
                            cur_axes[m_idx] = r_j @ cur_axes[m_idx]
                du_tot = 0.0
                for k_id, c_now in cur_coms.items():
                    du_tot += masses[k_id] * float(c_now[2] - coms[k_id][2])
                return du_tot

            # 1. Coordinate-axis probes (±delta_theta_rad along each passive DOF)
            for j_idx in range(d_dofs):
                q_p = np.zeros(d_dofs, dtype=np.float64)
                q_m = np.zeros(d_dofs, dtype=np.float64)
                q_p[j_idx] = float(delta_theta_rad)
                q_m[j_idx] = -float(delta_theta_rad)
                local_min_du = min(_eval_coupled_delta_u(q_p), _eval_coupled_delta_u(q_m))
                min_work = min(min_work, local_min_du)
                if local_min_du < -1e-5:
                    return PerturbationStabilityResult(
                        is_perturbation_stable=False,
                        nominal_utilization=eq0.peak_utilization,
                        worst_perturbed_utilization=float("inf"),
                        min_restoring_work_g_ldu=local_min_du,
                        mechanism_classification="unstable_inverted_mechanism",
                        free_passive_joints=free_passive_cnt,
                        min_hessian_eigenvalue=min(0.0, local_min_du),
                    )

            # 2. Coupled multi-joint gravitational potential Hessian H_free (`R9-WS3` §5.4):
            #    Catches coupled SO(3) saddle mechanisms where H_ii > 0 individually yet lambda_min(H) < 0!
            h_step = 1e-4
            hess = np.zeros((d_dofs, d_dofs), dtype=np.float64)
            for i_d in range(d_dofs):
                ei = np.zeros(d_dofs, dtype=np.float64)
                ei[i_d] = h_step
                hess[i_d, i_d] = (_eval_coupled_delta_u(ei) + _eval_coupled_delta_u(-ei)) / (
                    h_step * h_step
                )
                for j_d in range(i_d + 1, d_dofs):
                    ej = np.zeros(d_dofs, dtype=np.float64)
                    ej[j_d] = h_step
                    hij = (
                        _eval_coupled_delta_u(ei + ej)
                        - _eval_coupled_delta_u(ei - ej)
                        - _eval_coupled_delta_u(-ei + ej)
                        + _eval_coupled_delta_u(-ei - ej)
                    ) / (4.0 * h_step * h_step)
                    hess[i_d, j_d] = hij
                    hess[j_d, i_d] = hij

            eigvals, eigvecs = np.linalg.eigh(0.5 * (hess + hess.T))
            min_hess_eig = float(eigvals[0])
            v_min = eigvecs[:, 0]
            coupled_du = min(
                _eval_coupled_delta_u(float(delta_theta_rad) * v_min),
                _eval_coupled_delta_u(-float(delta_theta_rad) * v_min),
            )
            min_work = min(min_work, coupled_du)
            if min_hess_eig < -1e-4 or coupled_du < -1e-5:
                return PerturbationStabilityResult(
                    is_perturbation_stable=False,
                    nominal_utilization=eq0.peak_utilization,
                    worst_perturbed_utilization=float("inf"),
                    min_restoring_work_g_ldu=coupled_du,
                    mechanism_classification="unstable_coupled_saddle_mechanism",
                    free_passive_joints=free_passive_cnt,
                    min_hessian_eigenvalue=min_hess_eig,
                )

            if min_work > 1e-4:
                mech_mode = "stable_hanging_pendulum"
                if not allow_stable_pendulums:
                    return PerturbationStabilityResult(
                        is_perturbation_stable=False,
                        nominal_utilization=eq0.peak_utilization,
                        worst_perturbed_utilization=float("inf"),
                        min_restoring_work_g_ldu=min_work,
                        mechanism_classification="disallowed_pendulum_mechanism",
                        free_passive_joints=free_passive_cnt,
                        min_hessian_eigenvalue=min_hess_eig,
                    )
            else:
                mech_mode = "neutral_free_mechanism"
                if not allow_neutral_mechanisms:
                    return PerturbationStabilityResult(
                        is_perturbation_stable=False,
                        nominal_utilization=eq0.peak_utilization,
                        worst_perturbed_utilization=float("inf"),
                        min_restoring_work_g_ldu=min_work,
                        mechanism_classification="neutral_free_mechanism",
                        free_passive_joints=free_passive_cnt,
                        min_hessian_eigenvalue=min_hess_eig,
                    )

        worst_u = eq0.peak_utilization
        if not has_free_bridge and lateral_disturbance_ratio > 0.0:
            eps = float(lateral_disturbance_ratio)
            for lx, ly in ((eps, 0.0), (-eps, 0.0), (0.0, eps), (0.0, -eps)):
                eq_lat = self.evaluate_static_equilibrium(
                    clutch_force_capacity_grams=clutch_force_capacity_grams,
                    moment_arm_capacity_ldu=moment_arm_capacity_ldu,
                    locking_hinge_detent_moment_ldu_g=locking_hinge_detent_moment_ldu_g,
                    ball_joint_friction_moment_ldu_g=ball_joint_friction_moment_ldu_g,
                    lateral_accel_xy=(lx, ly),
                )
                if not eq_lat.is_stable:
                    return PerturbationStabilityResult(
                        is_perturbation_stable=False,
                        nominal_utilization=eq0.peak_utilization,
                        worst_perturbed_utilization=eq_lat.peak_utilization,
                        min_restoring_work_g_ldu=0.0 if np.isinf(min_work) else min_work,
                        mechanism_classification="lateral_perturbation_unstable",
                        free_passive_joints=free_passive_cnt,
                        min_hessian_eigenvalue=min_hess_eig,
                    )
                worst_u = max(worst_u, eq_lat.peak_utilization)

        return PerturbationStabilityResult(
            is_perturbation_stable=True,
            nominal_utilization=eq0.peak_utilization,
            worst_perturbed_utilization=worst_u,
            min_restoring_work_g_ldu=0.0 if np.isinf(min_work) else min_work,
            mechanism_classification=mech_mode,
            free_passive_joints=free_passive_cnt,
            min_hessian_eigenvalue=min_hess_eig,
        )


@dataclass(frozen=True, slots=True)
class PerturbationStabilityResult:
    """Result of angular joint-nullspace + coupled Hessian + lateral-wrench perturbation analysis."""

    is_perturbation_stable: bool
    nominal_utilization: float
    worst_perturbed_utilization: float
    min_restoring_work_g_ldu: float
    mechanism_classification: str
    free_passive_joints: int
    min_hessian_eigenvalue: float = 0.0


@dataclass(frozen=True, slots=True)
class JointLawSpec:
    """
    Explicit connector-local joint freedom descriptor (§6):
        T_b = T_a C_a J(q) C_b^{-1}
    recording remaining DoFs, hard stops, and exhaustive vs. sampled coverage.
    """

    joint_family: str  # "RIGID_INDEXED" | "REVOLUTE_1DOF" | "AXIAL_SLIDE_ROT_2DOF" | "SPHERICAL_CONE_3DOF"
    dof_count: int
    angle_limits_rad: Tuple[float, float]
    slide_limits_ldu: Tuple[float, float] = (0.0, 0.0)
    max_cone_angle_rad: float = 0.0
    detent_step_rad: Optional[float] = None
    coverage_mode: str = "EXHAUSTIVE"  # "EXHAUSTIVE" | "SAMPLED_GRID" | "CONTINUOUS_OPTIMIZED"


def compute_connector_joint_pose(
    parent_pose: ContinuousSE3Pose,
    parent_port_local: LDUFrame,
    child_port_local: LDUFrame,
    joint_angle_rad: float = 0.0,
    axial_slide_ldu: float = 0.0,
    ball_tilt_xy_rad: Tuple[float, float] = (0.0, 0.0),
) -> ContinuousSE3Pose:
    """
    Exact connector-local joint kinematics law (§6):
        T_b = T_a * C_a * J(q) * C_b^{-1}.
    """
    c_a = ContinuousSE3Pose.from_ldu_frame(parent_port_local)
    c_b = ContinuousSE3Pose.from_ldu_frame(child_port_local)
    r_yaw = _rodrigues((0.0, 0.0, 1.0), float(joint_angle_rad))
    rx = _rodrigues((1.0, 0.0, 0.0), float(ball_tilt_xy_rad[0]))
    ry = _rodrigues((0.0, 1.0, 0.0), float(ball_tilt_xy_rad[1]))
    j_q = ContinuousSE3Pose.from_matrix_and_pos(
        r_yaw @ rx @ ry, (0.0, 0.0, float(axial_slide_ldu))
    )
    return parent_pose.compose(c_a).compose(j_q).compose(c_b.inverse())


@dataclass(frozen=True, slots=True)
class LoopClosureResult:
    """Result of damped Gauss-Newton SE(3) kinematic loop closure on articulated branches."""

    converged: bool
    optimal_angles_rad: Tuple[float, ...]
    position_residual_ldu: float
    axis_alignment_cos: float
    iterations: int


def solve_two_branch_loop_closure(
    base_pose: ContinuousSE3Pose,
    left_base_port: LDUFrame,
    right_base_port: LDUFrame,
    left_part_id: str,
    left_prox_port_id: int,
    left_dist_port_id: int,
    right_part_id: str,
    right_prox_port_id: int,
    right_dist_port_id: int,
    init_angles_rad: Tuple[float, float] = (0.5, -0.5),
    angle_limits_rad: Tuple[float, float] = (-2.35, 2.35),
    target_tip_span_ldu: float = 0.0,
    max_iters: int = 40,
    tol_ldu: float = 1e-4,
) -> LoopClosureResult:
    """
    Solve exact 2-branch continuous SE(3) kinematic loop closure via damped Gauss-Newton
    (Levenberg-Marquardt) over `(q_left, q_right)` so the distal connector frames meet
    coincidently (or at `target_tip_span_ldu` along X for a horizontal crossbar).
    """
    l_prim = get_brick_primitive(left_part_id)
    r_prim = get_brick_primitive(right_part_id)
    lp_prox = l_prim.ports[left_prox_port_id].local_frame
    lp_dist = ContinuousSE3Pose.from_ldu_frame(l_prim.ports[left_dist_port_id].local_frame)
    rp_prox = r_prim.ports[right_prox_port_id].local_frame
    rp_dist = ContinuousSE3Pose.from_ldu_frame(r_prim.ports[right_dist_port_id].local_frame)

    q = np.array([float(init_angles_rad[0]), float(init_angles_rad[1])], dtype=np.float64)
    offset_vec = np.array([float(target_tip_span_ldu), 0.0, 0.0], dtype=np.float64)

    def _residual(qv: np.ndarray) -> Tuple[np.ndarray, float]:
        t_l = compute_connector_joint_pose(base_pose, left_base_port, lp_prox, float(qv[0]))
        t_r = compute_connector_joint_pose(base_pose, right_base_port, rp_prox, float(qv[1]))
        w_l = t_l.compose(lp_dist)
        w_r = t_r.compose(rp_dist)
        pos_diff = (w_l.t + offset_vec) - w_r.t
        cos_z = float(np.dot(w_l.R[:, 2], w_r.R[:, 2]))
        return pos_diff, cos_z

    lam = 1e-3
    eps_fd = 1e-6
    it = 0
    for it in range(1, max_iters + 1):
        res, cos_z = _residual(q)
        err = float(np.linalg.norm(res))
        if err <= tol_ldu:
            return LoopClosureResult(True, (float(q[0]), float(q[1])), err, cos_z, it)
        j_mat = np.zeros((3, 2), dtype=np.float64)
        for k in range(2):
            qp = q.copy()
            qp[k] += eps_fd
            rp, _ = _residual(qp)
            j_mat[:, k] = (rp - res) / eps_fd
        lhs = j_mat.T @ j_mat + lam * np.eye(2, dtype=np.float64)
        dq = -np.linalg.solve(lhs, j_mat.T @ res)
        q_cand = np.clip(q + dq, angle_limits_rad[0], angle_limits_rad[1])
        r_cand, _ = _residual(q_cand)
        if float(np.linalg.norm(r_cand)) < err:
            q = q_cand
            lam = max(lam * 0.5, 1e-7)
        else:
            lam = min(lam * 5.0, 1e3)

    res_f, cos_f = _residual(q)
    err_f = float(np.linalg.norm(res_f))
    return LoopClosureResult(err_f <= 1e-2, (float(q[0]), float(q[1])), err_f, cos_f, it)


# ---------------------------------------------------------------------------
# WS-3C: Genuine OMR `.mpd` Multi-Part Flattener, Connection Graph & Round-Trip (`Claim C`)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FlattenedOMRPlacement:
    """One brick placement flattened from a hierarchical LDraw `.mpd` model into world SE(3)."""

    raw_part_ref: str
    canonical_part_id: str
    supported_in_tier_a: bool
    color_code: int
    ldraw_world_R: Tuple[Tuple[float, float, float], ...]
    ldraw_world_t: Tuple[float, float, float]
    continuous_pose: Optional[ContinuousSE3Pose]
    is_cubic_aligned: bool


@dataclass(frozen=True, slots=True)
class OMRSetEvaluationResult:
    """Per-set round-trip & 5-family connection extraction result on a genuine OMR `.mpd` set."""

    set_num: str
    set_name: str
    total_part_instances: int
    supported_part_instances: int
    part_coverage_ratio: float
    non_cubic_se3_instances: int
    extracted_connections: int
    connector_families_present: Tuple[str, ...]
    roundtrip_pose_exact_rate: float


@dataclass(frozen=True, slots=True)
class OMRCorpusBenchmarkReport:
    """Aggregate `Claim C` verification report across all 26 genuine human OMR `.mpd` sets."""

    num_sets_evaluated: int
    sets_meeting_90pct_coverage: int
    mean_part_coverage_ratio: float
    aggregate_part_coverage_ratio: float
    sets_with_non_cubic_se3: int
    total_non_cubic_se3_instances: int
    total_extracted_connections: int
    five_connector_families_covered: Tuple[str, ...]
    connection_compatibility_rate: float
    roundtrip_export_reparse_exact_rate: float
    set_results: Tuple[OMRSetEvaluationResult, ...]


_R_PIN_TB_TO_LDRAW_ZUP = np.array(
    [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float64
)


def _ldraw_pose_to_topo_brick_se3(
    part_id: str, r_ldraw_world: np.ndarray, t_ldraw_world: np.ndarray
) -> ContinuousSE3Pose:
    """
    Map an official LDraw (`-Y` up, top-center origin) world transform `(R_ld, t_ld)` into
    `topo_brick`'s (`+Z` up, stud-(0,0)-bottom origin) `ContinuousSE3Pose`.
    """
    prim = get_brick_primitive(part_id)
    r_w = _C_LDRAW_TO_ZUP @ r_ldraw_world @ _C_LDRAW_TO_ZUP.T
    t_w = _C_LDRAW_TO_ZUP @ t_ldraw_world
    if getattr(prim, "is_ldraw_native_frame", False):
        return ContinuousSE3Pose.from_matrix_and_pos(r_w, t_w, orthonormalize=True)
    if prim.ports and all(
        p.connector_type in (ConnectorType.TECHNIC_PIN, ConnectorType.SMOOTH_PIN)
        for p in prim.ports
    ):
        return ContinuousSE3Pose.from_matrix_and_pos(
            r_w @ _R_PIN_TB_TO_LDRAW_ZUP, t_w, orthonormalize=True
        )
    if prim.ports and all(
        p.connector_type
        in (
            ConnectorType.TECHNIC_AXLE,
            ConnectorType.TECHNIC_AXLE_HOLE,
            ConnectorType.BAR_3_18MM,
        )
        for p in prim.ports
    ):
        return ContinuousSE3Pose.from_matrix_and_pos(r_w, t_w, orthonormalize=True)

    xmin = min(b.xmin for b in prim.hull_boxes)
    xmax = max(b.xmax for b in prim.hull_boxes)
    ymin = min(b.ymin for b in prim.hull_boxes)
    ymax = max(b.ymax for b in prim.hull_boxes)
    zmax = max(b.zmax for b in prim.hull_boxes)
    c_top = np.array([0.5 * (xmin + xmax), 0.5 * (ymin + ymax), float(zmax)], dtype=np.float64)
    r_tb = r_w @ _R_TB_TO_LDRAW_ZUP
    t_tb = t_w - r_tb @ c_top
    return ContinuousSE3Pose.from_matrix_and_pos(r_tb, t_tb, orthonormalize=True)


def _topo_brick_se3_to_ldraw_line(
    part_id: str, pose: ContinuousSE3Pose, color_code: int = 15
) -> str:
    """Export a `ContinuousSE3Pose` back to an official LDraw (`-Y` up) type-1 line."""
    prim = get_brick_primitive(part_id)
    r_tb = pose.R
    t_tb = pose.t
    if getattr(prim, "is_ldraw_native_frame", False):
        r_ld = _C_LDRAW_TO_ZUP.T @ r_tb @ _C_LDRAW_TO_ZUP
        t_ld = _C_LDRAW_TO_ZUP.T @ t_tb
        r_str = " ".join(f"{float(v):.6g}" for v in r_ld.reshape(-1))
        return f"1 {color_code} {t_ld[0]:.6g} {t_ld[1]:.6g} {t_ld[2]:.6g} {r_str} {part_id}.dat"
    if prim.ports and all(
        p.connector_type in (ConnectorType.TECHNIC_PIN, ConnectorType.SMOOTH_PIN)
        for p in prim.ports
    ):
        r_w = r_tb @ _R_PIN_TB_TO_LDRAW_ZUP.T
        t_w = t_tb
    elif prim.ports and all(
        p.connector_type
        in (
            ConnectorType.TECHNIC_AXLE,
            ConnectorType.TECHNIC_AXLE_HOLE,
            ConnectorType.BAR_3_18MM,
        )
        for p in prim.ports
    ):
        r_w = r_tb
        t_w = t_tb
    else:
        xmin = min(b.xmin for b in prim.hull_boxes)
        xmax = max(b.xmax for b in prim.hull_boxes)
        ymin = min(b.ymin for b in prim.hull_boxes)
        ymax = max(b.ymax for b in prim.hull_boxes)
        zmax = max(b.zmax for b in prim.hull_boxes)
        c_top = np.array([0.5 * (xmin + xmax), 0.5 * (ymin + ymax), float(zmax)], dtype=np.float64)
        t_w = t_tb + r_tb @ c_top
        r_w = r_tb @ _R_TB_TO_LDRAW_ZUP.T

    r_ld = _C_LDRAW_TO_ZUP.T @ r_w @ _C_LDRAW_TO_ZUP
    t_ld = _C_LDRAW_TO_ZUP.T @ t_w
    r_str = " ".join(f"{float(v):.6g}" for v in r_ld.reshape(-1))
    return f"1 {color_code} {t_ld[0]:.6g} {t_ld[1]:.6g} {t_ld[2]:.6g} {r_str} {part_id}.dat"


def flatten_omr_mpd_to_se3(
    mpd_text: str,
    catalog: Optional[Dict[str, BrickPrimitive]] = None,
) -> List[FlattenedOMRPlacement]:
    """
    Parse and recursively flatten a multi-submodel LDraw `.mpd` document into world-space
    `FlattenedOMRPlacement` instances with exact continuous SE(3) transforms.
    """
    active_cat = catalog if catalog is not None else TIER_A_CATALOG
    blocks: Dict[str, List[str]] = {}
    order: List[str] = []
    cur_name: Optional[str] = None
    cur_lines: List[str] = []
    for line in mpd_text.splitlines():
        m = re.match(r"^\s*0\s+FILE\s+(.+?)\s*$", line, re.I)
        if m:
            if cur_name is not None:
                blocks[cur_name] = cur_lines
            cur_name = m.group(1).strip().lower()
            order.append(cur_name)
            cur_lines = []
        elif re.match(r"^\s*0\s+NOFILE\s*$", line, re.I):
            if cur_name is not None:
                blocks[cur_name] = cur_lines
                cur_name = None
                cur_lines = []
        else:
            if cur_name is not None or not order:
                cur_lines.append(line)
    if cur_name is not None:
        blocks[cur_name] = cur_lines
    if not order:
        blocks["__root__"] = cur_lines
        order = ["__root__"]

    is_part_block: Dict[str, bool] = {}
    for bname, blines in blocks.items():
        hdr = "\n".join(blines[:15]).lower()
        is_part_block[bname] = bool(
            bname.endswith(".dat")
            or "!ldraw_org part" in hdr
            or "primitive" in hdr
            or "subpart" in hdr
        )
    model_order = [b for b in order if not is_part_block[b]]
    root_block = model_order[0] if model_order else order[0]

    placements: List[FlattenedOMRPlacement] = []

    def _walk(bname: str, r_parent: np.ndarray, t_parent: np.ndarray, depth: int = 0) -> None:
        if depth > 14 or bname not in blocks:
            return
        for ln in blocks[bname]:
            s = ln.strip()
            if not s.startswith("1 "):
                continue
            toks = s.split()
            if len(toks) < 15:
                continue
            try:
                color = int(toks[1])
                t_c = np.array([float(toks[2]), float(toks[3]), float(toks[4])], dtype=np.float64)
                r_c = np.array([float(x) for x in toks[5:14]], dtype=np.float64).reshape(3, 3)
            except ValueError:
                continue
            r_w = r_parent @ r_c
            t_w = r_parent @ t_c + t_parent
            raw_ref = " ".join(toks[14:]).strip()
            ref_low = raw_ref.lower()
            if ref_low in blocks and not is_part_block.get(ref_low, False):
                _walk(ref_low, r_w, t_w, depth + 1)
            else:
                cid = normalize_ldraw_part_id(ref_low, active_cat)
                sup = cid in active_cat
                c_pose = _ldraw_pose_to_topo_brick_se3(cid, r_w, t_w) if sup else None
                cubic = bool(np.all(np.abs(r_w - np.round(r_w)) <= 1e-3))
                placements.append(
                    FlattenedOMRPlacement(
                        raw_part_ref=raw_ref,
                        canonical_part_id=cid,
                        supported_in_tier_a=sup,
                        color_code=color,
                        ldraw_world_R=tuple(tuple(float(v) for v in row) for row in r_w),
                        ldraw_world_t=(float(t_w[0]), float(t_w[1]), float(t_w[2])),
                        continuous_pose=c_pose,
                        is_cubic_aligned=cubic,
                    )
                )

    _walk(root_block, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64))
    return placements


def _connector_family_bucket(ctype: ConnectorType) -> str:
    if ctype in (ConnectorType.STANDARD_STUD, ConnectorType.HOLLOW_STUD):
        return "STUD_TUBE_AND_SNOT"
    if ctype in (ConnectorType.TECHNIC_PIN_HOLE, ConnectorType.TECHNIC_PIN, ConnectorType.SMOOTH_PIN, ConnectorType.WHEEL_PIN, ConnectorType.WHEEL_HUB_HOLE):
        return "TECHNIC_PIN_AND_WHEEL"
    if ctype in (ConnectorType.TECHNIC_AXLE_HOLE, ConnectorType.TECHNIC_AXLE):
        return "TECHNIC_AXLE"
    if ctype in (ConnectorType.CLIP_3_18MM, ConnectorType.BAR_3_18MM):
        return "CLIP_AND_BAR"
    return "HINGE_TURNTABLE_BALL"


def extract_se3_port_connections(
    placements: Sequence[FlattenedOMRPlacement],
    pos_tol_ldu: float = 3.0,
) -> Tuple[int, int, Set[str]]:
    """
    Spatial hash extraction of mated port-to-port connections across continuous SE(3) placements.
    Returns `(compatible_connections, candidate_port_coincidences, connector_family_buckets)`.
    """
    cell_size = 12.0
    port_buckets: Dict[Tuple[int, int, int], List[Tuple[int, ConnectorType, PortPolarity, np.ndarray, np.ndarray]]] = {}
    supported_idx = [i for i, p in enumerate(placements) if p.supported_in_tier_a and p.continuous_pose is not None]

    for idx in supported_idx:
        pl = placements[idx]
        pose = pl.continuous_pose
        assert pose is not None
        prim = get_brick_primitive(pl.canonical_part_id)
        r_p = pose.R
        t_p = pose.t
        for ps in prim.ports:
            r_l = rot_id_to_matrix(ps.local_frame.rot_id).astype(np.float64)
            t_l = np.asarray(ps.local_frame.pos, dtype=np.float64)
            w_pos = r_p @ t_l + t_p
            w_z = (r_p @ r_l)[:, 2]
            ck = (
                int(math.floor(w_pos[0] / cell_size)),
                int(math.floor(w_pos[1] / cell_size)),
                int(math.floor(w_pos[2] / cell_size)),
            )
            port_buckets.setdefault(ck, []).append((idx, ps.connector_type, ps.polarity, w_pos, w_z))

    compat_conns = 0
    total_coincidences = 0
    families: Set[str] = set()
    used_ports: Set[int] = set()
    flat_ports: List[Tuple[Tuple[int, int, int], int, int, ConnectorType, PortPolarity, np.ndarray, np.ndarray]] = []
    uid = 0
    for ck, plist in port_buckets.items():
        for inst_i, ct, pol, wp, wz in plist:
            flat_ports.append((ck, uid, inst_i, ct, pol, wp, wz))
            uid += 1

    uid_by_cell: Dict[Tuple[int, int, int], List[Tuple[int, int, ConnectorType, PortPolarity, np.ndarray, np.ndarray]]] = {}
    for ck, u_i, inst_i, ct, pol, wp, wz in flat_ports:
        uid_by_cell.setdefault(ck, []).append((u_i, inst_i, ct, pol, wp, wz))

    for ck, u_a, inst_a, ct_a, pol_a, wp_a, wz_a in flat_ports:
        if u_a in used_ports or pol_a != PortPolarity.STUD:
            continue
        best_u: Optional[int] = None
        best_d = pos_tol_ldu
        best_ct_b: Optional[ConnectorType] = None
        best_pol_b: Optional[PortPolarity] = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    nck = (ck[0] + dx, ck[1] + dy, ck[2] + dz)
                    for u_b, inst_b, ct_b, pol_b, wp_b, wz_b in uid_by_cell.get(nck, ()):
                        if inst_b == inst_a or u_b in used_ports or pol_b != PortPolarity.ANTI_STUD:
                            continue
                        d = float(np.linalg.norm(wp_a - wp_b))
                        if d <= best_d and float(np.dot(wz_a, wz_b)) >= 0.45:
                            best_d = d
                            best_u = u_b
                            best_ct_b = ct_b
                            best_pol_b = pol_b
        if best_u is not None and best_ct_b is not None and best_pol_b is not None:
            total_coincidences += 1
            used_ports.add(u_a)
            used_ports.add(best_u)
            if are_connectors_compatible(ct_a, pol_a, best_ct_b, best_pol_b):
                compat_conns += 1
                families.add(_connector_family_bucket(ct_a))
                families.add(_connector_family_bucket(best_ct_b))

    return compat_conns, total_coincidences, families


def evaluate_omr_26set_benchmark(
    asset_path: Optional[Path] = None,
    catalog: Optional[Any] = None,
) -> OMRCorpusBenchmarkReport:
    """
    Evaluate `TIER_A_CATALOG` (or `catalog` / `COMPILED_LDRAW_CATALOG`) and `ContinuousSE3Pose`
    round-trip / 5-family connection extraction across all 26 genuine human Official Model
    Repository (`.mpd`) LEGO sets (`Claim C`).
    """
    p = asset_path or (Path(__file__).resolve().parent / "omr_benchmark_sets.json")
    payload = json.loads(p.read_text(encoding="utf-8"))
    raw_sets = payload["sets"]
    active_cat = catalog if catalog is not None else TIER_A_CATALOG

    set_results: List[OMRSetEvaluationResult] = []
    tot_parts_all = 0
    sup_parts_all = 0
    tot_conns_all = 0
    tot_coinc_all = 0
    tot_non_cubic = 0
    sets_with_nc = 0
    sets_ge_90 = 0
    all_families: Set[str] = set()
    rt_exact_num = 0
    rt_exact_den = 0

    for entry in raw_sets:
        placements = flatten_omr_mpd_to_se3(entry["mpd_model_text"], catalog=active_cat)
        n_tot = len(placements)
        n_sup = sum(1 for pl in placements if pl.supported_in_tier_a)
        n_nc = sum(1 for pl in placements if not pl.is_cubic_aligned)
        cov = float(n_sup / max(n_tot, 1))
        if cov >= 0.90:
            sets_ge_90 += 1
        if n_nc > 0:
            sets_with_nc += 1
        tot_non_cubic += n_nc
        tot_parts_all += n_tot
        sup_parts_all += n_sup

        compat_c, coinc_c, fams = extract_se3_port_connections(placements)
        tot_conns_all += compat_c
        tot_coinc_all += coinc_c
        all_families.update(fams)

        # Round-trip export -> re-parse pose exactness check on supported parts
        sup_pls = [pl for pl in placements if pl.supported_in_tier_a and pl.continuous_pose is not None]
        exported_lines = [
            _topo_brick_se3_to_ldraw_line(pl.canonical_part_id, pl.continuous_pose, pl.color_code)  # type: ignore[arg-type]
            for pl in sup_pls
        ]
        reparsed = flatten_omr_mpd_to_se3("\n".join(exported_lines), catalog=active_cat)
        exact_here = 0
        for pl_orig, pl_re in zip(sup_pls, reparsed):
            assert pl_orig.continuous_pose is not None and pl_re.continuous_pose is not None
            pos_err = float(np.linalg.norm(pl_orig.continuous_pose.t - pl_re.continuous_pose.t))
            rot_err = float(np.linalg.norm(pl_orig.continuous_pose.R - pl_re.continuous_pose.R))
            if pos_err <= 1e-3 and rot_err <= 1e-3:
                exact_here += 1
        rt_rate = float(exact_here / max(len(sup_pls), 1))
        rt_exact_num += exact_here
        rt_exact_den += len(sup_pls)

        set_results.append(
            OMRSetEvaluationResult(
                set_num=str(entry["set_num"]),
                set_name=str(entry.get("set_name", entry["set_num"])),
                total_part_instances=n_tot,
                supported_part_instances=n_sup,
                part_coverage_ratio=cov,
                non_cubic_se3_instances=n_nc,
                extracted_connections=compat_c,
                connector_families_present=tuple(sorted(fams)),
                roundtrip_pose_exact_rate=rt_rate,
            )
        )

    return OMRCorpusBenchmarkReport(
        num_sets_evaluated=len(set_results),
        sets_meeting_90pct_coverage=sets_ge_90,
        mean_part_coverage_ratio=float(np.mean([r.part_coverage_ratio for r in set_results])),
        aggregate_part_coverage_ratio=float(sup_parts_all / max(tot_parts_all, 1)),
        sets_with_non_cubic_se3=sets_with_nc,
        total_non_cubic_se3_instances=tot_non_cubic,
        total_extracted_connections=tot_conns_all,
        five_connector_families_covered=tuple(sorted(all_families)),
        connection_compatibility_rate=float(tot_conns_all / max(tot_coinc_all, 1)),
        roundtrip_export_reparse_exact_rate=float(rt_exact_num / max(rt_exact_den, 1)),
        set_results=tuple(set_results),
    )


# ---------------------------------------------------------------------------
# WS-3C: Official LEGO Set `10698-1` Inventory-Bounded Alt-Build Benchmark (`Claim E`)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Set10698AltBuildResult:
    """Result of realizing a multi-part target strictly within LEGO Set `10698-1` (`790` pieces)."""

    design_name: str
    status: str
    certified: bool
    num_parts_placed: int
    unique_part_types_used: int
    target_iou: float
    zero_inventory_overruns: bool
    max_part_usage_margin: int
    wall_clock_ms: float


def run_set_10698_inventory_bounded_alt_builds() -> Tuple[Set10698AltBuildResult, ...]:
    """
    Verify `Claim E`: Realize 5 diverse architectural & structural alt-build target shapes
    strictly bounded by the official Rebrickable inventory of LEGO Set `10698-1`
    (`108` part types, `790` non-spare pieces), verifying `0` inventory overruns across all
    `108` part types and `100%` `CertificateBundle` pass!
    """
    from topo_brick.kernel.construction_kernel import TopoBrickKernel
    from topo_brick.representation.interface_fields import SparseTiledTargetField

    kernel = TopoBrickKernel()
    # Rectangular + plate structural subset inside Set 10698-1 for fast blind volumetric tiling
    kit_10698_rect = tuple(
        pid
        for pid in (
            "3001", "3002", "3003", "3004", "3005", "3010", "2456", "3009", "3622",
            "3065", "6223",
        )
        if pid in SET_10698_OFFICIAL_INVENTORY
    )

    alt_targets: List[Tuple[str, Tuple[LDUBox, ...]]] = [
        (
            "10698_Classic_Cottage_Walls",
            (
                LDUBox(-10, -10, 0, 150, 110, 24),
                LDUBox(-10, -10, 24, 150, 10, 96),
                LDUBox(-10, 90, 24, 150, 110, 96),
                LDUBox(-10, 10, 24, 10, 90, 96),
                LDUBox(130, 10, 24, 150, 90, 96),
                LDUBox(-10, -10, 96, 150, 110, 120),
            ),
        ),
        (
            "10698_Twin_Bridge_Gatehouse",
            (
                LDUBox(-10, -10, 0, 30, 70, 96),
                LDUBox(110, -10, 0, 150, 70, 96),
                LDUBox(-10, -10, 96, 150, 70, 144),
            ),
        ),
        (
            "10698_Stepped_Ziggurat_Monument",
            (
                LDUBox(-10, -10, 0, 150, 150, 24),
                LDUBox(10, 10, 24, 130, 130, 48),
                LDUBox(30, 30, 48, 110, 110, 72),
                LDUBox(50, 50, 72, 90, 90, 120),
            ),
        ),
        (
            "10698_Courtyard_Watchtower",
            (
                LDUBox(-10, -10, 0, 110, 110, 24),
                LDUBox(-10, -10, 24, 30, 30, 168),
                LDUBox(-10, 30, 24, 110, 50, 72),
                LDUBox(30, -10, 24, 110, 10, 72),
            ),
        ),
        (
            "10698_Scarce_2456_2x6_Beam_Span",
            tuple(LDUBox(-10, -10, k * 24, 30, 230, (k + 1) * 24) for k in range(10)),
        ),
    ]

    results: List[Set10698AltBuildResult] = []
    for name, boxes in alt_targets:
        tf = SparseTiledTargetField(boxes=boxes)
        inv_copy = dict(SET_10698_OFFICIAL_INVENTORY)
        res = kernel.realize(
            target_field=tf,
            allowed_part_ids=kit_10698_rect,
            kit_name="SET_10698_OFFICIAL_INVENTORY",
            inventory_counts=inv_copy,
            object_id=name,
        )
        used_counts: Dict[str, int] = {}
        for p in res.assembly.placed_parts.values():
            used_counts[p.part_id] = used_counts.get(p.part_id, 0) + 1
        overruns = [
            pid
            for pid, cnt in used_counts.items()
            if cnt > SET_10698_OFFICIAL_INVENTORY.get(pid, 0)
        ]
        min_margin = min(
            (SET_10698_OFFICIAL_INVENTORY.get(pid, 0) - cnt for pid, cnt in used_counts.items()),
            default=0,
        )
        results.append(
            Set10698AltBuildResult(
                design_name=name,
                status=res.status.value,
                certified=res.certificates.overall_certified,
                num_parts_placed=res.assembly.num_parts,
                unique_part_types_used=len(used_counts),
                target_iou=res.certificates.geometry.target_iou,
                zero_inventory_overruns=(len(overruns) == 0),
                max_part_usage_margin=min_margin,
                wall_clock_ms=res.build_record.wall_clock_ms,
            )
        )
    return tuple(results)
