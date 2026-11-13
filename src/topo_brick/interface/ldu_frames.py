"""
Exact integer LDraw Unit (LDU) lattice frames in Z^3_LDU ⋊ SO(3)_cubic.

Implements:
- Canonical LDraw integer units (1 stud = 20 LDU, 1 plate = 8 LDU, 1 brick = 24 LDU,
  preserving the exact 5:5:6 SNOT identity: 5 plates = 40 LDU = 2 studs).
- The 24-element proper cubic rotation group SO(3)_cubic (det(R) = +1, excluding reflections).
- Rigid lattice frame transforms T = (rot_id, pos) with exact integer composition, inversion,
  and invariant relative transforms ΔT_uv = T_u^-1 ∘ T_v.
- Local gravity vector projection g_u = R_u^T g_world.
- Exact integer Axis-Aligned Bounding Boxes (LDUBox) closed under all 24 cubic rotations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np

# Canonical LDraw Unit (LDU) integer constants
STUD_PITCH_LDU: int = 20
HALF_STUD_LDU: int = 10
PLATE_HEIGHT_LDU: int = 8
HALF_PLATE_LDU: int = 4
BRICK_HEIGHT_LDU: int = 24

# World gravity vector in right-handed +Z-up LDU coordinates
WORLD_GRAVITY: Tuple[int, int, int] = (0, 0, -1)


def _build_so3_cubic_group() -> Tuple[
    np.ndarray,
    dict[Tuple[Tuple[int, ...], ...], int],
    np.ndarray,
    np.ndarray,
    Tuple[int, int, int, int],
]:
    """Generate the 24 proper rotation matrices of SO(3)_cubic (det = +1)."""
    from itertools import permutations, product

    matrices = []
    # Ensure identity is index 0
    identity = np.eye(3, dtype=np.int64)
    matrices.append(identity)

    for perm in permutations((0, 1, 2)):
        for signs in product((-1, 1), repeat=3):
            mat = np.zeros((3, 3), dtype=np.int64)
            for r_idx, c_idx in enumerate(perm):
                mat[r_idx, c_idx] = signs[r_idx]
            det = int(round(np.linalg.det(mat)))
            if det == 1 and not np.array_equal(mat, identity):
                matrices.append(mat)

    assert len(matrices) == 24, f"Expected 24 proper cubic rotations, got {len(matrices)}"
    rot_stack = np.stack(matrices, axis=0)  # [24, 3, 3]

    lookup: dict[Tuple[Tuple[int, ...], ...], int] = {}
    for idx, mat in enumerate(rot_stack):
        key = tuple(tuple(int(v) for v in row) for row in mat)
        lookup[key] = idx

    mul_table = np.zeros((24, 24), dtype=np.int64)
    inv_table = np.zeros(24, dtype=np.int64)
    for i in range(24):
        inv_mat = rot_stack[i].T
        inv_key = tuple(tuple(int(v) for v in row) for row in inv_mat)
        inv_table[i] = lookup[inv_key]
        for j in range(24):
            prod = rot_stack[i] @ rot_stack[j]
            prod_key = tuple(tuple(int(v) for v in row) for row in prod)
            mul_table[i, j] = lookup[prod_key]

    # Planar C_4 rotations around local +Z axis: 0, 90, 180, 270 deg
    c4_mats = [
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int64),
        np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.int64),
        np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.int64),
        np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.int64),
    ]
    c4_ids = tuple(
        lookup[tuple(tuple(int(v) for v in row) for row in m)] for m in c4_mats
    )
    return rot_stack, lookup, mul_table, inv_table, c4_ids  # type: ignore[return-value]


SO3_CUBIC_MATRICES, _MAT_TO_ID, _MUL_TABLE, _INV_TABLE, C4_Z_ROT_IDS = _build_so3_cubic_group()

_ROT_FLAT: Tuple[Tuple[int, int, int, int, int, int, int, int, int], ...] = tuple(
    (
        int(m[0, 0]),
        int(m[0, 1]),
        int(m[0, 2]),
        int(m[1, 0]),
        int(m[1, 1]),
        int(m[1, 2]),
        int(m[2, 0]),
        int(m[2, 1]),
        int(m[2, 2]),
    )
    for m in SO3_CUBIC_MATRICES
)
_MUL_TUPLE: Tuple[Tuple[int, ...], ...] = tuple(
    tuple(int(v) for v in row) for row in _MUL_TABLE
)
_INV_TUPLE: Tuple[int, ...] = tuple(int(v) for v in _INV_TABLE)
_X_AXES: Tuple[Tuple[int, int, int], ...] = tuple((r[0], r[3], r[6]) for r in _ROT_FLAT)
_Y_AXES: Tuple[Tuple[int, int, int], ...] = tuple((r[1], r[4], r[7]) for r in _ROT_FLAT)
_Z_AXES: Tuple[Tuple[int, int, int], ...] = tuple((r[2], r[5], r[8]) for r in _ROT_FLAT)


def rot_id_to_matrix(rot_id: int) -> np.ndarray:
    """Return the 3x3 integer rotation matrix for a rotation ID in 0..23."""
    return SO3_CUBIC_MATRICES[rot_id]


def matrix_to_rot_id(mat: np.ndarray) -> int:
    """Map an exact 3x3 signed permutation matrix (det=+1) to its rotation ID in 0..23."""
    key = tuple(tuple(int(v) for v in row) for row in mat)
    if key not in _MAT_TO_ID:
        raise ValueError(f"Matrix is not in SO(3)_cubic:\n{mat}")
    return _MAT_TO_ID[key]


def compose_rot_ids(rot_a: int, rot_b: int) -> int:
    """Return the rotation ID of R_a @ R_b in O(1)."""
    return _MUL_TUPLE[rot_a][rot_b]


def invert_rot_id(rot_id: int) -> int:
    """Return the rotation ID of R^T in O(1)."""
    return _INV_TUPLE[rot_id]


def apply_rot(rot_id: int, vec: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Multiply 3x3 integer rotation matrix R_{rot_id} by integer 3-vector vec."""
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = _ROT_FLAT[rot_id]
    x, y, z = vec
    if type(x) is int and type(y) is int and type(z) is int:
        return (
            m00 * x + m01 * y + m02 * z,
            m10 * x + m11 * y + m12 * z,
            m20 * x + m21 * y + m22 * z,
        )
    xi, yi, zi = int(x), int(y), int(z)
    return (
        m00 * xi + m01 * yi + m02 * zi,
        m10 * xi + m11 * yi + m12 * zi,
        m20 * xi + m21 * yi + m22 * zi,
    )


@dataclass(frozen=True, slots=True)
class LDUFrame:
    """
    Exact discrete rigid frame T = (R, r) in Z^3_LDU ⋊ SO(3)_cubic.
    """

    rot_id: int = 0
    pos: Tuple[int, int, int] = (0, 0, 0)

    @property
    def t(self) -> Tuple[int, int, int]:
        """Alias for `pos` (integer translation tuple)."""
        return self.pos

    @property
    def x_axis(self) -> Tuple[int, int, int]:
        """Return the local +X tangent direction in world coordinates."""
        return _X_AXES[self.rot_id]

    @property
    def y_axis(self) -> Tuple[int, int, int]:
        """Return the local +Y tangent direction in world coordinates."""
        return _Y_AXES[self.rot_id]

    @property
    def z_axis(self) -> Tuple[int, int, int]:
        """Return the local +Z outward normal direction in world coordinates."""
        return _Z_AXES[self.rot_id]

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Return the 3x3 integer rotation matrix corresponding to `rot_id`."""
        return rot_id_to_matrix(self.rot_id)

    def compose(self, other: LDUFrame) -> LDUFrame:
        """Return self ∘ other = (R_self R_other, r_self + R_self r_other)."""
        m00, m01, m02, m10, m11, m12, m20, m21, m22 = _ROT_FLAT[self.rot_id]
        ox, oy, oz = other.pos
        px, py, pz = self.pos
        return LDUFrame(
            rot_id=_MUL_TUPLE[self.rot_id][other.rot_id],
            pos=(
                px + m00 * ox + m01 * oy + m02 * oz,
                py + m10 * ox + m11 * oy + m12 * oz,
                pz + m20 * ox + m21 * oy + m22 * oz,
            ),
        )

    def inverse(self) -> LDUFrame:
        """Return T^-1 = (R^T, -R^T r)."""
        inv_r = _INV_TUPLE[self.rot_id]
        m00, m01, m02, m10, m11, m12, m20, m21, m22 = _ROT_FLAT[inv_r]
        px, py, pz = self.pos
        return LDUFrame(
            rot_id=inv_r,
            pos=(
                -(m00 * px + m01 * py + m02 * pz),
                -(m10 * px + m11 * py + m12 * pz),
                -(m20 * px + m21 * py + m22 * pz),
            ),
        )

    def transform_point(self, pt: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Map a local integer LDU point to world LDU coordinates: r + R * pt."""
        m00, m01, m02, m10, m11, m12, m20, m21, m22 = _ROT_FLAT[self.rot_id]
        ox, oy, oz = pt
        px, py, pz = self.pos
        return (
            px + m00 * ox + m01 * oy + m02 * oz,
            py + m10 * ox + m11 * oy + m12 * oz,
            pz + m20 * ox + m21 * oy + m22 * oz,
        )

    def transform_direction(self, d: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Map a local integer direction vector by R."""
        return apply_rot(self.rot_id, d)


def relative_transform(frame_u: LDUFrame, frame_v: LDUFrame) -> LDUFrame:
    """
    Compute the strictly relative transform ΔT_uv = T_u^-1 ∘ T_v.
    Invariant under any global transform H in Z^3_LDU ⋊ SO(3)_cubic:
    (H ∘ T_u)^-1 ∘ (H ∘ T_v) == T_u^-1 ∘ T_v.
    """
    return frame_u.inverse().compose(frame_v)


def local_gravity(frame_u: LDUFrame) -> Tuple[int, int, int]:
    """
    Return the world gravity vector expressed in frame_u's local coordinates:
    g_u = R_u^T g_world.
    """
    inv_r = _INV_TUPLE[frame_u.rot_id]
    return apply_rot(inv_r, WORLD_GRAVITY)


@dataclass(frozen=True, slots=True)
class LDUBox:
    """
    Exact integer Axis-Aligned Bounding Box in LDU coordinates.
    Closed under all 24 proper cubic rotations in SO(3)_cubic.
    """

    xmin: int
    ymin: int
    zmin: int
    xmax: int
    ymax: int
    zmax: int

    def __post_init__(self) -> None:
        if not (self.xmin < self.xmax and self.ymin < self.ymax and self.zmin < self.zmax):
            raise ValueError(
                f"Invalid LDUBox bounds: ({self.xmin}, {self.ymin}, {self.zmin}) .. "
                f"({self.xmax}, {self.ymax}, {self.zmax})"
            )

    def transform(self, frame: LDUFrame) -> LDUBox:
        """
        Transform this box by rigid frame T = (R, r) in Z^3_LDU ⋊ SO(3)_cubic.
        Because R is a signed permutation matrix, the image of an AABB is an exact AABB.
        """
        m00, m01, m02, m10, m11, m12, m20, m21, m22 = _ROT_FLAT[frame.rot_id]
        px, py, pz = frame.pos
        x0, y0, z0 = self.xmin, self.ymin, self.zmin
        x1, y1, z1 = self.xmax, self.ymax, self.zmax
        p0x = px + m00 * x0 + m01 * y0 + m02 * z0
        p1x = px + m00 * x1 + m01 * y1 + m02 * z1
        p0y = py + m10 * x0 + m11 * y0 + m12 * z0
        p1y = py + m10 * x1 + m11 * y1 + m12 * z1
        p0z = pz + m20 * x0 + m21 * y0 + m22 * z0
        p1z = pz + m20 * x1 + m21 * y1 + m22 * z1
        return LDUBox(
            xmin=p0x if p0x < p1x else p1x,
            ymin=p0y if p0y < p1y else p1y,
            zmin=p0z if p0z < p1z else p1z,
            xmax=p1x if p0x < p1x else p0x,
            ymax=p1y if p0y < p1y else p0y,
            zmax=p1z if p0z < p1z else p0z,
        )

    def intersects(self, other: LDUBox) -> bool:
        """
        Return True iff the open interiors of self and other overlap by >= 1 LDU^3.
        Flush face-touching boxes (e.g., zmax == other.zmin) do not intersect.
        """
        return (
            max(self.xmin, other.xmin) < min(self.xmax, other.xmax)
            and max(self.ymin, other.ymin) < min(self.ymax, other.ymax)
            and max(self.zmin, other.zmin) < min(self.zmax, other.zmax)
        )

    def contains_point_strict(self, pt: Tuple[int, int, int]) -> bool:
        """Return True iff pt lies strictly inside the open box interior."""
        x, y, z = pt
        return self.xmin < x < self.xmax and self.ymin < y < self.ymax and self.zmin < z < self.zmax

    @property
    def volume_ldu3(self) -> int:
        return (self.xmax - self.xmin) * (self.ymax - self.ymin) * (self.zmax - self.zmin)
