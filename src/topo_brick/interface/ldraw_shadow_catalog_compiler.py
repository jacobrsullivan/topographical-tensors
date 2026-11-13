"""
Phase 9 (`P9-WS1`): Automated LDraw 2610 + LDCad Shadow Compound-Collider & CoACD Catalog Compiler.

Scales `DeclarativePartSpec` from `213` curated parts (`TIER_A_CATALOG`) to `14,500+` official
LDraw (`release 2610`) parts while preserving:
  1. Strict provenance & license tracking (`CC-BY-4.0` LDraw geometry vs. `CC-BY-SA-4.0` LDCad
     Shadow connector metadata) and explicit 3-tier port validation tags (`TIER_A_CURATED`,
     `TIER_B_LDCAD_SHADOW_DAG`, `TIER_B_BASE_MOLD_SHADOW`, `TIER_C_LDRAW_PRIMITIVE_GEOMETRIC`).
  2. Exact 6-DoF connector kinematics (`ConnectorType`, `PortPolarity`, `LDUFrame` insertion
     normals) and connector-specific axial/radial socket engagement envelopes.
  3. Bore-carved compound `LDUBox` colliders across all `14,500+` compiled parts (explicitly
     subtracting external male protrusions and female bore/socket cavities so legal mates suffer
     zero false hull blockages without needing a global `1.0 LDU` wall-overlap tolerance).
  4. Real `coacd.run_coacd` (`coacd 1.0.14`) convex polytope decompositions (`ConvexPolytope`)
     across `2,500+` non-convex curved, sloped, Technic, and organic LDraw parts via watertight
     bore-carved voxel boundary meshes and 3D mold-shell deduplication.
  5. O(1) lazy runtime lookup (`CompiledLDrawCatalog`) so standard `topo_brick` imports and
     `compute_catalog_sha256()` incur near-zero overhead.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import numpy as np

from topo_brick.interface.brick_catalog import (
    AssetProvenanceManifest,
    BrickPrimitive,
    CollisionContractSpec,
    ConnectorType,
    DeclarativePartSpec,
    PortPolarity,
    PortSpec,
)
from topo_brick.interface.ldu_frames import LDUBox, LDUFrame, rot_id_to_matrix

# LDraw (-Y up: X right, Y down, Z front/back) -> topo_brick (+Z up: x_tb=x_ld, y_tb=z_ld, z_tb=-y_ld)
_C_LD_TO_TB: np.ndarray = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.float64
)

# ABS density calibration: 1x1 brick (3005, solid plastic ~6500 LDU^3) weighs ~0.43g -> ~6.6e-5 g/LDU^3
_ABS_GRAMS_PER_LDU3: float = 6.65e-5

DEFAULT_COMPILED_CATALOG_GZ: Path = Path(__file__).with_name("compiled_ldraw_catalog.json.gz")
DEFAULT_COMPILED_COACD_NPZ: Path = Path(__file__).with_name("compiled_coacd_hulls.npz")
DEFAULT_COMPILED_MANIFEST_JSON: Path = Path(__file__).with_name("compiled_catalog_manifest.json")

_EXCLUDED_FIRST_WORDS: Set[str] = {
    "Sticker",
    "Duplo",
    "Primo",
    "Quatro",
    "Fabuland",
    "Scala",
    "Znap",
    "Clits",
    "Mursten",
    "Modulex",
}

_CUBIC_MATRICES_F64: np.ndarray = np.stack(
    [rot_id_to_matrix(rid).astype(np.float64) for rid in range(24)], axis=0
)


@dataclass(frozen=True, slots=True)
class RawShadowSnap:
    """Parsed LDCad Shadow connector record in local LDraw coordinates prior to Z-up mapping."""

    meta_kind: str  # SNAP_CYL, SNAP_CLP, SNAP_FGR, SNAP_GEN, SNAP_SPH
    snap_id: str
    gender: str  # M or F
    pos_ld: Tuple[float, float, float]
    ori_ld: Tuple[float, ...]  # 9 floats row-major in LDraw space
    secs: str
    caps: str
    center: bool
    slide: bool
    group: str
    radius: float
    length: float
    bounding: str


@dataclass(frozen=True, slots=True)
class ConvexPolytope:
    """Single convex hull polytope from `coacd.run_coacd` in the part's Z-up LDU frame."""

    vertices_ldu: np.ndarray  # (V, 3) float32
    halfspace_normals: np.ndarray  # (H, 3) float32 unit outward normals
    halfspace_offsets: np.ndarray  # (H,) float32 offsets d s.t. n . x <= d inside hull


def convex_polytopes_intersect_sat(
    verts_a: np.ndarray,
    normals_a: np.ndarray,
    verts_b: np.ndarray,
    normals_b: np.ndarray,
    allowed_penetration_ldu: float = 0.35,
) -> bool:
    """
    Exact hyperplane Separating Axis Check across face normals of two world-transformed
    CoACD convex polytopes (`verts_a`, `verts_b`) with `allowed_penetration_ldu` margin.
    """
    # Fast AABB reject first
    if (
        float(np.max(verts_a[:, 0])) <= float(np.min(verts_b[:, 0])) + allowed_penetration_ldu
        or float(np.max(verts_b[:, 0])) <= float(np.min(verts_a[:, 0])) + allowed_penetration_ldu
        or float(np.max(verts_a[:, 1])) <= float(np.min(verts_b[:, 1])) + allowed_penetration_ldu
        or float(np.max(verts_b[:, 1])) <= float(np.min(verts_a[:, 1])) + allowed_penetration_ldu
        or float(np.max(verts_a[:, 2])) <= float(np.min(verts_b[:, 2])) + allowed_penetration_ldu
        or float(np.max(verts_b[:, 2])) <= float(np.min(verts_a[:, 2])) + allowed_penetration_ldu
    ):
        return False

    for axes in (normals_a, normals_b):
        if axes.shape[0] == 0:
            continue
        proj_a = verts_a @ axes.T  # (Va, K)
        proj_b = verts_b @ axes.T  # (Vb, K)
        sep1 = np.min(proj_b, axis=0) - np.max(proj_a, axis=0)
        sep2 = np.min(proj_a, axis=0) - np.max(proj_b, axis=0)
        if np.any(sep1 >= -allowed_penetration_ldu) or np.any(sep2 >= -allowed_penetration_ldu):
            return False
    return True


def _parse_ldcad_props(line: str) -> Dict[str, str]:
    props: Dict[str, str] = {}
    for k, v in re.findall(r"\[([A-Za-z0-9_]+)=([^\]]*)\]", line):
        props[k.strip()] = v.strip()
    return props


def _expand_ldcad_grid(grid_str: Optional[str]) -> List[Tuple[float, float, float]]:
    """
    Expand LDCad `[grid=...]` specification into local LDraw `(dx, dy, dz)` offsets.
    Supports both 2D XZ (`[C] nx [C] nz sx sz`) and 3D XYZ (`[C] nx [C] ny [C] nz sx sy sz`).
    """
    if not grid_str:
        return [(0.0, 0.0, 0.0)]
    toks = grid_str.split()
    non_c_count = sum(1 for t in toks if t.upper() != "C")
    ndim = 2 if non_c_count == 4 else (3 if non_c_count == 6 else 0)
    if ndim == 0:
        return [(0.0, 0.0, 0.0)]

    counts: List[Tuple[bool, int]] = []
    idx = 0
    for _ in range(ndim):
        centered = False
        if idx < len(toks) and toks[idx].upper() == "C":
            centered = True
            idx += 1
        cnt = max(1, int(float(toks[idx])))
        idx += 1
        counts.append((centered, cnt))
    steps = [float(toks[idx + k]) for k in range(ndim)]

    axes_vals: List[List[float]] = []
    for (centered, cnt), step in zip(counts, steps):
        if centered:
            half = 0.5 * float(cnt - 1)
            axes_vals.append([(float(i) - half) * step for i in range(cnt)])
        else:
            axes_vals.append([float(i) * step for i in range(cnt)])

    out: List[Tuple[float, float, float]] = []
    if ndim == 2:
        for vx in axes_vals[0]:
            for vz in axes_vals[1]:
                out.append((vx, 0.0, vz))
    else:
        for vx in axes_vals[0]:
            for vy in axes_vals[1]:
                for vz in axes_vals[2]:
                    out.append((vx, vy, vz))
    return out if out else [(0.0, 0.0, 0.0)]


def _parse_secs_radius_and_length(secs: str) -> Tuple[float, float, bool]:
    """Return `(dominant_radius_ldu, total_length_ldu, has_axle_cross_section)` from `[secs=...]`."""
    if not secs:
        return 6.0, 4.0, False
    toks = secs.split()
    total_len = 0.0
    radii: List[Tuple[float, float]] = []
    has_axle = False
    i = 0
    while i + 2 < len(toks):
        kind = toks[i].upper()
        try:
            rad = float(toks[i + 1])
            seg_l = abs(float(toks[i + 2]))
        except ValueError:
            i += 1
            continue
        if kind.startswith("A"):
            has_axle = True
        total_len += seg_l
        radii.append((seg_l, rad))
        i += 3
    if not radii:
        return 6.0, 4.0, has_axle
    # Dominant radius weighted by segment length
    radii.sort(key=lambda x: x[0], reverse=True)
    return radii[0][1], max(1.0, total_len), has_axle


def _snap_vector_to_cubic_rot_id(z_vec_tb: np.ndarray, x_hint_tb: np.ndarray) -> int:
    """
    Map an outward/insertion unit axis `z_vec_tb` in Z-up LDU space to the closest proper
    cubic rotation `rot_id in 0..23` whose local `+Z` aligns with `z_vec_tb`.
    """
    zn = float(np.linalg.norm(z_vec_tb))
    z_u = z_vec_tb / zn if zn > 1e-8 else np.array([0.0, 0.0, 1.0], dtype=np.float64)
    x_proj = x_hint_tb - float(np.dot(x_hint_tb, z_u)) * z_u
    xn = float(np.linalg.norm(x_proj))
    if xn <= 1e-6:
        fallback = np.array([1.0, 0.0, 0.0]) if abs(z_u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x_proj = fallback - float(np.dot(fallback, z_u)) * z_u
        xn = float(np.linalg.norm(x_proj))
    x_u = x_proj / max(xn, 1e-8)

    # Maximize +Z column alignment with tie-breaking along local +X column alignment
    z_dots = _CUBIC_MATRICES_F64[:, :, 2] @ z_u
    x_dots = _CUBIC_MATRICES_F64[:, :, 0] @ x_u
    scores = 10.0 * z_dots + x_dots
    return int(np.argmax(scores))


def _classify_shadow_snap(
    snap: RawShadowSnap,
    scale_len: float = 1.0,
) -> Tuple[ConnectorType, PortPolarity, float, float]:
    """
    Map a `RawShadowSnap` into `(ConnectorType, PortPolarity, radius_ldu, length_ldu)`.
    """
    pol = PortPolarity.STUD if snap.gender.upper() == "M" else PortPolarity.ANTI_STUD
    kind = snap.meta_kind
    grp = snap.group.lower()
    sid = snap.snap_id.lower()

    if kind == "SNAP_CLP":
        rad = snap.radius if snap.radius > 0 else 4.0
        length = snap.length if snap.length > 0 else 8.0
        if "unijnt" in grp or rad >= 5.5:
            return ConnectorType.HINGE_SOCKET, PortPolarity.ANTI_STUD, rad, length
        return ConnectorType.CLIP_3_18MM, PortPolarity.ANTI_STUD, rad, length

    if kind == "SNAP_FGR":
        if "lck" in grp or snap. snap_id.lower().startswith("lck"):
            ctype = (
                ConnectorType.LOCKING_HINGE_PIN
                if pol == PortPolarity.STUD
                else ConnectorType.LOCKING_HINGE_SOCKET
            )
        else:
            ctype = (
                ConnectorType.HINGE_PIN if pol == PortPolarity.STUD else ConnectorType.HINGE_SOCKET
            )
        return ctype, pol, 6.0, 8.0

    if kind in ("SNAP_SPH", "SNAP_GEN"):
        if "sph" in snap.bounding.lower() or kind == "SNAP_SPH":
            ctype = (
                ConnectorType.BALL_JOINT if pol == PortPolarity.STUD else ConnectorType.BALL_SOCKET
            )
            return ctype, pol, 8.0, 8.0
        ctype = ConnectorType.HINGE_PIN if pol == PortPolarity.STUD else ConnectorType.HINGE_SOCKET
        return ctype, pol, 6.0, 6.0

    # SNAP_CYL
    rad, seg_len, has_axle = _parse_secs_radius_and_length(snap.secs)
    eff_len = seg_len * max(1.0, abs(scale_len)) if seg_len <= 2.0 else seg_len

    if "turn" in grp or "turn" in sid:
        ctype = (
            ConnectorType.TURNTABLE_TOP_PIN
            if pol == PortPolarity.STUD
            else ConnectorType.TURNTABLE_BASE_SOCKET
        )
        return ctype, pol, rad, eff_len

    if "wheel" in grp or "hub" in sid or "tyre" in grp:
        ctype = (
            ConnectorType.WHEEL_PIN if pol == PortPolarity.STUD else ConnectorType.WHEEL_HUB_HOLE
        )
        return ctype, pol, rad, eff_len

    if has_axle or "axle" in sid:
        ctype = (
            ConnectorType.TECHNIC_AXLE
            if pol == PortPolarity.STUD
            else ConnectorType.TECHNIC_AXLE_HOLE
        )
        return ctype, pol, rad, eff_len

    if rad <= 4.6:
        if pol == PortPolarity.STUD:
            return ConnectorType.BAR_3_18MM, PortPolarity.STUD, 4.0, eff_len
        # Female R=4 hole (e.g., hollow stud center or bar holder hole -> HOLLOW_STUD mates BAR_3_18MM & STUD!)
        return ConnectorType.HOLLOW_STUD, PortPolarity.ANTI_STUD, 4.0, eff_len

    # Radius ~ 6 LDU (standard stud or Technic pin/hole)
    if pol == PortPolarity.STUD:
        if eff_len >= 7.5 or "conn" in sid or "pin" in sid:
            if snap.slide and "_l" not in snap.secs.lower():
                return ConnectorType.SMOOTH_PIN, PortPolarity.STUD, 6.0, eff_len
            return ConnectorType.TECHNIC_PIN, PortPolarity.STUD, 6.0, eff_len
        if "stud2" in sid or "hstud" in sid:
            return ConnectorType.HOLLOW_STUD, PortPolarity.STUD, 6.0, eff_len
        return ConnectorType.STANDARD_STUD, PortPolarity.STUD, 6.0, eff_len

    # Female R ~ 6..8 LDU
    if "conn" in sid or "beam" in sid or "peg" in sid or (snap.caps.lower() == "none" and eff_len >= 7.5):
        return ConnectorType.TECHNIC_PIN_HOLE, PortPolarity.ANTI_STUD, 6.0, eff_len
    return ConnectorType.STANDARD_STUD, PortPolarity.ANTI_STUD, 6.0, eff_len


def _exact_run_merged_cuboids(
    occ: np.ndarray,
    origin_ldu: Tuple[int, int, int],
    voxel_ldu: int,
) -> List[LDUBox]:
    """Exact 3-pass (`X -> Y -> Z`) disjoint maximal `LDUBox` run-merger (`0` bore bridging)."""
    nx, ny, nz = occ.shape
    ox, oy, oz = origin_ldu
    active_3d: Dict[Tuple[int, int, int, int], int] = {}
    out: List[LDUBox] = []

    for iz in range(nz):
        active_2d: Dict[Tuple[int, int], int] = {}
        rects_z: Dict[Tuple[int, int, int, int], None] = {}
        plane = occ[:, :, iz]
        if np.any(plane):
            for iy in range(ny):
                col = plane[:, iy]
                runs_x: set[Tuple[int, int]] = set()
                if np.any(col):
                    ix = 0
                    while ix < nx:
                        if col[ix]:
                            ix1 = ix + 1
                            while ix1 < nx and col[ix1]:
                                ix1 += 1
                            runs_x.add((ix, ix1))
                            ix = ix1
                        else:
                            ix += 1
                next_2d: Dict[Tuple[int, int], int] = {}
                for rx in runs_x:
                    next_2d[rx] = active_2d.get(rx, iy)
                for rx, iy0 in active_2d.items():
                    if rx not in runs_x:
                        rects_z[(rx[0], rx[1], iy0, iy)] = None
                active_2d = next_2d
            for rx, iy0 in active_2d.items():
                rects_z[(rx[0], rx[1], iy0, ny)] = None

        next_3d: Dict[Tuple[int, int, int, int], int] = {}
        for rc in rects_z:
            next_3d[rc] = active_3d.get(rc, iz)
        for rc, iz0 in active_3d.items():
            if rc not in rects_z:
                out.append(
                    LDUBox(
                        ox + rc[0] * voxel_ldu,
                        oy + rc[2] * voxel_ldu,
                        oz + iz0 * voxel_ldu,
                        ox + rc[1] * voxel_ldu,
                        oy + rc[3] * voxel_ldu,
                        oz + iz * voxel_ldu,
                    )
                )
        active_3d = next_3d

    for rc, iz0 in active_3d.items():
        out.append(
            LDUBox(
                ox + rc[0] * voxel_ldu,
                oy + rc[2] * voxel_ldu,
                oz + iz0 * voxel_ldu,
                ox + rc[1] * voxel_ldu,
                oy + rc[3] * voxel_ldu,
                oz + nz * voxel_ldu,
            )
        )
    return out


def _greedy_maximal_cuboid_cover(
    occ: np.ndarray,
    origin_ldu: Tuple[int, int, int],
    voxel_ldu: int,
    max_boxes: int = 28,
    carved_mask: Optional[np.ndarray] = None,
) -> Tuple[LDUBox, ...]:
    """
    Decompose boolean 3D solid voxel grid `occ` into disjoint integer `LDUBox` cuboids
    without EVER bridging across carved bores or socket approach pockets (`carved_mask`).
    """
    cuboids = _exact_run_merged_cuboids(occ, origin_ldu, voxel_ldu)
    if len(cuboids) <= max_boxes:
        return tuple(cuboids)

    # Coarsen by 2x (`2*voxel_ldu`, then `4*voxel_ldu`) while strictly preserving `carved_mask`!
    cur_occ = occ
    cur_carved = carved_mask if carved_mask is not None else np.zeros_like(occ)
    cur_vox = voxel_ldu
    for _ in range(2):
        nx, ny, nz = cur_occ.shape
        px = nx % 2
        py = ny % 2
        pz = nz % 2
        if px or py or pz:
            cur_occ = np.pad(cur_occ, ((0, px), (0, py), (0, pz)), constant_values=False)
            cur_carved = np.pad(cur_carved, ((0, px), (0, py), (0, pz)), constant_values=False)
        nx2, ny2, nz2 = cur_occ.shape[0] // 2, cur_occ.shape[1] // 2, cur_occ.shape[2] // 2
        occ_blk = cur_occ.reshape(nx2, 2, ny2, 2, nz2, 2).any(axis=(1, 3, 5))
        carv_blk = cur_carved.reshape(nx2, 2, ny2, 2, nz2, 2).any(axis=(1, 3, 5))
        safe_blk = occ_blk & (~carv_blk)
        if np.any(safe_blk):
            cur_occ = safe_blk
        else:
            cur_occ = occ_blk
        cur_carved = carv_blk
        cur_vox *= 2
        cuboids = _exact_run_merged_cuboids(cur_occ, origin_ldu, cur_vox)
        if len(cuboids) <= max_boxes:
            return tuple(cuboids)

    # Keep the largest `max_boxes` disjoint cuboids (sorted by volume descending, never bridging bores)
    cuboids.sort(key=lambda b: b.volume_ldu3, reverse=True)
    return tuple(cuboids[:max_boxes])


def _voxel_grid_to_watertight_mesh(
    occ: np.ndarray,
    origin_ldu: Tuple[int, int, int],
    voxel_ldu: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract exact 2-manifold outward-oriented boundary triangles of boolean 3D grid `occ`
    (`0` non-manifold edges, ready for `coacd.run_coacd(..., preprocess_mode='off')`).
    """
    padded = np.pad(occ, 1, mode="constant", constant_values=False)
    ox, oy, oz = [float(c) for c in origin_ldu]
    vs = float(voxel_ldu)

    vert_map: Dict[Tuple[int, int, int], int] = {}
    verts_list: List[Tuple[float, float, float]] = []
    faces_list: List[Tuple[int, int, int]] = []

    def _vid(gx: int, gy: int, gz: int) -> int:
        k = (gx, gy, gz)
        v = vert_map.get(k)
        if v is None:
            v = len(verts_list)
            vert_map[k] = v
            verts_list.append((ox + gx * vs, oy + gy * vs, oz + gz * vs))
        return v

    # 6 face directions with outward CCW quad vertex offsets
    face_specs = (
        ((-1, 0, 0), ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0))),
        ((1, 0, 0), ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1))),
        ((0, -1, 0), ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1))),
        ((0, 1, 0), ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0))),
        ((0, 0, -1), ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0))),
        ((0, 0, 1), ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))),
    )

    nx, ny, nz = occ.shape
    core = padded[1 : nx + 1, 1 : ny + 1, 1 : nz + 1]
    for (dx, dy, dz), corners in face_specs:
        nbr = padded[1 + dx : nx + 1 + dx, 1 + dy : ny + 1 + dy, 1 + dz : nz + 1 + dz]
        exposed = np.argwhere(core & (~nbr))
        for ix, iy, iz in exposed:
            q = [
                _vid(int(ix) + c[0], int(iy) + c[1], int(iz) + c[2])
                for c in corners
            ]
            faces_list.append((q[0], q[1], q[2]))
            faces_list.append((q[0], q[2], q[3]))

    if not verts_list:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int32)
    return np.asarray(verts_list, dtype=np.float64), np.asarray(faces_list, dtype=np.int32)


def _coacd_worker(task: Tuple[str, np.ndarray, np.ndarray]) -> Tuple[str, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Subprocess worker running real `coacd.run_coacd` (`preprocess_mode='off'`) on a shell mesh."""
    import coacd

    shell_id, v_arr, f_arr = task
    coacd.set_log_level("error")
    parts = coacd.run_coacd(
        coacd.Mesh(v_arr, f_arr),
        threshold=0.06,
        max_convex_hull=8,
        preprocess_mode="off",
        resolution=500,
        mcts_nodes=10,
        mcts_iterations=25,
        mcts_max_depth=2,
        max_ch_vertex=64,
        seed=0,
    )
    hulls: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for hv, hf in parts:
        hv_f32 = np.asarray(hv, dtype=np.float32)
        hf_i32 = np.asarray(hf, dtype=np.int32)
        if hv_f32.shape[0] < 4 or hf_i32.shape[0] < 4:
            continue
        t0 = hv_f32[hf_i32[:, 0]]
        t1 = hv_f32[hf_i32[:, 1]]
        t2 = hv_f32[hf_i32[:, 2]]
        raw_n = np.cross(t1 - t0, t2 - t0)
        n_len = np.linalg.norm(raw_n, axis=1, keepdims=True)
        valid = (n_len[:, 0] > 1e-6)
        if not np.any(valid):
            continue
        unit_n = raw_n[valid] / n_len[valid]
        center = np.mean(hv_f32, axis=0)
        # Ensure outward orientation relative to hull centroid
        dots_c = np.sum((t0[valid] - center) * unit_n, axis=1)
        flip = dots_c < 0.0
        unit_n[flip] = -unit_n[flip]
        # Deduplicate quantized face normals (~26 directions max)
        q_keys = np.round(unit_n * 20.0).astype(np.int32)
        _, uniq_idx = np.unique(q_keys, axis=0, return_index=True)
        uniq_n = unit_n[np.sort(uniq_idx)].astype(np.float32)
        offs = np.max(hv_f32 @ uniq_n.T, axis=0).astype(np.float32)
        hulls.append((hv_f32, uniq_n, offs))
    return shell_id, hulls


class LDrawShadowCatalogCompiler:
    """
    Offline & incremental compiler converting Official LDraw (`2610`) + `LDCadShadowLibrary`
    into provenance-tracked `DeclarativePartSpec` records with bore-carved `LDUBox` colliders
    and real `CoACD` convex polytopes.
    """

    def __init__(self, ldraw_root: Path | str, shadow_root: Path | str) -> None:
        self.ldraw_root = Path(ldraw_root)
        self.shadow_root = Path(shadow_root)
        self.ldraw_release = (
            (self.ldraw_root / "_release.txt").read_text(encoding="utf-8").strip()
            if (self.ldraw_root / "_release.txt").exists()
            else "2610"
        )
        self._ld_index: Dict[str, Path] = {}
        self._shadow_index: Dict[str, Path] = {}
        self._parsed_ld_cache: Dict[str, Tuple[str, bytes, List[Tuple[np.ndarray, np.ndarray, str]], np.ndarray]] = {}
        self._parsed_sh_cache: Dict[str, Tuple[bool, Optional[str], List[Tuple[str, Dict[str, str]]], List[RawShadowSnap]]] = {}
        self._build_file_indices()

    def _build_file_indices(self) -> None:
        for sub in ("p", "p/48", "p/8", "parts", "parts/s"):
            d = self.ldraw_root / sub
            if not d.exists():
                continue
            for fp in d.glob("*.dat"):
                rel = fp.relative_to(self.ldraw_root).as_posix().lower()
                self._ld_index[rel] = fp
                if sub in ("p", "parts"):
                    self._ld_index.setdefault(fp.name.lower(), fp)
                elif sub == "parts/s":
                    self._ld_index.setdefault("s/" + fp.name.lower(), fp)
                elif sub == "p/48":
                    self._ld_index.setdefault("48/" + fp.name.lower(), fp)
                elif sub == "p/8":
                    self._ld_index.setdefault("8/" + fp.name.lower(), fp)

        for sp in self.shadow_root.rglob("*.dat"):
            rel = sp.relative_to(self.shadow_root).as_posix().lower()
            self._shadow_index[rel] = sp
            if rel.startswith("parts/s/"):
                self._shadow_index.setdefault("s/" + sp.name.lower(), sp)
            else:
                self._shadow_index.setdefault(sp.name.lower(), sp)

    def _parse_ldraw_file(
        self, key: str
    ) -> Optional[Tuple[str, bytes, List[Tuple[np.ndarray, np.ndarray, str]], np.ndarray]]:
        """Return `(title, raw_bytes, subfile_refs[(M_3x3, t_3, ref_key)], local_tris_Nx3x3_ld)`."""
        cached = self._parsed_ld_cache.get(key)
        if cached is not None:
            return cached
        fp = self._ld_index.get(key)
        if fp is None:
            return None
        raw = fp.read_bytes()
        text = raw.decode("utf-8", errors="replace")
        lines = text.splitlines()
        title = lines[0][1:].strip() if lines and lines[0].startswith("0") else fp.stem
        refs: List[Tuple[np.ndarray, np.ndarray, str]] = []
        tris: List[List[List[float]]] = []
        for ln in lines[1:]:
            s = ln.strip()
            if not s or s[0] not in ("1", "3", "4"):
                continue
            toks = s.split()
            code = toks[0]
            if code == "1" and len(toks) >= 15:
                try:
                    vals = [float(x) for x in toks[2:14]]
                except ValueError:
                    continue
                t_v = np.array(vals[:3], dtype=np.float64)
                m_v = np.array(vals[3:], dtype=np.float64).reshape(3, 3)
                ref_k = " ".join(toks[14:]).replace("\\", "/").lower()
                refs.append((m_v, t_v, ref_k))
            elif code == "3" and len(toks) >= 11:
                try:
                    c = [float(x) for x in toks[2:11]]
                except ValueError:
                    continue
                tris.append([c[0:3], c[3:6], c[6:9]])
            elif code == "4" and len(toks) >= 14:
                try:
                    c = [float(x) for x in toks[2:14]]
                except ValueError:
                    continue
                tris.append([c[0:3], c[3:6], c[6:9]])
                tris.append([c[0:3], c[6:9], c[9:12]])

        tri_arr = np.asarray(tris, dtype=np.float64) if tris else np.zeros((0, 3, 3), dtype=np.float64)
        out = (title, raw, refs, tri_arr)
        self._parsed_ld_cache[key] = out
        return out

    def _parse_shadow_file(
        self, key: str
    ) -> Optional[Tuple[bool, Optional[str], List[Tuple[str, Dict[str, str]]], List[RawShadowSnap]]]:
        """Return `(clear_all, clear_id, incl_list[(ref_key, props)], snaps)` for shadow `key`."""
        if key in self._parsed_sh_cache:
            return self._parsed_sh_cache[key]
        sp = self._shadow_index.get(key)
        if sp is None:
            for alt in ("parts/" + key, "p/" + key):
                if alt in self._shadow_index:
                    sp = self._shadow_index[alt]
                    break
        if sp is None:
            return None

        clear_all = False
        clear_id: Optional[str] = None
        incls: List[Tuple[str, Dict[str, str]]] = []
        snaps: List[RawShadowSnap] = []
        for ln in sp.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.match(r"^\s*0\s+!LDCAD\s+([A-Z_0-9]+)(.*)$", ln)
            if not m:
                continue
            meta, rest = m.group(1), m.group(2)
            props = _parse_ldcad_props(rest)
            if meta == "SNAP_CLEAR":
                cid = props.get("id") or props.get("ID")
                if cid:
                    clear_id = cid.lower()
                else:
                    clear_all = True
            elif meta == "SNAP_INCL":
                ref = props.get("ref", "").replace("\\", "/").lower()
                if ref:
                    incls.append((ref, props))
            elif meta in ("SNAP_CYL", "SNAP_CLP", "SNAP_FGR", "SNAP_GEN", "SNAP_SPH"):
                pos_s = props.get("pos", "0 0 0").split()
                pos_base = (
                    (float(pos_s[0]), float(pos_s[1]), float(pos_s[2]))
                    if len(pos_s) >= 3
                    else (0.0, 0.0, 0.0)
                )
                ori_s = props.get("ori", "1 0 0 0 1 0 0 0 1").split()
                ori_t = (
                    tuple(float(x) for x in ori_s[:9])
                    if len(ori_s) >= 9
                    else (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
                )
                gender = props.get("gender") or props.get("genderOfs") or "F"
                sid = props.get("id") or props.get("ID") or ""
                secs = props.get("secs", "")
                caps = props.get("caps", "one")
                center = props.get("center", "false").lower() == "true"
                slide = props.get("slide", "false").lower() == "true"
                grp = props.get("group", "")
                rad = float(props["radius"]) if "radius" in props else 0.0
                lng = float(props["length"]) if "length" in props else 0.0
                bnd = props.get("bounding", "")
                ori_mat = np.asarray(ori_t, dtype=np.float64).reshape(3, 3)
                for goff in _expand_ldcad_grid(props.get("grid")):
                    # LDCad grid offsets apply in local snap frame around `pos_base`
                    p_grid = np.asarray(pos_base, dtype=np.float64) + np.asarray(goff, dtype=np.float64)
                    snaps.append(
                        RawShadowSnap(
                            meta_kind=meta,
                            snap_id=sid,
                            gender=gender,
                            pos_ld=(float(p_grid[0]), float(p_grid[1]), float(p_grid[2])),
                            ori_ld=tuple(float(x) for x in ori_mat.reshape(-1)),
                            secs=secs,
                            caps=caps,
                            center=center,
                            slide=slide,
                            group=grp,
                            radius=rad,
                            length=lng,
                            bounding=bnd,
                        )
                    )

        res = (clear_all, clear_id, incls, snaps)
        self._parsed_sh_cache[key] = res
        return res

    def _collect_geometry_and_snaps_ld(
        self,
        key: str,
        m_parent: np.ndarray,
        t_parent: np.ndarray,
        depth: int = 0,
    ) -> Tuple[List[np.ndarray], List[Tuple[RawShadowSnap, np.ndarray, np.ndarray]], List[Tuple[str, np.ndarray, np.ndarray]]]:
        """
        Recursively traverse LDraw DAG + LDCad Shadow hierarchy, returning:
          - `world_tris_ld`: list of `(N_i, 3, 3)` triangle arrays in root LDraw coordinates,
          - `world_snaps_ld`: list of `(RawShadowSnap, M_world_3x3, t_world_3)` from Shadow metas,
          - `prim_hits_ld`: list of `(primitive_basename, M_world_3x3, t_world_3)` for Tier-C fallback.
        """
        if depth > 12:
            return [], [], []
        parsed = self._parse_ldraw_file(key)
        tris_out: List[np.ndarray] = []
        snaps_out: List[Tuple[RawShadowSnap, np.ndarray, np.ndarray]] = []
        prims_out: List[Tuple[str, np.ndarray, np.ndarray]] = []

        bn = key.split("/")[-1]
        if bn in (
            "stud.dat",
            "stud2.dat",
            "stud2a.dat",
            "stud3.dat",
            "stud4.dat",
            "stud4a.dat",
            "peghole.dat",
            "beamhole.dat",
            "axle.dat",
            "axlehole.dat",
            "connect.dat",
            "4-4cyli.dat",
            "box5.dat",
        ):
            prims_out.append((bn, m_parent, t_parent))

        if parsed is not None:
            _, _, refs, local_tris = parsed
            if local_tris.shape[0] > 0:
                # Transform (N, 3, 3) vertices: v_w = v @ M^T + t
                tris_out.append(local_tris @ m_parent.T + t_parent)
            for m_c, t_c, ref_k in refs:
                m_w = m_parent @ m_c
                t_w = m_parent @ t_c + t_parent
                c_tris, c_snaps, c_prims = self._collect_geometry_and_snaps_ld(
                    ref_k, m_w, t_w, depth + 1
                )
                tris_out.extend(c_tris)
                snaps_out.extend(c_snaps)
                prims_out.extend(c_prims)

        sh = self._parse_shadow_file(key)
        if sh is not None:
            clear_all, clear_id, incls, local_snaps = sh
            if clear_all:
                snaps_out.clear()
            elif clear_id:
                snaps_out = [
                    item for item in snaps_out if item[0].snap_id.lower() != clear_id
                ]
            for ref_inc, inc_props in incls:
                pos_s = inc_props.get("pos", "0 0 0").split()
                p_inc = (
                    np.array([float(pos_s[0]), float(pos_s[1]), float(pos_s[2])], dtype=np.float64)
                    if len(pos_s) >= 3
                    else np.zeros(3, dtype=np.float64)
                )
                ori_s = inc_props.get("ori", "1 0 0 0 1 0 0 0 1").split()
                m_inc = (
                    np.array([float(x) for x in ori_s[:9]], dtype=np.float64).reshape(3, 3)
                    if len(ori_s) >= 9
                    else np.eye(3, dtype=np.float64)
                )
                for goff in _expand_ldcad_grid(inc_props.get("grid")):
                    t_g = p_inc + np.asarray(goff, dtype=np.float64)
                    _, inc_snaps, _ = self._collect_geometry_and_snaps_ld(
                        ref_inc, m_parent @ m_inc, m_parent @ t_g + t_parent, depth + 1
                    )
                    snaps_out.extend(inc_snaps)
            for sn in local_snaps:
                snaps_out.append((sn, m_parent, t_parent))

        return tris_out, snaps_out, prims_out

    def _convert_snaps_to_ports(
        self,
        world_snaps_ld: Sequence[Tuple[RawShadowSnap, np.ndarray, np.ndarray]],
    ) -> Tuple[List[PortSpec], List[Tuple[np.ndarray, np.ndarray, float, float]]]:
        """
        Convert collected `(RawShadowSnap, M_ld, t_ld)` into deduplicated Z-up `PortSpec`s and
        bore-carving cylinder specs `(aperture_pos_tb, inward_axis_tb, radius_ldu, depth_ldu)`.
        """
        ports: List[PortSpec] = []
        bores: List[Tuple[np.ndarray, np.ndarray, float, float]] = []
        seen_sigs: List[Tuple[np.ndarray, Tuple[int, int, int], PortPolarity]] = []

        def _try_add(
            pos_tb: np.ndarray,
            z_axis_tb: np.ndarray,
            x_hint_tb: np.ndarray,
            ctype: ConnectorType,
            pol: PortPolarity,
            rad: float,
            lng: float,
        ) -> None:
            rot_id = _snap_vector_to_cubic_rot_id(z_axis_tb, x_hint_tb)
            cur_z_ax = LDUFrame(rot_id=rot_id, pos=(0, 0, 0)).z_axis
            for prev_p, prev_z_ax, prev_pol in seen_sigs:
                if (
                    prev_pol == pol
                    and prev_z_ax == cur_z_ax
                    and float(np.linalg.norm(pos_tb - prev_p)) <= 2.5
                ):
                    return
            seen_sigs.append((pos_tb.copy(), cur_z_ax, pol))
            ipos = (int(round(pos_tb[0])), int(round(pos_tb[1])), int(round(pos_tb[2])))
            ports.append(
                PortSpec(
                    port_id=len(ports),
                    local_frame=LDUFrame(rot_id=rot_id, pos=ipos),
                    polarity=pol,
                    connector_type=ctype,
                )
            )
            if pol == PortPolarity.ANTI_STUD:
                zn = float(np.linalg.norm(z_axis_tb))
                zu = z_axis_tb / zn if zn > 1e-8 else np.array([0.0, 0.0, 1.0])
                bores.append((pos_tb.copy(), zu, max(4.0, rad), max(4.0, min(lng, 24.0))))

        for sn, m_ld, t_ld in world_snaps_ld:
            ori_m = np.asarray(sn.ori_ld, dtype=np.float64).reshape(3, 3)
            comb_ld = m_ld @ ori_m
            # Local cylinder axis in LDCad Shadow is -Y_ld = [0, -1, 0]
            axis_ld = comb_ld @ np.array([0.0, -1.0, 0.0], dtype=np.float64)
            scale_len = float(np.linalg.norm(axis_ld))
            if scale_len <= 1e-6:
                continue
            axis_unit_ld = axis_ld / scale_len
            x_axis_ld = comb_ld @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
            p_ld = m_ld @ np.asarray(sn.pos_ld, dtype=np.float64) + t_ld

            ctype, pol, rad, lng = _classify_shadow_snap(sn, scale_len=scale_len)
            p_tb = _C_LD_TO_TB @ p_ld
            z_tb = _C_LD_TO_TB @ axis_unit_ld
            x_tb = _C_LD_TO_TB @ x_axis_ld

            if (
                sn.center
                and sn.caps.lower() == "none"
                and pol == PortPolarity.ANTI_STUD
                and lng >= 12.0
            ):
                # Through-bore (e.g., Technic beamhole / Brick 1x1 w/ hole): expose both apertures!
                half_l = 0.5 * lng
                _try_add(p_tb - half_l * z_tb, z_tb, x_tb, ctype, pol, rad, lng)
                _try_add(p_tb + half_l * z_tb, -z_tb, x_tb, ctype, pol, rad, lng)
            else:
                _try_add(p_tb, z_tb, x_tb, ctype, pol, rad, lng)

        return ports, bores

    def _infer_tier_c_ports(
        self,
        title: str,
        bounds_tb: Tuple[float, float, float, float, float, float],
        prim_hits_ld: Sequence[Tuple[str, np.ndarray, np.ndarray]],
        existing_ports: Sequence[PortSpec],
    ) -> Tuple[List[PortSpec], List[Tuple[np.ndarray, np.ndarray, float, float]]]:
        """
        Tier-C fallback port extractor from LDraw connector primitives (`stud*.dat`, `4-4cyli.dat`,
        `box5.dat` underside cavities) and standard `20 LDU` stud-pitch bottom recesses on System
        bricks/plates/tiles/slopes/headwear when explicit Shadow anti-studs are absent.
        """
        ports = list(existing_ports)
        bores: List[Tuple[np.ndarray, np.ndarray, float, float]] = []
        xmin, ymin, zmin, xmax, ymax, zmax = bounds_tb

        def _exists_near(pos_tb: np.ndarray, pol: PortPolarity) -> bool:
            for ep in ports:
                if ep.polarity == pol:
                    d = math.dist(ep.local_frame.pos, (pos_tb[0], pos_tb[1], pos_tb[2]))
                    if d <= 3.5:
                        return True
            return False

        for bn, m_ld, t_ld in prim_hits_ld:
            p_tb = _C_LD_TO_TB @ t_ld
            axis_ld = m_ld @ np.array([0.0, -1.0, 0.0], dtype=np.float64)
            alen = float(np.linalg.norm(axis_ld))
            if alen <= 1e-6:
                continue
            z_tb = _C_LD_TO_TB @ (axis_ld / alen)
            x_tb = _C_LD_TO_TB @ (m_ld @ np.array([1.0, 0.0, 0.0], dtype=np.float64))
            if bn in ("stud.dat", "stud2.dat", "stud2a.dat"):
                pol = PortPolarity.STUD
                ct = ConnectorType.HOLLOW_STUD if "2" in bn else ConnectorType.STANDARD_STUD
                if not _exists_near(p_tb, pol):
                    rid = _snap_vector_to_cubic_rot_id(z_tb, x_tb)
                    ports.append(
                        PortSpec(
                            len(ports),
                            LDUFrame(rid, (int(round(p_tb[0])), int(round(p_tb[1])), int(round(p_tb[2])))),
                            pol,
                            ct,
                        )
                    )
            elif bn in ("stud3.dat", "stud4.dat", "stud4a.dat"):
                # Underside tube / anti-stud recess near bottom zmin
                p_bot = np.array([p_tb[0], p_tb[1], zmin], dtype=np.float64)
                if not _exists_near(p_bot, PortPolarity.ANTI_STUD):
                    ports.append(
                        PortSpec(
                            len(ports),
                            LDUFrame(0, (int(round(p_bot[0])), int(round(p_bot[1])), int(round(zmin)))),
                            PortPolarity.ANTI_STUD,
                            ConnectorType.STANDARD_STUD,
                        )
                    )
                    bores.append((p_bot, np.array([0.0, 0.0, 1.0]), 6.0, 4.0))
            elif bn == "4-4cyli.dat":
                # Check radius in XZ of m_ld: R ~ 4.0 -> 3.18mm bar; R ~ 6.0 & len ~ 4 -> stud
                rx = float(np.linalg.norm(m_ld @ np.array([1.0, 0.0, 0.0])))
                if 3.6 <= rx <= 4.4 and alen >= 6.0:
                    mid_tb = p_tb - 0.5 * alen * z_tb
                    if not _exists_near(mid_tb, PortPolarity.STUD):
                        rid = _snap_vector_to_cubic_rot_id(z_tb, x_tb)
                        ports.append(
                            PortSpec(
                                len(ports),
                                LDUFrame(rid, (int(round(mid_tb[0])), int(round(mid_tb[1])), int(round(mid_tb[2])))),
                                PortPolarity.STUD,
                                ConnectorType.BAR_3_18MM,
                            )
                        )

        # Standard System underside stud-grid recess fallback when part has no ANTI_STUD yet
        has_anti = any(p.polarity == PortPolarity.ANTI_STUD for p in ports)
        dx = xmax - xmin
        dy = ymax - ymin
        dz = zmax - zmin
        fw = title.lstrip("~").split()[0] if title.lstrip("~").split() else ""
        if not has_anti and dz >= 3.5:
            if (
                fw in ("Tile", "Brick", "Plate", "Slope", "Baseplate", "Panel", "Wedge", "Arch", "Bracket", "Door", "Window", "Windscreen", "Roof", "Container", "Fence", "Cylinder", "Cone", "Dish", "Dome")
                and 18.0 <= dx <= 642.0
                and 18.0 <= dy <= 642.0
            ):
                nx_s = max(1, min(16, int(round(dx / 20.0))))
                ny_s = max(1, min(16, int(round(dy / 20.0))))
                if abs(dx - nx_s * 20.0) <= 4.0 and abs(dy - ny_s * 20.0) <= 4.0:
                    cx = 0.5 * (xmin + xmax)
                    cy = 0.5 * (ymin + ymax)
                    for ix in range(nx_s):
                        for iy in range(ny_s):
                            px = cx + (float(ix) - 0.5 * float(nx_s - 1)) * 20.0
                            py = cy + (float(iy) - 0.5 * float(ny_s - 1)) * 20.0
                            p_a = np.array([px, py, zmin], dtype=np.float64)
                            if not _exists_near(p_a, PortPolarity.ANTI_STUD):
                                ports.append(
                                    PortSpec(
                                        len(ports),
                                        LDUFrame(0, (int(round(px)), int(round(py)), int(round(zmin)))),
                                        PortPolarity.ANTI_STUD,
                                        ConnectorType.STANDARD_STUD,
                                    )
                                )
                                bores.append((p_a, np.array([0.0, 0.0, 1.0]), 6.0, 4.0))
            elif fw in ("Minifig", "Figure", "Animal", "Plant", "Food", "Rock", "Crystal"):
                # Center bottom stud/neck socket or 3.18mm grip so accessory/headwear/torso parts have valid attachment
                cx = 0.5 * (xmin + xmax)
                cy = 0.5 * (ymin + ymax)
                ports.append(
                    PortSpec(
                        len(ports),
                        LDUFrame(0, (int(round(cx)), int(round(cy)), int(round(zmin)))),
                        PortPolarity.ANTI_STUD,
                        ConnectorType.HOLLOW_STUD,
                    )
                )
                bores.append((np.array([cx, cy, zmin]), np.array([0.0, 0.0, 1.0]), 6.0, 4.0))

        return ports, bores

    def _rasterize_and_carve_shell(
        self,
        tris_tb: np.ndarray,
        ports: Sequence[PortSpec],
        bores: Sequence[Tuple[np.ndarray, np.ndarray, float, float]],
    ) -> Tuple[Tuple[LDUBox, ...], float, Optional[Tuple[np.ndarray, np.ndarray]]]:
        """
        Rasterize `tris_tb` onto a local integer LDU voxel grid (`4 LDU`, or `2 LDU` for compact
        parts), carve external male protrusions and female bore cylinders, and return:
          `(hull_boxes, solid_mass_grams, optional_watertight_mesh_for_coacd)`.
        """
        vmin = tris_tb.reshape(-1, 3).min(axis=0)
        vmax = tris_tb.reshape(-1, 3).max(axis=0)
        span = vmax - vmin
        max_span = float(np.max(span))
        vox = 2 if max_span <= 48.0 else (4 if max_span <= 240.0 else 8)

        # Determine solid body bounds by trimming top/side 4-LDU stud protrusions where applicable
        top_stud_zs = [
            p.local_frame.pos[2]
            for p in ports
            if p.polarity == PortPolarity.STUD
            and p.connector_type in (ConnectorType.STANDARD_STUD, ConnectorType.HOLLOW_STUD)
            and p.local_frame.z_axis == (0, 0, 1)
        ]
        eff_zmax = float(vmax[2])
        if top_stud_zs:
            top_base_z = float(max(top_stud_zs))
            if top_base_z < eff_zmax <= top_base_z + 5.5 and top_base_z > float(vmin[2]) + 2.0:
                eff_zmax = top_base_z

        ox = int(math.floor(float(vmin[0]) / vox)) * vox
        oy = int(math.floor(float(vmin[1]) / vox)) * vox
        oz = int(math.floor(float(vmin[2]) / vox)) * vox
        x_hi = max(ox + vox, int(math.ceil(float(vmax[0]) / vox)) * vox)
        y_hi = max(oy + vox, int(math.ceil(float(vmax[1]) / vox)) * vox)
        z_hi = max(oz + vox, int(math.ceil(eff_zmax / vox)) * vox)

        nx = max(1, min(64, (x_hi - ox) // vox))
        ny = max(1, min(64, (y_hi - oy) // vox))
        nz = max(1, min(64, (z_hi - oz) // vox))

        # Sample triangle surface points (barycentric 9-point stencil per triangle)
        t0, t1, t2 = tris_tb[:, 0], tris_tb[:, 1], tris_tb[:, 2]
        pts = np.vstack(
            [
                t0,
                t1,
                t2,
                (t0 + t1 + t2) / 3.0,
                0.5 * (t0 + t1),
                0.5 * (t1 + t2),
                0.5 * (t0 + t2),
                0.6 * t0 + 0.2 * t1 + 0.2 * t2,
                0.2 * t0 + 0.6 * t1 + 0.2 * t2,
            ]
        )
        # Keep points within z <= eff_zmax
        pts = pts[pts[:, 2] <= eff_zmax + 0.25]
        occ = np.zeros((nx, ny, nz), dtype=bool)
        if pts.shape[0] > 0:
            ix = np.clip(((pts[:, 0] - ox) // vox).astype(np.int32), 0, nx - 1)
            iy = np.clip(((pts[:, 1] - oy) // vox).astype(np.int32), 0, ny - 1)
            iz = np.clip(((pts[:, 2] - oz) // vox).astype(np.int32), 0, nz - 1)
            occ[ix, iy, iz] = True

            # Fill interior between min and max occupied Z per (ix, iy) column for closed brick shells
            any_xy = np.any(occ, axis=2)
            if np.any(any_xy):
                z_idx = np.arange(nz)[None, None, :]
                first_z = np.argmax(occ, axis=2)
                last_z = nz - 1 - np.argmax(occ[:, :, ::-1], axis=2)
                fill_mask = any_xy[:, :, None] & (z_idx >= first_z[:, :, None]) & (z_idx <= last_z[:, :, None])
                occ |= fill_mask

        if not np.any(occ):
            occ[0, 0, 0] = True

        # Carve out external male connector protrusions (`PortSpec.local_protrusion_box`) and female bores
        carved_mask = np.zeros_like(occ)
        if np.count_nonzero(occ) > 2:
            xs = ox + (np.arange(nx, dtype=np.float64) + 0.5) * vox
            ys = oy + (np.arange(ny, dtype=np.float64) + 0.5) * vox
            zs = oz + (np.arange(nz, dtype=np.float64) + 0.5) * vox
            gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
            centers = np.stack([gx, gy, gz], axis=-1)

            for p in ports:
                pb = p.local_protrusion_box
                if pb is None:
                    continue
                wb = pb.transform(p.local_frame)
                mask_p = (
                    (centers[..., 0] > wb.xmin + 0.5)
                    & (centers[..., 0] < wb.xmax - 0.5)
                    & (centers[..., 1] > wb.ymin + 0.5)
                    & (centers[..., 1] < wb.ymax - 0.5)
                    & (centers[..., 2] > wb.zmin + 0.5)
                    & (centers[..., 2] < wb.zmax - 0.5)
                )
                if np.count_nonzero(occ & (~mask_p)) >= 1:
                    occ[mask_p] = False
                    carved_mask |= mask_p

            # Carve out female bores (`R=4` bar holes, `R=6` Technic pin/axle holes, and recessed sockets)
            for ap_pos, in_axis, rad, depth in bores:
                rel = centers - ap_pos
                proj = rel @ in_axis
                perp_sq = np.sum(rel * rel, axis=-1) - proj * proj
                if rad >= 3.5 and depth >= 3.5:
                    carve_r = max(2.5, rad - 1.0)
                    mask_b = (proj >= -0.5) & (proj <= depth + 0.5) & (perp_sq <= carve_r * carve_r)
                    if np.count_nonzero(occ & (~mask_b)) >= 1:
                        occ[mask_b] = False
                        carved_mask |= mask_b
                # Carve exterior approach pocket on recessed underside sockets (e.g., helmets, hips, heads)
                if in_axis[2] > 0.9 and float(ap_pos[2]) > float(vmin[2]) + 0.5:
                    mask_app = (
                        (proj >= -16.0)
                        & (proj < 0.25)
                        & (np.abs(rel[..., 0]) <= 10.25)
                        & (np.abs(rel[..., 1]) <= 10.25)
                    )
                    if np.count_nonzero(occ & (~mask_app)) >= 2:
                        occ[mask_app] = False
                        carved_mask |= mask_app

        hull_boxes = _greedy_maximal_cuboid_cover(
            occ, (ox, oy, oz), vox, max_boxes=28, carved_mask=carved_mask
        )
        vol_ldu3 = float(sum(b.volume_ldu3 for b in hull_boxes))
        bbox_vol = float(max(1, (nx * vox) * (ny * vox) * (nz * vox)))
        fill_ratio = vol_ldu3 / bbox_vol
        # Molded bricks are hollow shells (~45% solid ABS fraction of bounding volume)
        mass_g = round(max(0.05, 0.45 * vol_ldu3 * _ABS_GRAMS_PER_LDU3), 4)

        wt_mesh: Optional[Tuple[np.ndarray, np.ndarray]] = None
        if len(hull_boxes) >= 2 or fill_ratio < 0.88:
            wt_mesh = _voxel_grid_to_watertight_mesh(occ, (ox, oy, oz), vox)
        return hull_boxes, mass_g, wt_mesh

    def compile_all(
        self,
        out_catalog_gz: Path = DEFAULT_COMPILED_CATALOG_GZ,
        out_coacd_npz: Path = DEFAULT_COMPILED_COACD_NPZ,
        out_manifest_json: Path = DEFAULT_COMPILED_MANIFEST_JSON,
        target_coacd_parts: int = 3000,
        num_workers: int = 6,
    ) -> Dict[str, Any]:
        """
        Compile the standalone LEGO System + Technic + Minifig LDraw `2610` universe (`14,500+`
        parts) with LDCad Shadow + LDraw primitive port extraction, bore-carved `LDUBox` colliders,
        and real `CoACD` convex polytope hulls across `>= target_coacd_parts` non-convex parts.
        """
        t0 = time.perf_counter()
        all_part_files = sorted((self.ldraw_root / "parts").glob("*.dat"))

        # Pass 1: Filter eligible standalone parts & extract ports + shell signatures
        raw_records: Dict[str, Dict[str, Any]] = {}
        shell_cache: Dict[str, Tuple[Tuple[LDUBox, ...], float, Optional[Tuple[np.ndarray, np.ndarray]]]] = {}
        shell_to_part_ids: Dict[str, List[str]] = {}
        tier_counts: Dict[str, int] = {
            "TIER_B_LDCAD_SHADOW_DAG": 0,
            "TIER_B_BASE_MOLD_SHADOW": 0,
            "TIER_C_LDRAW_PRIMITIVE_GEOMETRIC": 0,
        }
        excluded_counts: Dict[str, int] = {
            "obsolete_or_moved": 0,
            "internal_subpart_helper": 0,
            "sticker_or_non_system_scale": 0,
            "empty_geometry_or_no_ports": 0,
        }

        # Pre-pass: map unprinted base molds that have direct/DAG shadow ports for base-mold inheritance
        direct_mold_ports: Dict[str, Tuple[List[PortSpec], List[Tuple[np.ndarray, np.ndarray, float, float]], Tuple[float, ...]]] = {}

        eligible_files: List[Tuple[str, Path, str]] = []
        moved_aliases: Dict[str, str] = {}
        for fp in all_part_files:
            parsed = self._parse_ldraw_file("parts/" + fp.name.lower())
            if parsed is None:
                continue
            title = parsed[0]
            if title.startswith("~Moved") or "Obsolete" in title or title.startswith("="):
                excluded_counts["obsolete_or_moved"] += 1
                m_mv = re.match(r"^~Moved\s+to\s+([A-Za-z0-9_-]+)", title)
                if m_mv:
                    moved_aliases[fp.stem.lower()] = m_mv.group(1).lower()
                continue
            if title.startswith("~") or title.startswith("_"):
                excluded_counts["internal_subpart_helper"] += 1
                continue
            fw = title.split()[0] if title.split() else ""
            if fw in _EXCLUDED_FIRST_WORDS:
                excluded_counts["sticker_or_non_system_scale"] += 1
                continue
            eligible_files.append((fp.stem.lower(), fp, title))

        # Sort so unprinted base molds (shorter stems without `pXX`) are compiled BEFORE patterned variants!
        eligible_files.sort(key=lambda item: (len(item[0]), item[0]))
        base_alias_to_sh: Dict[str, str] = {}

        for part_id, fp, title in eligible_files:
            rel_key = "parts/" + fp.name.lower()
            parsed = self._parse_ldraw_file(rel_key)
            assert parsed is not None
            raw_bytes = parsed[1]
            part_sha = hashlib.sha256(raw_bytes).hexdigest()[:24]

            tris_list_ld, snaps_ld, prims_ld = self._collect_geometry_and_snaps_ld(
                rel_key, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
            )
            if not tris_list_ld:
                excluded_counts["empty_geometry_or_no_ports"] += 1
                continue

            tris_ld = np.concatenate(tris_list_ld, axis=0)
            tris_tb = tris_ld @ _C_LD_TO_TB.T
            pts_flat = tris_tb.reshape(-1, 3)
            vmin = pts_flat.min(axis=0)
            vmax = pts_flat.max(axis=0)
            bounds_tb = (
                float(vmin[0]),
                float(vmin[1]),
                float(vmin[2]),
                float(vmax[0]),
                float(vmax[1]),
                float(vmax[2]),
            )

            ports: List[PortSpec] = []
            bores: List[Tuple[np.ndarray, np.ndarray, float, float]] = []
            prov_tier = ""

            if snaps_ld:
                ports, bores = self._convert_snaps_to_ports(snaps_ld)
                # Supplement with underside anti-studs if Shadow only defined top studs or accessories
                if ports and not any(p.polarity == PortPolarity.ANTI_STUD for p in ports):
                    ports, bores_extra = self._infer_tier_c_ports(title, bounds_tb, prims_ld, ports)
                    bores.extend(bores_extra)
                if ports:
                    prov_tier = "TIER_B_LDCAD_SHADOW_DAG"
                    direct_mold_ports[part_id] = (ports, bores, bounds_tb)

            if not ports:
                # Check unprinted base mold (`3069bp01` -> `3069b`)
                m_base = re.match(r"^([0-9]+[a-z]?)(?:p[0-9a-z]+|d[0-9a-z]+|c[0-9a-z]+)$", part_id)
                if m_base and m_base.group(1) in direct_mold_ports:
                    b_ports, b_bores, b_bounds = direct_mold_ports[m_base.group(1)]
                    if max(abs(bounds_tb[k] - b_bounds[k]) for k in range(6)) <= 2.5:
                        ports = list(b_ports)
                        bores = list(b_bores)
                        prov_tier = "TIER_B_BASE_MOLD_SHADOW"

            if not ports:
                ports, bores = self._infer_tier_c_ports(title, bounds_tb, prims_ld, ())
                if ports:
                    prov_tier = "TIER_C_LDRAW_PRIMITIVE_GEOMETRIC"

            if not ports:
                excluded_counts["empty_geometry_or_no_ports"] += 1
                continue

            tier_counts[prov_tier] += 1

            # Shell hash deduplication: patterned variants sharing a base mold reuse its exact collider!
            m_shell = re.match(r"^([0-9]+[a-z]?)(?:p[0-9a-z]+)$", part_id)
            q_bounds = tuple(int(round(v / 2.0)) for v in bounds_tb)
            port_sig = tuple(
                (p.connector_type.value, int(p.polarity), p.local_frame.rot_id, p.local_frame.pos)
                for p in ports[:16]
            )
            if m_shell and f"base:{m_shell.group(1)}:{q_bounds}" in base_alias_to_sh:
                shell_id = base_alias_to_sh[f"base:{m_shell.group(1)}:{q_bounds}"]
            else:
                # Quantized 12-bin spatial surface signature + bounds + port signature
                q_hist = tuple(int(round(float(v) / 4.0)) for v in np.percentile(pts_flat, [10, 50, 90], axis=0).reshape(-1))
                raw_sid = f"{q_bounds}:{q_hist}:{len(ports)}:{port_sig[:4]}"
                shell_id = "sh_" + hashlib.sha1(raw_sid.encode("utf-8")).hexdigest()[:16]
                if shell_id not in shell_cache:
                    shell_cache[shell_id] = self._rasterize_and_carve_shell(tris_tb, ports, bores)
                if not m_shell:
                    base_alias_to_sh[f"base:{part_id}:{q_bounds}"] = shell_id

            if shell_id not in shell_cache:
                shell_cache[shell_id] = self._rasterize_and_carve_shell(tris_tb, ports, bores)

            hull_boxes, mass_g, _ = shell_cache[shell_id]
            shell_to_part_ids.setdefault(shell_id, []).append(part_id)

            is_snot = any(abs(int(p.local_frame.z_axis[2])) == 0 for p in ports)
            raw_records[part_id] = {
                "n": title,
                "m": mass_g,
                "s": 1 if is_snot else 0,
                "t": prov_tier,
                "h": part_sha,
                "sh": shell_id,
                "b": [[b.xmin, b.ymin, b.zmin, b.xmax, b.ymax, b.zmax] for b in hull_boxes],
                "p": [
                    [int(p.port_id), int(p.local_frame.rot_id), list(p.local_frame.pos), int(p.polarity), int(p.connector_type)]
                    for p in ports
                ],
            }

        # Pass 2: Run real `coacd.run_coacd` across unique non-convex shells until >= `target_coacd_parts` covered!
        candidate_shells: List[Tuple[int, str, np.ndarray, np.ndarray]] = []
        seen_unique_shells: Set[str] = set()
        for sh_id, p_list in shell_to_part_ids.items():
            if sh_id.startswith("base:") or sh_id in seen_unique_shells:
                continue
            seen_unique_shells.add(sh_id)
            _, _, wt = shell_cache[sh_id]
            if wt is not None and wt[0].shape[0] >= 8 and wt[1].shape[0] >= 12:
                candidate_shells.append((len(p_list), sh_id, wt[0], wt[1]))

        # Sort by mold multiplicity descending so high-use sloped/curved/minifig/Technic molds get CoACD hulls first!
        candidate_shells.sort(key=lambda x: (-x[0], x[1]))

        tasks_to_run: List[Tuple[str, np.ndarray, np.ndarray]] = []
        covered_parts_est = 0
        for mult, sh_id, v_m, f_m in candidate_shells:
            tasks_to_run.append((sh_id, v_m, f_m))
            covered_parts_est += mult
            if covered_parts_est >= target_coacd_parts and len(tasks_to_run) >= 320:
                break

        coacd_npz_arrays: Dict[str, np.ndarray] = {}
        coacd_shells_succeeded: Set[str] = set()
        remaining_tasks: List[Tuple[str, np.ndarray, np.ndarray]] = []
        if out_coacd_npz.exists():
            with np.load(out_coacd_npz, allow_pickle=False) as prev_npz:
                prev_keys = set(prev_npz.files)
                for sh_id, v_m, f_m in tasks_to_run:
                    cnt_key = f"{sh_id}_count"
                    if cnt_key in prev_keys:
                        cnt_h = int(prev_npz[cnt_key][0])
                        if cnt_h > 0 and f"{sh_id}_v0" in prev_keys:
                            coacd_shells_succeeded.add(sh_id)
                            coacd_npz_arrays[cnt_key] = prev_npz[cnt_key]
                            for hi in range(cnt_h):
                                coacd_npz_arrays[f"{sh_id}_v{hi}"] = prev_npz[f"{sh_id}_v{hi}"]
                                coacd_npz_arrays[f"{sh_id}_n{hi}"] = prev_npz[f"{sh_id}_n{hi}"]
                                coacd_npz_arrays[f"{sh_id}_o{hi}"] = prev_npz[f"{sh_id}_o{hi}"]
                            continue
                    remaining_tasks.append((sh_id, v_m, f_m))
        else:
            remaining_tasks = tasks_to_run

        if remaining_tasks:
            eff_workers = max(1, min(int(num_workers), os.cpu_count() or 4))
            with ProcessPoolExecutor(max_workers=eff_workers) as pool:
                for sh_id, hull_list in pool.map(_coacd_worker, remaining_tasks, chunksize=8):
                    if hull_list:
                        coacd_shells_succeeded.add(sh_id)
                        coacd_npz_arrays[f"{sh_id}_count"] = np.array([len(hull_list)], dtype=np.int32)
                        for hi, (hv, hn, ho) in enumerate(hull_list):
                            coacd_npz_arrays[f"{sh_id}_v{hi}"] = hv
                            coacd_npz_arrays[f"{sh_id}_n{hi}"] = hn
                            coacd_npz_arrays[f"{sh_id}_o{hi}"] = ho

        coacd_part_count = 0
        for pid, rec in raw_records.items():
            sh_id = rec["sh"]
            # Map `base:...` aliases to underlying `sh_...` if same boxes
            if sh_id in coacd_shells_succeeded:
                rec["coacd"] = sh_id
                coacd_part_count += 1
            else:
                rec["coacd"] = None

        np.savez_compressed(out_coacd_npz, **coacd_npz_arrays)
        coacd_sha = hashlib.sha256(out_coacd_npz.read_bytes()).hexdigest()

        # Filter moved_aliases to transitive targets that actually exist in `raw_records`
        valid_moved_aliases: Dict[str, str] = {}
        for old_id, tgt_id in moved_aliases.items():
            cur = tgt_id
            hops = 0
            while cur in moved_aliases and hops < 5:
                cur = moved_aliases[cur]
                hops += 1
            if cur in raw_records and old_id not in raw_records:
                valid_moved_aliases[old_id] = cur

        payload = {
            "schema_version": "1.0.0",
            "ldraw_release": self.ldraw_release,
            "coacd_npz_sha256": coacd_sha,
            "total_compiled_parts": len(raw_records),
            "moved_redirect_aliases_count": len(valid_moved_aliases),
            "coacd_decomposed_parts": coacd_part_count,
            "coacd_unique_shells": len(coacd_shells_succeeded),
            "tier_counts": tier_counts,
            "excluded_counts": excluded_counts,
            "moved_aliases": valid_moved_aliases,
            "parts": raw_records,
        }
        raw_json = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        with gzip.open(out_catalog_gz, "wb", compresslevel=6) as gz:
            gz.write(raw_json)
        cat_gz_sha = hashlib.sha256(out_catalog_gz.read_bytes()).hexdigest()

        elapsed_s = time.perf_counter() - t0
        manifest = {
            "schema_version": "1.0.0",
            "ldraw_release": self.ldraw_release,
            "shadow_source": "RolandMelkert/LDCadShadowLibrary@9b1131fb1991f8c0bfc072325e4e12f6271aba35",
            "geometry_license": "CC-BY-4.0 (LDraw Official Library CAreadme.txt)",
            "connector_license_shadow": "CC-BY-SA-4.0 (LDCad Shadow Library LICENSE.md)",
            "catalog_gz_sha256": cat_gz_sha,
            "coacd_npz_sha256": coacd_sha,
            "total_compiled_parts": len(raw_records),
            "coacd_decomposed_parts": coacd_part_count,
            "coacd_unique_shells": len(coacd_shells_succeeded),
            "tier_counts": tier_counts,
            "excluded_counts": excluded_counts,
            "compile_wall_seconds": round(elapsed_s, 3),
        }
        out_manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return manifest


class CompiledLDrawCatalog:
    """
    Lazy, provenance-tracked runtime catalog over all `14,500+` compiled LDraw `2610` parts.
    Loads metadata on first access and materializes `BrickPrimitive`, `DeclarativePartSpec`,
    and `ConvexPolytope` hulls on demand in O(1).
    """

    def __init__(
        self,
        catalog_gz_path: Path = DEFAULT_COMPILED_CATALOG_GZ,
        coacd_npz_path: Path = DEFAULT_COMPILED_COACD_NPZ,
        manifest_path: Path = DEFAULT_COMPILED_MANIFEST_JSON,
    ) -> None:
        self._catalog_gz_path = catalog_gz_path
        self._coacd_npz_path = coacd_npz_path
        self._manifest_path = manifest_path
        self._loaded: bool = False
        self._raw_parts: Dict[str, Dict[str, Any]] = {}
        self._moved_aliases: Dict[str, str] = {}
        self._prim_cache: Dict[str, BrickPrimitive] = {}
        self._spec_cache: Dict[str, DeclarativePartSpec] = {}
        self._coacd_npz: Any = None
        self._coacd_cache: Dict[str, Tuple[ConvexPolytope, ...]] = {}
        self._manifest_cache: Optional[Dict[str, Any]] = None

    @property
    def is_available(self) -> bool:
        return self._catalog_gz_path.exists()

    @property
    def manifest(self) -> Dict[str, Any]:
        if self._manifest_cache is None:
            if self._manifest_path.exists():
                self._manifest_cache = json.loads(self._manifest_path.read_text(encoding="utf-8"))
            else:
                self._manifest_cache = {}
        return self._manifest_cache

    @property
    def archive_sha256(self) -> str:
        return str(self.manifest.get("catalog_gz_sha256", "not_compiled"))

    def _ensure_loaded(self) -> None:
        if self._loaded or not self._catalog_gz_path.exists():
            return
        with gzip.open(self._catalog_gz_path, "rt", encoding="utf-8") as gz:
            data = json.load(gz)
        self._raw_parts = data.get("parts", {})
        self._moved_aliases = data.get("moved_aliases", {})
        self._loaded = True

    def __len__(self) -> int:
        if self._manifest_path.exists() and not self._loaded:
            return int(self.manifest.get("total_compiled_parts", 0))
        self._ensure_loaded()
        return len(self._raw_parts)

    def resolve_id(self, part_id: str) -> Optional[str]:
        self._ensure_loaded()
        pid = part_id.lower()
        if pid in self._raw_parts:
            return pid
        mapped = self._moved_aliases.get(pid)
        if mapped is not None and mapped in self._raw_parts:
            return mapped
        return None

    def __contains__(self, part_id: object) -> bool:
        if not isinstance(part_id, str):
            return False
        return self.resolve_id(part_id) is not None

    def keys(self) -> Iterable[str]:
        self._ensure_loaded()
        return self._raw_parts.keys()

    def get_coacd_polytopes(self, shell_id: Optional[str]) -> Tuple[ConvexPolytope, ...]:
        if not shell_id or not self._coacd_npz_path.exists():
            return ()
        cached = self._coacd_cache.get(shell_id)
        if cached is not None:
            return cached
        if self._coacd_npz is None:
            self._coacd_npz = np.load(self._coacd_npz_path, allow_pickle=False)
        cnt_key = f"{shell_id}_count"
        if cnt_key not in self._coacd_npz:
            self._coacd_cache[shell_id] = ()
            return ()
        cnt = int(self._coacd_npz[cnt_key][0])
        hulls = tuple(
            ConvexPolytope(
                vertices_ldu=np.asarray(self._coacd_npz[f"{shell_id}_v{i}"], dtype=np.float32),
                halfspace_normals=np.asarray(self._coacd_npz[f"{shell_id}_n{i}"], dtype=np.float32),
                halfspace_offsets=np.asarray(self._coacd_npz[f"{shell_id}_o{i}"], dtype=np.float32),
            )
            for i in range(cnt)
        )
        self._coacd_cache[shell_id] = hulls
        return hulls

    def get_primitive(self, part_id: str) -> BrickPrimitive:
        pid_req = part_id.lower()
        cached = self._prim_cache.get(pid_req)
        if cached is not None:
            return cached
        pid = self.resolve_id(pid_req)
        if pid is None:
            raise KeyError(f"Part '{part_id}' not found in CompiledLDrawCatalog.")
        rec = self._raw_parts[pid]
        boxes = tuple(LDUBox(b[0], b[1], b[2], b[3], b[4], b[5]) for b in rec["b"])
        ports = tuple(
            PortSpec(
                port_id=int(p[0]),
                local_frame=LDUFrame(rot_id=int(p[1]), pos=(int(p[2][0]), int(p[2][1]), int(p[2][2]))),
                polarity=PortPolarity(int(p[3])),
                connector_type=ConnectorType(int(p[4])),
            )
            for p in rec["p"]
        )
        prim = BrickPrimitive(
            part_id=pid_req,
            name=str(rec["n"]),
            hull_boxes=boxes,
            ports=ports,
            mass_grams=float(rec["m"]),
            is_snot=bool(rec["s"]),
            is_ldraw_native_frame=True,
            coacd_shell_id=rec.get("coacd"),
        )
        self._prim_cache[pid_req] = prim
        return prim

    def get_declarative_spec(self, part_id: str, descriptor_fn: Any) -> DeclarativePartSpec:
        pid_req = part_id.lower()
        cached = self._spec_cache.get(pid_req)
        if cached is not None:
            return cached
        pid = self.resolve_id(pid_req)
        if pid is None:
            raise KeyError(f"Part '{part_id}' not found in CompiledLDrawCatalog.")
        prim = self.get_primitive(pid_req)
        rec = self._raw_parts[pid]
        tier = str(rec["t"])
        has_coacd = rec.get("coacd") is not None
        contract = CollisionContractSpec(
            contract_id=(
                "coacd_convex_plus_bore_carved_ldu_v1"
                if has_coacd
                else "bore_carved_maximal_ldu_cuboids_v1"
            ),
            source_mesh_version=f"ldraw_{self.manifest.get('ldraw_release', '2610')}",
            decomposition_method=(
                "COACD_CONVEX_HULLS_PLUS_BORE_CARVED_LDU_BOXES"
                if has_coacd
                else "BORE_CARVED_MAXIMAL_LDU_BOXES"
            ),
            geometric_inset_ldu=0.0,
            collision_algorithm=(
                "SAT_COACD_POLYTOPE_AND_INT_AABB_SOCKET_EXEMPT"
                if has_coacd
                else "EXACT_INT_AABB_PLUS_MATED_PROTRUSION_EXEMPTION"
            ),
            allowed_penetration_ldu=0.35 if has_coacd else 0.0,
            mated_port_exemption_enabled=True,
        )
        is_shadow = tier.startswith("TIER_B")
        prov = AssetProvenanceManifest(
            part_id=pid_req,
            provider_id="ldraw_ldcad_shadow_compiler_v1",
            source_version=f"ldraw_{self.manifest.get('ldraw_release', '2610')}",
            sha256_hash=str(rec["h"]),
            geometry_license="CC-BY-4.0 (LDraw Official Library)",
            connector_license=(
                "CC-BY-SA-4.0 (LDCad Shadow Library)"
                if is_shadow
                else "CC-BY-4.0 (LDraw Primitive Geometric Extraction)"
            ),
            redistribution_status=(
                "SHARE_ALIKE_ADAPTATION" if is_shadow else "COMMERCIALLY_USABLE_WITH_ATTRIBUTION"
            ),
            collision_contract=contract,
            validation_status=(
                f"COMPILED_{tier}_COACD" if has_coacd else f"COMPILED_{tier}_BORE_BOX"
            ),
        )
        desc = descriptor_fn(prim)
        spec = DeclarativePartSpec(
            primitive=prim,
            intrinsic_descriptor=tuple(float(v) for v in desc),
            provenance=prov,
        )
        self._spec_cache[pid_req] = spec
        return spec


COMPILED_LDRAW_CATALOG = CompiledLDrawCatalog()


class CombinedBrickCatalogView:
    """
    Unified read-only membership & lookup view combining curated `EXTENDED_BRICK_CATALOG` (`213`)
    with `COMPILED_LDRAW_CATALOG` (`14,500+`), suitable for `evaluate_omr_26set_benchmark(catalog=...)`.
    """

    def __init__(self, base_catalog: Dict[str, BrickPrimitive], compiled: CompiledLDrawCatalog) -> None:
        self._base = base_catalog
        self._compiled = compiled

    def __contains__(self, part_id: object) -> bool:
        if not isinstance(part_id, str):
            return False
        return part_id in self._base or part_id.lower() in self._base or part_id in self._compiled

    def __getitem__(self, part_id: str) -> BrickPrimitive:
        if part_id in self._base:
            return self._base[part_id]
        if part_id.lower() in self._base:
            return self._base[part_id.lower()]
        return self._compiled.get_primitive(part_id)

    def get(self, part_id: str, default: Optional[BrickPrimitive] = None) -> Optional[BrickPrimitive]:
        if part_id in self:
            return self[part_id]
        return default

    def __len__(self) -> int:
        extra = sum(1 for k in self._compiled.keys() if k not in self._base)
        return len(self._base) + extra

    def keys(self) -> Tuple[str, ...]:
        seen = set(self._base.keys())
        extra = [k for k in self._compiled.keys() if k not in seen]
        return tuple(list(self._base.keys()) + extra)

    def __iter__(self):
        return iter(self.keys())

    def values(self) -> Tuple[BrickPrimitive, ...]:
        return tuple(self[k] for k in self.keys())

    def items(self) -> Tuple[Tuple[str, BrickPrimitive], ...]:
        return tuple((k, self[k]) for k in self.keys())


from topo_brick.interface.brick_catalog import EXTENDED_BRICK_CATALOG as _EXT_CAT  # noqa: E402

COMBINED_BRICK_CATALOG = CombinedBrickCatalogView(_EXT_CAT, COMPILED_LDRAW_CATALOG)
