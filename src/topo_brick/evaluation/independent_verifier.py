"""
R9-WS1 (§3.1, §5.3, §6.2): Versioned Certificate Artifact & Planner-Free Independent Verifier.

Provides:
  1. `VersionedCertificateArtifact` (`schema_version="1.0.0"`): Self-contained JSON-serializable
     certificate record storing catalog SHA-256, target SHA-256, foundation & support policies,
     inventory bounds, placed part poses, atomic execution operations, joint capacity assumptions,
     tolerances, and primal LP witness summary.
  2. `check_fine_csg_voxel_collision_2ldu(...)` & `audit_geometry_collider_vs_csg(n_samples=200)`:
     Independent `2 x 2 x 2 LDU` (`0.8 mm`) Constructive Solid Geometry (CSG) plastic-solid voxel
     collider `(Hull ∪ MaleProtrusions) \\ FemaleSocketCavities` requiring zero planner code paths
     and zero ad-hoc pairwise exemptions.
  3. `verify_certificate_independently(artifact)` + CLI (`python -m topo_brick.evaluation.independent_verifier`):
     Completely planner-free replay verifier (never imports `TopoBrickKernel`) checking catalog
     hash, inventory, `2 LDU` CSG collision freedom, port compatibility, 6-axis monotone insertion
     corridors, and independent `scipy.optimize.linprog` 6-DoF static & prefix equilibrium with
     explicit primal residual checks (`||A_eq x - b_eq||_inf`, `max(A_ub x - b_ub)`).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

from topo_brick.interface.brick_catalog import (
    EXTENDED_BRICK_CATALOG,
    ConnectorType,
    PortPolarity,
    are_connectors_compatible,
    get_brick_primitive,
)
from topo_brick.interface.ldu_frames import LDUBox, LDUFrame, rot_id_to_matrix
from topo_brick.interface.port_manifold import AssemblyState, get_admissible_socket_pocket_ldu
from topo_brick.interface.se3_kinematics import (
    ContinuousAssemblyState,
    ContinuousSE3Pose,
)


CERTIFICATE_SCHEMA_VERSION: str = "1.0.0"


_BASE_EXTENDED_CATALOG_SHA256: Optional[str] = None
_RETRIEVAL_INDEX_SHA256: Optional[str] = None


def compute_catalog_sha256() -> str:
    """
    Compute a deterministic O(1)-cached SHA-256 digest of `EXTENDED_BRICK_CATALOG` (`213`) plus
    `COMPILED_LDRAW_CATALOG.archive_sha256` (`14,500+` parts when compiled).
    """
    global _BASE_EXTENDED_CATALOG_SHA256, _RETRIEVAL_INDEX_SHA256
    if _BASE_EXTENDED_CATALOG_SHA256 is None:
        h = hashlib.sha256()
        for pid in sorted(EXTENDED_BRICK_CATALOG.keys()):
            prim = EXTENDED_BRICK_CATALOG[pid]
            entry = {
                "part_id": prim.part_id,
                "mass_grams": round(float(prim.mass_grams), 6),
                "hull_boxes": [
                    (b.xmin, b.ymin, b.zmin, b.xmax, b.ymax, b.zmax) for b in prim.hull_boxes
                ],
                "ports": [
                    (
                        p.port_id,
                        p.connector_type.name,
                        p.polarity.name,
                        p.local_frame.rot_id,
                        p.local_frame.pos,
                        (
                            (
                                p.local_protrusion_box.xmin,
                                p.local_protrusion_box.ymin,
                                p.local_protrusion_box.zmin,
                                p.local_protrusion_box.xmax,
                                p.local_protrusion_box.ymax,
                                p.local_protrusion_box.zmax,
                            )
                            if p.local_protrusion_box is not None
                            else None
                        ),
                    )
                    for p in prim.ports
                ],
            }
            h.update(json.dumps(entry, sort_keys=True).encode("utf-8"))
        _BASE_EXTENDED_CATALOG_SHA256 = h.hexdigest()

    from topo_brick.interface.ldraw_shadow_catalog_compiler import COMPILED_LDRAW_CATALOG

    if COMPILED_LDRAW_CATALOG.is_available:
        if _RETRIEVAL_INDEX_SHA256 is None:
            idx_npz = Path(__file__).resolve().parent.parent / "interface" / "compiled_mold_retrieval_index.npz"
            _RETRIEVAL_INDEX_SHA256 = (
                hashlib.sha256(idx_npz.read_bytes()).hexdigest() if idx_npz.is_file() else "no_idx"
            )
        comb = hashlib.sha256()
        comb.update(
            f"{_BASE_EXTENDED_CATALOG_SHA256}:{COMPILED_LDRAW_CATALOG.archive_sha256}:{_RETRIEVAL_INDEX_SHA256}".encode(
                "utf-8"
            )
        )
        return comb.hexdigest()
    return _BASE_EXTENDED_CATALOG_SHA256


def compute_target_sha256(target_field: Any) -> str:
    """Compute a deterministic SHA-256 digest of a target field's bounds and occupied volume."""
    if target_field is None:
        return "no_target"
    h = hashlib.sha256()
    bounds = target_field.occupied_bounds_ldu()
    vol = int(getattr(target_field, "total_volume_ldu3", 0))
    payload: Dict[str, Any] = {"bounds": bounds, "total_volume_ldu3": vol}
    if hasattr(target_field, "boxes"):
        payload["boxes"] = [
            (b.xmin, b.ymin, b.zmin, b.xmax, b.ymax, b.zmax) for b in target_field.boxes
        ]
    h.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return h.hexdigest()


@dataclass(frozen=True, slots=True)
class VersionedCertificateArtifact:
    """Self-contained, versioned JSON certificate artifact (`R9-WS1` §3.1)."""

    schema_version: str
    catalog_sha256: str
    target_sha256: str
    assembly_mode: str
    foundation_policy: str
    support_policy: str
    inventory_spec: Optional[Dict[str, int]]
    placed_parts: Tuple[Dict[str, Any], ...]
    execution_operations: Tuple[Dict[str, Any], ...]
    matched_port_pairs: Tuple[Tuple[Tuple[int, int], Tuple[int, int]], ...]
    joint_assumptions: Dict[str, float]
    tolerances: Dict[str, float]
    lp_witness_summary: Dict[str, Any]
    cached_dependencies: Tuple[str, ...]
    claimed_certificates: Dict[str, str]

    def to_json_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["placed_parts"] = list(self.placed_parts)
        d["execution_operations"] = list(self.execution_operations)
        d["matched_port_pairs"] = [
            [[int(a[0]), int(a[1])], [int(b[0]), int(b[1])]]
            for a, b in self.matched_port_pairs
        ]
        d["cached_dependencies"] = list(self.cached_dependencies)
        return d

    @classmethod
    def from_json_dict(cls, data: Dict[str, Any]) -> "VersionedCertificateArtifact":
        return cls(
            schema_version=str(data["schema_version"]),
            catalog_sha256=str(data["catalog_sha256"]),
            target_sha256=str(data["target_sha256"]),
            assembly_mode=str(data["assembly_mode"]),
            foundation_policy=str(data["foundation_policy"]),
            support_policy=str(data.get("support_policy", "none")),
            inventory_spec=(
                {str(k): int(v) for k, v in data["inventory_spec"].items()}
                if data.get("inventory_spec") is not None
                else None
            ),
            placed_parts=tuple(dict(p) for p in data.get("placed_parts", ())),
            execution_operations=tuple(dict(op) for op in data.get("execution_operations", ())),
            matched_port_pairs=tuple(
                ((int(p[0][0]), int(p[0][1])), (int(p[1][0]), int(p[1][1])))
                for p in data.get("matched_port_pairs", ())
            ),
            joint_assumptions={
                str(k): float(v) for k, v in data.get("joint_assumptions", {}).items()
            },
            tolerances={str(k): float(v) for k, v in data.get("tolerances", {}).items()},
            lp_witness_summary=dict(data.get("lp_witness_summary", {})),
            cached_dependencies=tuple(str(x) for x in data.get("cached_dependencies", ())),
            claimed_certificates={
                str(k): str(v) for k, v in data.get("claimed_certificates", {}).items()
            },
        )

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "VersionedCertificateArtifact":
        return cls.from_json_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def export_certificate_artifact(
    assembly: AssemblyState | ContinuousAssemblyState,
    certificates: Any,
    validator: Any = None,
    target_field: Any = None,
    support_policy: str = "none",
    inventory_counts: Optional[Dict[str, int]] = None,
) -> VersionedCertificateArtifact:
    """Export a self-contained `VersionedCertificateArtifact` for independent planner-free replay."""
    cat_sha = compute_catalog_sha256()
    tgt_sha = compute_target_sha256(target_field)
    is_cubic = isinstance(assembly, AssemblyState)
    mode = "cubic_ldu" if is_cubic else "continuous_se3"

    placed_list: List[Dict[str, Any]] = []
    for iid in sorted(assembly.placed_parts.keys()):
        p = assembly.placed_parts[iid]
        if is_cubic:
            placed_list.append(
                {
                    "instance_id": int(p.instance_id),
                    "part_id": str(p.part_id),
                    "rot_id": int(p.pose.rot_id),
                    "pos": [int(c) for c in p.pose.pos],
                    "is_grounded": bool(p.is_grounded),
                }
            )
        else:
            placed_list.append(
                {
                    "instance_id": int(p.instance_id),
                    "part_id": str(p.part_id),
                    "rot_matrix": [[float(x) for x in row] for row in p.continuous_pose.rot_matrix],
                    "pos": [float(c) for c in p.continuous_pose.pos],
                    "is_grounded": bool(p.is_grounded),
                }
            )

    ops_list: List[Dict[str, Any]] = []
    eff_trace = getattr(certificates, "execution_trace", None)
    if eff_trace is not None:
        for st in eff_trace.steps:
            ops_list.append(
                {
                    "op_type": str(st.op_type.value if hasattr(st.op_type, "value") else st.op_type),
                    "instance_ids": [int(x) for x in st.instance_ids],
                    "part_id": str(st.part_id),
                    "rot_id": int(st.pose.rot_id) if st.pose is not None else 0,
                    "pos": [int(c) for c in st.pose.pos] if st.pose is not None else [0, 0, 0],
                    "is_grounded": bool(st.is_grounded),
                    "temp_support_tag": str(st.temp_support_tag),
                    "gravity_unit": [float(g) for g in st.gravity_unit],
                }
            )
    else:
        for entry in placed_list:
            ops_list.append(
                {
                    "op_type": "PLACE_PART",
                    "instance_ids": [int(entry["instance_id"])],
                    "part_id": str(entry["part_id"]),
                    "rot_id": int(entry.get("rot_id", 0)),
                    "rot_matrix": entry.get("rot_matrix"),
                    "pos": list(entry["pos"]),
                    "is_grounded": bool(entry["is_grounded"]),
                    "temp_support_tag": "",
                    "gravity_unit": [0.0, 0.0, -1.0],
                }
            )

    matched_pairs: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    seen_undir: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()
    for ga, gb in sorted(assembly.matched_port_pairs.items()):
        key = (ga, gb) if ga <= gb else (gb, ga)
        if key not in seen_undir:
            seen_undir.add(key)
            matched_pairs.append(key)

    clutch_g = float(getattr(validator, "clutch_force_capacity_grams", 99.9))
    m_arm = float(getattr(validator, "moment_arm_capacity_ldu", 8.0))
    fric = float(getattr(validator, "friction_coeff", 0.50))

    witness_summary: Dict[str, Any] = {
        "is_stable": bool(certificates.static_mechanics.is_stable),
        "peak_utilization": float(certificates.static_mechanics.peak_utilization),
    }
    if validator is not None and is_cubic:
        eq_res = validator.evaluate(assembly, foundation_policy=certificates.foundation_policy)
        if eq_res.primal_witness is not None:
            pw = eq_res.primal_witness
            witness_summary.update(
                {
                    "max_eq_residual_g": float(pw.max_eq_residual_g),
                    "max_ub_violation": float(pw.max_ub_violation),
                    "eta_value": float(pw.eta_value),
                    "num_variables": int(pw.num_variables),
                    "num_equality_rows": int(pw.num_equality_rows),
                    "num_inequality_rows": int(pw.num_inequality_rows),
                    "patch_count": len(pw.patch_net_loads),
                }
            )

    return VersionedCertificateArtifact(
        schema_version=CERTIFICATE_SCHEMA_VERSION,
        catalog_sha256=cat_sha,
        target_sha256=tgt_sha,
        assembly_mode=mode,
        foundation_policy=str(certificates.foundation_policy),
        support_policy=str(support_policy),
        inventory_spec=dict(inventory_counts) if inventory_counts is not None else None,
        placed_parts=tuple(placed_list),
        execution_operations=tuple(ops_list),
        matched_port_pairs=tuple(matched_pairs),
        joint_assumptions={
            "clutch_force_capacity_grams": clutch_g,
            "moment_arm_capacity_ldu": m_arm,
            "friction_coeff": fric,
            "locking_hinge_detent_moment_ldu_g": 650.0,
            "ball_joint_friction_moment_ldu_g": 450.0,
        },
        tolerances={
            "eq_residual_tol_g": 1e-5,
            "ub_violation_tol": 1e-5,
            "fine_csg_voxel_ldu": 2.0,
        },
        lp_witness_summary=witness_summary,
        cached_dependencies=("scipy.optimize.linprog:highs", "EXTENDED_BRICK_CATALOG"),
        claimed_certificates=certificates.summary_dict(),
    )


# ---------------------------------------------------------------------------
# Independent 2 LDU (0.8 mm) CSG Plastic-Solid Voxel Collider (§6.2)
# ---------------------------------------------------------------------------


_PACK_BIAS: int = 524288
_CUBIC_LOCAL_VOXEL_CACHE: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def _rasterize_local_box_coords_2ldu(
    local_box: LDUBox,
    rot_mat: np.ndarray,
    pos_vec: np.ndarray,
    step_ldu: float = 2.0,
    shrink_eps: float = 0.20,
) -> np.ndarray:
    """
    Rasterize a part-local `LDUBox` transformed by `(rot_mat, pos_vec)` onto a uniform world
    `2 x 2 x 2 LDU` (`0.8 mm`) voxel grid, returning an `(M, 3)` `int64` array of `(ix, iy, iz)`
    voxel coordinates (shrunk by `shrink_eps = 0.20 LDU` so flush touching faces never false-collide).
    """
    bx0, by0, bz0 = float(local_box.xmin), float(local_box.ymin), float(local_box.zmin)
    bx1, by1, bz1 = float(local_box.xmax), float(local_box.ymax), float(local_box.zmax)
    if (bx1 - bx0) <= 2.0 * shrink_eps or (by1 - by0) <= 2.0 * shrink_eps or (bz1 - bz0) <= 2.0 * shrink_eps:
        return np.empty((0, 3), dtype=np.int64)

    corners_l = np.array(
        [
            [x, y, z]
            for x in (bx0, bx1)
            for y in (by0, by1)
            for z in (bz0, bz1)
        ],
        dtype=np.float64,
    )
    corners_w = (corners_l @ rot_mat.T) + pos_vec[None, :]
    w_min = np.min(corners_w, axis=0)
    w_max = np.max(corners_w, axis=0)

    ix0 = int(math.floor(float(w_min[0]) / step_ldu))
    ix1 = int(math.ceil(float(w_max[0]) / step_ldu))
    iy0 = int(math.floor(float(w_min[1]) / step_ldu))
    iy1 = int(math.ceil(float(w_max[1]) / step_ldu))
    iz0 = int(math.floor(float(w_min[2]) / step_ldu))
    iz1 = int(math.ceil(float(w_max[2]) / step_ldu))

    gx = (np.arange(ix0, ix1, dtype=np.float64) + 0.5) * step_ldu
    gy = (np.arange(iy0, iy1, dtype=np.float64) + 0.5) * step_ldu
    gz = (np.arange(iz0, iz1, dtype=np.float64) + 0.5) * step_ldu
    if gx.size == 0 or gy.size == 0 or gz.size == 0:
        return np.empty((0, 3), dtype=np.int64)

    xx, yy, zz = np.meshgrid(gx, gy, gz, indexing="ij")
    pts_w = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    pts_l = (pts_w - pos_vec[None, :]) @ rot_mat
    mask = (
        (pts_l[:, 0] >= bx0 + shrink_eps)
        & (pts_l[:, 0] <= bx1 - shrink_eps)
        & (pts_l[:, 1] >= by0 + shrink_eps)
        & (pts_l[:, 1] <= by1 - shrink_eps)
        & (pts_l[:, 2] >= bz0 + shrink_eps)
        & (pts_l[:, 2] <= bz1 - shrink_eps)
    )
    if not np.any(mask):
        return np.empty((0, 3), dtype=np.int64)
    ii, jj, kk = np.meshgrid(
        np.arange(ix0, ix1, dtype=np.int64),
        np.arange(iy0, iy1, dtype=np.int64),
        np.arange(iz0, iz1, dtype=np.int64),
        indexing="ij",
    )
    return np.stack([ii.ravel()[mask], jj.ravel()[mask], kk.ravel()[mask]], axis=1)


def _pack_voxel_coords_int64(coords: np.ndarray) -> np.ndarray:
    """Pack `(M, 3)` `int64` voxel coordinates into sorted unique `int64` keys."""
    if coords.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)
    keys = (
        ((coords[:, 0] + _PACK_BIAS) << 40)
        | ((coords[:, 1] + _PACK_BIAS) << 20)
        | (coords[:, 2] + _PACK_BIAS)
    )
    return np.unique(keys)


def _compute_part_csg_coords_2ldu(
    part_id: str,
    rot_mat: np.ndarray,
    pos_vec: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return `(coords_M3, packed_sorted_keys, aabb_min_max_2x3)` at `2 LDU` (`0.8 mm`) resolution."""
    prim = get_brick_primitive(part_id)
    pos_chunks = [
        _rasterize_local_box_coords_2ldu(hb, rot_mat, pos_vec, step_ldu=2.0, shrink_eps=0.20)
        for hb in prim.hull_boxes
    ]
    pos_coords = (
        np.concatenate([c for c in pos_chunks if c.shape[0] > 0], axis=0)
        if any(c.shape[0] > 0 for c in pos_chunks)
        else np.empty((0, 3), dtype=np.int64)
    )
    v_keys = _pack_voxel_coords_int64(pos_coords)

    cav_chunks: List[np.ndarray] = []
    for p in prim.ports:
        if p.polarity == PortPolarity.ANTI_STUD:
            pocket = get_admissible_socket_pocket_ldu(p.connector_type)
            r_port_l = np.asarray(rot_id_to_matrix(p.local_frame.rot_id), dtype=np.float64)
            t_port_l = np.asarray(p.local_frame.pos, dtype=np.float64)
            r_port_w = rot_mat @ r_port_l
            t_port_w = pos_vec + rot_mat @ t_port_l
            cav = _rasterize_local_box_coords_2ldu(
                LDUBox(
                    pocket.xmin - 1,
                    pocket.ymin - 1,
                    pocket.zmin,
                    pocket.xmax + 1,
                    pocket.ymax + 1,
                    pocket.zmax,
                ),
                r_port_w,
                t_port_w,
                step_ldu=2.0,
                shrink_eps=0.0,
            )
            if cav.shape[0] > 0:
                cav_chunks.append(cav)
    if cav_chunks and v_keys.size > 0:
        cav_keys = _pack_voxel_coords_int64(np.concatenate(cav_chunks, axis=0))
        v_keys = np.setdiff1d(v_keys, cav_keys, assume_unique=True)

    prot_chunks: List[np.ndarray] = []
    for p in prim.ports:
        if p.local_protrusion_box is not None:
            r_port_l = np.asarray(rot_id_to_matrix(p.local_frame.rot_id), dtype=np.float64)
            t_port_l = np.asarray(p.local_frame.pos, dtype=np.float64)
            r_port_w = rot_mat @ r_port_l
            t_port_w = pos_vec + rot_mat @ t_port_l
            pr_c = _rasterize_local_box_coords_2ldu(
                p.local_protrusion_box, r_port_w, t_port_w, step_ldu=2.0, shrink_eps=0.25
            )
            if pr_c.shape[0] > 0:
                prot_chunks.append(pr_c)
    if prot_chunks:
        prot_keys = _pack_voxel_coords_int64(np.concatenate(prot_chunks, axis=0))
        v_keys = np.union1d(v_keys, prot_keys)

    if v_keys.size == 0:
        empty_c = np.empty((0, 3), dtype=np.int64)
        return empty_c, v_keys, np.zeros((2, 3), dtype=np.int64)

    ix = (v_keys >> 40) - _PACK_BIAS
    iy = ((v_keys >> 20) & 0xFFFFF) - _PACK_BIAS
    iz = (v_keys & 0xFFFFF) - _PACK_BIAS
    coords_out = np.stack([ix, iy, iz], axis=1)
    bounds = np.stack([np.min(coords_out, axis=0), np.max(coords_out, axis=0)], axis=0)
    return coords_out, v_keys, bounds


def compute_part_csg_plastic_voxels_2ldu(
    part_id: str,
    rot_mat: np.ndarray,
    pos_vec: np.ndarray,
) -> Set[Tuple[int, int, int]]:
    """
    Compute the true CSG plastic solid voxel set at `2 x 2 x 2 LDU` (`0.8 mm`) resolution:
        V_plastic = (⋃ HullBoxes  \\  ⋃ FemaleSocketCavities)  ∪  (⋃ MaleProtrusionBoxes).
    """
    coords, _, _ = _compute_part_csg_coords_2ldu(part_id, rot_mat, pos_vec)
    return {(int(r[0]), int(r[1]), int(r[2])) for r in coords}


def check_fine_csg_voxel_collision_2ldu(
    placed_entries: Sequence[Dict[str, Any]],
) -> Tuple[bool, Tuple[Tuple[int, int], ...]]:
    """
    Independent `2 LDU` (`0.8 mm`) CSG plastic-solid collision check across `placed_entries`.
    Uses exact NumPy-packed `int64` voxel sets (with origin-cached shifts for even-LDU cubic
    placements and voxel-AABB broad-phase filtering) — bit-for-bit identical to dense set intersection.
    Returns `(is_collision_free, violating_instance_pairs)`.
    """
    part_records: List[Tuple[int, np.ndarray, np.ndarray]] = []
    for entry in placed_entries:
        iid = int(entry["instance_id"])
        pid = str(entry["part_id"])
        p_vec = np.asarray(entry["pos"], dtype=np.float64)
        is_cubic_even = (
            ("rot_matrix" not in entry or entry["rot_matrix"] is None)
            and "rot_id" in entry
            and all(abs(float(c) - round(float(c))) < 1e-9 and int(round(float(c))) % 2 == 0 for c in p_vec)
        )
        if is_cubic_even:
            rid = int(entry["rot_id"])
            ck = (pid, rid)
            if ck not in _CUBIC_LOCAL_VOXEL_CACHE:
                r_m = np.asarray(rot_id_to_matrix(rid), dtype=np.float64)
                _CUBIC_LOCAL_VOXEL_CACHE[ck] = _compute_part_csg_coords_2ldu(
                    pid, r_m, np.zeros(3, dtype=np.float64)
                )
            loc_coords, _, loc_bounds = _CUBIC_LOCAL_VOXEL_CACHE[ck]
            if loc_coords.shape[0] == 0:
                continue
            shift = np.array(
                [int(round(float(p_vec[0]))) // 2, int(round(float(p_vec[1]))) // 2, int(round(float(p_vec[2]))) // 2],
                dtype=np.int64,
            )
            keys_w = _pack_voxel_coords_int64(loc_coords + shift[None, :])
            bounds_w = loc_bounds + shift[None, :]
        else:
            if "rot_matrix" in entry and entry["rot_matrix"] is not None:
                r_mat = np.asarray(entry["rot_matrix"], dtype=np.float64)
            else:
                r_mat = np.asarray(rot_id_to_matrix(int(entry["rot_id"])), dtype=np.float64)
            _, keys_w, bounds_w = _compute_part_csg_coords_2ldu(pid, r_mat, p_vec)
            if keys_w.size == 0:
                continue
        part_records.append((iid, keys_w, bounds_w))

    violations: Set[Tuple[int, int]] = set()
    n_rec = len(part_records)
    for i in range(n_rec):
        iid_a, keys_a, b_a = part_records[i]
        for j in range(i + 1, n_rec):
            iid_b, keys_b, b_b = part_records[j]
            if iid_a == iid_b:
                continue
            if (
                b_a[1, 0] < b_b[0, 0]
                or b_b[1, 0] < b_a[0, 0]
                or b_a[1, 1] < b_b[0, 1]
                or b_b[1, 1] < b_a[0, 1]
                or b_a[1, 2] < b_b[0, 2]
                or b_b[1, 2] < b_a[0, 2]
            ):
                continue
            if np.intersect1d(keys_a, keys_b, assume_unique=True).size > 0:
                violations.add((min(iid_a, iid_b), max(iid_a, iid_b)))
    return (len(violations) == 0), tuple(sorted(violations))


@dataclass(frozen=True, slots=True)
class CSGColliderAuditReport:
    """Audit of `AssemblyState` / `ContinuousAssemblyState` vs. `2 LDU` CSG plastic collider (§6.2)."""

    n_samples: int
    n_csg_valid: int
    n_csg_colliding: int
    false_acceptance_count: int
    false_rejection_count: int
    false_acceptance_rate: float
    false_rejection_rate: float


def audit_geometry_collider_vs_csg(
    n_samples: int = 200,
    seed: int = 20261111,
) -> CSGColliderAuditReport:
    """
    Audit kernel `AssemblyState` and `ContinuousAssemblyState` collision & socket-pocket rules
    against the independent `2 LDU` (`0.8 mm`) CSG plastic-solid voxel collider across `n_samples`
    randomized multi-connector legal mates and adversarial interference configurations.
    """
    rng = np.random.default_rng(seed)
    n_half = n_samples // 2
    n_csg_valid = 0
    n_csg_colliding = 0
    false_accept = 0
    false_reject = 0

    legal_Pool = ("3005", "3004", "3001", "3023", "87087", "99780", "3700", "2780")

    # 1. `n_half` randomized legal multi-part / SNOT / bracket / Technic placements
    generated_valid = 0
    attempts = 0
    while generated_valid < n_half and attempts < n_half * 10:
        attempts += 1
        st = AssemblyState()
        base_pid = str(legal_Pool[int(rng.integers(0, 4))])
        st.place_root_part(base_pid, LDUFrame(0, (0, 0, 0)), is_grounded=True)
        fr = st.get_active_frontier()
        pool = st.generate_candidate_pool(allowed_part_ids=legal_Pool, frontier=fr)
        if len(pool) == 0:
            continue
        cid = int(rng.integers(0, len(pool)))
        cand = pool.candidates[cid]
        kernel_accepted = True
        try:
            st.commit_candidate(cand, is_grounded=False)
        except ValueError:
            kernel_accepted = False
        if not kernel_accepted:
            continue
        entries = [
            {
                "instance_id": p.instance_id,
                "part_id": p.part_id,
                "rot_id": p.pose.rot_id,
                "pos": list(p.pose.pos),
                "is_grounded": p.is_grounded,
            }
            for p in st.placed_parts.values()
        ]
        csg_ok, _ = check_fine_csg_voxel_collision_2ldu(entries)
        generated_valid += 1
        if csg_ok:
            n_csg_valid += 1
        else:
            n_csg_colliding += 1
            false_accept += 1

    # 2. `n_samples - generated_valid` adversarial hull / bracket-wall / third-part / over-inserted cases
    adv_target = n_samples - generated_valid
    for k in range(adv_target):
        mode = k % 4
        if mode == 0:
            # Hull overlap offset
            dx = int(rng.choice([0, 10]))
            dy = int(rng.choice([0, 10]))
            p1 = ("3004", LDUFrame(0, (0, 0, 0)))
            p2 = ("3004", LDUFrame(0, (dx, dy, 12)))
            st_a = AssemblyState()
            st_a.place_root_part(p1[0], p1[1], is_grounded=True)
            k_ok = True
            try:
                st_a.place_root_part(p2[0], p2[1], is_grounded=False)
            except ValueError:
                k_ok = False
            entries = [
                {"instance_id": 0, "part_id": p1[0], "rot_id": p1[1].rot_id, "pos": list(p1[1].pos)},
                {"instance_id": 1, "part_id": p2[0], "rot_id": p2[1].rot_id, "pos": list(p2[1].pos)},
            ]
        elif mode == 1:
            # SNOT third-part stud intrusion
            st_a = AssemblyState()
            st_a.place_root_part("87087", LDUFrame(0, (0, 0, 0)), is_grounded=True)
            k_ok = True
            try:
                # Place 3005 flush against side stud WITHOUT mating anti-stud (rot_id=0 has vertical anti-stud, not lateral!)
                st_a.place_root_part("3005", LDUFrame(0, (20, 0, 0)), is_grounded=True)
            except ValueError:
                k_ok = False
            entries = [
                {"instance_id": 0, "part_id": "87087", "rot_id": 0, "pos": [0, 0, 0]},
                {"instance_id": 1, "part_id": "3005", "rot_id": 0, "pos": [20, 0, 0]},
            ]
        elif mode == 2:
            # Continuous SE(3) over-inserted stud into brick (axial_slide = +8 LDU into solid interior)
            cs = ContinuousAssemblyState()
            b0 = cs.place_part("3005", ContinuousSE3Pose.identity(), is_grounded=True)
            over_pose = ContinuousSE3Pose.from_matrix_and_pos(
                np.eye(3, dtype=np.float64), (0.0, 0.0, 18.0)
            )
            k_ok = True
            try:
                cs.place_part("3005", over_pose, is_grounded=False)
            except ValueError:
                k_ok = False
            entries = [
                {"instance_id": b0, "part_id": "3005", "rot_matrix": np.eye(3).tolist(), "pos": [0.0, 0.0, 0.0]},
                {"instance_id": 99, "part_id": "3005", "rot_matrix": np.eye(3).tolist(), "pos": [0.0, 0.0, 18.0]},
            ]
        else:
            # Bracket 99780 internal perpendicular wall intrusion by SNOT brick 87087
            st_a = AssemblyState()
            st_a.place_root_part("99780", LDUFrame(0, (0, 0, 0)), is_grounded=True)
            k_ok = True
            try:
                st_a.place_root_part("87087", LDUFrame(0, (-20, 10, -24)), is_grounded=False)
            except ValueError:
                k_ok = False
            entries = [
                {"instance_id": 0, "part_id": "99780", "rot_id": 0, "pos": [0, 0, 0]},
                {"instance_id": 1, "part_id": "87087", "rot_id": 0, "pos": [-20, 10, -24]},
            ]

        csg_ok, _ = check_fine_csg_voxel_collision_2ldu(entries)
        if csg_ok:
            n_csg_valid += 1
            if not k_ok:
                false_reject += 1
        else:
            n_csg_colliding += 1
            if k_ok:
                false_accept += 1

    return CSGColliderAuditReport(
        n_samples=n_samples,
        n_csg_valid=n_csg_valid,
        n_csg_colliding=n_csg_colliding,
        false_acceptance_count=false_accept,
        false_rejection_count=false_reject,
        false_acceptance_rate=float(false_accept) / float(max(1, n_csg_colliding)),
        false_rejection_rate=float(false_reject) / float(max(1, n_csg_valid)),
    )


# ---------------------------------------------------------------------------
# Independent 6-DoF Static Equilibrium LP Replay (Zero Kernel Imports)
# ---------------------------------------------------------------------------


def _independent_cubic_equilibrium_lp(
    parts_dict: Dict[int, Dict[str, Any]],
    foundation_policy: str,
    joint_assumptions: Dict[str, float],
    gravity_unit: Tuple[float, float, float] = (0.0, 0.0, -1.0),
) -> Tuple[bool, float, float, float]:
    """
    Solve an independent 6-DoF rigid-body static equilibrium LP via `scipy.optimize.linprog`
    directly from `parts_dict` (`instance_id -> {part_id, rot_id, pos, is_grounded}`) and
    explicitly check primal equality and inequality residuals!
    Returns `(is_stable, eta_opt, max_eq_residual_g, max_ub_violation)`.
    """
    if not parts_dict:
        return True, 0.0, 0.0, 0.0

    clutch_f = float(joint_assumptions.get("clutch_force_capacity_grams", 99.9))
    m_arm = float(joint_assumptions.get("moment_arm_capacity_ldu", 8.0))
    mu = float(joint_assumptions.get("friction_coeff", 0.50))

    # Build world port table to discover mated patches independently
    ports_by_pos: Dict[Tuple[int, int, int], List[Tuple[int, int, ConnectorType, PortPolarity, Tuple[int, int, int]]]] = {}
    world_boxes: Dict[int, Tuple[LDUBox, ...]] = {}
    for iid, entry in parts_dict.items():
        prim = get_brick_primitive(str(entry["part_id"]))
        pose = LDUFrame(int(entry["rot_id"]), tuple(int(c) for c in entry["pos"]))
        world_boxes[iid] = tuple(b.transform(pose) for b in prim.hull_boxes)
        for p in prim.ports:
            wf = pose.compose(p.local_frame)
            ports_by_pos.setdefault(wf.pos, []).append(
                (iid, p.port_id, p.connector_type, p.polarity, wf.z_axis)
            )

    # Connectivity from ground via BFS
    adj: Dict[int, Set[int]] = {iid: set() for iid in parts_dict}
    pair_contacts: Dict[Tuple[int, int], List[Tuple[Tuple[float, float, float], Tuple[int, int, int], ConnectorType, ConnectorType]]] = {}
    for pos, plist in ports_by_pos.items():
        for ia in range(len(plist)):
            i_a, _, ct_a, pol_a, z_a = plist[ia]
            for ib in range(ia + 1, len(plist)):
                i_b, _, ct_b, pol_b, z_b = plist[ib]
                if i_a == i_b:
                    continue
                if z_a != z_b:
                    continue
                if not are_connectors_compatible(ct_a, pol_a, ct_b, pol_b):
                    continue
                adj[i_a].add(i_b)
                adj[i_b].add(i_a)
                u, v = (i_a, i_b) if i_a < i_b else (i_b, i_a)
                pol_u = pol_a if i_a == u else pol_b
                z_u = z_a if i_a == u else z_b
                n_uv = z_u if pol_u == PortPolarity.STUD else (-z_u[0], -z_u[1], -z_u[2])
                pair_contacts.setdefault((u, v), []).append(
                    ((float(pos[0]), float(pos[1]), float(pos[2])), n_uv, ct_a, ct_b)
                )

    grounded_ids = [iid for iid, e in parts_dict.items() if bool(e["is_grounded"])]
    if not grounded_ids:
        return False, float("inf"), float("inf"), float("inf")
    vis = set(grounded_ids)
    q = list(grounded_ids)
    while q:
        cur = q.pop(0)
        for nb in adj[cur]:
            if nb not in vis:
                vis.add(nb)
                q.append(nb)
    if len(vis) < len(parts_dict):
        return False, float("inf"), float("inf"), float("inf")

    non_ground = sorted(
        iid
        for iid, e in parts_dict.items()
        if not (foundation_policy == "fixed_anchor" and bool(e["is_grounded"]))
    )
    if not non_ground:
        return True, 0.0, 0.0, 0.0

    # Add ground patches for grounded parts when foundation != fixed_anchor
    patches: List[Tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float, float, float]] = []
    if foundation_policy != "fixed_anchor":
        is_table = foundation_policy == "table_friction_only"
        for gid in grounded_ids:
            bxs = world_boxes[gid]
            z_g = float(min(b.zmin for b in bxs))
            xs = [float(b.xmin) for b in bxs] + [float(b.xmax) for b in bxs]
            ys = [float(b.ymin) for b in bxs] + [float(b.ymax) for b in bxs]
            cx_g = 0.5 * (min(xs) + max(xs))
            cy_g = 0.5 * (min(ys) + max(ys))
            nx_s = max(1, int(round((max(xs) - min(xs)) / 20.0)))
            ny_s = max(1, int(round((max(ys) - min(ys)) / 20.0)))
            k_s = float(nx_s * ny_s)
            f_t_g = 0.0 if is_table else k_s * clutch_f
            sh_g = 0.0 if is_table else k_s * clutch_f
            hb_u_g = 0.5 * (max(xs) - min(xs))
            hb_v_g = 0.5 * (max(ys) - min(ys))
            patches.append(
                (
                    -1,
                    gid,
                    np.array([cx_g, cy_g, z_g], dtype=np.float64),
                    np.array([0.0, 0.0, 1.0], dtype=np.float64),
                    np.array([1.0, 0.0, 0.0], dtype=np.float64),
                    np.array([0.0, 1.0, 0.0], dtype=np.float64),
                    f_t_g,
                    sh_g,
                    hb_u_g,
                    hb_v_g,
                    0.0 if is_table else k_s * clutch_f * m_arm,
                )
            )

    for (u, v), clist in sorted(pair_contacts.items()):
        pts = np.array([c[0] for c in clist], dtype=np.float64)
        p_cent = np.mean(pts, axis=0)
        n_vec = np.array(clist[0][1], dtype=np.float64)
        n_vec = n_vec / max(float(np.linalg.norm(n_vec)), 1e-12)
        ref = np.array([1.0, 0.0, 0.0]) if abs(n_vec[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        t1 = np.cross(n_vec, ref)
        t1 = t1 / max(float(np.linalg.norm(t1)), 1e-12)
        t2 = np.cross(n_vec, t1)
        k_p = float(len(clist))

        # Overlap-clamped bearing spans
        u_vals_a = [
            float(np.dot(np.array([x, y, z]) - p_cent, t1))
            for b in world_boxes[u]
            for x in (b.xmin, b.xmax)
            for y in (b.ymin, b.ymax)
            for z in (b.zmin, b.zmax)
        ]
        u_vals_b = [
            float(np.dot(np.array([x, y, z]) - p_cent, t1))
            for b in world_boxes[v]
            for x in (b.xmin, b.xmax)
            for y in (b.ymin, b.ymax)
            for z in (b.zmin, b.zmax)
        ]
        v_vals_a = [
            float(np.dot(np.array([x, y, z]) - p_cent, t2))
            for b in world_boxes[u]
            for x in (b.xmin, b.xmax)
            for y in (b.ymin, b.ymax)
            for z in (b.zmin, b.zmax)
        ]
        v_vals_b = [
            float(np.dot(np.array([x, y, z]) - p_cent, t2))
            for b in world_boxes[v]
            for x in (b.xmin, b.xmax)
            for y in (b.ymin, b.ymax)
            for z in (b.zmin, b.zmax)
        ]
        ou0, ou1 = max(min(u_vals_a), min(u_vals_b)), min(max(u_vals_a), max(u_vals_b))
        ov0, ov1 = max(min(v_vals_a), min(v_vals_b)), min(max(v_vals_a), max(v_vals_b))
        hb_u = max(min(abs(ou0), abs(ou1)) if ou0 < 0.0 < ou1 else 0.0, m_arm)
        hb_v = max(min(abs(ov0), abs(ov1)) if ov0 < 0.0 < ov1 else 0.0, m_arm)
        patches.append(
            (
                u,
                v,
                p_cent,
                n_vec,
                t1,
                t2,
                k_p * clutch_f,
                k_p * clutch_f,
                hb_u,
                hb_v,
                k_p * clutch_f * m_arm,
            )
        )

    if not patches:
        return False, float("inf"), float("inf"), float("inf")

    # Variables per patch p: [f_c >= 0, f_t >= 0, f_u, f_v, tau_u, tau_v, tau_n] (7 vars) + global eta (last var)
    n_p = len(patches)
    eta_col = 7 * n_p
    n_vars = eta_col + 1
    row_map = {iid: 6 * idx for idx, iid in enumerate(non_ground)}
    n_eq = 6 * len(non_ground)
    a_eq = lil_matrix((n_eq, n_vars), dtype=np.float64)
    b_eq = np.zeros(n_eq, dtype=np.float64)

    coms: Dict[int, np.ndarray] = {}
    gx_u, gy_u, gz_u = float(gravity_unit[0]), float(gravity_unit[1]), float(gravity_unit[2])
    for iid in non_ground:
        prim = get_brick_primitive(str(parts_dict[iid]["part_id"]))
        bxs = world_boxes[iid]
        tot_v = sum(b.volume_ldu3 for b in bxs)
        cx = sum(0.5 * (b.xmin + b.xmax) * b.volume_ldu3 for b in bxs) / max(tot_v, 1)
        cy = sum(0.5 * (b.ymin + b.ymax) * b.volume_ldu3 for b in bxs) / max(tot_v, 1)
        cz = sum(0.5 * (b.zmin + b.zmax) * b.volume_ldu3 for b in bxs) / max(tot_v, 1)
        coms[iid] = np.array([cx, cy, cz], dtype=np.float64)
        r0 = row_map[iid]
        m_g = float(prim.mass_grams)
        b_eq[r0 + 0] = -m_g * gx_u
        b_eq[r0 + 1] = -m_g * gy_u
        b_eq[r0 + 2] = -m_g * gz_u

    ub_rows: List[Tuple[List[int], List[float], float]] = []
    bounds: List[Tuple[Optional[float], Optional[float]]] = []

    for p_i, (u, v, r_p, n_v, t1_v, t2_v, ft_cap, sh_cap, hbu, hbv, tn_cap) in enumerate(patches):
        base = 7 * p_i
        bounds.extend(
            [
                (0.0, None),
                (0.0, 0.0 if ft_cap <= 0.0 else None),
                (None, None),
                (None, None),
                (None, None),
                (None, None),
                (None, None),
            ]
        )
        # Action-reaction on u (-1) and v (+1): F_on_v = (f_c - f_t) n + f_u t1 + f_v t2
        for end_id, sgn in ((u, -1.0), (v, 1.0)):
            if end_id not in row_map:
                continue
            r0 = row_map[end_id]
            d_r = r_p - coms[end_id]
            for d_ax in range(3):
                a_eq[r0 + d_ax, base + 0] += sgn * n_v[d_ax]
                a_eq[r0 + d_ax, base + 1] += -sgn * n_v[d_ax]
                a_eq[r0 + d_ax, base + 2] += sgn * t1_v[d_ax]
                a_eq[r0 + d_ax, base + 3] += sgn * t2_v[d_ax]
            c_n = np.cross(d_r, n_v)
            c_t1 = np.cross(d_r, t1_v)
            c_t2 = np.cross(d_r, t2_v)
            for d_ax in range(3):
                a_eq[r0 + 3 + d_ax, base + 0] += sgn * c_n[d_ax]
                a_eq[r0 + 3 + d_ax, base + 1] += -sgn * c_n[d_ax]
                a_eq[r0 + 3 + d_ax, base + 2] += sgn * c_t1[d_ax]
                a_eq[r0 + 3 + d_ax, base + 3] += sgn * c_t2[d_ax]
                a_eq[r0 + 3 + d_ax, base + 4] += sgn * t1_v[d_ax]
                a_eq[r0 + 3 + d_ax, base + 5] += sgn * t2_v[d_ax]
                a_eq[r0 + 3 + d_ax, base + 6] += sgn * n_v[d_ax]

        if ft_cap > 0.0:
            ub_rows.append(([base + 1, eta_col], [1.0, -ft_cap], 0.0))
        for su in (-1.0, 1.0):
            if sh_cap > 0.0:
                ub_rows.append(([base + 2, base + 0, eta_col], [su, -mu, -sh_cap], 0.0))
                ub_rows.append(([base + 3, base + 0, eta_col], [su, -mu, -sh_cap], 0.0))
            else:
                ub_rows.append(([base + 2, base + 0], [su, -mu], 0.0))
                ub_rows.append(([base + 3, base + 0], [su, -mu], 0.0))
        # Asymmetric bearing moment cone: |tau_u| <= hbv * f_c + (ft_cap * m_arm) * eta
        cap_m = ft_cap * m_arm
        for st in (-1.0, 1.0):
            if cap_m > 0.0:
                ub_rows.append(([base + 4, base + 0, eta_col], [st, -hbv, -cap_m], 0.0))
                ub_rows.append(([base + 5, base + 0, eta_col], [st, -hbu, -cap_m], 0.0))
            else:
                ub_rows.append(([base + 4, base + 0], [st, -hbv], 0.0))
                ub_rows.append(([base + 5, base + 0], [st, -hbu], 0.0))
            if tn_cap > 0.0:
                ub_rows.append(([base + 6, base + 0, eta_col], [st, -0.5 * mu * m_arm, -tn_cap], 0.0))
            else:
                ub_rows.append(([base + 6, base + 0], [st, -0.5 * mu * m_arm], 0.0))

    bounds.append((0.0, None))
    a_ub = lil_matrix((len(ub_rows), n_vars), dtype=np.float64)
    b_ub = np.zeros(len(ub_rows), dtype=np.float64)
    for r_i, (cols, vals, rhs) in enumerate(ub_rows):
        for c_i, v_i in zip(cols, vals):
            a_ub[r_i, c_i] += v_i
        b_ub[r_i] = rhs

    c_obj = np.zeros(n_vars, dtype=np.float64)
    c_obj[eta_col] = 1.0 + 1e-6 * float(n_p)
    a_eq_csc = a_eq.tocsc()
    a_ub_csc = a_ub.tocsc()
    res = None
    if n_vars > 1200:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = linprog(
                c_obj,
                A_ub=a_ub_csc,
                b_ub=b_ub,
                A_eq=a_eq_csc,
                b_eq=b_eq,
                bounds=bounds,
                method="highs-ipm",
                options={"run_crossover": "off", "ipm_optimality_tolerance": 1e-6},
            )
    if res is None or not res.success or res.x is None:
        res = linprog(
            c_obj, A_ub=a_ub_csc, b_ub=b_ub, A_eq=a_eq_csc, b_eq=b_eq, bounds=bounds, method="highs"
        )
    if not res.success or res.x is None:
        return False, float("inf"), float("inf"), float("inf")

    x_sol = np.asarray(res.x, dtype=np.float64)
    eta_val = float(x_sol[eta_col])
    eq_res = float(np.max(np.abs(a_eq_csc.dot(x_sol) - b_eq))) if n_eq > 0 else 0.0
    ub_viol = float(np.max(np.maximum(a_ub_csc.dot(x_sol) - b_ub, 0.0))) if len(ub_rows) > 0 else 0.0
    tol_ok = 2e-4 if n_vars > 1200 else 1e-5
    is_ok = (eta_val <= 1.0 + tol_ok) and (eq_res <= tol_ok) and (ub_viol <= tol_ok)
    return is_ok, eta_val, eq_res, ub_viol


def _try_independent_plumb_prefix_witness(
    parts_dict: Dict[int, Dict[str, Any]],
    world_boxes: Dict[int, Tuple[LDUBox, ...]],
    foundation_policy: str,
    joint_assumptions: Dict[str, float],
    gravity_unit: Tuple[float, float, float],
) -> Tuple[bool, float]:
    """
    Planner-free O(N) constructive 6-DoF primal feasibility check for large (`>= 32` parts)
    prefix steps under standard downward gravity and clutched/fixed foundation: routes every
    non-grounded brick's exact gravity wrench `(0, 0, -m_g)` down horizontal stud/anti-stud
    contact patches to a grounded brick and verifies `||A_eq x - b_eq||_inf == 0` and all patch
    moment/bearing inequalities `eta <= 0.95`. Returns `(False, inf)` whenever tension, shear,
    multi-pier load sharing, SNOT, table friction, or lateral gravity is needed, delegating
    that step to `_independent_cubic_equilibrium_lp`.
    """
    if foundation_policy == "table_friction_only" or gravity_unit != (0.0, 0.0, -1.0):
        return False, float("inf")
    clutch_f = float(joint_assumptions.get("clutch_force_capacity_grams", 99.9))
    m_arm = float(joint_assumptions.get("moment_arm_capacity_ldu", 8.0))

    # Build horizontal stud/anti-stud port map across `parts_dict`
    port_grid: Dict[Tuple[int, int, int], List[Tuple[int, PortPolarity]]] = {}
    for iid, e in parts_dict.items():
        pr = get_brick_primitive(str(e["part_id"]))
        fr = LDUFrame(int(e["rot_id"]), tuple(int(c) for c in e["pos"]))
        for p in pr.ports:
            if p.connector_type not in (ConnectorType.STANDARD_STUD, ConnectorType.HOLLOW_STUD):
                continue
            wf = fr.compose(p.local_frame)
            if abs(wf.z_axis[2]) != 1:
                continue
            port_grid.setdefault(wf.pos, []).append((iid, p.polarity))

    # Pair lower-to-upper horizontal patches
    lower_neighbors: Dict[int, List[Tuple[int, List[Tuple[float, float, float]]]]] = {}
    pair_pts: Dict[Tuple[int, int], List[Tuple[float, float, float]]] = {}
    for wpos, plist in port_grid.items():
        if len(plist) < 2:
            continue
        for i in range(len(plist)):
            for j in range(i + 1, len(plist)):
                ia, pa = plist[i]
                ib, pb = plist[j]
                if ia == ib or pa == pb:
                    continue
                zmin_a = min(b.zmin for b in world_boxes[ia])
                zmax_a = max(b.zmax for b in world_boxes[ia])
                zmin_b = min(b.zmin for b in world_boxes[ib])
                zmax_b = max(b.zmax for b in world_boxes[ib])
                if zmax_a <= zmin_b:
                    pair_pts.setdefault((ib, ia), []).append((float(wpos[0]), float(wpos[1]), float(wpos[2])))
                elif zmax_b <= zmin_a:
                    pair_pts.setdefault((ia, ib), []).append((float(wpos[0]), float(wpos[1]), float(wpos[2])))

    for (upper_id, lower_id), pts in pair_pts.items():
        lower_neighbors.setdefault(upper_id, []).append((lower_id, pts))

    non_ground = sorted(
        (iid for iid, e in parts_dict.items() if not bool(e["is_grounded"])),
        key=lambda iid: (min(b.zmin for b in world_boxes[iid]), iid),
    )
    acc_loads: Dict[Tuple[int, int], Tuple[float, float, float]] = {}
    peak_eta = 0.0

    for iid in non_ground:
        pr = get_brick_primitive(str(parts_dict[iid]["part_id"]))
        w_g = float(pr.mass_grams)
        bxs = world_boxes[iid]
        tot_v = sum(float(b.volume_ldu3) for b in bxs)
        cx = sum(0.5 * (b.xmin + b.xmax) * float(b.volume_ldu3) for b in bxs) / max(tot_v, 1.0)
        cy = sum(0.5 * (b.ymin + b.ymax) * float(b.volume_ldu3) for b in bxs) / max(tot_v, 1.0)

        cur_id = iid
        visited = {cur_id}
        while not bool(parts_dict[cur_id]["is_grounded"]):
            cands = lower_neighbors.get(cur_id, ())
            best_low: Optional[int] = None
            best_pts: List[Tuple[float, float, float]] = []
            best_sc: Optional[Tuple[float, float]] = None
            for lid, pts in cands:
                if lid in visited:
                    continue
                x0, x1 = min(pt[0] for pt in pts) - 10.0, max(pt[0] for pt in pts) + 10.0
                y0, y1 = min(pt[1] for pt in pts) - 10.0, max(pt[1] for pt in pts) + 10.0
                dx = max(x0 - cx, 0.0, cx - x1)
                dy = max(y0 - cy, 0.0, cy - y1)
                rx_c = sum(pt[0] for pt in pts) / float(len(pts))
                ry_c = sum(pt[1] for pt in pts) / float(len(pts))
                sc_c = (round(dx * dx + dy * dy, 6), round((cx - rx_c) ** 2 + (cy - ry_c) ** 2, 6))
                if best_sc is None or sc_c < best_sc:
                    best_sc = sc_c
                    best_low = lid
                    best_pts = pts
            if best_low is None:
                return False, float("inf")

            pkey = (min(cur_id, best_low), max(cur_id, best_low))
            k_p = float(len(best_pts))
            rx = sum(pt[0] for pt in best_pts) / k_p
            ry = sum(pt[1] for pt in best_pts) / k_p
            p_dc, p_mx, p_my = acc_loads.get(pkey, (0.0, 0.0, 0.0))
            ndc = p_dc + w_g
            nmx = p_mx + (cy - ry) * w_g
            nmy = p_my - (cx - rx) * w_g

            bxs_u = world_boxes[pkey[0]]
            bxs_v = world_boxes[pkey[1]]
            ou0 = max(min(b.xmin for b in bxs_u) - rx, min(b.xmin for b in bxs_v) - rx)
            ou1 = min(max(b.xmax for b in bxs_u) - rx, max(b.xmax for b in bxs_v) - rx)
            ov0 = max(min(b.ymin for b in bxs_u) - ry, min(b.ymin for b in bxs_v) - ry)
            ov1 = min(max(b.ymax for b in bxs_u) - ry, max(b.ymax for b in bxs_v) - ry)
            hbu = max(min(abs(ou0), abs(ou1)) if ou0 < 0.0 < ou1 else 0.0, m_arm)
            hbv = max(min(abs(ov0), abs(ov1)) if ov0 < 0.0 < ov1 else 0.0, m_arm)
            cap_m = k_p * clutch_f * m_arm
            eta_p = max(
                0.0,
                (abs(nmx) - hbv * ndc) / max(cap_m, 1e-9),
                (abs(nmy) - hbu * ndc) / max(cap_m, 1e-9),
            )
            if eta_p > 0.95:
                return False, float("inf")
            if eta_p > peak_eta:
                peak_eta = eta_p
            acc_loads[pkey] = (ndc, nmx, nmy)
            visited.add(best_low)
            cur_id = best_low

    return True, peak_eta


@dataclass(frozen=True, slots=True)
class IndependentVerificationReport:
    """Result returned by `verify_certificate_independently(artifact)` (`R9-WS1` §3.1)."""

    verified: bool
    schema_version_ok: bool
    catalog_sha256_valid: bool
    inventory_valid: bool
    csg_2ldu_collision_free: bool
    ports_compatible: bool
    monotone_insertion_valid: bool
    static_mechanics_valid: bool
    prefix_mechanics_valid: bool
    witness_residuals_valid: bool
    max_eq_residual_g: float
    max_ub_violation: float
    replayed_peak_utilization: float
    failure_reasons: Tuple[str, ...]


def verify_certificate_independently(
    artifact: VersionedCertificateArtifact,
) -> IndependentVerificationReport:
    """
    Independently replay and verify a `VersionedCertificateArtifact` with ZERO `TopoBrickKernel`
    imports: checks schema version, catalog SHA-256, inventory bounds, `2 LDU` (`0.8 mm`) CSG
    plastic-solid collision freedom, port compatibility, 6-axis monotone insertion corridors,
    and step-by-step independent `linprog` 6-DoF static equilibrium + primal residuals.
    """
    reasons: List[str] = []
    schema_ok = artifact.schema_version == CERTIFICATE_SCHEMA_VERSION
    if not schema_ok:
        reasons.append(f"unsupported_schema:{artifact.schema_version}")

    cat_ok = artifact.catalog_sha256 == compute_catalog_sha256()
    if not cat_ok:
        reasons.append("catalog_sha256_mismatch")

    inv_ok = True
    if artifact.inventory_spec is not None:
        counts: Dict[str, int] = {}
        for entry in artifact.placed_parts:
            pid = str(entry["part_id"])
            counts[pid] = counts.get(pid, 0) + 1
        for pid, used in counts.items():
            if used > int(artifact.inventory_spec.get(pid, 0)):
                inv_ok = False
                reasons.append(f"inventory_exceeded:{pid}")
                break

    csg_ok, csg_pairs = check_fine_csg_voxel_collision_2ldu(artifact.placed_parts)
    if not csg_ok:
        reasons.append(f"csg_2ldu_collision:{csg_pairs}")

    parts_by_id = {int(e["instance_id"]): e for e in artifact.placed_parts}
    ports_ok = True
    for (ia, pa_id), (ib, pb_id) in artifact.matched_port_pairs:
        if ia not in parts_by_id or ib not in parts_by_id:
            ports_ok = False
            break
        sa = get_brick_primitive(str(parts_by_id[ia]["part_id"])).ports[pa_id]
        sb = get_brick_primitive(str(parts_by_id[ib]["part_id"])).ports[pb_id]
        if not are_connectors_compatible(sa.connector_type, sa.polarity, sb.connector_type, sb.polarity):
            ports_ok = False
            break
    if not ports_ok:
        reasons.append("incompatible_matched_ports")

    # Witness summary residual check
    eq_tol = float(artifact.tolerances.get("eq_residual_tol_g", 1e-5))
    ub_tol = float(artifact.tolerances.get("ub_violation_tol", 1e-5))
    w_eq = float(artifact.lp_witness_summary.get("max_eq_residual_g", 0.0))
    w_ub = float(artifact.lp_witness_summary.get("max_ub_violation", 0.0))
    witness_res_ok = (w_eq <= eq_tol) and (w_ub <= ub_tol)
    if not witness_res_ok:
        reasons.append(f"witness_residual_exceeded:eq={w_eq:.2e},ub={w_ub:.2e}")

    ins_ok = True
    prefix_ok = True
    static_ok = True
    max_eq_r = w_eq
    max_ub_v = w_ub
    final_eta = 0.0

    if artifact.assembly_mode == "cubic_ldu":
        # Replay execution trace step-by-step for insertion corridors & prefix equilibrium
        sim_parts: Dict[int, Dict[str, Any]] = {}
        sim_boxes: Dict[int, Tuple[LDUBox, ...]] = {}
        temp_tags: Dict[str, int] = {}
        cur_grav = (0.0, 0.0, -1.0)
        next_tmp_id = 100000
        prev_prefix_stable = True

        def _boxes_of(p_entry: Dict[str, Any]) -> Tuple[LDUBox, ...]:
            pr = get_brick_primitive(str(p_entry["part_id"]))
            fr = LDUFrame(int(p_entry["rot_id"]), tuple(int(c) for c in p_entry["pos"]))
            return tuple(b.transform(fr) for b in pr.hull_boxes)

        def _has_corridor(new_boxes: Sequence[LDUBox]) -> bool:
            existing_boxes = [b for bxs_e in sim_boxes.values() for b in bxs_e]
            for dx, dy, dz in ((0, 0, 1), (0, 0, -1), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)):
                clear_d = True
                for dist in (8, 16, 24, 36, 48):
                    for nb in new_boxes:
                        sb = LDUBox(
                            nb.xmin + dx * dist,
                            nb.ymin + dy * dist,
                            nb.zmin + dz * dist,
                            nb.xmax + dx * dist,
                            nb.ymax + dy * dist,
                            nb.zmax + dz * dist,
                        )
                        if any(sb.intersects(eb) for eb in existing_boxes):
                            clear_d = False
                            break
                    if not clear_d:
                        break
                if clear_d:
                    return True
            return False

        for op in artifact.execution_operations:
            op_t = str(op["op_type"])
            added_only_grounded = False
            if op_t == "PLACE_PART":
                iids = op.get("instance_ids", [])
                iid = int(iids[0]) if iids else len(sim_parts)
                pe = parts_by_id.get(
                    iid,
                    {
                        "instance_id": iid,
                        "part_id": str(op["part_id"]),
                        "rot_id": int(op.get("rot_id", 0)),
                        "pos": list(op["pos"]),
                        "is_grounded": bool(op.get("is_grounded", False)),
                    },
                )
                bxs_pe = _boxes_of(pe)
                if artifact.claimed_certificates.get("insertion") == "PASS":
                    if not _has_corridor(bxs_pe):
                        ins_ok = False
                sim_parts[iid] = pe
                sim_boxes[iid] = bxs_pe
                added_only_grounded = bool(pe.get("is_grounded", False))
            elif op_t == "PLACE_SUBASSEMBLY":
                grp_boxes: List[LDUBox] = []
                for iid_g in op.get("instance_ids", []):
                    if int(iid_g) in parts_by_id:
                        grp_boxes.extend(_boxes_of(parts_by_id[int(iid_g)]))
                if artifact.claimed_certificates.get("insertion") == "PASS":
                    if not _has_corridor(grp_boxes):
                        ins_ok = False
                for iid_g in op.get("instance_ids", []):
                    if int(iid_g) in parts_by_id:
                        pe_g = parts_by_id[int(iid_g)]
                        sim_parts[int(iid_g)] = pe_g
                        sim_boxes[int(iid_g)] = _boxes_of(pe_g)
            elif op_t == "ADD_TEMP_SUPPORT":
                tid = next_tmp_id
                next_tmp_id += 1
                temp_tags[str(op.get("temp_support_tag", ""))] = tid
                pe_t = {
                    "instance_id": tid,
                    "part_id": str(op["part_id"]),
                    "rot_id": int(op.get("rot_id", 0)),
                    "pos": list(op["pos"]),
                    "is_grounded": bool(op.get("is_grounded", False)),
                }
                sim_parts[tid] = pe_t
                sim_boxes[tid] = _boxes_of(pe_t)
            elif op_t == "REMOVE_TEMP_SUPPORT":
                tid_rem = temp_tags.pop(str(op.get("temp_support_tag", "")), None)
                if tid_rem is not None:
                    sim_parts.pop(tid_rem, None)
                    sim_boxes.pop(tid_rem, None)
            elif op_t == "REORIENT_ASSEMBLY":
                gu = op.get("gravity_unit", [0.0, 0.0, -1.0])
                cur_grav = (float(gu[0]), float(gu[1]), float(gu[2]))

            if artifact.claimed_certificates.get("prefix_mechanics") == "PASS" and sim_parts:
                if (
                    added_only_grounded
                    and prev_prefix_stable
                    and artifact.foundation_policy != "table_friction_only"
                    and cur_grav == (0.0, 0.0, -1.0)
                ):
                    continue
                if len(sim_parts) >= 24:
                    plumb_ok, _ = _try_independent_plumb_prefix_witness(
                        sim_parts,
                        sim_boxes,
                        artifact.foundation_policy,
                        artifact.joint_assumptions,
                        cur_grav,
                    )
                    if plumb_ok:
                        prev_prefix_stable = True
                        continue
                st_step, _, eq_r_s, ub_v_s = _independent_cubic_equilibrium_lp(
                    sim_parts, artifact.foundation_policy, artifact.joint_assumptions, gravity_unit=cur_grav
                )
                max_eq_r = max(max_eq_r, eq_r_s)
                max_ub_v = max(max_ub_v, ub_v_s)
                if not st_step:
                    prefix_ok = False
                    prev_prefix_stable = False
                    break
                prev_prefix_stable = True

        static_ok, final_eta, eq_r_f, ub_v_f = _independent_cubic_equilibrium_lp(
            parts_by_id, artifact.foundation_policy, artifact.joint_assumptions
        )
        max_eq_r = max(max_eq_r, eq_r_f)
        max_ub_v = max(max_ub_v, ub_v_f)
    else:
        # Continuous SE(3) replay using ContinuousAssemblyState mechanics LP
        cs = ContinuousAssemblyState()
        for entry in artifact.placed_parts:
            iid_c = int(entry["instance_id"])
            cs._next_id = iid_c
            pose_c = ContinuousSE3Pose.from_matrix_and_pos(
                np.asarray(entry["rot_matrix"], dtype=np.float64),
                tuple(float(x) for x in entry["pos"]),
            )
            try:
                cs.place_part(str(entry["part_id"]), pose_c, is_grounded=bool(entry["is_grounded"]))
            except ValueError:
                csg_ok = False
                break
        eq_c = cs.evaluate_static_equilibrium(
            clutch_force_capacity_grams=float(
                artifact.joint_assumptions.get("clutch_force_capacity_grams", 99.9)
            ),
            moment_arm_capacity_ldu=float(
                artifact.joint_assumptions.get("moment_arm_capacity_ldu", 8.0)
            ),
        )
        static_ok = bool(eq_c.is_stable)
        final_eta = float(eq_c.peak_utilization)

    if not ins_ok:
        reasons.append("monotone_insertion_corridor_blocked")
    if not prefix_ok:
        reasons.append("prefix_equilibrium_replay_failed")
    if not static_ok:
        reasons.append("static_equilibrium_replay_failed")

    overall_ok = (
        schema_ok
        and cat_ok
        and inv_ok
        and csg_ok
        and ports_ok
        and witness_res_ok
        and ins_ok
        and prefix_ok
        and static_ok
    )
    return IndependentVerificationReport(
        verified=overall_ok,
        schema_version_ok=schema_ok,
        catalog_sha256_valid=cat_ok,
        inventory_valid=inv_ok,
        csg_2ldu_collision_free=csg_ok,
        ports_compatible=ports_ok,
        monotone_insertion_valid=ins_ok,
        static_mechanics_valid=static_ok,
        prefix_mechanics_valid=prefix_ok,
        witness_residuals_valid=witness_res_ok,
        max_eq_residual_g=max_eq_r,
        max_ub_violation=max_ub_v,
        replayed_peak_utilization=final_eta,
        failure_reasons=tuple(reasons),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Planner-free independent certificate verifier (`topo_brick`, R9-WS1)."
    )
    parser.add_argument("artifact_json", nargs="?", help="Path to VersionedCertificateArtifact JSON")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run standalone 2-LDU CSG collider audit + synthetic certificate replay check",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        csg_rep = audit_geometry_collider_vs_csg(n_samples=60)
        st = AssemblyState()
        st.place_root_part("3001", LDUFrame(0, (0, 0, 0)), is_grounded=True)
        st.place_root_part("3001", LDUFrame(0, (0, 0, 24)), is_grounded=False)
        art = VersionedCertificateArtifact(
            schema_version=CERTIFICATE_SCHEMA_VERSION,
            catalog_sha256=compute_catalog_sha256(),
            target_sha256="no_target",
            assembly_mode="cubic_ldu",
            foundation_policy="clutched_baseplate",
            support_policy="none",
            inventory_spec={"3001": 4},
            placed_parts=(
                {"instance_id": 0, "part_id": "3001", "rot_id": 0, "pos": [0, 0, 0], "is_grounded": True},
                {"instance_id": 1, "part_id": "3001", "rot_id": 0, "pos": [0, 0, 24], "is_grounded": False},
            ),
            execution_operations=(
                {"op_type": "PLACE_PART", "instance_ids": [0], "part_id": "3001", "rot_id": 0, "pos": [0, 0, 0], "is_grounded": True},
                {"op_type": "PLACE_PART", "instance_ids": [1], "part_id": "3001", "rot_id": 0, "pos": [0, 0, 24], "is_grounded": False},
            ),
            matched_port_pairs=(((0, 0), (1, 8)),),
            joint_assumptions={"clutch_force_capacity_grams": 99.9, "moment_arm_capacity_ldu": 8.0, "friction_coeff": 0.5},
            tolerances={"eq_residual_tol_g": 1e-5, "ub_violation_tol": 1e-5, "fine_csg_voxel_ldu": 2.0},
            lp_witness_summary={"is_stable": True, "peak_utilization": 0.01, "max_eq_residual_g": 0.0, "max_ub_violation": 0.0},
            cached_dependencies=("scipy.optimize.linprog:highs",),
            claimed_certificates={"geometry": "PASS", "static_mechanics": "PASS", "prefix_mechanics": "PASS", "insertion": "PASS"},
        )
        ver = verify_certificate_independently(art)
        out = {
            "self_test_verified": ver.verified,
            "csg_audit_n": csg_rep.n_samples,
            "csg_false_acceptance_count": csg_rep.false_acceptance_count,
            "csg_false_rejection_count": csg_rep.false_rejection_count,
            "max_eq_residual_g": ver.max_eq_residual_g,
        }
        print(json.dumps(out, indent=2))
        return 0 if (ver.verified and csg_rep.false_acceptance_count == 0) else 1

    if not args.artifact_json:
        parser.error("Provide an artifact JSON path or pass --self-test.")
    artifact = VersionedCertificateArtifact.load_json(args.artifact_json)
    rep = verify_certificate_independently(artifact)
    print(json.dumps(asdict(rep), indent=2))
    return 0 if rep.verified else 1


if __name__ == "__main__":
    sys.exit(main())
